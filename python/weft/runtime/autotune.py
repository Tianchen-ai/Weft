from __future__ import annotations

import threading
import math
import numbers
from pathlib import Path

from .cache import ArtifactCache
from .compiler import JITKernel, KernelSource, native_target
from .driver import Toolchain, file_identity
from .tuning import Config, Study, TuningPolicy
from .tuning_state import TuningState
from .configuration import resolve_configurations


class _NativeTrials:
    """One isolated writable snapshot shared by sequential candidate calls.

    All candidates have the same public ABI and bind the same caller objects.
    Read-only inputs remain borrowed. A trial never owns or closes the session's
    shared storage, and the session is released only after Study closes trials.
    """

    def __init__(self, validator, max_state_bytes):
        self.validator = validator
        self.max_state_bytes = max_state_bytes
        self.state = None

    def storage(self, bound):
        if self.state is None:
            self.state = TuningState(bound, max_bytes=self.max_state_bytes)
        return self.state

    def close(self):
        if self.state is not None:
            self.state.close()
            self.state = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class _NativeTrial:
    def __init__(self, compiled, keywords, session):
        self.compiled = compiled
        self.bound = compiled.interface.bind((), keywords)
        self.session = session
        self.call = None

    def _prepare(self):
        state = self.session.storage(self.bound)
        if self.call is None:
            self.call = self.compiled.prepare(**state.keywords())
        state.reset()

    def validate(self):
        if self.session.validator is None:
            return False
        self._prepare()
        # The validator receives both the untouched inputs and isolated trial
        # arguments. It must raise on failure; its work is outside native timing.
        self.call()
        verdict = self.session.validator(self.bound.keywords(), self.session.state.keywords())
        if verdict is not None:
            raise TypeError("a numerical validator must return None on success and raise on failure")
        return True

    def measure_ns(self):
        self._prepare()
        return self.call.time_ns()

    def close(self):
        self.call = None
        self.compiled.close()


class AutotunedKernel:
    """Lazy native selection over author-supplied complete configurations.

    Shapes, strides, Encoding, aliases and alignment are always workload keys.
    `key` adds scalar values or shape parameters that select different workloads.
    Input contents are deliberately not inspected. `validation_key` names the
    author's numerical contract and must change when that contract changes.
    """

    def __init__(self, definition, configs, *, key=(), policy=None, validator=None,
                 validation_key=None, toolchain=None, cache=None):
        self.definition = definition
        self.source = KernelSource.capture(definition)
        self._built = None
        self._initialize(configs, key=key, policy=policy, validator=validator,
                         validation_key=validation_key, toolchain=toolchain, cache=cache)

    @classmethod
    def from_candidates(cls, candidates, *, key=(), policy=None, validator=None,
                        validation_key=None, toolchain=None, cache=None):
        from .candidates import BuiltCandidates

        if not isinstance(candidates, BuiltCandidates):
            raise TypeError("prebuilt tuning requires BuiltCandidates")
        if candidates.artifacts and not candidates.native:
            raise ValueError("prebuilt tuning requires explicitly linked native artifacts")
        result = cls.__new__(cls)
        result.definition = None
        result.source = candidates.source
        result._built = candidates
        result._initialize(candidates.configs, key=key, policy=policy, validator=validator,
                           validation_key=validation_key, toolchain=toolchain, cache=cache)
        return result

    @classmethod
    def from_ir(cls, canonical, configs, *, key=(), policy=None, validator=None,
                validation_key=None, toolchain=None, cache=None):
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        result = cls.__new__(cls)
        result.definition = None
        result.source = KernelSource.from_ir(canonical, toolchain=selected)
        result._built = None
        result._initialize(configs, key=key, policy=policy, validator=validator,
                           validation_key=validation_key, toolchain=selected, cache=cache)
        return result

    def _initialize(self, configs, *, key, policy, validator, validation_key, toolchain, cache):
        self.policy = TuningPolicy() if policy is None else policy
        if not isinstance(self.policy, TuningPolicy):
            raise TypeError("policy must be a TuningPolicy")
        self.configs, self.call_signature = resolve_configurations(self.source, configs,
            max_candidates=self.policy.max_candidates)
        self.key = tuple(key)
        if isinstance(key, str) or len(set(self.key)) != len(self.key):
            raise ValueError("key must be a sequence of distinct scalar/shape names")
        scalars = {parameter.name for parameter in self.call_signature.arguments if parameter.kind == "scalar"}
        if set(self.key) - (scalars | set(self.call_signature.shape_parameters)):
            raise ValueError("key entries must name declared scalar or shape arguments")
        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable")
        if (validator is None) != (validation_key is None):
            raise ValueError("validator and its explicit validation_key must be supplied together")
        if validation_key is not None and (not isinstance(validation_key, str) or not validation_key):
            raise ValueError("validation_key must be a nonempty contract name")
        self.validator = validator
        self.validation_key = validation_key
        self.toolchain = Toolchain.from_environment() if toolchain is None else toolchain
        self.cache = cache if isinstance(cache, ArtifactCache) else ArtifactCache(cache)
        self._lock = threading.RLock()
        self._winners = {}
        self._retired = []
        self._tool_context = None
        self.last_result = None

    def _normalize(self, args, kwargs, allocate):
        return self.call_signature.normalize(args, kwargs, allocate_outputs=allocate)

    def _workload(self, call):
        buffers, views = [], []
        for parameter in self.call_signature.arguments:
            if parameter.kind != "view":
                continue
            buffer = call.buffers[parameter.name]
            buffer.validate(writable=parameter.writable)
            span = buffer.byte_span() if buffer.encoding.startswith("dense.") else buffer.nbytes
            aliases = []
            for other_name, other, other_span in buffers:
                if span and other_span and buffer.address < other.address + other_span and other.address < buffer.address + span:
                    aliases.append((other_name, buffer.address - other.address))
            views.append((parameter.name, buffer.encoding, buffer.shape, buffer.strides,
                          math.gcd(buffer.address, 256) if span else 256, tuple(aliases)))
            buffers.append((parameter.name, buffer, span))
        values = {**call.dimensions, **{parameter.name: call.values[parameter.name]
                                      for parameter in self.call_signature.arguments if parameter.kind == "scalar"}}
        keys = {}
        for name in self.key:
            value = values[name]
            if isinstance(value, numbers.Integral):
                keys[name] = int(value)
            elif isinstance(value, numbers.Real) and math.isfinite(value):
                keys[name] = float(value)
            else:
                raise TypeError(f"workload key {name} must be a finite scalar or integer shape")
        return tuple(views), tuple(keys.items())

    def _identity(self, workload, target, tools_identity):
        views, keys = workload
        return {"kind": "native-kernel", "canonical": self.source.canonical,
                "signature": self.source.signature, "interface": self.call_signature.semantic_dict(),
                "views": [{"name": name, "encoding": encoding, "shape": list(shape), "strides": list(strides),
                           "alignment": alignment,
                           "aliases": [{"argument": other, "byte_offset": offset} for other, offset in aliases]}
                          for name, encoding, shape, strides, alignment, aliases in views], "key": dict(keys),
                "validation": self.validation_key, "target": target.to_dict(),
                "toolchain": tools_identity}

    def _tools(self):
        if self._tool_context is None:
            # Loading a prebuilt collection needs capability discovery but no
            # system compiler. Executable identities replace compilation inputs
            # in the selection key; moving/replacing libraries causes a miss.
            selected = self.toolchain.resolve(native_library=self._built is None)
            identity = {"native": selected.identity(native_library=self._built is None),
                        "runtime": [file_identity(path) for path in sorted(Path(__file__).parent.glob("*.py"))]}
            if self._built is not None:
                identity["candidates"] = self._built.identity()
            self._tool_context = selected, identity
        return self._tool_context

    def _compile(self, config, target, toolchain):
        extensions = () if config.options.matrix_extension == "none" else (config.options.matrix_extension,)
        target.check_execution(extensions)
        artifact = (self.source.build(target=target.build_target, options=config.options,
                                      toolchain=toolchain, cache=self.cache) if self._built is None
                    else self._built.entry(config).require_artifact())
        # Trials own their handles; closing a loser cannot close another JIT's
        # shared loaded handle, nor a previously prepared call to the winner.
        return artifact.load(target=target, toolchain=toolchain)

    def _select(self, call, *, refresh=False):
        with self._lock:
            target = native_target(self.toolchain)
            view_workload = self._workload(call)
            # Source, validation contract, configurations and toolchain are
            # immutable for this callable. Only the exact dynamic workload and
            # execution target participate in the in-process winner lookup.
            # Persistent identity serialization is confined to a cold selection.
            # Tuple equality alone aliases 1 with 1.0 and +0.0 with -0.0,
            # unlike the persistent JSON identity. Preserve those distinctions
            # for scalar keys without serializing every View/target on a hit.
            scalar_key = tuple((name, "integer", value) if type(value) is int else
                               (name, "real", value.hex()) for name, value in view_workload[1])
            workload = (view_workload[0], scalar_key, target)
            previous = self._winners.get(workload)
            if not refresh and previous is not None and not previous[0].closed:
                self.last_result = previous[1]
                return previous
            toolchain, tool_identity = self._tools()
            identity = self._identity(view_workload, target, tool_identity)
            keywords = call.keywords(self.call_signature)

            with _NativeTrials(self.validator, self.policy.max_state_bytes) as session:
                def prepare(config):
                    compiled = self._compile(config, target, toolchain)
                    try:
                        if compiled.interface.semantic_dict() != self.call_signature.semantic_dict():
                            raise ValueError("compiled tuning candidate changes the common public interface")
                        return _NativeTrial(compiled, keywords, session)
                    except BaseException:
                        compiled.close()
                        raise

                result = Study(self.configs, identity=identity, timing_scope="native-entry-with-ctypes-ns",
                               policy=self.policy, cache=self.cache).run(prepare, refresh=refresh)
            selected = self._compile(result.winner, target, toolchain)
            try:
                selected.interface.bind_call(call)
            except BaseException:
                selected.close()
                raise
            # Existing PreparedCalls retain the old handle when a refresh picks
            # another binding. Explicit close() closes all handles we still own.
            if previous is not None:
                self._retired.append(previous[0])
            self._winners[workload] = (selected, result)
            self.last_result = result
            return selected, result

    def tune(self, *args, refresh=False, **kwargs):
        """Select without executing on caller-owned mutable storage."""
        call = self._normalize(args, kwargs, True)
        return self._select(call, refresh=refresh)[1]

    def prepare(self, *args, **kwargs):
        call = self._normalize(args, kwargs, False)
        return self._prepare_call(call)

    def prepare_run(self, *args, **kwargs):
        call = self._normalize(args, kwargs, True)
        return self._prepare_call(call)

    def _prepare_call(self, call):
        self.call_signature._require_call(call, native=True)
        compiled, _ = self._select(call)
        return compiled._prepare_call(call)

    def __call__(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)()

    def run(self, *args, **kwargs):
        return self.prepare_run(*args, **kwargs)()

    def fixed(self, config: Config | None = None):
        """Use a measured winner as an ordinary JIT/AOT binding, with no tuning."""
        if config is None:
            if self.last_result is None:
                raise ValueError("no workload has been selected; provide a Config explicitly")
            config = self.last_result.winner
        if config not in self.configs:
            raise ValueError("fixed configuration is outside this kernel's declared candidate set")
        if self._built is not None:
            return self._built.entry(config).require_artifact().load(toolchain=self.toolchain)
        result = JITKernel.from_source(self.source, options=config.options, toolchain=self.toolchain, cache=self.cache)
        result.definition = self.definition
        return result

    def build_candidates(self, *, target, native_library=True):
        """Compile ahead of time without native calls or choosing a winner."""
        from .candidates import BuiltCandidates

        if self._built is not None:
            if target != self._built.target:
                raise ValueError("prebuilt candidates cannot be rebuilt for another target")
            if native_library and not self._built.native:
                raise ValueError("link prebuilt candidates explicitly")
            return self._built
        return BuiltCandidates.build(self.source, self.configs, target=target,
            toolchain=self.toolchain, native_library=native_library, cache=self.cache,
            max_candidates=self.policy.max_candidates)

    @property
    def selections(self):
        """Current workload records, without triggering discovery or measurement."""
        with self._lock:
            return tuple(result for _, result in self._winners.values())

    @property
    def artifacts(self):
        """Distinct live winner artifacts; no choice is made between workloads."""
        with self._lock:
            result = {}
            for compiled, _ in self._winners.values():
                result[compiled.artifact.path] = compiled.artifact
            return tuple(result.values())

    def close(self):
        with self._lock:
            for compiled, _ in self._winners.values():
                compiled.close()
            for compiled in self._retired:
                compiled.close()
            self._retired.clear()
            self._winners.clear()
            self._tool_context = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def autotune(definition, configs, *, key=(), policy=None, validator=None,
             validation_key=None, toolchain=None, cache=None):
    return AutotunedKernel(definition, configs, key=key, policy=policy, validator=validator,
                           validation_key=validation_key, toolchain=toolchain, cache=cache)


def autotune_ir(canonical, configs, *, key=(), policy=None, validator=None,
                validation_key=None, toolchain=None, cache=None):
    return AutotunedKernel.from_ir(canonical, configs, key=key, policy=policy, validator=validator,
                                  validation_key=validation_key, toolchain=toolchain, cache=cache)
