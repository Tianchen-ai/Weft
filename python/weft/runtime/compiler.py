from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

from weft.api import KernelDefinition
from weft.frontend import FrontendCompiler

from .artifact import Artifact, declaration_header, validate_compilation
from .cache import ArtifactCache
from .compilation import CompilationPipeline
from .driver import CompilationError, Toolchain, file_identity, run_compiler
from .invocation import CompiledKernel
from .options import CompileOptions
from .target import NativeTarget, Target
from .signature import CallSignature, encode_signature
from .dispatch import dispatch_source


@dataclass(frozen=True)
class KernelSource:
    canonical: str
    signature: list[dict] | None
    description: dict

    @classmethod
    def capture(cls, definition: KernelDefinition):
        if not isinstance(definition, KernelDefinition):
            raise TypeError("expected an @weft.kernel definition")
        frontend = FrontendCompiler(definition)
        canonical = frontend.compile()
        return cls(canonical, encode_signature(definition.signature), frontend.interface_description)

    @classmethod
    def from_ir(cls, canonical, *, toolchain=None):
        """Read the public source contract from the canonical compiler authority.

        External frontends supply ordinary Canonical Kernel IR. No Python source
        parser, target discovery, physical selection or system compiler runs at
        this boundary. Python-only defaults are absent and arguments must be
        supplied explicitly. The final artifact ABI is checked against this
        source contract by the same compilation service as Python definitions.
        """
        if not isinstance(canonical, str) or not canonical.strip():
            raise TypeError("expected nonempty Canonical Kernel IR")
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        command = [selected.compiler_path(), "--emit=kernel-interface"]
        text = run_compiler(command, canonical, stage="canonical invocation interface")
        try:
            value = json.loads(text)
        except ValueError as error:
            raise CompilationError("compiler returned invalid canonical interface JSON",
                                   command=command, stdout=text) from error
        if not isinstance(value, dict) or set(value) != {"kind", "symbol", "interface"} or value["kind"] != "weft-kernel-interface":
            raise CompilationError("compiler returned an incomplete canonical invocation interface")
        description = value["interface"]
        if not isinstance(description, dict) or set(description) != {"arguments", "shape_parameters", "source_parameters"}:
            raise CompilationError("canonical invocation interface requires arguments, shapes and source domains")
        # Decode metadata now, independently of candidate compilation or native
        # loading. A malformed entry cannot become a partially initialized JIT.
        CallSignature(description)
        if not isinstance(description["source_parameters"], dict):
            raise CompilationError("canonical source domains must be a mapping")
        for name, choices in description["source_parameters"].items():
            if not isinstance(choices, list) or any(type(value) is not int or value <= 0 for value in choices):
                raise CompilationError(f"source domain {name!r} must contain positive integer choices")
        return cls(canonical, None, description)

    def options(self, options: CompileOptions) -> CompileOptions:
        domains = self.description["source_parameters"]
        unknown = set(options.meta) - domains.keys()
        if unknown:
            raise CompilationError(f"source bindings name undeclared auto parameters: {sorted(unknown)}")
        bindings = dict(options.meta)
        for name, choices in domains.items():
            if name not in bindings:
                if len(choices) != 1:
                    raise CompilationError(f"source auto parameter {name!r} needs an explicit binding")
                bindings[name] = choices[0]
            if choices and bindings[name] not in choices:
                raise CompilationError(f"{name}={bindings[name]} is outside its declared source domain {choices}")
        return replace(options, meta=bindings)

    def interface(self, bindings=None) -> CallSignature:
        return CallSignature(self.description, self.signature, bindings=bindings)

    def build(self, *, target: Target, options: CompileOptions, toolchain: Toolchain,
              native_library: bool = True, cache=None) -> Artifact:
        return _build(self.canonical, self.signature, description=self.description, target=target,
                      options=self.options(options), toolchain=toolchain, native_library=native_library, cache=cache)


def _cache(value) -> ArtifactCache:
    return value if isinstance(value, ArtifactCache) else ArtifactCache(value)


def _build(canonical: str, signature: list[dict] | None, *, target: Target,
           options: CompileOptions, toolchain: Toolchain,
           native_library: bool, cache, description: dict | None = None,
           physical_input: str | None = None, checkpoint: str | None = None) -> Artifact:
    if (physical_input is None) != (checkpoint is None):
        raise ValueError("a physical compilation input requires its explicit checkpoint")
    if checkpoint is not None and checkpoint not in ("contractions", "panel-transfers", "layout"):
        raise ValueError("unknown physical compilation checkpoint")
    entry = {None: "canonical", "contractions": "contraction-input", "layout": "layout-input",
             "panel-transfers": "panel-transfer-input"}[checkpoint]
    arguments = (["--emit=artifact", *target.arguments(), *options.arguments()] if checkpoint is None
                 else ["--emit=artifact", f"--resume-{entry}"])
    selected = toolchain.resolve(native_library=native_library)
    identity = {"operation": "compile-kernel", "canonical": canonical,
                "signature": signature, "target": target.to_dict(), "options": options.to_dict(),
                "interface": description,
                "compiler_arguments": arguments, "physical_input": physical_input,
                "runtime": [file_identity(path) for path in sorted(Path(__file__).parent.glob("*.py"))],
                "native_library": native_library, "toolchain": selected.identity(native_library=native_library)}

    def builder(staging: Path) -> None:
        command = [selected.compiler, *arguments]
        stage = f"Weft {entry} compilation for {target.march}"
        output = run_compiler(command, canonical if physical_input is None else physical_input,
                              stage=stage, structured=True)
        try:
            result = json.loads(output)
        except ValueError as error:
            raise CompilationError("compiler returned an invalid artifact JSON document",
                                   stage=stage, command=command, stdout=output) from error
        if not isinstance(result, dict) or "compilation" not in result:
            raise CompilationError("compiler artifact has no pass execution report",
                                   stage=stage, command=command, stdout=output)
        compilation = CompilationPipeline.from_dict(result["compilation"])
        if compilation.entry != entry:
            raise CompilationError("compiler artifact did not use the requested compilation entry point",
                                   stage=stage, command=command, stdout=output)
        metadata = validate_compilation(result, target, options)
        if description is not None and CallSignature(metadata, signature, bindings=metadata["bindings"]).semantic_dict() != CallSignature(description, signature, bindings=metadata["bindings"]).semantic_dict():
            raise CompilationError("compiler-emitted ABI disagrees with the canonical source interface")
        (staging / "canonical.mlir").write_text(canonical, encoding="utf-8")
        (staging / "physical.mlir").write_text(result["riscv_ir"], encoding="utf-8")
        (staging / "kernel.c").write_text(result["intrinsic_c"], encoding="utf-8")
        (staging / "kernel.h").write_text(declaration_header(metadata), encoding="utf-8")
        (staging / "dispatch.c").write_text(dispatch_source(metadata), encoding="utf-8")
        link_command = selected.link_library(staging / "kernel.c", staging / "kernel.so", target) if native_library else None
        manifest = {"kind": "weft-callable-artifact", "kernel": metadata,
                    "compilation": compilation.to_dict(),
                    "target": target.to_dict(), "options": options.to_dict(), "signature": signature,
                    "native_library": native_library, "dispatch": "weft_kernel_dispatch",
                    "compile_command": command, "link_command": link_command}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    directory, hit = _cache(cache).get_or_build(identity, builder)
    return Artifact(directory, cache_hit=hit)


def build(definition: KernelDefinition, *, target: Target, options: CompileOptions | None = None,
          toolchain: Toolchain | None = None, native_library: bool = True, cache=None) -> Artifact:
    """Compile an explicitly targeted artifact without executing it on the host."""
    return KernelSource.capture(definition).build(target=target,
                  options=CompileOptions() if options is None else options,
                  toolchain=Toolchain.from_environment() if toolchain is None else toolchain,
                  native_library=native_library, cache=cache)


def build_ir(canonical: str, *, target: Target, options: CompileOptions | None = None,
             toolchain: Toolchain | None = None, native_library: bool = True, cache=None) -> Artifact:
    """Compile Canonical Kernel IR supplied by an external frontend."""
    if not isinstance(canonical, str) or not canonical.strip():
        raise TypeError("build_ir requires nonempty Canonical Kernel IR")
    return _build(canonical, None, target=target, options=CompileOptions() if options is None else options,
                  toolchain=Toolchain.from_environment() if toolchain is None else toolchain,
                  native_library=native_library, cache=cache)


_native_targets = {}
_loaded: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_lock = threading.RLock()


def native_target(toolchain: Toolchain) -> NativeTarget:
    compiler = toolchain.compiler_path()
    identity = file_identity(compiler)
    key = (tuple(identity.items()), frozenset(os.sched_getaffinity(0)))
    with _lock:
        target = _native_targets.get(key)
        if target is None:
            target = NativeTarget.discover(toolchain)
            _native_targets[key] = target
    target.check_execution()
    return target


def compile(definition: KernelDefinition, *, options: CompileOptions | None = None,
            toolchain: Toolchain | None = None, cache=None) -> CompiledKernel:
    return compile_source(KernelSource.capture(definition), options=options, toolchain=toolchain, cache=cache)


def compile_source(source: KernelSource, *, options: CompileOptions | None = None,
                   toolchain: Toolchain | None = None, cache=None) -> CompiledKernel:
    selected = Toolchain.from_environment() if toolchain is None else toolchain
    target = native_target(selected)
    binding = CompileOptions() if options is None else options
    target.check_execution(() if binding.matrix_extension == "none" else (binding.matrix_extension,))
    artifact = source.build(target=target.build_target, options=binding,
                     toolchain=selected, cache=cache)
    key = (artifact.path, target)
    with _lock:
        loaded = _loaded.get(key)
        if loaded is None or loaded.closed:
            loaded = artifact.load(target=target, toolchain=selected)
            _loaded[key] = loaded
        return loaded


class JITKernel:
    """Lazy compilation of one explicit source/physical binding on the native CPU."""

    def __init__(self, definition: KernelDefinition, options: CompileOptions,
                 toolchain: Toolchain, cache=None) -> None:
        if not isinstance(definition, KernelDefinition):
            raise TypeError("jit expects an @weft.kernel definition")
        self.definition = definition
        self.options = options
        self.toolchain = toolchain
        self.cache = cache
        self.compiled: CompiledKernel | None = None
        self._source: KernelSource | None = None
        self._lock = threading.RLock()

    @classmethod
    def from_ir(cls, canonical, *, options=None, toolchain=None, cache=None):
        """Create the ordinary lazy native callable from a canonical frontend."""
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        source = KernelSource.from_ir(canonical, toolchain=selected)
        return cls.from_source(source, options=options, toolchain=selected, cache=cache)

    @classmethod
    def from_source(cls, source, *, options=None, toolchain=None, cache=None):
        if not isinstance(source, KernelSource):
            raise TypeError("JIT source must be a captured canonical KernelSource")
        result = cls.__new__(cls)
        result.definition = None
        result.options = source.options(CompileOptions() if options is None else options)
        result.toolchain = Toolchain.from_environment() if toolchain is None else toolchain
        result.cache = cache
        result.compiled = None
        result._source = source
        result._lock = threading.RLock()
        return result

    def materialize(self) -> CompiledKernel:
        with self._lock:
            if self.compiled is None or self.compiled.closed:
                self.compiled = compile_source(self.source, options=self.options,
                                        toolchain=self.toolchain, cache=self.cache)
            return self.compiled

    @property
    def source(self) -> KernelSource:
        with self._lock:
            if self._source is None:
                self._source = KernelSource.capture(self.definition)
            return self._source

    @property
    def call_signature(self) -> CallSignature:
        return self.source.interface(self.source.options(self.options).meta)

    def __call__(self, *args, **kwargs):
        return self.materialize()(*args, **kwargs)

    def prepare(self, *args, **kwargs):
        return self.materialize().prepare(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.materialize().run(*args, **kwargs)

    def prepare_run(self, *args, **kwargs):
        return self.materialize().prepare_run(*args, **kwargs)

    def _prepare_call(self, call):
        return self.materialize()._prepare_call(call)

    def specialize(self, **bindings: int) -> JITKernel:
        return self.with_options(self.options.specialize(**bindings))

    def with_options(self, options: CompileOptions) -> JITKernel:
        result = JITKernel.from_source(self.source, options=options, toolchain=self.toolchain, cache=self.cache)
        result.definition = self.definition
        return result

    def save(self, destination):
        return self.materialize().save(destination)

    def close(self) -> None:
        with self._lock:
            if self.compiled is not None:
                self.compiled.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def jit(definition: KernelDefinition, *, options: CompileOptions | None = None,
        toolchain: Toolchain | None = None, cache=None) -> JITKernel:
    return JITKernel(definition, CompileOptions() if options is None else options,
                     Toolchain.from_environment() if toolchain is None else toolchain, cache)


def jit_ir(canonical, *, options=None, toolchain=None, cache=None):
    return JITKernel.from_ir(canonical, options=options, toolchain=toolchain, cache=cache)
