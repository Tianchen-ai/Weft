"""Explicit native tuning inputs and numerical contracts for Python and CLI."""
from __future__ import annotations

import importlib
import json
import math
import runpy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .buffer import torch_module, view_metadata
from .tuning import ValidationError


@dataclass(frozen=True)
class NumericalCheck:
    """Expected output or updated state for one named argument.

    Floating-point comparison uses the caller's declared tolerances. Integer
    and Boolean results are exact. Expected arrays are independent references,
    never obtained by treating the first candidate as a numerical oracle.
    This helper does not choose a tolerance or relax it after a failed candidate.
    """

    expected: object
    rtol: float = 0.0
    atol: float = 0.0
    equal_nan: bool = False

    def __post_init__(self):
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.equal_nan) is not bool:
            raise TypeError("equal_nan must be an explicit Boolean")

    def validate(self, actual, *, name):
        torch = torch_module(actual)
        if torch is not None:
            if torch_module(self.expected) is None:
                raise TypeError(f"reference for {name} must be a Torch Tensor")
            if actual.device.type != "cpu" or self.expected.device.type != "cpu":
                raise NotImplementedError("native tuning references require CPU storage")
            if actual.dtype != self.expected.dtype or actual.shape != self.expected.shape:
                raise ValidationError(f"{name} reference shape or dtype differs from the actual output")
            floating = actual.dtype.is_floating_point
            if actual.dtype.is_complex:
                raise NotImplementedError("complex numerical contracts are unsupported")
            torch.testing.assert_close(actual, self.expected,
                                       rtol=self.rtol if floating else 0,
                                       atol=self.atol if floating else 0,
                                       equal_nan=self.equal_nan, check_stride=False)
            return
        # NumPy is optional for ordinary Weft calls. Import it only when the
        # caller elects this array-reference helper or archive input format.
        import numpy as np

        if not isinstance(actual, np.ndarray) or not isinstance(self.expected, np.ndarray):
            raise TypeError(f"reference checks for {name} require NumPy arrays or CPU Torch tensors")
        if actual.shape != self.expected.shape or actual.dtype != self.expected.dtype:
            raise ValidationError(f"{name} reference shape or dtype differs from the actual output")
        if actual.dtype.kind in "iub":
            np.testing.assert_array_equal(actual, self.expected, err_msg=name)
        elif actual.dtype.kind == "f":
            np.testing.assert_allclose(actual, self.expected, rtol=self.rtol, atol=self.atol,
                                       equal_nan=self.equal_nan, err_msg=name)
        else:
            raise NotImplementedError(f"reference dtype {actual.dtype} is unsupported")

    def contract(self):
        expected = self.expected
        return {"dtype": str(expected.dtype), "shape": list(expected.shape),
                "rtol": self.rtol, "atol": self.atol, "equal_nan": self.equal_nan}


class TuningWorkload:
    """Caller-owned arguments and an explicit numerical validation contract.

    A workload can be returned by a CLI factory or passed to run directly. All
    output and readwrite buffers are provided explicitly: constructing a workload
    does not compile, allocate outputs, discover hardware, or run a candidate.
    Trial writes remain isolated by the ordinary native autotuner.
    """

    def __init__(self, arguments, *, validator=None, validation_key=None, checks=None):
        if not isinstance(arguments, Mapping) or not arguments:
            raise TypeError("workload arguments must be a nonempty mapping")
        if any(not isinstance(name, str) or not name for name in arguments):
            raise ValueError("workload argument names must be nonempty strings")
        if checks is not None and validator is not None:
            raise ValueError("choose reference checks or a custom numerical validator")
        checks = {} if checks is None else dict(checks)
        if any(name not in arguments for name in checks):
            raise ValueError("reference checks must name provided output or state arguments")
        if any(not isinstance(check, NumericalCheck) for check in checks.values()):
            raise TypeError("checks must map argument names to NumericalCheck objects")
        if validator is not None and not callable(validator):
            raise TypeError("workload validator must be callable")
        if (bool(checks) or validator is not None) != (validation_key is not None):
            raise ValueError("numerical validation requires an explicit validation_key and vice versa")
        if validation_key is not None and (not isinstance(validation_key, str) or not validation_key):
            raise ValueError("validation_key must identify the author's numerical contract")
        self.arguments = MappingProxyType(dict(arguments))
        self.checks = MappingProxyType(checks)
        self.validation_key = validation_key
        self._validator = validator

    @property
    def validator(self):
        if self.checks:
            return self._check
        return self._validator

    def _check(self, original, trial):
        for name, check in self.checks.items():
            check.validate(trial[name], name=name)

    def check_interface(self, interface):
        """Ensure references cover the complete mutable result of this program."""
        if self.checks:
            mutable = {parameter.name for parameter in interface.arguments if parameter.writable}
            if set(self.checks) != mutable:
                raise ValueError(f"reference checks must cover exactly the writable arguments {sorted(mutable)}")
        # Metadata-only normalization retains stride, dtype and source-specialized
        # shape checks. Native binding subsequently checks actual memory aliases.
        call = interface.normalize((), dict(self.arguments), allow_outputs=False, metadata_only=True)
        for name, check in self.checks.items():
            expected = view_metadata(check.expected)
            actual = call.views[name]
            if expected.encoding != actual.encoding or expected.shape != actual.shape:
                raise ValueError(f"{name} reference shape or dtype disagrees with its invocation metadata")

    def contract_key(self):
        if not self.checks:
            return self.validation_key
        # Tolerances and expected shape/dtype are part of the contract even if
        # the caller reuses a human-readable validation key accidentally. Tensor
        # contents remain ordinary inputs and are never copied into cache keys.
        return json.dumps({"name": self.validation_key,
                           "checks": {name: check.contract() for name, check in self.checks.items()}},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)

    def autotune(self, definition, configs=None, *, key=(), policy=None, toolchain=None, cache=None):
        """Bind this workload's validator to a JIT or a precompiled collection."""
        from .autotune import autotune, autotune_ir
        from .candidates import BuiltCandidates

        arguments = dict(key=key, policy=policy, validator=self.validator,
                         validation_key=self.contract_key(), toolchain=toolchain, cache=cache)
        if isinstance(definition, BuiltCandidates):
            if configs is not None:
                raise ValueError("prebuilt candidates already contain their complete configurations")
            tuned = definition.autotune(**arguments)
        else:
            if configs is None:
                raise ValueError("JIT tuning requires explicit finite configurations")
            tuned = (autotune_ir(definition, configs, **arguments) if isinstance(definition, str)
                     else autotune(definition, configs, **arguments))
        self.check_interface(tuned.call_signature)
        return tuned

    def run(self, definition, configs=None, *, key=(), policy=None, toolchain=None,
            cache=None, refresh=False, selection=None, artifact=None):
        """Tune without executing on caller storage; optionally export the winner.

        Saving an artifact selects exactly the winner for this workload. It does
        not export the last winner of some other shape or erase source provenance.
        A returned TuningResult remains usable after loaded handles are closed.
        """
        with self.autotune(definition, configs, key=key, policy=policy,
                           toolchain=toolchain, cache=cache) as tuned:
            result = tuned.tune(**self.arguments, refresh=refresh)
            if artifact is not None:
                with tuned.fixed() as fixed:
                    fixed.save(artifact)
            if selection is not None:
                result.save(selection)
            return result

    @classmethod
    def from_archive(cls, path, *, scalars=None, expected=None, rtol=0.0, atol=0.0,
                     equal_nan=False, validation_key=None):
        """Load explicit dense arrays from NPZ with pickle disabled.

        NPZ represents independent dense arrays, not arbitrary alias/stride
        graphs. A Python workload factory is required for overlapping views,
        noncontiguous tensors, packed Buffer storage, or computed references.
        This method does not reinterpret one dtype as another or infer outputs.
        """
        import numpy as np

        def read(filename):
            with np.load(filename, allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    raise ValueError("array archive contains duplicate argument names")
                result = {name: archive[name] for name in archive.files}
            for name, value in result.items():
                if not value.dtype.isnative or value.dtype.kind not in "fiub":
                    raise NotImplementedError(f"archive argument {name} requires a native real numeric dtype")
            return result

        arguments = read(path)
        scalars = {} if scalars is None else dict(scalars)
        if arguments.keys() & scalars.keys():
            raise ValueError("scalar arguments cannot replace an array archive argument")
        if any(type(value) not in (bool, int, float) for value in scalars.values()):
            raise TypeError("scalar arguments must be JSON numbers or Booleans")
        arguments.update(scalars)
        references = {} if expected is None else read(expected)
        checks = {name: NumericalCheck(value, rtol=rtol, atol=atol, equal_nan=equal_nan)
                  for name, value in references.items()}
        return cls(arguments, checks=checks, validation_key=validation_key)


def load_workload(entry):
    """Call an explicitly named zero-argument Python workload factory."""
    location, separator, symbol = entry.rpartition(":")
    if not separator or not location or not symbol or not symbol.isidentifier():
        raise ValueError("workload must be module:factory or file.py:factory")
    namespace = runpy.run_path(location) if location.endswith(".py") else vars(importlib.import_module(location))
    if symbol not in namespace or not callable(namespace[symbol]):
        raise ValueError(f"workload entry {entry!r} is not a callable factory")
    workload = namespace[symbol]()
    if not isinstance(workload, TuningWorkload):
        raise TypeError("workload factory must return a TuningWorkload")
    return workload
