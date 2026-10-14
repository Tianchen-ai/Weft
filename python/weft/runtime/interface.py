from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from .abi import ABIBinding, ABIPlan, Argument
from .buffer import Buffer
from .signature import CallArguments, CallSignature, freeze
from .target import Target
from .storage_guards import StorageGuards


@dataclass(frozen=True)
class BoundCall:
    binding: ABIBinding
    capture: CallArguments
    outputs: tuple[object, ...]
    guards: StorageGuards = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "guards", StorageGuards(
            (view.buffer, view.argument.writable) for view in self.binding.views))

    @property
    def arguments(self):
        return self.binding.arguments

    @property
    def buffers(self):
        return self.binding.buffers

    @property
    def spans(self):
        return self.binding.spans

    @property
    def values(self):
        return self.capture.values

    @property
    def dimensions(self):
        return self.capture.dimensions

    @property
    def explicit_shapes(self):
        return self.capture.explicit_shapes

    def validate(self) -> None:
        self.guards.validate()

    def mark_written(self) -> None:
        self.guards.mark_written()

    def result(self):
        if len(self.outputs) == 1:
            return self.outputs[0]
        return self.outputs if self.outputs else None

    def keywords(self) -> dict:
        return {**self.values, **self.explicit_shapes}


class KernelInterface(CallSignature):
    parameter_type = Argument

    def __init__(self, metadata: dict, signature: list[dict] | None = None) -> None:
        super().__init__(metadata, signature, bindings=metadata["bindings"])
        self.metadata = freeze(metadata)
        self.target = Target(metadata["march"], metadata["abi"], metadata["vlen_bits"])

    @cached_property
    def _abi(self):
        # Cross-built/source-only artifacts may contain by-value types with no
        # ctypes ABI. Inspecting, saving or linking them must remain possible.
        return ABIPlan(self.arguments, self.shape_parameters, self.target.xlen)

    @property
    def ctypes(self) -> list:
        return list(self._abi.ctypes)

    def bind(self, args: tuple, kwargs: dict, *, allocate_outputs: bool = False) -> BoundCall:
        call = self.normalize(args, kwargs, allocate_outputs=allocate_outputs)
        return self.bind_call(call)

    def bind_call(self, call: CallArguments) -> BoundCall:
        """Reuse retained source Views, then prove this artifact's storage ABI."""
        self._require_call(call, native=True)
        binding = self._abi.bind(call)
        outputs = tuple(value.owner if isinstance(value := call.values[parameter.name], Buffer) else value
                        for parameter in self.outputs)
        result = BoundCall(binding, call, outputs)
        result.validate()
        return result

    def rebind_call(self, bound: BoundCall, updates: dict) -> BoundCall:
        call = self.rebind(bound.capture, updates)
        binding = self._abi.bind(call, previous=bound.binding)
        outputs = tuple(value.owner if isinstance(value := call.values[parameter.name], Buffer) else value
                        for parameter in self.outputs)
        result = BoundCall(binding, call, outputs)
        result.validate()
        return result
