from __future__ import annotations

import ctypes
import math
import os
import sys
from dataclasses import dataclass

from .buffer import Buffer, torch_module
from .abi import ABIPlan
from .native import NativeSignature


def workload_key(function, args, kwargs):
    """Keep dynamic View geometry and actual aliases in Triton's tuning key.

    Absolute allocation addresses are excluded. Only overlap offsets and the
    alignment class affect a choice, so equivalent fresh inputs reuse it.
    """
    supplied = function.signature.bind_partial(
        *args, **{name: value for name, value in kwargs.items() if name in function.arg_names})
    supplied.apply_defaults()
    views = []
    constants = {parameter.name for parameter in function.params if parameter.is_constexpr}
    for name, value in supplied.arguments.items():
        if name in constants or value is None or isinstance(value, (bool, int, float, str)):
            continue
        buffer = value if isinstance(value, Buffer) else Buffer(value)
        views.append((name, buffer))
    geometry = tuple((name, buffer.encoding, buffer.shape, buffer.strides, buffer.address % 256)
                     for name, buffer in views)
    # An opaque packed Encoding has no element byte stride. Its supplied byte
    # capacity is a conservative alias partition for the tuning key; each
    # candidate's typed ABI still proves its exact record/cohort span.
    spans = [buffer.byte_span() if buffer.encoding.startswith("dense.") else buffer.nbytes
             for _, buffer in views]
    aliases = []
    for index, (left_name, left) in enumerate(views):
        for right_index in range(index + 1, len(views)):
            right_name, right = views[right_index]
            if (spans[index] and spans[right_index]
                    and max(left.address, right.address) < min(left.address + spans[index],
                                                              right.address + spans[right_index])):
                aliases.append((left_name, right_name, right.address - left.address))
    from triton.runtime.driver import driver
    target = driver.for_model(function.model).get_current_target()
    environment = tuple((key, repr(kwargs[key])) for key in sorted(kwargs)
                        if key not in function.arg_names and key not in ("grid", "warmup"))
    grid = kwargs.get("grid")
    launch = tuple(grid) if isinstance(grid, (tuple, list)) else None
    return (geometry, tuple(aliases), str(target), tuple(sorted(os.sched_getaffinity(0))),
            os.environ.get("OMP_NUM_THREADS"), environment, launch)


class TrialCall:
    def __init__(self, kernel, supplied):
        signature = NativeSignature(kernel.metadata.kernel_abi, bindings=kernel.metadata.kernel_abi["bindings"])
        values = {argument.name: supplied[argument.name] for argument in signature.arguments
                  if not argument.name.startswith("__triton_")}
        values.update({argument.name: 1 if "_grid_" in argument.name else 0
                       for argument in signature.arguments if argument.name.startswith("__triton_")})
        call = signature.normalize((), values)
        binding = ABIPlan(signature.arguments, signature.shape_parameters, 64).bind(call)
        self.values, self.explicit_shapes, self.dimensions = call.values, call.explicit_shapes, call.dimensions
        self.spans = binding.spans

    def validate(self):
        for parameter, buffer, _ in self.spans:
            buffer.validate(writable=parameter.writable)


@dataclass
class _StorageGroup:
    begin: int
    end: int
    members: list


class _PrivateStorage:
    def __init__(self, begin: int, size: int, alignment: int):
        self.bytes = bytearray(size + 2 * alignment)
        self.buffer = Buffer(self.bytes)
        self.aligned_offset = (-self.buffer.address) % alignment
        self.payload_offset = self.aligned_offset + begin % alignment
        self.address = self.buffer.address + self.payload_offset
        self.size = size
        self._torch_root = None

    def tensor(self, prototype, offset: int):
        torch = torch_module(prototype)
        if self._torch_root is None:
            self._torch_root = torch.frombuffer(memoryview(self.bytes)[self.aligned_offset:], dtype=torch.uint8)
        byte_offset = self.payload_offset - self.aligned_offset + offset
        if byte_offset % prototype.element_size():
            raise ValueError("private storage cannot preserve the Tensor element alignment")
        return torch.empty(0, dtype=prototype.dtype, device="cpu").set_(
            self._torch_root.untyped_storage(), byte_offset // prototype.element_size(),
            tuple(prototype.shape), tuple(prototype.stride()))

    def view(self, prototype, buffer: Buffer, offset: int, required: int):
        if torch_module(prototype) is not None:
            return self.tensor(prototype, offset)
        numpy = sys.modules.get("numpy")
        if numpy is not None and isinstance(prototype, numpy.ndarray):
            return numpy.ndarray(prototype.shape, dtype=prototype.dtype, buffer=self.bytes,
                                 offset=self.payload_offset + offset, strides=prototype.strides)
        raw = memoryview(self.bytes)[self.payload_offset + offset:self.payload_offset + offset + required]
        return Buffer(raw, shape=buffer.shape, encoding=buffer.encoding,
                      strides=buffer.strides if buffer.encoding.startswith("dense.") else None)


class TuningState:
    """Trial storage with the caller's actual byte aliases, strides and alignment.

    The caller is never used as mutable trial storage. Read-only parameters
    sharing bytes with a writable parameter are rebound into the same private
    allocation. Initial copies visit only accessible logical byte ranges;
    resetting touches private storage, including its own padding, only.
    """

    def __init__(self, bound: TrialCall, *, max_bytes: int = 256 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("tuning state budget must be a positive byte count")
        bound.validate()
        self.values = dict(bound.values)
        self.explicit_shapes = dict(bound.explicit_shapes)
        self.dimensions = dict(bound.dimensions)
        self._original = bound
        self._copies: list[tuple[_PrivateStorage, _PrivateStorage]] = []
        self.bytes = 0
        spans = sorted(((buffer.address, buffer.address + size, parameter, buffer, size)
                        for parameter, buffer, size in bound.spans if size), key=lambda item: item[0])
        groups = []
        for begin, end, parameter, buffer, size in spans:
            if groups and begin < groups[-1].end:
                groups[-1].end = max(groups[-1].end, end)
                groups[-1].members.append((parameter, buffer, size))
            else:
                groups.append(_StorageGroup(begin, end, [(parameter, buffer, size)]))
        allocations = []
        for group in groups:
            if not any(parameter.writable for parameter, _, _ in group.members):
                continue
            alignment = math.lcm(256, *(parameter.alignment for parameter, _, _ in group.members))
            size = group.end - group.begin
            self.bytes += 2 * (size + 2 * alignment)
            allocations.append((group, alignment, size))
        # Empty outputs have independent owners too. Their alignment padding is
        # real allocated memory even though no logical element is copied. Count
        # it before allocating any group, so the budget bounds the entire shared
        # session rather than only its nonempty View payloads.
        empty = [(parameter, buffer) for parameter, buffer, required in bound.spans
                 if not required and parameter.writable]
        self.bytes += sum(2 * math.lcm(256, parameter.alignment) for parameter, _ in empty)
        if self.bytes > max_bytes:
            raise ValueError(f"tuning state needs {self.bytes} bytes, above its {max_bytes}-byte budget")
        for group, alignment, size in allocations:
            snapshot = _PrivateStorage(group.begin, size, alignment)
            private = _PrivateStorage(group.begin, size, alignment)
            for parameter, buffer, required in group.members:
                offset = buffer.address - group.begin
                for start, length in buffer.byte_regions(required):
                    ctypes.memmove(snapshot.address + offset + start, buffer.address + start, length)
                self.values[parameter.name] = private.view(bound.values[parameter.name], buffer, offset, required)
            self._copies.append((snapshot, private))
        # Empty outputs still get independent owners, without an address-based
        # alias claim about the shared null pointer used by many allocators.
        for parameter, buffer in empty:
            storage = _PrivateStorage(0, 0, math.lcm(256, parameter.alignment))
            self.values[parameter.name] = storage.view(bound.values[parameter.name], buffer, 0, 0)
        self.reset()

    def reset(self) -> None:
        if self._original is None:
            raise RuntimeError("tuning state is closed")
        self._original.validate()
        for snapshot, private in self._copies:
            ctypes.memmove(private.address, snapshot.address, private.size)

    def keywords(self) -> dict:
        if self._original is None:
            raise RuntimeError("tuning state is closed")
        return {**self.values, **self.explicit_shapes}

    def close(self) -> None:
        self.values.clear()
        self._copies.clear()
        self._original = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
