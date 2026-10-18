from __future__ import annotations

import ctypes
import math
import numbers
from dataclasses import dataclass

from .buffer import Buffer
from .errors import CompilationError
from .signature import Parameter


_SCALARS = {"i1": ctypes.c_bool, "i8": ctypes.c_int8, "u8": ctypes.c_uint8,
            "i16": ctypes.c_int16, "u16": ctypes.c_uint16, "i32": ctypes.c_int32,
            "u32": ctypes.c_uint32, "i64": ctypes.c_int64, "u64": ctypes.c_uint64,
            "f32": ctypes.c_float, "f64": ctypes.c_double, "index": ctypes.c_size_t}


@dataclass(frozen=True)
class Argument(Parameter):
    storage_bytes: int
    record_elements: int
    interleave_rows: int
    alignment: int

    @classmethod
    def from_dict(cls, value: dict):
        result = super().from_dict(value)
        if result.kind in ("view", "pointer") and min(result.storage_bytes, result.record_elements, result.alignment) <= 0:
            raise CompilationError(f"incomplete Encoding ABI for {result.name}")
        return result

    @property
    def ctype(self):
        if self.kind in ("view", "pointer"):
            return ctypes.c_void_p
        if self.dtype not in _SCALARS:
            raise NotImplementedError(f"Python by-value ABI does not support {self.dtype}; use the C declaration")
        return _SCALARS[self.dtype]


class ScalarABI:
    """Immutable scalar conversion facts resolved when the artifact is loaded."""

    def __init__(self, argument: Argument, xlen: int):
        self.name = argument.name
        self.dtype = argument.dtype
        self.ctype = argument.ctype
        self.floating = self.dtype.startswith("f")
        if not self.floating:
            bits = xlen if self.dtype == "index" else int(self.dtype[1:])
            unsigned = self.dtype.startswith("u") or self.dtype in ("index", "i1")
            self.minimum, self.maximum = (0, 1 << bits) if unsigned else (-(1 << (bits - 1)), 1 << (bits - 1))

    def convert(self, value):
        if self.floating:
            if not isinstance(value, numbers.Real):
                raise TypeError(f"{self.name} requires a real scalar")
            result = self.ctype(float(value))
            if math.isfinite(value) and not math.isfinite(result.value):
                raise OverflowError(f"{self.name} cannot be represented by {self.dtype}")
            return result
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"{self.name} requires an integer scalar")
        value = int(value)
        if not self.minimum <= value < self.maximum:
            raise OverflowError(f"{self.name}={value} does not fit {self.dtype}")
        return self.ctype(value)


@dataclass(frozen=True)
class ViewBinding:
    argument: Argument
    buffer: Buffer
    span: int
    pointer: ctypes.c_void_p
    strides: tuple


class ViewABI:
    """Encoding and pointer legality independent of Python call syntax."""

    def __init__(self, argument: Argument, xlen: int):
        self.argument = argument
        self.name = argument.name
        self.index_limit = 1 << xlen
        self.offset_limit = 1 << (xlen - 1)
        self.strided = argument.strided
        self.writable = argument.writable
        self.storage_bytes = argument.storage_bytes
        self.record_elements = argument.record_elements
        self.interleave_rows = argument.interleave_rows
        self.alignment = argument.alignment

    def bind(self, buffer: Buffer) -> ViewBinding:
        metadata = buffer.metadata
        shape, steps = metadata.shape, metadata.strides
        if any(extent >= self.index_limit for extent in shape):
            raise ValueError(f"{self.name} extent exceeds the target ABI")
        if shape and shape[-1] % self.record_elements:
            raise ValueError(f"{self.name} does not contain complete Encoding records")
        if self.interleave_rows and (len(shape) != 2 or shape[0] % self.interleave_rows):
            raise ValueError(f"{self.name} requires complete declared row cohorts")
        if self.argument.kind == "pointer":
            required = buffer.byte_span()
            strides = ()
        elif self.strided:
            required = buffer.byte_span()
            if self.writable:
                buffer.require_nonoverlapping()
            if any(stride * self.storage_bytes >= self.offset_limit for stride in steps):
                raise ValueError(f"{self.name} stride exceeds the signed pointer offset range")
            strides = tuple(ctypes.c_size_t(stride) for stride in steps)
        else:
            required = math.prod(shape) // self.record_elements * self.storage_bytes
            strides = ()
        if required > buffer.nbytes or required >= self.offset_limit:
            raise ValueError(f"{self.name} storage is shorter than its logical View or exceeds the ABI")
        if required and (not buffer.address or buffer.address % self.alignment):
            raise ValueError(f"{self.name} does not meet its Encoding alignment")
        if self.writable and metadata.readonly:
            raise TypeError(f"{self.name} requires writable storage")
        return ViewBinding(self.argument, buffer, required, ctypes.c_void_p(buffer.address), strides)


@dataclass(frozen=True)
class ABIBinding:
    arguments: tuple
    views: tuple[ViewBinding, ...]

    @property
    def buffers(self):
        return tuple((view.argument, view.buffer) for view in self.views)

    @property
    def spans(self):
        return tuple((view.argument, view.buffer, view.span) for view in self.views)


class ABIPlan:
    """Fixed marshalling and validation derived from the compiler's typed ABI.

    This host binding plan has no kernel layout, instruction, scheduling or
    source specialization authority. It still proves each invocation's storage.
    """

    def __init__(self, arguments, shape_parameters, xlen):
        self.arguments = arguments
        self.shapes = tuple(shape_parameters)
        self.index_limit = 1 << xlen
        self.scalars = tuple((index, ScalarABI(argument, xlen)) for index, argument in enumerate(arguments)
                             if argument.kind == "scalar")
        self.views = tuple((index, ViewABI(argument, xlen)) for index, argument in enumerate(arguments)
                           if argument.kind in ("view", "pointer"))
        self.alias_pairs = tuple((left, right) for right, (_, current) in enumerate(self.views)
                                 for left, (_, other) in enumerate(self.views[:right])
                                 if current.argument.alias_set != other.argument.alias_set)
        strides = sum(len(argument.shape) for argument in arguments if argument.strided)
        self.ctypes = tuple(argument.ctype for argument in arguments) + (ctypes.c_size_t,) * (len(self.shapes) + strides)

    def bind(self, call, previous: ABIBinding | None = None) -> ABIBinding:
        values = [None] * len(self.arguments)
        for index, scalar in self.scalars:
            values[index] = scalar.convert(call.values[scalar.name])
        views = []
        for view_index, (index, view) in enumerate(self.views):
            if view.name not in call.buffers:
                raise TypeError(f"missing kernel View argument: {view.name}")
            buffer = call.buffers[view.name]
            if previous is not None and previous.views[view_index].buffer is buffer:
                bound = previous.views[view_index]
            else:
                bound = view.bind(buffer)
            views.append(bound)
            values[index] = bound.pointer
        for left, right in self.alias_pairs:
            other, current = views[left], views[right]
            if current.span and other.span and (
                    current.buffer.address < other.buffer.address + other.span and
                    other.buffer.address < current.buffer.address + current.span):
                raise ValueError(f"{current.argument.name} aliases {other.argument.name} across declared alias groups")
        for name in self.shapes:
            if name not in call.dimensions:
                raise ValueError(f"kernel call needs explicit shape argument: {name}")
            dimension = call.dimensions[name]
            if type(dimension) is not int or not 0 <= dimension < self.index_limit:
                raise ValueError(f"shape {name} must fit the target unsigned index ABI")
            values.append(ctypes.c_size_t(dimension))
        for view in views:
            values.extend(view.strides)
        return ABIBinding(tuple(values), tuple(views))
