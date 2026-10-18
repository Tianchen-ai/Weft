"""Canonical values participating in Triton's common frontend protocols.

Positional Triton tensors keep Triton's types. Explicit Weft values retain
their logical axes and finite-width types across SSA joins and helper calls.
"""
from __future__ import annotations

from collections.abc import MutableMapping
import math

from triton.language import core as tl
from triton.language.weft.dtypes import DTypeCategory

from .ir import Value
from .types import LocalValueType, ScalarType, axes_of, element_type, shape_of


class LogicalType(tl.base_type):
    def __init__(self, canonical):
        self.canonical = canonical
        self.shape = shape_of(canonical)
        self.numel = math.prod(self.shape) if all(n > 0 for n in self.shape) else None

    def __eq__(self, other):
        return isinstance(other, LogicalType) and self.canonical == other.canonical

    def __hash__(self):
        return hash(self.canonical)

    def __repr__(self):
        return repr(self.canonical)

    def is_block(self):
        return isinstance(self.canonical, LocalValueType)

    def is_ptr(self):
        return False

    @property
    def scalar(self):
        element = element_type(self.canonical)
        if isinstance(element, ScalarType):
            return logical_dtype(element.dtype)
        return LogicalType(element)

    def to_ir(self, builder):
        return self.canonical

    def _flatten_ir_types(self, builder, types):
        types.append(self.canonical)

    def _unflatten_ir(self, handles, cursor):
        return wrap(handles[cursor]), cursor + 1

    def mangle(self):
        return repr(self.canonical)


class FiniteDType(tl.dtype):
    """The dtype protocol for explicit, non-Triton integer widths and index."""
    def __init__(self, dtype):
        self.canonical_dtype = dtype
        self.name = dtype.name
        self.primitive_bitwidth = 64 if dtype.category is DTypeCategory.INDEX else dtype.bits
        self.int_bitwidth = self.primitive_bitwidth
        self.int_signedness = (self.SIGNEDNESS.SIGNED if dtype.signedness == "signed"
                               else self.SIGNEDNESS.UNSIGNED)
        self.itemsize = (self.primitive_bitwidth + 7) // 8

    def is_int(self):
        return True

    def is_int_signed(self):
        return self.int_signedness == self.SIGNEDNESS.SIGNED

    def is_int_unsigned(self):
        return not self.is_int_signed()

    def to_ir(self, builder):
        return ScalarType(self.canonical_dtype)


def logical_dtype(dtype):
    from triton.language.weft import dtypes
    types = {dtypes.f16: tl.float16, dtypes.bf16: tl.bfloat16,
             dtypes.f32: tl.float32, dtypes.f64: tl.float64,
             dtypes.i1: tl.int1, dtypes.i8: tl.int8, dtypes.i16: tl.int16,
             dtypes.i32: tl.int32, dtypes.i64: tl.int64,
             dtypes.u8: tl.uint8, dtypes.u16: tl.uint16,
             dtypes.u32: tl.uint32, dtypes.u64: tl.uint64}
    if dtype in types:
        return types[dtype]
    if dtype.category in (DTypeCategory.INTEGER, DTypeCategory.INDEX):
        return FiniteDType(dtype)
    raise NotImplementedError(f"Triton value protocol for {dtype}")


class LogicalTensor(tl.tensor):
    def __init__(self, handle):
        super().__init__(handle, LogicalType(handle.type))


class LogicalObject(tl.base_value):
    def __init__(self, handle):
        self.handle = handle
        self.type = LogicalType(handle.type)

    def _flatten_ir(self, handles):
        handles.append(self.handle)

    def _set_name(self, builder, name):
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))


def wrap(value):
    if isinstance(value, Value):
        if isinstance(element_type(value.type), ScalarType):
            return LogicalTensor(value)
        return LogicalObject(value)
    if isinstance(value, tuple):
        return tl.tuple([wrap(item) for item in value])
    if value is None or isinstance(value, (bool, int, float, str)):
        return tl.constexpr(value)
    return value


def unwrap(value):
    if isinstance(value, (tl.tensor, LogicalObject)):
        return value.handle
    if isinstance(value, tl.constexpr):
        return value.value
    if isinstance(value, tl.tuple):
        return tuple(unwrap(item) for item in value.values)
    return value


class CanonicalScope(MutableMapping):
    """A view of the common CodeGenerator scope, not a second symbol table."""
    def __init__(self, generator):
        self.generator = generator
        self.scope = generator.lscope

    def __getitem__(self, name):
        return unwrap(self.scope[name])

    def __setitem__(self, name, value):
        wrapped = wrap(value)
        self.scope[name] = wrapped
        if self.generator.lscope is self.scope:
            self.generator.local_defs[name] = wrapped

    def __delitem__(self, name):
        del self.scope[name]

    def __iter__(self):
        return iter(self.scope)

    def __len__(self):
        return len(self.scope)

    def copy(self):
        return dict(self)
