"""Storage provenance and element offsets in Triton's value/SSA protocol.

A pointer tensor is a scalar storage origin plus an ordinary integer tensor.
Both components are explicit SSA operands at control joins and memory effects;
no address vector or implicit memory access is hidden in Python state.
"""
from dataclasses import dataclass

from triton.language import core as tl
from triton.language.weft import dtypes
from .types import ScalarType, value_type, shape_of
from .ir import Value


@dataclass(frozen=True)
class Address:
    base: Value
    offset: Value

    @property
    def type(self):
        from .types import axes_of
        return value_type(self.base.type, shape_of(self.offset.type), axes_of(self.offset.type))

    def get_loc(self):
        return self.offset.get_loc()

    def set_loc(self, location):
        self.offset.set_loc(location)


class AddressSSA:
    def _flatten_ir_types(self, builder, types):
        types.append(self.scalar.to_ir(builder))
        offset = ScalarType(dtypes.i64)
        types.append(builder.get_block_ty(offset, self.shape) if self.is_block() else offset)

    def _unflatten_ir(self, handles, cursor):
        return CanonicalTensor(Address(handles[cursor], handles[cursor + 1]), self), cursor + 2


class AddressScalarType(AddressSSA, tl.pointer_type):
    pass


class AddressBlockType(AddressSSA, tl.block_type):
    pass


class CanonicalTensor(tl.tensor):
    def __new__(cls, handle, ty):
        if not isinstance(handle, Address):
            return tl.tensor(handle, ty)
        return super().__new__(cls)

    def __init__(self, handle, ty):
        if isinstance(handle, Address):
            scalar = ty.scalar
            pointer = AddressScalarType(scalar.element_ty, scalar.address_space, scalar.const)
            ty = AddressBlockType(pointer, ty.shape) if ty.is_block() else pointer
        super().__init__(handle, ty)

    def _flatten_ir(self, handles):
        if isinstance(self.handle, Address):
            handles.extend((self.handle.base, self.handle.offset))
        else:
            super()._flatten_ir(handles)


def entry_pointer(base, builder, source_type=None):
    from .values import logical_dtype
    dtype = logical_dtype(base.type.element_type.dtype)
    return CanonicalTensor(Address(base, builder.get_int64(0)), source_type or tl.pointer_type(dtype))
