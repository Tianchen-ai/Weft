from __future__ import annotations

from dataclasses import dataclass

from .dtypes import DType


@dataclass(frozen=True, slots=True)
class LayoutAtom:
    category: str
    value: str


class _BitOrder:
    lsb_first = LayoutAtom("bit_order", "lsb_first")
    msb_first = LayoutAtom("bit_order", "msb_first")


class _ByteOrder:
    little = LayoutAtom("byte_order", "little")
    big = LayoutAtom("byte_order", "big")


bitorder = _BitOrder()
byteorder = _ByteOrder()
lo_first = "lo_first"
hi_first = "hi_first"


@dataclass(frozen=True, slots=True)
class FieldLayoutSpec:
    kind: str
    size: int
    order: str = ""
    fields: int = 0
    low_bits: int = 0


def grouped(*, elements: int) -> FieldLayoutSpec:
    if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
        raise TypeError("grouped expects a positive logical element count")
    return FieldLayoutSpec("grouped", elements)


def bit_layers(*, elements: int, order: str) -> FieldLayoutSpec:
    if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
        raise TypeError("bit_layers expects a positive logical layer extent")
    if order not in {lo_first, hi_first}:
        raise TypeError("bit_layers order must be lo_first or hi_first")
    return FieldLayoutSpec("layered", elements, order)


def pack_fields(*, group: int, fields: int, low_bits: int, order: str) -> FieldLayoutSpec:
    for name, value in {"group": group, "fields": fields, "low_bits": low_bits}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"pack_fields {name} must be a positive integer")
    if order not in {lo_first, hi_first}:
        raise TypeError("pack_fields order must be lo_first or hi_first")
    return FieldLayoutSpec("joined", group, order, fields, low_bits)


@dataclass(frozen=True, slots=True)
class PaddingSpec:
    bytes: int
    value: int = 0


def padding(bytes: int, value: int = 0) -> PaddingSpec:
    if isinstance(bytes, bool) or not isinstance(bytes, int) or bytes <= 0:
        raise TypeError("padding expects a positive byte count")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise TypeError("padding fill must be a byte")
    return PaddingSpec(bytes, value)


@dataclass(frozen=True, slots=True)
class ArraySpec:
    dtype: DType
    shape: tuple[object, ...]
    layouts: tuple[FieldLayoutSpec, ...] = ()

    def __matmul__(self, layout: object) -> ArraySpec:
        if not isinstance(layout, FieldLayoutSpec):
            raise TypeError("encoding field @ annotation must be a field layout")
        return ArraySpec(self.dtype, self.shape, self.layouts + (layout,))


@dataclass(frozen=True, slots=True)
class ViewSpec:
    encoding: object
    shape: tuple[object, ...]


class View:
    @classmethod
    def __class_getitem__(cls, parameters: object) -> ViewSpec:
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError("View expects View[Encoding, shape]")
        encoding, shape = parameters
        return ViewSpec(encoding, shape if isinstance(shape, tuple) else (shape,))


class static:
    """Marker for a build-time parameter in a derived Encoding definition."""

    @classmethod
    def __class_getitem__(cls, parameter: object) -> object:
        return parameter


@dataclass(frozen=True, slots=True)
class AutoSpec:
    choices: tuple[object, ...]

    @property
    def spelling(self) -> str:
        return "|".join(str(choice) for choice in self.choices)


def auto(*choices: object) -> AutoSpec:
    if not choices:
        raise TypeError("auto expects at least one candidate or one symbolic parameter")
    return AutoSpec(tuple(choices))
