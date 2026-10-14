from __future__ import annotations

import json
from dataclasses import dataclass

from weft.language.dtypes import DType, DTypeCategory


class ValueType:
    pass


@dataclass(frozen=True, slots=True)
class ScalarType(ValueType):
    dtype: DType


@dataclass(frozen=True, slots=True)
class EncodingType(ValueType):
    family: str
    kind: str
    layout_identity: str
    parameters: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ViewType(ValueType):
    encoding: EncodingType
    shape: tuple[int, ...]
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LocalValueType(ValueType):
    element_type: ValueType
    shape: tuple[int, ...]
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SliceType(ValueType):
    encoding: EncodingType
    shape: tuple[int, ...]
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DomainType(ValueType):
    axis_name: str
    domain_id: int
    parent_domain_id: int
    axis_id: int
    relation: str
    tail: str


@dataclass(frozen=True, slots=True)
class DomainPointType(ValueType):
    domain: DomainType


def emit_type(value_type: ValueType) -> str:
    if isinstance(value_type, ScalarType):
        dtype = value_type.dtype
        if dtype.category is DTypeCategory.BOOL:
            return "i1"
        if dtype.category is DTypeCategory.INDEX:
            return "index"
        if dtype.category is DTypeCategory.INTEGER:
            prefix = "si" if dtype.signedness == "signed" else "ui"
            return f"{prefix}{dtype.bits}"
        return dtype.name
    if isinstance(value_type, EncodingType):
        return (
            "!weft_kernel.encoding<"
            f"{json.dumps(value_type.family)}, {json.dumps(value_type.kind)}, "
            f"{json.dumps(value_type.layout_identity)}, "
            f"{_emit_i64_list(value_type.parameters)}>"
        )
    if isinstance(value_type, ViewType):
        return _emit_shaped("view", value_type.encoding, value_type.shape, value_type.axes)
    if isinstance(value_type, LocalValueType):
        return _emit_shaped("value", value_type.element_type, value_type.shape, value_type.axes)
    if isinstance(value_type, SliceType):
        return _emit_shaped("slice", value_type.encoding, value_type.shape, value_type.axes)
    if isinstance(value_type, DomainType):
        return (
            "!weft_kernel.domain<"
            f"{json.dumps(value_type.axis_name)}, {value_type.domain_id}, "
            f"{value_type.parent_domain_id}, {value_type.axis_id}, "
            f"{json.dumps(value_type.relation)}, "
            f"{json.dumps(value_type.tail)}>"
        )
    if isinstance(value_type, DomainPointType):
        return f"!weft_kernel.point<{emit_type(value_type.domain)}>"
    raise TypeError(f"cannot emit unknown Weft type {value_type!r}")


def _emit_i64_list(values: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _emit_shaped(
    mnemonic: str,
    element: ValueType,
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> str:
    dimensions = "[" + ", ".join(str(dimension) for dimension in shape) + "]"
    identities = "[" + ", ".join(str(axis) for axis in axes) + "]"
    return (
        f"!weft_kernel.{mnemonic}<"
        f"{emit_type(element)}, {dimensions}, {identities}>"
    )


def shape_of(value_type: ValueType) -> tuple[int, ...]:
    if isinstance(value_type, (ViewType, LocalValueType, SliceType)):
        return value_type.shape
    return ()


def axes_of(value_type: ValueType) -> tuple[int, ...]:
    if isinstance(value_type, (ViewType, LocalValueType, SliceType)):
        return value_type.axes
    return ()


def element_type(value_type: ValueType) -> ValueType:
    if isinstance(value_type, (ViewType, SliceType)):
        return value_type.encoding
    if isinstance(value_type, LocalValueType):
        return value_type.element_type
    return value_type


def value_type(
    element: ValueType, shape: tuple[int, ...], axes: tuple[int, ...]
) -> ValueType:
    if not shape:
        return element
    return LocalValueType(element, shape, axes)


def with_element(value: ValueType, element: ValueType) -> ValueType:
    if isinstance(value, LocalValueType):
        return LocalValueType(element, value.shape, value.axes)
    return element
