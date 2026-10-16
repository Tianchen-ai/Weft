from __future__ import annotations

import ast
import inspect
import json
import math
import sys
from dataclasses import dataclass
from types import FunctionType
from typing import Sequence

from weft.api import (
    DerivedEncodingDefinition,
    EncodingDefinition,
    InlineDefinition,
    KernelDefinition,
    OverloadSet,
)
from weft.diagnostics import FrontendError, SourceLocation
from weft.language.annotations import (
    AutoSpec,
    View,
    auto,
    bit_layers,
    bitorder,
    byteorder,
    grouped,
    pack_fields,
    padding,
)
from weft.language.builtins import Intrinsic, LevelConstructor
from weft.language.dtypes import (
    DType,
    DTypeCategory,
    f32,
    f64,
    i1,
    i8,
    i16,
    i32,
    i64,
    index,
    u8,
    u16,
    u32,
    u64,
)

from .ir import IRBuilder, Operation, Region, Value
from .source import FunctionSource
from .types import (
    DomainPointType,
    DomainType,
    EncodingType,
    LocalValueType,
    ScalarType,
    SliceType,
    ValueType,
    ViewType,
    axes_of,
    element_type,
    emit_type,
    shape_of,
    value_type,
    with_element,
)


def _string(value: str) -> str:
    return json.dumps(value)


def _strings(values: Sequence[str]) -> str:
    return "[" + ", ".join(_string(value) for value in values) + "]"


def _i64_array(values: Sequence[int]) -> str:
    if not values:
        return "array<i64>"
    return "array<i64: " + ", ".join(str(value) for value in values) + ">"


def _type_array(values: Sequence[ValueType]) -> str:
    return "[" + ", ".join(emit_type(value) for value in values) + "]"


def _layout_array(fields: Sequence[_FieldInfo]) -> str:
    rendered: list[str] = []
    for field in fields:
        atoms: list[str] = []
        for layout in field.layouts:
            values = [f"kind = {_string(layout.kind)}"]
            if layout.size:
                values.append(f"size = {layout.size} : i64")
            if layout.order:
                values.append(f"order = {_string(layout.order)}")
            if layout.fields:
                values.append(f"fields = {layout.fields} : i64")
            if layout.low_bits:
                values.append(f"low_bits = {layout.low_bits} : i64")
            if layout.role >= 0:
                values.append(f"role = {layout.role} : i64")
            atoms.append("{" + ", ".join(values) + "}")
        rendered.append("[" + ", ".join(atoms) + "]")
    return "[" + ", ".join(rendered) + "]"


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Name):
            self.names.add(node.value.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return


class _LoadedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return


def _assigned_names(statements: Sequence[ast.stmt]) -> set[str]:
    visitor = _AssignedNames()
    for statement in statements:
        visitor.visit(statement)
    return visitor.names


def _loaded_names(statements: Sequence[ast.stmt]) -> set[str]:
    visitor = _LoadedNames()
    for statement in statements:
        visitor.visit(statement)
    return visitor.names


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return {target.value.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for element in target.elts:
            result.update(_target_names(element))
        return result
    return set()


def _statement_uses_defs(statement: ast.stmt) -> tuple[set[str], set[str]]:
    uses = _loaded_names([statement])
    if isinstance(statement, ast.Assign):
        definitions: set[str] = set()
        for target in statement.targets:
            definitions.update(_target_names(target))
        return uses - definitions, definitions
    if isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        definitions = _target_names(statement.target)
        return uses, definitions
    if isinstance(statement, ast.For):
        definitions = _assigned_names(statement.body) | _target_names(statement.target)
        return uses, definitions
    if isinstance(statement, ast.While):
        return uses, _assigned_names(statement.body)
    if isinstance(statement, ast.If):
        return uses, _assigned_names(statement.body) & _assigned_names(statement.orelse)
    if isinstance(statement, ast.With):
        return uses, _assigned_names(statement.body)
    return uses, _assigned_names([statement])


def _names_needed_before_definition(statements: Sequence[ast.stmt]) -> set[str]:
    needed: set[str] = set()
    defined: set[str] = set()
    for statement in statements:
        uses, definitions = _statement_uses_defs(statement)
        needed.update(uses - defined)
        defined.update(definitions)
    return needed


@dataclass(frozen=True, slots=True)
class _LayoutInfo:
    kind: str
    size: int = 0
    order: str = ""
    fields: int = 0
    low_bits: int = 0
    role: int = -1


@dataclass(frozen=True, slots=True)
class _FieldInfo:
    name: str
    dtype: DType
    shape: tuple[int, ...]
    layouts: tuple[_LayoutInfo, ...]
    bit_offset: int
    storage_bits: int


@dataclass(frozen=True, slots=True)
class _PaddingInfo:
    bit_offset: int
    storage_bits: int
    fill: int


@dataclass(frozen=True, slots=True)
class _EncodingInfo:
    definition: EncodingDefinition | DerivedEncodingDefinition
    value_type: EncodingType
    bit_order: str
    byte_order: str
    alignment: int
    fields: tuple[_FieldInfo, ...]
    padding: tuple[_PaddingInfo, ...]
    storage_bits: int
    logical_extent: int | None


@dataclass(frozen=True, slots=True)
class _SliceInfo:
    base: Value
    selectors: tuple[Value, ...]
    selector_kinds: tuple[str, ...]


class FrontendCompiler:
    def __init__(self, definition: KernelDefinition[object, object]) -> None:
        self.definition = definition
        self.source = FunctionSource.from_definition(definition)
        self.builder = IRBuilder()
        self.block: Region | None = None
        self._kernel_body: Region | None = None
        self.env: dict[str, Value] = {}
        self.static_env: dict[str, object] = {}
        self.immutable_names: set[str] = set()
        self._birth_unavailable: dict[str, str] = {}
        self.active_domain: Value | None = None
        self._shape_ids: dict[str, int] = {}
        self._axis_ids: dict[str, int] = {}
        self._shape_symbols: dict[str, Value] = {}
        self._shape_multiples: dict[str, int] = {}
        self._auto_symbols: dict[str, Value] = {}
        self._auto_choices: dict[str, tuple[int, ...]] = {}
        self._next_shape_id = -1
        self._next_axis_id = 1
        self._next_domain_serial = 1
        self._declarations: list[Operation] = []
        self._declared_encodings: dict[str, _EncodingInfo] = {}
        self._declared_derives: set[tuple[str, tuple[int, ...]]] = set()
        self._declaring_derives: set[str] = set()
        self._slice_info: dict[Value, _SliceInfo] = {}
        self._field_record_extent: dict[Value, int] = {}
        self._derive_result: ViewType | None = None
        self._domain_partition_values: dict[int, Value] = {}
        self._domain_partition_spelling: dict[int, str | int] = {}
        self._view_roots: dict[Value, int] = {}
        self._argument_access: list[set[str]] = []

    def compile(self) -> str:
        function = self.source.function
        if function.args.posonlyargs or function.args.vararg or function.args.kwarg:
            raise FrontendError(
                "kernel parameters are annotated positional or keyword parameters",
                self.source.location(function),
            )
        parameters = function.args.args + function.args.kwonlyargs
        argument_types: list[ValueType] = []
        for parameter in parameters:
            if parameter.annotation is None:
                raise FrontendError(
                    f"kernel parameter {parameter.arg!r} requires a View annotation",
                    self.source.location(parameter),
                )
            annotation = self._parse_annotation(parameter.annotation)
            if not isinstance(annotation, (ViewType, ScalarType)):
                raise FrontendError(
                    "kernel parameters are View objects or explicit scalar values",
                    self.source.location(parameter),
                )
            argument_types.append(annotation)

        conflicts = set(self._shape_ids).intersection(parameter.arg for parameter in parameters)
        if conflicts:
            raise FrontendError(
                "kernel parameters conflict with inferred shape symbols: " + ", ".join(sorted(conflicts)),
                self.source.location(function),
            )

        shape_names = {shape_id: name for name, shape_id in self._shape_ids.items()}
        for argument_type in argument_types:
            if not isinstance(argument_type, ViewType) or not argument_type.shape:
                continue
            info = self._declared_encodings.get(argument_type.encoding.family)
            shape_name = shape_names.get(argument_type.shape[-1])
            if info is None or info.logical_extent is None or shape_name is None:
                continue
            self._shape_multiples[shape_name] = math.lcm(
                self._shape_multiples.get(shape_name, 1), info.logical_extent
            )

        body = self.builder.region(
            tuple(argument_types), tuple(parameter.arg for parameter in parameters)
        )
        self._kernel_body = body
        self.block = body
        self.env = {
            parameter.arg: argument
            for parameter, argument in zip(parameters, body.arguments)
        }
        self._argument_access = [set() for _ in body.arguments]
        for index, argument in enumerate(body.arguments):
            if isinstance(argument.type, ViewType):
                self._view_roots[argument] = index
        self._materialize_shape_symbols(function)
        root_type = DomainType("root", 0, -1, 0, "root", "exact")
        self.active_domain = self._emit(
            "weft_kernel.root_domain", function, result_types=(root_type,)
        )[0]
        self._compile_statements(function.body)
        if not body.terminated:
            self._emit("weft_kernel.return", function)

        unknown_alias_parameters = set(self.definition.alias_groups).difference(
            parameter.arg for parameter in parameters
        )
        if unknown_alias_parameters:
            unknown = ", ".join(sorted(unknown_alias_parameters))
            raise FrontendError(
                f"kernel alias_groups names unknown parameters: {unknown}",
                self.source.location(function),
            )
        alias_ids: dict[str, int] = {}
        next_alias_id = 1
        argument_alias_sets: list[int] = []
        for parameter, argument_type in zip(parameters, argument_types):
            group = self.definition.alias_groups.get(parameter.arg)
            if not isinstance(argument_type, ViewType):
                if group is not None:
                    raise FrontendError(
                        f"kernel alias group applies only to View parameter {parameter.arg}",
                        self.source.location(parameter),
                    )
                argument_alias_sets.append(-1)
                continue
            if group is None:
                argument_alias_sets.append(0)
                continue
            if group not in alias_ids:
                alias_ids[group] = next_alias_id
                next_alias_id += 1
            argument_alias_sets.append(alias_ids[group])

        strided = set(self.definition.strided)
        unknown_strides = strided.difference(parameter.arg for parameter in parameters)
        if unknown_strides:
            raise FrontendError(f"strided names unknown parameters: {sorted(unknown_strides)}", self.source.location(function))
        accesses = ["readwrite" if access == {"read", "write"} else
                    "read" if access == {"read"} else "write" if access == {"write"} else "none"
                    for access in self._argument_access]
        argument_info = []
        shape_names = {identity: name for name, identity in self._shape_ids.items()}
        for parameter, value_type, access, alias in zip(parameters, argument_types, accesses, argument_alias_sets):
            is_view = isinstance(value_type, ViewType)
            if parameter.arg in strided and (not is_view or value_type.encoding.kind != "dense"):
                raise FrontendError("runtime strides require a dense View parameter", self.source.location(parameter))
            argument_info.append({
                "name": parameter.arg, "kind": "view" if is_view else "scalar",
                "dtype": (value_type.encoding.family if value_type.encoding.kind == "dense" else "") if is_view else value_type.dtype.name,
                "encoding": value_type.encoding.layout_identity if is_view else "",
                "shape": [str(extent) if extent > 0 else shape_names[extent] for extent in value_type.shape] if is_view else [],
                "access": access, "alias_set": alias, "strided": parameter.arg in strided,
            })
        self.interface_description = {
            "arguments": argument_info,
            "shape_parameters": [name for name in self._shape_ids if name not in self._auto_symbols],
            "source_parameters": {name: list(choices) for name, choices in self._auto_choices.items()},
        }
        kernel = self.builder.operation(
            "weft_kernel.kernel",
            self.source.location(function),
            attributes={
                "sym_name": _string(self.definition.__name__),
                "arg_names": _strings([parameter.arg for parameter in parameters]),
                "arg_access": _strings(accesses),
                "arg_alias_sets": _i64_array(argument_alias_sets),
                "strided_args": _strings(sorted(strided)),
                "shape_symbols": _strings(list(self._shape_ids)),
                "source": _string(f"{self.source.filename}:{self.source.first_line}"),
            },
            regions=(body,),
        )
        return self.builder.module(
            self.definition.__module__, tuple(self._declarations) + (kernel,)
        )

    def _location(self, node: ast.AST) -> SourceLocation:
        return self.source.location(node)

    def _emit(
        self,
        name: str,
        node: ast.AST,
        *,
        operands: tuple[Value, ...] = (),
        result_types: tuple[ValueType, ...] = (),
        attributes: dict[str, str] | None = None,
        regions: tuple[Region, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> tuple[Value, ...]:
        if self.block is None:
            raise AssertionError("compiler has no insertion block")
        attrs = dict(attributes or {})
        return self.builder.emit(
            self.block,
            name,
            self._location(node),
            operands=operands,
            result_types=result_types,
            attributes=attrs,
            regions=regions,
            result_names=result_names,
        )

    def _shape_id(self, name: str) -> int:
        if name not in self._shape_ids:
            self._shape_ids[name] = self._next_shape_id
            self._next_shape_id -= 1
        return self._shape_ids[name]

    def _axis_id(self, name: str) -> int:
        key = name.lower()
        if key not in self._axis_ids:
            self._axis_ids[key] = self._next_axis_id
            self._next_axis_id += 1
        return self._axis_ids[key]

    def _materialize_shape_symbols(self, node: ast.AST) -> None:
        assert self.block is not None
        for name in tuple(self._shape_ids):
            value = self._emit(
                "weft_kernel.symbol",
                node,
                result_types=(ScalarType(index),),
                attributes={
                    "name": _string(name),
                    "kind": _string("shape"),
                    "choices": _i64_array(()),
                },
                result_names=(name,),
            )[0]
            self._shape_symbols[name] = value

    def _parse_annotation(self, annotation: ast.expr) -> ValueType:
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        if isinstance(annotation, (ast.Name, ast.Attribute)):
            static = self._resolve_static(annotation)
            if isinstance(static, DType):
                return ScalarType(static)
            if isinstance(static, EncodingDefinition):
                return self._declare_encoding(static).value_type
            if isinstance(static, DerivedEncodingDefinition):
                raise FrontendError(
                    "kernel ABI requires a concrete derived Encoding instance",
                    self._location(annotation),
                )
        if isinstance(annotation, ast.Subscript):
            owner = self._resolve_static(annotation.value)
            if isinstance(owner, DerivedEncodingDefinition):
                arguments = (
                    list(annotation.slice.elts)
                    if isinstance(annotation.slice, ast.Tuple)
                    else [annotation.slice]
                )
                values = tuple(self._eval_static(argument) for argument in arguments)
                if not values or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in values
                ):
                    raise FrontendError(
                        "derived Encoding parameters are positive static integers",
                        self._location(annotation),
                    )
                self._declare_derived(owner, values)
                identity = owner.__name__ + "[" + ",".join(str(value) for value in values) + "]"
                return EncodingType(
                    owner.__name__, "derived_instance", identity, values
                )
            if owner is View:
                items = (
                    list(annotation.slice.elts)
                    if isinstance(annotation.slice, ast.Tuple)
                    else [annotation.slice]
                )
                if len(items) != 2:
                    raise FrontendError(
                        "View annotation is View[Encoding, shape]",
                        self._location(annotation),
                    )
                encoding = self._parse_annotation(items[0])
                if not isinstance(encoding, EncodingType):
                    if isinstance(encoding, ScalarType):
                        encoding = EncodingType(
                            encoding.dtype.name,
                            "dense",
                            f"dense.{encoding.dtype.name}",
                            (),
                        )
                    else:
                        raise FrontendError("invalid View encoding", self._location(items[0]))
                shape_nodes = (
                    list(items[1].elts)
                    if isinstance(items[1], (ast.Tuple, ast.List))
                    else [items[1]]
                )
                shape: list[int] = []
                axes: list[int] = []
                for dimension in shape_nodes:
                    if (
                        isinstance(dimension, ast.Constant)
                        and isinstance(dimension.value, int)
                        and not isinstance(dimension.value, bool)
                    ):
                        if dimension.value <= 0:
                            raise FrontendError("View extent must be positive", self._location(dimension))
                        shape.append(dimension.value)
                        axes.append(self._next_axis_id)
                        self._next_axis_id += 1
                    elif isinstance(dimension, ast.Name):
                        shape.append(self._shape_id(dimension.id))
                        axes.append(self._axis_id(dimension.id))
                    else:
                        raise FrontendError(
                            "View shape uses integer or symbolic dimensions",
                            self._location(dimension),
                        )
                return ViewType(encoding, tuple(shape), tuple(axes))
        raise FrontendError("unsupported Weft annotation", self._location(annotation))

    def _resolve_static(self, node: ast.AST) -> object:
        if isinstance(node, ast.Name) and node.id in self.static_env:
            return self.static_env[node.id]
        return self.source.resolve(node)

    def _declare_encoding(self, definition: EncodingDefinition) -> _EncodingInfo:
        if definition.__name__ in self._declared_encodings:
            return self._declared_encodings[definition.__name__]
        tree = ast.parse(definition.source.text, filename=definition.source.filename)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1:
            raise FrontendError(
                "encoding source must contain one class",
                SourceLocation(definition.source.filename, definition.source.first_line, 0),
            )
        declaration = classes[0]
        module_bindings = definition.bindings
        bit_order = ""
        byte_order = ""
        alignment = 1
        elements: int | None = None
        layout_items: list[tuple[str, object]] = []
        for statement in declaration.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "layout"
                for target in statement.targets
            ):
                values = (
                    statement.value.elts
                    if isinstance(statement.value, ast.Tuple)
                    else [statement.value]
                )
                for value in values:
                    resolved = self._resolve_encoding_binding(value, module_bindings)
                    if resolved in (bitorder.lsb_first, bitorder.msb_first):
                        if bit_order:
                            raise FrontendError(
                                "encoding layout declares bit order more than once"
                            )
                        bit_order = resolved.value
                    elif resolved in (byteorder.little, byteorder.big):
                        if byte_order:
                            raise FrontendError(
                                "encoding layout declares byte order more than once"
                            )
                        byte_order = resolved.value
                    else:
                        raise FrontendError(
                            "encoding layout accepts bitorder and byteorder atoms"
                        )
            elif (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "elements"
            ):
                if elements is not None:
                    raise FrontendError("encoding elements may be declared only once")
                try:
                    literal = ast.literal_eval(statement.value)
                except (ValueError, TypeError) as error:
                    raise FrontendError(
                        "encoding elements must be a positive integer"
                    ) from error
                if isinstance(literal, bool) or not isinstance(literal, int) or literal <= 0:
                    raise FrontendError("encoding elements must be a positive integer")
                elements = literal
            elif (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "alignment"
            ):
                try:
                    literal = ast.literal_eval(statement.value)
                except (ValueError, TypeError) as error:
                    raise FrontendError("encoding alignment must be a positive byte count") from error
                if (
                    isinstance(literal, bool)
                    or not isinstance(literal, int)
                    or literal <= 0
                ):
                    raise FrontendError("encoding alignment must be a positive byte count")
                alignment = literal
            elif (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and self._resolve_encoding_binding(
                    statement.value.func, module_bindings
                )
                is padding
            ):
                call = statement.value
                if not 1 <= len(call.args) <= 2:
                    raise FrontendError("padding expects bytes and optional fill")
                try:
                    byte_count = self._resolve_encoding_binding(
                        call.args[0], module_bindings
                    )
                    fill = (
                        self._resolve_encoding_binding(call.args[1], module_bindings)
                        if len(call.args) == 2
                        else 0
                    )
                    for keyword in call.keywords:
                        if keyword.arg == "value":
                            fill = self._resolve_encoding_binding(
                                keyword.value, module_bindings
                            )
                        else:
                            raise FrontendError("padding accepts only value=")
                except (ValueError, TypeError) as error:
                    raise FrontendError("padding bytes and fill must be integer literals") from error
                if (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count <= 0
                    or isinstance(fill, bool)
                    or not isinstance(fill, int)
                    or not 0 <= fill <= 255
                ):
                    raise FrontendError("padding requires positive bytes and one-byte fill")
                layout_items.append(("padding", (byte_count, fill)))
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                dtype, shape, layouts = self._parse_encoding_field(
                    statement.annotation, module_bindings, definition
                )
                layout_items.append(("field", (statement.target.id, dtype, shape, layouts)))
        if not bit_order or not byte_order:
            raise FrontendError(
                "encoding layout must declare bit order and byte order",
                SourceLocation(definition.source.filename, definition.source.first_line, 0),
            )
        if elements is None:
            raise FrontendError(
                "encoding must declare a positive elements count",
                SourceLocation(definition.source.filename, definition.source.first_line, 0),
            )
        offset = 0
        fields: list[_FieldInfo] = []
        paddings: list[_PaddingInfo] = []
        item_index = 0
        while item_index < len(layout_items):
            kind, item = layout_items[item_index]
            if kind == "padding":
                byte_count, fill = item
                if offset % 8:
                    raise FrontendError("byte padding must begin at a byte boundary")
                storage_bits = int(byte_count) * 8
                paddings.append(_PaddingInfo(offset, storage_bits, int(fill)))
                offset += storage_bits
                item_index += 1
                continue
            name, dtype, shape, layouts = item
            count = math.prod(shape) if shape else 1
            if dtype.bits is None:
                raise FrontendError("encoding fields require fixed-width dtypes")
            self._verify_field_layout(dtype, shape, layouts, definition)
            if layouts[0].kind == "joined":
                joined = layouts[0]
                members: list[tuple[str, DType, tuple[int, ...], tuple[_LayoutInfo, ...]]] = []
                for member_index in range(joined.fields):
                    cursor = item_index + member_index
                    if cursor >= len(layout_items) or layout_items[cursor][0] != "field":
                        raise FrontendError("pack_fields requires consecutive logical fields")
                    member = layout_items[cursor][1]
                    member_name, member_dtype, member_shape, member_layouts = member
                    if (
                        member_dtype != dtype
                        or member_shape != shape
                        or len(member_layouts) != 1
                        or member_layouts[0] != joined
                    ):
                        raise FrontendError("pack_fields members must have one identical layout declaration")
                    members.append(member)
                storage_bits = joined.size * (joined.fields + 1) * 8
                for role, (member_name, member_dtype, member_shape, _) in enumerate(members):
                    concrete = _LayoutInfo(
                        joined.kind,
                        joined.size,
                        joined.order,
                        joined.fields,
                        joined.low_bits,
                        role,
                    )
                    fields.append(
                        _FieldInfo(
                            member_name,
                            member_dtype,
                            member_shape,
                            (concrete,),
                            offset,
                            storage_bits,
                        )
                    )
                offset += storage_bits
                item_index += joined.fields
                continue
            storage_bits = dtype.bits * count
            fields.append(_FieldInfo(name, dtype, shape, layouts, offset, storage_bits))
            offset += storage_bits
            item_index += 1
        encoding_type = EncodingType(
            definition.__name__, "base", definition.__name__, ()
        )
        info = _EncodingInfo(
            definition,
            encoding_type,
            bit_order,
            byte_order,
            alignment,
            tuple(fields),
            tuple(paddings),
            offset,
            elements,
        )
        self._declared_encodings[definition.__name__] = info
        location = SourceLocation(definition.source.filename, definition.source.first_line, 0)
        self._declarations.append(
            self.builder.operation(
                "weft_kernel.encoding_decl",
                location,
                attributes={
                    "sym_name": _string(definition.__name__),
                    "kind": _string("base"),
                    "layout_identity": _string(definition.__name__),
                    "bit_order": _string(bit_order),
                    "byte_order": _string(byte_order),
                    "alignment": str(alignment),
                    "elements": str(elements),
                    "storage_bits": str(offset),
                    "field_names": _strings([field.name for field in fields]),
                    "field_types": _type_array([ScalarType(field.dtype) for field in fields]),
                    "field_shapes": "["
                    + ", ".join(_i64_array(field.shape) for field in fields)
                    + "]",
                    "field_layouts": _layout_array(fields),
                    "field_bit_offsets": _i64_array([field.bit_offset for field in fields]),
                    "field_storage_bits": _i64_array([field.storage_bits for field in fields]),
                    "padding": _i64_array(
                        [
                            component
                            for padding in paddings
                            for component in (
                                padding.bit_offset,
                                padding.storage_bits,
                                padding.fill,
                            )
                        ]
                    ),
                },
            )
        )
        return info

    def _parse_encoding_field(
        self,
        annotation: ast.expr,
        bindings: dict[str, object],
        definition: EncodingDefinition,
    ) -> tuple[DType, tuple[int, ...], tuple[_LayoutInfo, ...]]:
        layouts: list[_LayoutInfo] = []
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        while isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.MatMult):
            layout_node = annotation.right
            annotation = annotation.left
            if not isinstance(layout_node, ast.Call):
                raise FrontendError("encoding field layout must use an explicit constructor")
            values = [
                self._resolve_encoding_binding(argument, bindings)
                for argument in layout_node.args
            ]
            keywords = {}
            for keyword in layout_node.keywords:
                if keyword.arg is None or keyword.arg in keywords:
                    raise FrontendError("encoding field layout needs distinct named parameters")
                keywords[keyword.arg] = self._resolve_encoding_binding(keyword.value, bindings)
            constructor = self._resolve_encoding_binding(layout_node.func, bindings)
            if constructor not in {grouped, bit_layers, pack_fields}:
                raise FrontendError(
                    "encoding field layout must be grouped, bit_layers, or pack_fields"
                )
            try:
                specification = constructor(*values, **keywords)
            except (TypeError, ValueError) as error:
                raise FrontendError(f"invalid encoding field layout parameters: {error}") from error
            if constructor is grouped:
                layout = _LayoutInfo("grouped", specification.size)
            elif constructor is bit_layers:
                layout = _LayoutInfo(
                    "layered", specification.size, specification.order
                )
            else:
                layout = _LayoutInfo(
                    "joined",
                    specification.size,
                    specification.order,
                    specification.fields,
                    specification.low_bits,
                )
            layouts.insert(0, layout)
        shape: tuple[int, ...] = ()
        if isinstance(annotation, ast.Subscript):
            dtype = self._resolve_encoding_binding(annotation.value, bindings)
            dimensions = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            try:
                dimension_values = tuple(
                    ast.literal_eval(dimension) for dimension in dimensions
                )
            except (ValueError, TypeError) as error:
                raise FrontendError("encoding array dimensions must be integers") from error
            if any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in dimension_values
            ):
                raise FrontendError(
                    "encoding array dimensions must be positive integers"
                )
            shape = dimension_values
        elif isinstance(annotation, (ast.Name, ast.Attribute)):
            dtype = self._resolve_encoding_binding(annotation, bindings)
        else:
            dtype = None
        if not isinstance(dtype, DType) or any(dimension <= 0 for dimension in shape):
            raise FrontendError(
                f"encoding {definition.__name__} fields require fixed Weft dtypes"
            )
        if not layouts:
            layouts.append(_LayoutInfo("natural"))
        return dtype, shape, tuple(layouts)

    @classmethod
    def _resolve_encoding_binding(
        cls, node: ast.expr, bindings: dict[str, object]
    ) -> object:
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if isinstance(node, ast.Attribute):
            owner = cls._resolve_encoding_binding(node.value, bindings)
            try:
                return getattr(owner, node.attr)
            except AttributeError as error:
                raise FrontendError(
                    f"unknown encoding declaration attribute {node.attr!r}"
                ) from error
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError) as error:
            raise FrontendError(
                "encoding declaration values must be static"
            ) from error

    def _verify_field_layout(
        self,
        dtype: DType,
        shape: tuple[int, ...],
        layouts: tuple[_LayoutInfo, ...],
        definition: EncodingDefinition,
    ) -> None:
        if layouts == (_LayoutInfo("natural"),):
            return
        if len(layouts) == 1 and layouts[0].kind == "joined":
            joined = layouts[0]
            if (
                len(shape) != 1
                or joined.size <= 0
                or joined.fields < 2
                or shape[0] != 2 * joined.size
                or joined.low_bits <= 0
                or dtype.bits is None
                or joined.low_bits >= dtype.bits
                or joined.low_bits * joined.fields != 8
                or dtype.bits + (dtype.bits - joined.low_bits) != 8
                or dtype.bits * shape[0] * joined.fields
                != joined.size * (joined.fields + 1) * 8
                or joined.order not in {"lo_first", "hi_first"}
            ):
                raise FrontendError("pack_fields parameters do not form one dense byte container")
            return
        if len(shape) != 1:
            raise FrontendError(
                f"encoding {definition.__name__} grouped/bit_layers fields must be rank one"
            )
        if len(layouts) != 2 or layouts[0].kind != "grouped" or layouts[1].kind != "layered":
            raise FrontendError("encoding field layout requires grouped(elements=...) followed by bit_layers(elements=..., order=...)")
        group, layer = layouts
        if group.size <= 0 or shape[0] % group.size:
            raise FrontendError("grouped extent must divide the logical field extent")
        if layer.size <= 0 or group.size % layer.size:
            raise FrontendError("bit_layers elements must divide its enclosing group")
        if layer.order not in {"lo_first", "hi_first"}:
            raise FrontendError("bit_layers order must be lo_first or hi_first")
        if dtype.bits is None or dtype.bits * (group.size // layer.size) != 8:
            raise FrontendError("bit_layers requires one byte per layer position")

    def _declare_derived(
        self, definition: DerivedEncodingDefinition, arguments: tuple[int, ...]
    ) -> None:
        name = definition.__name__
        instance = (name, arguments)
        if instance in self._declared_derives:
            return
        if name in self._declaring_derives:
            raise FrontendError(f"recursive derived encoding {name!r}")
        self._declaring_derives.add(name)
        source = FunctionSource.from_definition(definition)
        if len(source.function.args.args) != 1 or source.function.args.args[0].annotation is None:
            raise FrontendError("derived encoding takes one annotated View", source.location(source.function))
        static_parameters = source.function.args.kwonlyargs
        if len(static_parameters) != len(arguments):
            raise FrontendError(
                "derived Encoding instance does not bind every static parameter",
                source.location(source.function),
            )
        previous_source = self.source
        previous_static = self.static_env
        self.source = source
        self.static_env = dict(previous_static)
        self.static_env.update(
            {parameter.arg: value for parameter, value in zip(static_parameters, arguments)}
        )
        try:
            source_type = self._parse_annotation(source.function.args.args[0].annotation)
            if not isinstance(source_type, ViewType):
                raise FrontendError("derived encoding input must be a View", source.location(source.function))
            identity = name + "[" + ",".join(str(value) for value in arguments) + "]"
            symbol = name + "$" + "$".join(str(value) for value in arguments)
            result_encoding = EncodingType(
                name, "derived_instance", identity, arguments
            )
            result_type = ViewType(result_encoding, source_type.shape, source_type.axes)
            region = self.builder.region((source_type,), (source.function.args.args[0].arg,))
            saved = (self.block, self.env, self._derive_result, self.active_domain)
            self.block = region
            self.env = {source.function.args.args[0].arg: region.arguments[0]}
            self._derive_result = result_type
            self.active_domain = None
            if len(source.function.body) != 1 or not isinstance(source.function.body[0], ast.Return):
                raise FrontendError("derived encoding body is one return expression", source.location(source.function))
            return_node = source.function.body[0]
            if return_node.value is None:
                raise FrontendError("derived encoding must return a View", source.location(return_node))
            result = self._expect_value(self._compile_expr(return_node.value), return_node.value)
            if result.type != result_type:
                raise FrontendError("derived encoding result does not match its generated family", source.location(return_node))
            self._emit("weft_kernel.derive_yield", return_node, operands=(result,))
            self.block, self.env, self._derive_result, self.active_domain = saved
            self._declarations.append(
                self.builder.operation(
                    "weft_kernel.derive",
                    source.location(source.function),
                    attributes={
                        "sym_name": _string(symbol),
                        "source_family": _string(source_type.encoding.family),
                        "result_family": _string(name),
                        "layout_identity": _string(identity),
                        "parameter_names": _strings(
                            [parameter.arg for parameter in static_parameters]
                        ),
                        "parameter_values": _i64_array(arguments),
                    },
                    regions=(region,),
                )
            )
            source_info = self._declared_encodings.get(source_type.encoding.family)
            if source_info is not None:
                self._declared_encodings[name] = _EncodingInfo(
                    definition,
                    result_encoding,
                    source_info.bit_order,
                    source_info.byte_order,
                    source_info.alignment,
                    tuple(
                        _FieldInfo(
                            field.name,
                            field.dtype,
                            field.shape,
                            field.layouts,
                            field.bit_offset,
                            field.storage_bits,
                        )
                        for field in source_info.fields
                    ),
                    (),
                    source_info.storage_bits,
                    source_info.logical_extent,
                )
            self._declared_derives.add(instance)
        finally:
            self.source = previous_source
            self.static_env = previous_static
            self._declaring_derives.discard(name)

    def _compile_statements(self, statements: Sequence[ast.stmt]) -> None:
        for index_in_body, statement in enumerate(statements):
            live_after = _names_needed_before_definition(statements[index_in_body + 1 :])
            self._compile_statement(statement, live_after)

    def _compile_statement(self, statement: ast.stmt, live_after: set[str]) -> None:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    raise FrontendError("chained assignment is not part of Weft", self._location(statement))
                target = statement.targets[0]
                expression = statement.value
            else:
                if statement.value is None:
                    raise FrontendError("declaration requires a value", self._location(statement))
                target = statement.target
                expression = statement.value
            value = self._expect_value(self._compile_expr(expression), expression)
            self._bind_target(target, value)
            return
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name) or statement.target.id not in self.env:
                raise FrontendError("augmented assignment requires an existing local", self._location(statement))
            synthetic = ast.BinOp(statement.target, statement.op, statement.value)
            ast.copy_location(synthetic, statement)
            self._bind_name(
                statement.target.id,
                self._expect_value(self._compile_expr(synthetic), statement),
                statement.target,
            )
            return
        if isinstance(statement, ast.Expr):
            self._compile_expr(statement.value)
            return
        if isinstance(statement, ast.With):
            self._compile_level(statement)
            return
        if isinstance(statement, ast.For):
            self._compile_for(statement)
            return
        if isinstance(statement, ast.If):
            self._compile_if(statement, live_after)
            return
        if isinstance(statement, ast.While):
            self._compile_while(statement)
            return
        if isinstance(statement, ast.Return):
            raise FrontendError("return is valid only in an inline function", self._location(statement))
        if isinstance(statement, ast.Pass):
            return
        raise FrontendError(
            f"unsupported Weft statement {type(statement).__name__}", self._location(statement)
        )

    def _bind_name(self, name: str, value: Value, node: ast.AST) -> None:
        if name in self.immutable_names:
            raise FrontendError(
                f"staged value {name!r} is read-only in descendant code",
                self._location(node),
            )
        self.env[name] = value

    def _propagate_view_root(self, result: Value, source: Value) -> None:
        root = self._view_roots.get(source)
        if root is not None:
            self._view_roots[result] = root

    def _mark_view_access(self, value: Value, effect: str) -> None:
        root = self._view_roots.get(value)
        if root is not None:
            self._argument_access[root].add(effect)

    def _bind_target(self, target: ast.expr, value: Value) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, value, target)
            return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            if name not in self.env:
                raise FrontendError("indexed assignment requires an existing state value", self._location(target))
            current = self.env[name]
            indices, kinds, _, _ = self._compile_selectors(
                target.slice, current.type, current
            )
            updated = self._emit(
                "weft_kernel.update",
                target,
                operands=(current, value) + tuple(indices),
                result_types=(current.type,),
                attributes={"selectors": _strings(kinds)},
            )[0]
            self._bind_name(name, updated, target)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            raise FrontendError("tuple assignment is not part of the frozen frontend", self._location(target))
        raise FrontendError("assignment target must be a name or state element", self._location(target))

    def _compile_expr(self, expression: ast.expr) -> Value | None:
        if isinstance(expression, ast.Name):
            if expression.id in self.env:
                return self.env[expression.id]
            if expression.id in self._birth_unavailable:
                raise FrontendError(
                    self._birth_unavailable[expression.id], self._location(expression)
                )
            if expression.id in self._shape_symbols:
                return self._shape_symbols[expression.id]
            if expression.id in self._auto_symbols:
                return self._auto_symbols[expression.id]
            static = self._resolve_static(expression)
            if isinstance(static, (bool, int, float)):
                return self._constant(static, None, expression)
            raise FrontendError(
                f"static binding {static!r} is not a runtime value", self._location(expression)
            )
        if isinstance(expression, ast.Constant):
            if expression.value is None:
                return None
            return self._constant(expression.value, None, expression)
        if isinstance(expression, ast.Call):
            return self._compile_call(expression)
        if isinstance(expression, ast.Attribute):
            return self._compile_field(expression)
        if isinstance(expression, ast.Subscript):
            return self._compile_subscript(expression)
        if isinstance(expression, ast.BinOp):
            lhs, rhs = self._compile_binary_operands(expression.left, expression.right)
            operations = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.FloorDiv: "div",
                ast.Mod: "mod",
                ast.BitAnd: "and",
                ast.BitOr: "or",
                ast.BitXor: "xor",
                ast.LShift: "shl",
                ast.RShift: "shr",
            }
            kind = operations.get(type(expression.op))
            if kind is None:
                raise FrontendError("unsupported binary operator", self._location(expression))
            return self._binary_values(kind, lhs, rhs, expression)
        if isinstance(expression, ast.UnaryOp):
            if isinstance(expression.op, ast.USub) and isinstance(expression.operand, ast.Constant):
                return self._constant(-expression.operand.value, None, expression)
            operand = self._expect_value(self._compile_expr(expression.operand), expression.operand)
            if isinstance(expression.op, ast.USub):
                if not isinstance(element_type(operand.type), ScalarType):
                    raise FrontendError(
                        "numeric negation requires scalar elements",
                        self._location(expression),
                    )
                return self._emit(
                    "weft_kernel.unary",
                    expression,
                    operands=(operand,),
                    result_types=(operand.type,),
                    attributes={"kind": _string("neg")},
                )[0]
            raise FrontendError("unsupported unary operator", self._location(expression))
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                raise FrontendError("chained comparison is not supported", self._location(expression))
            lhs, rhs = self._compile_binary_operands(expression.left, expression.comparators[0])
            if not isinstance(element_type(lhs.type), ScalarType):
                raise FrontendError(
                    "comparison requires numeric scalar elements",
                    self._location(expression),
                )
            predicates = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
            }
            predicate = predicates.get(type(expression.ops[0]))
            if predicate is None:
                raise FrontendError("unsupported comparison", self._location(expression))
            shape, axes = self._broadcast_domain(lhs.type, rhs.type, expression)
            result_type = value_type(ScalarType(i1), shape, axes)
            return self._emit(
                "weft_kernel.compare",
                expression,
                operands=(lhs, rhs),
                result_types=(result_type,),
                attributes={"predicate": _string(predicate)},
            )[0]
        raise FrontendError(
            f"unsupported Weft expression {type(expression).__name__}", self._location(expression)
        )

    def _expect_value(self, value: Value | None, node: ast.AST) -> Value:
        if value is None:
            raise FrontendError("expression does not produce a value", self._location(node))
        if isinstance(value.type, SliceType) and not value.type.shape:
            return self._admit_value(value, node)
        return value

    def _constant(self, literal: object, dtype: DType | None, node: ast.AST) -> Value:
        if dtype is None:
            if isinstance(literal, bool):
                dtype = i1
            elif isinstance(literal, int):
                dtype = index
            elif isinstance(literal, float):
                dtype = f32
            else:
                raise FrontendError("runtime literals are bool, int, or float", self._location(node))
        value_type_ = ScalarType(dtype)
        if dtype.category is DTypeCategory.BOOL:
            spelling = "true" if literal else "false"
        elif dtype.category in {DTypeCategory.INTEGER, DTypeCategory.INDEX}:
            spelling = f"{int(literal)} : {emit_type(value_type_)}"
        else:
            floating = float(literal)
            if math.isnan(floating):
                raise FrontendError("NaN literals are not part of the frozen frontend", self._location(node))
            if math.isinf(floating):
                patterns = {
                    16: ("0x7C00", "0xFC00"),
                    32: ("0x7F800000", "0xFF800000"),
                    64: ("0x7FF0000000000000", "0xFFF0000000000000"),
                }
                if dtype.bits not in patterns:
                    raise FrontendError("infinity requires f16, f32, or f64", self._location(node))
                number = patterns[dtype.bits][floating < 0]
            else:
                number = f"{floating:.17g}"
                if "." not in number and "e" not in number.lower():
                    number += ".0"
            spelling = f"{number} : {emit_type(value_type_)}"
        operation = (
            "weft_kernel.constant"
            if dtype.category is DTypeCategory.INTEGER
            else "arith.constant"
        )
        return self._emit(
            operation,
            node,
            result_types=(value_type_,),
            attributes={"value": spelling},
        )[0]

    def _compile_binary_operands(self, lhs_node: ast.expr, rhs_node: ast.expr) -> tuple[Value, Value]:
        lhs = self._expect_value(self._compile_expr(lhs_node), lhs_node)
        rhs = self._expect_value(self._compile_expr(rhs_node), rhs_node)
        lhs_element = element_type(lhs.type)
        rhs_element = element_type(rhs.type)
        if lhs_element != rhs_element:
            if isinstance(lhs_node, ast.Constant) and isinstance(rhs_element, ScalarType):
                lhs = self._constant(lhs_node.value, rhs_element.dtype, lhs_node)
            elif isinstance(rhs_node, ast.Constant) and isinstance(lhs_element, ScalarType):
                rhs = self._constant(rhs_node.value, lhs_element.dtype, rhs_node)
            elif isinstance(lhs_element, ScalarType) and isinstance(rhs_element, ScalarType):
                target = self._promote_dtype(lhs_element.dtype, rhs_element.dtype)
                lhs = self._convert_value(lhs, target, lhs_node)
                rhs = self._convert_value(rhs, target, rhs_node)
            else:
                raise FrontendError("binary element types must match", self._location(lhs_node))
        return lhs, rhs

    def _promote_dtype(self, lhs: DType, rhs: DType) -> DType:
        floating = {DTypeCategory.FLOAT, DTypeCategory.BFLOAT}
        if lhs.category in floating or rhs.category in floating:
            return f64 if lhs == f64 or rhs == f64 else f32
        bits = max(lhs.bits or 64, rhs.bits or 64)
        signed = lhs.signedness == "signed" or rhs.signedness == "signed"
        table = {
            (True, 8): i8,
            (True, 16): i16,
            (True, 32): i32,
            (True, 64): i64,
            (False, 8): u8,
            (False, 16): u16,
            (False, 32): u32,
            (False, 64): u64,
        }
        width = next(width for width in (8, 16, 32, 64) if bits <= width)
        return table[(signed, width)]

    def _convert_value(self, value: Value, dtype: DType, node: ast.AST) -> Value:
        current = element_type(value.type)
        if current == ScalarType(dtype):
            return value
        if not isinstance(current, ScalarType):
            raise FrontendError("numeric conversion requires scalar elements", self._location(node))
        source = current.dtype
        integer_to_float = (
            source.category is DTypeCategory.INTEGER
            and dtype.category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT}
        )
        widens = (
            source.bits is not None
            and dtype.bits is not None
            and (
                dtype.bits > source.bits
                or (integer_to_float and dtype.bits >= source.bits)
            )
        )
        result = self._emit(
            "weft_kernel.widen" if widens else "weft_kernel.cast",
            node,
            operands=(value,),
            result_types=(with_element(value.type, ScalarType(dtype)),),
        )[0]
        return self._propagate_field_record_extent(result, value)

    def _propagate_field_record_extent(
        self, result: Value, *sources: Value
    ) -> Value:
        extents = {
            self._field_record_extent[source]
            for source in sources
            if source in self._field_record_extent
        }
        if len(extents) == 1:
            self._field_record_extent[result] = extents.pop()
        return result

    def _broadcast_domain(
        self, lhs: ValueType, rhs: ValueType, node: ast.AST
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        lhs_shape, rhs_shape = shape_of(lhs), shape_of(rhs)
        lhs_axes, rhs_axes = axes_of(lhs), axes_of(rhs)
        if not lhs_shape:
            return rhs_shape, rhs_axes
        if not rhs_shape:
            return lhs_shape, lhs_axes
        result: dict[int, int] = {}
        order: list[int] = []
        for shape, axes in ((lhs_shape, lhs_axes), (rhs_shape, rhs_axes)):
            for dimension, axis in zip(shape, axes):
                if axis not in result:
                    order.append(axis)
                    result[axis] = dimension
                elif result[axis] != dimension and result[axis] != 1 and dimension != 1:
                    raise FrontendError("logical axis extents do not broadcast", self._location(node))
                elif result[axis] == 1:
                    result[axis] = dimension
        return tuple(result[axis] for axis in order), tuple(order)

    def _binary_values(self, kind: str, lhs: Value, rhs: Value, node: ast.AST) -> Value:
        if element_type(lhs.type) != element_type(rhs.type):
            raise FrontendError("binary element types must match", self._location(node))
        if not isinstance(element_type(lhs.type), ScalarType):
            raise FrontendError(
                "binary operations require numeric scalar elements", self._location(node)
            )
        shape, axes = self._broadcast_domain(lhs.type, rhs.type, node)
        result_type = value_type(element_type(lhs.type), shape, axes)
        result = self._emit(
            "weft_kernel.binary",
            node,
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"kind": _string(kind)},
        )[0]
        return self._propagate_field_record_extent(result, lhs, rhs)

    def _compile_field(self, expression: ast.Attribute) -> Value:
        owner = self._expect_value(self._compile_expr(expression.value), expression.value)
        owner_element = element_type(owner.type)
        if not isinstance(owner_element, EncodingType):
            raise FrontendError("field access requires an encoded Value or View region", self._location(expression))
        info = self._declared_encodings.get(owner_element.family)
        if info is None:
            raise FrontendError(f"encoding {owner_element.family!r} is not declared", self._location(expression))
        field = next((field for field in info.fields if field.name == expression.attr), None)
        if field is None:
            raise FrontendError(
                f"encoding {owner_element.family!r} has no field {expression.attr!r}",
                self._location(expression),
            )
        shape = list(shape_of(owner.type))
        axes = list(axes_of(owner.type))
        if info.logical_extent is None or not shape:
            raise FrontendError(
                "encoded field access requires an explicit complete record axis",
                self._location(expression),
            )
        record_extent = shape.pop()
        record_axis = axes.pop()
        if record_extent > 0 and record_extent != info.logical_extent:
            raise FrontendError(
                "encoded field access must select exactly one storage record",
                self._location(expression),
            )
        for field_index, dimension in enumerate(field.shape):
            field_axis = record_axis if field_index + 1 == len(field.shape) else None
            if field_axis is None:
                field_axis = self._next_axis_id
                self._next_axis_id += 1
            if record_axis is None and field_index + 1 == len(field.shape):
                record_axis = field_axis
            shape.append(dimension)
            axes.append(field_axis)
        if isinstance(owner.type, (ViewType, SliceType)):
            result_type = SliceType(
                EncodingType(
                    field.dtype.name, "dense", f"dense.{field.dtype.name}", ()
                ),
                tuple(shape),
                tuple(axes),
            )
        else:
            result_type = value_type(ScalarType(field.dtype), tuple(shape), tuple(axes))
        result = self._emit(
            "weft_kernel.field",
            expression,
            operands=(owner,),
            result_types=(result_type,),
            attributes={"name": _string(expression.attr)},
        )[0]
        if isinstance(owner.type, (ViewType, SliceType)):
            self._propagate_view_root(result, owner)
        if info.logical_extent is not None:
            self._field_record_extent[result] = info.logical_extent
        return result

    def _compile_subscript(self, expression: ast.Subscript) -> Value:
        base = self._expect_value(self._compile_expr(expression.value), expression.value)
        selectors, kinds, result_shape, result_axes = self._compile_selectors(
            expression.slice, base.type, base
        )
        if isinstance(base.type, (ViewType, SliceType)):
            result_type: ValueType = SliceType(
                base.type.encoding, result_shape, result_axes
            )
            result = self._emit(
                "weft_kernel.slice",
                expression,
                operands=(base,) + tuple(selectors),
                result_types=(result_type,),
                attributes={"selectors": _strings(kinds)},
            )[0]
            self._slice_info[result] = _SliceInfo(base, tuple(selectors), tuple(kinds))
            self._propagate_view_root(result, base)
            return result
        if isinstance(base.type, LocalValueType):
            result_type = value_type(base.type.element_type, result_shape, result_axes)
            return self._emit(
                "weft_kernel.extract",
                expression,
                operands=(base,) + tuple(selectors),
                result_types=(result_type,),
                attributes={"selectors": _strings(kinds)},
            )[0]
        raise FrontendError("only View, slice, or local Value supports indexing", self._location(expression))

    def _compile_selectors(
        self,
        selector: ast.expr,
        base_type: ValueType,
        base_value: Value | None = None,
    ) -> tuple[list[Value], list[str], tuple[int, ...], tuple[int, ...]]:
        items = list(selector.elts) if isinstance(selector, ast.Tuple) else [selector]
        base_shape = list(shape_of(base_type))
        base_axes = list(axes_of(base_type))
        if len(items) == 1:
            candidate = items[0]
            if isinstance(candidate, ast.Name) and candidate.id in self.env:
                candidate_value = self.env[candidate.id]
                if isinstance(candidate_value.type, DomainPointType):
                    axis = candidate_value.type.domain.axis_id
                    if axis in base_axes and base_axes.index(axis) != 0:
                        items = [ast.Slice()] * base_axes.index(axis) + [candidate]
        if len(items) > len(base_shape):
            raise FrontendError("too many indices for logical value", self._location(selector))
        values: list[Value] = []
        kinds: list[str] = []
        result_shape: list[int] = []
        result_axes: list[int] = []
        for position, (dimension, axis) in enumerate(zip(base_shape, base_axes)):
            if position >= len(items):
                result_shape.append(dimension)
                result_axes.append(axis)
                continue
            item = items[position]
            if isinstance(item, ast.Slice) and item.lower is None and item.upper is None and item.step is None:
                kinds.append("all")
                result_shape.append(dimension)
                result_axes.append(axis)
                continue
            value = self._expect_value(self._compile_expr(item), item)
            values.append(value)
            if isinstance(value.type, DomainPointType):
                domain = value.type.domain
                if domain.axis_id != axis:
                    raise FrontendError("level point indexes a different logical axis", self._location(item))
                partition = self._domain_partition_spelling.get(domain.domain_id)
                if isinstance(partition, str):
                    local_dimension = self._shape_id(partition)
                elif isinstance(partition, int):
                    local_dimension = partition
                else:
                    local_dimension = dimension
                record_extent = (
                    self._field_record_extent.get(base_value)
                    if base_value is not None
                    else None
                )
                if (
                    dimension > 0
                    and record_extent is not None
                    and dimension < record_extent
                ) or (
                    dimension > 0
                    and local_dimension > 0
                    and dimension < local_dimension
                ):
                    kinds.append("group_index")
                else:
                    kinds.append("domain")
                    result_shape.append(local_dimension)
                    result_axes.append(axis)
            else:
                if value.type == ScalarType(index):
                    kinds.append("index")
                    continue
                if isinstance(value.type, LocalValueType):
                    selector_element = element_type(value.type)
                    if not isinstance(selector_element, ScalarType) or (
                        selector_element.dtype.category
                        not in {DTypeCategory.INTEGER, DTypeCategory.INDEX}
                        or selector_element.dtype.signedness == "signed"
                    ):
                        raise FrontendError(
                            "shaped gather indices must be unsigned integer or index Values",
                            self._location(item),
                        )
                    if not isinstance(base_type, LocalValueType):
                        raise FrontendError(
                            "shaped gather currently indexes a local Value",
                            self._location(item),
                        )
                    kinds.append("gather")
                    for gather_extent, gather_axis in zip(
                        shape_of(value.type), axes_of(value.type)
                    ):
                        if gather_axis in result_axes:
                            existing = result_shape[result_axes.index(gather_axis)]
                            if existing != gather_extent:
                                raise FrontendError(
                                    "gather index disagrees with an existing logical axis",
                                    self._location(item),
                                )
                            continue
                        result_shape.append(gather_extent)
                        result_axes.append(gather_axis)
                    continue
                raise FrontendError("scalar index must have index type", self._location(item))
        return values, kinds, tuple(result_shape), tuple(result_axes)

    def _compile_call(self, call: ast.Call) -> Value | None:
        callee = self._resolve_static(call.func)
        if isinstance(callee, DType):
            if len(call.args) != 1 or call.keywords:
                raise FrontendError("dtype constructor expects one value", self._location(call))
            argument = call.args[0]
            if isinstance(argument, ast.Constant):
                return self._constant(argument.value, callee, call)
            value = self._expect_value(self._compile_expr(argument), argument)
            if not isinstance(element_type(value.type), ScalarType):
                raise FrontendError("dtype constructor converts numeric Values", self._location(call))
            return self._convert_value(value, callee, call)
        if isinstance(callee, Intrinsic):
            method = getattr(self, f"_intrinsic_{callee.name}", None)
            if method is None:
                raise FrontendError(
                    f"W.{callee.name} has no canonical frontend rule", self._location(call)
                )
            return method(call)
        if isinstance(callee, OverloadSet):
            definition = self._select_overload(callee, call)
            return self._inline_function(definition, call)
        if isinstance(callee, InlineDefinition):
            return self._inline_function(callee, call)
        if isinstance(callee, FunctionType):
            return self._inline_function(InlineDefinition.capture(callee), call)
        raise FrontendError("call target is not a Weft op or inline function", self._location(call))

    def _arguments(
        self,
        call: ast.Call,
        positional: tuple[str, ...],
        defaults: dict[str, object],
    ) -> dict[str, ast.expr | object]:
        if len(call.args) > len(positional):
            raise FrontendError("too many positional arguments", self._location(call))
        result: dict[str, ast.expr | object] = dict(defaults)
        for name, value in zip(positional, call.args):
            result[name] = value
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg not in set(positional) | set(defaults):
                raise FrontendError("unknown or expanded keyword", self._location(keyword))
            result[keyword.arg] = keyword.value
        missing = [name for name in positional if name not in result]
        if missing:
            raise FrontendError(f"missing argument {missing[0]!r}", self._location(call))
        return result

    def _eval_static(self, node: ast.expr) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval_static(element) for element in node.elts)
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        if isinstance(node, ast.Call):
            callee = self._resolve_static(node.func)
            if callee is auto:
                return AutoSpec(tuple(self._eval_static(argument) for argument in node.args))
        return self._resolve_static(node)

    def _intrinsic_state(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("dtype", "shape"), {"init": None})
        dtype = self._eval_static(args["dtype"]) if isinstance(args["dtype"], ast.expr) else args["dtype"]
        if not isinstance(dtype, DType) or not isinstance(args["shape"], ast.expr):
            raise FrontendError("state expects a dtype and explicit logical shape", self._location(call))
        shape, axes = self._parse_local_shape(args["shape"])
        result_type = value_type(ScalarType(dtype), shape, axes)
        operands: tuple[Value, ...] = ()
        if isinstance(args["init"], ast.expr):
            init_node = args["init"]
            if isinstance(init_node, ast.Constant):
                initial = self._constant(init_node.value, dtype, init_node)
            elif (
                isinstance(init_node, ast.UnaryOp)
                and isinstance(init_node.op, ast.USub)
                and isinstance(init_node.operand, ast.Constant)
            ):
                initial = self._constant(-init_node.operand.value, dtype, init_node)
            else:
                initial = self._expect_value(self._compile_expr(init_node), init_node)
                initial_element = element_type(initial.type)
                if isinstance(initial_element, ScalarType) and initial_element.dtype != dtype:
                    initial = self._convert_value(initial, dtype, init_node)
            operands = (initial,)
        return self._emit(
            "weft_kernel.new",
            call,
            operands=operands,
            result_types=(result_type,),
            attributes={"initialized": "true" if operands else "false"},
        )[0]

    def _intrinsic_arange(self, call: ast.Call) -> Value:
        args = self._arguments(
            call, ("start", "end"), {"dtype": u32, "axis": None}
        )
        end = self._eval_static(args["end"])
        start = self._eval_static(args["start"]) if isinstance(args["start"], ast.expr) else args["start"]
        dtype = self._eval_static(args["dtype"]) if isinstance(args["dtype"], ast.expr) else args["dtype"]
        axis_name = (
            self._eval_static(args["axis"])
            if isinstance(args["axis"], ast.expr)
            else args["axis"]
        )
        if isinstance(start, bool) or not isinstance(start, int):
            raise FrontendError("arange start must be an integer", self._location(call))
        if isinstance(end, bool) or not isinstance(end, int) or end <= start:
            raise FrontendError("arange end must be an integer greater than start", self._location(call))
        extent = end - start
        if (
            not isinstance(dtype, DType)
            or dtype.category not in {DTypeCategory.INTEGER, DTypeCategory.INDEX}
            or dtype.signedness == "signed"
        ):
            raise FrontendError(
                "arange dtype must be an unsigned integer or index", self._location(call)
            )
        if axis_name is None:
            axis = self._next_axis_id
            self._next_axis_id += 1
        else:
            if not isinstance(axis_name, str):
                raise FrontendError(
                    "arange axis must be a logical axis name",
                    self._location(call),
                )
            axis = self._axis_id(axis_name)
        result_type = value_type(ScalarType(dtype), (extent,), (axis,))
        return self._emit(
            "weft_kernel.iota",
            call,
            result_types=(result_type,),
            attributes={"start": str(start), "end": str(start + extent)},
        )[0]

    def _parse_local_shape(self, node: ast.expr) -> tuple[tuple[int, ...], tuple[int, ...]]:
        items = list(node.elts) if isinstance(node, (ast.Tuple, ast.List)) else [node]
        shape: list[int] = []
        axes: list[int] = []
        active_points = [
            value.type.domain
            for value in self.env.values()
            if isinstance(value.type, DomainPointType)
        ]
        for item in items:
            if isinstance(item, ast.Constant) and isinstance(item.value, int):
                if item.value <= 0:
                    raise FrontendError(
                        "local shape extents must be positive",
                        self._location(item),
                    )
                shape.append(item.value)
                matching = [
                    domain.axis_id
                    for domain in active_points
                    if self._domain_partition_spelling.get(domain.domain_id)
                    == item.value
                ]
                if matching:
                    axes.append(matching[-1])
                else:
                    axes.append(self._next_axis_id)
                    self._next_axis_id += 1
            elif isinstance(item, ast.Name):
                shape.append(self._shape_id(item.id))
                matching = [
                    domain.axis_id
                    for domain in active_points
                    if self._domain_partition_spelling.get(domain.domain_id)
                    == item.id
                ]
                axes.append(matching[-1] if matching else self._axis_id(item.id))
            else:
                raise FrontendError("local shape uses integer or auto symbols", self._location(item))
        return tuple(shape), tuple(axes)

    def _intrinsic_stage(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("expr",), {})
        value = self._expect_value(self._compile_expr(args["expr"]), args["expr"])
        result = self._emit(
            "weft_kernel.materialize",
            call,
            operands=(value,),
            result_types=(value.type,),
        )[0]
        return self._propagate_field_record_extent(result, value)

    def _intrinsic_load(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("region",), {})
        region = self._compile_expr(args["region"])
        if region is None:
            raise FrontendError("load expects a View region", self._location(call))
        return self._admit_value(region, call)

    def _admit_value(self, region: Value, node: ast.AST) -> Value:
        if not isinstance(region.type, (SliceType, ViewType)):
            raise FrontendError("load expects a View region", self._location(node))
        encoding = region.type.encoding
        shape = list(region.type.shape)
        axes = list(region.type.axes)
        element: ValueType
        if encoding.kind == "dense":
            scalar_family = encoding.family
            static = getattr(sys.modules["weft.language"], scalar_family)
            element = ScalarType(static)
        else:
            element = encoding
        result_type = value_type(element, tuple(shape), tuple(axes))
        result = self._emit(
            "weft_kernel.admit",
            node,
            operands=(region,),
            result_types=(result_type,),
        )[0]
        self._mark_view_access(region, "read")
        return result

    def _intrinsic_store(self, call: ast.Call) -> None:
        args = self._arguments(call, ("region", "value"), {})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        region = self._compile_expr(args["region"])
        if not isinstance(region, Value):
            raise FrontendError("store destination is a View region", self._location(call))
        if not isinstance(region.type, (SliceType, ViewType)):
            raise FrontendError("store destination is a View region", self._location(call))
        self._mark_view_access(region, "write")
        self._emit("weft_kernel.commit", call, operands=(value, region))
        return None

    def _intrinsic_subview(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("region", "offsets", "extents"), {})
        region = self._expect_value(self._compile_expr(args["region"]), args["region"])
        if not isinstance(region.type, (SliceType, ViewType)):
            raise FrontendError("subview base must be a View region", self._location(call))

        dynamic_bounds: list[Value] = []

        def bounds(name: str) -> tuple[int, ...]:
            node = args[name]
            if isinstance(node, ast.Tuple):
                items = node.elts
            else:
                items = self._eval_static(node) if isinstance(node, ast.expr) else node
                if not isinstance(items, tuple):
                    raise FrontendError(f"subview {name} must be an explicit tuple", self._location(call))
            result: list[int] = []
            for item in items:
                literal = item.value if isinstance(item, ast.Constant) else item
                if type(literal) is int:
                    if literal < 0 or (name == "extents" and literal == 0):
                        raise FrontendError("subview bounds require non-negative offsets and positive extents", self._location(call))
                    result.append(literal)
                    continue
                if not isinstance(item, ast.expr):
                    raise FrontendError("subview bounds must be integer expressions", self._location(call))
                value = self._expect_value(self._compile_expr(item), item)
                if not isinstance(value.type, ScalarType) or value.type.dtype.category not in {
                    DTypeCategory.INTEGER, DTypeCategory.INDEX
                }:
                    raise FrontendError("subview dynamic bounds must be scalar integers", self._location(item))
                dynamic_bounds.append(self._convert_value(value, index, item))
                result.append(-1)
            return tuple(result)

        offsets = bounds("offsets")
        extents = bounds("extents")
        shape = shape_of(region.type)
        if len(offsets) != len(shape) or len(extents) != len(shape):
            raise FrontendError(
                "subview requires one offset and extent per base axis",
                self._location(call),
            )
        for offset, extent, base_extent in zip(offsets, extents, shape):
            if offset >= 0 and extent > 0 and base_extent > 0 and offset + extent > base_extent:
                raise FrontendError(
                    "subview offsets/extents exceed the base region",
                    self._location(call),
                )
        result = self._emit(
            "weft_kernel.subview",
            call,
            operands=(region, *dynamic_bounds),
            result_types=(SliceType(region.type.encoding,
                tuple(extent if extent > 0 else -axis for extent, axis in zip(extents, axes_of(region.type))),
                axes_of(region.type)),),
            attributes={"offsets": _i64_array(offsets), "extents": _i64_array(extents)},
        )[0]
        self._propagate_view_root(result, region)
        return result

    def _intrinsic_extent(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("region", "axis"), {})
        region = self._expect_value(self._compile_expr(args["region"]), args["region"])
        axis = self._eval_static(args["axis"])
        if not isinstance(region.type, (SliceType, ViewType)) or type(axis) is not int or not 0 <= axis < len(shape_of(region.type)):
            raise FrontendError("extent requires a View and an in-range static axis position", self._location(call))
        return self._emit("weft_kernel.extent", call, operands=(region,),
                          result_types=(ScalarType(index),), attributes={"axis": str(axis)})[0]

    def _intrinsic_reshape(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("value", "shape", "axes", "order"), {})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        if not isinstance(value.type, LocalValueType):
            raise FrontendError("reshape requires a shaped local Value", self._location(call))

        def static_tuple(name: str) -> tuple[object, ...]:
            node = args[name]
            result = self._eval_static(node) if isinstance(node, ast.expr) else node
            if not isinstance(result, tuple):
                raise FrontendError(
                    f"reshape {name} must be an explicit tuple",
                    self._location(node if isinstance(node, ast.AST) else call),
                )
            return result

        shape_node = args["shape"]
        symbolic_positions: set[int] = set()
        if isinstance(shape_node, ast.Tuple):
            symbolic_positions = {position for position, item in enumerate(shape_node.elts)
                                  if isinstance(item, ast.Name) and item.id in self._shape_ids}
            raw_shape = tuple(
                self._shape_ids[item.id]
                if isinstance(item, ast.Name) and item.id in self._shape_ids
                else self._eval_static(item)
                for item in shape_node.elts
            )
        else:
            raw_shape = static_tuple("shape")
        raw_axes = static_tuple("axes")
        raw_order = static_tuple("order")
        if not raw_shape or not all(
            isinstance(item, int) and not isinstance(item, bool) and
            (item > 0 or position in symbolic_positions)
            for position, item in enumerate(raw_shape)
        ):
            raise FrontendError(
                "reshape shape must contain positive integers or existing shape symbols",
                self._location(call),
            )
        if len(raw_axes) != len(raw_shape) or not all(
            isinstance(item, str) and item for item in raw_axes
        ):
            raise FrontendError(
                "reshape axes must name every output dimension",
                self._location(call),
            )
        if len(set(raw_axes)) != len(raw_axes):
            raise FrontendError("reshape output axes must be unique", self._location(call))
        if len(raw_order) != len(value.type.axes) or not all(
            isinstance(item, str) and item for item in raw_order
        ):
            raise FrontendError(
                "reshape order must name every input logical axis",
                self._location(call),
            )
        order = tuple(self._axis_id(str(item)) for item in raw_order)
        if len(set(order)) != len(order) or set(order) != set(value.type.axes):
            raise FrontendError(
                "reshape order must be a permutation of the input logical axes",
                self._location(call),
            )
        shape = tuple(int(item) for item in raw_shape)
        dynamic = any(extent < 0 for extent in (*shape, *value.type.shape))
        if dynamic and (shape != value.type.shape or order != value.type.axes):
            raise FrontendError(
                "dynamic reshape supports only a position-preserving axis rename with the same shape symbols",
                self._location(call),
            )
        if not dynamic and math.prod(shape) != math.prod(value.type.shape):
            raise FrontendError(
                "reshape cannot change the number of logical elements",
                self._location(call),
            )
        axes = tuple(self._axis_id(str(item)) for item in raw_axes)
        result_type = LocalValueType(value.type.element_type, shape, axes)
        return self._emit(
            "weft_kernel.reshape",
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes={"order": _i64_array(order)},
        )[0]

    def _intrinsic_widen(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("value", "dtype"), {})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        dtype = self._eval_static(args["dtype"]) if isinstance(args["dtype"], ast.expr) else args["dtype"]
        if not isinstance(dtype, DType):
            raise FrontendError("widen dtype must be a Weft dtype", self._location(call))
        if not isinstance(value.type, (ScalarType, LocalValueType)) or not isinstance(
            element_type(value.type), ScalarType
        ):
            raise FrontendError(
                "widen requires a numeric scalar or local Value", self._location(call)
            )
        result_type = with_element(value.type, ScalarType(dtype))
        result = self._emit(
            "weft_kernel.widen", call, operands=(value,), result_types=(result_type,)
        )[0]
        return self._propagate_field_record_extent(result, value)

    def _intrinsic_narrow(self, call: ast.Call) -> Value:
        args = self._arguments(
            call,
            ("value", "dtype"),
            {"rounding": "rne", "saturation": True},
        )
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        dtype = self._eval_static(args["dtype"]) if isinstance(args["dtype"], ast.expr) else args["dtype"]
        rounding = (
            self._eval_static(args["rounding"])
            if isinstance(args["rounding"], ast.expr)
            else args["rounding"]
        )
        saturation = (
            self._eval_static(args["saturation"])
            if isinstance(args["saturation"], ast.expr)
            else args["saturation"]
        )
        if not isinstance(dtype, DType):
            raise FrontendError("narrow dtype must be a Weft dtype", self._location(call))
        if not isinstance(value.type, (ScalarType, LocalValueType)) or not isinstance(
            element_type(value.type), ScalarType
        ):
            raise FrontendError(
                "narrow requires a numeric scalar or local Value", self._location(call)
            )
        if rounding not in {"rne", "rtz", "rdn", "rup", "dynamic"}:
            raise FrontendError(
                "narrow rounding must be rne, rtz, rdn, rup, or dynamic",
                self._location(call),
            )
        if not isinstance(saturation, bool):
            raise FrontendError("narrow saturation must be a boolean", self._location(call))
        result_type = with_element(value.type, ScalarType(dtype))
        return self._emit(
            "weft_kernel.narrow",
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes={
                "rounding": _string(rounding),
                "saturate": "true" if saturation else "false",
            },
        )[0]

    def _intrinsic_mac_pairs(self, call: ast.Call) -> Value:
        return self._mac_groups(call, 2, "mac_groups")

    def _intrinsic_mac_groups(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("a", "b"), {"n": None, "into": None})
        n = self._eval_static(args["n"]) if isinstance(args["n"], ast.expr) else args["n"]
        if isinstance(n, bool) or not isinstance(n, int):
            raise FrontendError("mac_groups requires integer n", self._location(call))
        return self._mac_groups(call, n, "mac_groups", args)

    def _mac_groups(
        self,
        call: ast.Call,
        group: int,
        operation: str,
        supplied: dict[str, ast.expr | object] | None = None,
    ) -> Value:
        args = supplied or self._arguments(call, ("a", "b"), {"into": None})
        lhs = self._expect_value(self._compile_expr(args["a"]), args["a"])
        rhs = self._expect_value(self._compile_expr(args["b"]), args["b"])
        into = self._eval_static(args["into"]) if isinstance(args["into"], ast.expr) else args["into"]
        if not isinstance(into, DType):
            raise FrontendError(f"{operation} requires an explicit into dtype", self._location(call))
        if not isinstance(element_type(lhs.type), ScalarType) or not isinstance(
            element_type(rhs.type), ScalarType
        ):
            raise FrontendError(
                f"{operation} requires numeric scalar elements", self._location(call)
            )
        lhs_shape, rhs_shape = shape_of(lhs.type), shape_of(rhs.type)
        lhs_axes, rhs_axes = axes_of(lhs.type), axes_of(rhs.type)
        if not lhs_axes or not rhs_axes or lhs_axes[-1] != rhs_axes[-1]:
            raise FrontendError(
                f"{operation} requires one final common grouped axis",
                self._location(call),
            )
        grouped_axis = lhs_axes[-1]
        broadcast_axes = tuple(
            [axis for axis in lhs_axes if axis != grouped_axis]
            + [axis for axis in rhs_axes if axis != grouped_axis and axis not in lhs_axes]
            + [grouped_axis]
        )
        extents: dict[int, int] = {}
        for axes, shape_ in ((lhs_axes, lhs_shape), (rhs_axes, rhs_shape)):
            for axis, extent in zip(axes, shape_):
                if axis in extents and extents[axis] != extent:
                    raise FrontendError(
                        f"{operation} operands disagree on a shared axis extent",
                        self._location(call),
                    )
                extents[axis] = extent
        shape = [extents[axis] for axis in broadcast_axes]
        if group <= 0:
            raise FrontendError(f"{operation} group must be positive", self._location(call))
        if shape[-1] > 0 and shape[-1] % group:
            raise FrontendError(
                f"{operation} input extent must be divisible by {group}",
                self._location(call),
            )
        shape[-1] = shape[-1] // group if shape[-1] > 0 else shape[-1]
        result_type = value_type(ScalarType(into), tuple(shape), broadcast_axes)
        attrs = {"group": str(group), "overflow": _string("wrap")}
        return self._emit(
            f"weft_kernel.{operation}",
            call,
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes=attrs,
        )[0]

    def _intrinsic_reduce(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("value",), {"op": "add", "axis": None})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        op = self._eval_static(args["op"]) if isinstance(args["op"], ast.expr) else args["op"]
        axis = self._eval_static(args["axis"]) if isinstance(args["axis"], ast.expr) else args["axis"]
        shape = list(shape_of(value.type))
        axes = list(axes_of(value.type))
        if not shape:
            raise FrontendError("reduce requires a shaped Value", self._location(call))
        if not isinstance(element_type(value.type), ScalarType):
            raise FrontendError(
                "reduce requires numeric scalar elements", self._location(call)
            )
        position = len(shape) - 1
        if isinstance(axis, str):
            axis_id = self._axis_ids.get(axis.lower())
            if axis_id not in axes:
                raise FrontendError("reduce axis is not present", self._location(call))
            position = axes.index(axis_id)
        elif isinstance(axis, int) and not isinstance(axis, bool):
            if axis < 0 or axis >= len(shape):
                raise FrontendError("reduce axis position is out of range", self._location(call))
            position = axis
        elif axis is not None:
            raise FrontendError("reduce axis is a logical name or position", self._location(call))
        if op not in {"add", "max", "min"}:
            raise FrontendError(
                "reduce op must be add, max, or min", self._location(call)
            )
        shape.pop(position)
        axes.pop(position)
        result_type = value_type(element_type(value.type), tuple(shape), tuple(axes))
        return self._emit(
            "weft_kernel.reduce",
            call,
            operands=(value,),
            result_types=(result_type,),
            attributes={"kind": _string(str(op)), "axis": str(position)},
        )[0]

    def _intrinsic_sum_pairs(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("value",), {})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        shape = list(shape_of(value.type))
        if not shape:
            raise FrontendError("sum_pairs requires a shaped Value", self._location(call))
        fold_element = element_type(value.type)
        if (
            not isinstance(fold_element, ScalarType)
            or fold_element.dtype.category is not DTypeCategory.INTEGER
        ):
            raise FrontendError(
                "sum_pairs requires fixed-width integer elements", self._location(call)
            )
        if shape[-1] > 0 and shape[-1] % 2:
            raise FrontendError("sum_pairs input extent must be even", self._location(call))
        shape[-1] = shape[-1] // 2 if shape[-1] > 0 else shape[-1]
        result_type = value_type(ScalarType(i32), tuple(shape), axes_of(value.type))
        return self._emit(
            "weft_kernel.fold2", call, operands=(value,), result_types=(result_type,)
        )[0]

    def _intrinsic_dot(self, call: ast.Call) -> Value:
        return self._compile_contract(call, "dot")

    def _intrinsic_reduce_dot(self, call: ast.Call) -> Value:
        return self._compile_contract(call, "reduce_dot")

    def _compile_contract(self, call: ast.Call, operation: str) -> Value:
        args = self._arguments(call, ("a", "b"), {"over": None, "acc_dtype": None})
        lhs = self._expect_value(self._compile_expr(args["a"]), args["a"])
        rhs = self._expect_value(self._compile_expr(args["b"]), args["b"])
        over = self._eval_static(args["over"]) if isinstance(args["over"], ast.expr) else args["over"]
        acc = self._eval_static(args["acc_dtype"]) if isinstance(args["acc_dtype"], ast.expr) else args["acc_dtype"]
        if acc is not None and not isinstance(acc, DType):
            raise FrontendError(f"{operation} acc_dtype must be a Weft dtype", self._location(call))
        names = (over,) if isinstance(over, str) else tuple(over or ())
        if not names:
            raise FrontendError(
                f"{operation} requires one or more explicit reduction axes",
                self._location(call),
            )
        reduction_axes: list[int] = []
        for name in names:
            axis = self._axis_ids.get(str(name).lower())
            if axis is None:
                raise FrontendError(f"unknown reduction axis {name!r}", self._location(call))
            reduction_axes.append(axis)
        if operation == "dot":
            lhs_free = set(axes_of(lhs.type)) - set(reduction_axes)
            rhs_free = set(axes_of(rhs.type)) - set(reduction_axes)
            if (
                len(reduction_axes) != 1
                or len(lhs_free) != 1
                or len(rhs_free) != 1
                or lhs_free & rhs_free
            ):
                raise FrontendError(
                    "dot requires two matrix blocks with one reduction axis and distinct row/column axes; "
                    "use reduce_dot for vector, batched, or multi-axis products",
                    self._location(call),
                )
        canonical_op = "outer_contract" if operation == "dot" else "contract"
        return self._contract_value(call, lhs, rhs, reduction_axes, canonical_op, acc)

    def _contract_value(
        self,
        call: ast.Call,
        lhs: Value,
        rhs: Value,
        reduction_axes: Sequence[int],
        operation: str,
        accumulator_dtype: DType | None,
    ) -> Value:
        lhs_axes, rhs_axes = axes_of(lhs.type), axes_of(rhs.type)
        lhs_element = element_type(lhs.type)
        rhs_element = element_type(rhs.type)
        if not isinstance(lhs_element, ScalarType) or not isinstance(
            rhs_element, ScalarType
        ):
            raise FrontendError(
                "contraction operands must have numeric scalar elements",
                self._location(call),
            )
        for axis in reduction_axes:
            if axis not in lhs_axes or axis not in rhs_axes:
                raise FrontendError("contraction axis must occur in both operands", self._location(call))
        output_shape: list[int] = []
        output_axes: list[int] = []
        for shape, axes in ((shape_of(lhs.type), lhs_axes), (shape_of(rhs.type), rhs_axes)):
            for dimension, axis in zip(shape, axes):
                if axis in reduction_axes or axis in output_axes:
                    continue
                output_shape.append(dimension)
                output_axes.append(axis)
        if accumulator_dtype is not None:
            result_element = ScalarType(accumulator_dtype)
        elif isinstance(lhs_element, ScalarType) and isinstance(
            rhs_element, ScalarType
        ):
            result_element = ScalarType(
                self._promote_dtype(lhs_element.dtype, rhs_element.dtype)
            )
        else:
            raise FrontendError(
                "contraction requires numeric operands and an explicit accumulator when promotion is ambiguous",
                self._location(call),
            )
        result_type = value_type(result_element, tuple(output_shape), tuple(output_axes))
        attributes = {"over": _i64_array(reduction_axes)}
        if accumulator_dtype is not None:
            attributes["acc_type"] = emit_type(ScalarType(accumulator_dtype))
        return self._emit(
            f"weft_kernel.{operation}",
            call,
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes=attributes,
        )[0]

    def _intrinsic_lookup(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("table", "idx"), {"bounds": None})
        table = self._expect_value(self._compile_expr(args["table"]), args["table"])
        if isinstance(table.type, (SliceType, ViewType)):
            table = self._admit_value(table, args["table"])
        if not isinstance(table.type, LocalValueType) or not isinstance(
            element_type(table.type), ScalarType
        ):
            raise FrontendError(
                "lookup table must be a shaped numeric local Value",
                self._location(call),
            )
        indices = self._expect_value(self._compile_expr(args["idx"]), args["idx"])
        index_element = element_type(indices.type)
        if (
            not isinstance(indices.type, (ScalarType, LocalValueType))
            or not isinstance(index_element, ScalarType)
            or index_element.dtype.category
            not in {DTypeCategory.INTEGER, DTypeCategory.INDEX}
            or index_element.dtype.signedness == "signed"
        ):
            raise FrontendError(
                "lookup indices must be unsigned integer or index Values",
                self._location(call),
            )
        bounds = (
            self._eval_static(args["bounds"])
            if isinstance(args["bounds"], ast.expr)
            else args["bounds"]
        )
        if bounds != "in_bounds":
            raise FrontendError(
                "lookup currently requires explicit bounds='in_bounds'",
                self._location(call),
            )
        result_type = value_type(element_type(table.type), shape_of(indices.type), axes_of(indices.type))
        return self._emit(
            "weft_kernel.lookup",
            call,
            operands=(table, indices),
            result_types=(result_type,),
            attributes={"bounds": _string(bounds)},
        )[0]

    def _intrinsic_interleave(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("view",), {"rows": None, "field_chunks": None})
        value = self._expect_value(self._compile_expr(args["view"]), args["view"])
        if not isinstance(value.type, ViewType) or self._derive_result is None:
            raise FrontendError("interleave is used by a derived encoding builder", self._location(call))
        rows = self._eval_static(args["rows"]) if isinstance(args["rows"], ast.expr) else args["rows"]
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or self._derive_result.encoding.parameters != (rows,)
        ):
            raise FrontendError(
                "interleave rows must equal the concrete derived Encoding parameter",
                self._location(call),
            )
        info = self._declared_encodings.get(value.type.encoding.family)
        if info is None:
            raise FrontendError("interleave requires a declared base Encoding", self._location(call))
        chunks_node = args["field_chunks"]
        if chunks_node is None:
            chunks = [1] * len(info.fields)
        else:
            try:
                declared = ast.literal_eval(chunks_node)
            except (ValueError, TypeError) as error:
                raise FrontendError("interleave field_chunks must be a static field-to-byte-count map",
                                    self._location(call)) from error
            if not isinstance(declared, dict) or set(declared) != {field.name for field in info.fields}:
                raise FrontendError("interleave field_chunks must name every field exactly once",
                                    self._location(call))
            chunks = [declared[field.name] for field in info.fields]
            if any(type(chunk) is not int or chunk <= 0 for chunk in chunks):
                raise FrontendError("interleave field chunks must be positive byte counts",
                                    self._location(call))
            if any(field.bit_offset % 8 or field.storage_bits % (8 * chunk)
                   for field, chunk in zip(info.fields, chunks)):
                raise FrontendError("interleave chunks must partition byte-addressable field storage",
                                    self._location(call))
        return self._emit(
            "weft_kernel.interleave",
            call,
            operands=(value,),
            result_types=(self._derive_result,),
            attributes={"rows": str(rows), "field_chunks": _i64_array(chunks)},
        )[0]

    def _intrinsic_maximum(self, call: ast.Call) -> Value:
        return self._pointwise_intrinsic(call, "max")

    def _intrinsic_minimum(self, call: ast.Call) -> Value:
        return self._pointwise_intrinsic(call, "min")

    def _intrinsic_maximum_propagating(self, call: ast.Call) -> Value:
        return self._pointwise_intrinsic(call, "maximum")

    def _intrinsic_minimum_propagating(self, call: ast.Call) -> Value:
        return self._pointwise_intrinsic(call, "minimum")

    def _intrinsic_select(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("condition", "true_value", "false_value"), {})
        condition = self._expect_value(self._compile_expr(args["condition"]), args["condition"])
        true_value, false_value = self._compile_binary_operands(args["true_value"], args["false_value"])
        if element_type(condition.type) != ScalarType(i1):
            raise FrontendError("select requires an i1 predicate", self._location(call))
        element = element_type(true_value.type)
        if not isinstance(element, ScalarType) or element != element_type(false_value.type):
            raise FrontendError("select branches require matching numeric elements", self._location(call))
        shape, axes = self._broadcast_domain(true_value.type, false_value.type, call)
        shape, axes = self._broadcast_domain(value_type(element, shape, axes), condition.type, call)
        result = self._emit(
            "weft_kernel.select", call,
            operands=(condition, true_value, false_value),
            result_types=(value_type(element, shape, axes),),
        )[0]
        return self._propagate_field_record_extent(result, true_value, false_value)

    def _pointwise_intrinsic(self, call: ast.Call, kind: str) -> Value:
        args = self._arguments(call, ("a", "b"), {})
        lhs, rhs = self._compile_binary_operands(args["a"], args["b"])
        if kind in {"maximum", "minimum"}:
            element = element_type(lhs.type)
            if not isinstance(element, ScalarType) or element.dtype.category not in {
                DTypeCategory.FLOAT, DTypeCategory.BFLOAT
            }:
                raise FrontendError("NaN-propagating extrema require floating-point elements", self._location(call))
        return self._binary_values(kind, lhs, rhs, call)

    def _intrinsic_exp(self, call: ast.Call) -> Value:
        return self._floating_unary_intrinsic(call, "exp")

    def _intrinsic_rsqrt(self, call: ast.Call) -> Value:
        return self._floating_unary_intrinsic(call, "rsqrt")

    def _intrinsic_log(self, call: ast.Call) -> Value:
        return self._floating_unary_intrinsic(call, "log")

    def _intrinsic_sqrt(self, call: ast.Call) -> Value:
        return self._floating_unary_intrinsic(call, "sqrt")

    def _floating_unary_intrinsic(self, call: ast.Call, kind: str) -> Value:
        args = self._arguments(call, ("value",), {})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        element = element_type(value.type)
        if (
            not isinstance(element, ScalarType)
            or element.dtype.category
            not in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT}
        ):
            raise FrontendError(f"{kind} requires floating-point elements", self._location(call))
        return self._emit(
            "weft_kernel.unary",
            call,
            operands=(value,),
            result_types=(value.type,),
            attributes={"kind": _string(kind)},
        )[0]

    def _intrinsic_abs(self, call: ast.Call) -> Value:
        args = self._arguments(call, ("value",), {})
        value = self._expect_value(self._compile_expr(args["value"]), args["value"])
        if not isinstance(element_type(value.type), ScalarType):
            raise FrontendError("abs requires numeric scalar elements", self._location(call))
        return self._emit(
            "weft_kernel.unary",
            call,
            operands=(value,),
            result_types=(value.type,),
            attributes={"kind": _string("abs")},
        )[0]

    def _birth_kind(self, statement: ast.stmt) -> str | None:
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
        if not isinstance(value, ast.Call):
            return None
        callee = self._resolve_static(value.func)
        if isinstance(callee, Intrinsic) and callee.name in {"state", "stage"}:
            return "state" if callee.name == "state" else "staged"
        return None

    def _compile_level(self, statement: ast.With) -> None:
        if len(statement.items) != 1:
            raise FrontendError("one with statement opens one Level", self._location(statement))
        item = statement.items[0]
        if not isinstance(item.context_expr, ast.Call) or not isinstance(item.optional_vars, ast.Name):
            raise FrontendError("Level syntax is with level.rows(...) as point", self._location(statement))
        constructor = self._resolve_static(item.context_expr.func)
        if not isinstance(constructor, LevelConstructor):
            raise FrontendError("with is reserved for a Weft Level", self._location(statement))
        if self.active_domain is None:
            raise FrontendError("Level requires an enclosing domain", self._location(statement))
        domain_type, extent, partition, multiplicity = self._level_domain(
            constructor.relation, item.context_expr
        )
        domain = self._emit(
            "weft_kernel.domain",
            item.context_expr,
            operands=(self.active_domain, extent, partition, multiplicity),
            result_types=(domain_type,),
        )[0]
        self._domain_partition_values[domain_type.domain_id] = partition
        point_type = DomainPointType(domain_type)

        state_statements = [s for s in statement.body if self._birth_kind(s) == "state"]
        staged_statements = [s for s in statement.body if self._birth_kind(s) == "staged"]
        body_statements = [s for s in statement.body if self._birth_kind(s) is None]
        initialization_unavailable = {
            name: f"Level initialization cannot reference body value {name!r}; state and stage "
            "initializers execute before the Level body"
            for name in _assigned_names(body_statements)
        }
        state_region, state_names, state_values = self._compile_birth_region(
            statement, point_type, item.optional_vars.id, state_statements, "state",
            initialization_unavailable, _assigned_names(staged_statements),
        )
        staged_region, staged_names, staged_values = self._compile_birth_region(
            statement, point_type, item.optional_vars.id, staged_statements, "stage",
            initialization_unavailable, _assigned_names(state_statements),
        )

        outer_env = self.env
        carried_names = sorted(_assigned_names(body_statements) & outer_env.keys())
        carried_values = tuple(outer_env[name] for name in carried_names)
        body = self.builder.region(
            (point_type,)
            + tuple(value.type for value in carried_values)
            + tuple(value.type for value in state_values)
            + tuple(value.type for value in staged_values),
            (item.optional_vars.id,)
            + tuple(carried_names)
            + tuple(state_names)
            + tuple(staged_names),
        )
        saved = (self.block, self.env, self.immutable_names, self.active_domain)
        self.block = body
        self.env = dict(outer_env)
        self.env[item.optional_vars.id] = body.arguments[0]
        cursor = 1
        for name in carried_names:
            self.env[name] = body.arguments[cursor]
            cursor += 1
        for name in state_names:
            self.env[name] = body.arguments[cursor]
            cursor += 1
        self.immutable_names = set(saved[2])
        for name, staged_value in zip(staged_names, staged_values):
            argument = body.arguments[cursor]
            self.env[name] = argument
            self._propagate_field_record_extent(argument, staged_value)
            self.immutable_names.add(name)
            cursor += 1
        self.active_domain = domain
        self._compile_statements(body_statements)
        handed = tuple(self.env[name] for name in carried_names)
        self._emit(
            "weft_kernel.handoff",
            statement,
            operands=handed,
        )
        self.block, self.env, self.immutable_names, self.active_domain = saved
        results = self._emit(
            "weft_kernel.level",
            statement,
            operands=(domain,) + carried_values,
            result_types=tuple(value.type for value in carried_values),
            regions=(state_region, staged_region, body),
            result_names=tuple(carried_names),
        )
        for name, result in zip(carried_names, results):
            self.env[name] = result

    def _compile_birth_region(
        self,
        owner: ast.With,
        point_type: DomainPointType,
        point_name: str,
        statements: list[ast.stmt],
        kind: str,
        unavailable: dict[str, str],
        sibling_names: set[str],
    ) -> tuple[Region, list[str], tuple[Value, ...]]:
        region = self.builder.region((point_type,), (point_name,))
        saved = (self.block, self.env, self.immutable_names)
        saved_unavailable = self._birth_unavailable
        self._birth_unavailable = dict(unavailable)
        self._birth_unavailable.update({
            name: f"{kind} initialization cannot reference {name!r} from the other initialization "
            "region; state and stage initializers have separate scopes"
            for name in sibling_names
        })
        self.block = region
        self.env = dict(saved[1])
        self.env[point_name] = region.arguments[0]
        names: list[str] = []
        values: list[Value] = []
        for statement in statements:
            target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
            if not isinstance(target, ast.Name):
                raise FrontendError(f"{kind} initialization binds one local name", self._location(statement))
            self._compile_statement(statement, set())
            names.append(target.id)
            values.append(self.env[target.id])
        self._emit("weft_kernel.births_yield", owner, operands=tuple(values))
        self.block, self.env, self.immutable_names = saved
        self._birth_unavailable = saved_unavailable
        return region, names, tuple(values)

    def _source_auto_value(self, specification: AutoSpec, node: ast.AST) -> tuple[Value, str | int]:
        if len(specification.choices) == 1 and isinstance(
            specification.choices[0], str
        ):
            name = specification.choices[0]
            choices: tuple[int, ...] = ()
            spelling: str | int = name
        elif specification.choices and all(
            isinstance(choice, int)
            and not isinstance(choice, bool)
            and choice > 0
            for choice in specification.choices
        ):
            choices = tuple(int(choice) for choice in specification.choices)
            name = "auto$" + "$".join(str(choice) for choice in choices)
            spelling = name
        else:
            raise FrontendError(
                "auto is one symbolic parameter or a finite list of positive integers",
                self._location(node),
            )
        existing = self._auto_symbols.get(name)
        if existing is not None:
            if self._auto_choices[name] != choices:
                raise FrontendError(
                    f"auto parameter {name!r} has inconsistent choices",
                    self._location(node),
                )
            return existing, spelling
        if self._kernel_body is None:
            raise AssertionError("source auto parameter requires a kernel body")
        operation = self.builder.operation(
            "weft_kernel.symbol",
            self._location(node),
            result_types=(ScalarType(index),),
            attributes={
                "name": _string(name),
                "kind": _string("source_auto"),
                "choices": _i64_array(choices),
            },
            result_names=(name,),
        )
        self._kernel_body.operations.insert(0, operation)
        value = operation.results[0]
        self._auto_symbols[name] = value
        self._auto_choices[name] = choices
        return value, spelling

    def _level_domain(
        self, relation: str, call: ast.Call
    ) -> tuple[DomainType, Value, Value, Value]:
        arguments = list(call.args)
        keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
        parent = self.active_domain.type
        assert isinstance(parent, DomainType)
        extent_symbol: str | None = None
        if arguments:
            source = arguments[0]
            if isinstance(source, ast.Name) and source.id in self.env and isinstance(
                self.env[source.id].type, DomainPointType
            ):
                source_domain = self.env[source.id].type.domain
                axis_name = source_domain.axis_name.split(".")[0]
                axis_id = source_domain.axis_id
                extent_value = self._domain_partition_values[source_domain.domain_id]
            elif isinstance(source, ast.Name):
                axis_name = source.id.lower()
                axis_id = self._axis_id(source.id)
                extent_value = self._shape_symbols[source.id]
                extent_symbol = source.id
            else:
                raise FrontendError("Level domain is a shape symbol or parent point", self._location(source))
        else:
            axis_name = parent.axis_name.split(".")[0]
            axis_id = parent.axis_id
            extent_value = self._domain_partition_values[parent.domain_id]
        parameter_name = "group" if relation in {"rows", "cols"} else "extent"
        if parameter_name not in keywords:
            public_relation = "subtiles" if relation == "subs" else relation
            raise FrontendError(f"level.{public_relation} requires {parameter_name}=", self._location(call))
        parameter = self._eval_static(keywords[parameter_name])
        if isinstance(parameter, AutoSpec):
            partition_value, partition_spelling = self._source_auto_value(
                parameter, keywords[parameter_name]
            )
        elif isinstance(parameter, int) and parameter > 0:
            partition_value = self._constant(
                parameter, index, keywords[parameter_name]
            )
            partition_spelling = parameter
        else:
            raise FrontendError("Level partition is a positive integer or auto declaration", self._location(call))
        multiplicity = self._emit(
            "arith.ceildivui",
            call,
            operands=(extent_value, partition_value),
            result_types=(ScalarType(index),),
        )[0]
        serial = self._next_domain_serial
        self._next_domain_serial += 1
        tail = "tail"
        if isinstance(partition_spelling, int):
            extent_spelling = self._domain_partition_spelling.get(parent.domain_id)
            if isinstance(extent_spelling, int) and extent_spelling % partition_spelling == 0:
                tail = "exact"
            elif (
                extent_symbol is not None
                and extent_symbol in self._shape_multiples
                and self._shape_multiples[extent_symbol] % partition_spelling == 0
            ):
                tail = "exact"
        domain = DomainType(
            f"{axis_name}.{serial}",
            serial,
            parent.domain_id,
            axis_id,
            relation,
            tail,
        )
        self._domain_partition_spelling[serial] = partition_spelling
        return domain, extent_value, partition_value, multiplicity

    def _compile_for(self, statement: ast.For) -> None:
        if statement.orelse or not isinstance(statement.target, ast.Name) or not isinstance(statement.iter, ast.Call):
            raise FrontendError("ordinary for uses Python range without for-else", self._location(statement))
        callee = self._resolve_static(statement.iter.func)
        if callee is not range:
            raise FrontendError("ordinary for iterates Python range", self._location(statement.iter))
        if not 1 <= len(statement.iter.args) <= 3 or statement.iter.keywords:
            raise FrontendError("range expects one to three positional bounds", self._location(statement.iter))
        args = statement.iter.args
        if len(args) == 1:
            lower = self._constant(0, index, statement)
            upper = self._expect_value(self._compile_expr(args[0]), args[0])
            step = self._constant(1, index, statement)
        else:
            lower = self._expect_value(self._compile_expr(args[0]), args[0])
            upper = self._expect_value(self._compile_expr(args[1]), args[1])
            step = self._expect_value(self._compile_expr(args[2]), args[2]) if len(args) == 3 else self._constant(1, index, statement)
        carried_names = sorted(_assigned_names(statement.body) & self.env.keys())
        init_values = tuple(self.env[name] for name in carried_names)
        body = self.builder.region(
            (ScalarType(index),) + tuple(value.type for value in init_values),
            (statement.target.id,) + tuple(carried_names),
        )
        saved = (self.block, self.env)
        self.block = body
        self.env = dict(saved[1])
        self.env[statement.target.id] = body.arguments[0]
        for name, argument in zip(carried_names, body.arguments[1:]):
            self.env[name] = argument
        self._compile_statements(statement.body)
        self._emit(
            "scf.yield",
            statement,
            operands=tuple(self.env[name] for name in carried_names),
        )
        self.block, self.env = saved
        results = self._emit(
            "scf.for",
            statement,
            operands=(lower, upper, step) + init_values,
            result_types=tuple(value.type for value in init_values),
            regions=(body,),
            result_names=tuple(carried_names),
        )
        for name, result in zip(carried_names, results):
            self.env[name] = result

    def _compile_if(self, statement: ast.If, live_after: set[str]) -> None:
        condition = self._expect_value(self._compile_expr(statement.test), statement.test)
        if condition.type != ScalarType(i1):
            raise FrontendError("if condition is scalar i1", self._location(statement.test))
        then_assigned = _assigned_names(statement.body)
        else_assigned = _assigned_names(statement.orelse)
        branch_defined = then_assigned | else_assigned
        missing = sorted(
            (branch_defined & live_after)
            - set(self.env)
            - (then_assigned & else_assigned)
        )
        if missing:
            raise FrontendError(
                f"if value {missing[0]!r} is live after the branch but is not defined on both paths",
                self._location(statement),
            )
        merged_names = sorted(
            branch_defined
            & (set(self.env) | live_after)
        )
        outer_env = self.env

        def branch(statements: Sequence[ast.stmt]) -> tuple[Region, tuple[Value, ...]]:
            region = self.builder.region(())
            saved = (self.block, self.env)
            self.block, self.env = region, dict(outer_env)
            self._compile_statements(statements)
            values = tuple(self.env[name] for name in merged_names)
            self._emit("scf.yield", statement, operands=values)
            self.block, self.env = saved
            return region, values

        then_region, then_values = branch(statement.body)
        else_region, else_values = branch(statement.orelse)
        if tuple(value.type for value in then_values) != tuple(value.type for value in else_values):
            raise FrontendError("if branch result types must match", self._location(statement))
        results = self._emit(
            "scf.if",
            statement,
            operands=(condition,),
            result_types=tuple(value.type for value in then_values),
            regions=(then_region, else_region),
            result_names=tuple(merged_names),
        )
        for name, result in zip(merged_names, results):
            self.env[name] = result

    def _compile_while(self, statement: ast.While) -> None:
        if statement.orelse:
            raise FrontendError("while-else is not part of Weft", self._location(statement))
        carried_names = sorted(_assigned_names(statement.body) & self.env.keys())
        init_values = tuple(self.env[name] for name in carried_names)
        condition_region = self.builder.region(
            tuple(value.type for value in init_values), tuple(carried_names)
        )
        body_region = self.builder.region(
            tuple(value.type for value in init_values), tuple(carried_names)
        )
        saved = (self.block, self.env)
        self.block, self.env = condition_region, dict(saved[1])
        for name, argument in zip(carried_names, condition_region.arguments):
            self.env[name] = argument
        condition = self._expect_value(self._compile_expr(statement.test), statement.test)
        if condition.type != ScalarType(i1):
            raise FrontendError("while condition is scalar i1", self._location(statement.test))
        self._emit(
            "scf.condition",
            statement,
            operands=(condition,) + condition_region.arguments,
        )
        self.block, self.env = body_region, dict(saved[1])
        for name, argument in zip(carried_names, body_region.arguments):
            self.env[name] = argument
        self._compile_statements(statement.body)
        self._emit(
            "scf.yield",
            statement,
            operands=tuple(self.env[name] for name in carried_names),
        )
        self.block, self.env = saved
        results = self._emit(
            "scf.while",
            statement,
            operands=init_values,
            result_types=tuple(value.type for value in init_values),
            regions=(condition_region, body_region),
            result_names=tuple(carried_names),
        )
        for name, result in zip(carried_names, results):
            self.env[name] = result

    def _type_signature(self, value_type_: ValueType) -> tuple[object, ...]:
        if isinstance(value_type_, (ViewType, SliceType)):
            return (
                "view",
                self._type_signature(value_type_.encoding),
                len(value_type_.shape),
            )
        if isinstance(value_type_, ScalarType):
            return ("scalar", value_type_.dtype.name)
        if isinstance(value_type_, EncodingType):
            return (
                "encoding",
                value_type_.family,
                value_type_.kind,
                value_type_.layout_identity,
                value_type_.parameters,
            )
        if isinstance(value_type_, LocalValueType):
            element = self._type_signature(value_type_.element_type)
            return ("value", element, len(value_type_.shape))
        return ("unknown",)

    def _expression_signature(self, expression: ast.expr) -> tuple[object, ...] | None:
        if isinstance(expression, ast.Constant):
            if isinstance(expression.value, bool):
                return ("scalar", i1.name)
            if isinstance(expression.value, int):
                return ("scalar", i32.name)
            if isinstance(expression.value, float):
                return ("scalar", f32.name)
        if (
            isinstance(expression, ast.UnaryOp)
            and isinstance(expression.op, (ast.UAdd, ast.USub))
            and isinstance(expression.operand, ast.Constant)
        ):
            return self._expression_signature(expression.operand)
        if isinstance(expression, ast.Call):
            try:
                callee = self._resolve_static(expression.func)
            except FrontendError:
                callee = None
            if isinstance(callee, DType):
                return ("scalar", callee.name)
        if isinstance(expression, ast.Name):
            if expression.id in self.env:
                return self._type_signature(self.env[expression.id].type)
            try:
                static = self._resolve_static(expression)
            except FrontendError:
                return None
        return None

    def _annotation_signature(
        self, source: FunctionSource, annotation: ast.expr | None
    ) -> tuple[object, ...] | None:
        if annotation is None:
            return None
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        if isinstance(annotation, ast.Name):
            static = source.resolve(annotation)
            if isinstance(static, DType):
                return ("scalar", static.name)
        if isinstance(annotation, ast.Subscript) and source.resolve(annotation.value) is View:
            items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            if len(items) != 2:
                return None
            encoding_node = items[0]
            if isinstance(encoding_node, ast.Subscript):
                encoding = source.resolve(encoding_node.value)
                parameters = (
                    list(encoding_node.slice.elts)
                    if isinstance(encoding_node.slice, ast.Tuple)
                    else [encoding_node.slice]
                )
            else:
                encoding = source.resolve(encoding_node)
                parameters = []
            if isinstance(encoding, DType):
                encoding_signature: tuple[object, ...] = (
                    "encoding",
                    encoding.name,
                    "dense",
                    f"dense.{encoding.name}",
                    (),
                )
            elif isinstance(encoding, EncodingDefinition):
                encoding_signature = (
                    "encoding",
                    encoding.__name__,
                    "base",
                    encoding.__name__,
                    (),
                )
            elif isinstance(encoding, DerivedEncodingDefinition):
                try:
                    values = tuple(int(ast.literal_eval(value)) for value in parameters)
                except (ValueError, TypeError):
                    return None
                identity = encoding.__name__ + "[" + ",".join(
                    str(value) for value in values
                ) + "]"
                encoding_signature = (
                    "encoding",
                    encoding.__name__,
                    "derived_instance",
                    identity,
                    values,
                )
            else:
                return None
            shape = items[1].elts if isinstance(items[1], (ast.Tuple, ast.List)) else [items[1]]
            return ("view", encoding_signature, len(shape))
        return None

    def _call_nodes_for_definition(
        self, definition: InlineDefinition, call: ast.Call
    ) -> dict[str, ast.expr] | None:
        parameters = list(definition.signature.parameters.values())
        if len(call.args) > len(parameters) or any(keyword.arg is None for keyword in call.keywords):
            return None
        nodes = {parameter.name: argument for parameter, argument in zip(parameters, call.args)}
        for keyword in call.keywords:
            assert keyword.arg is not None
            if keyword.arg not in definition.signature.parameters or keyword.arg in nodes:
                return None
            nodes[keyword.arg] = keyword.value
        for parameter in parameters:
            if parameter.name not in nodes and parameter.default is inspect.Parameter.empty:
                return None
        return nodes

    def _overload_matches(self, definition: InlineDefinition, call: ast.Call) -> bool:
        nodes = self._call_nodes_for_definition(definition, call)
        if nodes is None:
            return False
        source = FunctionSource.from_definition(definition)
        ast_parameters = {
            parameter.arg: parameter
            for parameter in source.function.args.args + source.function.args.kwonlyargs
        }
        for name, node in nodes.items():
            expected = self._annotation_signature(source, ast_parameters[name].annotation)
            actual = self._expression_signature(node)
            if expected is not None and actual != expected:
                return False
            parameter = definition.signature.parameters[name]
        return True

    def _select_overload(self, overloads: OverloadSet, call: ast.Call) -> InlineDefinition:
        matching = [
            definition
            for definition in overloads.definitions
            if self._overload_matches(definition, call)
        ]
        if len(matching) != 1:
            raise FrontendError(
                f"overload {overloads.name!r} is not uniquely selected by typed arguments",
                self._location(call),
            )
        return matching[0]

    def _inline_argument(
        self,
        expression: ast.expr,
        parameter: inspect.Parameter,
    ) -> Value | object:
        static_default = parameter.default
        if isinstance(static_default, (AutoSpec, DType, str, int)):
            return self._eval_static(expression)
        if isinstance(expression, (ast.Name, ast.Attribute, ast.Call)):
            try:
                static = self._eval_static(expression)
            except FrontendError:
                static = None
            if isinstance(static, (AutoSpec, DType)):
                return static
        return self._expect_value(self._compile_expr(expression), expression)

    def _inline_function(self, definition: InlineDefinition, call: ast.Call) -> Value | None:
        source = FunctionSource.from_definition(definition)
        parameters = source.function.args.args + source.function.args.kwonlyargs
        nodes = self._call_nodes_for_definition(definition, call)
        if nodes is None:
            raise FrontendError(f"inline function {definition.name} argument count mismatch", self._location(call))
        bound: dict[str, Value] = {}
        static_bound: dict[str, object] = {}
        for parameter_ast in parameters:
            parameter = definition.signature.parameters[parameter_ast.arg]
            if parameter_ast.arg in nodes:
                value = self._inline_argument(nodes[parameter_ast.arg], parameter)
            else:
                value = parameter.default
            if isinstance(value, Value):
                bound[parameter_ast.arg] = value
            else:
                static_bound[parameter_ast.arg] = value
        saved = (
            self.source, self.env, self.static_env, self.immutable_names,
            self._birth_unavailable,
        )
        self.source = source
        self.env = dict(self._shape_symbols)
        self.env.update(bound)
        self.static_env = dict(saved[2])
        self.static_env.update(static_bound)
        self.immutable_names = set()
        self._birth_unavailable = {}
        try:
            body = source.function.body
            if body and isinstance(body[-1], ast.Return):
                self._compile_statements(body[:-1])
                return_node = body[-1]
                assert isinstance(return_node, ast.Return)
                return (
                    self._expect_value(self._compile_expr(return_node.value), return_node.value)
                    if return_node.value is not None
                    else None
                )
            self._compile_statements(body)
            return None
        finally:
            (
                self.source, self.env, self.static_env, self.immutable_names,
                self._birth_unavailable,
            ) = saved


def lower_to_mlir(definition: KernelDefinition[object, object]) -> str:
    if not isinstance(definition, KernelDefinition):
        raise TypeError("lower_to_mlir expects an @weft.kernel definition")
    return FrontendCompiler(definition).compile()
