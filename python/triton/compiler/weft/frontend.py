"""Triton semantic implementation producing Canonical Kernel operations."""
from __future__ import annotations

import json

from triton.language import core as tl
from triton.language.semantic import TritonSemantic
from triton.language.weft import dtypes
from triton.language.weft.builtins import Intrinsic
from triton.language.weft.declarations import OverloadSet
from triton.language.weft.dtypes import DType

from .ir import Value
from .source import FunctionSource, normalize_static
from .types import ScalarType, ViewType, SliceType, LocalValueType, axes_of, element_type, shape_of, value_type
from .values import LogicalTensor, LogicalObject, LogicalType, FiniteDType, unwrap, wrap
from .pointers import CanonicalTensor, Address


def is_logical(value):
    return isinstance(value, tl.tensor) and isinstance(value.type, LogicalType)


class CanonicalSemantic(TritonSemantic):
    # Native function boundaries carry complete selected scalar/RVV parts.
    supports_noinline_tensors = True
    tensor = CanonicalTensor
    def __init__(self, extension):
        super().__init__(extension.builder)
        self.extension = extension
        self.closed_inductions = set()
        self.helper_returns = []

    @property
    def node(self):
        return self.extension.generator.cur_node

    def to_tensor(self, value, check_type=True):
        if isinstance(value, tl.tensor):
            return value
        if isinstance(value, LogicalObject):
            return value
        if isinstance(value, Value):
            return wrap(value)
        return super().to_tensor(value, check_type)

    def value_context(self, value):
        if isinstance(value, LogicalObject) and isinstance(value.handle.type, SliceType) and not value.handle.type.shape:
            return wrap(self.extension._admit_value(value.handle, self.node))
        return value

    def normalize_join_value(self, value, reference):
        if (isinstance(value, tl.tensor) and isinstance(reference, tl.tensor) and
                isinstance(value.handle, Value) and isinstance(reference.handle, Value) and
                value.handle.type == reference.handle.type):
            # Ordinary and explicit-model values can name the same Canonical
            # scalar/axes. Reconstruct the existing frontend protocol; no cast,
            # axis rewrite, or numerical operation is permitted at this join.
            return reference.type._unflatten_ir([value.handle], 0)[0]
        return value

    def is_loop_livein(self, value):
        return not (isinstance(value, tl.tensor) and isinstance(value.handle, Value) and
                    value.handle in self.closed_inductions)

    def finish_loop_induction(self, placeholder):
        # A completed loop's placeholder is not a dominating initial value for
        # a later loop that reuses the name. An actual escaped use remains an
        # undefined SSA operand and is rejected when constructing Canonical IR.
        self.closed_inductions.add(placeholder)

    def scalar_constant(self, value, dtype):
        canonical = dtype.canonical_dtype if isinstance(dtype, FiniteDType) else normalize_static(dtype)
        return tl.tensor(self.extension._constant(value, canonical, self.node), dtype)

    def cast(self, value, dtype, fp_downcast_rounding=None, bitcast=False):
        if bitcast:
            return self.bitcast(value, dtype)
        target = dtype.scalar if dtype.is_block() else dtype
        if target.is_ptr():
            if isinstance(value.handle, Address) and target == value.type.scalar:
                return value
            if isinstance(value.handle, Address) and target.address_space == value.type.scalar.address_space:
                before, after = value.type.scalar.element_ty, target.element_ty
                if max(8, before.primitive_bitwidth) != max(8, after.primitive_bitwidth):
                    raise NotImplementedError("pointer casts changing storage width require an explicit alignment contract")
                base = self.builder.op("weft_kernel.pointer_cast", (value.handle.base,), (target.to_ir(self.builder),))
                ty = tl.block_type(target, list(value.type.shape)) if value.type.is_block() else target
                return self.tensor(Address(base, value.handle.offset), ty)
            raise NotImplementedError("pointer reinterpretation requires an explicit storage contract")
        if isinstance(value.handle, Address):
            raise NotImplementedError("pointer to integer conversion has no Canonical storage provenance contract")
        if not is_logical(value) and not isinstance(target, FiniteDType) and not isinstance(value.dtype, FiniteDType):
            return super().cast(value, dtype, fp_downcast_rounding)
        canonical = target.canonical_dtype if isinstance(target, FiniteDType) else normalize_static(target)
        if target == tl.int1 and value.dtype != tl.int1:
            zero = self.extension._constant(0, element_type(value.handle.type).dtype, self.node)
            handle = self.builder.compare("ne", value.handle, zero)
            ty = tl.block_type(tl.int1, list(value.type.shape)) if value.type.is_block() else tl.int1
            return wrap(handle) if is_logical(value) else tl.tensor(handle, ty)
        if fp_downcast_rounding not in (None, "rtne", "rtz"):
            raise NotImplementedError(f"unsupported floating conversion rounding {fp_downcast_rounding}")
        if fp_downcast_rounding == "rtz":
            handle = self.builder.op("weft_kernel.narrow", (value.handle,),
                                     (value_type(ScalarType(canonical), shape_of(value.handle.type), axes_of(value.handle.type)),),
                                     {"rounding": '"rtz"', "saturate": "false"})
        else:
            handle = self.extension._convert_value(value.handle, canonical, self.node)
        if is_logical(value):
            return wrap(handle)
        result_type = tl.block_type(target, list(value.type.shape)) if value.type.is_block() else target
        return tl.tensor(handle, result_type)

    def bitcast(self, value, dtype):
        if not is_logical(value) and not isinstance(value.dtype, FiniteDType) and not isinstance(dtype.scalar, FiniteDType):
            return super().bitcast(value, dtype)
        target = normalize_static(dtype.scalar)
        result_type = value_type(ScalarType(target), shape_of(value.handle.type), axes_of(value.handle.type))
        handle = self.builder.op("weft_kernel.bitcast", (value.handle,), (result_type,))
        return wrap(handle) if is_logical(value) else tl.tensor(handle, value.type.with_element_ty(dtype))

    def assume(self, condition):
        if condition.type != tl.int1:
            raise TypeError("assume requires a scalar boolean condition")
        self.builder.op("weft_kernel.assume", (condition.handle,))

    def _axis_fact(self, value, values, kind):
        rank = max(1, len(value.shape)) if kind == "divisibility" else len(value.shape)
        if len(values) != rank or any(type(n) is not int or n <= 0 for n in values):
            raise ValueError(f"{kind} requires one positive integer per logical axis")
        handle = value.handle
        operands = (handle.base, handle.offset) if isinstance(handle, Address) else (handle,)
        self.builder.op("weft_kernel.value_fact", operands,
                        attrs={"kind": json.dumps(kind), "values": "array<i64" +
                               (": " + ", ".join(map(str, values)) if values else "") + ">"})
        return value

    def multiple_of(self, value, values):
        return self._axis_fact(value, values, "divisibility")

    def max_contiguous(self, value, values):
        return self._axis_fact(value, values, "contiguity")

    def max_constancy(self, value, values):
        return self._axis_fact(value, values, "constancy")

    def load(self, ptr, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy, is_volatile):
        if isinstance(ptr, LogicalObject) and isinstance(ptr.handle.type, (ViewType, SliceType)):
            if mask is not None or other is not None or boundary_check or padding_option or is_volatile:
                raise ValueError("View loads use their explicit domain; pointer masks require pointer operands")
            if cache_modifier or eviction_policy:
                raise NotImplementedError("cache modifiers for View loads")
            value = self.extension._admit_value(ptr.handle, self.node)
            named_axes = {identity for name, identity in self.extension._axis_ids.items()
                          if not name.startswith("$")}
            if (isinstance(element_type(value.type), ScalarType) and shape_of(value.type) and
                    all(n > 0 and n & (n - 1) == 0 for n in shape_of(value.type)) and
                    not named_axes.intersection(axes_of(value.type))):
                shape = shape_of(value.type)
                value = self.builder.reshape_axes(value, shape, self.builder.positional_axes(len(shape)))
                from .values import logical_dtype
                return tl.tensor(value, tl.block_type(logical_dtype(element_type(value.type).dtype), list(shape)))
            return wrap(value)
        return super().load(ptr, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy, is_volatile)

    def store(self, ptr, value, mask, boundary_check, cache_modifier, eviction_policy):
        if isinstance(ptr, LogicalObject) and isinstance(ptr.handle.type, (ViewType, SliceType)):
            if mask is not None or boundary_check or cache_modifier or eviction_policy:
                raise ValueError("View stores use their explicit domain; pointer masks require pointer operands")
            self.extension._mark_view_access(ptr.handle, "write")
            stored = value.handle
            if not is_logical(value) and shape_of(stored.type):
                if shape_of(stored.type) != shape_of(ptr.handle.type):
                    raise ValueError("a positional tensor stored into a View must match its logical shape")
                stored = self.builder.reshape_axes(stored, shape_of(ptr.handle.type), axes_of(ptr.handle.type))
            self.builder.op("weft_kernel.commit", (stored, ptr.handle))
            return None
        return super().store(ptr, value, mask, boundary_check, cache_modifier, eviction_policy)

    def intrinsic_call(self, node, callee):
        if isinstance(callee, (Intrinsic, DType, OverloadSet)):
            return True, wrap(self.extension._compile_call(node))
        return False, None

    def call_helper(self, fn, args, kwargs):
        if fn.is_gluon():
            raise TypeError("Gluon helpers have an explicit GPU programming model")
        if fn.explicit_model and fn.model != "weft":
            raise TypeError(f"cannot call an explicitly {fn.model} helper from the weft programming model")
        generator = self.extension.generator
        extension = self.extension
        bound = fn.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        source = FunctionSource.from_definition(fn)
        saved = {name: getattr(generator, name) for name in (
            "jit_fn", "gscope", "lscope", "local_defs", "pending_annotations", "return_vals", "return_ips",
            "file_name", "begin_line", "begin_col")}
        saved_extension = extension.source, extension.static_env, extension.immutable_names, extension._birth_unavailable
        insertion = generator._get_insertion_point_and_loc()
        generator.jit_fn = fn
        generator.gscope = fn.get_capture_scope()
        from types import ModuleType
        modules = generator.builder.module_map
        generator.gscope = {name: modules.get(value.__name__, value) if isinstance(value, ModuleType) else
                           getattr(modules[value.__module__], value.__name__) if getattr(value, "__module__", "") in modules else value
                           for name, value in generator.gscope.items()}
        generator.lscope = {}
        generator.local_defs = {}
        generator.pending_annotations = {}
        generator.return_vals, generator.return_ips = [], []
        generator.file_name, generator.begin_line, generator.begin_col = fn.file_name, fn.def_file_line_number - 1, fn.def_file_col_number
        extension.source = source
        extension.static_env = {}
        extension.immutable_names, extension._birth_unavailable = set(), {}
        dimensions = {}
        annotations = {p.arg: p.annotation for p in source.function.args.args + source.function.args.kwonlyargs}
        try:
            for name, value in bound.arguments.items():
                value = tl.constexpr(value) if not isinstance(value, tl.base_value) else value
                if isinstance(value, tl.constexpr):
                    extension.static_env[name] = normalize_static(value.value)
                elif isinstance(value, (tl.tensor, LogicalObject)):
                    extension._check_helper_value(source, annotations[name], value.handle, dimensions, self.node)
                generator.set_value(name, value)
            from .helpers import requires_function_region, compile_function_region
            if fn.noinline or requires_function_region(source.function.body):
                result = compile_function_region(self, fn, source.function.body)
            else:
                result = generator.visit_structured_inline_body(source.function.body)
            canonical_result = unwrap(result)
            if isinstance(canonical_result, Value):
                extension._check_helper_value(source, source.function.returns, canonical_result, dimensions, self.node)
            return result
        finally:
            for name, value in saved.items():
                setattr(generator, name, value)
            extension.source, extension.static_env, extension.immutable_names, extension._birth_unavailable = saved_extension
            # Keep the inlined operations in the caller's block. Only source
            # location and scope change across this source-function boundary.
            self.builder.set_loc(insertion[1])

    def helper_return(self, node):
        from triton.compiler.code_generator import flatten_values_to_ir
        value = self.extension.generator.visit(node.value)
        if value is None:
            value = tl.constexpr(None)
        self.builder.op("weft_kernel.helper_return", flatten_values_to_ir([value]))
        operation = self.builder.block.operations[-1]
        self.helper_returns[-1].append((value, self.builder.block, operation))

    def named_dot(self, lhs, rhs, over, acc_dtype):
        over = unwrap(over)
        names = (over,) if isinstance(over, str) else tuple(over)
        axes = [self.extension._axis_ids[name.lower()] for name in names]
        if len(axes) != 1 or len(shape_of(lhs.handle.type)) != 2 or len(shape_of(rhs.handle.type)) != 2:
            raise ValueError("named dot requires two matrices and one reduction axis")
        free_lhs, free_rhs = set(axes_of(lhs.handle.type)) - set(axes), set(axes_of(rhs.handle.type)) - set(axes)
        if free_lhs & free_rhs:
            raise ValueError("named dot requires distinct free axes; use reduce_dot for a shared batch")
        dtype = normalize_static(acc_dtype) if acc_dtype is not None else None
        return wrap(self.extension._contract_value(self.node, lhs.handle, rhs.handle, axes, "outer_contract", dtype))

    def reduction(self, inputs, axis, region_builder_fn):
        if not inputs:
            raise ValueError("reduction requires at least one input")
        if any(tuple(v.shape) != tuple(inputs[0].shape) for v in inputs):
            raise ValueError("reduction inputs must have the same shape")
        if axis is None:
            inputs = tuple(self.reshape(v, [v.numel.value], False) for v in inputs)
            axis = 0
        value = inputs[0]
        if not 0 <= axis < len(value.shape):
            raise ValueError("reduction axis is outside the input rank")
        region = self.builder.region(())

        class Combiner:
            def get_region(self, index):
                if index != 0:
                    raise IndexError(index)
                return region

        point, loc = self.builder.get_insertion_point(), self.builder.get_loc()
        try:
            region_builder_fn(Combiner())
        finally:
            self.builder.restore_insertion_point(point)
            self.builder.set_loc(loc)
        source = value.handle.type
        shape = tuple(n for i, n in enumerate(shape_of(source)) if i != axis)
        axes = tuple(a for i, a in enumerate(axes_of(source)) if i != axis)
        types = tuple(value_type(element_type(v.handle.type), shape, axes) for v in inputs)
        simple = None
        if len(inputs) == 1 and len(region.operations) == 2:
            operation, terminator = region.operations
            if (operation.name == "weft_kernel.binary" and operation.operands == region.arguments and
                    terminator.name == "weft_kernel.reduce_yield" and terminator.operands == operation.results):
                kind = json.loads(operation.attributes["kind"])
                if kind in ("add", "max", "min"):
                    simple = kind
        if simple is not None:
            results = (self.builder.op("weft_kernel.reduce", (value.handle,), types,
                                       {"kind": json.dumps(simple), "axis": str(axis)}),)
        else:
            results = self.builder.op("weft_kernel.reduce_region", tuple(v.handle for v in inputs), types,
                                      {"axis": str(axis)}, (region,))
            if len(inputs) == 1:
                results = (results,)
        return tuple(tl.tensor(self.builder.reshape_axes(result, shape, self.builder.positional_axes(len(shape))),
                               tl.block_type(v.dtype, list(shape)) if shape else v.dtype)
                     for v, result in zip(inputs, results))

    def logical_arange(self, start, end, dtype, axis):
        dtype = dtypes.u32 if dtype is None else normalize_static(dtype)
        if dtype.category not in (dtypes.DTypeCategory.INTEGER, dtypes.DTypeCategory.INDEX) or dtype.signedness == "signed":
            raise ValueError("named arange requires an unsigned integer or index dtype")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError("arange requires an increasing constant integer interval")
        identity = self.extension._axis_id(axis) if axis is not None else self.extension._next_axis_id
        if axis is None:
            self.extension._next_axis_id += 1
        ty = LocalValueType(ScalarType(dtype), (end - start,), (identity,))
        return wrap(self.builder.op("weft_kernel.iota", types=(ty,), attrs={"start": str(start), "end": str(end)}))

    def named_reduce(self, value, axis, op):
        ty = value.handle.type
        if not shape_of(ty) or not isinstance(element_type(ty), ScalarType):
            raise ValueError("logical reduction requires a shaped numeric Value")
        if op not in {"add", "max", "min"}:
            raise ValueError("logical reduction op must be add, max, or min")
        if axis is None:
            axis = len(axes_of(ty)) - 1
        if isinstance(axis, str):
            identity = self.extension._axis_ids.get(axis.lower())
        elif type(axis) is int and -len(axes_of(ty)) <= axis < len(axes_of(ty)):
            identity = axes_of(ty)[axis]
        else:
            raise ValueError("logical reduction needs an axis name or in-range position")
        if identity not in axes_of(ty):
            raise ValueError(f"reduction axis {axis!r} is absent from the input")
        result = value_type(element_type(ty), tuple(n for n, a in zip(shape_of(ty), axes_of(ty)) if a != identity),
                            tuple(a for a in axes_of(ty) if a != identity))
        return wrap(self.builder.op("weft_kernel.reduce", (value.handle,), (result,),
                                    {"axis": str(axes_of(ty).index(identity)), "kind": json.dumps(op)}))

    def logical_reduce(self, value, axis, keep_dims):
        if not is_logical(value) or keep_dims:
            return super().logical_reduce(value, axis, keep_dims)
        if axis is None:
            axis = len(shape_of(value.handle.type)) - 1
        return self.named_reduce(value, axis, "add")

    def named_reshape(self, value, shape, axes, order):
        ext = self.extension
        shape, axes, order = unwrap(shape), unwrap(axes), unwrap(order)
        if not all(isinstance(items, (tuple, list)) for items in (shape, axes, order)):
            raise TypeError("named reshape requires explicit shape, axes and order sequences")
        symbols = {symbol: ext._shape_ids[name] for bindings in (ext._shape_symbols, ext._auto_symbols)
                   for name, symbol in bindings.items()}
        shape = tuple(symbols[s] if isinstance(s, Value) else s for s in shape)
        if len(shape) != len(axes) or not all(isinstance(a, str) for a in (*axes, *order)):
            raise ValueError("named reshape axes/order must be names with one output name per dimension")
        output_axes = tuple(ext._axis_id(a) for a in axes)
        input_order = tuple(ext._axis_ids[a.lower()] for a in order)
        if sorted(input_order) != sorted(axes_of(value.handle.type)):
            raise ValueError("reshape order must name each input axis exactly once")
        return wrap(self.builder.reshape_axes(value.handle, shape, output_axes, input_order))

    def derived_interleave(self, value, rows, field_chunks):
        extension = self.extension
        handle = unwrap(value)
        result = extension._derive_result
        if not isinstance(handle.type, ViewType) or result is None:
            raise ValueError("artifact interleave belongs to a derived Encoding builder")
        if type(rows) is not int or rows <= 0 or result.encoding.parameters != (rows,):
            raise ValueError("interleave rows must equal the derived Encoding parameter")
        info = extension._declared_encodings[handle.type.encoding.family]
        if field_chunks is None:
            chunks = [1] * len(info.fields)
        else:
            if not isinstance(field_chunks, dict) or set(field_chunks) != {f.name for f in info.fields}:
                raise ValueError("interleave field_chunks must name every field once")
            chunks = [field_chunks[f.name] for f in info.fields]
            if any(type(n) is not int or n <= 0 for n in chunks) or any(
                    f.bit_offset % 8 or f.storage_bits % (8 * n) for f, n in zip(info.fields, chunks)):
                raise ValueError("interleave chunks must partition byte-addressable field storage")
        return wrap(self.builder.op("weft_kernel.interleave", (handle,), (result,),
                                    {"rows": str(rows), "field_chunks": "array<i64: " + ", ".join(map(str, chunks)) + ">"}))

    def _logical_binary(self, kind, lhs, rhs, comparison=False):
        ext = self.extension
        lhs, rhs = unwrap(lhs), unwrap(rhs)
        if not isinstance(lhs, Value):
            lhs = ext._constant(lhs, element_type(rhs.type).dtype, self.node)
        if not isinstance(rhs, Value):
            rhs = ext._constant(rhs, element_type(lhs.type).dtype, self.node)
        if element_type(lhs.type) != element_type(rhs.type):
            dtype = ext._promote_dtype(element_type(lhs.type).dtype, element_type(rhs.type).dtype)
            lhs, rhs = ext._convert_value(lhs, dtype, self.node), ext._convert_value(rhs, dtype, self.node)
        return wrap(self.builder.compare(kind, lhs, rhs) if comparison else self.builder.binary(kind, lhs, rhs))


def _binary(method, kind, comparison=False):
    upstream = getattr(TritonSemantic, method)
    def operation(self, lhs, rhs, *args, **kwargs):
        if is_logical(lhs) or is_logical(rhs):
            selected = kind
            if method in ("minimum", "maximum"):
                policy = args[0] if args else kwargs["propagate_nan"]
                if policy not in (tl.PropagateNan.ALL, tl.PropagateNan.NONE):
                    raise ValueError(f"unexpected NaN propagation policy {policy}")
                value = lhs if is_logical(lhs) else rhs
                if policy == tl.PropagateNan.ALL and value.dtype.is_floating():
                    selected = method
            return self._logical_binary(selected, lhs, rhs, comparison)
        return upstream(self, lhs, rhs, *args, **kwargs)
    return operation


for _method, _kind in {"add": "add", "sub": "sub", "mul": "mul", "truediv": "div", "floordiv": "div",
                       "mod": "mod", "and_": "and", "or_": "or", "xor_": "xor", "shl": "shl",
                       "lshr": "shr", "ashr": "shr", "minimum": "min", "maximum": "max"}.items():
    setattr(CanonicalSemantic, _method, _binary(_method, _kind))
for _method, _kind in {"equal": "eq", "not_equal": "ne", "less_than": "lt", "less_equal": "le",
                       "greater_than": "gt", "greater_equal": "ge"}.items():
    setattr(CanonicalSemantic, _method, _binary(_method, _kind, True))
