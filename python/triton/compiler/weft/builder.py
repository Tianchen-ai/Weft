"""Canonical operation builder for the shared Triton CodeGenerator.

This implements the frontend construction protocol, not a TTIR conversion.
The construction tree is handed to the in-process Canonical MLIR binding.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from triton.language.weft import dtypes
from triton.language.weft.diagnostics import SourceLocation

from .ir import IRBuilder
from .types import ScalarType, PointerType, LocalValueType, axes_of, element_type, shape_of, value_type
from .pointers import Address


@dataclass
class Location:
    filename: str
    line: int
    column: int
    name: str | None = None

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class CanonicalBuilder(IRBuilder):
    def __init__(self, extension):
        super().__init__()
        self.extension = extension
        self.block = None
        self.position = None
        self.location = None

    def create_loc(self, filename, line, column):
        return Location(filename, line, max(column - 1, 0))

    def set_loc(self, *args):
        self.location = args[0] if len(args) == 1 else self.create_loc(*args)

    def get_loc(self):
        return self.location

    def create_name_loc(self, name, location):
        location = location or self.location
        return Location(location.filename, location.line, location.column, name)

    def create_module(self):
        # The native module is constructed after the one Canonical tree closes.
        return None

    def get_insertion_block(self):
        return self.block

    def get_insertion_point(self):
        # Match OpBuilder's stable iterator semantics. A saved end iterator
        # remains after operations appended while visiting a nested region;
        # saving a list index would put the continuation before its own loop.
        anchor = (self.block.operations[self.position]
                  if self.block is not None and self.position is not None and
                  self.position < len(self.block.operations) else None)
        return self.block, anchor

    def restore_insertion_point(self, point):
        self.block, anchor = point
        if anchor is None:
            self.position = None
        else:
            self.position = next(index for index, operation in enumerate(self.block.operations) if operation is anchor)

    def set_insertion_point_to_start(self, block):
        self.block, self.position = block, 0

    def set_insertion_point_to_end(self, block):
        self.block, self.position = block, None

    def create_block(self):
        return self.region(())

    def create_block_with_parent(self, parent, types):
        if parent.arguments or parent.operations:
            raise ValueError("canonical structured region already has a block")
        parent.arguments = tuple(self.value(ty) for ty in types)
        self.set_insertion_point_to_start(parent)
        return parent

    def emit(self, block, name, location, **kwargs):
        operation = self.operation(name, location, **kwargs)
        for result in operation.results:
            result.set_loc(location)
        if block is self.block and self.position is not None:
            block.operations.insert(self.position, operation)
            self.position += 1
        else:
            if block.terminated:
                raise ValueError("cannot emit after a canonical terminator")
            block.operations.append(operation)
        return operation.results

    def op(self, name, operands=(), types=(), attrs=None, regions=()):
        location = self.location
        if location is None:
            raise ValueError("canonical construction requires a source location")
        source = SourceLocation(location.filename, location.line, location.column)
        results = self.emit(self.block, name, source, operands=tuple(operands), result_types=tuple(types),
                            attributes=attrs, regions=tuple(regions))
        return results[0] if len(results) == 1 else results

    def control(self, name, operands, types, regions):
        op = self.operation(name, self.location, operands=tuple(operands), result_types=tuple(types),
                            regions=tuple(regions))
        if self.position is None:
            self.block.operations.append(op)
        else:
            self.block.operations.insert(self.position, op)
            self.position += 1
        return op

    def create_if_op(self, types, condition, with_else):
        if not with_else:
            raise NotImplementedError("Canonical conditionals require both structured regions")
        regions = (self.region(()), self.region(()))
        return self.control("scf.if", (condition,), types, regions)

    def create_for_op(self, lower, upper, step, initial):
        # Canonical ordinary loops have index induction variables; the common
        # frontend still exposes the source integer dtype through explicit casts.
        bounds = tuple(self.create_int_cast(v, ScalarType(dtypes.index), True) for v in (lower, upper, step))
        region = self.region((ScalarType(dtypes.index),) + tuple(v.type for v in initial))
        return self.control("scf.for", bounds + tuple(initial), [v.type for v in initial], (region,))

    def close_regions(self, block):
        for operation in block.operations:
            for region in operation.regions:
                self.close_regions(region)
                if operation.name in ("scf.for", "scf.if") and not region.terminated:
                    if operation.results:
                        raise ValueError("structured Canonical region is missing its carried values")
                    region.operations.append(self.operation("scf.yield", operation.location))

    def create_while_op(self, types, initial):
        return self.control("scf.while", initial, types, (self.region(()), self.region(())))

    def create_yield_op(self, values):
        return self.op("scf.yield", values)

    def create_reduce_ret(self, *values):
        return self.op("weft_kernel.reduce_yield", values)

    def create_scan(self, values, axis, reverse):
        operation = self.control("weft_kernel.scan", values, [v.type for v in values], (self.region(()),))
        operation.attributes.update(axis=str(axis), reverse="true" if reverse else "false")
        return operation

    def create_scan_ret(self, *values):
        return self.op("weft_kernel.scan_yield", values)

    def create_condition_op(self, condition, values):
        return self.op("scf.condition", (condition, *values))

    def create_poison(self, ty):
        # The common generator replaces induction placeholders before native IR
        # construction. An unresolved use is rejected by the native SSA builder.
        return self.value(ty)

    def get_int32_attr(self, value):
        return f"{int(value)} : i32"

    def get_unit_attr(self):
        return "unit"

    def get_disable_loop_licm_attr(self):
        raise NotImplementedError("disable_licm has no Canonical loop contract")

    def positional_axes(self, rank):
        return tuple(self.extension._axis_id(f"$position{-i}") for i in range(rank, 0, -1))

    def create_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc):
        from triton._C.libtriton import ir
        if input_precision != ir.INPUT_PRECISION.IEEE:
            raise NotImplementedError("RISC-V dot requires IEEE input precision")
        # Triton's imprecise-accumulator bound applies to fp8; supported RISC-V
        # operands always accumulate in the declared result format.
        left_shape, right_shape = shape_of(lhs.type), shape_of(rhs.type)
        if left_shape[:-2] != right_shape[:-2]:
            raise ValueError("dot operands must have matching batch dimensions")
        ext = self.extension
        batch = tuple(ext._axis_id(f"$dot.batch{i}") for i in range(len(left_shape) - 2))
        m, n, k = (ext._axis_id("$dot." + name) for name in ("m", "n", "k"))
        left = self.reshape_axes(lhs, left_shape, batch + (m, k))
        right = self.reshape_axes(rhs, right_shape, batch + (k, n))
        product = ext._contract_value(ext.generator.cur_node, left, right, (k,),
                                      "contract" if batch else "outer_contract", element_type(acc.type).dtype)
        shape = left_shape[:-1] + right_shape[-1:]
        axes = self.positional_axes(len(shape))
        product = self.reshape_axes(product, shape, axes)
        return self.binary("add", product, self.reshape_axes(acc, shape, axes))

    def get_block_ty(self, element, shape):
        return LocalValueType(element, tuple(shape), self.positional_axes(len(shape)))

    def get_ptr_ty(self, element, address_space=1):
        if address_space != 1 or not isinstance(element, ScalarType):
            raise NotImplementedError("CPU pointers require a scalar element and the default address space")
        return PointerType(element)

    def create_addptr(self, pointer, offset):
        offset = self.create_int_cast(offset, value_type(ScalarType(dtypes.i64), shape_of(offset.type),
                                                        axes_of(offset.type)), True)
        origin = pointer.offset
        if origin.producer is not None and origin.producer.name == "weft_kernel.new":
            origin = origin.producer.operands[0]
        if (origin.producer is not None and origin.producer.name in ("weft_kernel.constant", "arith.constant") and
                int(origin.producer.attributes["value"].split(":")[0].strip(), 0) == 0):
            return Address(pointer.base, offset)
        return Address(pointer.base, self.binary("add", pointer.offset, offset))

    def pointer_memory(self, pointer, mask, value, *, store=False, cache=None, eviction=None):
        shape, axes = shape_of(pointer.offset.type), axes_of(pointer.offset.type)
        if mask is None:
            mask = self.get_int1(True)
        if shape:
            mask = self.create_broadcast(self.reshape_axes(mask, shape_of(mask.type),
                self.positional_axes(len(shape_of(mask.type)))), shape)
        if value is None:
            value = self.extension._constant(0, pointer.base.type.element_type.dtype,
                                             self.extension.generator.cur_node)
        ty = value_type(pointer.base.type.element_type, shape, axes)
        if shape and value.type != ty:
            value = self.create_broadcast(value, shape)
        return self.op("weft_kernel.pointer_store" if store else "weft_kernel.pointer_load",
                       (pointer.base, pointer.offset, mask, value), () if store else (ty,),
                       {"cache_hint": json.dumps(str(cache)), "eviction_hint": json.dumps(str(eviction))})

    def create_load(self, ptr, cache, eviction, volatile):
        return self.create_masked_load(ptr, None, None, cache, eviction, volatile)

    def create_masked_load(self, ptr, mask, other, cache, eviction, volatile):
        if volatile:
            raise NotImplementedError("volatile pointer loads require an explicit ordered-memory contract")
        return self.pointer_memory(ptr, mask, other, cache=cache, eviction=eviction)

    def create_store(self, ptr, value, cache, eviction):
        return self.create_masked_store(ptr, value, None, cache, eviction)

    def create_masked_store(self, ptr, value, mask, cache, eviction):
        self.pointer_memory(ptr, mask, value, store=True, cache=cache, eviction=eviction)

    def pointer_atomic(self, ptr, value, compare, mask, kind, sem, scope):
        from triton._C.libtriton import ir
        ordering = {ir.MEM_SEMANTIC.ACQUIRE: "acquire", ir.MEM_SEMANTIC.RELEASE: "release",
                    ir.MEM_SEMANTIC.ACQUIRE_RELEASE: "acq_rel", ir.MEM_SEMANTIC.RELAXED: "relaxed"}[sem]
        visibility = {ir.MEM_SYNC_SCOPE.CTA: "cta", ir.MEM_SYNC_SCOPE.GPU: "gpu",
                      ir.MEM_SYNC_SCOPE.SYSTEM: "sys"}[scope]
        if mask is None:
            mask = self.get_int1(True)
        if compare is None:
            compare = self.extension._constant(0, ptr.base.type.element_type.dtype, self.extension.generator.cur_node)
        ty = value_type(ptr.base.type.element_type, shape_of(ptr.offset.type), axes_of(ptr.offset.type))
        return self.op("weft_kernel.pointer_atomic", (ptr.base, ptr.offset, mask, value, compare), (ty,),
                       {"kind": json.dumps(kind), "ordering": json.dumps(ordering), "scope": json.dumps(visibility)})

    def create_atomic_rmw(self, op, ptr, value, mask, sem, scope):
        from triton._C.libtriton import ir
        kinds = {ir.ATOMIC_OP.ADD: "add", ir.ATOMIC_OP.FADD: "fadd", ir.ATOMIC_OP.AND: "and",
                 ir.ATOMIC_OP.OR: "or", ir.ATOMIC_OP.XOR: "xor", ir.ATOMIC_OP.XCHG: "xchg",
                 ir.ATOMIC_OP.MIN: "min", ir.ATOMIC_OP.MAX: "max", ir.ATOMIC_OP.UMIN: "umin",
                 ir.ATOMIC_OP.UMAX: "umax"}
        return self.pointer_atomic(ptr, value, None, mask, kinds[op], sem, scope)

    def create_atomic_cas(self, ptr, compare, value, sem, scope):
        return self.pointer_atomic(ptr, value, compare, None, "cas", sem, scope)

    def create_get_program_id(self, axis):
        return self.create_numeric_cast(self.extension._launch_arguments[f"__triton_pid_{axis}"],
                                        ScalarType(dtypes.i32))

    def create_get_num_programs(self, axis):
        return self.create_numeric_cast(self.extension._launch_arguments[f"__triton_grid_{axis}"],
                                        ScalarType(dtypes.i32))

    def get_null_value(self, ty):
        return self.extension._constant(0, ty.dtype, self.extension.generator.cur_node)

    def get_all_ones_value(self, ty):
        dtype = ty.dtype
        value = -1 if dtype.signedness == "signed" else (1 << dtype.bits) - 1
        return self.extension._constant(value, dtype, self.extension.generator.cur_node)

    def create_make_range(self, ty, start, end):
        # iota's Canonical index domain is unsigned; Triton arange exposes i32.
        raw_type = LocalValueType(ScalarType(dtypes.u32), ty.shape, ty.axes)
        value = self.op("weft_kernel.iota", types=(raw_type,), attrs={"start": str(start), "end": str(end)})
        return self.create_int_cast(value, ty, False)

    def create_int_cast(self, value, ty, signed):
        element = element_type(ty)
        if not isinstance(element, ScalarType):
            raise TypeError("integer cast requires a numeric destination")
        producer = value.producer
        source = element_type(value.type)
        if (element.dtype is dtypes.index and isinstance(source, ScalarType) and
                source.dtype.category in (dtypes.DTypeCategory.INTEGER, dtypes.DTypeCategory.INDEX) and
                producer is not None and producer.name in ("weft_kernel.constant", "arith.constant")):
            literal = int(producer.attributes["value"].split(":")[0].strip(), 0)
            return self.extension._constant(literal, dtypes.index, self.extension.generator.cur_node)
        return self.extension._convert_value(value, element.dtype, self.extension.generator.cur_node)

    def create_numeric_cast(self, value, ty):
        return self.extension._convert_value(value, element_type(ty).dtype, self.extension.generator.cur_node)

    def create_fp_to_fp(self, value, ty, rounding):
        from triton._C.libtriton import ir
        modes = {ir.ROUNDING_MODE.RTNE: "rne", ir.ROUNDING_MODE.RTZ: "rtz"}
        if rounding not in modes:
            raise NotImplementedError(f"unsupported floating conversion rounding {rounding}")
        return self.op("weft_kernel.narrow", (value,), (ty,),
                       {"rounding": json.dumps(modes[rounding]), "saturate": "false"})

    def create_bitcast(self, value, ty):
        return self.op("weft_kernel.bitcast", (value,), (ty,))

    def create_splat(self, ty, value):
        if isinstance(value, Address):
            return Address(value.base, self.create_splat(self.get_block_ty(ScalarType(dtypes.i64), ty.shape), value.offset))
        if not isinstance(ty, LocalValueType):
            raise TypeError("splat requires a block type")
        return self.op("weft_kernel.new", (value,), (ty,), {"initialized": "true"})

    def reshape_axes(self, value, shape, axes, order=None):
        if isinstance(value, Address):
            return Address(value.base, self.reshape_axes(value.offset, shape, axes, order))
        ty = value_type(element_type(value.type), tuple(shape), tuple(axes))
        if ty == value.type and (order is None or tuple(order) == axes_of(value.type)):
            return value
        if not shape:
            zero = self.extension._constant(0, dtypes.index, self.extension.generator.cur_node)
            return self.op("weft_kernel.extract", (value,) + (zero,) * len(shape_of(value.type)), (ty,),
                           {"selectors": json.dumps(["index"] * len(shape_of(value.type)))})
        if not shape_of(value.type):
            return self.create_splat(ty, value)
        order = axes_of(value.type) if order is None else order
        return self.op("weft_kernel.reshape", (value,), (ty,),
                       {"order": "array<i64: " + ", ".join(map(str, order)) + ">"})

    def create_reshape(self, value, shape, can_reorder=False):
        return self.reshape_axes(value, shape, self.positional_axes(len(shape)))

    def create_unsplat(self, value):
        return self.reshape_axes(value, (), ())

    def create_expand_dims(self, value, axis):
        shape = list(shape_of(value.type))
        shape.insert(axis, 1)
        return self.create_reshape(value, shape)

    def create_trans(self, value, order):
        shape = tuple(shape_of(value.type)[i] for i in order)
        axes = tuple(axes_of(value.type)[i] for i in order)
        return self.reshape_axes(value, shape, self.positional_axes(len(shape)), axes)

    def create_gather(self, source, indices, axis):
        if isinstance(source, Address):
            return Address(source.base, self.create_gather(source.offset, indices, axis))
        # The ordinary semantic interface has already checked ranks, axis and
        # matching free extents. Move the selected axis last so the existing
        # named projection can share every free coordinate with its indices.
        rank = len(shape_of(source.type))
        order = tuple(i for i in range(rank) if i != axis) + (axis,)
        source = self.create_trans(source, order)
        indices = self.create_trans(indices, order)
        dtype = element_type(indices.type).dtype
        unsigned = dtypes.u64 if dtype.bits > 32 else dtypes.u32
        indices = self.extension._convert_value(indices, unsigned, self.extension.generator.cur_node)
        index_axes = list(axes_of(source.type))
        index_axes[-1] = self.extension._next_axis_id
        self.extension._next_axis_id += 1
        indices = self.reshape_axes(indices, shape_of(indices.type), index_axes)
        ty = value_type(element_type(source.type), shape_of(indices.type), tuple(index_axes))
        gathered = self.op("weft_kernel.extract", (source, indices), (ty,),
                           {"selectors": json.dumps(["all"] * (rank - 1) + ["gather"])})
        gathered = self.reshape_axes(gathered, shape_of(indices.type), self.positional_axes(rank))
        return self.create_trans(gathered, tuple(order.index(i) for i in range(rank)))

    def create_join(self, lhs, rhs):
        if isinstance(lhs, Address) or isinstance(rhs, Address):
            if not isinstance(lhs, Address) or not isinstance(rhs, Address) or lhs.base is not rhs.base:
                raise NotImplementedError("joining pointers across different storage origins")
            return Address(lhs.base, self.create_join(lhs.offset, rhs.offset))
        if lhs.type != rhs.type:
            raise TypeError("join requires matching element types and broadcast domains")
        shape = shape_of(lhs.type) + (1,)
        axes = self.positional_axes(len(shape))
        lhs, rhs = (self.reshape_axes(value, shape, axes) for value in (lhs, rhs))
        lanes = self.op("weft_kernel.iota", types=(LocalValueType(ScalarType(dtypes.u32), (2,), (axes[-1],)),),
                        attrs={"start": "0", "end": "2"})
        zero = self.extension._constant(0, dtypes.u32, self.extension.generator.cur_node)
        return self.create_select(self.compare("eq", lanes, zero), lhs, rhs)

    def create_split(self, value):
        if isinstance(value, Address):
            return tuple(Address(value.base, offset) for offset in self.create_split(value.offset))
        shape, axes = shape_of(value.type), axes_of(value.type)
        if not shape or shape[-1] != 2:
            raise ValueError("split requires a final dimension of length two")
        ty = value_type(element_type(value.type), shape[:-1], axes[:-1])
        results = []
        for lane in (0, 1):
            index = self.extension._constant(lane, dtypes.index, self.extension.generator.cur_node)
            selected = self.op("weft_kernel.extract", (value, index), (ty,),
                               {"selectors": json.dumps(["all"] * (len(shape) - 1) + ["index"])})
            results.append(self.reshape_axes(selected, shape[:-1], self.positional_axes(len(shape) - 1)))
        return tuple(results)

    def create_broadcast(self, value, shape):
        if isinstance(value, Address):
            return Address(value.base, self.create_broadcast(value.offset, shape))
        # A typed select expands singleton coordinates without introducing any
        # arithmetic or changing NaN/signed-zero payloads.
        result_ty = self.get_block_ty(element_type(value.type), shape)
        if result_ty == value.type:
            return value
        if all(extent == 1 for extent in shape_of(value.type)):
            scalar = self.reshape_axes(value, (), ()) if shape_of(value.type) else value
            return self.create_splat(result_ty, scalar)
        predicate_ty = self.get_block_ty(ScalarType(dtypes.i1), shape)
        predicate = self.create_splat(predicate_ty, self.get_int1(True))
        return self.op("weft_kernel.select", (predicate, value, value), (result_ty,))

    def create_select(self, predicate, lhs, rhs):
        if isinstance(lhs, Address) or isinstance(rhs, Address):
            if not isinstance(lhs, Address) or not isinstance(rhs, Address) or lhs.base is not rhs.base:
                raise NotImplementedError("elementwise pointer selection across different storage origins")
            return Address(lhs.base, self.create_select(predicate, lhs.offset, rhs.offset))
        shape, axes = self.extension._broadcast_domain(lhs.type, rhs.type, self.extension.generator.cur_node)
        shape, axes = self.extension._broadcast_domain(value_type(element_type(lhs.type), shape, axes),
                                                       predicate.type, self.extension.generator.cur_node)
        return self.op("weft_kernel.select", (predicate, lhs, rhs),
                       (value_type(element_type(lhs.type), shape, axes),))

    def binary(self, kind, lhs, rhs):
        return self.extension._binary_values(kind, lhs, rhs, self.extension.generator.cur_node)

    def compare(self, predicate, lhs, rhs):
        shape, axes = self.extension._broadcast_domain(lhs.type, rhs.type, self.extension.generator.cur_node)
        return self.op("weft_kernel.compare", (lhs, rhs), (value_type(ScalarType(dtypes.i1), shape, axes),),
                       {"predicate": json.dumps(predicate)})

    def unary(self, kind, value):
        return self.op("weft_kernel.unary", (value,), (value.type,), {"kind": json.dumps(kind)})


# These are builder operation methods used by upstream TritonSemantic, not
# rewrites of user functions. Unsupported operations have no implementation.
for _method in ("fp_trunc", "fp_ext", "fp_to_si", "fp_to_ui", "si_to_fp", "ui_to_fp"):
    setattr(CanonicalBuilder, "create_" + _method, CanonicalBuilder.create_numeric_cast)


def _binary(kind):
    return lambda self, lhs, rhs: self.binary(kind, lhs, rhs)


for _kind, _methods in {
    "add": ("add", "fadd"), "sub": ("sub", "fsub"), "mul": ("mul", "fmul"),
    "div": ("sdiv", "udiv", "fdiv", "precise_divf"), "mod": ("srem", "urem", "frem"),
    "and": ("and",), "or": ("or",), "xor": ("xor",), "shl": ("shl",), "shr": ("lshr", "ashr"),
    "max": ("maxnumf", "maxsi", "maxui"), "min": ("minnumf", "minsi", "minui"),
    "maximum": ("maximumf",), "minimum": ("minimumf",),
}.items():
    for _method in _methods:
        setattr(CanonicalBuilder, "create_" + _method, _binary(_kind))


def _compare(predicate):
    return lambda self, lhs, rhs: self.compare(predicate, lhs, rhs)


for _predicate in ("eq", "ne", "lt", "le", "gt", "ge"):
    for _prefix in ("icmp", "icmpS", "icmpU", "fcmpO", "fcmpU"):
        setattr(CanonicalBuilder, "create_" + _prefix + _predicate.upper(), _compare(_predicate))


def _unary(kind):
    return lambda self, value: self.unary(kind, value)


for _method, _kind in {"fneg": "neg", "fabs": "abs", "iabs": "abs", "exp": "exp", "exp2": "exp2", "log": "log",
                        "sqrt": "sqrt", "precise_sqrt": "sqrt", "rsqrt": "rsqrt"}.items():
    setattr(CanonicalBuilder, "create_" + _method, _unary(_kind))


def _constant(dtype):
    return lambda self, value: self.extension._constant(value, dtype, self.extension.generator.cur_node)


def _type(dtype):
    return lambda self: ScalarType(dtype)


for _name, _dtype in {"int1": dtypes.i1, "int8": dtypes.i8, "uint8": dtypes.u8,
                      "int16": dtypes.i16, "uint16": dtypes.u16, "int32": dtypes.i32,
                      "uint32": dtypes.u32, "int64": dtypes.i64, "uint64": dtypes.u64,
                      "fp16": dtypes.f16, "bf16": dtypes.bf16, "fp32": dtypes.f32,
                      "fp64": dtypes.f64}.items():
    setattr(CanonicalBuilder, "get_" + _name, _constant(_dtype))
    setattr(CanonicalBuilder, "get_" + _name + "_ty", _type(_dtype))
for _name, _dtype in {"half": dtypes.f16, "float": dtypes.f32, "double": dtypes.f64}.items():
    setattr(CanonicalBuilder, "get_" + _name + "_ty", _type(_dtype))
