#ifndef WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
#define WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_Kernel_Dialect : Dialect {
  let name = "weft_kernel";
  let cppNamespace = "::weft::kernel";
  let summary = "Canonical Weft level, value, encoding, and operation semantics";
  let description = [{
    This dialect is the sole persistent Weft program representation. It records
    logical encodings, views, values, level ownership, ordinary ordered control,
    and explicit basic operations. It contains no hardware layout, vector width,
    register tile, fragment, instruction spelling, or built-in operator kernel.
  }];
  let useDefaultTypePrinterParser = 1;
}

class WeftKernel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WEFT_Kernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class WeftKernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Kernel_Dialect, mnemonic, traits>;

def WeftKernel_EncodingType : WeftKernel_Type<"Encoding", "encoding"> {
  let parameters = (ins
    StringRefParameter<"encoding family">:$family,
    StringRefParameter<"base, dense, derived_family, or derived_instance">:$kind,
    StringRefParameter<"fixed artifact layout identity, otherwise empty">:$layoutIdentity,
    "::mlir::DenseI64ArrayAttr":$parameters
  );
  let assemblyFormat =
      "`<` $family `,` $kind `,` $layoutIdentity `,` $parameters `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_ViewType : WeftKernel_Type<"View", "view"> {
  let parameters = (ins
    "::mlir::Type":$encoding,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds
  );
  let assemblyFormat =
      "`<` $encoding `,` $shape `,` $axisIds `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_PointerType : WeftKernel_Type<"Pointer", "pointer"> {
  let summary = "Typed storage origin without an implicit logical extent";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_ValueType : WeftKernel_Type<"Value", "value"> {
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds
  );
  let assemblyFormat =
      "`<` $elementType `,` $shape `,` $axisIds `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_SliceType : WeftKernel_Type<"Slice", "slice"> {
  let parameters = (ins
    "::mlir::Type":$encoding,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds
  );
  let assemblyFormat =
      "`<` $encoding `,` $shape `,` $axisIds `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_DomainType : WeftKernel_Type<"Domain", "domain"> {
  let parameters = (ins
    StringRefParameter<"axis name">:$axisName,
    "int64_t":$domainId,
    "int64_t":$parentDomainId,
    "int64_t":$axisId,
    StringRefParameter<"root, rows, cols, tiles, blocks, or subs">:$relation,
    StringRefParameter<"exact or tail">:$tail
  );
  let assemblyFormat =
      "`<` $axisName `,` $domainId `,` $parentDomainId `,` $axisId `,` $relation `,` $tail `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_PointType : WeftKernel_Type<"Point", "point"> {
  let parameters = (ins WeftKernel_DomainType:$domain);
  let assemblyFormat = "`<` $domain `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_EncodingDeclOp
    : WeftKernel_Op<"encoding_decl", [Symbol, Pure]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    StrAttr:$layout_identity,
    StrAttr:$bit_order,
    StrAttr:$byte_order,
    I64Attr:$alignment,
    I64Attr:$elements,
    I64Attr:$storage_bits,
    ArrayAttr:$field_names,
    ArrayAttr:$field_types,
    ArrayAttr:$field_shapes,
    ArrayAttr:$field_layouts,
    DenseI64ArrayAttr:$field_bit_offsets,
    DenseI64ArrayAttr:$field_storage_bits,
    DenseI64ArrayAttr:$padding
  );
  let hasVerifier = 1;
}

def WeftKernel_DeriveOp
    : WeftKernel_Op<"derive", [Symbol, IsolatedFromAbove, SingleBlock, Pure]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$source_family,
                       StrAttr:$result_family,
                       StrAttr:$layout_identity,
                       ArrayAttr:$parameter_names,
                       DenseI64ArrayAttr:$parameter_values);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_DeriveYieldOp : WeftKernel_Op<"derive_yield", [Terminator]> {
  let arguments = (ins WeftKernel_ViewType:$value);
  let hasVerifier = 1;
}

def WeftKernel_KernelOp
    : WeftKernel_Op<"kernel", [Symbol, IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       ArrayAttr:$arg_names,
                       ArrayAttr:$arg_access,
                       DenseI64ArrayAttr:$arg_alias_sets,
                       OptionalAttr<StrArrayAttr>:$strided_args,
                       ArrayAttr:$shape_symbols,
                       StrAttr:$source);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_ReturnOp : WeftKernel_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_HelperOp : WeftKernel_Op<"helper", [SingleBlock, NoTerminator, RecursiveMemoryEffects]> {
  let arguments = (ins StrAttr:$name, BoolAttr:$noinline);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_HelperReturnOp : WeftKernel_Op<"helper_return", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_RootDomainOp : WeftKernel_Op<"root_domain", [Pure]> {
  let results = (outs WeftKernel_DomainType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SymbolOp : WeftKernel_Op<"symbol", [Pure]> {
  let arguments = (ins StrAttr:$name, StrAttr:$kind,
                       DenseI64ArrayAttr:$choices);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftKernel_DomainOp : WeftKernel_Op<"domain", [Pure]> {
  let arguments = (ins WeftKernel_DomainType:$parent,
                       Index:$extent, Index:$partition,
                       Index:$multiplicity);
  let results = (outs WeftKernel_DomainType:$result);
  let hasVerifier = 1;
}

def WeftKernel_LevelOp
    : WeftKernel_Op<"level", [RecursiveMemoryEffects]> {
  let arguments = (ins WeftKernel_DomainType:$domain,
                       Variadic<AnyType>:$carried);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$state_births,
                         SizedRegion<1>:$staged_births,
                         SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_BirthsYieldOp : WeftKernel_Op<"births_yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_HandoffOp : WeftKernel_Op<"handoff", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

// MLIR arith.constant accepts only signless integers.  Weft integer types carry
// source signedness, so signed/unsigned scalar literals require a canonical op
// whose result type preserves that semantic fact.
def WeftKernel_ConstantOp : WeftKernel_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_IotaOp : WeftKernel_Op<"iota", [Pure]> {
  let arguments = (ins I64Attr:$start, I64Attr:$end);
  let results = (outs WeftKernel_ValueType:$result);
  let hasVerifier = 1;
}

// Source preconditions do not change the numerical value or memory effects.
def WeftKernel_AssumeOp : WeftKernel_Op<"assume", [MemoryEffects<[]>]> {
  let arguments = (ins I1:$condition);
}

def WeftKernel_ValueFactOp : WeftKernel_Op<"value_fact", [MemoryEffects<[]>]> {
  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$kind,
                       DenseI64ArrayAttr:$values);
  let hasVerifier = 1;
}

def WeftKernel_NewOp : WeftKernel_Op<"new", [Pure]> {
  let arguments = (ins Optional<AnyType>:$initial, BoolAttr:$initialized);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_LocalShapeOp : WeftKernel_Op<"local_shape", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Index:$extent, I64Attr:$id);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftKernel_EmptyOp : WeftKernel_Op<"empty", [MemoryEffects<[MemAlloc]>]> {
  let arguments = (ins Variadic<Index>:$extents);
  let results = (outs WeftKernel_ValueType:$result);
  let hasVerifier = 1;
}

def WeftKernel_MaterializeOp : WeftKernel_Op<"materialize", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_AdmitOp : WeftKernel_Op<"admit", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$region);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CommitOp : WeftKernel_Op<"commit", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$value, AnyType:$region);
  let hasVerifier = 1;
}

def WeftKernel_PointerCastOp : WeftKernel_Op<"pointer_cast", [Pure]> {
  let arguments = (ins WeftKernel_PointerType:$input);
  let results = (outs WeftKernel_PointerType:$result);
  let hasVerifier = 1;
}

def WeftKernel_PointerLoadOp : WeftKernel_Op<"pointer_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftKernel_PointerType:$base, AnyType:$offsets,
                       AnyType:$mask, AnyType:$other);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_PointerStoreOp : WeftKernel_Op<"pointer_store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins WeftKernel_PointerType:$base, AnyType:$offsets,
                       AnyType:$mask, AnyType:$value);
  let hasVerifier = 1;
}

def WeftKernel_PointerAtomicOp : WeftKernel_Op<"pointer_atomic", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins WeftKernel_PointerType:$base, AnyType:$offsets, AnyType:$mask,
                       AnyType:$value, AnyType:$compare, StrAttr:$kind, StrAttr:$ordering, StrAttr:$scope);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_AtomicRMWOp : WeftKernel_Op<"atomic_rmw", [
    MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Atomic scalar update returning the value before the update";
  let arguments = (ins AnyType:$value, WeftKernel_SliceType:$region,
                       StrAttr:$kind, StrAttr:$ordering);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SliceOp : WeftKernel_Op<"slice", [Pure]> {
  let arguments = (ins AnyType:$base, Variadic<AnyType>:$indices,
                       ArrayAttr:$selectors);
  let results = (outs WeftKernel_SliceType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SubviewOp : WeftKernel_Op<"subview", [Pure]> {
  let arguments = (ins AnyType:$base, Variadic<Index>:$dynamic_bounds,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$extents);
  let results = (outs WeftKernel_SliceType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ExtentOp : WeftKernel_Op<"extent", [Pure]> {
  let arguments = (ins AnyType:$base, I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftKernel_ReshapeOp : WeftKernel_Op<"reshape", [Pure]> {
  let arguments = (ins WeftKernel_ValueType:$input,
                       DenseI64ArrayAttr:$order);
  let results = (outs WeftKernel_ValueType:$result);
  let hasVerifier = 1;
}

def WeftKernel_FieldOp : WeftKernel_Op<"field", [Pure]> {
  let arguments = (ins AnyType:$owner, StrAttr:$name);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ExtractOp : WeftKernel_Op<"extract", [Pure]> {
  let arguments = (ins AnyType:$input, Variadic<AnyType>:$indices,
                       ArrayAttr:$selectors);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_UpdateOp : WeftKernel_Op<"update", [Pure]> {
  let arguments = (ins AnyType:$input, AnyType:$value,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_UnaryOp : WeftKernel_Op<"unary", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_BinaryOp : WeftKernel_Op<"binary", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$kind);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CompareOp : WeftKernel_Op<"compare", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SelectOp : WeftKernel_Op<"select", [Pure]> {
  let arguments = (ins AnyType:$condition, AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CastOp : WeftKernel_Op<"cast", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_BitcastOp : WeftKernel_Op<"bitcast", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_NarrowOp : WeftKernel_Op<"narrow", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$rounding,
                       BoolAttr:$saturate);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftKernel_GroupedMacOp<string mnemonic>
    : WeftKernel_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$group,
                       StrAttr:$overflow);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_MacGroupsOp : WeftKernel_GroupedMacOp<"mac_groups">;

def WeftKernel_WidenOp : WeftKernel_Op<"widen", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ReduceOp : WeftKernel_Op<"reduce", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind, I64Attr:$axis);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ReduceRegionOp : WeftKernel_Op<"reduce_region", [Pure, SingleBlock, IsolatedFromAbove]> {
  let arguments = (ins Variadic<WeftKernel_ValueType>:$inputs, I64Attr:$axis);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def WeftKernel_ReduceYieldOp : WeftKernel_Op<"reduce_yield", [Pure, Terminator, HasParent<"ReduceRegionOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def WeftKernel_ScanOp : WeftKernel_Op<"scan", [Pure, SingleBlock, IsolatedFromAbove]> {
  let arguments = (ins Variadic<WeftKernel_ValueType>:$inputs, I64Attr:$axis, BoolAttr:$reverse);
  let results = (outs Variadic<WeftKernel_ValueType>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def WeftKernel_ScanYieldOp : WeftKernel_Op<"scan_yield", [Pure, Terminator, HasParent<"ScanOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def WeftKernel_Fold2Op : WeftKernel_Op<"fold2", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftKernel_ContractLikeOp<string mnemonic>
    : WeftKernel_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       DenseI64ArrayAttr:$over,
                       OptionalAttr<TypeAttr>:$acc_type);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_DotOp : WeftKernel_ContractLikeOp<"dot">;
def WeftKernel_ContractOp : WeftKernel_ContractLikeOp<"contract">;
def WeftKernel_OuterContractOp
    : WeftKernel_ContractLikeOp<"outer_contract">;

def WeftKernel_LookupOp : WeftKernel_Op<"lookup", [Pure]> {
  let arguments = (ins AnyType:$table, AnyType:$indices, StrAttr:$bounds);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_InterleaveOp : WeftKernel_Op<"interleave", [Pure]> {
  let arguments = (ins WeftKernel_ViewType:$view, I64Attr:$rows,
                       DenseI64ArrayAttr:$field_chunks);
  let results = (outs WeftKernel_ViewType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
