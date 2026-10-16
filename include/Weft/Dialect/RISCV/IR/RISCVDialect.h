#ifndef WEFT_DIALECT_RISCV_IR_RISCVDIALECT_H
#define WEFT_DIALECT_RISCV_IR_RISCVDIALECT_H

#include "Weft/Dialect/RISCV/IR/RISCVOpsTraits.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#include "Weft/Dialect/RISCV/IR/RISCVOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVTypes.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVOps.h.inc"

namespace weft::riscv {

/// ISA extension use of the selected kernel, not its target capabilities.
mlir::FailureOr<llvm::SmallVector<llvm::StringRef>>
usedInstructionExtensions(KernelOp kernel);

bool knownIndexMultipleOf(mlir::Value value, int64_t divisor, unsigned depth = 0);

/// Returns whether an RVV layout is executable under the complete target
/// register contract, including fractional LMUL, VLMAX, and register groups.
bool supportsRVVLayout(TargetAttr target, LayoutAttr layout);
bool supportsRVVRegisterSlice(TargetAttr target, ValueType input, ValueType result,
                              llvm::ArrayRef<int64_t> offsets);
bool supportsRVVPartialPackedScale(TargetAttr target,
                                  llvm::ArrayRef<PartialSetType> inputs,
                                  llvm::ArrayRef<ValueType> scales,
                                  PartialSetType result);
bool supportsRVVMaskedNegate(TargetAttr target, ValueType data, ValueType mask);
bool supportsRVVWidenBinary(TargetAttr target, ValueType lhs, ValueType rhs,
                            ValueType result);
bool supportsRVVJoinedUnitGather(RVVRegularRepeatGatherOp operation);

/// A stream accumulator defines every physical lane, including the neutral
/// lanes never visited by a short input. Its memory operands retain tail VL.
ValueType streamAccumulatorType(ValueType window);

/// Returns the complete physical lane span of one RVV value.
std::optional<int64_t> rvvLaneCount(ValueType value);

/// Returns the RVV LMUL, in eighths, of a contiguous lane slice. Fractional
/// source groups retain their smallest addressable carrier; wider groups may
/// expose an m1 subgroup.
std::optional<int64_t> rvvLaneSliceLMULEighths(TargetAttr target,
                                             ValueType source,
                                             int64_t sliceLanes);

/// Smallest addressable group containing every contiguous repack window.
/// The selected carrier can be wider than a fractional result; the remaining
/// intra-group offset is an explicit slide, not a fractional vget.
std::optional<int64_t> rvvPartialRepackCarrierLMULEighths(
    TargetAttr target, ValueType source, ValueType result, int64_t split);

/// Returns the number of source lane pieces packed by one RVV part-to-lane
/// conversion. A moved lane axis may draw its additional pieces from issue-time
/// or register-replica coordinates; the complete source/result layouts may also
/// repartition unaffected coordinates. No logical coordinate may be lost or
/// duplicated.
std::optional<int64_t> rvvPartToLanePieces(ValueType source, ValueType result);

/// Storage projection identity and the address values it captures. Numeric
/// register extracts are not reloadable storage projections.
FieldOp fieldReadProjection(mlir::Value value,
                            llvm::SmallVectorImpl<mlir::Value> *indices = nullptr);
std::optional<int64_t> encodedFieldRawLMULEighths(ValueType result);
bool isContiguousInterleavedScalarRead(mlir::Value input, ValueType result,
                                      AccessAttr access);
llvm::StringRef fieldReadInstruction(mlir::Value input, ValueType result,
                                     AccessAttr access);
/// Upper bound for the selected fixed load/decode sequence, including its
/// output assembly. Address operands themselves are separate SSA live values.
std::optional<int64_t> fieldReadTemporaryGroups(
    mlir::Value input, ValueType result, AccessAttr access, mlir::ValueRange indices);
bool hasContiguousLayeredStreamStorage(FieldOp field, ValueType result,
                                      LayeredStreamGeometryAttr geometry);

/// Returns the bit offset of one logical field element under the declared
/// natural or grouped/layered storage geometry.
std::optional<int64_t> recordFieldRelativeBit(AccessAttr access,
                                              int64_t elementBits,
                                              int64_t logicalIndex);

/// Projects every result part onto the packed operand address identity used by
/// grouped MAC. Equal entries denote one typed packed supply shared by those
/// result consumers.
std::optional<llvm::SmallVector<int64_t>>
groupedMacSupplyProjection(ValueType packed, ValueType result);

/// Returns the logical bit offset owned by every physical RVV part of a
/// bitmask window. Window axes are flattened in their declared logical order;
/// retained register axes select the surrounding record and do not contribute
/// to the bit offset inside that record.
std::optional<llvm::SmallVector<int64_t>>
bitmaskWindowPartOffsets(ValueType result, llvm::ArrayRef<int64_t> windowAxes,
                         llvm::ArrayRef<int64_t> windowExtents);

} // namespace weft::riscv

#endif // WEFT_DIALECT_RISCV_IR_RISCVDIALECT_H
