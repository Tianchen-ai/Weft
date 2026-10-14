#ifndef WEFT_LIB_TARGET_RISCVPHYSICALSUPPORT_H
#define WEFT_LIB_TARGET_RISCVPHYSICALSUPPORT_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace weft::riscv_internal {

// Half storage and f16/f32 conversions are available with Zvfhmin; other
// selected half-vector operations require the full Zvfh instruction interface.
bool requiresFullVectorF16(mlir::Operation *operation);

mlir::FailureOr<int64_t> guardedLocalStorageBytes(riscv::KernelOp kernel);

bool hasOwnedLocalStorage(mlir::Value value);
std::optional<int64_t> ownedLocalStorageAlignment(mlir::Value value);
int64_t indexedEntrySourceAlignment(mlir::Value source);
int64_t indexedEntryAlignment(mlir::Value source, mlir::Value byteOffsets,
                              int64_t entryByteStride);

struct FieldFacts {
  llvm::StringRef mapping = "opaque";
  bool scalarPerRecord = false;
  int64_t logicalRank = 0;
  int64_t group = 0;
  int64_t layer = 0;
  int64_t joinFields = 0;
  int64_t joinLowBits = 0;
  int64_t joinRole = 0;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
  int64_t chunkBytes = 1;
  int64_t alignment = 1;
  llvm::StringRef order = "none";
};

struct IntegerRange {
  int64_t minimum;
  int64_t maximum;
};

struct WidenDotLaneSlicePlan {
  int64_t reductionLanes = 0;
  int64_t reductionStreams = 0;
  int64_t sliceLmulEighths = 0;
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> parts;
};

struct IndexedEntryRelation {
  mlir::Value entryIndices;
  int64_t payloadAxis = 0;
  int64_t payloadExtent = 0;
  int64_t entryStride = 0;
  int64_t indexDivisor = 1;
};

struct RegularIndexRelation {
  mlir::Value dynamicBase;
  int64_t base = 0;
  int64_t stride = 1;
  int64_t repeat = 1;
};

mlir::DenseI64ArrayAttr integers(mlir::Builder &builder,
                                 llvm::ArrayRef<int64_t> values);
mlir::ArrayAttr strings(mlir::Builder &builder,
                        llvm::ArrayRef<std::string> values);
mlir::DictionaryAttr dictionary(
    mlir::Builder &builder,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> values);
std::optional<int64_t> integer(mlir::DictionaryAttr dictionary,
                               llvm::StringRef name);
std::optional<llvm::StringRef> string(mlir::DictionaryAttr dictionary,
                                      llvm::StringRef name);
mlir::ArrayAttr array(mlir::DictionaryAttr dictionary, llvm::StringRef name);

riscv::LayoutAttr unassignedLayout(mlir::Builder &builder,
                                   llvm::ArrayRef<int64_t> axes);
riscv::LayoutAttr scalarLayout(mlir::Builder &builder, mlir::Type element,
                               llvm::ArrayRef<int64_t> shape,
                               llvm::ArrayRef<int64_t> axes,
                               llvm::StringRef validity = "full");
riscv::LayoutAttr projectLayout(mlir::Builder &builder, riscv::ValueType source,
                                riscv::LayoutAttr reference,
                                riscv::TargetAttr target);
riscv::AccessAttr unassignedAccess(mlir::Builder &builder);
riscv::AccessAttr denseAccess(mlir::Builder &builder,
                              riscv::MemDescType memory,
                              mlir::Type physicalValue);
riscv::ConversionAttr conversion(mlir::Builder &builder,
                                 llvm::StringRef kind,
                                 llvm::StringRef effect = "pure",
                                 int64_t temporaryGroups = 0);
riscv::ConversionAttr layoutConversion(mlir::Builder &builder,
                                       riscv::LayoutAttr source,
                                       riscv::LayoutAttr target);
bool selectLaneRepack(mlir::Builder &builder, riscv::ConvertLayoutOp operation);
riscv::ScheduleAttr schedule(mlir::Builder &builder, int64_t unroll,
                             int64_t pipelineDepth,
                             int64_t prefetchDistance = 0);
riscv::ImplementationAttr implementation(
    mlir::Builder &builder, llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef operation, llvm::ArrayRef<int64_t> parameters = {});
riscv::LeafAttr unselectedLeaf(mlir::Builder &builder);
riscv::LeafAttr readSnapshotLeaf(mlir::Builder &builder,
                                  riscv::TargetAttr target, int64_t bytes);
riscv::LeafAttr leaf(mlir::Builder &builder, llvm::StringRef engine,
                     llvm::StringRef family, llvm::StringRef instruction,
                     llvm::StringRef spelling, int64_t operandGroups,
                     int64_t resultGroups, int64_t temporaryGroups = 0,
                     int64_t fragmentGroups = 0,
                     llvm::StringRef mask = "none",
                     llvm::StringRef tail = "exact",
                     llvm::ArrayRef<int64_t> parameters = {},
                     int64_t localBytes = 0);
riscv::TargetAttr target(mlir::Builder &builder,
                         const RISCVTargetProfile &profile);

mlir::Type logicalElement(mlir::Type type);
llvm::ArrayRef<int64_t> logicalShape(mlir::Type type);
llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type);
unsigned logicalBitWidth(mlir::Type type);
riscv::LayoutAttr layoutOf(mlir::Type type);
int64_t issueRegisterGroups(mlir::Type type);
std::optional<int64_t> residentRegisterGroups(mlir::Value value);
int64_t issueRegisterGroups(mlir::Value value);
mlir::Value registerStorageIdentity(mlir::Value value);
struct RegisterPressure {
  int64_t vector;
  int64_t fragment;
  mlir::Operation *operation = nullptr;
  bool before = false;
  int64_t total() const { return vector + fragment; }
};
std::optional<RegisterPressure> physicalRegisterPressure(riscv::KernelOp kernel);
void selectFragmentRegisterReuse(riscv::KernelOp kernel);
mlir::Type withLayout(mlir::Type type, riscv::LayoutAttr layout);
int64_t physicalExtent(mlir::Value value, int64_t axis);
std::string terminalInstruction(mlir::Operation *operation);
bool isSameWidthIntegerCast(mlir::Operation *operation);
std::optional<int64_t> booleanComparisonTemporaryGroups(riscv::CompareOp operation);
std::optional<int64_t> localGatherTemporaryGroups(riscv::ExtractOp operation);
mlir::Value shapeExtentAt(mlir::Operation *anchor, int64_t symbolId);
mlir::FailureOr<llvm::SmallVector<mlir::Value>> valueExtentsAt(
    mlir::OpBuilder &builder, mlir::Operation *anchor, riscv::ValueType type);
std::optional<int64_t> storageFloatCastTemporaryGroups(mlir::Operation *operation);
bool isLocalDenseRowExtract(riscv::ExtractOp operation);
llvm::StringRef layeredStorageDecodeInstruction(int64_t shiftAmount,
                                                int64_t maskValue);

std::optional<int64_t> staticProduct(llvm::ArrayRef<int64_t> values);
std::optional<WidenDotLaneSlicePlan>
planWidenDotLaneSlices(riscv::ValueType operand, riscv::ValueType result,
                       llvm::ArrayRef<int64_t> reductionAxes,
                       riscv::TargetAttr target);
std::optional<IndexedEntryRelation>
analyzeIndexedEntryRelation(mlir::Value fullIndices, riscv::ValueType result,
                            llvm::ArrayRef<int64_t> retainedAxes,
                            llvm::ArrayRef<int64_t> retainedShape);
std::optional<RegularIndexRelation>
analyzeRegularIndexRelation(mlir::Value value, unsigned depth = 0);
std::string printType(mlir::Type type);

riscv::EncodingDeclOp findEncoding(mlir::Operation *operation,
                                   llvm::StringRef family);
llvm::StringRef baseEncodingFamily(mlir::Operation *operation,
                                   kernel::EncodingType encoding);
int64_t interleaveRows(mlir::Operation *operation,
                       kernel::EncodingType encoding);
FieldFacts fieldFacts(riscv::FieldOp operation);
std::optional<riscv::AccessAttr> contiguousInterleavedReadAccess(
    mlir::Value projection, riscv::ValueType result);
mlir::LogicalResult planFieldStore(mlir::IRRewriter &rewriter,
                                   riscv::StoreOp operation);

std::optional<int64_t> constantInt(mlir::Value value);
bool knownMultipleOf(mlir::Value value, int64_t divisor,
                     unsigned depth = 0);
std::optional<IntegerRange> integerRange(mlir::Value value,
                                         unsigned depth = 0);
std::optional<int64_t> maximumMagnitude(mlir::Value value);
mlir::Value stripRepresentationConversions(
    mlir::Value value,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr);
riscv::FieldOp sourceField(mlir::Value value);
riscv::LoadOp sourceLoad(mlir::Value value);
bool needsReadSnapshot(riscv::LoadOp load);
riscv::AccessAttr accessOf(mlir::Value value);
riscv::PhysicalPointOp originPoint(mlir::Value value, int64_t axis);
std::optional<riscv::StorageWindowPlanAttr>
storageWindowPlan(mlir::Builder &builder, riscv::FieldOp field,
                  int64_t reductionAxis, int64_t projectionBase,
                  int64_t projectionStride, int64_t projectionRepeat,
                  int64_t projectionExtent, int64_t offsetAlignment,
                  bool recordProjection = false);

void copyOrigin(mlir::Operation *source, mlir::Operation *target);

} // namespace weft::riscv_internal

#endif
