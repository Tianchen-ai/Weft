#ifndef WEFT_DIALECT_RISCV_IR_LAYOUT_H
#define WEFT_DIALECT_RISCV_IR_LAYOUT_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv {
struct RegisterLocation {
  int64_t part;
  int64_t lane;
};

// Bidirectional coordinates for an already selected register representation.
// The layout remains the only authority: part = replica-group * streams +
// time, and the grouped replica is the outermost coordinate inside a carrier.
// Scalar carriers can represent repeated lanes, but still have one C scalar
// per part. This class does not choose a layout or authorize a conversion.
class RegisterCoordinates {
public:
  static std::optional<RegisterCoordinates> get(LayoutAttr layout);

  int64_t parts() const { return partCount; }
  int64_t lanes() const { return laneCount; }
  int64_t streams() const { return streamCount; }
  llvm::ArrayRef<int64_t> extents() const { return axisExtents; }
  std::optional<llvm::SmallVector<int64_t>> logical(int64_t part,
                                                 int64_t lane = 0) const;
  std::optional<RegisterLocation> locate(llvm::ArrayRef<int64_t> logical) const;

private:
  RegisterCoordinates(LayoutAttr layout, int64_t streams, int64_t parts,
                      int64_t lanes, llvm::SmallVector<int64_t> extents)
      : layout(layout), streamCount(streams), partCount(parts), laneCount(lanes),
        axisExtents(std::move(extents)) {}
  LayoutAttr layout;
  int64_t streamCount, partCount, laneCount;
  llvm::SmallVector<int64_t> axisExtents;
};

// Project result coordinates onto the source axes. Every lane in this result
// carrier must select the same source part. Lane permutations/broadcasts still
// require their own typed operation; this function never creates one.
std::optional<int64_t> projectCarrierPart(ValueType source, ValueType result,
                                         int64_t resultPart);

struct RVVLayoutSlice {
  int64_t sourcePart;
  int64_t sourceOffset;
  int64_t resultPart;
  int64_t resultOffset;
  int64_t count;
};

int64_t replicaGroupAxis(LayoutAttr layout);
// A full-rank reshape that only renames/permutates complete logical axes.
// The returned positions map each result axis to its input axis.
std::optional<llvm::SmallVector<unsigned>> reshapeAxisPositions(
    ValueType input, ValueType result, llvm::ArrayRef<int64_t> order);
bool hasIdenticalReshapeCarrier(ValueType input, ValueType result,
                               llvm::ArrayRef<int64_t> order);
// Row-major reassociation that preserves the physical part and lane address.
bool hasIdenticalLinearCarrier(ValueType input, ValueType result);
// Whole-register renaming only: every destination lane keeps its source lane.
std::optional<llvm::SmallVector<int64_t>> reshapeCarrierParts(
    ValueType input, ValueType result, llvm::ArrayRef<int64_t> order);
std::optional<int64_t> rvvCarrierCount(LayoutAttr layout);
std::optional<llvm::SmallVector<int64_t>> rvvLaneCoordinates(
    LayoutAttr layout, int64_t part, int64_t lane);

// Exact ordered slices for a bounded, same-domain RVV representation change.
// No numerical permutation, memory access, or implicit layout choice occurs.
std::optional<llvm::SmallVector<RVVLayoutSlice>>
rvvLayoutRepackSlices(ValueType input, ValueType result);
bool isRVVSubregisterRepack(TargetAttr target, ValueType input, ValueType result);
int64_t rvvLayoutRepackTemporaryGroups(TargetAttr target, ValueType input, ValueType result);
std::optional<LayoutAttr> fragmentRVVHandoffLayout(FragmentType input,
                                                  TargetAttr target,
                                                  int64_t replicaGroup = 1);
bool isDirectFragmentRVVHandoff(FragmentType input, ValueType result);
bool isGroupedFragmentRVVHandoff(FragmentType input, ValueType result);
int64_t fragmentUnpackTemporaryGroups(FragmentType input, ValueType result);
bool needsRVVAxisBroadcast(mlir::Type input, mlir::Type result);
std::optional<int64_t> rvvBroadcastSourcePart(ValueType input, ValueType result,
                                            int64_t resultPart);
std::optional<int64_t> binaryBroadcastTemporaryGroups(BinaryOp operation);
std::optional<int64_t> selectTemporaryGroups(SelectOp operation);
std::optional<int64_t> unaryTemporaryGroups(llvm::StringRef kind, ValueType result);
std::optional<int64_t> unaryTemporaryGroups(UnaryOp operation, ValueType result,
                                          llvm::StringRef instruction = {});
std::optional<int64_t> binaryBroadcastInstructionCount(BinaryOp operation);
std::optional<int64_t> axisBroadcastInstructionCount(ValueType input, ValueType result);
std::optional<int64_t> axisBroadcastTemporaryGroups(RVVAxisBroadcastOp operation);
std::optional<int64_t> rvvRowScaleRows(BinaryOp operation);
std::optional<int64_t> rvvRowScaleTemporaryGroups(BinaryOp operation);
mlir::Value registerAliasSource(mlir::Value value);
std::optional<llvm::SmallVector<int64_t>> scalarExtractParts(ExtractOp operation);
std::optional<llvm::SmallVector<int64_t>> rvvRegularGatherIndices(
    ValueType input, ValueType result, llvm::ArrayRef<int64_t> pattern);
bool hasUniqueRegisterUse(mlir::Value value);
} // namespace weft::riscv

#endif
