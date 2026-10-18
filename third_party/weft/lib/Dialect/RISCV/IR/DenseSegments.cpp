#include "Weft/Dialect/RISCV/IR/DenseSegments.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/DenseRelations.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Verification.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <limits>

using namespace mlir;
using namespace weft::riscv;

std::optional<DenseSegmentFields> DenseSegmentFields::get(
    ValueRange descriptors, ValueType type, int64_t laneAxis) {
  const int64_t fields = descriptors.size();
  if (!llvm::is_contained(ArrayRef<int64_t>({2, 4, 8}), fields) ||
      fields * type.getLayout().getLmulEighths() > 64) return std::nullopt;
  auto tile = VectorTileGeometry::get(type, laneAxis);
  auto fieldsAtBase = denseSiblingFieldViews(descriptors);
  if (fieldsAtBase.size() != descriptors.size()) return std::nullopt;
  descriptors = fieldsAtBase;
  auto first = descriptors.front().getDefiningOp<SliceOp>();
  if (!tile || !first || !isAddressableDenseDescriptor(first.getBase())) return std::nullopt;
  auto base = first.getBase().getType();
  auto memory = first.getResult().getType();
  if (!tileMemoryElementMatches(base, type.getElementType()) ||
      !tileMemoryElementMatches(memory, type.getElementType()) ||
      memory.getStorageBits() != type.getElementType().getIntOrFloatBitWidth() ||
      first.getSelectors().size() != base.getAxisIds().size() ||
      memory.getAxisIds() != type.getAxisIds()) return std::nullopt;
  std::optional<unsigned> fieldPosition, fieldOperand;
  int64_t start = 0;
  unsigned operand = 0;
  for (auto [axis, selector] : llvm::enumerate(first.getSelectors())) {
    auto kind = cast<StringAttr>(selector).getValue();
    if (kind == "all") continue;
    if (kind != "index" && kind != "domain") return std::nullopt;
    if (operand >= first.getIndices().size()) return std::nullopt;
    auto initial = getConstantIntValue(first.getIndices()[operand]);
    bool different = false;
    for (Value descriptor : descriptors.drop_front()) {
      auto slice = descriptor.getDefiningOp<SliceOp>();
      if (!slice || slice.getBase() != first.getBase() || slice.getSelectors() != first.getSelectors() ||
          slice.getIndices().size() != first.getIndices().size() || descriptor.getType() != memory)
        return std::nullopt;
      if (slice.getIndices()[operand] != first.getIndices()[operand] &&
          !sameDenseIndex(slice.getIndices()[operand], first.getIndices()[operand])) different = true;
    }
    if (different) {
      if (fieldPosition || kind != "index" || !initial || *initial < 0 ||
          *initial > std::numeric_limits<int64_t>::max() - fields) return std::nullopt;
      fieldPosition = axis;
      fieldOperand = operand;
      start = *initial;
    }
    ++operand;
  }
  if (!fieldPosition || !fieldOperand || operand != first.getIndices().size()) return std::nullopt;
  for (auto [field, descriptor] : llvm::enumerate(descriptors)) {
    auto slice = descriptor.getDefiningOp<SliceOp>();
    if (getConstantIntValue(slice.getIndices()[*fieldOperand]) != start + int64_t(field))
      return std::nullopt;
  }
  return DenseSegmentFields{first.getBase(), base.getAxisIds()[*fieldPosition],
      *fieldPosition, start, fields, std::move(*tile)};
}

bool DenseSegmentFields::hasCompleteFields(Operation *at) const {
  auto extent = denseDescriptorExtent(base, fieldPosition);
  if (!extent) return false;
  if (auto count = getConstantIntValue(extent)) return *count >= firstField + fields;
  auto index = dyn_cast<Value>(extent);
  return index && denseIndexBoundsAt(index, at).lower >= uint64_t(firstField + fields);
}

bool DenseSegmentFields::hasReadStrides(Operation *at) const {
  return hasUnitStrideProof(at, base, fieldAxis);
}

bool DenseSegmentFields::canStore() const {
  // The current store realization covers complete contiguous records. An
  // arbitrary overlapping lane stride would change the original write order.
  auto memory = cast<MemDescType>(base.getType());
  if (firstField || memory.getShape()[fieldPosition] != fields) return false;
  for (const auto &start : tile.starts)
    for (auto [axis, offset] : llvm::enumerate(start))
      if (axis != tile.lanePosition && offset) return false;
  return true;
}

bool DenseSegmentFields::hasStoreStrides(Operation *at) const {
  return canStore() && hasReadStrides(at) && hasDenseStrideProof(at, base, tile.axis, fields);
}

namespace {
LogicalResult verifyTransfer(Operation *op, ValueRange memory, ValueRange values,
                              ValueRange extents, int64_t axis, LeafAttr leaf, bool store) {
  auto kernel = op->getParentOfType<KernelOp>();
  if (!kernel || memory.empty() || memory.size() != values.size())
    return op->emitOpError("dense segment transfer requires matching fields in a target kernel");
  auto type = dyn_cast<ValueType>(values.front().getType());
  if (!type || !llvm::all_of(values, [&](Value value) { return value.getType() == type; }))
    return op->emitOpError("dense segment transfer must preserve every field's exact numerical type and layout");
  auto geometry = DenseSegmentFields::get(memory, type, axis);
  if (!geometry || !kernel.getTarget().getHasSegmentMemory() ||
      !supportsRVVLayout(kernel.getTarget(), type.getLayout()) ||
      !geometry->hasCompleteFields(op) ||
      !(store ? geometry->hasStoreStrides(op) : geometry->hasReadStrides(op)))
    return op->emitOpError("dense segment transfer lacks exact adjacent fields, complete bounds or a dominating stride proof");
  for (Value descriptor : memory) {
    auto access = cast<MemDescType>(descriptor.getType()).getAccess();
    if ((access != "readwrite" && access != (store ? "write" : "read")) ||
        !tileExtentsMatch(descriptor, type, extents))
      return op->emitOpError("dense segment transfer must retain each descriptor's access and exact active extents");
  }
  int64_t groups = int64_t(memory.size()) * type.getLayout().getRegisterGroups();
  StringRef family = store ? "dense-segment-store" : "dense-segment-load";
  StringRef instruction = store ? "rvv.dense-segment-store" : "rvv.dense-segment-load";
  if (!weft::riscv::detail::exactLeaf(leaf, "transfer", family, instruction, "none", "exact") ||
      leaf.getOperandGroups() != (store ? groups : 0) ||
      leaf.getResultGroups() != (store ? 0 : groups) ||
      leaf.getTemporaryGroups() != (store ? groups : 0) ||
      leaf.getFragmentGroups() || leaf.getLocalBytes() || !leaf.getParameters().empty())
    return op->emitOpError("dense segment transfer requires its exact instruction and complete resident field resources");
  return success();
}
} // namespace

LogicalResult RVVDenseSegmentLoadOp::verify() {
  return verifyTransfer(*this, getSources(), getValues(), getExtents(), getAxis(), getLeaf(), false);
}

LogicalResult RVVDenseSegmentStoreOp::verify() {
  return verifyTransfer(*this, getDestinations(), getValues(), getExtents(), getAxis(), getLeaf(), true);
}
