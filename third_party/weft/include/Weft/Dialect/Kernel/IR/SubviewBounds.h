#ifndef WEFT_DIALECT_KERNEL_IR_SUBVIEWBOUNDS_H
#define WEFT_DIALECT_KERNEL_IR_SUBVIEWBOUNDS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace weft::kernel {

template <typename SubviewOp, typename SliceOp, typename FieldOp>
bool hasDynamicSubviewExtent(mlir::Value region) {
  while (mlir::Operation *producer = region.getDefiningOp()) {
    if (auto subview = mlir::dyn_cast<SubviewOp>(producer)) {
      if (llvm::is_contained(subview.getExtents(), int64_t{-1})) return true;
      region = subview.getBase();
    } else if (auto slice = mlir::dyn_cast<SliceOp>(producer)) region = slice.getBase();
    else if (auto field = mlir::dyn_cast<FieldOp>(producer)) region = field.getOwner();
    else return false;
  }
  return false;
}

// A -1 entry consumes an index SSA operand, offsets first and then extents.
// Symbolic result shapes describe the selected region, not the base extent.
inline mlir::LogicalResult verifySubviewBounds(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> baseShape,
    llvm::ArrayRef<int64_t> resultShape, mlir::DenseI64ArrayAttr offsets,
    mlir::DenseI64ArrayAttr extents, mlir::ValueRange dynamicBounds) {
  const size_t rank = baseShape.size();
  if (offsets.size() != rank || extents.size() != rank ||
      resultShape.size() != rank)
    return operation->emitOpError("subview requires one offset and extent per base axis");
  size_t dynamicCount = 0;
  for (int64_t offset : offsets.asArrayRef()) {
    if (offset < -1)
      return operation->emitOpError("subview offset must be non-negative or dynamic");
    dynamicCount += offset == -1;
  }
  for (int64_t extent : extents.asArrayRef()) {
    if (extent != -1 && extent <= 0)
      return operation->emitOpError("subview extent must be positive or dynamic");
    dynamicCount += extent == -1;
  }
  if (dynamicCount != dynamicBounds.size())
    return operation->emitOpError("subview dynamic bounds do not match their static markers");
  size_t cursor = 0;
  auto constant = [&](int64_t bound) -> std::optional<int64_t> {
    if (bound != -1) return bound;
    mlir::Value value = dynamicBounds[cursor++];
    llvm::APInt integer;
    if (mlir::matchPattern(value, mlir::m_ConstantInt(&integer)))
      return integer.getSExtValue();
    return std::nullopt;
  };
  llvm::SmallVector<std::optional<int64_t>> knownOffsets, knownExtents;
  for (int64_t bound : offsets.asArrayRef()) knownOffsets.push_back(constant(bound));
  for (int64_t bound : extents.asArrayRef()) knownExtents.push_back(constant(bound));
  for (size_t axis = 0; axis < rank; ++axis) {
    auto offset = knownOffsets[axis], extent = knownExtents[axis];
    if ((offset && *offset < 0) || (extent && *extent <= 0))
      return operation->emitOpError("subview bounds require non-negative offsets and positive extents");
    if (offset && extent && baseShape[axis] > 0 &&
        (*extent > baseShape[axis] || *offset > baseShape[axis] - *extent))
      return operation->emitOpError("subview region lies outside its base extent");
    if ((extents[axis] != -1 && resultShape[axis] != extents[axis]) ||
        (resultShape[axis] > 0 && (!extent || *extent != resultShape[axis])))
      return operation->emitOpError("subview result shape must describe the selected extents");
  }
  return mlir::success();
}

} // namespace weft::kernel

#endif
