#ifndef WEFT_DIALECT_KERNEL_IR_SHAPEBINDINGS_H
#define WEFT_DIALECT_KERNEL_IR_SHAPEBINDINGS_H

#include "mlir/IR/Operation.h"

namespace weft {
template <typename ShapeOp>
mlir::Value localShapeExtent(mlir::Operation *anchor, int64_t id) {
  for (mlir::Operation *scope = anchor; scope; scope = scope->getParentOp()) {
    for (mlir::Operation *previous = scope->getBlock() ? scope->getPrevNode() : nullptr; previous;
         previous = previous->getPrevNode())
      if (auto shape = mlir::dyn_cast<ShapeOp>(previous); shape && shape.getId() == id)
        return shape.getResult();
    if (scope->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) break;
  }
  return {};
}
} // namespace weft

#endif
