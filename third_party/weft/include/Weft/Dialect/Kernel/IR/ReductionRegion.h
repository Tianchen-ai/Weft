#ifndef WEFT_KERNEL_REDUCTIONREGION_H
#define WEFT_KERNEL_REDUCTIONREGION_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace weft {

// Canonical and Physical reductions share the tuple/axis/combiner contract.
// Target layouts and a particular reduction tree are deliberately absent here.
template <typename ValueType, typename YieldType, typename ScalarOperation>
mlir::LogicalResult verifyReductionRegion(mlir::Operation *op, int64_t axis,
                                          mlir::Region &region,
                                          ScalarOperation scalarOperation,
                                          bool preserveAxis = false) {
  const size_t count = op->getNumOperands();
  if (!count || count != op->getNumResults() || !llvm::hasSingleElement(region))
    return op->emitOpError("tuple reduction needs matching nonempty inputs/results and one combiner block");
  auto first = mlir::cast<ValueType>(op->getOperand(0).getType());
  if (axis < 0 || axis >= static_cast<int64_t>(first.getShape().size()))
    return op->emitOpError("reduction axis is outside the input rank");
  llvm::SmallVector<int64_t> shape, axes;
  for (auto [position, size] : llvm::enumerate(first.getShape().asArrayRef()))
    if (preserveAxis || position != static_cast<size_t>(axis)) {
      shape.push_back(size);
      axes.push_back(first.getAxisIds()[position]);
    }
  auto numeric = [](mlir::Type type) { return mlir::isa<mlir::IntegerType, mlir::FloatType>(type); };
  auto &body = region.front();
  if (body.getNumArguments() != count * 2)
    return op->emitOpError("combiner arguments must be the complete left tuple followed by the right tuple");
  auto yield = body.empty() ? YieldType() : mlir::dyn_cast<YieldType>(body.back());
  if (!yield || yield->getNumOperands() != count)
    return op->emitOpError("combiner must yield one scalar per tuple member");
  for (size_t i = 0; i < count; ++i) {
    auto input = mlir::cast<ValueType>(op->getOperand(i).getType());
    auto element = input.getElementType();
    auto output = mlir::dyn_cast<ValueType>(op->getResult(i).getType());
    bool matches = output ? llvm::equal(output.getShape().asArrayRef(), shape) &&
        llvm::equal(output.getAxisIds().asArrayRef(), axes) && output.getElementType() == element
        : shape.empty() && op->getResult(i).getType() == element;
    if (!numeric(element) || input.getShape() != first.getShape() || input.getAxisIds() != first.getAxisIds() || !matches ||
        body.getArgument(i).getType() != element || body.getArgument(count + i).getType() != element ||
        yield->getOperand(i).getType() != element)
      return op->emitOpError("tuple members must share a domain and preserve their own scalar dtype through the combiner");
  }
  for (auto &inner : body.without_terminator()) {
    if (!scalarOperation(&inner) || inner.getNumRegions() || inner.getNumResults() != 1 ||
        !llvm::all_of(inner.getOperandTypes(), numeric) || !llvm::all_of(inner.getResultTypes(), numeric))
      return inner.emitOpError("reduction combiner requires closed pure scalar numerical operations");
  }
  return mlir::success();
}
} // namespace weft
#endif
