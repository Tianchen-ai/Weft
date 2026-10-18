#ifndef WEFT_DIALECT_KERNEL_IR_STRIDEABI_H
#define WEFT_DIALECT_KERNEL_IR_STRIDEABI_H
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"

namespace weft {
inline std::string strideArgumentName(unsigned argument, unsigned axis) {
  return "_weft_stride_" + std::to_string(argument) + "_" + std::to_string(axis);
}
inline bool hasStrideArguments(mlir::ArrayAttr declared, mlir::Attribute name) {
  return declared && llvm::is_contained(declared, name);
}
template <typename ViewType>
mlir::LogicalResult verifyStrideABI(mlir::Operation *op, mlir::Block &body,
                                   mlir::ArrayAttr names, mlir::ArrayAttr shapes,
                                   mlir::ArrayAttr declared) {
  if (!declared) return mlir::success();
  llvm::SmallSet<llvm::StringRef, 8> seen;
  for (mlir::Attribute attr : declared) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(attr);
    auto found = llvm::find(names, attr);
    if (!name || found == names.end() || !seen.insert(name.getValue()).second)
      return op->emitOpError("strided_args must name distinct View arguments");
    auto index = static_cast<unsigned>(found - names.begin());
    auto view = mlir::dyn_cast<ViewType>(body.getArgument(index).getType());
    if (!view || mlir::cast<kernel::EncodingType>(view.getEncoding()).getKind() != "dense")
      return op->emitOpError("runtime strides require a dense View argument");
    for (unsigned axis = 0; axis < view.getShape().size(); ++axis) {
      auto symbol = mlir::StringAttr::get(op->getContext(), strideArgumentName(index, axis));
      if (llvm::is_contained(names, symbol) || llvm::is_contained(shapes, symbol))
        return op->emitOpError("argument or shape name collides with a stride ABI parameter");
    }
  }
  return mlir::success();
}
} // namespace weft
#endif
