#ifndef WEFT_KERNEL_ATOMIC_H
#define WEFT_KERNEL_ATOMIC_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"

namespace weft {
inline mlir::LogicalResult verifyAtomicContract(mlir::Operation *op, mlir::Type element,
    llvm::StringRef kind, llvm::StringRef ordering, llvm::StringRef scope) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  auto floating = mlir::dyn_cast<mlir::FloatType>(element);
  if ((!integer && !floating) || (element.getIntOrFloatBitWidth() != 16 &&
      element.getIntOrFloatBitWidth() != 32 && element.getIntOrFloatBitWidth() != 64))
    return op->emitOpError("pointer atomic storage must be a 16, 32 or 64 bit number");
  bool integral = kind == "add" || kind == "and" || kind == "or" || kind == "xor" ||
                  kind == "min" || kind == "max" || kind == "umin" || kind == "umax";
  if ((integral && !integer) || (kind == "fadd" && !floating) ||
      (!integral && kind != "fadd" && kind != "xchg" && kind != "cas"))
    return op->emitOpError("atomic operation and storage dtype disagree");
  if (ordering != "relaxed" && ordering != "acquire" && ordering != "release" && ordering != "acq_rel")
    return op->emitOpError("unsupported atomic ordering");
  if (scope != "cta" && scope != "gpu" && scope != "sys")
    return op->emitOpError("unsupported atomic synchronization scope");
  return mlir::success();
}
} // namespace weft
#endif
