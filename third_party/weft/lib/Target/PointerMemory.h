#ifndef WEFT_TARGET_POINTER_MEMORY_H
#define WEFT_TARGET_POINTER_MEMORY_H
#include "mlir/IR/BuiltinOps.h"
namespace weft::riscv_internal {
mlir::LogicalResult selectPointerMemory(mlir::ModuleOp module);
}
#endif
