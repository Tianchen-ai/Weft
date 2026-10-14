#ifndef WEFT_TARGET_CONTROL_STORAGE_H
#define WEFT_TARGET_CONTROL_STORAGE_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {
// Materialize a local representation at an ordered control edge. The edge
// copies completed numerical values; it never aliases a fresh iteration-local
// object into an enclosing scope or mutates a still-observable incoming value.
mlir::LogicalResult materializeControlStorage(mlir::ModuleOp module,
                                             mlir::IRRewriter &builder);
}
#endif
