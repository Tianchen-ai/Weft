#ifndef WEFT_TARGET_IME_FRAGMENT_SUPPLY_H
#define WEFT_TARGET_IME_FRAGMENT_SUPPLY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv_internal {
mlir::LogicalResult shareFragmentStorage(mlir::ModuleOp module, mlir::IRRewriter &rewriter);
void materializeFragmentRowSupply(riscv::KernelOp kernel, mlir::IRRewriter &rewriter);
void formFragmentWindowLoads(mlir::ModuleOp module, mlir::IRRewriter &rewriter);
}

#endif
