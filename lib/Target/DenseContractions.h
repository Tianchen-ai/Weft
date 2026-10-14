#ifndef WEFT_TARGET_DENSE_CONTRACTIONS_H
#define WEFT_TARGET_DENSE_CONTRACTIONS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {
// Realize closed floating contractions using bounded reduction-lane tiles and
// explicit multi-axis carries. Integer/extension/free-lane contracts have their
// separately selected numerical and operand-supply contracts.
mlir::LogicalResult materializeDenseContractions(mlir::ModuleOp module,
                                               mlir::IRRewriter &rewriter, int64_t maximumLMUL);
}

#endif
