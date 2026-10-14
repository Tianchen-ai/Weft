#pragma once

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace weft::riscv_internal {

// Local shape declarations write a compiler binding resource, not byte storage.
// Their lexical names are versioned together with the descriptor types that
// use them. They must never be treated as freely replayable arithmetic.
bool isPipelineShapeBinding(mlir::Operation *operation);
mlir::LogicalResult verifyPipelineShapeStages(mlir::scf::ForOp loop);
int64_t nextPipelineShapeIdentity(mlir::Operation *root);

// Type-bound extents are dependencies even when an operation does not repeat
// the declaration as an ordinary operand. Schedulers must keep that lexical
// binding, and issue cloning must give each definition a fresh identity.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> physicalShapeDependencies(mlir::Operation *operation);

mlir::Operation *clonePipelineOperation(mlir::IRRewriter &rewriter,
    mlir::Operation *source, mlir::IRMapping &values,
    llvm::DenseMap<mlir::Value, int64_t> &shapeVersions, int64_t &nextIdentity);

} // namespace weft::riscv_internal
