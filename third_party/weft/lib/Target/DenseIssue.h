#ifndef WEFT_TARGET_DENSE_ISSUE_H
#define WEFT_TARGET_DENSE_ISSUE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace weft::riscv_internal {

// One traversal of an existing numerical axis. Each callback receives its
// actual VL and selected validity; the numerical boundary remains unchanged.
using DenseIssueBody = llvm::function_ref<mlir::FailureOr<llvm::SmallVector<mlir::Value>>(
    mlir::Value offset, mlir::Value active, bool full, mlir::ValueRange carried)>;
mlir::FailureOr<llvm::SmallVector<mlir::Value>> emitDenseIssues(
    mlir::IRRewriter &builder, mlir::Operation *anchor, mlir::Value extent,
    int64_t width, mlir::ValueRange initial, DenseIssueBody body,
    int64_t pipelineDepth = 1, bool includePartial = true, bool registerPieces = false,
    bool separatePartial = false);
mlir::Value completeDenseExtent(mlir::IRRewriter &builder, mlir::Location location,
                                mlir::Value extent, int64_t width);

// Compute peak simultaneous carriers for one selected issue graph. Values
// outside this graph are inputs, not permission to reconstruct their producer.
struct DenseIssueCost {
  int64_t vectorGroups = 0;
  int64_t scalarValues = 0;
  int64_t operations = 0;
};
std::optional<DenseIssueCost> denseIssueCost(
    mlir::Builder &builder, llvm::ArrayRef<mlir::Operation *> order,
    llvm::ArrayRef<mlir::Value> results, llvm::ArrayRef<mlir::Value> exports,
    riscv::ValueType domain, llvm::ArrayRef<int64_t> tile, int64_t axis, riscv::TargetAttr target,
    int64_t accumulatorGroups = 0, int64_t maximumLaneSpan = 0);

mlir::Value denseReductionSeed(mlir::IRRewriter &builder, mlir::Location location,
    riscv::ValueType accumulator, llvm::StringRef kind);
mlir::Value denseReductionStep(mlir::IRRewriter &builder, mlir::Operation *origin,
    mlir::Value input, mlir::Value accumulator, mlir::Value active,
    llvm::StringRef kind);
mlir::Value denseReductionResult(mlir::IRRewriter &builder, mlir::Operation *origin,
    mlir::Value accumulator, int64_t axis, llvm::StringRef kind, mlir::Type resultType = {});

} // namespace weft::riscv_internal
#endif
