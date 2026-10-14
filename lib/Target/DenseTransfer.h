#ifndef WEFT_TARGET_DENSETRANSFER_H
#define WEFT_TARGET_DENSETRANSFER_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {

// Numeric capacity and accessible memory extent are independent. A selected
// issue may contain four lanes while its last window contains one element.
// These builders materialize that distinction as descriptor SSA and typed
// transfers; no descriptor pretends that padding belongs to the source object.
struct DenseTransferWindow {
  mlir::Value memory;
  llvm::SmallVector<mlir::Value> extents;
};

int64_t lastLocalShapeIdentity(mlir::ModuleOp module);
mlir::FailureOr<DenseTransferWindow> denseTransferWindow(
    mlir::IRRewriter &builder, mlir::Value memory,
    llvm::ArrayRef<mlir::Value> offsets, llvm::ArrayRef<int64_t> capacity,
    llvm::ArrayRef<mlir::Value> active,
    int64_t &shapeIdentity, mlir::Operation *origin);

mlir::FailureOr<mlir::Value> loadDenseWindow(mlir::IRRewriter &builder,
    const DenseTransferWindow &window, riscv::ValueType numeric,
    mlir::Operation *origin);
mlir::LogicalResult storeDenseWindow(mlir::IRRewriter &builder,
    mlir::Value numeric, const DenseTransferWindow &window,
    mlir::Operation *origin);

} // namespace weft::riscv_internal
#endif
