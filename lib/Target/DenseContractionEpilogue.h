#ifndef WEFT_TARGET_DENSE_CONTRACTION_EPILOGUE_H
#define WEFT_TARGET_DENSE_CONTRACTION_EPILOGUE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

namespace weft::riscv_internal {

struct ContractionEpilogueGraph {
  mlir::Value root;
  riscv::StoreOp store;
  llvm::SmallVector<mlir::Operation *> adapters;
  llvm::SmallVector<mlir::Value> inputs;
  mlir::Value storedValue() const { return store->getOperand(0); }
  mlir::Value destination() const { return store->getOperand(1); }
};

riscv::ValueType contractionEpiloguePiece(
    mlir::Builder &builder, mlir::Type type, llvm::ArrayRef<int64_t> axes,
    llvm::ArrayRef<int64_t> shape, int64_t laneAxis, int64_t lanes,
    riscv::TargetAttr target);

std::optional<int64_t> contractionEpiloguePeak(
    mlir::Builder &builder, const ContractionEpilogueGraph &graph,
    llvm::ArrayRef<int64_t> axes, llvm::ArrayRef<int64_t> shape,
    int64_t laneAxis, int64_t lanes, riscv::TargetAttr target);

// All reads have been proven stable at the original store frontier. The
// selected representation fixes scalar versus vector execution before this
// materializer runs; loading at first use shortens physical live ranges.
mlir::LogicalResult materializeContractionEpilogue(
    mlir::IRRewriter &builder, mlir::Operation *origin,
    const ContractionEpilogueGraph &graph, riscv::ContractionEpilogueAttr selected,
    mlir::Value input, mlir::ValueRange offsets, mlir::ValueRange active,
    const llvm::DenseMap<mlir::Value, mlir::Value> &readViews,
    int64_t &shapeIdentity);

} // namespace weft::riscv_internal
#endif
