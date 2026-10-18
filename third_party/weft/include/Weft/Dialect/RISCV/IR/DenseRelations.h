#ifndef WEFT_DIALECT_RISCV_IR_DENSERELATIONS_H
#define WEFT_DIALECT_RISCV_IR_DENSERELATIONS_H

#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "mlir/IR/ValueRange.h"

namespace weft::riscv {

// Relations between logical coordinates of two dense projections. None of
// these flags asserts that different coordinates denote different addresses.
// That additional fact requires an injective full-view mapping at the use.
struct DenseProjectionRelation {
  mlir::Value lhsBase;
  mlir::Value rhsBase;
  bool identical = false;
  bool pointwise = false;
  bool disjoint = false;
};

// Strip a common, coordinate-preserving subview suffix from sibling fields.
// Descriptor shape identities may differ when their extent SSA is equal.
llvm::SmallVector<mlir::Value> denseSiblingFieldViews(mlir::ValueRange views);

std::optional<DenseProjectionRelation> denseProjectionRelation(
    mlir::Value lhs, mlir::Value rhs);
bool hasInjectiveDenseMappingAt(mlir::Value fullView, mlir::Operation *at);
bool haveEqualDenseMappingsAt(mlir::Value lhs, mlir::Value rhs,
                              mlir::Operation *at);
bool haveDisjointDenseProjectionsAt(mlir::Value lhs, mlir::Value rhs,
                                   mlir::Operation *at);

} // namespace weft::riscv
#endif
