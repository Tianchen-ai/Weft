#ifndef WEFT_DIALECT_RISCV_IR_FMAISSUE_H
#define WEFT_DIALECT_RISCV_IR_FMAISSUE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace weft::riscv {

// Pointwise contraction has no reduction or closed scheduling frontier. The
// vector operand is lhs; rhs is either the identical carrier or a scalar.
llvm::StringRef pointwiseFMAInstruction(mlir::Type lhs, mlir::Type rhs,
                                      mlir::Type result);

// A finite set of independent lane-wise FMA updates. All operands are numeric
// values; memory supply, reduction traversal and initialization remain IR.
struct FMAIssueGeometry {
  bool scalar = false;
  bool widening = false;
  int64_t lanes = 0;
  int64_t inputGroups = 0;
  int64_t outputGroups = 0;
  llvm::SmallVector<int64_t> vectorParts;
  llvm::SmallVector<int64_t> multiplierParts;

  static std::optional<FMAIssueGeometry> get(ValueType vector,
                                            mlir::Type multiplier,
                                            ValueType result);
  llvm::StringRef instruction() const;
  int64_t parts() const { return vectorParts.size(); }
};

bool canReuseFMAAccumulator(mlir::Operation *operation, mlir::Value accumulator);
int64_t fmaIssueCopyGroups(RVVFMAIssueOp operation);

// End offsets partition the existing result parts. They fix the assembly
// operand frontier, never the author's shape or the reduction iteration order.
std::optional<llvm::SmallVector<int64_t>> selectFMAIssueGroups(
    const FMAIssueGeometry &geometry, int64_t registerBudget);
bool legalFMAIssueGroups(const FMAIssueGeometry &geometry,
                         llvm::ArrayRef<int64_t> ends, int64_t registerBudget);

} // namespace weft::riscv
#endif
