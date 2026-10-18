#ifndef WEFT_DIALECT_RISCV_IR_DENSEBOUNDS_H
#define WEFT_DIALECT_RISCV_IR_DENSEBOUNDS_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <cstdint>
#include <optional>

namespace weft::riscv {

struct DenseIndexBounds {
  uint64_t lower;
  uint64_t upper;

  std::optional<int64_t> constant() const;
};

// A bounded query of index SSA and explicit descriptor extents. Unknown facts
// remain the entire unsigned index range. This is not a memory-stability or
// speculative-execution proof and never follows a numeric load.
DenseIndexBounds denseIndexBounds(mlir::Value value);
// Facts at a use may additionally follow enclosing SCF branch predicates and
// induction bounds. They must not be substituted at an earlier definition.
DenseIndexBounds denseIndexBoundsAt(mlir::Value value, mlir::Operation *use);
std::optional<bool> denseIndexCompareAt(mlir::arith::CmpIPredicate predicate,
    mlir::Value lhs, mlir::Value rhs, mlir::Operation *use);
std::optional<bool> denseConditionAt(mlir::Value condition, mlir::Operation *use);
std::optional<int64_t> denseConstantIndex(mlir::Value value);

// Resolve an extent without inserting IR. The axis is a descriptor position,
// not a logical axis id. Unknown producers return an empty fold result.
mlir::OpFoldResult denseDescriptorExtent(mlir::Value memory, unsigned axis);
bool sameDenseExtent(mlir::Value extent, mlir::Value memory, unsigned axis);
bool sameDenseIndex(mlir::Value first, mlir::Value second);

} // namespace weft::riscv
#endif
