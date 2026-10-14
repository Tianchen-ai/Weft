#ifndef WEFT_DIALECT_RISCV_IR_CONTRACTION_H
#define WEFT_DIALECT_RISCV_IR_CONTRACTION_H

#include "Weft/Dialect/RISCV/IR/Layout.h"

namespace weft::riscv {

// A scalar/vector FMA over already projected numeric operands. The reduction
// coordinate has been consumed by explicit reads or register projections.
// Part maps are derived from typed coordinates, never from memory provenance.
struct ContractStepGeometry {
  llvm::SmallVector<int64_t> laneParts;
  llvm::SmallVector<int64_t> scalarParts;
  bool widening = false;
  bool floating = false;

  static std::optional<ContractStepGeometry> get(
      mlir::Type lhs, mlir::Type rhs, ValueType result,
      int64_t reductionAxis, llvm::StringRef laneOperand,
      llvm::StringRef instruction);
};

} // namespace weft::riscv
#endif
