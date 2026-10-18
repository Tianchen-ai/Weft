#ifndef WEFT_DIALECT_RISCV_IR_NUMERICS_H
#define WEFT_DIALECT_RISCV_IR_NUMERICS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <limits>

namespace weft::riscv {
class UnaryOp;

// Bounds cover every ordered value; NaNs are tracked independently. Facts are
// recomputed from numeric SSA, never inferred from an operator name or sampled
// inputs. They do not authorize reassociation or changing a rounding boundary.
struct FloatingBounds {
  double lower = -std::numeric_limits<double>::infinity();
  double upper = std::numeric_limits<double>::infinity();
  bool mayNaN = true;

  bool nonpositive() const { return upper <= 0; }
  bool positiveNormalF32() const;
};

FloatingBounds floatingBounds(mlir::Value value);
llvm::StringRef vectorMathInstruction(UnaryOp operation);
bool isVectorMathInstruction(llvm::StringRef instruction);
bool vectorMathInstructionIsLegal(UnaryOp operation, llvm::StringRef instruction);
} // namespace weft::riscv
#endif
