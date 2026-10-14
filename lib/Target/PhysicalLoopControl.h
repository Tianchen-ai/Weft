#ifndef WEFT_TARGET_PHYSICAL_LOOP_CONTROL_H
#define WEFT_TARGET_PHYSICAL_LOOP_CONTROL_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APInt.h"
#include <optional>

namespace weft::riscv_internal {

// Target-width facts about the existing ordered loop. These are arithmetic
// proofs, not a new loop plan: rewrites still install their choices in SCF.
struct PhysicalLoopControl {
  bool descending;
  unsigned width;
  std::optional<llvm::APInt> lower, upper, step;

  static mlir::FailureOr<PhysicalLoopControl> get(mlir::scf::ForOp loop);
  llvm::APInt coordinate(const llvm::APInt &value) const;
  std::optional<llvm::APInt> count() const;
  bool fitsCoordinate(const llvm::APInt &value) const;
  mlir::LogicalResult verifyFiniteEndpoint(mlir::scf::ForOp loop) const;
};

// Replayed physical IR must describe one direction, including when a Level and
// its lowered loop both carry direction attributes. A malformed attribute is
// never treated as an absent hint.
mlir::FailureOr<bool> descendingPhysicalLoop(mlir::scf::ForOp loop);

} // namespace weft::riscv_internal
#endif
