#ifndef WEFT_TARGET_VALUE_LIVENESS_H
#define WEFT_TARGET_VALUE_LIVENESS_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace weft::riscv_internal {

struct RegisterInterval {
  unsigned begin = 0;
  unsigned end = 0;
  int64_t groups = 0;
  bool fragment = false;
  mlir::Value value;
  bool captured = false;
};

struct RegisterPeak {
  int64_t vector = 0;
  int64_t fragment = 0;
  unsigned count = 0;
  mlir::Block *block = nullptr;
  unsigned point = 0;
  bool before = true;
  int64_t total() const { return vector + fragment; }
};

mlir::Operation *ancestorInBlock(mlir::Operation *operation, mlir::Block *block);

// A view of the current SSA graph. Rebuild after mutation. Storage aliases
// share one interval, including all their last uses in nested regions.
// Unknown or overflowing residency is a failed analysis, never zero cost.
class BlockRegisterLiveness {
public:
  static std::optional<BlockRegisterLiveness> get(
      mlir::Block &block, llvm::ArrayRef<mlir::Value> captures = {});
  llvm::ArrayRef<RegisterInterval> intervals() const { return live; }
  llvm::ArrayRef<mlir::Operation *> operations() const { return ordered; }
  std::optional<unsigned> position(mlir::Operation *operation) const;

  // Include child-region peaks and selected leaf temporaries. The operation
  // window is inclusive; values defined before it remain live when captured.
  std::optional<RegisterPeak> peak(unsigned first, unsigned last) const;
  std::optional<RegisterPeak> peak() const;

private:
  mlir::Block *block = nullptr;
  llvm::SmallVector<RegisterInterval> live;
  llvm::SmallVector<mlir::Operation *> ordered;
  llvm::DenseMap<mlir::Operation *, unsigned> positions;
};

std::optional<RegisterPeak> registerPeak(mlir::Block &block);

// Residency already live at an operation, including captures of enclosing
// blocks. Excluded SSA values must be removed by the caller's rewrite.
std::optional<int64_t> surroundingRegisterGroups(
    mlir::Operation *anchor, llvm::ArrayRef<mlir::Value> excluded = {});

} // namespace weft::riscv_internal
#endif
