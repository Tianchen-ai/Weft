#ifndef WEFT_DIALECT_RISCV_IR_STREAM_H
#define WEFT_DIALECT_RISCV_IR_STREAM_H

#include "Weft/Dialect/RISCV/IR/Layout.h"

namespace weft::riscv {

// One lane axis and explicit free-axis replicas. Coordinates are shared by
// verification and emission; neither consumer may invent another mapping.
struct StreamGeometry {
  int64_t axis;
  int64_t lanes;
  int64_t parts;
  llvm::SmallVector<int64_t> freeAxes;
  llvm::SmallVector<int64_t> freeShape;
  static std::optional<StreamGeometry> get(ValueType type, int64_t axis);
};

struct StreamContractGeometry {
  llvm::SmallVector<int64_t> lhsParts;
  llvm::SmallVector<int64_t> rhsParts;
  static std::optional<StreamContractGeometry> get(
      ValueType lhs, ValueType rhs, ValueType accumulator, int64_t axis);
};

// Maps each scalar output carrier to exactly one complete lane accumulator.
std::optional<llvm::SmallVector<int64_t>> streamFinalizeParts(
    ValueType input, mlir::Type output, int64_t axis);

} // namespace weft::riscv
#endif
