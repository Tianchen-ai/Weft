#pragma once

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/Builders.h"

namespace weft::riscv {

struct PanelTransferGeometry {
  mlir::Type element;
  llvm::SmallVector<int64_t, 2> axes;
  TargetAttr target;

  static std::optional<PanelTransferGeometry> get(DensePanelCopyOp operation);
  int64_t fieldAxis(int64_t copyAxis) const;
  ValueType numeric(mlir::Builder &builder, PanelTransferTileAttr tile) const;
  bool supports(PanelTransferTileAttr tile) const;
};

// Validation re-evaluates the selected point, never searches for a new choice.
mlir::LogicalResult verifyPanelTransferPlan(DensePanelCopyOp operation);
bool samePanelExtent(mlir::Value first, unsigned firstAxis,
                     mlir::Value second, unsigned secondAxis);

} // namespace weft::riscv
