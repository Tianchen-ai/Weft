#pragma once

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/PanelTransfer.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {
struct PanelTransferWork {
  int64_t loads = 0;
  int64_t stores = 0;
  int64_t registerTransfers = 0;
  int64_t rowIterations = 0;
  int64_t vectorIterations = 0;
  int64_t sourceAddresses = 0;

  // Width work and instruction count are distinct. Control/address counts
  // price the same explicitly emitted physical loops, never source traversal.
  std::optional<int64_t> total() const;
};

mlir::FailureOr<riscv::PanelTransferCaseAttr> selectPanelTransferCase(
    mlir::Builder &builder, const riscv::PanelTransferGeometry &geometry,
    int64_t copyAxis, int64_t maximumLMUL, int64_t maximumFields, bool segment,
    llvm::StringRef strategy);

// Full finite window counts include both field and lane remainders. Unknown
// extents cannot be substituted with a fabricated zero-transfer estimate.
std::optional<PanelTransferWork> panelTransferWork(
    riscv::PanelTransferCaseAttr selected, int64_t outer, int64_t inner);

// Both descriptors name the same rank-two logical window; only their dense
// storage strides differ. Selection and traversal are separate physical passes.
mlir::LogicalResult transferDensePanel(mlir::IRRewriter &builder,
    mlir::Value source, mlir::Value destination, int64_t copyAxis,
    int64_t maximumLMUL, int64_t maximumFields, mlir::Operation *anchor);
}
