#ifndef WEFT_TARGET_FIELDSTORES_H
#define WEFT_TARGET_FIELDSTORES_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {
mlir::LogicalResult materializeFieldStoreView(mlir::IRRewriter &rewriter,
                                             riscv::StoreOp operation);
}

#endif
