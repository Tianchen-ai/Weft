#ifndef WEFT_TARGET_IME_FRAGMENT_MATERIALIZATION_H
#define WEFT_TARGET_IME_FRAGMENT_MATERIALIZATION_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace weft::riscv_internal {
mlir::FailureOr<mlir::Value> materializeFragmentProduct(
    mlir::IRRewriter &rewriter, mlir::Operation *operation,
    riscv::ImplementationAttr implementation, riscv::ValueType resultType);
} // namespace weft::riscv_internal

#endif
