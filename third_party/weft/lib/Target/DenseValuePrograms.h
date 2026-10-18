#ifndef WEFT_TARGET_DENSE_VALUE_PROGRAMS_H
#define WEFT_TARGET_DENSE_VALUE_PROGRAMS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::riscv_internal {
// Strip-mine a closed numerical producer graph at its existing store/reduce
// boundary. The resulting loops and storage are ordinary Physical IR.
mlir::LogicalResult materializeDenseValuePrograms(mlir::ModuleOp module);
mlir::LogicalResult materializeDenseValueProgram(mlir::Operation *root);
}

#endif
