#ifndef WEFT_TARGET_PREDICATE_REGISTERS_H
#define WEFT_TARGET_PREDICATE_REGISTERS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::riscv_internal {
mlir::LogicalResult selectPredicateRegisters(mlir::ModuleOp module);
}
#endif
