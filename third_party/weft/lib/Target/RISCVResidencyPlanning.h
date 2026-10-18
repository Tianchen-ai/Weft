#ifndef WEFT_LIB_TARGET_RISCVRESIDENCYPLANNING_H
#define WEFT_LIB_TARGET_RISCVRESIDENCYPLANNING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft {

mlir::LogicalResult planRISCVResidency(mlir::ModuleOp module);

} // namespace weft

#endif
