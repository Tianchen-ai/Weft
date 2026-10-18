#ifndef WEFT_LIB_TARGET_RISCVINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVINTRINSICC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace weft {

struct RISCVKernelABI;

mlir::LogicalResult emitSelectedRISCVIntrinsicC(mlir::ModuleOp module,
                                                std::string &output,
                                                std::vector<RISCVKernelABI> &kernels);

} // namespace weft

#endif // WEFT_LIB_TARGET_RISCVINTRINSICC_H
