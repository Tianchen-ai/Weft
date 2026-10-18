#pragma once

#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv_internal {
std::vector<RISCVOperationDiagnostic> collectPhysicalOperations(riscv::KernelOp kernel);
}
