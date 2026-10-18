#pragma once

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv {

// A closed scalar packet preserves a selected load frontier. It reads the
// same coordinates as scalar_tile_load and has no additional memory effect.
// Its system-compiler boundary keeps all loaded FPRs resident before later
// vector loads; it does not emit a hardware fence or prefetch another issue.
mlir::LogicalResult verifyScalarPacket(ScalarTileLoadOp operation);

// Count already materialized floating scalar slots live at this load's exit,
// including the whole selected packet. Replica projections alias their exact
// root slots. A bounded analysis failure is not a zero-residency estimate.
std::optional<int64_t> scalarPacketResidency(ScalarTileLoadOp operation);

} // namespace weft::riscv
