#ifndef WEFT_TARGET_RISCVTARGETCATALOG_H
#define WEFT_TARGET_RISCVTARGETCATALOG_H

#include "Weft/Target/RISCVTargetProfile.h"
#include "llvm/Support/JSON.h"

namespace weft {

/// Compiler-supported local extensions, without probing or executing hardware.
llvm::json::Object riscvExtensionCatalogJSON();

/// Facts for an explicitly parsed target. This describes target legality, not
/// native availability, system-compiler support or a kernel's chosen operations.
bool riscvTargetCapabilitiesJSON(const RISCVTargetProfile &profile,
                                 llvm::json::Object &result, std::string &error);

} // namespace weft

#endif
