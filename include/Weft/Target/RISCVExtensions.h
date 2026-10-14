#ifndef WEFT_TARGET_RISCVEXTENSIONS_H
#define WEFT_TARGET_RISCVEXTENSIONS_H

#include "Weft/Target/RISCVFragment.h"

namespace weft {

/// A Linux-reported identity for an extension, independent of its local atoms.
/// Each entry is an exact conjunction; multiple entries are explicit alternatives.
struct RISCVNativeExtensionMatch {
  int64_t xlen;
  int64_t vlenBits;
  uint64_t vendorId;
  uint64_t architectureId;
  std::string isaFeature;
};

struct RISCVExtension {
  std::string name;
  std::string engine;
  std::string description;
  std::vector<RISCVNativeExtensionMatch> nativeMatches;
  llvm::ArrayRef<RISCVFragmentCapability> fragments;
};

/// Immutable built-in local extensions. Standard ISA extensions use LLVM's
/// normalized ISA parser; this registry never injects vendor tokens into march.
llvm::ArrayRef<RISCVExtension> riscvExtensions();
const RISCVExtension *findRISCVExtension(llvm::StringRef name);

/// Validate the complete registered contract before accepting a target.
/// Invalid registrations are compiler errors, not silently omitted capabilities.
bool validateRISCVExtensionRegistry(std::string &error);

bool matchesRISCVNativeExtension(const RISCVNativeExtensionMatch &match,
                                int64_t xlen, int64_t vlenBits,
                                uint64_t vendorId, uint64_t architectureId,
                                llvm::ArrayRef<llvm::StringRef> isaFeatures);

} // namespace weft

#endif
