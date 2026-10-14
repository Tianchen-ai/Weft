#include "Weft/Target/RISCVExtensions.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <set>
#include <tuple>

namespace weft {
const RISCVExtension &spacemitIME1Extension();

llvm::ArrayRef<RISCVExtension> riscvExtensions() {
  static const RISCVExtension extensions[] = {spacemitIME1Extension()};
  return extensions;
}

const RISCVExtension *findRISCVExtension(llvm::StringRef name) {
  for (const auto &extension : riscvExtensions())
    if (extension.name == name) return &extension;
  return nullptr;
}

const RISCVFragmentCapability *findRISCVFragmentCapability(llvm::StringRef name) {
  for (const auto &extension : riscvExtensions())
    for (const auto &fragment : extension.fragments)
      if (fragment.name == name) return &fragment;
  return nullptr;
}

bool matchesRISCVNativeExtension(const RISCVNativeExtensionMatch &match,
                                int64_t xlen, int64_t vlenBits,
                                uint64_t vendorId, uint64_t architectureId,
                                llvm::ArrayRef<llvm::StringRef> isaFeatures) {
  return match.xlen == xlen && match.vlenBits == vlenBits &&
         match.vendorId == vendorId && match.architectureId == architectureId &&
         llvm::is_contained(isaFeatures, match.isaFeature);
}

namespace {
bool validVLEN(int64_t value) {
  return value >= 128 && value <= 65536 && (value & (value - 1)) == 0;
}

bool validPacking(const RISCVFragmentPacking &packing, unsigned bits,
                  int64_t first, int64_t second) {
  if (packing.schema.empty() || packing.axisOrder.size() != 2 ||
      !((packing.axisOrder[0] == 0 && packing.axisOrder[1] == 1) ||
        (packing.axisOrder[0] == 1 && packing.axisOrder[1] == 0)) ||
      packing.tileOrder != "row_major" || packing.elementOrder != "row_major" ||
      packing.storageBits != bits || packing.alignment <= 0 ||
      (packing.alignment & (packing.alignment - 1))) return false;
  const int64_t shape[] = {first, second};
  return packing.rowsPerTile == shape[packing.axisOrder[0]] &&
         packing.columnsPerTile == shape[packing.axisOrder[1]];
}

std::string registryError() {
  llvm::StringSet<> extensionNames, instructionNames;
  for (const auto &extension : riscvExtensions()) {
    if (extension.name.empty() || extension.name == "none" ||
        !extensionNames.insert(extension.name).second)
      return "extension registration requires a distinct nonempty name";
    if (extension.engine != "matrix" || extension.description.empty() ||
        extension.fragments.empty() || extension.nativeMatches.empty())
      return extension.name + ": incomplete local extension registration";
    std::set<std::tuple<int64_t, int64_t, uint64_t, uint64_t, std::string>> matches;
    for (const auto &match : extension.nativeMatches) {
      if ((match.xlen != 32 && match.xlen != 64) || !validVLEN(match.vlenBits) ||
          match.isaFeature.empty() || match.isaFeature.find('_') != std::string::npos ||
          !matches.emplace(match.xlen, match.vlenBits, match.vendorId,
                           match.architectureId, match.isaFeature).second)
        return extension.name + ": invalid or duplicate native identity match";
    }
    for (const auto &fragment : extension.fragments) {
      if (fragment.name.empty() || !instructionNames.insert(fragment.name).second ||
          fragment.extension != extension.name)
        return extension.name + ": local instruction identity disagrees with its extension";
      if ((fragment.xlen != 32 && fragment.xlen != 64) || !validVLEN(fragment.vlenBits) ||
          !fragment.activeMFactor || fragment.activeMFactor > fragment.mFactor ||
          !fragment.nFactor || !fragment.kFactor ||
          !fragment.lhsResourceGroups || !fragment.rhsResourceGroups ||
          !fragment.accumulatorResourceGroups || !fragment.lhsElementBits ||
          !fragment.rhsElementBits || !fragment.accumulatorElementBits ||
          fragment.mmaGroups <= 0 || fragment.mmaChunks <= 0 ||
          fragment.lhsPackLocalBytes < 0 || fragment.rhsPackLocalBytes < 0 ||
          fragment.mmaLocalBytes < 0 || fragment.packTemporaryGroups < 0 ||
          fragment.unpackTemporaryGroups < 0)
        return fragment.name + ": incomplete shape, numerical or resource contract";
      if (!validPacking(fragment.lhsPacking, fragment.lhsElementBits,
                        fragment.activeMFactor, fragment.kFactor) ||
          !validPacking(fragment.rhsPacking, fragment.rhsElementBits,
                        fragment.kFactor, fragment.nFactor) ||
          !validPacking(fragment.accumulatorPacking, fragment.accumulatorElementBits,
                        fragment.activeMFactor, fragment.nFactor))
        return fragment.name + ": packing does not cover its declared logical atom";
      llvm::StringSet<> clobbers;
      for (const auto &clobber : fragment.clobbers)
        if (clobber.empty() || !clobbers.insert(clobber).second)
          return fragment.name + ": empty or duplicate assembly clobber";
      if (!llvm::any_of(extension.nativeMatches, [&](const auto &match) {
            return match.xlen == fragment.xlen && match.vlenBits == fragment.vlenBits;
          }))
        return fragment.name + ": no registered native target can execute this atom";
    }
    for (const auto &match : extension.nativeMatches)
      if (!llvm::any_of(extension.fragments, [&](const auto &fragment) {
            return match.xlen == fragment.xlen && match.vlenBits == fragment.vlenBits;
          }))
        return extension.name + ": native discovery advertises a target with no local implementation";
  }
  return {};
}
} // namespace

bool validateRISCVExtensionRegistry(std::string &error) {
  static const std::string diagnostic = registryError();
  if (diagnostic.empty()) return true;
  error = "invalid RISC-V extension registry: " + diagnostic;
  return false;
}
} // namespace weft
