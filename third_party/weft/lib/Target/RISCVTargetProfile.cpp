#include "Weft/Target/RISCVTargetProfile.h"
#include "Weft/Target/RISCVExtensions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

bool weft::RISCVTargetProfile::supportsSEW(unsigned sew) const {
  return llvm::is_contained(supportedSEW, sew);
}

bool weft::RISCVTargetProfile::supportsLMULEighths(int lmulEighths) const {
  return llvm::is_contained(legalLMULEighths, lmulEighths);
}

bool weft::RISCVTargetProfile::supportsFixedRVV() const {
  return hasRVV && vlenBits > 0 && vectorRegisters > 0;
}

bool weft::RISCVTargetProfile::supportsVectorShape(unsigned sew,
                                                   int lmulEighths) const {
  if (!supportsFixedRVV() || !supportsSEW(sew) ||
      !supportsLMULEighths(lmulEighths))
    return false;
  if (lmulEighths <= 0 ||
      (lmulEighths >= 8 && lmulEighths % 8 != 0))
    return false;
  uint64_t registerGroups =
      (static_cast<uint64_t>(lmulEighths) + 7) / 8;
  return registerGroups <= static_cast<uint64_t>(vectorRegisters);
}

bool weft::RISCVTargetProfile::supportsIndexedVectorMemory(
    unsigned elementSEW, int elementLMULEighths, unsigned indexSEW,
    int indexLMULEighths) const {
  if (!hasIndexedMemory ||
      (indexSEW != 16 && indexSEW != 32 && indexSEW != 64) ||
      !supportsVectorShape(elementSEW, elementLMULEighths) ||
      !supportsVectorShape(indexSEW, indexLMULEighths))
    return false;
  uint64_t elementLanes =
      static_cast<uint64_t>(vlenBits) * elementLMULEighths /
      (8 * static_cast<uint64_t>(elementSEW));
  uint64_t indexLanes =
      static_cast<uint64_t>(vlenBits) * indexLMULEighths /
      (8 * static_cast<uint64_t>(indexSEW));
  return elementLanes != 0 && elementLanes == indexLanes;
}

bool weft::RISCVTargetProfile::supportsSegmentVectorMemory(
    unsigned fields, unsigned sew, int lmulEighths) const {
  if (!hasSegmentMemory || fields < 2 || fields > 8 ||
      !supportsVectorShape(sew, lmulEighths))
    return false;
  if (static_cast<uint64_t>(fields) *
          static_cast<uint64_t>(lmulEighths) >
      64)
    return false;
  uint64_t registerGroups =
      (static_cast<uint64_t>(lmulEighths) + 7) / 8;
  return fields * registerGroups <=
         static_cast<uint64_t>(vectorRegisters);
}

bool weft::parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                                   int64_t vlenBits,
                                   llvm::StringRef matrixExtension,
                                   int64_t maxWideningCombineGroups,
                                   llvm::StringRef partialCombinePolicy,
                                   llvm::StringRef recordAxisPolicy,
                                   RISCVTargetProfile &profile,
                                   std::string &error) {
  if (march.empty()) {
    error = "--march is required for RISC-V target lowering";
    return false;
  }
  if (abi.empty()) {
    error = "--abi is required for RISC-V target lowering";
    return false;
  }
  profile = RISCVTargetProfile{};
  if (!validateRISCVExtensionRegistry(error)) return false;
  RISCVISAFacts isa;
  if (!parseRISCVISAFacts(march, vlenBits, isa, error))
    return false;
  if (isa.hasExtension("e")) {
    error = "the current Weft ABI set does not support the embedded E base ISA";
    return false;
  }
  profile.xlen = isa.xlen;
  profile.triple = "riscv" + std::to_string(isa.xlen) + "-unknown-linux-gnu";
  if (recordAxisPolicy == "within-record") {
    profile.recordAxisPolicy = RISCVRecordAxisPolicy::WithinRecord;
  } else if (recordAxisPolicy == "across-records") {
    profile.recordAxisPolicy = RISCVRecordAxisPolicy::AcrossRecords;
  } else {
    error =
        "--record-axis-policy must be within-record or across-records";
    return false;
  }
  std::optional<RISCVABI> parsedABI =
      llvm::StringSwitch<std::optional<RISCVABI>>(abi)
          .Case("lp64", RISCVABI::LP64)
          .Case("lp64f", RISCVABI::LP64F)
          .Case("lp64d", RISCVABI::LP64D)
          .Case("ilp32", RISCVABI::ILP32)
          .Case("ilp32f", RISCVABI::ILP32F)
          .Case("ilp32d", RISCVABI::ILP32D)
          .Default(std::nullopt);
  if (!parsedABI) {
    error = "--abi must be lp64/lp64f/lp64d or ilp32/ilp32f/ilp32d";
    return false;
  }
  bool abiIs64 = *parsedABI == RISCVABI::LP64 ||
                 *parsedABI == RISCVABI::LP64F ||
                 *parsedABI == RISCVABI::LP64D;
  if (abiIs64 != (profile.xlen == 64)) {
    error = profile.xlen == 64 ? "RV64 requires an lp64 ABI"
                               : "RV32 requires an ilp32 ABI";
    return false;
  }
  if (vlenBits < 0 || (vlenBits != 0 && vlenBits % 8 != 0)) {
    error = "--vlen-bits must be zero or a positive multiple of eight";
    return false;
  }
  if (maxWideningCombineGroups <= 0) {
    error = "--max-widening-combine-groups must be positive";
    return false;
  }
  profile.march = march.str();
  profile.abi = abi.str();
  profile.abiKind = *parsedABI;
  profile.hasF = isa.hasExtension("f");
  profile.hasD = isa.hasExtension("d");
  profile.hasScalarF16 = isa.hasExtension("zfh");
  profile.hasRVV = isa.hasExtension("v");
  bool hasEmbeddedVector = llvm::any_of(
      isa.extensions, [](const auto &extension) {
        return llvm::StringRef(extension.first).starts_with("zve");
      });
  if (hasEmbeddedVector && !profile.hasRVV) {
    error = "the current RISC-V backend requires the full V extension, not a Zve subset";
    return false;
  }
  // Zvfhmin admits widening/narrowing conversions, not half arithmetic.
  // The existing physical flag describes the latter and must require Zvfh.
  profile.hasVectorF16 = isa.hasExtension("zvfh");
  profile.hasVectorF16Conversion =
      isa.hasExtension("zvfhmin") || profile.hasVectorF16;
  bool abiNeedsF = *parsedABI == RISCVABI::LP64F ||
                   *parsedABI == RISCVABI::LP64D ||
                   *parsedABI == RISCVABI::ILP32F ||
                   *parsedABI == RISCVABI::ILP32D;
  bool abiNeedsD = *parsedABI == RISCVABI::LP64D ||
                   *parsedABI == RISCVABI::ILP32D;
  if ((abiNeedsF && !profile.hasF) || (abiNeedsD && !profile.hasD)) {
    error = "--abi floating-point register convention is not present in --march";
    return false;
  }
  if (!profile.hasRVV) {
    error = "the current Weft RISC-V physical compiler requires the full V extension";
    return false;
  }
  if (vlenBits <= 0) {
    error = "the current Weft RISC-V physical compiler requires an explicit positive --vlen-bits";
    return false;
  }
  if (vlenBits < 128 || vlenBits > 65536 || (vlenBits & (vlenBits - 1))) {
    error = "full V requires a power-of-two --vlen-bits from 128 to 65536";
    return false;
  }
  profile.vlenBits = vlenBits;
  profile.vectorRegisters = 32;
  if (maxWideningCombineGroups > profile.vectorRegisters) {
    error = "--max-widening-combine-groups exceeds the architectural vector register count";
    return false;
  }
  profile.maxWideningCombineGroups = maxWideningCombineGroups;
  profile.supportedSEW = {8, 16, 32, 64};
  profile.legalLMULEighths = {1, 2, 4, 8, 16, 32, 64};
  profile.hasIndexedMemory = true;
  profile.hasSegmentMemory = true;
  profile.hasWideningInteger = true;
  profile.hasWideningFloat = profile.hasF;
  if (partialCombinePolicy == "independent-multilevel") {
    profile.partialCombinePolicy =
        RISCVPartialCombinePolicy::IndependentMultilevel;
  } else if (partialCombinePolicy == "sequential") {
    profile.partialCombinePolicy = RISCVPartialCombinePolicy::Sequential;
  } else {
    error =
        "--partial-combine-policy must be independent-multilevel or sequential";
    return false;
  }
  if (matrixExtension != "none") {
    const auto *extension = findRISCVExtension(matrixExtension);
    if (!extension) {
      error = "unregistered matrix extension: " + matrixExtension.str();
      return false;
    }
    for (const auto &fragment : extension->fragments) {
      if (!profile.littleEndian || profile.xlen != fragment.xlen ||
          !profile.supportsFixedRVV() || profile.vlenBits != fragment.vlenBits ||
          !profile.supportsVectorShape(fragment.lhsElementBits, 8) ||
          !profile.supportsVectorShape(fragment.rhsElementBits, 8) ||
          !profile.supportsVectorShape(fragment.accumulatorElementBits, 8))
        continue;
      profile.fragmentCapabilities.push_back(fragment);
    }
    if (profile.fragmentCapabilities.empty()) {
      error = matrixExtension.str() + " has no local instruction implementation for this XLEN, VLEN and element-width profile";
      return false;
    }
  }
  return true;
}
