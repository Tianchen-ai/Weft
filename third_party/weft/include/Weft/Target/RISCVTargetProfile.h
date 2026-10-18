#ifndef WEFT_TARGET_RISCVTARGETPROFILE_H
#define WEFT_TARGET_RISCVTARGETPROFILE_H

#include "Weft/Target/RISCVFragment.h"
#include "Weft/Target/RISCVDataCache.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace weft {

enum class RISCVABI {
  LP64,
  LP64F,
  LP64D,
  ILP32,
  ILP32F,
  ILP32D,
};

enum class RISCVPartialCombinePolicy {
  IndependentMultilevel,
  Sequential,
};

enum class RISCVRecordAxisPolicy {
  WithinRecord,
  AcrossRecords,
};

/// Standard ISA facts after LLVM has resolved versions and implied extensions.
/// Vendor extension admission is separate from the standard march grammar.
struct RISCVISAFacts {
  unsigned xlen = 0;
  unsigned minVlenBits = 0;
  std::map<std::string, std::pair<unsigned, unsigned>> extensions;

  bool hasExtension(llvm::StringRef name) const;
};

bool parseRISCVISAFacts(llvm::StringRef march, int64_t vlenBits,
                       RISCVISAFacts &facts, std::string &error);

struct RISCVTargetProfile {
  std::string triple;
  std::string march;
  std::string abi;
  RISCVABI abiKind = RISCVABI::LP64;
  int64_t xlen = 0;
  bool littleEndian = true;
  bool hasRVV = false;
  bool hasF = false;
  bool hasD = false;
  bool hasScalarF16 = false;
  bool hasVectorF16 = false;
  bool hasVectorF16Conversion = false;
  bool hasIndexedMemory = false;
  bool hasSegmentMemory = false;
  bool hasWideningInteger = false;
  bool hasWideningFloat = false;
  int64_t vlenBits = 0;
  int64_t vectorRegisters = 0;
  int64_t maxPrivateStackBytes = 65536;
  std::optional<RISCVDataCache> dataCache;
  int64_t maxWideningCombineGroups = 2;
  std::vector<unsigned> supportedSEW;
  std::vector<int> legalLMULEighths;
  std::vector<RISCVFragmentCapability> fragmentCapabilities;
  RISCVPartialCombinePolicy partialCombinePolicy =
      RISCVPartialCombinePolicy::IndependentMultilevel;
  RISCVRecordAxisPolicy recordAxisPolicy =
      RISCVRecordAxisPolicy::WithinRecord;

  bool supportsSEW(unsigned sew) const;
  bool supportsLMULEighths(int lmulEighths) const;
  bool supportsFixedRVV() const;
  bool supportsVectorShape(unsigned sew, int lmulEighths) const;
  bool supportsIndexedVectorMemory(unsigned elementSEW,
                                   int elementLMULEighths,
                                   unsigned indexSEW,
                                   int indexLMULEighths) const;
  bool supportsSegmentVectorMemory(unsigned fields, unsigned sew,
                                   int lmulEighths) const;
};

bool parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                             int64_t vlenBits,
                             llvm::StringRef matrixExtension,
                             int64_t maxWideningCombineGroups,
                             llvm::StringRef partialCombinePolicy,
                             llvm::StringRef recordAxisPolicy,
                             RISCVTargetProfile &profile, std::string &error);

struct RISCVNativeTarget {
  RISCVTargetProfile profile;
  std::map<std::string, std::pair<unsigned, unsigned>> isaExtensions;
  std::vector<int> cpus;
  std::map<std::string, std::vector<int>> matrixExtensions;
  std::vector<RISCVCPUDataCache> dataCaches;
};

bool queryNativeRISCVTarget(RISCVNativeTarget &target, std::string &error);

bool parseRISCVISAExtensions(
    llvm::StringRef march, int64_t vlenBits,
    std::map<std::string, std::pair<unsigned, unsigned>> &extensions,
    std::string &error);

} // namespace weft

#endif // WEFT_TARGET_RISCVTARGETPROFILE_H
