#ifndef WEFT_TARGET_RISCVFRAGMENT_H
#define WEFT_TARGET_RISCVFRAGMENT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace weft {

enum class RISCVFragmentInstruction { SpacemitIME1I8MMA };
enum class RISCVFragmentSignedness { Signed, Unsigned };

struct RISCVFragmentPacking {
  std::string schema;
  std::vector<int64_t> axisOrder;
  int64_t rowsPerTile;
  int64_t columnsPerTile;
  std::string tileOrder;
  std::string elementOrder;
  int64_t storageBits;
  int64_t alignment;
};

struct RISCVFragmentCapability {
  RISCVFragmentInstruction instruction;
  std::string name;
  std::string extension;
  int64_t xlen;
  int64_t vlenBits;
  RISCVFragmentSignedness lhsSignedness;
  RISCVFragmentSignedness rhsSignedness;
  unsigned lhsElementBits;
  unsigned rhsElementBits;
  unsigned accumulatorElementBits;
  unsigned mFactor;
  unsigned nFactor;
  unsigned kFactor;
  unsigned lhsResourceGroups;
  unsigned rhsResourceGroups;
  unsigned accumulatorResourceGroups;
  RISCVFragmentPacking lhsPacking;
  RISCVFragmentPacking rhsPacking;
  RISCVFragmentPacking accumulatorPacking;
  int64_t mmaGroups;
  int64_t mmaChunks;
  std::vector<std::string> clobbers;
  bool volatileAsm;
  bool memoryClobber;
  int64_t lhsPackLocalBytes;
  int64_t rhsPackLocalBytes;
  int64_t mmaLocalBytes;
  int64_t packTemporaryGroups;
  int64_t unpackTemporaryGroups;
  RISCVFragmentSignedness accumulatorSignedness;
  unsigned activeMFactor;
};

const RISCVFragmentCapability *findRISCVFragmentCapability(llvm::StringRef name);
bool supportsRISCVFragmentTile(const RISCVFragmentCapability &capability,
                              int64_t m, int64_t n, int64_t k);

} // namespace weft

#endif
