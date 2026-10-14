#include "Weft/Target/RISCVFragment.h"
#include "Weft/Target/RISCVExtensions.h"

namespace {
llvm::ArrayRef<weft::RISCVFragmentCapability> fragmentContracts() {
  using namespace weft;
  static const RISCVFragmentCapability capabilities[] = {{
      RISCVFragmentInstruction::SpacemitIME1I8MMA,
      "spacemit-ime1-i8-mma", "spacemit-ime1", 64, 256,
      RISCVFragmentSignedness::Signed, RISCVFragmentSignedness::Signed,
      8, 8, 32, 4, 4, 8, 1, 1, 2,
      {"ime1-i8-m4-k8", {0, 1}, 4, 8, "row_major", "row_major", 8, 32},
      {"ime1-i8-n4-k8-transposed", {1, 0}, 4, 8,
       "row_major", "row_major", 8, 32},
      {"ime1-i32-m4-n4", {0, 1}, 4, 4, "row_major", "row_major", 32, 32},
      1, 1, {"t0"}, true, false,
      0, 0, 0, 1, 2, RISCVFragmentSignedness::Signed, 4}, {
      RISCVFragmentInstruction::SpacemitIME1I8MMA,
      "spacemit-ime1-i8-m1-mma", "spacemit-ime1", 64, 256,
      RISCVFragmentSignedness::Signed, RISCVFragmentSignedness::Signed,
      8, 8, 32, 4, 4, 8, 1, 1, 2,
      {"ime1-i8-m1-k8", {0, 1}, 1, 8, "row_major", "row_major", 8, 8},
      {"ime1-i8-n4-k8-transposed", {1, 0}, 4, 8,
       "row_major", "row_major", 8, 32},
      {"ime1-i32-m1-n4", {0, 1}, 1, 4, "row_major", "row_major", 32, 16},
      1, 1, {"t0"}, true, false,
      0, 0, 0, 1, 2, RISCVFragmentSignedness::Signed, 1}};
  return capabilities;
}
} // namespace

namespace weft {
const RISCVExtension &spacemitIME1Extension() {
  static const RISCVExtension extension{
      "spacemit-ime1", "matrix",
      "Signed i8 matrix atoms with i32 accumulation in shared RVV registers",
      {{64, 256, 0x710, 0x8000000058000001, "ime"}}, fragmentContracts()};
  return extension;
}
} // namespace weft

bool weft::supportsRISCVFragmentTile(const RISCVFragmentCapability &capability,
                                    int64_t m, int64_t n, int64_t k) {
  if (m <= 0 || n <= 0 || k != capability.kFactor ||
      m % capability.activeMFactor || n % capability.nFactor ||
      (capability.activeMFactor < capability.mFactor && m != capability.activeMFactor))
    return false;
  const int64_t rows = m / capability.activeMFactor;
  const int64_t columns = n / capability.nFactor;
  return rows <= 4 && columns <= 4 && rows * columns <= 4;
}
