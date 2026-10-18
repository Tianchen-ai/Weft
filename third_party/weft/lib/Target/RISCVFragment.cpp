#include "Weft/Target/RISCVFragment.h"

bool weft::supportsRISCVFragmentTile(const RISCVFragmentCapability &capability,
                                    int64_t m, int64_t n, int64_t k) {
  if (m <= 0 || n <= 0 || k != capability.kFactor ||
      !capability.activeMFactor || !capability.nFactor ||
      m % capability.activeMFactor || n % capability.nFactor)
    return false;
  const int64_t rows = m / capability.activeMFactor;
  const int64_t columns = n / capability.nFactor;
  return rows <= capability.maxRowAtoms && columns <= capability.maxColumnAtoms &&
         rows <= capability.maxProductAtoms / columns;
}
