#include "Weft/Target/RISCVDataCache.h"

#include <limits>

bool weft::RISCVDataCache::validate(std::string &error) const {
  if (sizeBytes <= 0 || lineBytes <= 0 || ways <= 0 || sets <= 0) {
    error = "data cache geometry requires positive size, line, ways and sets";
    return false;
  }
  if ((lineBytes & (lineBytes - 1)) || sizeBytes % lineBytes ||
      sizeBytes / lineBytes % ways || sizeBytes / lineBytes / ways != sets) {
    error = "data cache size must equal line bytes times ways times sets";
    return false;
  }
  if (sizeBytes > std::numeric_limits<int32_t>::max()) {
    error = "L1 data cache geometry exceeds the supported address-work domain";
    return false;
  }
  return true;
}
