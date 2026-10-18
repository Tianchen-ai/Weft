#ifndef WEFT_TARGET_RISCVDATACACHE_H
#define WEFT_TARGET_RISCVDATACACHE_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace weft {

// Cache geometry guides physical work estimates. It is neither an ISA
// execution requirement nor a capacity guarantee for private storage.
struct RISCVDataCache {
  int64_t sizeBytes;
  int64_t lineBytes;
  int64_t ways;
  int64_t sets;

  bool operator==(const RISCVDataCache &other) const {
    return sizeBytes == other.sizeBytes && lineBytes == other.lineBytes &&
           ways == other.ways && sets == other.sets;
  }
  bool operator!=(const RISCVDataCache &other) const { return !(*this == other); }
  bool validate(std::string &error) const;
};

struct RISCVCPUDataCache {
  int cpu;
  std::optional<RISCVDataCache> cache;
};

// Missing kernel cacheinfo is represented explicitly by an empty observation.
// Malformed or inconsistent published geometry remains an error.
bool discoverRISCVDataCaches(const std::vector<int> &cpus,
                            std::vector<RISCVCPUDataCache> &observations,
                            std::optional<RISCVDataCache> &common,
                            std::string &error);

} // namespace weft
#endif
