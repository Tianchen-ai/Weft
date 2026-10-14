#include "Weft/Target/RISCVDataCache.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
enum class Observation { Present, Absent, Error };

Observation read(const std::filesystem::path &path, std::string &text,
                 std::string &error) {
  std::error_code code;
  bool exists = std::filesystem::exists(path, code);
  if (code) {
    error = "cannot inspect native cache attribute " + path.string() + ": " + code.message();
    return Observation::Error;
  }
  if (!exists) return Observation::Absent;
  std::ifstream stream(path);
  if (!stream || !std::getline(stream, text) || stream.bad()) {
    error = "cannot read native cache attribute " + path.string();
    return Observation::Error;
  }
  text = llvm::StringRef(text).trim().str();
  if (text.empty()) {
    error = "empty native cache attribute " + path.string();
    return Observation::Error;
  }
  return Observation::Present;
}

bool required(const std::filesystem::path &path, std::string &text,
              std::string &error) {
  auto status = read(path, text, error);
  if (status == Observation::Absent)
    error = "incomplete native cacheinfo: missing " + path.string();
  return status == Observation::Present;
}

bool positive(const std::filesystem::path &path, int64_t &number,
              std::string &error, bool size = false) {
  std::string text;
  if (!required(path, text, error)) return false;
  llvm::StringRef spelling(text);
  int64_t multiplier = 1;
  if (size && spelling.ends_with("K")) {
    multiplier = 1024;
    spelling = spelling.drop_back();
  }
  if (spelling.empty() || spelling.getAsInteger(10, number) || number <= 0 ||
      number > std::numeric_limits<int64_t>::max() / multiplier) {
    error = "invalid positive native cache geometry in " + path.string();
    return false;
  }
  number *= multiplier;
  return true;
}

bool cpuCache(int cpu, std::optional<weft::RISCVDataCache> &result,
              std::string &error) {
  auto root = std::filesystem::path("/sys/devices/system/cpu") /
      ("cpu" + std::to_string(cpu)) / "cache";
  std::error_code code;
  bool exists = std::filesystem::exists(root, code);
  if (code) {
    error = "cannot inspect CPU cacheinfo: " + code.message();
    return false;
  }
  result.reset();
  if (!exists) return true;
  std::filesystem::directory_iterator iterator(root, code), end;
  if (code) {
    error = "cannot enumerate CPU cacheinfo: " + code.message();
    return false;
  }
  for (; iterator != end; iterator.increment(code)) {
    if (code) break;
    const auto &directory = iterator->path();
    std::string name = directory.filename().string();
    llvm::StringRef suffix(name);
    if (!suffix.consume_front("index") || suffix.empty() ||
        !std::all_of(suffix.begin(), suffix.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
    int64_t level;
    std::string kind;
    if (!positive(directory / "level", level, error) ||
        !required(directory / "type", kind, error)) return false;
    if (level != 1 || (kind != "Data" && kind != "Unified")) continue;
    if (result) {
      error = "CPU " + std::to_string(cpu) + " exposes more than one L1 data cache";
      return false;
    }
    weft::RISCVDataCache cache;
    if (!positive(directory / "size", cache.sizeBytes, error, true) ||
        !positive(directory / "coherency_line_size", cache.lineBytes, error) ||
        !positive(directory / "ways_of_associativity", cache.ways, error) ||
        !positive(directory / "number_of_sets", cache.sets, error)) return false;
    if (!cache.validate(error)) {
      error = directory.string() + ": " + error;
      return false;
    }
    result = cache;
  }
  if (code) {
    error = "failed enumerating CPU cacheinfo: " + code.message();
    return false;
  }
  return true;
}
} // namespace

bool weft::discoverRISCVDataCaches(const std::vector<int> &cpus,
    std::vector<RISCVCPUDataCache> &observations,
    std::optional<RISCVDataCache> &common, std::string &error) {
  observations.clear();
  common.reset();
  if (cpus.empty()) {
    error = "cache discovery requires the actual CPU affinity";
    return false;
  }
  bool homogeneous = true;
  for (int cpu : cpus) {
    std::optional<RISCVDataCache> cache;
    if (!cpuCache(cpu, cache, error)) return false;
    if (observations.empty()) common = cache;
    else homogeneous &= common == cache;
    observations.push_back({cpu, cache});
  }
  if (!homogeneous) common.reset();
  return true;
}
