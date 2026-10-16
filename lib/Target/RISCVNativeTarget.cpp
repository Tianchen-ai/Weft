#include "Weft/Target/RISCVTargetProfile.h"
#include "Weft/Target/RISCVExtensions.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

#if defined(__linux__) && defined(__riscv)
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

namespace {
bool discoverMatrixExtensions(weft::RISCVNativeTarget &result,
                              std::string &error) {
  std::ifstream input("/proc/cpuinfo");
  if (!input) {
    error = "cannot read per-CPU native matrix capability facts from /proc/cpuinfo";
    return false;
  }
  result.matrixExtensions.clear();
  std::set<int> seen;
  std::optional<int> processor;
  std::optional<uint64_t> vendor, architecture;
  std::string isa;
  bool commonFenceI = true;
  auto finish = [&]() {
    if (!processor ||
        std::find(result.cpus.begin(), result.cpus.end(), *processor) == result.cpus.end())
      return true;
    if (isa.empty() || !seen.insert(*processor).second) {
      error = "native CPU " + std::to_string(*processor) +
              " has missing or duplicate ISA facts in /proc/cpuinfo";
      return false;
    }
    llvm::SmallVector<llvm::StringRef> features;
    llvm::StringRef(isa).split(features, '_');
    commonFenceI &= llvm::is_contained(features, "zifencei");
    for (const auto &extension : weft::riscvExtensions()) {
      for (const auto &match : extension.nativeMatches) {
        if (std::find(features.begin(), features.end(), match.isaFeature) == features.end())
          continue;
        if (!vendor || !architecture) {
          error = "native matrix ISA feature has no vendor/architecture identity for CPU " +
                  std::to_string(*processor);
          return false;
        }
        if (weft::matchesRISCVNativeExtension(match, result.profile.xlen,
                result.profile.vlenBits, *vendor, *architecture, features)) {
          result.matrixExtensions[extension.name].push_back(*processor);
          break;
        }
      }
    }
    return true;
  };
  std::string text;
  while (std::getline(input, text)) {
    llvm::StringRef line(text);
    if (line.trim().empty()) {
      if (!finish())
        return false;
      processor.reset();
      vendor.reset();
      architecture.reset();
      isa.clear();
      continue;
    }
    auto [name, value] = line.split(':');
    name = name.trim();
    value = value.trim();
    if (name == "processor") {
      int cpu;
      if (processor || value.getAsInteger(10, cpu) || cpu < 0) {
        error = "malformed native processor identity in /proc/cpuinfo";
        return false;
      }
      processor = cpu;
    } else if (name == "isa") {
      if (!isa.empty() || value.empty()) {
        error = "missing or duplicate native ISA field in /proc/cpuinfo";
        return false;
      }
      isa = value.str();
    } else if (name == "mvendorid" || name == "marchid") {
      auto &identity = name == "mvendorid" ? vendor : architecture;
      uint64_t id;
      if (identity || value.getAsInteger(0, id)) {
        error = "malformed or duplicate native vendor/architecture identity in /proc/cpuinfo";
        return false;
      }
      identity = id;
    }
  }
  if (input.bad()) {
    error = "failed reading native matrix capability facts from /proc/cpuinfo";
    return false;
  }
  if (!finish())
    return false;
  if (seen.size() != result.cpus.size()) {
    error = "native CPU affinity contains processors with no ISA facts in /proc/cpuinfo";
    return false;
  }
  // Linux hwprobe/LLVM's feature vocabulary does not enumerate Zifencei.
  // Require the OS ISA declaration on every CPU in the discovered affinity.
  if (commonFenceI)
    result.isaExtensions.emplace("zifencei", std::make_pair(2U, 0U));
  return true;
}
} // namespace
#endif

bool weft::queryNativeRISCVTarget(RISCVNativeTarget &result,
                                  std::string &error) {
#if defined(__linux__) && defined(__riscv)
  constexpr int getVectorControl = 70; // Linux PR_RISCV_V_GET_CONTROL.
  constexpr unsigned long vectorCapability = 1UL << ('V' - 'A');
  const int control = prctl(getVectorControl, 0UL, 0UL, 0UL, 0UL);
  if (control < 0 || (control & 3) == 1 ||
      !(getauxval(AT_HWCAP) & vectorCapability)) {
    error = "native target requires Linux-enabled vector state";
    return false;
  }
  auto features = llvm::sys::getHostCPUFeatures();
  if (!features.lookup("v")) {
    error = "native target discovery requires hwprobe-confirmed full V support";
    return false;
  }
  std::vector<std::string> enabled;
  for (const auto &feature : features)
    if (feature.second &&
        llvm::RISCVISAInfo::isSupportedExtensionFeature(feature.first()))
      enabled.push_back("+" + feature.first().str());
  auto isa = llvm::RISCVISAInfo::parseFeatures(__riscv_xlen, enabled);
  if (!isa) {
    error = llvm::toString(isa.takeError());
    return false;
  }
  std::string march = "rv" + std::to_string(__riscv_xlen);
  // Preserve the reported extension vocabulary. Implied subextensions are
  // already covered by their parent requirement, not additional build flags.
  for (const auto &extension : (*isa)->getExtensions())
    if (extension.first.size() == 1 && features.lookup(extension.first))
      march += extension.first;
  for (const auto &extension : (*isa)->getExtensions())
    if (extension.first.size() > 1 && features.lookup(extension.first))
      march += "_" + extension.first;

  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) != 0) {
    error = "cannot discover native CPU affinity: " + std::string(strerror(errno));
    return false;
  }
  result.cpus.clear();
  uintptr_t bytes = 0;
  bool success = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &original))
      continue;
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    if (sched_setaffinity(0, sizeof(selected), &selected) != 0) {
      error = "cannot bind CPU for VLEN discovery: " + std::string(strerror(errno));
      success = false;
      break;
    }
    uintptr_t currentBytes;
    asm volatile("csrr %0, 0xc22" : "=r"(currentBytes));
    if (!currentBytes || (bytes && bytes != currentBytes)) {
      error = "native CPU affinity does not have one positive VLEN; select a homogeneous CPU set";
      success = false;
      break;
    }
    bytes = currentBytes;
    result.cpus.push_back(cpu);
  }
  if (sched_setaffinity(0, sizeof(original), &original) != 0) {
    error = "cannot restore CPU affinity after VLEN discovery: " + std::string(strerror(errno));
    return false;
  }
  if (!success || result.cpus.empty())
    return false;
  std::string abi = __riscv_xlen == 64 ? "lp64" : "ilp32";
#if defined(__riscv_float_abi_double)
  abi += "d";
#elif defined(__riscv_float_abi_single)
  abi += "f";
#endif
  if (!parseRISCVTargetProfile(march, abi, bytes * 8, "none", 2,
      "independent-multilevel", "within-record", result.profile, error))
    return false;
  enabled.push_back("+zvl" + std::to_string(bytes * 8) + "b");
  isa = llvm::RISCVISAInfo::parseFeatures(__riscv_xlen, enabled);
  if (!isa) {
    error = llvm::toString(isa.takeError());
    return false;
  }
  result.isaExtensions.clear();
  for (const auto &[name, version] : (*isa)->getExtensions())
    result.isaExtensions.emplace(name, std::make_pair(version.Major, version.Minor));
  // Zicbop consists entirely of ORI-to-x0 HINT encodings: execution on base I
  // is legal even when the implementation ignores prefetch. This is an
  // execution contract, not evidence of a cache engine or its latency.
  result.isaExtensions.emplace("zicbop", std::make_pair(1U, 0U));
  return discoverMatrixExtensions(result, error) &&
      discoverRISCVDataCaches(result.cpus, result.dataCaches, result.profile.dataCache, error);
#else
  error = "native target discovery requires a Weft compiler running on RISC-V Linux";
  return false;
#endif
}
