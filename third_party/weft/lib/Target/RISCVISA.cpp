#include "Weft/Target/RISCVTargetProfile.h"

#include "llvm/Support/Error.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

bool weft::RISCVISAFacts::hasExtension(llvm::StringRef name) const {
  return extensions.find(name.str()) != extensions.end();
}

bool weft::parseRISCVISAFacts(llvm::StringRef march, int64_t vlenBits,
                             RISCVISAFacts &facts, std::string &error) {
  auto isa = llvm::RISCVISAInfo::parseArchString(march, false);
  if (!isa) {
    error = llvm::toString(isa.takeError());
    return false;
  }
  if (vlenBits < (*isa)->getMinVLen()) {
    error = "fixed VLEN is smaller than the minimum required by --march";
    return false;
  }
  RISCVISAFacts parsed;
  parsed.xlen = (*isa)->getXLen();
  parsed.minVlenBits = (*isa)->getMinVLen();
  for (const auto &[name, version] : (*isa)->getExtensions())
    parsed.extensions.emplace(name, std::make_pair(version.Major, version.Minor));
  facts = std::move(parsed);
  return true;
}

bool weft::parseRISCVISAExtensions(
    llvm::StringRef march, int64_t vlenBits,
    std::map<std::string, std::pair<unsigned, unsigned>> &extensions,
    std::string &error) {
  RISCVISAFacts facts;
  if (!parseRISCVISAFacts(march, vlenBits, facts, error))
    return false;
  extensions = std::move(facts.extensions);
  return true;
}
