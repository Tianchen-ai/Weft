#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVExtensions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

using namespace weft;

bool RISCVCompilerOptions::validateBindings(std::string &error) const {
  auto finite = [&](llvm::StringRef name, int64_t value,
                    llvm::ArrayRef<int64_t> domain) {
    if (llvm::is_contained(domain, value)) return true;
    error = name.str() + "=" + std::to_string(value) + " is outside its finite domain (";
    for (auto [index, item] : llvm::enumerate(domain)) {
      if (index) error += ", ";
      error += std::to_string(item);
    }
    error += ")";
    return false;
  };
  if (!finite("lmul-eighths", lmulEighths, {1, 2, 4, 8, 16, 32, 64}) ||
      !finite("pipeline-depth", pipelineDepth, {1, 2}) ||
      !finite("scalar-load-prime", scalarLoadPrime, {0, 1}) ||
      !finite("pointwise-issue-copies", pointwiseIssueCopies, {1, 2, 4}) ||
      !finite("contraction-column-copies", contractionColumnCopies, {1, 2, 4}) ||
      !finite("panel-transfer-fields", panelTransferFields, {1, 2, 4, 8}) ||
      !finite("contraction-panel-k", contractionPanelK, {32, 64, 128, 256}) ||
      !finite("contraction-k-steps", contractionKSteps, {1, 2, 4})) return false;
  if (unroll <= 0) {
    error = "unroll must be a positive integer";
    return false;
  }
  if (panelTransferStrategy != "issue-work" && panelTransferStrategy != "max-fields" &&
      panelTransferStrategy != "single-field") {
    error = "panel-transfer-strategy must be issue-work, max-fields or single-field";
    return false;
  }
  if (localTileStrategy != "target-default" && localTileStrategy != "atom-partitioned" &&
      localTileStrategy != "register-grouped") {
    error = "local-tile-strategy must be target-default, atom-partitioned or register-grouped";
    return false;
  }
  for (const auto &binding : metaBindings)
    if (binding.getKey().empty() || binding.second <= 0) {
      error = "source meta bindings must have nonempty names and positive values";
      return false;
    }
  for (const auto &extension : requiredExtensions) {
    if (extension.empty()) {
      error = "required extensions must have nonempty names";
      return false;
    }
    if (!findRISCVExtension(extension)) {
      error = "unregistered required extension: " + extension;
      return false;
    }
  }
  return true;
}
