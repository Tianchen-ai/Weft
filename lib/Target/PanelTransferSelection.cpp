#include "DensePanelTransfer.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
bool better(riscv::PanelTransferTileAttr a, riscv::PanelTransferTileAttr b) {
  if (!b) return true;
  int64_t lhs = a.getWork() * b.getFields() * b.getLanes();
  int64_t rhs = b.getWork() * a.getFields() * a.getLanes();
  if (lhs != rhs) return lhs < rhs;
  if (a.getRegisters() != b.getRegisters()) return a.getRegisters() < b.getRegisters();
  if (a.getFields() != b.getFields()) return a.getFields() < b.getFields();
  return a.getLmulEighths() < b.getLmulEighths();
}

bool addProduct(int64_t &result, int64_t first, int64_t second) {
  int64_t product;
  return !llvm::MulOverflow(first, second, product) && !llvm::AddOverflow(result, product, result);
}

bool addPhase(PanelTransferWork &work, riscv::PanelTransferTileAttr tile,
              int64_t rows, int64_t inner) {
  if (!rows || !inner) return true;
  // A short final vector is one issued instruction. The selected register
  // group is retained, so this is a conservative width-work estimate for VL
  // tails, while the load/store/loop counts are exact for these finite extents.
  int64_t columns = inner / tile.getLanes() + (inner % tile.getLanes() != 0);
  int64_t issues;
  if (llvm::MulOverflow(rows, columns, issues)) return false;
  return addProduct(work.loads, issues, tile.getLoads()) &&
         addProduct(work.stores, issues, tile.getStores()) &&
         addProduct(work.registerTransfers, issues, 2 * tile.getRegisters()) &&
         addProduct(work.rowIterations, rows, 1) &&
         addProduct(work.vectorIterations, issues, 1) &&
         addProduct(work.sourceAddresses, rows, tile.getFields());
}
}

FailureOr<riscv::PanelTransferCaseAttr> weft::riscv_internal::selectPanelTransferCase(
    Builder &b, const riscv::PanelTransferGeometry &geometry, int64_t axis,
    int64_t maximumLMUL, int64_t maximumFields, bool segment, StringRef strategy) {
  if (strategy != "issue-work" && strategy != "max-fields" && strategy != "single-field") return failure();
  riscv::PanelTransferTileAttr best, tail;
  // At most four tuple sizes times seven architectural LMULs. Each candidate
  // is a complete local memory representation of the same logical window.
  // This owner is shared by contraction costing and actual transfer selection.
  for (int64_t fields : {1, 2, 4, 8}) {
    if (strategy == "single-field" && fields != 1) continue;
    if (fields > maximumFields || (fields > 1 && !segment)) continue;
    for (int64_t lmul : geometry.target.getLegalLMULEighths().asArrayRef()) {
      if (lmul > maximumLMUL || fields * lmul > 64) continue;
      int64_t lanes = geometry.target.getVlenBits() * lmul / (8 * geometry.element.getIntOrFloatBitWidth());
      if (lanes <= 1) continue;
      int64_t groups = fields * ((lmul + 7) / 8);
      auto candidate = riscv::PanelTransferTileAttr::get(b.getContext(), axis,
          fields, lanes, lmul, groups, fields, 1, fields + 1 + 2 * groups);
      if (!geometry.supports(candidate)) continue;
      if (fields == 1 && better(candidate, tail)) tail = candidate;
      bool wider = best && strategy == "max-fields" && candidate.getFields() != best.getFields();
      if (wider ? candidate.getFields() > best.getFields() : better(candidate, best)) best = candidate;
    }
  }
  if (!best || !tail) return failure();
  return riscv::PanelTransferCaseAttr::get(b.getContext(), best, best.getFields() == 1 ? best : tail);
}

std::optional<PanelTransferWork> weft::riscv_internal::panelTransferWork(
    riscv::PanelTransferCaseAttr selected, int64_t outer, int64_t inner) {
  if (outer < 0 || inner < 0) return std::nullopt;
  PanelTransferWork work;
  int64_t fields = selected.getMain().getFields();
  if (!addPhase(work, selected.getMain(), outer / fields, inner) ||
      !addPhase(work, selected.getTail(), outer % fields, inner)) return std::nullopt;
  return work;
}

std::optional<int64_t> PanelTransferWork::total() const {
  int64_t total = 0;
  for (int64_t value : {loads, stores, registerTransfers, sourceAddresses})
    if (value < 0 || llvm::AddOverflow(total, value, total)) return std::nullopt;
  if (!addProduct(total, rowIterations, 2) || !addProduct(total, vectorIterations, 2)) return std::nullopt;
  return total;
}
