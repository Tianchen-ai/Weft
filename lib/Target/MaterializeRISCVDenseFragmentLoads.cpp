#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace weft;

namespace {

struct DenseFragmentRead {
  riscv::LoadOp load;
  SmallVector<riscv::ConvertLayoutOp> conversions;
  riscv::AccessAttr access;
};

bool insideOriginalTransfer(riscv::LoadOp load) {
  Operation *child = load;
  for (Operation *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    auto guard = branch
        ? branch.getCondition().getDefiningOp<riscv::DenseRowWindowGuardOp>()
        : riscv::DenseRowWindowGuardOp();
    if (guard && guard.getSource() == load.getRegion() &&
        child->getParentRegion() == &branch.getElseRegion())
      return true;
  }
  return false;
}

std::optional<DenseFragmentRead> denseRead(riscv::IMEPackOp pack) {
  auto kernel = pack->getParentOfType<riscv::KernelOp>();
  if (!kernel || kernel.getResourcesMaterialized())
    return std::nullopt;
  DenseFragmentRead result;
  Value input = pack.getInput();
  auto packed = dyn_cast<riscv::ValueType>(input.getType());
  if (!packed || packed.getAxisIds().size() != 2 ||
      packed.getElementType() != pack.getResult().getType().getElementType() ||
      llvm::any_of(packed.getShape().asArrayRef(), [](int64_t extent) { return extent <= 0; }))
    return std::nullopt;
  while (auto conversion = input.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (!input.hasOneUse() || conversion->getBlock() != pack->getBlock() ||
        !isMemoryEffectFree(conversion) || conversion->getNumOperands() != 1)
      return std::nullopt;
    auto source = dyn_cast<riscv::ValueType>(conversion.getInput().getType());
    if (!source || source.getShape() != packed.getShape() ||
        source.getAxisIds() != packed.getAxisIds() ||
        source.getElementType() != packed.getElementType())
      return std::nullopt;
    result.conversions.push_back(conversion);
    input = conversion.getInput();
  }
  result.load = input.getDefiningOp<riscv::LoadOp>();
  if (!result.load || !input.hasOneUse() || result.load.getSnapshotStorage() ||
      result.load->getBlock() != pack->getBlock() || insideOriginalTransfer(result.load))
    return std::nullopt;
  auto access = riscv::fragmentDenseReadAccess(result.load.getRegion(),
                                              pack.getResult().getType());
  if (!access)
    return std::nullopt;
  result.access = *access;
  auto fragment = pack.getResult().getType();
  // This is a representation change of the original complete numeric read.
  // A pack of a projected or computed subset cannot authorize a new read.
  for (auto [position, axis] : llvm::enumerate(packed.getAxisIds().asArrayRef())) {
    auto found = llvm::find(fragment.getAxisIds().asArrayRef(), axis);
    if (found == fragment.getAxisIds().asArrayRef().end() ||
        packed.getShape()[position] != fragment.getShape()[found - fragment.getAxisIds().asArrayRef().begin()])
      return std::nullopt;
  }
  return result;
}

class MaterializeRISCVDenseFragmentLoadsPass
    : public PassWrapper<MaterializeRISCVDenseFragmentLoadsPass,
                         OperationPass<ModuleOp>> {
public:
  MaterializeRISCVDenseFragmentLoadsPass() = default;
  MaterializeRISCVDenseFragmentLoadsPass(
      const MaterializeRISCVDenseFragmentLoadsPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override {
    return "weft-riscv-materialize-dense-fragment-loads";
  }
  StringRef getDescription() const override {
    return "Materialize guarded whole-row transfers at the original dense fragment read";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    SmallVector<riscv::IMEPackOp> packs;
    getOperation().walk([&](riscv::IMEPackOp pack) { packs.push_back(pack); });
    IRRewriter rewriter(&getContext());
    for (auto pack : packs) {
      auto read = denseRead(pack);
      if (!read)
        continue;
      auto fragment = pack.getResult().getType();
      auto memory = read->load.getRegion().getType();
      auto packing = fragment.getPacking();
      int64_t rowAxis = fragment.getAxisIds()[packing.getAxisOrder()[0]];
      int64_t innerAxis = fragment.getAxisIds()[packing.getAxisOrder()[1]];
      int64_t rowPosition = llvm::find(memory.getAxisIds().asArrayRef(), rowAxis) -
                            memory.getAxisIds().asArrayRef().begin();
      int64_t innerPosition = llvm::find(memory.getAxisIds().asArrayRef(), innerAxis) -
                              memory.getAxisIds().asArrayRef().begin();
      int64_t rows = fragment.getShape()[packing.getAxisOrder()[0]];
      int64_t columns = packing.getColumnsPerTile();
      // Keep the memory event at its original position. Only pure representation
      // work moves into the two branches; neither branch re-reads an SSA value.
      rewriter.setInsertionPoint(read->load);
      auto guard = rewriter.create<riscv::DenseRowWindowGuardOp>(
          read->load.getLoc(), rewriter.getI1Type(), read->load.getRegion(),
          rowPosition, innerPosition, rows, columns, int64_t{8});
      auto branch = rewriter.create<scf::IfOp>(read->load.getLoc(),
          TypeRange{fragment}, guard.getResult(), true);
      riscv_internal::copyOrigin(read->load, guard);
      riscv_internal::copyOrigin(read->load, branch);
      rewriter.setInsertionPointToStart(branch.thenBlock());
      auto direct = rewriter.create<riscv::IMEFragmentDenseLoadOp>(
          read->load.getLoc(), fragment, read->load.getRegion(), read->access,
          riscv_internal::leaf(rewriter, "ime", "fragment-dense-load",
              "ime.dense-row-words", "ime.dense-row-words", 0, 0, 0,
              fragment.getResourceGroups(), "none", "agnostic"));
      riscv_internal::copyOrigin(read->load, direct);
      rewriter.create<scf::YieldOp>(read->load.getLoc(), direct.getResult());
      rewriter.setInsertionPointToStart(branch.elseBlock());
      IRMapping mapping;
      rewriter.clone(*read->load, mapping);
      for (auto conversion : llvm::reverse(read->conversions))
        rewriter.clone(*conversion, mapping);
      auto *original = rewriter.clone(*pack, mapping);
      rewriter.create<scf::YieldOp>(pack.getLoc(), original->getResults());
      rewriter.replaceOp(pack, branch.getResults());
      for (auto conversion : read->conversions)
        rewriter.eraseOp(conversion);
      rewriter.eraseOp(read->load);
      ++selectedTransfers;
    }
  }

private:
  Statistic selectedTransfers{this, "selected-transfers",
      "Dense numeric reads directly supplying complete fragment row windows"};
};

} // namespace

std::unique_ptr<Pass> weft::createMaterializeRISCVDenseFragmentLoadsPass() {
  return std::make_unique<MaterializeRISCVDenseFragmentLoadsPass>();
}
