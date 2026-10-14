#include "Weft/Target/RISCVPasses.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/ScalarPacket.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace weft;

namespace {
struct PacketUses {
  int64_t axis;
  int64_t steps;
  int64_t parts;
};

std::optional<PacketUses> packetUses(riscv::ScalarTileLoadOp load) {
  if (load.getPacket() || load.getGuarded()) return std::nullopt;
  auto type = load.getResult().getType();
  if (!type.getElementType().isF32() && !type.getElementType().isF64()) return std::nullopt;
  auto instruction = load.getLeaf().getInstruction();
  if (instruction != "scalar.tile-load.unit" && instruction != "scalar.tile-load.strided") return std::nullopt;
  auto parameters = load.getLeaf().getParameters();
  auto coordinates = riscv::RegisterCoordinates::get(type.getLayout());
  if (parameters.size() != 1 || !coordinates || coordinates->parts() < 4 || coordinates->parts() > 16)
    return std::nullopt;
  auto axes = type.getAxisIds().asArrayRef();
  auto position = llvm::find(axes, parameters[0]);
  if (position == axes.end()) return std::nullopt;
  int64_t steps = type.getShape()[position - axes.begin()];
  if (steps != 2 && steps != 4) return std::nullopt;
  // This frontier belongs to a complete supply packet consumed in this block.
  // A packet carried across pipeline stages has different residency and is
  // deliberately left to the existing pipeline's explicit versioned program.
  llvm::SmallPtrSet<Operation *, 8> projections;
  llvm::SmallPtrSet<Operation *, 8> updates;
  SmallVector<bool> covered(steps, false);
  for (Operation *user : load.getResult().getUsers()) {
    auto extract = dyn_cast<riscv::ExtractOp>(user);
    if (!extract || extract->getBlock() != load->getBlock() ||
        !load->isBeforeInBlock(extract) || extract.getSelectors().size() != axes.size() ||
        extract.getLeaf().getInstruction() != "scalar.replica-project") return std::nullopt;
    unsigned index = 0;
    for (auto [number, attribute] : llvm::enumerate(extract.getSelectors())) {
      auto selector = dyn_cast<StringAttr>(attribute);
      if (!selector) return std::nullopt;
      if (axes[number] == parameters[0]) {
        if (selector.getValue() != "index" || index >= extract.getIndices().size()) return std::nullopt;
        auto value = riscv::denseConstantIndex(extract.getIndices()[index++]);
        if (!value || *value < 0 || *value >= steps) return std::nullopt;
        covered[*value] = true;
      } else if (selector.getValue() != "all") return std::nullopt;
    }
    if (index != extract.getIndices().size() || extract.getResult().use_empty()) return std::nullopt;
    projections.insert(user);
    for (Operation *consumer : extract.getResult().getUsers()) {
      auto update = dyn_cast<riscv::RVVTileContractOp>(consumer);
      if (!update || update->getBlock() != load->getBlock() || !extract->isBeforeInBlock(update) ||
          update.getReductionAxis() != parameters[0]) return std::nullopt;
      Value scalar = update.getLaneOperand() == "lhs" ? update.getRhs() : update.getLhs();
      Value vector = update.getLaneOperand() == "lhs" ? update.getLhs() : update.getRhs();
      if (scalar != extract.getResult()) return std::nullopt;
      auto read = vector.getDefiningOp<riscv::RVVTileLoadOp>();
      if (!read || read->getBlock() != load->getBlock() || !load->isBeforeInBlock(read)) return std::nullopt;
      updates.insert(consumer);
    }
  }
  if (projections.size() != size_t(steps) || updates.size() != size_t(steps) ||
      llvm::is_contained(covered, false)) return std::nullopt;
  // All projected rows must feed one ordered accumulator chain. Independent
  // consumers are not an invitation to freeze a larger scalar live frontier.
  SmallVector<Operation *> ordered(updates.begin(), updates.end());
  llvm::sort(ordered, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (size_t i = 1; i < ordered.size(); ++i)
    if (cast<riscv::RVVTileContractOp>(ordered[i]).getAccumulator() != ordered[i - 1]->getResult(0))
      return std::nullopt;
  return PacketUses{parameters[0], steps, coordinates->parts()};
}

class SelectRISCVScalarPacketsPass
    : public PassWrapper<SelectRISCVScalarPacketsPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "weft-riscv-select-scalar-packets"; }
  StringRef getDescription() const override { return "Close scalar load packets before their ordered vector consumers"; }
  void runOnOperation() override {
    getOperation().walk([&](riscv::ScalarTileLoadOp load) {
      auto uses = packetUses(load);
      if (!uses) return;
      auto live = riscv::scalarPacketResidency(load);
      if (!live || *live > 32) return;
      if (load->getParentOfType<riscv::KernelOp>().getResourcesMaterialized()) {
        load.emitError("scalar packet selection requires an open physical resource boundary");
        signalPassFailure();
        return;
      }
      load.setPacketAttr(riscv::ScalarPacketAttr::get(&getContext(), uses->axis,
          uses->steps, uses->parts, uses->parts, *live));
      if (auto loop = load->getParentOfType<scf::ForOp>())
        loop->setAttr("weft.riscv.system_unroll", StringAttr::get(&getContext(), "disable"));
    });
  }
};
} // namespace

std::unique_ptr<Pass> weft::createSelectRISCVScalarPacketsPass() {
  return std::make_unique<SelectRISCVScalarPacketsPass>();
}
