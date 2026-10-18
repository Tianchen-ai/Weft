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

void selectScalarWindows(ModuleOp module, IRRewriter &rewriter) {
  SmallVector<riscv::ConvertLayoutOp> extracts;
  module.walk([&](riscv::ConvertLayoutOp op) {
    if (op.getConversion().getKind() == "extract" &&
        op.getResult().getType().getLayout().getCarrier() == "scalar")
      extracts.push_back(op);
  });
  unsigned selected = 0, packed = 0;
  for (auto extract : extracts) {
    auto load = extract.getInput().getDefiningOp<riscv::RVVReplicaStorageLoadOp>();
    if (!load || !load->hasOneUse() || load->getBlock() != extract->getBlock() ||
        load->getParentOfType<riscv::KernelOp>().getResourcesMaterialized() ||
        load.getPlan().getKind() != "unit" || load.getRecordRank() != 0)
      continue;
    auto target = load->getParentOfType<riscv::KernelOp>().getTarget();
    auto layout = load.getResult().getType().getLayout();
    auto field = load.getField().getDefiningOp<riscv::FieldOp>();
    auto type = extract.getResult().getType();
    auto integer = dyn_cast<IntegerType>(type.getElementType());
    auto source = riscv::RegisterCoordinates::get(load.getResult().getType().getLayout());
    auto result = riscv::RegisterCoordinates::get(type.getLayout());
    if (!field || load.getField().getType().getShape().size() != 1 ||
        field.getAccess().getMapping() != "natural" || field.getAccess().getForm() != "unit" ||
        !integer || integer.isSignless() ||
        (integer.getWidth() != 8 && integer.getWidth() != 16 && integer.getWidth() != 32) ||
        !source || !result || result->lanes() != 1 || result->parts() < 2 || result->parts() > 16 ||
        result->parts() > load.getField().getType().getShape()[0] ||
        type.getLayout().getValidity() != "full" ||
        load.getResult().getType().getLayout().getValidity() != "full" ||
        !llvm::equal(result->extents(), type.getShape().asArrayRef()) ||
        load.getWindowOffsets().size() * source->lanes() != static_cast<size_t>(result->parts()))
      continue;
    // Underfilled carriers can use direct scalar supply. Full carriers keep
    // vector traffic, with adjacent windows packed when one full carrier fits.
    const int64_t capacity = target.getVlenBits() * layout.getLmulEighths() / (8 * layout.getSew());
    // The original unit windows must cover this exact contiguous scalar range.
    // Recover the part permutation from typed coordinates, never from use order.
    bool closed = true;
    for (auto [window, offset] : llvm::enumerate(load.getWindowOffsets()))
      closed &= offset == static_cast<int64_t>(window) * source->lanes();
    SmallVector<int64_t> bases;
    for (int64_t part = 0; closed && part < result->parts(); ++part) {
      auto logical = result->logical(part);
      auto location = logical ? source->locate(*logical) : std::nullopt;
      if (!location || location->part >= static_cast<int64_t>(load.getWindowForPart().size())) {
        closed = false;
        break;
      }
      const int64_t window = load.getWindowForPart()[location->part];
      const int64_t offset = load.getWindowOffsets()[window] + location->lane;
      int64_t linear = 0;
      for (auto [coordinate, size] : llvm::zip(*logical, type.getShape().asArrayRef()))
        linear = linear * size + coordinate;
      closed &= offset == linear;
      bases.push_back(offset);
    }
    if (!closed) continue;
    rewriter.setInsertionPoint(load);
    if (source->lanes() == capacity) {
      if (load.getWindowOffsets().size() <= 1) continue;
      const int64_t numerator = result->parts() * layout.getSew() * 8;
      if (numerator % target.getVlenBits()) continue;
      const int64_t lmul = numerator / target.getVlenBits();
      if (lmul > 8 || !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul))
        continue;
      SmallVector<int64_t> one(type.getShape().size(), 1);
      auto ones = rewriter.getDenseI64ArrayAttr(one);
      auto combined = riscv::LayoutAttr::get(rewriter.getContext(), "rvv", type.getAxisIds(),
          ones, type.getShape(), ones, ones, ones, layout.getSew(), lmul, result->parts(), 1, "full");
      if (!riscv::supportsRVVLayout(target, combined)) continue;
      auto readType = riscv::ValueType::get(rewriter.getContext(), type.getElementType(),
          type.getShape(), type.getAxisIds(), combined);
      auto zero = rewriter.getDenseI64ArrayAttr({0});
      auto read = riscv::RVVReplicaStorageLoadOp::create(rewriter, load.getLoc(), readType,
          load.getField(), load.getLogicalBase(), load.getPlan(), uint64_t{0}, zero,
          rewriter.getDenseI64ArrayAttr({}), zero, zero, zero, zero, zero, zero,
          load.getAccess(), riscv_internal::leaf(rewriter, "rvv", "replica-storage-load",
              "rvv.replica-storage-load.natural", "rvv.replica-storage-load.natural",
              0, 1, 1, 0, "none", "exact"));
      riscv_internal::copyOrigin(load, read);
      extract.getInputMutable().assign(read.getResult());
      extract.setLeafAttr(riscv_internal::leaf(rewriter, "rvv", "layout-conversion",
          "rvv.extract", "rvv.extract", 1, 0, 0, 0, "none", "exact"));
      rewriter.eraseOp(load);
      ++packed;
      continue;
    }
    if (source->lanes() > capacity) continue;
    const int64_t sourceAxis = load.getField().getType().getAxisIds()[0];
    const int64_t resultAxis = type.getAxisIds().asArrayRef().back();
    auto read = riscv::RVVRegularRepeatScalarLoadOp::create(rewriter, load.getLoc(), type,
        field.getResult(), sourceAxis, resultAxis, load.getLogicalBase(), result->parts(), 1,
        rewriter.getDenseI64ArrayAttr(bases), field.getAccess(),
        riscv_internal::leaf(rewriter, "scalar", "regular-repeat-scalar-load",
            "scalar.regular-repeat-load", "scalar.regular-repeat-load", 0, 0, 0, 0,
            "none", "exact", {sourceAxis, resultAxis, result->parts(), 1}));
    riscv_internal::copyOrigin(load, read);
    rewriter.replaceOp(extract, read.getResult());
    rewriter.eraseOp(load);
    ++selected;
  }
  if (selected || packed)
    llvm::errs() << "weft-scalar-storage-windows: selected=" << selected
                 << " packed=" << packed << '\n';
}

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVScalarPacketsPass)
  StringRef getArgument() const override { return "weft-riscv-select-scalar-packets"; }
  StringRef getDescription() const override { return "Close scalar load packets before their ordered vector consumers"; }
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    selectScalarWindows(getOperation(), rewriter);
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
