#include "FragmentSupply.h"
#include "../RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#include <limits>

using namespace weft;

void weft::riscv_internal::formFragmentWindowLoads(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::IMEPackOp> packs;
  module.walk([&](riscv::IMEPackOp pack) { packs.push_back(pack); });
  int64_t selected = 0;
  for (auto pack : packs) {
    if (pack->getParentOfType<riscv::KernelOp>().getResourcesMaterialized()) continue;
    auto *conversion = pack.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<riscv::WidenOp, riscv::CastOp>(conversion) ||
        !conversion->hasOneUse() || conversion->getBlock() != pack->getBlock()) continue;
    auto read = conversion->getOperand(0).getDefiningOp<riscv::RVVReplicaStorageLoadOp>();
    auto fragment = pack.getResult().getType();
    if (!read || !read->hasOneUse() || read->getBlock() != pack->getBlock() ||
        pack.getInput().getType().getElementType() != fragment.getElementType() ||
        read.getPlan().getKind() != "layered" || read.getRecordRank() != 1) continue;
    auto access = riscv::fragmentWindowReadAccess(read.getField(), read.getLogicalBase(), fragment);
    auto field = read.getField().getDefiningOp<riscv::FieldOp>();
    auto owner = field ? sourceLoad(field.getOwner()) : riscv::LoadOp();
    if (!access || read.getAccess() != *access || !owner || needsReadSnapshot(owner) ||
        riscv::readCrossesWrite(owner, pack)) continue;
    const int64_t rows = fragment.getShape()[fragment.getPacking().getAxisOrder()[0]];
    const int64_t columns = fragment.getPacking().getColumnsPerTile();
    const int64_t bits = logicalBitWidth(read.getResult().getType());
    if (rows <= 1 || riscv::rvvLaneCount(read.getResult().getType()) != columns ||
        read.getPlan().getProjectionExtent() != columns ||
        read.getPlan().getProjectionBase() != 0 || read.getPlan().getProjectionStride() != 1 ||
        read.getPlan().getProjectionRepeat() != 1 ||
        read.getWindowOffsets().size() != static_cast<size_t>(rows) ||
        read.getWindowForPart().size() != static_cast<size_t>(rows) ||
        read.getRecordCoordinatesForWindow().size() != static_cast<size_t>(rows)) continue;
    bool uniform = true;
    for (int64_t row = 0; row < rows; ++row)
      uniform &= read.getWindowOffsets()[row] == 0 && read.getWindowForPart()[row] == row &&
          read.getRecordCoordinatesForWindow()[row] == row && read.getLayerForPart()[row] == 0 &&
          read.getPhysicalLayerForPart()[row] == 0 && read.getShiftOffsetForPart()[row] == 0 &&
          read.getShiftBaseFactorForPart()[row] == bits && read.getMaskValueForPart()[row] == (1LL << bits) - 1;
    if (!uniform) continue;
    rewriter.setInsertionPoint(pack);
    const char *spelling = riscv::supportsFragmentWordWindows(read.getField(), fragment)
        ? "ime.field-byte-windows.guarded-vlse64" : "ime.field-byte-windows";
    auto raw = riscv::IMEFragmentWindowLoadOp::create(rewriter, pack.getLoc(), fragment,
        read.getField(), read.getLogicalBase(), *access,
        leaf(rewriter, "ime", "fragment-window-load", spelling, spelling,
             0, 0, fragment.getPacking().getRowsPerTile(), fragment.getResourceGroups(), "none", "agnostic"));
    auto constant = [&](int64_t value) -> mlir::Value {
      return mlir::arith::ConstantIndexOp::create(rewriter, pack.getLoc(), value);
    };
    auto within = mlir::arith::RemUIOp::create(rewriter, pack.getLoc(), read.getLogicalBase(), constant(access->getGroupSize()));
    auto layer = mlir::arith::DivUIOp::create(rewriter, pack.getLoc(), within, constant(access->getLayerSize()));
    auto shift = mlir::arith::MulIOp::create(rewriter, pack.getLoc(), layer, constant(bits));
    auto decoded = riscv::IMEFragmentDecodeOp::create(rewriter, pack.getLoc(), fragment, raw, shift, bits, false,
        leaf(rewriter, "ime", "fragment-decode", "ime.decode.bits", "ime.decode.bits", 0, 0, 1, fragment.getResourceGroups()));
    copyOrigin(read, raw);
    copyOrigin(pack, decoded);
    rewriter.replaceOp(pack, decoded.getResult());
    rewriter.eraseOp(conversion);
    rewriter.eraseOp(read);
    ++selected;
  }
  if (selected) llvm::errs() << "weft-fragment-windows: selected=" << selected << '\n';
}

void weft::riscv_internal::materializeFragmentRowSupply(
    riscv::KernelOp kernel, mlir::IRRewriter &rewriter) {
  llvm::DenseMap<std::pair<mlir::Value, mlir::Type>, llvm::SmallVector<mlir::Value>> supplies;
  kernel.walk([&](riscv::IMEFragmentPointwiseOp operation) {
    auto input = mlir::dyn_cast<riscv::ValueType>(operation.getRhs().getType());
    auto fragment = operation.getLhs().getType();
    if (!input || !riscv::fragmentPointwiseTemporaryGroups(fragment, input) ||
        fragment.getPacking().getColumnsPerTile() != 8)
      return;
    auto &prior = supplies[{operation.getRhs(), fragment}];
    mlir::Value broadcast;
    for (auto value : prior)
      if (value.getDefiningOp()->getBlock() == operation->getBlock()) {
        broadcast = value;
        break;
    }
    if (!broadcast) {
      rewriter.setInsertionPoint(operation);
      if (auto *definition = operation.getRhs().getDefiningOp();
          definition && definition->getBlock() == operation->getBlock())
        rewriter.setInsertionPointAfter(definition);
      const int64_t parts = riscv::fragmentPartCount(fragment);
      const bool byteRepeat = parts == 1 &&
          riscv::supportsFragmentByteRepeat(fragment, input, kernel.getTarget());
      const bool partialRows = llvm::any_of(kernel.getTarget().getFragments(),
          [&](mlir::Attribute attribute) {
            auto capability = mlir::cast<riscv::FragmentCapabilityAttr>(attribute);
            return capability.getInstruction() == fragment.getFamily() &&
                capability.getActiveMFactor() < capability.getMFactor();
          });
      // Full-row accumulators leave little room for invariant gather indices.
      // Keep those indices inside the local issue instead of extending their
      // lifetime across the contraction's wider floating epilogue.
      const bool preparedIndices = !byteRepeat && parts > 1 && parts <= 4 && partialRows;
      auto instruction = byteRepeat ? "ime.rows.byte-repeat" :
                         preparedIndices ? "ime.rows.indices" : "ime.rows.local";
      auto supply = riscv::IMEFragmentRowBroadcastOp::create(rewriter, operation.getLoc(),
          fragment, operation.getRhs(), leaf(rewriter, "ime", "fragment-broadcast",
              instruction, instruction, 0, 0,
              byteRepeat || preparedIndices ? 2 * parts : parts + 1));
      copyOrigin(operation, supply);
      broadcast = supply.getResult();
      prior.push_back(broadcast);
    }
    operation.getRhsMutable().assign(broadcast);
    const bool grouped = riscv::supportsFragmentGroupedLoad(fragment, kernel.getTarget());
    auto instruction = ("ime.fragment." + operation.getKind()).str() + (grouped ? ".grouped" : "");
    operation.setLeafAttr(leaf(rewriter, "ime", "fragment-pointwise",
        instruction, instruction,
        0, 0, 0, fragment.getResourceGroups()));
  });
}

namespace {
std::optional<int64_t> constantIndex(mlir::Value value, unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value();
  auto *definition = value.getDefiningOp();
  if (!definition || definition->getNumOperands() != 2 ||
      !mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp>(definition))
    return std::nullopt;
  auto lhs = constantIndex(definition->getOperand(0), depth + 1);
  auto rhs = constantIndex(definition->getOperand(1), depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;
  __int128 result = mlir::isa<mlir::arith::AddIOp>(definition)
      ? static_cast<__int128>(*lhs) + *rhs : mlir::isa<mlir::arith::SubIOp>(definition)
      ? static_cast<__int128>(*lhs) - *rhs : static_cast<__int128>(*lhs) * *rhs;
  if (result < std::numeric_limits<int64_t>::min() || result > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(result);
}

std::optional<int64_t> relativeIndex(mlir::Value value, mlir::Value base, unsigned depth = 0) {
  if (value == base)
    return 0;
  if (depth > 12)
    return std::nullopt;
  auto add = value.getDefiningOp<mlir::arith::AddIOp>();
  if (!add)
    return std::nullopt;
  for (unsigned side = 0; side < 2; ++side) {
    auto offset = constantIndex(add->getOperand(side));
    auto relative = relativeIndex(add->getOperand(1 - side), base, depth + 1);
    if (offset && relative && *offset >= 0 && *relative >= 0 &&
        *relative <= std::numeric_limits<int64_t>::max() - *offset)
      return *relative + *offset;
  }
  return std::nullopt;
}

struct Supply {
  riscv::IMEFragmentStorageLoadOp read;
  riscv::FieldOp field;
  riscv::LoadOp owner;
  riscv::PhysicalPointOp point;
  riscv::PhysicalPointOp parent;
};

std::optional<Supply> supply(riscv::IMEFragmentStorageLoadOp read) {
  auto access = read.getAccess();
  auto fragment = read.getResult().getType();
  auto field = riscv::fieldReadProjection(read.getInput());
  auto owner = field ? riscv_internal::sourceLoad(field.getOwner()) : riscv::LoadOp();
  const int64_t innerAxis = fragment.getAxisIds()[fragment.getPacking().getAxisOrder()[1]];
  auto point = riscv_internal::originPoint(read.getInput(), innerAxis);
  auto parent = point ? point.getParent().getDefiningOp<riscv::PhysicalPointOp>() : riscv::PhysicalPointOp();
  if (access.getMapping() != "grouped_layered" || !field || !owner || !point || !parent ||
      owner.getSnapshotStorage() || access.getLayerSize() <= 0 ||
      access.getGroupSize() <= access.getLayerSize() ||
      access.getGroupSize() % access.getLayerSize() ||
      owner.getRegion().getType().getElements() % access.getGroupSize() ||
      !riscv::knownIndexMultipleOf(parent.getBase(), access.getGroupSize()) ||
      riscv::readCrossesWrite(owner, read))
    return std::nullopt;
  return Supply{read, field, owner, point, parent};
}

bool readonlyIssue(mlir::Operation &operation) {
  if (operation.getNumRegions() || operation.hasAttr("birth_id") ||
      mlir::isa<riscv::NewOp, riscv::MaterializeOp>(operation))
    return false;
  if (mlir::isa<riscv::IMEFragmentMMAOp>(operation))
    return true;
  if (mlir::isMemoryEffectFree(&operation) || mlir::isa<riscv::PhysicalPointOp>(operation))
    return true;
  auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::all_of(effects, [](const auto &effect) {
    return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}
}

mlir::LogicalResult weft::riscv_internal::shareFragmentStorage(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::scf::ForOp> issues;
  module.walk([&](mlir::scf::ForOp loop) {
    auto kernel = loop->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      return;
    if (!llvm::all_of(loop.getBody()->without_terminator(), readonlyIssue))
      return;
    auto lower = constantIndex(loop.getLowerBound());
    auto upper = constantIndex(loop.getUpperBound());
    auto step = constantIndex(loop.getStep());
    if (!lower || !upper || !step || *lower < 0 || *step != 1 ||
        *upper <= *lower || *upper - *lower < 2 || *upper - *lower > 4)
      return;
    for (auto read : loop.getBody()->getOps<riscv::IMEFragmentStorageLoadOp>()) {
      auto candidate = supply(read);
      if (!candidate || candidate->point->getBlock() != loop.getBody())
        continue;
      const int64_t layers = read.getAccess().getGroupSize() / read.getAccess().getLayerSize();
      auto base = candidate->point.getBase().getDefiningOp<mlir::arith::AddIOp>();
      // A peeled pipeline may begin partway through a group. Unroll only
      // when this bounded interval still contains a complete shared window.
      if (!base || layers <= 0 || layers > 4 ||
          *upper / layers - (*lower / layers + (*lower % layers != 0)) < 1 ||
          base.getLhs() != candidate->parent.getBase())
        continue;
      auto offset = base.getRhs().getDefiningOp<mlir::arith::MulIOp>();
      if (offset && offset.getLhs() == loop.getInductionVar() &&
          constantIndex(offset.getRhs()) == read.getAccess().getLayerSize()) {
        issues.push_back(loop);
        break;
      }
    }
  });
  for (auto loop : issues) {
    auto unrolled = mlir::loopUnrollByFactor(loop,
        *constantIndex(loop.getUpperBound()) - *constantIndex(loop.getLowerBound()));
    if (mlir::failed(unrolled))
      return loop.emitError("bounded packed-layer issue group could not be materialized");
  }

  llvm::SmallVector<riscv::IMEFragmentStorageLoadOp> reads;
  module.walk([&](riscv::IMEFragmentStorageLoadOp read) {
    auto kernel = read->getParentOfType<riscv::KernelOp>();
    if (kernel && !kernel.getResourcesMaterialized())
      reads.push_back(read);
  });
  llvm::SmallVector<Supply> previous;
  int64_t eliminated = 0;
  for (auto read : reads) {
    auto current = supply(read);
    if (!current)
      continue;
    auto offset = relativeIndex(current->point.getBase(), current->parent.getBase());
    if (!offset)
      continue;
    const int64_t group = read.getAccess().getGroupSize();
    const int64_t layer = read.getAccess().getLayerSize();
    bool shared = false;
    for (Supply &prior : previous) {
      if (prior.field != current->field || prior.owner != current->owner ||
          prior.parent != current->parent || prior.read->getBlock() != read->getBlock() ||
          prior.point.getResult().getType() != current->point.getResult().getType() ||
          prior.point.getActive() != current->point.getActive() ||
          prior.point.getPartition() != current->point.getPartition() ||
          prior.read.getResult().getType() != read.getResult().getType() ||
          prior.read.getAccess() != read.getAccess())
        continue;
      auto priorOffset = relativeIndex(prior.point.getBase(), prior.parent.getBase());
      if (!priorOffset || *priorOffset / group != *offset / group ||
          *priorOffset % layer != *offset % layer || riscv::readCrossesWrite(prior.owner, read))
        continue;
      rewriter.replaceOp(read, prior.read.getResult());
      ++eliminated;
      shared = true;
      break;
    }
    if (!shared) {
      previous.push_back(*current);
      if (previous.size() > 32)
        previous.erase(previous.begin());
    }
  }
  if (!issues.empty() && !eliminated)
    return module.emitError("bounded layer grouping did not expose its predicted common byte supply");
  if (eliminated)
    llvm::errs() << "weft-fragment-supply: issue-groups=" << issues.size()
                 << " raw-loads-shared=" << eliminated << '\n';
  return mlir::success();
}
