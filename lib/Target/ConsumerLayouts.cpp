#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/RISCVCompilation.h"
#include "RISCVPhysicalSupport.h"
#include "IME/FragmentSupply.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace weft;
using namespace weft::riscv_internal;

namespace {
bool pointwise(mlir::Operation *op) {
  return mlir::isa<riscv::BinaryOp, riscv::UnaryOp, riscv::CastOp,
                   riscv::WidenOp, riscv::NarrowOp>(op);
}

bool sameLaneDomain(mlir::Value value, riscv::ValueType anchor,
                    riscv::LayoutAttr layout) {
  auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!type || type.getLayout().getCarrier() != "rvv")
    return false;
  auto axes = anchor.getAxisIds().asArrayRef();
  int64_t previous = -1;
  for (auto [index, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || found - axes.begin() <= previous ||
        type.getShape()[index] != anchor.getShape()[found - axes.begin()])
      return false;
    previous = found - axes.begin();
  }
  // Dropping unit physical axes does not duplicate any supply lanes.
  for (auto [index, axis] : llvm::enumerate(axes))
    if (!llvm::is_contained(type.getAxisIds().asArrayRef(), axis) &&
        (layout.getTimeFactors()[index] != 1 || layout.getLaneFactors()[index] != 1 ||
         layout.getReplicaFactors()[index] != 1 || layout.getFragmentFactors()[index] != 1 ||
         layout.getLocalFactors()[index] != 1))
      return false;
  return true;
}

std::optional<llvm::SetVector<mlir::Value>> component(riscv::IMEUnpackOp anchor,
                                                    riscv::LayoutAttr layout) {
  llvm::SetVector<mlir::Value> values;
  auto type = anchor.getResult().getType();
  auto add = [&](mlir::Value value) {
    if (sameLaneDomain(value, type, layout))
      values.insert(value);
  };
  add(anchor.getResult());
  size_t edges = 0;
  for (size_t cursor = 0; cursor < values.size(); ++cursor) {
    if (values.size() > 128) {
      llvm::errs() << "weft-consumer-layout: rejected=component-value-limit limit=128\n";
      return std::nullopt;
    }
    auto value = values[cursor];
    auto *producer = value.getDefiningOp();
    if (producer && pointwise(producer))
      for (auto operand : producer->getOperands())
        if (auto *definition = operand.getDefiningOp();
            (definition && (pointwise(definition) || mlir::isa<riscv::NewOp, riscv::IMEUnpackOp, mlir::scf::ForOp>(definition))) ||
            mlir::isa<mlir::BlockArgument>(operand))
          add(operand);
    auto connectLoop = [&](mlir::scf::ForOp loop, unsigned index) {
      add(loop.getInitArgs()[index]);
      add(loop.getRegionIterArgs()[index]);
      add(loop.getResult(index));
      add(mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(index));
    };
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop = mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
      if (loop && argument.getArgNumber() > 0)
        connectLoop(loop, argument.getArgNumber() - 1);
    }
    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(producer))
      connectLoop(loop, mlir::cast<mlir::OpResult>(value).getResultNumber());
    for (mlir::OpOperand &use : value.getUses()) {
      if (++edges > 1024) {
        llvm::errs() << "weft-consumer-layout: rejected=component-edge-limit limit=1024\n";
        return std::nullopt;
      }
      auto *user = use.getOwner();
      if (pointwise(user))
        add(user->getResult(0));
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user); loop && use.getOperandNumber() >= 3)
        connectLoop(loop, use.getOperandNumber() - 3);
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user))
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(yield->getParentOp()))
          connectLoop(loop, use.getOperandNumber());
    }
  }
  return values;
}

mlir::FailureOr<mlir::Value> convert(mlir::Value value, riscv::ValueType type,
                                    mlir::Operation *at, mlir::IRRewriter &rewriter) {
  auto before = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!before)
    return mlir::failure();
  if (before == type)
    return value;
  llvm::SmallVector<mlir::Value> indices;
  if (auto field = riscv::fieldReadProjection(value, &indices)) {
    auto access = field.getAccess();
    const bool oneLaneAxis = type.getLayout().getReplicaGroup() == 1 &&
        llvm::count_if(type.getLayout().getLaneFactors().asArrayRef(), [](int64_t n) { return n > 1; }) <= 1;
    auto readType = oneLaneAxis ? type : before;
    auto temporary = riscv::fieldReadTemporaryGroups(value, readType, access, indices);
    if (!temporary)
      return mlir::failure();
    rewriter.setInsertionPoint(at);
    auto instruction = riscv::fieldReadInstruction(value, readType, access);
    auto read = rewriter.create<riscv::FieldReadOp>(at->getLoc(), readType, value, indices, access,
        leaf(rewriter, "transfer", "field-read", instruction, instruction, 0, 0, *temporary));
    copyOrigin(at, read);
    return oneLaneAxis ? mlir::FailureOr<mlir::Value>(read.getResult())
                       : convert(read.getResult(), type, at, rewriter);
  }
  auto edge = layoutConversion(rewriter, before.getLayout(), type.getLayout());
  if (before.getLayout().getCarrier() == "rvv" && type.getLayout().getCarrier() == "rvv") {
    if (!riscv::rvvLayoutRepackSlices(before, type))
      return mlir::failure();
    edge = conversion(rewriter, "lane_repack", "pure",
                       riscv::rvvLayoutRepackTemporaryGroups(at->getParentOfType<riscv::KernelOp>().getTarget(), before, type));
  }
  rewriter.setInsertionPoint(at);
  auto result = rewriter.create<riscv::ConvertLayoutOp>(at->getLoc(), type, value,
      edge, riscv::AccessAttr(), unselectedLeaf(rewriter));
  copyOrigin(at, result);
  return result.getResult();
}

bool rewriteComponent(riscv::IMEUnpackOp anchor, riscv::LayoutAttr layout,
                      mlir::IRRewriter &rewriter) {
  auto found = component(anchor, layout);
  if (!found) return false;
  auto &values = *found;
  auto target = anchor->getParentOfType<riscv::KernelOp>().getTarget();
  llvm::DenseMap<mlir::Value, riscv::ValueType> before;
  llvm::SmallPtrSet<mlir::Operation *, 32> changed;
  for (mlir::Value value : values) {
    if (auto *definition = value.getDefiningOp(); definition &&
        !pointwise(definition) && !mlir::isa<riscv::NewOp, riscv::IMEUnpackOp, mlir::scf::ForOp>(definition))
      return false;
    auto old = mlir::cast<riscv::ValueType>(value.getType());
    auto required = projectLayout(rewriter, old, layout, target);
    if (!required || !riscv::supportsRVVLayout(target, required))
      return false;
    before[value] = old;
    value.setType(withLayout(old, required));
    if (auto *definition = value.getDefiningOp())
      changed.insert(definition);
  }
  for (mlir::Value value : values) {
    llvm::SmallVector<mlir::OpOperand *> uses;
    for (mlir::OpOperand &use : value.getUses())
      uses.push_back(&use);
    for (auto *use : uses) {
      auto *user = use->getOwner();
      bool connected = changed.contains(user);
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user))
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(yield->getParentOp()))
          connected |= values.contains(loop.getResult(use->getOperandNumber()));
      if (connected)
        continue;
      auto converted = convert(value, before.lookup(value), user, rewriter);
      if (mlir::failed(converted))
        return false;
      use->set(*converted);
    }
  }
  for (auto *op : changed) {
    if (auto unpack = mlir::dyn_cast<riscv::IMEUnpackOp>(op)) {
      const int64_t temporary = riscv::fragmentUnpackTemporaryGroups(unpack.getInput().getType(),
                                                                    unpack.getResult().getType());
      unpack.setConversionAttr(conversion(rewriter, "fragment_to_rvv", "handoff", temporary));
      unpack.setLeafAttr(leaf(rewriter, "ime", "fragment-unpack", "ime.unpack.rvv", "ime.unpack.rvv", 0, 0, temporary));
      if (mlir::failed(unpack.verify()))
        return false;
      continue;
    }
    if (auto birth = mlir::dyn_cast<riscv::NewOp>(op);
        birth && layout.getReplicaGroup() > 1 && birth.getInitial() &&
        mlir::isa<mlir::FloatType, mlir::IntegerType>(birth.getInitial().getType())) {
      rewriter.setInsertionPoint(birth);
      auto splat = rewriter.create<riscv::RVVSplatOp>(birth.getLoc(),
          mlir::cast<riscv::ValueType>(birth.getResult().getType()), birth.getInitial(),
          leaf(rewriter, "rvv", "splat", "rvv.splat.local", "rvv.splat.local", 0, 0));
      copyOrigin(birth, splat);
      birth.getInitialMutable().assign(splat.getResult());
      continue;
    }
    if (!pointwise(op))
      continue;
    auto result = mlir::cast<riscv::ValueType>(op->getResult(0).getType());
    for (mlir::OpOperand &operand : op->getOpOperands()) {
      auto type = mlir::dyn_cast<riscv::ValueType>(operand.get().getType());
      if (!type)
        continue;
      auto required = projectLayout(rewriter, type, result.getLayout(), target);
      if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(op);
          binary && binary.getKind() == "mul" && operand.getOperandNumber() == 0 &&
          result.getElementType().isF32() && result.getLayout().getReplicaGroup() == 2 &&
          result.getAxisIds().size() == 2 && type.getAxisIds().size() == 1 &&
          type.getAxisIds()[0] == result.getAxisIds()[0] &&
          result.getLayout().getReplicaFactors()[0] == 1) {
        auto rhs = mlir::dyn_cast<riscv::ValueType>(binary.getRhs().getType());
        if (rhs && rhs.getAxisIds().size() == 1 && rhs.getAxisIds()[0] == result.getAxisIds()[1]) {
          auto one = integers(rewriter, {1});
          required = riscv::LayoutAttr::get(rewriter.getContext(), "scalar", type.getAxisIds(),
              one, one, integers(rewriter, {result.getLayout().getLaneFactors()[0]}), one, one,
              32, 0, 1, 0, result.getLayout().getValidity());
        }
      }
      if (!required)
        return false;
      auto projected = mlir::cast<riscv::ValueType>(withLayout(type, required));
      auto input = convert(operand.get(), projected, op, rewriter);
      if (mlir::failed(input))
        return false;
      operand.set(*input);
      auto inputLanes = staticProduct(required.getLaneFactors()).value() * required.getReplicaGroup();
      auto outputLanes = staticProduct(result.getLayout().getLaneFactors()).value() * result.getLayout().getReplicaGroup();
      if (required.getCarrier() != "rvv" || inputLanes == outputLanes)
        continue;
      if (mlir::isa<riscv::BinaryOp>(op))
        continue;
      if (type.getAxisIds().size() >= result.getAxisIds().size() || inputLanes > outputLanes)
        return false;
      rewriter.setInsertionPoint(op);
      auto broadcast = rewriter.create<riscv::RVVAxisBroadcastOp>(op->getLoc(), result, *input,
          leaf(rewriter, "rvv", "axis-broadcast", "rvv.axis-broadcast", "rvv.axis-broadcast", 0, 0,
               result.getLayout().getRegisterGroups() + 2 * ((result.getLayout().getLmulEighths() + 7) / 8),
               0, "none", "exact", {inputLanes, outputLanes}));
      copyOrigin(op, broadcast);
      operand.set(broadcast.getResult());
    }
    auto instruction = terminalInstruction(op);
    if (instruction.empty())
      return false;
    llvm::StringRef family = mlir::isa<riscv::WidenOp>(op) ? "widen"
        : mlir::isa<riscv::NarrowOp>(op) ? "narrow"
        : mlir::isa<riscv::CastOp>(op) ? "cast" : "pointwise";
    int64_t temporaries = 0;
    if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(op); binary && riscv::rvvRowScaleRows(binary))
      temporaries = *riscv::rvvRowScaleTemporaryGroups(binary);
    if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(op);
        binary && llvm::StringRef(instruction).ends_with(".broadcast")) {
      auto groups = riscv::binaryBroadcastTemporaryGroups(binary);
      if (!groups) return false;
      temporaries = *groups;
    }
    op->setAttr("leaf", leaf(rewriter, "rvv", family, instruction, instruction, 0, 0, temporaries));
    op->removeAttr("implementation");
  }
  return true;
}

struct IssueWork {
  int64_t arithmetic = 0;
  int64_t movement = 0;
  int64_t total() const { return arithmetic + movement; }
};
using Work = std::map<unsigned, IssueWork, std::greater<unsigned>>;
std::optional<Work> work(riscv::KernelOp kernel) {
  Work result;
  bool complete = true;
  kernel.walk([&](mlir::Operation *op) {
    int64_t count = 0;
    int64_t arithmetic = 0;
    if (pointwise(op)) {
      auto type = mlir::dyn_cast<riscv::ValueType>(op->getResult(0).getType());
      if (!type || type.getLayout().getCarrier() != "rvv")
        return;
      count = staticProduct(type.getLayout().getTimeFactors()).value() *
              staticProduct(type.getLayout().getReplicaFactors()).value() / type.getLayout().getReplicaGroup();
      arithmetic = count;
      count = 0;
      if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(op)) {
        if (auto rows = riscv::rvvRowScaleRows(binary)) {
          const int64_t group = type.getLayout().getReplicaGroup();
          count = arithmetic * (2 * (group - 1) + group * (*rows - 1));
          arithmetic *= group * *rows;
        } else {
          auto broadcasts = riscv::binaryBroadcastInstructionCount(binary);
          if (!broadcasts) { complete = false; return; }
          count += *broadcasts;
        }
      }
    } else if (auto unpack = mlir::dyn_cast<riscv::IMEUnpackOp>(op)) {
      auto handoff = riscv::fragmentRVVHandoffLayout(unpack.getInput().getType(), kernel.getTarget());
      if (!handoff) { complete = false; return; }
      auto from = mlir::cast<riscv::ValueType>(withLayout(unpack.getResult().getType(), *handoff));
      auto slices = riscv::rvvLayoutRepackSlices(from, unpack.getResult().getType());
      if (!slices) { complete = false; return; }
      for (auto slice : *slices)
        count += (slice.sourceOffset != 0) + (slice.resultOffset != 0);
    } else if (auto repack = mlir::dyn_cast<riscv::ConvertLayoutOp>(op);
               repack && repack.getConversion().getKind() == "lane_repack") {
      auto slices = riscv::rvvLayoutRepackSlices(repack.getInput().getType(), repack.getResult().getType());
      if (!slices) { complete = false; return; }
      if (!riscv::isRVVSubregisterRepack(kernel.getTarget(), repack.getInput().getType(), repack.getResult().getType()) &&
          !riscv::isRVVSubregisterRepack(kernel.getTarget(), repack.getResult().getType(), repack.getInput().getType()))
        for (auto slice : *slices)
          count += (slice.sourceOffset != 0) + (slice.resultOffset != 0);
    } else if (auto broadcast = mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(op)) {
      auto instructions = riscv::axisBroadcastInstructionCount(broadcast.getInput().getType(),
                                                               broadcast.getResult().getType());
      if (!instructions) { complete = false; return; }
      count = *instructions;
    } else if (auto zero = mlir::dyn_cast<riscv::IMEZeroOp>(op)) {
      count = riscv::fragmentPartCount(zero.getResult().getType());
    } else if (auto broadcast = mlir::dyn_cast<riscv::IMEFragmentRowBroadcastOp>(op)) {
      if (broadcast.getLeaf().getInstruction() == "ime.rows.byte-repeat") {
        arithmetic = 1;
        count = 1;
      } else {
        count = 2 * riscv::fragmentPartCount(broadcast.getResult().getType()) + 1;
      }
    } else if (auto arithmeticOp = mlir::dyn_cast<riscv::IMEFragmentPointwiseOp>(op)) {
      arithmetic = riscv::fragmentPartCount(arithmeticOp.getResult().getType());
      if (mlir::isa<riscv::ValueType>(arithmeticOp.getRhs().getType()))
        count = 4 * arithmetic - 1;
    } else if (auto birth = mlir::dyn_cast<riscv::NewOp>(op)) {
      auto type = mlir::dyn_cast<riscv::ValueType>(birth.getResult().getType());
      if (type && type.getLayout().getCarrier() == "rvv" && birth.getInitial() &&
          mlir::isa<mlir::FloatType, mlir::IntegerType>(birth.getInitial().getType()))
        count = *riscv::rvvCarrierCount(type.getLayout());
    } else if (auto splat = mlir::dyn_cast<riscv::RVVSplatOp>(op)) {
      auto parts = riscv::rvvCarrierCount(splat.getResult().getType().getLayout());
      if (!parts) { complete = false; return; }
      count = *parts;
    }
    unsigned depth = 0;
    for (auto *parent = op->getParentOp(); parent && parent != kernel.getOperation(); parent = parent->getParentOp())
      depth += mlir::isa<mlir::scf::ForOp>(parent);
    result[depth].arithmetic += arithmetic;
    result[depth].movement += count;
  });
  return complete ? std::optional<Work>(result) : std::nullopt;
}

bool lessWork(const Work &candidate, const Work &baseline) {
  Work levels = baseline;
  for (auto [depth, count] : candidate)
    levels.try_emplace(depth, IssueWork{});
  for (auto [depth, unused] : levels) {
    auto a = candidate.find(depth), b = baseline.find(depth);
    const int64_t av = a == candidate.end() ? 0 : a->second.total();
    const int64_t bv = b == baseline.end() ? 0 : b->second.total();
    if (av != bv)
      return av < bv;
  }
  return false;
}

std::optional<riscv::LayoutAttr> completeRowHandoffLayout(
    riscv::IMEUnpackOp anchor, riscv::TargetAttr target) {
  auto fragment = anchor.getInput().getType();
  auto base = riscv::fragmentRVVHandoffLayout(fragment, target);
  if (!base || fragment.getShape()[0] != 1 ||
      fragment.getPacking().getRowsPerTile() != 1 || riscv::fragmentPartCount(fragment) > 4)
    return std::nullopt;
  const int64_t columns = fragment.getShape()[1];
  const int64_t sew = base->getSew();
  const int64_t limit = riscv::fragmentPartResourceGroups(fragment) * 8;
  const int64_t required = (columns * sew * 8 + target.getVlenBits() - 1) / target.getVlenBits();
  auto one = mlir::DenseI64ArrayAttr::get(fragment.getContext(), {1, 1});
  auto lanes = mlir::DenseI64ArrayAttr::get(fragment.getContext(), {1, columns});
  for (int64_t lmul : target.getLegalLMULEighths().asArrayRef()) {
    if (lmul < required || lmul > limit)
      continue;
    auto layout = riscv::LayoutAttr::get(fragment.getContext(), "rvv", fragment.getAxisIds(),
        one, lanes, one, one, one, sew, lmul, columns, (lmul + 7) / 8, base->getValidity());
    if (riscv::supportsRVVLayout(target, layout))
      return layout;
  }
  return std::nullopt;
}

void materializeRegisterRowHandoff(riscv::IMEUnpackOp unpack,
                                   riscv::TargetAttr target,
                                   mlir::IRRewriter &rewriter) {
  auto result = unpack.getResult().getType();
  auto output = result.getLayout();
  auto natural = riscv::fragmentRVVHandoffLayout(unpack.getInput().getType(), target);
  if (!natural || result.getShape().size() != 2 || result.getShape()[0] != 1 ||
      natural->getLmulEighths() >= 8 || output.getLmulEighths() <= 8 ||
      natural->getValidity() != "full" || output.getValidity() != "full" ||
      natural->getReplicaGroup() != 1 || output.getReplicaGroup() != 1)
    return;
  const int64_t factor = 8 / natural->getLmulEighths();
  auto lanes = llvm::to_vector(natural->getLaneFactors().asArrayRef());
  auto replicas = llvm::to_vector(natural->getReplicaFactors().asArrayRef());
  if (factor < 2 || replicas[0] != 1 || replicas[1] % factor ||
      natural->getTimeFactors().asArrayRef() != output.getTimeFactors().asArrayRef())
    return;
  lanes[1] *= factor;
  replicas[1] /= factor;
  auto middle = riscv::LayoutAttr::get(rewriter.getContext(), "rvv", result.getAxisIds(),
      natural->getTimeFactors(), integers(rewriter, lanes), integers(rewriter, replicas),
      natural->getFragmentFactors(), natural->getLocalFactors(), natural->getSew(),
      8, natural->getVl() * factor, replicas[1], "full");
  auto sourceType = mlir::cast<riscv::ValueType>(withLayout(result, *natural));
  auto middleType = mlir::cast<riscv::ValueType>(withLayout(result, middle));
  if (!riscv::supportsRVVLayout(target, middle) ||
      !riscv::rvvLayoutRepackSlices(sourceType, middleType) ||
      !riscv::isRVVSubregisterRepack(target, result, middleType))
    return;
  // Form full physical registers with independent small slides, then join
  // those registers. A serial slide into the full row moves more active lanes
  // and makes all partial handoffs depend on the preceding one.
  unpack.getResult().setType(sourceType);
  const int64_t temporary = riscv::fragmentUnpackTemporaryGroups(unpack.getInput().getType(), sourceType);
  unpack.setConversionAttr(conversion(rewriter, "fragment_to_rvv", "handoff", temporary));
  unpack.setLeafAttr(leaf(rewriter, "ime", "fragment-unpack", "ime.unpack.rvv", "ime.unpack.rvv", 0, 0, temporary));
  rewriter.setInsertionPointAfter(unpack);
  auto pack = [&](mlir::Value value, riscv::ValueType type) {
    auto edge = rewriter.create<riscv::ConvertLayoutOp>(unpack.getLoc(), type, value,
        conversion(rewriter, "lane_repack", "pure",
            riscv::rvvLayoutRepackTemporaryGroups(target, mlir::cast<riscv::ValueType>(value.getType()), type)),
        riscv::AccessAttr(), unselectedLeaf(rewriter));
    copyOrigin(unpack, edge);
    return edge;
  };
  auto partial = pack(unpack.getResult(), middleType);
  auto complete = pack(partial.getResult(), result);
  unpack.getResult().replaceAllUsesExcept(complete.getResult(), partial);
}
}

static mlir::LogicalResult propagateConsumerLayouts(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::KernelOp> kernels(module.getOps<riscv::KernelOp>());
  for (auto kernel : kernels) {
    if (kernel.getResourcesMaterialized())
      continue;
    bool hasAnchor = false;
    kernel.walk([&](riscv::IMEUnpackOp) { hasAnchor = true; });
    if (!hasAnchor && kernel.getLocalTileStrategy() != "target-default") {
      rejectRISCVCandidate(kernel, RISCVFailureCategory::Unsupported,
                           "explicit local tile strategy has no applicable contraction");
      return mlir::failure();
    }
    if (!hasAnchor)
      continue;
    auto baseline = work(kernel);
    if (!baseline) {
      llvm::errs() << "weft-consumer-layout: rejected=unavailable-local-work-estimate\n";
      if (kernel.getLocalTileStrategy() != "target-default") {
        kernel.emitOpError("explicit local tile strategy has no bounded local work estimate");
        return mlir::failure();
      }
      continue;
    }
    mlir::OwningOpRef<riscv::KernelOp> trial(mlir::cast<riscv::KernelOp>(kernel->clone()));
    llvm::SmallVector<riscv::IMEUnpackOp> anchors;
    trial->walk([&](riscv::IMEUnpackOp unpack) { anchors.push_back(unpack); });
    bool changed = false, legal = true;
    bool grouped = false;
    bool completeRow = false;
    for (auto anchor : anchors) {
      auto layout = kernel.getLocalTileStrategy() != "atom-partitioned"
          ? riscv::fragmentRVVHandoffLayout(anchor.getInput().getType(), kernel.getTarget(), 2)
          : std::nullopt;
      grouped |= bool(layout);
      if (!layout && kernel.getLocalTileStrategy() == "register-grouped") {
        rejectRISCVCandidate(kernel, RISCVFailureCategory::Unsupported,
                             "register-grouped strategy is unsupported for the selected tile geometry");
        return mlir::failure();
      }
      if (!layout) {
        layout = completeRowHandoffLayout(anchor, kernel.getTarget());
        completeRow |= bool(layout);
      }
      if (riscv::isGroupedFragmentRVVHandoff(anchor.getInput().getType(), anchor.getResult().getType()) ||
          (!layout && riscv::isDirectFragmentRVVHandoff(anchor.getInput().getType(), anchor.getResult().getType())))
        continue;
      if (!layout)
        layout = riscv::fragmentRVVHandoffLayout(anchor.getInput().getType(), kernel.getTarget());
      if (!layout)
        continue;
      if (!rewriteComponent(anchor, *layout, rewriter)) { legal = false; break; }
      if (completeRow) materializeRegisterRowHandoff(anchor, kernel.getTarget(), rewriter);
      changed = true;
    }
    if ((grouped || completeRow) && legal) {
      materializeFragmentRowSupply(*trial, rewriter);
      trial->walk([&](riscv::IMEZeroOp zero) {
        if (riscv::fragmentRVVHandoffLayout(zero.getResult().getType(), kernel.getTarget(), 2))
          zero.setLeafAttr(leaf(rewriter, "ime", "fragment-init", "ime.zero.local", "ime.zero.local",
              0, 0));
        else if (completeRow) {
          auto type = zero.getResult().getType();
          const int64_t activeBits = type.getPacking().getRowsPerTile() *
              type.getPacking().getColumnsPerTile() * logicalBitWidth(type.getElementType());
          const int64_t lmul = activeBits * 8 / kernel.getTarget().getVlenBits();
          if (lmul > 0 && activeBits * 8 == lmul * kernel.getTarget().getVlenBits() &&
              lmul < riscv::fragmentPartResourceGroups(type) * 8 &&
              llvm::is_contained(kernel.getTarget().getLegalLMULEighths().asArrayRef(), lmul))
            zero.setLeafAttr(leaf(rewriter, "ime", "fragment-init", "ime.zero.local-prefix",
                "ime.zero.local-prefix", 0, 0, 0, 0, "none", "exact", {lmul}));
        }
      });
    }
    if (!legal && grouped) {
      kernel.emitOpError("selected register-grouped realization has an unsupported coordinate edge");
      return mlir::failure();
    }
    if (!changed)
      continue;
    rewriter.setInsertionPoint(kernel);
    rewriter.insert(trial->getOperation());
    std::string rejected;
    if (legal) {
      mlir::ScopedDiagnosticHandler diagnostics(module.getContext(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(rejected);
        diagnostic.print(stream);
        return mlir::success();
      });
      legal = mlir::succeeded(mlir::verify(trial->getOperation()));
    }
    selectFragmentRegisterReuse(*trial);
    auto pressure = legal ? physicalRegisterPressure(*trial) : std::nullopt;
    const bool resourceLegal = pressure && pressure->total() <= kernel.getTarget().getVectorRegisters();
    auto candidate = legal ? work(*trial) : std::nullopt;
    const bool selected = resourceLegal && candidate && (grouped || lessWork(*candidate, *baseline));
    llvm::errs() << "weft-consumer-layout: candidates=2 requested=" << kernel.getLocalTileStrategy() << " selected="
                 << (selected ? (grouped ? "register-grouped" : completeRow ? "complete-row" : "fragment-order") : "existing")
                 << " reason=" << (!legal ? "unsupported-coordinate-edge" : !resourceLegal ? "resident-budget" : !candidate ? "unknown-local-work" : grouped ? "target-register-group-rule" : selected ? "less-inner-issue-work" : "no-issue-work-reduction")
                 << " resident-groups=" << (pressure ? pressure->total() : -1)
                 << " scope=pointwise+broadcast+conversion+initializer estimate=pre-system-compiler cycle-estimate=unavailable work-by-loop-depth(alu,movement)=";
    for (auto [depth, count] : *baseline) {
      auto after = candidate ? candidate->find(depth) : Work::const_iterator();
      llvm::errs() << depth << ":(" << count.arithmetic << ',' << count.movement << ")->";
      if (candidate && after != candidate->end())
        llvm::errs() << '(' << after->second.arithmetic << ',' << after->second.movement << ')';
      else llvm::errs() << "unknown";
      llvm::errs() << ' ';
    }
    llvm::errs() << '\n';
    if (pressure && pressure->operation)
      llvm::errs() << "weft-consumer-layout: peak-entity=" << pressure->operation->getName()
                   << " side=" << (pressure->before ? "before" : "after")
                   << " vector=" << pressure->vector << " fragment=" << pressure->fragment << '\n';
    if (!rejected.empty()) llvm::errs() << "weft-consumer-layout: rejected=" << rejected << '\n';
    if (grouped && !selected) {
      kernel.emitOpError("selected register-grouped realization has no legal closed physical program");
      return mlir::failure();
    }
    if (!selected)
      continue;
    auto replacement = trial.release();
    rewriter.eraseOp(kernel);
  }
  return mlir::success();
}

namespace {
class PropagateRISCVConsumerLayoutsPass
    : public mlir::PassWrapper<PropagateRISCVConsumerLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override { return "weft-riscv-propagate-consumer-layouts"; }
  llvm::StringRef getDescription() const override {
    return "Propagate bounded producer representations through ordinary consumers and control carries";
  }
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    if (mlir::failed(propagateConsumerLayouts(getOperation(), rewriter)))
      signalPassFailure();
  }
};
}

std::unique_ptr<mlir::Pass> weft::createPropagateRISCVConsumerLayoutsPass() {
  return std::make_unique<PropagateRISCVConsumerLayoutsPass>();
}
