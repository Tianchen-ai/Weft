#include "Weft/Target/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "PipelineSchedule.h"
#include "PipelineShapeBindings.h"
#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
// Evaluate the actual expanded SSA in a disposable kernel. The selected result
// remains stage/order on the original loop; only the expander installs a new
// program. This prevents scheduler and emitter from owning different pipelines.
LogicalResult selectFrontiers(PipelineSchedule &schedule, IRRewriter &b,
                             unsigned &interleaved, int64_t &largestReduction) {
  schedule.write();
  if (failed(verifyPipelineShapeStages(schedule.loop))) return failure();
  auto kernel = schedule.loop->getParentOfType<riscv::KernelOp>();
  IRMapping mapping;
  OwningOpRef<riscv::KernelOp> trial(cast<riscv::KernelOp>(kernel->clone(mapping)));
  auto loop = cast<scf::ForOp>(mapping.lookup(schedule.loop.getInductionVar()).getParentBlock()->getParentOp());
  auto copy = PipelineSchedule::read(loop);
  if (failed(copy)) return failure();
  int64_t identity = nextPipelineShapeIdentity(trial->getOperation());
  auto expanded = expandPipeline(*copy, b, identity);
  if (failed(expanded)) return failure();
  auto before = registerPeak(expanded->guard.getThenRegion().front());
  if (!before) return schedule.loop.emitError("pipeline has no bounded expanded region residency");
  SmallVector<Value> initial(schedule.loop.getInitArgs());
  auto outside = surroundingRegisterGroups(schedule.loop, initial);
  if (!outside) return schedule.loop.emitError("pipeline has no bounded surrounding residency");
  // The surrounding values cannot be reclaimed by changing this issue order.
  // If they already consume the target budget, the resource owner still has to
  // realize any required spill; the scheduler uses the smallest positive local
  // budget to retire current operands before adding another version.
  int64_t available = std::max<int64_t>(1, kernel.getTarget().getVectorRegisters() - *outside);
  auto selected = orderPipelineFrontiers(*copy, *expanded, available);
  if (failed(selected)) return failure();
  auto after = registerPeak(expanded->guard.getThenRegion().front());
  if (!after) return schedule.loop.emitError("interleaved pipeline has no bounded expanded residency");
  // Both candidates have identical reads, arithmetic, stages and carriers. If
  // pressure is unchanged, keep maximal lookahead; there is no claimed latency
  // model that could justify a different tie-break.
  if (after->total() >= before->total()) return success();
  SmallVector<Operation *> order;
  for (Operation *operation : *selected) {
    auto originalPosition = operation->getAttrOfType<IntegerAttr>(pipelineOrder).getInt();
    order.push_back(schedule.ordered[originalPosition]);
  }
  schedule.ordered = std::move(order);
  if (failed(schedule.verify())) return failure();
  schedule.write();
  ++interleaved;
  largestReduction = std::max(largestReduction, before->total() - after->total());
  return success();
}

class ScheduleRISCVLevelsPass
    : public PassWrapper<ScheduleRISCVLevelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleRISCVLevelsPass)
  ScheduleRISCVLevelsPass() = default;
  ScheduleRISCVLevelsPass(const ScheduleRISCVLevelsPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "weft-riscv-schedule-levels"; }
  StringRef getDescription() const override {
    return "Schedule dependency-derived physical stages with bounded SSA residency";
  }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.schedule")) loops.push_back(loop);
    });
    IRRewriter b(&getContext());
    unsigned selected = 0;
    int64_t reduction = 0;
    for (auto loop : loops) {
      auto binding = loop->getAttrOfType<riscv::ScheduleAttr>("weft.riscv.schedule");
      if (binding.getPipelineDepth() == 1) continue;
      bool nested = false;
      loop.getBody()->walk([&](scf::ForOp child) {
        nested |= child != loop && child->hasAttr("weft.riscv.schedule");
      });
      if (nested) {
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      auto candidate = schedulePipeline(loop);
      if (failed(candidate)) { signalPassFailure(); return; }
      if (!*candidate) {
        // No independent supply is a fact of this graph. Consume the optional
        // local schedule without manufacturing a producer or copying a read.
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      if (failed(selectFrontiers(**candidate, b, selected, reduction))) {
        signalPassFailure();
        return;
      }
    }
    interleavedLoops += selected;
    maximumReduction = std::max<int64_t>(maximumReduction, reduction);
    if (selected)
      llvm::errs() << "weft-pipeline-frontier loops=" << selected
                   << " largest-region-register-reduction=" << reduction << '\n';
  }
private:
  Statistic interleavedLoops{this, "interleaved-loops", "Pipelines with a smaller expanded region register peak"};
  Statistic maximumReduction{this, "maximum-register-reduction", "Largest reduction in expanded region register groups"};
};

// Move an independent epilogue supply ahead of the contraction it will join.
// This is same-block scheduling: no iteration, Level or numerical boundary is
// changed. Keep the selected layouts/leaves and check their complete live set.
class ScheduleRISCVLocalSuppliesPass
    : public PassWrapper<ScheduleRISCVLocalSuppliesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleRISCVLocalSuppliesPass)
private:
  static bool collectRead(Value value, Operation *anchor,
                          llvm::SetVector<Operation *> &operations) {
    auto *op = value.getDefiningOp();
    if (!op || op->getBlock() != anchor->getBlock() || op->isBeforeInBlock(anchor))
      return true;
    if (operations.contains(op)) return true;
    if (operations.size() >= 24 || op == anchor || op->getNumRegions() ||
        op->hasAttr("birth_id") ||
        !(isa<riscv::IMEFragmentStorageLoadOp, riscv::PhysicalPointOp,
              riscv::FieldOp, riscv::ExtractOp, riscv::CastOp, riscv::WidenOp>(op) ||
          op->getName().getDialectNamespace() == "arith") ||
        !riscv::hasOnlyReadEffects(op))
      return false;
    for (Value operand : op->getOperands())
      if (!collectRead(operand, anchor, operations)) return false;
    operations.insert(op);
    return true;
  }

  static unsigned scheduleReads(ModuleOp module) {
    SmallVector<riscv::IMEFragmentStorageLoadOp> reads;
    module.walk([&](riscv::IMEFragmentStorageLoadOp op) { reads.push_back(op); });
    unsigned selected = 0;
    for (auto read : reads) {
      auto kernel = read->getParentOfType<riscv::KernelOp>();
      if (!kernel || kernel.getResourcesMaterialized() ||
          riscv::fragmentPartCount(read.getResult().getType()) < 2)
        continue;
      riscv::IMEFragmentMMAOp anchor;
      for (Operation *op = read->getPrevNode(); op; op = op->getPrevNode()) {
        if (auto mma = dyn_cast<riscv::IMEFragmentMMAOp>(op)) {
          if (!mma.getMemoryClobber()) anchor = mma;
          break;
        }
        if (!riscv::hasOnlyReadEffects(op)) break;
      }
      if (!anchor) continue;
      llvm::SetVector<Operation *> closure;
      if (!collectRead(read.getResult(), anchor, closure)) continue;
      SmallVector<std::pair<Operation *, Operation *>> positions;
      for (Operation *op : closure) {
        positions.emplace_back(op, op->getNextNode());
        op->moveBefore(anchor);
      }
      auto pressure = physicalRegisterPressure(kernel);
      if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters()) {
        for (auto [op, next] : llvm::reverse(positions)) op->moveBefore(next);
        continue;
      }
      ++selected;
    }
    return selected;
  }

  static riscv::IMEUnpackOp fragmentResult(Value value) {
    for (unsigned depth = 0; depth < 8; ++depth) {
      auto *op = value.getDefiningOp();
      if (auto unpack = dyn_cast_or_null<riscv::IMEUnpackOp>(op)) return unpack;
      if (!isa_and_nonnull<riscv::WidenOp, riscv::CastOp, riscv::ConvertLayoutOp>(op)) return {};
      value = op->getOperand(0);
    }
    return {};
  }

  static bool collect(Value value, Operation *anchor, llvm::SetVector<Operation *> &operations) {
    auto *op = value.getDefiningOp();
    if (!op || op->getBlock() != anchor->getBlock() || op->isBeforeInBlock(anchor)) return true;
    if (operations.contains(op)) return true;
    if (operations.size() >= 24 || op == anchor || op->getNumRegions() ||
        !isa<riscv::FieldOp, riscv::FieldReadOp, riscv::WidenOp, riscv::CastOp,
             riscv::BinaryOp, riscv::ConvertLayoutOp>(op) || !riscv::hasOnlyReadEffects(op)) return false;
    if (auto binary = dyn_cast<riscv::BinaryOp>(op); binary && binary.getKind() != "mul") return false;
    for (Value operand : op->getOperands())
      if (!collect(operand, anchor, operations)) return false;
    operations.insert(op);
    return true;
  }

public:
  StringRef getArgument() const override { return "weft-riscv-schedule-local-supplies"; }
  StringRef getDescription() const override {
    return "Overlap independent same-block numeric supplies with selected contractions";
  }
  void runOnOperation() override {
    SmallVector<riscv::BinaryOp> consumers;
    getOperation().walk([&](riscv::BinaryOp op) {
      if (op.getKind() == "mul") consumers.push_back(op);
    });
    unsigned selected = 0;
    for (auto consumer : consumers) {
      auto kernel = consumer->getParentOfType<riscv::KernelOp>();
      if (!kernel || kernel.getResourcesMaterialized()) continue;
      auto unpack = fragmentResult(consumer.getLhs());
      Value supply = consumer.getRhs();
      if (!unpack) { unpack = fragmentResult(consumer.getRhs()); supply = consumer.getLhs(); }
      if (!unpack || unpack->getBlock() != consumer->getBlock()) continue;
      auto first = unpack.getInput().getDefiningOp<riscv::IMEFragmentMMAOp>();
      if (!first || first->getBlock() != consumer->getBlock()) continue;
      while (auto prior = first.getAccumulator().getDefiningOp<riscv::IMEFragmentMMAOp>()) {
        if (prior->getBlock() != consumer->getBlock()) break;
        first = prior;
      }
      llvm::SetVector<Operation *> closure;
      if (!collect(supply, first, closure) || closure.empty()) continue;
      bool legal = true;
      for (Operation *op = first; op != consumer; op = op->getNextNode()) {
        if (auto mma = dyn_cast<riscv::IMEFragmentMMAOp>(op)) legal &= !mma.getMemoryClobber();
        else legal &= riscv::hasOnlyReadEffects(op);
      }
      if (!legal) continue;
      SmallVector<std::pair<Operation *, Operation *>> positions;
      for (Operation *op : closure) {
        positions.emplace_back(op, op->getNextNode());
        op->moveBefore(first);
      }
      auto pressure = physicalRegisterPressure(kernel);
      if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters()) {
        for (auto [op, next] : llvm::reverse(positions)) op->moveBefore(next);
        continue;
      }
      ++selected;
    }
    const auto reads = scheduleReads(getOperation());
    if (selected || reads)
      llvm::errs() << "weft-local-supply-schedule: selected=" << selected
                   << " reads=" << reads << '\n';
  }
};
} // namespace

std::unique_ptr<Pass> weft::createScheduleRISCVLevelsPass() {
  return std::make_unique<ScheduleRISCVLevelsPass>();
}

std::unique_ptr<Pass> weft::createScheduleRISCVLocalSuppliesPass() {
  return std::make_unique<ScheduleRISCVLocalSuppliesPass>();
}
