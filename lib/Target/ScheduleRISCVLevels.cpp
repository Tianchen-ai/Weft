#include "Weft/Target/RISCVPasses.h"
#include "PipelineSchedule.h"
#include "PipelineShapeBindings.h"
#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

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
} // namespace

std::unique_ptr<Pass> weft::createScheduleRISCVLevelsPass() {
  return std::make_unique<ScheduleRISCVLevelsPass>();
}
