#include "Weft/Target/RISCVPasses.h"
#include "PipelineSchedule.h"
#include "PipelineShapeBindings.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
void copyLoopAttributes(scf::ForOp source, Operation *target, bool levelIdentity) {
  for (NamedAttribute attribute : source->getAttrs()) {
    auto name = attribute.getName();
    if (name == "weft.riscv.schedule" || name == "weft.riscv.pipeline_peak") continue;
    if (!levelIdentity && name == "weft.riscv.level") continue;
    if (!isa<scf::ForOp>(target) &&
        (name == "weft.riscv.unroll_factor" || name == "weft.riscv.unroll_order")) continue;
    target->setAttr(name, attribute.getValue());
  }
  if (!levelIdentity && !target->hasAttr("weft.riscv.direction"))
    if (auto level = source->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      target->setAttr("weft.riscv.direction", StringAttr::get(target->getContext(), level.getDirection()));
}

// Each stage instance has its own index, numerical SSA and shape declaration
// map. Sharing one mapping would bind the consumer to a future load or to the
// next iteration's descriptor extent when stages are interleaved.
struct StageInstance {
  IRMapping values;
  llvm::DenseMap<Value, int64_t> shapes;

  Operation *clone(IRRewriter &b, Operation *operation, int64_t &identity) {
    return clonePipelineOperation(b, operation, values, shapes, identity);
  }
};
} // namespace

FailureOr<PipelineExpansion> riscv_internal::expandPipeline(
    const PipelineSchedule &schedule, IRRewriter &b, int64_t &shapeIdentity) {
  if (failed(schedule.verify()) || failed(verifyPipelineShapeStages(schedule.loop))) return failure();
  auto loop = schedule.loop;
  Location loc = loop.getLoc();
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(loop);
  auto predicate = schedule.descending ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ult;
  Value nonEmpty = arith::CmpIOp::create(b, loc, predicate, loop.getLowerBound(), loop.getUpperBound());
  auto guard = scf::IfOp::create(b, loc, loop.getResultTypes(), nonEmpty, true);
  copyLoopAttributes(loop, guard, true);

  b.setInsertionPointToStart(&guard.getThenRegion().front());
  StageInstance prologue;
  prologue.values.map(loop.getInductionVar(), loop.getLowerBound());
  for (auto [argument, initial] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
    prologue.values.map(argument, initial);
  for (Operation *operation : schedule.producers) prologue.clone(b, operation, shapeIdentity);

  SmallVector<Value> initial(loop.getInitArgs());
  if (schedule.transportInduction) initial.push_back(loop.getLowerBound());
  for (Value value : schedule.transported) initial.push_back(prologue.values.lookup(value));
  Value first = arith::AddIOp::create(b, loc, loop.getLowerBound(), loop.getStep());
  auto steady = scf::ForOp::create(b, loc, first, loop.getUpperBound(), loop.getStep(), initial);
  copyLoopAttributes(loop, steady, false);
  if (!steady.getBody()->empty()) b.eraseOp(steady.getBody()->getTerminator());
  b.setInsertionPointToStart(steady.getBody());

  const unsigned carryCount = loop.getInitArgs().size();
  const unsigned transportStart = carryCount + unsigned(schedule.transportInduction);
  StageInstance future, current;
  auto bindInvariants = [&](StageInstance &instance) {
    for (Operation *operation : schedule.invariants) {
      for (Value value : operation->getResults()) instance.values.map(value, prologue.values.lookup(value));
      operation->walk([&](riscv::LocalShapeOp shape) {
        auto version = prologue.shapes.find(shape.getResult());
        if (version != prologue.shapes.end()) instance.shapes[shape.getResult()] = version->second;
      });
    }
  };
  bindInvariants(future);
  bindInvariants(current);
  future.values.map(loop.getInductionVar(), steady.getInductionVar());
  if (schedule.transportInduction)
    current.values.map(loop.getInductionVar(), steady.getRegionIterArg(carryCount));
  for (auto [argument, value] : llvm::zip(loop.getRegionIterArgs(), steady.getRegionIterArgs().take_front(carryCount)))
    current.values.map(argument, value);
  for (auto [position, value] : llvm::enumerate(schedule.transported))
    current.values.map(value, steady.getRegionIterArg(transportStart + position));

  PipelineExpansion result;
  result.guard = guard;
  result.steady = steady;
  for (Operation *operation : schedule.ordered) {
    if (schedule.isInvariant(operation)) continue;
    StageInstance &instance = schedule.isProducer(operation) ? future : current;
    result.steadyOperations[operation] = instance.clone(b, operation, shapeIdentity);
  }
  SmallVector<Value> returned;
  for (Value value : yield.getOperands()) returned.push_back(current.values.lookupOrDefault(value));
  if (schedule.transportInduction) returned.push_back(steady.getInductionVar());
  for (Value value : schedule.transported) returned.push_back(future.values.lookup(value));
  scf::YieldOp::create(b, loc, returned);

  b.setInsertionPointAfter(steady);
  StageInstance epilogue;
  bindInvariants(epilogue);
  if (schedule.transportInduction)
    epilogue.values.map(loop.getInductionVar(), steady.getResult(carryCount));
  for (auto [argument, value] : llvm::zip(loop.getRegionIterArgs(), steady.getResults().take_front(carryCount)))
    epilogue.values.map(argument, value);
  for (auto [position, value] : llvm::enumerate(schedule.transported))
    epilogue.values.map(value, steady.getResult(transportStart + position));
  for (Operation *operation : schedule.consumers) epilogue.clone(b, operation, shapeIdentity);
  SmallVector<Value> finalValues;
  for (Value value : yield.getOperands()) finalValues.push_back(epilogue.values.lookupOrDefault(value));
  scf::YieldOp::create(b, loc, finalValues);
  b.setInsertionPointToStart(&guard.getElseRegion().front());
  scf::YieldOp::create(b, loc, loop.getInitArgs());
  // The caller owns replacement. Keeping the source alive here lets the
  // scheduler inspect a disposable expansion without dangling stage identities.
  return result;
}

namespace {
class PipelineRISCVLevelsPass
    : public PassWrapper<PipelineRISCVLevelsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineRISCVLevelsPass)
  StringRef getArgument() const override { return "weft-riscv-pipeline-levels"; }
  StringRef getDescription() const override {
    return "Expand a verified physical stage/order into versioned prologue, steady state and epilogue";
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    int64_t identity = nextPipelineShapeIdentity(getOperation());
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.schedule")) loops.push_back(loop);
    });
    for (auto loop : loops) {
      auto binding = loop->getAttrOfType<riscv::ScheduleAttr>("weft.riscv.schedule");
      if (binding.getPipelineDepth() == 1) {
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      auto schedule = PipelineSchedule::read(loop);
      if (failed(schedule)) { signalPassFailure(); return; }
      auto expanded = expandPipeline(*schedule, b, identity);
      if (failed(expanded)) { signalPassFailure(); return; }
      b.replaceOp(loop, expanded->guard.getResults());
    }
    bool remaining = false;
    getOperation().walk([&](Operation *operation) {
      if (operation->hasAttr("weft.riscv.schedule") || operation->hasAttr(pipelineStage) ||
          operation->hasAttr(pipelineOrder) || operation->hasAttr(pipelineInvariant) || operation->hasAttr("schedule")) {
        operation->emitError("physical schedule survived without structural pipeline expansion");
        remaining = true;
      }
    });
    if (remaining) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createPipelineRISCVLevelsPass() {
  return std::make_unique<PipelineRISCVLevelsPass>();
}
