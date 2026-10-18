#include "PipelineSchedule.h"
#include "PipelineShapeBindings.h"
#include "PhysicalLoopControl.h"
#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Target/RISCVCompilation.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
constexpr unsigned maximumOperations = 1024;
constexpr unsigned maximumEdges = 8192;

bool definedInside(Operation *container, Value value) {
  Operation *owner = value.getDefiningOp();
  if (!owner) owner = cast<BlockArgument>(value).getOwner()->getParentOp();
  return owner && (owner == container || container->isProperAncestor(owner));
}

SmallVector<Value> externalOperands(Operation *operation) {
  SmallVector<Value> result;
  llvm::DenseSet<Value> seen;
  operation->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      if (!definedInside(operation, operand) && seen.insert(operand).second)
        result.push_back(operand);
  });
  return result;
}

FailureOr<SmallVector<Value>> stageDependencies(Operation *operation) {
  auto result = externalOperands(operation);
  llvm::DenseSet<Value> seen(result.begin(), result.end());
  auto walked = operation->walk([&](Operation *nested) -> WalkResult {
    auto shapes = physicalShapeDependencies(nested);
    if (failed(shapes)) return WalkResult::interrupt();
    for (Value shape : *shapes)
      if (!definedInside(operation, shape) && seen.insert(shape).second) result.push_back(shape);
    return WalkResult::advance();
  });
  if (walked.wasInterrupted()) return failure();
  return result;
}

InFlightDiagnostic unsupportedPipeline(Operation *operation, bool admission,
                                       const llvm::Twine &message) {
  if (admission)
    return rejectRISCVCandidate(operation, RISCVFailureCategory::Unsupported, message);
  return operation->emitError(message);
}

LogicalResult readonlyUnit(Operation *operation, bool admission = false) {
  unsigned visited = 0;
  auto walked = operation->walk<WalkOrder::PreOrder>([&](Operation *nested) -> WalkResult {
    if (++visited > maximumEdges) {
      nested->emitError("pipeline stage exceeds its bounded physical operation domain");
      return WalkResult::interrupt();
    }
    if (isa<scf::YieldOp>(nested)) return WalkResult::advance();
    if (nested->getNumRegions()) {
      if (!isa<scf::ForOp, scf::IfOp>(nested)) {
        unsupportedPipeline(nested, admission,
                            "pipeline stage supports only closed read-only for/if regions");
        return WalkResult::interrupt();
      }
      if (nested->hasAttr("weft.riscv.schedule")) {
        nested->emitError("a pipeline stage has an unconsumed nested schedule");
        return WalkResult::interrupt();
      }
      if (nested->hasAttr("weft.riscv.level")) {
        unsupportedPipeline(nested, admission,
                            "a pipeline stage cannot clone a nested author Level");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    auto materialized = dyn_cast<riscv::RegisterMaterializeOp>(nested);
    bool physicalShare = materialized && materialized.getRealization() == "physical-share";
    if (!physicalShare && (nested->hasAttr("birth_id") || isa<riscv::NewOp, riscv::EmptyOp,
        riscv::RegisterMaterializeOp, riscv::MaterializeOp, riscv::StagedViewOp,
        riscv::LocalAllocOp, riscv::LocalBindOp>(nested))) {
      unsupportedPipeline(nested, admission,
                          "pipeline stage cannot duplicate a numerical or storage birth: ") << nested->getName();
      return WalkResult::interrupt();
    }
    if (!isPipelineShapeBinding(nested) && !riscv::hasOnlyReadEffects(nested)) {
      // A copied register-fragment accumulator writes only its new SSA result.
      // It has no external memory effect; stage dependencies preserve the
      // accumulator chain and later resource closure decides physical reuse.
      if (auto mma = dyn_cast<riscv::IMEFragmentMMAOp>(nested);
          mma && !mma.getMemoryClobber() && mma.getAccumulatorPolicy() == "copy")
        return WalkResult::advance();
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        if (llvm::any_of(instances, [](const auto &effect) {
              return isa<MemoryEffects::Write, MemoryEffects::Allocate,
                         MemoryEffects::Free>(effect.getEffect());
            })) {
          unsupportedPipeline(nested, admission,
                              "pipeline stage has a write or allocation effect without a reordering contract");
          return WalkResult::interrupt();
        }
      }
      nested->emitError("pipeline stage has a write or unknown effect without a reordering contract");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

// These operations bind coordinates and views. Moving their construction to
// the consuming stage does not replay a numeric read or a source birth.
bool replayableDescriptor(Operation *operation) {
  return isa<riscv::FieldOp, riscv::SliceOp, riscv::SubviewOp,
             riscv::ReaxisOp, riscv::LinearViewOp, riscv::ExtentOp>(operation);
}

bool contractionConsumer(Operation *operation) {
  if (!isa<riscv::RVVWidenMultiplyOp, riscv::RVVWidenDotOp,
           riscv::IMEFragmentMMAOp>(operation))
    return false;
  // Indexed storage projections still require their address/shape owner in
  // the consuming iteration. Only already-numeric or direct field supplies
  // establish this earlier frontier.
  for (Value operand : operation->getOperands()) {
    SmallVector<Value> indices;
    if (riscv::fieldReadProjection(operand, &indices) && !indices.empty())
      return false;
  }
  return true;
}

LogicalResult materializeContractionReads(scf::ForOp loop) {
  IRRewriter rewriter(loop.getContext());
  llvm::DenseMap<Value, Value> reads;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (!contractionConsumer(&operation)) continue;
    for (OpOperand &operand : operation.getOpOperands()) {
      Value input = operand.get();
      SmallVector<Value> indices;
      auto field = riscv::fieldReadProjection(input, &indices);
      auto type = dyn_cast<riscv::ValueType>(input.getType());
      if (!field || !type) continue;
      if (auto found = reads.find(input); found != reads.end()) {
        operand.set(found->second);
        continue;
      }
      auto access = input.getDefiningOp()->getAttrOfType<riscv::AccessAttr>("access");
      if (!access)
        return operation.emitError("contraction pipeline supply has no memory access");
      auto temporary = riscv::fieldReadTemporaryGroups(input, type, access, indices);
      if (!temporary)
        return operation.emitError("contraction pipeline supply has no closed numeric read");
      rewriter.setInsertionPoint(&operation);
      auto instruction = riscv::fieldReadInstruction(input, type, access);
      auto read = riscv::FieldReadOp::create(rewriter, operation.getLoc(), type,
          input, indices, access,
          leaf(rewriter, "transfer", "field-read", instruction, instruction,
               0, 0, *temporary, 0, "none", "exact"));
      copyOrigin(input.getDefiningOp(), read);
      reads[input] = read.getResult();
      operand.set(read.getResult());
    }
  }
  return success();
}

bool carryType(Type type) {
  if (type.isIndex() || isa<IntegerType, FloatType>(type)) return true;
  if (auto value = dyn_cast<riscv::ValueType>(type)) {
    auto carrier = value.getLayout().getCarrier();
    return (carrier == "rvv" || carrier == "scalar") &&
        llvm::all_of(value.getShape().asArrayRef(), [](int64_t size) { return size > 0; });
  }
  return isa<riscv::WindowType, riscv::FragmentType>(type);
}

LogicalResult loopContract(scf::ForOp loop, bool &descending,
                           bool admission = false) {
  auto control = PhysicalLoopControl::get(loop);
  if (failed(control) || failed(control->verifyFiniteEndpoint(loop))) return failure();
  descending = control->descending;
  auto schedule = loop->getAttrOfType<riscv::ScheduleAttr>("weft.riscv.schedule");
  if (!schedule || schedule.getPipelineDepth() != 2 || schedule.getBufferCount() != 2)
    return loop.emitError("pipeline requires the selected distance-one, two-stage schedule");
  if (loop.getInitArgs().empty())
    return unsupportedPipeline(loop, admission,
                               "pipeline requires a loop-carried numerical consumer");
  if (!llvm::hasSingleElement(loop.getRegion()) || !isa<scf::YieldOp>(loop.getBody()->getTerminator()))
    return loop.emitError("pipeline requires one ordered physical body and its explicit yield");
  for (Value initial : loop.getInitArgs())
    if (!carryType(initial.getType()))
      return unsupportedPipeline(loop, admission,
                                 "pipeline result has no supported scalar/register/window carry representation");
  return success();
}

void collectTransport(PipelineSchedule &schedule) {
  auto body = schedule.loop.getBody();
  auto yield = body->getTerminator();
  llvm::SmallPtrSet<Operation *, 32> consumers(schedule.consumers.begin(), schedule.consumers.end());
  schedule.transported.clear();
  schedule.transportInduction = false;
  for (Operation *consumer : schedule.consumers)
    for (Value operand : externalOperands(consumer))
      schedule.transportInduction |= operand == schedule.loop.getInductionVar();
  for (Value returned : yield->getOperands())
    schedule.transportInduction |= returned == schedule.loop.getInductionVar();
  for (Operation *producer : schedule.producers)
    if (!schedule.isInvariant(producer))
    for (Value result : producer->getResults()) {
      bool crosses = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        auto owner = ancestorInBlock(use.getOwner(), body);
        return owner == yield || consumers.contains(owner);
      });
      if (crosses) schedule.transported.push_back(result);
    }
}

LogicalResult verifyTransport(const PipelineSchedule &schedule) {
  PipelineSchedule expected(schedule);
  collectTransport(expected);
  if (expected.transported != schedule.transported || expected.transportInduction != schedule.transportInduction)
    return schedule.loop->emitError("pipeline transport does not match the physical use-def frontier");
  if (schedule.transported.empty() && schedule.invariants.empty())
    return schedule.loop->emitError("pipeline has no value crossing its producer/consumer stages");
  for (Value value : schedule.transported)
    if (!carryType(value.getType()))
      return value.getDefiningOp()->emitError("pipeline frontier cannot carry a descriptor, local object or dynamic numeric shape");
  return success();
}
} // namespace

bool PipelineSchedule::isProducer(Operation *operation) const {
  return llvm::is_contained(producers, operation);
}

bool PipelineSchedule::isInvariant(Operation *operation) const {
  return llvm::is_contained(invariants, operation);
}

LogicalResult PipelineSchedule::verify() const {
  auto loop = this->loop;
  bool direction;
  if (failed(loopContract(loop, direction))) return failure();
  if (direction != descending || producers.empty() || consumers.empty() ||
      ordered.size() != producers.size() + consumers.size() || ordered.size() > maximumOperations)
    return loop.emitError("pipeline stage membership or ordered domain is incomplete");
  llvm::DenseMap<Operation *, unsigned> positions;
  llvm::SmallPtrSet<Operation *, 32> producerSet(producers.begin(), producers.end());
  llvm::SmallPtrSet<Operation *, 32> consumerSet(consumers.begin(), consumers.end());
  if (producerSet.size() != producers.size() || consumerSet.size() != consumers.size())
    return loop.emitError("pipeline stage contains a duplicate operation");
  llvm::SmallPtrSet<Operation *, 32> invariantSet(invariants.begin(), invariants.end());
  if (invariantSet.size() != invariants.size() ||
      llvm::any_of(invariants, [&](Operation *operation) { return !producerSet.contains(operation); }))
    return loop.emitError("pipeline invariants must be a unique subset of the producer stage");
  for (auto [position, operation] : llvm::enumerate(ordered)) {
    if (operation->getBlock() != loop.getBody() || isa<scf::YieldOp>(operation) ||
        producerSet.contains(operation) == consumerSet.contains(operation) ||
        !positions.try_emplace(operation, position).second)
      return loop.emitError("pipeline order must contain every body operation exactly once");
  }
  unsigned edges = 0;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (!positions.contains(&operation))
      return operation.emitError("physical operation is missing from its pipeline order");
    if (failed(readonlyUnit(&operation))) return failure();
    bool producer = producerSet.contains(&operation);
    auto dependencies = stageDependencies(&operation);
    if (failed(dependencies)) return failure();
    for (Value operand : *dependencies) {
      if (++edges > maximumEdges) return loop.emitError("pipeline dependency graph exceeds its explicit edge bound");
      if (auto argument = dyn_cast<BlockArgument>(operand); argument && argument.getOwner() == loop.getBody()) {
        if (invariantSet.contains(&operation))
          return operation.emitError("pipeline invariant depends on an iteration coordinate or carry");
        if (producer && argument != loop.getInductionVar())
          return operation.emitError("next-iteration producer depends on an uncomputed loop carry");
        continue;
      }
      auto definition = ancestorInBlock(operand.getDefiningOp(), loop.getBody());
      if (!definition) continue;
      if (invariantSet.contains(&operation) && !invariantSet.contains(definition))
        return operation.emitError("pipeline invariant depends on an iteration-varying producer");
      bool sourceProducer = producerSet.contains(definition);
      if (producer && !sourceProducer)
        return operation.emitError("pipeline producer depends on its consumer stage");
      // A producer-to-consumer edge uses the previous iteration's buffered
      // value. All other edges are ordinary same-iteration SSA dependencies.
      if (sourceProducer == producer && positions.lookup(definition) >= positions.lookup(&operation))
        return operation.emitError("pipeline order uses a same-stage value before its definition");
    }
  }
  if (failed(verifyTransport(*this))) return failure();
  return success();
}

void PipelineSchedule::write() const {
  Builder builder(loop->getContext());
  for (auto [position, operation] : llvm::enumerate(ordered)) {
    operation->setAttr(pipelineStage, builder.getI64IntegerAttr(isProducer(operation) ? 0 : 1));
    operation->setAttr(pipelineOrder, builder.getI64IntegerAttr(position));
    if (isInvariant(operation)) operation->setAttr(pipelineInvariant, builder.getUnitAttr());
    else operation->removeAttr(pipelineInvariant);
  }
}

FailureOr<PipelineSchedule> PipelineSchedule::read(scf::ForOp loop) {
  PipelineSchedule result;
  result.loop = loop;
  if (failed(loopContract(loop, result.descending))) return failure();
  llvm::DenseMap<int64_t, Operation *> positions;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto stage = operation.getAttrOfType<IntegerAttr>(pipelineStage);
    auto order = operation.getAttrOfType<IntegerAttr>(pipelineOrder);
    if (!stage || (stage.getInt() != 0 && stage.getInt() != 1) || !order ||
        order.getInt() < 0 || !positions.try_emplace(order.getInt(), &operation).second)
      return operation.emitError("pipeline requires a unique total order and an exact stage for every operation"), failure();
    (stage.getInt() ? result.consumers : result.producers).push_back(&operation);
  }
  for (size_t position = 0; position < positions.size(); ++position) {
    auto found = positions.find(position);
    if (found == positions.end()) return loop.emitError("pipeline order must be a dense permutation of its body"), failure();
    result.ordered.push_back(found->second);
  }
  // Prologue/epilogue use their corresponding subsequences of the selected
  // total order, not the source block order left over from scheduling.
  result.producers.clear(); result.consumers.clear();
  for (Operation *operation : result.ordered)
    (operation->getAttrOfType<IntegerAttr>(pipelineStage).getInt() ? result.consumers : result.producers).push_back(operation);
  for (Operation *operation : result.ordered)
    if (operation->hasAttr(pipelineInvariant)) {
      if (!isa<UnitAttr>(operation->getAttr(pipelineInvariant)))
        return operation->emitError("pipeline invariant marker must be the selected unit attribute"), failure();
      result.invariants.push_back(operation);
    }
  collectTransport(result);
  if (failed(result.verify()) || failed(verifyPipelineShapeStages(loop))) return failure();
  return result;
}

FailureOr<std::optional<PipelineSchedule>> riscv_internal::schedulePipeline(scf::ForOp loop) {
  PipelineSchedule result;
  result.loop = loop;
  if (failed(loopContract(loop, result.descending, true))) return failure();
  // Freeze actual numeric reads before carrying the supply to another stage;
  // an encoded storage projection is not a buffered value.
  if (failed(materializeContractionReads(loop))) return failure();
  SmallVector<Operation *> operations;
  llvm::SmallPtrSet<Operation *, 32> consumerSet;
  unsigned edges = 0;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (operations.size() == maximumOperations)
      return loop.emitError("pipeline operation domain exceeds its explicit bound"), failure();
    if (failed(readonlyUnit(&operation, true))) return failure();
    operations.push_back(&operation);
    // A contraction starts the consuming stage even when its accumulator is
    // local to this iteration. Otherwise all dot work becomes a producer and
    // only the final scalar carry update overlaps the following iteration.
    if (contractionConsumer(&operation))
      consumerSet.insert(&operation);
    auto dependencies = stageDependencies(&operation);
    if (failed(dependencies)) return failure();
    for (Value operand : *dependencies) {
      if (++edges > maximumEdges) return loop.emitError("pipeline dependency graph exceeds its explicit edge bound"), failure();
      auto argument = dyn_cast<BlockArgument>(operand);
      if ((argument && argument.getOwner() == loop.getBody() && argument != loop.getInductionVar()) ||
          consumerSet.contains(ancestorInBlock(operand.getDefiningOp(), loop.getBody())))
        consumerSet.insert(&operation);
    }
  }
  auto yield = loop.getBody()->getTerminator();
  for (Value value : yield->getOperands())
    if (auto definition = value.getDefiningOp(); definition && definition->getBlock() == loop.getBody())
      consumerSet.insert(definition);
  // Closing a yielded producer into the consumer set can affect later users.
  // Source order is topological, so a single forward sweep closes that set.
  for (Operation *operation : operations) {
    auto dependencies = stageDependencies(operation);
    if (failed(dependencies)) return failure();
    for (Value operand : *dependencies)
      if (consumerSet.contains(ancestorInBlock(operand.getDefiningOp(), loop.getBody()))) consumerSet.insert(operation);
  }
  for (Operation *operation : llvm::reverse(operations)) {
    if (consumerSet.contains(operation) || !replayableDescriptor(operation)) continue;
    bool seen = false, onlyConsumers = true;
    for (Value value : operation->getResults())
      for (OpOperand &use : value.getUses()) {
        auto owner = ancestorInBlock(use.getOwner(), loop.getBody());
        if (!owner) continue;
        seen = true;
        onlyConsumers &= consumerSet.contains(owner);
      }
    if (seen && onlyConsumers) consumerSet.insert(operation);
  }
  for (Operation *operation : operations)
    (consumerSet.contains(operation) ? result.consumers : result.producers).push_back(operation);
  if (result.producers.empty() || result.consumers.empty()) return std::optional<PipelineSchedule>{};
  llvm::SmallPtrSet<Operation *, 32> invariantSet;
  for (Operation *operation : result.producers) {
    bool invariant = true;
    auto dependencies = stageDependencies(operation);
    if (failed(dependencies)) return failure();
    for (Value operand : *dependencies) {
      if (auto argument = dyn_cast<BlockArgument>(operand); argument && argument.getOwner() == loop.getBody()) {
        invariant = false;
        break;
      }
      if (auto owner = ancestorInBlock(operand.getDefiningOp(), loop.getBody()); owner && !invariantSet.contains(owner)) {
        invariant = false;
        break;
      }
    }
    if (invariant) { invariantSet.insert(operation); result.invariants.push_back(operation); }
  }
  collectTransport(result);
  if (result.transported.empty() && result.invariants.empty()) return std::optional<PipelineSchedule>{};
  for (Value value : result.transported)
    if (!carryType(value.getType()))
      return rejectRISCVCandidate(value.getDefiningOp(), RISCVFailureCategory::Unsupported,
          "pipeline frontier cannot carry a descriptor, local object or dynamic numeric shape"), failure();
  llvm::append_range(result.ordered, result.producers);
  llvm::append_range(result.ordered, result.consumers);
  if (failed(result.verify())) return failure();
  return std::optional<PipelineSchedule>(std::move(result));
}

namespace {
struct FrontierResidency {
  int64_t vector = 0, integer = 0, floating = 0;

  bool add(const FrontierResidency &other) {
    for (auto pair : {std::pair(&vector, other.vector), std::pair(&integer, other.integer),
                      std::pair(&floating, other.floating)}) {
      if (pair.second < 0 || *pair.first > std::numeric_limits<int64_t>::max() - pair.second) return false;
      *pair.first += pair.second;
    }
    return true;
  }
  void subtract(const FrontierResidency &other) {
    vector -= other.vector; integer -= other.integer; floating -= other.floating;
  }
  bool nonnegative() const { return vector >= 0 && integer >= 0 && floating >= 0; }
};

std::optional<FrontierResidency> frontierResidency(Value value, unsigned xlen) {
  auto vector = residentRegisterGroups(value);
  if (!vector) return std::nullopt;
  FrontierResidency result{*vector, 0, 0};
  if (*vector || matchPattern(value, m_Constant())) return result;
  Type element = value.getType();
  int64_t copies = 1;
  if (auto numeric = dyn_cast<riscv::ValueType>(element)) {
    if (numeric.getLayout().getCarrier() != "scalar") return result;
    auto count = staticProduct(numeric.getShape().asArrayRef());
    if (!count) return std::nullopt;
    copies = *count;
    element = numeric.getElementType();
  }
  if (isa<FloatType>(element)) result.floating = copies;
  else if (element.isIndex()) result.integer = copies;
  else if (auto integer = dyn_cast<IntegerType>(element)) {
    int64_t words = (integer.getWidth() + xlen - 1) / xlen;
    auto count = staticProduct({copies, words});
    if (!count) return std::nullopt;
    result.integer = *count;
  }
  return result;
}

struct FrontierValue {
  Value identity;
  Operation *birth = nullptr;
  FrontierResidency residency;
  llvm::SmallPtrSet<Operation *, 8> users;
  unsigned remaining = 0;
  bool live = false;
};

// A bounded, use-count view of the expanded steady state. It is deliberately
// rebuilt from real SSA, including loop carries and representation aliases;
// the source stage labels cannot tell us how many versions coexist here.
class FrontierLiveness {
public:
  static FailureOr<FrontierLiveness> get(scf::ForOp loop) {
    return get(*loop.getBody(), loop);
  }

  static FailureOr<FrontierLiveness> get(Block &body, Operation *owner) {
    FrontierLiveness result;
    result.body = &body;
    auto kernel = dyn_cast<riscv::KernelOp>(owner);
    if (!kernel) kernel = owner->getParentOfType<riscv::KernelOp>();
    if (!kernel) return owner->emitError("physical issue frontier has no target kernel"), failure();
    const unsigned xlen = kernel.getTarget().getMarch().starts_with("rv32") ? 32 : 64;
    auto add = [&](Value value) -> LogicalResult {
      value = registerStorageIdentity(value);
      if (result.indices.contains(value)) return success();
      auto residency = frontierResidency(value, xlen);
      if (!residency) return owner->emitError("issue frontier has an unknown or overflowing physical residency");
      if (!residency->vector && !residency->integer && !residency->floating) return success();
      FrontierValue record;
      record.identity = value;
      record.residency = *residency;
      auto defining = value.getDefiningOp();
      if (defining && defining->getBlock() == result.body) record.birth = defining;
      SmallVector<Value> aliases{value};
      llvm::DenseSet<Value> visited;
      while (!aliases.empty()) {
        Value alias = aliases.pop_back_val();
        if (!visited.insert(alias).second) continue;
        if (visited.size() > maximumEdges)
          return owner->emitError("issue frontier alias graph exceeds its explicit bound");
        for (OpOperand &use : alias.getUses()) {
          if (auto owner = ancestorInBlock(use.getOwner(), result.body)) record.users.insert(owner);
          for (Value output : use.getOwner()->getResults())
            if (registerStorageIdentity(output) == value) aliases.push_back(output);
        }
      }
      if (record.users.empty()) return success();
      record.remaining = record.users.size();
      record.live = !record.birth;
      if (record.live && !result.live.add(record.residency))
        return owner->emitError("issue frontier entry residency overflows its bounded counter");
      result.indices[value] = result.values.size();
      result.values.push_back(std::move(record));
      return success();
    };
    for (BlockArgument argument : result.body->getArguments())
      if (failed(add(argument))) return failure();
    for (Operation &operation : *result.body) {
      for (Value operand : externalOperands(&operation))
        if (failed(add(operand))) return failure();
      for (Value value : operation.getResults())
        if (failed(add(value))) return failure();
    }
    for (auto [index, value] : llvm::enumerate(result.values)) {
      if (value.birth) result.births[value.birth].push_back(index);
      for (Operation *user : value.users) result.uses[user].push_back(index);
    }
    return result;
  }

  struct Cost { FrontierResidency after; int64_t vectorPeak = 0; };

  FailureOr<Cost> cost(Operation *operation) const {
    Cost result{live, live.vector};
    auto used = uses.find(operation);
    if (used != uses.end())
      for (unsigned index : used->second) {
        const auto &value = values[index];
        if (!value.live || !value.remaining)
          return operation->emitError("pipeline frontier order reaches an unavailable value"), failure();
        if (value.remaining == 1) result.after.subtract(value.residency);
      }
    auto born = births.find(operation);
    if (born != births.end())
      for (unsigned index : born->second)
        if (!result.after.add(values[index].residency))
          return operation->emitError("pipeline frontier result residency overflows"), failure();
    if (!result.after.nonnegative()) return operation->emitError("pipeline frontier released an unavailable carrier"), failure();
    if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf")) {
      auto temporary = leaf.getTemporaryGroups() + leaf.getFragmentGroups();
      if (temporary < 0 || temporary > std::numeric_limits<int64_t>::max() - result.vectorPeak)
        return operation->emitError("pipeline leaf scratch residency overflows"), failure();
      result.vectorPeak += temporary;
    }
    for (Region &region : operation->getRegions())
      for (Block &block : region) {
        auto nested = registerPeak(block);
        if (!nested || nested->total() > std::numeric_limits<int64_t>::max() - live.vector)
          return operation->emitError("pipeline nested region has no bounded register estimate"), failure();
        result.vectorPeak = std::max(result.vectorPeak, live.vector + nested->total());
      }
    result.vectorPeak = std::max(result.vectorPeak, result.after.vector);
    return result;
  }

  LogicalResult apply(Operation *operation, const Cost &selected) {
    auto used = uses.find(operation);
    if (used != uses.end())
      for (unsigned index : used->second) {
        auto &value = values[index];
        if (!--value.remaining) value.live = false;
      }
    auto born = births.find(operation);
    if (born != births.end())
      for (unsigned index : born->second) values[index].live = true;
    live = selected.after;
    return success();
  }

private:
  Block *body = nullptr;
  FrontierResidency live;
  SmallVector<FrontierValue> values;
  llvm::DenseMap<Value, unsigned> indices;
  llvm::DenseMap<Operation *, SmallVector<unsigned>> uses, births;
};
} // namespace

FailureOr<SmallVector<Operation *>> riscv_internal::orderPipelineFrontiers(
    const PipelineSchedule &schedule, const PipelineExpansion &expansion, int64_t registerBudget) {
  auto steady = expansion.steady;
  if (registerBudget < 1) return steady.emitError("pipeline frontier has no available register budget"), failure();
  auto liveness = FrontierLiveness::get(expansion.steady);
  if (failed(liveness)) return failure();
  SmallVector<Operation *> ordered;
  SmallVector<Operation *> producers;
  for (Operation *operation : schedule.producers)
    if (schedule.isInvariant(operation)) ordered.push_back(operation);
    else producers.push_back(operation);
  unsigned producer = 0, consumer = 0;
  while (producer < producers.size() || consumer < schedule.consumers.size()) {
    Operation *p = producer < producers.size() ? producers[producer] : nullptr;
    Operation *c = consumer < schedule.consumers.size() ? schedule.consumers[consumer] : nullptr;
    FailureOr<FrontierLiveness::Cost> producerCost = failure(), consumerCost = failure();
    if (p) producerCost = liveness->cost(expansion.steadyOperations.lookup(p));
    if (c) consumerCost = liveness->cost(expansion.steadyOperations.lookup(c));
    if ((p && failed(producerCost)) || (c && failed(consumerCost))) return failure();
    // Keep the earliest legal lookahead. When a new input would exceed the
    // available carrier frontier, retire current consumers first. Integer and
    // FP counts are issue-order heuristics over the architectural register
    // files, not promises about the downstream allocator's ABI temporaries.
    bool takeProducer = p && (!c || (producerCost->vectorPeak <= registerBudget &&
        producerCost->after.integer <= 32 && producerCost->after.floating <= 32));
    Operation *original = takeProducer ? p : c;
    auto selected = takeProducer ? *producerCost : *consumerCost;
    if (failed(liveness->apply(expansion.steadyOperations.lookup(original), selected))) return failure();
    ordered.push_back(original);
    if (takeProducer) ++producer; else ++consumer;
  }
  for (Operation *operation : ordered)
    if (!schedule.isInvariant(operation))
      expansion.steadyOperations.lookup(operation)->moveBefore(steady.getBody()->getTerminator());
  return ordered;
}

FailureOr<SmallVector<Operation *>> riscv_internal::orderReadOnlyIssues(
    Operation *owner, Block &body, ArrayRef<Operation *> operations,
    ArrayRef<std::pair<int64_t, int64_t>> priorities, int64_t registerBudget) {
  if (operations.empty() || operations.size() != priorities.size() || operations.size() > 8192 || registerBudget < 1)
    return owner->emitError("read-only issue scheduling requires a bounded operation/priority domain and target budget"), failure();
  llvm::DenseMap<Operation *, unsigned> indices;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    if (operation->getBlock() != &body || operation->getNumRegions() ||
        (!isPipelineShapeBinding(operation) && !riscv::hasOnlyReadEffects(operation)) ||
        !indices.try_emplace(operation, index).second)
      return owner->emitError("read-only issue order requires unique flat pure/read operations in one block"), failure();
    if (index && operations[index - 1]->getNextNode() != operation)
      return owner->emitError("read-only issue scheduling cannot cross a gap in physical execution"), failure();
  }
  SmallVector<unsigned> degrees(operations.size(), 0);
  SmallVector<SmallVector<unsigned>> users(operations.size());
  unsigned edges = 0;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    llvm::DenseSet<unsigned> dependencies;
    SmallVector<Value> operands(operation->getOperands());
    auto shapes = physicalShapeDependencies(operation);
    if (failed(shapes)) return failure();
    llvm::append_range(operands, *shapes);
    for (Value operand : operands) {
      auto definition = operand.getDefiningOp();
      auto found = indices.find(definition);
      if (found == indices.end()) continue;
      if (found->second >= index)
        return operation->emitError("issue dependency does not dominate its original consumer"), failure();
      dependencies.insert(found->second);
    }
    for (unsigned dependency : dependencies) {
      if (++edges > 65536) return owner->emitError("issue scheduling dependency graph exceeds its explicit bound"), failure();
      users[dependency].push_back(index);
      ++degrees[index];
    }
  }
  auto liveness = FrontierLiveness::get(body, owner);
  if (failed(liveness)) return failure();
  // Fully unrolled windows may be embedded in a larger block. Advance the
  // lifetime state through its unchanged prefix, including values whose uses
  // extend beyond this window; those values cannot be borrowed as free space.
  for (Operation &operation : body) {
    if (&operation == operations.front()) break;
    auto cost = liveness->cost(&operation);
    if (failed(cost) || failed(liveness->apply(&operation, *cost))) return failure();
  }
  SmallVector<unsigned> ready;
  for (unsigned index = 0; index < degrees.size(); ++index)
    if (!degrees[index]) ready.push_back(index);
  SmallVector<Operation *> result;
  auto preferred = [&](unsigned a, unsigned b) {
    return priorities[a] == priorities[b] ? a < b : priorities[a] < priorities[b];
  };
  while (!ready.empty()) {
    unsigned best = 0;
    std::optional<FrontierLiveness::Cost> bestCost;
    bool bestFits = false;
    for (unsigned position = 0; position < ready.size(); ++position) {
      auto cost = liveness->cost(operations[ready[position]]);
      if (failed(cost)) return failure();
      bool fits = cost->vectorPeak <= registerBudget && cost->after.floating <= 32 && cost->after.integer <= 32;
      bool better = !bestCost || (fits != bestFits ? fits : fits ? preferred(ready[position], ready[best]) :
          std::tie(cost->vectorPeak, cost->after.vector, cost->after.floating, cost->after.integer) <
          std::tie(bestCost->vectorPeak, bestCost->after.vector, bestCost->after.floating, bestCost->after.integer));
      if (!better && bestCost && !fits && !bestFits &&
          std::tie(cost->vectorPeak, cost->after.vector, cost->after.floating, cost->after.integer) ==
          std::tie(bestCost->vectorPeak, bestCost->after.vector, bestCost->after.floating, bestCost->after.integer))
        better = preferred(ready[position], ready[best]);
      if (better) { best = position; bestCost = *cost; bestFits = fits; }
    }
    unsigned selected = ready[best];
    if (failed(liveness->apply(operations[selected], *bestCost))) return failure();
    result.push_back(operations[selected]);
    ready.erase(ready.begin() + best);
    for (unsigned user : users[selected]) if (!--degrees[user]) ready.push_back(user);
  }
  if (result.size() != operations.size())
    return owner->emitError("read-only issue scheduling found an incomplete dependency frontier"), failure();
  return result;
}
