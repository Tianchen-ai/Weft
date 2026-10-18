#include "DenseContractions.h"
#include "DenseContractionEpilogue.h"
#include "ContractionSupply.h"
#include "DenseIssue.h"
#include "DensePanelTransfer.h"
#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Target/RISCVCompilation.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/RISCVPipeline.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"

#include <functional>
#include <limits>
#include <tuple>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {

struct ContractDestination : ContractionEpilogueGraph {

  static bool pointwise(Operation *op) {
    return isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp, riscv::WidenOp,
               riscv::NarrowOp, riscv::CompareOp, riscv::SelectOp>(op);
  }

  static bool representation(riscv::ConvertLayoutOp operation) {
    auto relation = operation.getConversion();
    return relation.getEffect() == "pure" || relation.getKind() == "local_store" || relation.getKind() == "local_read";
  }

  static std::optional<ContractDestination> get(Operation *operation,
                                               const ContractionSupply (&supplies)[2]) {
    ContractDestination destination;
    Value value = operation->getResult(0);
    destination.root = value;
    SmallVector<Value> pending{value};
    llvm::SmallPtrSet<Operation *, 32> members;
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 64) return std::nullopt;
      for (OpOperand &use : pending[cursor].getUses()) {
        Operation *consumer = use.getOwner();
        if (consumer->getBlock() != operation->getBlock() ||
            !operation->isBeforeInBlock(consumer)) return std::nullopt;
        if (!members.insert(consumer).second) continue;
        if (auto store = dyn_cast<riscv::StoreOp>(consumer)) {
          if (use.getOperandNumber() != 0 || destination.store) return std::nullopt;
          destination.store = store;
          continue;
        }
        if (consumer->getNumResults() != 1 || consumer->getNumRegions() ||
            logicalAxes(consumer->getResult(0).getType()) != logicalAxes(value.getType()) ||
            logicalShape(consumer->getResult(0).getType()) != logicalShape(value.getType())) return std::nullopt;
        if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(consumer)) {
          if (!representation(conversion)) return std::nullopt;
        } else if (!pointwise(consumer)) return std::nullopt;
        destination.adapters.push_back(consumer);
        pending.push_back(consumer->getResult(0));
      }
    }
    if (!destination.store || !operation->isBeforeInBlock(destination.store))
      return std::nullopt;
    auto ordered = [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); };
    llvm::sort(destination.adapters, ordered);
    Value output = destination.store.getRegion();
    auto memory = cast<riscv::MemDescType>(output.getType());
    auto encoding = dyn_cast<kernel::EncodingType>(memory.getEncoding());
    if (!encoding || encoding.getKind() != "dense" || memory.getElements() != 1 || memory.getInterleaveRows() ||
        memory.getAxisIds().asArrayRef() != logicalAxes(value.getType()) ||
        memory.getShape().asArrayRef() != logicalShape(value.getType())) return std::nullopt;
    for (Operation *adapter : destination.adapters)
      for (Value input : adapter->getOperands()) {
        if (input == value || (input.getDefiningOp() && members.contains(input.getDefiningOp())) ||
            !isa<riscv::ValueType>(input.getType())) continue;
        if (!llvm::is_contained(destination.inputs, input)) destination.inputs.push_back(input);
      }
    // Every numerical observer must belong to this exact pointwise DAG ending
    // at one store. A second store or a reduction keeps the complete value.
    for (Operation *next = operation->getNextNode(); next != destination.store;
         next = next->getNextNode()) {
      if (!next) return std::nullopt;
      if (members.contains(next)) continue;
      // Physical storage may be allocated between the numerical producer and
      // its representation store. The allocation neither reads nor mutates an
      // input; its lexical lifetime stays exactly where it was placed.
      if (isa<riscv::LocalAllocOp>(next)) continue;
      if (!riscv::hasOnlyReadEffects(next, true)) return std::nullopt;
      bool observesOutput = false;
      next->walk([&](MemoryEffectOpInterface memory) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memory.getEffects(effects);
        for (auto &effect : effects)
          if (isa<MemoryEffects::Read>(effect.getEffect())) {
            Value read = effect.getValue();
            if (!read || riscv::mayAliasStorage(read, output)) observesOutput = true;
          }
      });
      if (observesOutput) return std::nullopt;
    }
    // A tiled store must not overwrite an input needed by a later tile. Frozen
    // numeric operands instead receive their own complete private snapshot.
    for (auto supply : supplies)
      if (supply.read && !supply.frozen && riscv::mayAliasStorage(supply.read.getRegion(), output))
        return std::nullopt;
    for (Value input : destination.inputs) {
      auto type = cast<riscv::ValueType>(input.getType());
      for (auto [axis, extent] : llvm::zip(type.getAxisIds().asArrayRef(), type.getShape().asArrayRef())) {
        auto found = llvm::find(logicalAxes(value.getType()), axis);
        if (found == logicalAxes(value.getType()).end()) return std::nullopt;
        if (extent != 1 && extent != logicalShape(value.getType())[found - logicalAxes(value.getType()).begin()])
          return std::nullopt;
      }
      auto supply = queryContractionSupply(input);
      if (!supply.read || supply.frozen || !supply.wideningCasts.empty() || supply.read.getSnapshotStorage() ||
          riscv::readCrossesWrite(supply.read, destination.store) ||
          riscv::mayAliasStorage(supply.read.getRegion(), output)) return std::nullopt;
      auto source = supply.read.getRegion().getType();
      auto encoding = dyn_cast<kernel::EncodingType>(source.getEncoding());
      if (!encoding || encoding.getKind() != "dense" || source.getElements() != 1 || source.getInterleaveRows() ||
          source.getShape().asArrayRef() != logicalShape(input.getType()) ||
          source.getAxisIds().asArrayRef() != ArrayRef<int64_t>(supply.valueToReadAxes)) return std::nullopt;
    }
    return destination;
  }
};

// All facts refer to an existing closed operation. This temporary query is
// consumed immediately; the resulting loops and carriers are the program.
struct ContractFacts {
  Operation *operation;
  Type outputType;
  riscv::ValueType result;
  riscv::ValueType inputs[2];
  ContractionSupply supplies[2];
  riscv::ScheduleAttr schedule;
  Operation *scheduleOwner = nullptr;
  riscv::TargetAttr target;
  SmallVector<int64_t> reductionAxes;
  int64_t axis = 0;
  int64_t laneAxis = 0;
  unsigned laneInput = 0;
  int64_t maximumLMUL;
  bool nativeHalf = false;
  Type storageTypes[2];
  std::optional<ContractDestination> destination;

  ContractFacts(Operation *op, int64_t maximumLMUL) : operation(op), maximumLMUL(maximumLMUL) {}

  ArrayRef<int64_t> freeShape() const { return logicalShape(outputType); }
  ArrayRef<int64_t> freeAxes() const { return logicalAxes(outputType); }
  Type element() const { return logicalElement(outputType); }
  bool isReduction(int64_t id) const { return llvm::is_contained(reductionAxes, id); }
  bool sameInput() const { return operation->getOperand(0) == operation->getOperand(1); }
  bool reductionLanes() const { return laneAxis == axis; }

  std::optional<int64_t> staticReductionExtent(int64_t id) const {
    auto position = llvm::find(inputs[0].getAxisIds().asArrayRef(), id);
    if (position == inputs[0].getAxisIds().asArrayRef().end()) return std::nullopt;
    int64_t size = inputs[0].getShape()[position - inputs[0].getAxisIds().asArrayRef().begin()];
    return size > 0 ? std::optional<int64_t>(size) : std::nullopt;
  }

  void selectNaturalRepresentation(std::optional<int64_t> singleton = {}) {
    laneAxis = axis;
    laneInput = 0;
    if (reductionAxes.size() != 1 || !result) return;
    for (unsigned input : {1u, 0u}) {
      int64_t candidate = inputs[input].getAxisIds().asArrayRef().back();
      if (isReduction(candidate) || singleton == candidate ||
          !llvm::is_contained(freeAxes(), candidate) ||
          llvm::is_contained(inputs[1 - input].getAxisIds().asArrayRef(), candidate)) continue;
      auto position = llvm::find(freeAxes(), candidate) - freeAxes().begin();
      if (freeShape()[position] == 1) continue;
      laneAxis = candidate;
      laneInput = input;
      break;
    }
  }

  bool analyze() {
    if (!isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(operation)) return false;
    auto implementation = operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
    auto over = operation->getAttrOfType<DenseI64ArrayAttr>("over");
    outputType = operation->getResult(0).getType();
    result = dyn_cast<riscv::ValueType>(outputType);
    schedule = operation->getAttrOfType<riscv::ScheduleAttr>("schedule");
    if ((!element().isF32() && !element().isF64()) || freeShape().size() > 4 ||
        llvm::is_contained(freeShape(), int64_t(0)) ||
        !implementation || implementation.getEngine() != "rvv" || !over || over.empty() || !schedule)
      return false;
    if ((implementation.getFamily() != "contract" || implementation.getOperation() != "rvv.vfmacc") &&
        (implementation.getFamily() != "widen-float-contract" || implementation.getOperation() != "rvv.vfwmacc"))
      return false;
    reductionAxes.assign(over.asArrayRef().begin(), over.asArrayRef().end());
    axis = reductionAxes.back();
    target = operation->getParentOfType<riscv::KernelOp>().getTarget();
    // The finite target binding applies to the nearest eligible physical
    // producer/consumer cluster, within the same closed contraction. It never
    // pipelines the author's private allocation or crosses the outer K add.
    for (Operation *parent = operation->getParentOp(); parent && !isa<riscv::KernelOp>(parent);
         parent = parent->getParentOp()) {
      if (auto enclosing = parent->getAttrOfType<riscv::ScheduleAttr>("weft.riscv.schedule")) {
        if (enclosing.getPipelineDepth() > 1) {
          Builder builder(operation->getContext());
          schedule = riscv_internal::schedule(builder, schedule.getUnroll(), enclosing.getPipelineDepth());
          scheduleOwner = parent;
          break;
        }
      }
    }
    for (unsigned index = 0; index < 2; ++index) {
      Value value = operation->getOperand(index);
      inputs[index] = dyn_cast<riscv::ValueType>(value.getType());
      auto type = inputs[index];
      if (!type || type.getShape().size() > 8 ||
          (type.getElementType() != element() && !(element().isF32() && type.getElementType().isF16())))
        return false;
      for (int64_t id : reductionAxes)
        if (!llvm::is_contained(type.getAxisIds().asArrayRef(), id)) return false;
      supplies[index] = queryContractionSupply(value);
      // A frozen numeric producer is snapshotted, never looked through to its
      // old memory. Stable read supplies preserve only exact widening casts.
      auto &supply = supplies[index];
      bool stable = supply.read && !supply.frozen && !supply.read.getSnapshotStorage() &&
                    !riscv::readCrossesWrite(supply.read, operation);
      if (stable) {
        auto memory = supply.read.getRegion().getType();
        stable = cast<kernel::EncodingType>(memory.getEncoding()).getKind() == "dense" &&
                 memory.getElements() == 1 && memory.getInterleaveRows() == 0 &&
                 logicalShape(value.getType()) == memory.getShape().asArrayRef() &&
                 ArrayRef<int64_t>(supply.valueToReadAxes) == memory.getAxisIds().asArrayRef();
      }
      auto storage = stable
          ? logicalElement(supplies[index].read.getResult().getType()) : type.getElementType();
      if (storage != element() && !(element().isF32() && (storage.isF16() || storage.isBF16()))) return false;
      storageTypes[index] = storage;
      if (index == 0) nativeHalf = storage.isF16();
      else nativeHalf &= storage.isF16();
    }
    // A surviving innermost coordinate can directly carry different outputs.
    // The other operand must broadcast over that coordinate. This selects a
    // representation of the existing closed contract, not an outer loop tree
    // or a persistent transpose. Dynamic strides retain an explicit guarded
    // unit/strided access choice in the physical program.
    selectNaturalRepresentation();
    destination = ContractDestination::get(operation, supplies);
    // A shared dynamic result uses the same bounded private storage as a
    // tiled static result. Its runtime capacity is checked by the resource
    // pass; the placeholder shape must never become a register extent.
    return true;
  }
};

struct TileChoice {
  SmallVector<int64_t> shape;
  int64_t lanes = 0, accumulatorLMUL = 0;
  int64_t registers = 0, work = 0, products = 0;
  int64_t loads = 0, conversions = 0, traversals = 0;
  int64_t overlappingGroups = 0;
  int64_t scalarSteps = 1;

  riscv::ContractionTileAttr attribute(const ContractFacts &facts) const {
    Builder b(facts.operation->getContext());
    return riscv::ContractionTileAttr::get(b.getContext(), b.getDenseI64ArrayAttr(shape),
        facts.laneAxis, facts.laneInput, lanes, accumulatorLMUL, scalarSteps,
        registers, loads, conversions);
  }

  static TileChoice read(riscv::ContractionTileAttr selected) {
    TileChoice choice;
    choice.shape.assign(selected.getShape().asArrayRef().begin(), selected.getShape().asArrayRef().end());
    choice.lanes = selected.getLanes();
    choice.accumulatorLMUL = selected.getLmulEighths();
    choice.scalarSteps = selected.getScalarSteps();
    choice.registers = selected.getRegisters();
    choice.loads = selected.getLoads();
    choice.conversions = selected.getConversions();
    return choice;
  }
};

struct PanelChoice {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> batchAxes;
  size_t lane = 0, row = 0;
  int64_t reduction = 0;
};

int64_t scalarSupplySteps(const ContractFacts &facts, int64_t copies) {
  auto cache = dyn_cast<riscv::DataCacheAttr>(facts.target.getDataCache());
  if (!cache || facts.reductionLanes() || copies <= 0) return 1;
  // The packet accompanies a transpose from two innermost-K supplies. For a
  // natural output-lane supply, retain the shorter scalar live range; its
  // vector-column replicas already amortize each scalar address and load.
  if (facts.inputs[facts.laneInput].getAxisIds().asArrayRef().back() != facts.axis) return 1;
  int64_t bytes = facts.storageTypes[1 - facts.laneInput].getIntOrFloatBitWidth() / 8;
  int64_t maximum = facts.operation->getParentOfType<riscv::KernelOp>().getContractionKSteps();
  // A finite scalar supply packet follows consecutive K coordinates. The
  // reported line bounds its useful span, while sixteen scalar values bound
  // the supply frontier independently of the vector accumulator residency.
  for (int64_t steps : {4, 2})
    if (steps <= maximum && steps * bytes <= cache.getLineBytes() && copies * steps <= 16)
      return steps;
  return 1;
}

std::optional<int64_t> privatePanelBudget(const ContractFacts &facts) {
  auto kernel = facts.operation->getParentOfType<riscv::KernelOp>();
  int64_t available = std::min<int64_t>(64 * 1024, facts.target.getMaxPrivateStackBytes());
  bool known = true;
  // A conservative reservation is sufficient for this optional representation.
  // Dynamic author snapshots keep their existing path; their runtime capacity
  // is not guessed to make a panel fit. Final liveness still verifies the IR.
  kernel.walk([&](Operation *operation) {
    if (auto allocation = dyn_cast<riscv::LocalAllocOp>(operation)) {
      auto type = allocation.getResult().getType();
      if (type.getSizeBytes() < 0) known = false;
      else if (type.getSizeBytes() >= available || type.getAlignment() - 1 >= available - type.getSizeBytes())
        available = 0;
      else available -= type.getSizeBytes() + type.getAlignment() - 1;
    }
    if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf"))
      available = std::max<int64_t>(0, available - leaf.getLocalBytes());
  });
  return known ? std::optional<int64_t>(available) : std::nullopt;
}

std::optional<PanelChoice> choosePanel(const ContractFacts &facts, const TileChoice &tile) {
  if (facts.reductionLanes() || !facts.destination || facts.freeAxes().size() < 2 ||
      facts.freeAxes().size() > 4) return std::nullopt;
  auto vectorAxes = facts.inputs[facts.laneInput].getAxisIds().asArrayRef();
  auto scalarAxes = facts.inputs[1 - facts.laneInput].getAxisIds().asArrayRef();
  PanelChoice panel;
  panel.lane = llvm::find(facts.freeAxes(), facts.laneAxis) - facts.freeAxes().begin();
  std::optional<size_t> row;
  for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
    bool vector = llvm::is_contained(vectorAxes, axis);
    bool scalar = llvm::is_contained(scalarAxes, axis);
    if (vector && scalar) panel.batchAxes.push_back(axis);
    else if (scalar) {
      if (row) return std::nullopt;
      row = position;
    } else if (!vector || position != panel.lane) return std::nullopt;
  }
  if (!row || vectorAxes.size() != panel.batchAxes.size() + 2 ||
      scalarAxes.size() != panel.batchAxes.size() + 2) return std::nullopt;
  panel.row = *row;
  size_t lane = panel.lane;
  // Either output coordinate can provide several microtiles. In particular,
  // one row can still reuse a K panel across adjacent column vectors instead
  // of traversing all K pages again for each small output vector.
  if (auto k = facts.staticReductionExtent(facts.axis); k && *k < 64) return std::nullopt;
  panel.shape.assign(facts.freeShape().size(), 1);
  panel.shape[*row] = facts.freeShape()[*row] > 0 ? std::min<int64_t>(facts.freeShape()[*row], 64) : 64;
  panel.shape[lane] = facts.freeShape()[lane] > 0 ? std::min<int64_t>(facts.freeShape()[lane], 64) : 64;
  if (panel.shape[*row] < 2 * tile.shape[*row] && panel.shape[lane] < 2 * tile.lanes)
    return std::nullopt;
  panel.reduction = facts.operation->getParentOfType<riscv::KernelOp>().getContractionPanelK();
  const int64_t inputBytes = facts.storageTypes[facts.laneInput].getIntOrFloatBitWidth() / 8;
  const int64_t outputBytes = facts.element().getIntOrFloatBitWidth() / 8;
  // This is a finite private representation of one closed operation. It does
  // not introduce an author stage, persistent pack, or a new reduction sum.
  // The final resource pass also accounts for surrounding allocations.
  auto budget = privatePanelBudget(facts);
  if (!budget) return std::nullopt;
  while (panel.reduction >= 32) {
    int64_t bytes = *staticProduct(panel.shape) * outputBytes +
        panel.reduction * panel.shape[lane] * inputBytes;
    if (bytes + 14 <= *budget) return panel;
    panel.reduction /= 2;
  }
  return std::nullopt;
}

int64_t panelCacheRefills(const ContractFacts &facts, const TileChoice &tile,
                         const PanelChoice &panel) {
  auto cache = dyn_cast<riscv::DataCacheAttr>(facts.target.getDataCache());
  if (!cache) return 0;
  int64_t vectorBytes = panel.reduction * panel.shape[panel.lane] *
      facts.storageTypes[facts.laneInput].getIntOrFloatBitWidth() / 8;
  int64_t partialBytes = *staticProduct(panel.shape) * facts.element().getIntOrFloatBitWidth() / 8;
  int64_t scalarRowBytes = panel.reduction *
      facts.storageTypes[1 - facts.laneInput].getIntOrFloatBitWidth() / 8;
  int64_t scalarBytes = tile.shape[panel.row] *
      ((scalarRowBytes + cache.getLineBytes() - 1) / cache.getLineBytes()) * cache.getLineBytes();
  if (vectorBytes + partialBytes + scalarBytes <= cache.getSizeBytes()) return 0;
  // A cyclic scan of a vector panel larger than the remaining L1 working
  // set cannot assume that the next row microtile reuses every line. Charge
  // another scan for each repeated traversal. This is a traffic ranking, not
  // an allocation limit, a latency claim or a proof of cache residency.
  int64_t traversals = (panel.shape[panel.row] + tile.shape[panel.row] - 1) / tile.shape[panel.row];
  int64_t lines = (vectorBytes + cache.getLineBytes() - 1) / cache.getLineBytes();
  int64_t registerBytes = facts.target.getVlenBits() / 8;
  int64_t lineGroups = (cache.getLineBytes() + registerBytes - 1) / registerBytes;
  return lines * lineGroups * (traversals - 1);
}

bool canVectorizeEpilogue(ContractFacts &facts, ArrayRef<int64_t> shape, int64_t lanes) {
  if (!facts.destination) return false;
  Builder b(facts.operation->getContext());
  return bool(contractionEpiloguePeak(b, *facts.destination, facts.freeAxes(), shape,
                                    facts.laneAxis, lanes, facts.target));
}

SmallVector<int64_t> epilogueShape(ContractFacts &facts, ArrayRef<int64_t> shape, int64_t lanes) {
  SmallVector<int64_t> piece(shape.size(), 1);
  int64_t span = std::min(lanes, facts.target.getVlenBits() / facts.element().getIntOrFloatBitWidth());
  piece[llvm::find(facts.freeAxes(), facts.laneAxis) - facts.freeAxes().begin()] = span;
  return piece;
}

int64_t outputEpilogueGroups(ContractFacts &facts, ArrayRef<int64_t> whole, int64_t lanes) {
  if (!facts.destination || facts.destination->adapters.empty()) return 0;
  auto shape = epilogueShape(facts, whole, lanes);
  lanes = *llvm::max_element(shape);
  if (!canVectorizeEpilogue(facts, shape, lanes)) return 0;
  Builder b(facts.operation->getContext());
  auto issue = contractionEpiloguePeak(b, *facts.destination, facts.freeAxes(), shape,
                                     facts.laneAxis, lanes, facts.target);
  if (!issue) return 0;
  int64_t peak = *issue;
  const int64_t elementBits = facts.element().getIntOrFloatBitWidth();
  const int64_t originalLanes = facts.target.getVlenBits();
  // The completed tile remains live while each small epilogue slice runs.
  // Tuple projection and aligned LMUL splitting alias those registers. The
  // separate epilogue working set prevents exp/select temporaries from
  // imposing their issue width on the contraction's reduction loop.
  auto elements = staticProduct(whole);
  int64_t retained = *elements * elementBits / originalLanes;
  retained = std::max<int64_t>(retained, *elements / lanes);
  return retained + peak;
}

// Compare actual issue work for the same author output window. Instruction
// count and register-width work are separate: m8 is one load, but transfers
// eight registers. This accounts for both the front end and the vector pipe
// without treating LMUL as free arithmetic or treating every register as an
// independently issued instruction.
enum class TileRejection { Unsupported, ResourceLimit, UnrepresentableCost };

std::optional<TileChoice> rejectTile(TileRejection &reason, TileRejection value) {
  reason = value;
  return std::nullopt;
}

std::optional<TileChoice> costTile(ContractFacts &facts, ArrayRef<int64_t> shape,
                                  int64_t lmul, int64_t surrounding,
                                  TileRejection &rejection) {
  const auto &target = facts.target;
  const int64_t bits = facts.element().getIntOrFloatBitWidth();
  const int64_t lanes = target.getVlenBits() * lmul / (bits * 8);
  Builder builder(facts.operation->getContext());
  auto legalCarrier = [&](int64_t sew, int64_t width) {
    auto one = builder.getDenseI64ArrayAttr({1});
    auto layout = riscv::LayoutAttr::get(builder.getContext(), "rvv",
        builder.getDenseI64ArrayAttr({facts.axis}), one, builder.getDenseI64ArrayAttr({lanes}),
        one, one, one, sew, width, lanes, (width + 7) / 8, "full");
    return riscv::supportsRVVLayout(target, layout);
  };
  if (lanes <= 1 || !legalCarrier(bits, lmul)) return std::nullopt;
  TileChoice choice;
  choice.shape.assign(shape.begin(), shape.end());
  choice.lanes = lanes;
  choice.accumulatorLMUL = lmul;
  auto outputCount = staticProduct(shape);
  if (!outputCount) return rejectTile(rejection, TileRejection::UnrepresentableCost);
  const int64_t outputs = *outputCount;
  const int64_t accumulatorGroups = (lmul + 7) / 8 * outputs;
  int64_t inputGroups = 0, computeInputs = 0, conversionPeak = 0;
  int64_t transferWork = 0, conversionWork = 0;
  for (unsigned input = 0; input < 2; ++input) {
    if (input == 1 && facts.sameInput()) continue;
    const int64_t storageBits = facts.storageTypes[input].getIntOrFloatBitWidth();
    const int64_t storageLMUL = lmul * storageBits / bits;
    if (storageLMUL * bits != lmul * storageBits || !legalCarrier(storageBits, storageLMUL))
      return std::nullopt;
    int64_t copies = 1;
    for (auto [position, axis] : llvm::enumerate(facts.freeAxes()))
      if (llvm::is_contained(facts.inputs[input].getAxisIds().asArrayRef(), axis)) copies *= shape[position];
    const int64_t loaded = copies * ((storageLMUL + 7) / 8);
    const bool promote = !facts.nativeHalf && facts.storageTypes[input] != facts.element();
    const int64_t converted = promote ? copies * ((lmul + 7) / 8) : loaded;
    choice.loads += copies;
    choice.conversions += promote ? copies : 0;
    transferWork += loaded + copies;
    conversionWork += promote ? converted + copies : 0;
    // Loads and exact conversions are emitted in operand order. A dead narrow
    // input does not remain resident during the subsequent operand's load.
    conversionPeak = std::max(conversionPeak, computeInputs + loaded + (promote ? converted : 0));
    inputGroups += loaded;
    computeInputs += converted;
  }
  int64_t supplyPeak = std::max(conversionPeak, computeInputs);
  if (facts.schedule.getPipelineDepth() == 2) {
    // Every unrolled producer crosses the iteration boundary. The scheduler
    // may retire a current issue before filling that part of the next window;
    // it cannot make the old buffered issues disappear at loop entry. Exact
    // widening scratch is charged while the last new issue is formed.
    supplyPeak = std::max(computeInputs * facts.schedule.getUnroll(),
        computeInputs * (facts.schedule.getUnroll() - 1) + conversionPeak);
  }
  // Final reduction's seed exists after the input operands have died. Charging
  // that seed on top of the compute phase rejects a legal 32-register tile.
  choice.registers = std::max(accumulatorGroups + supplyPeak, accumulatorGroups + 1);
  if (choice.registers > target.getVectorRegisters())
    return rejectTile(rejection, TileRejection::ResourceLimit);
  choice.overlappingGroups = std::max<int64_t>(0, surrounding + choice.registers - target.getVectorRegisters());
  choice.traversals = 1;
  for (auto [position, extent] : llvm::enumerate(facts.freeShape())) {
    if (extent < 0) continue;
    auto count = staticProduct({choice.traversals, extent / shape[position] + (extent % shape[position] != 0)});
    if (!count) return rejectTile(rejection, TileRejection::UnrepresentableCost);
    choice.traversals = *count;
  }
  int64_t activeLanes = lanes;
  if (auto extent = facts.staticReductionExtent(facts.axis)) activeLanes = std::min(activeLanes, *extent);
  const int64_t issueWork = transferWork + conversionWork + accumulatorGroups + outputs;
  // Normalize unknown dimensions to one complete chosen tile, while counting
  // every known dimension exactly. A ragged M axis does not make padding on
  // a fixed N=4 cohort disappear (a 3-column tile still executes twice).
  SmallVector<int64_t> useful;
  for (auto [position, extent] : llvm::enumerate(facts.freeShape()))
    useful.push_back(extent > 0 ? extent : shape[position]);
  auto count = staticProduct(useful);
  auto work = staticProduct({issueWork * facts.schedule.getUnroll() + 2, choice.traversals});
  auto products = count ? staticProduct({*count, activeLanes, facts.schedule.getUnroll()}) : std::nullopt;
  if (!work || !products) return rejectTile(rejection, TileRejection::UnrepresentableCost);
  choice.work = *work;
  choice.products = *products;
  return choice;
}

std::optional<TileChoice> costOutputTile(ContractFacts &facts, ArrayRef<int64_t> shape,
                                        int64_t lmul, int64_t surrounding, bool finalEpilogue,
                                        int64_t scalarSteps, TileRejection &rejection) {
  const int64_t bits = facts.element().getIntOrFloatBitWidth();
  const int64_t lanes = facts.target.getVlenBits() * lmul / (bits * 8);
  auto lanePosition = llvm::find(facts.freeAxes(), facts.laneAxis) - facts.freeAxes().begin();
  if (lanes <= 1 || shape[lanePosition] % lanes || shape[lanePosition] / lanes > 4) return std::nullopt;
  Builder b(facts.operation->getContext());
  SmallVector<int64_t> ones(shape.size(), 1), laneFactors(ones), replicas(shape);
  laneFactors[lanePosition] = lanes;
  replicas[lanePosition] = shape[lanePosition] / lanes;
  auto count = staticProduct(replicas);
  if (!count) return rejectTile(rejection, TileRejection::UnrepresentableCost);
  const int64_t groups = (lmul + 7) / 8;
  auto unit = b.getDenseI64ArrayAttr(ones);
  auto layout = riscv::LayoutAttr::get(b.getContext(), "rvv", b.getDenseI64ArrayAttr(facts.freeAxes()),
      unit, b.getDenseI64ArrayAttr(laneFactors), b.getDenseI64ArrayAttr(replicas), unit, unit,
      bits, lmul, lanes, *count * groups, "tail");
  if (!riscv::supportsRVVLayout(facts.target, layout)) return std::nullopt;
  TileChoice choice;
  choice.shape.assign(shape.begin(), shape.end());
  choice.lanes = lanes;
  choice.accumulatorLMUL = lmul;
  int64_t vectorCopies = replicas[lanePosition], scalarCopies = 1;
  for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
    if (axis == facts.laneAxis) continue;
    if (llvm::is_contained(facts.inputs[facts.laneInput].getAxisIds().asArrayRef(), axis)) vectorCopies *= shape[position];
    if (llvm::is_contained(facts.inputs[1 - facts.laneInput].getAxisIds().asArrayRef(), axis)) scalarCopies *= shape[position];
  }
  // Scalar replicas consume scalar registers even though they are absent from
  // the RVV group budget. Keep a bounded supply frontier and count every load.
  if (scalarCopies > 8) return std::nullopt;
  // Packets are selected for the compute phase with a private vector panel.
  // Direct external-memory traversal keeps one scalar step: its independent
  // dynamic strides would otherwise remain live for the whole packet.
  if (scalarSteps > (finalEpilogue ? 1 : scalarSupplySteps(facts, scalarCopies))) return std::nullopt;
  choice.scalarSteps = scalarSteps;
  int64_t storageBits = facts.storageTypes[facts.laneInput].getIntOrFloatBitWidth();
  int64_t storageLMUL = lmul * storageBits / bits;
  if (storageLMUL * bits != lmul * storageBits ||
      !llvm::is_contained(facts.target.getLegalLMULEighths().asArrayRef(), storageLMUL)) return std::nullopt;
  bool promote = !facts.nativeHalf && facts.storageTypes[facts.laneInput] != facts.element();
  int64_t loaded = vectorCopies * ((storageLMUL + 7) / 8);
  int64_t supplied = promote ? vectorCopies * groups : loaded;
  choice.registers = *count * groups + supplied + (promote ? loaded : 0);
  if (facts.schedule.getPipelineDepth() == 2) {
    int64_t issues = scalarSteps * facts.schedule.getUnroll();
    choice.registers = *count * groups + supplied * issues + (promote ? loaded : 0);
  }
  if (finalEpilogue)
    choice.registers = std::max(choice.registers, outputEpilogueGroups(facts, shape, lanes));
  if (choice.registers > facts.target.getVectorRegisters())
    return rejectTile(rejection, TileRejection::ResourceLimit);
  choice.overlappingGroups = std::max<int64_t>(0, surrounding + choice.registers - facts.target.getVectorRegisters());
  choice.loads = vectorCopies + scalarCopies;
  choice.conversions = (promote ? vectorCopies : 0) +
      (!facts.nativeHalf && facts.storageTypes[1 - facts.laneInput] != facts.element() ? scalarCopies : 0);
  choice.traversals = 1;
  SmallVector<int64_t> useful;
  for (auto [position, extent] : llvm::enumerate(facts.freeShape())) {
    useful.push_back(extent > 0 ? extent : shape[position]);
    if (extent > 0) {
      auto traversals = staticProduct({choice.traversals, (extent + shape[position] - 1) / shape[position]});
      if (!traversals) return rejectTile(rejection, TileRejection::UnrepresentableCost);
      choice.traversals = *traversals;
    }
  }
  // Every scalar row has its own address recurrence. Supplying adjacent K
  // values reduces loop control, but does not remove those row addresses;
  // column replicas reuse both the scalar value and its address work.
  const int64_t issues = choice.scalarSteps * facts.schedule.getUnroll();
  auto work = staticProduct({(choice.loads + loaded + scalarCopies + choice.conversions + *count * (groups + 1)) * issues +
                            scalarCopies + vectorCopies + 2,
                            choice.traversals});
  auto elements = staticProduct(useful);
  auto products = elements ? staticProduct({*elements, issues}) : std::nullopt;
  if (!work || !products) return rejectTile(rejection, TileRejection::UnrepresentableCost);
  choice.work = *work;
  choice.products = *products;
  return choice;
}

FailureOr<TileChoice> chooseTile(ContractFacts &facts, std::optional<size_t> singletonAxis = {},
                               bool finalEpilogue = true) {
  if (facts.schedule.getUnroll() < 1 || facts.schedule.getUnroll() > 16 ||
      (facts.schedule.getPipelineDepth() != 1 && facts.schedule.getPipelineDepth() != 2) ||
      facts.schedule.getBufferCount() != facts.schedule.getPipelineDepth())
    return facts.operation->emitError("dense contraction requires bounded unroll and a one- or two-stage schedule"), failure();
  SmallVector<Value> replaceable;
  for (Value value : facts.operation->getOperands().take_front(2))
    if (riscv::hasUniqueRegisterUse(value)) replaceable.push_back(registerStorageIdentity(value));
  if (facts.destination)
    for (Value value : facts.destination->inputs)
      if (riscv::hasUniqueRegisterUse(value)) replaceable.push_back(registerStorageIdentity(value));
  auto surrounding = surroundingRegisterGroups(facts.operation, replaceable);
  if (!surrounding)
    return facts.operation->emitError("dense contraction surrounding residency is not representable"), failure();
  std::optional<TileChoice> best;
  bool resourceRejected = false;
  bool unrepresentableCost = false;
  SmallVector<int64_t> dimensions(facts.freeShape().size(), 1);
  const int64_t bits = facts.element().getIntOrFloatBitWidth();
  int64_t maximum = std::numeric_limits<int64_t>::max();
  for (auto type : facts.inputs)
    maximum = std::min(maximum, facts.maximumLMUL * bits / type.getElementType().getIntOrFloatBitWidth());
  auto better = [&](const TileChoice &candidate) {
    if (!best) return true;
    if ((candidate.overlappingGroups == 0) != (best->overlappingGroups == 0))
      return candidate.overlappingGroups == 0;
    const __int128 left = __int128(candidate.work) * best->products;
    const __int128 right = __int128(best->work) * candidate.products;
    if (left != right) return left < right;
    return std::tie(candidate.traversals, candidate.loads, candidate.registers, candidate.accumulatorLMUL) <
           std::tie(best->traversals, best->loads, best->registers, best->accumulatorLMUL);
  };
  int64_t visited = 0;
  std::function<void(size_t)> visit = [&](size_t axis) {
    if (axis != dimensions.size()) {
      if (singletonAxis == axis) {
        dimensions[axis] = 1;
        visit(axis + 1);
        return;
      }
      if (!facts.reductionLanes() && facts.freeAxes()[axis] == facts.laneAxis) {
        dimensions[axis] = 1;
        visit(axis + 1);
        return;
      }
      if (!facts.reductionLanes() &&
          llvm::is_contained(facts.inputs[0].getAxisIds().asArrayRef(), facts.freeAxes()[axis]) &&
          llvm::is_contained(facts.inputs[1].getAxisIds().asArrayRef(), facts.freeAxes()[axis])) {
        // A shared batch coordinate addresses a different pair of matrices;
        // it supplies no cross-row reuse. Keep it as one physical traversal.
        dimensions[axis] = 1;
        visit(axis + 1);
        return;
      }
      int64_t previous = 0;
      for (int64_t size = 1; size <= 8; ++size) {
        int64_t extent = facts.freeShape()[axis] < 0 ? size : std::min(size, facts.freeShape()[axis]);
        if (extent == previous) continue;
        previous = dimensions[axis] = extent;
        visit(axis + 1);
      }
      return;
    }
    // Enumerate row replicas, column replicas and the target's legal LMULs.
    // Column replicas reuse scalar operands without widening each instruction.
    for (int64_t lmul : facts.target.getLegalLMULEighths().asArrayRef()) {
      if (lmul > maximum) continue;
      for (int64_t copies : {1, 2, 4}) {
        if (copies > facts.operation->getParentOfType<riscv::KernelOp>().getContractionColumnCopies()) continue;
        if (facts.reductionLanes() && copies != 1) continue;
        auto shape = dimensions;
        if (!facts.reductionLanes())
          shape[llvm::find(facts.freeAxes(), facts.laneAxis) - facts.freeAxes().begin()] =
              facts.target.getVlenBits() * lmul / (bits * 8) * copies;
        for (int64_t steps : {1, 2, 4}) {
          if ((facts.reductionLanes() || finalEpilogue) && steps != 1) continue;
          if (++visited > 30000) continue;
          TileRejection rejection = TileRejection::Unsupported;
          auto candidate = facts.reductionLanes() ? costTile(facts, shape, lmul, *surrounding, rejection)
              : costOutputTile(facts, shape, lmul, *surrounding, finalEpilogue, steps, rejection);
          if (!candidate) {
            resourceRejected |= rejection == TileRejection::ResourceLimit;
            unrepresentableCost |= rejection == TileRejection::UnrepresentableCost;
          }
          if (candidate && better(*candidate)) best = std::move(candidate);
        }
      }
    }
  };
  visit(0);
  if (visited > 30000)
    return facts.operation->emitError("dense contraction candidate domain exceeds its explicit bound"), failure();
  if (!best) {
    if (unrepresentableCost)
      return facts.operation->emitError("dense contraction candidate cost is not representable"), failure();
    if (resourceRejected)
      return rejectRISCVCandidate(facts.operation, RISCVFailureCategory::ResourceLimit,
          "dense contraction has no tile within the joint register budget"), failure();
    return rejectRISCVCandidate(facts.operation, RISCVFailureCategory::Unsupported,
        "dense contraction has no supported tile for its target layout and supply binding"), failure();
  }
  return *best;
}

struct RepresentationWork {
  // Each tile counter fits signed 64 bits, but comparing two panel ratios can
  // multiply four such counters. Keep the ranking exact without signed
  // overflow when source extents are large; this is only a work estimate.
  llvm::APInt work, products;

  explicit RepresentationWork(const TileChoice &tile)
      : work(256, tile.work), products(256, tile.products) {}

  bool lessThan(const RepresentationWork &other) const {
    return (work * other.products).ult(other.work * products);
  }
};

FailureOr<RepresentationWork> representationWork(ContractFacts &facts, const TileChoice &completed) {
  auto panel = choosePanel(facts, completed);
  if (!panel) return RepresentationWork(completed);
  auto compute = chooseTile(facts, {}, false);
  if (failed(compute)) return failure();
  // This is the phase that actually repeats over K. The final epilogue has
  // separate residency, so its smaller tile must not price the compute loop.
  const int64_t groups = (compute->accumulatorLMUL + 7) / 8;
  const int64_t outputs = *staticProduct(panel->shape);
  // Price the same finite transfer family that later writes the copy plan.
  // The old formula reused the FMA's LMUL and counted a unit read/write even
  // when packing transposed rows through a different segmented memory issue.
  // Dynamic stride branches remain runtime facts; rank the source's declared
  // innermost axis without asserting it will be contiguous on every call.
  auto sourceAxes = facts.inputs[facts.laneInput].getAxisIds().asArrayRef();
  int64_t copyAxis = sourceAxes.back();
  riscv::PanelTransferGeometry transfer{facts.storageTypes[facts.laneInput],
      {facts.axis, facts.laneAxis}, facts.target};
  Builder b(facts.operation->getContext());
  auto selectedTransfer = selectPanelTransferCase(b, transfer, copyAxis, facts.maximumLMUL,
      facts.operation->getParentOfType<riscv::KernelOp>().getPanelTransferFields(),
      copyAxis != facts.laneAxis && facts.target.getHasSegmentMemory(),
      facts.operation->getParentOfType<riscv::KernelOp>().getPanelTransferStrategy());
  if (failed(selectedTransfer))
    return facts.operation->emitError("contraction panel has no supported transfer issue"), failure();
  int64_t inner = copyAxis == facts.axis ? panel->reduction : panel->shape[panel->lane];
  int64_t outer = copyAxis == facts.axis ? panel->shape[panel->lane] : panel->reduction;
  auto transfers = panelTransferWork(*selectedTransfer, outer, inner);
  auto transferWork = transfers ? transfers->total() : std::nullopt;
  if (!transferWork) return facts.operation->emitError("contraction panel transfer cost is not representable"), failure();
  // The completed partial is independent of the input-copy representation.
  const int64_t traffic = *transferWork +
      2 * (groups + 1) * ((outputs + compute->lanes - 1) / compute->lanes) +
      panelCacheRefills(facts, *compute, *panel);
  RepresentationWork result(*compute);
  result.work = result.work * panel->reduction * outputs + llvm::APInt(256, traffic) * compute->products;
  result.products *= panel->reduction;
  result.products *= outputs;
  return result;
}

FailureOr<TileChoice> chooseRepresentation(ContractFacts &facts) {
  auto selected = chooseTile(facts);
  if (failed(selected) || !facts.reductionLanes() || !facts.destination || facts.reductionAxes.size() != 1 ||
      facts.freeAxes().size() != 2 || !staticProduct(facts.freeShape())) return selected;
  auto bestWork = representationWork(facts, *selected);
  if (failed(bestWork)) return failure();
  ContractFacts chosen(facts);
  // For two inputs with innermost K axes, a private transpose can trade K lanes for
  // either output coordinate. A naturally available output lane is retained:
  // exchanging it would require a strided scalar-K supply whose cache/TLB
  // traffic is not represented by this finite panel family.
  for (unsigned input : {1u, 0u}) {
    auto axes = facts.inputs[input].getAxisIds().asArrayRef();
    auto other = facts.inputs[1 - input].getAxisIds().asArrayRef();
    if (axes.size() != 2 || other.size() != 2) continue;
    for (int64_t axis : axes) {
      if (facts.isReduction(axis) || llvm::is_contained(other, axis)) continue;
      size_t position = llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin();
      if (facts.freeShape()[position] < 32 || facts.freeShape()[1 - position] < 16) continue;
      ContractFacts candidate(facts);
      candidate.laneAxis = axis;
      candidate.laneInput = input;
      auto tile = chooseTile(candidate);
      if (failed(tile)) return failure();
      auto panel = choosePanel(candidate, *tile);
      if (!panel) continue;
      auto work = representationWork(candidate, *tile);
      if (failed(work)) return failure();
      if (!work->lessThan(*bestWork)) continue;
      chosen = std::move(candidate);
      selected = *tile;
      bestWork = std::move(work);
    }
  }
  facts = std::move(chosen);
  return selected;
}

constexpr llvm::StringLiteral contractionPlanName = "weft.riscv.dense_contraction";

riscv::ContractionEpilogueAttr selectEpilogue(ContractFacts facts, const TileChoice &tile) {
  Builder b(facts.operation->getContext());
  auto scalar = [&] {
    return riscv::ContractionEpilogueAttr::get(b.getContext(), "scalar",
        b.getDenseI64ArrayAttr(tile.shape), 0, 1, 0, 0);
  };
  if (!facts.destination || facts.freeAxes().empty()) return scalar();
  if (!facts.reductionLanes() && facts.destination->adapters.empty())
    return riscv::ContractionEpilogueAttr::get(b.getContext(), "direct",
        b.getDenseI64ArrayAttr(tile.shape), facts.laneAxis, tile.lanes,
        tile.accumulatorLMUL,
        *staticProduct(tile.shape) / tile.lanes * ((tile.accumulatorLMUL + 7) / 8));
  int64_t lanes = tile.lanes;
  int64_t lmul = tile.accumulatorLMUL;
  if (facts.reductionLanes()) {
    if (facts.destination->adapters.empty()) return scalar();
    facts.laneAxis = facts.freeAxes().back();
    lanes = lmul = 0;
    int64_t width = tile.shape.back();
    int64_t bits = facts.element().getIntOrFloatBitWidth();
    for (int64_t candidate : facts.target.getLegalLMULEighths().asArrayRef()) {
      int64_t span = facts.target.getVlenBits() * candidate / (8 * bits);
      if (candidate <= 8 && candidate <= facts.maximumLMUL && span > 1 && width % span == 0) {
        lmul = candidate;
        lanes = span;
      }
    }
    if (!lmul) return scalar();
  }
  auto shape = epilogueShape(facts, tile.shape, lanes);
  lanes = *llvm::max_element(shape);
  lmul = 8 * lanes * facts.element().getIntOrFloatBitWidth() / facts.target.getVlenBits();
  int64_t groups = outputEpilogueGroups(facts, tile.shape, lanes);
  if (!canVectorizeEpilogue(facts, shape, lanes) || groups > facts.target.getVectorRegisters()) return scalar();
  return riscv::ContractionEpilogueAttr::get(b.getContext(), "vector", b.getDenseI64ArrayAttr(shape),
      facts.laneAxis, lanes, lmul, groups);
}

FailureOr<riscv::ContractionCaseAttr> selectCase(ContractFacts facts,
                                               std::optional<size_t> singleton = {}) {
  Builder b(facts.operation->getContext());
  if (singleton) facts.selectNaturalRepresentation(facts.freeAxes()[*singleton]);
  auto tile = singleton ? chooseTile(facts, singleton) : chooseRepresentation(facts);
  if (failed(tile)) return failure();
  auto selected = tile->attribute(facts);
  auto directEpilogue = selectEpilogue(facts, *tile);
  auto panel = singleton ? std::optional<PanelChoice>{} : choosePanel(facts, *tile);
  if (!panel)
    return riscv::ContractionCaseAttr::get(b.getContext(), singleton ? facts.freeAxes()[*singleton] : 0,
        selected, b.getDenseI64ArrayAttr({}), 0, selected, selected, directEpilogue, directEpilogue);
  auto update = chooseTile(facts, {}, false);
  if (failed(update)) return failure();
  auto updateSelected = update->attribute(facts);
  auto output = *tile;
  int64_t storeAxis = facts.destination->store.getRegion().getType().getAxisIds().asArrayRef().back();
  if (storeAxis != facts.laneAxis) {
    bool first = llvm::is_contained(facts.inputs[0].getAxisIds().asArrayRef(), storeAxis);
    bool second = llvm::is_contained(facts.inputs[1].getAxisIds().asArrayRef(), storeAxis);
    if (first != second) {
      facts.laneAxis = storeAxis;
      facts.laneInput = first ? 0 : 1;
      auto store = chooseTile(facts);
      if (failed(store)) return failure();
      output = *store;
    }
  }
  if (!facts.destination->adapters.empty()) {
    auto piece = epilogueShape(facts, output.shape, output.lanes);
    int64_t lanes = *llvm::max_element(piece);
    if (canVectorizeEpilogue(facts, piece, lanes)) {
      output.shape = std::move(piece);
      output.lanes = lanes;
      output.accumulatorLMUL = 8 * lanes * facts.element().getIntOrFloatBitWidth() / facts.target.getVlenBits();
    }
  }
  auto outputEpilogue = selectEpilogue(facts, output);
  // The panel output phase reloads completed numbers; it does not execute
  // another dot. Its tile witness must describe those transfers and the
  // epilogue, rather than retaining counters from the earlier compute tile.
  int64_t elements = *staticProduct(output.shape);
  int64_t parts = elements / output.lanes;
  int64_t rootGroups = parts * ((output.accumulatorLMUL + 7) / 8);
  output.loads = parts;
  output.conversions = 0;
  output.scalarSteps = 1;
  output.registers = rootGroups;
  if (outputEpilogue.getMode() == "scalar") {
    output.registers = rootGroups + 1;
    output.loads += elements * facts.destination->inputs.size();
    for (Operation *adapter : facts.destination->adapters)
      if (isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp>(adapter)) output.conversions += elements;
  } else if (outputEpilogue.getMode() == "vector") {
    output.registers = std::max(rootGroups, outputEpilogue.getRegisters());
    int64_t issues = elements / *staticProduct(outputEpilogue.getShape().asArrayRef());
    for (Value input : facts.destination->inputs) {
      auto piece = contractionEpiloguePiece(b, input.getType(), facts.freeAxes(),
          outputEpilogue.getShape().asArrayRef(), outputEpilogue.getLaneAxis(), outputEpilogue.getLanes(), facts.target);
      auto coordinates = riscv::RegisterCoordinates::get(piece.getLayout());
      output.loads += issues * coordinates->parts();
    }
    for (Operation *adapter : facts.destination->adapters)
      if (isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp>(adapter)) {
        auto piece = contractionEpiloguePiece(b, adapter->getResult(0).getType(), facts.freeAxes(),
            outputEpilogue.getShape().asArrayRef(), outputEpilogue.getLaneAxis(), outputEpilogue.getLanes(), facts.target);
        output.conversions += issues * riscv::RegisterCoordinates::get(piece.getLayout())->parts();
      }
  }
  return riscv::ContractionCaseAttr::get(b.getContext(), 0, selected,
      b.getDenseI64ArrayAttr(panel->shape), panel->reduction, updateSelected, output.attribute(facts),
      directEpilogue, outputEpilogue);
}

FailureOr<riscv::ContractionPlanAttr> selectPlan(ContractFacts &facts) {
  Builder b(facts.operation->getContext());
  SmallVector<Attribute> cases;
  if (facts.destination)
    for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
      if (facts.freeShape()[position] == 1) continue;
      bool lhs = llvm::is_contained(facts.inputs[0].getAxisIds().asArrayRef(), axis);
      bool rhs = llvm::is_contained(facts.inputs[1].getAxisIds().asArrayRef(), axis);
      if (lhs == rhs) continue;
      auto selected = selectCase(facts, position);
      if (failed(selected)) return failure();
      cases.push_back(*selected);
    }
  auto general = selectCase(facts);
  if (failed(general)) return failure();
  cases.push_back(*general);
  return riscv::ContractionPlanAttr::get(b.getContext(), b.getDenseI64ArrayAttr(facts.freeAxes()),
      b.getDenseI64ArrayAttr(facts.reductionAxes), b.getArrayAttr(cases), facts.maximumLMUL,
      facts.schedule.getUnroll(), facts.schedule.getPipelineDepth(), bool(facts.destination));
}

LogicalResult verifySelectedPlan(ContractFacts &facts, riscv::ContractionPlanAttr plan) {
  auto fail = [&](StringRef message) { return facts.operation->emitError(message); };
  if (plan.getOutputAxes().asArrayRef() != facts.freeAxes() ||
      plan.getReductionAxes().asArrayRef() != ArrayRef<int64_t>(facts.reductionAxes) ||
      plan.getMaximumLMUL() != facts.maximumLMUL ||
      plan.getUnroll() != facts.schedule.getUnroll() ||
      plan.getPipelineDepth() != facts.schedule.getPipelineDepth() ||
      plan.getStreamedStore() != bool(facts.destination))
    return fail("dense contraction selection disagrees with its current numerical graph, store frontier or binding");
  SmallVector<int64_t> expectedSingletons;
  if (facts.destination)
    for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
      bool first = llvm::is_contained(facts.inputs[0].getAxisIds().asArrayRef(), axis);
      bool second = llvm::is_contained(facts.inputs[1].getAxisIds().asArrayRef(), axis);
      if (facts.freeShape()[position] != 1 && first != second) expectedSingletons.push_back(axis);
    }
  expectedSingletons.push_back(0);
  if (plan.getCases().size() != expectedSingletons.size())
    return fail("dense contraction selection has an incomplete runtime extent partition");
  for (auto [attribute, singleton] : llvm::zip(plan.getCases(), expectedSingletons)) {
    auto selected = cast<riscv::ContractionCaseAttr>(attribute);
    if (selected.getSingletonAxis() != singleton)
      return fail("dense contraction selection reordered its extent predicates");
    for (auto tile : {selected.getTile(), selected.getUpdateTile(), selected.getOutputTile()}) {
      int64_t bits = facts.element().getIntOrFloatBitWidth();
      if (tile.getLanes() * bits * 8 != facts.target.getVlenBits() * tile.getLmulEighths() ||
          !llvm::is_contained(facts.target.getLegalLMULEighths().asArrayRef(), tile.getLmulEighths()) ||
          tile.getRegisters() > facts.target.getVectorRegisters())
        return fail("dense contraction tile is not legal for the exact target VLEN and resource file");
      for (auto input : facts.inputs)
        if (tile.getLmulEighths() * input.getElementType().getIntOrFloatBitWidth() > facts.maximumLMUL * bits)
          return fail("dense contraction tile exceeds its explicit input LMUL binding");
      int64_t vectorCopies = 1;
      for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
        int64_t size = tile.getShape()[position];
        if (axis == tile.getLaneAxis()) {
          vectorCopies = size / tile.getLanes();
          if (size % tile.getLanes() || !llvm::is_contained(ArrayRef<int64_t>({1, 2, 4}), vectorCopies) ||
              vectorCopies > facts.operation->getParentOfType<riscv::KernelOp>().getContractionColumnCopies())
            return fail("dense contraction column replicas are outside the declared finite target domain");
        } else if (size > 8) {
          return fail("dense contraction scalar replicas are outside the declared finite target domain");
        }
      }
      if (tile.getLaneAxis() != facts.axis) {
        auto axes = facts.inputs[tile.getLaneInput()].getAxisIds().asArrayRef();
        auto other = facts.inputs[1 - tile.getLaneInput()].getAxisIds().asArrayRef();
        auto position = llvm::find(facts.freeAxes(), tile.getLaneAxis()) - facts.freeAxes().begin();
        if (facts.reductionAxes.size() != 1 || !llvm::is_contained(axes, tile.getLaneAxis()) ||
            llvm::is_contained(other, tile.getLaneAxis()) ||
            tile.getShape()[position] % tile.getLanes())
          return fail("dense contraction output lanes have no exact one-sided operand projection");
      }
      if (singleton) {
        auto position = llvm::find(facts.freeAxes(), singleton) - facts.freeAxes().begin();
        if (tile.getShape()[position] != 1)
          return fail("dense contraction singleton case must have a single physical output coordinate");
      }
    }
    // Validate the chosen point in the finite domain. This computes its
    // resources, never searches for a replacement. A malformed selection must
    // fail instead of being quietly replaced by the compiler's favorite tile.
    for (bool update : {false, true}) {
      if (update && selected.getPanelShape().empty()) continue;
      auto tile = update ? selected.getUpdateTile() : selected.getTile();
      ContractFacts phase(facts);
      phase.laneAxis = tile.getLaneAxis();
      phase.laneInput = tile.getLaneInput();
      TileRejection rejection = TileRejection::Unsupported;
      auto cost = phase.reductionLanes()
          ? costTile(phase, tile.getShape().asArrayRef(), tile.getLmulEighths(), 0, rejection)
          : costOutputTile(phase, tile.getShape().asArrayRef(), tile.getLmulEighths(), 0,
                           !update, tile.getScalarSteps(), rejection);
      if (!cost || cost->lanes != tile.getLanes() || cost->registers != tile.getRegisters() ||
          cost->loads != tile.getLoads() || cost->conversions != tile.getConversions())
        return fail("dense contraction selection does not match its exact typed supply and residency cost");
    }
    for (bool panelOutput : {false, true}) {
      auto epilogue = panelOutput ? selected.getEpilogue() : selected.getDirectEpilogue();
      auto output = panelOutput ? selected.getOutputTile() : selected.getTile();
      if (epilogue.getMode() == "direct" && (!facts.destination || !facts.destination->adapters.empty() ||
          output.getLaneAxis() == facts.axis || epilogue.getShape() != output.getShape() ||
          epilogue.getLaneAxis() != output.getLaneAxis() || epilogue.getLanes() != output.getLanes() ||
          epilogue.getLmulEighths() != output.getLmulEighths()))
        return fail("direct contraction output may only store its unchanged completed vector tile");
      if (epilogue.getMode() == "vector") {
        if (!facts.destination || facts.destination->adapters.empty())
          return fail("vector contraction epilogue requires its proven pointwise consumer graph");
        for (auto [whole, piece] : llvm::zip(output.getShape().asArrayRef(), epilogue.getShape().asArrayRef()))
          if (whole % piece) return fail("selected epilogue issues do not partition their completed output tile");
        int64_t bits = facts.element().getIntOrFloatBitWidth();
        if (epilogue.getLanes() * bits * 8 != facts.target.getVlenBits() * epilogue.getLmulEighths() ||
            !llvm::is_contained(facts.target.getLegalLMULEighths().asArrayRef(), epilogue.getLmulEighths()))
          return fail("selected epilogue lane representation differs from the target vector geometry");
        Builder b(facts.operation->getContext());
        auto peak = contractionEpiloguePeak(b, *facts.destination, facts.freeAxes(),
            epilogue.getShape().asArrayRef(), epilogue.getLaneAxis(), epilogue.getLanes(), facts.target);
        if (!peak || *peak > epilogue.getRegisters())
          return fail("selected epilogue understates its numeric consumer residency");
      }
    }
    if (!selected.getPanelShape().empty()) {
      auto tile = selected.getTile();
      if (tile.getLaneAxis() == facts.axis || facts.reductionAxes.size() != 1)
        return fail("dense contraction private panel requires output lanes and one reduction coordinate");
      int64_t lane = llvm::find(facts.freeAxes(), tile.getLaneAxis()) - facts.freeAxes().begin();
      auto kernel = facts.operation->getParentOfType<riscv::KernelOp>();
      if (selected.getPanelK() > kernel.getContractionPanelK())
        return fail("dense contraction panel exceeds the instantiated finite K binding");
      unsigned rows = 0;
      for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
        bool first = llvm::is_contained(facts.inputs[tile.getLaneInput()].getAxisIds().asArrayRef(), axis);
        bool second = llvm::is_contained(facts.inputs[1 - tile.getLaneInput()].getAxisIds().asArrayRef(), axis);
        int64_t size = selected.getPanelShape()[position];
        if (size > 64 || (facts.freeShape()[position] > 0 && size > facts.freeShape()[position]))
          return fail("dense contraction panel exceeds its existing author output window");
        if (first && second) {
          if (size != 1) return fail("contraction panel cannot share supplies across different batch coordinates");
        } else if (second) ++rows;
        else if (!first || axis != tile.getLaneAxis())
          return fail("contraction panel has no exact row and column operand projections");
      }
      if (rows != 1) return fail("contraction panel requires one scalar row projection");
      auto outputs = staticProduct(selected.getPanelShape().asArrayRef());
      auto packed = staticProduct({selected.getPanelK(), selected.getPanelShape()[lane],
          facts.storageTypes[tile.getLaneInput()].getIntOrFloatBitWidth() / 8});
      // Previously materialized contractions own separate lexical panel
      // lifetimes. Summing their allocations here would invalidate a saved
      // second selection merely because the first one has been expanded.
      // Check this phase's capacity; the final storage-liveness pass verifies
      // simultaneous author snapshots and compiler allocations together.
      int64_t budget = std::min<int64_t>(64 * 1024, facts.target.getMaxPrivateStackBytes());
      if (!outputs || !packed ||
          *outputs * (facts.element().getIntOrFloatBitWidth() / 8) + *packed + 14 > budget)
        return fail("dense contraction selected panel exceeds its available private representation storage");
    }
  }
  return success();
}

class ContractProgram {
public:
  ContractProgram(ContractFacts &facts, riscv::ContractionPlanAttr plan, IRRewriter &b,
                  int64_t &identity, int64_t &shapeIdentity, llvm::DenseMap<Value, Value> &snapshots)
      : facts(facts), plan(plan), b(b), identity(identity),
        shapeIdentity(shapeIdentity), snapshots(snapshots), loc(facts.operation->getLoc()) {}

  LogicalResult run() {
    b.setInsertionPoint(facts.operation);
    for (unsigned index = 0; index < 2; ++index) {
      auto view = inputView(facts.operation->getOperand(index));
      if (failed(view)) return failure();
      memory[index] = *view;
    }
    if (facts.destination) {
      b.setInsertionPoint(facts.destination->store);
      // Close each epilogue read frontier before expanding any output stores.
      // The destination proof already excludes overlap with these inputs;
      // querying the partially rewritten graph for every microtile would
      // confuse generated stores with the original read-to-use interval and
      // exhaust a bounded alias query as the finite tail program grows.
      for (Value input : facts.destination->inputs) {
        auto descriptor = materializeContractionReadView(b, input, facts.destination->store);
        if (failed(descriptor) || !*descriptor)
          return facts.operation->emitError("contraction epilogue has no stable dense read frontier");
        epilogueMemory.try_emplace(input, *descriptor);
      }
    }
    // Extents belong to this exact descriptor, including a subview or private
    // snapshot. No ancestor Level point substitutes for that memory boundary.
    for (int64_t axis : facts.freeAxes()) {
      unsigned source = llvm::is_contained(cast<riscv::MemDescType>(memory[0].getType()).getAxisIds().asArrayRef(), axis) ? 0 : 1;
      freeExtents.push_back(extent(memory[source], axis));
    }
    for (int64_t axis : facts.reductionAxes) {
      Value bound = arith::MinUIOp::create(b, loc, extent(memory[0], axis), extent(memory[1], axis));
      // A private numeric snapshot reserves the nominal tile capacity. Its
      // initialized domain can still be a shorter active author Level.
      for (unsigned input = 0; input < 2; ++input)
        if (facts.inputs[input].getLayout().getValidity() == "tail")
          if (auto point = originPoint(facts.operation->getOperand(input), axis))
            bound = arith::MinUIOp::create(b, loc, bound, point.getActive());
      reductionExtents.push_back(bound);
    }
    reductionExtent = reductionExtents.back();
    reductionOffsets.assign(facts.reductionAxes.size(), constant(0));
    bool requiresStorage = false;
    for (Attribute attribute : plan.getCases())
      requiresStorage |= cast<riscv::ContractionCaseAttr>(attribute).getTile().getShape().asArrayRef() != facts.freeShape();
    if (requiresStorage && !facts.destination) {
      auto allocated = allocateLocalValue(b, facts.result, facts.operation, identity, false, "physical-share");
      if (failed(allocated)) return failure();
      storage = *allocated;
    }
    offsets.assign(facts.freeShape().size(), constant(0));
    active = freeExtents;
    if (failed(selectedCases(0)))
      return failure();
    b.setInsertionPoint(facts.operation);
    if (facts.destination) {
      b.eraseOp(facts.destination->store);
      for (Operation *adapter : llvm::reverse(facts.destination->adapters)) b.eraseOp(adapter);
      b.eraseOp(facts.operation);
      return success();
    }
    Value value = storage ? storage : result;
    if (value.getType() != facts.outputType) {
      auto source = cast<riscv::ValueType>(value.getType());
      auto conversion = riscv::ConvertLayoutOp::create(b, loc, facts.result, value,
          layoutConversion(b, source.getLayout(), facts.result.getLayout()),
          riscv::AccessAttr(), unselectedLeaf(b));
      copyOrigin(facts.operation, conversion);
      value = conversion;
    }
    b.replaceOp(facts.operation, value);
    return success();
  }

private:
  ContractFacts &facts;
  riscv::ContractionPlanAttr plan;
  riscv::ContractionCaseAttr selectedCase;
  riscv::ContractionEpilogueAttr selectedEpilogue;
  TileChoice choice;
  IRRewriter &b;
  int64_t &identity;
  int64_t &shapeIdentity;
  llvm::DenseMap<Value, Value> &snapshots;
  Location loc;
  Value memory[2], reductionExtent, storage, result;
  SmallVector<Value> freeExtents, offsets, active, reductionExtents, reductionOffsets;
  DenseMap<int64_t, Value> outputExtents[2];
  DenseMap<Value, Value> epilogueMemory;
  Value partialMemory;
  bool tiled = false;

  Value constant(int64_t value) { return arith::ConstantIndexOp::create(b, loc, value); }

  Value extent(Value descriptor, int64_t axis) {
    auto axes = cast<riscv::MemDescType>(descriptor.getType()).getAxisIds().asArrayRef();
    auto position = llvm::find(axes, axis) - axes.begin();
    return riscv::ExtentOp::create(b, loc, b.getIndexType(), descriptor, position);
  }

  FailureOr<Value> inputView(Value value) {
    auto stable = materializeContractionStorageView(b, value, facts.operation);
    if (failed(stable)) return failure();
    if (*stable) return *stable;
    auto type = cast<riscv::ValueType>(value.getType());
    if (type.getLayout().getCarrier() == "local") return localView(b, value, facts.operation);
    auto existing = snapshots.find(value);
    if (existing != snapshots.end()) return existing->second;
    // A numeric birth can feed many later closed dots (for example a staged
    // attention query). Snapshot once at that birth, before intervening writes;
    // do not allocate and copy it anew inside every consumer's K loop.
    OpBuilder::InsertionGuard guard(b);
    Operation *anchor = value.getDefiningOp() ? value.getDefiningOp()->getNextNode()
                                             : &cast<BlockArgument>(value).getOwner()->front();
    b.setInsertionPoint(anchor);
    auto allocated = allocateLocalValue(b, type, anchor, identity, false, "local-value-copy");
    if (failed(allocated)) return failure();
    auto descriptor = localView(b, *allocated, anchor);
    if (failed(descriptor)) return failure();
    auto store = riscv::StoreOp::create(b, loc, value, completeLocalSlice(b, *descriptor),
        unassignedAccess(b), unselectedLeaf(b), riscv::FieldStorePlanAttr());
    copyOrigin(facts.operation, store);
    snapshots.try_emplace(value, *descriptor);
    return *descriptor;
  }

  riscv::ValueType valueType(ArrayRef<int64_t> axes, ArrayRef<int64_t> shape,
                             Type element, int64_t lmul, bool accumulator = false, int64_t laneAxis = 0) {
    if (!laneAxis) laneAxis = facts.laneAxis;
    SmallVector<int64_t> lanes(axes.size(), 1), replicas(axes.size(), 1), one(axes.size(), 1);
    int64_t groups = 1;
    bool vector = false;
    for (size_t position = 0; position < axes.size(); ++position) {
      if (axes[position] == laneAxis && shape[position] > 1) {
        lanes[position] = choice.lanes;
        replicas[position] = shape[position] / choice.lanes;
        groups *= replicas[position];
        vector = true;
      }
      else { replicas[position] = shape[position]; groups *= shape[position]; }
    }
    if (vector) groups *= (lmul + 7) / 8;
    auto unit = b.getDenseI64ArrayAttr(one);
    auto layout = riscv::LayoutAttr::get(b.getContext(), vector ? "rvv" : "scalar", b.getDenseI64ArrayAttr(axes),
        unit, b.getDenseI64ArrayAttr(lanes), b.getDenseI64ArrayAttr(replicas), unit, unit,
        element.getIntOrFloatBitWidth(), vector ? lmul : 0, vector ? choice.lanes : 1,
        vector ? groups : 0, accumulator ? "full" : "tail");
    return riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape),
        b.getDenseI64ArrayAttr(axes), layout);
  }

  Type storageElement(Value descriptor) {
    auto encoding = cast<kernel::EncodingType>(cast<riscv::MemDescType>(descriptor.getType()).getEncoding());
    if (encoding.getFamily() == "f32") return b.getF32Type();
    if (encoding.getFamily() == "f16") return b.getF16Type();
    if (encoding.getFamily() == "bf16") return b.getBF16Type();
    if (encoding.getFamily() == "f64") return b.getF64Type();
    return {};
  }

  Value windowView(Value descriptor, ArrayRef<Value> starts, ArrayRef<Value> lengths) {
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    SmallVector<int64_t> shape(type.getShape().size()), markers(shape.size(), -1);
    SmallVector<int64_t> sizes(markers);
    SmallVector<Value> bounds(starts);
    for (size_t position = 0; position < shape.size(); ++position) {
      if (auto size = getConstantIntValue(lengths[position]); size && *size >= 0) {
        shape[position] = sizes[position] = *size;
        continue;
      }
      auto local = riscv::LocalShapeOp::create(b, loc, b.getIndexType(), lengths[position], ++shapeIdentity);
      shape[position] = -shapeIdentity;
      bounds.push_back(local);
    }
    auto viewType = riscv::MemDescType::get(b.getContext(), type.getEncoding(), b.getDenseI64ArrayAttr(shape),
        type.getAxisIds(), type.getStrides(), type.getOrigins(), type.getAlignment(), "slice", type.getAccess(),
        type.getAliasSet(), type.getLayoutIdentity(), type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    return riscv::SubviewOp::create(b, loc, viewType, descriptor, bounds,
        b.getDenseI64ArrayAttr(markers), b.getDenseI64ArrayAttr(sizes));
  }

  Value tileView(Value descriptor) {
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    SmallVector<Value> starts, lengths;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      if (facts.isReduction(axis)) {
        size_t reduction = llvm::find(facts.reductionAxes, axis) - facts.reductionAxes.begin();
        starts.push_back(constant(0)); lengths.push_back(reductionExtents[reduction]);
      } else {
        size_t outputAxis = llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin();
        bool broadcast = type.getShape()[position] == 1 && facts.freeShape()[outputAxis] != 1;
        starts.push_back(broadcast ? constant(0) : offsets[outputAxis]);
        lengths.push_back(broadcast ? constant(1) : active[outputAxis]);
      }
    }
    return windowView(descriptor, starts, lengths);
  }

  LogicalResult freeLoops(size_t axis) {
    return tileLoops(facts.freeShape(), freeExtents, axis, [&] { return reduction(); });
  }

  LogicalResult tileLoops(ArrayRef<int64_t> shape, ArrayRef<Value> bounds, size_t axis,
                          llvm::function_ref<LogicalResult()> body) {
    if (axis == choice.shape.size()) return body();
    if (!facts.reductionLanes() && facts.destination &&
        facts.destination->store.getRegion().getType().getAliasSet() >= 0)
      return outputTileLoops(shape, bounds, axis, body);
    if (choice.shape[axis] == shape[axis]) return tileLoops(shape, bounds, axis + 1, body);
    // A sole external store cannot observe padded output coordinates. Shared
    // complete numerical values retain their original full representation.
    bool boundedStore = facts.destination && facts.destination->store.getRegion().getType().getAliasSet() >= 0;
    Value upper = shape[axis] < 0 || boundedStore ? bounds[axis] : constant(shape[axis]);
    auto loop = scf::ForOp::create(b, loc, constant(0), upper, constant(choice.shape[axis]));
    loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(facts.operation, loop);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    offsets[axis] = loop.getInductionVar();
    active[axis] = arith::MinUIOp::create(b, loc,
        arith::SubIOp::create(b, loc, bounds[axis],
            arith::MinUIOp::create(b, loc, bounds[axis], offsets[axis])), constant(choice.shape[axis]));
    return tileLoops(shape, bounds, axis + 1, body);
  }

  LogicalResult outputTileLoops(ArrayRef<int64_t> shape, ArrayRef<Value> bounds, size_t axis,
                                 llvm::function_ref<LogicalResult()> body) {
    const int64_t width = choice.shape[axis];
    const bool lane = facts.freeAxes()[axis] == facts.laneAxis;
    Value upper = bounds[axis];
    Value zero = constant(0);
    Value complete = completeDenseExtent(b, loc, upper, width);
    auto savedOffsets = offsets, savedActive = active;
    auto emit = [&](Value begin, Value end, int64_t span, bool full) -> LogicalResult {
      auto first = riscv::denseConstantIndex(begin), last = riscv::denseConstantIndex(end);
      if (begin == end || (first && last && *first == *last)) return success();
      auto loop = scf::ForOp::create(b, loc, begin, end, constant(span));
      loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
      copyOrigin(facts.operation, loop);
      b.setInsertionPointToStart(loop.getBody());
      offsets[axis] = loop.getInductionVar();
      active[axis] = full ? constant(span) : Value(arith::MinUIOp::create(b, loc,
          arith::SubIOp::create(b, loc, end, loop.getInductionVar()), constant(span)));
      choice.shape[axis] = span;
      auto status = tileLoops(shape, bounds, axis + 1, body);
      choice.shape[axis] = width;
      offsets = savedOffsets;
      active = savedActive;
      b.setInsertionPointAfter(loop);
      return status;
    };
    if (width == 1) return emit(zero, upper, 1, true);
    if (failed(emit(zero, complete, width, true))) return failure();
    // Preserve the selected lane carrier. Replica tails use a bounded binary
    // decomposition: a remainder of six rows is a four-row and a two-row
    // microtile, not six traversals of the same K supply. Every generated row
    // is real, and the per-output FMA sequence remains unchanged.
    if (lane) return emit(complete, upper, choice.lanes, false);
    Value begin = complete;
    int64_t span = 1;
    while (span <= (width - 1) / 2) span *= 2;
    for (; span; span /= 2) {
      Value remaining = arith::SubIOp::create(b, loc, upper, begin);
      Value end = span == 1 ? upper : Value(arith::AddIOp::create(b, loc, begin,
          completeDenseExtent(b, loc, remaining, span)));
      if (failed(emit(begin, end, span, true))) return failure();
      begin = end;
    }
    return success();
  }

  FailureOr<Value> privateArray(ArrayRef<int64_t> axes, ArrayRef<int64_t> shape, Type element) {
    auto type = riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape),
        b.getDenseI64ArrayAttr(axes), scalarLayout(b, element, shape, axes));
    return allocateLocalValue(b, type, facts.operation, identity, false, "physical-share");
  }

  LogicalResult panelProgram(const PanelChoice &panel) {
    Value rowCount = arith::MinUIOp::create(b, loc, freeExtents[panel.row], constant(panel.shape[panel.row]));
    Value columnCount = arith::MinUIOp::create(b, loc, freeExtents[panel.lane], constant(panel.shape[panel.lane]));
    Value rows = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
        rowCount, constant(2 * choice.shape[panel.row]));
    Value columns = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
        columnCount, constant(2 * choice.lanes));
    Value terms = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, reductionExtent, constant(64));
    Value profitable = arith::AndIOp::create(b, loc, arith::OrIOp::create(b, loc, rows, columns), terms);
    auto branch = scf::IfOp::create(b, loc, profitable, true);
    copyOrigin(facts.operation, branch);
    auto savedOffsets = offsets, savedActive = active;
    b.setInsertionPointToStart(&branch.getThenRegion().front());
    if (failed(panelLoops(panel, 0))) return failure();
    offsets = savedOffsets; active = savedActive;
    b.setInsertionPointToStart(&branch.getElseRegion().front());
    if (failed(freeLoops(0))) return failure();
    offsets = savedOffsets; active = savedActive;
    b.setInsertionPointAfter(branch);
    return success();
  }

  void activateTile(riscv::ContractionTileAttr tile) {
    choice = TileChoice::read(tile);
    facts.laneAxis = tile.getLaneAxis();
    facts.laneInput = tile.getLaneInput();
    tiled = choice.shape != facts.freeShape();
  }

  std::optional<PanelChoice> selectedPanel() {
    if (selectedCase.getPanelShape().empty()) return std::nullopt;
    PanelChoice panel;
    panel.shape.assign(selectedCase.getPanelShape().asArrayRef().begin(),
                       selectedCase.getPanelShape().asArrayRef().end());
    panel.reduction = selectedCase.getPanelK();
    panel.lane = llvm::find(facts.freeAxes(), facts.laneAxis) - facts.freeAxes().begin();
    for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
      bool first = llvm::is_contained(facts.inputs[facts.laneInput].getAxisIds().asArrayRef(), axis);
      bool second = llvm::is_contained(facts.inputs[1 - facts.laneInput].getAxisIds().asArrayRef(), axis);
      if (first && second) panel.batchAxes.push_back(axis);
      else if (second) panel.row = position;
    }
    return panel;
  }

  LogicalResult selectedCases(size_t position) {
    selectedCase = cast<riscv::ContractionCaseAttr>(plan.getCases()[position]);
    selectedEpilogue = selectedCase.getDirectEpilogue();
    activateTile(selectedCase.getTile());
    if (!selectedCase.getSingletonAxis()) {
      auto panel = selectedPanel();
      return panel ? panelProgram(*panel) : freeLoops(0);
    }
    size_t row = llvm::find(facts.freeAxes(), selectedCase.getSingletonAxis()) - facts.freeAxes().begin();
    Value thin = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, freeExtents[row], constant(1));
    auto branch = scf::IfOp::create(b, loc, thin, true);
    copyOrigin(facts.operation, branch);
    auto savedOffsets = offsets, savedActive = active;
    b.setInsertionPointToStart(&branch.getThenRegion().front());
    if (failed(freeLoops(0))) return failure();
    offsets = savedOffsets; active = savedActive;
    b.setInsertionPointToStart(&branch.getElseRegion().front());
    if (failed(selectedCases(position + 1))) return failure();
    offsets = savedOffsets; active = savedActive;
    b.setInsertionPointAfter(branch);
    return success();
  }


  LogicalResult panelLoops(const PanelChoice &panel, size_t axis) {
    if (axis == panel.shape.size()) return panelBody(panel);
    if (panel.shape[axis] == facts.freeShape()[axis]) return panelLoops(panel, axis + 1);
    auto loop = scf::ForOp::create(b, loc, constant(0), freeExtents[axis], constant(panel.shape[axis]));
    loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(facts.operation, loop);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    offsets[axis] = loop.getInductionVar();
    active[axis] = arith::MinUIOp::create(b, loc,
        arith::SubIOp::create(b, loc, freeExtents[axis], offsets[axis]), constant(panel.shape[axis]));
    return panelLoops(panel, axis + 1);
  }

  // Pack only the current K/N window into a private dense representation. The
  // caller has already established the numerical supply frontier, so a frozen
  // source is read from its snapshot. Both the copy and its lifetime are IR.
  LogicalResult packPanel(Value source, Value destination, ArrayRef<int64_t> batchAxes) {
    for (int64_t axis : batchAxes) source = projectAxisIndex(source, axis, constant(0));
    int64_t copyAxis = cast<riscv::MemDescType>(source.getType()).getAxisIds().asArrayRef().back();
    return transferDensePanel(b, source, destination, copyAxis, facts.maximumLMUL,
        facts.operation->getParentOfType<riscv::KernelOp>().getPanelTransferFields(), facts.operation);
  }

  LogicalResult panelBody(const PanelChoice &panel) {
    auto completedChoice = choice;
    auto computationFacts = facts;
    SmallVector<Value> macroOffsets(offsets), macroActive(active);
    Value originalMemory[2] = {memory[0], memory[1]};
    Value originalReduction = reductionExtent;
    SmallVector<int64_t> partialAxes, partialShape;
    for (auto [axis, size] : llvm::zip(facts.freeAxes(), panel.shape))
      if (axis != facts.laneAxis) {
        partialAxes.push_back(axis);
        partialShape.push_back(size);
      }
    partialAxes.push_back(facts.laneAxis);
    partialShape.push_back(panel.shape[panel.lane]);
    auto partial = privateArray(partialAxes, partialShape, facts.element());
    if (failed(partial)) return failure();
    auto zero = riscv::ConstantOp::create(b, loc, facts.element(), b.getZeroAttr(facts.element()));
    if (failed(initializeLocalValue(b, *partial, zero, facts.operation))) return failure();
    auto partialView = localView(b, *partial, facts.operation);
    if (failed(partialView)) return failure();
    auto lanePosition = llvm::find(facts.freeAxes(), facts.laneAxis) - facts.freeAxes().begin();
    auto packed = privateArray({facts.axis, facts.laneAxis},
                               {panel.reduction, panel.shape[lanePosition]}, facts.storageTypes[facts.laneInput]);
    if (failed(packed)) return failure();
    auto packedView = localView(b, *packed, facts.operation);
    if (failed(packedView)) return failure();
    auto loop = scf::ForOp::create(b, loc, constant(0), originalReduction, constant(panel.reduction));
    loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(facts.operation, loop);
    b.setInsertionPointToStart(loop.getBody());
    reductionExtent = arith::MinUIOp::create(b, loc,
        arith::SubIOp::create(b, loc, originalReduction, loop.getInductionVar()), constant(panel.reduction));
    reductionExtents[0] = reductionExtent;
    for (unsigned input = 0; input < 2; ++input) {
      SmallVector<Value> starts, lengths;
      for (int64_t axis : cast<riscv::MemDescType>(memory[input].getType()).getAxisIds().asArrayRef()) {
        if (axis == facts.axis) {
          starts.push_back(loop.getInductionVar()); lengths.push_back(reductionExtent);
        } else {
          size_t position = llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin();
          starts.push_back(macroOffsets[position]); lengths.push_back(macroActive[position]);
        }
      }
      memory[input] = windowView(memory[input], starts, lengths);
    }
    Value packedWindow = windowView(*packedView, {constant(0), constant(0)},
                                    {reductionExtent, macroActive[lanePosition]});
    if (failed(packPanel(memory[facts.laneInput], packedWindow, panel.batchAxes)))
      return failure();
    memory[facts.laneInput] = packedWindow;
    partialMemory = *partialView;
    // A K panel closes into private storage before any author epilogue runs.
    // Its compute tile therefore does not retain registers for that epilogue.
    activateTile(selectedCase.getUpdateTile());
    offsets.assign(choice.shape.size(), constant(0));
    active = macroActive;
    if (failed(tileLoops(panel.shape, macroActive, 0, [&] { return outputReduction(); }))) return failure();
    b.setInsertionPointAfter(loop);
    activateTile(selectedCase.getOutputTile());
    selectedEpilogue = selectedCase.getEpilogue();
    memory[0] = originalMemory[0]; memory[1] = originalMemory[1];
    reductionExtent = originalReduction;
    reductionExtents[0] = originalReduction;
    // Private storage closes the accumulator phase. Its consumer can choose
    // the destination's lane axis independently, without retaining an FMA
    // carrier or moving an author numerical operation across the K boundary.
    offsets.assign(choice.shape.size(), constant(0));
    active = macroActive;
    // Reload completed output tiles after the final K panel. Bias and other
    // author epilogues run once here, never once per partial accumulation.
    if (failed(tileLoops(panel.shape, macroActive, 0, [&]() -> LogicalResult {
      auto type = outputPieceType(facts.freeAxes(), facts.element());
      Value value = readOutputTile(tileView(partialMemory), type);
      SmallVector<Value> localOffsets(offsets);
      for (size_t i = 0; i < offsets.size(); ++i)
        offsets[i] = arith::AddIOp::create(b, loc, macroOffsets[i], localOffsets[i]);
      auto status = finishOutputTile(value);
      offsets = localOffsets;
      return status;
    }))) return failure();
    partialMemory = {};
    facts = computationFacts;
    choice = completedChoice;
    selectedEpilogue = selectedCase.getDirectEpilogue();
    tiled = choice.shape != facts.freeShape();
    offsets = macroOffsets;
    active = macroActive;
    return success();
  }

  LogicalResult storeTile(Value tile) {
    if (facts.destination) return emitEpilogue(tile);
    auto coordinates = riscv::RegisterCoordinates::get(cast<riscv::ValueType>(tile.getType()).getLayout());
    if (!coordinates) return facts.operation->emitError("completed scalar contraction lost its register coordinates");
    SmallVector<Attribute> selectors(choice.shape.size(), b.getStringAttr("index"));
    for (int64_t part = 0; part < coordinates->parts(); ++part) {
      auto point = *coordinates->logical(part);
      SmallVector<Value> indices;
      Value flat = constant(0), valid;
      for (auto [axis, coordinate] : llvm::enumerate(point)) {
        indices.push_back(constant(coordinate));
        Value position = arith::AddIOp::create(b, loc, offsets[axis], indices.back());
        Value extent = facts.freeShape()[axis] < 0 ? freeExtents[axis] : constant(facts.freeShape()[axis]);
        flat = arith::AddIOp::create(b, loc,
            arith::MulIOp::create(b, loc, flat, extent), position);
        if (facts.freeShape()[axis] < 0 || facts.freeShape()[axis] % choice.shape[axis]) {
          Value bounded = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
              position, extent);
          valid = valid ? Value(arith::AndIOp::create(b, loc, valid, bounded)) : bounded;
        }
      }
      OpBuilder::InsertionGuard guard(b);
      if (valid) {
        auto branch = scf::IfOp::create(b, loc, valid, false);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
      }
      auto scalar = riscv::ExtractOp::create(b, loc, facts.element(), tile, indices,
          b.getArrayAttr(selectors), unassignedAccess(b), unselectedLeaf(b));
      auto store = riscv::LocalStoreOp::create(b, loc, scalar, storage, flat,
          leaf(b, "transfer", "local-store", "local.store.element", "local.store.element", 0, 0));
      copyOrigin(facts.operation, scalar);
      copyOrigin(facts.operation, store);
    }
    return success();
  }

  LogicalResult emitEpilogue(Value tile) {
    outputPoints();
    return materializeContractionEpilogue(b, facts.operation, *facts.destination,
        selectedEpilogue, tile, offsets, active, epilogueMemory, shapeIdentity);
  }

  FailureOr<Value> accumulate(ValueRange descriptors, ArrayRef<riscv::ValueType> operandTypes,
                              riscv::ValueType accumulator, Value initial, bool guarded, bool unitStride = false) {
    Value carry = initial;
    const int64_t step = choice.lanes * facts.schedule.getUnroll();
    auto capacity = facts.staticReductionExtent(facts.axis);
    bool hasFullIssue = !capacity || *capacity >= step;
    Value fullEnd = hasFullIssue ? completeDenseExtent(b, loc, reductionExtent, step) : constant(0);
    auto issue = [&](Value offset, Value length, Value previous, bool full) -> Value {
      Value operands[2];
      for (unsigned index = 0; index < 2; ++index) {
        if (index == 1 && facts.sameInput()) { operands[1] = operands[0]; continue; }
        auto type = full ? riscv::streamAccumulatorType(operandTypes[index]) : operandTypes[index];
        SmallVector<Value> starts, lengths;
        for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
          starts.push_back(axis == facts.axis ? offset : constant(0));
          lengths.push_back(axis == facts.axis ? length : guarded ? extent(descriptors[index], axis)
                                                                          : constant(type.getShape()[position]));
        }
        Value source = windowView(descriptors[index], starts, lengths);
        auto memory = cast<riscv::MemDescType>(source.getType());
        auto access = unitStride ? riscv::AccessAttr::get(b.getContext(), "unit", "dense",
            std::max<int64_t>(1, memory.getAlignment()), 0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1)
            : denseAccess(b, memory, type);
        auto load = riscv::RVVTileLoadOp::create(b, loc, type, source, tileBounds(source, type), facts.axis, access,
            leaf(b, "transfer", "tile-load", "rvv.tile-load", "rvv.tile-load", 0,
                 type.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
        copyOrigin(facts.operation, load);
        operands[index] = load;
        if (!facts.nativeHalf && type.getElementType() != facts.element()) {
          auto promoted = valueType(type.getAxisIds().asArrayRef(), type.getShape().asArrayRef(),
              facts.element(), choice.accumulatorLMUL, full);
          auto cast = riscv::CastOp::create(b, loc, promoted, load, unselectedLeaf(b));
          copyOrigin(facts.operation, cast);
          operands[index] = cast;
        }
      }
      auto instruction = facts.nativeHalf
          ? (full ? "rvv.stream-widen-contract-full-step" : "rvv.stream-widen-contract-step")
          : (full ? "rvv.stream-contract-full-step" : "rvv.stream-contract-step");
      auto update = riscv::RVVStreamContractStepOp::create(b, loc, accumulator, operands[0], operands[1], previous,
          length, facts.axis, leaf(b, "rvv", "stream-contract-step", instruction, instruction,
              cast<riscv::ValueType>(operands[0].getType()).getLayout().getRegisterGroups() +
              cast<riscv::ValueType>(operands[1].getType()).getLayout().getRegisterGroups() +
              accumulator.getLayout().getRegisterGroups(), accumulator.getLayout().getRegisterGroups()));
      copyOrigin(facts.operation, update);
      return update;
    };
    // Keep complete K issues free of per-load validity and active-length
    // calculations. The final at-most-unroll issues retain the same numerical
    // accumulator; no source contraction or Level boundary moves.
    // A statically short author window has no complete issue. Do not create
    // out-of-bounds static subviews in an unreachable zero-trip loop and rely
    // on a later simplifier to erase the invalid program.
    if (hasFullIssue) {
      auto main = scf::ForOp::create(b, loc, constant(0), fullEnd, constant(step), ValueRange{carry});
      main->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      main->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
      if (facts.schedule.getPipelineDepth() == 2) main->setAttr("weft.riscv.schedule", facts.schedule);
      copyOrigin(facts.operation, main);
      b.setInsertionPointToStart(main.getBody());
      carry = main.getRegionIterArg(0);
      for (int64_t strip = 0; strip < facts.schedule.getUnroll(); ++strip) {
        Value offset = main.getInductionVar();
        if (strip) offset = arith::AddIOp::create(b, loc, offset, constant(strip * choice.lanes));
        carry = issue(offset, constant(choice.lanes), carry, true);
      }
      scf::YieldOp::create(b, loc, carry);
      b.setInsertionPointAfter(main);
      carry = main.getResult(0);
    }
    auto tail = scf::ForOp::create(b, loc, fullEnd, reductionExtent, constant(choice.lanes), ValueRange{carry});
    tail->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    tail->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(facts.operation, tail);
    b.setInsertionPointToStart(tail.getBody());
    Value length = arith::MinUIOp::create(b, loc,
        arith::SubIOp::create(b, loc, reductionExtent, tail.getInductionVar()), constant(choice.lanes));
    carry = issue(tail.getInductionVar(), length, tail.getRegionIterArg(0), false);
    scf::YieldOp::create(b, loc, carry);
    b.setInsertionPointAfter(tail);
    return tail.getResult(0);
  }

  Value projectReductions(Value descriptor) {
    if (facts.reductionAxes.size() == 1) return descriptor;
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    SmallVector<int64_t> axes, shape, strides, origins;
    SmallVector<Value> indices;
    SmallVector<Attribute> selectors;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      if (facts.isReduction(axis) && axis != facts.axis) {
        size_t reduction = llvm::find(facts.reductionAxes, axis) - facts.reductionAxes.begin();
        indices.push_back(reductionOffsets[reduction]);
        selectors.push_back(b.getStringAttr("index"));
      } else {
        selectors.push_back(b.getStringAttr("all"));
        axes.push_back(axis); shape.push_back(type.getShape()[position]);
        strides.push_back(type.getStrides()[position]); origins.push_back(type.getOrigins()[position]);
      }
    }
    auto projected = riscv::MemDescType::get(b.getContext(), type.getEncoding(), b.getDenseI64ArrayAttr(shape),
        b.getDenseI64ArrayAttr(axes), b.getDenseI64ArrayAttr(strides), b.getDenseI64ArrayAttr(origins),
        type.getAlignment(), "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(),
        type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    auto slice = riscv::SliceOp::create(b, loc, projected, descriptor, indices, b.getArrayAttr(selectors));
    copyOrigin(facts.operation, slice);
    return slice;
  }

  Value projectReductionIndex(Value descriptor, Value index) {
    return projectAxisIndex(descriptor, facts.axis, index);
  }

  Value projectAxisIndex(Value descriptor, int64_t selectedAxis, Value index) {
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    SmallVector<int64_t> axes, shape, strides, origins;
    SmallVector<Attribute> selectors;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      bool reduced = axis == selectedAxis;
      selectors.push_back(b.getStringAttr(reduced ? "index" : "all"));
      if (reduced) continue;
      axes.push_back(axis); shape.push_back(type.getShape()[position]);
      strides.push_back(type.getStrides()[position]); origins.push_back(type.getOrigins()[position]);
    }
    auto projected = riscv::MemDescType::get(b.getContext(), type.getEncoding(), b.getDenseI64ArrayAttr(shape),
        b.getDenseI64ArrayAttr(axes), b.getDenseI64ArrayAttr(strides), b.getDenseI64ArrayAttr(origins),
        type.getAlignment(), "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(),
        type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    return riscv::SliceOp::create(b, loc, projected, descriptor, ValueRange{index}, b.getArrayAttr(selectors));
  }

  riscv::ValueType outputPieceType(ArrayRef<int64_t> axes, Type element, bool full = false) {
    SmallVector<int64_t> shape;
    for (int64_t axis : axes)
      shape.push_back(choice.shape[llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin()]);
    return valueType(axes, shape, element,
        choice.accumulatorLMUL * element.getIntOrFloatBitWidth() / facts.element().getIntOrFloatBitWidth(), full);
  }

  // Refine only points that already exist in the author's domain. The physical
  // window uses its own offset and active extent without introducing a Level.
  void outputPoints() {
    for (auto [position, axis] : llvm::enumerate(facts.freeAxes())) {
      riscv::PhysicalPointOp point;
      for (Value descriptor : memory)
        if ((point = originPoint(descriptor, axis))) break;
      if (!point) continue;
      Value base = arith::AddIOp::create(b, loc, point.getBase(), offsets[position]);
      riscv::PhysicalPointOp::create(b, loc, point.getType(), point.getParent(), base,
          active[position], constant(choice.shape[position]));
    }
  }

  SmallVector<Value> tileBounds(Value descriptor, riscv::ValueType type) {
    SmallVector<Value> bounds;
    for (int64_t axis : type.getAxisIds().asArrayRef()) bounds.push_back(extent(descriptor, axis));
    return bounds;
  }

  Value outputOperand(Value descriptor, unsigned input, Value k, bool guarded) {
    Value source = projectReductionIndex(descriptor, k);
    auto memoryType = cast<riscv::MemDescType>(source.getType());
    Type element = storageElement(source);
    auto type = outputPieceType(memoryType.getAxisIds().asArrayRef(), element);
    Value value;
    if (input == facts.laneInput) {
      if (!guarded) {
        SmallVector<Value> starts(type.getShape().size(), constant(0)), lengths;
        for (int64_t size : type.getShape().asArrayRef()) lengths.push_back(constant(size));
        source = windowView(source, starts, lengths);
        memoryType = cast<riscv::MemDescType>(source.getType());
      }
      auto loaded = riscv::RVVTileLoadOp::create(b, loc, type, source, tileBounds(source, type),
          facts.laneAxis, denseAccess(b, memoryType, type),
          leaf(b, "transfer", "tile-load", "rvv.tile-load", "rvv.tile-load", 0,
               type.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
      copyOrigin(facts.operation, loaded);
      value = loaded;
    } else {
      SmallVector<Value> bounds;
      for (int64_t axis : type.getAxisIds().asArrayRef()) bounds.push_back(outputExtents[input].lookup(axis));
      auto loaded = riscv::ScalarTileLoadOp::create(b, loc, type, source, bounds, guarded,
          leaf(b, "transfer", "scalar-tile-load", "scalar.tile-load", "scalar.tile-load", 0, 0, 0, 0, "none", "exact"), nullptr);
      copyOrigin(facts.operation, loaded);
      value = loaded;
    }
    if (!facts.nativeHalf && element != facts.element()) {
      auto cast = riscv::CastOp::create(b, loc,
          outputPieceType(type.getAxisIds().asArrayRef(), facts.element()), value, unselectedLeaf(b));
      copyOrigin(facts.operation, cast);
      value = cast;
    }
    return value;
  }

  Value scalarPacket(Value descriptor, Value start, int64_t steps, bool guarded) {
    auto memoryType = cast<riscv::MemDescType>(descriptor.getType());
    SmallVector<Value> starts, lengths;
    SmallVector<int64_t> shape;
    for (int64_t axis : memoryType.getAxisIds().asArrayRef()) {
      if (axis == facts.axis) {
        starts.push_back(start);
        lengths.push_back(constant(steps));
        shape.push_back(steps);
      } else {
        auto position = llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin();
        starts.push_back(constant(0));
        lengths.push_back(guarded ? extent(descriptor, axis) : constant(choice.shape[position]));
        shape.push_back(choice.shape[position]);
      }
    }
    Value source = windowView(descriptor, starts, lengths);
    auto type = valueType(memoryType.getAxisIds().asArrayRef(), shape, storageElement(source), 0);
    auto loaded = riscv::ScalarTileLoadOp::create(b, loc, type, source, tileBounds(source, type), guarded,
        leaf(b, "transfer", "scalar-tile-load", "scalar.tile-load", "scalar.tile-load", 0, 0, 0, 0, "none", "exact"), nullptr);
    copyOrigin(facts.operation, loaded);
    return loaded;
  }

  Value packetStep(Value packet, int64_t step) {
    auto input = cast<riscv::ValueType>(packet.getType());
    SmallVector<Attribute> selectors;
    SmallVector<int64_t> axes;
    for (int64_t axis : input.getAxisIds().asArrayRef()) {
      selectors.push_back(b.getStringAttr(axis == facts.axis ? "index" : "all"));
      if (axis != facts.axis) axes.push_back(axis);
    }
    auto type = outputPieceType(axes, input.getElementType());
    Value selected = riscv::ExtractOp::create(b, loc, type, packet, ValueRange{constant(step)},
        b.getArrayAttr(selectors), unassignedAccess(b), unselectedLeaf(b));
    copyOrigin(facts.operation, selected.getDefiningOp());
    if (!facts.nativeHalf && type.getElementType() != facts.element()) {
      selected = riscv::CastOp::create(b, loc, outputPieceType(axes, facts.element()), selected, unselectedLeaf(b));
      copyOrigin(facts.operation, selected.getDefiningOp());
    }
    return selected;
  }

  FailureOr<Value> outputAccumulation(ValueRange descriptors, riscv::ValueType type, Value initial, bool guarded) {
    StringRef instruction = facts.nativeHalf
        ? (guarded ? "rvv.tile-vfwmacc.vf" : "rvv.tile-vfwmacc-full.vf")
        : (guarded ? "rvv.tile-vfmacc.vf" : "rvv.tile-vfmacc-full.vf");
    SmallVector<Value> bounds(active);
    if (!guarded)
      for (size_t position = 0; position < bounds.size(); ++position) bounds[position] = constant(choice.shape[position]);
    auto update = [&](Value index, Value previous, Value supplied = {}) -> Value {
      Value operands[2];
      for (unsigned input = 0; input < 2; ++input)
        operands[input] = supplied && input != facts.laneInput ? supplied
            : outputOperand(descriptors[input], input, index, guarded);
      auto step = riscv::RVVTileContractOp::create(b, loc, type, operands[0], operands[1], previous,
          bounds, facts.laneAxis, facts.axis, facts.laneInput ? "rhs" : "lhs",
          leaf(b, "rvv", "tile-contract", instruction, instruction, 0, 0, 0, 0,
               "none", guarded ? "undisturbed" : "exact", {facts.axis, facts.laneAxis}));
      copyOrigin(facts.operation, step);
      return step;
    };
    auto loop = [&](Value begin, Value end, int64_t unroll, Value seed, bool pipeline, int64_t packetSize) {
      auto region = scf::ForOp::create(b, loc, begin, end, constant(unroll), ValueRange{seed});
      region->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      region->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
      if (pipeline && facts.schedule.getPipelineDepth() == 2)
        region->setAttr("weft.riscv.schedule", facts.schedule);
      copyOrigin(facts.operation, region);
      b.setInsertionPointToStart(region.getBody());
      Value carried = region.getRegionIterArg(0);
      Value packet;
      for (int64_t part = 0; part < unroll; ++part) {
        Value index = region.getInductionVar();
        if (part) index = arith::AddIOp::create(b, loc, index, constant(part));
        if (packetSize > 1 && part % packetSize == 0)
          packet = scalarPacket(descriptors[1 - facts.laneInput], index, packetSize, guarded);
        carried = update(index, carried, packet ? packetStep(packet, part % packetSize) : Value{});
      }
      scf::YieldOp::create(b, loc, carried);
      b.setInsertionPointAfter(region);
      return region.getResult(0);
    };
    int64_t packetSize = choice.scalarSteps;
    int64_t unroll = facts.schedule.getUnroll() * packetSize;
    auto capacity = facts.staticReductionExtent(facts.axis);
    bool hasFullIssue = !capacity || *capacity >= unroll;
    Value fullEnd = hasFullIssue ? completeDenseExtent(b, loc, reductionExtent, unroll) : constant(0);
    Value value = hasFullIssue ? loop(constant(0), fullEnd, unroll, initial, true, packetSize) : initial;
    if (unroll != 1) value = loop(fullEnd, reductionExtent, 1, value, false, 1);
    return value;
  }

  Value readOutputTile(Value source, riscv::ValueType type) {
    auto loaded = riscv::RVVTileLoadOp::create(b, loc, type, source, tileBounds(source, type), facts.laneAxis,
        denseAccess(b, cast<riscv::MemDescType>(source.getType()), type),
        leaf(b, "transfer", "tile-load", "rvv.tile-load", "rvv.tile-load", 0,
             type.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
    copyOrigin(facts.operation, loaded);
    return loaded;
  }

  void writeOutputTile(Value value, Value destination) {
    auto type = cast<riscv::ValueType>(value.getType());
    auto store = riscv::RVVTileStoreOp::create(b, loc, value, destination, tileBounds(destination, type), facts.laneAxis,
        denseAccess(b, cast<riscv::MemDescType>(destination.getType()), type),
        leaf(b, "transfer", "tile-store", "rvv.tile-store", "rvv.tile-store",
             type.getLayout().getRegisterGroups(), 0, 0, 0, "none", "exact"));
    copyOrigin(facts.operation, store);
  }


  LogicalResult outputReduction() {
    if (!partialMemory) outputPoints();
    Value descriptors[2] = {tileView(memory[0]), tileView(memory[1])};
    auto type = outputPieceType(facts.freeAxes(), facts.element(), !partialMemory);
    Value initial;
    if (partialMemory) initial = readOutputTile(tileView(partialMemory), type);
    else {
      auto zero = riscv::ConstantOp::create(b, loc, facts.element(), b.getZeroAttr(facts.element()));
      initial = riscv::RVVSplatOp::create(b, loc, type, zero,
          leaf(b, "rvv", "splat", "rvv.splat.local", "rvv.splat.local", 0, type.getLayout().getRegisterGroups()));
    }
    Value full;
    for (unsigned input = 0; input < 2; ++input) {
      outputExtents[input].clear();
      for (int64_t axis : cast<riscv::MemDescType>(descriptors[input].getType()).getAxisIds().asArrayRef()) {
        if (axis == facts.axis) continue;
        Value bound = extent(descriptors[input], axis);
        outputExtents[input][axis] = bound;
        size_t position = llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin();
        Value complete = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, bound, constant(choice.shape[position]));
        full = full ? Value(arith::AndIOp::create(b, loc, full, complete)) : complete;
      }
    }
    Value value;
    if (full) {
      auto branch = scf::IfOp::create(b, loc, TypeRange{type}, full, true);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      auto complete = outputAccumulation(descriptors, type, initial, false);
      if (failed(complete)) return failure();
      scf::YieldOp::create(b, loc, *complete);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      auto partial = outputAccumulation(descriptors, type, initial, true);
      if (failed(partial)) return failure();
      scf::YieldOp::create(b, loc, *partial);
      b.setInsertionPointAfter(branch);
      value = branch.getResult(0);
    } else {
      auto complete = outputAccumulation(descriptors, type, initial, false);
      if (failed(complete)) return failure();
      value = *complete;
    }
    if (partialMemory) {
      writeOutputTile(value, tileView(partialMemory));
      return success();
    }
    return finishOutputTile(value);
  }

  LogicalResult finishOutputTile(Value value) {
    if (facts.destination) return emitEpilogue(value);
    if (tiled) {
      auto descriptor = localView(b, storage, facts.operation);
      if (failed(descriptor)) return failure();
      writeOutputTile(value, tileView(*descriptor));
    } else result = value;
    return success();
  }

  FailureOr<Value> laneReduction(ValueRange views, riscv::ValueType accumulator, Value initial) {
    Value descriptors[2] = {projectReductions(views[0]), projectReductions(views[1])};
    riscv::ValueType operandTypes[2];
    for (unsigned index = 0; index < 2; ++index) {
      auto type = cast<riscv::MemDescType>(descriptors[index].getType());
      SmallVector<int64_t> operandShape;
      for (int64_t axis : type.getAxisIds().asArrayRef()) {
        if (axis == facts.axis) operandShape.push_back(choice.lanes);
        else {
          size_t position = llvm::find(facts.freeAxes(), axis) - facts.freeAxes().begin();
          operandShape.push_back(choice.shape[position]);
        }
      }
      Type element = storageElement(descriptors[index]);
      operandTypes[index] = valueType(type.getAxisIds().asArrayRef(), operandShape, element,
          choice.accumulatorLMUL * element.getIntOrFloatBitWidth() / facts.element().getIntOrFloatBitWidth());
    }
    Value full;
    for (unsigned input = 0; input < 2; ++input)
      for (auto [position, axis] : llvm::enumerate(operandTypes[input].getAxisIds().asArrayRef())) {
        if (axis == facts.axis) continue;
        Value complete = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
            extent(descriptors[input], axis), constant(operandTypes[input].getShape()[position]));
        full = full ? Value(arith::AndIOp::create(b, loc, full, complete)) : complete;
      }
    Value finalized;
    SmallVector<std::pair<Value, int64_t>> strides;
    bool canProbe = true;
    for (Value descriptor : descriptors) {
      auto type = cast<riscv::MemDescType>(descriptor.getType());
      auto axis = llvm::find(type.getAxisIds().asArrayRef(), facts.axis) - type.getAxisIds().asArrayRef().begin();
      if (type.getStrides()[axis] == 1) continue;
      auto source = riscv::denseArgumentStride(descriptor, facts.axis);
      if (!source) { canProbe = false; break; }
      if (!llvm::is_contained(strides, *source)) strides.push_back(*source);
    }
    Value condition = full;
    bool specialized = canProbe && !strides.empty();
    if (specialized)
      for (auto [argument, axis] : strides) {
        Value stride = riscv::StrideOp::create(b, loc, b.getIndexType(), argument, axis);
        Value unit = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, stride, constant(1));
        condition = condition ? Value(arith::AndIOp::create(b, loc, condition, unit)) : unit;
      }
    if (condition) {
      auto branch = scf::IfOp::create(b, loc, TypeRange{accumulator}, condition, true);
      copyOrigin(facts.operation, branch);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      auto complete = accumulate(descriptors, operandTypes, accumulator, initial, false, specialized);
      if (failed(complete)) return failure();
      scf::YieldOp::create(b, loc, *complete);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      auto partial = accumulate(descriptors, operandTypes, accumulator, initial, bool(full));
      if (failed(partial)) return failure();
      scf::YieldOp::create(b, loc, *partial);
      b.setInsertionPointAfter(branch);
      finalized = branch.getResult(0);
    } else {
      auto complete = accumulate(descriptors, operandTypes, accumulator, initial, false);
      if (failed(complete)) return failure();
      finalized = *complete;
    }
    return finalized;
  }

  FailureOr<Value> outerReductions(size_t depth, ValueRange descriptors,
                                   riscv::ValueType accumulator, Value initial) {
    if (depth + 1 == facts.reductionAxes.size()) return laneReduction(descriptors, accumulator, initial);
    auto loop = scf::ForOp::create(b, loc, constant(0), reductionExtents[depth], constant(1), ValueRange{initial});
    loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(facts.operation, loop);
    b.setInsertionPointToStart(loop.getBody());
    reductionOffsets[depth] = loop.getInductionVar();
    auto partial = outerReductions(depth + 1, descriptors, accumulator, loop.getRegionIterArg(0));
    if (failed(partial)) return failure();
    scf::YieldOp::create(b, loc, *partial);
    b.setInsertionPointAfter(loop);
    return loop.getResult(0);
  }

  LogicalResult reduction() {
    if (!facts.reductionLanes()) return outputReduction();
    Value descriptors[2] = {tiled ? tileView(memory[0]) : memory[0], tiled ? tileView(memory[1]) : memory[1]};
    SmallVector<int64_t> axes(facts.freeAxes()), shape(choice.shape);
    axes.push_back(facts.axis); shape.push_back(choice.lanes);
    auto accumulator = valueType(axes, shape, facts.element(), choice.accumulatorLMUL, true);
    auto initial = denseReductionSeed(b, loc, accumulator, "add");
    auto partial = outerReductions(0, descriptors, accumulator, initial);
    if (failed(partial)) return failure();
    Type scalarType = facts.result ? Type(valueType(facts.freeAxes(), choice.shape, facts.element(), 0))
                                  : facts.element();
    auto finalized = riscv::RVVStreamFinalizeOp::create(b, loc, scalarType, *partial, facts.axis, "add",
        leaf(b, "rvv", "stream-finalize", "rvv.stream-finalize", "rvv.stream-finalize",
             accumulator.getLayout().getRegisterGroups(), 0, 1));
    copyOrigin(facts.operation, finalized);
    if (facts.destination || tiled) return storeTile(finalized);
    else result = finalized;
    return success();
  }
};
}

LogicalResult riscv_internal::materializeDenseContractions(ModuleOp module, IRRewriter &rewriter,
                                                          int64_t maximumLMUL) {
  SmallVector<Operation *> operations;
  module.walk([&](Operation *op) {
    if (isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(op)) operations.push_back(op);
  });
  int64_t identity = lastLocalIdentity(module);
  llvm::DenseMap<Value, Value> snapshots;
  llvm::SmallPtrSet<Operation *, 8> consumedSchedules;
  int64_t shapeIdentity = 0;
  module.walk([&](riscv::KernelOp kernel) {
    shapeIdentity = std::max(shapeIdentity, int64_t(kernel.getShapeSymbols().size()));
  });
  module.walk([&](riscv::LocalShapeOp shape) { shapeIdentity = std::max(shapeIdentity, int64_t(shape.getId())); });
  for (Operation *operation : operations) {
    ContractFacts facts(operation, maximumLMUL);
    auto plan = operation->getAttrOfType<riscv::ContractionPlanAttr>(contractionPlanName);
    if (!facts.analyze()) {
      if (plan) return operation->emitError("selected dense contraction no longer has its proven supply and destination graph");
      continue;
    }
    if (!plan) return operation->emitError("dense contraction requires explicit physical selection before materialization");
    if (failed(verifySelectedPlan(facts, plan)) ||
        failed(ContractProgram(facts, plan, rewriter, identity, shapeIdentity, snapshots).run())) return failure();
    if (facts.scheduleOwner) consumedSchedules.insert(facts.scheduleOwner);
  }
  // A requested pipeline was bound to the contraction's physical issue
  // frontier. A short author window may have no complete issue at all; that
  // absence must not silently redirect the binding to the outer author loop.
  // Defer consumption until all contractions have read the enclosing binding.
  for (Operation *owner : consumedSchedules) owner->removeAttr("weft.riscv.schedule");
  return success();
}

namespace {
class SelectDenseContractionsPass
    : public PassWrapper<SelectDenseContractionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectDenseContractionsPass)
  explicit SelectDenseContractionsPass(int64_t bound = 8) { maximumLMUL = bound; }
  SelectDenseContractionsPass(const SelectDenseContractionsPass &other) : PassWrapper(other) {
    copyOptionValuesFrom(&other);
  }
  StringRef getArgument() const override { return "weft-riscv-select-dense-contractions"; }
  StringRef getDescription() const override {
    return "Select finite closed-contraction representations, extent cases and phase-local resources";
  }
  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](riscv::KernelOp kernel) {
      if (!llvm::is_contained(kernel.getTarget().getLegalLMULEighths().asArrayRef(), maximumLMUL.getValue())) {
        kernel.emitError("dense contraction selection requires a legal target LMUL binding");
        invalid = true;
      }
    });
    getOperation().walk([&](Operation *operation) {
      if (invalid) return;
      bool contraction = isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(operation);
      if (operation->hasAttr(contractionPlanName) &&
          (!contraction || !operation->getAttrOfType<riscv::ContractionPlanAttr>(contractionPlanName))) {
        operation->emitError("dense contraction selection must be a typed attribute on its closed numerical operation");
        invalid = true;
        return;
      }
      if (!contraction) return;
      ContractFacts facts(operation, maximumLMUL);
      auto previous = operation->getAttrOfType<riscv::ContractionPlanAttr>(contractionPlanName);
      if (!facts.analyze()) {
        if (previous) {
          operation->emitError("cannot reselect a contraction whose numerical or memory proof changed");
          invalid = true;
        }
        return;
      }
      if (previous) {
        invalid = failed(verifySelectedPlan(facts, previous));
        return;
      }
      auto plan = selectPlan(facts);
      if (failed(plan) || failed(verifySelectedPlan(facts, *plan))) { invalid = true; return; }
      operation->setAttr(contractionPlanName, *plan);
    });
    if (invalid) signalPassFailure();
  }
private:
  Option<int64_t> maximumLMUL{*this, "lmul-eighths", llvm::cl::init(8)};
};
}

std::unique_ptr<Pass> weft::createSelectRISCVDenseContractionsPass(int64_t maximumLMUL) {
  return std::make_unique<SelectDenseContractionsPass>(maximumLMUL);
}

FailureOr<std::string> weft::inspectRISCVContractions(ModuleOp module) {
  auto binding = readRISCVPipelineInput(module, RISCVCheckpoint::Contractions);
  if (failed(binding)) return failure();
  auto array = [](ArrayRef<int64_t> values) {
    llvm::json::Array result;
    for (int64_t value : values) result.push_back(value);
    return result;
  };
  auto tile = [&](riscv::ContractionTileAttr value) {
    return llvm::json::Object{{"shape", array(value.getShape().asArrayRef())},
        {"lane_axis", value.getLaneAxis()}, {"lane_input", value.getLaneInput()},
        {"lanes", value.getLanes()}, {"lmul_eighths", value.getLmulEighths()},
        {"scalar_steps", value.getScalarSteps()}, {"register_groups", value.getRegisters()},
        {"loads", value.getLoads()}, {"conversions", value.getConversions()}};
  };
  auto epilogue = [&](riscv::ContractionEpilogueAttr value) {
    return llvm::json::Object{{"mode", value.getMode()}, {"shape", array(value.getShape().asArrayRef())},
        {"lane_axis", value.getLaneAxis()}, {"lanes", value.getLanes()},
        {"lmul_eighths", value.getLmulEighths()}, {"register_groups", value.getRegisters()}};
  };
  llvm::json::Array kernels;
  bool invalid = false;
  module.walk([&](riscv::KernelOp kernel) {
    llvm::json::Array selections;
    kernel.walk([&](Operation *operation) {
      if (invalid) return;
      if (!isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(operation)) {
        if (operation->hasAttr(contractionPlanName)) {
          operation->emitError("contraction inspection found a selection on another physical operation");
          invalid = true;
        }
        return;
      }
      ContractFacts facts(operation, binding->lmulEighths);
      auto plan = operation->getAttrOfType<riscv::ContractionPlanAttr>(contractionPlanName);
      if (!facts.analyze()) {
        if (operation->hasAttr(contractionPlanName)) {
          operation->emitError("contraction inspection found a selection without its numerical proof");
          invalid = true;
        }
        return;
      }
      if (!plan || failed(verifySelectedPlan(facts, plan))) {
        if (!plan) operation->emitError("eligible contraction is missing its selected representation");
        invalid = true;
        return;
      }
      llvm::json::Array cases;
      for (Attribute attribute : plan.getCases()) {
        auto selected = cast<riscv::ContractionCaseAttr>(attribute);
        cases.push_back(llvm::json::Object{{"singleton_axis", selected.getSingletonAxis()},
            {"tile", tile(selected.getTile())},
            {"panel_shape", array(selected.getPanelShape().asArrayRef())}, {"panel_k", selected.getPanelK()},
            {"update_tile", tile(selected.getUpdateTile())}, {"output_tile", tile(selected.getOutputTile())},
            {"direct_epilogue", epilogue(selected.getDirectEpilogue())},
            {"epilogue", epilogue(selected.getEpilogue())}});
      }
      std::string element;
      llvm::raw_string_ostream printed(element);
      facts.element().print(printed);
      selections.push_back(llvm::json::Object{{"output_axes", array(facts.freeAxes())},
          {"output_shape", array(facts.freeShape())}, {"reduction_axes", array(facts.reductionAxes)},
          {"element", element}, {"streamed_store", plan.getStreamedStore()},
          {"unroll", plan.getUnroll()}, {"pipeline_depth", plan.getPipelineDepth()},
          {"cases", std::move(cases)}});
    });
    kernels.push_back(llvm::json::Object{{"name", kernel.getSymName()},
        {"vlen_bits", kernel.getTarget().getVlenBits()}, {"contractions", std::move(selections)}});
  });
  if (invalid) return failure();
  llvm::json::Object result{{"kind", "weft-contraction-selections"},
      {"scope", "static-physical-choices"}, {"kernels", std::move(kernels)}};
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
}
