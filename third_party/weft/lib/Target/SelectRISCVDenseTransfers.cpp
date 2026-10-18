#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "ContractionSupply.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/DenseSegments.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace weft;

namespace {

struct DenseRead {
  Operation *operation;
  Value source;
  riscv::ValueType type;
  int64_t laneAxis;
};

std::optional<DenseRead> denseRead(Operation *operation) {
  Value source, result;
  if (auto load = dyn_cast<riscv::LoadOp>(operation)) {
    if (load.getSnapshotStorage()) return std::nullopt;
    source = load.getRegion();
    result = load.getResult();
  } else if (auto load = dyn_cast<riscv::RVVTileLoadOp>(operation)) {
    if (load.getLeaf().getInstruction().contains(".segment")) return std::nullopt;
    source = load.getSource();
    result = load.getResult();
  } else return std::nullopt;
  auto type = dyn_cast<riscv::ValueType>(result.getType());
  auto memory = dyn_cast<riscv::MemDescType>(source.getType());
  if (!type || !memory || !riscv::isAddressableDenseDescriptor(source) ||
      !riscv::tileMemoryElementMatches(memory, type.getElementType()) ||
      memory.getStorageBits() != type.getElementType().getIntOrFloatBitWidth() ||
      (memory.getAccess() != "read" && memory.getAccess() != "readwrite") ||
      memory.getAxisIds().size() != type.getAxisIds().size()) return std::nullopt;
  for (auto [axis, lanes] : llvm::zip(type.getAxisIds().asArrayRef(),
                                     type.getLayout().getLaneFactors().asArrayRef())) {
    if (lanes <= 1 || !riscv::VectorTileGeometry::get(type, axis)) continue;
    for (int64_t id : type.getAxisIds().asArrayRef()) {
      auto position = llvm::find(memory.getAxisIds().asArrayRef(), id);
      if (position == memory.getAxisIds().asArrayRef().end() ||
          !riscv::denseDescriptorExtent(source, position - memory.getAxisIds().asArrayRef().begin()))
        return std::nullopt;
    }
    return DenseRead{operation, source, type, axis};
  }
  return std::nullopt;
}

struct SegmentRead {
  DenseRead read;
  riscv::VectorTileSegments segments;
  bool completeFields;
};

std::optional<SegmentRead> select(const DenseRead &read) {
  auto kernel = read.operation->getParentOfType<riscv::KernelOp>();
  if (!kernel || kernel.getResourcesMaterialized() ||
      !kernel.getTarget().getHasSegmentMemory() ||
      !riscv::supportsRVVLayout(kernel.getTarget(), read.type.getLayout())) return std::nullopt;
  auto memory = cast<riscv::MemDescType>(read.source.getType());
  // Enumerate only representations of the already selected resident parts.
  // A tuple reduces load issues; it does not change the lane axis or introduce
  // a larger tile, local pack, new read, or target-dependent author binding.
  std::optional<SegmentRead> best;
  for (int64_t fieldAxis : read.type.getAxisIds().asArrayRef()) {
    if (fieldAxis == read.laneAxis ||
        !riscv::hasUnitStrideProof(read.operation, read.source, fieldAxis)) continue;
    for (int64_t fields : {8, 4, 2}) {
      auto segments = riscv::VectorTileSegments::get(read.type, read.laneAxis, fieldAxis, fields);
      if (!segments) continue;
      auto position = llvm::find(memory.getAxisIds().asArrayRef(), fieldAxis) -
                      memory.getAxisIds().asArrayRef().begin();
      auto extent = riscv::denseDescriptorExtent(read.source, position);
      int64_t required = read.type.getShape()[segments->fieldPosition];
      bool complete = riscv::tileHasCompleteFields(read.operation, read.source, read.type, fieldAxis);
      // A known partial field group, including the false arm emitted by this
      // pass, cannot use a segment instruction: VL masks lanes, not fields.
      if (!complete) {
        if (auto constant = getConstantIntValue(extent); constant && *constant < required) continue;
        if (auto index = dyn_cast<Value>(extent);
            index && riscv::denseIndexBoundsAt(index, read.operation).upper < uint64_t(required)) continue;
      }
      if (!best || segments->groups.size() < best->segments.groups.size())
        best = SegmentRead{read, std::move(*segments), complete};
      break;
    }
  }
  return best;
}

SmallVector<Value> extents(IRRewriter &b, const DenseRead &read) {
  auto memory = cast<riscv::MemDescType>(read.source.getType());
  SmallVector<Value> result;
  for (int64_t axis : read.type.getAxisIds().asArrayRef()) {
    auto position = llvm::find(memory.getAxisIds().asArrayRef(), axis) -
                    memory.getAxisIds().asArrayRef().begin();
    auto extent = riscv::denseDescriptorExtent(read.source, position);
    if (auto fixed = getConstantIntValue(extent))
      result.push_back(arith::ConstantIndexOp::create(b, read.operation->getLoc(), *fixed));
    else
      result.push_back(cast<Value>(extent));
  }
  return result;
}

Value emitRead(IRRewriter &b, const SegmentRead &plan, ValueRange bounds) {
  const auto &read = plan.read;
  auto kernel = read.operation->getParentOfType<riscv::KernelOp>();
  bool full = riscv::tileIsCompleteMemory(kernel.getTarget(), read.source, read.type);
  StringRef instruction = full ? "rvv.tile-load.segment.full" : "rvv.tile-load.segment";
  auto memory = cast<riscv::MemDescType>(read.source.getType());
  auto access = riscv::AccessAttr::get(b.getContext(), "segment", "dense", memory.getAlignment(),
      0, plan.segments.fields, 0, 0, 0, 0, 0, 0, 0, "none", 1);
  auto leaf = riscv_internal::leaf(b, "transfer", "tile-load", instruction, instruction,
      0, read.type.getLayout().getRegisterGroups(), 0, 0, "none", "exact",
      {plan.segments.fieldAxis, plan.segments.fields});
  auto load = riscv::RVVTileLoadOp::create(b, read.operation->getLoc(), read.type,
      read.source, bounds, read.laneAxis, access, leaf);
  riscv_internal::copyOrigin(read.operation, load);
  return load;
}

struct SiblingAccess {
  Operation *operation;
  Value memory;
  Value value;
  riscv::ValueType type;
  int64_t axis;
  bool store;
};

std::optional<SiblingAccess> siblingAccess(Operation *operation) {
  if (auto read = denseRead(operation))
    return SiblingAccess{operation, read->source, operation->getResult(0), read->type, read->laneAxis, false};
  if (auto store = dyn_cast<riscv::StoreOp>(operation)) {
    auto type = dyn_cast<riscv::ValueType>(store.getValue().getType());
    if (!type || !riscv::isAddressableDenseDescriptor(store.getRegion()) ||
        !riscv::tileMemoryElementMatches(store.getRegion().getType(), type.getElementType()) ||
        store.getRegion().getType().getStorageBits() != type.getElementType().getIntOrFloatBitWidth()) return std::nullopt;
    for (auto [axis, lanes] : llvm::zip(type.getAxisIds().asArrayRef(), type.getLayout().getLaneFactors().asArrayRef()))
      if (lanes > 1 && riscv::VectorTileGeometry::get(type, axis))
        return SiblingAccess{operation, store.getRegion(), store.getValue(), type, axis, true};
  }
  if (auto store = dyn_cast<riscv::RVVTileStoreOp>(operation)) {
    auto type = store.getInput().getType();
    if (store.getLeaf().getInstruction().contains(".segment") ||
        !riscv::isAddressableDenseDescriptor(store.getDestination()) ||
        !riscv::tileMemoryElementMatches(store.getDestination().getType(), type.getElementType()))
      return std::nullopt;
    return SiblingAccess{operation, store.getDestination(), store.getInput(), type,
                         static_cast<int64_t>(store.getAxis()), true};
  }
  return std::nullopt;
}

// These predicates describe immutable descriptor facts, not a second program
// representation. The selected op's verifier reconstructs them from its inputs.
struct DescriptorRequirement {
  Value memory;
  int64_t axis;
  int64_t expected;
  bool extent;

  bool proven(Operation *at) const {
    if (!extent) return riscv::hasDenseStrideProof(at, memory, axis, expected);
    auto type = cast<riscv::MemDescType>(memory.getType());
    auto position = llvm::find(type.getAxisIds().asArrayRef(), axis) - type.getAxisIds().asArrayRef().begin();
    auto bound = riscv::denseDescriptorExtent(memory, position);
    if (auto fixed = getConstantIntValue(bound)) return *fixed >= expected;
    auto index = dyn_cast_if_present<Value>(bound);
    return index && riscv::denseIndexBoundsAt(index, at).lower >= uint64_t(expected);
  }

  bool possible(Operation *at) const {
    if (proven(at)) return true;
    auto type = cast<riscv::MemDescType>(memory.getType());
    auto position = llvm::find(type.getAxisIds().asArrayRef(), axis) - type.getAxisIds().asArrayRef().begin();
    if (!extent) return bool(riscv::denseArgumentStride(memory, axis));
    auto bound = riscv::denseDescriptorExtent(memory, position);
    if (auto fixed = getConstantIntValue(bound)) return *fixed >= expected;
    auto index = dyn_cast_if_present<Value>(bound);
    return index && riscv::denseIndexBoundsAt(index, at).upper >= uint64_t(expected);
  }

  bool matches(arith::CmpIOp compare) const {
    auto predicate = extent ? arith::CmpIPredicate::uge : arith::CmpIPredicate::eq;
    if (compare.getPredicate() != predicate) return false;
    for (unsigned side = 0; side < (extent ? 1u : 2u); ++side) {
      if (getConstantIntValue(compare->getOperand(1 - side)) != expected) continue;
      auto actual = compare->getOperand(side);
      if (!extent) {
        auto source = riscv::denseArgumentStride(memory, axis);
        auto stride = actual.getDefiningOp<riscv::StrideOp>();
        if (source && stride && stride.getView() == source->first && stride.getAxis() == source->second)
          return true;
      } else {
        auto type = cast<riscv::MemDescType>(memory.getType());
        auto position = llvm::find(type.getAxisIds().asArrayRef(), axis) - type.getAxisIds().asArrayRef().begin();
        if (riscv::sameDenseExtent(actual, memory, position)) return true;
      }
    }
    return false;
  }

  Value emit(IRRewriter &b, Location loc) const {
    Value actual;
    if (!extent) {
      auto source = *riscv::denseArgumentStride(memory, axis);
      actual = riscv::StrideOp::create(b, loc, b.getIndexType(), source.first, source.second);
    } else {
      auto type = cast<riscv::MemDescType>(memory.getType());
      auto position = llvm::find(type.getAxisIds().asArrayRef(), axis) - type.getAxisIds().asArrayRef().begin();
      auto bound = riscv::denseDescriptorExtent(memory, position);
      if (auto fixed = getConstantIntValue(bound)) actual = arith::ConstantIndexOp::create(b, loc, *fixed);
      else actual = cast<Value>(bound);
    }
    auto constant = arith::ConstantIndexOp::create(b, loc, expected);
    return arith::CmpIOp::create(b, loc, extent ? arith::CmpIPredicate::uge : arith::CmpIPredicate::eq,
                                  actual, constant);
  }
};

bool excludedByBranch(Operation *at, ArrayRef<DescriptorRequirement> requirements) {
  // A failed conjunction of our own prerequisites disproves this choice. This
  // also prevents replay from nesting another copy in the original-transfer arm.
  for (Operation *child = at, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    if (!branch || child->getParentRegion() != &branch.getElseRegion()) continue;
    SmallVector<Value> pending{branch.getCondition()};
    bool onlyRequirements = true;
    unsigned visited = 0;
    while (!pending.empty() && ++visited <= 32) {
      Value condition = pending.pop_back_val();
      if (auto both = condition.getDefiningOp<arith::AndIOp>()) {
        pending.append({both.getLhs(), both.getRhs()});
        continue;
      }
      auto compare = condition.getDefiningOp<arith::CmpIOp>();
      if (!compare || !llvm::any_of(requirements, [&](const auto &item) { return item.matches(compare); })) {
        onlyRequirements = false;
        break;
      }
    }
    if (onlyRequirements && pending.empty() && visited <= 32) return true;
  }
  return false;
}

SmallVector<SiblingAccess> adjacentFields(const SiblingAccess &first) {
  SmallVector<SiblingAccess> result{first};
  unsigned visited = 0;
  for (Operation *operation = first.operation->getNextNode(); operation && ++visited <= 32;
       operation = operation->getNextNode()) {
    if (auto access = siblingAccess(operation)) {
      if (access->store != first.store || access->type != first.type || access->axis != first.axis) break;
      Value pair[] = {result.back().memory, access->memory};
      if (!riscv::DenseSegmentFields::get(pair, first.type, first.axis)) break;
      result.push_back(*access);
      if (result.size() == 8) break;
      continue;
    }
    if (operation->getNumRegions() || !isMemoryEffectFree(operation) ||
        !isSpeculatable(operation)) break;
    // Read grouping crosses descriptor/scalar coordinate construction only.
    // It cannot make a future numerical input live across earlier computation.
    if (!first.store && llvm::any_of(operation->getResultTypes(), [](Type type) {
          return isa<riscv::ValueType, riscv::FragmentType, riscv::LocalType>(type);
        })) break;
  }
  while (!llvm::is_contained(ArrayRef<size_t>({1, 2, 4, 8}), result.size())) result.pop_back();
  return result;
}

bool sourcesAvailable(ArrayRef<SiblingAccess> accesses, DominanceInfo &dominance) {
  auto *first = accesses.front().operation;
  auto *last = accesses.back().operation;
  for (const auto &access : accesses) {
    auto *descriptor = access.memory.getDefiningOp();
    if (!descriptor || (!isa<riscv::SliceOp, riscv::SubviewOp>(descriptor)) ||
        (descriptor->getBlock() != first->getBlock() && !dominance.dominates(descriptor, first))) return false;
    for (Value operand : descriptor->getOperands())
      if (!dominance.dominates(operand, first)) return false;
    if (accesses.front().store) continue;
    for (Operation *user : access.value.getUsers()) {
      auto *anchor = first->getBlock()->findAncestorOpInBlock(*user);
      if (!anchor || anchor == first || anchor->isBeforeInBlock(last)) return false;
    }
  }
  return true;
}

SmallVector<Value> emitSiblingTransfer(IRRewriter &b, ArrayRef<SiblingAccess> accesses,
                                       ValueRange bounds) {
  const auto &first = accesses.front();
  SmallVector<Value> descriptors, values;
  SmallVector<Type> types;
  for (const auto &access : accesses) {
    descriptors.push_back(access.memory);
    values.push_back(access.value);
    types.push_back(access.type);
  }
  int64_t groups = int64_t(accesses.size()) * first.type.getLayout().getRegisterGroups();
  StringRef family = first.store ? "dense-segment-store" : "dense-segment-load";
  StringRef instruction = first.store ? "rvv.dense-segment-store" : "rvv.dense-segment-load";
  auto leaf = riscv_internal::leaf(b, "transfer", family, instruction, instruction,
      first.store ? groups : 0, first.store ? 0 : groups, first.store ? groups : 0, 0, "none", "exact");
  if (first.store) {
    auto op = riscv::RVVDenseSegmentStoreOp::create(b, first.operation->getLoc(),
        values, descriptors, bounds, first.axis, leaf);
    riscv_internal::copyOrigin(first.operation, op);
    return {};
  }
  auto op = riscv::RVVDenseSegmentLoadOp::create(b, first.operation->getLoc(),
      types, descriptors, bounds, first.axis, leaf);
  riscv_internal::copyOrigin(first.operation, op);
  SmallVector<Value> results;
  llvm::append_range(results, op.getValues());
  return results;
}

unsigned coalesceSiblingTransfers(ModuleOp module, IRRewriter &b) {
  SmallVector<Operation *> originals;
  module.walk([&](Operation *op) { if (siblingAccess(op)) originals.push_back(op); });
  llvm::SmallPtrSet<Operation *, 32> erased;
  unsigned selected = 0;
  DominanceInfo dominance(module);
  for (Operation *operation : originals) {
    if (erased.contains(operation)) continue;
    auto first = *siblingAccess(operation);
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() || !kernel.getTarget().getHasSegmentMemory()) continue;
    auto accesses = adjacentFields(first);
    if (accesses.size() == 1 || !sourcesAvailable(accesses, dominance)) continue;
    SmallVector<Value> descriptors;
    for (const auto &access : accesses) descriptors.push_back(access.memory);
    auto geometry = riscv::DenseSegmentFields::get(descriptors, first.type, first.axis);
    if (!geometry || (first.store && !geometry->canStore())) continue;
    auto *anchor = first.store ? accesses.back().operation : first.operation;
    SmallVector<DescriptorRequirement> requirements{
        {geometry->base, geometry->fieldAxis, 1, false},
        {geometry->base, geometry->fieldAxis, geometry->firstField + geometry->fields, true}};
    if (first.store) requirements.push_back({geometry->base, first.axis, geometry->fields, false});
    if (excludedByBranch(anchor, requirements) ||
        llvm::any_of(requirements, [&](const auto &item) { return !item.possible(anchor); })) continue;
    if (first.store) {
      SmallVector<Value> inputs;
      int64_t resident = 0;
      for (const auto &access : accesses)
        if (!llvm::is_contained(inputs, access.value)) {
          inputs.push_back(access.value);
          auto groups = riscv_internal::residentRegisterGroups(access.value);
          if (!groups) { resident = kernel.getTarget().getVectorRegisters() + 1; break; }
          resident += *groups;
        }
      auto surrounding = riscv_internal::surroundingRegisterGroups(anchor, inputs);
      int64_t temporary = int64_t(accesses.size()) * first.type.getLayout().getRegisterGroups();
      if (!surrounding || *surrounding + resident + temporary > kernel.getTarget().getVectorRegisters()) continue;
    }
    for (const auto &access : accesses) {
      auto *descriptor = access.memory.getDefiningOp();
      if (!dominance.dominates(descriptor, anchor)) descriptor->moveBefore(anchor);
    }
    b.setInsertionPoint(anchor);
    DenseRead read{anchor, first.memory, first.type, first.axis};
    auto bounds = extents(b, read);
    Value condition;
    for (const auto &item : requirements) {
      if (item.proven(anchor)) continue;
      Value predicate = item.emit(b, anchor->getLoc());
      condition = condition ? Value(arith::AndIOp::create(b, anchor->getLoc(), condition, predicate)) : predicate;
    }
    SmallVector<Value> replacement;
    if (condition) {
      SmallVector<Type> types;
      if (!first.store) for (const auto &access : accesses) types.push_back(access.type);
      auto branch = scf::IfOp::create(b, anchor->getLoc(), types, condition, true);
      riscv_internal::copyOrigin(anchor, branch);
      auto begin = [&](Block *block) {
        if (!block->empty()) b.eraseOp(block->getTerminator());
        b.setInsertionPointToStart(block);
      };
      begin(branch.thenBlock());
      scf::YieldOp::create(b, anchor->getLoc(), emitSiblingTransfer(b, accesses, bounds));
      begin(branch.elseBlock());
      SmallVector<Value> originalResults;
      for (const auto &access : accesses) {
        auto *copy = b.clone(*access.operation);
        llvm::append_range(originalResults, copy->getResults());
      }
      scf::YieldOp::create(b, anchor->getLoc(), originalResults);
      llvm::append_range(replacement, branch.getResults());
    } else replacement = emitSiblingTransfer(b, accesses, bounds);
    for (auto [index, access] : llvm::enumerate(accesses)) {
      erased.insert(access.operation);
      if (first.store) b.eraseOp(access.operation);
      else b.replaceOp(access.operation, replacement[index]);
    }
    ++selected;
  }
  return selected;
}

class SelectRISCVDenseTransfersPass
    : public PassWrapper<SelectRISCVDenseTransfersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVDenseTransfersPass)
  SelectRISCVDenseTransfersPass() = default;
  SelectRISCVDenseTransfersPass(const SelectRISCVDenseTransfersPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "weft-riscv-select-dense-transfers"; }
  StringRef getDescription() const override {
    return "Coalesce adjacent dense fields into bounded segment transfers at their original read";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    siblingTransfers += coalesceSiblingTransfers(getOperation(), b);
    SmallVector<DenseRead> reads;
    getOperation().walk([&](Operation *operation) {
      if (auto read = denseRead(operation)) reads.push_back(*read);
    });
    for (const auto &read : reads) {
      auto plan = select(read);
      if (!plan) continue;
      b.setInsertionPoint(read.operation);
      auto bounds = extents(b, read);
      if (plan->completeFields) {
        b.replaceOp(read.operation, emitRead(b, *plan, bounds));
      } else {
        auto capacity = arith::ConstantIndexOp::create(b, read.operation->getLoc(),
            read.type.getShape()[plan->segments.fieldPosition]);
        auto condition = arith::CmpIOp::create(b, read.operation->getLoc(),
            arith::CmpIPredicate::uge, bounds[plan->segments.fieldPosition], capacity);
        auto branch = scf::IfOp::create(b, read.operation->getLoc(),
            TypeRange{read.type}, condition, true);
        riscv_internal::copyOrigin(read.operation, branch);
        b.setInsertionPointToStart(branch.thenBlock());
        scf::YieldOp::create(b, read.operation->getLoc(), emitRead(b, *plan, bounds));
        b.setInsertionPointToStart(branch.elseBlock());
        auto original = b.clone(*read.operation);
        scf::YieldOp::create(b, read.operation->getLoc(), original->getResults());
        b.replaceOp(read.operation, branch.getResults());
        ++guardedTransfers;
      }
      removedLoadIssues += plan->segments.tile.parts() - plan->segments.groups.size();
      ++segmentedTransfers;
    }
  }
private:
  Statistic siblingTransfers{this, "sibling-transfers", "Adjacent field reads or writes sharing one segment instruction"};
  Statistic segmentedTransfers{this, "segment-transfers", "Selected adjacent dense field transfers"};
  Statistic guardedTransfers{this, "field-guards", "Explicit complete-field checks preserving partial rows"};
  Statistic removedLoadIssues{this, "removed-load-issues", "Per-invocation load issues removed on segment paths"};
};
} // namespace

std::unique_ptr<Pass> weft::createSelectRISCVDenseTransfersPass() {
  return std::make_unique<SelectRISCVDenseTransfersPass>();
}
