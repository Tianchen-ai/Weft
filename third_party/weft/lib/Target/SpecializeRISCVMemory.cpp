#include "Weft/Target/RISCVPasses.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <numeric>

using namespace mlir;
using namespace weft;

namespace {
struct Transfer {
  Operation *operation;
  Value memory;
  riscv::ValueType numeric;
  int64_t axis;
  std::pair<Value, int64_t> stride;
};

bool numericJoin(Type type) {
  if (auto value = dyn_cast<riscv::ValueType>(type))
    return value.getLayout().getCarrier() == "scalar" || value.getLayout().getCarrier() == "rvv";
  // Index and scalar predicates also carry physical proofs: checked byte
  // counts, descriptor extents and complete-window/stride guards. Replacing
  // their defining operation by a control join would erase that proof even
  // when the runtime number happens to be unchanged.
  if (auto integer = dyn_cast<IntegerType>(type)) return integer.getWidth() > 1;
  return isa<FloatType>(type);
}

std::optional<Transfer> transfer(Operation *operation) {
  auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
  if (!leaf || !leaf.getInstruction().ends_with(".dynamic-stride")) return std::nullopt;
  if (auto scalar = dyn_cast<riscv::ScalarTileLoadOp>(operation)) {
    if (leaf.getParameters().size() != 1) return std::nullopt;
    int64_t axis = leaf.getParameters()[0];
    auto stride = riscv::denseArgumentStride(scalar.getSource(), axis);
    if (!stride) return std::nullopt;
    return Transfer{operation, scalar.getSource(), scalar.getResult().getType(), axis, *stride};
  }
  Value memory, numeric;
  if (auto load = dyn_cast<riscv::LoadOp>(operation)) {
    if (load.getSnapshotStorage()) return std::nullopt;
    memory = load.getRegion(); numeric = load.getResult();
  } else if (auto store = dyn_cast<riscv::StoreOp>(operation)) {
    memory = store.getRegion(); numeric = store.getValue();
  } else if (auto load = dyn_cast<riscv::RVVTileLoadOp>(operation)) {
    memory = load.getSource(); numeric = load.getResult();
  } else if (auto store = dyn_cast<riscv::RVVTileStoreOp>(operation)) {
    memory = store.getDestination(); numeric = store.getInput();
  } else return std::nullopt;
  auto type = dyn_cast<riscv::ValueType>(numeric.getType());
  if (!type || !riscv::hasDynamicDenseStride(cast<riscv::MemDescType>(memory.getType()), type))
    return std::nullopt;
  for (auto [axis, lanes] : llvm::zip(type.getAxisIds().asArrayRef(), type.getLayout().getLaneFactors().asArrayRef())) {
    if (lanes <= 1) continue;
    auto stride = riscv::denseArgumentStride(memory, axis);
    if (stride) return Transfer{operation, memory, type, axis, *stride};
  }
  return std::nullopt;
}

// Duplicate a bounded physical issue traversal, never a storage birth or an
// author Level. The branch tests immutable ABI metadata and selects exactly one
// copy of the original ordered program, including its carried numerical values.
bool canSpecialize(scf::ForOp loop) {
  if (loop->hasAttr("weft.riscv.level") || !loop->hasAttr("weft.riscv.system_unroll")) return false;
  if (llvm::any_of(loop.getResultTypes(), [](Type type) { return !numericJoin(type); })) return false;
  unsigned cost = 0;
  bool legal = true;
  loop.walk([&](Operation *operation) {
    if (++cost > 128 || isa<riscv::LocalAllocOp, riscv::LocalBindOp,
        riscv::NewOp, riscv::EmptyOp, riscv::MaterializeOp,
        riscv::RegisterMaterializeOp, riscv::StagedViewOp>(operation) ||
        operation->hasAttr("birth_id") ||
        (operation != loop && operation->hasAttr("weft.riscv.level"))) legal = false;
    if (operation->getNumRegions() && !isa<scf::ForOp, scf::IfOp>(operation)) legal = false;
  });
  return legal;
}

void select(IRRewriter &b, const Transfer &edge, bool contiguous) {
  auto leaf = edge.operation->getAttrOfType<riscv::LeafAttr>("leaf");
  if (isa<riscv::ScalarTileLoadOp>(edge.operation)) {
    StringRef instruction = contiguous ? "scalar.tile-load.unit" : "scalar.tile-load.strided";
    edge.operation->setAttr("leaf", riscv_internal::leaf(b, "transfer", "scalar-tile-load",
        instruction, instruction, 0, 0, 0, 0, "none", "exact", {edge.axis}));
    return;
  }
  auto access = edge.operation->getAttrOfType<riscv::AccessAttr>("access");
  StringRef form = contiguous ? "unit" : "strided";
  edge.operation->setAttr("access", riscv::AccessAttr::get(b.getContext(), form,
      access.getMapping(), access.getAlignment(), access.getIndexSEW(), access.getSegmentFields(),
      access.getGroupSize(), access.getLayerSize(), access.getJoinFields(), access.getJoinLowBits(),
      access.getJoinRole(), access.getBitOffset(), access.getStorageBits(), access.getOrder(),
      access.getFieldChunkBytes()));
  std::string instruction = isa<riscv::RVVTileLoadOp>(edge.operation) ? "rvv.tile-load"
      : isa<riscv::RVVTileStoreOp>(edge.operation) ? "rvv.tile-store"
      : (isa<riscv::StoreOp>(edge.operation) ? "rvv.store." : "rvv.load.") + form.str();
  if (isa<riscv::RVVTileLoadOp, riscv::RVVTileStoreOp>(edge.operation) &&
      leaf.getInstruction().starts_with(instruction + ".full")) instruction += ".full";
  edge.operation->setAttr("leaf", riscv_internal::leaf(b, leaf.getEngine(), leaf.getFamily(),
      instruction, instruction, leaf.getOperandGroups(), leaf.getResultGroups(), leaf.getTemporaryGroups(),
      leaf.getFragmentGroups(), leaf.getMask(), leaf.getTail(), leaf.getParameters().asArrayRef(), leaf.getLocalBytes()));
}

void finalizeBranches(ModuleOp module, IRRewriter &b) {
  SmallVector<scf::IfOp> branches;
  module.walk([&](scf::IfOp branch) {
    if (branch.getNumResults() && !branch.getElseRegion().empty()) branches.push_back(branch);
  });
  for (auto branch : llvm::reverse(branches)) {
    SmallVector<riscv::RVVStreamFinalizeOp> consumers;
    for (Value result : branch.getResults()) {
      if (!result.hasOneUse()) break;
      auto finalize = dyn_cast<riscv::RVVStreamFinalizeOp>(*result.getUsers().begin());
      if (!finalize || finalize->getBlock() != branch->getBlock()) break;
      consumers.push_back(finalize);
    }
    if (consumers.size() != branch.getNumResults()) continue;
    // The reduction remains at the same closed-operation boundary and runs
    // once on the chosen path. Only scalar results cross the control join;
    // a complete vector tuple no longer survives solely for that join.
    for (Region &region : branch->getRegions()) {
      auto yield = cast<scf::YieldOp>(region.front().getTerminator());
      b.setInsertionPoint(yield);
      SmallVector<Value> results;
      for (auto [position, finalize] : llvm::enumerate(consumers)) {
        IRMapping mapping;
        mapping.map(finalize.getInput(), yield.getOperand(position));
        results.push_back(b.clone(*finalize, mapping)->getResult(0));
      }
      b.replaceOpWithNewOp<scf::YieldOp>(yield, results);
    }
    for (auto [position, finalize] : llvm::enumerate(consumers)) {
      branch.getResult(position).setType(finalize.getResult().getType());
      b.replaceOp(finalize, branch.getResult(position));
    }
  }
}

// Version a straight-line physical issue as a unit. Independent per-load
// branches otherwise survive inside author blocks, retaining both contiguous
// and strided address recurrences and joining their vector results repeatedly.
// The guard only reads immutable ABI metadata; every operation is still
// executed once, in its original order, on exactly one selected path.
struct AccessRegion {
  SmallVector<Operation *> operations;
  SmallVector<Value> results;
  llvm::SetVector<std::pair<Value, int64_t>> strides;

  unsigned codeSize() const {
    unsigned result = 0;
    for (Operation *root : operations)
      root->walk([&](Operation *) { ++result; });
    return result;
  }

  SmallVector<unsigned> strideOrder() const {
    SmallVector<int64_t> traffic(strides.size(), 0);
    for (Operation *root : operations)
      root->walk([&](Operation *operation) {
        if (auto edge = transfer(operation)) {
          unsigned slot = llvm::find(strides, edge->stride) - strides.begin();
          traffic[slot] += edge->numeric.getLayout().getCarrier() == "scalar"
              ? *riscv_internal::staticProduct(edge->numeric.getShape().asArrayRef())
              : std::max<int64_t>(1, edge->numeric.getLayout().getRegisterGroups());
        }
      });
    SmallVector<unsigned> order(strides.size());
    std::iota(order.begin(), order.end(), 0);
    // Rank the actual transferred register width, without pretending to know
    // dynamic loop trip counts or a target's memory latency. Stable ties keep
    // the original ordered program's first occurrence.
    llvm::stable_sort(order, [&](unsigned a, unsigned b) { return traffic[a] > traffic[b]; });
    return order;
  }

  bool contains(Operation *operation) const {
    return llvm::any_of(operations, [&](Operation *root) {
      return root == operation || root->isProperAncestor(operation);
    });
  }

  bool close() {
    for (Operation *root : operations) {
      root->walk([&](Operation *operation) {
        if (auto edge = transfer(operation)) strides.insert(edge->stride);
      });
      for (Value result : root->getResults()) {
        if (llvm::all_of(result.getUsers(), [&](Operation *user) { return contains(user); })) continue;
        if (!numericJoin(result.getType())) return false;
        results.push_back(result);
      }
    }
    return !strides.empty();
  }
};

std::optional<unsigned> regionCost(Operation *operation) {
  if (operation->hasTrait<OpTrait::IsTerminator>() || operation->hasAttr("weft.riscv.level") ||
      operation->hasAttr("birth_id") ||
      isa<riscv::PhysicalPointOp, riscv::RootPointOp, riscv::LocalAllocOp,
          riscv::LocalBindOp, riscv::NewOp, riscv::EmptyOp, riscv::MaterializeOp,
          riscv::RegisterMaterializeOp, riscv::StagedViewOp>(operation)) return std::nullopt;
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    if (!canSpecialize(loop)) return std::nullopt;
    unsigned size = 0;
    loop.walk([&](Operation *) { ++size; });
    return size;
  }
  if (operation->getNumRegions()) return std::nullopt;
  if (isMemoryEffectFree(operation) || transfer(operation) || riscv::hasOnlyReadEffects(operation)) return 1;
  // An ordinary store has an explicit descriptor and stays in place relative
  // to every earlier read/write. No alias assumption or store motion is used.
  if (auto store = dyn_cast<riscv::StoreOp>(operation)) {
    auto encoding = dyn_cast<kernel::EncodingType>(store.getRegion().getType().getEncoding());
    if (encoding && encoding.getKind() == "dense") return 1;
  }
  return std::nullopt;
}

AccessRegion accessRegion(Operation *first) {
  AccessRegion candidate;
  unsigned cost = 0;
  for (Operation *operation = first; operation; operation = operation->getNextNode()) {
    auto size = regionCost(operation);
    if (!size || cost + *size > 128) break;
    cost += *size;
    candidate.operations.push_back(operation);
  }
  // Only numeric values may cross a control join. Trim trailing descriptor
  // producers rather than changing their representation or inventing an ABI.
  while (!candidate.operations.empty()) {
    candidate.results.clear();
    candidate.strides.clear();
    if (candidate.close()) return candidate;
    candidate.operations.pop_back();
  }
  return candidate;
}

class StrideVersions {
public:
  StrideVersions(IRRewriter &builder, AccessRegion &region)
      : b(builder), region(region), anchor(region.operations.front()), loc(anchor->getLoc()),
        order(region.strideOrder()) {
    for (Value value : region.results) types.push_back(value.getType());
  }

  void run() {
    b.setInsertionPoint(anchor);
    Value condition;
    for (auto [argument, axis] : region.strides) {
      Value stride = riscv::StrideOp::create(b, loc, b.getIndexType(), argument, axis);
      Value one = arith::ConstantIndexOp::create(b, loc, 1);
      Value equal = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, stride, one);
      predicates.push_back(equal);
      condition = condition ? Value(arith::AndIOp::create(b, loc, condition, equal)) : equal;
    }
    auto branch = scf::IfOp::create(b, loc, types, condition, true);
    riscv_internal::copyOrigin(anchor, branch);
    begin(branch.getThenRegion());
    SmallVector<std::pair<Value, int64_t>> all(region.strides.begin(), region.strides.end());
    scf::YieldOp::create(b, loc, clone(all));
    begin(branch.getElseRegion());
    // The all-contiguous path is always available. Mixed strides receive a
    // bounded decision tree rather than forcing every other contiguous input
    // through a strided instruction because one argument has a different view.
    // At most five program copies and 512 original operations are duplicated.
    // This is a code-size bound, not a cycle estimate or a hidden tuning space.
    unsigned copies = std::max(1u, std::min(4u, 512u / region.codeSize() - 1u));
    auto values = mixed(0, {}, copies);
    scf::YieldOp::create(b, loc, values);
    for (auto [before, after] : llvm::zip(region.results, branch.getResults()))
      before.replaceUsesWithIf(after, [&](OpOperand &use) { return !region.contains(use.getOwner()); });
    for (Operation *original : llvm::reverse(region.operations)) b.eraseOp(original);
  }

private:
  IRRewriter &b;
  AccessRegion &region;
  Operation *anchor;
  Location loc;
  SmallVector<Type> types;
  SmallVector<Value> predicates;
  SmallVector<unsigned> order;

  void begin(Region &arm) {
    Block &body = arm.front();
    if (!body.empty()) b.eraseOp(body.getTerminator());
    b.setInsertionPointToStart(&body);
  }

  SmallVector<Value> clone(ArrayRef<std::pair<Value, int64_t>> units) {
    IRMapping mapping;
    for (Operation *original : region.operations) {
      auto *copy = b.clone(*original, mapping);
      copy->walk([&](Operation *operation) {
        if (auto edge = transfer(operation))
          select(b, *edge, llvm::is_contained(units, edge->stride));
      });
    }
    SmallVector<Value> result;
    for (Value value : region.results) result.push_back(mapping.lookup(value));
    return result;
  }

  SmallVector<Value> mixed(unsigned position, SmallVector<std::pair<Value, int64_t>> units,
                            unsigned copies) {
    if (position == order.size() || copies == 1) return clone(units);
    // The enclosing all-unit guard is false. If every previous stride tested
    // unit, the last one cannot be unit; no redundant final predicate is needed.
    if (position + 1 == order.size() && units.size() == position) return clone(units);
    unsigned slot = order[position];
    auto branch = scf::IfOp::create(b, loc, types, predicates[slot], true);
    riscv_internal::copyOrigin(anchor, branch);
    begin(branch.getThenRegion());
    auto contiguous = units;
    contiguous.push_back(region.strides[slot]);
    auto yes = mixed(position + 1, std::move(contiguous), copies / 2);
    scf::YieldOp::create(b, loc, yes);
    begin(branch.getElseRegion());
    auto no = mixed(position + 1, std::move(units), copies - copies / 2);
    scf::YieldOp::create(b, loc, no);
    b.setInsertionPointAfter(branch);
    return SmallVector<Value>(branch.getResults().begin(), branch.getResults().end());
  }
};

class SpecializeRISCVMemoryPass
    : public PassWrapper<SpecializeRISCVMemoryPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializeRISCVMemoryPass)
  StringRef getArgument() const override { return "weft-riscv-specialize-memory"; }
  StringRef getDescription() const override { return "Select dense memory forms with explicit loop-invariant stride guards"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    llvm::SetVector<Operation *> candidates;
    bool invalid = false;
    getOperation().walk([&](Operation *operation) {
      auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
      if (!leaf || !leaf.getInstruction().ends_with(".dynamic-stride")) return;
      auto edge = transfer(operation);
      if (!edge) {
        operation->emitError("dynamic dense memory choice has no exact ABI stride relation");
        invalid = true;
        return;
      }
      if (riscv::hasUnitStrideProof(operation, edge->memory, edge->axis)) {
        select(b, *edge, true);
        return;
      }
      Operation *root = operation;
      for (Operation *parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        if (!loop || !canSpecialize(loop)) break;
        root = loop;
      }
      candidates.insert(root);
    });
    if (invalid) { signalPassFailure(); return; }
    SmallVector<Operation *> roots;
    for (Operation *candidate : candidates) {
      bool nested = false;
      for (Operation *parent = candidate->getParentOp(); parent; parent = parent->getParentOp())
        if (candidates.contains(parent)) { nested = true; break; }
      if (!nested) roots.push_back(candidate);
    }
    llvm::SmallPtrSet<Operation *, 32> consumed;
    SmallVector<AccessRegion> regions;
    for (Operation *root : roots) {
      if (consumed.contains(root)) continue;
      auto region = accessRegion(root);
      if (region.operations.empty()) {
        root->emitError("dynamic dense access has no closed numeric specialization region");
        signalPassFailure();
        return;
      }
      for (Operation *candidate : roots)
        if (region.contains(candidate)) consumed.insert(candidate);
      regions.push_back(std::move(region));
    }
    // Build all regions before erasing a root, so the candidate set never holds
    // a dangling operation while another region is being inspected.
    for (auto &region : regions) StrideVersions(b, region).run();
    finalizeBranches(getOperation(), b);
  }
};
}

std::unique_ptr<Pass> weft::createSpecializeRISCVMemoryPass() {
  return std::make_unique<SpecializeRISCVMemoryPass>();
}
