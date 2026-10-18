#include "DenseValuePrograms.h"
#include "DenseProducerGraph.h"
#include "DenseIssue.h"
#include "DenseTransfer.h"
#include "LocalMemory.h"
#include "ValueLiveness.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/DenseRelations.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"

#include <functional>
#include <limits>
#include <memory>

using namespace mlir;
using namespace weft;
using riscv_internal::DenseProducerGraph;

namespace {

class DensePrograms {
public:
  explicit DensePrograms(ModuleOp module)
      : module(module), b(module.getContext()), identity(riscv_internal::lastLocalIdentity(module)),
        shapeIdentity(riscv_internal::lastLocalShapeIdentity(module)) {}

  LogicalResult run() {
    if (failed(projectedProducers())) return failure();
    consumed.clear();
    SmallVector<Operation *> roots;
    module.walk([&](Operation *op) {
      if (isa<riscv::StoreOp, riscv::ReduceOp>(op)) roots.push_back(op);
    });
    for (Operation *root : roots) {
      if (consumed.contains(root)) continue;
      Value value;
      if (auto store = dyn_cast<riscv::StoreOp>(root)) {
        auto type = dyn_cast<riscv::ValueType>(store.getValue().getType());
        if (!type || type.getShape() != store.getRegion().getType().getShape() ||
            type.getAxisIds() != store.getRegion().getType().getAxisIds()) continue;
        value = store.getValue();
      } else {
        auto reduce = cast<riscv::ReduceOp>(root);
        if (reduce.getKind() != "add" && reduce.getKind() != "max" && reduce.getKind() != "min") continue;
        value = reduce.getInput();
      }
      SmallVector<Operation *> candidates{root};
      auto selected = std::make_unique<DenseProducerGraph>(candidates);
      if (!selected->analyze(true, true)) selected.reset();
      unsigned distance = 0;
      for (Operation *next = root->getNextNode(); next && ++distance <= 256 && candidates.size() < 4;
           next = next->getNextNode()) {
        if (next->getNumRegions()) break;
        bool sink = isa<riscv::StoreOp, riscv::ReduceOp>(next);
        if (!sink && !riscv::hasOnlyReadEffects(next)) break;
        if (!sink || consumed.contains(next)) continue;
        if (auto reduction = dyn_cast<riscv::ReduceOp>(next);
            reduction && reduction.getKind() != "add" && reduction.getKind() != "min" && reduction.getKind() != "max") break;
        candidates.push_back(next);
        auto graph = std::make_unique<DenseProducerGraph>(candidates);
        if (graph->analyze(true, true)) selected = std::move(graph);
      }
      // Decide placement after finding the complete compatible consumer set.
      // A greedy first store can otherwise force a shared Boolean/value into
      // private storage even though the complete original graph fits registers.
      if (selected && selected->needsRealization()) {
        auto specialized = specializeAliasing(*selected);
        if (failed(specialized)) return failure();
        if (!*specialized && selected->aliasRequirements.empty() && failed(realize(*selected))) return failure();
      }
    }
    return success();
  }

  LogicalResult required(Operation *root) {
    DenseProducerGraph graph(ArrayRef<Operation *>{root});
    if (!graph.analyze(true))
      return root->emitError("whole-value consumer has no closed bounded producer graph: ") << graph.failureReason;
    return realize(graph);
  }

private:
  ModuleOp module;
  IRRewriter b;
  int64_t identity = 0;
  int64_t shapeIdentity = 0;
  llvm::SmallPtrSet<Operation *, 32> consumed;
  riscv::ValueType activeDomain;
  int64_t lanePosition = 0;
  int64_t laneWidth = 0;
  SmallVector<Value> coordinates;
  SmallVector<Value> domainExtents;
  SmallVector<Value> memoryExtents;
  SmallVector<int64_t> tileShape;
  llvm::DenseSet<Value> registerInputs;
  DenseMap<Value, SmallVector<Value>> registerOutputs;
  unsigned aliasSpecializations = 0;

  FailureOr<bool> specializeAliasing(DenseProducerGraph &graph) {
    // Only a closed pointwise store region is duplicated. Author Levels,
    // storage births and numerical values observed outside the region stay at
    // their original location. Both physical arms implement the same ordered
    // source operations; exactly one is executed on an invocation.
    if (aliasSpecializations >= 4 || !graph.exports.empty() || !graph.reductions.empty() ||
        graph.stores.size() != graph.sinks.size() || graph.order.size() > 96) return false;
    for (Operation *op : graph.order) {
      if (op->getNumRegions() || op->hasAttr("birth_id") ||
          (!isa<riscv::LoadOp, riscv::ConvertLayoutOp, riscv::IotaOp>(op) &&
           !riscv_internal::isDensePointwise(op))) return false;
      for (Operation *user : op->getResult(0).getUsers())
        if (!graph.members.contains(user) && !graph.ownsConsumer(user)) return false;
    }
    // A later materializer may visit the overlap arm again. Its original
    // snapshot is already a selected representation, not another search root.
    for (Operation *parent = graph.root->getParentOp(); parent; parent = parent->getParentOp()) {
      auto branch = dyn_cast<scf::IfOp>(parent);
      if (!branch) continue;
      SmallVector<Value> pending{branch.getCondition()};
      for (unsigned count = 0; !pending.empty() && count < 64; ++count) {
        Value value = pending.pop_back_val();
        if (value.getDefiningOp<riscv::DenseDisjointOp>() ||
            value.getDefiningOp<riscv::DenseInjectiveOp>() ||
            value.getDefiningOp<riscv::DenseElementwiseAliasOp>()) return false;
        if (auto both = value.getDefiningOp<arith::AndIOp>()) pending.append({both.getLhs(), both.getRhs()});
        if (auto either = value.getDefiningOp<arith::OrIOp>()) pending.append({either.getLhs(), either.getRhs()});
      }
    }
    llvm::SetVector<std::pair<Value, Value>> pairs;
    DenseMap<std::pair<Value, Value>, bool> sameProjection;
    auto addPair = [&](Value lhs, Value rhs, bool equal) {
      auto left = cast<riscv::MemoryViewOp>(lhs.getDefiningOp());
      auto right = cast<riscv::MemoryViewOp>(rhs.getDefiningOp());
      if (cast<BlockArgument>(left.getBase()).getArgNumber() > cast<BlockArgument>(right.getBase()).getArgNumber())
        std::swap(lhs, rhs);
      pairs.insert({lhs, rhs});
      auto [position, inserted] = sameProjection.try_emplace({lhs, rhs}, equal);
      if (!inserted) position->second &= equal;
    };
    for (auto requirement : graph.aliasRequirements)
      addPair(requirement.lhs, requirement.rhs, requirement.allowEqualMapping);
    for (auto read : graph.reads) {
      for (Operation *next = read->getNextNode(); next != graph.root; next = next->getNextNode()) {
        if (!next || (!graph.ownsConsumer(next) && !riscv::hasOnlyReadEffects(next, true))) return false;
      }
      for (auto store : graph.stores) {
        if (riscv::densePointwiseIndependentAt(read.getRegion(), store.getRegion(), graph.root)) continue;
        Value lhs = riscv::fullDenseArgumentView(read.getRegion());
        Value rhs = riscv::fullDenseArgumentView(store.getRegion());
        if (!lhs || !rhs) return false;
        auto relation = riscv::denseProjectionRelation(read.getRegion(), store.getRegion());
        bool equal = relation && relation->pointwise;
        if (lhs == rhs && !equal) return false;
        addPair(lhs, rhs, equal);
      }
    }
    if (pairs.empty() || pairs.size() > 4) return false;
    auto kernel = graph.root->getParentOfType<riscv::KernelOp>();
    Operation *last = nullptr;
    for (auto [lhs, rhs] : pairs)
      for (Value value : {lhs, rhs}) {
        Operation *view = value.getDefiningOp();
        if (view->getBlock() != &kernel.getBody().front()) return false;
        if (!last || last->isBeforeInBlock(view)) last = view;
      }
    // Bounds use immutable descriptor metadata. Compute the proof once before
    // every author traversal, without moving the reads or their effects.
    b.setInsertionPointAfter(last);
    Value condition;
    for (auto [lhs, rhs] : pairs) {
      Value disjoint;
      if (lhs == rhs) {
        if (!sameProjection.lookup({lhs, rhs}))
          return graph.root->emitError("same-storage streaming requires an injective coordinate relation"), failure();
        disjoint = riscv::DenseInjectiveOp::create(b, graph.root->getLoc(), b.getI1Type(), lhs);
      } else {
        disjoint = riscv::DenseDisjointOp::create(b, graph.root->getLoc(), b.getI1Type(), lhs, rhs);
      }
      if (lhs != rhs && sameProjection.lookup({lhs, rhs})) {
        Value alias = riscv::DenseElementwiseAliasOp::create(b, graph.root->getLoc(), b.getI1Type(), lhs, rhs);
        disjoint = arith::OrIOp::create(b, graph.root->getLoc(), disjoint, alias);
      }
      condition = condition ? Value(arith::AndIOp::create(b, graph.root->getLoc(), condition, disjoint)) : disjoint;
    }
    b.setInsertionPoint(graph.root);
    auto branch = scf::IfOp::create(b, graph.root->getLoc(), condition, true);
    riscv_internal::copyOrigin(graph.root, branch);
    SmallVector<Operation *> ordered(graph.order);
    for (Operation *sink : graph.sinks)
      if (!llvm::is_contained(ordered, sink)) ordered.push_back(sink);
    llvm::stable_sort(ordered, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    SmallVector<SmallVector<Operation *>, 2> arms;
    for (Region *region : {&branch.getThenRegion(), &branch.getElseRegion()}) {
      b.setInsertionPointToStart(&region->front());
      IRMapping mapping;
      SmallVector<Operation *> sinks;
      for (Operation *op : ordered) {
        Operation *copy = b.clone(*op, mapping);
        if (graph.ownsConsumer(op)) sinks.push_back(copy);
      }
      arms.push_back(std::move(sinks));
    }
    for (Operation *sink : graph.sinks) { consumed.insert(sink); b.eraseOp(sink); }
    for (Operation *op : llvm::reverse(graph.order)) { consumed.insert(op); b.eraseOp(op); }
    ++aliasSpecializations;
    for (auto [arm, sinks] : llvm::enumerate(arms)) {
      DenseProducerGraph version(sinks);
      if (version.analyze(true)) {
        if (failed(realize(version))) return failure();
      } else if (arm == 0) {
        return branch.emitError("selected dense alias arm lost its closed producer graph"), failure();
      }
      // An unmerged overlap arm remains the original ordered numerical graph.
      // The ordinary Value materializer closes its snapshots; streaming each
      // sink here would hide intervening writes from the later sink's analysis.
    }
    return true;
  }

  bool uniform(Value value) const {
    auto type = dyn_cast<riscv::ValueType>(value.getType());
    return type && type.getLayout().getCarrier() == "scalar" &&
        llvm::all_of(type.getShape().asArrayRef(), [](int64_t n) { return n == 1; });
  }

  bool registerCarrier(riscv::ValueType type, DenseProducerGraph &graph, int64_t extent) {
    auto layout = type.getLayout();
    if (type.getShape() != graph.domain.getShape() || type.getAxisIds() != graph.domain.getAxisIds() ||
        layout.getCarrier() != "rvv" || layout.getReplicaGroup() != 1 || layout.getVl() != graph.width ||
        layout.getLaneFactors()[graph.lanePosition] != graph.width ||
        layout.getTimeFactors()[graph.lanePosition] != extent / graph.width) return false;
    for (size_t position = 0; position < type.getShape().size(); ++position) {
      if (layout.getReplicaFactors()[position] != 1 || layout.getFragmentFactors()[position] != 1 ||
          layout.getLocalFactors()[position] != 1) return false;
      if (int64_t(position) != graph.lanePosition &&
          (type.getShape()[position] != 1 || layout.getTimeFactors()[position] != 1 ||
           layout.getLaneFactors()[position] != 1)) return false;
    }
    return true;
  }

  bool selectRegisterPieces(DenseProducerGraph &graph, Value upper) {
    registerInputs.clear();
    registerOutputs.clear();
    if (graph.issueCopies != 1) return false;
    auto extent = getConstantIntValue(upper);
    if (!extent || *extent <= 0 || *extent != graph.domain.getShape()[graph.lanePosition] ||
        *extent % graph.width || *extent / graph.width > 4) return false;
    auto enclosing = graph.root->getParentOfType<scf::ForOp>();
    if (!graph.reductions.empty() && !enclosing) return false;
    int64_t retainedInputs = 0;
    for (Value input : graph.inputs) {
      auto read = input.getDefiningOp<riscv::ConvertLayoutOp>();
      if (read && riscv_internal::isCompleteLocalRead(read)) continue;
      auto resident = riscv_internal::residentRegisterGroups(input);
      if (!resident || !*resident) continue;
      auto type = cast<riscv::ValueType>(input.getType());
      if (registerCarrier(type, graph, *extent)) {
        registerInputs.insert(input);
        retainedInputs += *resident - type.getLayout().getRegisterGroups();
      }
    }
    if (registerInputs.empty()) return false;
    int64_t retained = 0;
    for (Value value : graph.exports) {
      auto type = cast<riscv::ValueType>(value.getType());
      if (!registerCarrier(type, graph, *extent)) continue;
      registerOutputs.try_emplace(value);
      retained += type.getLayout().getRegisterGroups() * (*extent / graph.width);
    }
    // Completed pieces remain live while a later issue is evaluated. Account
    // for that extra residency before removing their private representation.
    auto target = graph.root->getParentOfType<riscv::KernelOp>().getTarget();
    if (graph.issueCost.vectorGroups + retained + retainedInputs > target.getVectorRegisters()) {
      registerInputs.clear();
      registerOutputs.clear();
      return false;
    }
    if (enclosing) enclosing->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    return true;
  }

  FailureOr<Value> registerPiece(Value input, riscv::ValueType type, Value offset, Location loc) {
    auto begin = getConstantIntValue(offset);
    auto source = cast<riscv::ValueType>(input.getType());
    if (!begin || *begin < 0 || *begin % type.getLayout().getVl()) return failure();
    int64_t part = *begin / type.getLayout().getVl();
    auto slice = riscv::RVVIssueSliceOp::create(b, loc, type, input,
        b.getDenseI64ArrayAttr({part}), b.getDenseI64ArrayAttr({0}),
        riscv_internal::leaf(b, "transfer", "issue-slice", "rvv.issue-slice", "rvv.issue-slice",
            source.getLayout().getRegisterGroups(), type.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
    return slice.getResult();
  }

  // A value on a strict subdomain has its own numerical evaluation. Inlining
  // its math into a larger lane domain can turn a vector exp/log into one
  // scalar libm call per replica and repeat it over every consumer window.
  // Close that existing producer once on its own axes before broadcasting its
  // representation. This neither replays a load nor invents an author Level.
  LogicalResult projectedProducers() {
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (!riscv_internal::isDensePointwise(op)) return;
      auto type = dyn_cast<riscv::ValueType>(op->getResult(0).getType());
      if (!type || !type.getElementType().isF32() || type.getShape().empty() ||
          type.getLayout().getCarrier() == "local" || uniform(op->getResult(0))) return;
      bool broadcast = llvm::any_of(op->getResult(0).getUsers(), [&](Operation *user) {
        if (!riscv_internal::isDensePointwise(user)) return false;
        auto output = dyn_cast<riscv::ValueType>(user->getResult(0).getType());
        return output && output.getShape().size() > type.getShape().size() &&
            llvm::all_of(type.getAxisIds().asArrayRef(), [&](int64_t axis) {
              return llvm::is_contained(output.getAxisIds().asArrayRef(), axis);
            });
      });
      if (broadcast) candidates.push_back(op);
    });
    for (Operation *root : llvm::reverse(candidates)) {
      if (consumed.contains(root)) continue;
      DenseProducerGraph graph(ArrayRef<Operation *>{root});
      if (!graph.analyze(true) || graph.width <= 1) continue;
      bool scalarMath = llvm::any_of(graph.order, [](Operation *op) {
        auto unary = dyn_cast<riscv::UnaryOp>(op);
        if (!unary || (unary.getKind() != "exp" && unary.getKind() != "exp2" && unary.getKind() != "log")) return false;
        auto type = dyn_cast<riscv::ValueType>(unary.getResult().getType());
        return type && type.getLayout().getCarrier() == "scalar";
      });
      if (scalarMath && failed(realize(graph))) return failure();
    }
    return success();
  }

  FailureOr<Value> freezeBoundary(Value value, Operation *consumer) {
    auto type = cast<riscv::ValueType>(value.getType());
    // Select the first point after the value exists. A loop-invariant source
    // then gets one private copy outside the loop; a carried version gets one
    // copy at entry to its own iteration. Static register domains alone reach
    // this path, so no consumer-only dynamic extent is needed at the birth.
    Operation *anchor = nullptr;
    if (auto argument = dyn_cast<BlockArgument>(value))
      anchor = &argument.getOwner()->front();
    else if (Operation *definition = value.getDefiningOp())
      anchor = definition->getNextNode();
    if (!anchor) return consumer->emitError("numeric boundary has no ordered storage point"), failure();
    DominanceInfo dominance(module);
    if (anchor != consumer && !dominance.properlyDominates(anchor, consumer))
      return consumer->emitError("numeric boundary storage does not dominate its consumer"), failure();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(anchor);
    auto snapshot = allocate(type, anchor, "physical-share");
    if (failed(snapshot)) return failure();
    auto region = riscv_internal::completeLocalSlice(b, snapshot->memory);
    auto stored = riscv::StoreOp::create(b, consumer->getLoc(), value, region,
        riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
    riscv_internal::copyOrigin(value.getDefiningOp() ? value.getDefiningOp() : consumer, stored);
    return snapshot->binding;
  }

  int64_t laneAxis() const { return activeDomain.getAxisIds()[lanePosition]; }

  Value index(Location loc, int64_t value) {
    return arith::ConstantIndexOp::create(b, loc, value);
  }

  Value activeExtent(DenseProducerGraph &graph, size_t position) {
    if (auto extent = riscv_internal::completePointExtent(graph.points[position]))
      return index(graph.root->getLoc(), *extent);
    return graph.points[position] ? graph.points[position].getActive() : domainExtents[position];
  }

  riscv::ValueType piece(riscv::ValueType source, int64_t width,
                         riscv::TargetAttr target, bool tail) {
    auto selected = tileShape;
    selected[lanePosition] = width;
    return riscv_internal::valueTileType(b, source, activeDomain.getAxisIds().asArrayRef(),
        selected, laneAxis(), target, tail, laneWidth);
  }

  FailureOr<riscv_internal::DenseTransferWindow> window(Value memory, Value offset,
      int64_t width, Operation *origin) {
    Location loc = origin->getLoc();
    auto old = cast<riscv::MemDescType>(memory.getType());
    SmallVector<int64_t> shape;
    SmallVector<Value> offsets, active;
    for (auto [position, axis] : llvm::enumerate(old.getAxisIds().asArrayRef())) {
      auto found = llvm::find(activeDomain.getAxisIds().asArrayRef(), axis);
      if (found == activeDomain.getAxisIds().asArrayRef().end())
        return origin->emitError("dense memory window has an axis outside the selected issue domain"), failure();
      size_t coordinate = found - activeDomain.getAxisIds().asArrayRef().begin();
      bool broadcast = old.getShape()[position] == 1;
      offsets.push_back(broadcast ? index(loc, 0) : axis == laneAxis() ? offset : coordinates[coordinate]);
      shape.push_back(broadcast ? 1 : axis == laneAxis() ? width : tileShape[coordinate]);
      if (broadcast) active.push_back(index(loc, 1));
      else {
        Value end = memoryExtents[coordinate];
        Value start = arith::MinUIOp::create(b, loc, end, offsets.back());
        Value remaining = arith::SubIOp::create(b, loc, end, start);
        active.push_back(arith::MinUIOp::create(b, loc, remaining, index(loc, shape.back())));
      }
    }
    return riscv_internal::denseTransferWindow(b, memory, offsets, shape, active, shapeIdentity, origin);
  }

  FailureOr<Value> load(Value memory, Value offset, riscv::ValueType type, Operation *origin) {
    auto position = llvm::find(type.getAxisIds().asArrayRef(), laneAxis());
    int64_t width = position == type.getAxisIds().asArrayRef().end() ? 1
        : type.getShape()[position - type.getAxisIds().asArrayRef().begin()];
    auto selected = window(memory, offset, width, origin);
    if (failed(selected)) return failure();
    return riscv_internal::loadDenseWindow(b, *selected, type, origin);
  }

  LogicalResult store(Value value, Value memory, Value offset, int64_t width, Operation *origin) {
    auto selected = window(memory, offset, width, origin);
    if (failed(selected)) return failure();
    return riscv_internal::storeDenseWindow(b, value, *selected, origin);
  }

  LogicalResult storeResults(DenseProducerGraph &graph, ValueRange values,
                             Value offset, int64_t width) {
    SmallVector<std::pair<unsigned, riscv_internal::DenseTransferWindow>> windows;
    for (auto [position, sink] : llvm::enumerate(graph.sinks)) {
      auto destination = dyn_cast<riscv::StoreOp>(sink);
      if (!destination) continue;
      auto selected = window(destination.getRegion(), offset, width, sink);
      if (failed(selected)) return failure();
      windows.emplace_back(position, *selected);
    }
    for (auto &[position, selected] : windows)
      if (failed(riscv_internal::storeDenseWindow(b, values[position], selected, graph.sinks[position])))
        return failure();
    return success();
  }

  LogicalResult storeReduction(Value value, Value memory, Operation *origin) {
    Location loc = origin->getLoc();
    if (isa<riscv::ValueType>(value.getType())) {
      return store(value, memory, index(loc, 0), 1, origin);
    }
    auto full = cast<riscv::MemDescType>(memory.getType());
    SmallVector<Value> indices;
    for (int64_t axis : full.getAxisIds().asArrayRef()) {
      auto found = llvm::find(activeDomain.getAxisIds().asArrayRef(), axis);
      if (found == activeDomain.getAxisIds().asArrayRef().end()) return failure();
      indices.push_back(coordinates[found - activeDomain.getAxisIds().asArrayRef().begin()]);
    }
    auto empty = b.getDenseI64ArrayAttr({});
    auto element = riscv::MemDescType::get(b.getContext(), full.getEncoding(), empty, empty,
        empty, empty, full.getAlignment(), "slice", full.getAccess(), full.getAliasSet(),
        full.getLayoutIdentity(), full.getStorageBits(), full.getElements(), full.getInterleaveRows());
    Value address = riscv::SliceOp::create(b, loc, element, memory, indices,
        b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))));
    riscv::StoreOp::create(b, loc, value, address, riscv_internal::unassignedAccess(b),
        riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
    return success();
  }

  Type reducedType(riscv::ValueType input) {
    SmallVector<int64_t> shape, axes;
    for (auto [extent, axis] : llvm::zip(input.getShape().asArrayRef(), input.getAxisIds().asArrayRef()))
      if (axis != laneAxis()) { shape.push_back(extent); axes.push_back(axis); }
    if (shape.empty()) return input.getElementType();
    auto project = [&](DenseI64ArrayAttr values) {
      SmallVector<int64_t> retained;
      for (auto [axis, value] : llvm::zip(input.getAxisIds().asArrayRef(), values.asArrayRef()))
        if (axis != laneAxis()) retained.push_back(value);
      return b.getDenseI64ArrayAttr(retained);
    };
    auto before = input.getLayout();
    auto layout = riscv::LayoutAttr::get(b.getContext(), "scalar", b.getDenseI64ArrayAttr(axes),
        project(before.getTimeFactors()), project(before.getLaneFactors()),
        project(before.getReplicaFactors()), project(before.getFragmentFactors()),
        project(before.getLocalFactors()), before.getSew(), 0, 1, 0, "full");
    return riscv::ValueType::get(b.getContext(), input.getElementType(),
        b.getDenseI64ArrayAttr(shape), b.getDenseI64ArrayAttr(axes), layout);
  }

  Value reductionInput(Value input, bool scalar, Location loc) {
    auto type = cast<riscv::ValueType>(input.getType());
    if (!scalar) return input;
    SmallVector<Value> indices;
    SmallVector<Attribute> selectors;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      bool retain = axis != laneAxis();
      if (!retain && type.getShape()[position] != 1) return {};
      selectors.push_back(b.getStringAttr(retain ? "all" : "index"));
      if (!retain) indices.push_back(index(loc, 0));
    }
    Type result = reducedType(type);
    return riscv::ExtractOp::create(b, loc, result, input, indices, b.getArrayAttr(selectors),
        riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
  }

  // The point only narrows physical execution of the existing axis. Its parent,
  // domain identity and birth ownership are inherited, not a new author Level.
  void scope(DenseProducerGraph &graph, Value offset, Value active, int64_t width) {
    for (auto [position, point] : llvm::enumerate(graph.points)) {
      if (!point) continue;
      bool lane = int64_t(position) == lanePosition;
      int64_t span = lane ? width : tileShape[position];
      Value extent = lane ? active : Value(arith::MinUIOp::create(b, graph.root->getLoc(),
          arith::SubIOp::create(b, graph.root->getLoc(), activeExtent(graph, position), coordinates[position]),
          index(graph.root->getLoc(), span)));
      Value base = arith::AddIOp::create(b, graph.root->getLoc(), point.getBase(),
          lane ? offset : coordinates[position]);
      riscv::PhysicalPointOp::create(b, graph.root->getLoc(), point.getType(), point.getParent(),
          base, extent, index(graph.root->getLoc(), span));
    }
  }

  scf::ForOp loop(Operation *anchor, Value lower, Value upper, int64_t step,
                  ValueRange initial = {}) {
    auto iteration = scf::ForOp::create(b, anchor->getLoc(), lower, upper,
        index(anchor->getLoc(), step), initial);
    iteration->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    iteration->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    riscv_internal::copyOrigin(anchor, iteration);
    return iteration;
  }

  struct Snapshot { Value binding; Value memory; };

  FailureOr<Snapshot> allocate(riscv::ValueType type, Operation *anchor,
                                StringRef schema) {
    auto binding = riscv_internal::allocateLocalValue(b, type, anchor, identity, false, schema);
    if (failed(binding)) return failure();
    auto memory = riscv_internal::localView(b, *binding, anchor);
    if (failed(memory)) return failure();
    return Snapshot{*binding, *memory};
  }

  // Traverse only axes present in this value. A rank-subdomain supply is read
  // once, rather than once for every broadcast coordinate of its consumer.
  LogicalResult rows(DenseProducerGraph &graph, riscv::ValueType type,
                     function_ref<LogicalResult()> emit) {
    auto saved = coordinates;
    std::function<LogicalResult(size_t)> visit = [&](size_t dimension) -> LogicalResult {
      if (dimension == type.getShape().size()) return emit();
      int64_t axis = type.getAxisIds()[dimension];
      auto found = llvm::find(activeDomain.getAxisIds().asArrayRef(), axis);
      size_t position = found - activeDomain.getAxisIds().asArrayRef().begin();
      if (int64_t(position) == lanePosition) return visit(dimension + 1);
      if (type.getShape()[dimension] == 1) {
        coordinates[position] = index(graph.root->getLoc(), 0);
        return visit(dimension + 1);
      }
      Value extent = activeExtent(graph, position);
      auto iteration = loop(graph.root, index(graph.root->getLoc(), 0), extent, tileShape[position]);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(iteration.getBody());
      coordinates[position] = iteration.getInductionVar();
      return visit(dimension + 1);
    };
    auto status = visit(0);
    coordinates = saved;
    return status;
  }

  FailureOr<Snapshot> snapshot(DenseProducerGraph &graph, Operation *read,
                              Value memory, riscv::ValueType type, Value upper) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(read);
    auto frozen = allocate(type, read, "local-value-copy");
    if (failed(frozen)) return failure();
    auto target = graph.root->getParentOfType<riscv::KernelOp>().getTarget();
    if (failed(rows(graph, type, [&]() -> LogicalResult {
      auto axis = llvm::find(type.getAxisIds().asArrayRef(), laneAxis());
      bool varies = axis != type.getAxisIds().asArrayRef().end() &&
          type.getShape()[axis - type.getAxisIds().asArrayRef().begin()] != 1;
      Value bound = varies ? upper : index(read->getLoc(), 1);
      Value vectorEnd = graph.point ? bound : riscv_internal::completeDenseExtent(b, read->getLoc(), bound, graph.width);
      auto copied = riscv_internal::emitDenseIssues(b, read, vectorEnd, graph.width, {},
          [&](Value offset, Value active, bool full, ValueRange) -> FailureOr<SmallVector<Value>> {
        scope(graph, offset, active, graph.width);
        auto tile = piece(type, graph.width, target, !full);
        if (!tile || (!full && !graph.point)) return failure();
        auto value = load(memory, offset, tile, read);
        if (failed(value) || failed(store(*value, frozen->memory, offset, graph.width, read))) return failure();
        return SmallVector<Value>{};
      }, 1, bool(graph.point));
      if (failed(copied)) return failure();
      if (!graph.point) {
        auto remainder = loop(read, vectorEnd, bound, 1);
        OpBuilder::InsertionGuard body(b);
        b.setInsertionPointToStart(remainder.getBody());
        auto tile = piece(type, 1, target, false);
        if (!tile) return failure();
        auto value = load(memory, remainder.getInductionVar(), tile, read);
        if (failed(value) || failed(store(*value, frozen->memory, remainder.getInductionVar(), 1, read)))
          return failure();
      }
      return success();
    }))) return failure();
    return *frozen;
  }

  FailureOr<SmallVector<Value>> expression(DenseProducerGraph &graph, const DenseMap<Value, Value> &supplies,
                               const DenseMap<Value, Value> &outputs,
                               Value offset, int64_t width, bool tail) {
    IRMapping mapping;
    auto target = graph.root->getParentOfType<riscv::KernelOp>().getTarget();
    DenseMap<Value, riscv_internal::DenseTransferWindow> windows;
    // Build metadata before numeric issues. Shape definitions are not
    // speculatable operations for a later scheduler to move across reads.
    // Selecting their position here exposes an uninterrupted read frontier
    // while keeping every original numeric producer's order below.
    for (Operation *op : graph.order) {
      Value value = op->getResult(0);
      Value memory = supplies.lookup(value);
      if (!memory) continue;
      auto type = piece(cast<riscv::ValueType>(value.getType()), width, target, tail);
      if (!type) return failure();
      auto position = llvm::find(type.getAxisIds().asArrayRef(), laneAxis());
      int64_t span = position == type.getAxisIds().asArrayRef().end() ? 1
          : type.getShape()[position - type.getAxisIds().asArrayRef().begin()];
      auto selected = window(memory, offset, span, op);
      if (failed(selected)) return failure();
      windows.try_emplace(value, *selected);
    }
    for (Value value : graph.inputs) {
      if (uniform(value)) {
        // A one-element shaped value is already one evaluated scalar SSA. Its
        // broadcast has no address traversal and needs no private-array copy.
        mapping.map(value, value);
        continue;
      }
      auto type = piece(cast<riscv::ValueType>(value.getType()), width, target, tail);
      if (!type) return failure();
      if (registerInputs.contains(value)) {
        auto selected = registerPiece(value, type, offset, graph.root->getLoc());
        if (failed(selected)) return failure();
        mapping.map(value, *selected);
      } else {
        if (!supplies.count(value)) return failure();
        auto loaded = load(supplies.lookup(value), offset, type, graph.root);
        if (failed(loaded)) return failure();
        mapping.map(value, *loaded);
      }
    }
    for (Operation *op : graph.order) {
      auto type = piece(cast<riscv::ValueType>(op->getResult(0).getType()), width, target, tail);
      if (!type) return failure();
      if (auto broadcast = dyn_cast<riscv::RVVAxisBroadcastOp>(op)) {
        // Pointwise operations already align their operands by logical axis.
        // Only internal single-consumer broadcasts enter this producer graph;
        // the subsequent layout pass closes the smaller issue's physical form.
        mapping.map(broadcast.getResult(), mapping.lookup(broadcast.getInput()));
        continue;
      }
      Value result;
      if (auto read = dyn_cast<riscv::LoadOp>(op)) {
        auto loaded = riscv_internal::loadDenseWindow(b, windows.at(read.getResult()), type, op);
        if (failed(loaded)) return failure();
        result = *loaded;
      } else if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op)) {
        if (Value memory = supplies.lookup(conversion.getResult())) {
          auto loaded = riscv_internal::loadDenseWindow(b, windows.at(conversion.getResult()), type, op);
          if (failed(loaded)) return failure();
          result = *loaded;
        } else {
          result = mapping.lookup(conversion.getInput());
          if (result.getType() != type) {
            auto input = dyn_cast<riscv::ValueType>(result.getType());
            bool embedded = input && input.getElementType() == type.getElementType() &&
                input.getAxisIds().size() < type.getAxisIds().size() &&
                !outputs.count(conversion.getResult()) && !registerOutputs.count(conversion.getResult()) &&
                !llvm::is_contained(graph.values, conversion.getResult());
            if (embedded)
              for (auto [position, axis] : llvm::enumerate(input.getAxisIds().asArrayRef())) {
                auto found = llvm::find(type.getAxisIds().asArrayRef(), axis);
                embedded &= found != type.getAxisIds().asArrayRef().end() &&
                    input.getShape()[position] == type.getShape()[found - type.getAxisIds().asArrayRef().begin()];
              }
            if (embedded && llvm::all_of(conversion.getResult().getUsers(), [](Operation *user) {
                  return isa<riscv::BinaryOp, riscv::CompareOp, riscv::SelectOp, riscv::ConvertLayoutOp>(user);
                })) {
              // A pure conversion following an internal broadcast must not
              // reintroduce the full tensor. Its pointwise consumers retain
              // the same axis embedding and close their own issue layouts.
              mapping.map(conversion.getResult(), result);
              continue;
            }
            conversion.emitError("bounded layout conversion has mismatched issue types; input=")
                << result.getType() << ", expected=" << type;
            return failure();
          }
        }
      } else if (auto iota = dyn_cast<riscv::IotaOp>(op)) {
        int64_t axis = type.getAxisIds()[0];
        auto position = llvm::find(activeDomain.getAxisIds().asArrayRef(), axis) -
            activeDomain.getAxisIds().asArrayRef().begin();
        Value coordinate = axis == laneAxis() ? offset : coordinates[position];
        Type element = type.getElementType();
        Value base = riscv::CastOp::create(b, op->getLoc(), element, coordinate,
            riscv_internal::unselectedLeaf(b));
        if (iota.getStart()) {
          Value start = riscv::ConstantOp::create(b, op->getLoc(), element,
              b.getIntegerAttr(element, iota.getStart()));
          base = riscv::BinaryOp::create(b, op->getLoc(), element, base, start,
              "add", riscv_internal::unselectedLeaf(b));
        }
        Value offsets = riscv::IotaOp::create(b, op->getLoc(), type, 0,
            type.getShape()[0], riscv_internal::unselectedLeaf(b));
        result = riscv::BinaryOp::create(b, op->getLoc(), type, offsets, base,
            "add", riscv_internal::unselectedLeaf(b));
      } else {
        Operation *computed = b.clone(*op, mapping);
        computed->getResult(0).setType(type);
        computed->setAttr("leaf", riscv_internal::unselectedLeaf(b));
        computed->removeAttr("implementation");
        if (auto lookup = dyn_cast<riscv::LookupOp>(computed))
          lookup.setAccessAttr(riscv_internal::unassignedAccess(b));
        result = computed->getResult(0);
      }
      riscv_internal::copyOrigin(op, result.getDefiningOp());
      mapping.map(op->getResult(0), result);
      if (Value output = outputs.lookup(op->getResult(0)); output && failed(store(result, output, offset, width, op)))
        return failure();
      if (auto saved = registerOutputs.find(op->getResult(0)); saved != registerOutputs.end())
        saved->second.push_back(result);
    }
    SmallVector<Value> results;
    for (Value value : graph.values) results.push_back(mapping.lookup(value));
    return results;
  }

  LogicalResult realize(DenseProducerGraph &graph) {
    if (!graph.aliasRequirements.empty())
      return graph.root->emitError("dense realization has undischarged alias proof obligations");
    Location loc = graph.root->getLoc();
    activeDomain = graph.domain;
    lanePosition = graph.lanePosition;
    laneWidth = graph.laneWidth;
    tileShape = graph.tileShape;
    Operation *first = graph.order.empty() ? graph.root : graph.order.front();
    for (Operation *op : graph.order)
      if (op->isBeforeInBlock(first)) first = op;
    b.setInsertionPoint(first);
    coordinates.assign(graph.domain.getShape().size(), index(loc, 0));
    auto extents = riscv_internal::valueExtentsAt(b, first, graph.domain);
    if (failed(extents)) return graph.root->emitError("dense issue domain has no actual extent for every logical axis");
    domainExtents = *extents;
    memoryExtents.clear();
    for (size_t position = 0; position < domainExtents.size(); ++position)
      memoryExtents.push_back(activeExtent(graph, position));
    Value upper = activeExtent(graph, lanePosition);
    bool registerPieces = selectRegisterPieces(graph, upper);
    DenseMap<Value, Value> supplies;
    DenseMap<Value, Value> frozen;
    DenseMap<Value, Value> outputs;
    for (Value input : graph.inputs) {
      if (uniform(input) || registerInputs.contains(input)) continue;
      b.setInsertionPoint(graph.root);
      auto type = cast<riscv::ValueType>(input.getType());
      Value storage = input;
      if (auto read = input.getDefiningOp<riscv::ConvertLayoutOp>();
          read && riscv_internal::isCompleteLocalRead(read) && !riscv::readCrossesWrite(read, graph.root))
        storage = read.getInput();
      if (cast<riscv::ValueType>(storage.getType()).getLayout().getCarrier() != "local") {
        auto frozen = freezeBoundary(input, graph.root);
        if (failed(frozen)) return failure();
        storage = *frozen;
      }
      auto memory = riscv_internal::localView(b, storage, graph.root);
      if (failed(memory)) return failure();
      if (llvm::any_of(graph.stores, [&](riscv::StoreOp destination) {
            return riscv::mayAliasStorageAt(storage, destination.getRegion(), graph.root);
          })) {
        auto frozen = snapshot(graph, graph.root, *memory, type, upper);
        if (failed(frozen)) return graph.root->emitError("overlapping local input has no complete snapshot");
        *memory = frozen->memory;
      }
      supplies[input] = *memory;
    }
    SmallVector<Value> capturedReads;
    for (auto read : graph.reads) {
      bool originalPoint = !graph.canStreamRead(read) ||
          llvm::any_of(graph.stores, [&](riscv::StoreOp store) {
            return !riscv::densePointwiseIndependentAt(read.getRegion(), store.getRegion(), graph.root);
          });
      bool shared = llvm::any_of(read.getResult().getUsers(), [&](Operation *user) {
            return !graph.ownsConsumer(user) && !graph.members.contains(user);
          });
      bool laterConsumers = llvm::all_of(read.getResult().getUsers(), [&](Operation *user) {
        if (graph.ownsConsumer(user) || graph.members.contains(user)) return true;
        auto next = graph.root->getBlock()->findAncestorOpInBlock(*user);
        return next && next != graph.root && graph.root->isBeforeInBlock(next);
      });
      // A read-only first consumer can fill the shared snapshot while using
      // each newly read piece. This removes a complete local reload traversal.
      // A store consumer that may overlap the input still needs the complete
      // original read before its first write, irrespective of common shapes.
      if (shared && !originalPoint && laterConsumers && graph.stores.empty()) {
        capturedReads.push_back(read.getResult());
        supplies[read.getResult()] = read.getRegion();
        continue;
      }
      bool preserve = originalPoint || shared;
      Value memory = read.getRegion();
      if (preserve) {
        auto copied = snapshot(graph, read, read.getRegion(),
            cast<riscv::ValueType>(read.getResult().getType()), upper);
        if (failed(copied)) return read.emitError("bounded producer snapshot has no closed storage");
        memory = copied->memory;
        frozen[read.getResult()] = copied->binding;
      }
      supplies[read.getResult()] = memory;
    }
    b.setInsertionPoint(graph.root);
    SmallVector<Value> saved(graph.exports);
    llvm::append_range(saved, capturedReads);
    for (Value value : saved) {
      if (registerOutputs.count(value)) continue;
      auto storage = allocate(cast<riscv::ValueType>(value.getType()), graph.root, "physical-share");
      if (failed(storage)) return graph.root->emitError("shared producer has no bounded private representation");
      frozen[value] = storage->binding;
      outputs[value] = storage->memory;
    }
    for (auto read : graph.localReads) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(read);
      auto memory = riscv_internal::localView(b, read.getInput(), read);
      if (failed(memory)) return read.emitError("bounded producer has no closed private source");
      // A local read is also a numerical snapshot. A later store through an
      // alias can change an element that a future issue has not read yet,
      // including when the two windows have the same shape but different
      // offsets. Complete that read before issuing the first destination write.
      if (llvm::any_of(graph.stores, [&](riscv::StoreOp store) {
            return riscv::mayAliasStorageAt(read.getInput(), store.getRegion(), graph.root);
          })) {
        auto copied = snapshot(graph, read, *memory,
            cast<riscv::ValueType>(read.getResult().getType()), upper);
        if (failed(copied)) return read.emitError("aliased private producer has no complete snapshot");
        frozen[read.getResult()] = copied->binding;
        *memory = copied->memory;
      }
      supplies[read.getResult()] = *memory;
    }
    SmallVector<Snapshot> reductionStorage;
    if (!graph.reductions.empty() && isa<riscv::ValueType>(graph.reductions.front().getResult().getType())) {
      for (auto reduction : graph.reductions) {
        auto storage = allocate(cast<riscv::ValueType>(reduction.getResult().getType()), reduction, "physical-share");
        if (failed(storage)) return failure();
        reductionStorage.push_back(*storage);
      }
    }
    SmallVector<Value> results;
    auto issue = [&]() -> LogicalResult {
      results.clear();
      auto target = graph.root->getParentOfType<riscv::KernelOp>().getTarget();
      Value extent = upper;
      if (!graph.point) {
        extent = riscv_internal::completeDenseExtent(b, loc, upper, graph.width);
      }
      SmallVector<Value> initial;
      for (auto reduction : graph.reductions) {
        auto accumulator = riscv::streamAccumulatorType(piece(graph.domain, graph.width, target, true));
        initial.push_back(riscv_internal::denseReductionSeed(b, loc, accumulator, reduction.getKind()));
      }
      auto traversal = riscv_internal::emitDenseIssues(b, graph.root, extent, graph.width, initial,
          [&](Value offset, Value active, bool full, ValueRange carries) -> FailureOr<SmallVector<Value>> {
        scope(graph, offset, active, graph.width);
        auto computed = expression(graph, supplies, outputs, offset, graph.width, !full);
        if (failed(computed)) return failure();
        if (failed(storeResults(graph, *computed, offset, graph.width))) return failure();
        SmallVector<Value> next;
        unsigned reductionPosition = 0;
        for (auto [position, sink] : llvm::enumerate(graph.sinks)) {
          if (auto reduction = dyn_cast<riscv::ReduceOp>(sink)) {
            Value reduced = reductionInput((*computed)[position], false, sink->getLoc());
            if (!reduced) return failure();
            next.push_back(riscv_internal::denseReductionStep(b, reduction, reduced,
                carries[reductionPosition++], active, reduction.getKind()));
          }
        }
        return next;
      }, 1, bool(graph.point), registerPieces);
      if (failed(traversal)) return graph.root->emitError("bounded producer has no common issue representation");
      for (auto [position, reduction] : llvm::enumerate(graph.reductions))
        results.push_back(riscv_internal::denseReductionResult(b, reduction,
            (*traversal)[position], laneAxis(), reduction.getKind(),
            reducedType(piece(graph.domain, graph.width, target, false))));
      if (!graph.point && !registerPieces) {
        auto remainder = loop(graph.root, extent, upper, 1, results);
        b.setInsertionPointToStart(remainder.getBody());
        auto scalar = expression(graph, supplies, outputs, remainder.getInductionVar(), 1, false);
        if (failed(scalar)) return failure();
        if (failed(storeResults(graph, *scalar, remainder.getInductionVar(), 1))) return failure();
        SmallVector<Value> next;
        unsigned reductionPosition = 0;
        for (auto [position, sink] : llvm::enumerate(graph.sinks)) {
          if (auto reduction = dyn_cast<riscv::ReduceOp>(sink)) {
            Value element = reductionInput((*scalar)[position], true, loc);
            if (!element) return failure();
            Value previous = remainder.getRegionIterArg(reductionPosition++);
            next.push_back(riscv::BinaryOp::create(b, loc, previous.getType(),
                previous, element, reduction.getKind(), riscv_internal::unselectedLeaf(b)));
          }
        }
        if (!next.empty()) scf::YieldOp::create(b, loc, next);
        results.assign(remainder.getResults().begin(), remainder.getResults().end());
        b.setInsertionPointAfter(remainder);
      }
      scope(graph, index(loc, 0), upper, graph.width);
      for (auto [position, storage] : llvm::enumerate(reductionStorage))
        if (failed(storeReduction(results[position], storage.memory, graph.root))) return failure();
      return success();
    };
    if (failed(rows(graph, graph.domain, issue))) return failure();
    DenseMap<Value, Value> sharedBridges;
    b.setInsertionPoint(graph.root);
    for (Value value : saved) {
      auto expected = cast<riscv::ValueType>(value.getType());
      if (auto pieces = registerOutputs.find(value); pieces != registerOutputs.end()) {
        auto assembled = riscv::RVVAssembleValueOp::create(b, loc, expected, pieces->second, laneAxis(),
            riscv_internal::leaf(b, "transfer", "assemble-value", "rvv.assemble-value", "rvv.assemble-value",
                expected.getLayout().getRegisterGroups() * pieces->second.size(),
                expected.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
        riscv_internal::copyOrigin(value.getDefiningOp(), assembled);
        sharedBridges[value] = assembled;
        continue;
      }
      Value storage = frozen.lookup(value);
      auto local = cast<riscv::ValueType>(storage.getType());
      auto bridge = riscv::ConvertLayoutOp::create(b, value.getLoc(), expected, storage,
          riscv_internal::layoutConversion(b, local.getLayout(), expected.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(b));
      riscv_internal::copyOrigin(value.getDefiningOp(), bridge);
      sharedBridges[value] = bridge;
    }
    for (auto [position, reduction] : llvm::enumerate(graph.reductions)) {
      Value result;
      if (reductionStorage.empty()) result = results[position];
      else {
        Value source = reductionStorage[position].binding;
        auto expected = cast<riscv::ValueType>(reduction.getResult().getType());
        auto local = cast<riscv::ValueType>(source.getType());
        result = riscv::ConvertLayoutOp::create(b, loc, expected, source,
            riscv_internal::layoutConversion(b, local.getLayout(), expected.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(b));
        riscv_internal::copyOrigin(reduction, result.getDefiningOp());
      }
      reduction.getResult().replaceAllUsesWith(result);
    }
    for (Operation *sink : graph.sinks)
      if (isa<riscv::StoreOp, riscv::ReduceOp>(sink)) {
        consumed.insert(sink);
        b.eraseOp(sink);
      }
    for (Operation *op : llvm::reverse(graph.order)) {
      if (!op->use_empty()) {
        if (Value bridge = sharedBridges.lookup(op->getResult(0))) {
          op->getResult(0).replaceAllUsesWith(bridge);
        } else if (Value snapshot = frozen.lookup(op->getResult(0))) {
          b.setInsertionPoint(op);
          auto expected = cast<riscv::ValueType>(op->getResult(0).getType());
          auto local = cast<riscv::ValueType>(snapshot.getType());
          auto bridge = riscv::ConvertLayoutOp::create(b, op->getLoc(), expected, snapshot,
              riscv_internal::layoutConversion(b, local.getLayout(), expected.getLayout()),
              riscv::AccessAttr(), riscv_internal::unselectedLeaf(b));
          riscv_internal::copyOrigin(op, bridge);
          op->getResult(0).replaceAllUsesWith(bridge);
        } else if (llvm::is_contained(graph.localReads, dyn_cast<riscv::ConvertLayoutOp>(op))) {
          continue;
        } else return op->emitError("bounded producer retained an unplanned consumer");
      }
      consumed.insert(op);
      b.eraseOp(op);
    }
    return success();
  }
};
}

LogicalResult riscv_internal::materializeDenseValuePrograms(ModuleOp module) {
  return DensePrograms(module).run();
}

LogicalResult riscv_internal::materializeDenseValueProgram(Operation *root) {
  return DensePrograms(root->getParentOfType<ModuleOp>()).required(root);
}
