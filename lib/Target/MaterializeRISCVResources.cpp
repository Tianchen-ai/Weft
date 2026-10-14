#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/RISCVCompilation.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/Predicate.h"

#include "RISCVPhysicalSupport.h"
#include "LocalMemory.h"
#include "ValueLiveness.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <iterator>
#include <limits>
#include <functional>

using namespace weft;

namespace {

int64_t groups(mlir::Value value) {
  return riscv_internal::residentRegisterGroups(value).value();
}

bool representableResidency(riscv::KernelOp kernel) {
  int64_t bound = 0;
  auto add = [&](std::optional<int64_t> groups) {
    if (!groups || *groups < 0 ||
        bound > std::numeric_limits<int64_t>::max() - *groups)
      return false;
    bound += *groups;
    return true;
  };
  return !kernel.walk([&](mlir::Operation *op) {
    for (mlir::Value value : op->getResults())
      if (!add(riscv_internal::residentRegisterGroups(value)))
        return mlir::WalkResult::interrupt();
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (mlir::Value argument : block.getArguments())
          if (!add(riscv_internal::residentRegisterGroups(argument)))
            return mlir::WalkResult::interrupt();
    if (auto leaf = op->getAttrOfType<riscv::LeafAttr>("leaf"))
      if (!add(leaf.getTemporaryGroups()) || !add(leaf.getFragmentGroups()))
        return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  }).wasInterrupted();
}

using Interval = riscv_internal::RegisterInterval;
using Peak = riscv_internal::RegisterPeak;
using riscv_internal::ancestorInBlock;

llvm::SmallVector<Interval> intervalsFor(mlir::Block &block) {
  // representableResidency checks the complete kernel before resource rewrites.
  auto liveness = riscv_internal::BlockRegisterLiveness::get(block).value();
  return llvm::to_vector(liveness.intervals());
}

Peak analyzeBlock(mlir::Block &block) {
  return riscv_internal::registerPeak(block).value();
}

mlir::LogicalResult closeLeafResources(riscv::KernelOp kernel, mlir::IRRewriter &rewriter) {
  auto walk = kernel.walk([&](mlir::Operation *operation) {
    auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
    if (!leaf)
      return mlir::WalkResult::advance();
    int64_t operandGroups = 0;
    int64_t resultGroups = 0;
    for (mlir::Value operand : operation->getOperands())
      operandGroups += riscv_internal::issueRegisterGroups(operand);
    for (mlir::Value result : operation->getResults())
      resultGroups += riscv_internal::issueRegisterGroups(result);
    std::optional<int64_t> temporaryGroups = leaf.getTemporaryGroups();
    if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation);
        binary && leaf.getInstruction().ends_with(".broadcast"))
      temporaryGroups = riscv::binaryBroadcastTemporaryGroups(binary);
    if (auto broadcast = mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(operation))
      temporaryGroups = riscv::axisBroadcastTemporaryGroups(broadcast);
    if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation);
        binary && leaf.getInstruction() == "rvv.mul.row-scale")
      temporaryGroups = riscv::rvvRowScaleTemporaryGroups(binary);
    if (!temporaryGroups) {
      operation->emitError("selected local leaf has no complete bounded resource relation");
      return mlir::WalkResult::interrupt();
    }
    operation->setAttr(
        "leaf", riscv_internal::leaf(
                    rewriter, leaf.getEngine(), leaf.getFamily(),
                    leaf.getInstruction(), leaf.getSpelling(), operandGroups,
                    resultGroups, *temporaryGroups,
                    leaf.getFragmentGroups(), leaf.getMask(), leaf.getTail(),
                    leaf.getParameters(), leaf.getLocalBytes()));
    return mlir::WalkResult::advance();
  });
  return mlir::failure(walk.wasInterrupted());
}

mlir::Operation *lastLocalUse(mlir::Value storage,
                              riscv::LocalReuseOp transfer = {}) {
  auto *last = storage.getDefiningOp();
  if (!last || !mlir::isa<riscv::LocalAllocOp, riscv::LocalReuseOp>(last))
    return nullptr;
  auto *block = last->getBlock();
  llvm::SmallVector<mlir::Value> pending{storage};
  llvm::DenseSet<mlir::Value> visited;
  for (unsigned i = 0; i < pending.size(); ++i) {
    mlir::Value value = pending[i];
    if (!visited.insert(value).second) continue;
    for (mlir::OpOperand &use : value.getUses()) {
      if (use.getOwner() == transfer.getOperation()) continue;
      if (riscv::isStorageMetadataUse(use)) continue;
      // An object may transfer ownership once. A second transfer cannot reuse
      // the old identity even when the first new occupant has become dead.
      if (mlir::isa<riscv::LocalReuseOp>(use.getOwner())) return nullptr;
      auto *at = ancestorInBlock(use.getOwner(), block);
      if (!at) return nullptr;
      if (last->isBeforeInBlock(at)) last = at;
      riscv::appendStorageAliases(use, pending);
    }
  }
  return last;
}

void reuseLocalSlots(riscv::KernelOp kernel, mlir::IRRewriter &rewriter) {
  mlir::DominanceInfo dominance(kernel);
  llvm::SmallVector<riscv::LocalAllocOp> allocations;
  kernel.walk([&](riscv::LocalAllocOp allocation) {
    auto type = allocation.getResult().getType();
    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type.getElementType()) &&
        llvm::all_of(allocation.getResult().getUsers(), [](mlir::Operation *user) {
          return mlir::isa<riscv::LocalBindOp>(user);
        }))
      allocations.push_back(allocation);
  });
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Value>> slots;
  unsigned reused = 0;
  for (riscv::LocalAllocOp allocation : allocations) {
    auto type = allocation.getResult().getType();
    auto &available = slots[allocation->getBlock()];
    bool selected = false;
    for (bool grow : {false, true}) {
      if (selected) break;
      for (mlir::Value &previous : available) {
        auto previousType = mlir::cast<riscv::LocalType>(previous.getType());
        auto previousAllocation = previous.getDefiningOp<riscv::LocalAllocOp>();
        auto previousReuse = previous.getDefiningOp<riscv::LocalReuseOp>();
        mlir::Value previousSize = previousAllocation
            ? previousAllocation.getSizeBytes() : previousReuse.getSizeBytes();
        if ((previousSize != allocation.getSizeBytes()) != grow) continue;
        if (previousType.getSizeBytes() != type.getSizeBytes() ||
            previousType.getElementType() != type.getElementType() ||
            previousType.getAlignment() < type.getAlignment() ||
            previousType.getOwnerDomainId() != type.getOwnerDomainId() ||
            previousType.getLifetimeEndDomainId() != type.getLifetimeEndDomainId())
          continue;
        auto *last = lastLocalUse(previous);
        if (!last || !last->isBeforeInBlock(allocation)) continue;
        mlir::Value storage = previous;
        while (auto reuse = storage.getDefiningOp<riscv::LocalReuseOp>())
          storage = reuse.getPrevious();
        auto slot = storage.getDefiningOp<riscv::LocalAllocOp>();
        if (previousSize != allocation.getSizeBytes()) {
          if (!slot || type.getSizeBytes() >= 0 ||
              !dominance.properlyDominates(allocation.getSizeBytes(), slot))
            continue;
          bool crossesStorageScope = false;
          for (auto *at = slot->getNextNode(); at != allocation; at = at->getNextNode()) {
            if (!at->getNumRegions()) continue;
            at->walk([&](mlir::Operation *nested) {
              auto leaf = nested->getAttrOfType<riscv::LeafAttr>("leaf");
              crossesStorageScope |= mlir::isa<riscv::LocalAllocOp>(nested) ||
                  (leaf && leaf.getLocalBytes() > 0);
            });
          }
          // Growing a parent slot across a child storage scope can raise that
          // child's peak even when the parent's sequential allocations shrink.
          if (crossesStorageScope) continue;
          if (slot->getIterator() != slot->getBlock()->begin())
            if (auto guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(
                    &*std::prev(slot->getIterator())))
              rewriter.eraseOp(guard);
          rewriter.setInsertionPoint(slot);
          mlir::Value capacity = rewriter.create<mlir::arith::MaxUIOp>(
              slot.getLoc(), slot.getSizeBytes(), allocation.getSizeBytes());
          slot->setOperand(0, capacity);
        }
        if (allocation->getIterator() != allocation->getBlock()->begin())
          if (auto guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(
                  &*std::prev(allocation->getIterator())))
            rewriter.eraseOp(guard);
        rewriter.setInsertionPoint(allocation);
        auto reuse = rewriter.create<riscv::LocalReuseOp>(
            allocation.getLoc(), type, previous, allocation.getSizeBytes());
        reuse->setAttrs(allocation->getAttrs());
        rewriter.replaceOp(allocation, reuse.getResult());
        previous = reuse.getResult();
        selected = true;
        ++reused;
        break;
      }
    }
    if (!selected) available.push_back(allocation.getResult());
  }
  if (reused)
    llvm::errs() << "weft-local-reuse: objects=" << allocations.size()
                 << " reused=" << reused << '\n';
}

mlir::LogicalResult verifyLocalLifetimes(riscv::KernelOp kernel) {
  if (kernel.walk([&](riscv::LocalReuseOp reuse) {
        auto *last = lastLocalUse(reuse.getPrevious(), reuse);
        if (!last || !last->isBeforeInBlock(reuse)) {
          reuse.emitError("private slot is reused before the previous object and all of its aliases are dead");
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }).wasInterrupted())
    return mlir::failure();
  auto walk = kernel.walk([&](riscv::LocalAllocOp allocation) {
    if (!riscv::storageAliasesRemainInScope(
            allocation.getResult(), allocation->getBlock())) {
      allocation.emitError("private allocation escapes its lexical storage lifetime");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return mlir::failure(walk.wasInterrupted());
}

mlir::LogicalResult closeDynamicLocalGuards(riscv::KernelOp kernel,
                                            mlir::IRRewriter &rewriter) {
  bool dynamic = false;
  kernel.walk([&](riscv::LocalAllocOp allocation) {
    dynamic |= allocation.getResult().getType().getSizeBytes() < 0;
  });
  if (!dynamic) return mlir::success();
  auto footprint = riscv_internal::staticLocalFootprint(kernel);
  if (!footprint) return kernel.emitError("static local storage footprint is not representable");
  int64_t staticBytes = footprint->bytes;
  const int64_t budget = kernel.getTarget().getMaxPrivateStackBytes();
  if (staticBytes >= budget) {
    rejectRISCVCandidate(kernel, RISCVFailureCategory::ResourceLimit,
        "static local storage leaves no target capacity for dynamic objects: ")
        << staticBytes << " bytes live, budget " << budget;
    return mlir::failure();
  }
  const int64_t limit = budget - staticBytes;
  rewriter.setInsertionPointToStart(&kernel.getBody().front());
  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(kernel.getLoc(), 0);
  std::function<mlir::Value(mlir::Block &, mlir::Value)> closeBlock;
  closeBlock = [&](mlir::Block &block, mlir::Value live) -> mlir::Value {
    for (mlir::Operation &operation : llvm::make_early_inc_range(block)) {
      if (auto allocation = mlir::dyn_cast<riscv::LocalAllocOp>(operation);
          allocation && allocation.getResult().getType().getSizeBytes() < 0) {
        riscv::LocalCapacityGuardOp guard;
        if (allocation->getIterator() != allocation->getBlock()->begin())
          guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(
              &*std::prev(allocation->getIterator()));
        auto exactLeaf = riscv_internal::leaf(
            rewriter, "scalar", "local-capacity",
            "scalar.local-capacity-guard", "scalar.local-capacity-guard", 0, 0);
        if (guard) rewriter.eraseOp(guard);
        rewriter.setInsertionPoint(allocation);
        mlir::Value padding = rewriter.create<mlir::arith::ConstantIndexOp>(allocation.getLoc(),
            allocation.getResult().getType().getAlignment() - 1);
        mlir::Value before = rewriter.create<mlir::arith::AddIOp>(allocation.getLoc(), live, padding);
        rewriter.create<riscv::LocalCapacityGuardOp>(
            allocation.getLoc(), allocation.getSizeBytes(), before, limit,
            allocation.getResult().getType().getBirthId(), exactLeaf);
        rewriter.setInsertionPointAfter(allocation);
        live = rewriter.create<mlir::arith::AddIOp>(allocation.getLoc(), before, allocation.getSizeBytes());
        continue;
      }
      if (auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(operation)) {
        if (!loop->hasAttr("weft.riscv.capacity_state")) {
          bool hasDynamicAllocation = false;
          loop.walk([&](riscv::LocalAllocOp allocation) {
            hasDynamicAllocation |=
                allocation.getResult().getType().getSizeBytes() < 0;
          });
          if (!hasDynamicAllocation) continue;
          rewriter.setInsertionPoint(loop);
          llvm::SmallVector<mlir::Type> types(loop.getResultTypes());
          llvm::SmallVector<mlir::Value> initial(loop.getInits());
          types.push_back(rewriter.getIndexType()); initial.push_back(live);
          auto expanded = rewriter.create<mlir::scf::WhileOp>(loop.getLoc(), types, initial);
          expanded->setAttrs(loop->getAttrs());
          expanded->setAttr("weft.riscv.capacity_state", rewriter.getUnitAttr());
          expanded.getBefore().takeBody(loop.getBefore()); expanded.getAfter().takeBody(loop.getAfter());
          expanded.getBeforeBody()->addArgument(rewriter.getIndexType(), loop.getLoc());
          expanded.getAfterBody()->addArgument(rewriter.getIndexType(), loop.getLoc());
          auto condition = mlir::cast<mlir::scf::ConditionOp>(expanded.getBeforeBody()->getTerminator());
          condition.getArgsMutable().append(live);
          auto yield = mlir::cast<mlir::scf::YieldOp>(expanded.getAfterBody()->getTerminator());
          yield.getResultsMutable().append(live);
          rewriter.replaceOp(loop, expanded.getResults().drop_back());
          loop = expanded;
        }
        loop.getInitsMutable()[loop.getInits().size() - 1].set(live);
        mlir::Value before = closeBlock(*loop.getBeforeBody(), live);
        auto condition = mlir::cast<mlir::scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
        condition.getArgsMutable()[condition.getArgs().size() - 1].set(before);
        closeBlock(*loop.getAfterBody(), loop.getAfterBody()->getArguments().back());
        auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getAfterBody()->getTerminator());
        yield.getResultsMutable()[yield.getNumOperands() - 1].set(live);
        continue;
      }
      for (mlir::Region &region : operation.getRegions())
        for (mlir::Block &nested : region) closeBlock(nested, live);
    }
    return live;
  };
  closeBlock(kernel.getBody().front(), zero);
  return mlir::success();
}

struct LocalFootprint {
  llvm::SmallVector<mlir::Value> sizes;
  int64_t padding = 0;
};

bool sameFootprint(mlir::Value value, const LocalFootprint &footprint) {
  struct Sum {
    llvm::DenseMap<mlir::Value, unsigned> terms;
    llvm::APInt constant{64, 0};
  };
  std::function<void(mlir::Value, Sum &)> collect = [&](mlir::Value value, Sum &sum) {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      sum.constant += llvm::APInt(64, constant.value()); return;
    }
    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
      collect(add.getLhs(), sum); collect(add.getRhs(), sum); return;
    }
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(argument.getOwner()->getParentOp());
      if (loop && loop->hasAttr("weft.riscv.capacity_state") && argument.getOwner() == loop.getAfterBody() &&
          argument.getArgNumber() + 1 == loop.getAfterBody()->getNumArguments()) {
        collect(mlir::cast<mlir::scf::ConditionOp>(loop.getBeforeBody()->getTerminator()).getArgs().back(), sum); return;
      }
    }
    ++sum.terms[value];
  };
  Sum actual, expected;
  collect(value, actual);
  expected.constant = llvm::APInt(64, footprint.padding);
  for (mlir::Value size : footprint.sizes) collect(size, expected);
  return actual.constant == expected.constant && actual.terms.size() == expected.terms.size() &&
      llvm::all_of(expected.terms, [&](const auto &entry) { return actual.terms.lookup(entry.first) == entry.second; });
}

mlir::FailureOr<int64_t> localStorageBytes(riscv::KernelOp kernel) {
  if (mlir::failed(verifyLocalLifetimes(kernel))) return mlir::failure();
  auto footprint = riscv_internal::staticLocalFootprint(kernel);
  if (!footprint) return kernel.emitError("static local storage footprint is not representable"), mlir::failure();
  int64_t bytes = footprint->bytes, dynamicLimit = 0;
  mlir::LogicalResult complete = mlir::success();
  std::function<LocalFootprint(mlir::Block &, LocalFootprint)> verifyBlock;
  verifyBlock = [&](mlir::Block &block, LocalFootprint live) -> LocalFootprint {
    for (mlir::Operation &operation : block) {
      if (auto allocation = mlir::dyn_cast<riscv::LocalAllocOp>(operation);
          allocation && allocation.getResult().getType().getSizeBytes() < 0) {
        riscv::LocalCapacityGuardOp guard;
        if (allocation->getIterator() != block.begin())
          guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(&*std::prev(allocation->getIterator()));
        live.padding += allocation.getResult().getType().getAlignment() - 1;
        if (!guard || guard.getSizeBytes() != allocation.getSizeBytes() ||
            guard.getBirthId() != allocation.getResult().getType().getBirthId() ||
            !sameFootprint(guard.getLiveBytes(), live)) {
          allocation.emitError("dynamic allocation guard does not account for its live lexical storage");
          complete = mlir::failure();
        } else dynamicLimit = std::max<int64_t>(dynamicLimit, guard.getLimitBytes());
        live.sizes.push_back(allocation.getSizeBytes());
        continue;
      }
      if (auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(operation)) {
        auto before = verifyBlock(*loop.getBeforeBody(), live);
        verifyBlock(*loop.getAfterBody(), before);
      } else {
        for (mlir::Region &region : operation.getRegions())
          for (mlir::Block &nested : region) verifyBlock(nested, live);
      }
    }
    return live;
  };
  verifyBlock(kernel.getBody().front(), {});
  if (mlir::failed(complete)) return mlir::failure();
  if (dynamicLimit > std::numeric_limits<int64_t>::max() - bytes)
    return kernel.emitError("combined local storage footprint is not representable"), mlir::failure();
  return bytes + dynamicLimit;
}

bool canPreserveProducerResult(mlir::Operation *operation) {
  // Spill preserves the read result; it never clones or moves the producer.
  return riscv::hasOnlyReadEffects(operation);
}

bool isIntegerPointwise(mlir::Operation *operation) {
  if (!mlir::isa<riscv::BinaryOp, riscv::CastOp, riscv::WidenOp,
                 riscv::NarrowOp>(operation) ||
      !mlir::isMemoryEffectFree(operation) || operation->getNumResults() != 1)
    return false;
  auto result = mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
  return result && result.getLayout().getCarrier() == "rvv" &&
         result.getLayout().getValidity() == "full" &&
         mlir::isa<mlir::IntegerType>(result.getElementType()) &&
         llvm::all_of(operation->getOperands(), [](mlir::Value operand) {
           return mlir::isa<mlir::IntegerType>(
               riscv_internal::logicalElement(operand.getType()));
         });
}

std::optional<int64_t> residentCost(mlir::Block &block) {
  int64_t cost = 0;
  for (const Interval &interval : intervalsFor(block)) {
    const int64_t length = interval.end - interval.begin;
    if (length && interval.groups >
                      (std::numeric_limits<int64_t>::max() - cost) / length)
      return std::nullopt;
    cost += interval.groups * length;
  }
  return cost;
}

bool sinkIntegerSlice(mlir::Operation *root, mlir::IRRewriter &rewriter,
                      int64_t &savedCost) {
  if (!isIntegerPointwise(root) || !root->getResult(0).hasOneUse())
    return false;
  mlir::Operation *consumer = *root->getResult(0).getUsers().begin();
  mlir::Block *block = root->getBlock();
  if (consumer->getBlock() != block || root->getNextNode() == consumer ||
      !root->isBeforeInBlock(consumer))
    return false;
  llvm::SmallPtrSet<mlir::Operation *, 32> slice;
  llvm::SmallVector<mlir::Value> worklist{root->getResult(0)};
  bool numericRead = false;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto *definition = value.getDefiningOp();
    if (!definition)
      return false;
    if (mlir::isa<riscv::ConstantOp, mlir::arith::ConstantOp>(definition))
      continue;
    // These are numeric SSA snapshots.  Their read sites stay where they were;
    // deferred Field/Extract/descriptor expressions are deliberately excluded.
    if (mlir::isa<riscv::FieldReadOp, riscv::RVVReplicaStorageLoadOp,
                  riscv::RegisterMaterializeOp>(definition)) {
      numericRead = true;
      continue;
    }
    if (definition->getBlock() != block || !value.hasOneUse() ||
        !isIntegerPointwise(definition))
      return false;
    if (slice.contains(definition))
      continue;
    if (slice.size() == 32)
      return false;
    slice.insert(definition);
    llvm::append_range(worklist, definition->getOperands());
  }
  if (!numericRead)
    return false;
  llvm::SmallVector<mlir::Operation *> ordered;
  llvm::SmallVector<mlir::Operation *> oldNext;
  bool crossedPartial = false;
  unsigned distance = 0;
  for (mlir::Operation &operation : *block) {
    if (&operation == consumer)
      break;
    if (slice.contains(&operation)) {
      ordered.push_back(&operation);
      oldNext.push_back(operation.getNextNode());
    }
    if (ordered.empty())
      continue;
    if (++distance > 256 || operation.getNumRegions() ||
        !canPreserveProducerResult(&operation))
      return false;
    crossedPartial |= llvm::any_of(operation.getResultTypes(), [](mlir::Type type) {
      return mlir::isa<riscv::PartialSetType>(type);
    });
  }
  if (!crossedPartial || ordered.size() != slice.size())
    return false;
  auto beforeCost = residentCost(*block);
  if (!beforeCost)
    return false;
  const int64_t beforePeak = analyzeBlock(*block).total();
  for (auto *operation : ordered)
    rewriter.moveOpBefore(operation, consumer);
  auto afterCost = residentCost(*block);
  if (afterCost && *afterCost < *beforeCost &&
      savedCost <= std::numeric_limits<int64_t>::max() -
                       (*beforeCost - *afterCost) &&
      analyzeBlock(*block).total() <= beforePeak) {
    savedCost += *beforeCost - *afterCost;
    return true;
  }
  for (auto [operation, next] : llvm::reverse(llvm::zip(ordered, oldNext)))
    rewriter.moveOpBefore(operation, next);
  return false;
}

bool prepareScalarFloatingSlice(mlir::Operation *root,
                                mlir::IRRewriter &rewriter,
                                unsigned &overlappedRegisterIssues) {
  auto factor = mlir::dyn_cast<riscv::BinaryOp>(root);
  if (!factor || factor.getKind() != "mul" || root->getNumResults() != 1 ||
      !mlir::isa<mlir::FloatType>(root->getResult(0).getType()) ||
      !root->getResult(0).hasOneUse() ||
      mlir::isa<riscv::RegisterMaterializeOp>(
          *root->getResult(0).getUsers().begin()))
    return false;
  auto *block = root->getBlock();
  llvm::SmallPtrSet<mlir::Operation *, 8> slice;
  llvm::SmallVector<mlir::Value> worklist{root->getResult(0)};
  bool hasField = false;
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    auto *definition = value.getDefiningOp();
    if (!definition || !mlir::isa<mlir::FloatType>(value.getType()))
      return false;
    if (mlir::isa<riscv::ConstantOp, mlir::arith::ConstantOp>(definition))
      continue;
    if (definition->getBlock() != block || !value.hasOneUse() ||
        !mlir::isMemoryEffectFree(definition) || definition->getNumRegions())
      return false;
    if (slice.contains(definition))
      continue;
    if (slice.size() == 8)
      return false;
    slice.insert(definition);
    if (mlir::isa<riscv::FieldOp>(definition)) {
      hasField = true;
      continue;
    }
    auto multiply = mlir::dyn_cast<riscv::BinaryOp>(definition);
    if (!mlir::isa<riscv::WidenOp, riscv::CastOp>(definition) &&
        (!multiply || multiply.getKind() != "mul"))
      return false;
    llvm::append_range(worklist, definition->getOperands());
  }
  if (!hasField)
    return false;
  llvm::SmallPtrSet<mlir::Operation *, 16> dependencies;
  for (auto *operation : slice)
    for (auto operand : operation->getOperands())
      if (auto *definition = operand.getDefiningOp();
          definition && definition->getBlock() == block &&
          !slice.contains(definition))
        dependencies.insert(definition);
  mlir::Operation *anchor = nullptr;
  unsigned distance = 0;
  unsigned vectorRegisterIssues = 0;
  bool crossesReduction = false;
  for (auto *cursor = root->getPrevNode(); cursor && distance < 64;
       cursor = cursor->getPrevNode(), ++distance) {
    if (slice.contains(cursor))
      continue;
    if (dependencies.contains(cursor) || cursor->getNumRegions() ||
        !canPreserveProducerResult(cursor))
      break;
    auto leaf = cursor->getAttrOfType<riscv::LeafAttr>("leaf");
    if (leaf && leaf.getEngine() == "rvv") {
      llvm::SmallDenseSet<mlir::Value, 8> scalarInputs;
      bool scalarTuple = false;
      for (mlir::Value operand : cursor->getOperands()) {
        auto type = mlir::dyn_cast<riscv::ValueType>(operand.getType());
        if ((!type || type.getLayout().getCarrier() == "scalar") &&
            mlir::isa<mlir::IntegerType, mlir::IndexType>(
                riscv_internal::logicalElement(operand.getType())) &&
            !operand.getDefiningOp<riscv::ConstantOp>() &&
            !operand.getDefiningOp<mlir::arith::ConstantOp>()) {
          scalarInputs.insert(operand);
          if (type)
            for (auto factors : {type.getLayout().getTimeFactors(),
                                 type.getLayout().getReplicaFactors()})
              scalarTuple |= llvm::any_of(factors.asArrayRef(),
                                          [](int64_t factor) { return factor > 1; });
        }
      }
      // Do not overlap address/conversion preparation with a multi-coefficient
      // GPR consumer. Its scalar inputs have not reached their last uses yet.
      if (scalarTuple || scalarInputs.size() > 1)
        break;
    }
    anchor = cursor;
    if (leaf && leaf.getEngine() == "rvv")
      vectorRegisterIssues += std::max<int64_t>(
          1, std::max(leaf.getOperandGroups(), leaf.getResultGroups()));
    crossesReduction |= mlir::isa<riscv::RVVFinalizeWidenDotOp,
                                  riscv::RVVPartialFinalizeOp,
                                  riscv::RVVWidenReduceOp, riscv::ReduceOp>(cursor);
    // Keep the shortest independent window that can cover this scalar slice
    // in the register-issue estimate; earlier placement only extends lifetime.
    if (crossesReduction && vectorRegisterIssues >= slice.size())
      break;
  }
  if (!anchor || !crossesReduction || vectorRegisterIssues < slice.size())
    return false;
  llvm::SmallVector<mlir::Operation *> ordered;
  for (auto &operation : *block)
    if (slice.contains(&operation))
      ordered.push_back(&operation);
  if (ordered.empty() || ordered.front()->isBeforeInBlock(anchor))
    return false;
  for (auto *operation : ordered)
    rewriter.moveOpBefore(operation, anchor);
  rewriter.setInsertionPointAfter(root);
  auto prepared = rewriter.create<riscv::RegisterMaterializeOp>(
      root->getLoc(), root->getResult(0).getType(), root->getResult(0),
      -1, -1, -1, "physical-share");
  riscv_internal::copyOrigin(root, prepared);
  root->getResult(0).replaceAllUsesExcept(prepared.getResult(), prepared);
  overlappedRegisterIssues += vectorRegisterIssues;
  return true;
}

mlir::LogicalResult rebindFieldReadProjections(
    riscv::KernelOp kernel, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::FieldReadOp> reads;
  kernel.walk([&](riscv::FieldReadOp read) { reads.push_back(read); });
  for (auto read : reads) {
    llvm::SmallVector<mlir::Value> captured;
    auto field = riscv::fieldReadProjection(read.getInput(), &captured);
    if (!field || captured.size() != read.getIndices().size())
      return read.emitError("spill rewriting lost the field address operand structure");
    if (llvm::equal(captured, read.getIndices()))
      continue;
    for (auto [before, after] : llvm::zip(captured, read.getIndices()))
      if (before.getType() != after.getType())
        return read.emitError("spill rewriting changed a field address operand type");

    // The read owns its use-site reloads. Clone only the pure address
    // projections, binding their index slots to this read's actual operands.
    llvm::SmallVector<riscv::ExtractOp> projections;
    mlir::Value value = read.getInput();
    while (auto extract = value.getDefiningOp<riscv::ExtractOp>()) {
      projections.push_back(extract);
      value = extract.getInput();
    }
    rewriter.setInsertionPoint(read);
    value = field.getResult();
    size_t cursor = captured.size();
    for (auto extract : llvm::reverse(projections)) {
      const size_t count = extract.getIndices().size();
      cursor -= count;
      auto copy = mlir::cast<riscv::ExtractOp>(rewriter.clone(*extract));
      copy.getInputMutable().assign(value);
      copy.getIndicesMutable().assign(read.getIndices().slice(cursor, count));
      value = copy.getResult();
    }
    read.getInputMutable().assign(value);
    for (auto extract : projections) {
      if (!extract.getResult().use_empty())
        break;
      llvm::SmallVector<riscv::ReloadOp> oldReloads;
      for (mlir::Value index : extract.getIndices())
        if (auto reload = index.getDefiningOp<riscv::ReloadOp>();
            reload && !llvm::is_contained(oldReloads, reload))
          oldReloads.push_back(reload);
      rewriter.eraseOp(extract);
      for (auto reload : oldReloads)
        if (reload.getResult().use_empty())
          rewriter.eraseOp(reload);
    }
  }
  return mlir::success();
}

bool shortenSharedReload(Peak &peak, mlir::IRRewriter &rewriter) {
  if (!peak.block || peak.point >= peak.block->getOperations().size())
    return false;
  mlir::Operation *at = &*std::next(peak.block->begin(), peak.point);
  riscv::ReloadOp selected;
  int64_t selectedGroups = 0;
  mlir::Block *block = peak.block;
  unsigned point = peak.point;
  while (block) {
    for (const Interval &interval : intervalsFor(*block)) {
      auto reload = interval.value.getDefiningOp<riscv::ReloadOp>();
      if (!reload || reload->getBlock() != block || interval.fragment ||
          interval.begin >= point || interval.end < point ||
          interval.groups <= selectedGroups)
        continue;
      // A private slot with one preceding spill and only reload users is immutable
      // here. Splitting its reads does not rematerialize the numerical producer.
      if (!reload.getSlot().getDefiningOp<riscv::LocalAllocOp>())
        continue;
      riscv::SpillOp writer;
      bool immutable = true;
      for (mlir::Operation *user : reload.getSlot().getUsers()) {
        if (auto spill = mlir::dyn_cast<riscv::SpillOp>(user)) {
          if (writer || spill.getSlot() != reload.getSlot()) {
            immutable = false;
            break;
          }
          writer = spill;
        } else if (!mlir::isa<riscv::ReloadOp>(user)) {
          immutable = false;
          break;
        }
      }
      if (!immutable || !writer || writer->getBlock() != block ||
          !writer->isBeforeInBlock(reload))
        continue;
      bool before = false, after = false, usedAtPeak = false;
      bool withinOwner = true, captured = false;
      mlir::Operation *ownerAt = ancestorInBlock(at, block);
      for (mlir::Operation *user : reload.getResult().getUsers()) {
        auto owner = ancestorInBlock(user, block);
        if (!owner) {
          withinOwner = false;
          break;
        }
        captured |= user->getBlock() != block;
        usedAtPeak |= user == at;
        before |= owner->isBeforeInBlock(ownerAt) ||
                  (!peak.before && owner == ownerAt);
        after |= ownerAt->isBeforeInBlock(owner);
      }
      if (!withinOwner ||
          (!captured && (!before || !after || (peak.before && usedAtPeak))))
        continue;
      selected = reload;
      selectedGroups = interval.groups;
    }
    mlir::Operation *parent = block->getParentOp();
    if (!parent || mlir::isa<riscv::KernelOp>(parent))
      break;
    block = parent->getBlock();
    if (!block)
      break;
    point = std::distance(block->begin(), parent->getIterator());
  }
  if (!selected)
    return false;
  llvm::SmallVector<mlir::Operation *> consumers;
  selected->getBlock()->getParentOp()->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *operation) {
        if (llvm::is_contained(operation->getOperands(), selected.getResult()))
          consumers.push_back(operation);
      });
  rewriter.moveOpBefore(selected, consumers.front());
  for (mlir::Operation *consumer : llvm::drop_begin(consumers)) {
    rewriter.setInsertionPoint(consumer);
    auto copy = rewriter.create<riscv::ReloadOp>(
        consumer->getLoc(), selected.getResult().getType(), selected.getSlot(),
        selected.getLeaf());
    riscv_internal::copyOrigin(selected, copy);
    consumer->replaceUsesOfWith(selected.getResult(), copy.getResult());
  }
  return true;
}

std::optional<Interval> selectVictim(
    const Peak &peak, const llvm::DenseSet<mlir::Value> &originalValues,
    const llvm::DenseSet<mlir::Value> &spilledValues) {
  std::optional<Interval> victim;
  int64_t bestScore = 0;
  mlir::Operation *peakOperation =
      peak.point < peak.block->getOperations().size()
          ? &*std::next(peak.block->begin(), peak.point)
          : nullptr;
  mlir::Block *block = peak.block;
  unsigned point = peak.point;
  bool before = peak.before;
  while (block) {
    for (const Interval &interval : intervalsFor(*block)) {
      if (!originalValues.contains(interval.value) ||
          spilledValues.contains(interval.value) ||
          interval.fragment || interval.end < point ||
          (!before && interval.end == point) ||
          riscv::fieldReadProjection(interval.value) || riscv::isPredicateRegister(interval.value))
        continue;
      mlir::Operation *definition = interval.value.getDefiningOp();
      if (definition && interval.begin >= point)
        continue;
      auto valueType = mlir::dyn_cast<riscv::ValueType>(interval.value.getType());
      if ((definition && definition->getBlock() != block) || !valueType ||
          !mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(
              valueType.getElementType()) ||
          valueType.getLayout().getCarrier() == "local" ||
          (definition &&
           (mlir::isa<riscv::ReloadOp>(definition) ||
            (mlir::isa<riscv::NewOp>(definition) &&
             !mlir::cast<riscv::NewOp>(definition).getInitialized()))))
        continue;

      bool withinOwner = !interval.value.use_empty();
      bool usedAtPeak = false;
      for (mlir::OpOperand &use : interval.value.getUses()) {
        withinOwner &= ancestorInBlock(use.getOwner(), block) != nullptr;
        usedAtPeak |= use.getOwner() == peakOperation;
        // An ancestor loop's init is consumed at its entry and remains a
        // register carry throughout the loop. Spilling that init does not shorten
        // the carry's lifetime at a peak in its body.
        if (block != peak.block && riscv::hasUniqueRegisterUse(interval.value) &&
            use.getOwner()->getBlock() == block &&
            ancestorInBlock(peakOperation, block) == use.getOwner())
          usedAtPeak = true;
      }
      // A use-site reload replaces the original at that exact point and cannot
      // reduce its peak. Only spill a value which is merely resident here.
      if (!withinOwner || (before && usedAtPeak))
        continue;
      const int64_t score =
          interval.groups * static_cast<int64_t>(interval.end - interval.begin);
      if (score > bestScore) {
        victim = interval;
        bestScore = score;
      }
    }
    mlir::Operation *parent = block->getParentOp();
    if (!parent || mlir::isa<riscv::KernelOp>(parent))
      break;
    block = parent->getBlock();
    if (!block)
      break;
    point = std::distance(block->begin(), parent->getIterator());
    before = true;
  }
  return victim;
}

class CloseRISCVLeafResourcesPass
    : public mlir::PassWrapper<CloseRISCVLeafResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-close-leaf-resources";
  }
  llvm::StringRef getDescription() const override {
    return "Close selected leaf operand/result resource fields before physical CSE";
  }
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    for (riscv::KernelOp kernel : getOperation().getOps<riscv::KernelOp>())
      if (mlir::failed(closeLeafResources(kernel, rewriter))) {
        signalPassFailure();
        return;
      }
  }
};

class MaterializeRISCVResourcesPass
    : public mlir::PassWrapper<MaterializeRISCVResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-resources";
  }
  llvm::StringRef getDescription() const override {
    return "Compute region-aware physical SSA liveness and materialize legal spills";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    for (riscv::KernelOp kernel : getOperation().getOps<riscv::KernelOp>()) {
      if (kernel.getResourcesMaterialized())
        continue;
      if (!representableResidency(kernel)) {
        kernel.emitError("physical numeric residency is not representable");
        failed = true;
        continue;
      }
      llvm::DenseMap<mlir::Block *, unsigned> preparedPerBlock;
      llvm::SmallVector<mlir::Operation *> scalarFloating;
      kernel.walk([&](mlir::Operation *operation) {
        if (auto materialize = mlir::dyn_cast<riscv::RegisterMaterializeOp>(operation);
            materialize && materialize.getRealization() == "physical-share" &&
            mlir::isa<mlir::FloatType>(
                riscv_internal::logicalElement(
                    materialize.getResult().getType())))
          ++preparedPerBlock[operation->getBlock()];
        if (operation->getNumResults() == 1 &&
            mlir::isa<mlir::FloatType>(operation->getResult(0).getType()))
          scalarFloating.push_back(operation);
      });
      unsigned preparedSlices = 0, overlap = 0;
      for (auto *operation : llvm::reverse(scalarFloating))
        if (preparedPerBlock.lookup(operation->getBlock()) < 4 &&
            prepareScalarFloatingSlice(operation, rewriter, overlap)) {
          ++preparedPerBlock[operation->getBlock()];
          ++preparedSlices;
        }
      if (preparedSlices)
        llvm::errs() << "weft-scalar-prepare: selected=" << preparedSlices
                     << " estimated-overlapped-register-issues=" << overlap
                     << '\n';
      llvm::SmallVector<mlir::Operation *> pointwise;
      kernel.walk([&](mlir::Operation *operation) {
        if (isIntegerPointwise(operation))
          pointwise.push_back(operation);
      });
      int64_t savedCost = 0;
      unsigned sunkSlices = 0;
      for (auto *operation : llvm::reverse(pointwise))
        if (sinkIntegerSlice(operation, rewriter, savedCost))
          ++sunkSlices;
      if (sunkSlices)
        llvm::errs() << "weft-resource-sink: candidates=" << pointwise.size()
                     << " selected=" << sunkSlices
                     << " estimated-register-residence-reduction=" << savedCost
                     << '\n';
      int64_t spillIdentity = 1;
      kernel.walk([&](riscv::LocalAllocOp allocation) {
        auto type = allocation.getResult().getType();
        spillIdentity = std::max({spillIdentity, type.getAliasSet() + 1,
                                  type.getBirthId() + 1});
      });
      llvm::DenseSet<mlir::Value> originalValues;
      kernel.walk([&](mlir::Operation *operation) {
        for (mlir::Value result : operation->getResults())
          if (groups(result) > 0)
            originalValues.insert(result);
        for (mlir::Region &region : operation->getRegions())
          for (mlir::Block &block : region)
            for (mlir::BlockArgument argument : block.getArguments())
              if (groups(argument) > 0)
                originalValues.insert(argument);
      });

      llvm::DenseSet<mlir::Value> spilledValues;
      bool closed = false;
      while (true) {
        riscv_internal::selectFragmentRegisterReuse(kernel);
        if (mlir::failed(closeLeafResources(kernel, rewriter))) {
          failed = true;
          break;
        }
        reuseLocalSlots(kernel, rewriter);
        Peak peak = analyzeBlock(kernel.getBody().front());
        if (mlir::failed(closeDynamicLocalGuards(kernel, rewriter))) {
          failed = true;
          break;
        }
        auto localBytes = localStorageBytes(kernel);
        if (mlir::failed(localBytes)) {
          failed = true;
          break;
        }
        kernel.setVectorRegisterPeak(peak.vector);
        kernel.setFragmentRegisterPeak(peak.fragment);
        kernel.setLocalStorageBytes(*localBytes);
        if (*localBytes > kernel.getTarget().getMaxPrivateStackBytes()) {
          rejectRISCVCandidate(kernel, RISCVFailureCategory::ResourceLimit)
              << "physical local storage requires " << *localBytes
              << " bytes, exceeding target budget "
              << kernel.getTarget().getMaxPrivateStackBytes();
          failed = true;
          break;
        }
        if (peak.total() <= kernel.getTarget().getVectorRegisters()) {
          closed = true;
          break;
        }
        if (shortenSharedReload(peak, rewriter)) {
          if (mlir::failed(rebindFieldReadProjections(kernel, rewriter))) {
            failed = true;
            break;
          }
          // Each immutable-slot read now has one immediate actual consumer.
          // It cannot be split again; no numerical producer is introduced.
          continue;
        }
        if (!peak.block) {
          kernel.emitError("resource peak has no physical block identity");
          failed = true;
          break;
        }
        llvm::SmallVector<Interval> intervals = intervalsFor(*peak.block);
        // Inserting reload sites can expose a different peak and increases the
        // number of issue points. Progress is instead bounded by consuming each
        // original SSA value once, including loop-body arguments. Closure still
        // requires the actual peak to satisfy the target register budget.
        auto victim = selectVictim(peak, originalValues, spilledValues);
        if (!victim) {
          mlir::Operation *at =
              peak.point < peak.block->getOperations().size()
                  ? &*std::next(peak.block->begin(), peak.point)
                  : nullptr;
          auto diagnostic = rejectRISCVCandidate(kernel, RISCVFailureCategory::ResourceLimit)
              << "physical program requires " << peak.vector << " vector and "
              << peak.fragment << " fragment groups "
              << (peak.before ? "before " : "after ")
              << (at ? at->getName().getStringRef() : llvm::StringRef("<block-end>"))
              << ", exceeding target budget "
              << kernel.getTarget().getVectorRegisters()
              << "; no resident value in the region ancestry has a profitable spill";
          for (const Interval &interval : intervals)
            if (interval.begin <= peak.point && peak.point <= interval.end)
              diagnostic << " [" << interval.value << ": " << interval.groups
                         << " groups, interval " << interval.begin << ".."
                         << interval.end << "]";
          for (mlir::Operation *parent = peak.block->getParentOp();
               parent && !mlir::isa<riscv::KernelOp>(parent);
               parent = parent->getParentOp()) {
            auto *owner = parent->getBlock();
            if (!owner) break;
            auto point = std::distance(owner->begin(), parent->getIterator());
            for (const Interval &interval : intervalsFor(*owner))
              if (interval.begin < point && point <= interval.end)
                diagnostic << " [ancestor " << interval.value << ": "
                           << interval.groups << " groups]";
          }
          failed = true;
          break;
        }

        mlir::Value value = victim->value;
        spilledValues.insert(value);
        // Preserve the completed numeric SSA result at its definition, including
        // results of effectful structured control. The producer is never moved,
        // cloned, or replayed; its effects do not make this snapshot a new read.
        auto valueType = mlir::cast<riscv::ValueType>(value.getType());
        mlir::Operation *definition = value.getDefiningOp();
        mlir::Location location =
            definition ? definition->getLoc()
                       : mlir::cast<mlir::BlockArgument>(value).getLoc();
        const int64_t vlenBytes = kernel.getTarget().getVlenBits() / 8;
        if (vlenBytes <= 0 || victim->groups >
                std::numeric_limits<int64_t>::max() / vlenBytes) {
          kernel.emitError("physical spill storage size is not representable");
          failed = true;
          break;
        }
        const int64_t sizeBytes = victim->groups * vlenBytes;
        const int64_t alignment =
            std::max<int64_t>(16, kernel.getTarget().getVlenBits() / 8);
        int64_t ownerDomain = 0;
        for (mlir::Operation *parent =
                 definition ? definition->getParentOp()
                            : mlir::cast<mlir::BlockArgument>(value).getOwner()->getParentOp();
             parent;
             parent = parent->getParentOp())
          if (auto level =
                  parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level")) {
            ownerDomain = level.getDomainId();
            break;
          }
        auto slotType = riscv::LocalType::get(
            rewriter.getContext(), valueType.getElementType(), valueType.getShape(),
            valueType.getAxisIds(), sizeBytes, alignment, spillIdentity, "spill",
            ownerDomain, spillIdentity, ownerDomain, "rvv-register-spill");
        if (definition)
          rewriter.setInsertionPoint(definition);
        else
          rewriter.setInsertionPointToStart(
              mlir::cast<mlir::BlockArgument>(value).getOwner());
        mlir::Value allocationBytes =
            rewriter.create<mlir::arith::ConstantIndexOp>(location, sizeBytes);
        auto slot = rewriter.create<riscv::LocalAllocOp>(
            location, slotType, allocationBytes);
        if (definition)
          rewriter.setInsertionPointAfter(definition);
        else
          rewriter.setInsertionPointAfter(slot);
        auto spillLeaf = riscv_internal::leaf(
            rewriter, "transfer", "spill", "rvv.spill", "rvv.spill",
            victim->groups, 0, 0, 0, "none", "exact");
        auto spill = rewriter.create<riscv::SpillOp>(
            location, value, slot.getResult(), spillLeaf);
        llvm::SmallVector<mlir::OpOperand *> uses;
        for (mlir::OpOperand &use : value.getUses())
          if (use.getOwner() != spill.getOperation())
            uses.push_back(&use);
        auto reloadLeaf = riscv_internal::leaf(
            rewriter, "transfer", "reload", "rvv.reload", "rvv.reload", 0,
            victim->groups, 0, 0, "none", "exact");
        llvm::DenseMap<mlir::Operation *, mlir::Value> reloads;
        for (mlir::OpOperand *use : uses) {
          auto found = reloads.find(use->getOwner());
          if (found != reloads.end()) {
            use->set(found->second);
            continue;
          }
          rewriter.setInsertionPoint(use->getOwner());
          auto reload = rewriter.create<riscv::ReloadOp>(
              use->getOwner()->getLoc(), valueType, slot.getResult(), reloadLeaf);
          use->set(reload.getResult());
          reloads[use->getOwner()] = reload.getResult();
        }
        if (mlir::failed(rebindFieldReadProjections(kernel, rewriter))) {
          failed = true;
          break;
        }
        ++spillIdentity;
      }
      if (!closed && !failed) {
        kernel.emitError(
            "resource materialization exhausted every physical SSA spill candidate");
        failed = true;
      } else if (closed) {
        kernel.setResourcesMaterialized(true);
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

mlir::FailureOr<int64_t>
weft::riscv_internal::guardedLocalStorageBytes(riscv::KernelOp kernel) {
  return localStorageBytes(kernel);
}

std::optional<weft::riscv_internal::RegisterPressure>
weft::riscv_internal::physicalRegisterPressure(riscv::KernelOp kernel) {
  if (!representableResidency(kernel))
    return std::nullopt;
  auto peak = analyzeBlock(kernel.getBody().front());
  auto *operation = peak.block && peak.point < peak.block->getOperations().size()
      ? &*std::next(peak.block->begin(), peak.point) : nullptr;
  return RegisterPressure{peak.vector, peak.fragment, operation, peak.before};
}

std::unique_ptr<mlir::Pass> weft::createMaterializeRISCVResourcesPass() {
  return std::make_unique<MaterializeRISCVResourcesPass>();
}

std::unique_ptr<mlir::Pass> weft::createCloseRISCVLeafResourcesPass() {
  return std::make_unique<CloseRISCVLeafResourcesPass>();
}
