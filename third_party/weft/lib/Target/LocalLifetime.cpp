#include "LocalLifetime.h"
#include "RISCVPhysicalSupport.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft;

namespace {

// All walks share one bound, including control back edges. An unresolved
// ownership graph is not an exclusive version. The query is rebuilt after IR
// mutation, so successful proofs never outlive the use-def graph they checked.
class LocalLifetime {
public:
  unsigned remaining = 2048;
  Operation *unproven = nullptr;
  llvm::DenseSet<Value> visitingPrivate;
  llvm::DenseSet<std::pair<Value, Value>> visitingDerived;
  llvm::DenseSet<std::pair<Value, Operation *>> visitingUses;
  llvm::DenseSet<std::pair<Value, Operation *>> visitingVersions;

  bool step(Operation *operation) {
    if (!remaining) return reject(operation);
    --remaining;
    return true;
  }
  bool reject(Operation *operation) { unproven = operation; return false; }

  bool privateState(Value value) {
    if (!step(value.getDefiningOp()) || !visitingPrivate.insert(value).second) return false;
    bool result = privateDefinition(value);
    visitingPrivate.erase(value);
    return result;
  }

  bool derivedState(Value value, Value carried) {
    if (value == carried) return true;
    if (!step(value.getDefiningOp()) || !visitingDerived.insert({value, carried}).second) return false;
    bool result = derivedDefinition(value, carried);
    visitingDerived.erase({value, carried});
    return result;
  }

  bool usesBefore(Value value, Operation *write, DominanceInfo &dominance) {
    if (!step(write) || !visitingUses.insert({value, write}).second) return reject(write);
    bool result = inspectUses(value, write, dominance);
    visitingUses.erase({value, write});
    return result;
  }

  bool versionBefore(Value value, Operation *write, DominanceInfo &dominance) {
    if (!step(write) || !visitingVersions.insert({value, write}).second) return reject(write);
    bool result = usesBefore(value, write, dominance) && transferBefore(value, write, dominance);
    visitingVersions.erase({value, write});
    return result;
  }

private:
  bool fresh(Value value) {
    return value.getDefiningOp<riscv::NewOp>() || value.getDefiningOp<riscv::EmptyOp>() ||
        value.getDefiningOp<riscv::MaterializeOp>();
  }

  bool privateDefinition(Value value) {
    if (value.getDefiningOp<riscv::NewOp>() || value.getDefiningOp<riscv::EmptyOp>() ||
        value.getDefiningOp<riscv::LocalBindOp>()) return true;
    if (auto materialize = value.getDefiningOp<riscv::MaterializeOp>())
      return materialize.getPlacement() != "reload";
    if (auto reshape = value.getDefiningOp<riscv::ReshapeOp>()) return privateState(reshape.getInput());
    if (auto update = value.getDefiningOp<riscv::UpdateOp>()) return privateState(update.getInput());
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()))
        return argument.getArgNumber() > 0 && privateState(loop.getInitArgs()[argument.getArgNumber() - 1]);
      if (auto loop = dyn_cast<scf::WhileOp>(argument.getOwner()->getParentOp())) {
        if (argument.getOwner() == loop.getBeforeBody()) return privateState(loop.getInits()[argument.getArgNumber()]);
        return privateState(loop.getConditionOp().getArgs()[argument.getArgNumber()]);
      }
      return false;
    }
    if (auto loop = value.getDefiningOp<scf::ForOp>())
      return privateState(loop.getInitArgs()[cast<OpResult>(value).getResultNumber()]);
    if (auto loop = value.getDefiningOp<scf::WhileOp>())
      return privateState(loop.getConditionOp().getArgs()[cast<OpResult>(value).getResultNumber()]);
    if (auto branch = value.getDefiningOp<scf::IfOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      return privateState(branch.thenYield().getOperand(position)) &&
          privateState(branch.elseYield().getOperand(position));
    }
    return false;
  }

  bool derivedDefinition(Value value, Value carried) {
    if (fresh(value)) return true;
    if (auto binding = value.getDefiningOp<riscv::LocalBindOp>())
      return binding.getStorage().hasOneUse() && bool(binding.getStorage().getDefiningOp<riscv::LocalAllocOp>());
    if (auto update = value.getDefiningOp<riscv::UpdateOp>()) return derivedState(update.getInput(), carried);
    if (auto reshape = value.getDefiningOp<riscv::ReshapeOp>()) return derivedState(reshape.getInput(), carried);
    if (auto loop = value.getDefiningOp<scf::ForOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      return derivedState(loop.getInitArgs()[position], carried) &&
          derivedState(loop.getYieldedValues()[position], loop.getRegionIterArgs()[position]);
    }
    if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      auto condition = loop.getConditionOp();
      auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
      return condition.getArgs().size() == loop.getInits().size() &&
          derivedState(loop.getInits()[position], carried) &&
          derivedState(condition.getArgs()[position], loop.getBeforeBody()->getArgument(position)) &&
          derivedState(yield.getOperand(position), loop.getAfterBody()->getArgument(position));
    }
    if (auto conditional = value.getDefiningOp<scf::IfOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      return llvm::all_of(conditional->getRegions(), [&](Region &region) {
        return !region.empty() && derivedState(cast<scf::YieldOp>(region.front().getTerminator()).getOperand(position), carried);
      });
    }
    return false;
  }

  bool exclusiveBranch(Operation *user, Operation *write) {
    for (Operation *current = write; current; current = current->getParentOp()) {
      auto conditional = dyn_cast_or_null<scf::IfOp>(current->getParentOp());
      if (!conditional || conditional.getElseRegion().empty()) continue;
      Region *other = current->getParentRegion() == &conditional.getThenRegion()
          ? &conditional.getElseRegion() : &conditional.getThenRegion();
      if (other->isAncestor(user->getParentRegion())) return true;
    }
    return false;
  }

  bool inspectUses(Value value, Operation *write, DominanceInfo &dominance) {
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user == write) {
        if (isa<scf::ForOp, scf::WhileOp, scf::YieldOp, scf::ConditionOp>(write) &&
            llvm::count(write->getOperands(), value) > 1) return reject(write);
        continue;
      }
      if (exclusiveBranch(user, write)) continue;
      Operation *before = write->getBlock()->findAncestorOpInBlock(*user);
      bool ordered = before && before != write && before->isBeforeInBlock(write);
      if (!ordered && !dominance.properlyDominates(user, write)) return reject(user);
      // Control transfers have a different version boundary and are checked
      // below. Ordinary address aliases retain the current write boundary.
      if (isa<riscv::LocalViewOp, riscv::SubviewOp, riscv::SliceOp,
              riscv::ReaxisOp, riscv::LinearViewOp, riscv::ReshapeOp>(user)) {
        SmallVector<Value> aliases;
        riscv::appendStorageAliases(use, aliases);
        for (Value alias : aliases)
          if (!usesBefore(alias, write, dominance)) return false;
      }
    }
    return true;
  }

  bool transferBefore(Value value, Operation *write, DominanceInfo &dominance) {
    if (!isa<riscv::ValueType>(value.getType())) return true;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
          loop && argument.getArgNumber() > 0) {
        unsigned position = argument.getArgNumber() - 1;
        if (!derivedState(loop.getYieldedValues()[position], argument)) return reject(loop);
        return versionBefore(loop.getInitArgs()[position], loop, dominance);
      }
      if (auto loop = dyn_cast<scf::WhileOp>(argument.getOwner()->getParentOp())) {
        unsigned position = argument.getArgNumber();
        auto condition = loop.getConditionOp();
        auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
        if (condition.getArgs().size() != loop.getInits().size() ||
            !derivedState(condition.getArgs()[position], loop.getBeforeBody()->getArgument(position)) ||
            !derivedState(yield.getOperand(position), loop.getAfterBody()->getArgument(position))) return reject(loop);
        return versionBefore(loop.getInits()[position], loop, dominance);
      }
      return reject(argument.getOwner()->getParentOp());
    }
    if (auto reshape = value.getDefiningOp<riscv::ReshapeOp>())
      return versionBefore(reshape.getInput(), reshape, dominance);
    if (auto update = value.getDefiningOp<riscv::UpdateOp>())
      return versionBefore(update.getInput(), update, dominance);
    if (auto loop = value.getDefiningOp<scf::ForOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      if (!derivedState(loop.getYieldedValues()[position], loop.getRegionIterArgs()[position])) return reject(loop);
      return versionBefore(loop.getInitArgs()[position], loop, dominance);
    }
    if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      auto condition = loop.getConditionOp();
      auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
      if (condition.getArgs().size() != loop.getInits().size() ||
          !derivedState(condition.getArgs()[position], loop.getBeforeBody()->getArgument(position)) ||
          !derivedState(yield.getOperand(position), loop.getAfterBody()->getArgument(position))) return reject(loop);
      return versionBefore(loop.getInits()[position], loop, dominance);
    }
    if (auto branch = value.getDefiningOp<scf::IfOp>()) {
      unsigned position = cast<OpResult>(value).getResultNumber();
      for (Region &region : branch->getRegions()) {
        auto yield = cast<scf::YieldOp>(region.front().getTerminator());
        if (!versionBefore(yield.getOperand(position), yield, dominance)) return false;
      }
      return true;
    }
    return derivedState(value, Value()) || reject(value.getDefiningOp());
  }
};
}

bool riscv_internal::isPrivateLocalState(Value value) {
  return LocalLifetime().privateState(value);
}

bool riscv_internal::hasExclusiveLocalVersion(Value value, Operation *boundary,
                                               DominanceInfo &dominance,
                                               Operation *&unproven) {
  LocalLifetime lifetime;
  bool result = lifetime.versionBefore(value, boundary, dominance);
  unproven = lifetime.unproven;
  return result;
}
