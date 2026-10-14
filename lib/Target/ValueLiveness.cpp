#include "ValueLiveness.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

Operation *riscv_internal::ancestorInBlock(Operation *operation, Block *block) {
  while (operation && operation->getBlock() != block) operation = operation->getParentOp();
  return operation;
}

namespace {
bool singleUseInBlock(Value value, Block *block, Operation *expected) {
  SmallVector<Value> pending{registerStorageIdentity(value)};
  llvm::DenseSet<Value> visited;
  unsigned uses = 0;
  while (!pending.empty()) {
    Value alias = pending.pop_back_val();
    if (!visited.insert(alias).second) continue;
    if (visited.size() > 128) return false;
    for (OpOperand &use : alias.getUses()) {
      bool representation = false;
      for (Value output : use.getOwner()->getResults())
        if (riscv::registerAliasSource(output) == alias) {
          pending.push_back(output);
          representation = true;
        }
      if (representation) continue;
      if (!ancestorInBlock(use.getOwner(), block)) continue;
      if (use.getOwner() != expected || ++uses != 1) return false;
    }
  }
  return uses == 1;
}

bool transfersIntoRegion(Operation *operation, Value value, bool captured = false) {
  if (!riscv::hasUniqueRegisterUse(value) &&
      !(captured && singleUseInBlock(value, operation->getBlock(), operation))) return false;
  if (auto loop = dyn_cast<scf::ForOp>(operation))
    return llvm::is_contained(loop.getInitArgs(), value);
  if (auto loop = dyn_cast<scf::WhileOp>(operation))
    return llvm::is_contained(loop.getInits(), value);
  return false;
}
}

std::optional<int64_t> riscv_internal::surroundingRegisterGroups(
    Operation *anchor, ArrayRef<Value> excluded) {
  llvm::DenseSet<Value> seen;
  int64_t total = 0;
  for (Operation *at = anchor; at && !isa<riscv::KernelOp>(at); at = at->getParentOp()) {
    auto liveness = BlockRegisterLiveness::get(*at->getBlock());
    if (!liveness) return std::nullopt;
    auto position = liveness->position(at);
    if (!position) return std::nullopt;
    for (const auto &interval : liveness->intervals()) {
      if ((!isa<BlockArgument>(interval.value) && interval.begin >= *position) || interval.end < *position ||
          llvm::is_contained(excluded, interval.value) || !seen.insert(interval.value).second) continue;
      // An initial value with only this control use has transferred to its
      // region argument. Counting it again in the ancestor keeps a dead birth
      // live beside its carry and can double an entire fractional-LMUL tuple.
      if (at != anchor && transfersIntoRegion(at, interval.value)) continue;
      if (interval.groups > std::numeric_limits<int64_t>::max() - total) return std::nullopt;
      total += interval.groups;
    }
  }
  return total;
}

std::optional<BlockRegisterLiveness> BlockRegisterLiveness::get(Block &block, ArrayRef<Value> captures) {
  BlockRegisterLiveness result;
  result.block = &block;
  for (Operation &op : block) {
    result.positions[&op] = result.ordered.size();
    result.ordered.push_back(&op);
  }
  int64_t total = 0;
  auto append = [&](Value value, unsigned begin, bool captured = false) {
    if (registerStorageIdentity(value) != value) return true;
    auto groups = residentRegisterGroups(value);
    if (!groups || *groups < 0 || *groups > std::numeric_limits<int64_t>::max() - total) return false;
    total += *groups;
    if (!*groups || value.use_empty()) return true;
    RegisterInterval interval{begin, begin, *groups, isa<riscv::FragmentType>(value.getType()), value, captured};
    bool usedInBlock = false;
    SmallVector<Value> pending{value};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
      Value alias = pending.pop_back_val();
      if (!visited.insert(alias).second) continue;
      for (OpOperand &use : alias.getUses()) {
        if (Operation *owner = ancestorInBlock(use.getOwner(), &block)) {
          usedInBlock = true;
          interval.end = std::max(interval.end, result.positions.lookup(owner));
        }
        for (Value output : use.getOwner()->getResults())
          if (registerStorageIdentity(output) == value) pending.push_back(output);
      }
    }
    if (!captured || usedInBlock) result.live.push_back(interval);
    return true;
  };
  for (BlockArgument argument : block.getArguments())
    if (!append(argument, 0)) return std::nullopt;
  for (Value capture : captures)
    if (!append(capture, 0, true)) return std::nullopt;
  for (auto [position, op] : llvm::enumerate(result.ordered))
    for (Value value : op->getResults())
      if (!append(value, position)) return std::nullopt;
  return result;
}

std::optional<unsigned> BlockRegisterLiveness::position(Operation *operation) const {
  auto found = positions.find(operation);
  if (found == positions.end()) return std::nullopt;
  return found->second;
}

namespace {
struct Groups {
  int64_t vector = 0;
  int64_t fragment = 0;
  bool add(const Groups &other) {
    if (other.vector > std::numeric_limits<int64_t>::max() - vector ||
        other.fragment > std::numeric_limits<int64_t>::max() - fragment) return false;
    vector += other.vector;
    fragment += other.fragment;
    return vector <= std::numeric_limits<int64_t>::max() - fragment;
  }
  void subtract(const Groups &other) {
    vector -= other.vector;
    fragment -= other.fragment;
  }
};

void record(RegisterPeak &peak, Groups groups, Block *block,
             unsigned point, bool before, unsigned count = 1) {
  if (groups.vector + groups.fragment > peak.total()) {
    peak = {groups.vector, groups.fragment, count, block, point, before};
  } else if (groups.vector + groups.fragment == peak.total()) {
    peak.count += count;
  }
}
}

std::optional<RegisterPeak> BlockRegisterLiveness::peak(unsigned first, unsigned last) const {
  if (ordered.empty()) return first == 0 && last == 0 ? std::optional<RegisterPeak>(RegisterPeak{}) : std::nullopt;
  if (first > last || last >= ordered.size()) return std::nullopt;

  // Sweep interval events rather than rescanning every value at every issue.
  // Before includes block arguments at entry and operands at their last use;
  // after includes a result only when it survives to a later operation.
  SmallVector<Groups> born(ordered.size() + 1), dead(ordered.size() + 1);
  Groups entry;
  for (const RegisterInterval &interval : live) {
    Groups groups = interval.fragment ? Groups{0, interval.groups} : Groups{interval.groups, 0};
    if (interval.captured || isa<BlockArgument>(interval.value)) {
      if (!entry.add(groups)) return std::nullopt;
      if (!dead[interval.end].add(groups)) return std::nullopt;
    } else if (interval.begin < interval.end) {
      if (!born[interval.begin].add(groups) || !dead[interval.end].add(groups)) return std::nullopt;
    }
  }

  RegisterPeak result;
  Groups before = entry;
  for (auto [point, operation] : llvm::enumerate(ordered)) {
    Groups after = before;
    after.subtract(dead[point]);
    if (!after.add(born[point])) return std::nullopt;
    if (point >= first && point <= last) {
      Groups issue = before;
      if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf")) {
        if (leaf.getTemporaryGroups() < 0 || leaf.getFragmentGroups() < 0 ||
            !issue.add({leaf.getTemporaryGroups(), leaf.getFragmentGroups()})) return std::nullopt;
      }
      record(result, issue, block, point, true);
      record(result, after, block, point, false);

      for (Region &region : operation->getRegions()) {
        for (Block &child : region) {
          Groups retained = before;
          SmallVector<Value> captures;
          for (const RegisterInterval &interval : live) {
            const bool born = interval.captured || isa<BlockArgument>(interval.value) || interval.begin < point;
            if (!born || interval.end != point) continue;
            // Only one arm executes. A value with no observer after the if
            // has a separate last use in each arm; do not keep it beside a
            // loop carry for the entire region merely because the other arm
            // contains a second syntactic use.
            const bool conditional = isa<scf::IfOp>(operation);
            const bool transfer = transfersIntoRegion(operation, interval.value, interval.captured);
            if (!conditional && !transfer) continue;
            Groups released = interval.fragment ? Groups{0, interval.groups} : Groups{interval.groups, 0};
            retained.subtract(released);
            if (conditional) captures.push_back(interval.value);
          }
          auto analysis = BlockRegisterLiveness::get(child, captures);
          auto childPeak = analysis ? analysis->peak() : std::nullopt;
          if (!childPeak || retained.vector < 0 || retained.fragment < 0) return std::nullopt;
          if (!retained.add({childPeak->vector, childPeak->fragment})) return std::nullopt;
          if (childPeak->total())
            record(result, retained, childPeak->block, childPeak->point,
                   childPeak->before, std::max(1u, childPeak->count));
        }
      }
    }
    before = after;
  }
  return result;
}

std::optional<RegisterPeak> BlockRegisterLiveness::peak() const {
  return peak(0, ordered.empty() ? 0 : ordered.size() - 1);
}

std::optional<RegisterPeak> riscv_internal::registerPeak(Block &block) {
  auto analysis = BlockRegisterLiveness::get(block);
  return analysis ? analysis->peak() : std::nullopt;
}
