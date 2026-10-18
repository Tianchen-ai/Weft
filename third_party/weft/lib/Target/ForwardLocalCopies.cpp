#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "DenseIssue.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/LocalTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;
using namespace weft;

namespace {

std::optional<int64_t> constantIndex(Value value, unsigned depth = 0) {
  if (depth > 16) return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) return constant.value();
  if (auto extent = value.getDefiningOp<riscv::ExtentOp>())
    if (auto view = extent.getBase().getDefiningOp<riscv::LocalViewOp>())
      return constantIndex(view.getExtents()[extent.getAxis()], depth + 1);
  Operation *op = value.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp>(op)) return std::nullopt;
  auto lhs = constantIndex(op->getOperand(0), depth + 1);
  auto rhs = constantIndex(op->getOperand(1), depth + 1);
  if (!lhs || !rhs || *lhs < 0 || *rhs < 0) return std::nullopt;
  __int128 result;
  if (isa<arith::AddIOp>(op)) result = __int128(*lhs) + *rhs;
  else if (isa<arith::SubIOp>(op)) result = __int128(*lhs) - *rhs;
  else if (isa<arith::MulIOp>(op)) result = __int128(*lhs) * *rhs;
  else {
    if (!*rhs) return std::nullopt;
    result = *lhs / *rhs;
  }
  if (result < 0 || result > std::numeric_limits<int64_t>::max()) return std::nullopt;
  return int64_t(result);
}

bool sameIndex(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto a = constantIndex(lhs), b = constantIndex(rhs);
  return a && b && a == b;
}

struct Rectangle {
  SmallVector<std::pair<int64_t, int64_t>> axes;
  int64_t elements = 1;
};

// A copy must cover each physical element exactly once. Checking the complete
// rectangular iteration domains admits vector strips plus scalar remainders,
// while rejecting partial initialization, repeated stores and sparse gathers.
std::optional<Rectangle> coverage(riscv::SubviewOp view, Operation *root) {
  if (view.getExtents().size() > 8 ||
      llvm::any_of(view.getExtents(), [](int64_t n) { return n <= 0; }) ||
      llvm::any_of(view.getOffsets(), [](int64_t n) { return n != -1; }) ||
      view.getDynamicBounds().size() != view.getExtents().size()) return std::nullopt;
  Rectangle result;
  llvm::DenseSet<Operation *> dimensions;
  for (auto [coordinate, width] : llvm::zip(view.getDynamicBounds(), view.getExtents())) {
    int64_t begin, end;
    if (auto constant = constantIndex(coordinate)) {
      begin = *constant;
      if (begin < 0 || width > std::numeric_limits<int64_t>::max() - begin) return std::nullopt;
      end = begin + width;
    } else {
      auto argument = dyn_cast<BlockArgument>(coordinate);
      auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
      if (!loop || loop.getInductionVar() != argument ||
          (loop != root && !root->isAncestor(loop)) || !dimensions.insert(loop).second) return std::nullopt;
      auto lower = constantIndex(loop.getLowerBound()), upper = constantIndex(loop.getUpperBound());
      auto step = constantIndex(loop.getStep());
      if (!lower || !upper || !step || *lower < 0 || *upper < *lower ||
          *step != width || (*upper - *lower) % width) return std::nullopt;
      begin = *lower;
      end = *upper;
    }
    result.axes.emplace_back(begin, end);
    if (end - begin && result.elements > std::numeric_limits<int64_t>::max() / (end - begin)) return std::nullopt;
    result.elements *= end - begin;
  }
  for (Operation *parent = view; parent && parent != root->getParentOp(); parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(parent); loop && !dimensions.contains(loop)) {
      auto lower = constantIndex(loop.getLowerBound()), upper = constantIndex(loop.getUpperBound());
      auto step = constantIndex(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower || *upper - *lower > *step)
        return std::nullopt;
    }
    if (parent == root) break;
  }
  return result;
}

bool overlap(const Rectangle &lhs, const Rectangle &rhs) {
  if (!lhs.elements || !rhs.elements) return false;
  for (auto [a, b] : llvm::zip(lhs.axes, rhs.axes))
    if (a.second <= b.first || b.second <= a.first) return false;
  return true;
}

// A dead source can transfer ownership; a retained source needs a bounded byte
// transfer. Both choices consume this same complete coverage proof. Identify one
// completed traversal by its destination descriptor, not by a kernel name or
// the allocation's other writes in earlier/later control-flow iterations.
class CopyIssues {
public:
  explicit CopyIssues(riscv::LocalViewOp destination) : destination(destination) {}

  bool analyze() {
    auto target = destination->getParentOfType<riscv::KernelOp>().getTarget();
    SmallVector<Value> pending{destination.getResult()};
    llvm::DenseSet<Value> visited;
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 256) return false;
      Value value = pending[cursor];
      if (!visited.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (riscv::isStorageMetadataUse(use)) continue;
        if (auto view = dyn_cast<riscv::SubviewOp>(user); view && view.getBase() == value) {
          pending.push_back(view.getResult());
        } else if (auto store = dyn_cast<riscv::StoreOp>(user); store && store.getRegion() == value) {
          if (writes.size() >= 32) return false;
          writes.push_back(store);
        } else return false;
      }
    }
    if (writes.empty()) return false;
    int64_t elements = 0;
    for (auto store : writes) {
      auto loaded = store.getValue().getDefiningOp<riscv::LoadOp>();
      auto output = store.getRegion().getDefiningOp<riscv::SubviewOp>();
      auto input = loaded ? loaded.getRegion().getDefiningOp<riscv::SubviewOp>() : riscv::SubviewOp();
      auto origin = input ? input.getBase().getDefiningOp<riscv::LocalViewOp>() : riscv::LocalViewOp();
      if (!loaded || !loaded.getResult().hasOneUse() || loaded.getSnapshotStorage() || !output || !input ||
          !origin || output.getBase() != destination.getResult() ||
          input.getExtents() != output.getExtents() || input.getOffsets() != output.getOffsets() ||
          input.getDynamicBounds().size() != output.getDynamicBounds().size()) return false;
      for (auto [from, to] : llvm::zip(input.getDynamicBounds(), output.getDynamicBounds()))
        if (!sameIndex(from, to)) return false;
      if (source && source != origin) return false;
      source = origin;
      auto value = dyn_cast<riscv::ValueType>(loaded.getResult().getType());
      if (!value || value.getLayout().getValidity() != "full" ||
          value.getShape() != input.getType().getShape() ||
          value.getAxisIds() != input.getType().getAxisIds()) return false;
      auto count = riscv_internal::staticProduct(value.getShape().asArrayRef());
      if (!count || *count <= 0) return false;
      Operation *root = destination->getBlock()->findAncestorOpInBlock(*store);
      if (!root || !destination->isBeforeInBlock(root)) return false;
      auto rectangle = coverage(output, root);
      if (!rectangle) return false;
      for (auto [range, extent] : llvm::zip(rectangle->axes, destination.getType().getShape().asArrayRef()))
        if (range.first < 0 || range.second > extent) return false;
      for (const auto &other : rectangles) if (overlap(*rectangle, other)) return false;
      if (rectangle->elements > std::numeric_limits<int64_t>::max() - elements ||
          rectangle->elements % *count) return false;
      elements += rectangle->elements;
      int64_t issues = rectangle->elements / *count;
      if (issues > std::numeric_limits<int64_t>::max() - originalIssues) return false;
      originalIssues += issues;
      rectangles.push_back(*rectangle);
      if (!llvm::is_contained(roots, root)) roots.push_back(root);
      loads.insert(loaded);
    }
    auto total = riscv_internal::staticProduct(destination.getType().getShape().asArrayRef());
    auto extent = riscv::localTransferBytes(source.getResult(), destination.getResult());
    if (!total || elements != *total || !extent || !*extent) return false;
    bytes = *extent;
    llvm::sort(roots, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    DominanceInfo dominance(destination->getParentOfType<riscv::KernelOp>());
    if (!dominance.properlyDominates(source.getOperation(), roots.front())) return false;
    for (Operation *root : roots) if (!copyOnly(root)) return false;
    // Preserve the original read and completion boundary. No intervening
    // access may change the source or observe a partially initialized target.
    for (Operation *op = roots.front(); op && op != roots.back(); op = op->getNextNode()) {
      if (llvm::is_contained(roots, op)) continue;
      if (op->getNumRegions()) return false;
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effects) { if (!isMemoryEffectFree(op)) return false; continue; }
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto effect : instances) {
        if (!isa<MemoryEffects::Read>(effect.getEffect()) || !effect.getValue() ||
            riscv::mayAliasStorage(effect.getValue(), destination.getResult())) return false;
      }
    }
    if (canReuse()) { reuse = true; return true; }
    auto resident = riscv_internal::surroundingRegisterGroups(roots.front());
    if (!resident) return false;
    int64_t bestIssues = originalIssues;
    for (int64_t lmul : {8, 16, 32}) {
      if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul) ||
          *resident > target.getVectorRegisters() - lmul / 8) continue;
      int64_t capacity = target.getVlenBits() * lmul / 64;
      int64_t issues = 1 + (bytes - 1) / capacity;
      if (issues < bestIssues) { bestIssues = issues; selectedLMUL = lmul; width = capacity; }
    }
    return selectedLMUL != 0;
  }

  LogicalResult rewrite(IRRewriter &b, llvm::DenseSet<Operation *> &erased) {
    Operation *anchor = roots.front();
    if (reuse) {
      b.setInsertionPoint(replacedAllocation);
      auto transfer = riscv::LocalReuseOp::create(b, replacedAllocation.getLoc(), replacedAllocation.getType(),
          reusedSource.getStorage(), replacedAllocation.getSizeBytes());
      riscv_internal::copyOrigin(replacedAllocation, transfer);
      b.replaceOp(replacedAllocation, transfer.getResult());
    } else {
      b.setInsertionPoint(anchor);
      Value extent = arith::ConstantIndexOp::create(b, anchor->getLoc(), bytes);
      auto selected = riscv_internal::leaf(b, "transfer", "copy", "rvv.local-copy", "rvv.local-copy",
          0, 0, selectedLMUL / 8, 0, "none", "exact", {selectedLMUL, width});
      auto result = riscv_internal::emitDenseIssues(b, anchor, extent, width, {},
          [&](Value offset, Value active, bool, ValueRange) -> FailureOr<SmallVector<Value>> {
        auto copy = riscv::LocalCopyOp::create(b, anchor->getLoc(), source, destination, offset, active, selected);
        riscv_internal::copyOrigin(anchor, copy);
        return SmallVector<Value>{};
      });
      if (failed(result)) return anchor->emitError("selected local copy has no bounded issue traversal");
    }
    for (Operation *root : roots) {
      root->walk([&](Operation *operation) { erased.insert(operation); });
      b.eraseOp(root);
    }
    return success();
  }

private:
  riscv::LocalViewOp destination, source;
  SmallVector<riscv::StoreOp> writes;
  SmallVector<Rectangle> rectangles;
  SmallVector<Operation *> roots;
  llvm::DenseSet<Operation *> loads;
  int64_t bytes = 0, width = 0, selectedLMUL = 0, originalIssues = 0;
  bool reuse = false;
  riscv::LocalBindOp reusedSource;
  riscv::LocalAllocOp replacedAllocation;

  bool canReuse() {
    auto binding = destination.getSource().getDefiningOp<riscv::LocalBindOp>();
    reusedSource = source.getSource().getDefiningOp<riscv::LocalBindOp>();
    if (!binding || !reusedSource || binding.getType() != reusedSource.getType()) return false;
    replacedAllocation = binding.getStorage().getDefiningOp<riscv::LocalAllocOp>();
    if (!replacedAllocation || !replacedAllocation.getResult().hasOneUse()) return false;
    auto type = binding.getStorage().getType(), previous = reusedSource.getStorage().getType();
    if ((type.getSchema() != "local-value-copy" && type.getSchema() != "physical-share" &&
         type.getSchema() != "staged-value" && type.getSchema() != "control-transfer") ||
        type.getSizeBytes() != previous.getSizeBytes() || type.getSizeBytes() != bytes ||
        type.getElementType() != previous.getElementType() || type.getAlignment() > previous.getAlignment() ||
        type.getOwnerDomainId() != previous.getOwnerDomainId() ||
        type.getLifetimeEndDomainId() != previous.getLifetimeEndDomainId()) return false;
    Operation *birth = reusedSource.getStorage().getDefiningOp();
    if (!isa_and_nonnull<riscv::LocalAllocOp, riscv::LocalReuseOp>(birth) ||
        birth->getBlock() != replacedAllocation->getBlock() || !birth->isBeforeInBlock(replacedAllocation)) return false;
    for (Operation *root : roots)
      if (root->getBlock() != replacedAllocation->getBlock() || !replacedAllocation->isBeforeInBlock(root)) return false;
    // This descriptor names one traversal. Ownership transfer additionally
    // proves that every alias of the entire destination is initialized by it.
    SmallVector<Value> pending{binding.getResult()};
    llvm::DenseSet<Value> visited;
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 1024) return false;
      Value value = pending[cursor];
      if (!visited.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        if (riscv::isStorageMetadataUse(use)) continue;
        SmallVector<Value> aliases;
        riscv::appendStorageAliases(use, aliases);
        if (!aliases.empty()) { llvm::append_range(pending, aliases); continue; }
        Operation *user = use.getOwner();
        if (auto store = dyn_cast<riscv::StoreOp>(user)) {
          if (!llvm::is_contained(writes, store)) return false;
          continue;
        }
        if (user->getNumRegions() || !riscv::hasOnlyReadEffects(user)) return false;
        if (!isMemoryEffectFree(user)) {
          auto at = replacedAllocation->getBlock()->findAncestorOpInBlock(*user);
          if (!at || !roots.back()->isBeforeInBlock(at)) return false;
        }
      }
    }
    pending.assign(1, reusedSource.getStorage());
    visited.clear();
    for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
      if (cursor >= 1024) return false;
      Value value = pending[cursor];
      if (!visited.insert(value).second) continue;
      for (OpOperand &use : value.getUses()) {
        if (riscv::isStorageMetadataUse(use)) continue;
        Operation *user = use.getOwner();
        if (isa<riscv::LocalReuseOp>(user)) return false;
        if (llvm::any_of(roots, [&](Operation *root) { return root == user || root->isAncestor(user); })) continue;
        Operation *at = replacedAllocation->getBlock()->findAncestorOpInBlock(*user);
        if (!at || at == replacedAllocation || !at->isBeforeInBlock(replacedAllocation)) return false;
        riscv::appendStorageAliases(use, pending);
      }
    }
    return true;
  }

  bool copyOnly(Operation *root) const {
    return !root->walk([&](Operation *op) {
      if (auto loop = dyn_cast<scf::ForOp>(op))
        return loop.getNumResults() == 0 ? WalkResult::advance() : WalkResult::interrupt();
      if (isa<scf::YieldOp>(op) || loads.contains(op)) return WalkResult::advance();
      if (auto store = dyn_cast<riscv::StoreOp>(op))
        return llvm::is_contained(writes, store) ? WalkResult::advance() : WalkResult::interrupt();
      if (op->getNumRegions() || !isMemoryEffectFree(op)) return WalkResult::interrupt();
      return WalkResult::advance();
    }).wasInterrupted();
  }
};
}

LogicalResult riscv_internal::forwardLocalCopies(ModuleOp module, IRRewriter &builder) {
  SmallVector<riscv::LocalViewOp> views;
  module.walk([&](riscv::LocalViewOp view) { views.push_back(view); });
  llvm::DenseSet<Operation *> erased;
  for (auto view : views) {
    if (erased.contains(view)) continue;
    CopyIssues copy(view);
    if (copy.analyze() && failed(copy.rewrite(builder, erased))) return failure();
  }
  return success();
}
