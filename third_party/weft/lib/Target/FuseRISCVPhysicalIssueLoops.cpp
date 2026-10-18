#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "ContractionSupply.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace weft;

namespace {

constexpr llvm::StringLiteral issueWindow = "weft.riscv.issue_window";
constexpr unsigned maximumCandidates = 32;

bool onlyReads(mlir::Operation *operation) {
  return riscv::hasOnlyReadEffects(operation);
}

unsigned readCount(riscv::KernelOp kernel) {
  unsigned count = 0;
  kernel.walk([&](mlir::MemoryEffectOpInterface operation) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    operation.getEffects(effects);
    if (llvm::any_of(effects, [](const auto &effect) {
          return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        }))
      ++count;
  });
  return count;
}

bool sameTraversal(mlir::scf::ForOp first, mlir::scf::ForOp second,
                   mlir::DominanceInfo &dominance) {
  auto window = first->getAttrOfType<mlir::DenseI64ArrayAttr>(issueWindow);
  if (!window || window.size() != 2 || window[0] <= 0 || window[1] <= 0 ||
      first->getAttrs() != second->getAttrs() ||
      first.getLowerBound() != second.getLowerBound() ||
      first.getUpperBound() != second.getUpperBound() ||
      first.getStep() != second.getStep())
    return false;
  for (mlir::Value initial : second.getInitArgs())
    if (!dominance.dominates(initial, first))
      return false;
  for (auto loop : {first, second}) {
    for (mlir::Operation &operation : *loop.getBody()) {
      if (!onlyReads(&operation))
        return false;
      if (loop != second)
        continue;
      for (mlir::Value operand : operation.getOperands()) {
        if (operand.getParentBlock() == second.getBody())
          continue;
        if (!dominance.dominates(operand, first))
          return false;
      }
    }
  }
  return true;
}

mlir::scf::ForOp fuse(mlir::scf::ForOp first, mlir::scf::ForOp second,
                     mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> initial(first.getInitArgs());
  llvm::append_range(initial, second.getInitArgs());
  rewriter.setInsertionPoint(first);
  auto combined = mlir::scf::ForOp::create(rewriter,
      first.getLoc(), first.getLowerBound(), first.getUpperBound(),
      first.getStep(), initial,
      [](mlir::OpBuilder &builder, mlir::Location location,
         mlir::Value, mlir::ValueRange carried) {
        mlir::scf::YieldOp::create(builder, location, carried);
      });
  combined->setAttrs(first->getAttrs());
  // One candidate combines exactly two existing traversals, never an unbounded
  // chain of increasingly wide live sets.
  combined->removeAttr(issueWindow);
  rewriter.eraseOp(combined.getBody()->getTerminator());
  rewriter.setInsertionPointToStart(combined.getBody());
  mlir::IRMapping mapping;
  mapping.map(first.getInductionVar(), combined.getInductionVar());
  mapping.map(second.getInductionVar(), combined.getInductionVar());
  mapping.map(first.getRegionIterArgs(),
              combined.getRegionIterArgs().take_front(first.getNumResults()));
  mapping.map(second.getRegionIterArgs(),
              combined.getRegionIterArgs().drop_front(first.getNumResults()));
  llvm::SmallVector<mlir::Value> yielded;
  for (auto loop : {first, second}) {
    for (mlir::Operation &operation : loop.getBody()->without_terminator())
      rewriter.clone(operation, mapping);
    for (mlir::Value value : loop.getBody()->getTerminator()->getOperands())
      yielded.push_back(mapping.lookupOrDefault(value));
  }
  mlir::scf::YieldOp::create(rewriter, combined.getLoc(), yielded);
  const unsigned firstResults = first.getNumResults();
  rewriter.replaceOp(first, combined.getResults().take_front(firstResults));
  rewriter.replaceOp(second, combined.getResults().drop_front(firstResults));
  return combined;
}

void scheduleReadFrontiers(mlir::scf::ForOp loop) {
  // Fusion can follow mechanical unrolling. Concatenating the two bodies would
  // keep the first issue's shared supplies alive across every later issue in
  // the first body. Complete consumers of an earlier read frontier first, while
  // retaining every SSA dependency (including each ordered accumulator chain).
  llvm::DenseMap<mlir::Operation *, unsigned> frontier, position, indegree;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>> users;
  unsigned nextRead = 0, nextPosition = 0;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    unsigned rank = 0;
    if (!mlir::isMemoryEffectFree(&operation))
      rank = ++nextRead;
    llvm::DenseSet<mlir::Operation *> dependencies;
    for (mlir::Value operand : operation.getOperands()) {
      auto *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != loop.getBody())
        continue;
      rank = std::max(rank, frontier.lookup(definition));
      dependencies.insert(definition);
    }
    frontier[&operation] = rank;
    position[&operation] = nextPosition++;
    indegree[&operation] = dependencies.size();
    for (auto *dependency : dependencies)
      users[dependency].push_back(&operation);
  }
  llvm::SmallVector<mlir::Operation *> ready, scheduled;
  for (mlir::Operation &operation : loop.getBody()->without_terminator())
    if (!indegree.lookup(&operation))
      ready.push_back(&operation);
  while (!ready.empty()) {
    auto selected = llvm::min_element(ready, [&](auto *lhs, auto *rhs) {
      return std::make_pair(frontier.lookup(lhs), position.lookup(lhs)) <
             std::make_pair(frontier.lookup(rhs), position.lookup(rhs));
    });
    auto *operation = *selected;
    ready.erase(selected);
    scheduled.push_back(operation);
    for (auto *user : users[operation])
      if (--indegree[user] == 0)
        ready.push_back(user);
  }
  for (auto *operation : scheduled)
    operation->moveBefore(loop.getBody()->getTerminator());
}

class FuseRISCVPhysicalIssueLoopsPass
    : public mlir::PassWrapper<FuseRISCVPhysicalIssueLoopsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseRISCVPhysicalIssueLoopsPass)
  FuseRISCVPhysicalIssueLoopsPass() = default;
  FuseRISCVPhysicalIssueLoopsPass(const FuseRISCVPhysicalIssueLoopsPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override {
    return "weft-riscv-fuse-physical-issue-loops";
  }
  llvm::StringRef getDescription() const override {
    return "Share supplies and specialize frozen physical issue traversals";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    for (riscv::KernelOp original :
         llvm::make_early_inc_range(getOperation().getOps<riscv::KernelOp>())) {
      auto kernel = original;
      if (kernel.getResourcesMaterialized())
        continue;
      unsigned candidates = 0;
      unsigned selectedPairs = 0, removedReads = 0, rejectedPairs = 0;
      int64_t selectedPeak = 0;
      bool changed = true;
      while (changed && candidates < maximumCandidates) {
        changed = false;
        llvm::SmallVector<mlir::scf::ForOp> loops;
        kernel.walk([&](mlir::scf::ForOp loop) {
          if (loop->hasAttr(issueWindow))
            loops.push_back(loop);
        });
        mlir::DominanceInfo dominance(kernel);
        for (auto first : loops) {
          mlir::Operation *next = first->getNextNode();
          while (next && !mlir::isa<mlir::scf::ForOp>(next) &&
                 next->getNumRegions() == 0 && mlir::isMemoryEffectFree(next))
            next = next->getNextNode();
          auto second = mlir::dyn_cast_or_null<mlir::scf::ForOp>(next);
          if (!second || !sameTraversal(first, second, dominance))
            continue;
          if (candidates == maximumCandidates)
            break;
          ++candidates;
          ++attemptedPairs;
          const unsigned before = readCount(kernel);
          mlir::IRMapping mapping;
          mlir::OwningOpRef<riscv::KernelOp> trial(
              mlir::cast<riscv::KernelOp>(kernel->clone(mapping)));
          auto clonedLoop = [&](mlir::scf::ForOp loop) {
            return mlir::cast<mlir::scf::ForOp>(
                mapping.lookup(loop.getInductionVar())
                    .getParentBlock()->getParentOp());
          };
          auto combined = fuse(clonedLoop(first), clonedLoop(second), rewriter);
          mlir::DominanceInfo trialDominance(trial->getOperation());
          mlir::eliminateCommonSubExpressions(
              rewriter, trialDominance, trial->getOperation());
          const unsigned after = readCount(*trial);
          if (after >= before)
            continue;
          scheduleReadFrontiers(combined);
          const auto pressure =
              riscv_internal::physicalRegisterPressure(*trial);
          if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters()) {
            ++resourceRejections;
            ++rejectedPairs;
            continue;
          }
          sharedReads += before - after;
          removedReads += before - after;
          ++selectedPairs;
          selectedPeak = std::max(selectedPeak, pressure->total());
          maximumSelectedPressure = std::max<int64_t>(
              maximumSelectedPressure, pressure->total());
          ++fusedPairs;
          rewriter.setInsertionPoint(kernel);
          auto selected = trial.release();
          rewriter.insert(selected.getOperation());
          rewriter.eraseOp(kernel);
          kernel = selected;
          changed = true;
          break;
        }
      }
      kernel.walk([](mlir::scf::ForOp loop) { loop->removeAttr(issueWindow); });
      // Consumer layouts and dead representations have now been resolved.
      // Query the complete current live set before instantiating frozen parts.
      llvm::SmallVector<mlir::scf::ForOp> numericLoops;
      kernel.walk<mlir::WalkOrder::PostOrder>(
          [&](mlir::scf::ForOp loop) { numericLoops.push_back(loop); });
      for (auto loop : numericLoops)
        if (mlir::failed(riscv_internal::specializeContractionParts(loop))) {
          signalPassFailure();
          return;
        }
      if (candidates)
        llvm::errs() << "weft-issue-fusion: candidates=" << candidates
                     << " selected=" << selectedPairs
                     << " shared-reads=" << removedReads
                     << " resource-rejections=" << rejectedPairs
                     << " estimated-peak-groups=" << selectedPeak << '\n';
    }
  }

private:
  Statistic attemptedPairs{this, "attempted-pairs", "Bounded physical issue fusion candidates"};
  Statistic fusedPairs{this, "fused-pairs", "Read-sharing issue loop pairs selected"};
  Statistic sharedReads{this, "shared-reads", "Static read operations removed by selected fusion"};
  Statistic resourceRejections{this, "resource-rejections", "Candidates exceeding the final register budget"};
  Statistic maximumSelectedPressure{this, "maximum-selected-pressure", "Largest selected region-aware register estimate"};
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createFuseRISCVPhysicalIssueLoopsPass() {
  return std::make_unique<FuseRISCVPhysicalIssueLoopsPass>();
}
