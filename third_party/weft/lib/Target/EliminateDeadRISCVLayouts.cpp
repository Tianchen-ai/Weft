#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"

#include <memory>

using namespace weft;

namespace {

bool eliminateOneDeadSCFForIterArgSet(mlir::ModuleOp module) {
  mlir::scf::ForOp loop;
  llvm::SmallVector<unsigned> deadIndices;
  module.walk([&](mlir::scf::ForOp candidate) {
    for (auto [index, iterArg, result] : llvm::enumerate(
             candidate.getRegionIterArgs(), candidate.getResults())) {
      if (iterArg.use_empty() && result.use_empty())
        deadIndices.push_back(index);
    }
    if (deadIndices.empty())
      return mlir::WalkResult::advance();
    loop = candidate;
    return mlir::WalkResult::interrupt();
  });
  if (!loop)
    return false;

  llvm::BitVector dead(loop.getInitArgs().size());
  for (unsigned index : deadIndices)
    dead.set(index);

  llvm::SmallVector<mlir::Value> keptInitialValues;
  llvm::SmallVector<unsigned> keptIndices;
  for (auto [index, value] : llvm::enumerate(loop.getInitArgs())) {
    if (dead.test(index))
      continue;
    keptInitialValues.push_back(value);
    keptIndices.push_back(index);
  }

  mlir::IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPoint(loop);
  auto replacement = mlir::scf::ForOp::create(rewriter,
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      keptInitialValues,
      [&](mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value induction, mlir::ValueRange iterArgs) {
        mlir::IRMapping mapping;
        mapping.map(loop.getInductionVar(), induction);
        for (auto [replacementIndex, originalIndex] :
             llvm::enumerate(keptIndices))
          mapping.map(loop.getRegionIterArg(originalIndex),
                      iterArgs[replacementIndex]);
        for (mlir::Operation &operation : loop.getBody()->without_terminator())
          builder.clone(operation, mapping);
        auto oldYield =
            mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        llvm::SmallVector<mlir::Value> keptYields;
        for (unsigned originalIndex : keptIndices)
          keptYields.push_back(
              mapping.lookupOrDefault(oldYield.getOperand(originalIndex)));
        mlir::scf::YieldOp::create(builder, location, keptYields);
      });
  replacement->setAttrs(loop->getAttrs());
  for (auto [replacementIndex, originalIndex] : llvm::enumerate(keptIndices))
    loop.getResult(originalIndex)
        .replaceAllUsesWith(replacement.getResult(replacementIndex));
  rewriter.eraseOp(loop);
  return true;
}

bool eliminateOneDeadSCFWhileInputSet(mlir::ModuleOp module) {
  mlir::scf::WhileOp loop;
  llvm::SmallVector<unsigned> deadIndices;
  module.walk([&](mlir::scf::WhileOp candidate) {
    for (auto [index, argument] : llvm::enumerate(candidate.getBeforeArguments()))
      if (argument.use_empty()) deadIndices.push_back(index);
    if (deadIndices.empty()) return mlir::WalkResult::advance();
    loop = candidate;
    return mlir::WalkResult::interrupt();
  });
  if (!loop) return false;

  // The before inputs/backedge and the condition outputs/after arguments are
  // independent signatures. An unused capacity carry need not survive merely
  // because the condition still forwards a useful, loop-invariant extent.
  llvm::BitVector dead(loop.getInits().size());
  for (unsigned index : deadIndices) dead.set(index);
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getAfterBody()->getTerminator());
  for (unsigned index : llvm::reverse(deadIndices)) {
    loop.getInitsMutable().erase(index, 1);
    yield->eraseOperands(index, 1);
  }
  loop.getBeforeBody()->eraseArguments(dead);
  return true;
}

class EliminateDeadRISCVLayoutsPass
    : public mlir::PassWrapper<EliminateDeadRISCVLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateDeadRISCVLayoutsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-eliminate-dead-layouts";
  }

  llvm::StringRef getDescription() const override {
    return "Erase unused pure layout conversions and dead physical producers";
  }

  void runOnOperation() override {
    bool changed = true;
    while (changed) {
      changed = false;
      while (eliminateOneDeadSCFForIterArgSet(getOperation()))
        changed = true;
      while (eliminateOneDeadSCFWhileInputSet(getOperation()))
        changed = true;
      llvm::SmallVector<riscv::ConvertLayoutOp> dead;
      getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
        if (conversion.getConversion().getEffect() == "pure" &&
            conversion.getResult().use_empty())
          dead.push_back(conversion);
      });
      for (riscv::ConvertLayoutOp conversion : dead) {
        conversion.erase();
        changed = true;
      }

      // Resource materialization may spill a value whose reload-side slice is
      // subsequently deleted as a dead physical producer.  The spill writes a
      // compiler-owned local slot and is unobservable once that slot has no
      // reader; retaining it would turn a dead representation choice into real
      // vector stores in the emitted kernel.
      llvm::SmallVector<riscv::SpillOp> unreadSpills;
      getOperation().walk([&](riscv::SpillOp spill) {
        const bool hasReader = llvm::any_of(
            spill.getSlot().getUsers(),
            [](mlir::Operation *user) { return !mlir::isa<riscv::SpillOp>(user); });
        if (!hasReader)
          unreadSpills.push_back(spill);
      });
      for (riscv::SpillOp spill : unreadSpills) {
        spill.erase();
        changed = true;
      }
      llvm::SmallVector<riscv::LocalAllocOp> deadSpillSlots;
      getOperation().walk([&](riscv::LocalAllocOp allocation) {
        if (allocation.getResult().use_empty() &&
            allocation.getResult().getType().getPurpose() == "spill")
          deadSpillSlots.push_back(allocation);
      });
      for (riscv::LocalAllocOp allocation : deadSpillSlots) {
        allocation.erase();
        changed = true;
      }

      llvm::SmallVector<mlir::Operation *> deadProducers;
      getOperation().walk([&](mlir::Operation *operation) {
        if (operation != getOperation() && mlir::isOpTriviallyDead(operation))
          deadProducers.push_back(operation);
      });
      for (mlir::Operation *operation : deadProducers) {
        operation->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createEliminateDeadRISCVLayoutsPass() {
  return std::make_unique<EliminateDeadRISCVLayoutsPass>();
}
