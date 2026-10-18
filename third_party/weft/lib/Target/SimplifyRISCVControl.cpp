#include "Weft/Target/RISCVPasses.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace weft;

namespace {
// This pass only deletes proven work or substitutes an already computed scalar
// SSA value. It never versions a body, changes a layout, or speculates a read.
// In particular, a known branch is resolved before visiting its dead arm: that
// arm need not acquire a legal memory specialization or resource assignment.
class ControlSimplifier {
public:
  explicit ControlSimplifier(IRRewriter &rewriter) : b(rewriter) {}

  void run(Region &region) {
    for (Block &block : region)
      for (auto iterator = block.begin(); iterator != block.end();) {
        Operation *operation = &*iterator++;
        if (auto branch = dyn_cast<scf::IfOp>(operation)) {
          if (auto truth = riscv::denseConditionAt(branch.getCondition(), branch)) {
            Region &selected = *truth ? branch.getThenRegion() : branch.getElseRegion();
            if (selected.empty()) { b.eraseOp(branch); ++removedBranches; continue; }
            run(selected);
            auto yield = cast<scf::YieldOp>(selected.front().getTerminator());
            SmallVector<Value> results(yield.getOperands());
            b.eraseOp(yield);
            b.inlineBlockBefore(&selected.front(), branch);
            b.replaceOp(branch, results);
            ++removedBranches;
            continue;
          }
        }
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
          auto direction = loop->getAttrOfType<StringAttr>("weft.riscv.direction");
          auto level = loop->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
          bool ascending = level ? level.getDirection() == "ascending" :
              direction && direction.getValue() == "ascending";
          auto empty = riscv::denseIndexCompareAt(arith::CmpIPredicate::uge,
              loop.getLowerBound(), loop.getUpperBound(), loop);
          if (ascending && empty && *empty) {
            b.replaceOp(loop, loop.getInitArgs());
            ++removedLoops;
            continue;
          }
        }
        for (Region &nested : operation->getRegions()) run(nested);
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
          if (inlineSingleIssue(loop)) continue;
          removeUnusedCarries(loop);
        }
        if (operation->getNumResults() == 1 && !operation->getNumRegions())
          foldScalar(operation);
      }
    // Worklist deletion, rather than repeated whole-module scans, also removes
    // arithmetic that became unused when an enclosing branch disappeared.
    for (Block &block : region) {
      SmallVector<Operation *> pending;
      llvm::SmallPtrSet<Operation *, 32> queued;
      auto enqueue = [&](Operation *operation) {
        if (operation && operation->getBlock() == &block && removable(operation) && queued.insert(operation).second)
          pending.push_back(operation);
      };
      for (Operation &operation : block) enqueue(&operation);
      while (!pending.empty()) {
        Operation *operation = pending.pop_back_val();
        queued.erase(operation);
        if (!removable(operation)) continue;
        SmallVector<Operation *> inputs;
        for (Value operand : operation->getOperands()) inputs.push_back(operand.getDefiningOp());
        b.eraseOp(operation);
        for (Operation *input : inputs) enqueue(input);
      }
    }
  }

  unsigned removedBranches = 0, removedLoops = 0, foldedScalars = 0;

private:
  IRRewriter &b;

  static bool removable(Operation *operation) {
    // Descriptor and numerical lifetime cleanup has its own owner. Scalar
    // arithmetic is the only dead graph this pass recursively prunes.
    return operation->getDialect() && operation->getDialect()->getNamespace() == "arith" &&
        !operation->getNumRegions() && operation->use_empty() && isMemoryEffectFree(operation);
  }

  bool inlineSingleIssue(scf::ForOp loop) {
    if (loop->hasAttr("weft.riscv.level") || loop->hasAttr("weft.riscv.schedule") ||
        !loop->hasAttr("weft.riscv.system_unroll")) return false;
    auto direction = loop->getAttrOfType<StringAttr>("weft.riscv.direction");
    if (!direction || direction.getValue() != "ascending") return false;
    auto lower = riscv::denseIndexBoundsAt(loop.getLowerBound(), loop).constant();
    auto upper = riscv::denseIndexBoundsAt(loop.getUpperBound(), loop).constant();
    auto step = riscv::denseIndexBoundsAt(loop.getStep(), loop).constant();
    if (!lower || !upper || !step || *lower < 0 || *upper <= *lower ||
        *step <= 0 || *upper - *lower > *step) return false;
    bool changesLifetime = false;
    loop.walk([&](Operation *operation) {
      if (operation != loop && (operation->hasAttr("weft.riscv.level") || operation->hasAttr("birth_id") ||
          isa<riscv::LocalAllocOp, riscv::LocalBindOp, riscv::NewOp, riscv::EmptyOp,
              riscv::MaterializeOp, riscv::RegisterMaterializeOp, riscv::StagedViewOp>(operation)))
        changesLifetime = true;
      // Stream steps retain their selected issue loop as the owner of the
      // downstream-unroll contract, including a one-issue remainder.
      if (isa<riscv::RVVStreamReduceStepOp, riscv::RVVStreamContractStepOp>(operation) &&
          operation->getParentOfType<scf::ForOp>() == loop) changesLifetime = true;
    });
    if (changesLifetime) return false;
    SmallVector<Value> inputs{loop.getLowerBound()};
    llvm::append_range(inputs, loop.getInitArgs());
    for (auto [argument, input] : llvm::zip(loop.getBody()->getArguments(), inputs))
      argument.replaceAllUsesWith(input);
    // Substitution can expose a false complete-window arm containing a static
    // subview that would be out of bounds if executed. Resolve that control
    // before the enclosing pass verifier sees the newly constant coordinates.
    run(loop.getRegion());
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    SmallVector<Value> results(yield.getOperands());
    b.eraseOp(yield);
    // MLIR still requires a replacement for every block argument, including
    // arguments whose uses were substituted above before control folding.
    b.inlineBlockBefore(loop.getBody(), loop, inputs);
    b.replaceOp(loop, results);
    ++removedLoops;
    return true;
  }

  void removeUnusedCarries(scf::ForOp loop) {
    // Keep the SCF signature and schedule stable. Forward invariant carries at
    // uses, but leave pruning the typed loop's argument list to its owner.
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [initial, argument, result, returned] :
         llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(), loop.getResults(), yield.getOperands())) {
      if (returned != argument && returned != initial) continue;
      if (!initial.getType().isIndex() && !initial.getType().isInteger(1)) continue;
      argument.replaceAllUsesWith(initial);
      result.replaceAllUsesWith(initial);
    }
  }

  void replace(Operation *operation, Value value) {
    if (operation->getResult(0) == value) return;
    b.replaceOp(operation, value);
    ++foldedScalars;
  }

  void foldScalar(Operation *operation) {
    Value result = operation->getResult(0);
    if (isa<arith::ConstantOp, riscv::ConstantOp, riscv::LocalShapeOp>(operation)) return;
    // Do not replace a memory read or a physical identity by a constant even
    // when its result happens to be used as an index. Only scalar arithmetic
    // and the read-only explicit descriptor extent participate here.
    bool arithmetic = operation->getDialect() && operation->getDialect()->getNamespace() == "arith";
    if (!arithmetic && !isa<riscv::ExtentOp>(operation)) return;
    b.setInsertionPoint(operation);
    if (result.getType().isInteger(1)) {
      if (auto known = riscv::denseConditionAt(result, operation))
        replace(operation, arith::ConstantIntOp::create(b, operation->getLoc(), *known, 1));
      return;
    }
    if (!result.getType().isIndex()) return;
    // Extents are descriptor facts and can be shared outside the branch. Their
    // unconditional fold preserves proof identity in memory verifiers.
    auto bounds = isa<riscv::ExtentOp>(operation) ? riscv::denseIndexBounds(result)
                                                : riscv::denseIndexBoundsAt(result, operation);
    if (auto constant = bounds.constant()) {
      replace(operation, arith::ConstantIndexOp::create(b, operation->getLoc(), *constant));
      return;
    }
    if (auto extent = dyn_cast<riscv::ExtentOp>(operation)) {
      if (auto value = dyn_cast_if_present<Value>(riscv::denseDescriptorExtent(extent.getBase(), extent.getAxis())))
        replace(operation, value);
      return;
    }
    if (auto minimum = dyn_cast<arith::MinUIOp>(operation)) {
      if (auto first = riscv::denseIndexCompareAt(arith::CmpIPredicate::ule,
              minimum.getLhs(), minimum.getRhs(), operation))
        replace(operation, *first ? minimum.getLhs() : minimum.getRhs());
      return;
    }
    if (auto maximum = dyn_cast<arith::MaxUIOp>(operation)) {
      if (auto first = riscv::denseIndexCompareAt(arith::CmpIPredicate::uge,
              maximum.getLhs(), maximum.getRhs(), operation))
        replace(operation, *first ? maximum.getLhs() : maximum.getRhs());
      return;
    }
    if (auto selection = dyn_cast<arith::SelectOp>(operation)) {
      if (auto truth = riscv::denseConditionAt(selection.getCondition(), operation))
        replace(operation, *truth ? selection.getTrueValue() : selection.getFalseValue());
      return;
    }
    if (auto addition = dyn_cast<arith::AddIOp>(operation)) {
      if (riscv::denseConstantIndex(addition.getLhs()) == 0) replace(operation, addition.getRhs());
      else if (riscv::denseConstantIndex(addition.getRhs()) == 0) replace(operation, addition.getLhs());
      return;
    }
    if (auto subtraction = dyn_cast<arith::SubIOp>(operation)) {
      if (riscv::denseConstantIndex(subtraction.getRhs()) == 0) replace(operation, subtraction.getLhs());
      return;
    }
    if (auto multiplication = dyn_cast<arith::MulIOp>(operation)) {
      if (riscv::denseConstantIndex(multiplication.getLhs()) == 1) replace(operation, multiplication.getRhs());
      else if (riscv::denseConstantIndex(multiplication.getRhs()) == 1) replace(operation, multiplication.getLhs());
      return;
    }
    if (auto division = dyn_cast<arith::DivUIOp>(operation))
      if (riscv::denseConstantIndex(division.getRhs()) == 1) replace(operation, division.getLhs());
  }
};

class SimplifyRISCVControlPass
    : public PassWrapper<SimplifyRISCVControlPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyRISCVControlPass)
  StringRef getArgument() const override { return "weft-riscv-simplify-control"; }
  StringRef getDescription() const override { return "Resolve proven physical bounds before memory versioning"; }
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    for (auto kernel : getOperation().getOps<riscv::KernelOp>()) {
      if (kernel.getResourcesMaterialized()) continue;
      ControlSimplifier simplifier(rewriter);
      simplifier.run(kernel.getBody());
    }
  }
};
}

std::unique_ptr<Pass> weft::createSimplifyRISCVControlPass() {
  return std::make_unique<SimplifyRISCVControlPass>();
}
