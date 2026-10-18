#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <memory>

namespace {

bool dependsOnDeferredField(mlir::Value value,
                            llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition ||
      mlir::isa<weft::riscv::RegisterMaterializeOp>(definition))
    return false;
  if (mlir::isa<weft::riscv::FieldOp>(definition))
    return true;
  if (!mlir::isMemoryEffectFree(definition) || !visited.insert(definition).second)
    return false;
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return dependsOnDeferredField(operand, visited);
  });
}

bool hasAtLeastIterations(mlir::LoopLikeOpInterface loop, int64_t count) {
  auto forLoop = mlir::dyn_cast<mlir::scf::ForOp>(loop.getOperation());
  if (!forLoop)
    return false;
  auto lower =
      forLoop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto upper =
      forLoop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step = forLoop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!lower || !upper || !step || step.value() <= 0 ||
      lower.value() < 0 || upper.value() <= lower.value()) return false;
  return (upper.value() - lower.value() - 1) / step.value() + 1 >= count;
}

void materializeDeferredScalarFieldsBeforeReadLoops(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<weft::riscv::FieldOp> fields;
  module.walk([&](weft::riscv::FieldOp field) { fields.push_back(field); });
  for (weft::riscv::FieldOp field : fields) {
    mlir::Type type = field.getResult().getType();
    if ((!type.isIndex() &&
         !mlir::isa<mlir::IntegerType, mlir::FloatType>(type)) ||
        llvm::any_of(field.getResult().getUsers(), [](mlir::Operation *user) {
          return mlir::isa<weft::riscv::RegisterMaterializeOp>(user);
        }))
      continue;

    mlir::Operation *target = nullptr;
    for (mlir::Operation *cursor = field->getPrevNode(); cursor;
         cursor = cursor->getPrevNode()) {
      if (cursor == field.getOwner().getDefiningOp())
        break;
      auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(cursor);
      if (loop) {
        if (!loop.isDefinedOutsideOfLoop(field.getOwner()) ||
            !hasAtLeastIterations(loop, 2) ||
            !weft::riscv::hasOnlyReadEffects(loop, true))
          break;
        target = cursor;
        continue;
      }
      if (!weft::riscv::hasOnlyReadEffects(cursor))
        break;
    }
    if (!target)
      continue;

    field->moveBefore(target);
    rewriter.setInsertionPointAfter(field);
    auto materialized = weft::riscv::RegisterMaterializeOp::create(rewriter,
        field.getLoc(), type, field.getResult(), -1, -1, -1,
        "physical-share");
    weft::riscv_internal::copyOrigin(field, materialized);
    field.getResult().replaceAllUsesExcept(materialized.getResult(),
                                           materialized.getOperation());
  }
}

class HoistRISCVLoopInvariantsPass
    : public mlir::PassWrapper<HoistRISCVLoopInvariantsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistRISCVLoopInvariantsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-hoist-loop-invariants";
  }
  llvm::StringRef getDescription() const override {
    return "Hoist side-effect-free physical coordinate and value expressions";
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::LoopLikeOpInterface> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::LoopLikeOpInterface loop) { loops.push_back(loop); });
    for (mlir::LoopLikeOpInterface loop : loops) {
      // A fully unrolled loop can retain its Level boundary with one physical
      // iteration. Hoisting its supplies saves no execution and makes every
      // decoded part resident across the whole body instead of its consumers.
      if (hasAtLeastIterations(loop, 1) && !hasAtLeastIterations(loop, 2))
        continue;
      // A memory read may only move out when this body is known to execute.
      // Unknown effects and guards remain barriers even if they have no
      // registered Write instance. Numeric SSA motion is a separate question.
      const bool canHoistReads = hasAtLeastIterations(loop, 1) &&
          weft::riscv::hasOnlyReadEffects(loop, true);
      auto regions = loop.getLoopRegions();
      mlir::moveLoopInvariantCode(
          regions,
          [&](mlir::Value value, mlir::Region *) {
            return loop.isDefinedOutsideOfLoop(value);
          },
          [&](mlir::Operation *operation, mlir::Region *) {
            if (operation->getDialect() &&
                operation->getDialect()->getNamespace() == "arith")
              return mlir::isMemoryEffectFree(operation);
            if (auto lookup =
                    mlir::dyn_cast<weft::riscv::LookupOp>(operation))
              return mlir::isa<weft::riscv::ValueType>(
                  lookup.getTable().getType());
            if (mlir::isa<weft::riscv::RVVReplicaStorageLoadOp>(operation))
              return canHoistReads && operation->getBlock() ==
                  mlir::cast<mlir::scf::ForOp>(loop.getOperation()).getBody();
            if (auto conversion =
                    mlir::dyn_cast<weft::riscv::ConvertLayoutOp>(operation))
              return conversion.getConversion().getEffect() == "pure";
            return mlir::isa<weft::riscv::ConstantOp,
                             weft::riscv::IotaOp,
                             weft::riscv::UnaryOp, weft::riscv::BinaryOp,
                             weft::riscv::CompareOp, weft::riscv::CastOp,
                             weft::riscv::NarrowOp, weft::riscv::WidenOp,
                             weft::riscv::LookupOp, weft::riscv::FieldOp,
                             weft::riscv::ExtractOp>(operation) &&
                   mlir::isMemoryEffectFree(operation);
          },
          [&](mlir::Operation *operation, mlir::Region *) {
            loop.moveOutOfLoop(operation);
          });
    }
    mlir::IRRewriter rewriter(&getContext());
    materializeDeferredScalarFieldsBeforeReadLoops(getOperation(), rewriter);
    llvm::SmallVector<mlir::Operation *> operations;
    getOperation().walk(
        [&](mlir::Operation *operation) { operations.push_back(operation); });
    for (mlir::Operation *operation : operations) {
      for (mlir::Value result : operation->getResults()) {
        mlir::Type type = result.getType();
        if (!type.isIndex() &&
            !mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
          continue;
        llvm::SmallPtrSet<mlir::Operation *, 8> visited;
        // Keep an outer scalar computation at its physical SSA definition.
        // An inline C expression would otherwise sink into every descendant
        // iteration even though LICM has already placed the producer outside.
        bool scalarComputation =
            mlir::isa<weft::riscv::UnaryOp, weft::riscv::BinaryOp,
                      weft::riscv::CompareOp, weft::riscv::SelectOp,
                      weft::riscv::CastOp, weft::riscv::NarrowOp,
                      weft::riscv::WidenOp>(operation);
        if (!scalarComputation && !dependsOnDeferredField(result, visited))
          continue;
        llvm::SmallVector<mlir::OpOperand *> nestedUses;
        for (mlir::OpOperand &use : result.getUses()) {
          mlir::Operation *ancestor = use.getOwner();
          while (ancestor && ancestor->getBlock() != operation->getBlock())
            ancestor = ancestor->getParentOp();
          if (ancestor && ancestor != use.getOwner())
            nestedUses.push_back(&use);
        }
        if (nestedUses.empty())
          continue;
        mlir::OpBuilder builder(operation);
        builder.setInsertionPointAfter(operation);
        auto materialized = weft::riscv::RegisterMaterializeOp::create(builder,
            operation->getLoc(), type, result, -1, -1, -1,
            "physical-share");
        weft::riscv_internal::copyOrigin(operation,
                                         materialized.getOperation());
        for (mlir::OpOperand *use : nestedUses)
          use->set(materialized.getResult());
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createHoistRISCVLoopInvariantsPass() {
  return std::make_unique<HoistRISCVLoopInvariantsPass>();
}
