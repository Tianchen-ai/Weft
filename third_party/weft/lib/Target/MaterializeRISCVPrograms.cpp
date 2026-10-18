#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

void copyProvenance(mlir::Operation *source, mlir::Operation *target) {
  for (llvm::StringRef name : {"canonical_op", "source_origin"})
    if (mlir::Attribute attribute = source->getAttr(name))
      target->setAttr(name, attribute);
}

class MaterializeRISCVProgramsPass final
    : public mlir::PassWrapper<MaterializeRISCVProgramsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVProgramsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-materialize-programs";
  }

  llvm::StringRef getDescription() const final {
    return "Materialize transient RISC-V macro programs into explicit control and SSA";
  }

  void runOnOperation() final {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    llvm::SmallVector<riscv::EncodedLocalPackOp> packs;
    getOperation().walk(
        [&](riscv::EncodedLocalPackOp operation) { packs.push_back(operation); });

    for (riscv::EncodedLocalPackOp operation : packs) {
      auto rowPoint = operation.getRowPoint().getDefiningOp<riscv::PhysicalPointOp>();
      auto recordPoint =
          operation.getRecordPoint().getDefiningOp<riscv::PhysicalPointOp>();
      riscv::LocalPackPlanAttr plan = operation.getPlan();
      if (!rowPoint || !recordPoint) {
        operation.emitError(
            "encoded local pack program requires explicit physical row and record points");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(operation);
      auto guard = riscv::IndexMultipleGuardOp::create(rewriter,
          operation.getLoc(), recordPoint.getActive(), plan.getRecordElements(),
          riscv_internal::leaf(rewriter, "scalar", "index-guard",
                               "scalar.index-multiple-guard",
                               "scalar.index-multiple-guard", 0, 0, 0));
      copyProvenance(operation, guard);

      mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter,
          operation.getLoc(), 0);
      mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter,
          operation.getLoc(), 1);
      mlir::Value rowsPerGroup =
          mlir::arith::ConstantIndexOp::create(rewriter,
              operation.getLoc(), plan.getInterleaveRows());
      mlir::Value elementsPerRecord =
          mlir::arith::ConstantIndexOp::create(rewriter,
              operation.getLoc(), plan.getRecordElements());
      mlir::Value bytesPerRecord =
          mlir::arith::ConstantIndexOp::create(rewriter,
              operation.getLoc(), plan.getRecordBytes());
      mlir::Value rowGroups = mlir::arith::CeilDivUIOp::create(rewriter,
          operation.getLoc(), rowPoint.getActive(), rowsPerGroup);
      mlir::Value blockCount = mlir::arith::DivUIOp::create(rewriter,
          operation.getLoc(), recordPoint.getActive(), elementsPerRecord);

      auto rowLoop = mlir::scf::ForOp::create(rewriter,
          operation.getLoc(), zero, rowGroups, one);
      rowLoop->setAttr("weft.riscv.direction",
                       rewriter.getStringAttr("ascending"));
      copyProvenance(operation, rowLoop);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value rowBase = mlir::arith::MulIOp::create(rewriter,
          operation.getLoc(), rowLoop.getInductionVar(), rowsPerGroup);
      mlir::Value rowsRemaining = mlir::arith::SubIOp::create(rewriter,
          operation.getLoc(), rowPoint.getActive(), rowBase);
      mlir::Value transferVL = mlir::arith::MinUIOp::create(rewriter,
          operation.getLoc(), rowsRemaining, rowsPerGroup);

      auto blockLoop = mlir::scf::ForOp::create(rewriter,
          operation.getLoc(), zero, blockCount, one);
      blockLoop->setAttr("weft.riscv.direction",
                         rewriter.getStringAttr("ascending"));
      copyProvenance(operation, blockLoop);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      auto byteLoop = mlir::scf::ForOp::create(rewriter,
          operation.getLoc(), zero, bytesPerRecord, one);
      byteLoop->setAttr("weft.riscv.direction",
                        rewriter.getStringAttr("ascending"));
      copyProvenance(operation, byteLoop);
      rewriter.setInsertionPointToStart(byteLoop.getBody());
      auto transfer =
          riscv::RVVEncodedLocalPackTransferOp::create(rewriter,
              operation.getLoc(), operation.getInput(), operation.getStorage(),
              rowLoop.getInductionVar(), rowBase, blockLoop.getInductionVar(),
              blockCount, byteLoop.getInductionVar(), transferVL, plan,
              operation.getTransferLayout(),
              riscv_internal::leaf(
                  rewriter, "transfer", "local-pack-transfer",
                  "rvv.local-pack.transfer.interleave",
                  "rvv.local-pack.transfer.interleave", 0, 0,
                  operation.getTransferLayout().getRegisterGroups()));
      copyProvenance(operation, transfer);

      rewriter.setInsertionPointAfter(rowLoop);
      auto binding = riscv::EncodedLocalBindOp::create(rewriter,
          operation.getLoc(), operation.getResult().getType(),
          operation.getInput(), operation.getStorage(), blockCount, plan);
      copyProvenance(operation, binding);
      operation.getResult().replaceAllUsesWith(binding.getResult());
      rewriter.eraseOp(operation);
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createMaterializeRISCVProgramsPass() {
  return std::make_unique<MaterializeRISCVProgramsPass>();
}
