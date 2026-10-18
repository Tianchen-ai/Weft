#ifndef WEFT_TARGET_RISCVPASSES_H
#define WEFT_TARGET_RISCVPASSES_H

#include "Weft/Target/RISCVCompiler.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace weft {
std::unique_ptr<mlir::Pass> createSelectRISCVScalarPacketsPass();
std::unique_ptr<mlir::Pass> createAlignRISCVLocalStoragePass();
std::unique_ptr<mlir::Pass> createSelectRISCVPanelTransfersPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVPanelTransfersPass();

std::unique_ptr<mlir::Pass>
createConvertWeftToRISCVPass(RISCVCompilerOptions options);
std::unique_ptr<mlir::Pass> createMaterializeRISCVReductionsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVAtomicsPass();
std::unique_ptr<mlir::Pass> createFinalizeRISCVHelpersPass();
std::unique_ptr<mlir::Pass> createSelectRISCVOperationsPass();
std::unique_ptr<mlir::Pass>
createPropagateRISCVLayoutsPass(int64_t lmulEighths);
std::unique_ptr<mlir::Pass> createPlanRISCVMemoryPass();
std::unique_ptr<mlir::Pass> createSpecializeRISCVPointerOffsetsPass();
std::unique_ptr<mlir::Pass> createPlanRISCVNestedMemoryPass();
std::unique_ptr<mlir::Pass> createSpecializeRISCVMemoryPass();
std::unique_ptr<mlir::Pass> createSelectRISCVDenseTransfersPass();
std::unique_ptr<mlir::Pass> createSelectRISCVFMAIssuesPass();
std::unique_ptr<mlir::Pass> createContractRISCVFloatingPointPass();
std::unique_ptr<mlir::Pass> createSimplifyRISCVControlPass();
std::unique_ptr<mlir::Pass> createSelectRISCVMemoryByteProjectionsPass();
std::unique_ptr<mlir::Pass> createCanonicalizeRISCVLayoutsPass();
std::unique_ptr<mlir::Pass> createPropagateRISCVConsumerLayoutsPass();
std::unique_ptr<mlir::Pass> createFuseRISCVBitplanesPass();
std::unique_ptr<mlir::Pass> createLowerRISCVCompositesPass(int64_t lmulEighths = 8);
std::unique_ptr<mlir::Pass> createPrepareRISCVCompositesPass();
std::unique_ptr<mlir::Pass> createSelectRISCVDenseContractionsPass(int64_t lmulEighths = 8);
std::unique_ptr<mlir::Pass> createMaterializeRISCVDenseFragmentLoadsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVLocalWindowsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVValueProgramsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVLocalProjectionsPass();
std::unique_ptr<mlir::Pass> createRetainRISCVFragmentAccumulatorsPass();
std::unique_ptr<mlir::Pass> createContractRISCVIntegerPointwisePass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVProgramsPass();
std::unique_ptr<mlir::Pass> createVectorizeRISCVRecordLoopsPass();
std::unique_ptr<mlir::Pass> createHoistRISCVLoopInvariantsPass();
std::unique_ptr<mlir::Pass> createScheduleRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createScheduleRISCVLocalSuppliesPass();
std::unique_ptr<mlir::Pass> createPipelineRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createUnrollRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createShareRISCVLayeredWindowsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVReplicaStorageLoadsPass();
std::unique_ptr<mlir::Pass> createPlanRISCVPartialTopologiesPass();
std::unique_ptr<mlir::Pass> createCoalesceRISCVPartialExtractionsPass();
std::unique_ptr<mlir::Pass>
createMaterializeRISCVPartialAccumulatorsPass();
std::unique_ptr<mlir::Pass> createFinalizeRISCVLeavesPass();
std::unique_ptr<mlir::Pass>
createSelectRISCVScalarLoadPrimesPass(bool enabled);
std::unique_ptr<mlir::Pass> createMaterializeRISCVReadSnapshotsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVResourcesPass();
std::unique_ptr<mlir::Pass> createCloseRISCVLeafResourcesPass();
std::unique_ptr<mlir::Pass> createFuseRISCVPhysicalIssueLoopsPass();
std::unique_ptr<mlir::Pass> createEliminateDeadRISCVLayoutsPass();
std::unique_ptr<mlir::Pass> createVerifyFinalRISCVPass();

} // namespace weft

#endif // WEFT_TARGET_RISCVPASSES_H
