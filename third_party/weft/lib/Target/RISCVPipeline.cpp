#include "Weft/Target/RISCVPipeline.h"

#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/SourceSpecialization.h"
#include "Weft/Dialect/RISCV/IR/PanelTransfer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <vector>

using namespace mlir;
using namespace weft;

namespace {

StringRef checkpointAttribute(RISCVCheckpoint checkpoint) {
  switch (checkpoint) {
  case RISCVCheckpoint::Contractions: return "weft.riscv.contraction_input";
  case RISCVCheckpoint::PanelTransfers: return "weft.riscv.panel_transfer_input";
  case RISCVCheckpoint::Layout: return "weft.riscv.layout_input";
  }
  llvm_unreachable("unknown physical checkpoint");
}

LogicalResult checkCheckpointBoundary(ModuleOp module, RISCVCheckpoint checkpoint) {
  bool foundKernel = false, closedResources = false, canonicalOperations = false;
  bool selectedContractions = false, panelCopies = false, unselectedPanels = false;
  module.walk([&](Operation *operation) {
    if (auto kernel = dyn_cast<riscv::KernelOp>(operation)) {
      foundKernel = true;
      closedResources |= kernel.getResourcesMaterialized();
    }
    canonicalOperations |= operation->getName().getDialectNamespace() == "weft_kernel";
    selectedContractions |= operation->hasAttr("weft.riscv.dense_contraction");
    if (auto copy = dyn_cast<riscv::DensePanelCopyOp>(operation)) {
      panelCopies = true;
      unselectedPanels |= !copy.getPlan();
    }
  });
  if (!foundKernel || closedResources || canonicalOperations)
    return module.emitError("pipeline input requires a physical program before final resource materialization");
  if (checkpoint != RISCVCheckpoint::Contractions && selectedContractions)
    return module.emitError("this input cannot retain an unmaterialized contraction selection");
  if (checkpoint == RISCVCheckpoint::Layout && panelCopies)
    return module.emitError("layout input cannot retain unmaterialized panel transfers");
  if (checkpoint == RISCVCheckpoint::PanelTransfers && unselectedPanels)
    return module.emitError("panel transfer input requires every copy to have a selected plan");
  return success();
}

template <RISCVCheckpoint boundary>
class MarkPipelineInputPass
    : public PassWrapper<MarkPipelineInputPass<boundary>, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkPipelineInputPass)
  MarkPipelineInputPass() = default;
  explicit MarkPipelineInputPass(const RISCVCompilerOptions &options) {
    scalarPrime = options.scalarLoadPrime != 0;
    lmul = boundary != RISCVCheckpoint::Layout ? options.lmulEighths : 0;
  }
  MarkPipelineInputPass(const MarkPipelineInputPass &other)
      : PassWrapper<MarkPipelineInputPass<boundary>, OperationPass<ModuleOp>>(other) {
    this->copyOptionValuesFrom(&other);
  }
  StringRef getArgument() const override {
    if (boundary == RISCVCheckpoint::PanelTransfers) return "weft-riscv-mark-panel-transfer-input";
    return boundary == RISCVCheckpoint::Contractions ? "weft-riscv-mark-contraction-input"
                                                     : "weft-riscv-mark-layout-input";
  }
  StringRef getDescription() const override { return "Seal a selected physical boundary and its remaining binding"; }
  void runOnOperation() override {
    if (failed(markRISCVPipelineInput(this->getOperation(), {boundary, lmul, scalarPrime})))
      this->signalPassFailure();
  }
private:
  typename Pass::template Option<bool> scalarPrime{*this, "scalar-load-prime", llvm::cl::init(false)};
  typename Pass::template Option<int64_t> lmul{*this, "lmul-eighths", llvm::cl::init(boundary != RISCVCheckpoint::Layout ? 8 : 0)};
};

template <RISCVCheckpoint boundary>
class ResumePipelineInputPass
    : public PassWrapper<ResumePipelineInputPass<boundary>, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResumePipelineInputPass)
  StringRef getArgument() const override {
    if (boundary == RISCVCheckpoint::PanelTransfers) return "weft-riscv-resume-panel-transfer-input";
    return boundary == RISCVCheckpoint::Contractions ? "weft-riscv-resume-contraction-input"
                                                     : "weft-riscv-resume-layout-input";
  }
  StringRef getDescription() const override { return "Consume a saved physical boundary and run the shared main suffix"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto binding = readRISCVPipelineInput(this->getOperation(), boundary);
    if (failed(binding)) { this->signalPassFailure(); return; }
    OpPassManager manager(ModuleOp::getOperationName());
    buildRISCVResumePipeline(manager, *binding);
    if (failed(this->runPipeline(manager, this->getOperation()))) this->signalPassFailure();
  }
};

template <RISCVCheckpoint boundary>
class ConsumePipelineInputPass
    : public PassWrapper<ConsumePipelineInputPass<boundary>, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConsumePipelineInputPass)
  ConsumePipelineInputPass() = default;
  explicit ConsumePipelineInputPass(const RISCVCompilerOptions &options) {
    scalarPrime = options.scalarLoadPrime != 0;
    lmul = boundary != RISCVCheckpoint::Layout ? options.lmulEighths : 0;
  }
  ConsumePipelineInputPass(const ConsumePipelineInputPass &other)
      : PassWrapper<ConsumePipelineInputPass<boundary>, OperationPass<ModuleOp>>(other) {
    this->copyOptionValuesFrom(&other);
  }
  StringRef getArgument() const override {
    if (boundary == RISCVCheckpoint::PanelTransfers) return "weft-riscv-consume-panel-transfer-input";
    return boundary == RISCVCheckpoint::Contractions ? "weft-riscv-consume-contraction-input"
                                                     : "weft-riscv-consume-layout-input";
  }
  StringRef getDescription() const override { return "Validate and consume the checkpoint's exact remaining binding"; }
  void runOnOperation() override {
    auto saved = readRISCVPipelineInput(this->getOperation(), boundary);
    if (failed(saved)) { this->signalPassFailure(); return; }
    if (saved->scalarLoadPrime != scalarPrime || saved->lmulEighths != lmul) {
      this->getOperation().emitError("pipeline replay cannot replace the checkpoint's remaining binding");
      this->signalPassFailure();
      return;
    }
    this->getOperation()->removeAttr(checkpointAttribute(boundary));
  }
private:
  typename Pass::template Option<bool> scalarPrime{*this, "scalar-load-prime", llvm::cl::init(false)};
  typename Pass::template Option<int64_t> lmul{*this, "lmul-eighths", llvm::cl::init(boundary != RISCVCheckpoint::Layout ? 8 : 0)};
};

enum class PassID {
  BindSource, Convert, PointerOffsets, Reductions, Atomics, Select, Layouts, LocalWindows, CanonicalizeLayouts, Memory,
  PrepareComposites, DenseContractions, Composites, SelectPanelTransfers, PanelTransfers,
  DenseFragments, Values, LocalProjections, Programs, RecordLoops, NestedMemory, Bitplanes,
  Hoist, Schedule, Pipeline, SCCP, Control, Fragments, ShareWindows,
  PartialTopology, PartialAccumulators, Unroll, PartialExtractions,
  DeadLayouts, ConsumerLayouts, ByteProjections, ReplicaLoads,
  SpecializeMemory, DenseTransfers, FMAIssues, PointwiseFMA, IntegerPointwise, ScalarPackets, Leaves, ScalarPrimes, Snapshots, LeafResources,
  LocalSupplies, CSE, FuseIssues, LocalAlignment, Resources, Verify, MarkLayout, ConsumeLayout, MarkContractions, ConsumeContractions,
  MarkPanelTransfers, ConsumePanelTransfers, Helpers,
};

struct Entry {
  PassID id;
  llvm::StringRef stage;
};

// This sequence is the executable main chain. Repeated passes are intentional:
// each repetition consumes physical operations exposed by the preceding owner.
constexpr Entry prefix[] = {
    {PassID::BindSource, "source-specialization"},
    {PassID::Convert, "canonical-to-physical"},
    {PassID::PointerOffsets, "pointer-addressing"},
    {PassID::Reductions, "reduction-materialization"},
    {PassID::Atomics, "atomic-materialization"},
    {PassID::Select, "representation"},
    {PassID::Layouts, "representation"},
    {PassID::LocalWindows, "representation"},
    {PassID::Select, "representation"},
    {PassID::CanonicalizeLayouts, "representation"},
    {PassID::Memory, "representation"},
    {PassID::CanonicalizeLayouts, "representation"},
    {PassID::Memory, "representation"},
    {PassID::Select, "representation"},
    {PassID::PrepareComposites, "program-materialization"},
    {PassID::DenseContractions, "program-materialization"},
    {PassID::Composites, "program-materialization"},
    {PassID::SelectPanelTransfers, "program-materialization"},
    {PassID::PanelTransfers, "program-materialization"},
    {PassID::DenseFragments, "program-materialization"},
    {PassID::Values, "program-materialization"},
    {PassID::Programs, "program-materialization"},
    {PassID::RecordLoops, "program-materialization"},
    {PassID::CanonicalizeLayouts, "program-materialization"},
    {PassID::NestedMemory, "program-materialization"},
    {PassID::Bitplanes, "program-materialization"},
    {PassID::Hoist, "scheduling"},
    {PassID::SCCP, "scheduling"},
    {PassID::Fragments, "scheduling"},
    {PassID::Schedule, "scheduling"},
    {PassID::Pipeline, "scheduling"},
    {PassID::SCCP, "scheduling"},
    {PassID::Control, "scheduling"},
    {PassID::ShareWindows, "partial-materialization"},
    {PassID::PartialTopology, "partial-materialization"},
    {PassID::PartialAccumulators, "partial-materialization"},
    {PassID::Unroll, "partial-materialization"},
    {PassID::ShareWindows, "partial-materialization"},
};

constexpr Entry suffix[] = {
    {PassID::PartialExtractions, "layout-finalization"},
    {PassID::LocalProjections, "layout-finalization"},
    {PassID::DeadLayouts, "layout-finalization"},
    {PassID::ConsumerLayouts, "layout-finalization"},
    {PassID::CanonicalizeLayouts, "layout-finalization"},
    {PassID::Memory, "layout-finalization"},
    {PassID::ByteProjections, "layout-finalization"},
    {PassID::ReplicaLoads, "layout-finalization"},
    {PassID::Hoist, "layout-finalization"},
    {PassID::Control, "layout-finalization"},
    {PassID::SpecializeMemory, "leaf-selection"},
    {PassID::Control, "leaf-selection"},
    {PassID::IntegerPointwise, "leaf-selection"},
    {PassID::Select, "leaf-selection"},
    {PassID::Leaves, "leaf-selection"},
    {PassID::LocalSupplies, "leaf-selection"},
    {PassID::DenseTransfers, "leaf-selection"},
    {PassID::ScalarPrimes, "leaf-selection"},
    {PassID::Snapshots, "resource-closure"},
    {PassID::LeafResources, "resource-closure"},
    {PassID::CSE, "resource-closure"},
    {PassID::Control, "resource-closure"},
    {PassID::FuseIssues, "resource-closure"},
    {PassID::PointwiseFMA, "resource-closure"},
    {PassID::ScalarPackets, "resource-closure"},
    {PassID::FMAIssues, "resource-closure"},
    {PassID::DeadLayouts, "resource-closure"},
    {PassID::LocalAlignment, "resource-closure"},
    {PassID::Resources, "resource-closure"},
    {PassID::DeadLayouts, "resource-closure"},
    {PassID::Helpers, "native-function-closure"},
    {PassID::Verify, "final-verification"},
};

constexpr Entry markLayout[] = {{PassID::MarkLayout, "layout-checkpoint"}};
constexpr Entry consumeLayout[] = {{PassID::ConsumeLayout, "layout-input"}};
constexpr Entry markContractions[] = {{PassID::MarkContractions, "contraction-checkpoint"}};
constexpr Entry consumeContractions[] = {{PassID::ConsumeContractions, "contraction-input"}};
constexpr Entry markPanelTransfers[] = {{PassID::MarkPanelTransfers, "panel-transfer-checkpoint"}};
constexpr Entry consumePanelTransfers[] = {{PassID::ConsumePanelTransfers, "panel-transfer-input"}};

llvm::ArrayRef<Entry> beforeCheckpoint(RISCVCheckpoint checkpoint) {
  auto entries = llvm::ArrayRef<Entry>(prefix);
  if (checkpoint == RISCVCheckpoint::Layout) return entries;
  auto materialization = llvm::find_if(entries, [checkpoint](const Entry &entry) {
    return entry.id == (checkpoint == RISCVCheckpoint::Contractions ? PassID::Composites : PassID::PanelTransfers);
  });
  return entries.take_front(materialization - entries.begin());
}

llvm::ArrayRef<Entry> afterCheckpoint(RISCVCheckpoint checkpoint) {
  return llvm::ArrayRef<Entry>(prefix).drop_front(beforeCheckpoint(checkpoint).size());
}

llvm::ArrayRef<Entry> checkpointMarker(RISCVCheckpoint checkpoint) {
  if (checkpoint == RISCVCheckpoint::PanelTransfers) return markPanelTransfers;
  return checkpoint == RISCVCheckpoint::Layout ? llvm::ArrayRef<Entry>(markLayout)
                                              : llvm::ArrayRef<Entry>(markContractions);
}

llvm::ArrayRef<Entry> checkpointConsumer(RISCVCheckpoint checkpoint) {
  if (checkpoint == RISCVCheckpoint::PanelTransfers) return consumePanelTransfers;
  return checkpoint == RISCVCheckpoint::Layout ? llvm::ArrayRef<Entry>(consumeLayout)
                                              : llvm::ArrayRef<Entry>(consumeContractions);
}

std::unique_ptr<Pass> makePass(PassID id, const RISCVCompilerOptions &options,
                               llvm::StringRef serializedTarget) {
  switch (id) {
  case PassID::BindSource:
    return createBindSourceParametersPass(options.metaBindings);
  case PassID::Convert: {
    RISCVCompilerOptions physicalOptions = options;
    physicalOptions.metaBindings.clear();
    return createConvertWeftToRISCVPass(serializedTarget, physicalOptions);
  }
  case PassID::Reductions: return createMaterializeRISCVReductionsPass();
  case PassID::Atomics: return createMaterializeRISCVAtomicsPass();
  case PassID::Helpers: return createFinalizeRISCVHelpersPass();
  case PassID::Select: return createSelectRISCVOperationsPass();
  case PassID::Layouts: return createPropagateRISCVLayoutsPass(options.lmulEighths);
  case PassID::LocalWindows: return createMaterializeRISCVLocalWindowsPass();
  case PassID::CanonicalizeLayouts: return createCanonicalizeRISCVLayoutsPass();
  case PassID::Memory: return createPlanRISCVMemoryPass();
  case PassID::PointerOffsets: return createSpecializeRISCVPointerOffsetsPass();
  case PassID::Composites: return createLowerRISCVCompositesPass(options.lmulEighths);
  case PassID::SelectPanelTransfers: return createSelectRISCVPanelTransfersPass();
  case PassID::PanelTransfers: return createMaterializeRISCVPanelTransfersPass();
  case PassID::PrepareComposites: return createPrepareRISCVCompositesPass();
  case PassID::DenseContractions: return createSelectRISCVDenseContractionsPass(options.lmulEighths);
  case PassID::DenseFragments: return createMaterializeRISCVDenseFragmentLoadsPass();
  case PassID::Values: return createMaterializeRISCVValueProgramsPass();
  case PassID::LocalProjections: return createMaterializeRISCVLocalProjectionsPass();
  case PassID::Programs: return createMaterializeRISCVProgramsPass();
  case PassID::RecordLoops: return createVectorizeRISCVRecordLoopsPass();
  case PassID::NestedMemory: return createPlanRISCVNestedMemoryPass();
  case PassID::Bitplanes: return createFuseRISCVBitplanesPass();
  case PassID::Hoist: return createHoistRISCVLoopInvariantsPass();
  case PassID::Schedule: return createScheduleRISCVLevelsPass();
  case PassID::Pipeline: return createPipelineRISCVLevelsPass();
  case PassID::SCCP: return createSCCPPass();
  case PassID::Control: return createSimplifyRISCVControlPass();
  case PassID::Fragments: return createRetainRISCVFragmentAccumulatorsPass();
  case PassID::ShareWindows: return createShareRISCVLayeredWindowsPass();
  case PassID::PartialTopology: return createPlanRISCVPartialTopologiesPass();
  case PassID::PartialAccumulators: return createMaterializeRISCVPartialAccumulatorsPass();
  case PassID::Unroll: return createUnrollRISCVLevelsPass();
  case PassID::PartialExtractions: return createCoalesceRISCVPartialExtractionsPass();
  case PassID::DeadLayouts: return createEliminateDeadRISCVLayoutsPass();
  case PassID::ConsumerLayouts: return createPropagateRISCVConsumerLayoutsPass();
  case PassID::ByteProjections: return createSelectRISCVMemoryByteProjectionsPass();
  case PassID::ReplicaLoads: return createMaterializeRISCVReplicaStorageLoadsPass();
  case PassID::SpecializeMemory: return createSpecializeRISCVMemoryPass();
  case PassID::DenseTransfers: return createSelectRISCVDenseTransfersPass();
  case PassID::FMAIssues: return createSelectRISCVFMAIssuesPass();
  case PassID::PointwiseFMA: return createContractRISCVFloatingPointPass();
  case PassID::LocalAlignment: return createAlignRISCVLocalStoragePass();
  case PassID::ScalarPackets: return createSelectRISCVScalarPacketsPass();
  case PassID::Leaves: return createFinalizeRISCVLeavesPass();
  case PassID::LocalSupplies: return createScheduleRISCVLocalSuppliesPass();
  case PassID::ScalarPrimes: return createSelectRISCVScalarLoadPrimesPass(options.scalarLoadPrime != 0);
  case PassID::IntegerPointwise: return createContractRISCVIntegerPointwisePass();
  case PassID::Snapshots: return createMaterializeRISCVReadSnapshotsPass();
  case PassID::LeafResources: return createCloseRISCVLeafResourcesPass();
  case PassID::CSE: return createCSEPass();
  case PassID::FuseIssues: return createFuseRISCVPhysicalIssueLoopsPass();
  case PassID::Resources: return createMaterializeRISCVResourcesPass();
  case PassID::Verify: return createVerifyFinalRISCVPass();
  case PassID::MarkLayout: return std::make_unique<MarkPipelineInputPass<RISCVCheckpoint::Layout>>(options);
  case PassID::ConsumeLayout: return std::make_unique<ConsumePipelineInputPass<RISCVCheckpoint::Layout>>(options);
  case PassID::MarkContractions: return std::make_unique<MarkPipelineInputPass<RISCVCheckpoint::Contractions>>(options);
  case PassID::ConsumeContractions: return std::make_unique<ConsumePipelineInputPass<RISCVCheckpoint::Contractions>>(options);
  case PassID::MarkPanelTransfers: return std::make_unique<MarkPipelineInputPass<RISCVCheckpoint::PanelTransfers>>(options);
  case PassID::ConsumePanelTransfers: return std::make_unique<ConsumePipelineInputPass<RISCVCheckpoint::PanelTransfers>>(options);
  }
  llvm_unreachable("unhandled RISC-V pipeline pass");
}

void append(OpPassManager &manager, llvm::ArrayRef<Entry> entries,
            const RISCVCompilerOptions &options,
            llvm::StringRef serializedTarget = {}) {
  for (const Entry &entry : entries)
    manager.addPass(makePass(entry.id, options, serializedTarget));
}

std::vector<RISCVPipelinePass> describe(llvm::ArrayRef<Entry> entries) {
  std::vector<RISCVPipelinePass> result;
  RISCVCompilerOptions options;
  for (const Entry &entry : entries)
    result.push_back({makePass(entry.id, options, {})->getArgument(), entry.stage});
  return result;
}

struct PhysicalPipelineOptions : PassPipelineOptions<PhysicalPipelineOptions> {
  Option<RISCVPipelineTarget> target{*this, "target",
      llvm::cl::desc("Complete #weft_riscv.target attribute")};
  ListOption<std::string> meta{*this, "meta",
      llvm::cl::desc("Source auto bindings NAME=INTEGER")};
  ListOption<std::string> requirements{*this, "require-extension",
      llvm::cl::desc("Required selected extension instructions")};
  Option<std::string> localTileStrategy{*this, "local-tile-strategy",
      llvm::cl::init("target-default")};
  Option<int64_t> lmul{*this, "lmul-eighths", llvm::cl::init(8)};
  Option<int64_t> unroll{*this, "unroll", llvm::cl::init(1)};
  Option<int64_t> pipeline{*this, "pipeline-depth", llvm::cl::init(1)};
  Option<int64_t> scalarPrime{*this, "scalar-load-prime", llvm::cl::init(0)};
  Option<int64_t> pointwiseCopies{*this, "pointwise-issue-copies", llvm::cl::init(1)};
  Option<int64_t> columnCopies{*this, "contraction-column-copies", llvm::cl::init(4)};
  Option<int64_t> transferFields{*this, "panel-transfer-fields", llvm::cl::init(4)};
  Option<std::string> transferStrategy{*this, "panel-transfer-strategy", llvm::cl::init("issue-work")};
  Option<int64_t> panelK{*this, "contraction-panel-k", llvm::cl::init(128)};
  Option<int64_t> kSteps{*this, "contraction-k-steps", llvm::cl::init(4)};

  LogicalResult bind(RISCVCompilerOptions &options,
                    function_ref<LogicalResult(const Twine &)> error) const {
    if (target.getValue().text.empty())
      return error("the RISC-V physical pipeline requires its complete target attribute");
    for (llvm::StringRef binding : meta) {
      auto [name, valueText] = binding.split('=');
      int64_t value = 0;
      if (name.empty() || valueText.getAsInteger(10, value) || value <= 0)
        return error("source meta binding must be NAME=POSITIVE_INTEGER");
      if (!options.metaBindings.try_emplace(name, value).second)
        return error("duplicate source meta binding: " + name);
    }
    options.requiredExtensions.assign(requirements.begin(), requirements.end());
    options.localTileStrategy = localTileStrategy;
    options.lmulEighths = lmul;
    options.unroll = unroll;
    options.pipelineDepth = pipeline;
    options.scalarLoadPrime = scalarPrime;
    options.pointwiseIssueCopies = pointwiseCopies;
    options.contractionColumnCopies = columnCopies;
    options.panelTransferFields = transferFields;
    options.panelTransferStrategy = transferStrategy;
    options.contractionPanelK = panelK;
    options.contractionKSteps = kSteps;
    std::string bindingError;
    if (!options.validateBindings(bindingError))
      return error(bindingError);
    return success();
  }
};

} // namespace

llvm::ArrayRef<RISCVPipelinePass>
weft::getRISCVPhysicalPipeline(bool stopBeforeLayout) {
  static const auto preparation = describe(prefix);
  static const auto complete = [] {
    auto result = describe(prefix);
    auto finalization = describe(suffix);
    result.insert(result.end(), finalization.begin(), finalization.end());
    return result;
  }();
  return stopBeforeLayout ? llvm::ArrayRef<RISCVPipelinePass>(preparation)
                          : llvm::ArrayRef<RISCVPipelinePass>(complete);
}

llvm::ArrayRef<RISCVPipelinePass> weft::getRISCVLayoutPipeline() {
  static const auto finalization = describe(suffix);
  return finalization;
}

llvm::ArrayRef<RISCVPipelinePass> weft::getRISCVPreparePipeline(RISCVCheckpoint checkpoint) {
  auto describeCheckpoint = [](RISCVCheckpoint boundary) {
    auto result = describe(beforeCheckpoint(boundary));
    auto marker = describe(checkpointMarker(boundary));
    result.insert(result.end(), marker.begin(), marker.end());
    return result;
  };
  static const auto layout = describeCheckpoint(RISCVCheckpoint::Layout);
  static const auto contractions = describeCheckpoint(RISCVCheckpoint::Contractions);
  static const auto panels = describeCheckpoint(RISCVCheckpoint::PanelTransfers);
  if (checkpoint == RISCVCheckpoint::PanelTransfers) return panels;
  return checkpoint == RISCVCheckpoint::Layout ? layout : contractions;
}

llvm::ArrayRef<RISCVPipelinePass> weft::getRISCVResumePipeline(RISCVCheckpoint checkpoint) {
  auto describeCheckpoint = [](RISCVCheckpoint boundary) {
    auto result = describe(checkpointConsumer(boundary));
    auto materialization = describe(afterCheckpoint(boundary));
    result.insert(result.end(), materialization.begin(), materialization.end());
    auto finalization = describe(suffix);
    result.insert(result.end(), finalization.begin(), finalization.end());
    return result;
  };
  static const auto layout = describeCheckpoint(RISCVCheckpoint::Layout);
  static const auto contractions = describeCheckpoint(RISCVCheckpoint::Contractions);
  static const auto panels = describeCheckpoint(RISCVCheckpoint::PanelTransfers);
  if (checkpoint == RISCVCheckpoint::PanelTransfers) return panels;
  return checkpoint == RISCVCheckpoint::Layout ? layout : contractions;
}

std::string weft::serializeRISCVPipelineTarget(const RISCVTargetProfile &target) {
  MLIRContext context;
  context.getOrLoadDialect<riscv::WEFTRISCVDialect>();
  Builder builder(&context);
  std::string result;
  llvm::raw_string_ostream output(result);
  mlir::Attribute(riscv_internal::target(builder, target)).print(output);
  return result;
}

void weft::buildRISCVPhysicalPipeline(OpPassManager &manager,
                                      const RISCVCompilerOptions &options,
                                      bool stopBeforeLayout) {
  append(manager, prefix, options, serializeRISCVPipelineTarget(options.target));
  if (!stopBeforeLayout)
    append(manager, suffix, options);
}

void weft::buildRISCVLayoutPipeline(OpPassManager &manager, bool scalarLoadPrime) {
  RISCVCompilerOptions options;
  options.scalarLoadPrime = scalarLoadPrime;
  append(manager, suffix, options);
}

void weft::buildRISCVPreparePipeline(OpPassManager &manager,
                                     const RISCVCompilerOptions &options, RISCVCheckpoint checkpoint) {
  append(manager, beforeCheckpoint(checkpoint), options, serializeRISCVPipelineTarget(options.target));
  append(manager, checkpointMarker(checkpoint), options);
}

void weft::buildRISCVResumePipeline(OpPassManager &manager, RISCVRemainingBinding binding) {
  RISCVCompilerOptions options;
  options.scalarLoadPrime = binding.scalarLoadPrime;
  options.lmulEighths = binding.lmulEighths;
  append(manager, checkpointConsumer(binding.checkpoint), options);
  append(manager, afterCheckpoint(binding.checkpoint), options);
  append(manager, suffix, options);
}

StringRef weft::checkpointName(RISCVCheckpoint checkpoint) {
  if (checkpoint == RISCVCheckpoint::PanelTransfers) return "panel-transfers";
  return checkpoint == RISCVCheckpoint::Layout ? "layout" : "contractions";
}

StringRef weft::checkpointEntry(RISCVCheckpoint checkpoint) {
  if (checkpoint == RISCVCheckpoint::PanelTransfers) return "panel-transfer-input";
  return checkpoint == RISCVCheckpoint::Layout ? "layout-input" : "contraction-input";
}

LogicalResult weft::markRISCVPipelineInput(ModuleOp module, RISCVRemainingBinding binding) {
  if (module->hasAttr(checkpointAttribute(RISCVCheckpoint::Layout)) ||
      module->hasAttr(checkpointAttribute(RISCVCheckpoint::PanelTransfers)) ||
      module->hasAttr(checkpointAttribute(RISCVCheckpoint::Contractions)))
    return module.emitError("pipeline input already carries a remaining binding");
  if (failed(checkCheckpointBoundary(module, binding.checkpoint))) return failure();
  Builder builder(module.getContext());
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("scalar_load_prime", builder.getBoolAttr(binding.scalarLoadPrime))};
  if (binding.checkpoint != RISCVCheckpoint::Layout)
    fields.push_back(builder.getNamedAttr("lmul_eighths", builder.getI64IntegerAttr(binding.lmulEighths)));
  else if (binding.lmulEighths != 0)
    return module.emitError("layout checkpoint cannot retain a consumed contraction LMUL binding");
  module->setAttr(checkpointAttribute(binding.checkpoint), builder.getDictionaryAttr(fields));
  return success(succeeded(readRISCVPipelineInput(module, binding.checkpoint)));
}

FailureOr<RISCVRemainingBinding> weft::readRISCVPipelineInput(ModuleOp module, RISCVCheckpoint checkpoint) {
  auto saved = module->getAttrOfType<DictionaryAttr>(checkpointAttribute(checkpoint));
  auto scalarPrime = saved ? saved.getAs<BoolAttr>("scalar_load_prime") : BoolAttr();
  auto lmul = saved ? saved.getAs<IntegerAttr>("lmul_eighths") : IntegerAttr();
  bool contractions = checkpoint == RISCVCheckpoint::Contractions;
  bool retainsLMUL = checkpoint != RISCVCheckpoint::Layout;
  bool other = false;
  for (auto boundary : {RISCVCheckpoint::Layout, RISCVCheckpoint::PanelTransfers, RISCVCheckpoint::Contractions})
    if (boundary != checkpoint) other |= module->hasAttr(checkpointAttribute(boundary));
  if (!saved || saved.size() != (retainsLMUL ? 2u : 1u) || !scalarPrime ||
      (retainsLMUL && (!lmul || !lmul.getType().isInteger(64))) ||
      other) {
    module.emitError("pipeline input requires the matching checkpoint and complete remaining binding");
    return failure();
  }
  if (failed(checkCheckpointBoundary(module, checkpoint))) return failure();
  bool legal = true;
  if (retainsLMUL) {
    module.walk([&](riscv::KernelOp kernel) {
      legal &= llvm::is_contained(kernel.getTarget().getLegalLMULEighths().asArrayRef(), lmul.getInt());
    });
    module.walk([&](Operation *operation) {
      if (auto plan = operation->getAttrOfType<riscv::ContractionPlanAttr>("weft.riscv.dense_contraction"))
        legal &= plan.getMaximumLMUL() == lmul.getInt();
    });
    module.walk([&](riscv::DensePanelCopyOp operation) { legal &= operation.getMaximumLmul() == lmul.getInt(); });
  }
  if (!legal) {
    module.emitError("pipeline input binding disagrees with its selected representations or target");
    return failure();
  }
  return RISCVRemainingBinding{checkpoint, retainsLMUL ? lmul.getInt() : 0, scalarPrime.getValue()};
}

void weft::registerRISCVPasses() {
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::StringSet<> registered;
    auto registerEntries = [&](llvm::ArrayRef<Entry> entries) {
      for (const Entry &entry : entries) {
        if (entry.id == PassID::CSE || entry.id == PassID::SCCP)
          continue;
        auto argument = makePass(entry.id, RISCVCompilerOptions{}, {})->getArgument();
        if (!registered.insert(argument).second)
          continue;
        registerPass([id = entry.id] {
          return makePass(id, RISCVCompilerOptions{}, {});
        });
      }
    };
    registerEntries(prefix);
    registerEntries(suffix);
    registerEntries(markLayout);
    registerEntries(consumeLayout);
    registerEntries(markContractions);
    registerEntries(consumeContractions);
    registerEntries(markPanelTransfers);
    registerEntries(consumePanelTransfers);
    registerPass([] { return std::make_unique<ResumePipelineInputPass<RISCVCheckpoint::Layout>>(); });
    registerPass([] { return std::make_unique<ResumePipelineInputPass<RISCVCheckpoint::Contractions>>(); });
    registerPass([] { return std::make_unique<ResumePipelineInputPass<RISCVCheckpoint::PanelTransfers>>(); });
  });
}

void weft::registerRISCVPipelines() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto registerPhysical = [](llvm::StringRef name, llvm::StringRef description,
                               std::optional<RISCVCheckpoint> checkpoint) {
      registerPassPipeline(name, description,
        [checkpoint](OpPassManager &manager, llvm::StringRef text,
           function_ref<LogicalResult(const Twine &)> error) {
          PhysicalPipelineOptions parsed;
          std::string diagnostics;
          llvm::raw_string_ostream output(diagnostics);
          if (failed(parsed.parseFromString(text, output)))
            return error(diagnostics);
          RISCVCompilerOptions options;
          if (failed(parsed.bind(options, error)))
            return failure();
          append(manager, checkpoint ? beforeCheckpoint(*checkpoint) : llvm::ArrayRef<Entry>(prefix),
                 options, parsed.target.getValue().text);
          if (checkpoint)
            append(manager, checkpointMarker(*checkpoint), options);
          else
            append(manager, suffix, options);
          return success();
        },
        [](function_ref<void(const detail::PassOptions &)> handler) {
          handler(PhysicalPipelineOptions{});
        });
    };
    registerPhysical("weft-riscv-physical",
        "Compile a bound canonical program through the complete RISC-V main chain", std::nullopt);
    registerPhysical("weft-riscv-prepare-layout",
        "Compile to the layout checkpoint with its saved remaining binding", RISCVCheckpoint::Layout);
    registerPhysical("weft-riscv-prepare-contractions",
        "Compile to selected contractions before materializing their loops and memory", RISCVCheckpoint::Contractions);
    registerPhysical("weft-riscv-prepare-panel-transfers",
        "Compile to selected panel copies before expanding their physical issue loops", RISCVCheckpoint::PanelTransfers);
    PassPipelineRegistration<>(
        "weft-riscv-finalize-layout", "Consume the existing layout checkpoint and run the shared finalization suffix",
        [](OpPassManager &manager) {
          manager.addPass(std::make_unique<ResumePipelineInputPass<RISCVCheckpoint::Layout>>());
        });
    PassPipelineRegistration<>(
        "weft-riscv-finalize-contractions", "Consume the contraction checkpoint through the shared main suffix",
        [](OpPassManager &manager) {
          manager.addPass(std::make_unique<ResumePipelineInputPass<RISCVCheckpoint::Contractions>>());
        });
    PassPipelineRegistration<>(
        "weft-riscv-finalize-panel-transfers", "Consume selected panel copies through the shared main suffix",
        [](OpPassManager &manager) {
          manager.addPass(std::make_unique<ResumePipelineInputPass<RISCVCheckpoint::PanelTransfers>>());
        });
  });
}
