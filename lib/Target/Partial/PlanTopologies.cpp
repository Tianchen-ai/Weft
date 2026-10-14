#include "Support.h"

namespace weft::riscv_partial {

static std::optional<int64_t> widenedIndependentResources(
    riscv::TargetAttr target, riscv::ValueType source,
    riscv::ValueType widened, int64_t slots, int64_t operandGroups) {
  if (!target || !source || !widened || slots < 2 ||
      (slots & (slots - 1)) || operandGroups < 0 ||
      !target.getHasWideningInteger() ||
      !riscv::supportsRVVLayout(target, source.getLayout()) ||
      !riscv::supportsRVVLayout(target, widened.getLayout()) ||
      widened.getLayout().getRegisterGroups() >
          target.getMaxWideningCombineGroups())
    return std::nullopt;
  const int64_t wideGroups = widened.getLayout().getRegisterGroups();
  auto sourceGroups = checkedProduct({slots, source.getLayout().getRegisterGroups()});
  auto firstGroups = checkedProduct({slots / 2, wideGroups});
  const int64_t budget = target.getVectorRegisters();
  if (!sourceGroups || !firstGroups || budget < 2 ||
      *sourceGroups > budget || *firstGroups > budget ||
      operandGroups > budget - *sourceGroups ||
      *sourceGroups > budget - *firstGroups || wideGroups > budget - 2)
    return std::nullopt;
  // First widening combines dominate later i32 pairwise live sets: the number
  // of slots halves at every subsequent stage while the carrier stays fixed.
  return std::max({operandGroups + *sourceGroups,
                   *sourceGroups + *firstGroups, wideGroups + 2});
}

class PlanRISCVPartialTopologiesPass
    : public mlir::PassWrapper<PlanRISCVPartialTopologiesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-plan-partial-topologies";
  }

  llvm::StringRef getDescription() const override {
    return "Select one typed widened-partial topology from operands, uses, and target resources";
  }

  void runOnOperation() override {
    mlir::Builder builder(&getContext());
    llvm::DenseMap<mlir::Operation *, ScaledTopologyFacts> scaledFacts;
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> scaledReductions;
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> replicaReductions;
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> nestedScaledReductions;
    llvm::DenseSet<mlir::Operation *> levelScaledDots;
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp reduce) { reductions.push_back(reduce); });
    for (riscv::ReduceOp reduce : reductions) {
      auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      auto dot = reduce.getInput().getDefiningOp<riscv::RVVWidenDotOp>();
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      if (reduce.getKind() == "add" && input && result && dot &&
          dot.getResult().hasOneUse() && dot.getOver().size() == 1 &&
          reduce.getAxis() >= 0 &&
          static_cast<size_t>(reduce.getAxis()) < input.getAxisIds().size() &&
          input.getLayout().getCarrier() == "scalar" &&
          result.getLayout().getCarrier() == "scalar" && resultElement &&
          resultElement.isSigned() && resultElement.getWidth() == 32) {
        const size_t position = static_cast<size_t>(reduce.getAxis());
        const int64_t reducedAxis = input.getAxisIds()[position];
        const int64_t replicas =
            input.getLayout().getReplicaFactors()[position];
        if (reducedAxis != dot.getOver()[0] && replicas > 1 &&
            input.getLayout().getTimeFactors()[position] == 1 &&
            input.getLayout().getLaneFactors()[position] == 1 &&
            input.getLayout().getFragmentFactors()[position] == 1 &&
            input.getLayout().getLocalFactors()[position] == 1)
          replicaReductions[dot.getOperation()] = reduce;
      }
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match)
        continue;
      if (match->dot.getOver().size() > 1) {
        auto found = nestedScaledReductions.find(match->dot.getOperation());
        auto previous =
            found == nestedScaledReductions.end()
                ? std::optional<ReplicaScaledDotReduction>()
                : matchReplicaScaledDotReduction(found->second);
        if (!match->reducedAxes.empty() &&
            (!previous || previous->reducedAxes.size() <
                              match->reducedAxes.size()))
          nestedScaledReductions[match->dot.getOperation()] = reduce;
        continue;
      }
      auto dotResultType =
          mlir::dyn_cast<riscv::ValueType>(match->dot.getResult().getType());
      if (!dotResultType)
        continue;
      ScaledTopologyFacts facts;
      facts.partialAxes = match->reducedAxes;
      llvm::sort(facts.partialAxes);
      facts.partialAxes.erase(
          std::unique(facts.partialAxes.begin(), facts.partialAxes.end()),
          facts.partialAxes.end());
      facts.outputAxes = remainingAxes(dotResultType, facts.partialAxes);
      facts.partialSlots =
          replicaProductForAxes(dotResultType, facts.partialAxes);
      facts.outputReplicas =
          replicaProductForAxes(dotResultType, facts.outputAxes);
      if (facts.partialSlots <= 1 || facts.outputReplicas <= 0)
        continue;
      auto found = scaledFacts.find(match->dot.getOperation());
      if (found == scaledFacts.end() ||
          found->second.partialAxes.size() < facts.partialAxes.size())
        scaledFacts[match->dot.getOperation()] = std::move(facts);
      scaledReductions[match->dot.getOperation()] = reduce;
    }
    getOperation().walk([&](mlir::scf::ForOp loop) {
      auto contributions = matchLevelScaledLoop(loop);
      if (contributions) {
        for (ScaledPartialContribution &contribution : *contributions)
          levelScaledDots.insert(contribution.dot.getOperation());
        return;
      }
      if (auto seed = matchLevelScaledLoopSeed(loop))
        levelScaledDots.insert(seed->dot.getOperation());
    });

    llvm::SmallVector<riscv::RVVWidenDotOp> dots;
    getOperation().walk(
        [&](riscv::RVVWidenDotOp dot) { dots.push_back(dot); });
    for (riscv::RVVWidenDotOp dot : dots) {
      dot->removeAttr("partial_layout_plan");
      dot->removeAttr("nested_partial_plan");
      dot->removeAttr("sequential_partial_plan");
      dot->removeAttr("scaled_partial_plan");
      dot->removeAttr("layered_partial_plan");
      dot->removeAttr("partial_combine_plan");
      auto result = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      llvm::SmallVector<int64_t> outputAxes =
          result ? llvm::SmallVector<int64_t>(result.getAxisIds().asArrayRef())
                 : llvm::SmallVector<int64_t>();
      const int64_t outputReplicas = result ? replicaProduct(result) : 1;
      const int64_t partialGroups =
          std::max<int64_t>(1, (2 * dot.getSliceLmulEighths() + 7) / 8);
      const int64_t streams = dot.getReductionStreams();
      const int64_t operandGroups =
          dot.getLhs().getType().getLayout().getRegisterGroups() +
          dot.getRhs().getType().getLayout().getRegisterGroups();
      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      const int64_t available =
          kernel ? kernel.getTarget().getVectorRegisters() - operandGroups : -1;
      const bool fused = dot.getFusedStreamsLegal();

      llvm::StringRef kind;
      int64_t rootOperand = -1;
      llvm::SmallVector<int64_t> partialAxes(dot.getOver());
      int64_t sourceSlots = streams;
      int64_t partialSlots = fused ? 1 : streams;
      int64_t replicas = outputReplicas;
      int64_t laneSplit = 1;
      int64_t combineArity = fused ? streams : 1;
      int64_t resources = std::max<int64_t>(1, partialSlots * partialGroups + 2);
      std::optional<SourceLanePlan> sourcePlan;
      std::optional<riscv::NestedPartialPlanAttr> nestedPlan;
      std::optional<LayeredTopologyFacts> layeredPlan;
      riscv::ValueType replicaAccumulator;
      riscv::ValueType sequentialLhsIssue;
      riscv::ValueType sequentialRhsIssue;
      riscv::ValueType sequentialAccumulator;
      std::optional<std::string> sequentialMultiply;
      llvm::StringRef sequentialLhsSupply = "slice";
      llvm::StringRef sequentialRhsSupply = "slice";
      bool sequentialPlanClosed = false;

      if (auto found = nestedScaledReductions.find(dot.getOperation());
          found != nestedScaledReductions.end()) {
        auto match = matchReplicaScaledDotReduction(found->second);
        int64_t windowAxis = 0;
        if (match)
          for (int64_t axis : match->reducedAxes) {
            auto candidateLhs = axisPosition(dot.getLhs().getType(), axis);
            auto candidateRhs = axisPosition(dot.getRhs().getType(), axis);
            auto candidateResult = result ? axisPosition(result, axis)
                                          : std::optional<size_t>();
            if (!candidateLhs || !candidateRhs || !candidateResult)
              continue;
            const int64_t lhsAxisTime =
                dot.getLhs().getType().getLayout().getTimeFactors()[*candidateLhs];
            const int64_t rhsAxisTime =
                dot.getRhs().getType().getLayout().getTimeFactors()[*candidateRhs];
            const int64_t lhsAxisLane =
                dot.getLhs().getType().getLayout().getLaneFactors()[*candidateLhs];
            const int64_t rhsAxisLane =
                dot.getRhs().getType().getLayout().getLaneFactors()[*candidateRhs];
            const int64_t axisExtent = result.getShape()[*candidateResult];
            if (lhsAxisTime <= 0 || lhsAxisTime != rhsAxisTime ||
                lhsAxisLane <= 0 || lhsAxisLane != rhsAxisLane ||
                axisExtent != lhsAxisTime * lhsAxisLane)
              continue;
            if (!windowAxis || lhsAxisTime > 1) {
              windowAxis = axis;
              if (lhsAxisTime > 1)
                break;
            }
          }
        auto lhsPosition = axisPosition(dot.getLhs().getType(), windowAxis);
        auto rhsPosition = axisPosition(dot.getRhs().getType(), windowAxis);
        auto resultPosition = result ? axisPosition(result, windowAxis)
                                     : std::optional<size_t>();
        const int64_t lhsTime = lhsPosition
                                    ? dot.getLhs()
                                          .getType()
                                          .getLayout()
                                          .getTimeFactors()[*lhsPosition]
                                    : 0;
        const int64_t rhsTime = rhsPosition
                                    ? dot.getRhs()
                                          .getType()
                                          .getLayout()
                                          .getTimeFactors()[*rhsPosition]
                                    : 0;
        const int64_t lhsWindow = lhsPosition
                                      ? dot.getLhs()
                                            .getType()
                                            .getLayout()
                                            .getLaneFactors()[*lhsPosition]
                                      : 0;
        const int64_t rhsWindow = rhsPosition
                                      ? dot.getRhs()
                                            .getType()
                                            .getLayout()
                                            .getLaneFactors()[*rhsPosition]
                                      : 0;
        const int64_t logicalExtent =
            resultPosition ? result.getShape()[*resultPosition] : 0;
        mlir::Type finalType = found->second.getResult().getType();
        auto finalElement = mlir::dyn_cast<mlir::IntegerType>(
            riscv_internal::logicalElement(finalType));
        auto finalValue = mlir::dyn_cast<riscv::ValueType>(finalType);
        const int64_t finalReplicaCount =
            finalValue
                ? product(finalValue.getLayout().getReplicaFactors())
                : 1;
        const bool closedScalarFinal =
            !finalValue ||
            (finalValue.getLayout().getCarrier() == "scalar" &&
             product(finalValue.getLayout().getTimeFactors()) == 1 &&
             product(finalValue.getLayout().getLaneFactors()) == 1 &&
             (finalReplicaCount == 1 ||
              (finalReplicaCount > 1 &&
               !reachesReductionThroughPureOps(found->second.getResult()))) &&
             product(finalValue.getLayout().getFragmentFactors()) == 1 &&
             product(finalValue.getLayout().getLocalFactors()) == 1);
        if (match && windowAxis > 0 && lhsPosition && rhsPosition &&
            resultPosition && lhsTime >= 1 && lhsTime == rhsTime &&
            lhsWindow >= 1 && lhsWindow == rhsWindow && logicalExtent > 1 &&
            logicalExtent == lhsTime * lhsWindow && finalElement &&
            finalElement.isSigned() && finalElement.getWidth() == 32 &&
            closedScalarFinal && dot.getPartialUnroll() > 0) {
          const bool issueOnlyWindow =
              hasGappedIndexedWindowRoot(dot.getLhs(), windowAxis) ||
              hasGappedIndexedWindowRoot(dot.getRhs(), windowAxis);
          int64_t plannedStreams = 0;
          int64_t plannedWindow = issueOnlyWindow ? 1 : lhsWindow;
          // The widest input representation can exceed the full-product or
          // reduced-slot budget even when smaller issue windows are legal.
          // Prefer fewer issues; only consider exact power-of-two partitions
          // of this same physical axis before freezing one complete plan.
          // A positive i64 window admits at most 63 halvings.
          int64_t candidates = 0;
          for (; candidates < 64; ++candidates) {
            plannedStreams = logicalExtent / plannedWindow;
            nestedPlan = planNestedPartialCarrier(
                builder, dot, *match, windowAxis, plannedStreams, plannedWindow);
            if (nestedPlan || plannedWindow <= 1 || plannedWindow % 2)
              break;
            plannedWindow /= 2;
          }
          if (!nestedPlan) {
            dot.emitError(
                "nested scaled contraction has no legal full-product carrier under the target resource contract; lhs=")
                << dot.getLhs().getType() << ", rhs=" << dot.getRhs().getType()
                << ", scale=" << match->scale.getType()
                 << ", window_axis=" << windowAxis << ", issue_streams="
                << lhsTime << ", window_extent=" << lhsWindow
                << ", reduction_lanes=" << dot.getReductionLanes()
                << ", reduction_streams=" << dot.getReductionStreams()
                << ", issue_window_candidates=" << candidates + 1
                << ", final_type=" << found->second.getResult().getType();
            signalPassFailure();
            return;
          }
          kind = "nested_scaled_stream";
          partialAxes.assign((*nestedPlan).getPartialAxes().asArrayRef().begin(),
                             (*nestedPlan).getPartialAxes().asArrayRef().end());
          outputAxes.assign((*nestedPlan).getOutputAxes().asArrayRef().begin(),
                            (*nestedPlan).getOutputAxes().asArrayRef().end());
          sourceSlots = plannedStreams;
          partialSlots =
              plannedStreams * (*nestedPlan).getPartialSlots();
          replicas = (*nestedPlan).getOutputReplicas();
          laneSplit = (*nestedPlan).getPartialSlots();
          combineArity = (*nestedPlan).getPartialSlots();
          resources = (*nestedPlan).getResourceGroups();
        }
      }

      if (kind.empty())
        if (auto found = replicaReductions.find(dot.getOperation());
          found != replicaReductions.end()) {
        riscv::ReduceOp reduce = found->second;
        auto input = mlir::cast<riscv::ValueType>(reduce.getInput().getType());
        auto reduced = mlir::cast<riscv::ValueType>(reduce.getResult().getType());
        const size_t position = static_cast<size_t>(reduce.getAxis());
        const int64_t reducedAxis = input.getAxisIds()[position];
        const int64_t reducedReplicas =
            input.getLayout().getReplicaFactors()[position];
        const int64_t survivingReplicas = replicaProduct(reduced);
        auto projectedLhs = projectReplicaReductionOperandType(
            builder, dot.getLhs().getType(), reducedAxis);
        auto projectedRhs = projectReplicaReductionOperandType(
            builder, dot.getRhs().getType(), reducedAxis);
        replicaAccumulator =
            projectedLhs && projectedRhs
                ? replicaReducedAccumulatorType(builder, projectedLhs,
                                                projectedRhs, dot.getOver()[0])
                : riscv::ValueType();
        const int64_t accumulatorGroups =
            replicaAccumulator
                ? replicaAccumulator.getLayout().getRegisterGroups()
                : 0;
        if (reducedReplicas > 1 && survivingReplicas > 0 && kernel &&
            replicaAccumulator && accumulatorGroups <= available &&
            riscv::supportsRVVLayout(kernel.getTarget(),
                                     replicaAccumulator.getLayout())) {
          kind = "replica_reduced";
          partialAxes.assign(1, reducedAxis);
          outputAxes.assign(reduced.getAxisIds().asArrayRef().begin(),
                            reduced.getAxisIds().asArrayRef().end());
          sourceSlots = reducedReplicas;
          partialSlots = reducedReplicas;
          replicas = survivingReplicas;
          combineArity = reducedReplicas;
          resources = std::max<int64_t>(1, operandGroups + accumulatorGroups + 2);
        }
        }
      if (kind.empty() && fused)
        if (auto layered = analyzeLayeredTopology(builder, dot)) {
          int64_t rootGroups = 0;
          for (const ProjectedRoot &root : layered->roots)
            rootGroups +=
                layered->windowTypes.lookup(root.value)
                    .getLayout()
                    .getRegisterGroups();
          const int64_t layeredResources =
              layered->accumulatorType.getLayout().getRegisterGroups() +
              layered->layeredWindowType.getLayout().getRegisterGroups() +
              rootGroups + 1;
          if (kernel &&
              layeredResources <= kernel.getTarget().getVectorRegisters()) {
            kind = "layered";
            rootOperand = layered->rootOperand;
            partialSlots = layered->streams;
            combineArity = layered->streams;
            resources = layeredResources;
            layeredPlan = std::move(*layered);
          }
        }
      if (kind.empty())
        if (auto found = scaledFacts.find(dot.getOperation());
            found != scaledFacts.end()) {
          const unsigned operandBits =
              riscv_internal::logicalBitWidth(dot.getLhs().getType().getElementType());
          const bool oneScaleGroupFillsVector =
              kernel && operandBits && dot.getReductionLanes() > 0 &&
              dot.getReductionLanes() * static_cast<int64_t>(operandBits) >=
                  kernel.getTarget().getVlenBits();
          // If one scale group already fills a target vector, reducing before
          // scaling avoids widening a full lane set only to immediately reduce
          // it.  Otherwise the physical partial stays in lanes, is widened and
          // scaled there, and is reduced only after the selected combine tree.
          // Scale dominance across a source Level is a legality fact, not a
          // reason to force either topology.
          const bool reduceBeforeScale = oneScaleGroupFillsVector;
          kind = reduceBeforeScale ? "reduced_scaled" : "scaled";
          partialAxes = found->second.partialAxes;
          outputAxes = found->second.outputAxes;
          partialSlots = found->second.partialSlots;
          replicas = found->second.outputReplicas;
          combineArity = partialSlots;
          resources =
              std::max<int64_t>(1, partialSlots * partialGroups + 2);
          if (auto source = planScaledSourceLanes(builder, dot, found->second)) {
            if (!reduceBeforeScale)
              kind = "scaled";
            sourceSlots = source->sourceSlots;
            laneSplit = source->laneSplit;
            // One wide source partial owns every lane slice produced by its
            // repack.  Keep those slices in one scale-combine chain instead
            // of rebuilding one scalar chain per reduced slot.
            combineArity = sourceSlots;
            sourcePlan = std::move(*source);
          }
        }
      auto levelScaledPartialGroups = checkedProduct(
          {outputReplicas, static_cast<int64_t>(dot.getPartialUnroll()),
           partialGroups});
      const int64_t levelScaledResources =
          levelScaledPartialGroups &&
                  *levelScaledPartialGroups <=
                      std::numeric_limits<int64_t>::max() - 2
              ? *levelScaledPartialGroups + 2
              : 0;
      if (kind.empty() && kernel && levelScaledResources > 0 &&
          levelScaledResources <= kernel.getTarget().getVectorRegisters() &&
          levelScaledDots.contains(dot.getOperation())) {
        kind = "level_scaled";
        partialSlots = dot.getPartialUnroll();
        combineArity = partialSlots % 2 == 0 ? 2 : partialSlots;
        resources = levelScaledResources;
      }
      auto topologyResultElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(dot.getResult().getType()));
      auto pairSlot = partialSlotType(builder, dot);
      auto widenedPairSlot =
          pairSlot ? widenPartialSlotType(builder, pairSlot)
                   : riscv::ValueType();
      auto wideningResources =
          kernel && pairSlot && widenedPairSlot
              ? widenedIndependentResources(kernel.getTarget(), pairSlot,
                                             widenedPairSlot, streams,
                                             operandGroups)
              : std::optional<int64_t>();
      const bool exactWideningTree =
          !fused && wideningResources &&
          topologyResultElement && topologyResultElement.isSigned() &&
          topologyResultElement.getWidth() == 32;
      const bool needsWidenedCombine = !fused && static_cast<bool>(widenedPairSlot);

      // Independent partials are a structural preference supplied by the
      // target profile. If narrow fusion is not exact, every i16 tree widens
      // its first pairwise stage, regardless of its number of streams. The
      // target carrier limit and the full first-stage live set gate selection.
      if (kind.empty() && kernel &&
          kernel.getTarget().getPartialCombinePolicy() ==
              "independent-multilevel" &&
          (exactWideningTree ||
           (!needsWidenedCombine && streams >= 4 && (streams & (streams - 1)) == 0 &&
            available >= 2 && partialGroups > 0 &&
            streams <= (available - 2) / partialGroups))) {
        kind = "independent";
        partialSlots = streams;
        combineArity = 2;
        resources = exactWideningTree
                        ? *wideningResources
                        : std::max<int64_t>(1, streams * partialGroups + 2);
      }
      if (kind.empty())
        kind = fused ? "sequential_fused" : "sequential_per_stream";

      // A sequential contraction is only real when its complete issue slice,
      // accumulator carrier, and simultaneous resource contract are selected
      // before materialization.  Fused and per-stream realizations share the
      // same carrier; they differ only in whether the carrier or the finalized
      // i32 value crosses an issue boundary.
      if (kind == "sequential_fused" || kind == "sequential_per_stream" ||
          kind == "level_scaled") {
        sequentialLhsIssue =
            sequentialIssueType(builder, dot, dot.getLhs().getType());
        sequentialRhsIssue =
            sequentialIssueType(builder, dot, dot.getRhs().getType());
        sequentialAccumulator = partialSlotType(builder, dot);
        sequentialMultiply = sequentialMultiplyInstruction(
            sequentialLhsIssue, sequentialRhsIssue);
        if (kind == "level_scaled" && kernel && dot.getOver().size() == 1 &&
            sequentialLhsIssue && sequentialRhsIssue) {
          const int64_t reductionAxis = dot.getOver()[0];
          auto lhsPosition = axisPosition(sequentialLhsIssue, reductionAxis);
          auto rhsPosition = axisPosition(sequentialRhsIssue, reductionAxis);
          if (replicaProduct(dot.getLhs().getType()) == 1 && lhsPosition &&
              supportsIssueStorageRematerialization(
                  builder, dot.getLhs(), reductionAxis,
                  sequentialLhsIssue.getShape()[*lhsPosition],
                  kernel.getTarget()))
            sequentialLhsSupply = "storage-rematerialize";
          if (replicaProduct(dot.getRhs().getType()) == 1 && rhsPosition &&
              supportsIssueStorageRematerialization(
                  builder, dot.getRhs(), reductionAxis,
                  sequentialRhsIssue.getShape()[*rhsPosition],
                  kernel.getTarget()))
            sequentialRhsSupply = "storage-rematerialize";
        }
        auto lhsParts = riscv_internal::staticProduct(
            sequentialLhsIssue
                ? sequentialLhsIssue.getLayout().getReplicaFactors().asArrayRef()
                : llvm::ArrayRef<int64_t>());
        auto rhsParts = riscv_internal::staticProduct(
            sequentialRhsIssue
                ? sequentialRhsIssue.getLayout().getReplicaFactors().asArrayRef()
                : llvm::ArrayRef<int64_t>());
        const int64_t plannedResources =
            sequentialLhsIssue && sequentialRhsIssue && sequentialAccumulator
                ? std::max<int64_t>(
                      sequentialLhsIssue.getLayout().getRegisterGroups() +
                          sequentialRhsIssue.getLayout().getRegisterGroups() +
                          sequentialAccumulator.getLayout().getRegisterGroups(),
                      sequentialAccumulator.getLayout().getRegisterGroups() + 2)
                : 0;
        sequentialPlanClosed =
            sequentialLhsIssue && sequentialRhsIssue && sequentialAccumulator &&
            sequentialMultiply &&
            lhsParts && rhsParts && *lhsParts == 1 && *rhsParts == 1 &&
            outputReplicas > 0 &&
            dot.getLhsParts().size() ==
                static_cast<size_t>(streams * outputReplicas) &&
            dot.getRhsParts().size() ==
                static_cast<size_t>(streams * outputReplicas) &&
            dot.getLhsLaneOffsets().size() ==
                static_cast<size_t>(outputReplicas) &&
            dot.getRhsLaneOffsets().size() ==
                static_cast<size_t>(outputReplicas) &&
            kernel && plannedResources <= kernel.getTarget().getVectorRegisters();
        if (!sequentialPlanClosed) {
          dot.emitError(
              "selected sequential topology has no closed issue-slice, free-axis, and accumulator carrier; lhs_issue=")
              << sequentialLhsIssue << ", rhs_issue=" << sequentialRhsIssue
              << ", accumulator=" << sequentialAccumulator
              << ", output_replicas=" << outputReplicas
              << ", lhs_issue_parts="
              << (lhsParts ? *lhsParts : int64_t{-1})
              << ", rhs_issue_parts="
              << (rhsParts ? *rhsParts : int64_t{-1})
              << ", lhs_plan_parts=" << dot.getLhsParts().size()
              << ", rhs_plan_parts=" << dot.getRhsParts().size()
              << ", streams=" << streams
              << ", resources=" << plannedResources;
          signalPassFailure();
          return;
        }
        if (kind != "level_scaled")
          resources = plannedResources;
      }

      dot->setAttr("partial_topology",
                   makePartialTopology(builder, dot, kind, rootOperand,
                                       partialAxes, outputAxes, sourceSlots,
                                       partialSlots, replicas, laneSplit,
                                       combineArity,
                                       resources));
      if (nestedPlan)
        dot->setAttr("nested_partial_plan", *nestedPlan);

      // The topology owner also freezes every intermediate physical value
      // layout used by materialization.  The following pass may instantiate
      // PartialSet containers from these types, but it must not reconstruct
      // lane/time/replica placement from factors a second time.
      riscv::ValueType partialSlot =
          sourcePlan ? sourcePlan->geometry.partialSlot
                     : partialSlotType(builder, dot);
      riscv::ValueType sourceSlot =
          sourcePlan ? sourcePlan->geometry.sourceSlot : riscv::ValueType();
      riscv::ValueType splitSourceSlot =
          sourcePlan ? sourcePlan->geometry.splitSourceSlot
                     : riscv::ValueType();
      riscv::ValueType widenedSlot =
          partialSlot ? widenPartialSlotType(builder, partialSlot)
                      : riscv::ValueType();
      riscv::ValueType reducedSlot =
          sourcePlan
              ? sourcePlan->geometry.reducedSlot
              : (partialSlot && dot.getOver().size() == 1
                     ? reducedPartialSlotType(builder, partialSlot,
                                              dot.getOver()[0])
                     : riscv::ValueType());
      auto vectorResult =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      if (dot.getResult().hasOneUse()) {
        auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *dot.getResult().getUsers().begin());
        if (conversion && conversion.getConversion().getEffect() == "pure" &&
            conversion.getConversion().getKind() == "register_to_lane")
          vectorResult =
              mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      }
      riscv::ValueType vectorSlot = kind == "replica_reduced"
                                        ? replicaAccumulator
                                    : vectorResult && dot.getOver().size() == 1
                                        ? vectorPartialSlotType(
                                              builder, dot.getLhs().getType(),
                                              vectorResult, dot.getOver()[0])
                                        : riscv::ValueType();
      if (kind != "replica_reduced" && vectorSlot &&
          vectorPartialSlotType(builder, dot.getRhs().getType(), vectorResult,
                                dot.getOver()[0]) != vectorSlot)
        vectorSlot = {};
      if (!partialSlot) {
        dot.emitError("selected partial topology has no typed partial layout");
        signalPassFailure();
        return;
      }
      riscv::PartialSetType sourceSetType;
      riscv::PartialSetType repackedSetType;
      riscv::PartialSetType partialSetType;
      riscv::PartialSetType scaledSetType;
      riscv::PartialSetType directSetType;
      riscv::PartialSetType reducedSetType;
      riscv::PartialSetType scaleCombinedSetType;
      riscv::PartialSetType fullScaledSetType;
      const int64_t reductionAxis =
          dot.getOver().empty() ? 0 : dot.getOver()[0];
      const bool completeSetGeometry =
          dot.getOver().size() == 1 && reductionAxis > 0 &&
          combineArity > 0 && partialSlots > 0 &&
          partialSlots % combineArity == 0 && !partialSlot.getShape().empty();
      if (completeSetGeometry) {
        const int64_t partialGroups =
            combineArity * partialSlot.getLayout().getRegisterGroups();
        partialSetType = riscv::PartialSetType::get(
            builder.getContext(), partialSlot, reductionAxis, combineArity,
            partialSlot.getShape()[0], partialGroups);
        if (widenedSlot) {
          const int64_t widenedGroups =
              widenedSlot.getLayout().getRegisterGroups();
          scaledSetType = riscv::PartialSetType::get(
              builder.getContext(), widenedSlot, reductionAxis, 1,
              partialSetType.getTermsPerSlot() * combineArity, widenedGroups);
          fullScaledSetType = riscv::PartialSetType::get(
              builder.getContext(), widenedSlot, reductionAxis, 1,
              partialSlot.getShape()[0] * partialSlots, widenedGroups);
        }
        if (kind == "reduced_scaled")
          directSetType = riscv::PartialSetType::get(
              builder.getContext(), partialSlot, reductionAxis, partialSlots,
              partialSlot.getShape()[0],
              partialSlots * partialSlot.getLayout().getRegisterGroups());
      }
      if (sourcePlan && completeSetGeometry) {
        const int64_t chunkPartialSlots = sourceSlots * laneSplit;
        const int64_t sourceSetGroups =
            sourceSlots * sourceSlot.getLayout().getRegisterGroups();
        const int64_t reducedSetGroups =
            chunkPartialSlots * reducedSlot.getLayout().getRegisterGroups();
        sourceSetType = riscv::PartialSetType::get(
            builder.getContext(), sourceSlot, reductionAxis, sourceSlots,
            sourceSlot.getShape()[0], sourceSetGroups);
        repackedSetType = riscv::PartialSetType::get(
            builder.getContext(), partialSlot, reductionAxis, chunkPartialSlots,
            sourceSetType.getTermsPerSlot() / laneSplit,
            chunkPartialSlots * partialSlot.getLayout().getRegisterGroups());
        reducedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, chunkPartialSlots,
            repackedSetType.getTermsPerSlot(), reducedSetGroups);
        scaleCombinedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, sourceSlots,
            reducedSetType.getTermsPerSlot() * laneSplit,
            sourceSlots * reducedSlot.getLayout().getRegisterGroups());
      } else if (completeSetGeometry && reducedSlot) {
        reducedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, partialSlots,
            partialSlot.getShape()[0],
            partialSlots * reducedSlot.getLayout().getRegisterGroups());
        const int64_t fanout = partialSlots / combineArity;
        scaleCombinedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, combineArity,
            reducedSetType.getTermsPerSlot() * fanout,
            combineArity * reducedSlot.getLayout().getRegisterGroups());
      }
      auto optionalType = [&](riscv::ValueType type) -> mlir::Type {
        return type ? mlir::Type(type)
                    : mlir::Type(mlir::NoneType::get(builder.getContext()));
      };
      auto optionalSetType = [&](riscv::PartialSetType type) -> mlir::Type {
        return type ? mlir::Type(type)
                    : mlir::Type(mlir::NoneType::get(builder.getContext()));
      };
      llvm::ArrayRef<int64_t> empty;
      mlir::Attribute selectedRepackLeaf = builder.getUnitAttr();
      if (sourcePlan) {
        auto repack = partialRepackLeaf(builder, kernel.getTarget(),
            sourceSetType, repackedSetType, laneSplit);
        if (!repack) {
          dot.emitError("partial source plan has no legal repack carrier");
          signalPassFailure();
          return;
        }
        selectedRepackLeaf = *repack;
      }
      dot->setAttr(
          "partial_layout_plan",
          riscv::PartialLayoutPlanAttr::get(
              builder.getContext(),
              optionalType(sourcePlan ? sourcePlan->lhsType
                                      : riscv::ValueType()),
              optionalType(sourcePlan ? sourcePlan->rhsType
                                      : riscv::ValueType()),
              optionalType(sourceSlot),
              optionalType(splitSourceSlot), partialSlot,
              optionalType(widenedSlot), optionalType(reducedSlot),
              optionalType(vectorSlot),
              optionalSetType(sourceSetType), optionalSetType(repackedSetType),
              optionalSetType(partialSetType), optionalSetType(scaledSetType),
              optionalSetType(directSetType), optionalSetType(reducedSetType),
              optionalSetType(scaleCombinedSetType),
              optionalSetType(fullScaledSetType),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->lhsParts
                                                      : empty),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->rhsParts
                                                      : empty),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->lhsOffsets
                                                      : empty),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->rhsOffsets
                                                      : empty), selectedRepackLeaf));

      if (kind == "sequential_fused" || kind == "sequential_per_stream" ||
          kind == "level_scaled") {
        auto lhsIssue = sequentialLhsIssue;
        auto rhsIssue = sequentialRhsIssue;
        auto accumulator = sequentialAccumulator;
        auto accumulatorElement = accumulator
                                      ? mlir::dyn_cast<mlir::IntegerType>(
                                            accumulator.getElementType())
                                      : mlir::IntegerType();
        const int64_t issueCount = dot.getReductionStreams();
        llvm::SmallVector<int64_t> lhsLaneOffsets;
        llvm::SmallVector<int64_t> rhsLaneOffsets;
        lhsLaneOffsets.reserve(static_cast<size_t>(issueCount * outputReplicas));
        rhsLaneOffsets.reserve(static_cast<size_t>(issueCount * outputReplicas));
        for (int64_t output = 0; output < outputReplicas; ++output)
          for (int64_t issue = 0; issue < issueCount; ++issue) {
            lhsLaneOffsets.push_back(dot.getLhsLaneOffsets()[output]);
            rhsLaneOffsets.push_back(dot.getRhsLaneOffsets()[output]);
          }
        const int64_t plannedResources =
            lhsIssue && rhsIssue && accumulator
                ? std::max<int64_t>(
                      lhsIssue.getLayout().getRegisterGroups() +
                          rhsIssue.getLayout().getRegisterGroups() +
                          accumulator.getLayout().getRegisterGroups(),
                      accumulator.getLayout().getRegisterGroups() + 2)
                : 0;
        if (!sequentialPlanClosed || !lhsIssue || !rhsIssue || !accumulator ||
            !accumulatorElement ||
            (kind != "level_scaled" && plannedResources != resources) || !kernel ||
            plannedResources > kernel.getTarget().getVectorRegisters()) {
          dot.emitError(
              "selected sequential topology has no closed issue-slice and accumulator plan");
          signalPassFailure();
          return;
        }
        const llvm::StringRef realization =
            kind == "level_scaled"
                ? llvm::StringRef("fused_partial")
                : kind == "sequential_fused" ? llvm::StringRef("fused")
                                              : llvm::StringRef("per_stream");
        const llvm::StringRef finalizeInstruction =
            kind == "level_scaled"
                ? llvm::StringRef("none")
                : accumulatorElement.getWidth() == 16
                      ? llvm::StringRef("rvv.vwredsum.partial")
                      : llvm::StringRef("rvv.vredsum.partial");
        dot->setAttr(
            "sequential_partial_plan",
            riscv::SequentialPartialPlanAttr::get(
                builder.getContext(), realization,
                issueCount,
                builder.getDenseI64ArrayAttr(dot.getOver()), lhsIssue,
                rhsIssue, accumulator, sequentialLhsSupply,
                sequentialRhsSupply,
                builder.getDenseI64ArrayAttr(dot.getLhsParts()),
                builder.getDenseI64ArrayAttr(dot.getRhsParts()),
                builder.getDenseI64ArrayAttr(lhsLaneOffsets),
                builder.getDenseI64ArrayAttr(rhsLaneOffsets),
                *sequentialMultiply, "rvv.vwmacc.partial", finalizeInstruction,
                plannedResources));
      }

      if (kind == "scaled" || kind == "reduced_scaled") {
        auto found = scaledReductions.find(dot.getOperation());
        auto match = found != scaledReductions.end()
                         ? matchReplicaScaledDotReduction(found->second)
                         : std::optional<ReplicaScaledDotReduction>();
        auto scaleSource =
            match ? mlir::dyn_cast<riscv::ValueType>(match->scale.getType())
                  : riscv::ValueType();
        auto scalarScale =
            scaleSource ? scalarReplicaType(builder, scaleSource)
                        : riscv::ValueType();
        llvm::DenseSet<mlir::Operation *> rematerializationVisited;
        const bool canRematerialize =
            match && supportsScalarReplicaRematerialization(
                         match->scale, rematerializationVisited);
        const llvm::StringRef scaleSupply =
            canRematerialize ? "scalar-rematerialize" : "vector-convert";
        auto narrowScale =
            kind == "scaled" && laneSplit == 1 && match
                ? plannedNarrowScaleType(builder, match->scale, scalarScale)
                : riscv::ValueType();
        auto multiplyInstruction =
            partialMultiplyInstruction(dot.getLhs().getType(),
                                       dot.getRhs().getType());
        auto finalSet = laneSplit > 1 || kind == "reduced_scaled"
                            ? scaleCombinedSetType
                            : fullScaledSetType;
        auto finalizeInstruction =
            finalSet ? partialFinalizeInstruction(finalSet, reductionAxis)
                     : std::optional<llvm::StringRef>();
        const llvm::StringRef reduceInstruction =
            laneSplit > 1 || kind == "reduced_scaled"
                ? llvm::StringRef("rvv.partial-reduce.widen")
                : llvm::StringRef("none");
        int64_t plannedResources = resources;
        if (kind == "scaled" && laneSplit == 1 && partialSetType &&
            scaledSetType && combineArity > 0 &&
            partialSlots % combineArity == 0) {
          const int64_t chunks = partialSlots / combineArity;
          plannedResources = std::max<int64_t>(
              plannedResources,
              partialSetType.getResourceGroups() +
                  chunks * scaledSetType.getResourceGroups() +
                  (narrowScale
                       ? narrowScale.getLayout().getRegisterGroups()
                       : 0));
        }
        if (!match || !scalarScale || !multiplyInstruction || !finalSet ||
            !finalizeInstruction ||
            (kind == "scaled" && laneSplit == 1 && !narrowScale) || !kernel ||
            plannedResources > kernel.getTarget().getVectorRegisters()) {
          dot.emitError(
              "selected scaled topology has no closed scale-supply and reduction plan");
          signalPassFailure();
          return;
        }
        if (plannedResources != resources) {
          auto topology = dot.getPartialTopology();
          dot->setAttr(
              "partial_topology",
              riscv::PartialTopologyAttr::get(
                  builder.getContext(), topology.getKind(),
                  topology.getRootOperand(), topology.getPartialAxes(),
                  topology.getOutputAxes(), topology.getSourceSlots(),
                  topology.getPartialSlots(), topology.getOutputReplicas(),
                  topology.getLaneSplit(), topology.getCombineArity(),
                  topology.getSlotOrder(), plannedResources));
        }
        dot->setAttr(
            "scaled_partial_plan",
            riscv::ScaledPartialPlanAttr::get(
                builder.getContext(), kind, scalarScale,
                narrowScale
                    ? mlir::Type(narrowScale)
                    : mlir::Type(mlir::NoneType::get(builder.getContext())),
                scaleSupply, *multiplyInstruction, reduceInstruction,
                *finalizeInstruction, plannedResources));
      }

      if (kind == "layered") {
        if (!layeredPlan) {
          dot.emitError("selected layered topology has no typed physical plan");
          signalPassFailure();
          return;
        }
        auto root = llvm::find_if(
            layeredPlan->roots, [&](const ProjectedRoot &candidate) {
              return candidate.value == layeredPlan->layered.value;
            });
        if (root == layeredPlan->roots.end()) {
          dot.emitError("selected layered topology lost its typed root identity");
          signalPassFailure();
          return;
        }
        const int64_t rootIndex = static_cast<int64_t>(
            root - layeredPlan->roots.begin());
        llvm::SmallVector<mlir::Attribute> rootWindowTypes;
        llvm::SmallVector<mlir::Attribute> rootStoragePlans;
        llvm::SmallVector<mlir::Attribute> rootWindowInstructions;
        rootWindowTypes.reserve(layeredPlan->roots.size());
        rootStoragePlans.reserve(layeredPlan->roots.size());
        rootWindowInstructions.reserve(layeredPlan->roots.size());
        for (const ProjectedRoot &candidate : layeredPlan->roots) {
          auto windowType = layeredPlan->windowTypes.lookup(candidate.value);
          if (!windowType) {
            dot.emitError(
                "selected layered topology has an untyped projected root");
            signalPassFailure();
            return;
          }
          auto storagePlan = riscv_internal::storageWindowPlan(
              builder, candidate.field, layeredPlan->reductionAxis,
              candidate.base, candidate.stride, candidate.repeat,
              candidate.extent, layeredPlan->layered.geometry.getLaneCount());
          if (!storagePlan) {
            dot.emitError(
                "selected layered topology has an unplanned projected storage root");
            signalPassFailure();
            return;
          }
          rootWindowTypes.push_back(mlir::TypeAttr::get(windowType));
          rootStoragePlans.push_back(*storagePlan);
          rootWindowInstructions.push_back(builder.getStringAttr(
              candidate.access.getMapping() == "grouped_layered"
                  ? "rvv.storage-window.layered"
                  : "rvv.storage-window.natural"));
        }
        const int64_t rawGroups =
            layeredPlan->layeredWindowType.getLayout().getRegisterGroups();
        auto storageType = riscv::LayeredWindowType::get(
            builder.getContext(),
            mlir::cast<riscv::ValueType>(
                layeredPlan->layered.field.getResult().getType()),
            layeredPlan->layeredWindowType, layeredPlan->reductionAxis,
            layeredPlan->layers, layeredPlan->windowsPerLayer, rawGroups);
        llvm::SmallVector<mlir::Attribute> decodeInstructions;
        for (auto [shift, mask] : llvm::zip(
                 layeredPlan->layered.geometry
                     .getShiftAmountForLogicalLayer()
                     .asArrayRef(),
                 layeredPlan->layered.geometry
                     .getMaskValueForLogicalLayer()
                     .asArrayRef()))
          decodeInstructions.push_back(builder.getStringAttr(
              riscv_internal::layeredStorageDecodeInstruction(shift, mask)));
        auto accumulatorElement = mlir::dyn_cast<mlir::IntegerType>(
            layeredPlan->accumulatorType.getElementType());
        if (!accumulatorElement ||
            (accumulatorElement.getWidth() != 16 &&
             accumulatorElement.getWidth() != 32)) {
          dot.emitError(
              "selected layered topology has no typed final reduction leaf");
          signalPassFailure();
          return;
        }
        const llvm::StringRef finalizeInstruction =
            accumulatorElement.getWidth() == 16 ? "rvv.vwredsum.partial"
                                                : "rvv.vredsum.partial";
        dot->setAttr(
            "layered_partial_plan",
            riscv::LayeredPartialPlanAttr::get(
                builder.getContext(), rootIndex,
                layeredPlan->layeredWindowType, storageType,
                layeredPlan->lhsIssueType, layeredPlan->rhsIssueType,
                layeredPlan->accumulatorType,
                builder.getArrayAttr(rootWindowTypes),
                builder.getArrayAttr(rootStoragePlans),
                builder.getArrayAttr(rootWindowInstructions),
                builder.getArrayAttr(decodeInstructions), finalizeInstruction,
                layeredPlan->layered.geometry, resources));
      }
    }

    // A nested partial plan owns the contraction's issue unroll.  Lowering
    // records the same requested factor on the original issue loop so
    // non-partial contractions can still use generic loop unrolling.  Once a
    // nested plan has been selected, consume that loop attribute here: the
    // materializer will create the planned issue loop and attach the frozen
    // NestedPartialPlanAttr::issue_unroll exactly once.  A mixed loop would
    // otherwise have two incompatible schedule owners, so reject it instead
    // of silently unrolling the whole canonical contraction as well.
    llvm::SmallVector<mlir::scf::ForOp> issueLoops;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.unroll_factor"))
        issueLoops.push_back(loop);
    });
    for (mlir::scf::ForOp loop : issueLoops) {
      llvm::SmallVector<riscv::RVVWidenDotOp> ownedDots;
      loop.walk([&](riscv::RVVWidenDotOp dot) {
        if (dot->getParentOfType<mlir::scf::ForOp>() == loop)
          ownedDots.push_back(dot);
      });
      const bool consumesIssueUnroll = llvm::any_of(
          ownedDots, [](riscv::RVVWidenDotOp dot) {
            return static_cast<bool>(dot.getNestedPartialPlanAttr());
          });
      if (!consumesIssueUnroll)
        continue;
      if (ownedDots.empty() ||
          !llvm::all_of(ownedDots, [](riscv::RVVWidenDotOp dot) {
            return static_cast<bool>(dot.getNestedPartialPlanAttr());
          })) {
        loop.emitError(
            "one physical issue loop mixes planner-owned and generic unroll contracts");
        signalPassFailure();
        return;
      }
      const int64_t requested =
          loop
              ->getAttrOfType<mlir::IntegerAttr>(
                  "weft.riscv.unroll_factor")
              .getInt();
      for (riscv::RVVWidenDotOp dot : ownedDots) {
        auto plan = dot.getNestedPartialPlanAttr();
        if (dot.getPartialUnroll() != requested ||
            plan.getIssueUnroll() !=
                std::min<int64_t>(requested, plan.getIssueStreams())) {
          dot.emitError(
              "nested partial plan did not consume the issue-loop unroll contract exactly once");
          signalPassFailure();
          return;
        }
      }
      loop->removeAttr("weft.riscv.unroll_factor");
    }

    // Freeze the complete combine program after every dot has its selected
    // topology and typed layout plan.  The following materialization pass may
    // validate these facts against the still-present SSA graph, but it must not
    // choose a vector/scalar realization or rebuild intermediate set types.
    for (riscv::RVVWidenDotOp dot : dots) {
      if (!dot || !dot->getBlock() ||
          dot.getPartialTopology().getKind() != "independent")
        continue;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto lhsStreams = riscv_internal::staticProduct(
          lhs.getLayout().getTimeFactors().asArrayRef());
      auto rhsStreams = riscv_internal::staticProduct(
          rhs.getLayout().getTimeFactors().asArrayRef());
      auto layoutPlan = dot.getPartialLayoutPlanAttr();
      auto slotType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getPartialSlotType())
                     : riscv::ValueType();
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          resultValue ? resultValue.getElementType() : dot.getResult().getType());
      auto outputParts = resultValue
                             ? riscv_internal::staticProduct(
                                   resultValue.getLayout()
                                       .getReplicaFactors()
                                       .asArrayRef())
                             : std::optional<int64_t>(1);
      auto multiplyInstruction = partialMultiplyInstruction(lhs, rhs);
      if (!lhsStreams || !rhsStreams || *lhsStreams != *rhsStreams ||
          *lhsStreams < 2 || !layoutPlan || !slotType || !resultElement ||
          !resultElement.isSigned() || resultElement.getWidth() != 32 ||
          !outputParts || *outputParts <= 0 || !multiplyInstruction) {
        dot.emitError(
            "selected independent topology has no complete typed combine inputs");
        signalPassFailure();
        return;
      }
      const int64_t slots = *lhsStreams;
      riscv::ConvertLayoutOp vectorConsumer;
      riscv::ValueType vectorResult;
      if (dot.getResult().hasOneUse()) {
        vectorConsumer = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *dot.getResult().getUsers().begin());
        if (vectorConsumer &&
            vectorConsumer.getConversion().getEffect() == "pure" &&
            vectorConsumer.getConversion().getKind() == "register_to_lane")
          vectorResult =
              mlir::dyn_cast<riscv::ValueType>(vectorConsumer.getResult().getType());
      }
      auto vectorSlotType =
          mlir::dyn_cast<riscv::ValueType>(layoutPlan.getVectorSlotType());
      auto lhsReplicas = riscv_internal::staticProduct(
          lhs.getLayout().getReplicaFactors().asArrayRef());
      auto rhsReplicas = riscv_internal::staticProduct(
          rhs.getLayout().getReplicaFactors().asArrayRef());
      auto vectorResultStreams =
          vectorResult
              ? riscv_internal::staticProduct(
                    vectorResult.getLayout().getTimeFactors().asArrayRef())
              : std::optional<int64_t>();
      auto vectorResultReplicas =
          vectorResult
              ? riscv_internal::staticProduct(
                    vectorResult.getLayout().getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      const bool preservesFreeLane =
          vectorConsumer && vectorResult && vectorSlotType && lhsReplicas &&
          rhsReplicas && *lhsReplicas == 1 && *rhsReplicas == 1 &&
          vectorResultStreams && vectorResultReplicas &&
          *vectorResultStreams == 1 && *vectorResultReplicas == 1;
      riscv::ValueType selectedSlot =
          preservesFreeLane ? vectorSlotType : slotType;
      auto reductionAxis =
          primaryReductionLaneAxis(selectedSlot, dot.getOver());
      auto termsPerSourceSlot = laneProductForAxes(selectedSlot, dot.getOver());
      const int64_t slotGroups = selectedSlot.getLayout().getRegisterGroups();
      const int64_t setGroups = slots * slotGroups;
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      if (!reductionAxis || !termsPerSourceSlot || !kernel || setGroups <= 0 ||
          operandGroups < 0 ||
          kernel.getTarget().getVectorRegisters() < operandGroups + 2 ||
          setGroups >
              kernel.getTarget().getVectorRegisters() - operandGroups - 2) {
        dot.emitError(
            "selected independent topology exceeds its typed source-set resource contract");
        signalPassFailure();
        return;
      }
      const auto reductionPosition =
          axisPosition(selectedSlot, *reductionAxis);
      if (!reductionPosition) {
        dot.emitError(
            "selected independent topology lost its reduction axis in the partial carrier");
        signalPassFailure();
        return;
      }
      auto sourceSet = riscv::PartialSetType::get(
          builder.getContext(), selectedSlot, *reductionAxis, slots,
          *termsPerSourceSlot, setGroups);
      llvm::SmallVector<mlir::Attribute> combineTypes;
      llvm::SmallVector<int64_t> combineArities;
      riscv::PartialSetType finalSet = sourceSet;
      int64_t remainingSlots = slots;
      int64_t termsPerSlot = sourceSet.getTermsPerSlot();
      riscv::ValueType combineSlot = selectedSlot;
      auto selectedElement = mlir::dyn_cast<mlir::IntegerType>(
          selectedSlot.getElementType());
      const bool widenFirstCombine =
          !dot.getFusedStreamsLegal() && selectedElement &&
          selectedElement.isSigned() && selectedElement.getWidth() == 16 &&
          resultElement.isSigned() && resultElement.getWidth() == 32;
      if (widenFirstCombine) {
        combineSlot = widenPartialSlotType(builder, selectedSlot);
        auto wideResources = widenedIndependentResources(
            kernel.getTarget(), selectedSlot, combineSlot, slots, operandGroups);
        if (!wideResources ||
            *wideResources > dot.getPartialTopology().getResourceGroups()) {
          dot.emitError(
              "selected independent topology has no resource-legal first i32 combine stage");
          signalPassFailure();
          return;
        }
      }
      while (remainingSlots > 1) {
        const int64_t arity = remainingSlots % 2 == 0 ? 2 : remainingSlots;
        remainingSlots /= arity;
        termsPerSlot *= arity;
        const int64_t combineGroups =
            combineSlot.getLayout().getRegisterGroups();
        finalSet = riscv::PartialSetType::get(
            builder.getContext(), combineSlot, *reductionAxis, remainingSlots,
            termsPerSlot, remainingSlots * combineGroups);
        combineTypes.push_back(mlir::TypeAttr::get(finalSet));
        combineArities.push_back(arity);
      }
      llvm::SmallVector<int64_t> lhsParts;
      llvm::SmallVector<int64_t> rhsParts;
      const int64_t plannedOutputs = preservesFreeLane ? 1 : *outputParts;
      for (int64_t outputPart = 0; outputPart < plannedOutputs; ++outputPart) {
        int64_t lhsBase = 0;
        int64_t rhsBase = 0;
        if (!preservesFreeLane) {
          auto lhsReplica = resultValue
                                ? projectReplica(lhs, resultValue, outputPart)
                                : std::optional<int64_t>(0);
          auto rhsReplica = resultValue
                                ? projectReplica(rhs, resultValue, outputPart)
                                : std::optional<int64_t>(0);
          if (!lhsReplica || !rhsReplica) {
            dot.emitError(
                "selected independent topology has no complete output-replica map");
            signalPassFailure();
            return;
          }
          lhsBase = *lhsReplica * slots;
          rhsBase = *rhsReplica * slots;
        }
        for (int64_t stream = 0; stream < slots; ++stream) {
          lhsParts.push_back(preservesFreeLane ? stream : lhsBase + stream);
          rhsParts.push_back(preservesFreeLane ? stream : rhsBase + stream);
        }
      }
      auto finalizeInstruction =
          preservesFreeLane
              ? std::optional<llvm::StringRef>("rvv.partial-finalize.widen")
              : partialFinalizeInstruction(finalSet, *reductionAxis);
      if (!finalizeInstruction) {
        dot.emitError(
            "selected independent topology has no typed finalization leaf");
        signalPassFailure();
        return;
      }
      llvm::ArrayRef<int64_t> empty;
      dot->setAttr(
          "partial_combine_plan",
          riscv::PartialCombinePlanAttr::get(
              builder.getContext(),
              preservesFreeLane ? "independent_vector"
                                : "independent_scalar",
              sourceSet, mlir::NoneType::get(builder.getContext()), finalSet,
              builder.getArrayAttr(combineTypes),
              builder.getDenseI64ArrayAttr(combineArities),
              builder.getDenseI64ArrayAttr(lhsParts),
              builder.getDenseI64ArrayAttr(rhsParts),
              builder.getDenseI64ArrayAttr(empty), *multiplyInstruction,
              *finalizeInstruction, plannedOutputs,
              dot.getPartialTopology().getResourceGroups()));
    }

    llvm::SmallVector<mlir::scf::ForOp> plannedLevelLoops;
    getOperation().walk(
        [&](mlir::scf::ForOp loop) { plannedLevelLoops.push_back(loop); });
    for (mlir::scf::ForOp loop : plannedLevelLoops) {
      auto matched = matchLevelScaledLoop(loop);
      if (!matched)
        if (auto seed = matchLevelScaledLoopSeed(loop))
          matched = llvm::SmallVector<ScaledPartialContribution>{*seed};
      if (!matched ||
          !llvm::all_of(*matched, [](const ScaledPartialContribution &item) {
            return hasTopology(item.dot, "level_scaled");
          }))
        continue;
      mlir::Type carry = loop.getRegionIterArg(0).getType();
      auto carryType = mlir::dyn_cast<riscv::ValueType>(carry);
      auto carryInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(carry));
      auto outputParts = carryType
                             ? riscv_internal::staticProduct(
                                   carryType.getLayout()
                                       .getReplicaFactors()
                                       .asArrayRef())
                             : carryInteger && carryInteger.isSigned() &&
                                       carryInteger.getWidth() == 32
                                   ? std::optional<int64_t>(1)
                                   : std::optional<int64_t>();
      auto firstLayout = matched->front().dot.getPartialLayoutPlanAttr();
      auto firstTopology = matched->front().dot.getPartialTopology();
      auto slotType =
          firstLayout ? mlir::dyn_cast<riscv::ValueType>(
                            firstLayout.getPartialSlotType())
                      : riscv::ValueType();
      auto reducedSet =
          firstLayout ? mlir::dyn_cast<riscv::PartialSetType>(
                            firstLayout.getReducedSetType())
                      : riscv::PartialSetType();
      auto finalSet =
          firstLayout ? mlir::dyn_cast<riscv::PartialSetType>(
                            firstLayout.getScaleCombinedSetType())
                      : riscv::PartialSetType();
      const int64_t slots = firstTopology.getPartialSlots();
      const int64_t reductionAxis = matched->front().dot.getOver()[0];
      const bool scalarCarry =
          carryType ? carryType.getLayout().getCarrier() == "scalar"
                    : static_cast<bool>(carryInteger);
      if (!scalarCarry || !outputParts || *outputParts <= 0 || !slotType ||
          !reducedSet || !finalSet || slots <= 0) {
        loop.emitError(
            "selected level-scaled topology has no complete typed combine carrier");
        signalPassFailure();
        return;
      }
      auto sourceSet = riscv::PartialSetType::get(
          builder.getContext(), slotType, reductionAxis, slots,
          slotType.getShape()[0],
          slots * slotType.getLayout().getRegisterGroups());
      auto finalizeInstruction =
          partialFinalizeInstruction(finalSet, reductionAxis);
      std::optional<std::string> commonMultiply;
      for (ScaledPartialContribution &contribution : *matched) {
        auto selected = partialMultiplyInstruction(contribution.dot.getLhs().getType(),
                                                   contribution.dot.getRhs().getType());
        if (!selected || (commonMultiply && *commonMultiply != *selected)) {
          loop.emitError(
              "selected level-scaled topology has inconsistent multiply leaves");
          signalPassFailure();
          return;
        }
        commonMultiply = *selected;
      }
      if (!commonMultiply || !finalizeInstruction) {
        loop.emitError(
            "selected level-scaled topology has no complete leaf selection");
        signalPassFailure();
        return;
      }
      llvm::ArrayRef<int64_t> empty;
      for (ScaledPartialContribution &contribution : *matched) {
        llvm::SmallVector<int64_t> lhsParts;
        llvm::SmallVector<int64_t> rhsParts;
        llvm::SmallVector<int64_t> scaleParts;
        for (int64_t outputPart = 0; outputPart < *outputParts; ++outputPart) {
          auto singletonPart = [](riscv::ValueType value)
              -> std::optional<int64_t> {
            auto parts = riscv_internal::staticProduct(
                value.getLayout().getReplicaFactors().asArrayRef());
            return parts && *parts == 1 ? std::optional<int64_t>(0)
                                        : std::nullopt;
          };
          auto lhsPart = carryType
                             ? projectReplica(contribution.dot.getLhs().getType(),
                                              carryType, outputPart)
                             : singletonPart(contribution.dot.getLhs().getType());
          auto rhsPart = carryType
                             ? projectReplica(contribution.dot.getRhs().getType(),
                                              carryType, outputPart)
                             : singletonPart(contribution.dot.getRhs().getType());
          auto scaleType =
              mlir::dyn_cast<riscv::ValueType>(contribution.scale.getType());
          auto scalePart =
              carryType && scaleType
                  ? projectReplica(scaleType, carryType, outputPart)
                  : !carryType && !scaleType
                        ? std::optional<int64_t>(0)
                        : scaleType ? singletonPart(scaleType) : std::nullopt;
          if (!lhsPart || !rhsPart || !scalePart) {
            loop.emitError(
                "selected level-scaled topology has no complete output-replica map");
            signalPassFailure();
            return;
          }
          lhsParts.push_back(*lhsPart);
          rhsParts.push_back(*rhsPart);
          scaleParts.push_back(*scalePart);
        }
        contribution.dot->setAttr(
            "partial_combine_plan",
            riscv::PartialCombinePlanAttr::get(
                builder.getContext(), "level_scaled", sourceSet, reducedSet,
                finalSet, builder.getArrayAttr({}),
                builder.getDenseI64ArrayAttr(empty),
                builder.getDenseI64ArrayAttr(lhsParts),
                builder.getDenseI64ArrayAttr(rhsParts),
                builder.getDenseI64ArrayAttr(scaleParts), *commonMultiply,
                *finalizeInstruction, *outputParts,
                firstTopology.getResourceGroups()));
      }
    }

    planPartialAddTrees(getOperation(), false);
  }
};

} // namespace weft::riscv_partial

std::unique_ptr<mlir::Pass>
weft::createPlanRISCVPartialTopologiesPass() {
  return std::make_unique<weft::riscv_partial::PlanRISCVPartialTopologiesPass>();
}
