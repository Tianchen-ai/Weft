#include "Support.h"

namespace weft::riscv_partial {

class MaterializeRISCVPartialAccumulatorsPass
    : public mlir::PassWrapper<MaterializeRISCVPartialAccumulatorsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-partial-accumulators";
  }

  llvm::StringRef getDescription() const override {
    return "Materialize sequential storage windows and loop-carried widened partial accumulators";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    int64_t nextPartialBirthId = 0;
    getOperation().walk([&](mlir::Operation *operation) {
      if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
        nextPartialBirthId =
            std::max(nextPartialBirthId,
                     static_cast<int64_t>(partial.getBirthId()) + 1);
      if (auto capture = mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
        nextPartialBirthId =
            std::max(nextPartialBirthId,
                     static_cast<int64_t>(capture.getBirthId()) + 1);
      if (auto collect = mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
        nextPartialBirthId =
            std::max(nextPartialBirthId,
                     static_cast<int64_t>(collect.getBirthId()) + 1);
    });

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    for (riscv::ExtractOp extract : extracts) {
      auto candidate = analyzeIssueStorageWindowCandidate(rewriter, extract);
      if (!candidate)
        continue;
      rewriter.setInsertionPoint(extract);
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          extract.getLoc(), 0);
      llvm::StringRef tail = candidate->resultType.getLayout().getValidity() == "tail"
                                 ? "agnostic"
                                 : "exact";
      auto window = rewriter.create<riscv::RVVStorageWindowOp>(
          extract.getLoc(), candidate->resultType, candidate->field.getResult(),
          candidate->point.getResult(), zero, candidate->plan,
          extract.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "storage-window",
                               "rvv.storage-window.layered",
                               "rvv.storage-window.layered",
                               mlir::cast<riscv::ValueType>(
                                   candidate->field.getResult().getType())
                                   .getLayout()
                                   .getRegisterGroups(),
                               candidate->resultType.getLayout().getRegisterGroups(),
                               0, 0,
                               "none", tail));
      riscv_internal::copyOrigin(extract, window);
      extract.getResult().replaceAllUsesWith(window.getResult());
      rewriter.eraseOp(extract);
    }

    llvm::SmallVector<riscv::BinaryOp> wideningProducts;
    getOperation().walk([&](riscv::BinaryOp binary) {
      if (binary.getKind() == "mul")
        wideningProducts.push_back(binary);
    });
    for (riscv::BinaryOp binary : wideningProducts) {
      auto lhsWiden = binary.getLhs().getDefiningOp<riscv::WidenOp>();
      auto rhsWiden = binary.getRhs().getDefiningOp<riscv::WidenOp>();
      auto lhs = lhsWiden
                     ? mlir::dyn_cast<riscv::ValueType>(
                           lhsWiden.getInput().getType())
                     : riscv::ValueType();
      auto rhs = rhsWiden
                     ? mlir::dyn_cast<riscv::ValueType>(
                           rhsWiden.getInput().getType())
                     : riscv::ValueType();
      auto result =
          mlir::dyn_cast<riscv::ValueType>(binary.getResult().getType());
      auto lhsElement =
          lhs ? mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType())
              : mlir::IntegerType();
      auto rhsElement =
          rhs ? mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType())
              : mlir::IntegerType();
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      auto kernel = binary->getParentOfType<riscv::KernelOp>();
      if (!lhs || !rhs || !result || !lhsElement || !rhsElement ||
          !resultElement || lhsElement.isSignless() || rhsElement.isSignless() ||
          resultElement.isSignless() ||
          lhsElement.getWidth() > lhs.getLayout().getSew() ||
          rhsElement.getWidth() > rhs.getLayout().getSew() ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          resultElement.getWidth() != 2 * lhs.getLayout().getSew() ||
          lhs.getShape() != rhs.getShape() || lhs.getShape() != result.getShape() ||
          lhs.getAxisIds() != rhs.getAxisIds() ||
          lhs.getAxisIds() != result.getAxisIds() ||
          lhs.getLayout().getTimeFactors() !=
              rhs.getLayout().getTimeFactors() ||
          lhs.getLayout().getTimeFactors() !=
              result.getLayout().getTimeFactors() ||
          lhs.getLayout().getLaneFactors() !=
              rhs.getLayout().getLaneFactors() ||
          lhs.getLayout().getLaneFactors() !=
              result.getLayout().getLaneFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              rhs.getLayout().getReplicaFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              result.getLayout().getReplicaFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              rhs.getLayout().getFragmentFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              result.getLayout().getFragmentFactors() ||
          lhs.getLayout().getLocalFactors() !=
              rhs.getLayout().getLocalFactors() ||
          lhs.getLayout().getLocalFactors() !=
              result.getLayout().getLocalFactors() ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          result.getLayout().getSew() != resultElement.getWidth() ||
          lhs.getLayout().getLmulEighths() !=
              rhs.getLayout().getLmulEighths() ||
          result.getLayout().getLmulEighths() !=
              2 * lhs.getLayout().getLmulEighths() ||
          lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
          lhs.getLayout().getVl() != result.getLayout().getVl() || !kernel ||
          !kernel.getTarget().getHasWideningInteger() ||
          !llvm::is_contained(
              kernel.getTarget().getLegalLMULEighths().asArrayRef(),
              result.getLayout().getLmulEighths()))
        continue;
      llvm::StringRef instruction;
      if (lhsElement.isUnsigned() && rhsElement.isUnsigned())
        instruction = "rvv.vwmulu.vv";
      else if (lhsElement.isSigned() && rhsElement.isSigned())
        instruction = "rvv.vwmul.vv";
      else if (lhsElement.isSigned())
        instruction = "rvv.vwmulsu.vv";
      else
        instruction = "rvv.vwmulsu.vv.swap";
      rewriter.setInsertionPoint(binary);
      auto fused = rewriter.create<riscv::RVVWidenMultiplyOp>(
          binary.getLoc(), result, lhsWiden.getInput(), rhsWiden.getInput(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-multiply", instruction, instruction,
              lhs.getLayout().getRegisterGroups() +
                  rhs.getLayout().getRegisterGroups(),
              result.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
      riscv_internal::copyOrigin(binary, fused);
      binary.getResult().replaceAllUsesWith(fused.getResult());
      rewriter.eraseOp(binary);
      if (lhsWiden.getResult().use_empty())
        rewriter.eraseOp(lhsWiden);
      if (rhsWiden && rhsWiden != lhsWiden && rhsWiden.getResult().use_empty())
        rewriter.eraseOp(rhsWiden);
    }

    llvm::SmallVector<riscv::BinaryOp> scalarWideningProducts;
    getOperation().walk([&](riscv::BinaryOp binary) {
      if (binary.getKind() == "mul")
        scalarWideningProducts.push_back(binary);
    });
    for (riscv::BinaryOp binary : scalarWideningProducts) {
      riscv::WidenOp widen =
          binary.getLhs().getDefiningOp<riscv::WidenOp>();
      mlir::Value scalar = binary.getRhs();
      if (!widen) {
        widen = binary.getRhs().getDefiningOp<riscv::WidenOp>();
        scalar = binary.getLhs();
      }
      auto input = widen ? mlir::dyn_cast<riscv::ValueType>(
                               widen.getInput().getType())
                         : riscv::ValueType();
      auto widened = widen ? mlir::dyn_cast<riscv::ValueType>(
                                 widen.getResult().getType())
                           : riscv::ValueType();
      auto result =
          mlir::dyn_cast<riscv::ValueType>(binary.getResult().getType());
      auto constant = scalar.getDefiningOp<riscv::ConstantOp>();
      auto integer = constant
                         ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                         : mlir::IntegerAttr();
      auto inputElement =
          input ? mlir::dyn_cast<mlir::IntegerType>(input.getElementType())
                : mlir::IntegerType();
      auto scalarElement = mlir::dyn_cast<mlir::IntegerType>(scalar.getType());
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      bool scalarFits = false;
      if (integer && inputElement)
        scalarFits = inputElement.isSigned()
                         ? integer.getValue().isSignedIntN(inputElement.getWidth())
                         : integer.getValue().isIntN(inputElement.getWidth());
      if (!input || !widened || !result || !constant || !integer ||
          !inputElement || !scalarElement || !resultElement || !scalarFits ||
          inputElement.isSignless() || scalarElement.isSignless() ||
          resultElement.isSignless() ||
          scalarElement != widened.getElementType() ||
          resultElement != widened.getElementType() ||
          result.getShape() != input.getShape() ||
          result.getAxisIds() != input.getAxisIds() ||
          result.getLayout().getTimeFactors() !=
              input.getLayout().getTimeFactors() ||
          result.getLayout().getLaneFactors() !=
              input.getLayout().getLaneFactors() ||
          result.getLayout().getReplicaFactors() !=
              input.getLayout().getReplicaFactors() ||
          result.getLayout().getFragmentFactors() !=
              input.getLayout().getFragmentFactors() ||
          result.getLayout().getLocalFactors() !=
              input.getLayout().getLocalFactors() ||
          resultElement.getWidth() != 2 * inputElement.getWidth() ||
          (inputElement.isUnsigned() && scalarElement.isSigned()))
        continue;
      llvm::StringRef instruction;
      if (inputElement.isUnsigned())
        instruction = "rvv.vwmulu.vx";
      else if (scalarElement.isSigned())
        instruction = "rvv.vwmul.vx";
      else
        instruction = "rvv.vwmulsu.vx";
      rewriter.setInsertionPoint(binary);
      auto narrowConstant = rewriter.create<riscv::ConstantOp>(
          binary.getLoc(), inputElement,
          rewriter.getIntegerAttr(inputElement, integer.getInt()));
      auto fused = rewriter.create<riscv::RVVWidenScalarMultiplyOp>(
          binary.getLoc(), result, widen.getInput(), narrowConstant.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-scalar-multiply", instruction,
              instruction, input.getLayout().getRegisterGroups(),
              result.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
      riscv_internal::copyOrigin(binary, fused);
      binary.getResult().replaceAllUsesWith(fused.getResult());
      rewriter.eraseOp(binary);
      if (widen.getResult().use_empty())
        rewriter.eraseOp(widen);
      if (constant.getResult().use_empty())
        rewriter.eraseOp(constant);
    }

    llvm::SmallVector<riscv::RVVWidenAccumulateOp> zeroSeededAccumulates;
    getOperation().walk([&](riscv::RVVWidenAccumulateOp accumulate) {
      zeroSeededAccumulates.push_back(accumulate);
    });
    for (riscv::RVVWidenAccumulateOp accumulate : zeroSeededAccumulates) {
      auto splat =
          accumulate.getAccumulator().getDefiningOp<riscv::RVVSplatOp>();
      auto constant = splat
                          ? splat.getScalar().getDefiningOp<riscv::ConstantOp>()
                          : riscv::ConstantOp();
      auto integer = constant
                         ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                         : mlir::IntegerAttr();
      if (!splat || !constant || !integer || integer.getInt() != 0 ||
          !splat.getResult().hasOneUse())
        continue;
      auto lhs = accumulate.getLhs().getType();
      auto rhs = accumulate.getRhs().getType();
      auto result = accumulate.getResult().getType();
      auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
      auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
      auto resultElement =
          mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
      auto kernel = accumulate->getParentOfType<riscv::KernelOp>();
      if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
          rhsElement.isSignless() || !resultElement ||
          resultElement.isSignless() ||
          (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
          lhs.getShape() != rhs.getShape() || lhs.getShape() != result.getShape() ||
          lhs.getAxisIds() != rhs.getAxisIds() ||
          lhs.getAxisIds() != result.getAxisIds() ||
          lhs.getLayout().getTimeFactors() !=
              rhs.getLayout().getTimeFactors() ||
          lhs.getLayout().getTimeFactors() !=
              result.getLayout().getTimeFactors() ||
          lhs.getLayout().getLaneFactors() !=
              rhs.getLayout().getLaneFactors() ||
          lhs.getLayout().getLaneFactors() !=
              result.getLayout().getLaneFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              rhs.getLayout().getReplicaFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              result.getLayout().getReplicaFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              rhs.getLayout().getFragmentFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              result.getLayout().getFragmentFactors() ||
          lhs.getLayout().getLocalFactors() !=
              rhs.getLayout().getLocalFactors() ||
          lhs.getLayout().getLocalFactors() !=
              result.getLayout().getLocalFactors() ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          resultElement.getWidth() != 2 * lhs.getLayout().getSew() ||
          result.getLayout().getSew() != resultElement.getWidth() ||
          lhs.getLayout().getLmulEighths() !=
              rhs.getLayout().getLmulEighths() ||
          result.getLayout().getLmulEighths() !=
              2 * lhs.getLayout().getLmulEighths() ||
          lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
          lhs.getLayout().getVl() != result.getLayout().getVl() || !kernel ||
          !kernel.getTarget().getHasWideningInteger() ||
          !llvm::is_contained(
              kernel.getTarget().getLegalLMULEighths().asArrayRef(),
              result.getLayout().getLmulEighths()))
        continue;
      llvm::StringRef instruction;
      if (lhsElement.isSigned() && rhsElement.isSigned())
        instruction = "rvv.vwmul.vv";
      else if (lhsElement.isSigned())
        instruction = "rvv.vwmulsu.vv";
      else
        instruction = "rvv.vwmulsu.vv.swap";
      rewriter.setInsertionPoint(accumulate);
      auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
          accumulate.getLoc(), accumulate.getResult().getType(),
          accumulate.getLhs(), accumulate.getRhs(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-multiply", instruction, instruction,
              lhs.getLayout().getRegisterGroups() +
                  rhs.getLayout().getRegisterGroups(),
              accumulate.getResult().getType().getLayout().getRegisterGroups(),
              0, 0, "none", "exact"));
      riscv_internal::copyOrigin(accumulate, product);
      accumulate.getResult().replaceAllUsesWith(product.getResult());
      rewriter.eraseOp(accumulate);
      if (splat.getResult().use_empty())
        rewriter.eraseOp(splat);
      if (constant.getResult().use_empty())
        rewriter.eraseOp(constant);
    }

    llvm::SmallVector<riscv::ReduceOp> nestedScaledReductions;
    llvm::DenseSet<mlir::Operation *> matchedNestedDots;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      auto match = matchReplicaScaledDotReduction(reduce);
      auto plan = match ? match->dot.getNestedPartialPlanAttr()
                        : riscv::NestedPartialPlanAttr();
      if (match && plan && hasTopology(match->dot, "nested_scaled_stream") &&
          match->reducedAxes == llvm::SmallVector<int64_t>(
                                     plan.getPartialAxes().asArrayRef())) {
        nestedScaledReductions.push_back(reduce);
        matchedNestedDots.insert(match->dot.getOperation());
      }
    });
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      if (!hasTopology(dot, "nested_scaled_stream") ||
          matchedNestedDots.contains(dot.getOperation()))
        return;
      auto diagnostic = dot.emitError(
          "nested topology lost its scale/reduction chain before materialization; users=");
      for (mlir::Operation *user : dot.getResult().getUsers())
        diagnostic << user->getName().getStringRef() << " ";
    });
    llvm::SmallVector<mlir::Value> nestedDeadRoots;
    llvm::SmallVector<std::pair<mlir::scf::ForOp, mlir::Value>>
        deferredPostLoopScales;
    for (riscv::ReduceOp reduce : nestedScaledReductions) {
      if (!reduce || !reduce->getBlock())
        continue;
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match || match->reducedAxes.empty() || match->dot.getOver().size() < 2)
        continue;
      riscv::RVVWidenDotOp dot = match->dot;
      auto nestedPlan = dot.getNestedPartialPlanAttr();
      const int64_t windowAxis =
          nestedPlan ? nestedPlan.getWindowAxis() : int64_t{0};
      const int64_t streams =
          nestedPlan ? nestedPlan.getIssueStreams() : int64_t{0};
      const int64_t windowExtent =
          nestedPlan ? nestedPlan.getWindowExtent() : int64_t{0};
      const int64_t partialSlots =
          nestedPlan ? nestedPlan.getPartialSlots() : int64_t{0};
      auto partialAxes = nestedPlan ? nestedPlan.getPartialAxes()
                                    : mlir::DenseI64ArrayAttr();
      auto partialAxisExtents =
          nestedPlan ? nestedPlan.getPartialAxisExtents()
                     : mlir::DenseI64ArrayAttr();
      auto combineSetTypes = nestedPlan ? nestedPlan.getCombineSetTypes()
                                        : mlir::ArrayAttr();
      auto combineAxes = nestedPlan ? nestedPlan.getCombineAxes()
                                    : mlir::DenseI64ArrayAttr();
      auto combineArities = nestedPlan ? nestedPlan.getCombineArities()
                                       : mlir::DenseI64ArrayAttr();
      auto issueLhsType =
          nestedPlan ? mlir::dyn_cast<riscv::ValueType>(
                           nestedPlan.getIssueLhsType())
                     : riscv::ValueType();
      auto issueRhsType =
          nestedPlan ? mlir::dyn_cast<riscv::ValueType>(
                           nestedPlan.getIssueRhsType())
                     : riscv::ValueType();
      auto scaleReplicaType =
          nestedPlan ? mlir::dyn_cast<riscv::ValueType>(
                           nestedPlan.getScaleReplicaType())
                     : riscv::ValueType();
      auto sourceSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getSourceSetType())
                     : riscv::PartialSetType();
      auto repackedSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getRepackedSetType())
                     : riscv::PartialSetType();
      auto reducedSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getReducedSetType())
                     : riscv::PartialSetType();
      auto scaleCombinedSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getScaleCombinedSetType())
                     : riscv::PartialSetType();
      mlir::Type finalType = reduce.getResult().getType();
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(finalType));
      auto finalValue = mlir::dyn_cast<riscv::ValueType>(finalType);
      const int64_t outputReplicas =
          nestedPlan ? nestedPlan.getOutputReplicas() : int64_t{0};
      auto finalReplicas =
          finalValue ? riscv_internal::staticProduct(
                           finalValue.getLayout().getReplicaFactors().asArrayRef())
                     : std::optional<int64_t>(1);
      const bool closedScalarFinal =
          !finalValue ||
          (finalValue.getLayout().getCarrier() == "scalar" &&
           product(finalValue.getLayout().getTimeFactors()) == 1 &&
           product(finalValue.getLayout().getLaneFactors()) == 1 &&
           product(finalValue.getLayout().getReplicaFactors()) > 0 &&
           product(finalValue.getLayout().getFragmentFactors()) == 1 &&
           product(finalValue.getLayout().getLocalFactors()) == 1);
      if (!nestedPlan || windowAxis <= 0 || streams <= 0 || windowExtent <= 0 ||
          partialSlots < windowExtent || partialSlots % windowExtent ||
          outputReplicas <= 0 || !finalReplicas ||
          *finalReplicas != outputReplicas ||
          nestedPlan.getLhsSourceParts().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getRhsSourceParts().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getLhsLaneOffsets().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getRhsLaneOffsets().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getScaleReplicas().size() !=
              static_cast<size_t>(outputReplicas * partialSlots) ||
          !partialAxes || !partialAxisExtents || !combineSetTypes ||
          !combineAxes || !combineArities ||
          partialAxes.size() != partialAxisExtents.size() ||
          combineSetTypes.size() != combineAxes.size() ||
          combineSetTypes.size() != combineArities.size() ||
          !issueLhsType || !issueRhsType || !scaleReplicaType ||
          !sourceSetType || !repackedSetType || !reducedSetType ||
          !scaleCombinedSetType || !resultElement || !resultElement.isSigned() ||
          resultElement.getWidth() != 32 || !closedScalarFinal) {
        dot.emitError(
            "nested topology reached materialization without its complete typed carrier plan");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(reduce);
      auto zero = rewriter.create<riscv::ConstantOp>(
          reduce.getLoc(), resultElement,
          rewriter.getIntegerAttr(resultElement, 0));
      llvm::SmallVector<mlir::Value> initial(
          static_cast<size_t>(outputReplicas), zero.getResult());
      auto lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), 0);
      auto upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), streams);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          reduce.getLoc(), lower, upper, step, initial,
          [&](mlir::OpBuilder &builder, mlir::Location location,
              mlir::Value, mlir::ValueRange carried) {
            builder.create<mlir::scf::YieldOp>(location, carried);
          });
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll",
                    rewriter.getStringAttr("disable"));
      loop->setAttr("weft.riscv.issue_window",
                    rewriter.getDenseI64ArrayAttr({windowAxis, windowExtent}));
      if (nestedPlan.getIssueUnroll() > 1) {
        loop->setAttr("weft.riscv.unroll_factor",
                      rewriter.getI64IntegerAttr(nestedPlan.getIssueUnroll()));
        auto sourceSet =
            mlir::dyn_cast<riscv::PartialSetType>(nestedPlan.getSourceSetType());
        auto kernel = dot->getParentOfType<riscv::KernelOp>();
        // Operation-major ordering is closed only when every issue owns one
        // complete product carrier.  Reordering a multi-lane issue window
        // extends several partial lifetimes at once instead of merely grouping
        // equivalent target states.
        if (sourceSet && kernel && nestedPlan.getWindowExtent() == 1 &&
            nestedPlan.getIssueUnroll() * sourceSet.getResourceGroups() <
                kernel.getTarget().getVectorRegisters())
          loop->setAttr("weft.riscv.unroll_order",
                        rewriter.getStringAttr("operation-major"));
      }
      if (auto defaultYield = mlir::dyn_cast<mlir::scf::YieldOp>(
              loop.getBody()->getTerminator()))
        rewriter.eraseOp(defaultYield);
      rewriter.setInsertionPointToStart(loop.getBody());

      llvm::DenseMap<mlir::Value, mlir::Value> clones;
      auto target = dot->getParentOfType<riscv::KernelOp>().getTarget();
      auto materializeIssueOperand = [&](mlir::Value source,
                                         riscv::ValueType planned)
          -> mlir::FailureOr<mlir::Value> {
        auto sourceType = mlir::dyn_cast<riscv::ValueType>(source.getType());
        // One issue with an output cohort already consumes the operand's
        // complete logical domain.  Keep that SSA supply and let the explicit
        // layout edge below move its output replicas into the planned carrier;
        // cloning its internal storage windows would project coordinates that
        // are not being iterated.
        if (streams == 1 && outputReplicas > 1 && sourceType &&
            sourceType.getShape() == planned.getShape() &&
            sourceType.getAxisIds() == planned.getAxisIds())
          return source;
        return cloneIssueWindow(source, windowAxis, windowExtent,
                                loop.getInductionVar(), target, rewriter,
                                clones);
      };
      const bool deferScale =
          nestedPlan.getScaleSupplyStage() == "after-partial-reduce";
      const bool sharedScale =
          nestedPlan.getScaleSupplyStage() == "shared-before-issue";
      auto materializeScaleWindow = [&]() -> mlir::FailureOr<mlir::Value> {
        if (!sharedScale)
          return cloneIssueWindow(match->scale, windowAxis, windowExtent,
                                  loop.getInductionVar(), target, rewriter,
                                  clones);
        mlir::Value scaleSource = match->scale;
        auto source = mlir::cast<riscv::ValueType>(scaleSource.getType());
        auto type = issueWindowType(rewriter, target, source, windowAxis,
                                    windowExtent);
        if (!type || source.getShape().size() != 1 ||
            source.getAxisIds()[0] != windowAxis ||
            source.getShape()[0] != streams * windowExtent ||
            source.getLayout().getTimeFactors()[0] != 1 ||
            source.getLayout().getLaneFactors()[0] != source.getShape()[0] ||
            source.getLayout().getValidity() != "full")
          return mlir::failure();
        auto layout = riscv::LayoutAttr::get(
            rewriter.getContext(), "rvv", type.getAxisIds(),
            type.getLayout().getTimeFactors(), type.getLayout().getLaneFactors(),
            type.getLayout().getReplicaFactors(),
            type.getLayout().getFragmentFactors(),
            type.getLayout().getLocalFactors(), source.getLayout().getSew(),
            source.getLayout().getLmulEighths(), windowExtent,
            source.getLayout().getRegisterGroups(), "full");
        type = riscv::ValueType::get(
            rewriter.getContext(), source.getElementType(), type.getShape(),
            type.getAxisIds(), layout);
        auto indexElement = mlir::IntegerType::get(
            rewriter.getContext(), source.getLayout().getSew(),
            mlir::IntegerType::Unsigned);
        auto indexType = riscv::ValueType::get(
            rewriter.getContext(), indexElement, type.getShape(),
            type.getAxisIds(), type.getLayout());
        auto local = rewriter.create<riscv::IotaOp>(
            reduce.getLoc(), indexType, 0, windowExtent,
            riscv_internal::unselectedLeaf(rewriter));
        auto width = rewriter.create<mlir::arith::ConstantIndexOp>(
            reduce.getLoc(), windowExtent);
        auto offset = rewriter.create<mlir::arith::MulIOp>(
            reduce.getLoc(), loop.getInductionVar(), width.getResult());
        auto indexOffset = rewriter.create<riscv::CastOp>(
            reduce.getLoc(), indexElement, offset.getResult(),
            riscv_internal::unselectedLeaf(rewriter));
        auto indices = rewriter.create<riscv::BinaryOp>(
            reduce.getLoc(), indexType, local.getResult(), indexOffset.getResult(),
            "add", riscv_internal::unselectedLeaf(rewriter));
        auto slice = rewriter.create<riscv::LookupOp>(
            reduce.getLoc(), type, scaleSource, indices.getResult(), "in_bounds",
            riscv_internal::unassignedAccess(rewriter),
            riscv_internal::unselectedLeaf(rewriter));
        riscv_internal::copyOrigin(reduce, slice);
        return slice.getResult();
      };
      std::optional<mlir::Value> lhsSlice;
      std::optional<mlir::Value> rhsSlice;
      std::optional<mlir::Value> earlyScaleSlice;
      int64_t failedSupply = -1;
      for (int64_t supply :
           nestedPlan.getIssueMaterializationOrder().asArrayRef()) {
        if (supply == 2 && deferScale)
          continue;
        auto slice = supply == 0
                         ? materializeIssueOperand(dot.getLhs(), issueLhsType)
                     : supply == 1
                         ? materializeIssueOperand(dot.getRhs(), issueRhsType)
                         : materializeScaleWindow();
        if (mlir::failed(slice)) {
          failedSupply = supply;
          break;
        }
        if (supply == 0)
          lhsSlice = *slice;
        else if (supply == 1)
          rhsSlice = *slice;
        else
          earlyScaleSlice = *slice;
      }
      auto lhsType = lhsSlice
                         ? mlir::dyn_cast<riscv::ValueType>(lhsSlice->getType())
                         : riscv::ValueType();
      auto rhsType = rhsSlice
                         ? mlir::dyn_cast<riscv::ValueType>(rhsSlice->getType())
                         : riscv::ValueType();
      if (!lhsSlice || !rhsSlice || !lhsType || !rhsType ||
          (!deferScale && !earlyScaleSlice)) {
        dot.emitError(
            "nested issue-window cloning did not produce its planned supplies; lhs=")
            << lhsType << ", planned_lhs=" << issueLhsType
            << ", rhs=" << rhsType << ", planned_rhs=" << issueRhsType
            << ", failed_supply=" << failedSupply
            << ", scale=" << match->scale;
        rewriter.eraseOp(loop);
        failed = true;
        continue;
      }
      auto materializeIssueLayout = [&](mlir::Value value,
                                        riscv::ValueType source,
                                        riscv::ValueType planned) -> mlir::Value {
        if (source == planned)
          return value;
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            dot.getLoc(), planned, value,
            riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                             planned.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
        if (mlir::Operation *definition = value.getDefiningOp())
          riscv_internal::copyOrigin(definition, conversion);
        return conversion.getResult();
      };
      mlir::Value issueLhs =
          materializeIssueLayout(*lhsSlice, lhsType, issueLhsType);
      mlir::Value issueRhs =
          materializeIssueLayout(*rhsSlice, rhsType, issueRhsType);
      auto materializeScaleReplica = [&](mlir::Value scale)
          -> mlir::FailureOr<mlir::Value> {
        auto sourceType = mlir::dyn_cast<riscv::ValueType>(scale.getType());
        if (!sourceType)
          return mlir::failure();
        if (nestedPlan.getScaleSupply() == "scalar-rematerialize") {
          llvm::DenseMap<mlir::Value, mlir::Value> rematerialized;
          auto rematerializedScale =
              rematerializeScalarReplicas(scale, rewriter, rematerialized);
          if (mlir::failed(rematerializedScale) ||
              (*rematerializedScale).getType() != scaleReplicaType)
            return mlir::failure();
          llvm::DenseSet<mlir::Value> stops{*rematerializedScale};
          llvm::DenseSet<mlir::Operation *> candidates;
          collectDeadChainCandidates(scale, stops, candidates);
          sweepDeadChainCandidates(candidates, rewriter);
          return *rematerializedScale;
        }
        if (sourceType == scaleReplicaType)
          return scale;
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            reduce.getLoc(), scaleReplicaType, scale,
            riscv_internal::layoutConversion(rewriter, sourceType.getLayout(),
                                             scaleReplicaType.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
        if (mlir::Operation *definition = scale.getDefiningOp())
          riscv_internal::copyOrigin(definition, conversion);
        return conversion.getResult();
      };
      std::optional<mlir::Value> scalarScale;
      if (!deferScale) {
        auto materializedScale = materializeScaleReplica(*earlyScaleSlice);
        if (mlir::failed(materializedScale)) {
          dot.emitError(
              "nested carrier materialization disagrees with its selected scale supply");
          rewriter.eraseOp(loop);
          failed = true;
          continue;
        }
        scalarScale = *materializedScale;
      }
      const int64_t reductionAxis = sourceSetType.getReductionAxis();
      auto zeroOperand = rewriter.getDenseI64ArrayAttr({0});
      llvm::SmallVector<int64_t> windowOrder;
      windowOrder.reserve(static_cast<size_t>(partialSlots));
      for (int64_t window = 0; window < partialSlots; ++window)
        windowOrder.push_back(window);
      llvm::SmallVector<mlir::Value> accumulatedOutputs;
      accumulatedOutputs.reserve(static_cast<size_t>(outputReplicas));
      for (int64_t output = 0; output < outputReplicas; ++output) {
        auto lhsPart = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getLhsSourceParts().asArrayRef().slice(output, 1));
        auto rhsPart = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getRhsSourceParts().asArrayRef().slice(output, 1));
        auto lhsOffset = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getLhsLaneOffsets().asArrayRef().slice(output, 1));
        auto rhsOffset = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getRhsLaneOffsets().asArrayRef().slice(output, 1));
        auto sourceSet = rewriter.create<riscv::RVVPartialSetOp>(
            dot.getLoc(), sourceSetType, mlir::ValueRange{issueLhs},
            mlir::ValueRange{issueRhs}, zeroOperand, zeroOperand, lhsPart,
            rhsPart, lhsOffset, rhsOffset, reductionAxis,
            nestedPlan.getMultiplyInstruction(), ownerDomain(reduce),
            nextPartialBirthId++, ownerDomain(reduce),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set",
                issueLhsType.getLayout().getRegisterGroups() +
                    issueRhsType.getLayout().getRegisterGroups(),
                sourceSetType.getResourceGroups(), 0, 0, "none", "exact",
                {reductionAxis, 1}));
        riscv_internal::copyOrigin(dot, sourceSet);
        sinkReplicaSupplies(sourceSet);
        mlir::Value reductionInput = sourceSet.getResult();
        if (partialSlots > 1) {
          auto repackLeaf = mlir::dyn_cast<riscv::LeafAttr>(nestedPlan.getRepackLeaf());
          if (!repackLeaf) {
            dot.emitError("selected partial repack has no legal carrier");
            signalPassFailure();
            return;
          }
          auto repacked = rewriter.create<riscv::RVVPartialRepackOp>(
              dot.getLoc(), repackedSetType, sourceSet.getResult(), partialSlots,
              repackLeaf);
          riscv_internal::copyOrigin(dot, repacked);
          reductionInput = repacked.getResult();
        }
        auto scaleParts = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getScaleReplicas().asArrayRef().slice(
                output * partialSlots, partialSlots));
        auto partialReduced = rewriter.create<riscv::RVVPartialReduceOp>(
            dot.getLoc(), reducedSetType, reductionInput,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-reduce",
                nestedPlan.getReduceInstruction(),
                nestedPlan.getReduceInstruction(),
                repackedSetType.getResourceGroups(),
                reducedSetType.getResourceGroups(), 1, 0, "none", "exact",
                {reductionAxis, partialSlots}));
        riscv_internal::copyOrigin(dot, partialReduced);
        if (!scalarScale) {
          // Reuse reads and their shared pointwise decode, but not width/layout
          // conversions. Sharing a widen would prevent use-local scalar
          // conversion from consuming its narrow source across the product.
          llvm::DenseMap<mlir::Value, mlir::Value> deferredScaleClones;
          for (auto [source, cloned] : clones) {
            auto *producer = cloned.getDefiningOp();
            auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(producer);
            const bool fieldRead =
                extract && riscv_internal::sourceField(extract.getInput());
            if (producer && (fieldRead || mlir::isa<riscv::BinaryOp>(producer) ||
                             (!mlir::isMemoryEffectFree(producer) &&
                              mayMoveReadAcross(producer))))
              deferredScaleClones.try_emplace(source, cloned);
          }
          auto scale = cloneIssueWindow(
              match->scale, windowAxis, windowExtent, loop.getInductionVar(),
              target, rewriter, deferredScaleClones);
          auto materializedScale =
              mlir::succeeded(scale)
                  ? materializeScaleReplica(*scale)
                  : mlir::FailureOr<mlir::Value>(mlir::failure());
          if (mlir::failed(materializedScale)) {
            dot.emitError(
                "nested carrier could not materialize its deferred scale supply");
            failed = true;
            break;
          }
          scalarScale = *materializedScale;
        }
        llvm::SmallVector<mlir::Value> scales(
            static_cast<size_t>(partialSlots), *scalarScale);
        auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
            reduce.getLoc(), scaleCombinedSetType, partialReduced.getResult(),
            scales, scaleParts, rewriter.getDenseI64ArrayAttr(windowOrder),
            rewriter.getDenseI64ArrayAttr({partialAxes.asArrayRef().back()}),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-scale-combine",
                "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                reducedSetType.getResourceGroups(),
                scaleCombinedSetType.getResourceGroups(), 1, 0, "none",
                "exact",
                {reductionAxis, partialSlots, scaleCombinedSetType.getSlots(),
                 partialSlots / scaleCombinedSetType.getSlots()}));
        riscv_internal::copyOrigin(dot, combined);
        mlir::Value combinedPartials = combined.getResult();
        for (auto [typeAttr, logicalAxis, arity] :
             llvm::zip(combineSetTypes, combineAxes.asArrayRef(),
                       combineArities.asArrayRef())) {
          auto combinedType = mlir::cast<riscv::PartialSetType>(
              mlir::cast<mlir::TypeAttr>(typeAttr).getValue());
          auto topologyCombined =
              rewriter.create<riscv::RVVPartialCombineOp>(
                  reduce.getLoc(), combinedType, combinedPartials, arity,
                  "pairwise", rewriter.getDenseI64ArrayAttr({logicalAxis}),
                  riscv_internal::leaf(
                      rewriter, "rvv", "partial-combine",
                      "rvv.partial-combine", "rvv.partial-combine",
                      mlir::cast<riscv::PartialSetType>(
                          combinedPartials.getType())
                          .getResourceGroups(),
                      combinedType.getResourceGroups(), 0, 0, "none", "exact",
                      {reductionAxis, arity, combinedType.getSlots()}));
          riscv_internal::copyOrigin(dot, topologyCombined);
          combinedPartials = topologyCombined.getResult();
        }
        auto finalSetType =
            mlir::cast<riscv::PartialSetType>(combinedPartials.getType());
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            reduce.getLoc(), resultElement, combinedPartials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                nestedPlan.getFinalizeInstruction(),
                nestedPlan.getFinalizeInstruction(),
                finalSetType.getResourceGroups(), 0, 1, 0, "none",
                "exact", {reductionAxis, 1}));
        riscv_internal::copyOrigin(dot, finalized);
        auto accumulated = rewriter.create<riscv::BinaryOp>(
            reduce.getLoc(), resultElement, loop.getRegionIterArg(output),
            finalized.getResult(), "add",
            riscv_internal::unselectedLeaf(rewriter));
        accumulatedOutputs.push_back(accumulated.getResult());
      }
      if (failed) {
        rewriter.eraseOp(loop);
        continue;
      }
      rewriter.create<mlir::scf::YieldOp>(reduce.getLoc(), accumulatedOutputs);

      rewriter.setInsertionPointAfter(loop);
      nestedDeadRoots.push_back(reduce.getInput());
      mlir::Value replacement;
      if (finalValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            reduce.getLoc(), finalValue, loop.getResults(),
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(reduce, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = loop.getResult(0);
      }
      reduce.getResult().replaceAllUsesWith(replacement);
      if (!match->scale.hasOneUse())
        deferredPostLoopScales.emplace_back(loop, match->scale);
      rewriter.eraseOp(reduce);
    }
    llvm::DenseSet<mlir::Value> nestedCleanupStops;
    llvm::DenseSet<mlir::Operation *> nestedCleanupCandidates;
    for (mlir::Value root : nestedDeadRoots)
      collectDeadChainCandidates(root, nestedCleanupStops,
                                 nestedCleanupCandidates);
    sweepDeadChainCandidates(nestedCleanupCandidates, rewriter);
    for (auto [loop, scale] : deferredPostLoopScales)
      placePureSliceAtFirstPostLoopUse(loop, scale);

    llvm::SmallVector<riscv::ReduceOp> replicaPartialReductions;
    llvm::SmallVector<mlir::Value> replicaCleanupRoots;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      if (mlir::isa<mlir::IntegerType>(
              riscv_internal::logicalElement(reduce.getResult().getType())))
        replicaPartialReductions.push_back(reduce);
    });
    for (riscv::ReduceOp reduce : replicaPartialReductions) {
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match)
        continue;
      riscv::RVVWidenDotOp dot = match->dot;
      const bool reduceBeforeScale = hasTopology(dot, "reduced_scaled");
      if (!hasTopology(dot, "scaled") && !reduceBeforeScale)
        continue;
      llvm::SmallVector<int64_t> matchedAxes(match->reducedAxes);
      llvm::sort(matchedAxes);
      matchedAxes.erase(std::unique(matchedAxes.begin(), matchedAxes.end()),
                        matchedAxes.end());
      if (matchedAxes != llvm::SmallVector<int64_t>(
                             dot.getPartialTopology().getPartialAxes().asArrayRef()))
        continue;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto dotResult = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto scaledType =
          mlir::dyn_cast<riscv::ValueType>(match->scaleMultiply.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(reduce.getResult().getType()));
      const int64_t reductionAxis = dot.getOver()[0];
      auto lhsReduction = axisPosition(lhs, reductionAxis);
      auto rhsReduction = axisPosition(rhs, reductionAxis);
      const int64_t slots = dot.getPartialTopology().getPartialSlots();
      const int64_t outputParts =
          dot.getPartialTopology().getOutputReplicas();
      auto finalValue =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      llvm::SmallVector<int64_t> partialAxes(
          dot.getPartialTopology().getPartialAxes().asArrayRef());
      llvm::SmallVector<int64_t> outputAxes(
          dot.getPartialTopology().getOutputAxes().asArrayRef());
      auto layoutPlan = dot.getPartialLayoutPlanAttr();
      auto scalePlan = dot.getScaledPartialPlanAttr();
      auto slotType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getPartialSlotType())
                     : riscv::ValueType();
      bool complete = layoutPlan && scalePlan &&
                      scalePlan.getRealization() ==
                          dot.getPartialTopology().getKind() &&
                      dotResult && scaledType && resultElement &&
                      resultElement.isSigned() && resultElement.getWidth() == 32 &&
                      lhsReduction && rhsReduction && slotType && slots >= 2 &&
                      ((slots & (slots - 1)) == 0) && outputParts > 0 &&
                      dotResult.getLayout().getCarrier() == "scalar" &&
                      dotResult.getShape() == scaledType.getShape() &&
                      dotResult.getAxisIds() == scaledType.getAxisIds() &&
                      llvm::all_of(dotResult.getAxisIds().asArrayRef(),
                                   [&](int64_t axis) {
                                     return llvm::is_contained(partialAxes, axis) ||
                                            llvm::is_contained(outputAxes, axis);
                                   }) &&
                      replicaProductForAxes(dotResult, partialAxes) == slots &&
                      replicaProductForAxes(dotResult, outputAxes) == outputParts &&
                      ((!finalValue && outputAxes.empty() && outputParts == 1) ||
                       (finalValue &&
                        finalValue.getAxisIds().asArrayRef() ==
                            llvm::ArrayRef<int64_t>(outputAxes))) &&
                      lhs.getLayout().getTimeFactors()[*lhsReduction] == 1 &&
                      rhs.getLayout().getTimeFactors()[*rhsReduction] == 1;
      if (complete) {
        for (size_t position = 0; position < dotResult.getShape().size(); ++position)
          complete &= dotResult.getLayout().getTimeFactors()[position] == 1 &&
                      dotResult.getLayout().getLaneFactors()[position] == 1 &&
                      dotResult.getLayout().getReplicaFactors()[position] > 0;
      }
      if (!complete)
        continue;

      rewriter.setInsertionPoint(reduce);
      mlir::Value scalarScale;
      auto plannedScalarScale = mlir::dyn_cast<riscv::ValueType>(
          scalePlan.getScalarScaleType());
      if (scalePlan.getScaleSupply() == "scalar-rematerialize") {
        llvm::DenseMap<mlir::Value, mlir::Value> rematerialized;
        auto rematerializedScale = rematerializeScalarReplicas(
            match->scale, rewriter, rematerialized);
        if (mlir::succeeded(rematerializedScale) &&
            (*rematerializedScale).getType() == plannedScalarScale)
          scalarScale = *rematerializedScale;
      } else if (scalePlan.getScaleSupply() == "vector-convert") {
        auto source =
            mlir::dyn_cast<riscv::ValueType>(match->scale.getType());
        if (source && plannedScalarScale) {
          if (source == plannedScalarScale) {
            scalarScale = match->scale;
          } else {
          auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
              reduce.getLoc(), plannedScalarScale, match->scale,
              riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                               plannedScalarScale.getLayout()),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(rewriter));
          if (mlir::Operation *definition = match->scale.getDefiningOp())
            riscv_internal::copyOrigin(definition, conversion);
          scalarScale = conversion.getResult();
          }
        }
      }
      auto scalarScaleType =
          scalarScale
              ? mlir::dyn_cast<riscv::ValueType>(scalarScale.getType())
              : riscv::ValueType();
      auto scaleParts =
          scalarScaleType
              ? riscv_internal::staticProduct(
                    scalarScaleType.getLayout()
                        .getReplicaFactors()
                        .asArrayRef())
              : std::optional<int64_t>();
      auto scaleElement =
          scalarScaleType
              ? mlir::dyn_cast<mlir::IntegerType>(
                    scalarScaleType.getElementType())
              : mlir::IntegerType();
      if (!scalarScaleType || scalarScaleType != plannedScalarScale ||
          scalarScaleType.getLayout().getCarrier() != "scalar" ||
          !scaleParts || *scaleParts < slots || !scaleElement ||
          !scaleElement.isSigned() || scaleElement.getWidth() != 32)
        continue;

      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      auto widenedSlot = mlir::dyn_cast<riscv::ValueType>(
          layoutPlan.getWidenedSlotType());
      auto plannedNarrowScale = mlir::dyn_cast<riscv::ValueType>(
          scalePlan.getNarrowScaleType());
      auto narrowScale = plannedNarrowScale
                             ? materializeExactNarrowScale(scalarScale, rewriter)
                             : std::optional<mlir::Value>();
      auto narrowScaleType =
          narrowScale
              ? mlir::dyn_cast<riscv::ValueType>((*narrowScale).getType())
              : riscv::ValueType();
      auto narrowScaleParts =
          narrowScaleType
              ? riscv_internal::staticProduct(
                    narrowScaleType.getLayout().getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      const int64_t combineArity =
          dot.getPartialTopology().getCombineArity();
      const int64_t laneSplit = dot.getPartialTopology().getLaneSplit();
      const int64_t sourceSlots = dot.getPartialTopology().getSourceSlots();
      auto plannedSourceSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getSourceSetType());
      auto plannedRepackedSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getRepackedSetType());
      auto plannedReducedSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getReducedSetType());
      auto plannedScaleCombinedSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getScaleCombinedSetType());
      std::optional<ScaledSourceLaneGeometry> sourceGeometry;
      if (laneSplit > 1) {
        auto sourceSlot =
            mlir::dyn_cast<riscv::ValueType>(layoutPlan.getSourceSlotType());
        auto splitSourceSlot = mlir::dyn_cast<riscv::ValueType>(
            layoutPlan.getSplitSourceSlotType());
        auto partialSlot =
            mlir::dyn_cast<riscv::ValueType>(layoutPlan.getPartialSlotType());
        auto reducedSlot =
            mlir::dyn_cast<riscv::ValueType>(layoutPlan.getReducedSlotType());
        if (sourceSlot && splitSourceSlot && partialSlot && reducedSlot &&
            plannedSourceSet && plannedRepackedSet && plannedReducedSet &&
            plannedScaleCombinedSet)
          sourceGeometry = ScaledSourceLaneGeometry{
              sourceSlot, splitSourceSlot, partialSlot, reducedSlot,
              plannedSourceSet.getResourceGroups(),
              plannedReducedSet.getResourceGroups()};
      }
      if (sourceGeometry) {
        slotType = sourceGeometry->partialSlot;
      }
      const int64_t scaleChunkCount =
          combineArity > 0 && slots % combineArity == 0
              ? slots / combineArity
              : 0;
      const int64_t scaleChains = combineArity;
      const int64_t scaleFanout =
          scaleChains > 0 ? slots / scaleChains : 0;
      auto setType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getPartialSetType());
      auto scaledSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getScaledSetType());
      auto directSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getDirectSetType());
      auto reducedSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getReducedSetType());
      auto scaleCombinedSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getScaleCombinedSetType());
      auto fullScaledSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getFullScaledSetType());
      const int64_t setGroups =
          setType ? setType.getResourceGroups() : 0;
      const int64_t scaledGroups =
          scaledSetType ? scaledSetType.getResourceGroups() : 0;
      const int64_t narrowScaleGroups =
          narrowScaleType ? narrowScaleType.getLayout().getRegisterGroups() : 0;
      auto sourceSlot = sourceGeometry ? sourceGeometry->sourceSlot
                                       : riscv::ValueType();
      auto sourceSetType = sourceGeometry ? plannedSourceSet
                                          : riscv::PartialSetType();
      auto repackedSetType = sourceGeometry ? plannedRepackedSet
                                            : riscv::PartialSetType();
      auto reducedSlot = mlir::dyn_cast<riscv::ValueType>(
          layoutPlan.getReducedSlotType());
      const bool wideSourceLegal =
          laneSplit > 1
              ? (sourceGeometry && sourceSetType && repackedSetType &&
                 reducedSlot && reducedSetType && scaleCombinedSetType &&
                 sourceSetType.getTermsPerSlot() % laneSplit == 0)
              : !reduceBeforeScale ||
                    (directSetType && reducedSetType && scaleCombinedSetType);
      const bool narrowScaleLegal =
          reduceBeforeScale || laneSplit > 1 ||
          (widenedSlot && narrowScale && narrowScaleType == plannedNarrowScale &&
           narrowScaleParts &&
           *narrowScaleParts >= slots && scaleChunkCount > 0 && setGroups > 0 &&
           scaledGroups > 0 && scaledSetType && fullScaledSetType);
      if (!kernel || !wideSourceLegal || !narrowScaleLegal ||
          scalePlan.getResourceGroups() >
              kernel.getTarget().getVectorRegisters())
        continue;

      rewriter.setInsertionPoint(reduce);
      mlir::Value sourceLhs = dot.getLhs();
      mlir::Value sourceRhs = dot.getRhs();
      if (laneSplit > 1) {
        auto plannedLhs = mlir::dyn_cast<riscv::ValueType>(
            layoutPlan.getSourceLhsType());
        auto plannedRhs = mlir::dyn_cast<riscv::ValueType>(
            layoutPlan.getSourceRhsType());
        if (!plannedLhs || !plannedRhs)
          continue;
        auto materializeOperandLayout = [&](mlir::Value input,
                                            riscv::ValueType planned)
            -> mlir::Value {
          if (input.getType() == planned)
            return input;
          auto source = mlir::cast<riscv::ValueType>(input.getType());
          auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
              reduce.getLoc(), planned, input,
              riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                               planned.getLayout()),
              riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
          if (mlir::Operation *definition = input.getDefiningOp())
            riscv_internal::copyOrigin(definition, conversion);
          return conversion.getResult();
        };
        sourceLhs = materializeOperandLayout(sourceLhs, plannedLhs);
        sourceRhs = materializeOperandLayout(sourceRhs, plannedRhs);
      }
      llvm::SmallVector<mlir::Value> scalarParts;
      bool outputComplete = true;
      for (int64_t outputPart = 0; outputPart < outputParts; ++outputPart) {
        llvm::SmallVector<int64_t> lhsParts;
        llvm::SmallVector<int64_t> rhsParts;
        llvm::SmallVector<int64_t> lhsLaneOffsets;
        llvm::SmallVector<int64_t> rhsLaneOffsets;
        llvm::SmallVector<int64_t> scaleReplicas;
        for (int64_t slot = 0; slot < slots; ++slot) {
          auto resultPart = composeReplicaPart(dotResult, outputAxes, outputPart,
                                               partialAxes, slot);
          const int64_t planned =
              resultPart ? *resultPart * dot.getReductionStreams() : -1;
          auto lhsPart =
              planned >= 0 &&
                      planned < static_cast<int64_t>(dot.getLhsParts().size())
                  ? std::optional<int64_t>(dot.getLhsParts()[planned])
                  : std::nullopt;
          auto rhsPart =
              planned >= 0 &&
                      planned < static_cast<int64_t>(dot.getRhsParts().size())
                  ? std::optional<int64_t>(dot.getRhsParts()[planned])
                  : std::nullopt;
          auto scalePart = resultPart
                               ? projectReplica(scalarScaleType, dotResult,
                                                *resultPart)
                               : std::optional<int64_t>();
          auto lhsLaneOffset =
              resultPart &&
                      *resultPart <
                          static_cast<int64_t>(dot.getLhsLaneOffsets().size())
                  ? std::optional<int64_t>(
                        dot.getLhsLaneOffsets()[*resultPart])
                  : std::nullopt;
          auto rhsLaneOffset =
              resultPart &&
                      *resultPart <
                          static_cast<int64_t>(dot.getRhsLaneOffsets().size())
                  ? std::optional<int64_t>(
                        dot.getRhsLaneOffsets()[*resultPart])
                  : std::nullopt;
          if (!lhsPart || !rhsPart || !scalePart || !lhsLaneOffset ||
              !rhsLaneOffset) {
            outputComplete = false;
            break;
          }
          lhsParts.push_back(*lhsPart);
          rhsParts.push_back(*rhsPart);
          lhsLaneOffsets.push_back(*lhsLaneOffset);
          rhsLaneOffsets.push_back(*rhsLaneOffset);
          scaleReplicas.push_back(*scalePart);
        }
        if (!outputComplete)
          break;

        auto combineReducedPartials = [&](mlir::Value reducedPartials) {
          llvm::SmallVector<mlir::Value> scales(slots, scalarScale);
          auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
              reduce.getLoc(), scaleCombinedSetType, reducedPartials, scales,
              rewriter.getDenseI64ArrayAttr(scaleReplicas),
              dot.getPartialTopology().getSlotOrder(),
              rewriter.getDenseI64ArrayAttr(partialAxes),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-scale-combine",
                  "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                  reducedSetType.getResourceGroups(),
                  scaleCombinedSetType.getResourceGroups(), 1, 0, "none",
                  "exact",
                  {reductionAxis, slots, scaleChains, scaleFanout}));
          riscv_internal::copyOrigin(dot, combined);
          return combined.getResult();
        };

        mlir::Value partials;
        if (laneSplit > 1) {
          if (!layoutPlan) {
            outputComplete = false;
            break;
          }
          const int64_t begin = outputPart * sourceSlots;
          auto sourceLhsParts = layoutPlan.getSourceLhsParts()
                                    .asArrayRef()
                                    .slice(begin, sourceSlots);
          auto sourceRhsParts = layoutPlan.getSourceRhsParts()
                                    .asArrayRef()
                                    .slice(begin, sourceSlots);
          auto sourceLhsOffsets = layoutPlan.getSourceLhsOffsets()
                                      .asArrayRef()
                                      .slice(begin, sourceSlots);
          auto sourceRhsOffsets = layoutPlan.getSourceRhsOffsets()
                                      .asArrayRef()
                                      .slice(begin, sourceSlots);
          llvm::SmallVector<int64_t> sourceOperands(sourceSlots, 0);
          auto sourceSet = rewriter.create<riscv::RVVPartialSetOp>(
              reduce.getLoc(), sourceSetType,
              mlir::ValueRange{sourceLhs}, mlir::ValueRange{sourceRhs},
              rewriter.getDenseI64ArrayAttr(sourceOperands),
              rewriter.getDenseI64ArrayAttr(sourceOperands),
              rewriter.getDenseI64ArrayAttr(sourceLhsParts),
              rewriter.getDenseI64ArrayAttr(sourceRhsParts),
              rewriter.getDenseI64ArrayAttr(sourceLhsOffsets),
              rewriter.getDenseI64ArrayAttr(sourceRhsOffsets), reductionAxis,
              scalePlan.getMultiplyInstruction(),
              ownerDomain(reduce), nextPartialBirthId++, ownerDomain(reduce),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-set", "rvv.partial-set",
                  "rvv.partial-set", 0, sourceSetType.getResourceGroups(), 0,
                  0, "none", "exact", {reductionAxis, sourceSlots}));
          riscv_internal::copyOrigin(dot, sourceSet);
          sinkReplicaSupplies(sourceSet);
          auto repackLeaf = mlir::dyn_cast<riscv::LeafAttr>(layoutPlan.getRepackLeaf());
          if (!repackLeaf) {
            dot.emitError("selected partial repack has no legal carrier");
            signalPassFailure();
            return;
          }
          auto repacked = rewriter.create<riscv::RVVPartialRepackOp>(
              reduce.getLoc(), repackedSetType, sourceSet.getResult(), laneSplit,
              repackLeaf);
          riscv_internal::copyOrigin(dot, repacked);
          auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
              reduce.getLoc(), reducedSetType, repacked.getResult(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-reduce",
                  scalePlan.getReduceInstruction(),
                  scalePlan.getReduceInstruction(),
                  repackedSetType.getResourceGroups(),
                  reducedSetType.getResourceGroups(), 1, 0, "none", "exact",
                  {reductionAxis, slots}));
          riscv_internal::copyOrigin(dot, reduced);
          partials = combineReducedPartials(reduced.getResult());
        } else if (reduceBeforeScale) {
          llvm::SmallVector<int64_t> operandSlots(slots, 0);
          auto set = rewriter.create<riscv::RVVPartialSetOp>(
              reduce.getLoc(), directSetType, mlir::ValueRange{dot.getLhs()},
              mlir::ValueRange{dot.getRhs()},
              rewriter.getDenseI64ArrayAttr(operandSlots),
              rewriter.getDenseI64ArrayAttr(operandSlots),
              rewriter.getDenseI64ArrayAttr(lhsParts),
              rewriter.getDenseI64ArrayAttr(rhsParts),
              rewriter.getDenseI64ArrayAttr(lhsLaneOffsets),
              rewriter.getDenseI64ArrayAttr(rhsLaneOffsets), reductionAxis,
              scalePlan.getMultiplyInstruction(),
              ownerDomain(reduce), nextPartialBirthId++, ownerDomain(reduce),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-set", "rvv.partial-set",
                  "rvv.partial-set", 0, directSetType.getResourceGroups(), 0,
                  0, "none", "exact", {reductionAxis, slots}));
          riscv_internal::copyOrigin(dot, set);
          sinkReplicaSupplies(set);
          auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
              reduce.getLoc(), reducedSetType, set.getResult(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-reduce",
                  scalePlan.getReduceInstruction(),
                  scalePlan.getReduceInstruction(),
                  directSetType.getResourceGroups(),
                  reducedSetType.getResourceGroups(), 1, 0, "none", "exact",
                  {reductionAxis, slots}));
          riscv_internal::copyOrigin(dot, reduced);
          partials = combineReducedPartials(reduced.getResult());
        } else {
          llvm::SmallVector<mlir::Value> chunkPartials;
          for (int64_t first = 0; first < slots; first += combineArity) {
            llvm::SmallVector<int64_t> operandSlots(combineArity, 0);
            auto slice = [&](llvm::ArrayRef<int64_t> values) {
              return rewriter.getDenseI64ArrayAttr(
                  values.slice(first, combineArity));
            };
            auto set = rewriter.create<riscv::RVVPartialSetOp>(
                reduce.getLoc(), setType, mlir::ValueRange{dot.getLhs()},
                mlir::ValueRange{dot.getRhs()},
                rewriter.getDenseI64ArrayAttr(operandSlots),
                rewriter.getDenseI64ArrayAttr(operandSlots), slice(lhsParts),
                slice(rhsParts), slice(lhsLaneOffsets), slice(rhsLaneOffsets),
                reductionAxis, scalePlan.getMultiplyInstruction(),
                ownerDomain(reduce), nextPartialBirthId++, ownerDomain(reduce),
                riscv_internal::leaf(
                    rewriter, "rvv", "partial-set", "rvv.partial-set",
                    "rvv.partial-set", 0, setGroups, 0, 0, "none", "exact",
                    {reductionAxis, combineArity}));
            riscv_internal::copyOrigin(dot, set);
            sinkReplicaSupplies(set);
            llvm::SmallVector<mlir::Value> scales(combineArity, *narrowScale);
            auto scaled = rewriter.create<riscv::RVVPartialWidenScaleOp>(
                reduce.getLoc(), scaledSetType, set.getResult(), scales,
                slice(scaleReplicas), combineArity,
                rewriter.getDenseI64ArrayAttr(partialAxes),
                riscv_internal::leaf(
                    rewriter, "rvv", "partial-widen-scale",
                    "rvv.partial-widen-scale", "rvv.partial-widen-scale",
                    setGroups + narrowScaleGroups, scaledGroups, 0, 0, "none",
                    "exact", {reductionAxis, combineArity, combineArity}));
            riscv_internal::copyOrigin(dot, scaled);
            chunkPartials.push_back(scaled.getResult());
          }

          partials = chunkPartials.front();
          if (chunkPartials.size() > 1) {
            auto merged = rewriter.create<riscv::RVVPartialMergeOp>(
                reduce.getLoc(), fullScaledSetType, chunkPartials, "pairwise",
                riscv_internal::leaf(
                    rewriter, "rvv", "partial-merge", "rvv.partial-merge",
                    "rvv.partial-merge", scaleChunkCount * scaledGroups,
                    scaledGroups, 0, 0, "none", "exact",
                    {reductionAxis, scaleChunkCount, scaleChunkCount,
                     slots * setType.getTermsPerSlot()}));
            riscv_internal::copyOrigin(dot, merged);
            partials = merged.getResult();
          }
        }
        auto finalType = mlir::cast<riscv::PartialSetType>(partials.getType());
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            reduce.getLoc(), resultElement, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                scalePlan.getFinalizeInstruction(),
                scalePlan.getFinalizeInstruction(),
                finalType.getResourceGroups(), 0, 0, 0, "none",
                "exact", {reductionAxis, 1}));
        riscv_internal::copyOrigin(dot, finalized);
        scalarParts.push_back(finalized.getResult());
      }
      if (!outputComplete ||
          scalarParts.size() != static_cast<size_t>(outputParts))
        continue;
      mlir::Value replacement;
      if (finalValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            reduce.getLoc(), finalValue, scalarParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = scalarParts.front();
      }
      reduce.getResult().replaceAllUsesWith(replacement);

      mlir::Value oldScale = match->scale;
      mlir::Value oldDotSide = match->dotSide;
      llvm::SmallVector<mlir::Operation *> replacedChain;
      llvm::DenseSet<mlir::Operation *> seenChain;
      auto rememberChainOp = [&](mlir::Operation *operation) {
        if (operation && seenChain.insert(operation).second)
          replacedChain.push_back(operation);
      };
      for (riscv::ReduceOp reduction : match->reductions)
        rememberChainOp(reduction);
      for (riscv::ConvertLayoutOp conversion : match->reductionConversions)
        rememberChainOp(conversion);
      rememberChainOp(match->scaleMultiply);
      bool erasedAny = false;
      do {
        erasedAny = false;
        for (mlir::Operation *&operation : replacedChain) {
          if (!operation ||
              !llvm::all_of(operation->getResults(), [](mlir::Value result) {
                return result.use_empty();
              }))
            continue;
          rewriter.eraseOp(operation);
          operation = nullptr;
          erasedAny = true;
        }
      } while (erasedAny);
      if (llvm::any_of(replacedChain,
                       [](mlir::Operation *operation) { return operation; })) {
        reduce.emitError(
            "scaled partial rewrite left a live reduction-chain operation");
        failed = true;
        break;
      }
      replicaCleanupRoots.push_back(oldScale);
      replicaCleanupRoots.push_back(oldDotSide);
      continue;

    }
    llvm::DenseSet<mlir::Operation *> replicaCleanupVisited;
    llvm::DenseSet<mlir::Value> replicaCleanupStops;
    for (mlir::Value root : replicaCleanupRoots)
      eraseDeadChain(root, replicaCleanupStops, replicaCleanupVisited,
                     rewriter);

    // The planner sees one unreplicated loop-carried contribution and freezes
    // the complete issue cohort before any cloning.  Expand that exact cohort
    // here so the materializer consumes its typed plan rather than asking the
    // generic Level unroller to duplicate an unplanned reduction graph.
    llvm::SmallVector<mlir::scf::ForOp> levelScaledSeeds;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      auto seed = matchLevelScaledLoopSeed(loop);
      if (seed && hasTopology(seed->dot, "level_scaled"))
        levelScaledSeeds.push_back(loop);
    });
    for (mlir::scf::ForOp loop : levelScaledSeeds) {
      auto seed = matchLevelScaledLoopSeed(loop);
      const int64_t factor = seed ? seed->dot.getPartialTopology().getPartialSlots()
                                  : int64_t{0};
      auto expanded = seed ? expandPlannedLevelScaledIssueLoop(loop, factor,
                                                               rewriter)
                           : mlir::FailureOr<mlir::scf::ForOp>(mlir::failure());
      if (mlir::failed(expanded)) {
        loop.emitError(
            "selected level-scaled topology cannot mechanically expand its exact issue cohort");
        failed = true;
      }
    }
    llvm::SmallVector<mlir::scf::ForOp> partialLoops;
    getOperation().walk(
        [&](mlir::scf::ForOp loop) { partialLoops.push_back(loop); });
    for (mlir::scf::ForOp loop : partialLoops) {
      auto matchedContributions = matchLevelScaledLoop(loop);
      if (!matchedContributions ||
          !llvm::all_of(*matchedContributions,
                        [](const ScaledPartialContribution &contribution) {
                          return hasTopology(contribution.dot, "level_scaled");
                        }))
        continue;
      llvm::SmallVector<ScaledPartialContribution> contributions =
          std::move(*matchedContributions);
      const int64_t slots = contributions.front().dot.getPartialUnroll();
      auto yield =
          mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());

      const int64_t reductionAxis = contributions.front().dot.getOver()[0];
      mlir::Type carry = loop.getRegionIterArg(0).getType();
      auto carryType = mlir::dyn_cast<riscv::ValueType>(carry);
      auto carryElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(carry));
      auto carryParts =
          carryType
              ? riscv_internal::staticProduct(
                    carryType.getLayout().getReplicaFactors().asArrayRef())
              : carryElement && carryElement.isSigned() &&
                        carryElement.getWidth() == 32
                    ? std::optional<int64_t>(1)
                    : std::optional<int64_t>();
      auto firstLayoutPlan =
          contributions.front().dot.getPartialLayoutPlanAttr();
      auto firstTopology = contributions.front().dot.getPartialTopology();
      auto firstCombinePlan =
          contributions.front().dot->getAttrOfType<riscv::PartialCombinePlanAttr>(
              "partial_combine_plan");
      auto slotType =
          firstLayoutPlan
              ? mlir::dyn_cast<riscv::ValueType>(
                    firstLayoutPlan.getPartialSlotType())
              : riscv::ValueType();
      const bool scalarCarry =
          carryType ? carryType.getLayout().getCarrier() == "scalar"
                    : static_cast<bool>(carryElement);
      bool complete = firstLayoutPlan && firstCombinePlan && scalarCarry &&
                      carryParts && *carryParts > 0 && carryElement &&
                      carryElement.isSigned() && carryElement.getWidth() == 32 &&
                      firstCombinePlan.getRealization() == "level_scaled" &&
                      firstCombinePlan.getOutputParts() == *carryParts &&
                      slotType;
      for (ScaledPartialContribution &contribution : contributions) {
        auto combinePlan = contribution.dot->getAttrOfType<
            riscv::PartialCombinePlanAttr>("partial_combine_plan");
        auto issuePlan = contribution.dot.getSequentialPartialPlanAttr();
        auto issueAccumulator =
            issuePlan
                ? mlir::dyn_cast<riscv::ValueType>(issuePlan.getAccumulatorType())
                : riscv::ValueType();
        complete &= contribution.dot.getPartialUnroll() == slots &&
                    contribution.dot.getOver().size() == 1 &&
                    contribution.dot.getOver()[0] == reductionAxis &&
                    contribution.dot.getPartialTopology() == firstTopology &&
                    contribution.dot.getResult().getType() == carry &&
                    contribution.scaleMultiply.getResult().getType() == carry &&
                    contribution.carryAdd.getResult().getType() == carry &&
                    issuePlan &&
                    issuePlan.getRealization() == "fused_partial" &&
                    issuePlan.getIssueCount() > 0 &&
                    issueAccumulator == slotType &&
                    contribution.dot.getPartialLayoutPlanAttr() &&
                    combinePlan &&
                    combinePlan.getRealization() == "level_scaled" &&
                    combinePlan.getSourceSetType() ==
                        firstCombinePlan.getSourceSetType() &&
                    combinePlan.getReducedSetType() ==
                        firstCombinePlan.getReducedSetType() &&
                    combinePlan.getFinalSetType() ==
                        firstCombinePlan.getFinalSetType() &&
                    combinePlan.getMultiplyInstruction() ==
                        firstCombinePlan.getMultiplyInstruction() &&
                    combinePlan.getFinalizeInstruction() ==
                        firstCombinePlan.getFinalizeInstruction() &&
                    combinePlan.getLhsParts().size() ==
                        static_cast<size_t>(*carryParts) &&
                    combinePlan.getRhsParts().size() ==
                        static_cast<size_t>(*carryParts) &&
                    combinePlan.getScaleParts().size() ==
                        static_cast<size_t>(*carryParts) &&
                    mlir::dyn_cast<riscv::ValueType>(
                        contribution.dot.getPartialLayoutPlanAttr()
                            .getPartialSlotType()) == slotType;
      }
      if (!complete)
        continue;

      struct OutputPartialOperands {
        llvm::SmallVector<mlir::Value> partials;
        llvm::SmallVector<mlir::Value> scales;
        llvm::SmallVector<int64_t> scaleReplicas;
      };
      llvm::SmallVector<OutputPartialOperands, 1> outputOperands(*carryParts);
      auto setType = mlir::cast<riscv::PartialSetType>(
          firstCombinePlan.getSourceSetType());
      auto reducedType = mlir::cast<riscv::PartialSetType>(
          firstCombinePlan.getReducedSetType());
      auto combinedType = mlir::cast<riscv::PartialSetType>(
          firstCombinePlan.getFinalSetType());
      const int64_t chains = firstTopology.getCombineArity();
      if (chains <= 0 || slots % chains ||
          firstTopology.getSlotOrder().size() != static_cast<size_t>(slots))
        continue;
      const int64_t fanout = slots / chains;

      rewriter.setInsertionPoint(yield);
      for (ScaledPartialContribution &contribution : contributions) {
        auto combinePlan = contribution.dot->getAttrOfType<
            riscv::PartialCombinePlanAttr>("partial_combine_plan");
        auto issuePlan = contribution.dot.getSequentialPartialPlanAttr();
        auto lhsIssue = mlir::cast<riscv::ValueType>(issuePlan.getLhsIssueType());
        auto rhsIssue = mlir::cast<riscv::ValueType>(issuePlan.getRhsIssueType());
        const int64_t issueCount = issuePlan.getIssueCount();
        auto kernel = contribution.dot->getParentOfType<riscv::KernelOp>();
        auto materializeIssueOperand =
            [&](mlir::Value source, riscv::ValueType issueType,
                llvm::StringRef supply, mlir::DenseI64ArrayAttr sourceParts,
                mlir::DenseI64ArrayAttr laneOffsets, size_t index,
                int64_t issue) -> mlir::FailureOr<mlir::Value> {
          if (supply == "slice") {
            auto slice = rewriter.create<riscv::RVVIssueSliceOp>(
                yield.getLoc(), issueType, source,
                rewriter.getDenseI64ArrayAttr({sourceParts[index]}),
                rewriter.getDenseI64ArrayAttr({laneOffsets[index]}),
                riscv_internal::leaf(
                    rewriter, "transfer", "issue-slice", "rvv.issue-slice",
                    "rvv.issue-slice", 0,
                    issueType.getLayout().getRegisterGroups(), 0, 0, "none",
                    "exact"));
            riscv_internal::copyOrigin(contribution.dot, slice);
            return slice.getResult();
          }
          auto position = axisPosition(issueType, reductionAxis);
          if (supply != "storage-rematerialize" || !kernel || !position)
            return mlir::failure();
          auto issueIndex = rewriter.create<mlir::arith::ConstantIndexOp>(
              yield.getLoc(), issue);
          llvm::DenseMap<mlir::Value, mlir::Value> clones;
          auto projected = cloneIssueWindow(
              source, reductionAxis, issueType.getShape()[*position],
              issueIndex.getResult(), kernel.getTarget(), rewriter, clones);
          if (mlir::failed(projected) || (*projected).getType() != issueType)
            return mlir::failure();
          return *projected;
        };
        for (int64_t outputPart = 0; outputPart < *carryParts; ++outputPart) {
          mlir::Value partial;
          for (int64_t issue = 0; issue < issueCount; ++issue) {
            const size_t index =
                static_cast<size_t>(outputPart * issueCount + issue);
            auto lhsSlice = materializeIssueOperand(
                contribution.dot.getLhs(), lhsIssue,
                issuePlan.getLhsIssueSupply(),
                issuePlan.getLhsSourceParts(), issuePlan.getLhsLaneOffsets(),
                index, issue);
            auto rhsSlice = materializeIssueOperand(
                contribution.dot.getRhs(), rhsIssue,
                issuePlan.getRhsIssueSupply(),
                issuePlan.getRhsSourceParts(), issuePlan.getRhsLaneOffsets(),
                index, issue);
            if (mlir::failed(lhsSlice) || mlir::failed(rhsSlice)) {
              contribution.dot.emitError(
                  "selected issue supply cannot materialize its frozen storage/view program; output_part=")
                  << outputPart << ", issue=" << issue
                  << ", lhs_supply=" << issuePlan.getLhsIssueSupply()
                  << ", lhs_source=" << contribution.dot.getLhs().getType()
                  << ", lhs_issue=" << lhsIssue
                  << ", rhs_supply=" << issuePlan.getRhsIssueSupply()
                  << ", rhs_source=" << contribution.dot.getRhs().getType()
                  << ", rhs_issue=" << rhsIssue;
              failed = true;
              break;
            }
            if (!partial) {
              auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
                  yield.getLoc(), slotType, *lhsSlice, *rhsSlice,
                  riscv_internal::leaf(
                      rewriter, "rvv", "widen-multiply",
                      issuePlan.getMultiplyInstruction(),
                      issuePlan.getMultiplyInstruction(),
                      lhsIssue.getLayout().getRegisterGroups() +
                          rhsIssue.getLayout().getRegisterGroups(),
                      slotType.getLayout().getRegisterGroups(), 0, 0, "none",
                      "exact"));
              riscv_internal::copyOrigin(contribution.dot, product);
              partial = product.getResult();
            } else {
              auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
                  yield.getLoc(), slotType, *lhsSlice, *rhsSlice, partial,
                  issuePlan.getReductionAxes(),
                  riscv_internal::leaf(
                      rewriter, "rvv", "widen-accumulate",
                      issuePlan.getAccumulateInstruction(),
                      issuePlan.getAccumulateInstruction(),
                      lhsIssue.getLayout().getRegisterGroups() +
                          rhsIssue.getLayout().getRegisterGroups() +
                          slotType.getLayout().getRegisterGroups(),
                      slotType.getLayout().getRegisterGroups(), 0, 0, "none",
                      "agnostic"));
              riscv_internal::copyOrigin(contribution.dot, accumulate);
              partial = accumulate.getResult();
            }
          }
          if (failed)
            break;
          OutputPartialOperands &operands = outputOperands[outputPart];
          operands.partials.push_back(partial);
          operands.scales.push_back(contribution.scale);
          operands.scaleReplicas.push_back(
              combinePlan.getScaleParts()[outputPart]);
        }
        if (failed)
          break;
      }
      if (failed)
        continue;

      llvm::SmallVector<mlir::Value> scalarParts;
      for (OutputPartialOperands &operands : outputOperands) {
        auto set = rewriter.create<riscv::RVVPartialCollectOp>(
            yield.getLoc(), setType, operands.partials,
            ownerDomain(yield), nextPartialBirthId++, ownerDomain(yield),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-collect", "rvv.partial-collect",
                "rvv.partial-collect", setType.getResourceGroups(),
                setType.getResourceGroups(), 0, 0, "none",
                slotType.getLayout().getValidity() == "tail" ? "agnostic"
                                                               : "exact",
                {reductionAxis, slots}));
        riscv_internal::copyOrigin(contributions.front().dot, set);
        auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
            yield.getLoc(), reducedType, set.getResult(),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-reduce",
                mlir::cast<mlir::IntegerType>(slotType.getElementType())
                            .getWidth() == 16
                    ? "rvv.partial-reduce.widen"
                    : "rvv.partial-reduce",
                mlir::cast<mlir::IntegerType>(slotType.getElementType())
                            .getWidth() == 16
                    ? "rvv.partial-reduce.widen"
                    : "rvv.partial-reduce",
                setType.getResourceGroups(), reducedType.getResourceGroups(), 1,
                0, "none", "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(contributions.front().dot, reduced);
        auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
            yield.getLoc(), combinedType, reduced.getResult(), operands.scales,
            rewriter.getDenseI64ArrayAttr(operands.scaleReplicas),
            firstTopology.getSlotOrder(),
            firstTopology.getPartialAxes(),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-scale-combine",
                "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                reducedType.getResourceGroups(),
                combinedType.getResourceGroups(), 1, 0, "none", "exact",
                {reductionAxis, slots, chains, fanout}));
        riscv_internal::copyOrigin(contributions.front().dot, combined);
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            yield.getLoc(), carryElement, combined.getResult(),
            reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                firstCombinePlan.getFinalizeInstruction(),
                firstCombinePlan.getFinalizeInstruction(),
                combinedType.getResourceGroups(), 0, 0, 0, "none", "exact",
                {reductionAxis, chains}));
        riscv_internal::copyOrigin(contributions.front().dot, finalized);
        scalarParts.push_back(finalized.getResult());
      }
      mlir::Value materializedCarry;
      if (carryType) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            yield.getLoc(), carryType, scalarParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(contributions.front().dot, assembled);
        materializedCarry = assembled.getResult();
      } else {
        materializedCarry = scalarParts.front();
      }
      riscv::BinaryOp finalAdd = contributions.back().carryAdd;
      finalAdd->moveBefore(yield);
      rewriter.modifyOpInPlace(finalAdd, [&] {
        finalAdd->setOperand(0, loop.getRegionIterArg(0));
        finalAdd->setOperand(1, materializedCarry);
      });

      llvm::SmallVector<mlir::Value> issueCleanupRoots;
      for (ScaledPartialContribution &contribution :
           llvm::reverse(contributions)) {
        issueCleanupRoots.push_back(contribution.dot.getLhs());
        issueCleanupRoots.push_back(contribution.dot.getRhs());
        if (contribution.carryAdd != finalAdd)
          rewriter.eraseOp(contribution.carryAdd);
        rewriter.eraseOp(contribution.scaleMultiply);
        rewriter.eraseOp(contribution.dot);
      }
      llvm::DenseSet<mlir::Operation *> issueCleanupVisited;
      llvm::DenseSet<mlir::Value> issueCleanupStops;
      for (mlir::Value root : issueCleanupRoots)
        eraseDeadChain(root, issueCleanupStops, issueCleanupVisited, rewriter);
    }

    // A canonical reduction over one surviving replica axis of a widened dot
    // is one register-local accumulation chain followed by one lane reduction.
    // The topology planner freezes the accumulator layout; this pass only
    // projects each replica contribution and instantiates that closed chain.
    llvm::SmallVector<riscv::ReduceOp> replicaReductions;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      auto dot = reduce.getInput().getDefiningOp<riscv::RVVWidenDotOp>();
      if (dot && hasTopology(dot, "replica_reduced"))
        replicaReductions.push_back(reduce);
    });
    for (riscv::ReduceOp reduce : replicaReductions) {
      auto dot = reduce.getInput().getDefiningOp<riscv::RVVWidenDotOp>();
      auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      auto topology = dot ? dot.getPartialTopologyAttr()
                          : riscv::PartialTopologyAttr();
      auto layoutPlan = dot ? dot.getPartialLayoutPlanAttr()
                            : riscv::PartialLayoutPlanAttr();
      auto accumulatorType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getVectorSlotType())
                     : riscv::ValueType();
      if (!dot || !result || !resultElement ||
          !resultElement.isSigned() || resultElement.getWidth() != 32 ||
          !topology || topology.getPartialAxes().size() != 1 ||
          !accumulatorType || dot.getReductionStreams() != 1)
        continue;
      const int64_t replicaAxis = topology.getPartialAxes()[0];
      auto projectedLhs = projectReplicaReductionOperandType(
          rewriter, dot.getLhs().getType(), replicaAxis);
      auto projectedRhs = projectReplicaReductionOperandType(
          rewriter, dot.getRhs().getType(), replicaAxis);
      if (!projectedLhs || !projectedRhs ||
          replicaReducedAccumulatorType(rewriter, projectedLhs, projectedRhs,
                                        dot.getOver()[0]) != accumulatorType)
        continue;

      rewriter.setInsertionPoint(reduce);
      const int64_t reductionAxis = dot.getOver()[0];
      auto partialElement =
          mlir::cast<mlir::IntegerType>(accumulatorType.getElementType());
      auto zero = rewriter.create<riscv::ConstantOp>(
          reduce.getLoc(), partialElement,
          rewriter.getIntegerAttr(partialElement, 0));
      auto initial = rewriter.create<riscv::RVVSplatOp>(
          reduce.getLoc(), accumulatorType, zero.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
              accumulatorType.getLayout().getRegisterGroups()));
      mlir::Value carried = initial.getResult();
      for (int64_t replica = 0; replica < topology.getSourceSlots(); ++replica) {
        auto index = rewriter.create<mlir::arith::ConstantIndexOp>(
            reduce.getLoc(), replica);
        auto lhs = rewriter.create<riscv::ProjectReductionOperandOp>(
            reduce.getLoc(), projectedLhs, dot.getLhs(), index, replicaAxis,
            riscv_internal::leaf(
                rewriter, "transfer", "reduction-projection",
                "rvv.project-reduction-operand",
                "rvv.project-reduction-operand", 0, 0, 1, 0));
        auto rhs = rewriter.create<riscv::ProjectReductionOperandOp>(
            reduce.getLoc(), projectedRhs, dot.getRhs(), index, replicaAxis,
            riscv_internal::leaf(
                rewriter, "transfer", "reduction-projection",
                "rvv.project-reduction-operand",
                "rvv.project-reduction-operand", 0, 0, 1, 0));
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            reduce.getLoc(), accumulatorType, lhs.getResult(), rhs.getResult(),
            carried, rewriter.getDenseI64ArrayAttr({reductionAxis}),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial",
                projectedLhs.getLayout().getRegisterGroups() +
                    projectedRhs.getLayout().getRegisterGroups() +
                    accumulatorType.getLayout().getRegisterGroups(),
                accumulatorType.getLayout().getRegisterGroups(), 0, 0, "none",
                "agnostic"));
        riscv_internal::copyOrigin(dot, lhs);
        riscv_internal::copyOrigin(dot, rhs);
        riscv_internal::copyOrigin(dot, accumulate);
        carried = accumulate.getResult();
      }
      auto finalized = rewriter.create<riscv::RVVFinalizeWidenDotOp>(
          reduce.getLoc(), result, carried,
          rewriter.getDenseI64ArrayAttr({reductionAxis}),
          riscv_internal::leaf(
              rewriter, "rvv", "finalize-widen-dot",
              partialElement.getWidth() == 16 ? "rvv.vwredsum.partial"
                                              : "rvv.vredsum.partial",
              partialElement.getWidth() == 16 ? "rvv.vwredsum.partial"
                                              : "rvv.vredsum.partial",
              accumulatorType.getLayout().getRegisterGroups(), 0, 2, 0,
              "none", "exact", {reductionAxis}));
      riscv_internal::copyOrigin(reduce, finalized);
      reduce.getResult().replaceAllUsesWith(finalized.getResult());
      rewriter.eraseOp(reduce);
      rewriter.eraseOp(dot);
    }

    // Instantiate the already-selected sequential program.  Each issue slice
    // is a typed projection of an existing vector part.  A fused realization
    // carries one widened accumulator across issues; a per-stream realization
    // finalizes each issue and combines the explicit i32 SSA results.
    llvm::SmallVector<riscv::RVVWidenDotOp> sequentialDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      auto plan = dot.getSequentialPartialPlanAttr();
      if (plan && plan.getRealization() != "fused_partial")
        sequentialDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : sequentialDots) {
      auto plan = dot.getSequentialPartialPlanAttr();
      auto lhsIssue = mlir::dyn_cast<riscv::ValueType>(plan.getLhsIssueType());
      auto rhsIssue = mlir::dyn_cast<riscv::ValueType>(plan.getRhsIssueType());
      auto accumulator =
          mlir::dyn_cast<riscv::ValueType>(plan.getAccumulatorType());
      auto partialElement =
          accumulator
              ? mlir::dyn_cast<mlir::IntegerType>(accumulator.getElementType())
              : mlir::IntegerType();
      const int64_t issueCount = plan.getIssueCount();
      const int64_t outputParts =
          issueCount > 0
              ? static_cast<int64_t>(plan.getLhsSourceParts().size()) /
                    issueCount
              : 0;
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          resultValue ? resultValue.getElementType() : dot.getResult().getType());
      const bool fused = plan.getRealization() == "fused";
      const bool perStream = plan.getRealization() == "per_stream";
      const bool matchingTopology =
          (fused && hasTopology(dot, "sequential_fused")) ||
          (perStream && hasTopology(dot, "sequential_per_stream"));
      if (!lhsIssue || !rhsIssue || !accumulator || !partialElement ||
          !resultElement || !resultElement.isSigned() ||
          resultElement.getWidth() != 32 || !matchingTopology ||
          issueCount <= 0 || outputParts <= 0 ||
          plan.getLhsSourceParts().size() !=
              plan.getRhsSourceParts().size() ||
          plan.getLhsSourceParts().size() != plan.getLhsLaneOffsets().size() ||
          plan.getRhsSourceParts().size() != plan.getRhsLaneOffsets().size()) {
        dot.emitError(
            "selected sequential topology lost its closed issue program");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(dot);
      auto materializeIssueOperands = [&](int64_t output, int64_t issue) {
        const size_t index = static_cast<size_t>(output * issueCount + issue);
        auto lhsParts = rewriter.getDenseI64ArrayAttr(
            {plan.getLhsSourceParts()[index]});
        auto rhsParts = rewriter.getDenseI64ArrayAttr(
            {plan.getRhsSourceParts()[index]});
        auto lhsOffsets = rewriter.getDenseI64ArrayAttr(
            {plan.getLhsLaneOffsets()[index]});
        auto rhsOffsets = rewriter.getDenseI64ArrayAttr(
            {plan.getRhsLaneOffsets()[index]});
        auto lhsSlice = rewriter.create<riscv::RVVIssueSliceOp>(
            dot.getLoc(), lhsIssue, dot.getLhs(), lhsParts, lhsOffsets,
            riscv_internal::leaf(rewriter, "transfer", "issue-slice",
                                 "rvv.issue-slice", "rvv.issue-slice", 0,
                                 lhsIssue.getLayout().getRegisterGroups(), 0, 0,
                                 "none", "exact"));
        auto rhsSlice = rewriter.create<riscv::RVVIssueSliceOp>(
            dot.getLoc(), rhsIssue, dot.getRhs(), rhsParts, rhsOffsets,
            riscv_internal::leaf(rewriter, "transfer", "issue-slice",
                                 "rvv.issue-slice", "rvv.issue-slice", 0,
                                 rhsIssue.getLayout().getRegisterGroups(), 0, 0,
                                 "none", "exact"));
        riscv_internal::copyOrigin(dot, lhsSlice);
        riscv_internal::copyOrigin(dot, rhsSlice);
        return std::make_pair(lhsSlice.getResult(), rhsSlice.getResult());
      };
      auto materializeIssue = [&](int64_t output, int64_t issue,
                                  mlir::Value carried) -> mlir::Value {
        auto [lhsSlice, rhsSlice] = materializeIssueOperands(output, issue);
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            dot.getLoc(), accumulator, lhsSlice, rhsSlice, carried,
            plan.getReductionAxes(),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate",
                plan.getAccumulateInstruction(),
                plan.getAccumulateInstruction(),
                lhsIssue.getLayout().getRegisterGroups() +
                    rhsIssue.getLayout().getRegisterGroups() +
                    accumulator.getLayout().getRegisterGroups(),
                accumulator.getLayout().getRegisterGroups(), 0, 0, "none",
                "agnostic"));
        riscv_internal::copyOrigin(dot, accumulate);
        return accumulate.getResult();
      };
      auto makeInitial = [&]() {
        auto zero = rewriter.create<riscv::ConstantOp>(
            dot.getLoc(), partialElement,
            rewriter.getIntegerAttr(partialElement, 0));
        auto initial = rewriter.create<riscv::RVVSplatOp>(
            dot.getLoc(), accumulator, zero.getResult(),
            riscv_internal::leaf(
                rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                accumulator.getLayout().getRegisterGroups()));
        riscv_internal::copyOrigin(dot, initial);
        return initial.getResult();
      };
      auto finalize = [&](mlir::Value partial) {
        auto finalized = rewriter.create<riscv::RVVFinalizeWidenDotOp>(
            dot.getLoc(), resultElement, partial,
            plan.getReductionAxes(),
            riscv_internal::leaf(
                rewriter, "rvv", "finalize-widen-dot",
                plan.getFinalizeInstruction(), plan.getFinalizeInstruction(),
                accumulator.getLayout().getRegisterGroups(), 0, 2, 0, "none",
                "exact", plan.getReductionAxes().asArrayRef()));
        riscv_internal::copyOrigin(dot, finalized);
        return finalized.getResult();
      };

      llvm::SmallVector<mlir::Value> materializedOutputs;
      for (int64_t output = 0; output < outputParts; ++output) {
        mlir::Value outputValue;
        if (issueCount == 1) {
          auto [lhsSlice, rhsSlice] = materializeIssueOperands(output, 0);
          auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
              dot.getLoc(), accumulator, lhsSlice, rhsSlice,
              riscv_internal::leaf(
                  rewriter, "rvv", "widen-multiply",
                  plan.getMultiplyInstruction(),
                  plan.getMultiplyInstruction(),
                  lhsIssue.getLayout().getRegisterGroups() +
                      rhsIssue.getLayout().getRegisterGroups(),
                  accumulator.getLayout().getRegisterGroups(), 0, 0, "none",
                  "exact"));
          riscv_internal::copyOrigin(dot, product);
          outputValue = finalize(product.getResult());
        } else if (fused) {
          mlir::Value carried = makeInitial();
          for (int64_t issue = 0; issue < issueCount; ++issue)
            carried = materializeIssue(output, issue, carried);
          outputValue = finalize(carried);
        } else {
          for (int64_t issue = 0; issue < issueCount; ++issue) {
            mlir::Value contribution = finalize(
                materializeIssue(output, issue, makeInitial()));
            if (!outputValue) {
              outputValue = contribution;
              continue;
            }
            auto combined = rewriter.create<riscv::BinaryOp>(
                dot.getLoc(), resultElement, outputValue, contribution, "add",
                riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(dot, combined);
            outputValue = combined.getResult();
          }
        }
        materializedOutputs.push_back(outputValue);
      }
      mlir::Value replacement;
      if (resultValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            dot.getLoc(), resultValue, materializedOutputs,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = materializedOutputs.front();
      }
      dot.getResult().replaceAllUsesWith(replacement);
      rewriter.eraseOp(dot);
    }

    llvm::SmallVector<riscv::RVVWidenDotOp> independentDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      independentDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : independentDots) {
      if (!hasTopology(dot, "independent") || dot.getOver().empty())
        continue;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto plan = dot->getAttrOfType<riscv::PartialCombinePlanAttr>(
          "partial_combine_plan");
      auto sourceSet =
          plan ? mlir::dyn_cast<riscv::PartialSetType>(plan.getSourceSetType())
               : riscv::PartialSetType();
      const int64_t reductionAxis =
          sourceSet ? sourceSet.getReductionAxis() : int64_t{0};
      auto finalSet =
          plan ? mlir::dyn_cast<riscv::PartialSetType>(plan.getFinalSetType())
               : riscv::PartialSetType();
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          resultValue ? resultValue.getElementType() : dot.getResult().getType());
      if (!plan || !sourceSet || !finalSet || reductionAxis <= 0 ||
          !resultElement ||
          !resultElement.isSigned() || resultElement.getWidth() != 32 ||
          plan.getOutputParts() <= 0 ||
          plan.getLhsParts().size() != plan.getRhsParts().size() ||
          plan.getLhsParts().size() !=
              static_cast<size_t>(plan.getOutputParts() * sourceSet.getSlots()) ||
          plan.getCombineSetTypes().size() !=
              plan.getCombineArities().size()) {
        dot.emitError(
            "selected independent topology reached materialization without a closed combine plan");
        failed = true;
        continue;
      }
      riscv::ConvertLayoutOp vectorConsumer;
      riscv::ValueType vectorResult;
      if (plan.getRealization() == "independent_vector" &&
          dot.getResult().hasOneUse()) {
        vectorConsumer = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *dot.getResult().getUsers().begin());
        if (vectorConsumer &&
            vectorConsumer.getConversion().getEffect() == "pure" &&
            vectorConsumer.getConversion().getKind() == "register_to_lane")
          vectorResult =
              mlir::dyn_cast<riscv::ValueType>(vectorConsumer.getResult().getType());
      }
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      if ((plan.getRealization() == "independent_vector") !=
              static_cast<bool>(vectorResult) ||
          (plan.getRealization() != "independent_vector" &&
           plan.getRealization() != "independent_scalar")) {
        dot.emitError(
            "selected independent realization no longer matches its consumer");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(dot);
      llvm::SmallVector<mlir::Value> materializedParts;
      const int64_t slots = sourceSet.getSlots();
      for (int64_t outputPart = 0; outputPart < plan.getOutputParts();
           ++outputPart) {
        llvm::SmallVector<int64_t> operandSlots(slots, 0);
        auto lhsParts = plan.getLhsParts().asArrayRef().slice(
            static_cast<size_t>(outputPart * slots), static_cast<size_t>(slots));
        auto rhsParts = plan.getRhsParts().asArrayRef().slice(
            static_cast<size_t>(outputPart * slots), static_cast<size_t>(slots));
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            dot.getLoc(), sourceSet, mlir::ValueRange{dot.getLhs()},
            mlir::ValueRange{dot.getRhs()},
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(lhsParts),
            rewriter.getDenseI64ArrayAttr(rhsParts),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots), reductionAxis,
            plan.getMultiplyInstruction(),
            ownerDomain(dot), nextPartialBirthId++, ownerDomain(dot),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", operandGroups,
                sourceSet.getResourceGroups(), 0, 0, "none", "exact",
                {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, set);
        sinkReplicaSupplies(set);
        mlir::Value partials = set.getResult();
        for (auto [typeAttr, arity] :
             llvm::zip(plan.getCombineSetTypes(),
                       plan.getCombineArities().asArrayRef())) {
          auto inputSet =
              mlir::cast<riscv::PartialSetType>(partials.getType());
          auto combinedType = mlir::cast<riscv::PartialSetType>(
              mlir::cast<mlir::TypeAttr>(typeAttr).getValue());
          auto inputElement = mlir::dyn_cast<mlir::IntegerType>(
              inputSet.getPartialType().getElementType());
          auto combinedElement = mlir::dyn_cast<mlir::IntegerType>(
              combinedType.getPartialType().getElementType());
          const bool wideningCombine =
              inputElement && combinedElement && inputElement.isSigned() &&
              combinedElement.isSigned() && inputElement.getWidth() == 16 &&
              combinedElement.getWidth() == 32;
          const llvm::StringRef combineInstruction =
              wideningCombine ? "rvv.partial-combine.widen"
                              : "rvv.partial-combine";
          auto combine = rewriter.create<riscv::RVVPartialCombineOp>(
              dot.getLoc(), combinedType, partials, arity, "pairwise",
              dot.getPartialTopology().getPartialAxes(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-combine", combineInstruction,
                  combineInstruction, inputSet.getResourceGroups(),
                  combinedType.getResourceGroups(), 0, 0, "none", "exact",
                  {reductionAxis, arity, combinedType.getSlots()}));
          riscv_internal::copyOrigin(dot, combine);
          partials = combine.getResult();
        }
        mlir::Type finalizedType = vectorResult
                                       ? mlir::Type(vectorResult)
                                       : mlir::Type(resultElement);
        auto finalize = rewriter.create<riscv::RVVPartialFinalizeOp>(
            dot.getLoc(), finalizedType, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                plan.getFinalizeInstruction(), plan.getFinalizeInstruction(),
                mlir::cast<riscv::PartialSetType>(partials.getType())
                    .getResourceGroups(),
                vectorResult ? vectorResult.getLayout().getRegisterGroups() : 0,
                2, 0, "none", "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, finalize);
        materializedParts.push_back(finalize.getResult());
      }
      mlir::Value replacement;
      if (vectorResult) {
        replacement = materializedParts.front();
        vectorConsumer.getResult().replaceAllUsesWith(replacement);
        rewriter.eraseOp(vectorConsumer);
      } else if (resultValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            dot.getLoc(), resultValue, materializedParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = materializedParts.front();
      }
      if (!vectorResult)
        dot.getResult().replaceAllUsesWith(replacement);
      rewriter.eraseOp(dot);
    }

    if (!materializePartialAddTrees(getOperation(), rewriter, nextPartialBirthId))
      failed = true;
    llvm::SmallVector<riscv::RVVWidenDotOp> dots;
    getOperation().walk(
        [&](riscv::RVVWidenDotOp dot) { dots.push_back(dot); });
    for (riscv::RVVWidenDotOp dot : dots) {
      if (!hasTopology(dot, "layered") || dot.getOver().size() != 1)
        continue;
      auto plan = dot.getLayeredPartialPlanAttr();
      llvm::DenseSet<mlir::Operation *> lhsVisited;
      llvm::DenseSet<mlir::Operation *> rhsVisited;
      llvm::SmallVector<ProjectedRoot> roots;
      collectProjectedRoots(dot.getLhs(), dot.getOver()[0], lhsVisited, roots);
      collectProjectedRoots(dot.getRhs(), dot.getOver()[0], rhsVisited, roots);
      if (!plan || plan.getRootIndex() < 0 ||
          plan.getRootIndex() >= static_cast<int64_t>(roots.size()) ||
          plan.getRootWindowTypes().size() != roots.size() ||
          plan.getRootStoragePlans().size() != roots.size() ||
          plan.getRootWindowInstructions().size() != roots.size()) {
        dot.emitError(
            "selected layered partial topology no longer satisfies its typed plan");
        failed = true;
        continue;
      }
      const int64_t reductionAxis = plan.getGeometry().getAxis();
      ProjectedRoot layered = roots[static_cast<size_t>(plan.getRootIndex())];
      if (!layered.layered || !layered.field || !layered.origin ||
          layered.geometry != plan.getGeometry()) {
        dot.emitError(
            "selected layered partial root disagrees with its frozen typed geometry");
        failed = true;
        continue;
      }
      llvm::DenseMap<mlir::Value, riscv::ValueType> windowTypes;
      llvm::DenseMap<mlir::Value, riscv::StorageWindowPlanAttr> storagePlans;
      llvm::DenseMap<mlir::Value, mlir::StringAttr> windowInstructions;
      bool completeRootTypes = true;
      for (auto [root, typeAttributeValue, storagePlanValue,
                 instructionValue] :
           llvm::zip(roots, plan.getRootWindowTypes(),
                     plan.getRootStoragePlans(),
                     plan.getRootWindowInstructions())) {
        auto typeAttribute = mlir::dyn_cast<mlir::TypeAttr>(typeAttributeValue);
        auto windowType =
            typeAttribute
                ? mlir::dyn_cast<riscv::ValueType>(typeAttribute.getValue())
                : riscv::ValueType();
        auto storagePlan =
            mlir::dyn_cast<riscv::StorageWindowPlanAttr>(storagePlanValue);
        auto instruction = mlir::dyn_cast<mlir::StringAttr>(instructionValue);
        completeRootTypes &= static_cast<bool>(windowType) &&
                             static_cast<bool>(storagePlan) &&
                             static_cast<bool>(instruction);
        if (windowType && storagePlan && instruction) {
          windowTypes[root.value] = windowType;
          storagePlans[root.value] = storagePlan;
          windowInstructions[root.value] = instruction;
        }
      }
      if (!completeRootTypes) {
        dot.emitError("selected layered partial roots lost their typed windows");
        failed = true;
        continue;
      }
      riscv::AccessAttr access = layered.access;
      const int64_t group = plan.getGeometry().getGroupSize();
      const int64_t layerExtent = plan.getGeometry().getLayerSize();
      auto storageType =
          mlir::cast<riscv::LayeredWindowType>(plan.getStorageWindowType());
      const int64_t layers = storageType.getLayers();
      if (plan.getDecodeInstructions().size() != static_cast<size_t>(layers)) {
        dot.emitError(
            "selected layered partial plan lost its decode instruction sequence");
        failed = true;
        continue;
      }
      const int64_t lanes = plan.getGeometry().getLaneCount();
      const int64_t windowsPerLayer = storageType.getWindowsPerLayer();
      const int64_t windowCount =
          static_cast<int64_t>(plan.getGeometry().getGroupForWindow().size());
      auto layeredWindowType =
          mlir::cast<riscv::ValueType>(plan.getDecodedWindowType());
      auto accumulatorType =
          mlir::cast<riscv::ValueType>(plan.getAccumulatorType());
      auto partialInteger = mlir::cast<mlir::IntegerType>(
          accumulatorType.getElementType());

      rewriter.setInsertionPoint(dot);
      auto partialElement =
          mlir::cast<mlir::IntegerType>(accumulatorType.getElementType());
      auto zeroAttr = rewriter.getIntegerAttr(partialElement, 0);
      auto zero = rewriter.create<riscv::ConstantOp>(dot.getLoc(), partialElement,
                                                      zeroAttr);
      auto initial = rewriter.create<riscv::RVVSplatOp>(
          dot.getLoc(), accumulatorType, zero.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
              accumulatorType.getLayout().getRegisterGroups()));
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), 0);
      mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), windowCount);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          dot.getLoc(), lower, upper, step, mlir::ValueRange{initial.getResult()});
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll",
                    rewriter.getStringAttr("disable"));
      rewriter.setInsertionPointToStart(loop.getBody());
      mlir::Value windowIndex = loop.getInductionVar();
      const int64_t rawGroups =
          layeredWindowType.getLayout().getRegisterGroups();
      auto storagePlan = storagePlans.lookup(layered.value);
      if (!storagePlan) {
        dot.emitError(
            "selected layered partial root has no closed storage-load plan");
        failed = true;
        continue;
      }
      auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
          dot.getLoc(), storageType, layered.field.getResult(),
          layered.origin.getResult(), windowIndex, storagePlan, access,
          riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-load",
                "rvv.layered-storage-load", "rvv.layered-storage-load",
                mlir::cast<riscv::ValueType>(layered.field.getResult().getType())
                    .getLayout()
                    .getRegisterGroups(),
                rawGroups, rawGroups, 0, "none",
              layeredWindowType.getLayout().getValidity() == "tail" ? "agnostic"
                                                                     : "exact"));

      mlir::Value windowsPerLayerValue =
          rewriter.create<mlir::arith::ConstantIndexOp>(dot.getLoc(),
                                                        windowsPerLayer);
      mlir::Value groupValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), group);
      mlir::Value laneValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), lanes);
      mlir::Value groupIndex = rewriter.create<mlir::arith::DivUIOp>(
          dot.getLoc(), windowIndex, windowsPerLayerValue);
      mlir::Value withinWindow = rewriter.create<mlir::arith::RemUIOp>(
          dot.getLoc(), windowIndex, windowsPerLayerValue);
      mlir::Value groupBase = rewriter.create<mlir::arith::MulIOp>(
          dot.getLoc(), groupIndex, groupValue);
      mlir::Value withinBase = rewriter.create<mlir::arith::MulIOp>(
          dot.getLoc(), withinWindow, laneValue);

      mlir::Value carried = loop.getRegionIterArg(0);
      for (int64_t layer = 0; layer < layers; ++layer) {
        const int64_t physicalLayer =
            plan.getGeometry().getPhysicalLayerForLogicalLayer()[
                static_cast<size_t>(layer)];
        const int64_t shiftAmount =
            plan.getGeometry().getShiftAmountForLogicalLayer()[
                static_cast<size_t>(layer)];
        const int64_t maskValue =
            plan.getGeometry().getMaskValueForLogicalLayer()[
                static_cast<size_t>(layer)];
        auto instructionAttr = mlir::dyn_cast<mlir::StringAttr>(
            plan.getDecodeInstructions()[static_cast<size_t>(layer)]);
        if (!instructionAttr) {
          dot.emitError(
              "selected layered partial plan has an untyped decode instruction");
          failed = true;
          break;
        }
        llvm::StringRef instruction = instructionAttr.getValue();
        auto decoded = rewriter.create<riscv::RVVLayeredStorageDecodeOp>(
            dot.getLoc(), layeredWindowType, storage.getResult(), layer,
            physicalLayer, shiftAmount, maskValue,
            riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-decode",
                instruction, instruction,
                rawGroups, layeredWindowType.getLayout().getRegisterGroups(), 1,
                0, "none",
                layeredWindowType.getLayout().getValidity() == "tail"
                    ? "agnostic"
                    : "exact"));
        mlir::Value layerValue = rewriter.create<mlir::arith::ConstantIndexOp>(
            dot.getLoc(), layer * layerExtent);
        mlir::Value layerBase = rewriter.create<mlir::arith::AddIOp>(
            dot.getLoc(), groupBase, layerValue);
        mlir::Value logicalOffset = rewriter.create<mlir::arith::AddIOp>(
            dot.getLoc(), layerBase, withinBase);

        llvm::DenseMap<mlir::Value, mlir::Value> replacements;
        replacements[layered.value] = decoded.getResult();
        for (ProjectedRoot &root : roots) {
          if (root.value == layered.value)
            continue;
          riscv::ValueType windowType = windowTypes.lookup(root.value);
          llvm::StringRef rootTail =
              windowType.getLayout().getValidity() == "tail" ? "agnostic"
                                                               : "exact";
          auto rootPlan = storagePlans.lookup(root.value);
          auto rootInstruction = windowInstructions.lookup(root.value);
          if (!rootPlan || !rootInstruction) {
            dot.emitError(
                "projected partial root has no closed typed storage-window plan");
            failed = true;
            break;
          }
          auto window = rewriter.create<riscv::RVVStorageWindowOp>(
              dot.getLoc(), windowType, root.field.getResult(),
              root.origin.getResult(), logicalOffset, rootPlan,
              root.field.getAccess(),
              riscv_internal::leaf(
                  rewriter, "rvv", "storage-window",
                  rootInstruction.getValue(), rootInstruction.getValue(),
                  mlir::cast<riscv::ValueType>(root.field.getResult().getType())
                      .getLayout()
                      .getRegisterGroups(),
                  windowType.getLayout().getRegisterGroups(), 1, 0, "none",
                  rootTail));
          replacements[root.value] = window.getResult();
        }

        llvm::DenseMap<mlir::Value, mlir::Value> clones;
        llvm::DenseMap<mlir::Value, bool> dependence;
        auto lhsSlice = cloneWindowSlice(dot.getLhs(), replacements,
                                         reductionAxis, rewriter, clones,
                                         dependence);
        auto rhsSlice = cloneWindowSlice(dot.getRhs(), replacements,
                                         reductionAxis, rewriter, clones,
                                         dependence);
        if (mlir::failed(lhsSlice) || mlir::failed(rhsSlice)) {
          failed = true;
          break;
        }

        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            dot.getLoc(), accumulatorType, *lhsSlice, *rhsSlice, carried,
            rewriter.getDenseI64ArrayAttr({reductionAxis}),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial",
                mlir::cast<riscv::ValueType>(lhsSlice->getType())
                        .getLayout()
                        .getRegisterGroups() +
                    mlir::cast<riscv::ValueType>(rhsSlice->getType())
                        .getLayout()
                        .getRegisterGroups() +
                    accumulatorType.getLayout().getRegisterGroups(),
                accumulatorType.getLayout().getRegisterGroups(), 0, 0, "none",
                "agnostic"));
        carried = accumulate.getResult();
      }
      if (failed) {
        rewriter.eraseOp(loop);
        break;
      }
      rewriter.create<mlir::scf::YieldOp>(dot.getLoc(), carried);
      rewriter.setInsertionPointAfter(loop);
      auto finalized = rewriter.create<riscv::RVVFinalizeWidenDotOp>(
          dot.getLoc(), dot.getResult().getType(), loop.getResult(0),
          rewriter.getDenseI64ArrayAttr({reductionAxis}),
          riscv_internal::leaf(rewriter, "rvv", "finalize-widen-dot",
                               plan.getFinalizeInstruction(),
                               plan.getFinalizeInstruction(),
                               accumulatorType.getLayout().getRegisterGroups(),
                               0, 2));
      riscv_internal::copyOrigin(dot, finalized);
      mlir::Value oldLhs = dot.getLhs();
      mlir::Value oldRhs = dot.getRhs();
      dot.getResult().replaceAllUsesWith(finalized.getResult());
      rewriter.eraseOp(dot);
      llvm::DenseSet<mlir::Value> stops;
      for (ProjectedRoot &root : roots)
        stops.insert(root.field.getResult());
      llvm::DenseSet<mlir::Operation *> erased;
      eraseDeadChain(oldLhs, stops, erased, rewriter);
      eraseDeadChain(oldRhs, stops, erased, rewriter);
    }

    // Every selected widening-dot must now be an explicit issue/product/
    // reduction program.  Leaving the composite here would make the terminal
    // emitter reconstruct hidden SSA values and their resource lifetime.
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      dot.emitError(
          "selected widening dot was not materialized into the final physical program; topology=")
          << dot.getPartialTopology();
      failed = true;
    });

    llvm::SmallVector<riscv::RVVPartialSetOp> partialSets;
    getOperation().walk(
        [&](riscv::RVVPartialSetOp partial) { partialSets.push_back(partial); });
    for (riscv::RVVPartialSetOp partial : partialSets)
      sinkReplicaSupplies(partial);

    if (failed)
      signalPassFailure();
  }
};

} // namespace weft::riscv_partial

std::unique_ptr<mlir::Pass>
weft::createMaterializeRISCVPartialAccumulatorsPass() {
  return std::make_unique<weft::riscv_partial::MaterializeRISCVPartialAccumulatorsPass>();
}
