#include "Support.h"

namespace weft::riscv_partial {

void planPartialAddTrees(mlir::ModuleOp module, bool extractionsOnly) {
  mlir::Builder builder(module.getContext());
  // Scalar add trees combine multiple already-planned partial leaves.  Plan
  // the common carrier and every repack before materialization so equivalent
  // add spelling cannot trigger a second layout search in the next pass.
  llvm::SmallVector<riscv::BinaryOp> partialAddRoots;
  module.walk([&](riscv::BinaryOp add) {
    auto kernel = add->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      return;
    add->removeAttr("partial_add_tree_plan");
    if (add.getKind() == "add" &&
        mlir::isa<mlir::IntegerType>(add.getResult().getType()))
      partialAddRoots.push_back(add);
  });
  for (riscv::BinaryOp root : partialAddRoots) {
    llvm::SmallVector<riscv::BinaryOp> adds;
    llvm::SmallVector<PartialAddLeaf> leaves;
    if (!collectPartialAddTree(root.getResult(), root, adds, leaves) ||
        leaves.size() < 2)
      continue;
    if (extractionsOnly && !llvm::all_of(leaves, [&](PartialAddLeaf leaf) {
          if (!leaf.finalize || leaf.finalize->getBlock() != root->getBlock())
            return false;
          auto set = leaf.finalize.getInput().getType();
          auto partial = set.getPartialType();
          auto element = mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
          return element && element.isSigned() && element.getWidth() == 32 &&
                 partial.getShape().asArrayRef() == llvm::ArrayRef<int64_t>({1}) &&
                 partial.getLayout().getCarrier() == "rvv" &&
                 partial.getLayout().getVl() == 1 &&
                 partial.getLayout().getRegisterGroups() == 1 &&
                 leaf.finalize.getLeaf().getInstruction() ==
                     "rvv.partial-finalize.extract";
        }))
      continue;
    llvm::SmallVector<riscv::PartialSetType> leafTypes;
    bool complete = true;
    for (PartialAddLeaf leaf : leaves) {
      auto type = partialTypeForAddLeaf(builder, leaf);
      if (!type) {
        complete = false;
        break;
      }
      leafTypes.push_back(*type);
    }
    if (!complete)
      continue;
    riscv::PartialSetType targetType;
    llvm::SmallVector<int64_t> splits;
    llvm::SmallVector<size_t> order(leaves.size());
    std::iota(order.begin(), order.end(), 0);
    llvm::sort(order, [&](size_t lhsIndex, size_t rhsIndex) {
      auto lhsSet = leafTypes[lhsIndex];
      auto rhsSet = leafTypes[rhsIndex];
      auto lhsPosition = axisPosition(lhsSet.getPartialType(),
                                      lhsSet.getReductionAxis());
      auto rhsPosition = axisPosition(rhsSet.getPartialType(),
                                      rhsSet.getReductionAxis());
      return lhsSet.getPartialType().getLayout().getLaneFactors()[*lhsPosition] >
             rhsSet.getPartialType().getLayout().getLaneFactors()[*rhsPosition];
    });
    for (size_t targetIndex : order) {
      auto candidate = leafTypes[targetIndex];
      llvm::SmallVector<int64_t> candidateSplits;
      bool compatible = true;
      for (riscv::PartialSetType source : leafTypes) {
        auto split = partialSplitFactor(source, candidate);
        if (!split || *split > 2) {
          compatible = false;
          break;
        }
        candidateSplits.push_back(*split);
      }
      if (compatible) {
        targetType = candidate;
        splits = std::move(candidateSplits);
        break;
      }
    }
    if (!targetType)
      continue;
    llvm::SmallVector<mlir::Attribute> leafTypeAttrs;
    llvm::SmallVector<mlir::Attribute> normalizedTypeAttrs;
    llvm::SmallVector<mlir::Attribute> multiplyAttrs;
    llvm::SmallVector<mlir::Attribute> repackLeaves;
    int64_t totalSlots = 0;
    int64_t totalTerms = 0;
    int64_t totalResources = 0;
    for (auto [leaf, source, split] : llvm::zip(leaves, leafTypes, splits)) {
      auto normalized =
          split == 1
              ? source
              : riscv::PartialSetType::get(
                    builder.getContext(), targetType.getPartialType(),
                    targetType.getReductionAxis(), source.getSlots() * split,
                    source.getTermsPerSlot() / split,
                    source.getSlots() * split *
                        targetType.getPartialType().getLayout().getRegisterGroups());
      leafTypeAttrs.push_back(mlir::TypeAttr::get(source));
      normalizedTypeAttrs.push_back(mlir::TypeAttr::get(normalized));
      mlir::Attribute repack = builder.getUnitAttr();
      if (split > 1) {
        auto selected = partialRepackLeaf(builder,
            root->getParentOfType<riscv::KernelOp>().getTarget(),
            source, normalized, split);
        if (!selected) {
          complete = false;
          break;
        }
        repack = *selected;
      }
      repackLeaves.push_back(repack);
      if (leaf.dot) {
        auto multiply = partialMultiplyInstruction(leaf.dot.getLhs().getType(),
                                                   leaf.dot.getRhs().getType());
        if (!multiply) {
          complete = false;
          break;
        }
        multiplyAttrs.push_back(builder.getStringAttr(*multiply));
      } else {
        multiplyAttrs.push_back(builder.getStringAttr(""));
      }
      const int64_t maximum = std::numeric_limits<int64_t>::max();
      if (normalized.getSlots() > maximum - totalSlots ||
          normalized.getTermsPerSlot() >
              (maximum - totalTerms) / normalized.getSlots() ||
          normalized.getResourceGroups() > maximum - totalResources) {
        complete = false;
        break;
      }
      totalSlots += normalized.getSlots();
      totalTerms += normalized.getSlots() * normalized.getTermsPerSlot();
      totalResources += normalized.getResourceGroups();
    }
    if (!complete)
      continue;
    auto mergedType = riscv::PartialSetType::get(
        builder.getContext(), targetType.getPartialType(),
        targetType.getReductionAxis(), 1, totalTerms,
        targetType.getPartialType().getLayout().getRegisterGroups());
    auto finalizeInstruction =
        partialFinalizeInstruction(mergedType, targetType.getReductionAxis());
    if (!finalizeInstruction)
      continue;
    if (extractionsOnly && totalResources >
                               root->getParentOfType<riscv::KernelOp>()
                                       .getTarget().getVectorRegisters() -
                                   mergedType.getResourceGroups())
      continue;
    root->setAttr(
        "partial_add_tree_plan",
        riscv::PartialAddTreePlanAttr::get(
            builder.getContext(), builder.getArrayAttr(leafTypeAttrs),
            builder.getArrayAttr(normalizedTypeAttrs),
            builder.getArrayAttr(multiplyAttrs),
            builder.getDenseI64ArrayAttr(splits), mergedType,
            *finalizeInstruction, totalSlots, totalTerms, totalResources,
            builder.getArrayAttr(repackLeaves)));
  }
  // Only the late, already-reduced i32 relation admits maximal extraction
  // trees here. Do not enlarge the earlier narrow-product merge domain.
  if (!extractionsOnly)
    return;
  // Rewriting a child first would invalidate the parent plan's leaf identities
  // before it can be materialized.
  for (riscv::BinaryOp root : partialAddRoots)
    if (root.getResult().hasOneUse() &&
        (*root.getResult().getUsers().begin())->hasAttr("partial_add_tree_plan"))
      root->removeAttr("partial_add_tree_plan");
}

bool materializePartialAddTrees(mlir::ModuleOp module,
                               mlir::IRRewriter &rewriter,
                               int64_t &nextPartialBirthId) {
  bool failed = false;
  // Scalar add trees can expose independently materialized widened partials
  // that share one logical reduction axis but use different legal RVV lane
  // widths. Normalize the wider partials to the widest existing common
  // representation, join their slots, and reduce once.
  llvm::SmallVector<riscv::BinaryOp> partialAddRoots;
  module.walk([&](riscv::BinaryOp add) {
    if (add.getKind() == "add" &&
        mlir::isa<mlir::IntegerType>(add.getResult().getType()))
      partialAddRoots.push_back(add);
  });
  for (riscv::BinaryOp root : partialAddRoots) {
    if (!root || !root->getBlock())
      continue;
    auto plan = root->getAttrOfType<riscv::PartialAddTreePlanAttr>(
        "partial_add_tree_plan");
    if (!plan)
      continue;
    llvm::SmallVector<riscv::BinaryOp> adds;
    llvm::SmallVector<PartialAddLeaf> leaves;
    if (!collectPartialAddTree(root.getResult(), root, adds, leaves) ||
        leaves.size() != plan.getLeafSetTypes().size() ||
        leaves.size() != plan.getNormalizedSetTypes().size() ||
        leaves.size() != plan.getLeafMultiplyInstructions().size() ||
        leaves.size() != plan.getRepackLeaves().size() ||
        leaves.size() != plan.getSplits().size()) {
      root.emitError(
          "partial add tree no longer matches its frozen typed leaf plan");
      failed = true;
      continue;
    }

    rewriter.setInsertionPoint(root);
    llvm::SmallVector<mlir::Value> normalized;
    bool complete = true;
    for (size_t index = 0; index < leaves.size(); ++index) {
      PartialAddLeaf leaf = leaves[index];
      auto sourceType = mlir::cast<riscv::PartialSetType>(
          mlir::cast<mlir::TypeAttr>(plan.getLeafSetTypes()[index])
              .getValue());
      auto normalizedType = mlir::cast<riscv::PartialSetType>(
          mlir::cast<mlir::TypeAttr>(plan.getNormalizedSetTypes()[index])
              .getValue());
      auto currentType = partialTypeForAddLeaf(rewriter, leaf);
      const int64_t split = plan.getSplits()[index];
      auto multiply = mlir::cast<mlir::StringAttr>(
          plan.getLeafMultiplyInstructions()[index]);
      if (!currentType || *currentType != sourceType) {
        complete = false;
        break;
      }
      mlir::Value value;
      mlir::Operation *origin = nullptr;
      if (leaf.finalize) {
        if (!multiply.getValue().empty()) {
          complete = false;
          break;
        }
        value = leaf.finalize.getInput();
        origin = leaf.finalize.getOperation();
      } else {
        if (multiply.getValue().empty()) {
          complete = false;
          break;
        }
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            root.getLoc(), sourceType,
            mlir::ValueRange{leaf.dot.getLhs()},
            mlir::ValueRange{leaf.dot.getRhs()},
            rewriter.getDenseI64ArrayAttr({0}),
            rewriter.getDenseI64ArrayAttr({0}),
            rewriter.getDenseI64ArrayAttr({0}),
            rewriter.getDenseI64ArrayAttr({0}),
            rewriter.getDenseI64ArrayAttr({0}),
            rewriter.getDenseI64ArrayAttr({0}),
            sourceType.getReductionAxis(), multiply.getValue(),
            ownerDomain(root), nextPartialBirthId++, ownerDomain(root),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", 0, sourceType.getResourceGroups(), 0, 0,
                "none", "exact", {sourceType.getReductionAxis(), 1}));
        riscv_internal::copyOrigin(leaf.dot, set);
        sinkReplicaSupplies(set);
        value = set.getResult();
        origin = leaf.dot.getOperation();
      }
      if (split > 1) {
        auto repackLeaf = mlir::dyn_cast<riscv::LeafAttr>(plan.getRepackLeaves()[index]);
        if (!repackLeaf) {
          root->emitError("selected partial repack has no legal carrier");
          return false;
        }
        auto repack = rewriter.create<riscv::RVVPartialRepackOp>(
            root.getLoc(), normalizedType, value, split,
            repackLeaf);
        riscv_internal::copyOrigin(origin, repack);
        value = repack.getResult();
      } else if (normalizedType != sourceType) {
        complete = false;
        break;
      }
      normalized.push_back(value);
    }
    if (!complete || normalized.size() != leaves.size()) {
      root.emitError(
          "partial add tree materialization disagrees with its frozen typed plan");
      failed = true;
      continue;
    }
    auto combinedType =
        mlir::cast<riscv::PartialSetType>(plan.getMergedSetType());
    auto combine = rewriter.create<riscv::RVVPartialMergeOp>(
        root.getLoc(), combinedType, normalized, "pairwise",
        riscv_internal::leaf(
            rewriter, "rvv", "partial-merge", "rvv.partial-merge",
            "rvv.partial-merge", plan.getTotalResourceGroups(),
            combinedType.getResourceGroups(), 0, 0, "none", "exact",
            {combinedType.getReductionAxis(),
             static_cast<int64_t>(normalized.size()), plan.getTotalSlots(),
             plan.getTotalTerms()}));
    riscv_internal::copyOrigin(root, combine);
    auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
        root.getLoc(), root.getResult().getType(), combine.getResult(),
        combinedType.getReductionAxis(),
        riscv_internal::leaf(
            rewriter, "rvv", "partial-finalize",
            plan.getFinalizeInstruction(), plan.getFinalizeInstruction(),
            combinedType.getResourceGroups(), 0, 2, 0, "none", "exact",
            {combinedType.getReductionAxis(), 1}));
    riscv_internal::copyOrigin(root, finalized);

    root.getResult().replaceAllUsesWith(finalized.getResult());
    for (riscv::BinaryOp add : adds)
      if (add && add->getBlock() && add.getResult().use_empty())
        rewriter.eraseOp(add);
    for (PartialAddLeaf leaf : leaves) {
      if (leaf.finalize && leaf.finalize->getBlock() &&
          leaf.finalize.getResult().use_empty())
        rewriter.eraseOp(leaf.finalize);
      if (leaf.dot && leaf.dot->getBlock() && leaf.dot.getResult().use_empty())
        rewriter.eraseOp(leaf.dot);
    }
  }

  return !failed;
}

struct PackedScaleWindow {
  riscv::ConvertLayoutOp conversion;
  riscv::ValueType type;
};

std::optional<PackedScaleWindow>
packedScaleWindow(riscv::RVVPartialScaleCombineOp combine,
                   mlir::IRRewriter &rewriter) {
  auto kernel = combine->getParentOfType<riscv::KernelOp>();
  auto input = combine.getInput().getType();
  auto result = combine.getResult().getType();
  if (!kernel || kernel.getResourcesMaterialized() || result.getSlots() != 1 ||
      input.getSlots() < 2 || input.getSlots() > 8 ||
      combine.getLogicalReductionAxes().size() != 1 ||
      combine.getScales().size() != static_cast<size_t>(input.getSlots()))
    return std::nullopt;
  mlir::Value scalarScale = combine.getScales().front();
  if (!llvm::all_of(combine.getScales(),
                    [&](mlir::Value value) { return value == scalarScale; }))
    return std::nullopt;
  auto conversion = scalarScale.getDefiningOp<riscv::ConvertLayoutOp>();
  if (!conversion || conversion.getConversion().getEffect() != "pure" ||
      conversion.getResult().getType().getLayout().getCarrier() != "scalar" ||
      !llvm::all_of(scalarScale.getUsers(), [&](mlir::Operation *user) {
        return user == combine.getOperation();
      }))
    return std::nullopt;
  auto source = conversion.getInput().getType();
  auto before = source.getLayout();
  if (before.getCarrier() != "rvv" || source.getShape().size() != 1 ||
      source.getShape()[0] != input.getSlots() ||
      source.getAxisIds()[0] != combine.getLogicalReductionAxes()[0])
    return std::nullopt;
  for (int64_t lane = 0; lane < input.getSlots(); ++lane)
    if (combine.getScaleReplicas()[combine.getSlotOrder()[lane]] != lane)
      return std::nullopt;
  auto layout = riscv::LayoutAttr::get(
      rewriter.getContext(), "rvv", source.getAxisIds(),
      before.getTimeFactors(), before.getLaneFactors(),
      before.getReplicaFactors(), before.getFragmentFactors(),
      before.getLocalFactors(), 32, 8, before.getVl(), 1,
      before.getValidity());
  auto packedType = riscv::ValueType::get(
      rewriter.getContext(), source.getElementType(), source.getShape(),
      source.getAxisIds(), layout);
  if (!riscv::supportsRVVLayout(kernel.getTarget(), before) ||
      !riscv::supportsRVVPartialPackedScale(kernel.getTarget(), {input},
                                           {packedType}, result))
    return std::nullopt;
  return PackedScaleWindow{conversion, packedType};
}

mlir::Value materializePackedScale(PackedScaleWindow window,
                                   mlir::Operation *origin,
                                   mlir::IRRewriter &rewriter) {
  auto source = window.conversion.getInput().getType();
  mlir::Value scale = window.conversion.getInput();
  if (source != window.type) {
    auto edge = rewriter.create<riscv::ConvertLayoutOp>(
        origin->getLoc(), window.type, scale,
        riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                          window.type.getLayout()),
        riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, edge);
    scale = edge.getResult();
  }
  return scale;
}

void packReducedScaleCombines(mlir::ModuleOp module,
                              mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::RVVPartialScaleCombineOp> combines;
  module.walk([&](riscv::RVVPartialScaleCombineOp op) { combines.push_back(op); });
  for (auto combine : combines) {
    auto window = packedScaleWindow(combine, rewriter);
    if (!window)
      continue;
    auto input = combine.getInput().getType();
    auto result = combine.getResult().getType();
    auto conversion = window->conversion;
    auto layout = window->type.getLayout();
    // Compare two already-legal local representations.  Count the pack slides,
    // multiply, zero seed, reduction and its VL setup; a reduction has one
    // additional dependency unit.  Scalar rematerialization may avoid scale
    // extraction entirely, so use that candidate's lower bound when legal.
    const int64_t packedIssues = input.getSlots() + 4;
    llvm::DenseSet<mlir::Operation *> visited;
    const bool scalarSupply = supportsScalarReplicaRematerialization(
        conversion.getInput(), visited);
    if (visited.size() >= 32)
      continue;
    const int64_t scalarIssues = scalarSupply ? input.getSlots()
                                              : input.getSlots() * 3 - 1;
    if (packedIssues >= scalarIssues)
      continue;
    rewriter.setInsertionPoint(combine);
    mlir::Value scale = materializePackedScale(*window, combine, rewriter);
    auto packed = rewriter.create<riscv::RVVPartialPackedScaleOp>(
        combine.getLoc(), result, mlir::ValueRange{combine.getInput()},
        mlir::ValueRange{scale},
        combine.getSlotOrderAttr(), combine.getLogicalReductionAxesAttr(),
        riscv_internal::leaf(
            rewriter, "rvv", "partial-packed-scale", "rvv.partial-packed-scale",
            "rvv.partial-packed-scale",
            input.getResourceGroups() + layout.getRegisterGroups(),
            result.getResourceGroups(), 3, 0, "none", "exact"));
    packed->setAttr("weft.riscv.packed_scale_cost",
                    rewriter.getDenseI64ArrayAttr({packedIssues, scalarIssues}));
    riscv_internal::copyOrigin(combine, packed);
    rewriter.replaceOp(combine, packed.getResult());
    if (conversion.getResult().use_empty())
      rewriter.eraseOp(conversion);
  }
}

void packAdjacentScaleCombines(mlir::ModuleOp module,
                               mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::RVVPartialMergeOp> merges;
  module.walk([&](riscv::RVVPartialMergeOp op) { merges.push_back(op); });
  unsigned candidates = 0;
  for (auto merge : merges) {
    auto kernel = merge->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() ||
        merge.getInputs().size() > 8)
      continue;
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<riscv::RVVPartialScaleCombineOp> obsolete;
    llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
    for (size_t index = 0; index < merge.getInputs().size();) {
      auto first = merge.getInputs()[index]
                       .getDefiningOp<riscv::RVVPartialScaleCombineOp>();
      auto second = index + 1 < merge.getInputs().size()
          ? merge.getInputs()[index + 1]
                .getDefiningOp<riscv::RVVPartialScaleCombineOp>()
          : riscv::RVVPartialScaleCombineOp();
      auto firstWindow = first ? packedScaleWindow(first, rewriter) : std::nullopt;
      auto secondWindow = second ? packedScaleWindow(second, rewriter) : std::nullopt;
      if (!firstWindow || !secondWindow || first == second ||
          first->getBlock() != merge->getBlock() ||
          second->getBlock() != merge->getBlock() ||
          !first->isBeforeInBlock(second) ||
          !first.getResult().hasOneUse() || !second.getResult().hasOneUse() ||
          first.getLogicalReductionAxes() != second.getLogicalReductionAxes() ||
          candidates >= 32) {
        inputs.push_back(merge.getInputs()[index++]);
        continue;
      }
      ++candidates;
      auto a = first.getInput().getType();
      auto b = second.getInput().getType();
      const int64_t slots = a.getSlots() + b.getSlots();
      const int64_t firstTerms = first.getResult().getType().getTermsPerSlot();
      const int64_t secondTerms = second.getResult().getType().getTermsPerSlot();
      if (secondTerms > std::numeric_limits<int64_t>::max() - firstTerms) {
        inputs.push_back(merge.getInputs()[index++]);
        continue;
      }
      auto result = riscv::PartialSetType::get(
          rewriter.getContext(), a.getPartialType(), a.getReductionAxis(),
          1, firstTerms + secondTerms, 1);
      if (!riscv::supportsRVVPartialPackedScale(
              kernel.getTarget(), {a, b}, {firstWindow->type, secondWindow->type},
              result)) {
        inputs.push_back(merge.getInputs()[index++]);
        continue;
      }
      // Two independent scale windows need one additional scale-pack slide.
      // Include the old merge add as well as both legal scalar lower bounds.
      int64_t scalarIssues = 1;
      bool bounded = true;
      for (auto window : {*firstWindow, *secondWindow}) {
        llvm::DenseSet<mlir::Operation *> visited;
        const bool scalarSupply = supportsScalarReplicaRematerialization(
            window.conversion.getInput(), visited);
        bounded &= visited.size() < 32;
        const int64_t width = window.type.getShape()[0];
        scalarIssues += scalarSupply ? width : 3 * width - 1;
      }
      const int64_t packedIssues = slots + 5;
      if (!bounded || packedIssues >= scalarIssues) {
        inputs.push_back(merge.getInputs()[index++]);
        continue;
      }
      // Consume a completed pair at its last original consumer. Keeping all
      // pairs until the final merge would extend the large-product live sets.
      rewriter.setInsertionPoint(second);
      llvm::SmallVector<mlir::Value> scales{
          materializePackedScale(*firstWindow, first, rewriter),
          materializePackedScale(*secondWindow, second, rewriter)};
      llvm::SmallVector<int64_t> order(first.getSlotOrder());
      for (int64_t slot : second.getSlotOrder())
        order.push_back(a.getSlots() + slot);
      auto packed = rewriter.create<riscv::RVVPartialPackedScaleOp>(
          merge.getLoc(), result,
          mlir::ValueRange{first.getInput(), second.getInput()}, scales,
          rewriter.getDenseI64ArrayAttr(order), first.getLogicalReductionAxesAttr(),
          riscv_internal::leaf(
              rewriter, "rvv", "partial-packed-scale", "rvv.partial-packed-scale",
              "rvv.partial-packed-scale", a.getResourceGroups() + b.getResourceGroups() + 2,
              1, 4, 0, "none", "exact"));
      packed->setAttr("weft.riscv.packed_scale_cost",
                      rewriter.getDenseI64ArrayAttr({packedIssues, scalarIssues}));
      riscv_internal::copyOrigin(merge, packed);
      inputs.push_back(packed.getResult());
      obsolete.append({first, second});
      conversions.append({firstWindow->conversion, secondWindow->conversion});
      index += 2;
    }
    if (obsolete.empty())
      continue;
    if (inputs.size() == 1) {
      rewriter.replaceOp(merge, inputs.front());
    } else {
      int64_t slots = 0, resources = 0;
      for (mlir::Value value : inputs) {
        auto type = mlir::cast<riscv::PartialSetType>(value.getType());
        slots += type.getSlots();
        resources += type.getResourceGroups();
      }
      auto result = merge.getResult().getType();
      merge.getInputsMutable().assign(inputs);
      merge.setLeafAttr(riscv_internal::leaf(
          rewriter, "rvv", "partial-merge", "rvv.partial-merge", "rvv.partial-merge",
          resources, result.getResourceGroups(), 0, 0, "none", "exact",
          {result.getReductionAxis(), static_cast<int64_t>(inputs.size()), slots,
           result.getTermsPerSlot()}));
    }
    for (auto operation : obsolete)
      rewriter.eraseOp(operation);
    for (auto conversion : conversions)
      if (conversion.getResult().use_empty())
        rewriter.eraseOp(conversion);
  }
}

class CoalesceRISCVPartialExtractionsPass
    : public mlir::PassWrapper<CoalesceRISCVPartialExtractionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-coalesce-partial-extractions";
  }
  llvm::StringRef getDescription() const override {
    return "Combine already-reduced i32 partials before extracting scalar add trees";
  }
  void runOnOperation() override {
    planPartialAddTrees(getOperation(), true);
    mlir::IRRewriter rewriter(&getContext());
    // The extraction-only plans reuse existing partials and create no births.
    int64_t unusedBirthId = 0;
    if (!materializePartialAddTrees(getOperation(), rewriter, unusedBirthId)) {
      signalPassFailure();
      return;
    }
    packReducedScaleCombines(getOperation(), rewriter);
    packAdjacentScaleCombines(getOperation(), rewriter);
  }
};

} // namespace weft::riscv_partial

std::unique_ptr<mlir::Pass> weft::createCoalesceRISCVPartialExtractionsPass() {
  return std::make_unique<weft::riscv_partial::CoalesceRISCVPartialExtractionsPass>();
}
