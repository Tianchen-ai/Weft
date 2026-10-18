#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

#include <memory>

using namespace weft;

namespace {

bool rematerializable(mlir::Operation *operation) {
  return mlir::isa<riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                   riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation) &&
         operation->getNumResults() == 1;
}

bool isPureRepresentationConversion(riscv::ConvertLayoutOp conversion) {
  return conversion.getConversion().getEffect() == "pure";
}

bool isRegisterCarrierResize(riscv::ConvertLayoutOp conversion) {
  auto before = conversion.getInput().getType().getLayout();
  auto after = conversion.getResult().getType().getLayout();
  return before.getCarrier() == "rvv" && after.getCarrier() == "rvv" &&
         before.getLmulEighths() != after.getLmulEighths() &&
         before.getAxisIds() == after.getAxisIds() &&
         before.getTimeFactors() == after.getTimeFactors() &&
         before.getLaneFactors() == after.getLaneFactors() &&
         before.getReplicaFactors() == after.getReplicaFactors() &&
         before.getFragmentFactors() == after.getFragmentFactors() &&
         before.getLocalFactors() == after.getLocalFactors() &&
         before.getSew() == after.getSew() && before.getVl() == after.getVl() &&
         before.getValidity() == after.getValidity();
}

bool hasFinalResourceContract(riscv::ConvertLayoutOp conversion) {
  auto kernel = conversion->getParentOfType<riscv::KernelOp>();
  return kernel && kernel.getResourcesMaterialized();
}

bool preservesReadOrder(mlir::Operation *from, mlir::Operation *to) {
  if (from->getBlock() != to->getBlock())
    return false;
  for (mlir::Operation *cursor = from->getNextNode(); cursor != to;
       cursor = cursor->getNextNode()) {
    if (!cursor)
      return false;
    if (!riscv::hasOnlyReadEffects(cursor))
      return false;
  }
  return true;
}

bool canReprojectIndex(mlir::Value value, riscv::LayoutAttr required,
                       riscv::TargetAttr target, mlir::Builder &builder,
                       unsigned &remaining) {
  if (!remaining)
    return false;
  --remaining;
  auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!type || type.getLayout() == required)
    return true;
  auto edge = riscv_internal::layoutConversion(builder, type.getLayout(), required);
  if (type.getLayout().getCarrier() != "rvv" ||
      required.getCarrier() != "rvv" ||
      (edge.getKind() != "register_to_lane" &&
       edge.getKind() != "time_to_lane") ||
      riscv::rvvPartToLanePieces(
          type, mlir::cast<riscv::ValueType>(
                    riscv_internal::withLayout(type, required))))
    return true;
  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !rematerializable(producer) ||
      (!value.hasOneUse() && !mlir::isa<riscv::IotaOp>(producer)))
    return false;
  for (mlir::Value operand : producer->getOperands()) {
    auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
    if (!operandType)
      continue;
    auto projected = riscv_internal::projectLayout(
        builder, operandType, required, target);
    if (!projected ||
        !canReprojectIndex(operand, projected, target, builder, remaining))
      return false;
  }
  return true;
}

class CanonicalizeRISCVLayoutsPass
    : public mlir::PassWrapper<CanonicalizeRISCVLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeRISCVLayoutsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-canonicalize-layouts";
  }
  llvm::StringRef getDescription() const override {
    return "Eliminate and rematerialize explicit physical layout conversions";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    getOperation().walk([&](riscv::ExtractOp extract) {
      if (extract->getParentOfType<riscv::KernelOp>().getResourcesMaterialized() ||
          extract.getIndices().size() != 1) return;
      auto point = extract.getIndices()[0].getDefiningOp<riscv::PhysicalPointOp>();
      auto parent = extract.getInput().getDefiningOp<riscv::ExtractOp>();
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (!point || !parent || parent.getIndices().size() != 1 ||
          !result || result.getAxisIds() != parent.getInput().getType().getAxisIds() ||
          point.getParent() != parent.getIndices()[0] ||
          parent.getSelectors() != extract.getSelectors()) return;
      unsigned domains = 0;
      for (auto [axis, selector] : llvm::enumerate(extract.getSelectors())) {
        auto kind = mlir::cast<mlir::StringAttr>(selector).getValue();
        if (kind == "domain") {
          ++domains;
          int64_t whole = parent.getInput().getType().getShape()[axis];
          int64_t window = mlir::cast<riscv::ValueType>(parent.getResult().getType()).getShape()[axis];
          int64_t child = result.getShape()[axis];
          auto parentPoint = point.getParent().getDefiningOp<riscv::PhysicalPointOp>();
          if (whole <= 0 || window <= 0 || child <= 0 || whole % window || window % child || !parentPoint ||
              riscv_internal::constantInt(parentPoint.getPartition()) != window ||
              riscv_internal::constantInt(point.getPartition()) != child) return;
        }
        else if (kind != "all") return;
      }
      if (domains != 1) return;
      // Physical points retain absolute coordinates and their explicit parent.
      // A child of this exact projection can select the original SSA directly.
      extract.getInputMutable().assign(parent.getInput());
      extract.setAccessAttr(riscv_internal::unassignedAccess(rewriter));
      extract.setLeafAttr(riscv_internal::unselectedLeaf(rewriter));
    });
    llvm::SmallVector<riscv::ExtractOp> unitProjections;
    getOperation().walk([&](riscv::ExtractOp extract) {
      auto input = extract.getInput().getType();
      if (!mlir::isa<riscv::ValueType>(extract.getResult().getType()) ||
          !riscv_internal::staticProduct(input.getShape().asArrayRef()) ||
          extract->getParentOfType<riscv::KernelOp>().getResourcesMaterialized()) return;
      unsigned cursor = 0;
      for (auto [axis, selector] : llvm::enumerate(extract.getSelectors())) {
        auto kind = mlir::cast<mlir::StringAttr>(selector).getValue();
        if (kind == "all") continue;
        if (kind != "index" || input.getShape()[axis] != 1 ||
            !mlir::matchPattern(extract.getIndices()[cursor++], mlir::m_Zero())) return;
      }
      if (cursor) unitProjections.push_back(extract);
    });
    for (auto extract : unitProjections) {
      rewriter.setInsertionPoint(extract);
      auto projected = riscv::ReshapeOp::create(rewriter, extract.getLoc(), extract.getResult().getType(),
          extract.getInput(), extract.getInput().getType().getAxisIds(), mlir::DenseI64ArrayAttr());
      riscv_internal::copyOrigin(extract, projected);
      rewriter.replaceOp(extract, projected.getResult());
    }
    getOperation().walk([&](riscv::ExtractOp extract) {
      auto input = extract.getInput().getType();
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      auto kernel = extract->getParentOfType<riscv::KernelOp>();
      if (!result || kernel.getResourcesMaterialized() ||
          riscv::fieldReadProjection(extract.getInput()) ||
          input.getElementType() != result.getElementType() || input.getAxisIds() != result.getAxisIds() ||
          extract.getIndices().size() != 1) return;
      auto point = extract.getIndices()[0].getDefiningOp<riscv::PhysicalPointOp>();
      if (!point) return;
      auto before = input.getLayout(), after = result.getLayout();
      if (before.getCarrier() != "rvv" || after.getCarrier() != "rvv" ||
          before.getValidity() != after.getValidity() ||
          (before.getValidity() != "full" && before.getValidity() != "tail") ||
          before.getReplicaGroup() != 1 || after.getReplicaGroup() != 1 ||
          before.getSew() != after.getSew()) return;
      int64_t selected = -1;
      for (auto [axis, selector] : llvm::enumerate(extract.getSelectors())) {
        auto kind = mlir::cast<mlir::StringAttr>(selector).getValue();
        if (kind == "domain" && selected == -1) selected = axis;
        else if (kind != "all") return;
      }
      if (selected < 0 || point.getResult().getType().getDomain().getAxisId() != input.getAxisIds()[selected]) return;
      int64_t parent = input.getShape()[selected], child = result.getShape()[selected];
      int64_t oldLanes = before.getLaneFactors()[selected], newLanes = after.getLaneFactors()[selected];
      if (parent <= 0 || child <= 0 || parent % child ||
          riscv_internal::constantInt(point.getPartition()) != child ||
          (before.getValidity() == "tail" &&
           riscv_internal::constantInt(point.getActive()) != child) ||
          oldLanes <= newLanes || newLanes <= 1 || oldLanes % newLanes || child % newLanes ||
          (oldLanes % child && child % oldLanes)) return;
      for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
        if (before.getFragmentFactors()[axis] != 1 || before.getLocalFactors()[axis] != 1 ||
            after.getFragmentFactors()[axis] != 1 || after.getLocalFactors()[axis] != 1 ||
            before.getReplicaFactors()[axis] != after.getReplicaFactors()[axis]) return;
        if (axis != selected && (input.getShape()[axis] != result.getShape()[axis] ||
            before.getTimeFactors()[axis] != after.getTimeFactors()[axis] ||
            before.getLaneFactors()[axis] != 1 || after.getLaneFactors()[axis] != 1)) return;
      }
      auto time = llvm::to_vector(before.getTimeFactors().asArrayRef());
      auto lane = llvm::to_vector(before.getLaneFactors().asArrayRef());
      if (time[selected] <= 0 || time[selected] > std::numeric_limits<int64_t>::max() / (oldLanes / newLanes) ||
          after.getTimeFactors()[selected] != child / newLanes || before.getReplicaFactors()[selected] != 1) return;
      time[selected] *= oldLanes / newLanes;
      lane[selected] = newLanes;
      auto replicas = riscv_internal::staticProduct(before.getReplicaFactors().asArrayRef());
      int64_t groups = (after.getLmulEighths() + 7) / 8;
      if (!replicas || *replicas <= 0 || groups <= 0 || *replicas > std::numeric_limits<int64_t>::max() / groups) return;
      auto layout = riscv::LayoutAttr::get(rewriter.getContext(), "rvv", input.getAxisIds(),
          rewriter.getDenseI64ArrayAttr(time), rewriter.getDenseI64ArrayAttr(lane),
          before.getReplicaFactors(), before.getFragmentFactors(), before.getLocalFactors(),
          before.getSew(), after.getLmulEighths(), newLanes,
          groups * *replicas, before.getValidity());
      auto supplied = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(input, layout));
      if (!riscv::rvvLayoutRepackSlices(input, supplied)) return;
      // Split the existing numeric carrier into the domain's issue width. The
      // extract then selects complete parts without re-reading or widening it.
      // Tail validity on unchanged replica axes survives this exact lane cut.
      rewriter.setInsertionPoint(extract);
      auto conversion = riscv::ConvertLayoutOp::create(rewriter, extract.getLoc(), supplied, extract.getInput(),
          riscv_internal::conversion(rewriter, "lane_repack", "pure",
              riscv::rvvLayoutRepackTemporaryGroups(extract->getParentOfType<riscv::KernelOp>().getTarget(), input, supplied)),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(extract, conversion);
      extract.getInputMutable().assign(conversion.getResult());
    });
    llvm::SmallVector<riscv::ReduceOp> unitReductions;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      if (input && result && input.getShape()[reduce.getAxis()] == 1 &&
          riscv_internal::staticProduct(input.getShape().asArrayRef()) &&
          !reduce->getParentOfType<riscv::KernelOp>().getResourcesMaterialized())
        unitReductions.push_back(reduce);
    });
    for (auto reduce : unitReductions) {
      rewriter.setInsertionPoint(reduce);
      auto input = mlir::cast<riscv::ValueType>(reduce.getInput().getType());
      auto projected = riscv::ReshapeOp::create(rewriter, reduce.getLoc(),
          reduce.getResult().getType(), reduce.getInput(), input.getAxisIds(), mlir::DenseI64ArrayAttr());
      riscv_internal::copyOrigin(reduce, projected);
      rewriter.replaceOp(reduce, projected.getResult());
    }
    llvm::SmallVector<riscv::ReshapeOp> renames;
    getOperation().walk([&](riscv::ReshapeOp reshape) { renames.push_back(reshape); });
    for (auto reshape : renames) {
      auto input = reshape.getInput().getType(), result = reshape.getResult().getType();
      auto load = reshape.getInput().getDefiningOp<riscv::LoadOp>();
      if (!load || !load.getResult().hasOneUse() || load.getSnapshotStorage() ||
          input.getShape() != result.getShape() ||
          !llvm::equal(reshape.getOrder(), input.getAxisIds().asArrayRef()) ||
          !preservesReadOrder(load, reshape)) continue;
      auto kernel = reshape->getParentOfType<riscv::KernelOp>();
      if (kernel.getResourcesMaterialized()) continue;
      auto source = load.getRegion().getType();
      auto encoding = mlir::dyn_cast<kernel::EncodingType>(source.getEncoding());
      if (!encoding || encoding.getKind() != "dense" || source.getElements() != 1 || source.getInterleaveRows()) continue;
      auto descriptor = riscv::MemDescType::get(rewriter.getContext(), source.getEncoding(), source.getShape(),
          result.getAxisIds(), source.getStrides(), source.getOrigins(), source.getAlignment(),
          source.getAddressClass(), source.getAccess(), source.getAliasSet(), source.getLayoutIdentity(),
          source.getStorageBits(), source.getElements(), source.getInterleaveRows());
      rewriter.setInsertionPoint(load);
      auto region = riscv::ReaxisOp::create(rewriter, load.getLoc(), descriptor, load.getRegion());
      auto selected = riscv::LoadOp::create(rewriter, load.getLoc(), result, region, mlir::Value(),
          riscv_internal::unassignedAccess(rewriter), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(load, selected);
      rewriter.replaceOp(reshape, selected.getResult());
      rewriter.eraseOp(load);
    }
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
      getOperation().walk(
          [&](riscv::ConvertLayoutOp operation) { conversions.push_back(operation); });
      for (riscv::ConvertLayoutOp conversion : conversions) {
        if (!conversion)
          continue;
        // Resource materialization freezes the final physical program.  A
        // backward-rematerialization after that point would change layouts,
        // selected leaves, and the measured live-set without rerunning their
        // owners.  Keep only the dominance-safe identical-conversion CSE below
        // available when this pass is replayed on independently parsed final
        // RISC-V IR.
        if (hasFinalResourceContract(conversion))
          continue;
        if (auto previous =
                conversion.getInput().getDefiningOp<riscv::ConvertLayoutOp>()) {
          if (isPureRepresentationConversion(previous) &&
              isPureRepresentationConversion(conversion)) {
            if (previous.getInput().getType() ==
                conversion.getResult().getType()) {
              conversion.getResult().replaceAllUsesWith(previous.getInput());
              rewriter.eraseOp(conversion);
              if (previous.getResult().use_empty())
                rewriter.eraseOp(previous);
              changed = true;
              continue;
            }
            if (previous.getConversion().getKind() == "lane_repack" ||
                conversion.getConversion().getKind() == "lane_repack")
              continue;
            // Compose two pure, layout-only edges before attempting backward
            // rematerialization.  Without this, an iota or pointwise producer
            // can remain hidden behind an intermediate representation and the
            // terminal emitter would be forced to interpret a conversion
            // chain.  Memory-carrying conversions keep their explicit edges.
            if (!previous->hasAttr("source_access") &&
                !conversion->hasAttr("source_access")) {
              auto source = mlir::cast<riscv::ValueType>(
                  previous.getInput().getType());
              auto target = mlir::cast<riscv::ValueType>(
                  conversion.getResult().getType());
              auto composedEdge = riscv_internal::layoutConversion(
                  rewriter, source.getLayout(), target.getLayout());
              // A closed pack may need a larger intermediate carrier before
              // a lane-preserving resize. Do not compose away that contract.
              if (isRegisterCarrierResize(conversion) &&
                  riscv::rvvPartToLanePieces(source, previous.getResult().getType()) &&
                  (composedEdge.getKind() == "register_to_lane" ||
                   composedEdge.getKind() == "time_to_lane") &&
                  !riscv::rvvPartToLanePieces(source, target))
                continue;
              rewriter.setInsertionPoint(conversion);
              auto composed = riscv::ConvertLayoutOp::create(rewriter,
                  conversion.getLoc(), target, previous.getInput(),
                  composedEdge,
                  riscv::AccessAttr(),
                  riscv_internal::unselectedLeaf(rewriter));
              conversion.getResult().replaceAllUsesWith(composed.getResult());
              rewriter.eraseOp(conversion);
              if (previous.getResult().use_empty())
                rewriter.eraseOp(previous);
              changed = true;
              continue;
            }
          }
        }
        // Complete hardware subregisters need no lane shuffle. Freeze that
        // coordinate mapping before backward propagation can narrow its
        // producer; the explicit edge owns the result's live register parts.
        if (conversion.getConversion().getKind() == "tuple" &&
            isPureRepresentationConversion(conversion)) {
          auto target = conversion->getParentOfType<riscv::KernelOp>().getTarget();
          auto source = conversion.getInput().getType();
          auto result = conversion.getResult().getType();
          if (riscv::isRVVSubregisterRepack(target, source, result) ||
              riscv::isRVVSubregisterRepack(target, result, source)) {
            conversion.setConversionAttr(riscv_internal::conversion(rewriter,
                "lane_repack", "pure", riscv::rvvLayoutRepackTemporaryGroups(target, source, result)));
            conversion.setLeafAttr(riscv_internal::unselectedLeaf(rewriter));
            changed = true;
          }
        }
        // A selected coordinate exchange may cancel its exact inverse, but
        // must not rematerialize an arbitrary producer in a new layout.
        if (conversion.getConversion().getKind() == "lane_repack")
          continue;
        if (!isPureRepresentationConversion(conversion))
          continue;
        // Lane-preserving register resizing is already a closed conversion.
        // projectLayout chooses a minimum carrier, so rematerializing through
        // this edge could silently restore smaller operand LMULs than the
        // selected result instruction requires.
        if (isRegisterCarrierResize(conversion))
          continue;
        mlir::Operation *producer = conversion.getInput().getDefiningOp();
        if (auto share =
                mlir::dyn_cast_or_null<riscv::RegisterMaterializeOp>(producer);
            share && share.getRealization() == "physical-share" &&
            share.getInput().hasOneUse() && share.getResult().hasOneUse() &&
            preservesReadOrder(share, conversion)) {
          // A formerly shared scalar supply may become single-use after
          // partial materialization. It no longer needs a sharing boundary.
          conversion.getInputMutable().assign(share.getInput());
          rewriter.eraseOp(share);
          producer = conversion.getInput().getDefiningOp();
          changed = true;
        }
        if (auto gather =
                mlir::dyn_cast_or_null<riscv::RVVRegularRepeatGatherOp>(producer);
            gather && gather.getResults().size() == 1 &&
            gather.getResults()[0].hasOneUse() &&
            gather.getIndices().size() == 1 && preservesReadOrder(gather, conversion)) {
          auto target = conversion.getResult().getType();
          auto layout = target.getLayout();
          auto bases = gather.getPartBases().size() == 1
                           ? mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                                 gather.getPartBases()[0])
                           : mlir::DenseI64ArrayAttr();
          auto axes = target.getAxisIds().asArrayRef();
          auto sourceAxis = llvm::find(axes, gather.getReductionAxis());
          const size_t position = static_cast<size_t>(sourceAxis - axes.begin());
          const int64_t parts = sourceAxis != axes.end()
                                    ? target.getShape()[position] : 0;
          const bool singletonFreeAxes = llvm::all_of(
              llvm::enumerate(target.getShape().asArrayRef()), [&](auto entry) {
                return entry.index() == position ||
                       ((entry.value() <= 0 || entry.value() == 1) &&
                        layout.getReplicaFactors()[entry.index()] == 1);
              });
          if (layout.getCarrier() == "scalar" && parts > 0 &&
              singletonFreeAxes &&
              llvm::all_of(layout.getTimeFactors().asArrayRef(),
                           [](int64_t factor) { return factor == 1; }) &&
              llvm::all_of(layout.getLaneFactors().asArrayRef(),
                           [](int64_t factor) { return factor == 1; }) &&
              layout.getReplicaFactors()[position] == parts && bases &&
              bases.size() == 1 && bases[0] == 0 &&
              parts % gather.getRepeat() == 0 &&
              parts / gather.getRepeat() == gather.getSourceCount()) {
            llvm::SmallVector<int64_t> partBases;
            for (int64_t part = 0; part < parts; ++part)
              partBases.push_back(part / gather.getRepeat());
            rewriter.setInsertionPoint(gather);
            auto scalar = riscv::RVVRegularRepeatScalarLoadOp::create(rewriter,
                gather.getLoc(), target, gather.getField(), gather.getSourceAxis(),
                gather.getReductionAxis(), gather.getSourceBase(),
                gather.getSourceCount(), gather.getRepeat(),
                rewriter.getDenseI64ArrayAttr(partBases), gather.getAccess(),
                riscv_internal::leaf(
                    rewriter, "scalar", "regular-repeat-scalar-load",
                    "scalar.regular-repeat-load", "scalar.regular-repeat-load",
                    0, 0, 0, 0, "none", "exact",
                    {static_cast<int64_t>(gather.getSourceAxis()),
                     static_cast<int64_t>(gather.getReductionAxis()),
                     static_cast<int64_t>(gather.getSourceCount()),
                     static_cast<int64_t>(gather.getRepeat())}));
            riscv_internal::copyOrigin(gather, scalar);
            conversion.getResult().replaceAllUsesWith(scalar.getResult());
            rewriter.eraseOp(conversion);
            rewriter.eraseOp(gather);
            changed = true;
            continue;
          }
        }
        if (auto load = mlir::dyn_cast_or_null<riscv::LoadOp>(producer);
            load && load.getResult().hasOneUse() && !load.getSnapshotStorage() &&
            load.getAccess().getForm() == "unit" &&
            preservesReadOrder(load, conversion)) {
          auto source = mlir::dyn_cast<riscv::ValueType>(load.getResult().getType());
          auto target = conversion.getResult().getType();
          auto encoding = mlir::cast<kernel::EncodingType>(
              load.getRegion().getType().getEncoding());
          auto kernel = load->getParentOfType<riscv::KernelOp>();
          auto slice = load.getRegion().getDefiningOp<riscv::SliceOp>();
          auto point = slice && slice.getIndices().size() == 1
              ? slice.getIndices().front().getDefiningOp<riscv::PhysicalPointOp>()
              : riscv::PhysicalPointOp();
          if (source && kernel && encoding.getKind() == "dense" &&
              source.getShape() == target.getShape() &&
              source.getAxisIds() == target.getAxisIds() &&
              source.getElementType() == target.getElementType()) {
            auto before = source.getLayout();
            auto after = target.getLayout();
            bool linearPartition = false;
            if (point && source.getShape().size() == 1 && source.getShape()[0] > 0 &&
                conversion.getResult().hasOneUse() &&
                rematerializable(*conversion.getResult().getUsers().begin())) {
              const int64_t oldLanes = before.getLaneFactors()[0];
              const int64_t newLanes = after.getLaneFactors()[0];
              const int64_t extent = source.getShape()[0];
              linearPartition = oldLanes > 0 && newLanes > 0 && newLanes != oldLanes &&
                  (newLanes % oldLanes == 0 || oldLanes % newLanes == 0) &&
                  extent % oldLanes == 0 && extent % newLanes == 0 &&
                  before.getVl() == oldLanes && after.getVl() == newLanes &&
                  before.getTimeFactors()[0] == extent / oldLanes &&
                  after.getTimeFactors()[0] == extent / newLanes &&
                  before.getReplicaFactors()[0] == 1 && after.getReplicaFactors()[0] == 1 &&
                  before.getFragmentFactors()[0] == 1 && after.getFragmentFactors()[0] == 1 &&
                  before.getLocalFactors()[0] == 1 && after.getLocalFactors()[0] == 1;
            }
            auto ones = [](auto factors) {
              return llvm::all_of(factors.asArrayRef(), [](int64_t value) { return value == 1; });
            };
            const bool rowColumnInterchange = source.getShape().size() == 2 && slice &&
                slice.getIndices().size() == 2 &&
                llvm::all_of(slice.getIndices(), [](mlir::Value index) {
                  return index.getDefiningOp<riscv::PhysicalPointOp>() != nullptr;
                }) &&
                before.getLaneFactors() != after.getLaneFactors() &&
                before.getLaneFactors() == after.getReplicaFactors() &&
                before.getReplicaFactors() == after.getLaneFactors() &&
                llvm::count_if(before.getLaneFactors().asArrayRef(), [](int64_t v) { return v > 1; }) == 1 &&
                llvm::count_if(after.getLaneFactors().asArrayRef(), [](int64_t v) { return v > 1; }) == 1 &&
                ones(before.getTimeFactors()) && ones(after.getTimeFactors()) &&
                ones(before.getFragmentFactors()) && ones(after.getFragmentFactors()) &&
                ones(before.getLocalFactors()) && ones(after.getLocalFactors());
            if (before.getCarrier() == "rvv" && after.getCarrier() == "rvv" &&
                before.getSew() == after.getSew() && (linearPartition || rowColumnInterchange) &&
                before.getValidity() == after.getValidity() &&
                riscv::supportsRVVLayout(kernel.getTarget(), after)) {
              // Keep the same read point/domain and let memory planning select its new lane axis.
              rewriter.setInsertionPoint(load);
              auto projectedLoad = riscv::LoadOp::create(rewriter,
                  load.getLoc(), target, load.getRegion(), mlir::Value(),
                  riscv_internal::unassignedAccess(rewriter),
                  riscv_internal::unselectedLeaf(rewriter));
              riscv_internal::copyOrigin(load, projectedLoad);
              conversion.getResult().replaceAllUsesWith(projectedLoad.getResult());
              rewriter.eraseOp(conversion);
              rewriter.eraseOp(load);
              changed = true;
              continue;
            }
          }
        }
        if (auto lookup = mlir::dyn_cast_or_null<riscv::LookupOp>(producer);
            lookup && mlir::isa<riscv::MemDescType>(lookup.getTable().getType()) &&
            lookup.getResult().hasOneUse() &&
            conversion.getResult().getType().getLayout().getCarrier() ==
                "scalar" && preservesReadOrder(lookup, conversion)) {
          auto indexType =
              mlir::dyn_cast<riscv::ValueType>(lookup.getIndices().getType());
          auto kernel = conversion->getParentOfType<riscv::KernelOp>();
          auto target = conversion.getResult().getType();
          auto required = indexType && kernel
              ? riscv_internal::projectLayout(rewriter, indexType,
                                               target.getLayout(),
                                               kernel.getTarget())
              : riscv::LayoutAttr();
          if (!required)
            continue;
          rewriter.setInsertionPoint(conversion);
          mlir::Value indices = lookup.getIndices();
          auto requiredType = riscv_internal::withLayout(indexType, required);
          if (requiredType != indexType) {
            auto edge = riscv::ConvertLayoutOp::create(rewriter,
                conversion.getLoc(), requiredType, indices,
                riscv_internal::layoutConversion(
                    rewriter, indexType.getLayout(), required),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(lookup, edge);
            indices = edge.getResult();
          }
          auto rematerialized = riscv::LookupOp::create(rewriter,
              lookup.getLoc(), target, lookup.getTable(), indices,
              lookup.getBounds(), riscv_internal::unassignedAccess(rewriter),
              riscv_internal::unselectedLeaf(rewriter));
          riscv_internal::copyOrigin(lookup, rematerialized);
          conversion.getResult().replaceAllUsesWith(rematerialized.getResult());
          rewriter.eraseOp(conversion);
          rewriter.eraseOp(lookup);
          changed = true;
          continue;
        }
        if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(producer)) {
          if (!conversion.getInput().hasOneUse() ||
              (!extract->hasAttr("index_pattern") &&
               extract.getAccess().getForm() != "indexed"))
            continue;
          auto targetType =
              mlir::cast<riscv::ValueType>(conversion.getResult().getType());
          auto kernel = conversion->getParentOfType<riscv::KernelOp>();
          if (auto field = riscv::fieldReadProjection(extract.getResult())) {
            auto facts = riscv_internal::fieldFacts(field);
            auto fieldType = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
            bool crossesLayer = false;
            if (fieldType && facts.mapping == "grouped_layered" && facts.layer > 0 &&
                facts.logicalRank > 0 &&
                facts.logicalRank <= static_cast<int64_t>(fieldType.getAxisIds().size())) {
              for (int64_t axis : fieldType.getAxisIds().asArrayRef().take_back(
                       static_cast<size_t>(facts.logicalRank))) {
                auto found = llvm::find(targetType.getAxisIds().asArrayRef(), axis);
                if (found != targetType.getAxisIds().asArrayRef().end())
                  crossesLayer |= targetType.getLayout().getLaneFactors()[
                      found - targetType.getAxisIds().asArrayRef().begin()] > facts.layer;
              }
            }
            // A wider numeric pack is legal without making the storage read
            // itself a cross-layer access. Keep that representation edge.
            if (crossesLayer)
              continue;
          }
          if (extract->hasAttr("index_pattern")) {
            auto pattern = extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern");
            if (extract.getAccess().getForm() == "register" &&
                !riscv::rvvRegularGatherIndices(extract.getInput().getType(), targetType,
                                                 pattern.asArrayRef()))
              continue;
            rewriter.setInsertionPoint(conversion);
            auto rematerialized = riscv::ExtractOp::create(rewriter,
                extract.getLoc(), conversion.getResult().getType(),
                extract.getInput(), extract.getIndices(), extract.getSelectors(),
                extract.getAccess(), extract.getLeaf());
            for (auto attribute : extract->getAttrs())
              if (attribute.getName() != "operandSegmentSizes")
                rematerialized->setAttr(attribute.getName(),
                                        attribute.getValue());
            rematerialized->setAttr("leaf", riscv_internal::unselectedLeaf(rewriter));
            rematerialized->removeAttr("implementation");
            conversion.getResult().replaceAllUsesWith(rematerialized.getResult());
            rewriter.eraseOp(conversion);
            rewriter.eraseOp(extract);
            changed = true;
            continue;
          }
          if (!kernel)
            continue;
          llvm::SmallVector<riscv::LayoutAttr> requiredLayouts;
          requiredLayouts.reserve(extract.getIndices().size());
          bool legalRematerialization = true;
          unsigned indexProjectionBudget = 32;
          for (mlir::Value index : extract.getIndices()) {
            auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
            if (!indexType) {
              requiredLayouts.push_back(riscv::LayoutAttr());
              continue;
            }
            auto required = riscv_internal::projectLayout(
                rewriter, indexType, targetType.getLayout(),
                kernel.getTarget());
            if (!required) {
              legalRematerialization = false;
              break;
            }
            // An index made from pure iota/pointwise values can be rebuilt in
            // the consumer layout even when concatenating its old parts would
            // change axis order. The same bounded path must close every input.
            if (!canReprojectIndex(index, required, kernel.getTarget(), rewriter,
                                   indexProjectionBudget)) {
              legalRematerialization = false;
              break;
            }
            requiredLayouts.push_back(required);
          }
          // Backward rematerialization is an optimization, not a legality
          // escape.  Preserve the original typed conversion when projecting
          // an index would require an unsupported cross-axis lane shuffle.
          if (!legalRematerialization)
            continue;
          llvm::SmallVector<mlir::Value> indices;
          indices.reserve(extract.getIndices().size());
          rewriter.setInsertionPoint(conversion);
          for (auto [index, required] :
               llvm::zip(extract.getIndices(), requiredLayouts)) {
            auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
            if (!indexType) {
              indices.push_back(index);
              continue;
            }
            mlir::Type requiredType =
                riscv_internal::withLayout(indexType, required);
            if (requiredType == index.getType()) {
              indices.push_back(index);
              continue;
            }
            auto edge = riscv::ConvertLayoutOp::create(rewriter,
                conversion.getLoc(), requiredType, index,
                riscv_internal::layoutConversion(
                    rewriter, indexType.getLayout(), required),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(extract, edge);
            indices.push_back(edge.getResult());
          }
          auto rematerialized = riscv::ExtractOp::create(rewriter,
              extract.getLoc(), conversion.getResult().getType(),
              extract.getInput(), indices, extract.getSelectors(),
              extract.getAccess(), extract.getLeaf());
          for (auto attribute : extract->getAttrs())
            if (attribute.getName() != "operandSegmentSizes")
              rematerialized->setAttr(attribute.getName(), attribute.getValue());
          rematerialized->setAttr("leaf", riscv_internal::unselectedLeaf(rewriter));
          rematerialized->removeAttr("implementation");
          conversion.getResult().replaceAllUsesWith(rematerialized.getResult());
          rewriter.eraseOp(conversion);
          rewriter.eraseOp(extract);
          changed = true;
          continue;
        }
        if (!producer || !rematerializable(producer) ||
            (!conversion.getInput().hasOneUse() &&
             !mlir::isa<riscv::IotaOp>(producer)))
          continue;
        auto targetType = conversion.getResult().getType();
        auto targetValue = mlir::cast<riscv::ValueType>(targetType);
        bool integerReinterpret = false;
        if (auto cast = mlir::dyn_cast<riscv::CastOp>(producer)) {
          auto inputType =
              mlir::dyn_cast<riscv::ValueType>(cast.getInput().getType());
          auto castType =
              mlir::dyn_cast<riscv::ValueType>(cast.getResult().getType());
          auto sourceInteger = inputType
              ? mlir::dyn_cast<mlir::IntegerType>(inputType.getElementType())
              : mlir::IntegerType();
          auto targetInteger = castType
              ? mlir::dyn_cast<mlir::IntegerType>(castType.getElementType())
              : mlir::IntegerType();
          integerReinterpret = sourceInteger && targetInteger &&
              sourceInteger.getWidth() == targetInteger.getWidth();
          if (inputType && castType &&
              riscv_internal::logicalBitWidth(inputType.getElementType()) ==
                  riscv_internal::logicalBitWidth(castType.getElementType()) &&
              castType.getLayout() != targetValue.getLayout() &&
              !integerReinterpret)
            continue;
        }
        rewriter.setInsertionPoint(conversion);
        llvm::SmallVector<mlir::Value> operands;
        for (mlir::Value operand : producer->getOperands()) {
          auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
          if (!operandType) {
            operands.push_back(operand);
            continue;
          }
          auto kernel = conversion->getParentOfType<riscv::KernelOp>();
          const bool sameElementPointwise =
              ((mlir::isa<riscv::BinaryOp, riscv::UnaryOp>(producer) &&
                operandType.getElementType() == targetValue.getElementType()) ||
               (mlir::isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp>(producer) &&
                operandType.getLayout().getSew() == targetValue.getLayout().getSew())) &&
              operandType.getShape() == targetValue.getShape() &&
              operandType.getAxisIds() == targetValue.getAxisIds();
          riscv::LayoutAttr required = (integerReinterpret || sameElementPointwise)
              ? targetType.getLayout()
              : riscv_internal::projectLayout(
                    rewriter, operandType, targetType.getLayout(),
                    kernel.getTarget());
          if (!required) {
            conversion.emitError(
                "rematerialized producer has no legal operand layout projection");
            signalPassFailure();
            return;
          }
          mlir::Type requiredType = riscv_internal::withLayout(operandType, required);
          mlir::Value projected = operand;
          if (requiredType != operand.getType()) {
            auto edge = riscv::ConvertLayoutOp::create(rewriter,
                conversion.getLoc(), requiredType, operand,
                riscv_internal::layoutConversion(
                    rewriter, operandType.getLayout(), required),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(producer, edge);
            projected = edge.getResult();
            operandType = mlir::cast<riscv::ValueType>(requiredType);
          }

          bool properSubdomain =
              operandType.getAxisIds().size() < targetValue.getAxisIds().size() &&
              operandType.getLayout().getCarrier() == "rvv" &&
              targetValue.getLayout().getCarrier() == "rvv" &&
              operandType.getLayout().getSew() == targetValue.getLayout().getSew();
          for (auto [position, axis] :
               llvm::enumerate(operandType.getAxisIds().asArrayRef())) {
            auto found = llvm::find(targetValue.getAxisIds().asArrayRef(), axis);
            properSubdomain &=
                found != targetValue.getAxisIds().asArrayRef().end() &&
                operandType.getShape()[position] ==
                    targetValue.getShape()[static_cast<size_t>(
                        found - targetValue.getAxisIds().asArrayRef().begin())];
          }
          auto inputLanes = riscv_internal::staticProduct(
              operandType.getLayout().getLaneFactors().asArrayRef());
          auto targetLanes = riscv_internal::staticProduct(
              targetValue.getLayout().getLaneFactors().asArrayRef());
          if (properSubdomain && inputLanes && targetLanes &&
              *targetLanes > *inputLanes) {
            auto broadcastType = riscv::ValueType::get(
                rewriter.getContext(), operandType.getElementType(),
                targetValue.getShape(), targetValue.getAxisIds(),
                targetValue.getLayout());
            auto broadcast = riscv::RVVAxisBroadcastOp::create(rewriter,
                conversion.getLoc(), broadcastType, projected,
                riscv_internal::leaf(
                    rewriter, "rvv", "axis-broadcast",
                    "rvv.axis-broadcast", "rvv.axis-broadcast",
                    operandType.getLayout().getRegisterGroups(),
                    targetValue.getLayout().getRegisterGroups(), 1, 0, "none",
                    "exact", {*inputLanes, *targetLanes}));
            riscv_internal::copyOrigin(producer, broadcast);
            projected = broadcast.getResult();
          }
          operands.push_back(projected);
        }
        mlir::OperationState state(producer->getLoc(), producer->getName());
        state.addOperands(operands);
        state.addTypes(targetType);
        state.addAttributes(producer->getAttrs());
        mlir::Operation *rematerialized = rewriter.create(state);
        // A leaf belongs to its solved representation, not just its source op.
        rematerialized->setAttr("leaf", riscv_internal::unselectedLeaf(rewriter));
        rematerialized->removeAttr("implementation");
        conversion.getResult().replaceAllUsesWith(rematerialized->getResult(0));
        rewriter.eraseOp(conversion);
        if (producer->use_empty())
          rewriter.eraseOp(producer);
        changed = true;
      }

      // Sharing conversions can expose a single-use producer.  Feed that
      // change back into this same rematerialization fixed point.
      getOperation().walk([&](mlir::Block *block) {
        llvm::SmallVector<riscv::ConvertLayoutOp> available;
        for (mlir::Operation &operation : llvm::make_early_inc_range(*block)) {
          auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
          if (!conversion || !isPureRepresentationConversion(conversion))
            continue;
          auto found = llvm::find_if(available, [&](riscv::ConvertLayoutOp prior) {
            return prior.getInput() == conversion.getInput() &&
                   prior.getResult().getType() ==
                       conversion.getResult().getType() &&
                   prior->getAttr("conversion") ==
                       conversion->getAttr("conversion") &&
                   prior->getAttr("source_access") ==
                       conversion->getAttr("source_access") &&
                   prior->getAttr("leaf") == conversion->getAttr("leaf");
          });
          if (found == available.end()) {
            available.push_back(conversion);
            continue;
          }
          conversion.getResult().replaceAllUsesWith(found->getResult());
          rewriter.eraseOp(conversion);
          changed = true;
        }
      });
    }

    llvm::SmallVector<riscv::ConvertLayoutOp> packs;
    getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
      packs.push_back(conversion);
    });
    for (auto conversion : packs) {
      const auto kind = conversion.getConversion().getKind();
      if (hasFinalResourceContract(conversion) ||
          !isPureRepresentationConversion(conversion) ||
          (kind != "register_to_lane" && kind != "time_to_lane"))
        continue;
      auto source = conversion.getInput().getType();
      auto target = conversion.getResult().getType();
      if (riscv_internal::selectLaneRepack(rewriter, conversion)) continue;
      auto sourceLanes = riscv::rvvLaneCount(source);
      auto targetLanes = riscv::rvvLaneCount(target);
      if (!sourceLanes || !targetLanes || *sourceLanes <= 0 ||
          *targetLanes <= *sourceLanes || *targetLanes % *sourceLanes ||
          riscv::rvvPartToLanePieces(source, target))
        continue;
      const int64_t pieces = *targetLanes / *sourceLanes;
      const int64_t sourceLMUL = source.getLayout().getLmulEighths();
      if (sourceLMUL <= 0 || pieces > 64 / sourceLMUL)
        continue;
      const int64_t packLMUL = sourceLMUL * pieces;
      auto layout = target.getLayout();
      if (packLMUL == layout.getLmulEighths())
        continue;
      auto replicas = riscv_internal::staticProduct(
          layout.getReplicaFactors().asArrayRef());
      auto kernel = conversion->getParentOfType<riscv::KernelOp>();
      if (!replicas || *replicas <= 0 || !kernel ||
          *replicas > kernel.getTarget().getVectorRegisters() /
                          ((packLMUL + 7) / 8))
        continue;
      auto packLayout = riscv::LayoutAttr::get(
          rewriter.getContext(), layout.getCarrier(), layout.getAxisIds(),
          layout.getTimeFactors(), layout.getLaneFactors(),
          layout.getReplicaFactors(), layout.getFragmentFactors(),
          layout.getLocalFactors(), layout.getSew(), packLMUL, layout.getVl(),
          ((packLMUL + 7) / 8) * *replicas, layout.getValidity());
      auto packType = mlir::cast<riscv::ValueType>(
          riscv_internal::withLayout(target, packLayout));
      if (!riscv::supportsRVVLayout(kernel.getTarget(), packLayout) ||
          !riscv::rvvPartToLanePieces(source, packType))
        continue;
      // The pack width is fixed by its source pieces. A consumer can require
      // either less or more register capacity: preserve the closed pack first,
      // then resize without changing logical lanes or the read point.
      rewriter.setInsertionPoint(conversion);
      auto pack = riscv::ConvertLayoutOp::create(rewriter,
          conversion.getLoc(), packType, conversion.getInput(),
          riscv_internal::layoutConversion(rewriter, source.getLayout(), packLayout),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(conversion, pack);
      conversion.getInputMutable().assign(pack.getResult());
      conversion.setConversionAttr(
          riscv_internal::layoutConversion(rewriter, packLayout, layout));
      conversion.setLeafAttr(riscv_internal::unselectedLeaf(rewriter));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createCanonicalizeRISCVLayoutsPass() {
  return std::make_unique<CanonicalizeRISCVLayoutsPass>();
}
