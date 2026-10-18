#include "ContractionSupply.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace weft;

namespace {
void selectStaticPart(riscv::ProjectReductionOperandOp projection,
                      int64_t index, mlir::IRRewriter &rewriter) {
  auto input = projection.getInput().getType();
  auto result = projection.getResult().getType();
  auto layout = input.getLayout(), output = result.getLayout();
  if (layout.getCarrier() != "rvv" || output.getCarrier() != "rvv" ||
      layout.getValidity() != "full" || layout.getReplicaGroup() != 1 ||
      result != riscv_internal::projectedReductionType(
                    rewriter, input, projection.getReductionAxis()))
    return;
  for (auto factors : {layout.getReplicaFactors(), layout.getFragmentFactors(),
                       layout.getLocalFactors()})
    if (llvm::any_of(factors.asArrayRef(), [](int64_t factor) { return factor != 1; }))
      return;
  for (auto [position, axis] : llvm::enumerate(input.getAxisIds().asArrayRef())) {
    if (axis == projection.getReductionAxis()) {
      if (layout.getLaneFactors()[position] != 1 || index < 0 ||
          index >= layout.getTimeFactors()[position]) return;
    } else if (layout.getTimeFactors()[position] != 1 ||
               layout.getLaneFactors()[position] != input.getShape()[position]) {
      return;
    }
  }
  rewriter.setInsertionPoint(projection);
  auto slice = riscv::RVVIssueSliceOp::create(rewriter,
      projection.getLoc(), result, projection.getInput(),
      rewriter.getDenseI64ArrayAttr({index}), rewriter.getDenseI64ArrayAttr({0}),
      riscv_internal::leaf(rewriter, "transfer", "issue-slice", "rvv.issue-slice",
          "rvv.issue-slice", layout.getRegisterGroups(), output.getRegisterGroups(),
          0, 0, "none", "exact"));
  riscv_internal::copyOrigin(projection, slice);
  rewriter.replaceOp(projection, slice.getResult());
}
} // namespace

mlir::LogicalResult riscv_internal::specializeContractionParts(
    mlir::scf::ForOp loop) {
  auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || *lower != 0 || !upper || *upper <= 1 || !step || *step != 1 ||
      llvm::any_of(loop.getBody()->without_terminator(), [](mlir::Operation &op) {
        return op.getNumRegions() != 0;
      }))
    return mlir::success();

  bool hasFrozenParts = false;
  for (auto projection : loop.getBody()->getOps<riscv::ProjectReductionOperandOp>()) {
    if (projection.getReductionIndex() != loop.getInductionVar())
      return mlir::success();
    if (!loop.isDefinedOutsideOfLoop(projection.getInput()))
      continue;
    auto supply = queryContractionSupply(projection.getInput());
    auto type = projection.getInput().getType();
    auto axes = type.getAxisIds().asArrayRef();
    auto found = llvm::find(axes, projection.getReductionAxis());
    if (!supply.frozen || found == axes.end()) continue;
    size_t axis = found - axes.begin();
    auto layout = type.getLayout();
    auto extent = staticProduct({layout.getTimeFactors()[axis],
                                 layout.getLaneFactors()[axis]});
    hasFrozenParts |= type.getShape()[axis] == *upper && extent == upper &&
        layout.getCarrier() == "rvv" && layout.getValidity() == "full" &&
        layout.getTimeFactors()[axis] > 1 &&
        layout.getReplicaFactors()[axis] == 1 &&
        layout.getFragmentFactors()[axis] == 1 && layout.getLocalFactors()[axis] == 1;
  }
  if (!hasFrozenParts) return mlir::success();
  auto kernel = loop->getParentOfType<riscv::KernelOp>();
  auto pressure = physicalRegisterPressure(kernel);
  if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters())
    return mlir::success();
  // This preserves the existing per-member order and carries. Constant K
  // coordinates remove dynamic stream selection from numeric projections;
  // no read or snapshot crosses its original enclosing control scope.
  llvm::SmallVector<std::pair<riscv::ProjectReductionOperandOp, int64_t>> parts;
  if (mlir::failed(mlir::loopUnrollByFactor(loop, *upper,
          [&](unsigned iteration, mlir::Operation *operation, mlir::OpBuilder) {
            if (auto projection = mlir::dyn_cast<riscv::ProjectReductionOperandOp>(operation))
              parts.emplace_back(projection, iteration);
          })))
    return loop.emitError("frozen contraction parts could not be instantiated");
  mlir::IRRewriter rewriter(kernel.getContext());
  for (auto [projection, index] : parts) {
    rewriter.setInsertionPoint(projection);
    projection.getReductionIndexMutable().assign(
        mlir::arith::ConstantIndexOp::create(rewriter, projection.getLoc(), index));
    selectStaticPart(projection, index, rewriter);
  }
  return mlir::success();
}

riscv::LoadOp riscv_internal::ContractionSupply::directRead(
    mlir::Operation *consumer) const {
  auto load = read;
  if (!load || !wideningCasts.empty() || load.getSnapshotStorage() || riscv::readCrossesWrite(load, consumer) ||
      logicalAxes(value.getType()) != logicalAxes(load.getResult().getType()) ||
      logicalShape(value.getType()) != logicalShape(load.getResult().getType()) ||
      llvm::ArrayRef<int64_t>(valueToReadAxes) != logicalAxes(value.getType()))
    return {};
  return load;
}

std::optional<int64_t>
riscv_internal::ContractionSupply::readAxis(int64_t valueAxis) const {
  if (!read) return std::nullopt;
  auto axes = logicalAxes(value.getType());
  auto position = llvm::find(axes, valueAxis);
  if (position == axes.end() || valueToReadAxes.size() != axes.size())
    return std::nullopt;
  return valueToReadAxes[position - axes.begin()];
}

riscv_internal::ContractionSupply
riscv_internal::queryContractionSupply(mlir::Value value) {
  ContractionSupply result;
  result.value = value;
  result.valueToReadAxes = llvm::to_vector(logicalAxes(value.getType()));
  bool geometry = true;
  mlir::Value current = value;
  auto freeze = [&](mlir::Value input) {
    result.frozen = true;
    auto birth = queryContractionSupply(input);
    result.origin = birth.origin;
    if (!geometry || !birth.read || birth.read.getSnapshotStorage()) return;
    auto memory = birth.read.getRegion().getType();
    if (mlir::cast<kernel::EncodingType>(memory.getEncoding()).getKind() != "dense")
      return;
    auto axes = memory.getAxisIds().asArrayRef();
    for (auto [position, axis] : llvm::enumerate(result.valueToReadAxes)) {
      auto sourceAxis = birth.readAxis(axis);
      if (!sourceAxis) continue;
      auto found = llvm::find(axes, *sourceAxis);
      if (found != axes.end() && memory.getStrides()[found - axes.begin()] == 1)
        result.birthUnitAxes.push_back(logicalAxes(value.getType())[position]);
    }
  };
  while (true) {
    result.frontier = current;
    if (geometry) {
      auto type = mlir::dyn_cast<riscv::ValueType>(current.getType());
      auto axes = type ? type.getAxisIds().asArrayRef() : llvm::ArrayRef<int64_t>();
      for (auto [position, axis] : llvm::enumerate(result.valueToReadAxes)) {
        auto found = llvm::find(axes, axis);
        if (found == axes.end()) continue;
        size_t slot = found - axes.begin();
        auto layout = type.getLayout();
        if (layout.getCarrier() == "rvv" &&
            layout.getLaneFactors()[slot] == physicalExtent(current, axis) &&
            llvm::all_of(llvm::enumerate(layout.getLaneFactors().asArrayRef()),
                         [&](auto factor) {
                           return factor.index() == slot || factor.value() == 1;
                         })) {
          int64_t originalAxis = logicalAxes(value.getType())[position];
          if (!llvm::is_contained(result.contiguousAxes, originalAxis))
            result.contiguousAxes.push_back(originalAxis);
        }
      }
    }
    if (auto materialize = current.getDefiningOp<riscv::RegisterMaterializeOp>()) {
      freeze(materialize.getInput());
      break;
    }
    if (auto materialize = current.getDefiningOp<riscv::MaterializeOp>()) {
      freeze(materialize.getInput());
      break;
    }
    if (auto load = current.getDefiningOp<riscv::LoadOp>()) {
      result.origin = load;
      auto sourceElement = result.wideningCasts.empty()
                               ? logicalElement(value.getType())
                               : logicalElement(result.wideningCasts.back()->getOperand(0).getType());
      if (geometry && sourceElement == logicalElement(load.getResult().getType())) {
        result.read = load;
        result.access = load.getAccess();
        result.memorySource = true;
        auto memory = load.getRegion().getType();
        if (mlir::cast<kernel::EncodingType>(memory.getEncoding()).getKind() == "dense")
          for (auto [position, axis] : llvm::enumerate(result.valueToReadAxes)) {
            auto axes = memory.getAxisIds().asArrayRef();
            auto found = llvm::find(axes, axis);
            int64_t originalAxis = logicalAxes(value.getType())[position];
            if (found != axes.end() && memory.getStrides()[found - axes.begin()] == 1 &&
                !llvm::is_contained(result.contiguousAxes, originalAxis))
              result.contiguousAxes.push_back(originalAxis);
          }
      }
      break;
    }
    if (auto conversion = current.getDefiningOp<riscv::ConvertLayoutOp>()) {
      if (conversion.getConversion().getEffect() != "pure") break;
      current = conversion.getInput();
      continue;
    }
    auto promotion = current.getDefiningOp();
    if (mlir::isa_and_nonnull<riscv::CastOp, riscv::WidenOp>(promotion)) {
      auto input = promotion->getOperand(0);
      auto output = promotion->getResult(0);
      auto source = logicalElement(input.getType()), target = logicalElement(output.getType());
      if ((source.isF16() || source.isBF16()) && target.isF32() &&
          logicalShape(input.getType()) == logicalShape(output.getType()) &&
          logicalAxes(input.getType()) == logicalAxes(output.getType())) {
        result.wideningCasts.push_back(promotion);
        current = input;
        continue;
      }
      if (mlir::isa<riscv::CastOp>(promotion)) break;
    }
    if (auto reshape = current.getDefiningOp<riscv::ReshapeOp>()) {
      auto input = reshape.getInput().getType();
      auto output = reshape.getResult().getType();
      auto outputAxes = output.getAxisIds().asArrayRef();
      auto order = reshape.getOrder();
      geometry &= riscv::reshapeAxisPositions(input, output, order).has_value();
      if (geometry)
        for (int64_t &axis : result.valueToReadAxes) {
          auto position = llvm::find(outputAxes, axis);
          if (position == outputAxes.end()) {
            geometry = false;
            break;
          }
          axis = order[position - outputAxes.begin()];
        }
      current = reshape.getInput();
      continue;
    }
    if (auto extract = current.getDefiningOp<riscv::ExtractOp>()) {
      auto owner = queryContractionSupply(extract.getInput());
      result.origin = owner.origin;
      result.frontier = owner.frontier;
      result.frozen = owner.frozen;
      result.memorySource = owner.hasMemorySupply();
      if (!result.frozen) result.access = accessOf(current);
      break;
    }
    // Encoded projections keep their current access relation. Their origin is
    // useful to the encoded lowering, but is not a dense reload permission.
    result.origin = sourceLoad(current);
    result.access = accessOf(current);
    break;
  }
  if (!result.read) result.valueToReadAxes.clear();
  return result;
}

mlir::FailureOr<mlir::Value> riscv_internal::materializeContractionReadView(
    mlir::IRRewriter &rewriter, mlir::Value value, mlir::Operation *consumer) {
  if (!queryContractionSupply(value).wideningCasts.empty()) return mlir::Value();
  return materializeContractionStorageView(rewriter, value, consumer);
}

mlir::FailureOr<mlir::Value> riscv_internal::materializeContractionStorageView(
    mlir::IRRewriter &rewriter, mlir::Value value, mlir::Operation *consumer) {
  auto supply = queryContractionSupply(value);
  if (!supply.read || supply.frozen || supply.read.getSnapshotStorage() ||
      riscv::readCrossesWrite(supply.read, consumer))
    return mlir::Value();
  auto memory = supply.read.getRegion().getType();
  if (mlir::cast<kernel::EncodingType>(memory.getEncoding()).getKind() != "dense" ||
      memory.getElements() != 1 || memory.getInterleaveRows() != 0 ||
      logicalShape(value.getType()) != memory.getShape().asArrayRef() ||
      llvm::ArrayRef<int64_t>(supply.valueToReadAxes) != memory.getAxisIds().asArrayRef())
    return mlir::Value();
  auto axes = logicalAxes(value.getType());
  if (axes == memory.getAxisIds().asArrayRef())
    return mlir::Value(supply.read.getRegion());
  auto renamed = riscv::MemDescType::get(
      rewriter.getContext(), memory.getEncoding(), memory.getShape(),
      integers(rewriter, axes), memory.getStrides(), memory.getOrigins(),
      memory.getAlignment(), memory.getAddressClass(), memory.getAccess(),
      memory.getAliasSet(), memory.getLayoutIdentity(), memory.getStorageBits(),
      memory.getElements(), memory.getInterleaveRows());
  rewriter.setInsertionPoint(consumer);
  return mlir::Value(riscv::ReaxisOp::create(rewriter,
      consumer->getLoc(), renamed, supply.read.getRegion()));
}

riscv::ValueType riscv_internal::projectedReductionType(
    mlir::Builder &builder, riscv::ValueType input, int64_t reductionAxis) {
  llvm::SmallVector<int64_t> shape, axes, time, lane, replica, fragment, local;
  auto source = input.getLayout();
  for (size_t index = 0; index < input.getAxisIds().size(); ++index) {
    if (input.getAxisIds()[index] == reductionAxis) continue;
    shape.push_back(input.getShape()[index]);
    axes.push_back(input.getAxisIds()[index]);
    time.push_back(source.getTimeFactors()[index]);
    lane.push_back(source.getLaneFactors()[index]);
    replica.push_back(source.getReplicaFactors()[index]);
    fragment.push_back(source.getFragmentFactors()[index]);
    local.push_back(source.getLocalFactors()[index]);
  }
  int64_t laneCount = 1, replicas = 1;
  for (int64_t factor : lane) laneCount *= factor;
  for (int64_t factor : replica) replicas *= factor;
  llvm::StringRef carrier = laneCount > 1 ? "rvv" : "scalar";
  int64_t lmul = carrier == "rvv" ? source.getLmulEighths() : 0;
  int64_t groups = carrier == "rvv" ? ((lmul + 7) / 8) * replicas : 0;
  auto axisAttr = integers(builder, axes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), carrier, axisAttr, integers(builder, time),
      integers(builder, lane), integers(builder, replica),
      integers(builder, fragment), integers(builder, local), source.getSew(),
      lmul, carrier == "rvv" ? laneCount : 1, groups, source.getValidity());
  return riscv::ValueType::get(builder.getContext(), input.getElementType(),
                             integers(builder, shape), axisAttr, layout);
}

mlir::FailureOr<mlir::Value> riscv_internal::projectContractionRead(
    mlir::IRRewriter &rewriter, mlir::Value value, mlir::Type projectedType,
    llvm::ArrayRef<std::pair<int64_t, mlir::Value>> coordinates,
    mlir::Operation *consumer) {
  auto input = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!input || logicalElement(projectedType) != input.getElementType())
    return consumer->emitError("dense projection must preserve the numeric element type"), mlir::failure();
  llvm::DenseMap<int64_t, mlir::Value> selected;
  for (auto [axis, index] : coordinates) {
    if (!index || !index.getType().isIndex() || !selected.try_emplace(axis, index).second ||
        !llvm::is_contained(input.getAxisIds().asArrayRef(), axis))
      return consumer->emitError("dense projection requires unique input axes and scalar index coordinates"), mlir::failure();
  }
  llvm::SmallVector<int64_t> axes, shape;
  for (auto [axis, extent] : llvm::zip(input.getAxisIds().asArrayRef(), input.getShape().asArrayRef())) {
    if (selected.count(axis)) continue;
    axes.push_back(axis);
    shape.push_back(extent);
  }
  if (logicalAxes(projectedType) != llvm::ArrayRef<int64_t>(axes) ||
      logicalShape(projectedType) != llvm::ArrayRef<int64_t>(shape))
    return consumer->emitError("dense projection result must retain every unselected logical axis and extent"), mlir::failure();

  // Provenance alone is insufficient. This exact numeric birth must still
  // have an unchanged dense storage supply at the consumer's program point.
  auto supply = queryContractionSupply(value);
  if (!supply.read || supply.frozen || supply.read.getSnapshotStorage() ||
      riscv::readCrossesWrite(supply.read, consumer)) return mlir::Value();
  auto memory = supply.read.getRegion().getType();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(memory.getEncoding());
  if (!encoding || encoding.getKind() != "dense" || memory.getElements() != 1 ||
      memory.getInterleaveRows() || supply.valueToReadAxes.size() != input.getAxisIds().size() ||
      memory.getAxisIds().size() != input.getAxisIds().size()) return mlir::Value();

  auto target = consumer->getParentOfType<riscv::KernelOp>().getTarget();
  auto withElement = [&](mlir::Type element) -> mlir::Type {
    auto shaped = mlir::dyn_cast<riscv::ValueType>(projectedType);
    if (!shaped) return element;
    auto seed = riscv::ValueType::get(rewriter.getContext(), element,
        shaped.getShape(), shaped.getAxisIds(), shaped.getLayout());
    auto layout = projectLayout(rewriter, seed, shaped.getLayout(), target);
    return layout ? withLayout(seed, layout) : mlir::Type();
  };
  const auto sourceElement = logicalElement(supply.read.getResult().getType());
  if (supply.wideningCasts.empty() && sourceElement != input.getElementType()) return mlir::Value();
  mlir::Type loadedType = sourceElement == input.getElementType() ? projectedType : withElement(sourceElement);
  if (!loadedType)
    return consumer->emitError("dense projection storage element has no legal carrier"), mlir::failure();

  llvm::SmallVector<int64_t> namedAxes, retainedAxes, retainedShape, strides, origins;
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<mlir::Attribute> selectors;
  for (auto [position, axis] : llvm::enumerate(memory.getAxisIds().asArrayRef())) {
    auto found = llvm::find(supply.valueToReadAxes, axis);
    if (found == supply.valueToReadAxes.end()) return mlir::Value();
    size_t numeric = found - supply.valueToReadAxes.begin();
    const int64_t name = input.getAxisIds()[numeric];
    // A renamed axis is a one-to-one storage relation, not permission to read
    // another extent or to transpose the unprojected part of the value.
    if (memory.getShape()[position] != input.getShape()[numeric]) return mlir::Value();
    namedAxes.push_back(name);
    if (auto coordinate = selected.find(name); coordinate != selected.end()) {
      indices.push_back(coordinate->second);
      selectors.push_back(rewriter.getStringAttr("index"));
    } else {
      retainedAxes.push_back(name);
      retainedShape.push_back(memory.getShape()[position]);
      strides.push_back(memory.getStrides()[position]);
      origins.push_back(memory.getOrigins()[position]);
      selectors.push_back(rewriter.getStringAttr("all"));
    }
  }
  if (retainedAxes != axes || retainedShape != shape || indices.size() != coordinates.size()) return mlir::Value();
  mlir::Value descriptor = supply.read.getRegion();
  if (memory.getAxisIds().asArrayRef() != llvm::ArrayRef<int64_t>(namedAxes)) {
    auto renamed = riscv::MemDescType::get(rewriter.getContext(), memory.getEncoding(), memory.getShape(),
        integers(rewriter, namedAxes), memory.getStrides(), memory.getOrigins(), memory.getAlignment(),
        memory.getAddressClass(), memory.getAccess(), memory.getAliasSet(), memory.getLayoutIdentity(),
        memory.getStorageBits(), memory.getElements(), memory.getInterleaveRows());
    auto rename = riscv::ReaxisOp::create(rewriter, consumer->getLoc(), renamed, descriptor);
    copyOrigin(consumer, rename);
    descriptor = rename;
  }
  auto sliceType = riscv::MemDescType::get(rewriter.getContext(), memory.getEncoding(), integers(rewriter, shape),
      integers(rewriter, axes), integers(rewriter, strides), integers(rewriter, origins), memory.getAlignment(),
      "slice", memory.getAccess(), memory.getAliasSet(), memory.getLayoutIdentity(), memory.getStorageBits(),
      memory.getElements(), memory.getInterleaveRows());
  auto access = denseAccess(rewriter, sliceType, loadedType);
  if (access.getForm() == "indexed" && !target.getHasIndexedMemory())
    return consumer->emitError("dense projection requires indexed memory unsupported by target"), mlir::failure();
  auto slice = riscv::SliceOp::create(rewriter, consumer->getLoc(), sliceType, descriptor,
      indices, rewriter.getArrayAttr(selectors));
  auto load = riscv::LoadOp::create(rewriter, consumer->getLoc(), loadedType, slice,
      mlir::Value(), access, unselectedLeaf(rewriter));
  copyOrigin(consumer, slice);
  copyOrigin(consumer, load);
  mlir::Value result = load;
  // The original exact promotions remain explicit numeric operations. The
  // descriptor always describes the original bytes, including bf16 storage.
  for (auto promotion : llvm::reverse(supply.wideningCasts)) {
    auto type = withElement(logicalElement(promotion->getResult(0).getType()));
    if (!type)
      return consumer->emitError("dense projection promotion has no legal carrier"), mlir::failure();
    auto converted = riscv::CastOp::create(rewriter, promotion->getLoc(), type, result, unselectedLeaf(rewriter));
    copyOrigin(promotion, converted);
    result = converted;
  }
  if (result.getType() != projectedType)
    return consumer->emitError("dense projection did not reconstruct the requested numeric representation"), mlir::failure();
  return result;
}
