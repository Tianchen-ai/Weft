#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/Kernel/IR/ReductionRegion.h"
#include "Weft/Target/RISCVDataCache.h"
#include "Weft/Target/RISCVTargetProfile.h"
#include "Weft/Target/RISCVExtensions.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include "Weft/Dialect/Kernel/IR/StrideABI.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/Numerics.h"
#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/LocalTransfer.h"
#include "Verification.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>

using namespace weft::riscv;
using namespace weft::riscv::detail;

#include "Weft/Dialect/RISCV/IR/RISCVOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"

bool weft::riscv::supportsRVVLayout(TargetAttr target, LayoutAttr layout) {
  int64_t elen = 0;
  for (int64_t supported : target.getSupportedSEW().asArrayRef())
    elen = std::max(elen, supported);
  const bool positivePhysicalShape =
      layout && layout.getSew() > 0 && layout.getLmulEighths() > 0 &&
      layout.getVl() > 0 && layout.getRegisterGroups() > 0;
  const bool legalFractionalLMUL =
      positivePhysicalShape && elen > 0 &&
      layout.getLmulEighths() <=
          std::numeric_limits<int64_t>::max() / elen &&
      layout.getSew() <= std::numeric_limits<int64_t>::max() / 8 &&
      layout.getLmulEighths() * elen >= 8 * layout.getSew();
  const bool vlFits =
      positivePhysicalShape && target.getVlenBits() > 0 &&
      layout.getVl() <=
          std::numeric_limits<int64_t>::max() / layout.getSew() &&
      layout.getVl() * layout.getSew() <=
          std::numeric_limits<int64_t>::max() / 8 &&
      target.getVlenBits() <=
          std::numeric_limits<int64_t>::max() / layout.getLmulEighths() &&
      layout.getVl() * layout.getSew() * 8 <=
          target.getVlenBits() * layout.getLmulEighths();
  return layout && layout.getCarrier() == "rvv" && target.getHasRVV() &&
         target.getVlenBits() > 0 && target.getVectorRegisters() > 0 &&
         llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                            layout.getSew()) &&
         llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                            layout.getLmulEighths()) &&
         legalFractionalLMUL && vlFits &&
         layout.getRegisterGroups() <= target.getVectorRegisters();
}

std::optional<int64_t> weft::riscv::rvvLaneCount(ValueType value) {
  if (!value || value.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  int64_t lanes = 1;
  for (int64_t factor : value.getLayout().getLaneFactors().asArrayRef()) {
    if (factor <= 0 || lanes > std::numeric_limits<int64_t>::max() / factor)
      return std::nullopt;
    lanes *= factor;
  }
  const int64_t group = value.getLayout().getReplicaGroup();
  if (group <= 0 || lanes > std::numeric_limits<int64_t>::max() / group)
    return std::nullopt;
  return lanes * group;
}

std::optional<int64_t>
weft::riscv::rvvLaneSliceLMULEighths(TargetAttr target, ValueType source,
                                   int64_t sliceLanes) {
  auto lanes = rvvLaneCount(source);
  const int64_t sourceLMUL =
      source ? source.getLayout().getLmulEighths() : int64_t{0};
  if (!lanes || *lanes <= 0 || sliceLanes <= 0 || sliceLanes > *lanes ||
      !target || !supportsRVVLayout(target, source.getLayout()))
    return std::nullopt;
  const int64_t minimum = std::min<int64_t>(8, sourceLMUL);
  int64_t sliceLMUL = sourceLMUL;
  while (sliceLMUL > minimum) {
    const int64_t smaller = sliceLMUL / 2;
    const int64_t capacity = target.getVlenBits() * smaller /
                             (8 * source.getLayout().getSew());
    // Every complete window must stay inside the selected register group.
    if (capacity < sliceLanes ||
        (capacity % sliceLanes &&
         *lanes / sliceLanes * sliceLanes > capacity) ||
        !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), smaller))
      break;
    sliceLMUL = smaller;
  }
  return sliceLMUL;
}

std::optional<int64_t> weft::riscv::rvvPartialRepackCarrierLMULEighths(
    TargetAttr target, ValueType source, ValueType result, int64_t split) {
  if (!target || !source || !result || split <= 1 ||
      !supportsRVVLayout(target, source.getLayout()) ||
      !supportsRVVLayout(target, result.getLayout()) ||
      source.getLayout().getSew() != result.getLayout().getSew() ||
      source.getLayout().getVl() % split ||
      source.getLayout().getVl() / split != result.getLayout().getVl())
    return std::nullopt;
  const int64_t sourceLMUL = source.getLayout().getLmulEighths();
  int64_t carrier = std::min(sourceLMUL,
      std::max<int64_t>(8, result.getLayout().getLmulEighths()));
  const int64_t window = result.getLayout().getVl();
  for (; carrier <= sourceLMUL; carrier *= 2) {
    const int64_t capacity = target.getVlenBits() * carrier /
                             (8 * source.getLayout().getSew());
    if (capacity <= 0)
      return std::nullopt;
    bool contained = true;
    for (int64_t chunk = 0; chunk < split; ++chunk)
      contained &= (chunk * window) % capacity + window <= capacity;
    if (contained)
      return carrier;
  }
  return std::nullopt;
}

std::optional<int64_t>
weft::riscv::rvvPartToLanePieces(ValueType source, ValueType result) {
  if (!source || !result || source.getElementType() != result.getElementType() ||
      source.getShape() != result.getShape() ||
      source.getAxisIds() != result.getAxisIds() ||
      source.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "rvv" ||
      source.getLayout().getSew() != result.getLayout().getSew())
    return std::nullopt;
  auto sourceLayout = source.getLayout();
  auto resultLayout = result.getLayout();
  auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
  auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
  auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
  auto sourceFragment = sourceLayout.getFragmentFactors().asArrayRef();
  auto sourceLocal = sourceLayout.getLocalFactors().asArrayRef();
  auto resultTime = resultLayout.getTimeFactors().asArrayRef();
  auto resultLane = resultLayout.getLaneFactors().asArrayRef();
  auto resultReplica = resultLayout.getReplicaFactors().asArrayRef();
  auto resultFragment = resultLayout.getFragmentFactors().asArrayRef();
  auto resultLocal = resultLayout.getLocalFactors().asArrayRef();
  const size_t rank = source.getAxisIds().size();
  if (sourceTime.size() != rank || sourceLane.size() != rank ||
      sourceReplica.size() != rank || sourceFragment.size() != rank ||
      sourceLocal.size() != rank || resultTime.size() != rank ||
      resultLane.size() != rank || resultReplica.size() != rank ||
      resultFragment.size() != rank || resultLocal.size() != rank)
    return std::nullopt;

  auto represented = [](int64_t time, int64_t lane,
                        int64_t replica) -> std::optional<int64_t> {
    if (time <= 0 || lane <= 0 || replica <= 0 ||
        time > std::numeric_limits<int64_t>::max() / lane)
      return std::nullopt;
    const int64_t timeLane = time * lane;
    if (timeLane > std::numeric_limits<int64_t>::max() / replica)
      return std::nullopt;
    return timeLane * replica;
  };
  int64_t pieces = 1;
  bool movedAnyAxis = false;
  bool hasEarlierLaneAxis = false;
  for (size_t position = 0; position < rank; ++position) {
    auto sourceExtent = represented(sourceTime[position], sourceLane[position],
                                    sourceReplica[position]);
    auto resultExtent = represented(resultTime[position], resultLane[position],
                                    resultReplica[position]);
    if (!sourceExtent || !resultExtent || *sourceExtent != *resultExtent ||
        sourceFragment[position] != resultFragment[position] ||
        sourceLocal[position] != resultLocal[position])
      return std::nullopt;
    const bool moved =
        resultLane[position] > sourceLane[position] &&
        (sourceReplica[position] > resultReplica[position] ||
         sourceTime[position] > resultTime[position]);
    // Concatenating complete source vectors cannot interleave an earlier
    // lane axis with pieces taken from a later logical axis.
    if (moved && hasEarlierLaneAxis)
      return std::nullopt;
    hasEarlierLaneAxis |= sourceLane[position] > 1;
    if (!moved) {
      if (sourceLane[position] != resultLane[position])
        return std::nullopt;
      continue;
    }
    if (sourceLane[position] <= 0 ||
        resultLane[position] % sourceLane[position])
      return std::nullopt;
    const int64_t factor = resultLane[position] / sourceLane[position];
    if (factor <= 1 || pieces > std::numeric_limits<int64_t>::max() / factor)
      return std::nullopt;
    pieces *= factor;
    movedAnyAxis = true;
  }
  auto sourceLanes = rvvLaneCount(source);
  auto resultLanes = rvvLaneCount(result);
  if (!movedAnyAxis || pieces <= 1 || (pieces & (pieces - 1)) || !sourceLanes ||
      !resultLanes || *sourceLanes > std::numeric_limits<int64_t>::max() / pieces ||
      *resultLanes != *sourceLanes * pieces ||
      sourceLayout.getLmulEighths() <= 0 ||
      sourceLayout.getLmulEighths() >
          std::numeric_limits<int64_t>::max() / pieces ||
      resultLayout.getLmulEighths() !=
          sourceLayout.getLmulEighths() * pieces)
    return std::nullopt;
  return pieces;
}

std::optional<llvm::SmallVector<int64_t>>
weft::riscv::groupedMacSupplyProjection(ValueType packed, ValueType result) {
  if (!packed || !result || packed.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  auto product = [](llvm::ArrayRef<int64_t> factors) -> std::optional<int64_t> {
    int64_t value = 1;
    for (int64_t factor : factors) {
      if (factor <= 0 ||
          value > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      value *= factor;
    }
    return value;
  };
  auto timeParts = product(result.getLayout().getTimeFactors().asArrayRef());
  auto registerParts = product(result.getLayout().getReplicaFactors().asArrayRef());
  if (!timeParts || !registerParts || *timeParts <= 0 || *registerParts <= 0 ||
      *registerParts > std::numeric_limits<int64_t>::max() / *timeParts)
    return std::nullopt;

  int64_t laneAxis = 0;
  for (auto [axis, factor] :
       llvm::zip(result.getAxisIds().asArrayRef(),
                 result.getLayout().getLaneFactors().asArrayRef())) {
    if (factor <= 1)
      continue;
    if (laneAxis != 0)
      return std::nullopt;
    laneAxis = axis;
  }
  if (laneAxis <= 0)
    return std::nullopt;

  llvm::SmallVector<int64_t> mapping;
  llvm::SmallVector<llvm::SmallVector<int64_t>> supplyKeys;
  const int64_t resultParts = *timeParts * *registerParts;
  mapping.reserve(static_cast<size_t>(resultParts));
  for (int64_t part = 0; part < resultParts; ++part) {
    int64_t stream = part % *timeParts;
    int64_t replica = part / *timeParts;
    llvm::SmallVector<int64_t> timeCoordinates(result.getAxisIds().size(), 0);
    llvm::SmallVector<int64_t> replicaCoordinates(result.getAxisIds().size(), 0);
    for (int64_t position = static_cast<int64_t>(result.getAxisIds().size()) - 1;
         position >= 0; --position) {
      const int64_t time = result.getLayout().getTimeFactors()[position];
      const int64_t copies = result.getLayout().getReplicaFactors()[position];
      if (time <= 0 || copies <= 0)
        return std::nullopt;
      timeCoordinates[position] = stream % time;
      stream /= time;
      replicaCoordinates[position] = replica % copies;
      replica /= copies;
    }
    if (stream != 0 || replica != 0)
      return std::nullopt;

    llvm::SmallVector<int64_t> key;
    for (size_t position = 0; position < result.getAxisIds().size(); ++position) {
      const int64_t axis = result.getAxisIds()[position];
      if (!llvm::is_contained(packed.getAxisIds().asArrayRef(), axis))
        continue;
      key.push_back(replicaCoordinates[position]);
      if (axis == laneAxis)
        key.push_back(timeCoordinates[position]);
    }
    auto found = llvm::find(supplyKeys, key);
    if (found == supplyKeys.end()) {
      mapping.push_back(static_cast<int64_t>(supplyKeys.size()));
      supplyKeys.push_back(std::move(key));
    } else {
      mapping.push_back(static_cast<int64_t>(found - supplyKeys.begin()));
    }
  }
  if (supplyKeys.empty())
    return std::nullopt;
  return mapping;
}

namespace {

mlir::LogicalResult verifyShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes,
    bool allowEmpty = false) {
  if (!allowEmpty && shape.empty())
    return emitError() << "physical shaped value must have positive rank";
  if (shape.size() != axes.size())
    return emitError() << "logical shape and axis identity rank must match";
  llvm::DenseSet<int64_t> seen;
  for (auto [extent, axis] : llvm::zip(shape, axes)) {
    if (extent == 0)
      return emitError() << "logical extents are positive or symbolic";
    if (axis <= 0 || !seen.insert(axis).second)
      return emitError() << "logical axis identities are positive and unique";
  }
  return mlir::success();
}

bool isScalar(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

bool isPhysicalValue(mlir::Type type) {
  return isScalar(type) || mlir::isa<ValueType, FragmentType>(type);
}

std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> factors) {
  int64_t result = 1;
  for (int64_t factor : factors) {
    if (factor <= 0 ||
        result > std::numeric_limits<int64_t>::max() / factor)
      return std::nullopt;
    result *= factor;
  }
  return result;
}

llvm::ArrayRef<int64_t> shapeOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto memory = mlir::dyn_cast<MemDescType>(type))
    return memory.getShape().asArrayRef();
  if (auto local = mlir::dyn_cast<LocalType>(type))
    return local.getShape().asArrayRef();
  if (auto fragment = mlir::dyn_cast<FragmentType>(type))
    return fragment.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> axesOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto memory = mlir::dyn_cast<MemDescType>(type))
    return memory.getAxisIds().asArrayRef();
  if (auto local = mlir::dyn_cast<LocalType>(type))
    return local.getAxisIds().asArrayRef();
  if (auto fragment = mlir::dyn_cast<FragmentType>(type))
    return fragment.getAxisIds().asArrayRef();
  return {};
}

mlir::Type elementOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getElementType();
  if (auto fragment = mlir::dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  if (auto local = mlir::dyn_cast<LocalType>(type))
    return local.getElementType();
  return type;
}

bool sameLogicalDomain(mlir::Type lhs, mlir::Type rhs) {
  return shapeOf(lhs) == shapeOf(rhs) && axesOf(lhs) == axesOf(rhs);
}

bool sameStorageGeometry(AccessAttr lhs, AccessAttr rhs) {
  return lhs && rhs && lhs.getMapping() == rhs.getMapping() &&
         lhs.getGroupSize() == rhs.getGroupSize() &&
         lhs.getLayerSize() == rhs.getLayerSize() &&
         lhs.getJoinFields() == rhs.getJoinFields() &&
         lhs.getJoinLowBits() == rhs.getJoinLowBits() &&
         lhs.getJoinRole() == rhs.getJoinRole() &&
         lhs.getBitOffset() == rhs.getBitOffset() &&
         lhs.getStorageBits() == rhs.getStorageBits() &&
         lhs.getOrder() == rhs.getOrder();
}

bool canRead(llvm::StringRef access) {
  return access == "read" || access == "readwrite";
}

bool canWrite(llvm::StringRef access) {
  return access == "write" || access == "readwrite";
}

std::optional<int64_t> checkedPositiveProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

mlir::LogicalResult verifyStaticRepresentation(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, LayoutAttr layout) {
  auto time = layout.getTimeFactors().asArrayRef();
  auto lane = layout.getLaneFactors().asArrayRef();
  auto replica = layout.getReplicaFactors().asArrayRef();
  auto fragment = layout.getFragmentFactors().asArrayRef();
  auto local = layout.getLocalFactors().asArrayRef();
  if (shape.size() != time.size())
    return emitError() << "physical representation rank does not match its shape";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] < 0)
      continue;
    int64_t factors[] = {time[index], lane[index], replica[index],
                         fragment[index], local[index]};
    auto represented = checkedPositiveProduct(factors);
    if (!represented || *represented != shape[index])
      return emitError() << "physical factors overflow or do not represent "
                         << shape[index] << " coordinates at dimension "
                         << index;
  }
  return mlir::success();
}

bool sameAxisMapping(ValueType lhs, size_t lhsIndex, ValueType rhs,
                     size_t rhsIndex) {
  auto left = lhs.getLayout();
  auto right = rhs.getLayout();
  auto leftLanes = rvvLaneCount(lhs), rightLanes = rvvLaneCount(rhs);
  if ((left.getReplicaGroup() > 1 || right.getReplicaGroup() > 1) &&
      leftLanes && rightLanes && *leftLanes == *rightLanes && left.getLaneFactors()[lhsIndex] > 1) {
    auto leftStride = checkedProduct(left.getLaneFactors().asArrayRef().drop_front(lhsIndex + 1));
    auto rightStride = checkedProduct(right.getLaneFactors().asArrayRef().drop_front(rhsIndex + 1));
    if (!leftStride || !rightStride || *leftStride != *rightStride)
      return false;
  }
  const int64_t leftGroup = replicaGroupAxis(left) == static_cast<int64_t>(lhsIndex) ? left.getReplicaGroup() : 1;
  const int64_t rightGroup = replicaGroupAxis(right) == static_cast<int64_t>(rhsIndex) ? right.getReplicaGroup() : 1;
  return leftGroup == rightGroup && left.getTimeFactors()[lhsIndex] == right.getTimeFactors()[rhsIndex] &&
         left.getLaneFactors()[lhsIndex] == right.getLaneFactors()[rhsIndex] &&
         left.getReplicaFactors()[lhsIndex] ==
             right.getReplicaFactors()[rhsIndex] &&
         left.getFragmentFactors()[lhsIndex] ==
             right.getFragmentFactors()[rhsIndex] &&
         left.getLocalFactors()[lhsIndex] == right.getLocalFactors()[rhsIndex];
}

bool hasLegalRVVIndexedEMUL(LayoutAttr indexLayout, int64_t dataSEW,
                            int64_t dataLMULEighths) {
  if (!indexLayout || indexLayout.getCarrier() != "rvv" ||
      indexLayout.getSew() <= 0 || indexLayout.getLmulEighths() <= 0 ||
      dataSEW <= 0 || dataLMULEighths <= 0 ||
      indexLayout.getLmulEighths() >
          std::numeric_limits<int64_t>::max() / dataSEW ||
      dataLMULEighths >
          std::numeric_limits<int64_t>::max() / indexLayout.getSew())
    return false;
  // RVV indexed memory operations use the data LMUL and index EEW to
  // determine the index EMUL.  The selected index value must carry exactly
  // that EMUL; spelling an arbitrary index/result pair is not a legal leaf.
  return indexLayout.getLmulEighths() * dataSEW ==
         dataLMULEighths * indexLayout.getSew();
}

bool hasLegalRVVIndexedEMUL(ValueType indices, ValueType result) {
  return indices && result && result.getLayout().getCarrier() == "rvv" &&
         hasLegalRVVIndexedEMUL(indices.getLayout(), result.getLayout().getSew(),
                                result.getLayout().getLmulEighths());
}

std::optional<int64_t> physicalPartCount(ValueType value) {
  auto time =
      checkedPositiveProduct(value.getLayout().getTimeFactors().asArrayRef());
  auto replicas = checkedPositiveProduct(
      value.getLayout().getReplicaFactors().asArrayRef());
  if (!time || !replicas ||
      *time > std::numeric_limits<int64_t>::max() / *replicas)
    return std::nullopt;
  return *time * *replicas;
}

std::optional<llvm::SmallVector<int64_t>>
computeBitmaskWindowPartOffsets(
    ValueType result, llvm::ArrayRef<int64_t> windowAxes,
    llvm::ArrayRef<int64_t> windowExtents) {
  if (!result || windowAxes.empty() ||
      windowAxes.size() != windowExtents.size())
    return std::nullopt;
  auto layout = result.getLayout();
  auto axes = result.getAxisIds().asArrayRef();
  auto shape = result.getShape().asArrayRef();
  llvm::SmallVector<size_t> positions;
  llvm::DenseSet<int64_t> uniqueAxes;
  for (auto [axis, extent] : llvm::zip(windowAxes, windowExtents)) {
    auto position = llvm::find(axes, axis);
    if (extent <= 0 || position == axes.end() ||
        !uniqueAxes.insert(axis).second)
      return std::nullopt;
    const size_t index = static_cast<size_t>(position - axes.begin());
    if (shape[index] != extent)
      return std::nullopt;
    positions.push_back(index);
  }

  llvm::DenseSet<size_t> windowPositions(positions.begin(), positions.end());
  for (size_t position = 0; position < axes.size(); ++position)
    if (!windowPositions.contains(position) &&
        layout.getLaneFactors()[position] != 1)
      return std::nullopt;
  auto lanes = checkedPositiveProduct(layout.getLaneFactors().asArrayRef());
  auto windowElements = checkedPositiveProduct(windowExtents);
  if (!lanes || !windowElements)
    return std::nullopt;

  auto streams =
      checkedPositiveProduct(layout.getTimeFactors().asArrayRef());
  auto replicas =
      checkedPositiveProduct(layout.getReplicaFactors().asArrayRef());
  if (!streams || !replicas ||
      *streams > std::numeric_limits<int64_t>::max() / *replicas)
    return std::nullopt;
  const int64_t parts = *streams * *replicas;
  llvm::SmallVector<int64_t> offsets;
  offsets.reserve(parts);
  for (int64_t part = 0; part < parts; ++part) {
    int64_t remainingStream = part % *streams;
    int64_t remainingReplica = part / *streams;
    llvm::SmallVector<int64_t> timeCoordinates(axes.size(), 0);
    llvm::SmallVector<int64_t> replicaCoordinates(axes.size(), 0);
    for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
         position >= 0; --position) {
      const int64_t time = layout.getTimeFactors()[position];
      const int64_t replica = layout.getReplicaFactors()[position];
      if (time <= 0 || replica <= 0)
        return std::nullopt;
      timeCoordinates[position] = remainingStream % time;
      remainingStream /= time;
      replicaCoordinates[position] = remainingReplica % replica;
      remainingReplica /= replica;
    }
    if (remainingStream || remainingReplica)
      return std::nullopt;

    int64_t offset = 0;
    for (auto [position, extent] : llvm::zip(positions, windowExtents)) {
      const int64_t lane = layout.getLaneFactors()[position];
      const int64_t replica = layout.getReplicaFactors()[position];
      if (lane <= 0 || replica <= 0 || (lane > 1 && replica > 1))
        return std::nullopt;
      int64_t coordinate = 0;
      if (lane > 1) {
        if (timeCoordinates[position] >
            std::numeric_limits<int64_t>::max() / lane)
          return std::nullopt;
        coordinate = timeCoordinates[position] * lane;
      } else {
        if (timeCoordinates[position] >
            std::numeric_limits<int64_t>::max() / replica)
          return std::nullopt;
        coordinate =
            timeCoordinates[position] * replica + replicaCoordinates[position];
      }
      if (coordinate < 0 || coordinate >= extent ||
          offset > (std::numeric_limits<int64_t>::max() - coordinate) / extent)
        return std::nullopt;
      offset = offset * extent + coordinate;
    }
    if (offset > *windowElements - *lanes)
      return std::nullopt;
    for (int64_t linearLane = 0; linearLane < *lanes; ++linearLane) {
      int64_t remainingLane = linearLane;
      llvm::SmallVector<int64_t> laneCoordinates(axes.size(), 0);
      for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
           position >= 0; --position) {
        const int64_t lane = layout.getLaneFactors()[position];
        if (lane <= 0)
          return std::nullopt;
        laneCoordinates[position] = remainingLane % lane;
        remainingLane /= lane;
      }
      if (remainingLane)
        return std::nullopt;
      int64_t logicalOffset = 0;
      for (auto [position, extent] : llvm::zip(positions, windowExtents)) {
        const int64_t lane = layout.getLaneFactors()[position];
        const int64_t replica = layout.getReplicaFactors()[position];
        int64_t coordinate =
            lane > 1
                ? timeCoordinates[position] * lane + laneCoordinates[position]
                : timeCoordinates[position] * replica +
                      replicaCoordinates[position];
        if (coordinate < 0 || coordinate >= extent ||
            logicalOffset >
                (std::numeric_limits<int64_t>::max() - coordinate) / extent)
          return std::nullopt;
        logicalOffset = logicalOffset * extent + coordinate;
      }
      if (logicalOffset != offset + linearLane)
        return std::nullopt;
    }
    offsets.push_back(offset);
  }
  return offsets;
}

bool isNumericPhysical(mlir::Type type) {
  mlir::Type element = elementOf(type);
  return element.isIndex() ||
         mlir::isa<mlir::IntegerType, mlir::FloatType>(element);
}

FieldOp sourceField(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto field = mlir::dyn_cast<FieldOp>(definition))
      return field;
    if (auto extract = mlir::dyn_cast<ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    if (auto conversion = mlir::dyn_cast<ConvertLayoutOp>(definition)) {
      value = conversion.getInput();
      continue;
    }
    if (auto materialize = mlir::dyn_cast<RegisterMaterializeOp>(definition)) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return {};
}

LoadOp sourceLoad(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto load = mlir::dyn_cast<LoadOp>(definition))
      return load;
    if (auto conversion = mlir::dyn_cast<ConvertLayoutOp>(definition)) {
      value = conversion.getInput();
      continue;
    }
    if (auto materialize = mlir::dyn_cast<RegisterMaterializeOp>(definition)) {
      value = materialize.getInput();
      continue;
    }
    if (auto extract = mlir::dyn_cast<ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    if (auto pack = mlir::dyn_cast<EncodedLocalBindOp>(definition)) {
      value = pack.getInput();
      continue;
    }
    if (auto pack = mlir::dyn_cast<EncodedLocalPackOp>(definition)) {
      value = pack.getInput();
      continue;
    }
    break;
  }
  return {};
}

mlir::Value sourcePoint(mlir::Value value, int64_t axis) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (auto point = mlir::dyn_cast<PointType>(current.getType());
        point && point.getDomain().getAxisId() == axis)
      return current;
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      continue;
    llvm::append_range(worklist, definition->getOperands());
  }
  return {};
}

std::optional<int64_t> doubledPositive(int64_t value) {
  if (value <= 0 || value > std::numeric_limits<int64_t>::max() / 2)
    return std::nullopt;
  return value * 2;
}

int64_t localPackInterleaveRows(mlir::Value owner);

bool hasGroupedMacOperandGeometry(mlir::Operation *operation, mlir::Value lhs,
                                  mlir::Value rhs, int64_t reductionAxis,
                                  GroupedMacPlanAttr plan,
                                  LayoutAttr loadLayout,
                                  LayoutAttr partialLayout,
                                  LayoutAttr resultLayout) {
  auto lhsType = mlir::dyn_cast<ValueType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<ValueType>(rhs.getType());
  auto lhsInteger = lhsType
                        ? mlir::dyn_cast<mlir::IntegerType>(
                              lhsType.getElementType())
                        : mlir::IntegerType();
  auto rhsInteger = rhsType
                        ? mlir::dyn_cast<mlir::IntegerType>(
                              rhsType.getElementType())
                        : mlir::IntegerType();
  FieldOp lhsField = sourceField(lhs);
  FieldOp rhsField = sourceField(rhs);
  LoadOp lhsLoad = lhsField ? sourceLoad(lhsField.getOwner()) : LoadOp();
  LoadOp rhsLoad = rhsField ? sourceLoad(rhsField.getOwner()) : LoadOp();
  auto lhsMemory = lhsLoad ? lhsLoad.getRegion().getType() : MemDescType();
  auto rhsMemory = rhsLoad ? rhsLoad.getRegion().getType() : MemDescType();
  int64_t lhsInterleaveRows =
      lhsField ? localPackInterleaveRows(lhsField.getOwner()) : 0;
  if (lhsInterleaveRows <= 0 && lhsMemory)
    lhsInterleaveRows = lhsMemory.getInterleaveRows();
  AccessAttr lhsAccess = lhsField ? lhsField.getAccess() : AccessAttr();
  AccessAttr rhsAccess = rhsField ? rhsField.getAccess() : AccessAttr();
  mlir::Value lhsPoint = sourcePoint(lhs, reductionAxis);
  mlir::Value rhsPoint = sourcePoint(rhs, reductionAxis);
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel || !lhsType || !rhsType || !lhsInteger || !rhsInteger ||
      !lhsInteger.isUnsigned() || lhsInteger.getWidth() >= 8 ||
      !rhsInteger.isSigned() || rhsInteger.getWidth() != 8 || !lhsField ||
      !rhsField || !lhsLoad || !rhsLoad || !lhsPoint || !rhsPoint ||
      lhsPoint != rhsPoint || !plan ||
      lhsAccess.getMapping() != "grouped_layered" ||
      rhsAccess.getMapping() != "natural" || plan.getGroup() <= 0 ||
      lhsAccess.getGroupSize() <= 0 || lhsAccess.getLayerSize() <= 0 ||
      lhsAccess.getGroupSize() % lhsAccess.getLayerSize() ||
      lhsAccess.getLayerSize() % plan.getGroup() || lhsAccess.getBitOffset() % 8 ||
      rhsAccess.getBitOffset() % 8 ||
      (lhsAccess.getOrder() != "lo_first" &&
       lhsAccess.getOrder() != "hi_first") ||
      !supportsRVVLayout(kernel.getTarget(), lhsType.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), loadLayout) ||
      !supportsRVVLayout(kernel.getTarget(), partialLayout) ||
      !kernel.getTarget().getHasWideningInteger())
    return false;
  if (plan.getReductionAxis() != reductionAxis ||
      plan.getStorageGroup() != lhsAccess.getGroupSize() ||
      plan.getStorageLayer() != lhsAccess.getLayerSize() ||
      plan.getLogicalWidth() != lhsInteger.getWidth() ||
      plan.getLhsBitOffsetBytes() != lhsAccess.getBitOffset() / 8 ||
      plan.getRhsBitOffsetBytes() != rhsAccess.getBitOffset() / 8 ||
      plan.getRecordElements() != lhsMemory.getElements() ||
      plan.getRecordElements() != rhsMemory.getElements() ||
      plan.getInterleaveRows() != lhsInterleaveRows ||
      plan.getTermByteStride() !=
          (lhsInterleaveRows > 0 ? lhsInterleaveRows : 1))
    return false;
  int64_t laneAxis = 0;
  bool reductionLane = false;
  for (auto [axis, factor] :
       llvm::zip(lhsType.getAxisIds().asArrayRef(),
                 lhsType.getLayout().getLaneFactors().asArrayRef())) {
    if (axis == reductionAxis && factor > 1)
      reductionLane = true;
    if (axis != reductionAxis && factor > 1) {
      laneAxis = axis;
      break;
    }
  }
  bool hasCohortStride = false;
  if (lhsMemory && laneAxis > 0)
    for (auto [axis, stride] :
         llvm::zip(lhsMemory.getAxisIds().asArrayRef(),
                   lhsMemory.getStrides().asArrayRef()))
      if (axis == laneAxis && stride != 0)
        hasCohortStride = true;
  const bool result =
      lhsMemory && ((plan.getLoadForm() == "unit" &&
                     lhsInterleaveRows > 0) ||
                    (plan.getLoadForm() == "strided" && hasCohortStride &&
                     plan.getRowStrideAxis() == laneAxis));
  (void)reductionLane;
  auto sameCoordinates = [](LayoutAttr lhsLayout, LayoutAttr rhsLayout) {
    return lhsLayout.getAxisIds() == rhsLayout.getAxisIds() &&
           lhsLayout.getTimeFactors() == rhsLayout.getTimeFactors() &&
           lhsLayout.getLaneFactors() == rhsLayout.getLaneFactors() &&
           lhsLayout.getReplicaFactors() == rhsLayout.getReplicaFactors() &&
           lhsLayout.getFragmentFactors() == rhsLayout.getFragmentFactors() &&
           lhsLayout.getLocalFactors() == rhsLayout.getLocalFactors() &&
           lhsLayout.getVl() == rhsLayout.getVl() &&
           lhsLayout.getValidity() == rhsLayout.getValidity();
  };
  auto resultType = mlir::dyn_cast<ValueType>(
      mlir::cast<RVVGroupedMacLoadOp>(operation)
          .getResult()
          .getType()
          .getResultType());
  if (!resultType || resultType.getLayout() != resultLayout ||
      resultLayout.getSew() != 32 ||
      resultLayout.getLmulEighths() !=
          2 * partialLayout.getLmulEighths())
    return false;
  auto expectedGroups = [](LayoutAttr layout) {
    auto replicas = checkedPositiveProduct(
        layout.getReplicaFactors().asArrayRef());
    return replicas ? ((layout.getLmulEighths() + 7) / 8) * *replicas : -1;
  };
  if (loadLayout.getRegisterGroups() != expectedGroups(loadLayout) ||
      partialLayout.getRegisterGroups() != expectedGroups(partialLayout) ||
      resultLayout.getRegisterGroups() != expectedGroups(resultLayout))
    return false;
  for (auto [position, axis] :
       llvm::enumerate(lhsType.getAxisIds().asArrayRef())) {
    auto resultAxis = llvm::find(resultLayout.getAxisIds().asArrayRef(), axis);
    if (resultAxis == resultLayout.getAxisIds().asArrayRef().end()) {
      if (axis != reductionAxis ||
          loadLayout.getTimeFactors()[position] !=
              std::max<int64_t>(1, lhsType.getShape()[position]) ||
          loadLayout.getLaneFactors()[position] != 1 ||
          loadLayout.getReplicaFactors()[position] != 1 ||
          loadLayout.getFragmentFactors()[position] != 1 ||
          loadLayout.getLocalFactors()[position] != 1)
        return false;
      continue;
    }
    const size_t resultPosition = static_cast<size_t>(
        resultAxis - resultLayout.getAxisIds().asArrayRef().begin());
    if (loadLayout.getTimeFactors()[position] !=
            resultLayout.getTimeFactors()[resultPosition] ||
        loadLayout.getLaneFactors()[position] !=
            resultLayout.getLaneFactors()[resultPosition] ||
        loadLayout.getReplicaFactors()[position] !=
            resultLayout.getReplicaFactors()[resultPosition] ||
        loadLayout.getFragmentFactors()[position] !=
            resultLayout.getFragmentFactors()[resultPosition] ||
        loadLayout.getLocalFactors()[position] !=
            resultLayout.getLocalFactors()[resultPosition])
      return false;
  }
  for (auto [position, axis] :
       llvm::enumerate(rhsType.getAxisIds().asArrayRef())) {
    if (axis == reductionAxis)
      continue;
    auto resultAxis = llvm::find(resultLayout.getAxisIds().asArrayRef(), axis);
    if (resultAxis == resultLayout.getAxisIds().asArrayRef().end())
      return false;
    const size_t resultPosition = static_cast<size_t>(
        resultAxis - resultLayout.getAxisIds().asArrayRef().begin());
    auto rhsLayout = rhsType.getLayout();
    if (rhsLayout.getTimeFactors()[position] !=
            resultLayout.getTimeFactors()[resultPosition] ||
        rhsLayout.getLaneFactors()[position] !=
            resultLayout.getLaneFactors()[resultPosition] ||
        rhsLayout.getReplicaFactors()[position] !=
            resultLayout.getReplicaFactors()[resultPosition] ||
        rhsLayout.getFragmentFactors()[position] !=
            resultLayout.getFragmentFactors()[resultPosition] ||
        rhsLayout.getLocalFactors()[position] !=
            resultLayout.getLocalFactors()[resultPosition])
      return false;
  }
  for (int64_t axis : resultLayout.getAxisIds().asArrayRef())
    if (!llvm::is_contained(lhsType.getAxisIds().asArrayRef(), axis) &&
        !llvm::is_contained(rhsType.getAxisIds().asArrayRef(), axis))
      return false;
  return result && loadLayout.getAxisIds() == lhsType.getAxisIds() &&
         partialLayout.getAxisIds() == lhsType.getAxisIds() &&
         sameCoordinates(loadLayout, partialLayout) &&
         loadLayout.getSew() == 8 && partialLayout.getSew() == 16 &&
         partialLayout.getLmulEighths() ==
             2 * loadLayout.getLmulEighths() &&
         loadLayout.getVl() == resultLayout.getVl();
}

mlir::LogicalResult verifyBroadcastDomain(mlir::Operation *operation,
                                          mlir::Type lhs, mlir::Type rhs,
                                          mlir::Type result) {
  llvm::SmallVector<int64_t> expectedAxes(axesOf(lhs));
  for (int64_t axis : axesOf(rhs))
    if (!llvm::is_contained(expectedAxes, axis))
      expectedAxes.push_back(axis);
  if (!llvm::equal(expectedAxes, axesOf(result)))
    return operation->emitOpError(
        "pointwise result axes must follow lhs order then rhs-only axes; lhs=")
           << lhs << ", rhs=" << rhs << ", result=" << result;
  for (auto [axis, extent] : llvm::zip(axesOf(result), shapeOf(result))) {
    std::optional<int64_t> expectedExtent;
    for (mlir::Type operand : {lhs, rhs}) {
      auto axes = axesOf(operand);
      auto found = llvm::find(axes, axis);
      if (found == axes.end())
        continue;
      int64_t operandExtent =
          shapeOf(operand)[static_cast<size_t>(found - axes.begin())];
      if (!expectedExtent || *expectedExtent == 1)
        expectedExtent = operandExtent;
      else if (operandExtent != 1 && operandExtent != *expectedExtent)
        return operation->emitOpError(
            "pointwise operands disagree on a shared logical axis extent");
    }
    if (!expectedExtent || extent != *expectedExtent)
      return operation->emitOpError(
          "pointwise result extent must equal its broadcast operand extent");
  }
  auto resultValue = mlir::dyn_cast<ValueType>(result);
  if (!resultValue)
    return mlir::success();
  if (auto binary = mlir::dyn_cast<BinaryOp>(operation); binary && rvvRowScaleRows(binary))
    return mlir::success();
  for (mlir::Type operandType : {lhs, rhs}) {
    auto operand = mlir::dyn_cast<ValueType>(operandType);
    if (!operand)
      continue;
    for (auto [position, axis] :
         llvm::enumerate(operand.getAxisIds().asArrayRef())) {
      auto found = llvm::find(resultValue.getAxisIds().asArrayRef(), axis);
      if (found == resultValue.getAxisIds().asArrayRef().end())
        return operation->emitOpError(
            "pointwise physical operand axis is absent from its result");
      size_t resultPosition = static_cast<size_t>(
          found - resultValue.getAxisIds().asArrayRef().begin());
      const int64_t operandExtent = operand.getShape()[position];
      const int64_t resultExtent = resultValue.getShape()[resultPosition];
      if (operandExtent == 1 && resultExtent != 1)
        continue;
      if (!sameAxisMapping(operand, position, resultValue, resultPosition))
        return operation->emitOpError(
            "pointwise non-broadcast axis requires an explicit layout conversion");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyProjection(mlir::Operation *operation,
                                     mlir::Type base,
                                     mlir::ValueRange indices,
                                     mlir::ArrayAttr selectors,
                                     mlir::Type projected,
                                     bool allowGather) {
  if (selectors.size() > axesOf(base).size())
    return operation->emitOpError("projection has more selectors than input axes");
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  size_t indexCursor = 0;
  constexpr int64_t dynamicProjection = std::numeric_limits<int64_t>::min();
  for (size_t position = 0; position < axesOf(base).size(); ++position) {
    llvm::StringRef selector =
        position < selectors.size()
            ? mlir::cast<mlir::StringAttr>(selectors[position]).getValue()
            : llvm::StringRef("all");
    if (selector == "all") {
      expectedAxes.push_back(axesOf(base)[position]);
      expectedShape.push_back(shapeOf(base)[position]);
      continue;
    }
    if (indexCursor >= indices.size())
      return operation->emitOpError(
          "projection selector has no corresponding index operand");
    mlir::Value index = indices[indexCursor++];
    if (selector == "domain" || selector == "group_index") {
      auto point = mlir::dyn_cast<PointType>(index.getType());
      if (!point || point.getDomain().getAxisId() != axesOf(base)[position])
        return operation->emitOpError(
            "Level projection point must select the corresponding logical axis");
      if (selector == "domain") {
        expectedAxes.push_back(axesOf(base)[position]);
        expectedShape.push_back(dynamicProjection);
      }
      continue;
    }
    if (selector == "index") {
      if (!index.getType().isIndex())
        return operation->emitOpError(
            "scalar projection index must have index type");
      continue;
    }
    if (selector != "gather" || !allowGather)
      return operation->emitOpError("unknown or disallowed projection selector");
    auto gather = mlir::dyn_cast<ValueType>(index.getType());
    auto integer = gather
                       ? mlir::dyn_cast<mlir::IntegerType>(gather.getElementType())
                       : mlir::IntegerType();
    if (!integer || integer.isSigned())
      return operation->emitOpError(
          "gather projection requires shaped unsigned indices");
    for (auto [axis, extent] : llvm::zip(gather.getAxisIds().asArrayRef(),
                                        gather.getShape().asArrayRef())) {
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return operation->emitOpError(
            "gather projection disagrees on a shared logical axis extent");
      }
    }
  }
  if (indexCursor != indices.size() ||
      !llvm::equal(expectedAxes, axesOf(projected)) ||
      expectedShape.size() != shapeOf(projected).size())
    return operation->emitOpError(
        "projection operands and result logical axes do not match");
  for (auto [expected, actual] : llvm::zip(expectedShape, shapeOf(projected)))
    if (expected != dynamicProjection && expected != actual)
      return operation->emitOpError(
          "projection result has an incorrect logical extent");
  return mlir::success();
}

mlir::LogicalResult verifyStringArray(mlir::Operation *operation,
                                      mlir::ArrayAttr values,
                                      llvm::StringRef name) {
  if (llvm::any_of(values, [](mlir::Attribute value) {
        return !mlir::isa<mlir::StringAttr>(value);
      }))
    return operation->emitOpError() << name << " must contain only strings";
  return mlir::success();
}

mlir::LogicalResult verifyConversion(mlir::Operation *operation,
                                     mlir::Type input, mlir::Type result) {
  if (!isPhysicalValue(input) || !isPhysicalValue(result) ||
      !sameLogicalDomain(input, result))
    return operation->emitOpError(
        "physical conversion preserves the logical value domain");
  return verifyLeafOperation(operation);
}

std::optional<int64_t> computeRecordFieldRelativeBit(AccessAttr access,
                                                     int64_t elementBits,
                                                     int64_t logicalIndex) {
  if (!access || elementBits <= 0 || logicalIndex < 0)
    return std::nullopt;
  if (access.getMapping() == "natural") {
    if (logicalIndex > std::numeric_limits<int64_t>::max() / elementBits)
      return std::nullopt;
    return logicalIndex * elementBits;
  }
  if (access.getMapping() != "grouped_layered" ||
      access.getGroupSize() <= 0 || access.getLayerSize() <= 0 ||
      access.getGroupSize() % access.getLayerSize() ||
      (access.getOrder() != "lo_first" && access.getOrder() != "hi_first"))
    return std::nullopt;
  const int64_t group = access.getGroupSize();
  const int64_t layer = access.getLayerSize();
  const int64_t layers = group / layer;
  if (layers <= 0 || group > std::numeric_limits<int64_t>::max() / elementBits)
    return std::nullopt;
  const int64_t groupBits = group * elementBits;
  const int64_t groupIndex = logicalIndex / group;
  if (groupIndex > std::numeric_limits<int64_t>::max() / groupBits)
    return std::nullopt;
  const int64_t within = logicalIndex % group;
  const int64_t logicalLayer = within / layer;
  const int64_t physicalLayer =
      access.getOrder() == "lo_first" ? logicalLayer
                                      : layers - 1 - logicalLayer;
  const int64_t withinLayer = within % layer;
  const int64_t layerStrideBits = layers * elementBits;
  if (withinLayer > std::numeric_limits<int64_t>::max() / layerStrideBits)
    return std::nullopt;
  if (physicalLayer >
      std::numeric_limits<int64_t>::max() / elementBits)
    return std::nullopt;
  const int64_t groupOffset = groupIndex * groupBits;
  const int64_t layerOffset = withinLayer * layerStrideBits;
  const int64_t physicalOffset = physicalLayer * elementBits;
  if (groupOffset > std::numeric_limits<int64_t>::max() - layerOffset)
    return std::nullopt;
  const int64_t prefix = groupOffset + layerOffset;
  if (prefix > std::numeric_limits<int64_t>::max() - physicalOffset)
    return std::nullopt;
  return prefix + physicalOffset;
}

bool isClosedRecordCohortValue(ValueType value, RecordCohortType cohort) {
  if (!value || !cohort || value.getShape().size() != 1 ||
      value.getAxisIds().size() != 1 ||
      value.getShape()[0] != cohort.getWidth() ||
      value.getAxisIds()[0] != cohort.getDomain().getAxisId())
    return false;
  LayoutAttr layout = value.getLayout();
  return layout.getCarrier() == "rvv" && layout.getTimeFactors().size() == 1 &&
         layout.getTimeFactors()[0] == 1 &&
         layout.getLaneFactors().size() == 1 &&
         layout.getLaneFactors()[0] == cohort.getWidth() &&
         layout.getReplicaFactors().size() == 1 &&
         layout.getReplicaFactors()[0] == 1 &&
         layout.getFragmentFactors().size() == 1 &&
         layout.getFragmentFactors()[0] == 1 &&
         layout.getLocalFactors().size() == 1 &&
         layout.getLocalFactors()[0] == 1 &&
         layout.getVl() == cohort.getWidth() && layout.getValidity() == "full";
}

std::optional<int64_t> exactRecordByteStride(MemDescType memory,
                                             RecordCohortType cohort) {
  if (!memory || !cohort || memory.getShape().size() != 1 ||
      memory.getAxisIds().size() != 1 || memory.getStrides().size() != 1 ||
      memory.getAxisIds()[0] != cohort.getDomain().getAxisId() ||
      memory.getStrides()[0] != 1 || memory.getElements() <= 0 ||
      memory.getStorageBits() <= 0 || memory.getStorageBits() % 8 ||
      cohort.getPartition() % memory.getElements())
    return std::nullopt;
  const int64_t recordBytes = memory.getStorageBits() / 8;
  const int64_t recordsPerPoint = cohort.getPartition() / memory.getElements();
  if (recordsPerPoint <= 0 ||
      recordBytes > std::numeric_limits<int64_t>::max() / recordsPerPoint)
    return std::nullopt;
  return recordBytes * recordsPerPoint;
}

EncodingDeclOp findEncodingDeclaration(mlir::Operation *operation,
                                       llvm::StringRef family) {
  EncodingDeclOp result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
  return result;
}

llvm::StringRef baseEncodingFamily(mlir::Operation *operation,
                                   weft::kernel::EncodingType encoding) {
  if (encoding.getKind() == "base" || encoding.getKind() == "dense")
    return encoding.getFamily();
  llvm::StringRef result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DerivedEncodingOp derived) {
      if (result.empty() && derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derived.getSourceFamily();
    });
  return result;
}

int64_t derivedInterleaveRows(mlir::Operation *operation,
                              weft::kernel::EncodingType encoding) {
  if (!encoding || encoding.getKind() != "derived_instance")
    return 0;
  int64_t rows = 0;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DerivedEncodingOp derived) {
      if (derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        rows = derived.getInterleaveRows();
    });
  return rows;
}

int64_t localPackInterleaveRows(mlir::Value owner) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (owner) {
    mlir::Operation *definition = owner.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return 0;
    if (auto pack = mlir::dyn_cast<EncodedLocalBindOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto pack = mlir::dyn_cast<EncodedLocalPackOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto materialize = mlir::dyn_cast<MaterializeOp>(definition)) {
      if (auto plan = materialize.getLocalPackPlanAttr())
        return plan.getInterleaveRows();
    }
    if (auto extract = mlir::dyn_cast<ExtractOp>(definition)) {
      owner = extract.getInput();
      continue;
    }
    if (auto convert = mlir::dyn_cast<ConvertLayoutOp>(definition)) {
      owner = convert.getInput();
      continue;
    }
    if (auto materialize = mlir::dyn_cast<RegisterMaterializeOp>(definition)) {
      owner = materialize.getInput();
      continue;
    }
    return 0;
  }
  return 0;
}

} // namespace

std::optional<int64_t>
weft::riscv::recordFieldRelativeBit(AccessAttr access, int64_t elementBits,
                                    int64_t logicalIndex) {
  return computeRecordFieldRelativeBit(access, elementBits, logicalIndex);
}

std::optional<llvm::SmallVector<int64_t>>
weft::riscv::bitmaskWindowPartOffsets(
    ValueType result, llvm::ArrayRef<int64_t> windowAxes,
    llvm::ArrayRef<int64_t> windowExtents) {
  return computeBitmaskWindowPartOffsets(result, windowAxes, windowExtents);
}

mlir::LogicalResult LayoutAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef carrier, mlir::DenseI64ArrayAttr axisIds,
    mlir::DenseI64ArrayAttr timeFactors,
    mlir::DenseI64ArrayAttr laneFactors,
    mlir::DenseI64ArrayAttr replicaFactors,
    mlir::DenseI64ArrayAttr fragmentFactors,
    mlir::DenseI64ArrayAttr localFactors, int64_t sew,
    int64_t lmulEighths, int64_t vl, int64_t registerGroups,
    llvm::StringRef validity, int64_t replicaGroup) {
  if (replicaGroup != 1 && replicaGroup != 2)
    return emitError() << "replica group must be one or two adjacent tiles";
  if (replicaGroup != 1 &&
      (carrier != "rvv" ||
       llvm::any_of(timeFactors.asArrayRef(), [](int64_t n) { return n != 1; }) ||
       llvm::any_of(fragmentFactors.asArrayRef(), [](int64_t n) { return n != 1; }) ||
       llvm::any_of(localFactors.asArrayRef(), [](int64_t n) { return n != 1; })))
    return emitError() << "grouped replicas require a register-only, single-time RVV layout";
  if (carrier != "unassigned" && carrier != "scalar" && carrier != "rvv" &&
      carrier != "ime" && carrier != "local")
    return emitError() << "unknown physical carrier";
  const size_t rank = axisIds.size();
  if (timeFactors.size() != rank || laneFactors.size() != rank ||
      replicaFactors.size() != rank || fragmentFactors.size() != rank ||
      localFactors.size() != rank)
    return emitError() << "every logical axis requires all five physical factors";
  llvm::DenseSet<int64_t> seen;
  for (int64_t axis : axisIds.asArrayRef())
    if (axis <= 0 || !seen.insert(axis).second)
      return emitError() << "layout axis identities are positive and unique";
  auto validFactors = [&](mlir::DenseI64ArrayAttr factors) {
    return llvm::all_of(factors.asArrayRef(), [&](int64_t factor) {
      return carrier == "unassigned" ? factor >= 0 : factor > 0;
    });
  };
  if (!validFactors(timeFactors) || !validFactors(laneFactors) ||
      !validFactors(replicaFactors) || !validFactors(fragmentFactors) ||
      !validFactors(localFactors))
    return emitError() << "physical factors must be positive after layout assignment";
  if (replicaGroup != 1) {
    auto last = llvm::find_if(llvm::reverse(replicaFactors.asArrayRef()),
                             [](int64_t n) { return n > 1; });
    if (last == llvm::reverse(replicaFactors.asArrayRef()).end() || *last % replicaGroup)
      return emitError() << "grouping must partition the innermost replica axis";
  }
  if (carrier == "unassigned") {
    if (sew != 0 || lmulEighths != 0 || vl != 0 || registerGroups != 0 ||
        validity != "unassigned")
      return emitError() << "unassigned layout cannot carry physical decisions";
    return mlir::success();
  }
  for (auto factors : {timeFactors, laneFactors, replicaFactors,
                       fragmentFactors, localFactors})
    if (!checkedPositiveProduct(factors.asArrayRef()))
      return emitError()
             << "physical factor product overflows the representable program geometry";
  if (sew <= 0 || vl <= 0 || registerGroups < 0)
    return emitError() << "assigned layout requires positive SEW/vl and resources";
  if (carrier == "rvv" && lmulEighths <= 0)
    return emitError() << "RVV layout requires a legal positive LMUL";
  if (carrier != "rvv" && lmulEighths != 0)
    return emitError() << "only an RVV layout carries LMUL";
  if (validity != "full" && validity != "tail" && validity != "mask")
    return emitError() << "assigned layout has unknown validity semantics";
  return mlir::success();
}

mlir::LogicalResult DataCacheAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t sizeBytes, int64_t lineBytes, int64_t ways, int64_t sets) {
  weft::RISCVDataCache cache{sizeBytes, lineBytes, ways, sets};
  std::string error;
  if (!cache.validate(error)) return emitError() << error;
  return mlir::success();
}

mlir::LogicalResult TargetAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef triple, llvm::StringRef march, llvm::StringRef abi,
    bool hasRVV, bool hasVectorF16, bool hasVectorF16Conversion, bool hasIndexedMemory,
    bool hasSegmentMemory, bool hasWideningInteger, bool hasWideningFloat,
    int64_t vlenBits, int64_t vectorRegisters,
    int64_t maxPrivateStackBytes, mlir::DenseI64ArrayAttr supportedSEW,
    mlir::DenseI64ArrayAttr legalLMULEighths,
    int64_t maxWideningCombineGroups,
    llvm::StringRef partialCombinePolicy, llvm::StringRef recordAxisPolicy,
    mlir::ArrayAttr fragments, mlir::Attribute dataCache) {
  if (triple.empty() || march.empty() || abi.empty() || vlenBits <= 0 ||
      vectorRegisters <= 0 || maxPrivateStackBytes < 0 || supportedSEW.empty() ||
      legalLMULEighths.empty() || maxWideningCombineGroups <= 0 ||
      maxWideningCombineGroups > vectorRegisters)
    return emitError() << "RISC-V target facts must be complete";
  if (!hasRVV || (!hasWideningInteger && !hasWideningFloat))
    return emitError()
           << "current RISC-V physical dialect requires RVV and at least one widening class";
  if (vlenBits < 128 || vlenBits > 65536 || (vlenBits & (vlenBits - 1)))
    return emitError() << "full V requires a power-of-two VLEN from 128 to 65536 bits";
  RISCVTargetProfile expected;
  std::string error;
  if (!parseRISCVTargetProfile(march, abi, vlenBits, "none",
          maxWideningCombineGroups, partialCombinePolicy, recordAxisPolicy,
          expected, error))
    return emitError() << "invalid target profile: " << error;
  if (hasRVV != expected.hasRVV ||
      hasVectorF16 != expected.hasVectorF16 ||
      hasVectorF16Conversion != expected.hasVectorF16Conversion ||
      hasIndexedMemory != expected.hasIndexedMemory ||
      hasSegmentMemory != expected.hasSegmentMemory ||
      hasWideningInteger != expected.hasWideningInteger ||
      hasWideningFloat != expected.hasWideningFloat)
    return emitError() << "serialized vector capability facts disagree with the target ISA";
  auto sameDomain = [](mlir::DenseI64ArrayAttr actual,
                       const auto &domain) {
    if (actual.size() != domain.size()) return false;
    llvm::SmallDenseSet<int64_t> distinct;
    for (int64_t value : actual.asArrayRef())
      if (!llvm::is_contained(domain, value) || !distinct.insert(value).second)
        return false;
    return true;
  };
  if (vectorRegisters != expected.vectorRegisters ||
      !sameDomain(supportedSEW, expected.supportedSEW) ||
      !sameDomain(legalLMULEighths, expected.legalLMULEighths))
    return emitError() << "full-V register count, SEW and LMUL domains disagree with hardware facts";
  if (triple != expected.triple)
    return emitError() << "target triple disagrees with the authoritative ISA/ABI profile";
  if (partialCombinePolicy != "independent-multilevel" &&
      partialCombinePolicy != "sequential")
    return emitError() << "unknown partial combine structural priority";
  if (recordAxisPolicy != "within-record" &&
      recordAxisPolicy != "across-records")
    return emitError() << "unknown record-axis placement structural priority";
  if (!mlir::isa<mlir::UnitAttr, DataCacheAttr>(dataCache))
    return emitError() << "data cache facts must be an explicit geometry or unit for unavailable";
  if (llvm::any_of(supportedSEW.asArrayRef(), [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(legalLMULEighths.asArrayRef(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(fragments, [](mlir::Attribute value) {
        return !mlir::isa<FragmentCapabilityAttr>(value);
      }))
    return emitError() << "target capability domains are malformed";
  for (mlir::Attribute attribute : fragments) {
    auto fragment = mlir::cast<FragmentCapabilityAttr>(attribute);
    const auto *contract = weft::findRISCVFragmentCapability(fragment.getInstruction());
    if (!contract || vlenBits != contract->vlenBits ||
        !march.starts_with("rv" + std::to_string(contract->xlen)) ||
        !triple.starts_with("riscv" + std::to_string(contract->xlen)))
      return emitError() << "fragment ISA/VLEN requirements disagree with the target";
    for (int64_t bits : {fragment.getLhsBits(), fragment.getRhsBits(),
                         fragment.getAccumulatorBits()})
      if (!llvm::is_contained(supportedSEW.asArrayRef(), bits))
        return emitError() << "fragment handoff requires an unsupported RVV SEW";
  }
  return mlir::success();
}

mlir::LogicalResult AccessAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef form, llvm::StringRef mapping, int64_t alignment,
    int64_t indexSEW, int64_t segmentFields, int64_t groupSize,
    int64_t layerSize, int64_t joinFields, int64_t joinLowBits,
    int64_t joinRole, int64_t bitOffset, int64_t storageBits,
    llvm::StringRef order, int64_t fieldChunkBytes) {
  if (form != "unassigned" && form != "unit" && form != "strided" &&
      form != "indexed" && form != "segment" && form != "register" &&
      form != "local")
    return emitError() << "unknown physical memory form";
  if (mapping != "dense" && mapping != "natural" &&
      mapping != "grouped_layered" && mapping != "joined" &&
      mapping != "opaque")
    return emitError() << "unknown storage mapping";
  if (alignment <= 0 || indexSEW < 0 || segmentFields < 0 || groupSize < 0 ||
      layerSize < 0 || joinFields < 0 || joinLowBits < 0 || joinRole < 0 ||
      bitOffset < 0 || storageBits < 0 || fieldChunkBytes <= 0)
    return emitError() << "memory access parameters are invalid";
  if (fieldChunkBytes != 1 &&
      (mapping == "dense" || mapping == "opaque" || bitOffset % 8 ||
       storageBits <= 0 || storageBits % 8 || storageBits / 8 % fieldChunkBytes))
    return emitError() << "field chunks must partition byte-addressable field storage";
  if (order != "none" && order != "lo_first" && order != "hi_first")
    return emitError() << "unknown bit-layer order";
  if (form == "indexed" && indexSEW == 0)
    return emitError() << "indexed access requires an index SEW";
  if (form == "segment" && segmentFields < 2)
    return emitError() << "segment access requires at least two fields";
  if (mapping == "joined" &&
      (groupSize <= 0 || joinFields <= 1 || joinLowBits <= 0 ||
       joinRole >= joinFields))
    return emitError() << "joined storage mapping requires complete field geometry";
  if (mapping == "grouped_layered" &&
      (groupSize <= 0 || layerSize <= 0 || groupSize % layerSize ||
       storageBits <= 0 || (order != "lo_first" && order != "hi_first")))
    return emitError()
           << "grouped/layered storage mapping requires complete group, layer, bit, and order geometry";
  return mlir::success();
}

mlir::LogicalResult FieldStorePlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t logicalBits, int64_t storeBits,
    int64_t lanes, int64_t sourcesPerStore, mlir::DenseI64ArrayAttr byteOffsets,
    mlir::DenseI64ArrayAttr sourceParts, mlir::DenseI64ArrayAttr bitShifts) {
  if ((kind != "scalar" && kind != "scalar_bytes" && kind != "natural" && kind != "layered") ||
      logicalBits <= 0 || logicalBits > 32 ||
      (storeBits != 8 && storeBits != 16 && storeBits != 32) || lanes <= 0 ||
      sourcesPerStore <= 0 || sourcesPerStore > 8 || byteOffsets.empty() ||
      sourceParts.size() != byteOffsets.size() * sourcesPerStore ||
      bitShifts.size() != sourceParts.size())
    return emitError() << "field store plan has incomplete bounded packing geometry";
  for (int64_t offset : byteOffsets.asArrayRef())
    if (offset < 0)
      return emitError() << "field store byte offsets must be nonnegative";
  for (auto [part, shift] : llvm::zip(sourceParts.asArrayRef(), bitShifts.asArrayRef()))
    if (part < 0 || shift < 0 || shift >= 32)
      return emitError() << "field store source/bit projection is invalid";
  if (kind == "layered") {
    if ((logicalBits != 1 && logicalBits != 2 && logicalBits != 4) ||
        storeBits != 8 || sourcesPerStore != 8 / logicalBits)
      return emitError() << "layered field store must assemble complete bytes";
  } else if (sourcesPerStore != 1 ||
             (kind != "scalar_bytes" && storeBits != logicalBits) ||
             (kind == "scalar_bytes" && (storeBits != 8 || lanes != 1)) ||
             (kind == "scalar" && (lanes != 1 || byteOffsets.size() != 1))) {
    return emitError() << "natural field store plan changes the element representation";
  }
  return mlir::success();
}

mlir::LogicalResult ScheduleAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t unroll, int64_t pipelineDepth,
    int64_t bufferCount, int64_t prefetchDistance) {
  if (kind != "sequential" && kind != "pipelined")
    return emitError() << "unknown physical schedule kind";
  if (unroll <= 0 || pipelineDepth <= 0 || bufferCount <= 0 ||
      prefetchDistance < 0)
    return emitError() << "schedule parameters must be positive";
  if ((pipelineDepth == 1) != (kind == "sequential") ||
      bufferCount < pipelineDepth)
    return emitError() << "schedule kind, depth, and buffer count disagree";
  return mlir::success();
}

mlir::LogicalResult PartialTopologyAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t rootOperand,
    mlir::DenseI64ArrayAttr partialAxes, mlir::DenseI64ArrayAttr outputAxes,
    int64_t sourceSlots, int64_t partialSlots, int64_t outputReplicas,
    int64_t laneSplit, int64_t combineArity,
    mlir::DenseI64ArrayAttr slotOrder, int64_t resourceGroups) {
  if (kind != "unassigned" && kind != "sequential_fused" &&
      kind != "sequential_per_stream" && kind != "independent" &&
      kind != "replica_reduced" &&
      kind != "scaled" && kind != "reduced_scaled" &&
      kind != "level_scaled" && kind != "nested_scaled_stream" &&
      kind != "merged" && kind != "layered")
    return emitError() << "unknown partial topology kind";
  if (kind == "unassigned") {
    if (rootOperand != -1 || !partialAxes.empty() || !outputAxes.empty() ||
        sourceSlots != 0 || partialSlots != 0 || outputReplicas != 0 ||
        laneSplit != 0 || combineArity != 0 || !slotOrder.empty() ||
        resourceGroups != 0)
      return emitError()
             << "unassigned partial topology cannot carry physical decisions";
    return mlir::success();
  }
  if (rootOperand < -1 || sourceSlots <= 0 || partialSlots <= 0 ||
      outputReplicas <= 0 || laneSplit <= 0 || combineArity <= 0 ||
      resourceGroups <= 0 ||
      slotOrder.size() != static_cast<size_t>(partialSlots))
    return emitError() << "selected partial topology has incomplete geometry";
  llvm::SmallVector<int64_t> ordered(slotOrder.asArrayRef());
  llvm::sort(ordered);
  for (auto [index, slot] : llvm::enumerate(ordered))
    if (slot != static_cast<int64_t>(index))
      return emitError() << "partial topology slot order is not a permutation";
  llvm::SmallVector<int64_t> uniquePartialAxes(partialAxes.asArrayRef());
  llvm::SmallVector<int64_t> uniqueOutputAxes(outputAxes.asArrayRef());
  llvm::sort(uniquePartialAxes);
  llvm::sort(uniqueOutputAxes);
  if (std::adjacent_find(uniquePartialAxes.begin(), uniquePartialAxes.end()) !=
          uniquePartialAxes.end() ||
      std::adjacent_find(uniqueOutputAxes.begin(), uniqueOutputAxes.end()) !=
          uniqueOutputAxes.end())
    return emitError() << "partial topology axes must be unique";
  if (llvm::any_of(partialAxes.asArrayRef(), [&](int64_t axis) {
        return axis == 0 || llvm::is_contained(outputAxes.asArrayRef(), axis);
      }) ||
      llvm::any_of(outputAxes.asArrayRef(),
                   [](int64_t axis) { return axis == 0; }))
    return emitError()
           << "partial and output axes must be nonzero and disjoint";
  const bool ownsLaneSplit = kind == "scaled" || kind == "reduced_scaled" ||
                             kind == "nested_scaled_stream";
  if (laneSplit > 1 &&
      (!ownsLaneSplit || sourceSlots * laneSplit != partialSlots))
    return emitError()
           << "lane-split topology must close one scaled source carrier and its partial slots";
  if ((kind == "independent" || kind == "replica_reduced" ||
       kind == "scaled" ||
       kind == "reduced_scaled" || kind == "level_scaled" ||
       kind == "nested_scaled_stream" ||
       kind == "layered") &&
      (combineArity > partialSlots || partialSlots % combineArity))
    return emitError()
           << "partial topology combine arity must divide its partial slots";
  const bool ownsRoot = kind == "layered";
  if (ownsRoot != (rootOperand >= 0))
    return emitError()
           << "only a layered topology may own one root operand";
  return mlir::success();
}

mlir::LogicalResult PackedPlaneMergePlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t logicalAxis, int64_t logicalElements, int64_t lowBits,
    int64_t highBits, int64_t groupSize, int64_t lowLayerBytes,
    int64_t highLayerBytes, int64_t lowByteOffset, int64_t highByteOffset,
    int64_t insertBit) {
  if (logicalAxis <= 0 || logicalElements <= 0 || logicalElements % 4 ||
      lowBits <= 0 || highBits <= 0 || lowBits >= 8 || highBits >= 8 ||
      groupSize != logicalElements || lowLayerBytes <= 0 ||
      highLayerBytes <= 0 || lowLayerBytes % 4 || highLayerBytes % 4 ||
      groupSize * lowBits != lowLayerBytes * 8 ||
      groupSize * highBits != highLayerBytes * 8 || lowByteOffset < 0 ||
      highByteOffset < 0 || insertBit < lowBits || insertBit + highBits > 8)
    return emitError()
           << "packed plane merge plan requires two byte-parallel sub-byte layers over one four-element-aligned logical group";
  return mlir::success();
}

static bool matchesPartialRepackLeaf(mlir::Attribute attribute,
                                    PartialSetType source,
                                    PartialSetType result, int64_t split) {
  if (split == 1)
    return source == result && mlir::isa<mlir::UnitAttr>(attribute);
  auto leaf = mlir::dyn_cast<LeafAttr>(attribute);
  if (!source || !result || !leaf || split <= 1 ||
      leaf.getEngine() != "rvv" || leaf.getFamily() != "partial-repack" ||
      leaf.getSpelling() != leaf.getInstruction() ||
      leaf.getOperandGroups() != source.getResourceGroups() ||
      leaf.getResultGroups() != result.getResourceGroups())
    return false;
  auto parameters = leaf.getParameters().asArrayRef();
  if (parameters.size() < 2 || parameters[0] != source.getReductionAxis() ||
      parameters[1] != split)
    return false;
  if (leaf.getInstruction() == "rvv.partial-repack.split")
    return parameters.size() == 2 && leaf.getTemporaryGroups() == 0;
  if (leaf.getInstruction() != "rvv.partial-repack.slice" || parameters.size() != 3)
    return false;
  const int64_t carrier = parameters[2];
  return carrier > 0 && !(carrier & (carrier - 1)) &&
      carrier >= result.getPartialType().getLayout().getLmulEighths() &&
      carrier <= source.getPartialType().getLayout().getLmulEighths() &&
      leaf.getTemporaryGroups() == result.getResourceGroups() +
          std::max<int64_t>(1, carrier / 8);
}

mlir::LogicalResult PartialLayoutPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type sourceLhsType, mlir::Type sourceRhsType,
    mlir::Type sourceSlotType, mlir::Type splitSourceSlotType,
    mlir::Type partialSlotType, mlir::Type widenedSlotType,
    mlir::Type reducedSlotType, mlir::Type vectorSlotType,
    mlir::Type sourceSetType, mlir::Type repackedSetType,
    mlir::Type partialSetType, mlir::Type scaledSetType,
    mlir::Type directSetType, mlir::Type reducedSetType,
    mlir::Type scaleCombinedSetType, mlir::Type fullScaledSetType,
    mlir::DenseI64ArrayAttr sourceLhsParts,
    mlir::DenseI64ArrayAttr sourceRhsParts,
    mlir::DenseI64ArrayAttr sourceLhsOffsets,
    mlir::DenseI64ArrayAttr sourceRhsOffsets, mlir::Attribute repackLeaf) {
  auto sourceLhs = mlir::dyn_cast<weft::riscv::ValueType>(sourceLhsType);
  auto sourceRhs = mlir::dyn_cast<weft::riscv::ValueType>(sourceRhsType);
  auto source = mlir::dyn_cast<weft::riscv::ValueType>(sourceSlotType);
  auto split = mlir::dyn_cast<weft::riscv::ValueType>(splitSourceSlotType);
  auto partial = mlir::dyn_cast<weft::riscv::ValueType>(partialSlotType);
  auto widened = mlir::dyn_cast<weft::riscv::ValueType>(widenedSlotType);
  auto reduced = mlir::dyn_cast<weft::riscv::ValueType>(reducedSlotType);
  auto vector = mlir::dyn_cast<weft::riscv::ValueType>(vectorSlotType);
  auto sourceSet = mlir::dyn_cast<weft::riscv::PartialSetType>(sourceSetType);
  auto repackedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(repackedSetType);
  auto partialSet = mlir::dyn_cast<weft::riscv::PartialSetType>(partialSetType);
  auto scaledSet = mlir::dyn_cast<weft::riscv::PartialSetType>(scaledSetType);
  auto directSet = mlir::dyn_cast<weft::riscv::PartialSetType>(directSetType);
  auto reducedSet = mlir::dyn_cast<weft::riscv::PartialSetType>(reducedSetType);
  auto scaleCombinedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(scaleCombinedSetType);
  auto fullScaledSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(fullScaledSetType);
  auto optionalPhysicalValue = [](mlir::Type type) {
    return mlir::isa<weft::riscv::ValueType, mlir::NoneType>(type);
  };
  auto optionalPartialSet = [](mlir::Type type) {
    return mlir::isa<weft::riscv::PartialSetType, mlir::NoneType>(type);
  };
  if (!partial || !optionalPhysicalValue(sourceLhsType) ||
      !optionalPhysicalValue(sourceRhsType) ||
      !optionalPhysicalValue(sourceSlotType) ||
      !optionalPhysicalValue(splitSourceSlotType) ||
      !optionalPhysicalValue(widenedSlotType) ||
      !optionalPhysicalValue(reducedSlotType) ||
      !optionalPhysicalValue(vectorSlotType) ||
      !optionalPartialSet(sourceSetType) ||
      !optionalPartialSet(repackedSetType) ||
      !optionalPartialSet(partialSetType) ||
      !optionalPartialSet(scaledSetType) ||
      !optionalPartialSet(directSetType) ||
      !optionalPartialSet(reducedSetType) ||
      !optionalPartialSet(scaleCombinedSetType) ||
      !optionalPartialSet(fullScaledSetType))
    return emitError()
           << "partial layout plan requires one concrete partial slot and typed optional slots";
  if (partial.getLayout().getCarrier() != "rvv" ||
      (widened && widened.getLayout().getCarrier() != "rvv") ||
      (reduced && reduced.getLayout().getCarrier() != "rvv") ||
      (vector && vector.getLayout().getCarrier() != "rvv"))
    return emitError() << "partial layout plan values must use the RVV carrier";
  const bool hasSourcePlan = static_cast<bool>(source) || static_cast<bool>(split) ||
                             static_cast<bool>(sourceLhs) ||
                             static_cast<bool>(sourceRhs);
  if (hasSourcePlan != (static_cast<bool>(source) && static_cast<bool>(split) &&
                        static_cast<bool>(sourceLhs) &&
                        static_cast<bool>(sourceRhs) && sourceSet &&
                        repackedSet && reducedSet && scaleCombinedSet) ||
      (source && (sourceLhs.getLayout().getCarrier() != "rvv" ||
                  sourceRhs.getLayout().getCarrier() != "rvv" ||
                  source.getLayout().getCarrier() != "rvv" ||
                  split.getLayout().getCarrier() != "rvv")))
    return emitError() << "partial source layout plan is incomplete";
  if (!hasSourcePlan &&
      (sourceSet || repackedSet ||
       !sourceLhsParts.empty() || !sourceRhsParts.empty() ||
       !sourceLhsOffsets.empty() || !sourceRhsOffsets.empty()))
    return emitError()
           << "partial layout plan without source coalescing cannot carry source maps";
  if (hasSourcePlan &&
      (sourceLhsParts.empty() ||
       sourceLhsParts.size() != sourceRhsParts.size() ||
       sourceLhsParts.size() != sourceLhsOffsets.size() ||
       sourceLhsParts.size() != sourceRhsOffsets.size() ||
       llvm::any_of(sourceLhsParts.asArrayRef(),
                    [](int64_t value) { return value < 0; }) ||
       llvm::any_of(sourceRhsParts.asArrayRef(),
                    [](int64_t value) { return value < 0; }) ||
       llvm::any_of(sourceLhsOffsets.asArrayRef(),
                    [](int64_t value) { return value < 0; }) ||
       llvm::any_of(sourceRhsOffsets.asArrayRef(),
                    [](int64_t value) { return value < 0; })))
    return emitError()
           << "partial layout plan requires one complete non-negative source map";
  if (hasSourcePlan &&
      (sourceSet.getPartialType() != source ||
       repackedSet.getPartialType() != partial ||
       reducedSet.getPartialType() != reduced ||
       scaleCombinedSet.getPartialType() != reduced ||
       sourceSet.getReductionAxis() != repackedSet.getReductionAxis() ||
       sourceSet.getReductionAxis() != reducedSet.getReductionAxis() ||
       sourceSet.getReductionAxis() != scaleCombinedSet.getReductionAxis() ||
       sourceSet.getSlots() <= 0 || repackedSet.getSlots() <= 0 ||
       repackedSet.getSlots() % sourceSet.getSlots() ||
       reducedSet.getSlots() != repackedSet.getSlots() ||
       scaleCombinedSet.getSlots() != sourceSet.getSlots()))
    return emitError()
           << "partial source layout plan has inconsistent typed set geometry";
  if ((hasSourcePlan && !matchesPartialRepackLeaf(repackLeaf, sourceSet,
          repackedSet, repackedSet.getSlots() / sourceSet.getSlots())) ||
      (!hasSourcePlan && !mlir::isa<mlir::UnitAttr>(repackLeaf)))
    return emitError() << "partial source layout plan has no matching frozen repack leaf";
  auto partialMatches = [](weft::riscv::PartialSetType set,
                           weft::riscv::ValueType value) {
    return !set || (value && set.getPartialType() == value &&
                    set.getSlots() > 0 && set.getTermsPerSlot() > 0 &&
                    set.getResourceGroups() > 0);
  };
  if (!partialMatches(partialSet, partial) ||
      !partialMatches(scaledSet, widened) ||
      !partialMatches(directSet, partial) ||
      !partialMatches(reducedSet, reduced) ||
      !partialMatches(scaleCombinedSet, reduced) ||
      !partialMatches(fullScaledSet, widened))
    return emitError()
           << "partial layout plan contains a set with a mismatched slot type";
  return mlir::success();
}

static bool isWideningPartialCombineStage(weft::riscv::PartialSetType source,
                                          weft::riscv::PartialSetType result,
                                          int64_t arity) {
  auto sourceValue = source.getPartialType();
  auto resultValue = result.getPartialType();
  auto sourceElement =
      mlir::dyn_cast<mlir::IntegerType>(sourceValue.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(resultValue.getElementType());
  auto sourceLayout = sourceValue.getLayout();
  auto resultLayout = resultValue.getLayout();
  return arity == 2 && sourceElement && resultElement &&
         sourceElement.isSigned() && resultElement.isSigned() &&
         sourceElement.getWidth() == 16 && resultElement.getWidth() == 32 &&
         sourceValue.getShape() == resultValue.getShape() &&
         sourceValue.getAxisIds() == resultValue.getAxisIds() &&
         sourceLayout.getCarrier() == "rvv" &&
         resultLayout.getCarrier() == "rvv" &&
         sourceLayout.getAxisIds() == resultLayout.getAxisIds() &&
         sourceLayout.getTimeFactors() == resultLayout.getTimeFactors() &&
         sourceLayout.getLaneFactors() == resultLayout.getLaneFactors() &&
         sourceLayout.getReplicaFactors() == resultLayout.getReplicaFactors() &&
         sourceLayout.getFragmentFactors() ==
             resultLayout.getFragmentFactors() &&
         sourceLayout.getLocalFactors() == resultLayout.getLocalFactors() &&
         sourceLayout.getSew() == 16 && resultLayout.getSew() == 32 &&
         resultLayout.getLmulEighths() ==
             sourceLayout.getLmulEighths() * 2 &&
         sourceLayout.getVl() == resultLayout.getVl() &&
         sourceLayout.getValidity() == resultLayout.getValidity();
}

mlir::LogicalResult PartialCombinePlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef realization, mlir::Type sourceSetType,
    mlir::Type reducedSetType, mlir::Type finalSetType,
    mlir::ArrayAttr combineSetTypes,
    mlir::DenseI64ArrayAttr combineArities,
    mlir::DenseI64ArrayAttr lhsParts, mlir::DenseI64ArrayAttr rhsParts,
    mlir::DenseI64ArrayAttr scaleParts, llvm::StringRef multiplyInstruction,
    llvm::StringRef finalizeInstruction, int64_t outputParts,
    int64_t resourceGroups) {
  auto source = mlir::dyn_cast<weft::riscv::PartialSetType>(sourceSetType);
  auto reduced = mlir::dyn_cast<weft::riscv::PartialSetType>(reducedSetType);
  auto final = mlir::dyn_cast<weft::riscv::PartialSetType>(finalSetType);
  if ((realization != "level_scaled" &&
       realization != "independent_vector" &&
       realization != "independent_scalar") ||
      !source || !final || outputParts <= 0 || resourceGroups <= 0 ||
      multiplyInstruction.empty() || finalizeInstruction.empty() ||
      lhsParts.size() != rhsParts.size())
    return emitError()
           << "partial combine plan requires one complete typed realization";
  if (realization == "level_scaled") {
    if (!reduced || !combineSetTypes.empty() || !combineArities.empty() ||
        lhsParts.size() != static_cast<size_t>(outputParts) ||
        scaleParts.size() != static_cast<size_t>(outputParts) ||
        source.getReductionAxis() != reduced.getReductionAxis() ||
        source.getReductionAxis() != final.getReductionAxis())
      return emitError()
             << "level-scaled combine plan requires one source/reduced/final set and one part map per output";
  } else {
    if (reduced || !scaleParts.empty() ||
        combineSetTypes.size() != combineArities.size() ||
        lhsParts.size() !=
            static_cast<size_t>(outputParts * source.getSlots()))
      return emitError()
             << "independent combine plan requires one flattened source map and one typed set per combine stage";
    weft::riscv::PartialSetType previous = source;
    for (auto [typeAttr, arity] :
         llvm::zip(combineSetTypes, combineArities.asArrayRef())) {
      auto typed = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
      auto next = typed ? mlir::dyn_cast<weft::riscv::PartialSetType>(
                              typed.getValue())
                        : weft::riscv::PartialSetType();
      const bool preservesType =
          next && next.getPartialType() == previous.getPartialType();
      const bool widensType =
          next && isWideningPartialCombineStage(previous, next, arity);
      if (!next || arity <= 1 || previous.getSlots() % arity ||
          (!preservesType && !widensType) ||
          next.getReductionAxis() != previous.getReductionAxis() ||
          next.getSlots() != previous.getSlots() / arity ||
          next.getTermsPerSlot() != previous.getTermsPerSlot() * arity)
        return emitError()
               << "independent combine plan contains an inconsistent typed stage";
      previous = next;
    }
    if (previous != final || final.getSlots() != 1)
      return emitError()
             << "independent combine stages must close to the declared final set";
  }
  if (llvm::any_of(lhsParts.asArrayRef(), [](int64_t value) { return value < 0; }) ||
      llvm::any_of(rhsParts.asArrayRef(), [](int64_t value) { return value < 0; }) ||
      llvm::any_of(scaleParts.asArrayRef(),
                   [](int64_t value) { return value < 0; }))
    return emitError() << "partial combine part maps must be non-negative";
  return mlir::success();
}

mlir::LogicalResult PartialAddTreePlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::ArrayAttr leafSetTypes, mlir::ArrayAttr normalizedSetTypes,
    mlir::ArrayAttr leafMultiplyInstructions,
    mlir::DenseI64ArrayAttr splits, mlir::Type mergedSetType,
    llvm::StringRef finalizeInstruction, int64_t totalSlots,
    int64_t totalTerms, int64_t totalResourceGroups, mlir::ArrayAttr repackLeaves) {
  auto merged = mlir::dyn_cast<weft::riscv::PartialSetType>(mergedSetType);
  const size_t leaves = leafSetTypes.size();
  if (leaves < 2 || normalizedSetTypes.size() != leaves ||
      repackLeaves.size() != leaves ||
      leafMultiplyInstructions.size() != leaves || splits.size() != leaves ||
      !merged || merged.getSlots() != 1 || finalizeInstruction.empty() ||
      totalSlots <= 0 || totalTerms <= 0 || totalResourceGroups <= 0)
    return emitError()
           << "partial add-tree plan requires one complete typed plan for every leaf";
  int64_t computedSlots = 0;
  int64_t computedTerms = 0;
  int64_t computedResources = 0;
  for (size_t index = 0; index < leaves; ++index) {
    auto sourceAttr = mlir::dyn_cast<mlir::TypeAttr>(leafSetTypes[index]);
    auto normalizedAttr =
        mlir::dyn_cast<mlir::TypeAttr>(normalizedSetTypes[index]);
    auto instruction =
        mlir::dyn_cast<mlir::StringAttr>(leafMultiplyInstructions[index]);
    auto source = sourceAttr
                      ? mlir::dyn_cast<weft::riscv::PartialSetType>(
                            sourceAttr.getValue())
                      : weft::riscv::PartialSetType();
    auto normalized = normalizedAttr
                          ? mlir::dyn_cast<weft::riscv::PartialSetType>(
                                normalizedAttr.getValue())
                          : weft::riscv::PartialSetType();
    const int64_t split = splits[index];
    if (!source || !normalized || !instruction || split <= 0 || split > 2 ||
        source.getSlots() * split != normalized.getSlots() ||
        source.getTermsPerSlot() != normalized.getTermsPerSlot() * split ||
        normalized.getResourceGroups() != normalized.getSlots() *
            normalized.getPartialType().getLayout().getRegisterGroups() ||
        normalized.getPartialType() != merged.getPartialType() ||
        normalized.getReductionAxis() != merged.getReductionAxis() ||
        !matchesPartialRepackLeaf(repackLeaves[index], source, normalized, split))
      return emitError()
             << "partial add-tree leaf disagrees with its normalized typed representation";
    computedSlots += normalized.getSlots();
    computedTerms += normalized.getSlots() * normalized.getTermsPerSlot();
    computedResources += normalized.getResourceGroups();
  }
  if (computedSlots != totalSlots || computedTerms != totalTerms ||
      computedResources != totalResourceGroups ||
      merged.getTermsPerSlot() != totalTerms)
    return emitError()
           << "partial add-tree resource and term totals do not close";
  return mlir::success();
}

mlir::LogicalResult NestedPartialPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t windowAxis, int64_t issueStreams, int64_t windowExtent,
    mlir::DenseI64ArrayAttr partialAxes,
    mlir::DenseI64ArrayAttr partialAxisExtents, int64_t partialSlots,
    int64_t issueUnroll, mlir::DenseI64ArrayAttr outputAxes,
    int64_t outputReplicas, mlir::DenseI64ArrayAttr issueMaterializationOrder,
    mlir::DenseI64ArrayAttr lhsSourceParts,
    mlir::DenseI64ArrayAttr rhsSourceParts,
    mlir::DenseI64ArrayAttr lhsLaneOffsets,
    mlir::DenseI64ArrayAttr rhsLaneOffsets,
    mlir::DenseI64ArrayAttr scaleReplicas, mlir::Type issueLhsType,
    mlir::Type issueRhsType,
    mlir::Type sourceSlotType, mlir::Type splitSlotType,
    mlir::Type reducedSlotType, mlir::Type scaleReplicaType,
    mlir::Type sourceSetType, mlir::Type repackedSetType,
    mlir::Type reducedSetType, mlir::Type scaleCombinedSetType,
    mlir::ArrayAttr combineSetTypes, mlir::DenseI64ArrayAttr combineAxes,
    mlir::DenseI64ArrayAttr combineArities,
    llvm::StringRef scaleSupply, llvm::StringRef scaleSupplyStage,
    llvm::StringRef multiplyInstruction,
    llvm::StringRef reduceInstruction,
    llvm::StringRef finalizeInstruction, int64_t resourceGroups,
    mlir::Attribute repackLeaf) {
  auto issueLhs = mlir::dyn_cast<weft::riscv::ValueType>(issueLhsType);
  auto issueRhs = mlir::dyn_cast<weft::riscv::ValueType>(issueRhsType);
  auto sourceSlot = mlir::dyn_cast<weft::riscv::ValueType>(sourceSlotType);
  auto splitSlot = mlir::dyn_cast<weft::riscv::ValueType>(splitSlotType);
  auto reducedSlot = mlir::dyn_cast<weft::riscv::ValueType>(reducedSlotType);
  auto scaleReplica = mlir::dyn_cast<weft::riscv::ValueType>(scaleReplicaType);
  auto scaleElement =
      scaleReplica
          ? mlir::dyn_cast<mlir::IntegerType>(scaleReplica.getElementType())
          : mlir::IntegerType();
  auto sourceSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(sourceSetType);
  auto repackedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(repackedSetType);
  auto reducedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(reducedSetType);
  auto scaleCombinedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(scaleCombinedSetType);
  llvm::SmallVector<weft::riscv::PartialSetType> combineSets;
  bool combineSetTypesLegal = true;
  for (mlir::Attribute attribute : combineSetTypes) {
    auto typeAttribute = mlir::dyn_cast<mlir::TypeAttr>(attribute);
    auto set = typeAttribute
                   ? mlir::dyn_cast<weft::riscv::PartialSetType>(
                         typeAttribute.getValue())
                   : weft::riscv::PartialSetType();
    combineSetTypesLegal &= static_cast<bool>(set);
    if (set)
      combineSets.push_back(set);
  }
  const bool exactMultiply =
      multiplyInstruction == "rvv.vwmul.vv" ||
      multiplyInstruction == "rvv.vwmul.vv.reinterpret-rhs" ||
      multiplyInstruction == "rvv.vwmul.vv.reinterpret-lhs" ||
      multiplyInstruction == "rvv.vwmulsu.vv" ||
      multiplyInstruction == "rvv.vwmulsu.vv.swap";
  llvm::DenseSet<int64_t> uniquePartialAxes;
  int64_t computedPartialSlots = 1;
  bool partialAxesLegal = !partialAxes.empty() &&
                          partialAxes.size() == partialAxisExtents.size();
  for (auto [axis, extent] :
       llvm::zip(partialAxes.asArrayRef(), partialAxisExtents.asArrayRef())) {
    partialAxesLegal &=
        axis > 0 && extent > 0 && uniquePartialAxes.insert(axis).second &&
        computedPartialSlots <= std::numeric_limits<int64_t>::max() / extent;
    if (partialAxesLegal)
      computedPartialSlots *= extent;
  }
  auto windowPosition = llvm::find(partialAxes.asArrayRef(), windowAxis);
  const bool windowExtentLegal =
      windowPosition != partialAxes.asArrayRef().end() &&
      partialAxisExtents[static_cast<size_t>(
          windowPosition - partialAxes.asArrayRef().begin())] == windowExtent;
  llvm::SmallVector<int64_t> sortedIssueMaterializationOrder(
      issueMaterializationOrder.asArrayRef().begin(),
      issueMaterializationOrder.asArrayRef().end());
  llvm::sort(sortedIssueMaterializationOrder);
  const bool issueMaterializationOrderLegal =
      sortedIssueMaterializationOrder == llvm::SmallVector<int64_t>({0, 1, 2});
  if (windowAxis <= 0 || issueStreams <= 0 || windowExtent <= 0 ||
      !partialAxesLegal || !windowExtentLegal ||
      computedPartialSlots != partialSlots ||
      !issueMaterializationOrderLegal ||
      !combineSetTypesLegal ||
      combineSets.size() != combineAxes.size() ||
      combineSets.size() != combineArities.size() ||
      combineSets.size() + 1 != partialAxes.size() ||
      issueUnroll <= 0 || issueUnroll > issueStreams || resourceGroups <= 0 ||
      outputReplicas <= 0 ||
      !issueLhs || !issueRhs || !sourceSlot || !splitSlot || !reducedSlot ||
      !scaleReplica || !sourceSet || !repackedSet || !reducedSet ||
      !scaleCombinedSet || !exactMultiply ||
      reduceInstruction != "rvv.partial-reduce.widen" ||
      (finalizeInstruction != "rvv.partial-finalize.extract" &&
       finalizeInstruction != "rvv.partial-finalize.reduce"))
    return emitError()
           << "nested partial plan requires one complete issue/product/reduction topology";
  if (scaleSupply != "vector-convert" &&
      scaleSupply != "scalar-rematerialize")
    return emitError()
           << "nested partial plan requires one selected scale supply form";
  if ((scaleSupplyStage != "before-product" &&
       scaleSupplyStage != "after-partial-reduce" &&
       scaleSupplyStage != "shared-before-issue") ||
      (scaleSupplyStage != "before-product" && outputReplicas != 1) ||
      (scaleSupplyStage == "shared-before-issue" &&
       (scaleSupply != "vector-convert" || issueStreams <= 1)))
    return emitError()
           << "nested partial plan requires one scale stage consistent with output multiplicity";
  if (issueLhs.getLayout().getCarrier() != "rvv" ||
      issueRhs.getLayout().getCarrier() != "rvv" ||
      sourceSlot.getLayout().getCarrier() != "rvv" ||
      splitSlot.getLayout().getCarrier() != "rvv" ||
      reducedSlot.getLayout().getCarrier() != "rvv" ||
      scaleReplica.getLayout().getCarrier() != "scalar" || !scaleElement ||
      !scaleElement.isSigned() || scaleElement.getWidth() != 32)
    return emitError()
           << "nested partial plan requires RVV issue/product values and scalar scale replicas";
  auto lhsLanes = checkedPositiveProduct(
      issueLhs.getLayout().getLaneFactors().asArrayRef());
  auto rhsLanes = checkedPositiveProduct(
      issueRhs.getLayout().getLaneFactors().asArrayRef());
  auto lhsTime = checkedPositiveProduct(
      issueLhs.getLayout().getTimeFactors().asArrayRef());
  auto rhsTime = checkedPositiveProduct(
      issueRhs.getLayout().getTimeFactors().asArrayRef());
  auto lhsReplicas = checkedPositiveProduct(
      issueLhs.getLayout().getReplicaFactors().asArrayRef());
  auto rhsReplicas = checkedPositiveProduct(
      issueRhs.getLayout().getReplicaFactors().asArrayRef());
  auto scaleReplicaParts = checkedPositiveProduct(
      scaleReplica.getLayout().getReplicaFactors().asArrayRef());
  auto lhsParts = physicalPartCount(issueLhs);
  auto rhsParts = physicalPartCount(issueRhs);
  const bool scaleMapSizeLegal =
      outputReplicas <= std::numeric_limits<int64_t>::max() / partialSlots &&
      scaleReplicas.size() ==
          static_cast<size_t>(outputReplicas * partialSlots);
  auto validParts = [](mlir::DenseI64ArrayAttr values, int64_t count,
                       std::optional<int64_t> available) {
    return available && values.size() == static_cast<size_t>(count) &&
           llvm::all_of(values.asArrayRef(), [&](int64_t value) {
             return value >= 0 && value < *available;
           });
  };
  llvm::DenseSet<int64_t> uniqueOutputAxes;
  const bool outputAxesLegal =
      llvm::all_of(outputAxes.asArrayRef(), [&](int64_t axis) {
        return axis > 0 &&
               !llvm::is_contained(partialAxes.asArrayRef(), axis) &&
               uniqueOutputAxes.insert(axis).second;
      });
  if (!lhsLanes || !rhsLanes || !lhsTime || !rhsTime || !lhsReplicas ||
      !rhsReplicas || !scaleReplicaParts || *lhsLanes != *rhsLanes ||
      *lhsLanes != sourceSet.getTermsPerSlot() || *lhsTime != 1 ||
      *rhsTime != 1 || *lhsReplicas <= 0 || *rhsReplicas <= 0 ||
      !outputAxesLegal || (outputReplicas > 1 && outputAxes.empty()) ||
      !validParts(lhsSourceParts, outputReplicas, lhsParts) ||
      !validParts(rhsSourceParts, outputReplicas, rhsParts) ||
      lhsLaneOffsets.size() != static_cast<size_t>(outputReplicas) ||
      rhsLaneOffsets.size() != static_cast<size_t>(outputReplicas) ||
      !llvm::all_of(lhsLaneOffsets.asArrayRef(),
                    [](int64_t offset) { return offset == 0; }) ||
      !llvm::all_of(rhsLaneOffsets.asArrayRef(),
                    [](int64_t offset) { return offset == 0; }) ||
      !scaleMapSizeLegal ||
      !validParts(scaleReplicas, outputReplicas * partialSlots,
                  scaleReplicaParts) ||
      !llvm::all_of(scaleReplica.getLayout().getTimeFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) ||
      !llvm::all_of(scaleReplica.getLayout().getLaneFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }))
    return emitError()
           << "nested partial issue values must be one complete lane product with closed output-part and scale-replica maps";
  if (sourceSlot.getAxisIds().size() != 1 ||
      splitSlot.getAxisIds() != sourceSlot.getAxisIds() ||
      reducedSlot.getAxisIds() != sourceSlot.getAxisIds() ||
      sourceSlot.getShape().size() != 1 || splitSlot.getShape().size() != 1 ||
      reducedSlot.getShape().size() != 1 ||
      sourceSlot.getShape()[0] != splitSlot.getShape()[0] * partialSlots ||
      reducedSlot.getShape()[0] != 1 ||
      sourceSlot.getLayout().getLmulEighths() !=
          splitSlot.getLayout().getLmulEighths() * partialSlots ||
      sourceSlot.getLayout().getVl() !=
          splitSlot.getLayout().getVl() * partialSlots)
    return emitError()
           << "nested partial product carrier must split exactly into typed reduction windows";
  const int64_t reductionAxis = sourceSlot.getAxisIds()[0];
  auto combinedPartial = scaleCombinedSet.getPartialType();
  const int64_t scaleCombineArity = partialAxisExtents.asArrayRef().back();
  const int64_t expectedScaleSlots = partialSlots / scaleCombineArity;
  const int64_t expectedScaleTerms =
      reducedSet.getTermsPerSlot() * scaleCombineArity;
  int64_t expectedSlots = expectedScaleSlots;
  int64_t expectedTerms = expectedScaleTerms;
  bool combineStagesLegal = true;
  weft::riscv::PartialSetType previousSet = scaleCombinedSet;
  int64_t combinePeak = scaleCombinedSet.getResourceGroups();
  for (size_t stage = 0; stage < combineSets.size(); ++stage) {
    const size_t axisPosition = partialAxes.size() - 2 - stage;
    const int64_t expectedAxis = partialAxes[axisPosition];
    const int64_t expectedArity = partialAxisExtents[axisPosition];
    auto nextSet = combineSets[stage];
    combineStagesLegal &=
        combineAxes[stage] == expectedAxis &&
        combineArities[stage] == expectedArity && expectedArity > 0 &&
        expectedSlots % expectedArity == 0;
    if (!combineStagesLegal)
      break;
    expectedSlots /= expectedArity;
    expectedTerms *= expectedArity;
    combineStagesLegal &=
        nextSet.getPartialType() == combinedPartial &&
        nextSet.getReductionAxis() == reductionAxis &&
        nextSet.getSlots() == expectedSlots &&
        nextSet.getTermsPerSlot() == expectedTerms &&
        nextSet.getResourceGroups() ==
            expectedSlots * combinedPartial.getLayout().getRegisterGroups();
    combinePeak =
        std::max(combinePeak, previousSet.getResourceGroups() +
                                  nextSet.getResourceGroups());
    previousSet = nextSet;
  }
  auto finalSet = combineSets.empty() ? scaleCombinedSet : combineSets.back();
  if (sourceSet.getPartialType() != sourceSlot || sourceSet.getSlots() != 1 ||
      sourceSet.getReductionAxis() != reductionAxis ||
      repackedSet.getPartialType() != splitSlot ||
      repackedSet.getReductionAxis() != reductionAxis ||
      repackedSet.getSlots() != partialSlots ||
      repackedSet.getTermsPerSlot() * partialSlots !=
          sourceSet.getTermsPerSlot() ||
      repackedSet.getResourceGroups() != partialSlots *
          splitSlot.getLayout().getRegisterGroups() ||
      !matchesPartialRepackLeaf(repackLeaf, sourceSet, repackedSet, partialSlots) ||
      reducedSet.getPartialType() != reducedSlot ||
      reducedSet.getReductionAxis() != reductionAxis ||
      reducedSet.getSlots() != partialSlots ||
      reducedSet.getTermsPerSlot() != repackedSet.getTermsPerSlot() ||
      combinedPartial != reducedSlot ||
      scaleCombinedSet.getReductionAxis() != reductionAxis ||
      scaleCombinedSet.getSlots() != expectedScaleSlots ||
      scaleCombinedSet.getTermsPerSlot() != expectedScaleTerms ||
      scaleCombinedSet.getResourceGroups() !=
          expectedScaleSlots * combinedPartial.getLayout().getRegisterGroups() ||
      !combineStagesLegal || expectedSlots != 1 ||
      finalSet.getTermsPerSlot() != sourceSet.getTermsPerSlot() ||
      (finalSet.getPartialType().getLayout().getLaneFactors()[0] == 1) !=
          (finalizeInstruction == "rvv.partial-finalize.extract") ||
      resourceGroups < issueLhs.getLayout().getRegisterGroups() +
                           issueRhs.getLayout().getRegisterGroups() +
                           sourceSet.getResourceGroups() ||
      resourceGroups <= combinePeak)
    return emitError()
           << "nested partial plan set types disagree with the selected full-product carrier";
  return mlir::success();
}

mlir::LogicalResult SequentialPartialPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef realization, int64_t issueCount,
    mlir::DenseI64ArrayAttr reductionAxes,
    mlir::Type lhsIssueType, mlir::Type rhsIssueType,
    mlir::Type accumulatorType, llvm::StringRef lhsIssueSupply,
    llvm::StringRef rhsIssueSupply, mlir::DenseI64ArrayAttr lhsSourceParts,
    mlir::DenseI64ArrayAttr rhsSourceParts,
    mlir::DenseI64ArrayAttr lhsLaneOffsets,
    mlir::DenseI64ArrayAttr rhsLaneOffsets,
    llvm::StringRef multiplyInstruction,
    llvm::StringRef accumulateInstruction,
    llvm::StringRef finalizeInstruction, int64_t resourceGroups) {
  auto lhs = mlir::dyn_cast<weft::riscv::ValueType>(lhsIssueType);
  auto rhs = mlir::dyn_cast<weft::riscv::ValueType>(rhsIssueType);
  auto accumulator =
      mlir::dyn_cast<weft::riscv::ValueType>(accumulatorType);
  auto lhsElement =
      lhs ? mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType())
          : mlir::IntegerType();
  auto rhsElement =
      rhs ? mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType())
          : mlir::IntegerType();
  llvm::StringRef expectedMultiply;
  if (lhsElement && rhsElement) {
    if (lhsElement.isSigned() && rhsElement.isSigned())
      expectedMultiply = "rvv.vwmul.vv";
    else if (lhsElement.isSigned())
      expectedMultiply = "rvv.vwmulsu.vv";
    else if (rhsElement.isSigned())
      expectedMultiply = "rvv.vwmulsu.vv.swap";
  }
  auto accumulatorElement =
      accumulator
          ? mlir::dyn_cast<mlir::IntegerType>(accumulator.getElementType())
          : mlir::IntegerType();
  llvm::DenseSet<int64_t> seenReductionAxes;
  bool validReductionAxes = !reductionAxes.empty();
  for (int64_t axis : reductionAxes.asArrayRef())
    validReductionAxes &= axis > 0 && seenReductionAxes.insert(axis).second;
  const bool fusedPartial = realization == "fused_partial";
  if ((realization != "fused" && realization != "per_stream" &&
       !fusedPartial) ||
      issueCount <= 0 || !validReductionAxes || !lhs || !rhs || !accumulator ||
      !lhsElement || !rhsElement || !accumulatorElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      accumulatorElement.isSignless() ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      lhsElement.getWidth() > 16 || rhsElement.getWidth() > 16 ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      accumulatorElement.getWidth() !=
          2 * std::max<unsigned>(8, lhsElement.getWidth()) ||
      (lhsIssueSupply != "slice" &&
       lhsIssueSupply != "storage-rematerialize") ||
      (rhsIssueSupply != "slice" &&
       rhsIssueSupply != "storage-rematerialize") ||
      lhsSourceParts.empty() || lhsSourceParts.size() != rhsSourceParts.size() ||
      lhsSourceParts.size() != lhsLaneOffsets.size() ||
      rhsSourceParts.size() != rhsLaneOffsets.size() ||
      lhsSourceParts.size() % static_cast<size_t>(issueCount) ||
      multiplyInstruction != expectedMultiply ||
      accumulateInstruction != "rvv.vwmacc.partial" ||
      (fusedPartial ? finalizeInstruction != "none"
                    : (finalizeInstruction != "rvv.vwredsum.partial" &&
                       finalizeInstruction != "rvv.vredsum.partial")) ||
      resourceGroups <= 0)
    return emitError()
           << "sequential partial plan requires one closed issue-slice, widened product/accumulator, and final-reduction program";
  auto lhsPhysicalParts = physicalPartCount(lhs);
  auto rhsPhysicalParts = physicalPartCount(rhs);
  const int64_t expectedResources = std::max<int64_t>(
      lhs.getLayout().getRegisterGroups() +
          rhs.getLayout().getRegisterGroups() +
          accumulator.getLayout().getRegisterGroups(),
      accumulator.getLayout().getRegisterGroups() + 2);
  auto closedIssueCarrier = [&](weft::riscv::ValueType value) {
    if (value.getAxisIds() != reductionAxes ||
        value.getShape().size() != reductionAxes.size())
      return false;
    for (size_t position = 0; position < value.getShape().size(); ++position)
      if (value.getShape()[position] <= 0 ||
          value.getLayout().getTimeFactors()[position] != 1 ||
          value.getLayout().getLaneFactors()[position] != value.getShape()[position] ||
          value.getLayout().getReplicaFactors()[position] != 1 ||
          value.getLayout().getFragmentFactors()[position] != 1 ||
          value.getLayout().getLocalFactors()[position] != 1)
        return false;
    return true;
  };
  if (lhs.getLayout().getCarrier() != "rvv" ||
      rhs.getLayout().getCarrier() != "rvv" ||
      accumulator.getLayout().getCarrier() != "rvv" ||
      !closedIssueCarrier(lhs) || !closedIssueCarrier(rhs) ||
      !closedIssueCarrier(accumulator) || lhs.getShape() != rhs.getShape() ||
      lhs.getShape() != accumulator.getShape() ||
      accumulator.getLayout().getSew() !=
          2 * lhs.getLayout().getSew() ||
      accumulator.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
      lhs.getLayout().getVl() != accumulator.getLayout().getVl() ||
      !lhsPhysicalParts || !rhsPhysicalParts ||
      *lhsPhysicalParts != 1 || *rhsPhysicalParts != 1 ||
      llvm::any_of(lhsLaneOffsets.asArrayRef(),
                   [](int64_t offset) { return offset < 0; }) ||
      llvm::any_of(rhsLaneOffsets.asArrayRef(),
                   [](int64_t offset) { return offset < 0; }) ||
      resourceGroups != expectedResources)
    return emitError()
           << "sequential partial plan carrier or resource contract is inconsistent";
  return mlir::success();
}

mlir::LogicalResult ScaledPartialPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef realization, mlir::Type scalarScaleType,
    mlir::Type narrowScaleType, llvm::StringRef scaleSupply,
    llvm::StringRef multiplyInstruction, llvm::StringRef reduceInstruction,
    llvm::StringRef finalizeInstruction, int64_t resourceGroups) {
  auto scalarScale =
      mlir::dyn_cast<weft::riscv::ValueType>(scalarScaleType);
  auto narrowScale =
      mlir::dyn_cast<weft::riscv::ValueType>(narrowScaleType);
  auto scalarElement =
      scalarScale
          ? mlir::dyn_cast<mlir::IntegerType>(scalarScale.getElementType())
          : mlir::IntegerType();
  auto narrowElement =
      narrowScale
          ? mlir::dyn_cast<mlir::IntegerType>(narrowScale.getElementType())
          : mlir::IntegerType();
  const bool exactMultiply =
      multiplyInstruction == "rvv.vwmul.vv" ||
      multiplyInstruction == "rvv.vwmul.vv.reinterpret-rhs" ||
      multiplyInstruction == "rvv.vwmul.vv.reinterpret-lhs" ||
      multiplyInstruction == "rvv.vwmulsu.vv" ||
      multiplyInstruction == "rvv.vwmulsu.vv.swap";
  if ((realization != "scaled" && realization != "reduced_scaled") ||
      !scalarScale || !scalarElement || !scalarElement.isSigned() ||
      scalarElement.getWidth() != 32 ||
      scalarScale.getLayout().getCarrier() != "scalar" ||
      (scaleSupply != "scalar-rematerialize" &&
       scaleSupply != "vector-convert") ||
      !exactMultiply ||
      (reduceInstruction != "none" &&
       reduceInstruction != "rvv.partial-reduce.widen") ||
      (finalizeInstruction != "rvv.partial-finalize.extract" &&
       finalizeInstruction != "rvv.partial-finalize.reduce") ||
      resourceGroups <= 0)
    return emitError()
           << "scaled partial plan requires one scalar scale supply and closed multiply/reduce/finalize program";
  if (narrowScale &&
      (!narrowElement || !narrowElement.isSigned() ||
       narrowElement.getWidth() > 16 ||
       narrowScale.getLayout().getCarrier() != "scalar" ||
       narrowScale.getShape() != scalarScale.getShape() ||
       narrowScale.getAxisIds() != scalarScale.getAxisIds() ||
       narrowScale.getLayout().getReplicaFactors() !=
           scalarScale.getLayout().getReplicaFactors()))
    return emitError()
           << "scaled partial narrow scale disagrees with the selected scalar replica carrier";
  if (realization == "reduced_scaled" &&
      (narrowScale || reduceInstruction != "rvv.partial-reduce.widen"))
    return emitError()
           << "reduced-scaled partial plan must reduce widened products before scalar scaling";
  return mlir::success();
}

mlir::LogicalResult LayeredStreamGeometryAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int64_t axis,
    int64_t groupSize, int64_t layerSize, int64_t laneCount,
    int64_t streamCount, int64_t replicaCount,
    mlir::DenseI64ArrayAttr windowForStream,
    mlir::DenseI64ArrayAttr representativeStreamForWindow,
    mlir::DenseI64ArrayAttr physicalLayerForStream,
    mlir::DenseI64ArrayAttr physicalLayerForLogicalLayer,
    mlir::DenseI64ArrayAttr shiftAmountForLogicalLayer,
    mlir::DenseI64ArrayAttr maskValueForLogicalLayer,
    mlir::DenseI64ArrayAttr shiftAmountForStream,
    mlir::DenseI64ArrayAttr maskValueForStream,
    mlir::DenseI64ArrayAttr groupForWindow,
    mlir::DenseI64ArrayAttr withinForWindow) {
  if (axis <= 0 || groupSize <= 0 || layerSize <= 0 ||
      groupSize % layerSize || laneCount <= 0 || streamCount <= 0 ||
      replicaCount <= 0 ||
      windowForStream.size() != static_cast<size_t>(streamCount) ||
      representativeStreamForWindow.size() != groupForWindow.size() ||
      physicalLayerForStream.size() != static_cast<size_t>(streamCount) ||
      shiftAmountForStream.size() != static_cast<size_t>(streamCount) ||
      maskValueForStream.size() != static_cast<size_t>(streamCount) ||
      groupForWindow.empty() || groupForWindow.size() != withinForWindow.size())
    return emitError() << "layered stream geometry is incomplete";
  const int64_t layers = groupSize / layerSize;
  if (physicalLayerForLogicalLayer.size() != static_cast<size_t>(layers) ||
      shiftAmountForLogicalLayer.size() != static_cast<size_t>(layers) ||
      maskValueForLogicalLayer.size() != static_cast<size_t>(layers))
    return emitError()
           << "layered stream geometry requires one complete decode plan per logical layer";
  llvm::SmallVector<int64_t> logicalLayerMap(
      physicalLayerForLogicalLayer.asArrayRef());
  llvm::sort(logicalLayerMap);
  for (auto [index, layer] : llvm::enumerate(logicalLayerMap))
    if (layer != static_cast<int64_t>(index))
      return emitError()
             << "layered stream logical-to-physical layer map must be a permutation";
  for (auto [layer, shiftAmount, maskValue] :
       llvm::zip(physicalLayerForLogicalLayer.asArrayRef(),
                 shiftAmountForLogicalLayer.asArrayRef(),
                 maskValueForLogicalLayer.asArrayRef()))
    if (shiftAmount < 0 || maskValue < 0 ||
        (layer == 0) != (shiftAmount == 0))
      return emitError()
             << "layered stream logical decode plan disagrees with its physical layer";
  for (int64_t window : windowForStream.asArrayRef())
    if (window < 0 || window >= static_cast<int64_t>(groupForWindow.size()))
      return emitError() << "layered stream references an absent raw window";
  for (auto [window, representative] :
       llvm::enumerate(representativeStreamForWindow.asArrayRef()))
    if (representative < 0 || representative >= streamCount ||
        windowForStream[static_cast<size_t>(representative)] !=
            static_cast<int64_t>(window))
      return emitError()
             << "layered stream raw window lacks an exact representative stream";
  for (int64_t layer : physicalLayerForStream.asArrayRef())
    if (layer < 0 || layer >= layers)
      return emitError() << "layered stream physical layer is out of range";
  for (auto [layer, shiftAmount, maskValue] :
       llvm::zip(physicalLayerForStream.asArrayRef(),
                 shiftAmountForStream.asArrayRef(),
                 maskValueForStream.asArrayRef())) {
    auto logical = llvm::find(physicalLayerForLogicalLayer.asArrayRef(), layer);
    if (logical == physicalLayerForLogicalLayer.asArrayRef().end())
      return emitError()
             << "layered stream physical layer has no logical decode owner";
    const size_t logicalIndex = static_cast<size_t>(
        logical - physicalLayerForLogicalLayer.asArrayRef().begin());
    if (shiftAmount != shiftAmountForLogicalLayer[logicalIndex] ||
        maskValue != maskValueForLogicalLayer[logicalIndex])
      return emitError()
             << "layered stream decode plan disagrees with its logical layer";
  }
  for (int64_t within : withinForWindow.asArrayRef())
    if (within < 0 || within + laneCount > layerSize)
      return emitError() << "layered raw window crosses a storage layer";
  return mlir::success();
}

mlir::LogicalResult StorageWindowPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t reductionAxis, int64_t projectionBase,
    int64_t projectionStride, int64_t projectionRepeat,
    int64_t projectionExtent, int64_t offsetAlignment,
    int64_t recordElements, int64_t byteOffset, int64_t elementBits,
    int64_t groupSize, int64_t layerSize, int64_t physicalLayerBase,
    int64_t physicalLayerStep, int64_t shiftBase, int64_t shiftStep,
    int64_t maskValue) {
  if (reductionAxis <= 0 || projectionBase < 0 || projectionStride <= 0 ||
      projectionRepeat <= 0 || projectionExtent <= 0 || offsetAlignment <= 0 ||
      recordElements <= 0 || byteOffset < 0 || elementBits <= 0)
    return emitError() << "storage window plan has incomplete logical or storage geometry";
  if (kind == "unit" || kind == "strided" || kind == "repeat") {
    const bool exactKind =
        (kind == "unit" && projectionRepeat == 1 && projectionStride == 1) ||
        (kind == "strided" && projectionRepeat == 1 && projectionStride > 1) ||
        (kind == "repeat" && projectionRepeat > 1);
    if (!exactKind || groupSize != 0 || layerSize != 0 ||
        physicalLayerBase != 0 || physicalLayerStep != 0 || shiftBase != 0 ||
        shiftStep != 0 || maskValue != 0)
      return emitError()
             << "natural storage window plan disagrees with its selected load form";
    return mlir::success();
  }
  if (kind == "interleaved_natural" || kind == "interleaved_joined") {
    if (projectionBase != 0 || projectionStride != 1 ||
        projectionRepeat != 1 || groupSize != 0 || layerSize != 0 ||
        physicalLayerBase != 0 || physicalLayerStep != 0 || shiftBase != 0 ||
        shiftStep != 0 || maskValue != 0)
      return emitError()
             << "interleaved record projection disagrees with its selected load form";
    return mlir::success();
  }
  if (kind != "layered")
    return emitError() << "unknown storage window implementation kind";
  const int64_t layers =
      layerSize > 0 && groupSize > 0 ? groupSize / layerSize : 0;
  const bool exactLayerMap =
      layers > 1 &&
      ((physicalLayerBase == 0 && physicalLayerStep == 1) ||
       (physicalLayerBase == layers - 1 && physicalLayerStep == -1));
  if (projectionBase != 0 || projectionStride != 1 ||
      projectionRepeat != 1 || groupSize <= 0 || layerSize <= 0 ||
      groupSize % layerSize || elementBits >= 8 || !exactLayerMap ||
      shiftBase != physicalLayerBase * elementBits ||
      shiftStep != physicalLayerStep * elementBits ||
      maskValue != ((int64_t{1} << elementBits) - 1))
    return emitError()
           << "layered storage window plan has incomplete address or decode geometry";
  return mlir::success();
}

mlir::LogicalResult LocalPackPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t rowAxis, int64_t recordAxis,
    int64_t interleaveRows, int64_t recordBytes, int64_t recordElements,
    int64_t localBytes) {
  auto interleavedRecordBytes =
      checkedPositiveProduct({interleaveRows, recordBytes});
  if (kind != "interleave" || rowAxis <= 0 || recordAxis <= 0 ||
      rowAxis == recordAxis || interleaveRows <= 1 || recordBytes <= 0 ||
      recordElements <= 0 || localBytes <= 0 ||
      !interleavedRecordBytes || localBytes % *interleavedRecordBytes != 0)
    return emitError()
           << "local pack plan requires one closed record-interleave geometry";
  return mlir::success();
}

mlir::LogicalResult LayeredPartialPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t rootIndex, mlir::Type decodedWindowType,
    mlir::Type storageWindowType, mlir::Type lhsIssueType,
    mlir::Type rhsIssueType, mlir::Type accumulatorType,
    mlir::ArrayAttr rootWindowTypes, mlir::ArrayAttr rootStoragePlans,
    mlir::ArrayAttr rootWindowInstructions,
    mlir::ArrayAttr decodeInstructions,
    llvm::StringRef finalizeInstruction,
    LayeredStreamGeometryAttr geometry, int64_t resourceGroups) {
  auto decoded =
      mlir::dyn_cast<weft::riscv::ValueType>(decodedWindowType);
  auto storage =
      mlir::dyn_cast<weft::riscv::LayeredWindowType>(storageWindowType);
  auto lhsIssue = mlir::dyn_cast<weft::riscv::ValueType>(lhsIssueType);
  auto rhsIssue = mlir::dyn_cast<weft::riscv::ValueType>(rhsIssueType);
  auto accumulator =
      mlir::dyn_cast<weft::riscv::ValueType>(accumulatorType);
  if (rootIndex < 0 || rootIndex >= static_cast<int64_t>(rootWindowTypes.size()) ||
      !decoded || !storage || !lhsIssue || !rhsIssue || !accumulator ||
      !geometry || resourceGroups <= 0 ||
      rootWindowTypes.empty() || rootStoragePlans.size() != rootWindowTypes.size() ||
      rootWindowInstructions.size() != rootWindowTypes.size() ||
      decodeInstructions.size() !=
          geometry.getShiftAmountForLogicalLayer().size() ||
      decoded.getLayout().getCarrier() != "rvv" ||
      accumulator.getLayout().getCarrier() != "rvv" ||
      storage.getResultType() != decoded ||
      storage.getReductionAxis() != geometry.getAxis() ||
      storage.getLayers() != geometry.getGroupSize() / geometry.getLayerSize() ||
      storage.getWindowsPerLayer() * geometry.getLaneCount() !=
          geometry.getLayerSize() ||
      (finalizeInstruction != "rvv.vwredsum.partial" &&
       finalizeInstruction != "rvv.vredsum.partial"))
    return emitError()
           << "layered partial plan requires one closed typed storage, decode, accumulator, and root plan";
  auto lhsElement =
      mlir::dyn_cast<mlir::IntegerType>(lhsIssue.getElementType());
  auto rhsElement =
      mlir::dyn_cast<mlir::IntegerType>(rhsIssue.getElementType());
  auto accumulatorElement =
      mlir::dyn_cast<mlir::IntegerType>(accumulator.getElementType());
  auto lhsReduction = llvm::find(lhsIssue.getAxisIds().asArrayRef(),
                                 geometry.getAxis());
  auto rhsReduction = llvm::find(rhsIssue.getAxisIds().asArrayRef(),
                                 geometry.getAxis());
  if (!lhsElement || !rhsElement || !accumulatorElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      accumulatorElement.isSignless() ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      accumulatorElement.getWidth() !=
          2 * std::max<unsigned>(8, lhsElement.getWidth()) ||
      lhsIssue.getLayout().getCarrier() != "rvv" ||
      rhsIssue.getLayout().getCarrier() != "rvv" ||
      lhsReduction == lhsIssue.getAxisIds().asArrayRef().end() ||
      rhsReduction == rhsIssue.getAxisIds().asArrayRef().end())
    return emitError()
           << "layered partial issue and accumulator element contracts do not close";
  const size_t lhsReductionPosition = static_cast<size_t>(
      lhsReduction - lhsIssue.getAxisIds().asArrayRef().begin());
  const size_t rhsReductionPosition = static_cast<size_t>(
      rhsReduction - rhsIssue.getAxisIds().asArrayRef().begin());
  if (lhsIssue.getShape()[lhsReductionPosition] != geometry.getLaneCount() ||
      rhsIssue.getShape()[rhsReductionPosition] != geometry.getLaneCount() ||
      lhsIssue.getLayout().getTimeFactors()[lhsReductionPosition] != 1 ||
      rhsIssue.getLayout().getTimeFactors()[rhsReductionPosition] != 1 ||
      lhsIssue.getLayout().getLaneFactors()[lhsReductionPosition] !=
          geometry.getLaneCount() ||
      rhsIssue.getLayout().getLaneFactors()[rhsReductionPosition] !=
          geometry.getLaneCount() ||
      lhsIssue.getLayout().getReplicaFactors()[lhsReductionPosition] != 1 ||
      rhsIssue.getLayout().getReplicaFactors()[rhsReductionPosition] != 1 ||
      lhsIssue.getLayout().getSew() != rhsIssue.getLayout().getSew() ||
      lhsIssue.getLayout().getLmulEighths() !=
          rhsIssue.getLayout().getLmulEighths() ||
      lhsIssue.getLayout().getVl() != rhsIssue.getLayout().getVl() ||
      accumulator.getLayout().getSew() !=
          2 * lhsIssue.getLayout().getSew() ||
      accumulator.getLayout().getLmulEighths() !=
          2 * lhsIssue.getLayout().getLmulEighths() ||
      accumulator.getLayout().getVl() != lhsIssue.getLayout().getVl())
    return emitError()
           << "layered partial issue windows disagree with the selected RVV carrier";

  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFreeAxes = [&](weft::riscv::ValueType issue,
                            size_t reductionPosition) -> bool {
    for (size_t position = 0; position < issue.getAxisIds().size(); ++position) {
      if (position == reductionPosition)
        continue;
      if (issue.getLayout().getTimeFactors()[position] != 1 ||
          issue.getLayout().getLaneFactors()[position] != 1 ||
          issue.getLayout().getReplicaFactors()[position] !=
              issue.getShape()[position] ||
          issue.getLayout().getFragmentFactors()[position] != 1 ||
          issue.getLayout().getLocalFactors()[position] != 1)
        return false;
      auto found = llvm::find(expectedAxes, issue.getAxisIds()[position]);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(issue.getAxisIds()[position]);
        expectedShape.push_back(issue.getShape()[position]);
        continue;
      }
      if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
          issue.getShape()[position])
        return false;
    }
    return true;
  };
  if (!appendFreeAxes(lhsIssue, lhsReductionPosition) ||
      !appendFreeAxes(rhsIssue, rhsReductionPosition))
    return emitError()
           << "layered partial free axes require exact register-replica ownership";
  expectedAxes.push_back(geometry.getAxis());
  expectedShape.push_back(geometry.getLaneCount());
  if (!llvm::equal(expectedAxes, accumulator.getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, accumulator.getShape().asArrayRef()))
    return emitError()
           << "layered partial accumulator does not preserve its free and reduction axes";
  for (size_t position = 0; position < expectedAxes.size(); ++position) {
    const bool reduction = position + 1 == expectedAxes.size();
    if (accumulator.getLayout().getTimeFactors()[position] != 1 ||
        accumulator.getLayout().getLaneFactors()[position] !=
            (reduction ? geometry.getLaneCount() : 1) ||
        accumulator.getLayout().getReplicaFactors()[position] !=
            (reduction ? 1 : expectedShape[position]) ||
        accumulator.getLayout().getFragmentFactors()[position] != 1 ||
        accumulator.getLayout().getLocalFactors()[position] != 1)
      return emitError()
             << "layered partial accumulator has an inconsistent time/lane/replica mapping";
  }
  int64_t expectedResources =
      accumulator.getLayout().getRegisterGroups() + storage.getResourceGroups() + 1;
  for (size_t index = 0; index < rootWindowTypes.size(); ++index) {
    mlir::Attribute typeValue = rootWindowTypes[index];
    mlir::Attribute planValue = rootStoragePlans[index];
    auto instruction =
        mlir::dyn_cast<mlir::StringAttr>(rootWindowInstructions[index]);
    auto typeAttribute = mlir::dyn_cast<mlir::TypeAttr>(typeValue);
    auto type = typeAttribute
                    ? mlir::dyn_cast<weft::riscv::ValueType>(
                          typeAttribute.getValue())
                    : weft::riscv::ValueType();
    auto plan = mlir::dyn_cast<weft::riscv::StorageWindowPlanAttr>(planValue);
    const llvm::StringRef expectedWindowInstruction =
        plan && plan.getKind() == "layered" ? "rvv.storage-window.layered"
                                            : "rvv.storage-window.natural";
    if (!type || type.getLayout().getCarrier() != "rvv" || !plan ||
        !instruction || instruction.getValue() != expectedWindowInstruction ||
        plan.getReductionAxis() != geometry.getAxis() ||
        (static_cast<int64_t>(index) == rootIndex &&
         (plan.getKind() != "layered" ||
          plan.getGroupSize() != geometry.getGroupSize() ||
          plan.getLayerSize() != geometry.getLayerSize())))
      return emitError()
             << "layered partial root requires one typed RVV window and storage plan";
    expectedResources += type.getLayout().getRegisterGroups();
  }
  for (auto [instructionValue, shift, mask] : llvm::zip(
           decodeInstructions,
           geometry.getShiftAmountForLogicalLayer().asArrayRef(),
           geometry.getMaskValueForLogicalLayer().asArrayRef())) {
    auto instruction = mlir::dyn_cast<mlir::StringAttr>(instructionValue);
    const llvm::StringRef expected =
        shift > 0 ? (mask > 0
                         ? llvm::StringRef(
                               "rvv.layered-storage-decode.shift-mask")
                         : llvm::StringRef("rvv.layered-storage-decode.shift"))
                  : (mask > 0
                         ? llvm::StringRef("rvv.layered-storage-decode.mask")
                         : llvm::StringRef(
                               "rvv.layered-storage-decode.identity"));
    if (!instruction || instruction.getValue() != expected)
      return emitError()
             << "layered partial decode instruction disagrees with its typed storage geometry";
  }
  if ((accumulatorElement.getWidth() == 16) !=
          (finalizeInstruction == "rvv.vwredsum.partial") ||
      resourceGroups != expectedResources)
    return emitError()
           << "layered partial final reduction or resource contract disagrees with its program";
  return mlir::success();
}

mlir::LogicalResult GroupedMacPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t group, int64_t unroll,
    int64_t reductionAxis, int64_t storageGroup, int64_t storageLayer,
    int64_t logicalWidth, int64_t lhsBitOffsetBytes,
    int64_t rhsBitOffsetBytes, int64_t recordElements,
    int64_t interleaveRows, llvm::StringRef loadForm,
    int64_t rowStrideAxis, int64_t termByteStride,
    int64_t physicalLayerBase, int64_t physicalLayerStep,
    int64_t shiftBase, int64_t shiftStep, int64_t maskValue,
    mlir::DenseI64ArrayAttr termOrder) {
  const int64_t layers =
      storageLayer > 0 && storageGroup > 0 ? storageGroup / storageLayer : 0;
  const bool exactLayerMap =
      layers > 0 &&
      ((physicalLayerBase == 0 && physicalLayerStep == 1) ||
       (physicalLayerBase == layers - 1 && physicalLayerStep == -1));
  if ((kind != "reduce" && kind != "compact" && kind != "fragmented") ||
      group <= 0 || unroll <= 0 || reductionAxis <= 0 || storageGroup <= 0 ||
      storageLayer <= 0 || storageGroup % storageLayer ||
      storageLayer % group || logicalWidth <= 0 || logicalWidth >= 8 ||
      lhsBitOffsetBytes < 0 || rhsBitOffsetBytes < 0 || recordElements <= 0 ||
      interleaveRows < 0 || termByteStride <= 0 || !exactLayerMap ||
      shiftBase != physicalLayerBase * logicalWidth ||
      shiftStep != physicalLayerStep * logicalWidth ||
      (loadForm != "unit" && loadForm != "strided") ||
      (loadForm == "unit" &&
       (interleaveRows <= 0 || rowStrideAxis != 0 ||
        termByteStride != interleaveRows)) ||
      (loadForm == "strided" &&
       (interleaveRows != 0 || rowStrideAxis <= 0 || termByteStride != 1)) ||
      maskValue != ((int64_t{1} << logicalWidth) - 1))
    return emitError() << "grouped MAC plan has incomplete typed storage geometry";
  auto terms = checkedPositiveProduct({group, unroll});
  if (!terms || termOrder.size() != static_cast<size_t>(*terms))
    return emitError() << "grouped MAC plan has no complete term order";
  llvm::SmallVector<int64_t> sorted(termOrder.asArrayRef());
  llvm::sort(sorted);
  for (auto [index, term] : llvm::enumerate(sorted))
    if (term != static_cast<int64_t>(index))
      return emitError() << "grouped MAC term order must be a permutation";
  for (int64_t slot = 0; slot < unroll; ++slot) {
    llvm::SmallVector<int64_t> slotTerms;
    slotTerms.reserve(static_cast<size_t>(group));
    for (int64_t term = 0; term < group; ++term)
      slotTerms.push_back(termOrder[static_cast<size_t>(slot * group + term)]);
    llvm::sort(slotTerms);
    for (int64_t term = 0; term < group; ++term)
      if (slotTerms[static_cast<size_t>(term)] != slot * group + term)
        return emitError()
               << "grouped MAC term order must preserve each unrolled slot";
  }
  if (kind == "compact" && storageLayer % *terms)
    return emitError()
           << "compact grouped MAC plan crosses a selected storage layer";
  if (kind == "fragmented" && storageLayer % *terms == 0)
    return emitError()
           << "fragmented grouped MAC plan duplicates one compact storage window";
  return mlir::success();
}

mlir::LogicalResult LevelAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t domainId, int64_t axisId, llvm::StringRef relation,
    int64_t stateBirthCount, int64_t stagedBirthCount, int64_t handoffCount,
    llvm::StringRef direction) {
  if (domainId <= 0 || axisId <= 0)
    return emitError() << "physical Level requires positive domain and axis identities";
  if (relation != "rows" && relation != "cols" && relation != "tiles" &&
      relation != "blocks" && relation != "subs")
    return emitError() << "unknown physical Level relation";
  if (stateBirthCount < 0 || stagedBirthCount < 0 || handoffCount < 0)
    return emitError() << "physical Level counts cannot be negative";
  if (direction != "ascending" && direction != "descending")
    return emitError() << "physical Level requires one ordered direction";
  return mlir::success();
}

mlir::LogicalResult ImplementationAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef operation, mlir::DenseI64ArrayAttr parameters) {
  if (engine != "scalar" && engine != "rvv" && engine != "ime" &&
      engine != "transfer")
    return emitError() << "unknown physical implementation engine";
  if (family.empty() || operation.empty())
    return emitError() << "physical implementation anchor must be complete";
  if (llvm::any_of(parameters.asArrayRef(),
                   [](int64_t value) { return value < 0; }))
    return emitError() << "physical implementation parameters cannot be negative";
  return mlir::success();
}

mlir::LogicalResult LeafAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef instruction, llvm::StringRef spelling,
    int64_t operandGroups, int64_t resultGroups, int64_t temporaryGroups,
    int64_t fragmentGroups, llvm::StringRef mask, llvm::StringRef tail,
    mlir::DenseI64ArrayAttr parameters, int64_t localBytes) {
  if (engine != "unselected" && engine != "scalar" && engine != "rvv" &&
      engine != "ime" && engine != "transfer")
    return emitError() << "unknown physical engine";
  if (operandGroups < 0 || resultGroups < 0 || temporaryGroups < 0 ||
      fragmentGroups < 0 || localBytes < 0)
    return emitError() << "leaf resource counts cannot be negative";
  if (mask != "none" && mask != "explicit" && mask != "tail")
    return emitError() << "unknown leaf mask policy";
  if (tail != "exact" && tail != "agnostic" && tail != "undisturbed")
    return emitError() << "unknown leaf tail policy";
  if (engine == "unselected") {
    if (!family.empty() || !instruction.empty() || !spelling.empty())
      return emitError() << "unselected leaf cannot claim a target operation";
  } else if (family.empty() || instruction.empty() || spelling.empty()) {
    return emitError() << "selected leaf contract must be complete";
  }
  if (llvm::any_of(parameters.asArrayRef(), [](int64_t value) { return value < 0; }))
    return emitError() << "leaf parameters cannot be negative";
  return mlir::success();
}

mlir::LogicalResult ConversionAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, llvm::StringRef effect, int64_t temporaryGroups) {
  if (kind != "splat" && kind != "extract" && kind != "reshape" &&
      kind != "tuple" && kind != "register_to_lane" &&
      kind != "time_to_lane" && kind != "lane_to_register" && kind != "lane_repack" &&
      kind != "rvv_to_fragment" &&
      kind != "fragment_to_rvv" && kind != "local_load" &&
      kind != "local_store")
    return emitError() << "unknown physical conversion kind";
  if (effect != "pure" && effect != "read" && effect != "write" &&
      effect != "handoff")
    return emitError() << "unknown physical conversion effect";
  if (temporaryGroups < 0)
    return emitError() << "conversion temporary resources cannot be negative";
  return mlir::success();
}

mlir::LogicalResult ValueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds, LayoutAttr layout) {
  if (!elementType || !layout)
    return emitError() << "physical value requires element and layout types";
  // A reduction-operand projection may eliminate its final logical axis while
  // still requiring an explicit physical carrier (for example, one scalar
  // register tuple selected from RVV lanes).  Such a value has rank zero in
  // the logical program, but its LayoutAttr remains meaningful and must round
  // trip through textual physical IR.  Operations that require a shaped
  // domain verify positive rank locally.
  if (failed(
          verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true)))
    return mlir::failure();
  if (layout.getAxisIds() != axisIds)
    return emitError() << "physical layout must preserve every logical axis identity";
  if (layout.getCarrier() != "unassigned" &&
      failed(verifyStaticRepresentation(emitError, shape.asArrayRef(), layout)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult RecordCohortType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DomainType domain, int64_t width, int64_t partition) {
  if (!domain || domain.getRelation() == "root" || domain.getDomainId() <= 0)
    return emitError() << "record cohort requires one non-root Level domain";
  if (width <= 1 || partition <= 0)
    return emitError()
           << "record cohort width must exceed one and partition must be positive";
  return mlir::success();
}

mlir::LogicalResult MemDescType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds, mlir::DenseI64ArrayAttr strides,
    mlir::DenseI64ArrayAttr origins, int64_t alignment,
    llvm::StringRef addressClass, llvm::StringRef access, int64_t aliasSet,
    llvm::StringRef layoutIdentity, int64_t storageBits, int64_t elements,
    int64_t interleaveRows) {
  if (!mlir::isa<weft::kernel::EncodingType>(encoding))
    return emitError() << "memory descriptor requires a canonical Encoding identity";
  if (failed(verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true)))
    return mlir::failure();
  if (strides.size() != shape.size() || origins.size() != shape.size())
    return emitError() << "descriptor stride/origin rank must match logical rank";
  if (alignment <= 0 || storageBits <= 0 || elements <= 0 || interleaveRows < 0)
    return emitError() << "descriptor storage facts are invalid";
  if (addressClass != "pinned" && addressClass != "slice" &&
      addressClass != "local" && addressClass != "pointer")
    return emitError() << "unknown descriptor address class";
  if (access != "none" && access != "read" && access != "write" &&
      access != "readwrite")
    return emitError() << "unknown descriptor access mode";
  if (addressClass == "pinned" && (aliasSet < 0 || layoutIdentity.empty()))
    return emitError() << "pinned descriptor requires alias and layout identity";
  return mlir::success();
}

mlir::LogicalResult LocalType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds, int64_t sizeBytes, int64_t alignment,
    int64_t aliasSet, llvm::StringRef purpose, int64_t ownerDomainId,
    int64_t birthId, int64_t lifetimeEndDomainId, llvm::StringRef schema) {
  if (!elementType || failed(verifyShape(emitError, shape.asArrayRef(),
                                         axisIds.asArrayRef(), true)))
    return mlir::failure();
  if ((sizeBytes == 0 || sizeBytes < -1) || alignment <= 0 || (alignment & (alignment - 1)) || aliasSet < 0)
    return emitError()
           << "local storage size is positive or dynamic (-1), alignment is a positive power of two, and alias is nonnegative";
  if (purpose != "spill" && purpose != "pack" && purpose != "pipeline" &&
      purpose != "handoff")
    return emitError() << "unknown local storage purpose";
  if (ownerDomainId < 0 || birthId < 0 || lifetimeEndDomainId < ownerDomainId ||
      schema.empty())
    return emitError() << "local storage requires explicit owner, birth, lifetime, and schema";
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(elementType);
  if (encoding && (encoding.getKind() != "base" || purpose != "pack" ||
                   schema != "encoded-interleave" || sizeBytes <= 0))
    return emitError() << "encoded local storage requires a fixed-size interleaved base-record pack";
  if (sizeBytes >= 0) {
    int64_t logicalElements = 1;
    bool fullyStaticShape = true;
    for (int64_t extent : shape.asArrayRef()) {
      if (extent < 0) {
        fullyStaticShape = false;
        continue;
      }
      if (logicalElements > std::numeric_limits<int64_t>::max() / extent)
        return emitError() << "local storage logical shape overflows";
      logicalElements *= extent;
    }
    if (fullyStaticShape) {
      // Encoding elements are logical coordinates within records, not fixed
      // width scalars. EncodedLocalPack/Bind verify the exact byte count from
      // record geometry and the selected local packing plan.
      if (encoding) return mlir::success();
      const unsigned bits = elementBitWidth(elementType);
      if (!bits)
        return emitError() << "local storage element type has no fixed width";
      if (logicalElements >
          (std::numeric_limits<int64_t>::max() - 7) / bits)
        return emitError() << "local storage byte size overflows";
      const int64_t minimumBytes = (logicalElements * bits + 7) / 8;
      if (sizeBytes < minimumBytes)
        return emitError()
               << "local storage does not cover its complete logical value";
    }
  }
  return mlir::success();
}

mlir::LogicalResult FragmentType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, llvm::StringRef role, FragmentPackingAttr packing,
    mlir::Type elementType,
    mlir::DenseI64ArrayAttr shape, mlir::DenseI64ArrayAttr axisIds,
    LayoutAttr layout, int64_t resourceGroups) {
  if (family.empty() ||
      (role != "lhs" && role != "rhs" && role != "accumulator") ||
      !packing || !elementType || resourceGroups <= 0 || !layout ||
      layout.getCarrier() != "ime")
    return emitError() << "fragment requires one complete IME representation";
  if (failed(verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef())) ||
      layout.getAxisIds() != axisIds)
    return mlir::failure();
  if (llvm::any_of(shape.asArrayRef(), [](int64_t extent) {
        return extent <= 0;
      }))
    return emitError() << "extension fragment shape must be fully static";
  const auto *contract = weft::findRISCVFragmentCapability(family);
  if (!contract)
    return emitError() << "fragment family has no registered target contract";
  auto capability = weft::riscv::fragmentCapabilityAttr(elementType.getContext(), *contract);
  const bool lhs = role == "lhs", rhs = role == "rhs";
  const int64_t first = lhs ? capability.getActiveMFactor()
                            : rhs ? capability.getKFactor() : capability.getActiveMFactor();
  const int64_t second = lhs ? capability.getKFactor() : capability.getNFactor();
  const int64_t groups = lhs ? capability.getLhsResourceGroups()
                             : rhs ? capability.getRhsResourceGroups() : capability.getAccumulatorResourceGroups();
  const auto expectedPacking = lhs ? capability.getLhsPacking()
                                   : rhs ? capability.getRhsPacking() : capability.getAccumulatorPacking();
  const int64_t expectedBits = lhs ? capability.getLhsBits()
                                  : rhs ? capability.getRhsBits() : capability.getAccumulatorBits();
  const bool signednessMatches = lhs || rhs
      ? integerSignednessMatches(elementType, lhs ? capability.getLhsSignedness() : capability.getRhsSignedness())
      : weft::riscv::fragmentAccumulatorTypeMatches(capability, elementType);
  if (shape.size() != 2 ||
      !supportsRISCVFragmentTile(*contract, rhs ? contract->activeMFactor : shape[0],
                                lhs ? contract->nFactor : shape[1],
                                lhs ? shape[1] : rhs ? shape[0] : contract->kFactor) ||
      packing != expectedPacking ||
      resourceGroups != groups * (shape[0] / first) * (shape[1] / second) ||
      elementBitWidth(elementType) != expectedBits || !signednessMatches)
    return emitError() << "fragment shape, element, packing and resources must match its registered role";
  for (size_t axis = 0; axis < 2; ++axis) {
    const int64_t atom = axis == 0 ? first : second;
    if (layout.getTimeFactors()[axis] != 1 || layout.getLaneFactors()[axis] != 1 ||
        layout.getLocalFactors()[axis] != 1 || layout.getFragmentFactors()[axis] != atom ||
        layout.getReplicaFactors()[axis] != shape[axis] / atom)
      return emitError() << "fragment layout must explicitly separate atom coordinates and bounded replicas";
  }
  const unsigned bits = elementBitWidth(elementType);
  if (!bits || packing.getStorageBits() < bits)
    return emitError()
           << "fragment packing storage width must cover its logical element";
  auto order = packing.getAxisOrder().asArrayRef();
  if (shape.size() != 2 || order.size() != 2 ||
      shape[order[0]] % packing.getRowsPerTile() ||
      shape[order[1]] % packing.getColumnsPerTile())
    return emitError()
           << "fragment logical shape must tile exactly under its physical axis order";
  if (failed(verifyStaticRepresentation(emitError, shape.asArrayRef(), layout)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult WindowType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, mlir::Type lhsType, mlir::Type rhsType,
    mlir::Type resultType,
    int64_t reductionAxis, int64_t slots, int64_t termsPerSlot,
    int64_t resultParts, LayoutAttr partialLayout, LayoutAttr resultLayout,
    int64_t resourceGroups) {
  if (family != "grouped-mac" && family != "encoded-dot" &&
      family != "contract")
    return emitError() << "unknown physical operand-window family";
  auto lhs = mlir::dyn_cast<ValueType>(lhsType);
  auto rhs = mlir::dyn_cast<ValueType>(rhsType);
  auto result = mlir::dyn_cast<ValueType>(resultType);
  if (!lhs || !rhs || !result ||
      reductionAxis <= 0 || slots <= 0 ||
      termsPerSlot <= 0 || resultParts <= 0 || !partialLayout || !resultLayout ||
      partialLayout.getCarrier() != "rvv" || resultLayout.getCarrier() != "rvv" ||
      result.getLayout() != resultLayout || resourceGroups <= 0)
    return emitError() << "physical operand window requires complete typed resources";
  if (!llvm::is_contained(lhs.getAxisIds().asArrayRef(), reductionAxis) ||
      !llvm::is_contained(rhs.getAxisIds().asArrayRef(), reductionAxis) ||
      llvm::is_contained(result.getAxisIds().asArrayRef(), reductionAxis))
    return emitError()
           << "operand window must consume one shared reduction axis and preserve only free axes";
  auto extentFor = [&](ValueType value, int64_t axis) {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
    return found == value.getAxisIds().asArrayRef().end()
               ? int64_t(0)
               : value.getShape()[static_cast<size_t>(
                     found - value.getAxisIds().asArrayRef().begin())];
  };
  const int64_t lhsReduction = extentFor(lhs, reductionAxis);
  const int64_t rhsReduction = extentFor(rhs, reductionAxis);
  if (lhsReduction > 0 && rhsReduction > 0 && lhsReduction != rhsReduction)
    return emitError()
           << "operand window reduction-axis extents must agree";
  llvm::DenseMap<int64_t, int64_t> freeExtents;
  auto collectFree = [&](ValueType value) -> mlir::LogicalResult {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (axis == reductionAxis)
        continue;
      auto [found, inserted] = freeExtents.try_emplace(axis, extent);
      if (inserted)
        continue;
      if (found->second != extent)
        return emitError()
               << "operand window operands disagree on a shared free-axis extent";
    }
    return mlir::success();
  };
  if (failed(collectFree(lhs)) || failed(collectFree(rhs)))
    return mlir::failure();
  if (freeExtents.size() != result.getAxisIds().size())
    return emitError()
           << "operand window result must preserve every free logical axis";
  for (auto [axis, extent] : llvm::zip(result.getAxisIds().asArrayRef(),
                                      result.getShape().asArrayRef())) {
    auto found = freeExtents.find(axis);
    if (found == freeExtents.end() || found->second != extent)
      return emitError()
             << "operand window result must preserve each free-axis extent in its result order";
  }
  if (partialLayout.getAxisIds() != lhs.getAxisIds() &&
      partialLayout.getAxisIds() != result.getAxisIds())
    return emitError()
           << "operand window partial layout must describe its lhs or result domain";
  auto timeParts =
      checkedPositiveProduct(resultLayout.getTimeFactors().asArrayRef());
  auto registerParts =
      checkedPositiveProduct(resultLayout.getReplicaFactors().asArrayRef());
  const bool partsOverflow =
      !timeParts || !registerParts ||
      *timeParts > std::numeric_limits<int64_t>::max() / *registerParts;
  const int64_t expectedParts =
      partsOverflow ? -1 : *timeParts * *registerParts;
  if (resultParts != expectedParts)
    return emitError()
           << "physical operand window result parts disagree with its layout";
  return mlir::success();
}

mlir::LogicalResult LayeredWindowType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ValueType fieldType, ValueType resultType, int64_t reductionAxis,
    int64_t layers, int64_t windowsPerLayer, int64_t resourceGroups) {
  if (!fieldType || !resultType || reductionAxis <= 0 || layers <= 1 ||
      windowsPerLayer <= 0 || resourceGroups <= 0 ||
      fieldType.getElementType() != resultType.getElementType() ||
      resultType.getLayout().getCarrier() != "rvv")
    return emitError()
           << "layered storage window requires typed field/result values and positive geometry";
  auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(), reductionAxis);
  auto resultAxis = llvm::find(resultType.getAxisIds().asArrayRef(), reductionAxis);
  if (fieldAxis == fieldType.getAxisIds().asArrayRef().end() ||
      resultAxis == resultType.getAxisIds().asArrayRef().end())
    return emitError()
           << "layered storage window must preserve its reduction-axis identity";
  const size_t resultPosition = static_cast<size_t>(
      resultAxis - resultType.getAxisIds().asArrayRef().begin());
  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - fieldType.getAxisIds().asArrayRef().begin());
  if (resultType.getLayout().getTimeFactors()[resultPosition] != 1 ||
      resultType.getLayout().getLaneFactors()[resultPosition] <= 1 ||
      resultType.getShape()[resultPosition] !=
          resultType.getLayout().getLaneFactors()[resultPosition] ||
      resultType.getShape()[resultPosition] > fieldType.getShape()[fieldPosition] ||
      resultType.getLayout().getVl() < resultType.getShape()[resultPosition] ||
      resourceGroups != resultType.getLayout().getRegisterGroups())
    return emitError()
           << "layered storage window result must be one complete, resource-exact RVV lane window";
  if (fieldType.getAxisIds() != resultType.getAxisIds())
    return emitError()
           << "layered storage window must preserve every logical axis identity";
  for (size_t position = 0; position < resultType.getShape().size(); ++position) {
    if (position == resultPosition)
      continue;
    if (fieldType.getShape()[position] != resultType.getShape()[position] ||
        resultType.getLayout().getTimeFactors()[position] != 1 ||
        resultType.getLayout().getLaneFactors()[position] != 1 ||
        (resultType.getShape()[position] > 0
             ? resultType.getLayout().getReplicaFactors()[position] !=
                   resultType.getShape()[position]
             : resultType.getLayout().getReplicaFactors()[position] <= 0) ||
        resultType.getLayout().getFragmentFactors()[position] != 1 ||
        resultType.getLayout().getLocalFactors()[position] != 1)
      return emitError()
             << "layered storage window free axes must be preserved as register replicas";
  }
  return mlir::success();
}

mlir::LogicalResult PartialSetType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ValueType partialType, int64_t reductionAxis, int64_t slots,
    int64_t termsPerSlot, int64_t resourceGroups) {
  if (!partialType || reductionAxis <= 0 || slots <= 0 || termsPerSlot <= 0 ||
      resourceGroups <= 0 || partialType.getLayout().getCarrier() != "rvv")
    return emitError()
           << "partial set requires one RVV partial type and positive topology";
  auto axis = llvm::find(partialType.getAxisIds().asArrayRef(), reductionAxis);
  if (axis == partialType.getAxisIds().asArrayRef().end())
    return emitError() << "partial set type does not retain its reduction axis";
  const size_t position = static_cast<size_t>(
      axis - partialType.getAxisIds().asArrayRef().begin());
  if (partialType.getLayout().getTimeFactors()[position] != 1 ||
      partialType.getLayout().getLaneFactors()[position] !=
          partialType.getShape()[position] ||
      partialType.getLayout().getReplicaFactors()[position] != 1 ||
      partialType.getLayout().getFragmentFactors()[position] != 1 ||
      partialType.getLayout().getLocalFactors()[position] != 1)
    return emitError()
           << "partial set reduction axis must be one complete lane partial";
  for (size_t other = 0; other < partialType.getShape().size(); ++other) {
    if (other == position)
      continue;
    const int64_t lanes = partialType.getLayout().getLaneFactors()[other];
    const int64_t replicas =
        partialType.getLayout().getReplicaFactors()[other];
    if (partialType.getLayout().getTimeFactors()[other] != 1 || lanes <= 0 ||
        replicas <= 0 ||
        (partialType.getShape()[other] > 0 &&
         lanes * replicas != partialType.getShape()[other]) ||
        partialType.getLayout().getFragmentFactors()[other] != 1 ||
        partialType.getLayout().getLocalFactors()[other] != 1)
      return emitError()
             << "partial-set free axes must remain exact lane/register coordinates";
  }
  if (partialType.getLayout().getRegisterGroups() <= 0 ||
      slots > std::numeric_limits<int64_t>::max() /
                  partialType.getLayout().getRegisterGroups() ||
      resourceGroups !=
          slots * partialType.getLayout().getRegisterGroups())
    return emitError()
           << "partial set resource groups disagree with its independent slots";
  return mlir::success();
}

mlir::LogicalResult DomainType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef axisName, int64_t domainId, int64_t parentDomainId,
    int64_t axisId, llvm::StringRef relation, llvm::StringRef tail) {
  if (axisName.empty() ||
      (relation != "root" && relation != "rows" && relation != "cols" &&
       relation != "tiles" && relation != "blocks" && relation != "subs") ||
      (tail != "exact" && tail != "tail"))
    return emitError() << "invalid physical domain identity";
  if (relation == "root") {
    if (domainId != 0 || parentDomainId != -1 || axisId != 0)
      return emitError() << "root domain identity is fixed";
  } else if (domainId <= 0 || parentDomainId < 0 || axisId <= 0) {
    return emitError() << "child domain identities must be positive";
  }
  return mlir::success();
}

mlir::LogicalResult PointType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DomainType domain) {
  if (!domain)
    return emitError() << "point requires one physical domain";
  return mlir::success();
}

mlir::LogicalResult EncodingDeclOp::verify() {
  const size_t count = getFieldNames().size();
  if (getKind() != "base" || getLayoutIdentity() != getSymName() ||
      getAlignment() <= 0 || getElements() <= 0 || getStorageBits() <= 0 ||
      count == 0 || getFieldTypes().size() != count ||
      getFieldShapes().size() != count || getFieldLayouts().size() != count ||
      getFieldBitOffsets().size() != count ||
      getFieldStorageBits().size() != count)
    return emitOpError("physical Encoding declaration is incomplete");
  if (failed(verifyStringArray(*this, getFieldNames(), "field_names")))
    return mlir::failure();
  int64_t covered = 0;
  for (auto [offset, width] : llvm::zip(getFieldBitOffsets(),
                                       getFieldStorageBits())) {
    if (offset < 0 || width <= 0 || offset + width > getStorageBits())
      return emitOpError("Encoding field storage span is invalid");
    covered = std::max(covered, offset + width);
  }
  for (size_t index = 0; index < getPadding().size(); index += 3) {
    if (index + 2 >= getPadding().size())
      return emitOpError("padding is an offset,width,fill sequence");
    covered = std::max(covered, getPadding()[index] + getPadding()[index + 1]);
  }
  if (covered != getStorageBits())
    return emitOpError("Encoding fields and padding must cover the record");
  return mlir::success();
}

mlir::LogicalResult DerivedEncodingOp::verify() {
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily() || getLayoutIdentity().empty() ||
      getParameterNames().empty() ||
      getParameterNames().size() != getParameterValues().size() ||
      getInterleaveRows() <= 0)
    return emitOpError("derived Encoding declaration is incomplete");
  if (failed(verifyStringArray(*this, getParameterNames(), "parameter_names")))
    return mlir::failure();
  llvm::DenseSet<llvm::StringRef> names;
  for (auto [nameAttr, value] :
       llvm::zip(getParameterNames(), getParameterValues())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    if (name.empty() || !names.insert(name).second || value <= 0)
      return emitOpError(
          "derived Encoding parameters must be named uniquely and be positive");
  }
  if (getParameterValues().size() != 1 ||
      getParameterValues()[0] != getInterleaveRows())
    return emitOpError(
        "the supported derived Encoding is exactly one parameterized interleave");
  auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
  unsigned sourceDeclarations = 0;
  EncodingDeclOp sourceDeclaration;
  unsigned identicalDerivations = 0;
  if (module)
    module.walk([&](mlir::Operation *operation) {
      if (auto declaration = mlir::dyn_cast<EncodingDeclOp>(operation)) {
        if (declaration.getSymName() == getSourceFamily()) {
          ++sourceDeclarations;
          sourceDeclaration = declaration;
        }
        return;
      }
      auto derived = mlir::dyn_cast<DerivedEncodingOp>(operation);
      if (derived && derived.getResultFamily() == getResultFamily() &&
          derived.getLayoutIdentity() == getLayoutIdentity() &&
          derived.getParameterValues() == getParameterValues())
        ++identicalDerivations;
    });
  if (sourceDeclarations != 1 || identicalDerivations != 1)
    return emitOpError(
        "derived Encoding must uniquely reference one base declaration and one concrete identity");
  if (mlir::failed(weft::kernel::fieldInterleaveSpans(
          sourceDeclaration.getStorageBits(), sourceDeclaration.getFieldBitOffsets(),
          sourceDeclaration.getFieldStorageBits(), sourceDeclaration.getPadding(),
          getFieldChunks())))
    return emitOpError("derived field interleave chunks do not close its byte storage relation");
  return mlir::success();
}

mlir::LogicalResult ArtifactPackOp::verify() {
  if (!mlir::isa<mlir::ModuleOp>(getOperation()->getParentOp()))
    return emitOpError("artifact pack must be a module-level target entry");
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily() || getLayoutIdentity().empty() ||
      getRecordBytes() <= 0 || getElements() <= 0 || getInterleaveRows() <= 0)
    return emitOpError("artifact pack identity and storage geometry must be complete");
  mlir::Block &size = getSizeBody().front();
  if (size.getNumArguments() != 2 ||
      !llvm::all_of(size.getArgumentTypes(),
                    [](mlir::Type type) { return type.isIndex(); }) ||
      !mlir::isa<ArtifactSizeYieldOp>(size.getTerminator()))
    return emitOpError("artifact packed-size region takes M/K and yields one index");
  mlir::Block &pack = getPackBody().front();
  if (pack.getNumArguments() != 4 ||
      !mlir::isa<MemDescType>(pack.getArgument(0).getType()) ||
      !mlir::isa<MemDescType>(pack.getArgument(1).getType()) ||
      !pack.getArgument(2).getType().isIndex() ||
      !pack.getArgument(3).getType().isIndex() ||
      !mlir::isa<ArtifactReturnOp>(pack.getTerminator()))
    return emitOpError(
        "artifact pack region takes source/target descriptors and M/K indices");
  auto source = mlir::cast<MemDescType>(pack.getArgument(0).getType());
  auto target = mlir::cast<MemDescType>(pack.getArgument(1).getType());
  auto sourceEncoding = mlir::cast<weft::kernel::EncodingType>(source.getEncoding());
  auto targetEncoding = mlir::cast<weft::kernel::EncodingType>(target.getEncoding());
  if (sourceEncoding.getFamily() != getSourceFamily() ||
      targetEncoding.getFamily() != getResultFamily() ||
      targetEncoding.getLayoutIdentity() != getLayoutIdentity() ||
      target.getInterleaveRows() != getInterleaveRows())
    return emitOpError("artifact pack region descriptors disagree with its Encoding identity");
  DerivedEncodingOp declaration;
  if (auto module = getOperation()->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DerivedEncodingOp candidate) {
      if (candidate.getResultFamily() == getResultFamily() &&
          candidate.getLayoutIdentity() == getLayoutIdentity() &&
          candidate.getParameterValues() == targetEncoding.getParameters().asArrayRef())
        declaration = candidate;
    });
  if (!declaration || declaration.getFieldChunks() != getFieldChunks())
    return emitOpError("artifact pack chunks disagree with its declared storage transform");
  return mlir::success();
}

mlir::LogicalResult ArtifactSizeYieldOp::verify() {
  auto parent = getOperation()->getParentOfType<ArtifactPackOp>();
  if (!parent || getOperation()->getParentRegion() != &parent.getSizeBody())
    return emitOpError("artifact_size_yield terminates only an artifact size region");
  return mlir::success();
}

mlir::LogicalResult ArtifactReturnOp::verify() {
  auto parent = getOperation()->getParentOfType<ArtifactPackOp>();
  if (!parent || getOperation()->getParentRegion() != &parent.getPackBody())
    return emitOpError("artifact_return terminates only an artifact pack region");
  return mlir::success();
}

mlir::LogicalResult StorageLoadOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  if (!integer || integer.getWidth() != 8 || !integer.isSignless())
    return emitOpError("raw artifact storage load produces one signless storage byte");
  if (!getOperation()->getParentOfType<ArtifactPackOp>() ||
      !canRead(getSource().getType().getAccess()) ||
      !exactLeaf(getLeaf(), "transfer", "artifact-storage",
                 "scalar.storage.load.u8", "none", "exact"))
    return emitOpError("raw storage load is private to a typed artifact builder");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult StorageStoreOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getValue().getType());
  if (!integer || integer.getWidth() != 8 || !integer.isSignless())
    return emitOpError("raw artifact storage store consumes one signless storage byte");
  if (!getOperation()->getParentOfType<ArtifactPackOp>() ||
      !canWrite(getDestination().getType().getAccess()) ||
      !exactLeaf(getLeaf(), "transfer", "artifact-storage",
                 "scalar.storage.store.u8", "none", "exact"))
    return emitOpError("raw storage store is private to a typed artifact builder");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult KernelOp::verify() {
  if (getPanelTransferStrategy() != "issue-work" && getPanelTransferStrategy() != "max-fields" &&
      getPanelTransferStrategy() != "single-field")
    return emitOpError("panel transfer strategy is outside its finite target domain");
  if (!llvm::is_contained(llvm::ArrayRef<int64_t>({1, 2, 4}), int64_t(getContractionColumnCopies())) ||
      !llvm::is_contained(llvm::ArrayRef<int64_t>({1, 2, 4, 8}), int64_t(getPanelTransferFields())) ||
      !llvm::is_contained(llvm::ArrayRef<int64_t>({32, 64, 128, 256}), int64_t(getContractionPanelK())) ||
      !llvm::is_contained(llvm::ArrayRef<int64_t>({1, 2, 4}), int64_t(getContractionKSteps())))
    return emitOpError("dense physical binding is outside its finite target parameter domain");
  if (getPointwiseIssueCopies() != 1 && getPointwiseIssueCopies() != 2 && getPointwiseIssueCopies() != 4)
    return emitOpError("pointwise issue copies must be 1, 2 or 4");
  if (getLocalTileStrategy() != "target-default" && getLocalTileStrategy() != "atom-partitioned" &&
      getLocalTileStrategy() != "register-grouped")
    return emitOpError("unsupported local tile strategy");
  mlir::Block &body = getBody().front();
  if (getArgNames().size() != body.getNumArguments() ||
      getArgAccess().size() != body.getNumArguments() ||
      getArgAliasSets().size() != body.getNumArguments() ||
      getParameterNames().size() != getParameterValues().size())
    return emitOpError("physical kernel ABI and parameter bindings disagree");
  if (failed(verifyStringArray(*this, getArgNames(), "arg_names")) ||
      failed(verifyStringArray(*this, getArgAccess(), "arg_access")) ||
      failed(verifyStringArray(*this, getShapeSymbols(), "shape_symbols")) ||
      failed(verifyStringArray(*this, getParameterNames(), "parameter_names")))
    return mlir::failure();
  llvm::SmallDenseSet<llvm::StringRef> requirements;
  for (mlir::Attribute attribute : getRequiredExtensions()) {
    auto extension = mlir::cast<mlir::StringAttr>(attribute).getValue();
    if (!requirements.insert(extension).second)
      return emitOpError("duplicate required extension: ") << extension;
    if (!findRISCVExtension(extension))
      return emitOpError("unregistered required extension: ") << extension;
    if (!llvm::any_of(getTarget().getFragments(), [&](mlir::Attribute attribute) {
          auto capability = mlir::cast<FragmentCapabilityAttr>(attribute);
          const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
          return contract && contract->extension == extension;
        }))
      return emitOpError("required extension is absent from the target capabilities: ") << extension;
  }
  for (mlir::BlockArgument argument : body.getArguments()) {
    auto descriptor = mlir::dyn_cast<MemDescType>(argument.getType());
    if (!descriptor) {
      if (!isScalar(argument.getType()) ||
          getArgAliasSets()[argument.getArgNumber()] != -1 ||
          mlir::cast<mlir::StringAttr>(getArgAccess()[argument.getArgNumber()]).getValue() != "none")
        return emitOpError("physical scalar parameters require scalar types without memory effects or alias sets");
      continue;
    }
    if (descriptor.getAliasSet() !=
        getArgAliasSets()[argument.getArgNumber()])
      return emitOpError(
          "physical kernel ABI alias facts must match argument memdescs");
  }
  if (!mlir::isa<ReturnOp>(body.getTerminator()))
    return emitOpError("physical kernel body must end in weft_riscv.return");
  if (failed(weft::verifyStrideABI<MemDescType>(*this, body, getArgNames(),
                                               getShapeSymbols(), getStridedArgsAttr())))
    return mlir::failure();
  if (getVectorRegisterPeak() < 0 || getFragmentRegisterPeak() < 0 ||
      getLocalStorageBytes() < 0)
    return emitOpError("physical resource summary cannot be negative");
  return mlir::success();
}

mlir::LogicalResult HelperOp::verify() {
  if (getName().empty() || getBody().front().getNumArguments())
    return emitOpError("physical helper requires a named closure body");
  return mlir::success();
}

mlir::LogicalResult HelperReturnOp::verify() {
  auto helper = (*this)->getParentOfType<HelperOp>();
  if (!helper || getValues().size() != helper.getNumResults())
    return emitOpError("helper return signature is incomplete");
  for (auto [value, result] : llvm::zip(getValues(), helper.getResults())) {
    if (elementOf(value.getType()) != elementOf(result.getType()) ||
        shapeOf(value.getType()) != shapeOf(result.getType()))
      return emitOpError("helper return must preserve its declared numerical type and shape");
  }
  return mlir::success();
}

void HelperReturnOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), weft::kernel::FunctionControlResource::get());
}

mlir::LogicalResult ReturnOp::verify() {
  if (!mlir::isa_and_nonnull<KernelOp>(getOperation()->getParentOp()) ||
      !getValues().empty())
    return emitOpError("physical kernel returns only through explicit stores");
  return mlir::success();
}

mlir::LogicalResult RootDomainOp::verify() {
  return getResult().getType().getRelation() == "root"
             ? mlir::success()
             : emitOpError("root_domain must produce the root domain");
}

mlir::LogicalResult SymbolOp::verify() {
  return getName().empty() ? emitOpError("shape symbol name must not be empty")
                           : mlir::success();
}

mlir::LogicalResult StrideOp::verify() {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(getView());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!argument || !kernel || argument.getOwner() != &kernel.getBody().front() ||
      getAxis() < 0 || getAxis() >= static_cast<int64_t>(getView().getType().getShape().size()))
    return emitOpError("stride must query an axis of a pinned kernel argument");
  if (!weft::hasStrideArguments(kernel.getStridedArgsAttr(), kernel.getArgNames()[argument.getArgNumber()]))
    return emitOpError("stride requires a declared strided View ABI");
  if (getView().getType().getStrides()[getAxis()] != -1)
    return emitOpError("runtime stride cannot replace a known static stride");
  return mlir::success();
}

mlir::LogicalResult DomainOp::verify() {
  DomainType parent = getParent().getType();
  DomainType child = getResult().getType();
  if (child.getRelation() == "root" ||
      child.getParentDomainId() != parent.getDomainId())
    return emitOpError("physical child domain must preserve parent identity");
  return mlir::success();
}

mlir::LogicalResult LoopOp::verify() {
  if (getStateBirthCount() < 0 || getStagedBirthCount() < 0 ||
      getHandoffCount() != static_cast<int64_t>(getResults().size()))
    return emitOpError("physical loop birth/handoff identity is incomplete");
  if (getCarried().size() != getResults().size())
    return emitOpError("physical loop carried operands and results must match");
  if (getDomain().getType().getParentDomainId() !=
      getParentPoint().getType().getDomain().getDomainId())
    return emitOpError(
        "physical loop must consume the explicit point of its parent domain");
  for (auto [operand, result] : llvm::zip(getCarried(), getResults()))
    if (operand.getType() != result.getType())
      return emitOpError("physical loop carry types must be stable");
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1 + getCarried().size() ||
      body.getArgument(0).getType() !=
          PointType::get(getContext(), getDomain().getType()))
    return emitOpError("physical loop body signature is malformed");
  for (auto [argument, carried] :
       llvm::zip(body.getArguments().drop_front(), getCarried()))
    if (argument.getType() != carried.getType())
      return emitOpError("physical loop body carry type mismatch");
  auto yield = mlir::dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().getTypes() != getResults().getTypes())
    return emitOpError("physical loop yield must match loop results");
  int64_t stateBirths = 0;
  int64_t stagedBirths = 0;
  const int64_t owner = getDomain().getType().getDomainId();
  for (mlir::Operation &operation : body.without_terminator()) {
    if (auto state = mlir::dyn_cast<NewOp>(operation);
        state && state.getOwnerDomainId() == owner)
      ++stateBirths;
    if (auto staged = mlir::dyn_cast<MaterializeOp>(operation);
        staged && staged.getOwnerDomainId() == owner)
      ++stagedBirths;
  }
  if (stateBirths != getStateBirthCount() ||
      stagedBirths != getStagedBirthCount())
    return emitOpError(
        "physical loop birth identities disagree with its typed body");
  return mlir::success();
}

mlir::LogicalResult YieldOp::verify() {
  return mlir::isa_and_nonnull<LoopOp>(getOperation()->getParentOp())
             ? mlir::success()
             : emitOpError("weft_riscv.yield terminates a physical loop");
}

mlir::LogicalResult RootPointOp::verify() {
  if (getDomain().getType().getRelation() != "root" ||
      getResult().getType().getDomain() != getDomain().getType())
    return emitOpError("root_point must preserve the root domain identity");
  return mlir::success();
}

mlir::LogicalResult PhysicalPointOp::verify() {
  auto parent = getParent().getType().getDomain();
  auto result = getResult().getType().getDomain();
  if (result.getDomainId() <= 0 ||
      result.getParentDomainId() != parent.getDomainId())
    return emitOpError("physical point requires a non-root domain identity");
  return mlir::success();
}

mlir::LogicalResult RecordCohortOp::verify() {
  auto origin = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto partition =
      origin ? origin.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
             : mlir::arith::ConstantIndexOp();
  auto active =
      origin ? origin.getActive().getDefiningOp<mlir::arith::ConstantIndexOp>()
             : mlir::arith::ConstantIndexOp();
  RecordCohortType cohort = getResult().getType();
  if (!origin || origin.getResult().getType().getDomain() != cohort.getDomain() ||
      !partition || !active || partition.value() != cohort.getPartition() ||
      active.value() != partition.value())
    return emitOpError(
        "record cohort must preserve one fully active physical Level point and its static partition");
  return mlir::success();
}

mlir::LogicalResult ConstantOp::verify() {
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(getValue());
  if (!typed || typed.getType() != getResult().getType() ||
      !isScalar(getResult().getType()))
    return emitOpError("physical constant attribute and scalar result must match");
  return mlir::success();
}

mlir::LogicalResult ReduceRegionOp::verify() {
  return weft::verifyReductionRegion<ValueType, ReduceYieldOp>(*this, getAxis(), getCombine(),
      [](mlir::Operation *op) {
        return mlir::isa<ConstantOp, mlir::arith::ConstantOp, UnaryOp, BinaryOp,
            CompareOp, SelectOp, CastOp, BitcastOp, WidenOp, NarrowOp>(op);
      });
}

mlir::LogicalResult ScanOp::verify() {
  return weft::verifyReductionRegion<ValueType, ScanYieldOp>(*this, getAxis(), getCombine(),
      [](mlir::Operation *op) {
        return mlir::isa<ConstantOp, mlir::arith::ConstantOp, UnaryOp, BinaryOp,
            CompareOp, SelectOp, CastOp, BitcastOp, WidenOp, NarrowOp>(op);
      }, true);
}

mlir::LogicalResult IotaOp::verify() {
  mlir::Type element = elementOf(getResult().getType());
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  if (getEnd() <= getStart() || getResult().getType().getShape().size() != 1 ||
      getResult().getType().getShape()[0] != getEnd() - getStart() ||
      (!element.isIndex() && (!integer || integer.isSigned())))
    return emitOpError(
        "physical iota must match one unsigned-integer or index domain");
  auto layout = getResult().getType().getLayout();
  if (layout.getCarrier() == "scalar" &&
      llvm::any_of(layout.getLaneFactors().asArrayRef(), [](int64_t n) { return n != 1; }))
    return emitOpError("varying scalar iota coordinates must use explicit time or register parts");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult NewOp::verify() {
  if (!isPhysicalValue(getResult().getType()) ||
      getInitialized() != static_cast<bool>(getInitial()) ||
      !getInitialized() || !getInitial() ||
      (getPlacement() != "scalar" && getPlacement() != "register" &&
       getPlacement() != "local"))
    return emitOpError("physical state birth has an invalid placement or initial value");
  if (getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId())
    return emitOpError("physical state birth requires explicit owner and lifetime");
  if (auto fragment = mlir::dyn_cast<FragmentType>(getResult().getType())) {
    if (getInitial().getType() != fragment || fragment.getRole() != "accumulator" ||
        getPlacement() != "register")
      return emitOpError("fragment state birth requires its exact accumulator initializer and register lifetime");
    return mlir::success();
  }
  if (auto result = mlir::dyn_cast<ValueType>(getResult().getType())) {
    if (auto initial = mlir::dyn_cast<ValueType>(getInitial().getType())) {
      if (!sameLogicalDomain(initial, result) ||
          initial.getElementType() != result.getElementType() ||
          initial.getLayout() != result.getLayout())
        return emitOpError(
            "physical shaped state initializer requires an explicit closed layout conversion");
    } else if (!isScalar(getInitial().getType()) ||
               elementOf(getInitial().getType()) != result.getElementType()) {
      return emitOpError(
          "physical state initializer must be layout-identical or scalar-broadcastable");
    }
    return mlir::success();
  }
  if (!isScalar(getResult().getType()) ||
      getInitial().getType() != getResult().getType())
    return emitOpError(
        "scalar physical state initializer and result types must match exactly");
  return mlir::success();
}

mlir::LogicalResult MaterializeOp::verify() {
  if (!isPhysicalValue(getInput().getType()) ||
      !isPhysicalValue(getResult().getType()) ||
      !sameLogicalDomain(getInput().getType(), getResult().getType()) ||
      elementOf(getInput().getType()) != elementOf(getResult().getType()) ||
      (getPlacement() != "shared" && getPlacement() != "register" &&
       getPlacement() != "local" && getPlacement() != "reload"))
    return emitOpError("physical materialize preserves value identity and has one placement");
  if (getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() || getSchema().empty())
    return emitOpError("physical staged birth requires explicit owner, lifetime, and schema");
  if (getLocalPackPlan() && getPlacement() != "local")
    return emitOpError("a selected local pack plan requires local placement");
  if (getLocalPackPlan() &&
      !mlir::isa<kernel::EncodingType>(elementOf(getResult().getType())))
    return emitOpError("a local encoded pack plan requires an encoded value");
  return mlir::success();
}

mlir::LogicalResult StagedViewOp::verify() {
  if (getSource().getType() != getResult().getType() || getSchema().empty() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId())
    return emitOpError(
        "reload staged view must preserve one typed descriptor and carry its owner/lifetime identity");
  return mlir::success();
}

mlir::LogicalResult MemoryViewOp::verify() {
  const int64_t rank = getBase().getType().getShape().size();
  if (getBase().getType() != getResult().getType() ||
      static_cast<int64_t>(getExtents().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank ||
      static_cast<int64_t>(getOrigins().size()) != rank)
    return emitOpError(
        "memory view requires one explicit extent, stride, and origin per descriptor axis");
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(getBase());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (argument && kernel && argument.getOwner() == &kernel.getBody().front() &&
      weft::hasStrideArguments(kernel.getStridedArgsAttr(),
                               kernel.getArgNames()[argument.getArgNumber()])) {
    for (auto [axis, value] : llvm::enumerate(getStrides())) {
      auto stride = value.getDefiningOp<StrideOp>();
      if (getBase().getType().getStrides()[axis] != -1 || !stride ||
          stride.getView() != getBase() || stride.getAxis() != static_cast<int64_t>(axis))
        return emitOpError("strided entry view must consume its declared ABI stride for every axis");
    }
  }
  return mlir::success();
}

mlir::LogicalResult SliceOp::verify() {
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")) ||
      getBase().getType().getEncoding() != getResult().getType().getEncoding())
    return emitOpError("physical slice preserves pinned Encoding identity");
  auto base = getBase().getType();
  auto result = getResult().getType();
  if (result.getAddressClass() != "slice" ||
      result.getAccess() != base.getAccess() ||
      result.getAliasSet() != base.getAliasSet() ||
      result.getLayoutIdentity() != base.getLayoutIdentity() ||
      result.getAlignment() != base.getAlignment() ||
      result.getStorageBits() != base.getStorageBits() ||
      result.getElements() != base.getElements() ||
      result.getInterleaveRows() != base.getInterleaveRows())
    return emitOpError(
        "physical slice must preserve its base storage, alias, and access facts");
  if (getSelectors().size() > base.getShape().size())
    return emitOpError("physical slice has more selectors than base axes");
  if (failed(verifyProjection(*this, base, getIndices(), getSelectors(), result,
                              false)))
    return mlir::failure();
  int64_t indexOperands = 0;
  llvm::SmallVector<int64_t> expectedStrides;
  llvm::SmallVector<int64_t> expectedOrigins;
  for (size_t dimension = 0; dimension < base.getShape().size(); ++dimension) {
    llvm::StringRef kind = "all";
    if (dimension < getSelectors().size())
      kind = mlir::cast<mlir::StringAttr>(getSelectors()[dimension]).getValue();
    if (kind != "all")
      ++indexOperands;
    if (kind == "index" || kind == "group_index")
      continue;
    if (kind != "all" && kind != "domain")
      return emitOpError("physical memory slice selector is not addressable");
    expectedStrides.push_back(base.getStrides()[dimension]);
    expectedOrigins.push_back(base.getOrigins()[dimension]);
  }
  if (indexOperands != static_cast<int64_t>(getIndices().size()) ||
      expectedStrides.size() != result.getShape().size() ||
      result.getStrides().asArrayRef() != llvm::ArrayRef(expectedStrides) ||
      result.getOrigins().asArrayRef() != llvm::ArrayRef(expectedOrigins))
    return emitOpError(
        "physical slice indices and projected descriptor strides disagree");
  return mlir::success();
}

mlir::LogicalResult SubviewOp::verify() {
  MemDescType base = getBase().getType();
  MemDescType result = getResult().getType();
  if (base.getEncoding() != result.getEncoding() ||
      base.getAxisIds() != result.getAxisIds() ||
      base.getStrides() != result.getStrides() ||
      base.getOrigins() != result.getOrigins() ||
      result.getAddressClass() != "slice" ||
      base.getAccess() != result.getAccess() ||
      base.getAliasSet() != result.getAliasSet() ||
      base.getLayoutIdentity() != result.getLayoutIdentity() ||
      base.getAlignment() != result.getAlignment() ||
      base.getStorageBits() != result.getStorageBits() ||
      base.getElements() != result.getElements() ||
      base.getInterleaveRows() != result.getInterleaveRows())
    return emitOpError(
        "physical subview preserves base storage facts and narrows only its logical extents");
  return weft::kernel::verifySubviewBounds(*this, base.getShape(), result.getShape(),
      getOffsetsAttr(), getExtentsAttr(), getDynamicBounds());
}

mlir::LogicalResult ExtentOp::verify() {
  if (getAxis() < 0 || static_cast<size_t>(getAxis()) >= getBase().getType().getShape().size())
    return emitOpError("extent requires an in-range descriptor axis position");
  return mlir::success();
}

mlir::LogicalResult ReshapeOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  auto inputCount = checkedProduct(input.getShape().asArrayRef());
  auto resultCount = checkedProduct(result.getShape().asArrayRef());
  bool axisRename = input.getShape() == result.getShape() &&
                    llvm::equal(getOrder(), input.getAxisIds().asArrayRef());
  bool unitProjection = kernel::isUnitAxisProjection(input.getShape(), input.getAxisIds(),
      result.getShape(), result.getAxisIds(), getOrder());
  if (input.getElementType() != result.getElementType() ||
      (!axisRename && !unitProjection && (!inputCount || !resultCount || *inputCount != *resultCount)))
    return emitOpError(
        "physical reshape preserves element type and static element count, ordered shape, or the non-unit axis domain");
  auto order = getOrder();
  auto inputAxes = input.getAxisIds().asArrayRef();
  if (order.size() != inputAxes.size())
    return emitOpError("physical reshape order must cover every input axis");
  llvm::DenseSet<int64_t> seen;
  for (int64_t axis : order)
    if (!llvm::is_contained(inputAxes, axis) || !seen.insert(axis).second)
      return emitOpError(
          "physical reshape order must be an input-axis permutation");

  LayoutAttr source = input.getLayout();
  LayoutAttr target = result.getLayout();
  if (source.getCarrier() == "unassigned" || target.getCarrier() == "unassigned") {
    if (source.getCarrier() != target.getCarrier())
      return emitOpError(
          "physical reshape layouts must be assigned together");
    return mlir::success();
  }
  if (auto selected = getPartOrder()) {
    auto expected = reshapeCarrierParts(input, result, order);
    if (!expected || !llvm::equal(*selected, *expected))
      return emitOpError("reshape part_order must preserve every lane and logical coordinate");
  } else if (!llvm::equal(order, inputAxes) &&
      !hasIdenticalReshapeCarrier(input, result, order))
    return emitOpError(
        "reshape axis permutation must preserve every physical carrier coordinate");
  auto sameFactorCount = [&](mlir::DenseI64ArrayAttr lhs,
                             mlir::DenseI64ArrayAttr rhs) {
    return checkedProduct(lhs.asArrayRef()) == checkedProduct(rhs.asArrayRef());
  };
  if (source.getCarrier() != target.getCarrier() ||
      source.getSew() != target.getSew() ||
      source.getLmulEighths() != target.getLmulEighths() ||
      source.getVl() != target.getVl() ||
      source.getRegisterGroups() != target.getRegisterGroups() ||
      source.getValidity() != target.getValidity() ||
      !sameFactorCount(source.getTimeFactors(), target.getTimeFactors()) ||
      !sameFactorCount(source.getLaneFactors(), target.getLaneFactors()) ||
      !sameFactorCount(source.getReplicaFactors(), target.getReplicaFactors()) ||
      !sameFactorCount(source.getFragmentFactors(), target.getFragmentFactors()) ||
      !sameFactorCount(source.getLocalFactors(), target.getLocalFactors()))
    return emitOpError(
        "physical reshape requires one linearly identical carrier partition; input=")
           << input << ", result=" << result;
  return mlir::success();
}

mlir::LogicalResult ReaxisOp::verify() {
  auto source = getBase().getType(), result = getResult().getType();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(source.getEncoding());
  auto renamed = MemDescType::get(getContext(), source.getEncoding(), source.getShape(),
      result.getAxisIds(), source.getStrides(), source.getOrigins(), source.getAlignment(),
      source.getAddressClass(), source.getAccess(), source.getAliasSet(), source.getLayoutIdentity(),
      source.getStorageBits(), source.getElements(), source.getInterleaveRows());
  if (!encoding || encoding.getKind() != "dense" || source.getElements() != 1 ||
      source.getInterleaveRows() != 0 || renamed != result)
    return emitOpError("descriptor reaxis only renames ordered axes of an unchanged dense storage region");
  return mlir::success();
}

static bool exactReadSnapshotLeaf(mlir::Operation *operation, LeafAttr leaf) {
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel)
    return false;
  auto target = kernel.getTarget();
  const bool vector = target.getHasRVV();
  auto parameters = leaf.getParameters().asArrayRef();
  return exactLeaf(leaf, "transfer", "load",
                   vector ? "rvv.read-snapshot" : "scalar.read-snapshot",
                   "none", "exact") &&
         leaf.getOperandGroups() == 0 && leaf.getResultGroups() == 0 &&
         leaf.getFragmentGroups() == 0 && leaf.getLocalBytes() == 0 &&
         (vector ? parameters.size() == 2 &&
                       (parameters[0] == 8 || parameters[0] == 16 || parameters[0] == 32) &&
                       leaf.getTemporaryGroups() == parameters[0] / 8 &&
                       parameters[1] == target.getVlenBits() * parameters[0] / 64 &&
                       parameters[1] > 0 &&
                       llvm::is_contained(
                           target.getLegalLMULEighths().asArrayRef(), parameters[0]) &&
                       llvm::is_contained(target.getSupportedSEW().asArrayRef(), 8)
                 : parameters.empty() && leaf.getTemporaryGroups() == 0);
}

void LoadOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
        mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getRegionMutable());
  if (getSnapshotStorage())
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(1));
}

mlir::LogicalResult LoadOp::verify() {
  if (!canRead(getRegion().getType().getAccess()) ||
      !isPhysicalValue(getResult().getType()) ||
      !sameLogicalDomain(getRegion().getType(), getResult().getType()))
    return emitOpError("physical load produces a scalar or shaped physical value");
  auto encoding =
      mlir::cast<weft::kernel::EncodingType>(getRegion().getType().getEncoding());
  mlir::Type resultElement = elementOf(getResult().getType());
  if (encoding.getKind() == "dense") {
    if (std::max(8u, elementBitWidth(resultElement)) != getRegion().getType().getStorageBits())
      return emitOpError(
          "dense physical load result width must match its memory Encoding");
  } else if (resultElement != encoding) {
    return emitOpError(
        "encoded physical load must preserve the pinned Encoding identity");
  }
  if (mlir::Value allocation = getSnapshotStorage()) {
    auto memory = getRegion().getType();
    auto value = mlir::dyn_cast<ValueType>(getResult().getType());
    auto storage = mlir::cast<LocalType>(allocation.getType());
    const int64_t extent = value && value.getShape().size() == 1
                               ? value.getShape()[0]
                               : 0;
    const int64_t recordBytes = memory.getStorageBits() / 8;
    if (encoding.getKind() != "base" || !value ||
        value.getLayout().getCarrier() != "local" ||
        value.getLayout().getValidity() != "full" || extent <= 0 ||
        extent % memory.getElements() || memory.getStorageBits() % 8 ||
        recordBytes <= 0 || memory.getInterleaveRows() != 0 ||
        memory.getStrides().size() != 1 ||
        memory.getStrides()[0] != 1 ||
        extent / memory.getElements() >
            std::numeric_limits<int64_t>::max() / recordBytes ||
        storage.getSizeBytes() != extent / memory.getElements() * recordBytes ||
        !storage.getElementType().isInteger(8) || !storage.getShape().empty() ||
        !storage.getAxisIds().empty() || storage.getPurpose() != "pack" ||
        storage.getSchema() != "read-snapshot" ||
        storage.getAlignment() < memory.getAlignment() ||
        !allocation.getDefiningOp<LocalAllocOp>() || !allocation.hasOneUse() ||
        !exactReadSnapshotLeaf(*this, getLeaf()))
      return emitOpError(
          "encoded read snapshot requires an allocated complete contiguous record range");
  } else if (getLeaf().getInstruction().ends_with("read-snapshot")) {
    return emitOpError("record snapshot leaf requires explicit local storage");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult StoreOp::verify() {
  if (!canWrite(getRegion().getType().getAccess()))
    return emitOpError("physical store descriptor is not writable");
  if (!isPhysicalValue(getValue().getType()) ||
      !sameLogicalDomain(getValue().getType(), getRegion().getType()))
    return emitOpError("physical store value and descriptor domains must match");
  if (auto plan = getFieldStorePlanAttr()) {
    auto field = getRegion().getDefiningOp<FieldOp>();
    auto value = mlir::dyn_cast<ValueType>(getValue().getType());
    auto element = value ? value.getElementType() : getValue().getType();
    auto access = getAccess();
    const bool scalar = !value || value.getLayout().getCarrier() == "scalar";
    auto count = value ? checkedPositiveProduct(value.getShape().asArrayRef())
                       : std::optional<int64_t>(1);
    if (!field || !count || elementBitWidth(element) != plan.getLogicalBits() ||
        (scalar != (plan.getKind() == "scalar" || plan.getKind() == "scalar_bytes")) ||
        (!scalar && (value.getShape().size() != 1 || value.getLayout().getCarrier() != "rvv" ||
                     value.getLayout().getVl() != plan.getLanes() ||
                     value.getLayout().getReplicaFactors()[0] != 1)) ||
        !checkedPositiveProduct({*count, plan.getLogicalBits()}) ||
        *count * plan.getLogicalBits() != access.getStorageBits() ||
        getLeaf().getFamily() != "encoded-field-store" ||
        access.getBitOffset() != field.getAccess().getBitOffset() ||
        access.getStorageBits() != field.getAccess().getStorageBits() ||
        access.getFieldChunkBytes() != field.getAccess().getFieldChunkBytes())
      return emitOpError("field store plan disagrees with its complete logical field and physical carrier");
    const int64_t parts = scalar ? 1 : *count / plan.getLanes();
    const int64_t bytesPerStore = plan.getLanes() * (plan.getStoreBits() / 8);
    llvm::DenseSet<int64_t> visitedParts;
    int64_t cursor = 0;
    for (auto [i, offset] : llvm::enumerate(plan.getByteOffsets().asArrayRef())) {
      if (offset != cursor || bytesPerStore > access.getStorageBits() / 8 - cursor)
        return emitOpError("field store groups must cover byte storage exactly once in order");
      cursor += bytesPerStore;
      for (int64_t layer = 0; layer < plan.getSourcesPerStore(); ++layer) {
        const size_t index = i * plan.getSourcesPerStore() + layer;
        const int64_t part = plan.getSourceParts()[index];
        const int64_t shift = plan.getBitShifts()[index];
        if (part >= parts)
          return emitOpError("field store source part exceeds its selected physical carrier");
        if (plan.getKind() == "scalar_bytes") {
          if (part != 0 || shift != offset * 8)
            return emitOpError("scalar byte store must preserve little-endian bit positions");
        } else if (plan.getKind() == "layered") {
          const int64_t logical = part * plan.getLanes();
          if (access.getMapping() != "grouped_layered" || access.getLayerSize() <= 0 ||
              access.getGroupSize() != access.getLayerSize() * plan.getSourcesPerStore() ||
              access.getLayerSize() % plan.getLanes() ||
              offset != logical / access.getGroupSize() * access.getLayerSize() +
                            logical % access.getLayerSize() ||
              shift != (access.getOrder() == "lo_first"
                            ? logical % access.getGroupSize() / access.getLayerSize()
                            : plan.getSourcesPerStore() - 1 - logical % access.getGroupSize() / access.getLayerSize()) * plan.getLogicalBits() ||
              !visitedParts.insert(part).second)
            return emitOpError("layered field store does not invert its declared bit layout");
        } else if (part != static_cast<int64_t>(i) || shift != 0) {
          return emitOpError("natural field store does not preserve source part order");
        }
      }
    }
    if (cursor != access.getStorageBits() / 8)
      return emitOpError("field store plan leaves bytes unwritten");
    const int64_t expectedTemporaries = !scalar && plan.getKind() == "layered"
                                           ? 2 * ((value.getLayout().getLmulEighths() + 7) / 8)
                                           : 0;
    if (getLeaf().getTemporaryGroups() != expectedTemporaries)
      return emitOpError("field store packing temporaries disagree with its selected carrier");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult AtomicRMWOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getValue().getType());
  auto memory = getRegion().getType();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(memory.getEncoding());
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!integer || integer.isSignless() ||
      (integer.getWidth() != 32 && integer.getWidth() != 64) ||
      getResult().getType() != integer || !memory.getShape().empty() ||
      !encoding || encoding.getKind() != "dense" ||
      memory.getAccess() != "readwrite" ||
      memory.getAlignment() < integer.getWidth() / 8 ||
      !owner || (integer.getWidth() == 64 &&
                 !owner.getTarget().getMarch().starts_with("rv64")))
    return emitOpError("atomic update requires an aligned scalar readwrite integer address supported by XLEN");
  std::string family = (integer.isUnsigned() ? "u" : "i") +
                       std::to_string(integer.getWidth());
  if (encoding.getFamily() != family || getKind() != "add" ||
      !exactLeaf(getLeaf(), "scalar", "atomic-rmw", "scalar.atomic-add",
                 "none", "exact"))
    return emitOpError("atomic update dtype/kind and selected leaf disagree");
  if (getOrdering() != "relaxed" && getOrdering() != "acquire" &&
      getOrdering() != "release" && getOrdering() != "acq_rel" &&
      getOrdering() != "seq_cst")
    return emitOpError("unsupported atomic memory ordering");
  return mlir::success();
}

mlir::LogicalResult FieldOp::verify() {
  mlir::Type ownerType = getOwner().getType();
  mlir::Type encodingType;
  if (auto memory = mlir::dyn_cast<MemDescType>(ownerType))
    encodingType = memory.getEncoding();
  else if (auto value = mlir::dyn_cast<ValueType>(ownerType))
    encodingType = value.getElementType();
  auto encoding = mlir::dyn_cast<weft::kernel::EncodingType>(encodingType);
  if (getName().empty() || !encoding ||
      (!isPhysicalValue(getResult().getType()) &&
       !mlir::isa<MemDescType>(getResult().getType())))
    return emitOpError("physical field access requires a named physical result");
  llvm::StringRef family = baseEncodingFamily(*this, encoding);
  auto declaration = findEncodingDeclaration(*this, family);
  if (!declaration)
    return emitOpError("physical field owner has no matching Encoding declaration");
  size_t fieldIndex = declaration.getFieldNames().size();
  for (auto [index, name] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(name).getValue() == getName()) {
      fieldIndex = index;
      break;
    }
  if (fieldIndex == declaration.getFieldNames().size())
    return emitOpError("physical field name is absent from its Encoding");
  if (getAccess().getForm() != "unassigned") {
    std::optional<int64_t> chunkBytes;
    if (encoding.getKind() == "base")
      chunkBytes = 1;
    else if (auto module = getOperation()->getParentOfType<mlir::ModuleOp>())
      module.walk([&](DerivedEncodingOp derived) {
        if (derived.getResultFamily() == encoding.getFamily() &&
            derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
            derived.getParameterValues() == encoding.getParameters().asArrayRef() &&
            fieldIndex < derived.getFieldChunks().size())
          chunkBytes = derived.getFieldChunks()[fieldIndex];
      });
    if (!chunkBytes || getAccess().getFieldChunkBytes() != *chunkBytes ||
        getAccess().getBitOffset() != declaration.getFieldBitOffsets()[fieldIndex] ||
        getAccess().getStorageBits() != declaration.getFieldStorageBits()[fieldIndex])
      return emitOpError("selected field edge disagrees with its declared byte storage transform");
  }
  mlir::Type fieldType =
      mlir::cast<mlir::TypeAttr>(declaration.getFieldTypes()[fieldIndex]).getValue();
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[fieldIndex])
          .asArrayRef();
  if (shapeOf(ownerType).empty() ||
      shapeOf(ownerType).back() != declaration.getElements())
    return emitOpError(
        "physical encoded field owner must end in one complete record axis");
  llvm::SmallVector<int64_t> expectedShape(shapeOf(ownerType).drop_back());
  expectedShape.append(fieldShape.begin(), fieldShape.end());
  auto resultShape = shapeOf(getResult().getType());
  auto ownerAxes = axesOf(ownerType);
  auto resultAxes = axesOf(getResult().getType());
  if (!llvm::equal(expectedShape, resultShape) ||
      resultAxes.size() < ownerAxes.size() - 1 ||
      !llvm::equal(resultAxes.take_front(ownerAxes.size() - 1),
                   ownerAxes.drop_back()) ||
      (!fieldShape.empty() &&
       (resultAxes.empty() || resultAxes.back() != ownerAxes.back())))
    return emitOpError(
        "physical field result must replace the record axis with its declared field shape");
  if (auto result = mlir::dyn_cast<ValueType>(getResult().getType())) {
    if (result.getElementType() != fieldType)
      return emitOpError(
          "physical field result element must match its Encoding declaration");
  } else if (isScalar(getResult().getType())) {
    if (!fieldShape.empty() || getResult().getType() != fieldType)
      return emitOpError(
          "scalar physical field result must match one scalar Encoding field");
  } else {
    auto memory = mlir::cast<MemDescType>(getResult().getType());
    auto dense = mlir::cast<weft::kernel::EncodingType>(memory.getEncoding());
    if (dense.getKind() != "dense" ||
        memory.getStorageBits() != elementBitWidth(fieldType))
      return emitOpError(
          "physical field descriptor must use the declared dense element width");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult ExtractOp::verify() {
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")) ||
      !isPhysicalValue(getResult().getType()) ||
      elementOf(getInput().getType()) != elementOf(getResult().getType()))
    return emitOpError("physical extract requires typed selectors and result");
  auto pattern = getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
      "index_pattern");
  if (pattern && getAccess().getForm() == "register") {
    auto result = mlir::dyn_cast<ValueType>(getResult().getType());
    auto indices = rvvRegularGatherIndices(getInput().getType(), result, pattern.asArrayRef());
    auto baseType = getIndices().size() == 1
        ? mlir::dyn_cast<mlir::IntegerType>(getIndices().front().getType())
        : mlir::IntegerType();
    if (!indices || getSelectors().size() != 1 ||
        mlir::cast<mlir::StringAttr>(getSelectors()[0]).getValue() != "regular" ||
        getIndices().size() > 1 ||
        (!getIndices().empty() && !getIndices().front().getType().isIndex() &&
         (!baseType || !baseType.isUnsigned())))
      return emitOpError("regular register gather requires one uniform scalar index per complete carrier");
    if (getLeaf().getEngine() != "unselected" &&
        (getLeaf().getParameters().asArrayRef() != llvm::ArrayRef<int64_t>(*indices) ||
         !exactLeaf(getLeaf(), "rvv", "extract", "rvv.extract.regular-scalar", "none", "exact")))
      return emitOpError("regular register gather leaf must match its carrier index mapping");
    return verifyLeafOperation(*this);
  }
  if (pattern) {
    auto input = getInput().getType();
    auto result = mlir::dyn_cast<ValueType>(getResult().getType());
    size_t regularDimension = getSelectors().size();
    unsigned regularCount = 0;
    for (auto [dimension, selectorAttribute] :
         llvm::enumerate(getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "regular") {
        ++regularCount;
        regularDimension = dimension;
      } else if (selector != "all") {
        return emitOpError(
            "regular physical extract currently permits only all/regular selectors");
      }
    }
    if (!result || pattern.size() != 3 || regularCount != 1 ||
        !getIndices().empty() || getAccess().getForm() != "indexed" ||
        input.getShape().size() != result.getShape().size() ||
        input.getAxisIds() != result.getAxisIds() ||
        regularDimension >= input.getShape().size() || pattern[0] < 0 ||
        pattern[1] <= 0 || pattern[2] <= 0 ||
        result.getShape()[regularDimension] <= 0)
      return emitOpError(
          "regular physical extract requires one closed base/stride/repeat mapping");
    for (size_t dimension = 0; dimension < input.getShape().size(); ++dimension)
      if (dimension != regularDimension &&
          input.getShape()[dimension] != result.getShape()[dimension])
        return emitOpError(
            "regular physical extract must preserve every non-indexed axis");
    const int64_t steps =
        (result.getShape()[regularDimension] - 1) / pattern[2];
    if (steps > (std::numeric_limits<int64_t>::max() - pattern[0]) /
                    pattern[1])
      return emitOpError("regular physical extract index mapping overflows");
    const int64_t last = pattern[0] + steps * pattern[1];
    if (last < pattern[0] || last >= input.getShape()[regularDimension])
      return emitOpError(
          "regular physical extract index mapping exceeds its source axis");
  } else if (failed(verifyProjection(*this, getInput().getType(), getIndices(),
                                     getSelectors(), getResult().getType(),
                                     true))) {
    return mlir::failure();
  }
  if (getAccess().getForm() == "indexed") {
    auto result = mlir::dyn_cast<ValueType>(getResult().getType());
    for (mlir::Value index : getIndices())
      if (auto typedIndex = mlir::dyn_cast<ValueType>(index.getType());
          typedIndex && typedIndex.getLayout().getCarrier() == "rvv" &&
          (!result || !hasLegalRVVIndexedEMUL(typedIndex, result)))
        return emitOpError(
            "indexed extract index EMUL is incompatible with its data layout");
  }
  if (getAccess().getForm() == "register") {
    if (getLeaf().getInstruction() == "scalar.replica-project") {
      auto parts = scalarExtractParts(*this);
      if (!parts || llvm::ArrayRef<int64_t>(*parts) != getLeaf().getParameters().asArrayRef() ||
          !exactLeaf(getLeaf(), "scalar", "extract", "scalar.replica-project", "none", "exact") ||
          getLeaf().getOperandGroups() || getLeaf().getResultGroups() || getLeaf().getTemporaryGroups() ||
          getLeaf().getFragmentGroups() || getLeaf().getLocalBytes())
        return emitOpError("scalar projection requires an exact constant coordinate map");
      return verifyLeafOperation(*this);
    }
    auto result = mlir::dyn_cast<ValueType>(getResult().getType());
    auto input = getInput().getType();
    unsigned gatherCount = 0;
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= getIndices().size())
        return emitOpError("register gather selector has no index operand");
      if (selector == "gather") {
        ++gatherCount;
        gatherIndex = getIndices()[cursor];
      }
      ++cursor;
    }
    auto indices = gatherIndex
                       ? mlir::dyn_cast<ValueType>(gatherIndex.getType())
                       : ValueType();
    auto time = checkedPositiveProduct(input.getLayout().getTimeFactors().asArrayRef());
    auto replicas =
        checkedPositiveProduct(input.getLayout().getReplicaFactors().asArrayRef());
    auto resultTime = result ? checkedPositiveProduct(
                                   result.getLayout().getTimeFactors().asArrayRef())
                             : std::optional<int64_t>();
    auto resultReplicas =
        result ? checkedPositiveProduct(
                     result.getLayout().getReplicaFactors().asArrayRef())
               : std::optional<int64_t>();
    auto indexTime = indices ? checkedPositiveProduct(
                                   indices.getLayout().getTimeFactors().asArrayRef())
                             : std::optional<int64_t>();
    auto indexReplicas =
        indices ? checkedPositiveProduct(
                      indices.getLayout().getReplicaFactors().asArrayRef())
                : std::optional<int64_t>();
    if (result && indices && input.getLayout().getCarrier() == "rvv" &&
        result.getLayout().getCarrier() == "scalar" &&
        indices.getLayout().getCarrier() == "scalar") {
      auto sourceLanes = checkedPositiveProduct(
          input.getLayout().getLaneFactors().asArrayRef());
      auto resultLanes = checkedPositiveProduct(
          result.getLayout().getLaneFactors().asArrayRef());
      auto indexLanes = checkedPositiveProduct(
          indices.getLayout().getLaneFactors().asArrayRef());
      auto laneCount = getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "lane_gather_count");
      auto sourceParts =
          getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "lane_gather_source_parts");
      auto indexParts =
          getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "lane_gather_index_parts");
      if (gatherCount != 1 || !time || *time != 1 || !sourceLanes ||
          *sourceLanes <= 1 || !replicas || *replicas <= 0 || !resultTime ||
          *resultTime != 1 || !resultLanes || *resultLanes != 1 ||
          !resultReplicas || *resultReplicas <= 0 || !indexTime ||
          *indexTime != 1 || !indexLanes || *indexLanes != 1 ||
          !indexReplicas || *indexReplicas <= 0 || !laneCount ||
          laneCount.getInt() != *sourceLanes || !sourceParts || !indexParts ||
          sourceParts.size() != static_cast<size_t>(*resultReplicas) ||
          indexParts.size() != static_cast<size_t>(*resultReplicas) ||
          !exactLeaf(getLeaf(), "rvv", "extract",
                     "rvv.extract.lane-to-replica", "none", "exact"))
        return emitOpError(
            "lane-to-replica extract requires one complete RVV source and a typed source/index map per result replica");
      for (auto [sourcePart, indexPart] :
           llvm::zip(sourceParts.asArrayRef(), indexParts.asArrayRef()))
        if (sourcePart < 0 || sourcePart >= *replicas || indexPart < 0 ||
            indexPart >= *indexReplicas)
          return emitOpError(
              "lane-to-replica extract references an absent source or index part");
      return verifyLeafOperation(*this);
    }
    if (result && indices && input.getLayout().getCarrier() == "scalar" &&
        result.getLayout().getCarrier() == "scalar" &&
        indices.getLayout().getCarrier() == "scalar") {
      auto arity = getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "replica_gather_arity");
      auto candidates = getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "replica_gather_candidates");
      auto keys = getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "replica_gather_keys");
      if (gatherCount != 1 || !time || *time != 1 || !resultTime ||
          *resultTime != 1 || !indexTime || *indexTime != 1 || !replicas ||
          !resultReplicas || !indexReplicas || *resultReplicas <= 1 ||
          *indexReplicas <= 1 || *resultReplicas % *indexReplicas || !arity ||
          arity.getInt() <= 0 ||
          !candidates ||
          candidates.size() !=
              static_cast<size_t>(*resultReplicas * arity.getInt()) ||
          !keys || keys.size() != candidates.size() ||
          !exactLeaf(getLeaf(), "scalar", "extract",
                     "scalar.replica-gather", "none", "exact"))
        return emitOpError(
            "scalar register extract requires one typed replica candidate matrix and exact gather leaf");
      for (int64_t resultPart = 0; resultPart < *resultReplicas; ++resultPart) {
        llvm::DenseSet<int64_t> row;
        llvm::DenseSet<int64_t> keyRow;
        for (int64_t candidate = 0; candidate < arity.getInt(); ++candidate) {
          const int64_t offset = resultPart * arity.getInt() + candidate;
          int64_t source = candidates[offset];
          int64_t key = keys[offset];
          if (source < 0 || source >= *replicas || !row.insert(source).second ||
              key < 0 || !keyRow.insert(key).second)
            return emitOpError(
                "scalar register extract rows require distinct non-negative keys and distinct in-range source replicas");
        }
      }
      return verifyLeafOperation(*this);
    }
    if (!result || !indices || gatherCount != 1 ||
        input.getLayout().getCarrier() != "rvv" ||
        result.getLayout().getCarrier() != "rvv" ||
        indices.getLayout().getCarrier() != "rvv" || !time || *time != 1 ||
        !replicas || *replicas != 1 ||
        input.getLayout().getSew() != result.getLayout().getSew() ||
        input.getLayout().getLmulEighths() !=
            result.getLayout().getLmulEighths() ||
        indices.getLayout().getSew() != result.getLayout().getSew() ||
        indices.getLayout().getLmulEighths() !=
            result.getLayout().getLmulEighths() ||
        !exactLeaf(getLeaf(), "rvv", "extract", "rvv.extract.vrgather",
                   "none", "exact"))
      return emitOpError(
                 "register extract requires one complete RVV source, index, result, and exact gather leaf; input=")
             << input << ", index="
             << (gatherIndex ? gatherIndex.getType() : mlir::Type())
             << ", result=" << getResult().getType();
  }
  return verifyLeafOperation(*this);
}

bool UpdateOp::coversWholeState() {
  auto input = getInput().getType();
  auto value = mlir::dyn_cast<ValueType>(getValue().getType());
  return value && getIndices().empty() && getSelectors().size() == input.getShape().size() &&
      llvm::all_of(getSelectors(), [](mlir::Attribute selector) {
        auto text = mlir::dyn_cast<mlir::StringAttr>(selector);
        return text && text.getValue() == "all";
      }) && value.getElementType() == input.getElementType() &&
      value.getShape() == input.getShape() && value.getAxisIds() == input.getAxisIds() &&
      getResult().getType() == input;
}

mlir::LogicalResult UpdateOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return emitOpError()
           << "physical update preserves its shaped state representation; input="
           << getInput().getType() << ", result=" << getResult().getType();
  if (elementOf(getValue().getType()) != getInput().getType().getElementType())
    return emitOpError("physical update value element type must match its state");
  if (failed(verifyProjection(*this, getInput().getType(), getIndices(),
                              getSelectors(), getValue().getType(), true)))
    return mlir::failure();
  return verifyLeafOperation(*this);
}

mlir::LogicalResult BitcastOp::verify() {
  const auto sourceWidth = elementBitWidth(elementOf(getInput().getType()));
  const auto targetWidth = elementBitWidth(elementOf(getResult().getType()));
  if (sourceWidth < 8 || sourceWidth != targetWidth ||
      !isNumericPhysical(getInput().getType()) ||
      !isNumericPhysical(getResult().getType()))
    return emitOpError("bitcast requires same-width numeric elements");
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto result = mlir::dyn_cast<ValueType>(getResult().getType());
  if (bool(input) != bool(result) || (input && input.getLayout() != result.getLayout()))
    return emitOpError("bitcast requires identical physical coordinates; layout transport must be explicit");
  if (!sameLogicalDomain(getInput().getType(), getResult().getType()))
    return emitOpError("bitcast preserves its complete logical domain");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult UnaryOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      !isNumericPhysical(getInput().getType()) ||
      (getKind() != "neg" && getKind() != "abs" && getKind() != "exp" && getKind() != "exp2" &&
       getKind() != "rsqrt" && getKind() != "sqrt" && getKind() != "log") ||
      ((getKind() == "exp" || getKind() == "exp2" || getKind() == "rsqrt" || getKind() == "sqrt" || getKind() == "log") &&
       !mlir::isa<mlir::FloatType>(elementOf(getInput().getType()))))
    return emitOpError("physical unary operation preserves its value type");
  if (isVectorMathInstruction(getLeaf().getInstruction())) {
    auto type = mlir::dyn_cast<ValueType>(getResult().getType());
    if (!vectorMathInstructionIsLegal(*this, getLeaf().getInstruction()))
      return emitOpError("selected vector math requires a proven input range");
    auto groups = type ? unaryTemporaryGroups(*this, type, getLeaf().getInstruction()) : std::nullopt;
    if (!groups || getLeaf().getTemporaryGroups() != *groups)
      return emitOpError("vector math temporary resources must scale with its selected LMUL");
  }
  if (getLeaf().getInstruction() == "scalar.exp-approx-f32" ||
      getLeaf().getInstruction() == "scalar.log-approx-f32") {
    auto type = mlir::dyn_cast<ValueType>(getResult().getType());
    if (!elementOf(getResult().getType()).isF32() ||
        (type && type.getLayout().getCarrier() != "scalar") ||
        getLeaf().getInstruction() != ("scalar." + getKind() + "-approx-f32").str() ||
        getLeaf().getTemporaryGroups() != 0 || getLeaf().getLocalBytes() != 0)
      return emitOpError("scalar f32 math requires its selected inlined numerical sequence");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult BinaryOp::verify() {
  constexpr llvm::StringLiteral allowed[] = {
      "add", "sub", "mul", "div", "mod", "and", "or",
      "xor", "shl", "shr", "max", "min", "maximum", "minimum"};
  if (!isPhysicalValue(getLhs().getType()) || !isPhysicalValue(getRhs().getType()) ||
      !isPhysicalValue(getResult().getType()) ||
      !isNumericPhysical(getLhs().getType()) ||
      !isNumericPhysical(getRhs().getType()) ||
      !isNumericPhysical(getResult().getType()) ||
      !llvm::is_contained(allowed, getKind()) ||
      elementOf(getLhs().getType()) != elementOf(getRhs().getType()) ||
      elementOf(getLhs().getType()) != elementOf(getResult().getType()))
    return emitOpError("physical binary operation requires physical values");
  if ((getKind() == "maximum" || getKind() == "minimum") &&
      !mlir::isa<mlir::FloatType>(elementOf(getResult().getType())))
    return emitOpError("NaN-propagating extrema require floating-point elements");
  if (failed(verifyBroadcastDomain(*this, getLhs().getType(),
                                   getRhs().getType(), getResult().getType())))
    return mlir::failure();
  return verifyLeafOperation(*this);
}

mlir::LogicalResult SelectOp::verify() {
  mlir::Type result = getResult().getType();
  if (!isPhysicalValue(result) || !elementOf(getCondition().getType()).isInteger(1) ||
      !isNumericPhysical(getTrueValue().getType()) || !isNumericPhysical(getFalseValue().getType()) ||
      elementOf(getTrueValue().getType()) != elementOf(result) ||
      elementOf(getFalseValue().getType()) != elementOf(result))
    return emitOpError("physical select requires an i1 predicate and matching numeric elements");
  llvm::SmallVector<int64_t> axes, shape;
  for (mlir::Value value : {getTrueValue(), getFalseValue(), getCondition()}) {
    if (!isPhysicalValue(value.getType())) return emitOpError("select requires physical values");
    for (auto [axis, extent] : llvm::zip(axesOf(value.getType()), shapeOf(value.getType()))) {
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) { axes.push_back(axis); shape.push_back(extent); }
      else {
        int64_t &previous = shape[found - axes.begin()];
        if (previous == 1) previous = extent;
        else if (extent != 1 && extent != previous)
          return emitOpError("select operands disagree on a shared axis extent");
      }
    }
  }
  if (!llvm::equal(axes, axesOf(result)) || !llvm::equal(shape, shapeOf(result)))
    return emitOpError("select result must preserve its ordered broadcast domain");
  if (getLeaf().getEngine() != "unselected") {
    auto groups = selectTemporaryGroups(*this);
    if (!groups || getLeaf().getTemporaryGroups() != *groups)
      return emitOpError("select physical mapping or temporary resource contract is not closed");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult CompareOp::verify() {
  constexpr llvm::StringLiteral predicates[] = {"eq", "ne", "lt",
                                                 "le", "gt", "ge"};
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
      elementOf(getResult().getType()));
  if (!isPhysicalValue(getLhs().getType()) || !isPhysicalValue(getRhs().getType()) ||
      !isPhysicalValue(getResult().getType()) ||
      !isNumericPhysical(getLhs().getType()) ||
      !isNumericPhysical(getRhs().getType()) ||
      elementOf(getLhs().getType()) != elementOf(getRhs().getType()) ||
      !resultElement || resultElement.getWidth() != 1 ||
      !llvm::is_contained(predicates, getPredicate()))
    return emitOpError("physical compare requires physical values");
  if (failed(verifyBroadcastDomain(*this, getLhs().getType(),
                                   getRhs().getType(), getResult().getType())))
    return mlir::failure();
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVCompareSelectOp::verify() {
  auto type = getResult().getType();
  auto layout = type.getLayout();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  constexpr llvm::StringLiteral predicates[] = {"eq", "ne", "lt", "le", "gt", "ge"};
  bool numeric = type.getElementType().isF32() || type.getElementType().isF64() ||
      (integer && !integer.isSignless() && (integer.getWidth() == 8 || integer.getWidth() == 16 ||
                                           integer.getWidth() == 32 || integer.getWidth() == 64));
  auto matches = [&](mlir::Type operand) { return operand == type || operand == type.getElementType(); };
  if (!kernel || !numeric || !llvm::is_contained(predicates, getPredicate()) ||
      layout.getCarrier() != "rvv" || layout.getReplicaGroup() != 1 ||
      !supportsRVVLayout(kernel.getTarget(), layout) ||
      getLhs().getType() != type || getFalseValue().getType() != type ||
      !matches(getRhs().getType()) || !matches(getTrueValue().getType()))
    return emitOpError("direct compare/select requires one common RVV domain and exactly typed scalar operands");
  auto coordinates = RegisterCoordinates::get(layout);
  if (!coordinates || coordinates->streams() != 1 || coordinates->lanes() != layout.getVl() ||
      layout.getLmulEighths() <= 0 || (layout.getSew() * 8) % layout.getLmulEighths())
    return emitOpError("direct compare/select has no single-issue mask geometry");
  int64_t ratio = layout.getSew() * 8 / layout.getLmulEighths();
  if (!llvm::is_contained(llvm::ArrayRef<int64_t>({1, 2, 4, 8, 16, 32, 64}), ratio))
    return emitOpError("direct compare/select mask ratio is outside the ISA domain");
  auto selected = getLeaf();
  bool rhsVector = getRhs().getType() == type, trueVector = getTrueValue().getType() == type;
  bool floating = mlir::isa<mlir::FloatType>(type.getElementType());
  std::string instruction = std::string("rvv.compare-select.") +
      (rhsVector ? "vv" : floating ? "vf" : "vx") + "." +
      (trueVector ? "vvm" : floating ? "vfm" : "vxm");
  if (selected.getEngine() != "rvv" || selected.getFamily() != "compare-select" ||
      selected.getInstruction() != instruction || selected.getSpelling() != instruction ||
      selected.getOperandGroups() != (2 + int(rhsVector) + int(trueVector)) * layout.getRegisterGroups() ||
      selected.getResultGroups() != layout.getRegisterGroups() || selected.getTemporaryGroups() != 1 ||
      selected.getFragmentGroups() != 0 || selected.getLocalBytes() != 0)
    return emitOpError("direct compare/select requires its exact one-mask resource contract");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult CastOp::verify() {
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto result = mlir::dyn_cast<ValueType>(getResult().getType());
  auto sourceInteger = input
      ? mlir::dyn_cast<mlir::IntegerType>(input.getElementType())
      : mlir::IntegerType();
  auto resultInteger = result
      ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
      : mlir::IntegerType();
  if (sourceInteger && resultInteger &&
      sourceInteger.getWidth() == resultInteger.getWidth() &&
      input.getLayout().getCarrier() == "rvv" &&
      result.getLayout().getCarrier() == "rvv" &&
      input.getLayout() != result.getLayout())
    return emitOpError(
        "same-width RVV integer cast requires identical physical layouts; representation changes require convert_layout");
  return verifyConversion(*this, getInput().getType(), getResult().getType());
}

mlir::LogicalResult NarrowOp::verify() {
  return verifyConversion(*this, getInput().getType(), getResult().getType());
}

mlir::LogicalResult WidenOp::verify() {
  return verifyConversion(*this, getInput().getType(), getResult().getType());
}

mlir::LogicalResult MacGroupsOp::verify() {
  if (getGroup() == 0 ||
      getGroup() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      getOverflow() != "wrap")
    return emitOpError("physical mac_groups requires positive group and wrap semantics");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult ReduceOp::verify() {
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  if (!input || !isPhysicalValue(getResult().getType()) ||
      (getKind() != "add" && getKind() != "max" && getKind() != "min") ||
      getAxis() < 0 || static_cast<size_t>(getAxis()) >= input.getShape().size() ||
      elementOf(getInput().getType()) != elementOf(getResult().getType()))
    return emitOpError("physical reduce requires physical values");
  llvm::SmallVector<int64_t> expectedShape(input.getShape().asArrayRef());
  llvm::SmallVector<int64_t> expectedAxes(input.getAxisIds().asArrayRef());
  expectedShape.erase(expectedShape.begin() + getAxis());
  expectedAxes.erase(expectedAxes.begin() + getAxis());
  if (shapeOf(getResult().getType()) != llvm::ArrayRef<int64_t>(expectedShape) ||
      axesOf(getResult().getType()) != llvm::ArrayRef<int64_t>(expectedAxes))
    return emitOpError(
        "physical reduce must remove exactly its selected axis and preserve every free axis");
  if (auto result = mlir::dyn_cast<ValueType>(getResult().getType()))
    for (auto [resultPosition, axis] :
         llvm::enumerate(result.getAxisIds().asArrayRef())) {
      auto found = llvm::find(input.getAxisIds().asArrayRef(), axis);
      if (found == input.getAxisIds().asArrayRef().end() ||
          !sameAxisMapping(input,
                           static_cast<size_t>(
                               found - input.getAxisIds().asArrayRef().begin()),
                           result, resultPosition))
        return emitOpError()
               << "physical reduce must preserve every free-axis representation; input="
               << input << ", result=" << result
               << ", free_axis=" << axis;
    }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult Fold2Op::verify() {
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto result = mlir::dyn_cast<ValueType>(getResult().getType());
  if (!input || !result)
    return emitOpError("physical fold2 consumes and produces shaped values");
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto inputShape = input.getShape().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  if (inputShape.empty() || inputShape.size() != resultShape.size() ||
      inputShape.back() <= 0 || inputShape.back() % 2 ||
      resultShape.back() != inputShape.back() / 2 ||
      !inputElement || !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 ||
      input.getAxisIds() != result.getAxisIds() ||
      !llvm::equal(inputShape.drop_back(), resultShape.drop_back()))
    return emitOpError(
        "physical fold2 must halve the final axis and produce signed i32");
  if ((input.getLayout().getCarrier() != "unassigned" &&
       input.getLayout().getCarrier() != "local") ||
      (result.getLayout().getCarrier() != "unassigned" &&
       result.getLayout().getCarrier() != "rvv") ||
      (getAccess().getForm() != "unassigned" &&
       (getAccess().getMapping() != "natural" ||
        getAccess().getBitOffset() % 8)))
    return emitOpError()
           << "physical fold2 requires a closed natural-field to RVV realization"
           << " (input=" << input.getLayout().getCarrier()
           << ", result=" << result.getLayout().getCarrier()
           << ", access=" << getAccess().getForm() << "/"
           << getAccess().getMapping() << ", bit_offset="
           << getAccess().getBitOffset() << ")";
  return verifyLeafOperation(*this);
}

static mlir::LogicalResult verifyContractOp(mlir::Operation *operation) {
  auto lhs = mlir::dyn_cast<ValueType>(operation->getOperand(0).getType());
  auto rhs = mlir::dyn_cast<ValueType>(operation->getOperand(1).getType());
  auto result = mlir::dyn_cast<ValueType>(operation->getResult(0).getType());
  auto over = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
  if (!lhs || !rhs || !over || over.empty())
    return operation->emitOpError(
        "physical contract requires shaped operands and reduction axes");
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : over.asArrayRef())
    if (axis <= 0 || !reduction.insert(axis).second ||
        !llvm::is_contained(lhs.getAxisIds().asArrayRef(), axis) ||
        !llvm::is_contained(rhs.getAxisIds().asArrayRef(), axis))
      return operation->emitOpError(
          "physical contract reduction axes must be positive, unique, and shared");
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFree = [&](ValueType value) -> mlir::LogicalResult {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (reduction.contains(axis))
        continue;
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return operation->emitOpError(
            "physical contract operands disagree on a shared free-axis extent");
      }
    }
    return mlir::success();
  };
  if (failed(appendFree(lhs)) || failed(appendFree(rhs)))
    return mlir::failure();
  for (int64_t axis : over.asArrayRef()) {
    auto lhsIt = llvm::find(lhs.getAxisIds().asArrayRef(), axis);
    auto rhsIt = llvm::find(rhs.getAxisIds().asArrayRef(), axis);
    int64_t lhsExtent = lhs.getShape()[static_cast<size_t>(
        lhsIt - lhs.getAxisIds().asArrayRef().begin())];
    int64_t rhsExtent = rhs.getShape()[static_cast<size_t>(
        rhsIt - rhs.getAxisIds().asArrayRef().begin())];
    if (lhsExtent > 0 && rhsExtent > 0 && lhsExtent != rhsExtent)
      return operation->emitOpError(
          "physical contract reduction-axis extents disagree");
  }
  if (expectedAxes.empty()) {
    if (result || !isNumericPhysical(operation->getResult(0).getType()))
      return operation->emitOpError(
          "physical scalar contract must eliminate every logical axis");
  } else if (!result ||
             !llvm::equal(expectedAxes, result.getAxisIds().asArrayRef()) ||
             !llvm::equal(expectedShape, result.getShape().asArrayRef())) {
    return operation->emitOpError(
        "physical contract result must preserve ordered free axes and extents");
  }
  if (auto accumulator = operation->getAttrOfType<mlir::TypeAttr>("acc_type");
      accumulator && accumulator.getValue() !=
                         elementOf(operation->getResult(0).getType()))
    return operation->emitOpError(
        "physical contract accumulator type must equal its result element type");
  auto laneOperand = operation->getAttrOfType<mlir::StringAttr>("lane_operand");
  auto laneForm = operation->getAttrOfType<mlir::StringAttr>("lane_memory_form");
  if (!laneOperand || !laneForm ||
      (laneOperand.getValue() != "unassigned" &&
       laneOperand.getValue() != "lhs" && laneOperand.getValue() != "rhs") ||
      (laneForm.getValue() != "unassigned" && laneForm.getValue() != "unit" &&
       laneForm.getValue() != "strided" && laneForm.getValue() != "indexed" &&
       laneForm.getValue() != "segment" && laneForm.getValue() != "register"))
    return operation->emitOpError(
        "physical contract lane access must be unassigned or explicitly closed");
  return verifyLeafOperation(operation);
}

mlir::LogicalResult DotOp::verify() { return verifyContractOp(*this); }
mlir::LogicalResult ContractOp::verify() { return verifyContractOp(*this); }
mlir::LogicalResult OuterContractOp::verify() { return verifyContractOp(*this); }

void LookupOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
        mlir::MemoryEffects::Effect>> &effects) {
  if (mlir::isa<MemDescType>(getTable().getType()))
    effects.emplace_back(mlir::MemoryEffects::Read::get(), &getTableMutable());
}

mlir::LogicalResult LookupOp::verify() {
  auto indexElement = mlir::dyn_cast<mlir::IntegerType>(
      elementOf(getIndices().getType()));
  if (getBounds() != "in_bounds" ||
      (!elementOf(getIndices().getType()).isIndex() &&
       (!indexElement || indexElement.isSigned())) ||
      !sameLogicalDomain(getIndices().getType(), getResult().getType()) ||
      !isNumericPhysical(getResult().getType()) ||
      (!mlir::isa<ValueType, MemDescType>(getTable().getType())))
    return emitOpError("physical lookup currently requires in-bounds semantics");
  if (auto table = mlir::dyn_cast<ValueType>(getTable().getType());
      table && table.getElementType() != elementOf(getResult().getType()))
    return emitOpError("physical lookup table and result element types disagree");
  if (auto table = mlir::dyn_cast<MemDescType>(getTable().getType())) {
    auto encoding =
        mlir::cast<weft::kernel::EncodingType>(table.getEncoding());
    if (encoding.getKind() != "dense" ||
        table.getStorageBits() != elementBitWidth(elementOf(getResult().getType())))
      return emitOpError(
          "physical lookup descriptor must match the result element width");
  }
  llvm::StringRef carrier = "scalar";
  if (auto indices = mlir::dyn_cast<ValueType>(getIndices().getType()))
    carrier = indices.getLayout().getCarrier();
  auto tableValue = mlir::dyn_cast<ValueType>(getTable().getType());
  auto resultValue = mlir::dyn_cast<ValueType>(getResult().getType());
  // ConvertWeftToRISCV creates a typed lookup before layout propagation and
  // memory planning have assigned its access form and terminal leaf.  Keep
  // that transient state verifiable only while both decisions are explicitly
  // unassigned; PlanRISCVMemory must close them together before final
  // verification.
  if (getAccess().getForm() == "unassigned" &&
      getLeaf().getEngine() == "unselected")
    return verifyLeafOperation(*this);
  if (auto indices = mlir::dyn_cast<ValueType>(getIndices().getType()))
    if (auto result = mlir::dyn_cast<ValueType>(getResult().getType()))
      for (size_t position = 0; position < indices.getAxisIds().size(); ++position)
        if (!sameAxisMapping(indices, position, result, position))
          return emitOpError()
                 << "physical lookup indices/result require an explicit layout conversion; indices="
                 << indices << ", result=" << result;
  if (tableValue) {
    auto tableLayout = tableValue.getLayout();
    auto indexValue = mlir::dyn_cast<ValueType>(getIndices().getType());
    if (!indexValue || !resultValue || carrier != "rvv" ||
        tableLayout.getCarrier() != "rvv" ||
        resultValue.getLayout().getCarrier() != "rvv" ||
        tableLayout.getSew() != resultValue.getLayout().getSew() ||
        tableLayout.getLmulEighths() != resultValue.getLayout().getLmulEighths() ||
        indexValue.getLayout().getSew() != resultValue.getLayout().getSew() ||
        indexValue.getLayout().getLmulEighths() !=
            resultValue.getLayout().getLmulEighths() ||
        getAccess().getForm() != "register" ||
        !exactLeaf(getLeaf(), "rvv", "lookup", "rvv.vrgather", "none",
                   "exact"))
      return emitOpError(
          "register lookup requires one layout-compatible RVV table, index, result, and exact gather leaf");
    auto tableParts = checkedPositiveProduct(tableLayout.getTimeFactors().asArrayRef());
    auto tableReplicas =
        checkedPositiveProduct(tableLayout.getReplicaFactors().asArrayRef());
    if (!tableParts || !tableReplicas || *tableParts != 1 ||
        *tableReplicas != 1)
      return emitOpError(
          "register lookup table must fit one complete RVV register value");
    return verifyLeafOperation(*this);
  }
  if ((carrier == "scalar" &&
       (getAccess().getForm() != "unit" ||
        !exactLeaf(getLeaf(), "scalar", "lookup", "scalar.lookup", "none",
                   "exact"))) ||
      (carrier == "rvv" &&
       (getAccess().getForm() != "indexed" ||
        !resultValue ||
        !hasLegalRVVIndexedEMUL(
            mlir::dyn_cast<ValueType>(getIndices().getType()), resultValue) ||
        !exactLeaf(getLeaf(), "rvv", "lookup", "rvv.vluxei", "none",
                   "exact"))) ||
      (carrier != "scalar" && carrier != "rvv"))
    return emitOpError(
        "physical lookup access, index carrier, and exact leaf must agree");
  return verifyLeafOperation(*this);
}

static mlir::LogicalResult verifyByteReadSource(mlir::Operation *operation,
                                               mlir::Value value) {
  if (auto memory = mlir::dyn_cast<MemDescType>(value.getType())) {
    auto encoding = mlir::cast<weft::kernel::EncodingType>(memory.getEncoding());
    if (encoding.getKind() != "dense" || !canRead(memory.getAccess()) ||
        memory.getShape().size() != 1 || memory.getStrides()[0] != 1 ||
        (memory.getStorageBits() != 8 && memory.getStorageBits() != 16 &&
         memory.getStorageBits() != 32))
      return operation->emitError("byte read requires one contiguous readable byte or word descriptor");
  } else {
    auto field = value.getDefiningOp<FieldOp>();
    auto source = mlir::dyn_cast<ValueType>(value.getType());
    auto word = source ? mlir::dyn_cast<mlir::IntegerType>(source.getElementType())
                       : mlir::IntegerType();
    if (!field || !source || !word || !word.isUnsigned() ||
        (word.getWidth() != 16 && word.getWidth() != 32) ||
        source.getShape().size() != 1 || source.getShape()[0] <= 0 ||
        source.getLayout().getCarrier() != "local" ||
        source.getLayout().getValidity() != "full" ||
        field.getAccess().getMapping() != "natural" ||
        field.getAccess().getBitOffset() % 8 ||
        field.getAccess().getStorageBits() / word.getWidth() != source.getShape()[0] ||
        field.getAccess().getStorageBits() % word.getWidth())
      return operation->emitError("byte read requires one complete byte-aligned natural word field");
  }
  return mlir::success();
}

mlir::LogicalResult RVVByteGatherOp::verify() {
  auto offsets = getByteOffsets().getType();
  auto index = mlir::dyn_cast<mlir::IntegerType>(offsets.getElementType());
  auto result = getResult().getType();
  auto byte = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel || !kernel.getTarget().getHasRVV() ||
      !kernel.getTarget().getHasIndexedMemory() ||
      !supportsRVVLayout(kernel.getTarget(), offsets.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), result.getLayout()))
    return emitOpError("byte gather requires supported RVV indexed-memory carriers");
  if (!index || !index.isUnsigned() ||
      (index.getWidth() != 16 && index.getWidth() != 32) || !byte ||
      !byte.isUnsigned() || byte.getWidth() != 8 ||
      offsets.getShape() != result.getShape() ||
      offsets.getAxisIds() != result.getAxisIds() ||
      offsets.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "rvv" ||
      offsets.getLayout().getValidity() != "full" ||
      result.getLayout().getValidity() != "full" ||
      !hasLegalRVVIndexedEMUL(offsets.getLayout(), 8,
                             result.getLayout().getLmulEighths()))
    return emitOpError("byte gather requires matching full RVV axes and unsigned byte offsets");
  for (size_t axis = 0; axis < result.getShape().size(); ++axis)
    if (!sameAxisMapping(offsets, axis, result, axis))
      return emitOpError("byte gather must preserve every index-axis mapping");
  if (mlir::failed(verifyByteReadSource(*this, getSource())))
    return mlir::failure();
  if (getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "natural" ||
      getAccess().getIndexSEW() != index.getWidth() ||
      getAccess().getAlignment() != 1 ||
      !exactLeaf(getLeaf(), "rvv", "byte-gather", "rvv.byte-gather", "none", "exact"))
    return emitOpError("byte gather requires its exact indexed byte leaf");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVByteWindowsLoadOp::verify() {
  if (mlir::failed(verifyByteReadSource(*this, getSource())))
    return mlir::failure();
  auto bases = getByteBases();
  auto base = bases.empty() ? mlir::IntegerType()
                           : mlir::dyn_cast<mlir::IntegerType>(bases.front().getType());
  auto result = getResult().getType();
  auto layout = result.getLayout();
  auto byte = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  const int64_t width = getWindowBytesAttr().getInt();
  auto lanes = checkedPositiveProduct(result.getShape().asArrayRef());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!base || !base.isUnsigned() ||
      (base.getWidth() != 16 && base.getWidth() != 32) ||
      !byte || byte.isSignless() || byte.getWidth() != 8 || !kernel ||
      !kernel.getTarget().getHasRVV() ||
      !supportsRVVLayout(kernel.getTarget(), layout) || layout.getSew() != 8 ||
      !lanes || *lanes != layout.getVl() ||
      layout.getCarrier() != "rvv" || layout.getValidity() != "full" ||
      !llvm::equal(result.getShape().asArrayRef(), layout.getLaneFactors().asArrayRef()))
    return emitOpError("byte windows require scalar unsigned bases and one full RVV byte carrier");
  for (auto factors : {layout.getTimeFactors(), layout.getReplicaFactors(),
                       layout.getFragmentFactors(), layout.getLocalFactors()})
    if (llvm::any_of(factors.asArrayRef(), [](int64_t factor) { return factor != 1; }))
      return emitOpError("byte windows cannot own traversal, replicas, fragments, or local storage");
  const int64_t domain = int64_t{1} << base.getWidth();
  if (bases.empty() || bases.size() > 4 ||
      width <= 0 || width > domain || (width & (width - 1)) ||
      static_cast<int64_t>(bases.size()) * width != layout.getVl())
    return emitOpError("byte windows require one to four complete power-of-two windows");
  for (mlir::Value windowBase : bases)
    if (windowBase.getType() != base)
      return emitOpError("byte window bases must share one scalar unsigned integer type");
  const int64_t temporary = bases.size() == 1 ? 0 : layout.getRegisterGroups();
  if (getAccess().getForm() != "unit" || getAccess().getMapping() != "natural" ||
      getAccess().getAlignment() != 1 || getAccess().getIndexSEW() != 0 ||
      !exactLeaf(getLeaf(), "rvv", "byte-windows-load", "rvv.byte-windows-pack", "none", "exact") ||
      getLeaf().getTemporaryGroups() != temporary ||
      getLeaf().getLocalBytes() != 0 || getLeaf().getFragmentGroups() != 0)
    return emitOpError("byte windows require exact byte loads and the bounded pack temporary");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVIndexedEntryLoadOp::verify() {
  mlir::Type offsetType = getEntryOffsets().getType();
  auto offsets = mlir::dyn_cast<ValueType>(offsetType);
  auto result = getResult().getType();
  mlir::Type offsetElement = offsets ? offsets.getElementType() : offsetType;
  auto indexElement =
      mlir::dyn_cast<mlir::IntegerType>(offsetElement);
  const bool scalarEntries =
      !offsets || offsets.getLayout().getCarrier() == "scalar";
  const bool vectorEntries =
      offsets && offsets.getLayout().getCarrier() == "rvv";
  if ((!offsetElement.isIndex() &&
       (!indexElement || indexElement.isSigned())) ||
      (!scalarEntries && !vectorEntries) ||
      result.getLayout().getCarrier() != "rvv")
    return emitOpError(
        "indexed entry load requires unsigned scalar or RVV byte offsets and an RVV result");
  const unsigned payloadBits = elementBitWidth(result.getElementType());
  const int64_t payloadElementBytes =
      payloadBits && payloadBits % 8 == 0
          ? static_cast<int64_t>(payloadBits / 8)
          : 0;
  const bool payloadBytesOverflow =
      getPayloadExtent() <= 0 || payloadElementBytes <= 0 ||
      getPayloadExtent() >
          std::numeric_limits<int64_t>::max() / payloadElementBytes;
  const int64_t payloadBytes =
      payloadBytesOverflow ? 0 : getPayloadExtent() * payloadElementBytes;
  if (payloadBytesOverflow || getEntryByteStride() < payloadBytes)
    return emitOpError(
        "indexed entry load requires one positive contiguous payload within each byte stride");

  llvm::SmallVector<int64_t> sourceAxes;
  llvm::SmallVector<int64_t> sourceShape;
  std::optional<size_t> sourcePayloadPosition;
  if (auto descriptor = mlir::dyn_cast<MemDescType>(getSource().getType())) {
    auto encoding =
        mlir::cast<weft::kernel::EncodingType>(descriptor.getEncoding());
    if (encoding.getKind() != "dense" ||
        (descriptor.getShape().size() != 1 &&
         descriptor.getShape().size() != 2) ||
        descriptor.getStorageBits() != elementBitWidth(result.getElementType()))
      return emitOpError(
          "indexed entry load requires a dense descriptor matching the payload element width");
    sourceAxes.append(descriptor.getAxisIds().asArrayRef().begin(),
                      descriptor.getAxisIds().asArrayRef().end());
    sourceShape.append(descriptor.getShape().asArrayRef().begin(),
                       descriptor.getShape().asArrayRef().end());
  } else if (auto value = mlir::dyn_cast<ValueType>(getSource().getType())) {
    if (value.getLayout().getCarrier() != "local" ||
        value.getElementType() != result.getElementType())
      return emitOpError(
          "indexed entry value source must be one local payload with matching element type");
    sourceAxes.append(value.getAxisIds().asArrayRef().begin(),
                      value.getAxisIds().asArrayRef().end());
    sourceShape.append(value.getShape().asArrayRef().begin(),
                       value.getShape().asArrayRef().end());
  } else {
    return emitOpError(
        "indexed entry load source must be a dense descriptor or local value");
  }
  auto sourceAxis = llvm::find(sourceAxes, getSourceAxis());
  if (sourceAxis == sourceAxes.end())
    return emitOpError("indexed entry load source axis is absent from its source");
  const size_t sourcePosition =
      static_cast<size_t>(sourceAxis - sourceAxes.begin());
  if (auto descriptor = mlir::dyn_cast<MemDescType>(getSource().getType());
      descriptor && descriptor.getShape().size() == 2) {
    const int64_t descriptorEntryBytes =
        payloadElementBytes > 0 && descriptor.getStrides()[0] > 0 &&
                descriptor.getStrides()[0] <=
                    std::numeric_limits<int64_t>::max() /
                        payloadElementBytes
            ? descriptor.getStrides()[0] * payloadElementBytes
            : 0;
    if (sourcePosition != 0 || descriptor.getShape()[1] != getPayloadExtent() ||
        descriptorEntryBytes != getEntryByteStride() ||
        descriptor.getStrides()[1] != 1)
      return emitOpError(
          "indexed entry table must be contiguous entry-major payload storage");
    sourcePayloadPosition = 1;
  }

  llvm::SmallVector<int64_t> entryAxes;
  llvm::SmallVector<int64_t> entryShape;
  if (offsets) {
    entryAxes.append(offsets.getAxisIds().asArrayRef().begin(),
                     offsets.getAxisIds().asArrayRef().end());
    entryShape.append(offsets.getShape().asArrayRef().begin(),
                      offsets.getShape().asArrayRef().end());
  }
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (size_t position = 0; position < sourceAxes.size(); ++position) {
    if (position == sourcePosition ||
        (sourcePayloadPosition && position == *sourcePayloadPosition))
      continue;
    expectedAxes.push_back(sourceAxes[position]);
    expectedShape.push_back(sourceShape[position]);
  }
  const size_t retainedSourceAxes = expectedAxes.size();
  for (auto [axis, extent] : llvm::zip(entryAxes, entryShape)) {
    if (llvm::is_contained(expectedAxes, axis) || axis == getPayloadAxis())
      return emitOpError(
          "indexed entry axes must be disjoint from retained source and payload axes");
    expectedAxes.push_back(axis);
    expectedShape.push_back(extent);
  }
  expectedAxes.push_back(getPayloadAxis());
  expectedShape.push_back(getPayloadExtent());
  if (!llvm::equal(expectedAxes, resultAxes) ||
      !llvm::equal(expectedShape, resultShape))
    return emitOpError(
        "indexed entry load result must retain source axes and append entry plus payload axes");
  for (size_t position = 0; position < entryAxes.size(); ++position)
    if (!sameAxisMapping(offsets, position, result,
                         retainedSourceAxes + position))
      return emitOpError(
          "indexed entry load must preserve every outer entry-axis representation");

  const size_t payloadPosition = resultAxes.size() - 1;
  auto layout = result.getLayout();
  if (!payloadElementBytes)
    return emitOpError(
        "indexed entry load requires a byte-addressable payload element");
  if (layout.getTimeFactors()[payloadPosition] != 1 ||
      layout.getLaneFactors()[payloadPosition] != getPayloadExtent() ||
      layout.getReplicaFactors()[payloadPosition] != 1 ||
      layout.getFragmentFactors()[payloadPosition] != 1 ||
      layout.getLocalFactors()[payloadPosition] != 1)
    return emitOpError(
        "indexed entry payload axis must be one complete RVV lane dimension");

  if (getAccess().getMapping() != "natural")
    return emitOpError(
        "indexed entry load requires one natural entry-payload mapping");
  if (scalarEntries) {
    const bool naturallyAligned =
        getAccess().getAlignment() >= payloadElementBytes;
    const llvm::StringRef instruction =
        naturallyAligned ? "rvv.indexed-entry-load"
                         : "rvv.indexed-entry-byte-load";
    if (getAccess().getForm() != "unit" ||
        !exactLeaf(getLeaf(), "rvv", "indexed-entry-load", instruction,
                   "none", "exact"))
      return emitOpError(
          "scalar indexed entries require one alignment-legal exact unit-payload RVV leaf");
  } else {
    const uint64_t packedBits =
        payloadBits * static_cast<uint64_t>(getPayloadExtent());
    if (!indexElement ||
        (indexElement.getWidth() != 16 && indexElement.getWidth() != 32 &&
         indexElement.getWidth() != 64) ||
        !hasLegalRVVIndexedEMUL(offsets.getLayout(), packedBits,
                                result.getLayout().getLmulEighths()) ||
        payloadBits == 0 ||
        (packedBits != 32 && packedBits != 64) ||
        getEntryByteStride() != payloadBytes ||
        getAccess().getForm() != "indexed" ||
        !exactLeaf(getLeaf(), "rvv", "indexed-entry-gather",
                   "rvv.indexed-entry-gather", "none", "exact"))
      return emitOpError(
          "RVV indexed entries require one exact contiguous-entry gather leaf")
             << "; entry_byte_stride=" << getEntryByteStride()
             << ", payload_extent=" << getPayloadExtent()
             << ", access=" << getAccess() << ", leaf=" << getLeaf()
             << ", offset_type=" << offsets << ", result_type=" << result;
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVUnitEntryWindowLoadOp::verify() {
  ValueType source = getSource().getType();
  ValueType result = getResult().getType();
  auto field = getSource().getDefiningOp<FieldOp>();
  auto baseValue = mlir::dyn_cast<ValueType>(getEntryBase().getType());
  mlir::Type baseElement = baseValue ? baseValue.getElementType()
                                     : getEntryBase().getType();
  auto baseInteger = mlir::dyn_cast<mlir::IntegerType>(baseElement);
  bool scalarBase = baseElement.isIndex() ||
                    (baseInteger && !baseInteger.isSigned());
  if (baseValue) {
    auto layout = baseValue.getLayout();
    scalarBase = scalarBase && layout.getCarrier() == "scalar" &&
                 llvm::all_of(layout.getTimeFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLaneFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getReplicaFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getFragmentFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLocalFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; });
  }
  if (!field || source.getLayout().getCarrier() != "local" || !scalarBase ||
      result.getLayout().getCarrier() != "rvv" ||
      source.getElementType() != result.getElementType() ||
      getEntryAxes().empty() ||
      getEntryAxes().size() != getEntryExtents().size() ||
      llvm::any_of(getEntryExtents(),
                   [](int64_t extent) { return extent <= 0; }) ||
      getPayloadExtent() <= 0 ||
      getEntryStride() != getPayloadExtent() ||
      llvm::is_contained(getEntryAxes(), getPayloadAxis()) ||
      getAccess().getForm() != "unit" ||
      getAccess().getMapping() != "natural" ||
      !exactLeaf(getLeaf(), "rvv", "unit-entry-window-load",
                 "rvv.unit-entry-window-load", "none", "exact"))
    return emitOpError(
        "unit entry window load requires one byte-contiguous typed local field, scalar base, and exact RVV leaf");

  llvm::DenseSet<int64_t> entryAxes;
  for (int64_t axis : getEntryAxes())
    if (!entryAxes.insert(axis).second)
      return emitOpError("unit entry window entry axes must be unique");

  auto sourceAxes = source.getAxisIds().asArrayRef();
  auto sourceShape = source.getShape().asArrayRef();
  auto sourceAxis = llvm::find(sourceAxes, getSourceAxis());
  if (sourceAxis == sourceAxes.end() ||
      llvm::any_of(getEntryAxes(), [&](int64_t axis) {
        return llvm::is_contained(sourceAxes, axis);
      }) ||
      (llvm::is_contained(sourceAxes, getPayloadAxis()) &&
       getPayloadAxis() != getSourceAxis()))
    return emitOpError(
        "unit entry window entry axes must be new and a reused payload axis must be the projected source axis");
  const size_t sourcePosition =
      static_cast<size_t>(sourceAxis - sourceAxes.begin());
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (size_t position = 0; position < sourceAxes.size(); ++position) {
    if (position == sourcePosition)
      continue;
    expectedAxes.push_back(sourceAxes[position]);
    expectedShape.push_back(sourceShape[position]);
  }
  const size_t retainedAxes = expectedAxes.size();
  expectedAxes.append(getEntryAxes().begin(), getEntryAxes().end());
  expectedShape.append(getEntryExtents().begin(), getEntryExtents().end());
  expectedAxes.push_back(getPayloadAxis());
  expectedShape.push_back(getPayloadExtent());
  if (!llvm::equal(expectedAxes, result.getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, result.getShape().asArrayRef()))
    return emitOpError(
        "unit entry window result must retain source axes and append entry plus payload axes");
  auto layout = result.getLayout();
  for (size_t position = 0; position < retainedAxes; ++position)
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
          "unit entry window retained axes must remain register coordinates");
  llvm::SmallVector<int64_t> windowExtents(getEntryExtents().begin(),
                                           getEntryExtents().end());
  windowExtents.push_back(getPayloadExtent());
  bool striped = false;
  int64_t precedingLaneSpan = 1;
  for (size_t offset = 0; offset < windowExtents.size(); ++offset) {
    const size_t position = retainedAxes + offset;
    const int64_t extent = windowExtents[offset];
    const int64_t time = layout.getTimeFactors()[position];
    const int64_t lane = layout.getLaneFactors()[position];
    if (time <= 0 || lane <= 0 || layout.getReplicaFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
                 "unit entry and payload axes must form one complete RVV lane window")
             << "; entry_axes=" << getEntryAxes()
             << ", payload_axis=" << getPayloadAxis()
             << ", result_type=" << result;
    if (time == 1) {
      if (lane != extent)
        return emitOpError(
                   "unit entry window lane factors do not cover one complete contiguous strip")
               << "; result_type=" << result;
    } else {
      if (striped || precedingLaneSpan != 1 || lane <= 1 ||
          time > std::numeric_limits<int64_t>::max() / lane ||
          time * lane != extent)
        return emitOpError(
                   "unit entry window permits one primary time-striped entry axis")
               << "; result_type=" << result;
      striped = true;
    }
    if (precedingLaneSpan > std::numeric_limits<int64_t>::max() / lane)
      return emitOpError("unit entry window lane span overflows");
    precedingLaneSpan *= lane;
  }
  return verifyLeafOperation(*this);
}

void ConvertLayoutOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
        mlir::MemoryEffects::Effect>> &effects) {
  llvm::StringRef effect = getConversion().getEffect();
  if (effect == "read") {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable());
    return;
  }
  if (effect == "write") {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::cast<mlir::OpResult>(getResult()));
    return;
  }
  if (effect == "handoff") {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::cast<mlir::OpResult>(getResult()));
  }
}

FieldOp weft::riscv::fieldReadProjection(
    mlir::Value value, llvm::SmallVectorImpl<mlir::Value> *indices) {
  while (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (extract.getAccess().getForm() == "register")
      return {};
    if (indices)
      indices->append(extract.getIndices().begin(), extract.getIndices().end());
    value = extract.getInput();
  }
  return value.getDefiningOp<FieldOp>();
}

std::optional<int64_t>
weft::riscv::encodedFieldRawLMULEighths(ValueType result) {
  if (!result || result.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  const int64_t bits = std::max<int64_t>(8, elementBitWidth(result.getElementType()));
  static constexpr int64_t legalLMULValues[] = {1, 2, 4, 8, 16, 32, 64};
  const auto legalLMUL = llvm::ArrayRef<int64_t>(legalLMULValues);
  const int64_t resultLMUL = result.getLayout().getLmulEighths();
  if (!llvm::is_contained(legalLMUL, resultLMUL))
    return std::nullopt;
  const int64_t numerator = resultLMUL * 8;
  if (numerator % bits)
    return std::nullopt;
  const int64_t lmul = numerator / bits;
  if (!llvm::is_contained(legalLMUL, lmul))
    return std::nullopt;
  return lmul;
}

namespace {
bool isInterleavedScalarField(FieldOp field) {
  auto type = field ? mlir::dyn_cast<ValueType>(field.getResult().getType()) : ValueType();
  auto load = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  return type && type.getShape().size() == 1 && load && !load.getSnapshotStorage() &&
         load.getRegion().getType().getShape().size() == 2 &&
         load.getRegion().getType().getInterleaveRows() > 0 &&
         type.getAxisIds()[0] == load.getRegion().getType().getAxisIds()[0];
}
}

bool weft::riscv::isContiguousInterleavedScalarRead(
    mlir::Value input, ValueType result, AccessAttr access) {
  llvm::SmallVector<mlir::Value> indices;
  auto field = fieldReadProjection(input, &indices);
  if (!isInterleavedScalarField(field) || !indices.empty() || !result || !access ||
      access.getMapping() != "natural" || access.getBitOffset() % 8 ||
      !sameLogicalDomain(input.getType(), result))
    return false;
  const int64_t bits = elementBitWidth(result.getElementType());
  auto layout = result.getLayout();
  if ((bits != 8 && bits != 16 && bits != 32) || access.getStorageBits() != bits ||
      access.getFieldChunkBytes() != bits / 8 || layout.getCarrier() != "rvv" ||
      layout.getReplicaGroup() != 1 || layout.getLaneFactors()[0] <= 1 ||
      layout.getFragmentFactors()[0] != 1 || layout.getLocalFactors()[0] != 1 ||
      (layout.getValidity() != "full" && layout.getValidity() != "tail"))
    return false;
  auto load = field.getOwner().getDefiningOp<LoadOp>();
  auto slice = load.getRegion().getDefiningOp<SliceOp>();
  auto view = slice ? slice.getBase().getDefiningOp<MemoryViewOp>() : MemoryViewOp();
  auto encoding = mlir::dyn_cast<weft::kernel::EncodingType>(elementOf(load.getResult().getType()));
  auto declaration = encoding ? findEncodingDeclaration(field, baseEncodingFamily(field, encoding)) : EncodingDeclOp();
  if (!view || !mlir::isa<mlir::BlockArgument>(view.getBase()) || !declaration ||
      declaration.getByteOrder() != "little")
    return false;
  const int64_t cohort = load.getRegion().getType().getInterleaveRows();
  auto represented = checkedPositiveProduct({layout.getTimeFactors()[0],
      layout.getReplicaFactors()[0], layout.getLaneFactors()[0]});
  if (!represented)
    return false;
  const int64_t extent = *represented;
  if (extent > cohort || cohort % extent ||
      (result.getShape()[0] > 0 && result.getShape()[0] != extent) ||
      extent % layout.getLaneFactors()[0])
    return false;
  for (mlir::Value index : slice.getIndices()) {
    auto point = index.getDefiningOp<PhysicalPointOp>();
    if (!point || point.getResult().getType().getDomain().getAxisId() != result.getAxisIds()[0])
      continue;
    auto constant = point.getPartition().getDefiningOp<mlir::arith::ConstantOp>();
    auto partition = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()) : mlir::IntegerAttr();
    return partition && partition.getInt() == extent && knownIndexMultipleOf(point.getBase(), extent);
  }
  return false;
}

namespace {
bool isInterleavedIndexedByteRead(mlir::Value input, ValueType result, AccessAttr access) {
  auto extract = input.getDefiningOp<ExtractOp>();
  auto field = fieldReadProjection(input);
  auto encoding = field ? mlir::dyn_cast<weft::kernel::EncodingType>(elementOf(field.getOwner().getType()))
                        : weft::kernel::EncodingType();
  int64_t rows = field ? localPackInterleaveRows(field.getOwner()) : 0;
  if (!rows && encoding) rows = derivedInterleaveRows(field, encoding);
  if (!extract || !field || rows <= 1 || !result || !access ||
      access.getForm() != "indexed" || access.getMapping() != "natural" ||
      access.getFieldChunkBytes() != 1 || access.getBitOffset() % 8 ||
      extract.getIndices().size() != 1 ||
      result.getLayout().getCarrier() != "rvv") return false;
  auto index = mlir::dyn_cast<ValueType>(extract.getIndices()[0].getType());
  int64_t bits = elementBitWidth(result.getElementType());
  auto lanes = checkedPositiveProduct(result.getLayout().getLaneFactors());
  auto indexLanes = index ? checkedPositiveProduct(index.getLayout().getLaneFactors()) : std::nullopt;
  return index && index.getLayout().getCarrier() == "rvv" &&
      lanes && indexLanes && *lanes == *indexLanes &&
      (bits == 8 || bits == 16 || bits == 32) && result.getLayout().getSew() == bits &&
      encodedFieldRawLMULEighths(result).has_value();
}

bool isRecordElementRead(mlir::Value input, ValueType result, AccessAttr access) {
  auto projection = input.getDefiningOp<ExtractOp>();
  auto field = fieldReadProjection(input);
  if (!projection || !field || !result || !access ||
      access.getForm() != "strided" ||
      (access.getMapping() != "natural" && access.getMapping() != "grouped_layered") ||
      projection.getIndices().size() != 1 || !projection.getIndices()[0].getType().isIndex()) return false;
  auto inputType = projection.getInput().getType();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto encoding = mlir::dyn_cast<weft::kernel::EncodingType>(elementOf(field.getOwner().getType()));
  if (!integer || !integer.isUnsigned() ||
      (integer.getWidth() != 1 && integer.getWidth() != 2 && integer.getWidth() != 4 && integer.getWidth() != 8) ||
      !encoding || encoding.getKind() != "base" ||
      projection.getSelectors().size() != inputType.getShape().size() ||
      inputType.getAxisIds().size() != result.getAxisIds().size() + 1 ||
      inputType.getAxisIds().asArrayRef().drop_back() != result.getAxisIds().asArrayRef()) return false;
  for (auto [position, selector] : llvm::enumerate(projection.getSelectors())) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(selector);
    if (!name || name.getValue() != (position + 1 == projection.getSelectors().size() ? "index" : "all")) return false;
  }
  auto layout = result.getLayout();
  if (layout.getCarrier() != "rvv" || layout.getSew() != 8 || layout.getReplicaGroup() != 1 ||
      llvm::count_if(layout.getLaneFactors().asArrayRef(), [](int64_t n) { return n > 1; }) != 1 ||
      llvm::any_of(layout.getFragmentFactors().asArrayRef(), [](int64_t n) { return n != 1; }) ||
      llvm::any_of(layout.getLocalFactors().asArrayRef(), [](int64_t n) { return n != 1; })) return false;
  // Each logical element must be contained in one byte. Wider, joined and
  // cross-byte encodings retain their separate storage-window contracts.
  if (access.getBitOffset() % 8 || access.getStorageBits() % integer.getWidth() ||
      inputType.getShape().asArrayRef().back() != access.getStorageBits() / integer.getWidth()) return false;
  if (access.getMapping() == "grouped_layered")
    return access.getGroupSize() > 0 && access.getLayerSize() > 0 &&
           access.getGroupSize() % access.getLayerSize() == 0 &&
           access.getOrder() == "lo_first";
  return true;
}

bool isRecordGatherRead(mlir::Value input, ValueType result, AccessAttr access) {
  auto extract = input.getDefiningOp<ExtractOp>();
  auto field = extract ? extract.getInput().getDefiningOp<FieldOp>() : FieldOp();
  auto source = field ? mlir::dyn_cast<ValueType>(field.getResult().getType()) : ValueType();
  auto encoding = field ? mlir::dyn_cast<weft::kernel::EncodingType>(elementOf(field.getOwner().getType()))
                        : weft::kernel::EncodingType();
  if (!extract || !source || !encoding || encoding.getKind() != "base" ||
      localPackInterleaveRows(field.getOwner()) || !result || !access ||
      access.getForm() != "strided" || access.getMapping() != "natural" ||
      access.getBitOffset() % 8 || elementBitWidth(result.getElementType()) != 8 ||
      !mlir::isa<mlir::IntegerType>(result.getElementType()) ||
      extract.getIndices().size() != 1 || source.getShape().size() < 2 ||
      result.getLayout().getCarrier() != "rvv") return false;
  auto index = mlir::dyn_cast<ValueType>(extract.getIndices()[0].getType());
  if (!index || index.getLayout().getCarrier() != "scalar" ||
      llvm::any_of(index.getLayout().getLaneFactors().asArrayRef(), [](int64_t n) { return n != 1; }) ||
      extract.getSelectors().size() != source.getShape().size()) return false;
  for (auto [position, selector] : llvm::enumerate(extract.getSelectors()))
    if (mlir::cast<mlir::StringAttr>(selector).getValue() !=
        (position + 1 == source.getShape().size() ? "gather" : "all")) return false;
  auto prefix = source.getAxisIds().asArrayRef().drop_back();
  if (result.getAxisIds().size() != prefix.size() + index.getAxisIds().size() ||
      !llvm::equal(result.getAxisIds().asArrayRef().take_front(prefix.size()), prefix) ||
      !llvm::equal(result.getAxisIds().asArrayRef().drop_front(prefix.size()), index.getAxisIds().asArrayRef())) return false;
  for (size_t i = prefix.size(); i < result.getAxisIds().size(); ++i)
    if (result.getLayout().getLaneFactors()[i] != 1) return false;
  return llvm::count_if(result.getLayout().getLaneFactors().asArrayRef(), [](int64_t n) { return n > 1; }) == 1;
}
}

llvm::StringRef weft::riscv::fieldReadInstruction(
    mlir::Value input, ValueType result, AccessAttr access) {
  if (isInterleavedIndexedByteRead(input, result, access)) return "encoded.field-read.indexed-bytes";
  if (isRecordGatherRead(input, result, access)) return "encoded.field-read.record-gather";
  if (isRecordElementRead(input, result, access)) return "encoded.field-read.record-element";
  if (result && access && result.getLayout().getCarrier() == "rvv" &&
      access.getForm() == "unit" && access.getMapping() == "natural" &&
      isInterleavedScalarField(fieldReadProjection(input)))
    return access.getAlignment() < elementBitWidth(result.getElementType()) / 8
               ? "encoded.field-read.unit-bytes" : "encoded.field-read.unit";
  return "encoded.field-read";
}

std::optional<int64_t> weft::riscv::fieldReadTemporaryGroups(
    mlir::Value input, ValueType result, AccessAttr access, mlir::ValueRange indices) {
  if (result && result.getLayout().getCarrier() == "scalar")
    return 0;
  auto rawLMUL = encodedFieldRawLMULEighths(result);
  if (!rawLMUL || !access)
    return std::nullopt;
  const int64_t rawGroups = (*rawLMUL + 7) / 8;
  const int64_t valueGroups = (result.getLayout().getLmulEighths() + 7) / 8;
  int64_t indexGroups = 0;
  llvm::DenseSet<mlir::Value> seenIndices;
  for (mlir::Value index : indices)
    if (auto value = mlir::dyn_cast<ValueType>(index.getType());
        value && value.getLayout().getCarrier() == "rvv" && seenIndices.insert(index).second) {
      if (!llvm::is_contained(llvm::ArrayRef<int64_t>({1, 2, 4, 8, 16, 32, 64}),
                              value.getLayout().getLmulEighths()))
        return std::nullopt;
      indexGroups = std::max(indexGroups, (value.getLayout().getLmulEighths() + 7) / 8);
    }
  int64_t localPeak = 0;
  if (isInterleavedIndexedByteRead(input, result, access)) {
    localPeak = indexGroups + 3 * valueGroups;
  } else if (access.getMapping() == "joined") {
    // Head/low/high may coexist for record-strided joined storage; each
    // shift/mask reuses its own byte carrier. Interleaved storage needs no more.
    localPeak = 3 * rawGroups;
  } else if (access.getMapping() == "grouped_layered") {
    // Quotient/remainder, byte offset and layer offset use index-width carriers;
    // narrowing the shift and decoding the byte retain one additional carrier.
    localPeak = std::max(3 * indexGroups, 2 * rawGroups);
  } else if (fieldReadInstruction(input, result, access) != "encoded.field-read") {
    localPeak = valueGroups;
  } else if (access.getMapping() == "natural") {
    localPeak = elementBitWidth(result.getElementType()) > 8
                    ? std::max(2 * valueGroups, indexGroups + valueGroups)
                    : valueGroups;
    if (access.getAlignment() < elementBitWidth(result.getElementType()) / 8)
      localPeak = std::max(localPeak, 4 * valueGroups);
  } else {
    return std::nullopt;
  }
  auto streams = checkedPositiveProduct(result.getLayout().getTimeFactors().asArrayRef());
  auto outputs = streams ? checkedPositiveProduct(
      {result.getLayout().getRegisterGroups(), *streams}) : std::nullopt;
  if (!outputs || *outputs < valueGroups)
    return std::nullopt;
  const int64_t scratch = localPeak;
  const int64_t priorOutputs = *outputs - valueGroups;
  if (priorOutputs > std::numeric_limits<int64_t>::max() - scratch)
    return std::nullopt;
  return priorOutputs + scratch;
}

mlir::LogicalResult FieldReadOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  llvm::SmallVector<mlir::Value> indices;
  auto field = fieldReadProjection(getInput(), &indices);
  auto temporaryGroups = fieldReadTemporaryGroups(getInput(), result, getAccess(), getIndices());
  auto instruction = fieldReadInstruction(getInput(), result, getAccess());
  if (!field || !sameLogicalDomain(input, result) ||
      input.getElementType() != result.getElementType() ||
      (input.getLayout().getCarrier() != "scalar" &&
       input.getLayout().getCarrier() != "local" &&
       input.getLayout().getCarrier() != "rvv") ||
      (result.getLayout().getCarrier() != "rvv" &&
       result.getLayout().getCarrier() != "scalar") ||
      elementBitWidth(result.getElementType()) == 0 ||
      !exactLeaf(getLeaf(), "transfer", "field-read", instruction,
                 "none", "exact") ||
      getLeaf().getLocalBytes() != 0 || getLeaf().getFragmentGroups() != 0 ||
      !llvm::equal(indices, getIndices()) || !temporaryGroups ||
      getLeaf().getTemporaryGroups() != *temporaryGroups)
    return emitOpError("field read requires a storage projection and a numeric result with closed resources");
  auto access = getAccess();
  auto storage = field.getAccess();
  if (access.getForm() == "unit" && result.getLayout().getCarrier() == "rvv" &&
      isInterleavedScalarField(field) &&
      !isContiguousInterleavedScalarRead(getInput(), result, access))
    return emitOpError("unit interleaved scalar read must remain inside its aligned row cohort");
  if ((access.getForm() != "unit" && access.getForm() != "strided" &&
       access.getForm() != "indexed" && access.getForm() != "segment") ||
      access.getMapping() != storage.getMapping() ||
      access.getAlignment() > storage.getAlignment() ||
      access.getBitOffset() != storage.getBitOffset() ||
      access.getStorageBits() != storage.getStorageBits() ||
      access.getGroupSize() != storage.getGroupSize() ||
      access.getLayerSize() != storage.getLayerSize() ||
      access.getOrder() != storage.getOrder() ||
      access.getJoinFields() != storage.getJoinFields() ||
      access.getJoinLowBits() != storage.getJoinLowBits() ||
      access.getJoinRole() != storage.getJoinRole() ||
      access.getFieldChunkBytes() != storage.getFieldChunkBytes())
    return emitOpError("field read must preserve its encoded storage mapping");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel ||
      (access.getForm() == "indexed" && !kernel.getTarget().getHasIndexedMemory()) ||
      (access.getForm() == "segment" && !kernel.getTarget().getHasSegmentMemory()))
    return emitOpError("field read memory form is unsupported by the target");
  return mlir::success();
}

mlir::LogicalResult ConvertLayoutOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  if (input.getElementType() != result.getElementType() ||
      input.getShape() != result.getShape() || input.getAxisIds() != result.getAxisIds() ||
      input.getLayout() == result.getLayout())
    return emitOpError()
           << "convert_layout changes only a non-identical physical representation; input="
           << input << ", result=" << result
           << ", conversion=" << getConversion();
  llvm::StringRef source = input.getLayout().getCarrier();
  llvm::StringRef target = result.getLayout().getCarrier();
  llvm::StringRef kind = getConversion().getKind();
  llvm::StringRef effect = getConversion().getEffect();
  if ((input.getLayout().getReplicaGroup() > 1 || result.getLayout().getReplicaGroup() > 1) &&
      kind != "lane_repack")
    return emitOpError("replica-grouped conversion requires an explicit ordered lane repack");
  if ((source == "scalar" && target == "rvv" && kind != "splat" &&
       kind != "register_to_lane" && kind != "time_to_lane") ||
      (source == "rvv" && target == "scalar" && kind != "extract") ||
      (source == "local" && target != "local" && kind != "local_load") ||
      (source != "local" && target == "local" && kind != "local_store") ||
      (source == target && kind != "reshape" && kind != "tuple" &&
       kind != "register_to_lane" && kind != "time_to_lane" &&
       kind != "lane_to_register" && kind != "lane_repack") ||
      (kind == "local_load" && effect != "read") ||
      (kind == "local_store" && effect != "write") ||
      (kind != "local_load" && kind != "local_store" && effect != "pure") ||
      kind == "rvv_to_fragment" || kind == "fragment_to_rvv")
    return emitOpError() << "convert_layout kind '" << kind
                         << "' does not match " << source << " -> " << target;
  if (kind == "lane_repack") {
    auto kernel = getOperation()->getParentOfType<KernelOp>();
    if (!kernel || !rvvLayoutRepackSlices(input, result) ||
        getConversion().getTemporaryGroups() != rvvLayoutRepackTemporaryGroups(kernel.getTarget(), input, result))
      return emitOpError("lane repack requires a bounded complete ordered slice relation and its temporary resources");
  }
  if (source == "scalar" && target == "rvv" &&
      kind == "register_to_lane") {
    auto sourceLayout = input.getLayout();
    auto targetLayout = result.getLayout();
    auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
    auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    auto sourceFragment = sourceLayout.getFragmentFactors().asArrayRef();
    auto sourceLocal = sourceLayout.getLocalFactors().asArrayRef();
    auto targetTime = targetLayout.getTimeFactors().asArrayRef();
    auto targetLane = targetLayout.getLaneFactors().asArrayRef();
    auto targetReplica = targetLayout.getReplicaFactors().asArrayRef();
    auto targetFragment = targetLayout.getFragmentFactors().asArrayRef();
    auto targetLocal = targetLayout.getLocalFactors().asArrayRef();
    bool hasLaneDimension = false;
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      hasLaneDimension |= targetLane[dimension] > 1;
      if (sourceLane[dimension] != 1 ||
          sourceFragment[dimension] != 1 || sourceLocal[dimension] != 1 ||
          targetFragment[dimension] != 1 || targetLocal[dimension] != 1)
        return emitOpError(
            "scalar register-to-lane conversion only uses scalar time/replica coordinates and RVV lanes");
    }
    if (!hasLaneDimension)
      return emitOpError(
          "scalar register-to-lane conversion requires one non-unit target lane axis");
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (targetLane[dimension] > 1) {
        auto represented = checkedPositiveProduct(
            {targetTime[dimension], targetLane[dimension],
             targetReplica[dimension]});
        if (!represented || sourceTime[dimension] != 1 ||
            sourceReplica[dimension] != *represented)
          return emitOpError(
              "scalar register-to-lane conversion must preserve every complete moved axis");
        continue;
      }
      auto sourceRepresented = checkedPositiveProduct(
          {sourceTime[dimension], sourceReplica[dimension]});
      auto targetRepresented = checkedPositiveProduct(
          {targetTime[dimension], targetReplica[dimension]});
      if (targetLane[dimension] != 1 || !sourceRepresented ||
          !targetRepresented || *sourceRepresented != *targetRepresented)
        return emitOpError(
            "scalar register-to-lane conversion must preserve every non-lane logical axis");
    }
  }
  if (source == "scalar" && target == "rvv" && kind == "time_to_lane") {
    auto sourceLayout = input.getLayout();
    auto targetLayout = result.getLayout();
    auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
    auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    auto sourceFragment = sourceLayout.getFragmentFactors().asArrayRef();
    auto sourceLocal = sourceLayout.getLocalFactors().asArrayRef();
    auto targetTime = targetLayout.getTimeFactors().asArrayRef();
    auto targetLane = targetLayout.getLaneFactors().asArrayRef();
    auto targetReplica = targetLayout.getReplicaFactors().asArrayRef();
    auto targetFragment = targetLayout.getFragmentFactors().asArrayRef();
    auto targetLocal = targetLayout.getLocalFactors().asArrayRef();
    bool hasLaneDimension = false;
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      hasLaneDimension |= targetLane[dimension] > 1;
      if (sourceLane[dimension] != 1 ||
          sourceFragment[dimension] != 1 || sourceLocal[dimension] != 1 ||
          targetFragment[dimension] != 1 || targetLocal[dimension] != 1)
        return emitOpError(
            "scalar time-to-lane conversion only moves issue-time parts into RVV lanes");
    }
    if (!hasLaneDimension)
      return emitOpError(
          "scalar time-to-lane conversion requires one non-unit target lane axis");
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (targetLane[dimension] > 1) {
        auto represented = checkedPositiveProduct(
            {targetTime[dimension], targetLane[dimension],
             targetReplica[dimension]});
        auto sourceRepresented = checkedPositiveProduct(
            {sourceTime[dimension], sourceReplica[dimension]});
        if (!represented || !sourceRepresented ||
            *sourceRepresented != *represented ||
            sourceReplica[dimension] != targetReplica[dimension] ||
            sourceTime[dimension] <= targetTime[dimension])
          return emitOpError(
              "scalar time-to-lane conversion must preserve every complete moved axis");
        continue;
      }
      auto sourceRepresented = checkedPositiveProduct(
          {sourceTime[dimension], sourceReplica[dimension]});
      auto targetRepresented = checkedPositiveProduct(
          {targetTime[dimension], targetReplica[dimension]});
      if (targetLane[dimension] != 1 || !sourceRepresented ||
          !targetRepresented || *sourceRepresented != *targetRepresented)
        return emitOpError(
            "scalar time-to-lane conversion must preserve every non-lane logical axis");
    }
  }
  if (auto access = (*this)->getAttrOfType<AccessAttr>("source_access")) {
    if (effect != "pure" && effect != "read")
      return emitOpError(
          "source_access is only legal on a read-only representation conversion");
    if (access.getForm() != "unit" && access.getForm() != "strided" &&
        access.getForm() != "indexed" && access.getForm() != "segment")
      return emitOpError(
          "source_access must name one exact target memory form");
  }
  if (getLeaf().getEngine() != "unselected") {
    llvm::StringRef expected =
        kind == "splat"          ? "rvv.splat"
        : kind == "extract"      ? "rvv.extract"
        : kind == "local_load"   ? "rvv.local-load"
        : kind == "local_store"  ? "rvv.local-store"
        : kind == "tuple"        ? "rvv.tuple-convert"
        : kind == "register_to_lane" ? "rvv.register-to-lane"
        : kind == "time_to_lane" ? "rvv.time-to-lane"
        : kind == "lane_to_register" ? "rvv.lane-to-register"
        : kind == "lane_repack" ? "rvv.lane-repack"
                                      : "rvv.layout-reshape";
    if (getLeaf().getEngine() != "rvv" ||
        getLeaf().getFamily() != "layout-conversion" ||
        getLeaf().getInstruction() != expected)
      return emitOpError(
          "convert_layout leaf does not implement its typed conversion kind");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult LocalAllocOp::verify() {
  LocalType type = getResult().getType();
  if ((type.getSizeBytes() == 0 || type.getSizeBytes() < -1) ||
      type.getAlignment() <= 0 ||
      type.getPurpose().empty() || type.getSchema().empty())
    return emitOpError("local allocation requires size, alignment, purpose, and schema");
  if (type.getSizeBytes() >= 0) {
    auto constant = getSizeBytes().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!constant || constant.value() != type.getSizeBytes())
      return emitOpError(
          "static local allocation size SSA value must equal its LocalType size");
  }
  return mlir::success();
}

mlir::LogicalResult LocalCapacityGuardOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel || getLimitBytes() <= 0 || getBirthId() < 0 ||
      getLimitBytes() > kernel.getTarget().getMaxPrivateStackBytes() ||
      !exactLeaf(getLeaf(), "scalar", "local-capacity",
                 "scalar.local-capacity-guard", "none", "exact"))
    return emitOpError(
        "dynamic local capacity guard requires one target-bounded scalar trap contract");
  auto next = std::next(getOperation()->getIterator());
  if (next == getOperation()->getBlock()->end())
    return emitOpError("dynamic local capacity guard must immediately precede its allocation");
  auto allocation = mlir::dyn_cast<LocalAllocOp>(&*next);
  if (!allocation || allocation.getSizeBytes() != getSizeBytes() ||
      allocation.getResult().getType().getSizeBytes() != -1 ||
      allocation.getResult().getType().getBirthId() != getBirthId())
    return emitOpError(
        "dynamic local capacity guard must uniquely guard the following allocation");
  return mlir::success();
}

mlir::LogicalResult LocalReuseOp::verify() {
  LocalType previous = getPrevious().getType();
  LocalType result = getResult().getType();
  mlir::Value previousSize;
  if (auto allocation = getPrevious().getDefiningOp<LocalAllocOp>())
    previousSize = allocation.getSizeBytes();
  else if (auto reuse = getPrevious().getDefiningOp<LocalReuseOp>())
    previousSize = reuse.getSizeBytes();
  mlir::Value storage = getPrevious();
  while (auto reuse = storage.getDefiningOp<LocalReuseOp>())
    storage = reuse.getPrevious();
  auto allocation = storage.getDefiningOp<LocalAllocOp>();
  llvm::SmallVector<mlir::Value> capacities;
  if (allocation) capacities.push_back(allocation.getSizeBytes());
  bool covered = false;
  llvm::DenseSet<mlir::Value> visited;
  while (!capacities.empty()) {
    mlir::Value capacity = capacities.pop_back_val();
    if (!visited.insert(capacity).second) continue;
    covered |= capacity == getSizeBytes();
    auto fixedCapacity = capacity.getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto fixedRequest = getSizeBytes().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (fixedCapacity && fixedRequest && fixedRequest.value() >= 0)
      covered |= fixedCapacity.value() >= fixedRequest.value();
    if (auto maximum = capacity.getDefiningOp<mlir::arith::MaxUIOp>()) {
      capacities.push_back(maximum.getLhs());
      capacities.push_back(maximum.getRhs());
    }
  }
  if (!previousSize || getPrevious().getDefiningOp()->getBlock() != getOperation()->getBlock() ||
      !covered || previous.getSizeBytes() != result.getSizeBytes() ||
      previous.getElementType() != result.getElementType() ||
      previous.getAlignment() < result.getAlignment() ||
      previous.getOwnerDomainId() != result.getOwnerDomainId() ||
      previous.getLifetimeEndDomainId() != result.getLifetimeEndDomainId() ||
      !mlir::isa<mlir::IntegerType, mlir::FloatType>(previous.getElementType()) ||
      !mlir::isa<mlir::IntegerType, mlir::FloatType>(result.getElementType()) ||
      !llvm::all_of(getPrevious().getUsers(), [](mlir::Operation *user) {
        return mlir::isa<LocalBindOp, LocalReuseOp>(user);
      }) || !llvm::all_of(getResult().getUsers(), [](mlir::Operation *user) {
        return mlir::isa<LocalBindOp, LocalReuseOp>(user);
      }))
    return emitOpError("local reuse requires a same-scope numeric local binding with proven capacity and sufficient alignment");
  return mlir::success();
}

mlir::LogicalResult DenseSnapshotOp::verify() {
  auto source = getSource().getType();
  auto result = getResult().getType();
  auto storage = getStorage().getType();
  auto encoding = mlir::cast<weft::kernel::EncodingType>(source.getEncoding());
  const int64_t elementBytes = source.getStorageBits() / 8;
  if (encoding.getKind() != "dense" || source.getShape().size() != 1 ||
      source.getShape()[0] <= 0 || source.getStrides()[0] != 1 ||
      source.getStorageBits() % 8 || elementBytes <= 0 ||
      source.getShape()[0] > std::numeric_limits<int64_t>::max() / elementBytes ||
      source.getElements() != 1 || source.getInterleaveRows() != 0 ||
      result.getEncoding() != source.getEncoding() ||
      result.getShape() != source.getShape() ||
      result.getAxisIds() != source.getAxisIds() ||
      result.getStrides() != source.getStrides() || result.getOrigins()[0] != 0 ||
      result.getStorageBits() != source.getStorageBits() ||
      result.getElements() != 1 || result.getInterleaveRows() != 0 ||
      result.getAddressClass() != "local" || result.getAccess() != "read" ||
      result.getAliasSet() != storage.getAliasSet() ||
      result.getAlignment() != storage.getAlignment() ||
      !storage.getElementType().isInteger(8) ||
      !storage.getShape().empty() || !storage.getAxisIds().empty() ||
      storage.getSizeBytes() != source.getShape()[0] * elementBytes ||
      storage.getPurpose() != "pack" || storage.getSchema() != "read-snapshot" ||
      !getStorage().getDefiningOp<LocalAllocOp>() ||
      !exactReadSnapshotLeaf(*this, getLeaf()))
    return emitOpError(
        "dense snapshot requires one complete static contiguous table and exact private byte storage");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult LocalShapeOp::verify() {
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!owner || getId() <= static_cast<int64_t>(owner.getShapeSymbols().size()))
    return emitOpError("local shape identity must not alias a public ABI shape");
  if (auto constant = getExtent().getDefiningOp<mlir::arith::ConstantIndexOp>();
      constant && constant.value() < 0)
    return emitOpError("local shape extent cannot be negative");
  return mlir::success();
}

void LocalShapeOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), kernel::LocalShapeResource::get());
}

mlir::LogicalResult EmptyOp::verify() {
  auto value = getResult().getType();
  if (!isNumericPhysical(value) || value.getShape().size() != getExtents().size() ||
      (value.getLayout().getCarrier() != "unassigned" && value.getLayout().getCarrier() != "local") ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId())
    return emitOpError("empty storage requires explicit extents, private lifetime, and a local carrier");
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!owner) return mlir::failure();
  for (auto [shape, extent] : llvm::zip(value.getShape().asArrayRef(), getExtents()))
    if (shape < -static_cast<int64_t>(owner.getShapeSymbols().size()) &&
        weft::localShapeExtent<LocalShapeOp>(*this, -shape) != extent)
      return emitOpError("empty storage lost its dominating local shape definition");
  return mlir::success();
}

mlir::LogicalResult LocalSizeOp::verify() {
  if (getElementBytes() <= 0 ||
      !exactLeaf(getLeaf(), "scalar", "local-size", "local.size.checked", "none", "exact"))
    return emitOpError("local size requires a positive element width and checked size leaf");
  return mlir::success();
}

mlir::LogicalResult LocalBindOp::verify() {
  LocalType storage = getStorage().getType();
  ValueType result = getResult().getType();
  if (result.getLayout().getCarrier() != "local" ||
      storage.getElementType() != result.getElementType() ||
      storage.getShape() != result.getShape() ||
      storage.getAxisIds() != result.getAxisIds() ||
      getExtents().size() != result.getShape().size())
    return emitOpError(
        "local bind must expose exactly the logical domain owned by its storage");
  if (storage.getSizeBytes() >= 0) {
    auto elements = getElements().getDefiningOp<mlir::arith::ConstantIndexOp>();
    int64_t expectedElements = 1;
    bool symbolic = false;
    for (int64_t extent : result.getShape().asArrayRef()) {
      if (extent < 0) {
        symbolic = true;
        continue;
      }
      if (extent == 0)
        return emitOpError("static local storage has a zero logical extent");
      if (expectedElements > std::numeric_limits<int64_t>::max() / extent)
        return emitOpError("static local storage element count overflows");
      expectedElements *= extent;
    }
    if (symbolic) {
      auto physicalElements = checkedPositiveProduct(
          result.getLayout().getLocalFactors().asArrayRef());
      if (!physicalElements)
        return emitOpError(
            "symbolic local binding requires one fixed local-coordinate extent");
      expectedElements = *physicalElements;
    }
    if (!elements || elements.value() != expectedElements)
      return emitOpError(
          "static local binding element count must equal its represented physical shape");
    const unsigned bits = std::max(8u, elementBitWidth(result.getElementType()));
    if (!bits || bits % 8 ||
        expectedElements >
            std::numeric_limits<int64_t>::max() / (bits / 8) ||
        storage.getSizeBytes() != expectedElements * (bits / 8))
      return emitOpError(
          "static local binding byte size must exactly preserve its physical elements")
          << "; storage has " << storage.getSizeBytes() << " bytes for "
          << expectedElements << " elements of " << result.getElementType()
          << " (" << bits << " storage bits each)";
  }
  return verifyLocalBindingStorage(*this);
}

mlir::LogicalResult LocalViewOp::verify() {
  auto source = getSource().getType();
  auto result = getResult().getType();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(result.getEncoding());
  std::string family;
  if (source.getElementType().isIndex()) family = "i64";
  else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(source.getElementType()))
    family = (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
  else { llvm::raw_string_ostream stream(family); source.getElementType().print(stream); }
  const int64_t rank = source.getShape().size();
  if (source.getLayout().getCarrier() != "local" ||
      source.getShape() != result.getShape() || source.getAxisIds() != result.getAxisIds() ||
      getExtents().size() != rank || getStrides().size() != rank ||
      !encoding || encoding.getKind() != "dense" || encoding.getFamily() != family ||
      result.getLayoutIdentity() != "dense." + family ||
      result.getAddressClass() != "local" || result.getAccess() != "readwrite" ||
      result.getAliasSet() != -1 || result.getElements() != 1 || result.getInterleaveRows() != 0 ||
      result.getStorageBits() != std::max(8u, elementBitWidth(source.getElementType())) ||
      result.getAlignment() != result.getStorageBits() / 8 ||
      llvm::any_of(result.getOrigins().asArrayRef(), [](int64_t origin) { return origin != 0; }))
    return emitOpError("local view must preserve one byte-addressable dense local state domain");
  auto constant = [](mlir::Value value) -> std::optional<int64_t> {
    if (auto op = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) return op.value();
    return std::nullopt;
  };
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  int64_t staticStride = 1;
  for (int64_t axis = rank - 1; axis >= 0; --axis) {
    int64_t extent = source.getShape()[axis];
    if (extent > 0) {
      if (constant(getExtents()[axis]) != extent)
        return emitOpError("local view static extent does not match its SSA binding");
    } else {
      if (!kernel || extent >= 0 || extent == std::numeric_limits<int64_t>::min())
        return emitOpError("local view dynamic extent has no valid shape identity");
      if (extent < -int64_t(kernel.getShapeSymbols().size())) {
        if (weft::localShapeExtent<LocalShapeOp>(*this, -extent) != getExtents()[axis])
          return emitOpError("local view dynamic extent must use its dominating local shape binding");
      } else {
        auto symbol = getExtents()[axis].getDefiningOp<SymbolOp>();
        if (!symbol || symbol.getName() != mlir::cast<mlir::StringAttr>(kernel.getShapeSymbols()[-extent - 1]).getValue())
          return emitOpError("local view dynamic extent must name its shape symbol, independently of axis identity");
      }
    }
    if (result.getStrides()[axis] != staticStride)
      return emitOpError("local view must expose contiguous row-major storage strides");
    if (staticStride) {
      if (constant(getStrides()[axis]) != staticStride)
        return emitOpError("local view static stride does not match its SSA binding");
    } else {
      auto multiply = getStrides()[axis].getDefiningOp<mlir::arith::MulIOp>();
      bool product = multiply &&
          ((multiply.getLhs() == getStrides()[axis + 1] && multiply.getRhs() == getExtents()[axis + 1]) ||
           (multiply.getRhs() == getStrides()[axis + 1] && multiply.getLhs() == getExtents()[axis + 1]));
      product |= constant(getStrides()[axis + 1]) == 1 && getStrides()[axis] == getExtents()[axis + 1];
      if (!product) return emitOpError("local view dynamic stride must preserve its trailing extent product");
    }
    if (staticStride && extent > 0) {
      if (extent > std::numeric_limits<int64_t>::max() / staticStride)
        return emitOpError("local view static stride overflows index storage");
      staticStride *= extent;
    } else staticStride = 0;
  }
  return mlir::success();
}

mlir::LogicalResult LocalLoadOp::verify() {
  if (getSource().getType().getLayout().getCarrier() != "local" ||
      elementOf(getResult().getType()) != getSource().getType().getElementType() ||
      !isScalar(getResult().getType()) ||
      !exactLeaf(getLeaf(), "transfer", "local-load", "local.load.element",
                 "none", "exact"))
    return emitOpError("local load reads one scalar element from a local value");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult LocalStoreOp::verify() {
  if (getDestination().getType().getLayout().getCarrier() != "local" ||
      elementOf(getValue().getType()) !=
          getDestination().getType().getElementType() ||
      !isScalar(getValue().getType()) ||
      !exactLeaf(getLeaf(), "transfer", "local-store",
                 "local.store.element", "none", "exact"))
    return emitOpError("local store writes one scalar element into a local value");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVLocalMaterializeOp::verify() {
  ValueType input = getInput().getType();
  ValueType destination = getDestination().getType();
  auto localElements = checkedPositiveProduct(
      destination.getLayout().getLocalFactors().asArrayRef());
  const unsigned bits = elementBitWidth(input.getElementType());
  int64_t laneDimensions = 0;
  bool mappingClosed = input.getShape().size() == destination.getShape().size();
  for (size_t dimension = 0; mappingClosed && dimension < input.getShape().size();
       ++dimension) {
    const int64_t lane = input.getLayout().getLaneFactors()[dimension];
    const int64_t replica = input.getLayout().getReplicaFactors()[dimension];
    const int64_t local = destination.getLayout().getLocalFactors()[dimension];
    if (lane > 1) {
      ++laneDimensions;
      mappingClosed &= dimension + 1 == input.getShape().size() && lane == local &&
                       replica == 1;
    } else {
      mappingClosed &= replica == local;
    }
    mappingClosed &= input.getLayout().getTimeFactors()[dimension] == 1 &&
                     input.getLayout().getFragmentFactors()[dimension] == 1 &&
                     input.getLayout().getLocalFactors()[dimension] == 1;
  }
  const int64_t localBytes =
      localElements && bits
          ? (*localElements * static_cast<int64_t>(bits) + 7) / 8
          : -1;
  if (input.getLayout().getCarrier() != "rvv" ||
      destination.getLayout().getCarrier() != "local" ||
      !sameLogicalDomain(input, destination) ||
      input.getElementType() != destination.getElementType() ||
      laneDimensions != 1 || !mappingClosed || localBytes <= 0 ||
      !getDestination().getDefiningOp<LocalBindOp>() ||
      !exactLeaf(getLeaf(), "transfer", "local-materialize",
                 "rvv.local-materialize", "none", "exact") ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != 0 || getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 ||
      getLeaf().getLocalBytes() != localBytes)
    return emitOpError(
        "RVV local materialize requires one complete innermost-lane value and an exact local representation");
  return mlir::success();
}

mlir::LogicalResult EncodedLocalPackOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  LocalType storage = getStorage().getType();
  LocalPackPlanAttr plan = getPlan();
  LayoutAttr transfer = getTransferLayout();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(input.getElementType());
  auto row = getRowPoint().getType().getDomain();
  auto record = getRecordPoint().getType().getDomain();
  auto rowPosition = llvm::find(result.getAxisIds().asArrayRef(), plan.getRowAxis());
  auto recordPosition =
      llvm::find(result.getAxisIds().asArrayRef(), plan.getRecordAxis());
  auto representedExtent = [&](decltype(rowPosition) position)
      -> std::optional<int64_t> {
    if (position == result.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t axis = static_cast<size_t>(
        position - result.getAxisIds().asArrayRef().begin());
    int64_t extent = 1;
    for (mlir::DenseI64ArrayAttr factors : {
             result.getLayout().getTimeFactors(),
             result.getLayout().getLaneFactors(),
             result.getLayout().getReplicaFactors(),
             result.getLayout().getFragmentFactors(),
             result.getLayout().getLocalFactors()}) {
      const int64_t factor = factors[axis];
      if (factor <= 0 ||
          extent > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      extent *= factor;
    }
    return extent;
  };
  auto rowExtent = representedExtent(rowPosition);
  auto recordExtent = representedExtent(recordPosition);
  std::optional<int64_t> expectedLocalBytes;
  if (rowExtent && recordExtent && *recordExtent % plan.getRecordElements() == 0) {
    auto records = checkedPositiveProduct(
        {*rowExtent, *recordExtent / plan.getRecordElements()});
    if (records)
      expectedLocalBytes =
          checkedPositiveProduct({*records, plan.getRecordBytes()});
  }
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const bool transferLayout =
      kernel && transfer && transfer.getCarrier() == "rvv" &&
      transfer.getAxisIds().asArrayRef() ==
          llvm::ArrayRef<int64_t>({plan.getRowAxis()}) &&
      transfer.getTimeFactors().asArrayRef() == llvm::ArrayRef<int64_t>({1}) &&
      transfer.getLaneFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({plan.getInterleaveRows()}) &&
      transfer.getReplicaFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getFragmentFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getLocalFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getSew() == 8 && transfer.getLmulEighths() == 8 &&
      transfer.getVl() == plan.getInterleaveRows() &&
      transfer.getRegisterGroups() == 1 &&
      supportsRVVLayout(kernel.getTarget(), transfer);
  if (!encoding || encoding.getKind() != "base" ||
      !sameLogicalDomain(input, result) ||
      input.getElementType() != result.getElementType() ||
      result.getLayout().getCarrier() != "local" ||
      storage.getPurpose() != "pack" ||
      storage.getElementType() != result.getElementType() ||
      storage.getShape() != result.getShape() ||
      storage.getAxisIds() != result.getAxisIds() ||
      storage.getSizeBytes() != plan.getLocalBytes() ||
      !expectedLocalBytes || *expectedLocalBytes != plan.getLocalBytes() ||
      row.getAxisId() != plan.getRowAxis() ||
      record.getAxisId() != plan.getRecordAxis() ||
      !rowExtent || !recordExtent || *rowExtent % plan.getInterleaveRows() ||
      *recordExtent % plan.getRecordElements() ||
      !getStorage().getDefiningOp<LocalAllocOp>() ||
      !transferLayout ||
      !exactLeaf(getLeaf(), "transfer", "local-pack",
                 "rvv.local-pack.interleave", "none", "exact") ||
      getLeaf().getOperandGroups() != 0 || getLeaf().getResultGroups() != 0 ||
      getLeaf().getTemporaryGroups() != transfer.getRegisterGroups() ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError(
        "encoded local pack requires one allocated, lifetime-owned interleaved record view");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult IndexMultipleGuardOp::verify() {
  if (getMultiple() <= 0 ||
      !exactLeaf(getLeaf(), "scalar", "index-guard",
                 "scalar.index-multiple-guard", "none", "exact") ||
      getLeaf().getOperandGroups() != 0 || getLeaf().getResultGroups() != 0 ||
      getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError(
        "index multiple guard requires one positive exact scalar divisibility contract");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult EncodedLocalBindOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  LocalType storage = getStorage().getType();
  LocalPackPlanAttr plan = getPlan();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(input.getElementType());
  auto representedExtent = [&](ValueType value,
                               int64_t axis) -> std::optional<int64_t> {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
    if (found == value.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t position = static_cast<size_t>(
        found - value.getAxisIds().asArrayRef().begin());
    int64_t extent = 1;
    for (mlir::DenseI64ArrayAttr factors : {
             value.getLayout().getTimeFactors(),
             value.getLayout().getLaneFactors(),
             value.getLayout().getReplicaFactors(),
             value.getLayout().getFragmentFactors(),
             value.getLayout().getLocalFactors()}) {
      const int64_t factor = factors[position];
      if (factor <= 0 ||
          extent > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      extent *= factor;
    }
    return extent;
  };
  auto rowExtent = representedExtent(result, plan.getRowAxis());
  auto recordExtent = representedExtent(result, plan.getRecordAxis());
  if (!encoding || encoding.getKind() != "base" ||
      !sameLogicalDomain(input, result) ||
      input.getElementType() != result.getElementType() ||
      result.getLayout().getCarrier() != "local" ||
      storage.getPurpose() != "pack" ||
      storage.getElementType() != result.getElementType() ||
      storage.getShape() != result.getShape() ||
      storage.getAxisIds() != result.getAxisIds() ||
      storage.getSizeBytes() != plan.getLocalBytes() ||
      !rowExtent || !recordExtent || *rowExtent <= 0 || *recordExtent <= 0 ||
      *rowExtent % plan.getInterleaveRows() ||
      *recordExtent % plan.getRecordElements() ||
      !getStorage().getDefiningOp<LocalAllocOp>())
    return emitOpError(
        "encoded local bind requires one complete allocated interleaved record view");
  return mlir::success();
}

mlir::LogicalResult RVVEncodedLocalPackTransferOp::verify() {
  ValueType input = getInput().getType();
  LocalType storage = getStorage().getType();
  LocalPackPlanAttr plan = getPlan();
  LayoutAttr transfer = getTransferLayout();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(input.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const bool transferLayout =
      kernel && transfer && transfer.getCarrier() == "rvv" &&
      transfer.getAxisIds().asArrayRef() ==
          llvm::ArrayRef<int64_t>({plan.getRowAxis()}) &&
      transfer.getTimeFactors().asArrayRef() == llvm::ArrayRef<int64_t>({1}) &&
      transfer.getLaneFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({plan.getInterleaveRows()}) &&
      transfer.getReplicaFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getFragmentFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getLocalFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getSew() == 8 && transfer.getLmulEighths() == 8 &&
      transfer.getVl() == plan.getInterleaveRows() &&
      transfer.getRegisterGroups() == 1 &&
      supportsRVVLayout(kernel.getTarget(), transfer);
  if (!encoding || encoding.getKind() != "base" ||
      storage.getPurpose() != "pack" ||
      storage.getElementType() != input.getElementType() ||
      storage.getShape() != input.getShape() ||
      storage.getAxisIds() != input.getAxisIds() ||
      storage.getSizeBytes() != plan.getLocalBytes() ||
      !getStorage().getDefiningOp<LocalAllocOp>() || !transferLayout ||
      !exactLeaf(getLeaf(), "transfer", "local-pack-transfer",
                 "rvv.local-pack.transfer.interleave", "none", "exact") ||
      getLeaf().getOperandGroups() != 0 || getLeaf().getResultGroups() != 0 ||
      getLeaf().getTemporaryGroups() != transfer.getRegisterGroups() ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError(
        "encoded local pack transfer requires one exact interleaved byte-window copy");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult SpillOp::verify() {
  if (fieldReadProjection(getInput()))
    return emitOpError("spill requires a numeric SSA value, not an encoded storage projection");
  auto value = mlir::dyn_cast<ValueType>(getInput().getType());
  LocalType slot = getSlot().getType();
  if (!value || slot.getPurpose() != "spill" ||
      value.getElementType() != slot.getElementType() ||
      value.getShape() != slot.getShape() || value.getAxisIds() != slot.getAxisIds() ||
      getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError("spill slot must exactly preserve its physical value domain");
  if (value.getLayout().getCarrier() == "scalar") {
    auto timeParts =
        checkedProduct(value.getLayout().getTimeFactors().asArrayRef());
    auto replicaParts =
        checkedProduct(value.getLayout().getReplicaFactors().asArrayRef());
    const unsigned bits = elementBitWidth(value.getElementType());
    if (!exactLeaf(getLeaf(), "transfer", "spill", "scalar.tuple-spill",
                   "none", "exact") ||
        !timeParts || *timeParts != 1 || !replicaParts || *replicaParts <= 0 ||
        !bits || bits % 8 ||
        *replicaParts >
            std::numeric_limits<int64_t>::max() / (bits / 8) ||
        slot.getSizeBytes() != *replicaParts * (bits / 8))
      return emitOpError(
          "scalar tuple spill requires one exact compact local representation");
    return verifyLeafOperation(*this);
  }
  if (!exactLeaf(getLeaf(), "transfer", "spill", "rvv.spill", "none",
                 "exact"))
    return emitOpError("non-scalar spill requires the exact RVV spill leaf");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto streamParts =
      checkedProduct(value.getLayout().getTimeFactors().asArrayRef());
  const int64_t groups = value.getLayout().getRegisterGroups();
  const int64_t vlenBytes = kernel ? kernel.getTarget().getVlenBits() / 8 : 0;
  if (!kernel || !streamParts || *streamParts <= 0 || groups <= 0 ||
      vlenBytes <= 0 || groups > std::numeric_limits<int64_t>::max() / *streamParts ||
      groups * *streamParts >
          std::numeric_limits<int64_t>::max() / vlenBytes ||
      slot.getSizeBytes() != groups * *streamParts * vlenBytes)
    return emitOpError(
        "spill slot size must preserve every selected RVV time part");
  return verifyLeafOperation(*this);
}
mlir::LogicalResult ReloadOp::verify() {
  auto value = mlir::dyn_cast<ValueType>(getResult().getType());
  LocalType slot = getSlot().getType();
  if (!value || slot.getPurpose() != "spill" ||
      value.getElementType() != slot.getElementType() ||
      value.getShape() != slot.getShape() || value.getAxisIds() != slot.getAxisIds() ||
      !exactLeaf(getLeaf(), "transfer", "reload", "rvv.reload", "none",
                 "exact") ||
      getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError("reload must restore the exact spilled logical domain");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto streamParts =
      checkedProduct(value.getLayout().getTimeFactors().asArrayRef());
  const int64_t groups = value.getLayout().getRegisterGroups();
  const int64_t vlenBytes = kernel ? kernel.getTarget().getVlenBits() / 8 : 0;
  if (!kernel || !streamParts || *streamParts <= 0 || groups <= 0 ||
      vlenBytes <= 0 || groups > std::numeric_limits<int64_t>::max() / *streamParts ||
      groups * *streamParts >
          std::numeric_limits<int64_t>::max() / vlenBytes ||
      slot.getSizeBytes() != groups * *streamParts * vlenBytes)
    return emitOpError(
        "reload slot size must preserve every selected RVV time part");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RegisterMaterializeOp::verify() {
  const bool physicalShare = getRealization() == "physical-share";
  auto value = mlir::dyn_cast<ValueType>(getResult().getType());
  const bool scalarPhysical =
      isScalar(getResult().getType()) ||
      (value && value.getLayout().getCarrier() == "scalar");
  if (!isPhysicalValue(getInput().getType()) ||
      getInput().getType() != getResult().getType() ||
      (physicalShare
           ? (!scalarPhysical || getOwnerDomainId() != -1 || getBirthId() != -1 ||
              getLifetimeEndDomainId() != -1)
           : (getOwnerDomainId() < 0 || getBirthId() < 0 ||
              getLifetimeEndDomainId() < getOwnerDomainId())) ||
      (!physicalShare && getRealization() != "share" &&
       getRealization() != "reload" &&
       getRealization() != "rematerialize"))
    return emitOpError("register materialize requires stable value identity and lifetime");
  return mlir::success();
}

mlir::LogicalResult RVVBitplaneMergeOp::verify() {
  ValueType low = getLow().getType();
  ValueType plane = getPlane().getType();
  ValueType result = getResult().getType();
  auto lowElement = mlir::dyn_cast<mlir::IntegerType>(low.getElementType());
  auto planeElement =
      mlir::dyn_cast<mlir::IntegerType>(plane.getElementType());
  auto window = getPlane().getDefiningOp<RVVBitmaskWindowLoadOp>();
  auto field = window ? window.getField().getDefiningOp<FieldOp>()
                      : getPlane().getDefiningOp<FieldOp>();
  const bool logicalPlane =
      planeElement && !planeElement.isSigned() && planeElement.getWidth() == 1 &&
      plane.getShape() == low.getShape() && plane.getAxisIds() == low.getAxisIds() &&
      field && field.getAccess().getForm() == "indexed" &&
      field.getAccess().getMapping() == "grouped_layered" &&
      field.getAccess().getGroupSize() == 8 &&
      field.getAccess().getLayerSize() == 1 &&
      (!window ||
       (window.getAccess().getMapping() == field.getAccess().getMapping() &&
        window.getAccess().getGroupSize() ==
            field.getAccess().getGroupSize() &&
        window.getAccess().getLayerSize() ==
            field.getAccess().getLayerSize() &&
        window.getAccess().getOrder() == field.getAccess().getOrder() &&
        window.getAccess().getBitOffset() ==
            field.getAccess().getBitOffset() &&
        window.getResult().getType() == plane));
  const bool bytePlane =
      planeElement && !planeElement.isSigned() && planeElement.getWidth() == 8 &&
      plane.getLayout().getCarrier() == "local";
  if (!lowElement || lowElement.isSigned() || lowElement.getWidth() != 8 ||
      (!logicalPlane && !bytePlane) || low != result ||
      low.getLayout().getCarrier() != "rvv" || getInsertBit() <= 0 ||
      getInsertBit() >= 8)
    return emitOpError(
        "RVV bitplane merge requires one unsigned-eight-bit RVV value and one typed logical-bit or local-byte plane");
  if (low.getShape().size() != plane.getShape().size() ||
      low.getAxisIds() != plane.getAxisIds())
    return emitOpError(
        "RVV bitplane merge requires matching logical axes for values and storage plane");
  int64_t bitplaneAxes = 0;
  int64_t bitplaneAxis = 0;
  if (bytePlane) {
    for (auto [axis, lowExtent, planeExtent] :
         llvm::zip(low.getAxisIds().asArrayRef(), low.getShape().asArrayRef(),
                   plane.getShape().asArrayRef())) {
      if (lowExtent == planeExtent)
        continue;
      if (planeExtent <= 0 || lowExtent != planeExtent * 8)
        return emitOpError(
            "RVV bitplane merge plane must pack exactly eight logical values per byte");
      ++bitplaneAxes;
      bitplaneAxis = axis;
    }
    if (bitplaneAxes != 1)
      return emitOpError("RVV bitplane merge requires one byte-packed logical axis");
  }
  int64_t laneAxis = 0;
  int64_t laneAxes = 0;
  for (auto [axis, lane] :
       llvm::zip(low.getLayout().getAxisIds().asArrayRef(),
                 low.getLayout().getLaneFactors().asArrayRef()))
    if (lane > 1) {
      laneAxis = axis;
      ++laneAxes;
    }
  if (logicalPlane) {
    if (laneAxes != 1)
      return emitOpError(
          "RVV logical bitplane merge must map one encoded logical axis to lanes");
    bitplaneAxis = laneAxis;
  }
  auto packed = llvm::find(low.getAxisIds().asArrayRef(), bitplaneAxis);
  const size_t packedPosition = static_cast<size_t>(
      packed - low.getAxisIds().asArrayRef().begin());
  const bool laneMask =
      getLeaf().getInstruction() == "rvv.bitplane-merge.mask" &&
      laneAxes == 1 && laneAxis == bitplaneAxis &&
      low.getLayout().getVl() > 0 && low.getLayout().getVl() % 8 == 0;
  const bool stridedByte =
      getLeaf().getInstruction() == "rvv.bitplane-merge.strided" &&
      laneAxes == 1 && laneAxis != bitplaneAxis &&
      packed != low.getAxisIds().asArrayRef().end() &&
      low.getLayout().getLaneFactors()[packedPosition] == 1 &&
      low.getLayout().getReplicaFactors()[packedPosition] == 1 &&
      low.getLayout().getFragmentFactors()[packedPosition] == 1 &&
      low.getLayout().getLocalFactors()[packedPosition] == 1 &&
      low.getLayout().getTimeFactors()[packedPosition] ==
          low.getShape()[packedPosition];
  bool otherTime = false;
  for (auto [axis, time] :
       llvm::zip(low.getAxisIds().asArrayRef(),
                 low.getLayout().getTimeFactors().asArrayRef()))
    otherTime |= axis != bitplaneAxis && time != 1;
  if (!laneMask && !(stridedByte && !otherTime))
    return emitOpError(
        "RVV bitplane merge layout does not match its selected lane-mask or strided-byte realization");
  if (!field ||
      ((!logicalPlane && field.getAccess().getMapping() != "natural") ||
       (logicalPlane && field.getAccess().getMapping() != "grouped_layered")) ||
      field.getAccess().getBitOffset() % 8)
    return emitOpError(
        "RVV bitplane merge requires one byte-aligned encoded bitplane field");
  if ((!exactLeaf(getLeaf(), "rvv", "bitplane-merge",
                  "rvv.bitplane-merge.mask", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "bitplane-merge",
                  "rvv.bitplane-merge.strided", "none", "agnostic")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({static_cast<int64_t>(getInsertBit())}) ||
      getLeaf().getLocalBytes() != 0)
    return emitOpError("RVV bitplane merge has no exact selected leaf");
  return mlir::success();
}

mlir::LogicalResult PackedPlaneMergeOp::verify() {
  ValueType low = getLowField().getType();
  ValueType high = getHighField().getType();
  ValueType result = getResult().getType();
  auto lowElement = mlir::dyn_cast<mlir::IntegerType>(low.getElementType());
  auto highElement = mlir::dyn_cast<mlir::IntegerType>(high.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto lowField = getLowField().getDefiningOp<FieldOp>();
  auto highField = getHighField().getDefiningOp<FieldOp>();
  auto plan = getPlan();
  auto axis = llvm::find(result.getAxisIds().asArrayRef(), plan.getLogicalAxis());
  if (axis == result.getAxisIds().asArrayRef().end())
    return emitOpError("packed plane merge result has no planned logical axis");
  const size_t axisPosition = static_cast<size_t>(
      axis - result.getAxisIds().asArrayRef().begin());
  if (!lowElement || lowElement.isSigned() || !highElement ||
      highElement.isSigned() || !resultElement || resultElement.isSigned() ||
      resultElement.getWidth() != 8 || !lowField || !highField ||
      lowField.getOwner() != highField.getOwner() || low.getShape() != result.getShape() ||
      high.getShape() != result.getShape() || low.getAxisIds() != result.getAxisIds() ||
      high.getAxisIds() != result.getAxisIds() ||
      lowElement.getWidth() != plan.getLowBits() ||
      highElement.getWidth() != plan.getHighBits() ||
      lowField.getAccess().getMapping() != "grouped_layered" ||
      highField.getAccess().getMapping() != "grouped_layered" ||
      lowField.getAccess().getOrder() != "lo_first" ||
      highField.getAccess().getOrder() != "lo_first" ||
      lowField.getAccess().getGroupSize() != plan.getGroupSize() ||
      highField.getAccess().getGroupSize() != plan.getGroupSize() ||
      lowField.getAccess().getLayerSize() != plan.getLowLayerBytes() ||
      highField.getAccess().getLayerSize() != plan.getHighLayerBytes() ||
      lowField.getAccess().getBitOffset() / 8 != plan.getLowByteOffset() ||
      highField.getAccess().getBitOffset() / 8 != plan.getHighByteOffset() ||
      lowField.getAccess().getBitOffset() % 8 ||
      highField.getAccess().getBitOffset() % 8 ||
      result.getLayout().getCarrier() != "scalar" ||
      result.getShape()[axisPosition] != plan.getLogicalElements() ||
      result.getLayout().getTimeFactors()[axisPosition] != 1 ||
      result.getLayout().getLaneFactors()[axisPosition] != 1 ||
      result.getLayout().getReplicaFactors()[axisPosition] !=
          plan.getLogicalElements())
    return emitOpError(
        "packed plane merge requires two byte-aligned grouped/layered fields and one scalar logical tuple");
  for (size_t index = 0; index < result.getShape().size(); ++index)
    if (index != axisPosition &&
        (result.getLayout().getTimeFactors()[index] != 1 ||
         result.getLayout().getLaneFactors()[index] != 1 ||
         result.getLayout().getFragmentFactors()[index] != 1 ||
         result.getLayout().getLocalFactors()[index] != 1))
      return emitOpError(
          "packed plane merge may only preserve non-scale axes as scalar register replicas");
  const bool scalarWords =
      exactLeaf(getLeaf(), "scalar", "packed-plane-merge",
                "scalar.packed-plane-merge.words", "none", "exact") &&
      getLeaf().getOperandGroups() == 0 && getLeaf().getResultGroups() == 0 &&
      getLeaf().getTemporaryGroups() == 0 && getLeaf().getFragmentGroups() == 0 &&
      getLeaf().getLocalBytes() == 0;
  if (!scalarWords ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({plan.getLogicalAxis(),
                                   plan.getLogicalElements(),
                                   plan.getLowBits(), plan.getHighBits(),
                                   plan.getInsertBit()}))
    return emitOpError("packed plane merge has no exact selected leaf; leaf=")
           << getLeaf() << ", plan=" << plan;
  return mlir::success();
}

mlir::LogicalResult RVVBitmaskDecodeOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto origin = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto point = getPoint().getDefiningOp<PhysicalPointOp>();
  const int64_t axis = getPoint().getType().getDomain().getAxisId();
  auto position = llvm::find(result.getAxisIds().asArrayRef(), axis);
  const bool hasAxis = position != result.getAxisIds().asArrayRef().end();
  const size_t axisPosition = hasAxis
                                  ? static_cast<size_t>(
                                        position -
                                        result.getAxisIds().asArrayRef().begin())
                                  : 0;
  const int64_t extent = hasAxis ? result.getShape()[axisPosition] : 0;
  auto partition = point
                       ? point.getPartition().getDefiningOp<
                             mlir::arith::ConstantIndexOp>()
                       : mlir::arith::ConstantIndexOp();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  bool compatibleShape = field.getAxisIds() == result.getAxisIds() &&
                         field.getShape().size() == result.getShape().size();
  if (compatibleShape)
    for (size_t index = 0; index < field.getShape().size(); ++index) {
      if (index == axisPosition)
        compatibleShape &= field.getShape()[index] >= extent;
      else
        compatibleShape &= field.getShape()[index] == result.getShape()[index];
    }
  bool onlyPointAxisIsStreamed = hasAxis;
  if (onlyPointAxisIsStreamed)
    for (size_t index = 0; index < result.getShape().size(); ++index)
      if (index != axisPosition)
        onlyPointAxisIsStreamed &=
            result.getLayout().getTimeFactors()[index] == 1 &&
            result.getLayout().getLaneFactors()[index] == 1;

  bool ownerCoversOriginRecord =
      sourceField &&
      static_cast<bool>(sourceField.getOwner().getDefiningOp<LoadOp>());
  if (sourceField && origin) {
    if (auto owner = sourceField.getOwner().getDefiningOp<ExtractOp>()) {
      auto ownerInput = mlir::dyn_cast<ValueType>(owner.getInput().getType());
      size_t cursor = 0;
      for (auto [position, selectorAttribute] :
           llvm::enumerate(owner.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= owner.getIndices().size())
          break;
        mlir::Value index = owner.getIndices()[cursor++];
        if (ownerInput && position < ownerInput.getAxisIds().size() &&
            ownerInput.getAxisIds()[position] == axis && selector == "domain" &&
            index == getOrigin())
          ownerCoversOriginRecord = true;
      }
    }
  }
  if (!fieldElement || fieldElement.isSigned() || fieldElement.getWidth() != 1 ||
      !resultElement || resultElement.isSigned() ||
      resultElement.getWidth() != 1 || !sourceField || !origin || !point ||
      !partition || !hasAxis || extent <= 0 || partition.value() != extent ||
      point.getParent() != getOrigin() ||
      point.getResult().getType().getDomain().getAxisId() !=
          origin.getResult().getType().getDomain().getAxisId() ||
      !ownerCoversOriginRecord || !compatibleShape ||
      !onlyPointAxisIsStreamed ||
      getAccess() != sourceField.getAccess() ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" ||
      getAccess().getGroupSize() <= 0 || getAccess().getLayerSize() <= 0 ||
      getAccess().getGroupSize() != getAccess().getLayerSize() * 8 ||
      getAccess().getOrder() != "lo_first" ||
      getAccess().getBitOffset() % 8 ||
      extent % getAccess().getGroupSize() ||
      result.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getLaneFactors()[axisPosition] %
          getAccess().getGroupSize() ||
      result.getLayout().getTimeFactors()[axisPosition] *
              result.getLayout().getLaneFactors()[axisPosition] !=
          extent ||
      (validity != "full" && validity != "tail") ||
      (!exactLeaf(getLeaf(), "rvv", "bitmask-decode", "rvv.bitmask-decode",
                  "none", tail) &&
       !exactLeaf(getLeaf(), "rvv", "bitmask-decode", "rvv.bitmask-decode-mask",
                  "none", tail)))
    return emitOpError(
               "RVV bitmask decode requires a byte-aligned contiguous logical-u1 field, "
               "a record owner covering the sub-Level origin, an explicitly anchored "
               "sub-Level point, and a complete time/lane result; field=")
           << field << ", result=" << result << ", origin=" << getOrigin().getType()
           << ", point=" << getPoint().getType() << ", access=" << getAccess();
  return mlir::success();
}

mlir::LogicalResult RVVSignedBitmaskReduceOp::verify() {
  ValueType data = getData().getType();
  ValueType mask = getMask().getType();
  auto dataElement = mlir::dyn_cast<mlir::IntegerType>(data.getElementType());
  auto maskElement = mlir::dyn_cast<mlir::IntegerType>(mask.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  auto maskLoad = getMask().getDefiningOp<RVVBitmaskWindowLoadOp>();
  auto position = llvm::find(data.getAxisIds().asArrayRef(), getReductionAxis());
  const bool hasAxis = position != data.getAxisIds().asArrayRef().end();
  const size_t axisPosition =
      hasAxis ? static_cast<size_t>(position - data.getAxisIds().asArrayRef().begin())
              : 0;
  const int64_t extent = hasAxis ? data.getShape()[axisPosition] : 0;
  llvm::StringRef validity = data.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  bool onlyReductionAxisIsStreamed = hasAxis;
  if (onlyReductionAxisIsStreamed)
    for (size_t index = 0; index < data.getShape().size(); ++index)
      if (index != axisPosition)
        onlyReductionAxisIsStreamed &=
            data.getLayout().getTimeFactors()[index] == 1 &&
            data.getLayout().getLaneFactors()[index] == 1;
  if (!dataElement || !dataElement.isSigned() || dataElement.getWidth() != 8 ||
      !maskElement || maskElement.isSigned() || maskElement.getWidth() != 1 ||
      !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 || !hasAxis || extent <= 0 ||
      extent * 128 >= 32768 || mask.getShape() != data.getShape() ||
      mask.getAxisIds() != data.getAxisIds() ||
      mask.getLayout() != data.getLayout() || !onlyReductionAxisIsStreamed ||
      !maskLoad ||
      !exactLeaf(maskLoad.getLeaf(), "rvv", "bitmask-window-load",
                 "rvv.bitmask-window-mask", "none", tail) ||
      data.getLayout().getCarrier() != "rvv" ||
      data.getLayout().getTimeFactors()[axisPosition] *
              data.getLayout().getLaneFactors()[axisPosition] !=
          extent ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "signed-bitmask-reduce",
                 "rvv.signed-bitmask-reduce.i8-i16", "none", tail))
    return emitOpError(
        "RVV signed-bitmask reduction requires one signed-i8 lane value, one materialized logical-u1 mask in the same exact lane carrier, a proven i16-safe reduction extent, and the exact selected leaf");
  return mlir::success();
}

bool weft::riscv::supportsRVVMaskedNegate(TargetAttr target, ValueType data,
                                       ValueType mask) {
  if (!target || !data || !mask || data.getShape() != mask.getShape() ||
      data.getAxisIds() != mask.getAxisIds())
    return false;
  auto element = mlir::dyn_cast<mlir::IntegerType>(data.getElementType());
  auto maskElement = mlir::dyn_cast<mlir::IntegerType>(mask.getElementType());
  auto layout = data.getLayout();
  auto maskLayout = mask.getLayout();
  auto lanes = rvvLaneCount(data);
  if (!element || !element.isSigned() ||
      (element.getWidth() != 8 && element.getWidth() != 16 &&
       element.getWidth() != 32 && element.getWidth() != 64) ||
      !maskElement || !maskElement.isUnsigned() || maskElement.getWidth() != 1 ||
      layout.getSew() != element.getWidth() ||
      !supportsRVVLayout(target, layout) ||
      !supportsRVVLayout(target, maskLayout) || !lanes ||
      *lanes != layout.getVl() ||
      llvm::any_of(layout.getFragmentFactors().asArrayRef(),
                   [](int64_t factor) { return factor != 1; }) ||
      llvm::any_of(layout.getLocalFactors().asArrayRef(),
                   [](int64_t factor) { return factor != 1; }))
    return false;
  return layout.getTimeFactors() == maskLayout.getTimeFactors() &&
         layout.getLaneFactors() == maskLayout.getLaneFactors() &&
         layout.getReplicaFactors() == maskLayout.getReplicaFactors() &&
         layout.getFragmentFactors() == maskLayout.getFragmentFactors() &&
         layout.getLocalFactors() == maskLayout.getLocalFactors() &&
         layout.getVl() == maskLayout.getVl() &&
         layout.getValidity() == maskLayout.getValidity() &&
         (layout.getValidity() == "full" || layout.getValidity() == "tail") &&
         layout.getSew() * maskLayout.getLmulEighths() ==
             maskLayout.getSew() * layout.getLmulEighths();
}

mlir::LogicalResult RVVMaskedNegateOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto maskLoad = getMask().getDefiningOp<RVVBitmaskWindowLoadOp>();
  auto maskDecode = getMask().getDefiningOp<RVVBitmaskDecodeOp>();
  auto data = getData().getType();
  llvm::StringRef tail =
      data.getLayout().getValidity() == "tail" ? "agnostic" : "exact";
  const int64_t temporaryGroups =
      std::max<int64_t>(1, (data.getLayout().getLmulEighths() + 7) / 8);
  if (!kernel ||
      !supportsRVVMaskedNegate(kernel.getTarget(), data, getMask().getType()) ||
      getResult().getType() != data ||
      !((maskLoad && exactLeaf(maskLoad.getLeaf(), "rvv", "bitmask-window-load",
                               "rvv.bitmask-window-mask", "none", tail)) ||
        (maskDecode && exactLeaf(maskDecode.getLeaf(), "rvv", "bitmask-decode",
                                 "rvv.bitmask-decode-mask", "none", tail))) ||
      !exactLeaf(getLeaf(), "rvv", "masked-negate", "rvv.masked-negate",
                 "none", tail) ||
      getLeaf().getTemporaryGroups() != temporaryGroups ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError(
        "RVV masked negate requires signed integer data, one materialized u1 mask with identical pointwise geometry and mask ratio, an unchanged result type, and the exact temporary carrier");
  return mlir::success();
}

mlir::LogicalResult RVVBitmaskWindowLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto baseValue = mlir::dyn_cast<ValueType>(getByteBase().getType());
  mlir::Type baseElement =
      baseValue ? baseValue.getElementType() : getByteBase().getType();
  auto baseInteger = mlir::dyn_cast<mlir::IntegerType>(baseElement);
  bool scalarBase = baseElement.isIndex() ||
                    (baseInteger && !baseInteger.isSigned());
  if (baseValue) {
    auto layout = baseValue.getLayout();
    scalarBase = scalarBase && layout.getCarrier() == "scalar" &&
                 llvm::all_of(layout.getTimeFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLaneFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getReplicaFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getFragmentFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLocalFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; });
  }
  auto sourceAccess = sourceField ? sourceField.getAccess() : AccessAttr();
  auto encoding = sourceField ? mlir::dyn_cast<weft::kernel::EncodingType>(
      elementOf(sourceField.getOwner().getType())) : weft::kernel::EncodingType();
  int64_t rows = sourceField ? localPackInterleaveRows(sourceField.getOwner()) : 0;
  if (!rows && encoding) rows = derivedInterleaveRows(sourceField, encoding);
  const bool strided = rows > 1;
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  const bool vectorLeaf =
      exactLeaf(getLeaf(), "rvv", "bitmask-window-load",
                "rvv.bitmask-window-load", "none", tail);
  const bool maskLeaf =
      exactLeaf(getLeaf(), "rvv", "bitmask-window-load",
                "rvv.bitmask-window-mask", "none", tail);
  const bool maskUse =
      maskLeaf && getResult().hasOneUse() &&
      mlir::isa<RVVBitplaneMergeOp, RVVSignedBitmaskReduceOp, RVVMaskedNegateOp>(
          *getResult().getUsers().begin());
  if (!sourceField || !fieldElement || fieldElement.isSigned() ||
      fieldElement.getWidth() != 1 || !resultElement ||
      resultElement.isSigned() || resultElement.getWidth() != 1 ||
      (field.getLayout().getCarrier() != "local" &&
       field.getLayout().getCarrier() != "scalar") ||
      result.getLayout().getCarrier() != "rvv" || !scalarBase ||
      getWindowAxes().empty() ||
      getWindowAxes().size() != getWindowExtents().size() ||
      llvm::any_of(getWindowExtents(),
                   [](int64_t extent) { return extent <= 0; }) ||
      getAccess().getForm() != (strided ? "strided" : "unit") ||
      (strided && getAccess().getFieldChunkBytes() != 1) ||
      getAccess().getMapping() != "grouped_layered" ||
      getAccess().getGroupSize() <= 0 || getAccess().getLayerSize() <= 0 ||
      getAccess().getGroupSize() != getAccess().getLayerSize() * 8 ||
      getAccess().getOrder() != "lo_first" || getAccess().getBitOffset() % 8 ||
      sourceAccess.getMapping() != getAccess().getMapping() ||
      sourceAccess.getGroupSize() != getAccess().getGroupSize() ||
      sourceAccess.getLayerSize() != getAccess().getLayerSize() ||
      sourceAccess.getOrder() != getAccess().getOrder() ||
      sourceAccess.getBitOffset() != getAccess().getBitOffset() ||
      (validity != "full" && validity != "tail") ||
      (!vectorLeaf && !maskUse))
    return emitOpError()
           << "RVV bitmask window load requires one byte-aligned logical-u1 field, "
              "one scalar byte base, complete lane window axes, and the exact RVV leaf; "
              "field_def="
           << (getField().getDefiningOp()
                   ? getField().getDefiningOp()->getName().getStringRef()
                   : llvm::StringRef("block-argument"))
           << ", field=" << field << ", byte_base=" << getByteBase().getType()
           << ", scalar_base=" << scalarBase << ", result=" << result
           << ", access=" << getAccess();

  llvm::DenseSet<int64_t> windowAxes;
  for (int64_t axis : getWindowAxes())
    if (!windowAxes.insert(axis).second)
      return emitOpError("RVV bitmask window axes must be unique");
  auto windowElements =
      checkedPositiveProduct(getWindowExtents());
  if (!windowElements || *windowElements % getAccess().getGroupSize())
    return emitOpError(
        "RVV bitmask window extent must cover complete packed groups");

  auto expectedPartOffsets = bitmaskWindowPartOffsets(
      result, getWindowAxes(), getWindowExtents());
  if (!expectedPartOffsets ||
      getPartBitOffsets() != llvm::ArrayRef<int64_t>(*expectedPartOffsets) ||
      llvm::any_of(*expectedPartOffsets, [&](int64_t offset) {
        return offset < 0 || offset >= *windowElements || offset % 8;
      }))
    return emitOpError()
           << "RVV bitmask window part offsets disagree with its typed "
              "time/lane/replica mapping; actual="
           << getPartBitOffsets() << ", expected="
           << (expectedPartOffsets
                   ? llvm::ArrayRef<int64_t>(*expectedPartOffsets)
                   : llvm::ArrayRef<int64_t>())
           << ", result=" << result << ", window_axes=" << getWindowAxes()
           << ", window_extents=" << getWindowExtents();

  auto fieldAxes = field.getAxisIds().asArrayRef();
  auto fieldShape = field.getShape().asArrayRef();
  auto sourceAxis = llvm::find(fieldAxes, getSourceAxis());
  if (sourceAxis == fieldAxes.end())
    return emitOpError("RVV bitmask source axis is absent from its field");
  const bool replacesSourceAxis =
      getWindowAxes().size() == 1 && getWindowAxes()[0] == getSourceAxis();
  if (!replacesSourceAxis &&
      llvm::any_of(getWindowAxes(), [&](int64_t axis) {
        return llvm::is_contained(fieldAxes, axis);
      }))
    return emitOpError(
        "RVV bitmask appended window axes must be disjoint from retained field axes");
  const size_t sourcePosition =
      static_cast<size_t>(sourceAxis - fieldAxes.begin());
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (size_t position = 0; position < fieldAxes.size(); ++position) {
    if (position == sourcePosition)
      continue;
    expectedAxes.push_back(fieldAxes[position]);
    expectedShape.push_back(fieldShape[position]);
  }
  const size_t retainedAxes = expectedAxes.size();
  if (replacesSourceAxis) {
    expectedAxes.insert(expectedAxes.begin() + sourcePosition,
                        getWindowAxes().begin(), getWindowAxes().end());
    expectedShape.insert(expectedShape.begin() + sourcePosition,
                         getWindowExtents().begin(), getWindowExtents().end());
  } else {
    expectedAxes.append(getWindowAxes().begin(), getWindowAxes().end());
    expectedShape.append(getWindowExtents().begin(), getWindowExtents().end());
  }
  if (!llvm::equal(expectedAxes, result.getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, result.getShape().asArrayRef()))
    return emitOpError(
        "RVV bitmask window result must retain source axes and append its logical window axes");

  auto layout = result.getLayout();
  for (size_t position = 0; position < result.getAxisIds().size(); ++position) {
    if (replacesSourceAxis && position == sourcePosition)
      continue;
    if (!replacesSourceAxis && position >= retainedAxes)
      continue;
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
          "RVV bitmask window retained axes must remain register coordinates");
  }
  for (size_t offset = 0; offset < getWindowExtents().size(); ++offset) {
    const size_t position = replacesSourceAxis ? sourcePosition
                                               : retainedAxes + offset;
    if (layout.getTimeFactors()[position] <= 0 ||
        layout.getLaneFactors()[position] <= 0 ||
        layout.getReplicaFactors()[position] <= 0 ||
        layout.getTimeFactors()[position] *
                layout.getLaneFactors()[position] *
                layout.getReplicaFactors()[position] !=
            getWindowExtents()[offset] ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
          "RVV bitmask window axes must form one complete time/lane/replica window");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVGroupedMacLoadOp::verify() {
  auto window = getResult().getType();
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  if (!lhs || lhs.getLayout().getCarrier() != "rvv")
    return emitOpError("grouped MAC lhs must use an RVV representation");
  if (window.getFamily() != "grouped-mac" || !getPlan() ||
      (getPlan().getKind() != "compact" &&
       getPlan().getKind() != "fragmented"))
    return emitOpError("grouped MAC window has invalid structural parameters");
  if (window.getReductionAxis() != getPlan().getReductionAxis() ||
      window.getSlots() != getPlan().getUnroll() ||
      window.getTermsPerSlot() != getPlan().getGroup())
    return emitOpError("grouped MAC window shape disagrees with its operation");
  if (window.getLhsType() != getLhs().getType() ||
      window.getRhsType() != getRhs().getType() ||
      window.getPartialLayout() != getPartialLayout())
    return emitOpError("grouped MAC window type disagrees with its operands");
  auto windowResult = mlir::dyn_cast<ValueType>(window.getResultType());
  auto supplyPlan = groupedMacSupplyProjection(lhs, windowResult);
  auto physicalParts = physicalPartCount(windowResult);
  if (!supplyPlan ||
      getPackedSupplyForResult() != llvm::ArrayRef<int64_t>(*supplyPlan) ||
      !physicalParts || *physicalParts != window.getResultParts())
    return emitOpError(
        "grouped MAC window requires one typed packed-supply mapping over all issue and register parts");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("grouped MAC load must be nested in one target kernel");
  auto target = kernel.getTarget();
  if (!hasGroupedMacOperandGeometry(
          getOperation(), getLhs(), getRhs(), getPlan().getReductionAxis(),
          getPlan(), getLoadLayout(), getPartialLayout(),
          window.getResultLayout()) ||
      !supportsRVVLayout(target, window.getResultLayout()))
    return emitOpError(
        "grouped MAC load has no complete typed field, cohort-storage, "
        "RVV-layout, or shared reduction-point realization");
  if ((!exactLeaf(getLeaf(), "rvv", "grouped-mac-load",
                  "rvv.grouped-mac-load.u8-s8", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "grouped-mac-load",
                  "rvv.grouped-mac-load.u8-s8", "none", "exact")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {getPlan().getGroup(), getPlan().getUnroll(),
               getPlan().getReductionAxis()}))
    return emitOpError("grouped MAC load leaf disagrees with its window");
  return mlir::success();
}

mlir::LogicalResult RVVGroupedMacStepOp::verify() {
  auto window = getWindow().getType();
  if (window.getFamily() != "grouped-mac" ||
      getAccumulator().getType() != getResult().getType() ||
      window.getResultType() != getResult().getType() ||
      window.getResultLayout() != getResult().getType().getLayout() ||
      window.getResultLayout().getCarrier() != "rvv" ||
      (!exactLeaf(getLeaf(), "rvv", "grouped-mac-step",
                  "rvv.vwmaccsu.vx.grouped", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "grouped-mac-step",
                  "rvv.vwmaccsu.vx.grouped", "none", "exact")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({window.getTermsPerSlot(), window.getSlots(),
                                   window.getReductionAxis()}))
    return emitOpError("grouped MAC compute step contract is incomplete");
  return mlir::success();
}

mlir::LogicalResult RVVWidenDotOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto shapedResult = mlir::dyn_cast<ValueType>(getResult().getType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
      shapedResult ? shapedResult.getElementType() : getResult().getType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("RVV widening dot must be nested in one target kernel");
  auto target = kernel.getTarget();
  auto partialLMULValue = doubledPositive(getSliceLmulEighths());
  const int64_t partialLMUL = partialLMULValue.value_or(-1);
  llvm::SmallVector<int64_t> freeAxes;
  for (int64_t axis : lhs.getAxisIds().asArrayRef())
    if (!llvm::is_contained(getOver(), axis))
      freeAxes.push_back(axis);
  for (int64_t axis : rhs.getAxisIds().asArrayRef())
    if (!llvm::is_contained(getOver(), axis) &&
        !llvm::is_contained(freeAxes, axis))
      freeAxes.push_back(axis);
  const bool legalResult =
      (!shapedResult && isScalar(getResult().getType()) && freeAxes.empty()) ||
      (shapedResult && shapedResult.getLayout().getCarrier() == "scalar" &&
       shapedResult.getAxisIds().asArrayRef() ==
           llvm::ArrayRef<int64_t>(freeAxes));
  const int64_t outputParts =
      shapedResult ? physicalPartCount(shapedResult).value_or(-1) : 1;
  auto laneProduct = [](ValueType value,
                        llvm::ArrayRef<int64_t> selectedAxes) {
    int64_t result = 1;
    for (auto [axis, factor] :
         llvm::zip(value.getAxisIds().asArrayRef(),
                   value.getLayout().getLaneFactors().asArrayRef())) {
      if (!llvm::is_contained(selectedAxes, axis))
        continue;
      if (factor <= 0 || result > std::numeric_limits<int64_t>::max() / factor)
        return int64_t{-1};
      result *= factor;
    }
    return result;
  };
  auto allLanes = [](ValueType value) {
    return checkedPositiveProduct(
               value.getLayout().getLaneFactors().asArrayRef())
        .value_or(-1);
  };
  const int64_t lhsReductionLanes = laneProduct(lhs, getOver());
  const int64_t rhsReductionLanes = laneProduct(rhs, getOver());
  const int64_t lhsLanes = allLanes(lhs);
  const int64_t rhsLanes = allLanes(rhs);
  auto validOffsets = [&](llvm::ArrayRef<int64_t> offsets, int64_t lanes) {
    if (offsets.size() != static_cast<size_t>(outputParts) || lanes <= 0 ||
        getReductionLanes() <= 0)
      return false;
    return llvm::all_of(offsets, [&](int64_t offset) {
      return offset >= 0 && offset % getReductionLanes() == 0 &&
             offset <= lanes - getReductionLanes();
    });
  };
  auto lhsParts = physicalPartCount(lhs);
  auto rhsParts = physicalPartCount(rhs);
  auto validParts = [&](llvm::ArrayRef<int64_t> parts,
                        std::optional<int64_t> available) {
    if (!available || getReductionStreams() <= 0 ||
        outputParts > std::numeric_limits<int64_t>::max() /
                          getReductionStreams() ||
        parts.size() != static_cast<size_t>(outputParts * getReductionStreams()))
      return false;
    return llvm::all_of(parts, [&](int64_t part) {
      return part >= 0 && part < *available;
    });
  };
  const bool closedLaneSlices =
      outputParts > 0 && getReductionStreams() > 0 &&
      getReductionLanes() == lhsReductionLanes &&
      getReductionLanes() == rhsReductionLanes && lhsLanes > 0 &&
      rhsLanes > 0 &&
      getSliceLmulEighths() <= lhs.getLayout().getLmulEighths() &&
      getSliceLmulEighths() <= rhs.getLayout().getLmulEighths() &&
      lhs.getLayout().getLmulEighths() * getReductionLanes() <=
          getSliceLmulEighths() * lhsLanes &&
      rhs.getLayout().getLmulEighths() * getReductionLanes() <=
          getSliceLmulEighths() * rhsLanes &&
      validOffsets(getLhsLaneOffsets(), lhsLanes) &&
      validOffsets(getRhsLaneOffsets(), rhsLanes) &&
      validParts(getLhsParts(), lhsParts) &&
      validParts(getRhsParts(), rhsParts);
  auto factorFor = [](LayoutAttr layout, mlir::DenseI64ArrayAttr factors,
                      int64_t axis) {
    auto found = llvm::find(layout.getAxisIds().asArrayRef(), axis);
    if (found == layout.getAxisIds().asArrayRef().end())
      return int64_t{0};
    return factors[static_cast<size_t>(
        found - layout.getAxisIds().asArrayRef().begin())];
  };
  bool compatibleReductionMapping =
      !getOver().empty() && lhs.getLayout().getCarrier() == "rvv" &&
      rhs.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getSew() == rhs.getLayout().getSew();
  for (int64_t axis : getOver()) {
    auto completeMappedAxis = [&](ValueType value) {
      auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
      if (found == value.getAxisIds().asArrayRef().end())
        return false;
      const size_t position = static_cast<size_t>(
          found - value.getAxisIds().asArrayRef().begin());
      const int64_t extent = value.getShape()[position];
      auto layout = value.getLayout();
      return extent > 0 &&
             layout.getTimeFactors()[position] *
                     layout.getLaneFactors()[position] ==
                 extent &&
             layout.getReplicaFactors()[position] == 1 &&
             layout.getFragmentFactors()[position] == 1 &&
             layout.getLocalFactors()[position] == 1;
    };
    compatibleReductionMapping &=
        completeMappedAxis(lhs) && completeMappedAxis(rhs) &&
        factorFor(lhs.getLayout(), lhs.getLayout().getLaneFactors(), axis) ==
            factorFor(rhs.getLayout(), rhs.getLayout().getLaneFactors(), axis) &&
        factorFor(lhs.getLayout(), lhs.getLayout().getTimeFactors(), axis) ==
            factorFor(rhs.getLayout(), rhs.getLayout().getTimeFactors(), axis);
  }
  auto selectedCoordinateOrder = [&](ValueType value,
                                     mlir::DenseI64ArrayAttr factors) {
    llvm::SmallVector<int64_t> order;
    for (auto [axis, factor] :
         llvm::zip(value.getAxisIds().asArrayRef(), factors.asArrayRef()))
      if (factor > 1 && llvm::is_contained(getOver(), axis))
        order.push_back(axis);
    return order;
  };
  compatibleReductionMapping &=
      selectedCoordinateOrder(lhs, lhs.getLayout().getLaneFactors()) ==
          selectedCoordinateOrder(rhs, rhs.getLayout().getLaneFactors()) &&
      selectedCoordinateOrder(lhs, lhs.getLayout().getTimeFactors()) ==
          selectedCoordinateOrder(rhs, rhs.getLayout().getTimeFactors());
  llvm::StringRef topologyKind = getPartialTopology().getKind();
  auto partialLayoutPlan = getPartialLayoutPlanAttr();
  auto nestedPartialPlan = getNestedPartialPlanAttr();
  auto sequentialPartialPlan = getSequentialPartialPlanAttr();
  auto scaledPartialPlan = getScaledPartialPlanAttr();
  auto layeredPartialPlan = getLayeredPartialPlanAttr();
  auto combinePartialPlan =
      (*this)->getAttrOfType<PartialCombinePlanAttr>("partial_combine_plan");
  const bool topologyAssigned = topologyKind != "unassigned";
  const bool scaledIssues =
      topologyKind == "reduced_scaled" && getReductionStreams() > 1;
  const bool collectsPartials = topologyKind == "level_scaled" || scaledIssues;
  const bool sequentialPlanRequired =
      getReductionStreams() > 0 &&
      (topologyKind == "sequential_fused" ||
       topologyKind == "sequential_per_stream" ||
       collectsPartials);
  // LowerRISCVComposites creates a legal but deliberately unassigned physical
  // contraction.  The topology pass owns every plan attribute and the final
  // verifier rejects any surviving unassigned op.  Do not require a topology
  // plan at this intermediate verifier boundary; once assigned, require the
  // complete mutually consistent plan set below.
  bool layoutPlanClosed = !topologyAssigned;
  if (topologyAssigned) {
    if (partialLayoutPlan)
      layoutPlanClosed =
          (topologyKind == "nested_scaled_stream") ==
              static_cast<bool>(nestedPartialPlan) &&
          sequentialPlanRequired ==
              static_cast<bool>(sequentialPartialPlan) &&
          (topologyKind == "scaled" || topologyKind == "reduced_scaled") ==
              static_cast<bool>(scaledPartialPlan) &&
          (topologyKind == "layered") ==
              static_cast<bool>(layeredPartialPlan);
    else
      layoutPlanClosed = false;
  }
  if (topologyAssigned && partialLayoutPlan) {
    auto partialSlot = mlir::dyn_cast<ValueType>(
        partialLayoutPlan.getPartialSlotType());
    int64_t plannedLanes = 1;
    bool completePartialAxes =
        partialSlot && partialSlot.getLayout().getCarrier() == "rvv" &&
        partialSlot.getAxisIds().asArrayRef() == getOver() &&
        partialSlot.getShape().size() == getOver().size();
    if (completePartialAxes)
      for (auto [extent, lane] :
           llvm::zip(partialSlot.getShape().asArrayRef(),
                     partialSlot.getLayout().getLaneFactors().asArrayRef())) {
        if (extent <= 0 || extent != lane ||
            plannedLanes > getReductionLanes() / extent) {
          completePartialAxes = false;
          break;
        }
        plannedLanes *= extent;
      }
    layoutPlanClosed &=
        completePartialAxes && plannedLanes == getReductionLanes();
    const int64_t sourceSlots = getPartialTopology().getSourceSlots();
    const int64_t split = getPartialTopology().getLaneSplit();
    const int64_t expected =
        getPartialTopology().getOutputReplicas() * sourceSlots;
    const int64_t sourceLanes = split * getReductionLanes();
    auto sourceLhs = mlir::dyn_cast<ValueType>(
        partialLayoutPlan.getSourceLhsType());
    auto sourceRhs = mlir::dyn_cast<ValueType>(
        partialLayoutPlan.getSourceRhsType());
    auto mapComplete = [&](mlir::DenseI64ArrayAttr values) {
      return expected > 0 &&
             values.size() == static_cast<size_t>(expected);
    };
    const bool sourcePlanRequired = split > 1 && !nestedPartialPlan;
    layoutPlanClosed &=
        sourcePlanRequired == static_cast<bool>(sourceLhs) &&
        sourcePlanRequired == static_cast<bool>(sourceRhs) &&
        sourcePlanRequired == mlir::isa<ValueType>(
                                  partialLayoutPlan.getSourceSlotType()) &&
        sourcePlanRequired == mlir::isa<ValueType>(
                                  partialLayoutPlan.getSplitSourceSlotType());
    auto sameLogicalValue = [](ValueType source, ValueType planned) {
      return source && planned &&
             source.getElementType() == planned.getElementType() &&
             source.getShape() == planned.getShape() &&
             source.getAxisIds() == planned.getAxisIds();
    };
    if (sourcePlanRequired)
      layoutPlanClosed &=
          sourceSlots > 0 && sourceLanes > 0 &&
          sameLogicalValue(lhs, sourceLhs) &&
          sameLogicalValue(rhs, sourceRhs) &&
          sourceLhs.getLayout().getCarrier() == "rvv" &&
          sourceRhs.getLayout().getCarrier() == "rvv" &&
          supportsRVVLayout(target, sourceLhs.getLayout()) &&
          supportsRVVLayout(target, sourceRhs.getLayout()) &&
          mapComplete(partialLayoutPlan.getSourceLhsParts()) &&
          mapComplete(partialLayoutPlan.getSourceRhsParts()) &&
          mapComplete(partialLayoutPlan.getSourceLhsOffsets()) &&
          mapComplete(partialLayoutPlan.getSourceRhsOffsets());
    if (layoutPlanClosed && sourcePlanRequired) {
      auto plannedLhsParts = physicalPartCount(sourceLhs);
      auto plannedRhsParts = physicalPartCount(sourceRhs);
      auto plannedLhsLanes = allLanes(sourceLhs);
      auto plannedRhsLanes = allLanes(sourceRhs);
      if (!plannedLhsParts || !plannedRhsParts ||
          plannedLhsLanes < sourceLanes || plannedRhsLanes < sourceLanes) {
        layoutPlanClosed = false;
      } else {
        for (auto [lhsPart, rhsPart, lhsOffset, rhsOffset] : llvm::zip(
                 partialLayoutPlan.getSourceLhsParts().asArrayRef(),
                 partialLayoutPlan.getSourceRhsParts().asArrayRef(),
                 partialLayoutPlan.getSourceLhsOffsets().asArrayRef(),
                 partialLayoutPlan.getSourceRhsOffsets().asArrayRef()))
          layoutPlanClosed &= lhsPart >= 0 && lhsPart < *plannedLhsParts &&
                              rhsPart >= 0 && rhsPart < *plannedRhsParts &&
                              lhsOffset >= 0 && rhsOffset >= 0 &&
                              lhsOffset % sourceLanes == 0 &&
                              rhsOffset % sourceLanes == 0 &&
                              lhsOffset <= plannedLhsLanes - sourceLanes &&
                              rhsOffset <= plannedRhsLanes - sourceLanes;
      }
    }
  }
  if (topologyAssigned && nestedPartialPlan) {
    auto windowAxis = llvm::find(lhs.getAxisIds().asArrayRef(),
                                 nestedPartialPlan.getWindowAxis());
    auto rhsWindowAxis = llvm::find(rhs.getAxisIds().asArrayRef(),
                                    nestedPartialPlan.getWindowAxis());
    std::optional<size_t> resultWindowPosition;
    if (shapedResult) {
      auto found = llvm::find(shapedResult.getAxisIds().asArrayRef(),
                              nestedPartialPlan.getWindowAxis());
      if (found != shapedResult.getAxisIds().asArrayRef().end())
        resultWindowPosition = static_cast<size_t>(
            found - shapedResult.getAxisIds().asArrayRef().begin());
    }
    auto issueLhs = mlir::dyn_cast<ValueType>(nestedPartialPlan.getIssueLhsType());
    auto issueRhs = mlir::dyn_cast<ValueType>(nestedPartialPlan.getIssueRhsType());
    auto sourceSet = mlir::dyn_cast<PartialSetType>(
        nestedPartialPlan.getSourceSetType());
    const bool axesPresent =
        windowAxis != lhs.getAxisIds().asArrayRef().end() &&
        rhsWindowAxis != rhs.getAxisIds().asArrayRef().end() && shapedResult &&
        resultWindowPosition;
    llvm::SmallVector<int64_t> expectedOutputAxes;
    llvm::SmallVector<int64_t> expectedOutputReplicaFactors;
    if (shapedResult)
      for (auto [axis, factor] :
           llvm::zip(shapedResult.getAxisIds().asArrayRef(),
                     shapedResult.getLayout().getReplicaFactors().asArrayRef()))
        if (!llvm::is_contained(
                nestedPartialPlan.getPartialAxes().asArrayRef(), axis)) {
          expectedOutputAxes.push_back(axis);
          expectedOutputReplicaFactors.push_back(factor);
        }
    auto expectedOutputReplicas =
        checkedPositiveProduct(expectedOutputReplicaFactors);
    llvm::SmallVector<int64_t> plannedOutputAxes(
        nestedPartialPlan.getOutputAxes().asArrayRef().begin(),
        nestedPartialPlan.getOutputAxes().asArrayRef().end());
    llvm::SmallVector<int64_t> topologyOutputAxes(
        getPartialTopology().getOutputAxes().asArrayRef().begin(),
        getPartialTopology().getOutputAxes().asArrayRef().end());
    llvm::SmallVector<int64_t> sortedExpectedOutputAxes(expectedOutputAxes);
    const bool topologyAndPlanAgree =
        plannedOutputAxes == topologyOutputAxes;
    llvm::sort(plannedOutputAxes);
    llvm::sort(topologyOutputAxes);
    llvm::sort(sortedExpectedOutputAxes);
    const bool outputAxesClosed =
        topologyAndPlanAgree && plannedOutputAxes == sortedExpectedOutputAxes &&
        topologyOutputAxes == sortedExpectedOutputAxes;
    layoutPlanClosed &= axesPresent && issueLhs && issueRhs && sourceSet;
    if (layoutPlanClosed) {
      const size_t lhsPosition = static_cast<size_t>(
          windowAxis - lhs.getAxisIds().asArrayRef().begin());
      const size_t rhsPosition = static_cast<size_t>(
          rhsWindowAxis - rhs.getAxisIds().asArrayRef().begin());
      const size_t resultPosition = *resultWindowPosition;
      auto closesIssuePartition = [&](ValueType source, ValueType issue,
                                      size_t position) {
        const int64_t sourceTime =
            source.getLayout().getTimeFactors()[position];
        const int64_t sourceLanes =
            source.getLayout().getLaneFactors()[position];
        const int64_t streams = nestedPartialPlan.getIssueStreams();
        const int64_t window = nestedPartialPlan.getWindowExtent();
        if (sourceTime <= 0 || sourceLanes <= 0 || streams <= 0 || window <= 0 ||
            sourceTime > std::numeric_limits<int64_t>::max() / sourceLanes ||
            streams > std::numeric_limits<int64_t>::max() / window ||
            sourceTime * sourceLanes != streams * window ||
            issue.getElementType() != source.getElementType() ||
            issue.getAxisIds() != source.getAxisIds() ||
            issue.getShape().size() != source.getShape().size() ||
            issue.getShape()[position] != window ||
            issue.getLayout().getTimeFactors()[position] != 1 ||
            issue.getLayout().getLaneFactors()[position] != window ||
            issue.getLayout().getReplicaFactors()[position] != 1)
          return false;
        for (size_t axisPosition = 0; axisPosition < source.getShape().size();
             ++axisPosition) {
          if (axisPosition == position)
            continue;
          const bool reductionAxis =
              llvm::is_contained(getOver(), source.getAxisIds()[axisPosition]);
          const int64_t sourceTime =
              source.getLayout().getTimeFactors()[axisPosition];
          const int64_t sourceLanes =
              source.getLayout().getLaneFactors()[axisPosition];
          const int64_t issueTime =
              issue.getLayout().getTimeFactors()[axisPosition];
          const int64_t issueLanes =
              issue.getLayout().getLaneFactors()[axisPosition];
          if (issue.getShape()[axisPosition] != source.getShape()[axisPosition] ||
              issue.getLayout().getReplicaFactors()[axisPosition] !=
                  source.getLayout().getReplicaFactors()[axisPosition] ||
              issue.getLayout().getFragmentFactors()[axisPosition] !=
                  source.getLayout().getFragmentFactors()[axisPosition] ||
              issue.getLayout().getLocalFactors()[axisPosition] !=
                  source.getLayout().getLocalFactors()[axisPosition])
            return false;
          if (reductionAxis) {
            if (sourceTime <= 0 || sourceLanes <= 0 || issueTime != 1 ||
                sourceTime > std::numeric_limits<int64_t>::max() / sourceLanes ||
                issueLanes != sourceTime * sourceLanes)
              return false;
          } else if (issueTime != sourceTime || issueLanes != sourceLanes) {
            return false;
          }
        }
        return true;
      };
      layoutPlanClosed &=
          closesIssuePartition(lhs, issueLhs, lhsPosition) &&
          closesIssuePartition(rhs, issueRhs, rhsPosition) &&
          expectedOutputReplicas &&
          outputAxesClosed &&
          nestedPartialPlan.getOutputReplicas() == *expectedOutputReplicas &&
          getPartialTopology().getOutputReplicas() == *expectedOutputReplicas &&
          getPartialTopology().getPartialAxes().asArrayRef() ==
              nestedPartialPlan.getPartialAxes().asArrayRef() &&
          shapedResult.getShape()[resultPosition] ==
              nestedPartialPlan.getIssueStreams() *
                  nestedPartialPlan.getWindowExtent() &&
          getPartialTopology().getSourceSlots() ==
              nestedPartialPlan.getIssueStreams() &&
          getPartialTopology().getPartialSlots() ==
              nestedPartialPlan.getIssueStreams() *
                  nestedPartialPlan.getPartialSlots() &&
          getPartialTopology().getLaneSplit() ==
              nestedPartialPlan.getPartialSlots() &&
          getPartialTopology().getCombineArity() ==
              nestedPartialPlan.getPartialSlots() &&
          getPartialTopology().getResourceGroups() ==
              nestedPartialPlan.getResourceGroups() &&
          llvm::is_contained(getOver(), sourceSet.getReductionAxis()) &&
          supportsRVVLayout(target, issueLhs.getLayout()) &&
          supportsRVVLayout(target, issueRhs.getLayout()) &&
          nestedPartialPlan.getResourceGroups() <= target.getVectorRegisters();
    }
  }
  if (topologyAssigned && sequentialPartialPlan) {
    auto lhsIssue = mlir::dyn_cast<ValueType>(
        sequentialPartialPlan.getLhsIssueType());
    auto rhsIssue = mlir::dyn_cast<ValueType>(
        sequentialPartialPlan.getRhsIssueType());
    auto accumulator = mlir::dyn_cast<ValueType>(
        sequentialPartialPlan.getAccumulatorType());
    auto lhsIssueParts = lhsIssue ? physicalPartCount(lhsIssue) : std::nullopt;
    auto rhsIssueParts = rhsIssue ? physicalPartCount(rhsIssue) : std::nullopt;
    const int64_t issueCount = sequentialPartialPlan.getIssueCount();
    auto sourceMatchesIssue = [&](ValueType source, ValueType issue) {
      if (!issue || source.getElementType() != issue.getElementType() ||
          issue.getAxisIds().asArrayRef() != getOver() ||
          issue.getShape().size() != getOver().size())
        return false;
      for (size_t issuePosition = 0; issuePosition < getOver().size();
           ++issuePosition) {
        auto sourcePosition = llvm::find(source.getAxisIds().asArrayRef(),
                                         getOver()[issuePosition]);
        if (sourcePosition == source.getAxisIds().asArrayRef().end())
          return false;
        const size_t position = static_cast<size_t>(
            sourcePosition - source.getAxisIds().asArrayRef().begin());
        if (source.getShape()[position] !=
                issue.getShape()[issuePosition] *
                    source.getLayout().getTimeFactors()[position] ||
            issue.getShape()[issuePosition] !=
                issue.getLayout().getLaneFactors()[issuePosition] ||
            issue.getLayout().getLaneFactors()[issuePosition] !=
                source.getLayout().getLaneFactors()[position])
          return false;
      }
      for (size_t position = 0; position < source.getAxisIds().size();
           ++position)
        if (!llvm::is_contained(getOver(), source.getAxisIds()[position]) &&
            (source.getLayout().getFragmentFactors()[position] != 1 ||
             source.getLayout().getLocalFactors()[position] != 1))
          return false;
      return issue.getLayout().getSew() == source.getLayout().getSew() &&
             issue.getLayout().getLmulEighths() == getSliceLmulEighths() &&
             issue.getLayout().getVl() == getReductionLanes();
    };
    bool laneOffsetsClosed =
        sequentialPartialPlan.getLhsSourceParts().asArrayRef() == getLhsParts() &&
        sequentialPartialPlan.getRhsSourceParts().asArrayRef() == getRhsParts() &&
        sequentialPartialPlan.getLhsSourceParts().size() ==
            static_cast<size_t>(outputParts * issueCount) &&
        sequentialPartialPlan.getRhsSourceParts().size() ==
            static_cast<size_t>(outputParts * issueCount) &&
        getLhsLaneOffsets().size() == static_cast<size_t>(outputParts) &&
        getRhsLaneOffsets().size() == static_cast<size_t>(outputParts) &&
        sequentialPartialPlan.getLhsLaneOffsets().size() ==
            sequentialPartialPlan.getLhsSourceParts().size() &&
        sequentialPartialPlan.getRhsLaneOffsets().size() ==
            sequentialPartialPlan.getRhsSourceParts().size();
    if (laneOffsetsClosed)
      for (int64_t output = 0; output < outputParts; ++output)
        for (int64_t issue = 0; issue < issueCount; ++issue) {
          const size_t index = static_cast<size_t>(output * issueCount + issue);
          laneOffsetsClosed &=
              sequentialPartialPlan.getLhsLaneOffsets()[index] ==
                  getLhsLaneOffsets()[output] &&
              sequentialPartialPlan.getRhsLaneOffsets()[index] ==
                  getRhsLaneOffsets()[output];
        }
    const llvm::StringRef expectedRealization =
        collectsPartials
            ? llvm::StringRef("fused_partial")
            : topologyKind == "sequential_fused"
                  ? llvm::StringRef("fused")
                  : llvm::StringRef("per_stream");
    const bool issueSupplyConsumed =
        expectedRealization == "fused_partial" ||
        (sequentialPartialPlan.getLhsIssueSupply() == "slice" &&
         sequentialPartialPlan.getRhsIssueSupply() == "slice");
    const bool sequentialPlanClosed =
        !getOver().empty() && issueCount == getReductionStreams() &&
        sequentialPartialPlan.getRealization() == expectedRealization &&
        issueSupplyConsumed &&
        sequentialPartialPlan.getReductionAxes().asArrayRef() == getOver() &&
        lhsIssue &&
        rhsIssue && accumulator &&
        lhsIssueParts && rhsIssueParts && issueCount > 0 &&
        sequentialPartialPlan.getLhsSourceParts().asArrayRef() ==
            getLhsParts() &&
        sequentialPartialPlan.getRhsSourceParts().asArrayRef() ==
            getRhsParts() &&
        sequentialPartialPlan.getLhsSourceParts().size() ==
            static_cast<size_t>(issueCount * outputParts) &&
        sequentialPartialPlan.getRhsSourceParts().size() ==
            static_cast<size_t>(issueCount * outputParts) &&
        *lhsIssueParts == 1 && *rhsIssueParts == 1 && laneOffsetsClosed &&
        sourceMatchesIssue(lhs, lhsIssue) && sourceMatchesIssue(rhs, rhsIssue) &&
        (!collectsPartials ||
         ((issueCount == 1 || getFusedStreamsLegal()) && partialLayoutPlan &&
          accumulator == partialLayoutPlan.getPartialSlotType())) &&
        (!scaledIssues || getPartialTopology().getLaneSplit() == 1) &&
        (collectsPartials ||
         getPartialTopology().getResourceGroups() ==
             sequentialPartialPlan.getResourceGroups()) &&
        sequentialPartialPlan.getResourceGroups() <= target.getVectorRegisters();
    layoutPlanClosed &= sequentialPlanClosed;
  }
  if (topologyAssigned && scaledPartialPlan) {
    auto narrowScale = mlir::dyn_cast<ValueType>(
        scaledPartialPlan.getNarrowScaleType());
    auto finalSet =
        partialLayoutPlan
            ? mlir::dyn_cast<PartialSetType>(
                  getPartialTopology().getLaneSplit() > 1 ||
                          topologyKind == "reduced_scaled"
                      ? partialLayoutPlan.getScaleCombinedSetType()
                      : partialLayoutPlan.getFullScaledSetType())
            : PartialSetType();
    auto finalLanes = finalSet ? rvvLaneCount(finalSet.getPartialType())
                               : std::optional<int64_t>();
    auto finalInstruction =
        finalSet && finalLanes
            ? (*finalLanes == 1
                   ? llvm::StringRef("rvv.partial-finalize.extract")
                   : llvm::StringRef("rvv.partial-finalize.reduce"))
            : llvm::StringRef();
    const llvm::StringRef reduceInstruction =
        getPartialTopology().getLaneSplit() > 1 ||
                topologyKind == "reduced_scaled"
            ? llvm::StringRef("rvv.partial-reduce.widen")
            : llvm::StringRef("none");
    layoutPlanClosed &=
        scaledPartialPlan.getRealization() == topologyKind && finalSet &&
        scaledPartialPlan.getReduceInstruction() == reduceInstruction &&
        scaledPartialPlan.getFinalizeInstruction() == finalInstruction &&
        (topologyKind != "scaled" ||
         getPartialTopology().getLaneSplit() > 1 || narrowScale) &&
        (topologyKind != "reduced_scaled" || !narrowScale) &&
        getPartialTopology().getResourceGroups() ==
            scaledPartialPlan.getResourceGroups() &&
        scaledPartialPlan.getResourceGroups() <= target.getVectorRegisters();
  }
  if (topologyAssigned && topologyKind == "independent") {
    auto source = combinePartialPlan
                      ? mlir::dyn_cast<PartialSetType>(combinePartialPlan.getSourceSetType())
                      : PartialSetType();
    auto element = source ? mlir::dyn_cast<mlir::IntegerType>(
                                source.getPartialType().getElementType())
                          : mlir::IntegerType();
    layoutPlanClosed &=
        source && combinePartialPlan.getResourceGroups() ==
                      getPartialTopology().getResourceGroups() &&
        combinePartialPlan.getResourceGroups() <= target.getVectorRegisters();
    if (element && element.getWidth() == 16 && !getFusedStreamsLegal()) {
      auto stages = combinePartialPlan.getCombineSetTypes();
      auto arities = combinePartialPlan.getCombineArities().asArrayRef();
      auto firstAttr = stages.empty() ? mlir::TypeAttr()
                                     : mlir::dyn_cast<mlir::TypeAttr>(stages[0]);
      auto first = firstAttr ? mlir::dyn_cast<PartialSetType>(firstAttr.getValue())
                             : PartialSetType();
      layoutPlanClosed &=
          first && !arities.empty() &&
          isWideningPartialCombineStage(source, first, arities[0]) &&
          supportsRVVLayout(target, first.getPartialType().getLayout()) &&
          first.getPartialType().getLayout().getRegisterGroups() <=
              target.getMaxWideningCombineGroups();
    }
  }
  if (topologyAssigned && layeredPartialPlan)
    layoutPlanClosed &= getPartialTopology().getRootOperand() >= 0 &&
                        getOver().size() == 1 &&
                        layeredPartialPlan.getGeometry().getAxis() == getOver()[0] &&
                        getPartialTopology().getResourceGroups() ==
                            layeredPartialPlan.getResourceGroups() &&
                        layeredPartialPlan.getResourceGroups() <=
                            target.getVectorRegisters();
  const bool validTopology =
      topologyKind == "unassigned" || topologyKind == "sequential_fused" ||
      topologyKind == "sequential_per_stream" ||
      topologyKind == "independent" || topologyKind == "replica_reduced" ||
      topologyKind == "scaled" ||
      topologyKind == "reduced_scaled" ||
      topologyKind == "level_scaled" ||
      topologyKind == "nested_scaled_stream" || topologyKind == "merged" ||
      topologyKind == "layered";
  if (!validTopology || !layoutPlanClosed || getPartialUnroll() <= 0 ||
      getSliceLmulEighths() <= 0 || !closedLaneSlices ||
      !lhsElement || !rhsElement || !resultElement ||
      lhsElement.getWidth() > 16 || rhsElement.getWidth() > 16 ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      resultElement.getWidth() != 32 || !legalResult ||
      getOver().empty() ||
      !llvm::is_contained(lhs.getAxisIds().asArrayRef(), getOver()[0]) ||
      !llvm::is_contained(rhs.getAxisIds().asArrayRef(), getOver()[0]) ||
      !compatibleReductionMapping ||
      !target.getHasWideningInteger() ||
      !supportsRVVLayout(target, lhs.getLayout()) ||
      !supportsRVVLayout(target, rhs.getLayout()) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                          getSliceLmulEighths()) ||
      !partialLMULValue ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), partialLMUL) ||
      !exactLeaf(getLeaf(), "rvv", "widen-dot",
                 "rvv.vwmul-vwredsum", "none", "exact"))
    return emitOpError()
           << "RVV widening dot requires matching <=16-bit integer vectors with "
              "at least one signed operand, one selected partial topology, "
              "one or more mapped reduction axes, a signed-i32 scalar result, "
              "and a legal doubled "
              "LMUL; lhs="
           << lhs << ", rhs=" << rhs << ", result=" << getResult().getType()
           << ", over=" << getOver() << ", partial_lmul=" << partialLMUL
           << ", leaf=" << getLeaf();
  return mlir::success();
}

bool weft::riscv::supportsRVVWidenBinary(TargetAttr target, ValueType lhs,
                                       ValueType rhs, ValueType result) {
  if (!target || !lhs || lhs != rhs || !result)
    return false;
  auto narrow = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto wide = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto source = lhs.getLayout();
  auto destination = result.getLayout();
  return narrow && wide && !narrow.isSignless() && !wide.isSignless() &&
         narrow.getSignedness() == wide.getSignedness() &&
         narrow.getWidth() >= 8 && wide.getWidth() == 2 * narrow.getWidth() &&
         lhs.getShape() == result.getShape() &&
         lhs.getAxisIds() == result.getAxisIds() &&
         source.getCarrier() == "rvv" && destination.getCarrier() == "rvv" &&
         source.getAxisIds() == destination.getAxisIds() &&
         source.getTimeFactors() == destination.getTimeFactors() &&
         source.getLaneFactors() == destination.getLaneFactors() &&
         source.getReplicaFactors() == destination.getReplicaFactors() &&
         source.getFragmentFactors() == destination.getFragmentFactors() &&
         source.getLocalFactors() == destination.getLocalFactors() &&
         source.getValidity() == destination.getValidity() &&
         (source.getValidity() == "full" || source.getValidity() == "tail") &&
         source.getSew() == narrow.getWidth() &&
         destination.getSew() == wide.getWidth() &&
         destination.getLmulEighths() == 2 * source.getLmulEighths() &&
         source.getVl() == destination.getVl() &&
         target.getHasWideningInteger() &&
         supportsRVVLayout(target, source) &&
         supportsRVVLayout(target, destination);
}

mlir::LogicalResult RVVWidenAddOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto element = mlir::dyn_cast<mlir::IntegerType>(
      getLhs().getType().getElementType());
  const llvm::StringRef instruction = element && element.isUnsigned()
                                          ? "rvv.vwaddu.vv" : "rvv.vwadd.vv";
  const llvm::StringRef tail = getResult().getType().getLayout().getValidity() == "tail"
                                   ? "agnostic" : "exact";
  if (!kernel || !supportsRVVWidenBinary(kernel.getTarget(), getLhs().getType(),
                                      getRhs().getType(), getResult().getType()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-add", instruction, "none", tail))
    return emitOpError(
        "RVV widening add requires identical signedness-qualified narrow vectors and a coordinate-identical doubled-width result");
  return mlir::success();
}

mlir::LogicalResult RVVNarrowShiftRightOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto input = getInput().getType();
  auto result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  const bool vectorAmount = getAmount().getType() == result;
  const llvm::StringRef instruction =
      vectorAmount ? "rvv.vnsrl.wv" : "rvv.vnsrl.wx";
  const llvm::StringRef tail =
      result.getLayout().getValidity() == "tail" ? "agnostic" : "exact";
  const int64_t operandGroups = input.getLayout().getRegisterGroups() +
      (vectorAmount ? result.getLayout().getRegisterGroups() : 0);
  if (!kernel || !element || !element.isUnsigned() ||
      !supportsRVVWidenBinary(kernel.getTarget(), result, result, input) ||
      (!vectorAmount && getAmount().getType() != input.getElementType()) ||
      !exactLeaf(getLeaf(), "rvv", "narrow-shift-right", instruction, "none", tail) ||
      getLeaf().getOperandGroups() != operandGroups ||
      getLeaf().getResultGroups() != result.getLayout().getRegisterGroups() ||
      getLeaf().getTemporaryGroups() != 0 || getLeaf().getFragmentGroups() != 0 ||
      getLeaf().getLocalBytes() != 0)
    return emitOpError(
        "RVV narrowing logical shift requires coordinate-identical unsigned vectors with doubled input width/LMUL, a result-width vector or input-width scalar shift amount, and exact leaf resources");
  return mlir::success();
}

mlir::LogicalResult RVVWidenMultiplyOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  ValueType result = getResult().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  llvm::StringRef instruction;
  if (lhsElement && rhsElement) {
    if (lhsElement.isUnsigned() && rhsElement.isUnsigned())
      instruction = "rvv.vwmulu.vv";
    else if (lhsElement.isSigned() && rhsElement.isSigned())
      instruction = "rvv.vwmul.vv";
    else if (lhsElement.isSigned())
      instruction = "rvv.vwmulsu.vv";
    else
      instruction = "rvv.vwmulsu.vv.swap";
  }
  const bool sameCoordinates =
      lhs.getShape() == rhs.getShape() && lhs.getShape() == result.getShape() &&
      lhs.getAxisIds() == rhs.getAxisIds() &&
      lhs.getAxisIds() == result.getAxisIds() &&
      lhs.getLayout().getCarrier() == "rvv" &&
      rhs.getLayout().getCarrier() == "rvv" &&
      result.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getAxisIds() == rhs.getLayout().getAxisIds() &&
      lhs.getLayout().getAxisIds() == result.getLayout().getAxisIds() &&
      lhs.getLayout().getTimeFactors() == rhs.getLayout().getTimeFactors() &&
      lhs.getLayout().getTimeFactors() == result.getLayout().getTimeFactors() &&
      lhs.getLayout().getLaneFactors() == rhs.getLayout().getLaneFactors() &&
      lhs.getLayout().getLaneFactors() == result.getLayout().getLaneFactors() &&
      lhs.getLayout().getReplicaFactors() ==
          rhs.getLayout().getReplicaFactors() &&
      lhs.getLayout().getReplicaFactors() ==
          result.getLayout().getReplicaFactors() &&
      lhs.getLayout().getFragmentFactors() ==
          rhs.getLayout().getFragmentFactors() &&
      lhs.getLayout().getFragmentFactors() ==
          result.getLayout().getFragmentFactors() &&
      lhs.getLayout().getLocalFactors() == rhs.getLayout().getLocalFactors() &&
      lhs.getLayout().getLocalFactors() == result.getLayout().getLocalFactors();
  if (!kernel || !lhsElement || !rhsElement || !resultElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      resultElement.isSignless() || lhsElement.getWidth() > 16 ||
      rhsElement.getWidth() > 16 ||
      lhsElement.getWidth() > lhs.getLayout().getSew() ||
      rhsElement.getWidth() > rhs.getLayout().getSew() ||
      resultElement.getWidth() != 2 * lhs.getLayout().getSew() ||
      !sameCoordinates ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      result.getLayout().getSew() !=
          static_cast<int64_t>(resultElement.getWidth()) ||
      rhs.getLayout().getLmulEighths() !=
          lhs.getLayout().getLmulEighths() ||
      result.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
      lhs.getLayout().getVl() != result.getLayout().getVl() ||
      !kernel.getTarget().getHasWideningInteger() ||
      !supportsRVVLayout(kernel.getTarget(), lhs.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), rhs.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), result.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-multiply", instruction, "none",
                 "exact"))
    return emitOpError(
        "RVV widening multiply requires coordinate-identical integer operands and one exact doubled-width result layout");
  return mlir::success();
}

mlir::LogicalResult RVVWidenScalarMultiplyOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType result = getResult().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(getRhs().getType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  llvm::StringRef instruction;
  if (lhsElement && rhsElement) {
    if (lhsElement.isUnsigned() && rhsElement.isUnsigned())
      instruction = "rvv.vwmulu.vx";
    else if (lhsElement.isSigned() && rhsElement.isSigned())
      instruction = "rvv.vwmul.vx";
    else if (lhsElement.isSigned())
      instruction = "rvv.vwmulsu.vx";
  }
  const bool sameCoordinates =
      lhs.getShape() == result.getShape() &&
      lhs.getAxisIds() == result.getAxisIds() &&
      lhs.getLayout().getCarrier() == "rvv" &&
      result.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getAxisIds() == result.getLayout().getAxisIds() &&
      lhs.getLayout().getTimeFactors() ==
          result.getLayout().getTimeFactors() &&
      lhs.getLayout().getLaneFactors() ==
          result.getLayout().getLaneFactors() &&
      lhs.getLayout().getReplicaFactors() ==
          result.getLayout().getReplicaFactors() &&
      lhs.getLayout().getFragmentFactors() ==
          result.getLayout().getFragmentFactors() &&
      lhs.getLayout().getLocalFactors() ==
          result.getLayout().getLocalFactors();
  if (!kernel || !lhsElement || !rhsElement || !resultElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      resultElement.isSignless() || !isScalar(getRhs().getType()) ||
      lhsElement != rhsElement || lhsElement.getWidth() > 16 ||
      lhsElement.getWidth() != lhs.getLayout().getSew() ||
      resultElement.getWidth() != 2 * lhsElement.getWidth() ||
      result.getLayout().getSew() !=
          static_cast<int64_t>(resultElement.getWidth()) ||
      result.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != result.getLayout().getVl() ||
      !sameCoordinates || !kernel.getTarget().getHasWideningInteger() ||
      !supportsRVVLayout(kernel.getTarget(), lhs.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), result.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-scalar-multiply", instruction,
                 "none", "exact"))
    return emitOpError(
        "RVV widening scalar multiply requires one narrow scalar and one exact doubled-width result layout");
  return mlir::success();
}

mlir::LogicalResult RVVMultiplyHighScalarOp::verify() {
  auto lhs = getLhs().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel || !element || !element.isUnsigned() ||
      (element.getWidth() != 8 && element.getWidth() != 16) ||
      getRhs().getType() != element || getResult().getType() != lhs ||
      lhs.getLayout().getCarrier() != "rvv" ||
      lhs.getLayout().getSew() != element.getWidth() ||
      !kernel.getTarget().getHasRVV() ||
      !supportsRVVLayout(kernel.getTarget(), lhs.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "multiply-high-scalar", "rvv.vmulhu.vx",
                 "none", "exact"))
    return emitOpError(
        "RVV unsigned multiply-high requires matching u8/u16 vector and scalar "
        "operands and an unchanged result layout");
  return mlir::success();
}

mlir::LogicalResult RVVRegularRepeatIndexOp::verify() {
  if (getReductionAxis() <= 0 || getRepeat() <= 0 || getResults().empty() ||
      getPartBases().size() != getResults().size())
    return emitOpError(
        "regular-repeat index requires one reduction axis, repeat factor, and base array per result");
  const bool powerOfTwo = (getRepeat() & (getRepeat() - 1)) == 0;
  llvm::StringRef instruction = powerOfTwo ? "rvv.regular-repeat-index.pow2"
                                           : "rvv.regular-repeat-index.div";
  if (!exactLeaf(getLeaf(), "rvv", "regular-repeat-index", instruction,
                 "none", "exact"))
    return emitOpError(
        "regular-repeat index leaf disagrees with its repeat geometry");
  mlir::Type firstType = getResults().front().getType();
  for (auto [resultValue, basesAttribute] :
       llvm::zip(getResults(), getPartBases())) {
    ValueType result = mlir::cast<ValueType>(resultValue.getType());
    auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
    auto bases = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(basesAttribute);
    auto axis = llvm::find(result.getAxisIds().asArrayRef(), getReductionAxis());
    auto timeParts = checkedPositiveProduct(
        result.getLayout().getTimeFactors().asArrayRef());
    if (resultValue.getType() != firstType || !element || !element.isUnsigned() ||
        result.getLayout().getCarrier() != "rvv" ||
        result.getLayout().getSew() != static_cast<int64_t>(element.getWidth()) ||
        axis == result.getAxisIds().asArrayRef().end() || !bases || !timeParts ||
        bases.size() != static_cast<size_t>(*timeParts))
      return emitOpError(
          "regular-repeat index results must share one unsigned RVV coordinate type");
    const size_t position = static_cast<size_t>(
        axis - result.getAxisIds().asArrayRef().begin());
    const int64_t lanes = result.getLayout().getLaneFactors()[position];
    if (lanes <= getRepeat() || result.getShape()[position] !=
                                    lanes * result.getLayout().getTimeFactors()[position])
      return emitOpError(
          "regular-repeat index is only valid when one source relation spans multiple lanes");
    for (int64_t base : bases.asArrayRef())
      if (base < 0)
        return emitOpError("regular-repeat index base must be non-negative");
  }
  return mlir::success();
}

bool weft::riscv::supportsRVVJoinedUnitGather(
    RVVRegularRepeatGatherOp operation) {
  auto access = operation.getAccess();
  auto base = operation.getSourceBase().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto field = operation.getField().getType();
  auto kernel = operation->getParentOfType<KernelOp>();
  const int64_t group = access.getGroupSize();
  const int64_t fields = access.getJoinFields();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  const int64_t width = element ? element.getWidth() : 0;
  const int64_t lowBits = access.getJoinLowBits();
  auto axis = llvm::find(field.getAxisIds().asArrayRef(), operation.getSourceAxis());
  if (access.getMapping() != "joined" || access.getForm() != "indexed" ||
      !base || base.value() != 0 || !kernel || group <= 0 || group > 128 ||
      (group & (group - 1)) != 0 ||
      fields <= 1 || fields > 8 || access.getBitOffset() % 8 ||
      access.getJoinRole() != fields - 1 ||
      !element || !element.isUnsigned() || lowBits <= 0 || lowBits >= width ||
      fields * lowBits > 8 || 2 * width - lowBits > 8 ||
      access.getStorageBits() < (fields + 1) * group * 8 ||
      operation.getSourceCount() != static_cast<uint64_t>(2 * group) ||
      axis == field.getAxisIds().asArrayRef().end() ||
      field.getShape()[axis - field.getAxisIds().asArrayRef().begin()] != 2 * group ||
      operation.getResults().empty())
    return false;
  for (mlir::Value value : operation.getResults()) {
    auto type = mlir::dyn_cast<ValueType>(value.getType());
    auto lanes = rvvLaneCount(type);
    if (!type || type != operation.getResults().front().getType() ||
        !lanes || *lanes <= static_cast<int64_t>(operation.getRepeat()) ||
        *lanes < 2 * group || type.getLayout().getSew() != 8 ||
        type.getLayout().getValidity() != "full" ||
        !supportsRVVLayout(kernel.getTarget(), type.getLayout()) ||
        type.getLayout().getRegisterGroups() > kernel.getTarget().getVectorRegisters() / 4)
      return false;
  }
  return true;
}

mlir::LogicalResult RVVRegularRepeatGatherOp::verify() {
  ValueType field = getField().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  mlir::Type sourceBaseType = elementOf(getSourceBase().getType());
  auto sourceBaseInteger = mlir::dyn_cast<mlir::IntegerType>(sourceBaseType);
  const bool scalarSourceBase =
      isScalar(getSourceBase().getType()) &&
      (sourceBaseType.isIndex() ||
       (sourceBaseInteger && !sourceBaseInteger.isSigned()));
  const int64_t sourceAxis = getSourceAxis();
  const int64_t axis = getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), sourceAxis);
  const bool natural = getAccess().getMapping() == "natural" &&
                       getAccess().getForm() == "unit" &&
                       (fieldElement &&
                        (fieldElement.getWidth() == 8 ||
                         fieldElement.getWidth() == 16 ||
                         fieldElement.getWidth() == 32));
  const bool joined =
      getAccess().getMapping() == "joined" &&
      getAccess().getForm() == "indexed" && fieldElement &&
      fieldElement.isUnsigned() && fieldElement.getWidth() > 0 &&
      fieldElement.getWidth() <= 8 && getAccess().getGroupSize() > 0 &&
      getAccess().getJoinFields() > 1 && getAccess().getJoinLowBits() > 0 &&
      getAccess().getJoinLowBits() < fieldElement.getWidth() &&
      getAccess().getJoinRole() >= 0 &&
      getAccess().getJoinRole() < getAccess().getJoinFields() &&
      (getAccess().getOrder() == "lo_first" ||
       getAccess().getOrder() == "hi_first");
  if (!sourceField || !fieldElement || fieldElement.isSignless() ||
      sourceAxis <= 0 || axis <= 0 ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      getAccess() != sourceField.getAccess() ||
      (!natural && !joined) || getAccess().getBitOffset() % 8 ||
      !scalarSourceBase || getSourceCount() <= 0 || getRepeat() <= 0 ||
      getResults().empty() || getPartBases().size() != getResults().size())
    return emitOpError()
           << "regular-repeat gather requires one natural encoded field and closed load/gather geometry; field="
           << field << ", source-field=" << static_cast<bool>(sourceField)
           << ", element=" << field.getElementType()
           << ", source-axis=" << sourceAxis << ", result-axis=" << axis
           << ", access=" << getAccess() << ", source-base-type="
           << getSourceBase().getType() << ", source-count=" << getSourceCount()
           << ", repeat=" << getRepeat() << ", results=" << getResults().size()
           << ", indices=" << getIndices().size();
  const bool powerOfTwo =
      getRepeat() > 0 && (getRepeat() & (getRepeat() - 1)) == 0;
  auto firstResult = mlir::cast<ValueType>(getResults().front().getType());
  auto firstAxis = llvm::find(firstResult.getAxisIds().asArrayRef(), axis);
  const int64_t firstLanes =
      firstAxis == firstResult.getAxisIds().asArrayRef().end()
          ? 0
          : firstResult.getLayout().getLaneFactors()[static_cast<size_t>(
                firstAxis - firstResult.getAxisIds().asArrayRef().begin())];
  auto ownerEncoding = mlir::dyn_cast<weft::kernel::EncodingType>(
      elementOf(sourceField.getOwner().getType()));
  if (natural && firstLanes > getRepeat() &&
      (localPackInterleaveRows(sourceField.getOwner()) > 1 ||
       derivedInterleaveRows(*this, ownerEncoding) > 1))
    return emitOpError("contiguous regular-repeat words cannot read byte-interleaved records");
  const bool joinedIndexed = joined && firstLanes > getRepeat();
  const bool joinedGroupPowerOfTwo =
      joined && (getAccess().getGroupSize() &
                 (getAccess().getGroupSize() - 1)) == 0;
  llvm::StringRef instruction =
      joinedIndexed
          ? joinedGroupPowerOfTwo
                ? "rvv.regular-repeat-joined-gather.pow2"
                : "rvv.regular-repeat-joined-gather.div"
          : joined ? "rvv.regular-repeat-joined-broadcast"
             : firstLanes <= getRepeat()
                   ? "rvv.regular-repeat-broadcast"
                   : powerOfTwo ? "rvv.regular-repeat-gather.pow2"
                                : "rvv.regular-repeat-gather.div";
  if (getLeaf().getInstruction() == "rvv.regular-repeat-joined-unit") {
    if (!supportsRVVJoinedUnitGather(*this) ||
        getLeaf().getTemporaryGroups() !=
            4 * firstResult.getLayout().getRegisterGroups())
      return emitOpError("joined unit gather requires complete in-bounds byte windows and exact scratch groups");
    instruction = "rvv.regular-repeat-joined-unit";
  }
  if (!exactLeaf(getLeaf(), "rvv", "regular-repeat-gather", instruction,
                 "none", "exact"))
    return emitOpError(
        "regular-repeat gather leaf disagrees with its repeat-index geometry");
  llvm::SmallVector<int64_t> expectedParameters{
      sourceAxis, axis, static_cast<int64_t>(getSourceCount()),
      static_cast<int64_t>(getRepeat())};
  if (instruction == "rvv.regular-repeat-joined-unit") {
    const int64_t width = fieldElement.getWidth();
    const int64_t low = getAccess().getJoinLowBits();
    const int64_t role = getAccess().getOrder() == "lo_first"
                             ? getAccess().getJoinRole()
                             : getAccess().getJoinFields() - 1 - getAccess().getJoinRole();
    expectedParameters.push_back((role + 1) * low == 8 ? 0 : (int64_t{1} << low) - 1);
    expectedParameters.push_back(2 * width - low == 8 ? 0 : (int64_t{1} << (width - low)) - 1);
  }
  if (getLeaf().getParameters().asArrayRef() != llvm::ArrayRef<int64_t>(expectedParameters))
    return emitOpError(
        "regular-repeat gather leaf does not preserve its selected source/result axis relation");

  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - field.getAxisIds().asArrayRef().begin());
  if (field.getShape()[fieldPosition] <= 0 ||
      getSourceCount() > field.getShape()[fieldPosition])
    return emitOpError("regular-repeat gather source window is out of bounds");

  if ((firstLanes <= getRepeat() && !getIndices().empty()) ||
      (firstLanes > getRepeat() &&
       getIndices().size() != getResults().size()))
    return emitOpError(
        "regular-repeat gather must carry typed indices exactly when lanes span repeated source elements");
  for (auto [resultNumber, resultValue] : llvm::enumerate(getResults())) {
    ValueType result = mlir::cast<ValueType>(resultValue.getType());
    auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
    auto bases = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
        getPartBases()[resultNumber]);
    auto timeParts = checkedPositiveProduct(
        result.getLayout().getTimeFactors().asArrayRef());
    auto registerParts = checkedPositiveProduct(
        result.getLayout().getReplicaFactors().asArrayRef());
    if (!bases || !timeParts || !registerParts ||
        bases.size() != static_cast<size_t>(*timeParts) ||
        resultAxis == result.getAxisIds().asArrayRef().end() ||
        result.getElementType() != field.getElementType() ||
        result.getLayout().getCarrier() != "rvv" ||
        result.getLayout().getSew() !=
            static_cast<int64_t>(std::max<unsigned>(8, fieldElement.getWidth())) ||
        *registerParts <= 0)
      return emitOpError(
          "regular-repeat gather result coordinates are incomplete");
    const size_t resultPosition = static_cast<size_t>(
        resultAxis - result.getAxisIds().asArrayRef().begin());
    const int64_t lanes = result.getLayout().getLaneFactors()[resultPosition];
    if (lanes <= 1 || result.getLayout().getVl() < lanes ||
        result.getShape()[resultPosition] !=
            lanes * result.getLayout().getTimeFactors()[resultPosition] ||
        (getRepeat() % lanes != 0 && lanes % getRepeat() != 0))
      return emitOpError(
          "regular-repeat gather result does not have one complete lane/time projection");
    for (int64_t base : bases.asArrayRef())
      if (base < 0 ||
          base + ((lanes - 1) / getRepeat()) >= getSourceCount())
        return emitOpError(
            "regular-repeat gather index addresses outside its selected source window");
    if (firstLanes <= getRepeat())
      continue;
    mlir::Value index = getIndices()[resultNumber];
    ValueType indexType = mlir::cast<ValueType>(index.getType());
    auto indexElement =
        mlir::dyn_cast<mlir::IntegerType>(indexType.getElementType());
    auto indexOp = index.getDefiningOp<RVVRegularRepeatIndexOp>();
    if (!indexElement || !indexElement.isUnsigned() ||
        indexType.getShape() != result.getShape() ||
        indexType.getAxisIds() != result.getAxisIds() ||
        indexType.getLayout() != result.getLayout() || !indexOp ||
        indexOp.getReductionAxis() != axis ||
        indexOp.getRepeat() != getRepeat())
      return emitOpError(
          "regular-repeat gather typed index does not match its result coordinates");
    auto indexBases = mlir::cast<mlir::DenseI64ArrayAttr>(
        indexOp.getPartBases()[
            mlir::cast<mlir::OpResult>(index).getResultNumber()]);
    if (indexBases != bases)
      return emitOpError(
          "regular-repeat gather index bases disagree with the selected source window");
  }
  return mlir::success();
}

mlir::LogicalResult RVVRegularRepeatScalarLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  mlir::Type sourceBaseType = elementOf(getSourceBase().getType());
  auto sourceBaseInteger = mlir::dyn_cast<mlir::IntegerType>(sourceBaseType);
  const bool scalarSourceBase =
      isScalar(getSourceBase().getType()) &&
      (sourceBaseType.isIndex() ||
       (sourceBaseInteger && !sourceBaseInteger.isSigned()));
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), getSourceAxis());
  auto resultAxis =
      llvm::find(result.getAxisIds().asArrayRef(), getReductionAxis());
  const bool natural = getAccess().getMapping() == "natural" &&
                       getAccess().getForm() == "unit" && fieldElement &&
                       (fieldElement.getWidth() == 8 ||
                        fieldElement.getWidth() == 16 ||
                        fieldElement.getWidth() == 32);
  const bool joined =
      getAccess().getMapping() == "joined" &&
      getAccess().getForm() == "indexed" && fieldElement &&
      fieldElement.isUnsigned() && fieldElement.getWidth() > 0 &&
      fieldElement.getWidth() <= 8 && getAccess().getGroupSize() > 0 &&
      getAccess().getJoinFields() > 1 && getAccess().getJoinLowBits() > 0 &&
      getAccess().getJoinLowBits() < fieldElement.getWidth() &&
      getAccess().getJoinRole() >= 0 &&
      getAccess().getJoinRole() < getAccess().getJoinFields() &&
      (getAccess().getOrder() == "lo_first" ||
       getAccess().getOrder() == "hi_first");
  auto timeParts = checkedPositiveProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  auto replicas = checkedPositiveProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  const auto partBases = getPartBases();
  if (!sourceField || !fieldElement || fieldElement.isSignless() ||
      getSourceAxis() <= 0 || getReductionAxis() <= 0 ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      getAccess() != sourceField.getAccess() ||
      (!natural && !joined) || getAccess().getBitOffset() % 8 ||
      !scalarSourceBase || getSourceCount() <= 0 || getRepeat() <= 0 ||
      result.getElementType() != field.getElementType() ||
      result.getLayout().getCarrier() != "scalar" || !timeParts || !replicas ||
      *timeParts > std::numeric_limits<int64_t>::max() / *replicas ||
      partBases.size() != static_cast<size_t>(*timeParts * *replicas) ||
      !llvm::all_of(result.getLayout().getFragmentFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) ||
      !llvm::all_of(result.getLayout().getLocalFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }))
    return emitOpError(
        "regular-repeat scalar load requires one typed field window and one source base per scalar physical part");
  const size_t resultPosition = static_cast<size_t>(
      resultAxis - result.getAxisIds().asArrayRef().begin());
  const int64_t lanes = result.getLayout().getLaneFactors()[resultPosition];
  const bool flatWindow = natural && getRepeat() == 1 &&
      result.getShape().size() > 1;
  if (getSourceCount() >
      std::numeric_limits<int64_t>::max() / getRepeat())
    return emitOpError("regular-repeat scalar load logical extent overflows");
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    const bool validExtent =
        flatWindow ? result.getShape()[position] > 0 : position == resultPosition
            ? result.getShape()[position] == getSourceCount() * getRepeat()
            : result.getShape()[position] <= 0 ||
                  result.getShape()[position] == 1;
    if (!validExtent)
      return emitOpError()
             << "regular-repeat scalar load may preserve only singleton non-source axes; result="
             << result << ", reduction-position=" << resultPosition
             << ", source-count=" << getSourceCount()
             << ", repeat=" << getRepeat();
  }
  const int64_t sourceTime =
      result.getLayout().getTimeFactors()[resultPosition];
  const int64_t sourceReplicas =
      result.getLayout().getReplicaFactors()[resultPosition];
  llvm::SmallVector<int64_t> expectedPartBases;
  if (flatWindow) {
    auto extent = checkedPositiveProduct(result.getShape().asArrayRef());
    auto coordinates = RegisterCoordinates::get(result.getLayout());
    if (!extent || *extent != getSourceCount() || !coordinates ||
        result.getLayout().getValidity() != "full" ||
        coordinates->lanes() != 1 || coordinates->parts() != *extent ||
        !llvm::equal(coordinates->extents(), result.getShape().asArrayRef()))
      return emitOpError("scalar storage window must cover its complete logical domain");
    for (int64_t part = 0; part < coordinates->parts(); ++part) {
      auto logical = coordinates->logical(part);
      if (!logical) return emitOpError("scalar storage window has no part coordinate");
      int64_t offset = 0;
      for (auto [coordinate, size] : llvm::zip(*logical, result.getShape().asArrayRef()))
        offset = offset * size + coordinate;
      expectedPartBases.push_back(offset);
    }
  } else if (sourceReplicas == 1 && *replicas == 1 &&
      *timeParts == sourceTime && lanes > 0 &&
      lanes <= getRepeat() && getRepeat() % lanes == 0) {
    expectedPartBases.reserve(sourceTime);
    for (int64_t stream = 0; stream < sourceTime; ++stream)
      expectedPartBases.push_back(stream * lanes / getRepeat());
  } else if (sourceTime == 1 && *timeParts == 1 && lanes == 1 &&
             sourceReplicas == getSourceCount() * getRepeat() &&
             *replicas == sourceReplicas) {
    expectedPartBases.reserve(sourceReplicas);
    for (int64_t replica = 0; replica < sourceReplicas; ++replica)
      expectedPartBases.push_back(replica / getRepeat());
  } else {
    return emitOpError(
        "regular-repeat scalar load supports one scalar per aligned lane window or one scalar per logical replica");
  }
  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - field.getAxisIds().asArrayRef().begin());
  if (lanes <= 0 || lanes > getRepeat() || getRepeat() % lanes ||
      field.getShape()[fieldPosition] <= 0 ||
      getSourceCount() > field.getShape()[fieldPosition] ||
      llvm::any_of(partBases, [&](int64_t base) {
        return base < 0 || base >= getSourceCount();
      }) ||
      !llvm::equal(partBases, expectedPartBases) ||
      !exactLeaf(getLeaf(), "scalar", "regular-repeat-scalar-load",
                 "scalar.regular-repeat-load", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({static_cast<int64_t>(getSourceAxis()),
                                   static_cast<int64_t>(getReductionAxis()),
                                   static_cast<int64_t>(getSourceCount()),
                                   static_cast<int64_t>(getRepeat())}))
    return emitOpError(
        "regular-repeat scalar load field bounds or selected leaf are incomplete");
  return mlir::success();
}

mlir::LogicalResult RVVStorageWindowOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  StorageWindowPlanAttr plan = getPlan();
  const int64_t axis = plan.getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
  auto pointDomain = getOrigin().getType().getDomain();
  auto pointAxis = pointDomain.getAxisId();
  LoadOp load = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  MemDescType memory = load ? load.getRegion().getType() : MemDescType();
  auto timeParts = checkedPositiveProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  llvm::StringRef mapping = getAccess().getMapping();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  llvm::StringRef instruction =
      mapping == "grouped_layered" ? "rvv.storage-window.layered"
                                    : "rvv.storage-window.natural";
  if (!sourceField || !memory ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      !getLogicalOffset().getType().isIndex() || axis <= 0 || pointAxis != axis ||
      field.getElementType() != result.getElementType() ||
      field.getAxisIds() != result.getAxisIds() ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      pointDomain.getTail() != "exact" ||
      result.getLayout().getCarrier() != "rvv" || !timeParts || *timeParts <= 0 ||
      (mapping != "natural" && mapping != "grouped_layered") ||
      (validity != "full" && validity != "tail") ||
      plan.getRecordElements() != memory.getElements() ||
      getAccess().getBitOffset() % 8 ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() !=
          static_cast<int64_t>(elementBitWidth(field.getElementType())) ||
      !exactLeaf(getLeaf(), "rvv", "storage-window", instruction,
                 "none", tail))
    return emitOpError()
           << "RVV storage window requires one typed field/point/offset edge and "
              "one complete lane result; field="
           << field << ", result=" << result << ", axis=" << axis
           << ", point-axis=" << pointAxis << ", access=" << getAccess()
           << ", field-access="
           << (sourceField ? sourceField.getAccess() : AccessAttr())
           << ", point-tail=" << pointDomain.getTail() << ", leaf=" << getLeaf()
           << ", plan=" << plan;
  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - field.getAxisIds().asArrayRef().begin());
  const size_t resultPosition = static_cast<size_t>(
      resultAxis - result.getAxisIds().asArrayRef().begin());
  const int64_t lanes =
      result.getLayout().getLaneFactors()[resultPosition];
  const int64_t projectionBase = plan.getProjectionBase();
  const int64_t projectionStride = plan.getProjectionStride();
  const int64_t projectionRepeat = plan.getProjectionRepeat();
  const int64_t projectionExtent = plan.getProjectionExtent();
  const int64_t offsetAlignment = plan.getOffsetAlignment();
  const bool projectionBounds =
      projectionBase >= 0 && projectionStride > 0 && projectionRepeat > 0 &&
      projectionExtent > 0 && field.getShape()[fieldPosition] > 0 &&
      projectionBase <= field.getShape()[fieldPosition] - 1 &&
      (projectionExtent - 1) / projectionRepeat <=
          (field.getShape()[fieldPosition] - 1 - projectionBase) /
              projectionStride;
  if (lanes <= 0 || !projectionBounds ||
      projectionExtent < lanes * *timeParts ||
      offsetAlignment != lanes ||
      (projectionRepeat % lanes != 0 && lanes % projectionRepeat != 0) ||
      result.getShape()[resultPosition] !=
          result.getLayout().getLaneFactors()[resultPosition] *
              result.getLayout().getTimeFactors()[resultPosition] ||
      result.getLayout().getVl() < lanes)
    return emitOpError()
           << "RVV storage window result must cover one bounded, lane-aligned "
              "projected subrange; field="
           << field << ", result=" << result << ", lanes=" << lanes
           << ", projection=<" << projectionBase << "," << projectionStride
           << "," << projectionRepeat << "," << projectionExtent
           << ">, offset-alignment=" << offsetAlignment;
  if (mapping == "grouped_layered") {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
    const int64_t group = getAccess().getGroupSize();
    const int64_t layer = getAccess().getLayerSize();
    const int64_t layers = layer > 0 ? group / layer : 0;
    const int64_t physicalLayerBase =
        getAccess().getOrder() == "lo_first" ? 0 : layers - 1;
    const int64_t physicalLayerStep =
        getAccess().getOrder() == "lo_first" ? 1 : -1;
    if (!integer || integer.isSigned() || integer.getWidth() <= 0 ||
        integer.getWidth() >= 8 || group <= 0 || layer <= 0 || group % layer ||
        layer % lanes || getAccess().getBitOffset() % 8 ||
        plan.getKind() != "layered" || plan.getGroupSize() != group ||
        plan.getLayerSize() != layer ||
        plan.getPhysicalLayerBase() != physicalLayerBase ||
        plan.getPhysicalLayerStep() != physicalLayerStep ||
        plan.getShiftBase() != physicalLayerBase * integer.getWidth() ||
        plan.getShiftStep() != physicalLayerStep * integer.getWidth() ||
        plan.getMaskValue() != (int64_t{1} << integer.getWidth()) - 1 ||
        projectionBase != 0 || projectionStride != 1 || projectionRepeat != 1 ||
        projectionExtent != lanes * *timeParts)
      return emitOpError(
          "layered RVV storage window requires one byte-aligned contiguous lane window inside a storage layer");
  } else {
    llvm::StringRef expectedKind =
        projectionRepeat > 1
            ? "repeat"
            : projectionStride == 1 ? "unit" : "strided";
    if (plan.getKind() != expectedKind)
      return emitOpError(
          "natural RVV storage window plan disagrees with its selected access form");
  }
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    if (position == resultPosition)
      continue;
    if (field.getShape()[position] != result.getShape()[position] ||
        result.getLayout().getLaneFactors()[position] != 1 ||
        result.getLayout().getTimeFactors()[position] != 1 ||
        (result.getShape()[position] > 0
             ? result.getLayout().getReplicaFactors()[position] !=
                   result.getShape()[position]
             : result.getLayout().getReplicaFactors()[position] <= 0) ||
        result.getLayout().getFragmentFactors()[position] != 1 ||
        result.getLayout().getLocalFactors()[position] != 1)
      return emitOpError(
          "RVV storage window must preserve every free axis as register replicas");
  }
  return mlir::success();
}

mlir::LogicalResult RVVLayeredRecordLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto physicalPoint = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto pointPartition =
      physicalPoint
          ? physicalPoint.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
          : mlir::arith::ConstantIndexOp();
  StorageWindowPlanAttr plan = getPlan();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  LoadOp load = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  MemDescType memory = load ? load.getRegion().getType() : MemDescType();
  const int64_t reductionAxis = plan ? plan.getReductionAxis() : 0;
  auto fieldReduction =
      llvm::find(field.getAxisIds().asArrayRef(), reductionAxis);
  auto resultReduction =
      llvm::find(result.getAxisIds().asArrayRef(), reductionAxis);
  int64_t laneAxis = 0;
  int64_t lanes = 0;
  for (auto [axis, factor] : llvm::zip(
           result.getAxisIds().asArrayRef(),
           result.getLayout().getLaneFactors().asArrayRef())) {
    if (factor <= 1)
      continue;
    if (laneAxis != 0)
      return emitOpError(
          "layered record load result must have exactly one SIMD lane axis");
    laneAxis = axis;
    lanes = factor;
  }
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || !physicalPoint || !pointPartition || !plan || !memory ||
      !integer || integer.isSigned() || integer.getWidth() <= 0 ||
      integer.getWidth() >= 8 || !getLogicalOffset().getType().isIndex() ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      getAccess().getMapping() != "grouped_layered" ||
      (getAccess().getForm() != "strided" &&
       getAccess().getForm() != "indexed") ||
      getOrigin().getType().getDomain().getAxisId() != reductionAxis ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      fieldReduction == field.getAxisIds().asArrayRef().end() ||
      resultReduction != result.getAxisIds().asArrayRef().end() ||
      field.getElementType() != result.getElementType() ||
      field.getShape().size() != result.getShape().size() + 1 ||
      result.getLayout().getCarrier() != "rvv" || laneAxis <= 0 || lanes <= 1 ||
      result.getLayout().getSew() != 8 ||
      (validity != "full" && validity != "tail") ||
      plan.getKind() != "layered" || plan.getProjectionBase() != 0 ||
      plan.getProjectionStride() != 1 || plan.getProjectionRepeat() != 1 ||
      plan.getProjectionExtent() <= 0 || plan.getOffsetAlignment() != 1 ||
      pointPartition.value() != plan.getProjectionExtent() ||
      plan.getRecordElements() != memory.getElements() ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() != integer.getWidth() ||
      plan.getGroupSize() != getAccess().getGroupSize() ||
      plan.getLayerSize() != getAccess().getLayerSize() ||
      plan.getGroupSize() <= 0 || plan.getLayerSize() <= 0 ||
      plan.getGroupSize() % plan.getLayerSize() ||
      integer.getWidth() * (plan.getGroupSize() / plan.getLayerSize()) != 8 ||
      getAccess().getBitOffset() % 8 ||
      !exactLeaf(getLeaf(), "rvv", "layered-record-load",
                 "rvv.layered-record-load", "none", tail))
    return emitOpError(
        "layered record load requires one typed reduction subview and one independent record-lane result");

  const size_t reductionPosition = static_cast<size_t>(
      fieldReduction - field.getAxisIds().asArrayRef().begin());
  if (field.getShape()[reductionPosition] < plan.getProjectionExtent())
    return emitOpError(
        "layered record load projection exceeds the encoded field extent");
  size_t resultPosition = 0;
  for (size_t fieldPosition = 0; fieldPosition < field.getShape().size();
       ++fieldPosition) {
    if (fieldPosition == reductionPosition)
      continue;
    if (resultPosition >= result.getShape().size() ||
        field.getAxisIds()[fieldPosition] !=
            result.getAxisIds()[resultPosition] ||
        field.getShape()[fieldPosition] != result.getShape()[resultPosition])
      return emitOpError(
          "layered record load must preserve every non-reduction logical axis");
    ++resultPosition;
  }
  if (resultPosition != result.getShape().size())
    return emitOpError(
        "layered record load result has an extra logical axis");
  auto lane = llvm::find(result.getAxisIds().asArrayRef(), laneAxis);
  const size_t lanePosition = static_cast<size_t>(
      lane - result.getAxisIds().asArrayRef().begin());
  if (result.getShape()[lanePosition] !=
          result.getLayout().getTimeFactors()[lanePosition] * lanes ||
      result.getLayout().getReplicaFactors()[lanePosition] != 1 ||
      result.getLayout().getFragmentFactors()[lanePosition] != 1 ||
      result.getLayout().getLocalFactors()[lanePosition] != 1 ||
      result.getLayout().getVl() < lanes)
    return emitOpError(
        "layered record load lane axis must cover its result through time and SIMD lanes");
  return mlir::success();
}

mlir::LogicalResult RVVLayeredStorageLoadOp::verify() {
  ValueType field = getField().getType();
  LayeredWindowType window = getResult().getType();
  ValueType result = window.getResultType();
  StorageWindowPlanAttr plan = getPlan();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  const int64_t group = plan ? plan.getGroupSize() : 0;
  const int64_t layer = plan ? plan.getLayerSize() : 0;
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto physicalPoint = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto pointPartition =
      physicalPoint
          ? physicalPoint.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
          : mlir::arith::ConstantIndexOp();
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(),
                               plan ? plan.getReductionAxis() : 0);
  const int64_t lanes =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getLaneFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(),
                              plan ? plan.getReductionAxis() : 0);
  const int64_t fieldExtent =
      fieldAxis == field.getAxisIds().asArrayRef().end()
          ? 0
          : field.getShape()[static_cast<size_t>(
                fieldAxis - field.getAxisIds().asArrayRef().begin())];
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || getAccess() != sourceField.getAccess() || !plan || !integer ||
      integer.isSigned() || getOrigin().getType().getDomain().getAxisId() !=
                                plan.getReductionAxis() ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      !getWindowIndex().getType().isIndex() ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      layer <= 0 || group % layer || layers <= 1 || lanes <= 1 ||
      !pointPartition || pointPartition.value() % group ||
      layer % lanes || integer.getWidth() * layers != 8 ||
      plan.getKind() != "layered" ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() != integer.getWidth() ||
      plan.getGroupSize() != getAccess().getGroupSize() ||
      plan.getLayerSize() != getAccess().getLayerSize() ||
      plan.getProjectionBase() < 0 || plan.getProjectionExtent() <= 0 ||
      plan.getProjectionBase() % group || plan.getProjectionExtent() % group ||
      fieldExtent <= 0 || plan.getProjectionBase() > fieldExtent ||
      plan.getProjectionExtent() > fieldExtent - plan.getProjectionBase() ||
      getAccess().getBitOffset() % 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      window.getFieldType() != field || window.getReductionAxis() !=
                                            plan.getReductionAxis() ||
      window.getLayers() != layers ||
      window.getWindowsPerLayer() != layer / lanes ||
      (!exactLeaf(getLeaf(), "rvv", "layered-storage-load",
                  "rvv.layered-storage-load", "none", tail) &&
       !exactLeaf(getLeaf(), "rvv", "layered-storage-load",
                  "rvv.layered-storage-load.scalar-prime", "none", tail)))
    return emitOpError(
        "layered storage load requires one grouped/layered field and closed byte-window geometry");
  return mlir::success();
}

mlir::LogicalResult RVVLayeredStorageDecodeOp::verify() {
  LayeredWindowType window = getWindow().getType();
  ValueType result = getResult().getType();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  auto load = getWindow().getDefiningOp<RVVLayeredStorageLoadOp>();
  StorageWindowPlanAttr plan = load ? load.getPlan() : StorageWindowPlanAttr();
  const int64_t expectedPhysicalLayer =
      plan ? plan.getPhysicalLayerBase() +
                 getLayer() * plan.getPhysicalLayerStep()
           : -1;
  auto integer =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  const int64_t width = integer ? integer.getWidth() : 0;
  const int64_t expectedShift =
      plan ? plan.getShiftBase() + getLayer() * plan.getShiftStep() : -1;
  const int64_t expectedMask =
      integer && (expectedPhysicalLayer + 1) * width != 8
          ? (int64_t(1) << width) - 1
          : 0;
  llvm::StringRef expectedInstruction =
      expectedShift > 0
          ? (expectedMask > 0 ? "rvv.layered-storage-decode.shift-mask"
                              : "rvv.layered-storage-decode.shift")
          : (expectedMask > 0 ? "rvv.layered-storage-decode.mask"
                              : "rvv.layered-storage-decode.identity");
  if (getLayer() < 0 || getLayer() >= window.getLayers() || !load || !plan ||
      plan.getKind() != "layered" ||
      getPhysicalLayer() != expectedPhysicalLayer ||
      !integer || integer.isSigned() || width <= 0 || width > 8 ||
      width != plan.getElementBits() ||
      getShiftAmount() != expectedShift || getMaskValue() != expectedMask ||
      result != window.getResultType() ||
      !exactLeaf(getLeaf(), "rvv", "layered-storage-decode",
                 expectedInstruction, "none", tail))
    return emitOpError(
        "layered storage decode requires one selected layer of its typed window");
  return mlir::success();
}

mlir::LogicalResult RVVReplicaStorageLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  if (!sourceField)
    return emitOpError("requires its original encoded field projection as the storage source");
  auto storageLoad = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  auto memory = storageLoad ? storageLoad.getRegion().getType() : MemDescType();
  mlir::Type fieldElement = field.getElementType();
  auto fieldInteger = mlir::dyn_cast<mlir::IntegerType>(fieldElement);
  const int64_t fieldBits = elementBitWidth(fieldElement);
  mlir::Type baseType = getLogicalBase().getType();
  const bool scalarBase = baseType.isIndex() || mlir::isa<mlir::IntegerType>(baseType);
  StorageWindowPlanAttr plan = getPlan();
  if (!plan)
    return emitOpError(
        "replica storage load requires one selected storage-window plan");
  const int64_t laneAxis = plan.getReductionAxis();
  const int64_t logicalSpan = plan.getProjectionExtent();
  const int64_t logicalBaseMultiple = plan.getOffsetAlignment();
  const int64_t recordRank = getRecordRank();
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), laneAxis);
  auto parts = physicalPartCount(result);
  const bool natural = plan.getKind() == "unit";
  const bool strided = plan.getKind() == "strided";
  const bool layered = plan.getKind() == "layered";
  const bool interleavedNatural = plan.getKind() == "interleaved_natural";
  const bool interleavedJoined = plan.getKind() == "interleaved_joined";
  const bool interleaved = interleavedNatural || interleavedJoined;
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  const int64_t layers = layered && layer > 0 ? group / layer : 1;
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  llvm::StringRef instruction = natural
                                    ? "rvv.replica-storage-load.natural"
                                : strided
                                    ? "rvv.replica-storage-load.strided"
                                : layered
                                    ? "rvv.replica-storage-load.layered"
                                : interleavedNatural
                                    ? "rvv.replica-storage-load.interleaved-natural"
                                    : "rvv.replica-storage-load.interleaved-joined";
  auto ownerEncoding = mlir::dyn_cast<weft::kernel::EncodingType>(
      sourceField ? elementOf(sourceField.getOwner().getType()) : mlir::Type());
  int64_t interleave =
      sourceField ? localPackInterleaveRows(sourceField.getOwner()) : 0;
  if (interleave <= 0 && sourceField)
    interleave = derivedInterleaveRows(*this, ownerEncoding);
  const bool exactBaseLeaf =
      exactLeaf(getLeaf(), "rvv", "replica-storage-load", instruction,
                "none", tail);
  const bool exactScalarPrimeLeaf = exactLeaf(
      getLeaf(), "rvv", "replica-storage-load",
      (instruction + ".scalar-prime").str(), "none", tail);
  if (!sourceField || !storageLoad || !memory ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      fieldBits <= 0 || (fieldInteger && fieldInteger.isSignless()) || !scalarBase ||
      laneAxis <= 0 || plan.getProjectionBase() != 0 ||
      ((natural || layered || interleaved) &&
       plan.getProjectionStride() != 1) ||
      (strided && plan.getProjectionStride() <= 1) ||
      plan.getProjectionRepeat() != 1 ||
      plan.getRecordElements() != memory.getElements() ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() != fieldBits ||
      ((natural || strided) && getAccess().getMapping() != "natural") ||
      (layered && getAccess().getMapping() != "grouped_layered") ||
      (interleavedNatural && getAccess().getMapping() != "natural") ||
      (interleavedJoined && getAccess().getMapping() != "joined") ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      field.getElementType() != result.getElementType() ||
      result.getLayout().getCarrier() != "rvv" || !parts || *parts <= 0 ||
      logicalBaseMultiple <= 0 || logicalSpan <= 0 ||
      getWindowOffsets().empty() ||
      recordRank < 0 || recordRank > static_cast<int64_t>(field.getShape().size()) ||
      getRecordCoordinatesForWindow().size() !=
          getWindowOffsets().size() * static_cast<size_t>(recordRank) ||
      getWindowForPart().size() != static_cast<size_t>(*parts) ||
      getLayerForPart().size() != static_cast<size_t>(*parts) ||
      getPhysicalLayerForPart().size() != static_cast<size_t>(*parts) ||
      getShiftOffsetForPart().size() != static_cast<size_t>(*parts) ||
      getShiftBaseFactorForPart().size() != static_cast<size_t>(*parts) ||
      getMaskValueForPart().size() != static_cast<size_t>(*parts) ||
      (!natural && !strided && !layered && !interleaved) ||
      (validity != "full" && validity != "tail") ||
      (!exactBaseLeaf && !exactScalarPrimeLeaf) ||
      (exactScalarPrimeLeaf &&
       (!layered || getWindowOffsets().size() != 1)))
    return emitOpError()
           << "replica storage load requires one scalar base and a closed field-to-register window map; field="
           << field << ", result=" << result << ", source_field="
           << static_cast<bool>(sourceField) << ", access=" << getAccess()
           << ", source_access="
           << (sourceField ? sourceField.getAccess() : AccessAttr())
           << ", storage_load=" << static_cast<bool>(storageLoad)
           << ", memory=" << memory << ", interleave=" << interleave
           << ", result_axis="
           << (resultAxis == result.getAxisIds().asArrayRef().end()
                   ? -1
                   : static_cast<int64_t>(resultAxis -
                                          result.getAxisIds().asArrayRef().begin()))
           << ", plan=" << plan
           << ", window_offsets=" << getWindowOffsets()
           << ", record_rank=" << recordRank
           << ", record_coordinates_for_window="
           << getRecordCoordinatesForWindow()
           << ", window_for_part=" << getWindowForPart()
           << ", layer_for_part=" << getLayerForPart()
           << ", physical_layer_for_part=" << getPhysicalLayerForPart()
           << ", shift_offset_for_part=" << getShiftOffsetForPart()
           << ", shift_base_factor_for_part=" << getShiftBaseFactorForPart()
           << ", mask_value_for_part=" << getMaskValueForPart()
           << ", parts=" << (parts ? *parts : -1)
           << ", base_type=" << baseType << ", leaf=" << getLeaf();

  const size_t lanePosition = static_cast<size_t>(
      resultAxis - result.getAxisIds().asArrayRef().begin());
  const int64_t primaryLanes =
      result.getLayout().getLaneFactors()[lanePosition];
  auto physicalLanes =
      checkedPositiveProduct(result.getLayout().getLaneFactors().asArrayRef());
  if (primaryLanes <= 0 || !physicalLanes || *physicalLanes <= 0 ||
      result.getShape()[lanePosition] !=
          primaryLanes * result.getLayout().getTimeFactors()[lanePosition] ||
      result.getLayout().getReplicaFactors()[lanePosition] != 1 ||
      result.getLayout().getFragmentFactors()[lanePosition] != 1 ||
      result.getLayout().getLocalFactors()[lanePosition] != 1)
    return emitOpError(
        "replica storage load lane axis must be one complete SIMD window");
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    const int64_t time = result.getLayout().getTimeFactors()[position];
    const int64_t lane = result.getLayout().getLaneFactors()[position];
    const int64_t replica = result.getLayout().getReplicaFactors()[position];
    const int64_t axis = result.getAxisIds()[position];
    auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
    const bool preservedFieldAxis =
        fieldAxis != field.getAxisIds().asArrayRef().end() &&
        field.getShape()[static_cast<size_t>(
            fieldAxis - field.getAxisIds().asArrayRef().begin())] ==
            result.getShape()[position];
    int64_t factors[] = {time, lane, replica};
    auto represented = checkedPositiveProduct(factors);
    if ((lane > 1 && replica != 1) ||
        (time > 1 && replica > 1) || !represented ||
        (result.getShape()[position] > 0 &&
         *represented != result.getShape()[position]) ||
        (result.getShape()[position] <= 0 &&
         (validity != "tail" ||
          (*represented != 1 && !preservedFieldAxis))) ||
        result.getLayout().getFragmentFactors()[position] != 1 ||
        result.getLayout().getLocalFactors()[position] != 1)
      return emitOpError(
          "replica storage load must preserve every logical axis as lane, issue time, or register replicas");
  }
  auto streams = checkedPositiveProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  if (!streams || *streams <= 0)
    return emitOpError(
        "replica storage load must define a positive issue-time decomposition");
  if (*parts % *streams)
    return emitOpError(
        "replica storage load has an incomplete stream/replica product");
  auto loadLanes = checkedPositiveProduct(
      result.getLayout().getLaneFactors().asArrayRef());
  if (!loadLanes || *loadLanes <= 0 ||
      result.getLayout().getVl() != *loadLanes)
    return emitOpError(
        "replica storage load must use exactly its typed load-lane extent as vl");
  for (size_t window = 0; window < getWindowOffsets().size(); ++window) {
    for (int64_t position = 0; position < recordRank; ++position) {
      const int64_t coordinate = getRecordCoordinatesForWindow()[
          window * static_cast<size_t>(recordRank) + static_cast<size_t>(position)];
      const int64_t extent = field.getShape()[static_cast<size_t>(position)];
      int64_t width = 1;
      auto mapped = llvm::find(result.getAxisIds().asArrayRef(),
                               field.getAxisIds()[static_cast<size_t>(position)]);
      if (mapped != result.getAxisIds().asArrayRef().end())
        width = result.getLayout().getLaneFactors()[static_cast<size_t>(
            mapped - result.getAxisIds().asArrayRef().begin())];
      if (coordinate < 0 || width <= 0 ||
          (extent > 0 && coordinate > extent - width))
        return emitOpError()
               << "replica storage load window has an out-of-range record coordinate; window="
               << window << ", position=" << position
               << ", coordinate=" << coordinate << ", width=" << width
               << ", extent=" << extent << ", record_rank=" << recordRank
               << ", field=" << field << ", result=" << result;
    }
  }
  for (int64_t window : getWindowForPart())
    if (window < 0 ||
        window >= static_cast<int64_t>(getWindowOffsets().size()))
      return emitOpError(
          "replica storage load part references an invalid source window");

  if (interleaved) {
    if (interleave <= 0 || recordRank != 1 ||
        result.getAxisIds()[lanePosition] != field.getAxisIds()[0] ||
        result.getShape()[lanePosition] > interleave ||
        logicalSpan != result.getShape()[lanePosition] ||
        (getAccess().getForm() != "strided" &&
         getAccess().getForm() != "indexed") ||
        (interleavedNatural && fieldBits != 8 && fieldBits != 16 &&
         fieldBits != 32) ||
        (interleavedJoined &&
         (!fieldInteger || !fieldInteger.isUnsigned() || fieldBits > 8 ||
          getAccess().getGroupSize() <= 0 || getAccess().getJoinFields() <= 1 ||
          getAccess().getJoinLowBits() <= 0 ||
          getAccess().getJoinLowBits() >= fieldBits ||
          getAccess().getJoinRole() < 0 ||
          getAccess().getJoinRole() >= getAccess().getJoinFields() ||
          (getAccess().getOrder() != "lo_first" &&
           getAccess().getOrder() != "hi_first"))))
      return emitOpError(
          "interleaved replica storage load requires one explicit record-axis projection and closed byte assembly");
  } else if (natural || strided) {
    if ((natural && getAccess().getForm() != "indexed") ||
        (strided && getAccess().getForm() != "strided") ||
        getAccess().getBitOffset() % 8 ||
        !fieldInteger ||
        (fieldBits != 8 && fieldBits != 16 && fieldBits != 32) ||
        result.getLayout().getSew() != fieldBits)
      return emitOpError(
          "natural or strided replica storage load requires one byte-addressable source relation");
  } else {
    if (getAccess().getForm() != "indexed" || group <= 0 || layer <= 0 ||
        group % layer || layers <= 1 || layer % *physicalLanes ||
        !fieldInteger || fieldInteger.isSigned() || fieldBits <= 0 ||
        layers > 8 / fieldBits ||
        (logicalSpan > group && (8 % fieldBits || layers != 8 / fieldBits)) ||
        result.getLayout().getSew() != 8 || getAccess().getBitOffset() % 8 ||
        (getAccess().getOrder() != "lo_first" &&
         getAccess().getOrder() != "hi_first") ||
        (logicalSpan <= group ? group % logicalSpan : logicalSpan % group) ||
        (logicalSpan > group && logicalBaseMultiple % group))
      return emitOpError(
          "layered replica storage load requires closed grouped/layered byte geometry");
  }


  for (int64_t offset : getWindowOffsets()) {
    const bool bounded = natural || strided || interleaved
                             ? offset >= 0 &&
                                   offset + (*loadLanes - 1) *
                                                plan.getProjectionStride() <
                                       logicalSpan
                             : offset >= 0 &&
                                   offset % layer <= layer - *physicalLanes &&
                                   offset <= (logicalSpan > group ? logicalSpan / layers : layer) -
                                                 *physicalLanes;
    if (!bounded)
      return emitOpError("replica storage load window offset is out of bounds");
  }
  for (auto [window, storageLayer] :
       llvm::zip(getWindowForPart(), getLayerForPart())) {
    if (window < 0 || window >= static_cast<int64_t>(getWindowOffsets().size()) ||
        storageLayer < 0 || storageLayer >= layers ||
        ((natural || strided || interleaved) && storageLayer != 0))
      return emitOpError(
          "replica storage load part map references an invalid window or layer");
  }
  const int64_t elementBits = fieldBits;
  const int64_t fullMask = (int64_t{1} << elementBits) - 1;
  const bool dynamicBase =
      layered && logicalBaseMultiple % group != 0;
  for (auto [logicalLayer, physicalLayer, shiftOffset, shiftBaseFactor,
             maskValue] :
       llvm::zip(getLayerForPart(), getPhysicalLayerForPart(),
                 getShiftOffsetForPart(), getShiftBaseFactorForPart(),
                 getMaskValueForPart())) {
    if (natural || strided || interleaved) {
      if (physicalLayer != 0 || shiftOffset != 0 || shiftBaseFactor != 0 ||
          maskValue != 0)
        return emitOpError(
            "natural replica storage load cannot carry a physical layer");
      continue;
    }
    const int64_t expectedPhysical =
        dynamicBase ? 0
                    : (getAccess().getOrder() == "lo_first"
                           ? logicalLayer
                           : layers - 1 - logicalLayer);
    const int64_t expectedOffset =
        dynamicBase
            ? (getAccess().getOrder() == "lo_first"
                   ? logicalLayer * elementBits
                   : (layers - 1 - logicalLayer) * elementBits)
            : expectedPhysical * elementBits;
    const int64_t expectedFactor =
        dynamicBase
            ? (getAccess().getOrder() == "lo_first" ? elementBits
                                                      : -elementBits)
            : 0;
    const int64_t expectedMask =
        dynamicBase || (expectedPhysical + 1) * elementBits != 8 ? fullMask : 0;
    if (logicalLayer < 0 || logicalLayer >= layers ||
        physicalLayer != expectedPhysical || shiftOffset != expectedOffset ||
        shiftBaseFactor != expectedFactor || maskValue != expectedMask)
      return emitOpError(
          "replica storage physical layer disagrees with the selected base proof");
  }
  return mlir::success();
}

mlir::LogicalResult RVVSegmentPairLoadOp::verify() {
  auto field = getField().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto load = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto type = getFirst().getType();
  auto layout = type.getLayout();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  if (!sourceField || !load || !kernel ||
      field.getShape().size() != 1 || field.getShape()[0] <= 0 ||
      field.getLayout().getCarrier() != "local" ||
      type != getSecond().getType() || type.getShape().size() != 1 ||
      type.getElementType() != field.getElementType() || !integer ||
      integer.isSignless() ||
      (integer.getWidth() != 8 && integer.getWidth() != 16 &&
       integer.getWidth() != 32) || layout.getCarrier() != "rvv" ||
      layout.getValidity() != "full" ||
      layout.getSew() != static_cast<int64_t>(integer.getWidth()) ||
      layout.getLaneFactors()[0] <= 1 ||
      layout.getReplicaFactors()[0] != 1 ||
      layout.getFragmentFactors()[0] != 1 ||
      layout.getLocalFactors()[0] != 1 || layout.getLmulEighths() > 32 ||
      !kernel.getTarget().getHasSegmentMemory() ||
      !supportsRVVLayout(kernel.getTarget(), layout))
    return emitOpError(
        "segment pair load requires two identical full one-axis RVV results and one natural record field");
  const int64_t elements = type.getShape()[0];
  const int64_t lanes = layout.getLaneFactors()[0];
  const int64_t base = getLogicalBase();
  const int64_t tupleGroups =
      2 * std::max<int64_t>(1, (layout.getLmulEighths() + 7) / 8);
  const int64_t parts = layout.getTimeFactors()[0];
  auto memory = load.getRegion().getType();
  if (elements <= 0 || elements % lanes ||
      parts != elements / lanes ||
      parts > kernel.getTarget().getVectorRegisters() / tupleGroups ||
      layout.getVl() != lanes || base < 0 || base >= field.getShape()[0] ||
      elements > (field.getShape()[0] - base) / 2 ||
      memory.getInterleaveRows() != 0 ||
      localPackInterleaveRows(sourceField.getOwner()) > 0 ||
      getAccess().getForm() != "segment" ||
      getAccess().getSegmentFields() != 2 || getAccess().getIndexSEW() != 0 ||
      getAccess().getMapping() != "natural" ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      getAccess().getBitOffset() % 8 ||
      getAccess().getAlignment() != integer.getWidth() / 8 ||
      sourceField.getAccess().getAlignment() % getAccess().getAlignment() ||
      !exactLeaf(getLeaf(), "rvv", "segment-pair-load", "rvv.vlseg2",
                 "none", "exact") ||
      getLeaf().getTemporaryGroups() != parts * tupleGroups)
    return emitOpError(
        "segment pair load must cover exactly field[base + 2*i] and field[base + 2*i + 1] without widening the read range");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVRecordStorageLoadOp::verify() {
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto sourceLoad = sourceField ? ::sourceLoad(sourceField.getOwner()) : LoadOp();
  auto sourceSlice =
      sourceLoad ? sourceLoad.getRegion().getDefiningOp<SliceOp>() : SliceOp();
  RecordCohortType cohort = getCohort().getType();
  auto cohortOp = getCohort().getDefiningOp<RecordCohortOp>();
  auto origin = cohortOp ? cohortOp.getOrigin().getDefiningOp<PhysicalPointOp>()
                         : PhysicalPointOp();
  ValueType result = getResult().getType();
  mlir::Type fieldElement = sourceField
                                ? elementOf(sourceField.getResult().getType())
                                : mlir::Type();
  const int64_t fieldBits = sourceField ? elementBitWidth(fieldElement) : 0;
  const bool layered = getAccess().getMapping() == "grouped_layered";
  const int64_t expectedStorageWidth = layered ? 8 : fieldBits;
  mlir::Type expectedElement =
      layered ? mlir::IntegerType::get(getContext(), 8,
                                       mlir::IntegerType::Unsigned)
              : fieldElement;
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto target = kernel ? kernel.getTarget() : TargetAttr();
  auto expectedRecordStride =
      sourceLoad ? exactRecordByteStride(sourceLoad.getRegion().getType(), cohort)
                 : std::optional<int64_t>();
  auto sourceEncoding = sourceLoad
                            ? mlir::dyn_cast<weft::kernel::EncodingType>(
                                  sourceLoad.getRegion().getType().getEncoding())
                            : weft::kernel::EncodingType();
  if (!sourceField || !sourceLoad || !sourceSlice || !cohort || !cohortOp ||
      !origin || !kernel)
    return emitOpError(
        "record storage load requires explicit field, source-load, source-point, cohort, and target owners");
  if (!sourceEncoding || sourceEncoding.getKind() == "dense" ||
      sourceSlice.getIndices().size() != 1 ||
      sourceSlice.getIndices().front() != origin.getResult() ||
      sourceSlice.getSelectors().size() != 1 ||
      mlir::cast<mlir::StringAttr>(sourceSlice.getSelectors()[0]).getValue() !=
          "domain")
    return emitOpError(
        "record storage load source must be one encoded record at the cohort origin point");
  if (!sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      getAccess().getForm() != "strided" ||
      (getAccess().getMapping() != "natural" && !layered) ||
      getAccess().getBitOffset() % 8 || getStorageIndex() < 0 ||
      getStorageWidth() != expectedStorageWidth ||
      !expectedRecordStride ||
      getRecordByteStride() != *expectedRecordStride ||
      (expectedStorageWidth != 8 && expectedStorageWidth != 16 &&
       expectedStorageWidth != 32 && expectedStorageWidth != 64))
    return emitOpError(
        "record storage load requires one byte-addressable strided field geometry");
  if (result.getElementType() != expectedElement ||
      result.getLayout().getSew() != expectedStorageWidth ||
      !isClosedRecordCohortValue(result, cohort) ||
      !supportsRVVLayout(target, result.getLayout()))
    return emitOpError(
        "record storage load result must be one target-legal closed Level cohort");
  if (getLogicalIndices().empty() ||
      !exactLeaf(getLeaf(), "rvv", "record-storage-load",
                 "rvv.record-storage-load.strided", "none", "exact"))
    return emitOpError(
        "record storage load requires non-empty logical ownership and its exact selected leaf");

  int64_t logicalElements = 1;
  if (auto fieldValue =
          mlir::dyn_cast<ValueType>(sourceField.getResult().getType())) {
    auto product = checkedPositiveProduct(fieldValue.getShape().asArrayRef());
    if (!product)
      return emitOpError("record storage field has no static logical extent");
    logicalElements = *product;
  }
  llvm::SmallVector<int64_t> expectedIndices;
  for (int64_t logical = 0; logical < logicalElements; ++logical) {
    auto bit = recordFieldRelativeBit(getAccess(), fieldBits, logical);
    if (!bit || *bit < 0 || *bit % 8 + fieldBits > expectedStorageWidth)
      return emitOpError(
          "record storage field element crosses its selected storage unit");
    if (*bit / 8 == getStorageIndex())
      expectedIndices.push_back(logical);
  }
  if (expectedIndices != getLogicalIndices() ||
      getStorageIndex() >
          (getAccess().getStorageBits() - expectedStorageWidth) / 8)
    return emitOpError(
        "record storage load logical indices do not exactly name one source storage unit");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVRecordStorageDecodeOp::verify() {
  auto load = getInput().getDefiningOp<RVVRecordStorageLoadOp>();
  auto sourceField = load ? load.getField().getDefiningOp<FieldOp>() : FieldOp();
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  mlir::Type resultElement = result.getElementType();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(resultElement);
  const int64_t elementBits = integer ? integer.getWidth() : 0;
  auto bit = sourceField ? recordFieldRelativeBit(
                              load.getAccess(), elementBits, getLogicalIndex())
                         : std::optional<int64_t>();
  const int64_t expectedShift = bit ? *bit % 8 : -1;
  const int64_t expectedMask =
      elementBits > 0 && elementBits < 8 ? (int64_t(1) << elementBits) - 1
                                         : 0;
  llvm::StringRef instruction =
      expectedShift > 0
          ? (expectedMask > 0 ? "rvv.record-storage-decode.shift-mask"
                              : "rvv.record-storage-decode.shift")
          : (expectedMask > 0 ? "rvv.record-storage-decode.mask"
                              : "rvv.record-storage-decode.identity");
  if (!load || !sourceField || !integer || integer.isSigned() ||
      elementBits <= 0 || elementBits >= 8 || !bit ||
      *bit / 8 != load.getStorageIndex() ||
      !llvm::is_contained(load.getLogicalIndices(),
                          getLogicalIndex()) ||
      getShiftAmount() != expectedShift || getMaskValue() != expectedMask ||
      input.getElementType() !=
          mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::Unsigned) ||
      resultElement != elementOf(sourceField.getResult().getType()) ||
      input.getShape() != result.getShape() ||
      input.getAxisIds() != result.getAxisIds() ||
      input.getLayout() != result.getLayout() ||
      !exactLeaf(getLeaf(), "rvv", "record-storage-decode", instruction,
                 "none", "exact"))
    return emitOpError(
        "record storage decode must select one typed sub-byte field element from its raw storage unit");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVRecordStoreOp::verify() {
  ValueType value = getValue().getType();
  MemDescType destination = getDestination().getType();
  RecordCohortType cohort = getCohort().getType();
  auto encoding = mlir::dyn_cast<weft::kernel::EncodingType>(
      destination.getEncoding());
  auto expectedRecordStride = exactRecordByteStride(destination, cohort);
  if (!encoding || encoding.getKind() != "dense" ||
      !canWrite(destination.getAccess()) || destination.getShape().size() != 1 ||
      destination.getAxisIds().size() != 1 ||
      destination.getAxisIds()[0] != cohort.getDomain().getAxisId() ||
      destination.getElements() != 1 ||
      destination.getStorageBits() != elementBitWidth(value.getElementType()) ||
      !isClosedRecordCohortValue(value, cohort) || getElementOffset() < 0 ||
      getElementOffset() >= cohort.getPartition() ||
      !expectedRecordStride ||
      getRecordByteStride() != *expectedRecordStride ||
      getAccess().getForm() != "strided" ||
      getAccess().getMapping() != "dense" ||
      !exactLeaf(getLeaf(), "transfer", "record-store",
                 "rvv.record-store.strided", "none", "exact"))
    return emitOpError(
        "record store requires one writable dense destination and a closed strided Level cohort");
  return verifyLeafOperation(*this);
}

bool weft::riscv::supportsRVVRegisterSlice(
    TargetAttr target, ValueType input, ValueType result,
    llvm::ArrayRef<int64_t> offsets) {
  if (!target || !input || !result || offsets.empty())
    return false;
  auto source = input.getLayout();
  auto destination = result.getLayout();
  auto sourceLanes = rvvLaneCount(input);
  auto resultLanes = rvvLaneCount(result);
  if (!supportsRVVLayout(target, source) || !supportsRVVLayout(target, destination) ||
      input.getElementType() != result.getElementType() ||
      elementBitWidth(input.getElementType()) != source.getSew() ||
      source.getSew() != destination.getSew() ||
      source.getValidity() != "full" || destination.getValidity() != "full" ||
      destination.getLmulEighths() < 8 ||
      source.getLmulEighths() <= destination.getLmulEighths() ||
      source.getLmulEighths() % destination.getLmulEighths() ||
      !sourceLanes || !resultLanes || source.getVl() != *sourceLanes)
    return false;
  const int64_t capacity = target.getVlenBits() / destination.getSew() *
                           (destination.getLmulEighths() / 8);
  return capacity > 0 && *resultLanes == capacity &&
         destination.getVl() == capacity &&
         llvm::all_of(offsets, [&](int64_t offset) {
           return offset >= 0 && offset % capacity == 0 &&
                  offset <= *sourceLanes - capacity;
         });
}

mlir::LogicalResult RVVIssueSliceOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  auto inputParts = physicalPartCount(input);
  auto resultParts = physicalPartCount(result);
  auto inputLanes = checkedPositiveProduct(
      input.getLayout().getLaneFactors().asArrayRef());
  auto resultLanes = checkedPositiveProduct(
      result.getLayout().getLaneFactors().asArrayRef());
  const bool registerSlice = getLeaf().getInstruction() == "rvv.issue-slice.split";
  if (input.getElementType() != result.getElementType() ||
      input.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "rvv" || !inputParts || !resultParts ||
      *resultParts <= 0 || !inputLanes || !resultLanes ||
      getSourceParts().size() != static_cast<size_t>(*resultParts) ||
      getLaneOffsets().size() != static_cast<size_t>(*resultParts) ||
      llvm::any_of(getSourceParts(), [&](int64_t part) {
        return part < 0 || part >= *inputParts;
      }) ||
      llvm::any_of(getLaneOffsets(), [&](int64_t offset) {
        return offset < 0 || offset % *resultLanes ||
               offset > *inputLanes - *resultLanes;
      }) ||
      !exactLeaf(getLeaf(), "transfer", "issue-slice",
                 registerSlice ? "rvv.issue-slice.split" : "rvv.issue-slice",
                 "none", "exact"))
    return emitOpError(
        "RVV issue slice requires one exact physical-part projection");

  bool closedAxes = !result.getAxisIds().empty();
  for (size_t resultPosition = 0;
       resultPosition < result.getAxisIds().size(); ++resultPosition) {
    auto inputAxis = llvm::find(input.getAxisIds().asArrayRef(),
                                result.getAxisIds()[resultPosition]);
    if (inputAxis == input.getAxisIds().asArrayRef().end()) {
      closedAxes = false;
      continue;
    }
    const size_t inputPosition = static_cast<size_t>(
        inputAxis - input.getAxisIds().asArrayRef().begin());
    const int64_t inputShape = input.getShape()[inputPosition];
    const int64_t resultShape = result.getShape()[resultPosition];
    const int64_t inputTime =
        input.getLayout().getTimeFactors()[inputPosition];
    const int64_t inputLanes =
        input.getLayout().getLaneFactors()[inputPosition];
    const int64_t resultLanes =
        result.getLayout().getLaneFactors()[resultPosition];
    const bool timeSlice =
        resultShape > 0 && inputShape == resultShape * inputTime &&
        result.getLayout().getTimeFactors()[resultPosition] == 1 &&
        resultLanes == inputLanes && resultLanes == resultShape;
    const bool laneSlice =
        resultShape > 0 && inputTime == 1 &&
        result.getLayout().getTimeFactors()[resultPosition] == 1 &&
        inputShape == inputLanes && resultShape == resultLanes &&
        inputLanes >= resultLanes && inputLanes % resultLanes == 0;
    closedAxes &=
        (timeSlice || laneSlice) &&
        result.getLayout().getReplicaFactors()[resultPosition] == 1 &&
        result.getLayout().getFragmentFactors()[resultPosition] == 1 &&
        result.getLayout().getLocalFactors()[resultPosition] == 1;
  }
  for (size_t inputPosition = 0; inputPosition < input.getAxisIds().size();
       ++inputPosition)
    if (!llvm::is_contained(result.getAxisIds().asArrayRef(),
                            input.getAxisIds()[inputPosition]))
      closedAxes &= input.getLayout().getFragmentFactors()[inputPosition] == 1 &&
                    input.getLayout().getLocalFactors()[inputPosition] == 1;
  if (!closedAxes || input.getLayout().getSew() != result.getLayout().getSew() ||
      result.getLayout().getLmulEighths() <= 0 ||
      input.getLayout().getLmulEighths() <
          result.getLayout().getLmulEighths() ||
      input.getLayout().getLmulEighths() %
          result.getLayout().getLmulEighths() ||
      result.getLayout().getVl() != *resultLanes)
    return emitOpError(
        "RVV issue slice must select one typed contiguous lane window and remove only non-fragment/non-local coordinates");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (registerSlice && (!kernel || getLeaf().getTemporaryGroups() != 0 ||
                       !supportsRVVRegisterSlice(kernel.getTarget(), input, result,
                                                  getLaneOffsets())))
    return emitOpError("register issue slice requires aligned complete target register groups");
  return mlir::success();
}

mlir::LogicalResult RVVAssembleValueOp::verify() {
  auto result = getResult().getType();
  auto layout = result.getLayout();
  auto position = llvm::find(result.getAxisIds().asArrayRef(), getAxis());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel || getPieces().empty() || getPieces().size() > 4 || position == result.getAxisIds().asArrayRef().end() ||
      layout.getCarrier() != "rvv" || layout.getReplicaGroup() != 1 || !supportsRVVLayout(kernel.getTarget(), layout))
    return emitOpError("value assembly requires a bounded sequence of typed RVV issues");
  size_t dimension = position - result.getAxisIds().asArrayRef().begin();
  int64_t lanes = layout.getVl();
  auto allOne = [](mlir::DenseI64ArrayAttr factors) {
    return llvm::all_of(factors.asArrayRef(), [](int64_t n) { return n == 1; });
  };
  if (lanes <= 1 || result.getShape()[dimension] != lanes * int64_t(getPieces().size()) ||
      layout.getLaneFactors()[dimension] != lanes ||
      layout.getTimeFactors()[dimension] != int64_t(getPieces().size()) ||
      !allOne(layout.getReplicaFactors()) || !allOne(layout.getFragmentFactors()) || !allOne(layout.getLocalFactors()))
    return emitOpError("value assembly must cover its complete issue-time axis exactly once");
  for (size_t i = 0; i < result.getShape().size(); ++i)
    if (i != dimension && (result.getShape()[i] != 1 || layout.getTimeFactors()[i] != 1 || layout.getLaneFactors()[i] != 1))
      return emitOpError("value assembly cannot invent a traversal of free coordinates");
  int64_t groups = 0;
  for (mlir::Value value : getPieces()) {
    auto piece = mlir::cast<ValueType>(value.getType());
    auto physical = piece.getLayout();
    if (piece.getElementType() != result.getElementType() || piece.getAxisIds() != result.getAxisIds() ||
        piece.getShape().size() != result.getShape().size() || physical.getCarrier() != "rvv" ||
        physical.getReplicaGroup() != 1 || physical.getValidity() != "full" || physical.getSew() != layout.getSew() ||
        physical.getLmulEighths() != layout.getLmulEighths() || physical.getVl() != lanes ||
        physical.getLaneFactors() != layout.getLaneFactors() || !allOne(physical.getTimeFactors()) ||
        !allOne(physical.getReplicaFactors()) || !allOne(physical.getFragmentFactors()) || !allOne(physical.getLocalFactors()))
      return emitOpError("value assembly changes an issue's element type or register geometry");
    for (size_t i = 0; i < result.getShape().size(); ++i)
      if (piece.getShape()[i] != (i == dimension ? lanes : 1))
        return emitOpError("value assembly piece is not one complete contiguous issue");
    groups += physical.getRegisterGroups();
  }
  auto selected = getLeaf();
  if (!exactLeaf(selected, "transfer", "assemble-value", "rvv.assemble-value", "none", "exact") ||
      selected.getOperandGroups() != groups || selected.getResultGroups() != layout.getRegisterGroups() ||
      selected.getTemporaryGroups() != 0 || selected.getLocalBytes() != 0 || selected.getFragmentGroups() != 0)
    return emitOpError("value assembly is a register tuple handoff with no arithmetic or storage");
  return verifyLeafOperation(*this);
}

namespace {
ReloadOp singleChainReload(SpillOp spill) {
  auto allocation = spill.getSlot().getDefiningOp<LocalAllocOp>();
  if (!allocation || allocation.getResult().getType().getPurpose() != "spill" ||
      allocation->getBlock() != spill->getBlock() ||
      !allocation->isBeforeInBlock(spill))
    return {};
  ReloadOp reload;
  for (mlir::Operation *user : spill.getSlot().getUsers()) {
    if (user == spill.getOperation())
      continue;
    auto candidate = mlir::dyn_cast<ReloadOp>(user);
    if (!candidate || reload || candidate->getBlock() != spill->getBlock() ||
        !spill->isBeforeInBlock(candidate) ||
        candidate.getResult().getType() != spill.getInput().getType() ||
        !candidate.getResult().hasOneUse())
      return {};
    reload = candidate;
  }
  return reload;
}

mlir::Value chainValueBeforeSpill(mlir::Value value) {
  while (auto reload = value.getDefiningOp<ReloadOp>()) {
    auto allocation = reload.getSlot().getDefiningOp<LocalAllocOp>();
    if (!allocation || allocation.getResult().getType().getPurpose() != "spill" ||
        allocation->getBlock() != reload->getBlock())
      break;
    SpillOp writer;
    bool uniqueWrite = true;
    for (mlir::Operation *user : reload.getSlot().getUsers()) {
      if (auto spill = mlir::dyn_cast<SpillOp>(user)) {
        if (writer || spill->getBlock() != reload->getBlock() ||
            !allocation->isBeforeInBlock(spill) ||
            spill.getInput().getType() != reload.getResult().getType()) {
          uniqueWrite = false;
          break;
        }
        writer = spill;
      } else if (auto other = mlir::dyn_cast<ReloadOp>(user);
                 !other || other->getBlock() != reload->getBlock() ||
                 other.getResult().getType() != reload.getResult().getType()) {
        uniqueWrite = false;
        break;
      }
    }
    if (!uniqueWrite || !writer ||
        llvm::any_of(reload.getSlot().getUsers(), [&](mlir::Operation *user) {
          return user != writer.getOperation() && !writer->isBeforeInBlock(user);
        }))
      break;
    // A shared seed can be spilled once and reloaded for several chains.
    // Recovering its SSA origin does not make any accumulate result shareable:
    // the forward closed-chain check still requires one consumer per link.
    value = writer.getInput();
  }
  return value;
}

mlir::Value chainValueAfterSpill(mlir::Value value) {
  while (value.hasOneUse()) {
    auto spill = mlir::dyn_cast<SpillOp>(*value.getUsers().begin());
    auto reload = spill ? singleChainReload(spill) : ReloadOp();
    if (!reload)
      break;
    value = reload.getResult();
  }
  return value;
}
} // namespace

mlir::LogicalResult RVVWidenAccumulateOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  ValueType accumulator = getAccumulator().getType();
  ValueType result = getResult().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(accumulator.getElementType());
  llvm::ArrayRef<int64_t> reductionAxes = getReductionAxes();
  llvm::DenseSet<int64_t> seenReductionAxes;
  bool validReductionAxes = !reductionAxes.empty();
  for (int64_t axis : reductionAxes)
    validReductionAxes &= axis > 0 && seenReductionAxes.insert(axis).second;
  auto isReductionAxis = [&](int64_t axis) {
    return llvm::is_contained(reductionAxes, axis);
  };
  auto oneTimePart = [](ValueType value) {
    return checkedPositiveProduct(value.getLayout().getTimeFactors().asArrayRef()) ==
           std::optional<int64_t>(1);
  };
  auto positionOf = [](ValueType value,
                       int64_t axis) -> std::optional<size_t> {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
    if (found == value.getAxisIds().asArrayRef().end())
      return std::nullopt;
    return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
  };
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFree = [&](ValueType value) -> bool {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (isReductionAxis(axis))
        continue;
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return false;
      }
    }
    return true;
  };
  const bool compatibleFreeAxes = appendFree(lhs) && appendFree(rhs);
  bool compatibleReductionAxes = validReductionAxes;
  for (int64_t axis : reductionAxes) {
    auto lhsPosition = positionOf(lhs, axis);
    auto rhsPosition = positionOf(rhs, axis);
    auto accumulatorPosition = positionOf(accumulator, axis);
    compatibleReductionAxes &= lhsPosition && rhsPosition && accumulatorPosition;
    if (!accumulatorPosition)
      continue;
    expectedAxes.push_back(axis);
    expectedShape.push_back(accumulator.getShape()[*accumulatorPosition]);
    compatibleReductionAxes &=
        accumulator.getShape()[*accumulatorPosition] ==
            accumulator.getLayout().getLaneFactors()[*accumulatorPosition] &&
        accumulator.getLayout().getReplicaFactors()[*accumulatorPosition] == 1;
  }
  auto operandMapsToPartial = [&](ValueType operand) {
    for (auto [position, axis] :
         llvm::enumerate(operand.getAxisIds().asArrayRef())) {
      auto partialPosition = llvm::find(accumulator.getAxisIds().asArrayRef(), axis);
      if (partialPosition == accumulator.getAxisIds().asArrayRef().end())
        return false;
      const size_t target = static_cast<size_t>(
          partialPosition - accumulator.getAxisIds().asArrayRef().begin());
      if (isReductionAxis(axis)) {
        if (operand.getLayout().getLaneFactors()[position] !=
                accumulator.getLayout().getLaneFactors()[target] ||
            operand.getShape()[position] != accumulator.getShape()[target])
          return false;
        continue;
      }
      if (operand.getLayout().getTimeFactors()[position] != 1 ||
          operand.getLayout().getLaneFactors()[position] != 1 ||
          operand.getLayout().getReplicaFactors()[position] !=
              accumulator.getLayout().getReplicaFactors()[target] ||
          operand.getLayout().getFragmentFactors()[position] != 1 ||
          operand.getLayout().getLocalFactors()[position] != 1)
        return false;
    }
    return true;
  };
  auto laneOrder = [&](ValueType value) {
    llvm::SmallVector<int64_t> order;
    for (auto [axis, lane] :
         llvm::zip(value.getAxisIds().asArrayRef(),
                   value.getLayout().getLaneFactors().asArrayRef()))
      if (lane > 1 && isReductionAxis(axis))
        order.push_back(axis);
    return order;
  };
  compatibleReductionAxes &=
      laneOrder(lhs) == laneOrder(rhs) &&
      laneOrder(lhs) == laneOrder(accumulator);
  auto loop = getOperation()->getParentOfType<mlir::scf::ForOp>();
  bool loopCarried = loop && getOperation()->getBlock() == loop.getBody();
  unsigned carryIndex = 0;
  mlir::Value chain = chainValueBeforeSpill(getAccumulator());
  while (loopCarried) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(chain)) {
      if (argument.getOwner() != loop.getBody() || argument.getArgNumber() == 0) {
        loopCarried = false;
        break;
      }
      carryIndex = argument.getArgNumber() - 1;
      break;
    }
    auto previous = chain.getDefiningOp<RVVWidenAccumulateOp>();
    if (!previous || previous.getOperation()->getBlock() != loop.getBody() ||
        previous.getReductionAxes() != getReductionAxes()) {
      loopCarried = false;
      break;
    }
    chain = chainValueBeforeSpill(previous.getAccumulator());
  }
  if (loopCarried) {
    mlir::Value current = getResult();
    while (true) {
      current = chainValueAfterSpill(current);
      if (!current.hasOneUse()) {
        loopCarried = false;
        break;
      }
      mlir::Operation *user = *current.getUsers().begin();
      if (auto next = mlir::dyn_cast<RVVWidenAccumulateOp>(user)) {
        if (next.getAccumulator() != current ||
            next.getReductionAxes() != getReductionAxes()) {
          loopCarried = false;
          break;
        }
        current = next.getResult();
        continue;
      }
      auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user);
      loopCarried = yield && carryIndex < yield.getNumOperands() &&
                    yield.getOperand(carryIndex) == current;
      break;
    }
  }
  bool explicitChain = false;
  if (!loopCarried) {
    mlir::Value first = chainValueBeforeSpill(getAccumulator());
    while (auto previous = first.getDefiningOp<RVVWidenAccumulateOp>()) {
      if (previous.getOperation()->getBlock() != getOperation()->getBlock() ||
          previous.getReductionAxes() != getReductionAxes()) {
        first = {};
        break;
      }
      first = chainValueBeforeSpill(previous.getAccumulator());
    }
    auto splatSeed = first ? first.getDefiningOp<RVVSplatOp>() : RVVSplatOp();
    auto productSeed =
        first ? first.getDefiningOp<RVVWidenMultiplyOp>()
              : RVVWidenMultiplyOp();
    // A pure seed may be shared from any dominating block.  Requiring it to
    // live beside every consumer rejects the legal CSE/LICM form where one
    // zero vector (or one widened product) feeds several explicit partial
    // chains in nested regions.  SSA dominance already guarantees that such a
    // definition is available here; the closed-chain contract starts at this
    // accumulate result, not at a block-local copy of the seed.
    explicitChain = static_cast<bool>(splatSeed) ||
                    static_cast<bool>(productSeed);
    mlir::Value current = getResult();
    while (explicitChain) {
      // Resource materialization may save one link in a private slot.  Follow
      // only a single writer/reload with the exact type and ordered lifetime;
      // extra writers, aliases or numeric consumers do not form this chain.
      current = chainValueAfterSpill(current);
      if (!current.hasOneUse()) {
        explicitChain = false;
        break;
      }
      mlir::Operation *user = *current.getUsers().begin();
      if (auto next = mlir::dyn_cast<RVVWidenAccumulateOp>(user)) {
        if (next.getAccumulator() != current ||
            next.getReductionAxes() != getReductionAxes() ||
            next.getOperation()->getBlock() != getOperation()->getBlock()) {
          explicitChain = false;
          break;
        }
        current = next.getResult();
        continue;
      }
      if (auto capture = mlir::dyn_cast<RVVPartialCaptureOp>(user))
        explicitChain = capture.getInput() == current;
      else if (auto collect = mlir::dyn_cast<RVVPartialCollectOp>(user))
        explicitChain = llvm::is_contained(collect.getInputs(), current);
      else if (auto finalize = mlir::dyn_cast<RVVFinalizeWidenDotOp>(user))
        explicitChain = finalize.getPartial() == current &&
                        finalize.getReductionAxes() == getReductionAxes();
      else
        explicitChain = false;
      break;
    }
  }
  if (!lhsElement || !rhsElement || !partialElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      !partialElement.isSigned() ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      partialElement.getWidth() !=
          2 * std::max<unsigned>(8, lhsElement.getWidth()) ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      accumulator != result || !compatibleReductionAxes ||
      !oneTimePart(lhs) || !oneTimePart(rhs) || !oneTimePart(accumulator) ||
      lhs.getLayout().getCarrier() != "rvv" ||
      rhs.getLayout().getCarrier() != "rvv" ||
      accumulator.getLayout().getCarrier() != "rvv" ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
      lhs.getLayout().getVl() != accumulator.getLayout().getVl() ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      accumulator.getLayout().getSew() != 2 * lhs.getLayout().getSew() ||
      lhs.getLayout().getLmulEighths() != rhs.getLayout().getLmulEighths() ||
      accumulator.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      !compatibleFreeAxes ||
      accumulator.getAxisIds().asArrayRef() !=
          llvm::ArrayRef<int64_t>(expectedAxes) ||
      accumulator.getShape().asArrayRef() !=
          llvm::ArrayRef<int64_t>(expectedShape) ||
      !operandMapsToPartial(lhs) || !operandMapsToPartial(rhs) ||
      (!loopCarried && !explicitChain) ||
      !exactLeaf(getLeaf(), "rvv", "widen-accumulate",
                 "rvv.vwmacc.partial", "none", "agnostic"))
    return emitOpError(
        "RVV widened accumulation requires matching multi-axis issue operands and one closed partial chain");
  return mlir::success();
}

mlir::LogicalResult RVVFinalizeWidenDotOp::verify() {
  ValueType partial = getPartial().getType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  auto resultValue = mlir::dyn_cast<ValueType>(getResult().getType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
      resultValue ? resultValue.getElementType() : getResult().getType());
  llvm::ArrayRef<int64_t> reductionAxes = getReductionAxes();
  llvm::DenseSet<int64_t> seenReductionAxes;
  bool validReductionAxes = !reductionAxes.empty();
  for (int64_t axis : reductionAxes)
    validReductionAxes &= axis > 0 && seenReductionAxes.insert(axis).second;
  auto isReductionAxis = [&](int64_t axis) {
    return llvm::is_contained(reductionAxes, axis);
  };
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (auto [axis, extent] : llvm::zip(partial.getAxisIds().asArrayRef(),
                                      partial.getShape().asArrayRef()))
    if (!isReductionAxis(axis)) {
      expectedAxes.push_back(axis);
      expectedShape.push_back(extent);
    }
  const bool legalResult =
      (!resultValue && expectedAxes.empty() && isScalar(getResult().getType())) ||
      (resultValue && resultValue.getLayout().getCarrier() == "scalar" &&
       llvm::equal(expectedAxes, resultValue.getAxisIds().asArrayRef()) &&
       llvm::equal(expectedShape, resultValue.getShape().asArrayRef()));
  mlir::Value partialSource = chainValueBeforeSpill(getPartial());
  auto loop = partialSource.getDefiningOp<mlir::scf::ForOp>();
  auto explicitChain = partialSource.getDefiningOp<RVVWidenAccumulateOp>();
  auto explicitProduct = partialSource.getDefiningOp<RVVWidenMultiplyOp>();
  bool completeMapping = validReductionAxes &&
                         partial.getLayout().getCarrier() == "rvv" &&
                         (loop || (explicitChain &&
                                   explicitChain.getReductionAxes() ==
                                       getReductionAxes()) ||
                          explicitProduct);
  for (int64_t axis : reductionAxes)
    completeMapping &= llvm::is_contained(partial.getAxisIds().asArrayRef(), axis);
  if (completeMapping)
    for (size_t position = 0; position < partial.getShape().size(); ++position) {
      const bool reductionDimension =
          isReductionAxis(partial.getAxisIds()[position]);
      completeMapping &= partial.getLayout().getTimeFactors()[position] == 1 &&
                         partial.getLayout().getFragmentFactors()[position] == 1 &&
                         partial.getLayout().getLocalFactors()[position] == 1 &&
                         (reductionDimension
                              ? partial.getLayout().getLaneFactors()[position] ==
                                        partial.getShape()[position] &&
                                    partial.getLayout().getReplicaFactors()[position] == 1
                              : partial.getLayout().getLaneFactors()[position] == 1 &&
                                    (partial.getShape()[position] > 0
                                         ? partial.getLayout().getReplicaFactors()[position] ==
                                               partial.getShape()[position]
                                         : partial.getLayout().getReplicaFactors()[position] > 0));
    }
  if (completeMapping && resultValue)
    for (size_t position = 0; position < resultValue.getShape().size(); ++position) {
      auto partialPosition = llvm::find(partial.getAxisIds().asArrayRef(),
                                        resultValue.getAxisIds()[position]);
      if (partialPosition == partial.getAxisIds().asArrayRef().end()) {
        completeMapping = false;
        break;
      }
      const size_t sourcePosition = static_cast<size_t>(
          partialPosition - partial.getAxisIds().asArrayRef().begin());
      completeMapping &=
          resultValue.getLayout().getTimeFactors()[position] == 1 &&
          resultValue.getLayout().getLaneFactors()[position] == 1 &&
          resultValue.getLayout().getReplicaFactors()[position] ==
              partial.getLayout().getReplicaFactors()[sourcePosition] &&
          resultValue.getLayout().getFragmentFactors()[position] == 1 &&
          resultValue.getLayout().getLocalFactors()[position] == 1;
    }
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  llvm::StringRef instruction =
      partialElement && partialElement.getWidth() == 16
          ? llvm::StringRef("rvv.vwredsum.partial")
          : llvm::StringRef("rvv.vredsum.partial");
  if (!partialElement || !partialElement.isSigned() ||
      (partialElement.getWidth() != 16 && partialElement.getWidth() != 32) ||
      !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      !completeMapping || !legalResult || !kernel ||
      !supportsRVVLayout(kernel.getTarget(), partial.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "finalize-widen-dot",
                 instruction, "none", "exact"))
    return emitOpError(
        "final widened dot reduction must remove every selected lane axis and preserve every free replica");
  return mlir::success();
}

mlir::LogicalResult RVVPartialSetOp::verify() {
  auto enclosingKernel = getOperation()->getParentOfType<KernelOp>();
  if (!enclosingKernel)
    return emitOpError("RVV partial set requires an enclosing target kernel");
  PartialSetType result = getResult().getType();
  ValueType partial = result.getPartialType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  auto positionOf = [&](ValueType value) -> std::optional<size_t> {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), getReductionAxis());
    if (found == value.getAxisIds().asArrayRef().end())
      return std::nullopt;
    return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
  };
  auto partialAxis = positionOf(partial);
  if (getLhs().empty() || getRhs().empty() || !partialElement ||
      !partialElement.isSigned() || !partialAxis ||
      getLhsOperands().size() != static_cast<size_t>(result.getSlots()) ||
      getRhsOperands().size() != static_cast<size_t>(result.getSlots()) ||
      getLhsParts().size() != static_cast<size_t>(result.getSlots()) ||
      getRhsParts().size() != static_cast<size_t>(result.getSlots()) ||
      getLhsLaneOffsets().size() != static_cast<size_t>(result.getSlots()) ||
      getRhsLaneOffsets().size() != static_cast<size_t>(result.getSlots()) ||
      getReductionAxis() != result.getReductionAxis() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() ||
      !exactLeaf(getLeaf(), "rvv", "partial-set", "rvv.partial-set",
                 "none", "exact"))
    return emitOpError(
        "RVV partial set requires paired stream operands and one closed result topology");
  auto preservesPartialAxes = [&](ValueType operand, int64_t part, int64_t offset) {
    if (partial.getAxisIds().size() == 1)
      return true;
    auto source = RegisterCoordinates::get(operand.getLayout());
    auto target = RegisterCoordinates::get(partial.getLayout());
    auto base = source ? source->logical(part, offset) : std::nullopt;
    if (!source || !target || !base || target->parts() != 1 || target->lanes() > 4096)
      return false;
    for (int64_t axis : partial.getAxisIds().asArrayRef())
      if (!llvm::is_contained(operand.getAxisIds().asArrayRef(), axis)) return false;
    // Each selected slice represents one product. Extra lane axes may select
    // another independent product, but must be fixed throughout this slice.
    for (int64_t lane = 0; lane < target->lanes(); ++lane) {
      auto actual = source->logical(part, offset + lane);
      auto expected = target->logical(0, lane);
      if (!actual || !expected) return false;
      for (auto [position, axis] : llvm::enumerate(operand.getAxisIds().asArrayRef())) {
        auto found = llvm::find(partial.getAxisIds().asArrayRef(), axis);
        int64_t delta = found == partial.getAxisIds().asArrayRef().end()
            ? 0 : (*expected)[found - partial.getAxisIds().asArrayRef().begin()];
        if ((*actual)[position] - (*base)[position] != delta) return false;
      }
    }
    return true;
  };
  for (int64_t slot = 0; slot < result.getSlots(); ++slot) {
    const int64_t lhsOperand = getLhsOperands()[slot];
    const int64_t rhsOperand = getRhsOperands()[slot];
    if (lhsOperand < 0 || lhsOperand >= static_cast<int64_t>(getLhs().size()) ||
        rhsOperand < 0 || rhsOperand >= static_cast<int64_t>(getRhs().size()))
      return emitOpError(
          "RVV partial set slot references an absent operand");
    ValueType lhs = mlir::cast<ValueType>(getLhs()[lhsOperand].getType());
    ValueType rhs = mlir::cast<ValueType>(getRhs()[rhsOperand].getType());
    auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
    auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
    auto lhsAxis = positionOf(lhs);
    auto rhsAxis = positionOf(rhs);
    auto lhsParts = physicalPartCount(lhs);
    auto rhsParts = physicalPartCount(rhs);
    auto lhsLanes = rvvLaneCount(lhs);
    auto rhsLanes = rvvLaneCount(rhs);
    auto partialLanes = rvvLaneCount(partial);
    const int64_t sliceLanes = partialLanes.value_or(-1);
    auto target = enclosingKernel.getTarget();
    auto lhsSliceLMUL = rvvLaneSliceLMULEighths(target, lhs, sliceLanes);
    auto rhsSliceLMUL = rvvLaneSliceLMULEighths(target, rhs, sliceLanes);
    const int64_t primaryLanes =
        partial.getLayout().getLaneFactors()[*partialAxis];
    // reductionAxis names the primary physical lane axis, while termsPerSlot
    // covers the complete product carrier.  A multi-axis carrier therefore
    // carries the product of all lane factors, not just the primary factor.
    const bool closedTerms = primaryLanes > 0 && sliceLanes > 0 &&
                             result.getTermsPerSlot() == sliceLanes;
    const llvm::StringRef instruction = getMultiplyInstruction();
    const bool exactMultiply =
        (instruction == "rvv.vwmul.vv" && lhsElement && rhsElement &&
         lhsElement.isSigned() && rhsElement.isSigned()) ||
        (instruction == "rvv.vwmul.vv.reinterpret-rhs" && lhsElement &&
         rhsElement && lhsElement.isSigned() && rhsElement.isUnsigned() &&
         rhsElement.getWidth() < rhs.getLayout().getSew()) ||
        (instruction == "rvv.vwmul.vv.reinterpret-lhs" && lhsElement &&
         rhsElement && lhsElement.isUnsigned() && rhsElement.isSigned() &&
         lhsElement.getWidth() < lhs.getLayout().getSew()) ||
        (instruction == "rvv.vwmulsu.vv" && lhsElement && rhsElement &&
         lhsElement.isSigned() && rhsElement.isUnsigned()) ||
        (instruction == "rvv.vwmulsu.vv.swap" && lhsElement && rhsElement &&
         lhsElement.isUnsigned() && rhsElement.isSigned());
    if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
        rhsElement.isSignless() ||
        (!lhsElement.isSigned() && !rhsElement.isSigned()) || !lhsAxis ||
        !rhsAxis || lhs.getLayout().getCarrier() != "rvv" ||
        rhs.getLayout().getCarrier() != "rvv" ||
        std::max<unsigned>(8, lhsElement.getWidth()) !=
            std::max<unsigned>(8, rhsElement.getWidth()) ||
        partialElement.getWidth() !=
            2 * std::max<unsigned>(8, lhsElement.getWidth()) ||
        lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
        lhs.getLayout().getTimeFactors()[*lhsAxis] <= 0 ||
        rhs.getLayout().getTimeFactors()[*rhsAxis] <= 0 ||
        sliceLanes <= 0 || !lhsLanes || !rhsLanes ||
        sliceLanes > *lhsLanes || sliceLanes > *rhsLanes ||
        getLhsLaneOffsets()[slot] < 0 || getRhsLaneOffsets()[slot] < 0 ||
        getLhsLaneOffsets()[slot] % sliceLanes ||
        getRhsLaneOffsets()[slot] % sliceLanes ||
        getLhsLaneOffsets()[slot] > *lhsLanes - sliceLanes ||
        getRhsLaneOffsets()[slot] > *rhsLanes - sliceLanes ||
        partial.getLayout().getSew() != 2 * lhs.getLayout().getSew() ||
        !lhsSliceLMUL || !rhsSliceLMUL || *lhsSliceLMUL != *rhsSliceLMUL ||
        partial.getLayout().getLmulEighths() != 2 * *lhsSliceLMUL ||
        !closedTerms ||
        !preservesPartialAxes(lhs, getLhsParts()[slot], getLhsLaneOffsets()[slot]) ||
        !preservesPartialAxes(rhs, getRhsParts()[slot], getRhsLaneOffsets()[slot]) ||
        !lhsParts || !rhsParts || getLhsParts()[slot] < 0 ||
        getRhsParts()[slot] < 0 || getLhsParts()[slot] >= *lhsParts ||
        getRhsParts()[slot] >= *rhsParts || !exactMultiply)
      return emitOpError(
                 "RVV partial set operand pair does not produce the declared widened slots")
             << "; slot=" << slot << ", partial=" << partial
             << ", lhs=" << lhs << ", rhs=" << rhs
             << ", lhs_slice_lmul=" << lhsSliceLMUL.value_or(-1)
             << ", rhs_slice_lmul=" << rhsSliceLMUL.value_or(-1)
             << ", lhs_offset=" << getLhsLaneOffsets()[slot]
             << ", rhs_offset=" << getRhsLaneOffsets()[slot]
             << ", lhs_part=" << getLhsParts()[slot]
             << ", rhs_part=" << getRhsParts()[slot]
             << ", reduction_axis=" << getReductionAxis()
             << ", slots=" << result.getSlots()
             << ", terms_per_slot=" << result.getTermsPerSlot()
             << ", multiply=" << instruction;
  }
  return mlir::success();
}

mlir::LogicalResult RVVPartialCaptureOp::verify() {
  ValueType input = getInput().getType();
  PartialSetType result = getResult().getType();
  auto axis = llvm::find(input.getAxisIds().asArrayRef(),
                         result.getReductionAxis());
  auto parts = physicalPartCount(input);
  llvm::StringRef tail = input.getLayout().getValidity() == "tail"
                             ? llvm::StringRef("agnostic")
                             : llvm::StringRef("exact");
  if (input.getLayout().getCarrier() != "rvv" || !parts || *parts != 1 ||
      axis == input.getAxisIds().asArrayRef().end() ||
      result.getPartialType() != input || result.getSlots() != 1 ||
      result.getTermsPerSlot() !=
          input.getLayout().getLaneFactors()[static_cast<size_t>(
              axis - input.getAxisIds().asArrayRef().begin())] ||
      result.getResourceGroups() != input.getLayout().getRegisterGroups() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != result.getResourceGroups() ||
      !exactLeaf(getLeaf(), "rvv", "partial-capture",
                 "rvv.partial-capture", "none", tail))
    return emitOpError(
        "RVV partial capture requires one complete register-resident lane partial");
  return mlir::success();
}

mlir::LogicalResult RVVPartialCollectOp::verify() {
  PartialSetType result = getResult().getType();
  ValueType partial = result.getPartialType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  auto lanes = rvvLaneCount(partial);
  auto parts = physicalPartCount(partial);
  auto axis = llvm::find(partial.getAxisIds().asArrayRef(),
                         result.getReductionAxis());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  llvm::StringRef tail = partial.getLayout().getValidity() == "tail"
                             ? llvm::StringRef("agnostic")
                             : llvm::StringRef("exact");
  bool inputsClosed = !getInputs().empty() &&
                      getInputs().size() == static_cast<size_t>(result.getSlots());
  for (mlir::Value input : getInputs())
    inputsClosed &= input.getType() == partial;
  if (!inputsClosed || !partialElement || !partialElement.isSigned() ||
      (partialElement.getWidth() != 16 && partialElement.getWidth() != 32) ||
      partial.getLayout().getCarrier() != "rvv" || !lanes ||
      *lanes <= 0 || !parts || *parts != 1 ||
      axis == partial.getAxisIds().asArrayRef().end() ||
      result.getTermsPerSlot() != *lanes ||
      result.getResourceGroups() !=
          result.getSlots() * partial.getLayout().getRegisterGroups() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() || !kernel ||
      !supportsRVVLayout(kernel.getTarget(), partial.getLayout()) ||
      getLeaf().getOperandGroups() != result.getResourceGroups() ||
      getLeaf().getResultGroups() != result.getResourceGroups() ||
      !exactLeaf(getLeaf(), "rvv", "partial-collect",
                 "rvv.partial-collect", "none", tail))
    return emitOpError(
        "RVV partial collect requires one explicit register-resident product per typed slot");
  return mlir::success();
}

mlir::LogicalResult RVVPartialRepackOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType source = input.getPartialType();
  ValueType target = result.getPartialType();
  auto sourceElement =
      mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
  auto targetElement =
      mlir::dyn_cast<mlir::IntegerType>(target.getElementType());
  auto sourceAxis = llvm::find(source.getAxisIds().asArrayRef(),
                               input.getReductionAxis());
  auto targetAxis = llvm::find(target.getAxisIds().asArrayRef(),
                               input.getReductionAxis());
  const int64_t split = static_cast<int64_t>(getSplitFactor());
  bool compatibleAxes = source.getAxisIds() == target.getAxisIds() &&
                        source.getShape().size() == target.getShape().size();
  if (compatibleAxes && sourceAxis != source.getAxisIds().asArrayRef().end() &&
      targetAxis != target.getAxisIds().asArrayRef().end()) {
    const size_t sourcePosition = static_cast<size_t>(
        sourceAxis - source.getAxisIds().asArrayRef().begin());
    const size_t targetPosition = static_cast<size_t>(
        targetAxis - target.getAxisIds().asArrayRef().begin());
    compatibleAxes &= sourcePosition == targetPosition;
    for (size_t position = 0; position < source.getShape().size(); ++position) {
      if (position == sourcePosition) {
        compatibleAxes &= source.getShape()[position] ==
                              target.getShape()[position] * split &&
                          source.getLayout().getLaneFactors()[position] ==
                              target.getLayout().getLaneFactors()[position] * split &&
                          source.getLayout().getTimeFactors()[position] == 1 &&
                          target.getLayout().getTimeFactors()[position] == 1 &&
                          source.getLayout().getReplicaFactors()[position] == 1 &&
                          target.getLayout().getReplicaFactors()[position] == 1;
      } else {
        compatibleAxes &= source.getShape()[position] == target.getShape()[position] &&
                          source.getLayout().getTimeFactors()[position] ==
                              target.getLayout().getTimeFactors()[position] &&
                          source.getLayout().getLaneFactors()[position] ==
                              target.getLayout().getLaneFactors()[position] &&
                          source.getLayout().getReplicaFactors()[position] ==
                              target.getLayout().getReplicaFactors()[position];
      }
    }
  } else {
    compatibleAxes = false;
  }
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (split <= 1 || !sourceElement || !targetElement ||
      sourceElement != targetElement || !compatibleAxes ||
      source.getLayout().getCarrier() != "rvv" ||
      target.getLayout().getCarrier() != "rvv" ||
      source.getLayout().getSew() != target.getLayout().getSew() ||
      source.getLayout().getLmulEighths() !=
          target.getLayout().getLmulEighths() * split ||
      source.getLayout().getVl() != target.getLayout().getVl() * split ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() * split ||
      result.getTermsPerSlot() * split != input.getTermsPerSlot() ||
      result.getResourceGroups() != result.getSlots() *
          target.getLayout().getRegisterGroups() || !kernel ||
      !supportsRVVLayout(kernel.getTarget(), source.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), target.getLayout()))
    return emitOpError(
        "RVV partial repack requires an exact lane/LMUL split with independent slot resources");
  auto carrier = rvvPartialRepackCarrierLMULEighths(
      kernel.getTarget(), source, target, split);
  if (!carrier)
    return emitOpError("RVV partial repack windows have no addressable carrier");
  const int64_t capacity = kernel.getTarget().getVlenBits() * *carrier /
                           (8 * target.getLayout().getSew());
  const bool registerView = *carrier == target.getLayout().getLmulEighths() &&
      target.getLayout().getLmulEighths() >= 8 && target.getLayout().getVl() == capacity;
  const llvm::StringRef instruction = registerView
      ? "rvv.partial-repack.split" : "rvv.partial-repack.slice";
  llvm::SmallVector<int64_t> parameters{input.getReductionAxis(), split};
  if (!registerView)
    parameters.push_back(*carrier);
  const int64_t temporaries = registerView ? 0 :
      result.getResourceGroups() + std::max<int64_t>(1, *carrier / 8);
  if (!exactLeaf(getLeaf(), "rvv", "partial-repack", instruction, "none", "exact") ||
      getLeaf().getParameters().asArrayRef() != llvm::ArrayRef<int64_t>(parameters) ||
      getLeaf().getTemporaryGroups() != temporaries)
    return emitOpError("RVV partial repack leaf disagrees with its carrier and temporary resources");
  return mlir::success();
}

mlir::LogicalResult RVVPartialMergeOp::verify() {
  PartialSetType result = getResult().getType();
  if (getInputs().size() < 2)
    return emitOpError("RVV partial merge requires at least two input sets");
  int64_t slots = 0;
  int64_t terms = 0;
  int64_t resources = 0;
  for (mlir::Value inputValue : getInputs()) {
    PartialSetType input = mlir::cast<PartialSetType>(inputValue.getType());
    if (input.getPartialType() != result.getPartialType() ||
        input.getReductionAxis() != result.getReductionAxis())
      return emitOpError(
          "RVV partial merge inputs must share one partial layout and reduction axis");
    slots += input.getSlots();
    terms += input.getSlots() * input.getTermsPerSlot();
    resources += input.getResourceGroups();
  }
  const int64_t resultGroups =
      result.getPartialType().getLayout().getRegisterGroups();
  if (getTopology() != "pairwise" || result.getSlots() != 1 ||
      result.getTermsPerSlot() != terms ||
      result.getResourceGroups() != resultGroups ||
      getLeaf().getOperandGroups() != resources ||
      getLeaf().getResultGroups() != resultGroups ||
      !exactLeaf(getLeaf(), "rvv", "partial-merge", "rvv.partial-merge",
                 "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({result.getReductionAxis(),
                                   static_cast<int64_t>(getInputs().size()),
                                   slots, terms}))
    return emitOpError(
        "RVV partial merge must pairwise combine every compatible slot into one typed partial");
  return mlir::success();
}

mlir::LogicalResult RVVPartialReduceOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType inputPartial = input.getPartialType();
  ValueType resultPartial = result.getPartialType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(inputPartial.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(resultPartial.getElementType());
  auto resultAxis = llvm::find(resultPartial.getAxisIds().asArrayRef(),
                               input.getReductionAxis());
  if (!inputElement || !resultElement || !inputElement.isSigned() ||
      !resultElement.isSigned() ||
      (inputElement.getWidth() != 16 && inputElement.getWidth() != 32) ||
      resultElement.getWidth() != 32 ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() ||
      result.getTermsPerSlot() != input.getTermsPerSlot() ||
      resultAxis == resultPartial.getAxisIds().asArrayRef().end())
    return emitOpError(
        "RVV partial reduction requires one topology-preserving i16/i32 to i32 reduction");
  const size_t position = static_cast<size_t>(
      resultAxis - resultPartial.getAxisIds().asArrayRef().begin());
  llvm::StringRef instruction = inputElement.getWidth() == 16
                                    ? "rvv.partial-reduce.widen"
                                    : "rvv.partial-reduce";
  if (resultPartial.getShape().size() != 1 ||
      resultPartial.getShape()[position] != 1 ||
      resultPartial.getLayout().getLaneFactors()[position] != 1 ||
      resultPartial.getLayout().getSew() != 32 ||
      resultPartial.getLayout().getLmulEighths() != 8 ||
      resultPartial.getLayout().getRegisterGroups() != 1 ||
      !exactLeaf(getLeaf(), "rvv", "partial-reduce", instruction, "none",
                 "exact"))
    return emitOpError(
        "RVV partial reduction result must retain a singleton typed reduction slot; result=")
           << resultPartial << ", instruction=" << instruction
           << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVPartialScaleCombineOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType partial = input.getPartialType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  llvm::DenseSet<int64_t> logicalAxes;
  const bool logicalAxesLegal =
      !getLogicalReductionAxes().empty() &&
      llvm::all_of(getLogicalReductionAxes(), [&](int64_t axis) {
        return axis > 0 && logicalAxes.insert(axis).second;
      });
  if (!element || !element.isSigned() || element.getWidth() != 32 ||
      !logicalAxesLegal ||
      getScales().size() != static_cast<size_t>(input.getSlots()) ||
      getScaleReplicas().size() != static_cast<size_t>(input.getSlots()) ||
      getSlotOrder().size() != static_cast<size_t>(input.getSlots()) ||
      result.getPartialType() != partial ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() <= 0 || input.getSlots() % result.getSlots() ||
      result.getTermsPerSlot() !=
          input.getTermsPerSlot() * input.getSlots() / result.getSlots() ||
      !exactLeaf(getLeaf(), "rvv", "partial-scale-combine",
                 "rvv.partial-scale-combine", "none", "exact"))
    return emitOpError(
        "RVV scaled partial combine requires one scale per slot and one closed fanout");
  llvm::SmallVector<bool> seen(input.getSlots(), false);
  for (int64_t slot : getSlotOrder()) {
    if (slot < 0 || slot >= input.getSlots() || seen[slot])
      return emitOpError(
          "RVV scaled partial combine slot order must be a permutation");
    seen[slot] = true;
  }
  for (auto [scale, replica] :
       llvm::zip(getScales(), getScaleReplicas())) {
    auto type = mlir::dyn_cast<ValueType>(scale.getType());
    auto direct = mlir::dyn_cast<mlir::IntegerType>(scale.getType());
    auto scaleElement = type
                            ? mlir::dyn_cast<mlir::IntegerType>(
                                  type.getElementType())
                            : direct;
    auto replicas = type ? checkedPositiveProduct(
                               type.getLayout().getReplicaFactors().asArrayRef())
                         : direct ? std::optional<int64_t>(1)
                                  : std::optional<int64_t>();
    const bool closedCarrier =
        type ? type.getLayout().getCarrier() == "scalar" &&
                   llvm::all_of(type.getLayout().getTimeFactors().asArrayRef(),
                                [](int64_t factor) { return factor == 1; }) &&
                   llvm::all_of(type.getLayout().getLaneFactors().asArrayRef(),
                                [](int64_t factor) { return factor == 1; })
             : static_cast<bool>(direct);
    if (!scaleElement || !scaleElement.isSigned() ||
        scaleElement.getWidth() != 32 || !closedCarrier || !replicas ||
        replica < 0 || replica >= *replicas)
      return emitOpError(
          "RVV scaled partial combine scales require in-bounds signed-i32 scalar replicas");
  }
  return mlir::success();
}

bool weft::riscv::supportsRVVPartialPackedScale(
    TargetAttr target, llvm::ArrayRef<PartialSetType> inputs,
    llvm::ArrayRef<ValueType> scales, PartialSetType result) {
  if (!target || inputs.empty() || inputs.size() > 8 ||
      scales.empty() || scales.size() > 8 || !result)
    return false;
  auto partial = result.getPartialType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  auto layout = partial.getLayout();
  const auto ones = [](auto values) {
    return llvm::all_of(values, [](int64_t factor) { return factor == 1; });
  };
  if (!element || !element.isSigned() || element.getWidth() != 32 ||
      result.getSlots() != 1 || layout.getCarrier() != "rvv" ||
      layout.getSew() != 32 || layout.getLmulEighths() != 8 ||
      layout.getVl() != 1 || layout.getValidity() != "full" ||
      !ones(layout.getTimeFactors().asArrayRef()) ||
      !ones(layout.getLaneFactors().asArrayRef()) ||
      !ones(layout.getReplicaFactors().asArrayRef()) ||
      !ones(layout.getFragmentFactors().asArrayRef()) ||
      !ones(layout.getLocalFactors().asArrayRef()) ||
      !supportsRVVLayout(target, layout))
    return false;
  int64_t slots = 0, terms = 0, resources = 0;
  for (auto input : inputs) {
    if (!input || input.getSlots() <= 0 ||
        input.getSlots() > 8 - slots ||
        input.getPartialType() != partial ||
        input.getReductionAxis() != result.getReductionAxis() ||
        input.getTermsPerSlot() >
            (std::numeric_limits<int64_t>::max() - terms) / input.getSlots())
      return false;
    const int64_t width = input.getSlots();
    slots += width;
    terms += width * input.getTermsPerSlot();
    resources += input.getResourceGroups();
  }
  int64_t scaleAxis = -1, scaleLanes = 0;
  for (auto scale : scales) {
    if (!scale || scale.getShape().size() != 1)
      return false;
    const int64_t width = scale.getShape()[0];
    auto scaleLayout = scale.getLayout();
    if (scale.getElementType() != element || width <= 0 || width > slots - scaleLanes ||
        (scaleAxis != -1 && scale.getAxisIds()[0] != scaleAxis) ||
        scaleLayout.getCarrier() != "rvv" || scaleLayout.getSew() != 32 ||
        scaleLayout.getLmulEighths() != 8 || scaleLayout.getVl() != width ||
        scaleLayout.getValidity() != "full" ||
        scaleLayout.getLaneFactors()[0] != width ||
        !ones(scaleLayout.getTimeFactors().asArrayRef()) ||
        !ones(scaleLayout.getReplicaFactors().asArrayRef()) ||
        !ones(scaleLayout.getFragmentFactors().asArrayRef()) ||
        !ones(scaleLayout.getLocalFactors().asArrayRef()) ||
        !supportsRVVLayout(target, scaleLayout))
      return false;
    scaleAxis = scale.getAxisIds()[0];
    scaleLanes += width;
    resources += scaleLayout.getRegisterGroups();
  }
  const int64_t temporaries = scales.size() == 1 ? 3 : 4;
  return slots >= 2 && slots <= target.getVlenBits() / 32 &&
         scaleLanes == slots && terms == result.getTermsPerSlot() &&
         resources + result.getResourceGroups() + temporaries <=
             target.getVectorRegisters();
}

mlir::LogicalResult RVVPartialPackedScaleOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  llvm::SmallVector<PartialSetType> inputs;
  llvm::SmallVector<ValueType> scales;
  int64_t slots = 0, operandGroups = 0;
  for (mlir::Value value : getInputs()) {
    auto input = mlir::cast<PartialSetType>(value.getType());
    inputs.push_back(input);
    if (input.getSlots() > 8 - slots)
      return emitOpError("packed scale supports at most eight reduced slots");
    slots += input.getSlots();
    operandGroups += input.getResourceGroups();
  }
  for (mlir::Value value : getScales()) {
    auto scale = mlir::cast<ValueType>(value.getType());
    scales.push_back(scale);
    operandGroups += scale.getLayout().getRegisterGroups();
  }
  const int64_t temporaries = scales.size() == 1 ? 3 : 4;
  if (!kernel || !supportsRVVPartialPackedScale(
                     kernel.getTarget(), inputs, scales, getResult().getType()) ||
      getSlotOrder().size() != static_cast<size_t>(slots) ||
      getLogicalReductionAxes().size() != 1 ||
      getLogicalReductionAxes()[0] != scales.front().getAxisIds()[0] ||
      getLeaf().getOperandGroups() != operandGroups ||
      getLeaf().getResultGroups() != getResult().getType().getResourceGroups() ||
      getLeaf().getTemporaryGroups() != temporaries ||
      !exactLeaf(getLeaf(), "rvv", "partial-packed-scale",
                 "rvv.partial-packed-scale", "none", "exact"))
    return emitOpError(
        "packed scale requires aligned full i32 VL1 slot sets and m1 scale windows with closed pack resources");
  llvm::SmallVector<bool> seen(slots, false);
  for (int64_t slot : getSlotOrder()) {
    if (slot < 0 || slot >= slots || seen[slot])
      return emitOpError("packed scale slot order must be a permutation");
    seen[slot] = true;
  }
  return mlir::success();
}

mlir::LogicalResult RVVPartialWidenScaleOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType inputPartial = input.getPartialType();
  ValueType resultPartial = result.getPartialType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(inputPartial.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(resultPartial.getElementType());
  llvm::DenseSet<int64_t> logicalAxes;
  const bool logicalAxesLegal =
      !getLogicalReductionAxes().empty() &&
      llvm::all_of(getLogicalReductionAxes(), [&](int64_t axis) {
        return axis > 0 && logicalAxes.insert(axis).second;
      });
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const bool sameCoordinates =
      inputPartial.getAxisIds() == resultPartial.getAxisIds() &&
      inputPartial.getShape() == resultPartial.getShape() &&
      inputPartial.getLayout().getAxisIds() ==
          resultPartial.getLayout().getAxisIds() &&
      inputPartial.getLayout().getTimeFactors() ==
          resultPartial.getLayout().getTimeFactors() &&
      inputPartial.getLayout().getLaneFactors() ==
          resultPartial.getLayout().getLaneFactors() &&
      inputPartial.getLayout().getReplicaFactors() ==
          resultPartial.getLayout().getReplicaFactors() &&
      inputPartial.getLayout().getFragmentFactors() ==
          resultPartial.getLayout().getFragmentFactors() &&
      inputPartial.getLayout().getLocalFactors() ==
          resultPartial.getLayout().getLocalFactors();
  if (!logicalAxesLegal || !inputElement || !inputElement.isSigned() ||
      inputElement.getWidth() != 16 || !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      inputPartial.getLayout().getCarrier() != "rvv" ||
      resultPartial.getLayout().getCarrier() != "rvv" || !sameCoordinates ||
      getCombineArity() <= 0 || input.getSlots() % getCombineArity() ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() / getCombineArity() ||
      result.getTermsPerSlot() !=
          input.getTermsPerSlot() * getCombineArity() ||
      resultPartial.getLayout().getSew() !=
          2 * inputPartial.getLayout().getSew() ||
      resultPartial.getLayout().getLmulEighths() !=
          2 * inputPartial.getLayout().getLmulEighths() ||
      resultPartial.getLayout().getVl() != inputPartial.getLayout().getVl() ||
      resultPartial.getLayout().getRegisterGroups() !=
          std::max<int64_t>(
              1, (resultPartial.getLayout().getLmulEighths() + 7) / 8) ||
      result.getResourceGroups() !=
          result.getSlots() *
              resultPartial.getLayout().getRegisterGroups() ||
      getScales().size() != static_cast<size_t>(input.getSlots()) ||
      getScaleReplicas().size() != static_cast<size_t>(input.getSlots()) ||
      !kernel || !kernel.getTarget().getHasWideningInteger() ||
      !supportsRVVLayout(kernel.getTarget(), inputPartial.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), resultPartial.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "partial-widen-scale",
                 "rvv.partial-widen-scale", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {input.getReductionAxis(), input.getSlots(),
               static_cast<int64_t>(getCombineArity())}))
    return emitOpError(
        "RVV partial widen-scale requires one signed narrow scalar per i16 slot and an exact i32 topology");
  for (auto [scale, replica] :
       llvm::zip(getScales(), getScaleReplicas())) {
    ValueType type = mlir::cast<ValueType>(scale.getType());
    auto scaleElement =
        mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
    auto replicas = checkedPositiveProduct(
        type.getLayout().getReplicaFactors().asArrayRef());
    if (!scaleElement || !scaleElement.isSigned() ||
        scaleElement.getWidth() <= 0 || scaleElement.getWidth() > 16 ||
        type.getLayout().getCarrier() != "scalar" || !replicas ||
        replica < 0 || replica >= *replicas ||
        !llvm::all_of(type.getLayout().getTimeFactors().asArrayRef(),
                      [](int64_t factor) { return factor == 1; }) ||
        !llvm::all_of(type.getLayout().getLaneFactors().asArrayRef(),
                      [](int64_t factor) { return factor == 1; }))
      return emitOpError(
          "RVV partial widen-scale operands require in-bounds signed narrow scalar replicas");
  }
  return mlir::success();
}

mlir::LogicalResult RVVPartialCombineOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  llvm::DenseSet<int64_t> logicalAxes;
  const bool logicalAxesLegal =
      !getLogicalReductionAxes().empty() &&
      llvm::all_of(getLogicalReductionAxes(), [&](int64_t axis) {
        return axis > 0 && logicalAxes.insert(axis).second;
      });
  const bool preservesType =
      result.getPartialType() == input.getPartialType();
  const bool widensType =
      isWideningPartialCombineStage(input, result, getArity());
  const llvm::StringRef instruction =
      widensType ? "rvv.partial-combine.widen" : "rvv.partial-combine";
  if (!logicalAxesLegal || getArity() <= 1 || getTopology() != "pairwise" ||
      input.getSlots() % getArity() ||
      (!preservesType && !widensType) ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() / getArity() ||
      result.getTermsPerSlot() != input.getTermsPerSlot() * getArity() ||
      !exactLeaf(getLeaf(), "rvv", "partial-combine", instruction, "none",
                 "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({input.getReductionAxis(),
                                   static_cast<int64_t>(getArity()),
                                   result.getSlots()}))
    return emitOpError(
        "RVV partial combine requires an exact pairwise divisive tree level over one partial set");
  return mlir::success();
}

mlir::LogicalResult RVVPartialFinalizeOp::verify() {
  PartialSetType input = getInput().getType();
  auto result = mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  auto resultValue = mlir::dyn_cast<ValueType>(getResult().getType());
  auto partial = mlir::dyn_cast<mlir::IntegerType>(
      input.getPartialType().getElementType());
  auto axis = llvm::find(input.getPartialType().getAxisIds().asArrayRef(),
                         getReductionAxis());
  if ((!result && !resultValue) || !partial || !partial.isSigned() ||
      (partial.getWidth() != 16 && partial.getWidth() != 32) ||
      getReductionAxis() != input.getReductionAxis() ||
      axis == input.getPartialType().getAxisIds().asArrayRef().end())
    return emitOpError(
        "RVV partial finalize requires one signed widened set and an i32 result");
  const size_t position = static_cast<size_t>(
      axis - input.getPartialType().getAxisIds().asArrayRef().begin());
  if (resultValue) {
    auto resultElement =
        mlir::dyn_cast<mlir::IntegerType>(resultValue.getElementType());
    bool preservesFreeAxes =
        resultElement && resultElement.isSigned() &&
        resultElement.getWidth() == 32 && partial.getWidth() == 16 &&
        input.getSlots() == 1 &&
        input.getPartialType().getShape()[position] == 1 &&
        input.getPartialType().getLayout().getTimeFactors()[position] == 1 &&
        input.getPartialType().getLayout().getLaneFactors()[position] == 1 &&
        input.getPartialType().getLayout().getReplicaFactors()[position] == 1 &&
        resultValue.getLayout().getCarrier() == "rvv" &&
        resultValue.getAxisIds().size() + 1 ==
            input.getPartialType().getAxisIds().size() &&
        resultValue.getLayout().getSew() == 32 &&
        resultValue.getLayout().getLmulEighths() ==
            input.getPartialType().getLayout().getLmulEighths() * 2 &&
        resultValue.getLayout().getVl() ==
            input.getPartialType().getLayout().getVl();
    size_t resultPosition = 0;
    for (size_t partialPosition = 0;
         preservesFreeAxes &&
         partialPosition < input.getPartialType().getAxisIds().size();
         ++partialPosition) {
      if (partialPosition == position)
        continue;
      preservesFreeAxes &=
          resultPosition < resultValue.getAxisIds().size() &&
          resultValue.getAxisIds()[resultPosition] ==
              input.getPartialType().getAxisIds()[partialPosition] &&
          resultValue.getShape()[resultPosition] ==
              input.getPartialType().getShape()[partialPosition] &&
          resultValue.getLayout().getTimeFactors()[resultPosition] ==
              input.getPartialType().getLayout().getTimeFactors()[partialPosition] &&
          resultValue.getLayout().getLaneFactors()[resultPosition] ==
              input.getPartialType().getLayout().getLaneFactors()[partialPosition] &&
          resultValue.getLayout().getReplicaFactors()[resultPosition] ==
              input.getPartialType().getLayout().getReplicaFactors()[partialPosition] &&
          resultValue.getLayout().getFragmentFactors()[resultPosition] ==
              input.getPartialType().getLayout().getFragmentFactors()[partialPosition] &&
          resultValue.getLayout().getLocalFactors()[resultPosition] ==
              input.getPartialType().getLayout().getLocalFactors()[partialPosition];
      ++resultPosition;
    }
    if (!preservesFreeAxes ||
        !exactLeaf(getLeaf(), "rvv", "partial-finalize",
                   "rvv.partial-finalize.widen", "none", "exact"))
      return emitOpError(
          "RVV vector partial finalize must widen one singleton reduction coordinate and preserve every free physical axis");
    return mlir::success();
  }
  if (!result || !result.isSigned() || result.getWidth() != 32)
    return emitOpError(
        "RVV scalar partial finalize requires one signed i32 result");
  auto lanes = rvvLaneCount(input.getPartialType());
  llvm::StringRef instruction = lanes && *lanes == 1
                                    ? "rvv.partial-finalize.extract"
                                    : "rvv.partial-finalize.reduce";
  if (!lanes || !exactLeaf(getLeaf(), "rvv", "partial-finalize", instruction,
                           "none", "exact"))
    return emitOpError()
           << "RVV partial finalize leaf disagrees with its typed lane topology; "
              "expected_instruction="
           << instruction << ", reduction_axis=" << getReductionAxis()
           << ", partial=" << input.getPartialType() << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVAssembleReplicasOp::verify() {
  ValueType result = getResult().getType();
  auto parts = checkedPositiveProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  if (result.getLayout().getCarrier() != "scalar" || !parts ||
      *parts != static_cast<int64_t>(getValues().size()) ||
      llvm::any_of(getValues(), [&](mlir::Value value) {
        return value.getType() != result.getElementType();
      }) ||
      !exactLeaf(getLeaf(), "scalar", "assemble-replicas",
                 "scalar.assemble-replicas", "none", "exact"))
    return emitOpError(
        "scalar replica assembly requires one scalar value per result replica");
  return mlir::success();
}

mlir::LogicalResult RVVWidenReduceOp::verify() {
  auto input = getInput().getType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  auto parts = physicalPartCount(input);
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError(
        "RVV widening reduction must be nested in one target kernel");
  auto target = kernel.getTarget();
  if (!inputElement || inputElement.isSignless() || !resultElement ||
      resultElement.isSignless() || getKind() != "add" ||
      resultElement.getWidth() != inputElement.getWidth() * 2 ||
      inputElement.getWidth() < 8 || inputElement.getWidth() > 16 ||
      !isScalar(getResult().getType()) || getAxis() < 0 ||
      static_cast<size_t>(getAxis()) >= input.getShape().size() ||
      input.getLayout().getCarrier() != "rvv" ||
      checkedPositiveProduct(input.getLayout().getTimeFactors().asArrayRef()) !=
          std::optional<int64_t>(1) || !parts || *parts != 1 ||
      !target.getHasRVV() || !target.getHasWideningInteger() ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                          input.getLayout().getSew()) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                          input.getLayout().getLmulEighths()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-reduce", "rvv.vwredsum", "none",
                 "exact"))
    return emitOpError(
        "RVV widening reduction requires one signedness-qualified integer "
        "vector, an add reduction axis, and a double-width scalar result");
  return mlir::success();
}

mlir::LogicalResult RVVPartitionedWidenReduceStoreOp::verify() {
  auto input = getInput().getType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto destination = getDestination().getType();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(destination.getEncoding());
  auto destinationField = getDestination().getDefiningOp<FieldOp>();
  auto parts = physicalPartCount(input);
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError(
        "partitioned widening reduction store must be nested in one target kernel");
  auto target = kernel.getTarget();
  std::string destinationFamily;
  if (inputElement)
    destinationFamily =
        std::string(inputElement.isUnsigned() ? "u" : "i") +
        std::to_string(inputElement.getWidth() * 2);
  auto axis = llvm::find(destination.getAxisIds().asArrayRef(),
                         getDestinationAxis());
  if (!inputElement || !inputElement.isSigned() || inputElement.getWidth() != 8 ||
      input.getShape().size() != 1 || getPartition() != 16 || getCount() <= 1 ||
      input.getShape()[0] != getPartition() * getCount() ||
      input.getLayout().getCarrier() != "rvv" ||
      input.getLayout().getValidity() != "full" || !parts ||
      getCount() % *parts || !encoding || !destinationField ||
      destinationField.getAccess().getMapping() != "natural" ||
      destinationField.getAccess().getBitOffset() % 8 ||
      encoding.getKind() != "dense" ||
      encoding.getFamily() != destinationFamily ||
      axis == destination.getAxisIds().asArrayRef().end() ||
      getAccess().getForm() != "unit" || getAccess().getMapping() != "dense" ||
      !target.getHasRVV() || !target.getHasWideningInteger() ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                          input.getLayout().getSew()) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                          input.getLayout().getLmulEighths()) ||
      !exactLeaf(getLeaf(), "rvv", "partitioned-widen-reduce-store",
                 "rvv.partitioned-vwredsum-store", "none", "exact"))
    return emitOpError()
           << "partitioned widening reduction store requires one full signed-i8 "
              "RVV axis split into 16-element partitions, an aligned natural "
              "i16 field, and a closed unit destination edge on a "
              "widening-capable target; input="
           << input << ", physical_parts=" << (parts ? *parts : -1)
           << ", partition=" << getPartition() << ", count=" << getCount()
           << ", destination=" << destination << ", field_access="
           << (destinationField ? destinationField.getAccess() : AccessAttr())
           << ", store_access=" << getAccess() << ", target=" << target;
  return mlir::success();
}

mlir::LogicalResult RVVLayeredWindowOp::verify() {
  auto field = getField().getType();
  auto first = getFirst().getType();
  auto second = getSecond().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto plan = getPlan();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  const int64_t axis = getPoint().getType().getDomain().getAxisId();
  auto resultAxis = llvm::find(first.getAxisIds().asArrayRef(), axis);
  const bool hasLayerAxis =
      resultAxis != first.getAxisIds().asArrayRef().end() &&
      first.getShape()[static_cast<size_t>(
          resultAxis - first.getAxisIds().asArrayRef().begin())] == layer;
  auto physicalPoint = getPoint().getDefiningOp<PhysicalPointOp>();
  auto partition = physicalPoint
                       ? physicalPoint.getPartition().getDefiningOp<
                             mlir::arith::ConstantIndexOp>()
                       : mlir::arith::ConstantIndexOp();
  llvm::StringRef validity = first.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!element || element.isSigned() || field.getShape().size() != first.getShape().size() ||
      first != second || first.getElementType() != field.getElementType() ||
      first.getAxisIds() != field.getAxisIds() || !hasLayerAxis ||
      first.getLayout().getCarrier() != "rvv" ||
      (validity != "full" && validity != "tail") ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      getAccess().getGroupSize() != group || getAccess().getLayerSize() != layer ||
      plan.getKind() != "layered" || plan.getReductionAxis() != axis ||
      plan.getProjectionBase() != 0 || plan.getProjectionStride() != 1 ||
      plan.getProjectionRepeat() != 1 || plan.getProjectionExtent() != layer ||
      layer <= 0 || group != layer * 2 || element.getWidth() * 2 > 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      getPhysicalLayerForResult().size() != 2 ||
      getShiftForResult().size() != 2 || getMaskForResult().size() != 2 ||
      !partition || partition.value() != layer ||
      !exactLeaf(getLeaf(), "rvv", "layered-window", "rvv.layered-window",
                 "none", tail))
    return emitOpError(
        "RVV layered window requires two equal full/tail layer results, one "
        "two-layer packed field, and an identical exact layer-sized point");
  llvm::SmallVector<int64_t> ordered(getPhysicalLayerForResult());
  llvm::sort(ordered);
  if (ordered != llvm::SmallVector<int64_t>({0, 1}))
    return emitOpError(
        "RVV layered window physical result layers must form one permutation");
  for (size_t index = 0; index < getPhysicalLayerForResult().size(); ++index) {
    const int64_t physicalLayer = getPhysicalLayerForResult()[index];
    const int64_t shift = getShiftForResult()[index];
    const int64_t mask = getMaskForResult()[index];
    if (physicalLayer !=
            plan.getPhysicalLayerBase() +
                static_cast<int64_t>(index) * plan.getPhysicalLayerStep() ||
        shift != physicalLayer * static_cast<int64_t>(element.getWidth()) ||
        mask != ((physicalLayer + 1) *
                            static_cast<int64_t>(element.getWidth()) !=
                        8
                    ? (int64_t{1} << element.getWidth()) - 1
                    : 0))
      return emitOpError(
          "RVV layered window result decode plan disagrees with its typed layer");
  }
  return mlir::success();
}

bool weft::riscv::hasContiguousLayeredStreamStorage(
    FieldOp field, ValueType result, LayeredStreamGeometryAttr geometry) {
  auto load = field ? sourceLoad(field.getOwner()) : LoadOp();
  if (!load || !geometry)
    return false;
  auto memory = load.getRegion().getType();
  if (memory.getInterleaveRows() == 0)
    return true;
  // Each window is a unit byte load within one record's field chunk. Its
  // replicas are record rows; its streams are record-local packed layers.
  auto access = field.getAccess();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  const int64_t chunk = access.getFieldChunkBytes();
  const int64_t lanes = geometry.getLaneCount();
  if (!integer || chunk <= 0 || lanes <= 0 || lanes > chunk ||
      result.getShape().size() != 2 || memory.getAxisIds() != result.getAxisIds() ||
      geometry.getAxis() != result.getAxisIds()[1] ||
      result.getShape()[1] <= 0 ||
      access.getStorageBits() / integer.getWidth() != result.getShape()[1] ||
      result.getLayout().getLaneFactors()[0] != 1 ||
      geometry.getGroupForWindow().size() != geometry.getWithinForWindow().size())
    return false;
  for (auto [group, within] : llvm::zip(geometry.getGroupForWindow().asArrayRef(),
                                      geometry.getWithinForWindow().asArrayRef())) {
    const int64_t layer = geometry.getLayerSize();
    if (group < 0 || layer <= 0 || within < 0 || within >= layer ||
        group > (std::numeric_limits<int64_t>::max() - within) / layer)
      return false;
    const int64_t byte = group * layer + within;
    if (byte % chunk > chunk - lanes || byte > access.getStorageBits() / 8 - lanes)
      return false;
  }
  return true;
}

mlir::LogicalResult RVVLayeredStreamOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto sourceField = getField().getDefiningOp<FieldOp>();
  if (!hasContiguousLayeredStreamStorage(sourceField, result, getGeometry()))
    return emitOpError("requires unit byte windows inside the declared record or interleaved field chunk");
  const int64_t group = getAccess().getGroupSize();
  const int64_t layer = getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  bool hasStreamAxis = false;
  bool onlyOneStreamAxis = true;
  for (size_t index = 0; index < field.getShape().size(); ++index) {
    const int64_t extent = field.getShape()[index];
    const int64_t time = field.getLayout().getTimeFactors()[index];
    const int64_t lanes = field.getLayout().getLaneFactors()[index];
    const bool streamAxis =
        group > 0 && layer > 0 && extent > 0 && extent % group == 0 &&
        lanes > 1 && layer % lanes == 0 && time > 1 &&
        time * lanes == extent;
    if (streamAxis) {
      onlyOneStreamAxis &= !hasStreamAxis;
      hasStreamAxis = true;
    } else {
      onlyOneStreamAxis &= time == 1 && lanes == 1;
    }
  }
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  auto geometry = getGeometry();
  bool exactLayerOps =
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getShiftAmountForStream().size() &&
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getMaskValueForStream().size();
  if (exactLayerOps && element)
    for (auto [physicalLayer, shiftAmount, maskValue] :
         llvm::zip(geometry.getPhysicalLayerForStream().asArrayRef(),
                   geometry.getShiftAmountForStream().asArrayRef(),
                   geometry.getMaskValueForStream().asArrayRef()))
      exactLayerOps &=
          shiftAmount == physicalLayer * element.getWidth() &&
          maskValue ==
              ((physicalLayer + 1) * element.getWidth() != 8
                   ? (int64_t(1) << element.getWidth()) - 1
                   : 0);
  if (!element || element.isSigned() || !sourceField ||
      getAccess() != sourceField.getAccess() || field != result ||
      field.getLayout().getCarrier() != "rvv" || !hasStreamAxis ||
      !onlyOneStreamAxis ||
      (validity != "full" && validity != "tail") ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      layer <= 0 || group % layer || layers <= 1 ||
      element.getWidth() * layers > 8 || getAccess().getBitOffset() % 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      geometry.getGroupSize() != group || geometry.getLayerSize() != layer ||
      geometry.getStreamCount() !=
          checkedPositiveProduct(
              result.getLayout().getTimeFactors().asArrayRef()).value_or(-1) ||
      geometry.getReplicaCount() !=
          checkedPositiveProduct(
              result.getLayout().getReplicaFactors().asArrayRef()).value_or(-1) ||
      !exactLayerOps ||
      !exactLeaf(getLeaf(), "rvv", "layered-stream", "rvv.layered-stream",
                 "none", tail))
    return emitOpError(
        "RVV layered stream requires one byte-aligned packed field whose "
        "grouped/layered geometry is represented by one complete time/lane axis");
  return mlir::success();
}

mlir::LogicalResult RVVProjectedLayeredStreamOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto sourceField = getField().getDefiningOp<FieldOp>();
  const int64_t axis = getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
  const int64_t group = getAccess().getGroupSize();
  const int64_t layer = getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto physicalPoint = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto pointPartition =
      physicalPoint
          ? physicalPoint.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
          : mlir::arith::ConstantIndexOp();
  const int64_t fieldExtent =
      fieldAxis == field.getAxisIds().asArrayRef().end()
          ? 0
          : field.getShape()[static_cast<size_t>(
                fieldAxis - field.getAxisIds().asArrayRef().begin())];
  const int64_t resultExtent =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getShape()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  const int64_t lanes =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getLaneFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  const int64_t time =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getTimeFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  auto geometry = getGeometry();
  bool exactLayerOps =
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getShiftAmountForStream().size() &&
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getMaskValueForStream().size();
  if (exactLayerOps && element)
    for (auto [physicalLayer, shiftAmount, maskValue] :
         llvm::zip(geometry.getPhysicalLayerForStream().asArrayRef(),
                   geometry.getShiftAmountForStream().asArrayRef(),
                   geometry.getMaskValueForStream().asArrayRef()))
      exactLayerOps &=
          shiftAmount == physicalLayer * element.getWidth() &&
          maskValue ==
              ((physicalLayer + 1) * element.getWidth() != 8
                   ? (int64_t(1) << element.getWidth()) - 1
                   : 0);
  bool preservesFreeAxes = field.getAxisIds() == result.getAxisIds() &&
                          field.getShape().size() == result.getShape().size();
  if (preservesFreeAxes)
    for (size_t position = 0; position < result.getShape().size(); ++position) {
      if (result.getAxisIds()[position] == axis)
        continue;
      preservesFreeAxes &=
          field.getShape()[position] == result.getShape()[position] &&
          result.getLayout().getTimeFactors()[position] == 1 &&
          result.getLayout().getLaneFactors()[position] == 1 &&
          (result.getShape()[position] > 0
               ? result.getLayout().getReplicaFactors()[position] ==
                     result.getShape()[position]
               : result.getLayout().getReplicaFactors()[position] > 0) &&
          result.getLayout().getFragmentFactors()[position] == 1 &&
          result.getLayout().getLocalFactors()[position] == 1;
    }
  if (!element || element.isSigned() || !sourceField ||
      getAccess() != sourceField.getAccess() || axis <= 0 ||
      getOrigin().getType().getDomain().getAxisId() != axis ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      field.getElementType() != result.getElementType() || !preservesFreeAxes ||
      getProjectionBase() < 0 || getProjectionStride() != 1 ||
      getProjectionRepeat() != 1 || getProjectionExtent() != resultExtent ||
      resultExtent <= 0 || fieldExtent <= 0 ||
      getProjectionBase() > fieldExtent ||
      resultExtent > fieldExtent - getProjectionBase() ||
      group <= 0 || layer <= 0 || group % layer || layers <= 1 ||
      getProjectionBase() % group || resultExtent % group ||
      !pointPartition || pointPartition.value() % group ||
      result.getLayout().getCarrier() != "rvv" || lanes <= 1 || time <= 1 ||
      time > std::numeric_limits<int64_t>::max() / lanes ||
      time * lanes != resultExtent || layer % lanes ||
      element.getWidth() * layers > 8 || getAccess().getBitOffset() % 8 ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      geometry.getAxis() != axis || geometry.getGroupSize() != group ||
      geometry.getLayerSize() != layer || geometry.getLaneCount() != lanes ||
      geometry.getStreamCount() !=
          checkedPositiveProduct(
              result.getLayout().getTimeFactors().asArrayRef()).value_or(-1) ||
      geometry.getReplicaCount() !=
          checkedPositiveProduct(
              result.getLayout().getReplicaFactors().asArrayRef()).value_or(-1) ||
      !exactLayerOps ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "projected-layered-stream",
                 "rvv.projected-layered-stream", "none", tail))
    return emitOpError(
        "projected layered stream requires one typed field/point projection and one closed grouped/layered time-lane mapping");
  return mlir::success();
}

mlir::LogicalResult RVVSplatOp::verify() {
  if (!isScalar(getScalar().getType()) ||
      (!exactLeaf(getLeaf(), "rvv", "splat", "rvv.splat", "none", "exact") &&
       !exactLeaf(getLeaf(), "rvv", "splat", "rvv.splat.local", "none", "exact")) ||
      getResult().getType().getLayout().getCarrier() != "rvv" ||
      elementOf(getScalar().getType()) !=
          getResult().getType().getElementType())
    return emitOpError()
           << "RVV splat requires a matching scalar and RVV result; scalar="
           << getScalar().getType() << ", result=" << getResult().getType()
           << ", leaf engine=" << getLeaf().getEngine();
  return mlir::success();
}

mlir::LogicalResult RVVAxisBroadcastOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  auto inputLayout = input.getLayout();
  auto resultLayout = result.getLayout();
  auto inputLanes = checkedProduct(inputLayout.getLaneFactors().asArrayRef());
  auto resultLanes = checkedProduct(resultLayout.getLaneFactors().asArrayRef());
  bool preservesSourceDomain = input.getElementType() == result.getElementType();
  for (auto [position, axis] :
       llvm::enumerate(input.getAxisIds().asArrayRef())) {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (found == result.getAxisIds().asArrayRef().end()) {
      preservesSourceDomain = false;
      break;
    }
    size_t resultPosition = static_cast<size_t>(
        found - result.getAxisIds().asArrayRef().begin());
    preservesSourceDomain &=
        input.getShape()[position] == result.getShape()[resultPosition];
  }
  bool addsOnlySingletonAxes = true;
  for (auto [position, axis] :
       llvm::enumerate(result.getAxisIds().asArrayRef())) {
    if (llvm::is_contained(input.getAxisIds().asArrayRef(), axis))
      continue;
    addsOnlySingletonAxes &=
        result.getShape()[position] == 1 &&
        resultLayout.getTimeFactors()[position] == 1 &&
        resultLayout.getLaneFactors()[position] == 1 &&
        resultLayout.getReplicaFactors()[position] == 1 &&
        resultLayout.getFragmentFactors()[position] == 1 &&
        resultLayout.getLocalFactors()[position] == 1;
  }
  const bool properLaneExpansion =
      inputLanes && resultLanes && *resultLanes > *inputLanes;
  const bool singletonAxisExpansion =
      inputLanes && resultLanes && *resultLanes == *inputLanes &&
      addsOnlySingletonAxes &&
      inputLayout.getLmulEighths() == resultLayout.getLmulEighths();
  if (!preservesSourceDomain || input.getAxisIds().size() >=
                                    result.getAxisIds().size() ||
      inputLayout.getCarrier() != "rvv" ||
      resultLayout.getCarrier() != "rvv" || !inputLanes || !resultLanes ||
      *inputLanes <= 0 ||
      (!properLaneExpansion && !singletonAxisExpansion) ||
      inputLayout.getSew() != resultLayout.getSew() ||
      inputLayout.getLmulEighths() > resultLayout.getLmulEighths() ||
      !exactLeaf(getLeaf(), "rvv", "axis-broadcast",
                 "rvv.axis-broadcast", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({*inputLanes, *resultLanes}))
    return emitOpError(
        "RVV axis broadcast must embed one proper logical sub-domain in a wider lane layout or add only singleton physical axes");
  if (!axisBroadcastTemporaryGroups(*this))
    return emitOpError("RVV axis broadcast has no bounded ordered part/axis relation");
  return mlir::success();
}

mlir::LogicalResult ProjectReductionOperandOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  llvm::SmallVector<int64_t> expectedShape;
  llvm::SmallVector<int64_t> expectedAxes;
  bool found = false;
  for (auto [shape, axis] :
       llvm::zip(input.getShape().asArrayRef(), input.getAxisIds().asArrayRef())) {
    if (axis == getReductionAxis()) {
      if (found)
        return emitOpError("reduction operand axis identity is not unique");
      found = true;
      continue;
    }
    expectedShape.push_back(shape);
    expectedAxes.push_back(axis);
  }
  if (!found || input.getElementType() != result.getElementType() ||
      result.getShape().asArrayRef() != llvm::ArrayRef(expectedShape) ||
      result.getAxisIds().asArrayRef() != llvm::ArrayRef(expectedAxes) ||
      result.getLayout().getCarrier() == "ime" ||
      result.getLayout().getCarrier() == "local" ||
      !exactLeaf(getLeaf(), "transfer", "reduction-projection",
                 "rvv.project-reduction-operand", "none", "exact"))
    return emitOpError(
        "reduction operand projection must remove exactly one logical axis");
  auto inputLayout = input.getLayout();
  auto resultLayout = result.getLayout();
  for (auto [position, axis] :
       llvm::enumerate(result.getAxisIds().asArrayRef())) {
    auto found = llvm::find(input.getAxisIds().asArrayRef(), axis);
    if (found == input.getAxisIds().asArrayRef().end())
      return emitOpError("reduction projection introduced a new free axis");
    size_t sourcePosition =
        static_cast<size_t>(found - input.getAxisIds().asArrayRef().begin());
    if (resultLayout.getTimeFactors()[position] !=
            inputLayout.getTimeFactors()[sourcePosition] ||
        resultLayout.getLaneFactors()[position] !=
            inputLayout.getLaneFactors()[sourcePosition] ||
        resultLayout.getReplicaFactors()[position] !=
            inputLayout.getReplicaFactors()[sourcePosition] ||
        resultLayout.getFragmentFactors()[position] !=
            inputLayout.getFragmentFactors()[sourcePosition] ||
        resultLayout.getLocalFactors()[position] !=
            inputLayout.getLocalFactors()[sourcePosition])
      return emitOpError(
          "reduction projection must preserve every free-axis physical coordinate");
  }
  size_t reduced = static_cast<size_t>(
      llvm::find(input.getAxisIds().asArrayRef(), getReductionAxis()) -
      input.getAxisIds().asArrayRef().begin());
  const int64_t lanes = inputLayout.getLaneFactors()[reduced];
  const int64_t streams = inputLayout.getTimeFactors()[reduced];
  const bool scalarProjection = lanes == 1 &&
      inputLayout.getCarrier() == "scalar" && resultLayout.getCarrier() == "scalar" &&
      inputLayout.getReplicaFactors()[reduced] > 0 && streams > 0 &&
      (inputLayout.getReplicaFactors()[reduced] > 1 || streams > 1);
  const bool timeProjection = lanes == 1 && streams > 1 &&
      inputLayout.getCarrier() == "rvv" && resultLayout.getCarrier() == "rvv" &&
      inputLayout.getLmulEighths() == resultLayout.getLmulEighths() &&
      inputLayout.getVl() == resultLayout.getVl();
  const bool laneProjection = lanes > 1 && streams > 0 &&
      inputLayout.getCarrier() == "rvv" &&
      resultLayout.getCarrier() == "scalar" &&
      llvm::all_of(resultLayout.getLaneFactors().asArrayRef(),
                   [](int64_t factor) { return factor == 1; }) &&
      llvm::all_of(resultLayout.getTimeFactors().asArrayRef(),
                   [](int64_t factor) { return factor == 1; });
  if (inputLayout.getSew() != resultLayout.getSew() ||
      (!scalarProjection && inputLayout.getReplicaFactors()[reduced] != 1) ||
      inputLayout.getFragmentFactors()[reduced] != 1 ||
      inputLayout.getLocalFactors()[reduced] != 1 ||
      (!scalarProjection && !timeProjection && !laneProjection))
    return emitOpError(
        "reduction projection requires an explicit lane or time coordinate; "
        "singleton physical factors do not imply a singleton logical axis");
  return mlir::success();
}

void WEFTRISCVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Weft/Dialect/RISCV/IR/RISCVAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Weft/Dialect/RISCV/IR/RISCVTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"
      >();
}
