#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/Numerics.h"
#include "Weft/Dialect/RISCV/IR/Predicate.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <algorithm>
#include <cstdint>
#include <numeric>

using namespace weft;

bool riscv_internal::hasOwnedLocalStorage(mlir::Value value) {
  return riscv::storageOrigins(value).isOwnedLocal();
}

std::optional<int64_t>
riscv_internal::ownedLocalStorageAlignment(mlir::Value value) {
  auto origins = riscv::storageOrigins(value);
  if (!origins.isOwnedLocal()) return std::nullopt;
  int64_t alignment = 0;
  for (auto allocation : origins.allocations) {
    auto type = mlir::dyn_cast<riscv::LocalType>(allocation.getType());
    if (!type || type.getAlignment() <= 0) return std::nullopt;
    alignment = std::gcd(alignment, type.getAlignment());
  }
  return alignment;
}

int64_t riscv_internal::indexedEntrySourceAlignment(mlir::Value source) {
  if (auto descriptor = mlir::dyn_cast<riscv::MemDescType>(source.getType()))
    return descriptor.getAlignment();
  if (auto field = sourceField(source))
    return fieldFacts(field).alignment;
  if (auto load = sourceLoad(source))
    return load.getRegion().getType().getAlignment();
  if (auto alignment = ownedLocalStorageAlignment(source)) {
    const unsigned bits = logicalBitWidth(source.getType());
    if (bits && bits % 8 == 0)
      return std::gcd(*alignment, static_cast<int64_t>(bits / 8));
  }
  return 1;
}

int64_t riscv_internal::indexedEntryAlignment(mlir::Value source,
                                             mlir::Value byteOffsets,
                                             int64_t entryByteStride) {
  if (entryByteStride <= 0)
    return 1;
  int64_t available = std::gcd(
      std::max<int64_t>(1, indexedEntrySourceAlignment(source)), entryByteStride);
  int64_t alignment = 1;
  while (available % 2 == 0) {
    alignment *= 2;
    available /= 2;
  }
  // The stride describes the table, not arbitrary offset SSA. Preserve only
  // low bits proven on the actual address supplied to this memory operation.
  while (alignment > 1 && !knownMultipleOf(byteOffsets, alignment))
    alignment /= 2;
  return alignment;
}

namespace {

std::optional<int64_t> resolvedPhysicalIndex(mlir::Value value,
                                             mlir::Operation *scope) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto symbol = value.getDefiningOp<riscv::SymbolOp>()) {
    auto kernel = scope->getParentOfType<riscv::KernelOp>();
    if (!kernel)
      return std::nullopt;
    for (auto [name, binding] :
         llvm::zip(kernel.getParameterNames(), kernel.getParameterValues()))
      if (mlir::cast<mlir::StringAttr>(name).getValue() == symbol.getName())
        return binding;
  }
  return std::nullopt;
}

int64_t domainPartition(riscv::DomainType domain, mlir::Operation *scope) {
  int64_t result = 0;
  auto module = scope->getParentOfType<mlir::ModuleOp>();
  module.walk([&](riscv::DomainOp operation) {
    if (operation.getResult().getType().getDomainId() == domain.getDomainId())
      if (auto value = resolvedPhysicalIndex(operation.getPartition(), scope))
        result = *value;
  });
  return result;
}

bool checkedRegularAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedRegularSubtract(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedRegularScale(int64_t value, int64_t factor, int64_t &result) {
  if (factor <= 0 || value > std::numeric_limits<int64_t>::max() / factor ||
      value < std::numeric_limits<int64_t>::min() / factor)
    return false;
  result = value * factor;
  return true;
}

std::optional<riscv_internal::RegularIndexRelation>
analyzeRegularIndexRelationImpl(mlir::Value value, unsigned depth) {
  if (depth > 16)
    return std::nullopt;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    return analyzeRegularIndexRelationImpl(conversion.getInput(), depth + 1);
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    if (iota.getStart() >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    return riscv_internal::RegularIndexRelation{
        {}, static_cast<int64_t>(iota.getStart()), 1, 1};
  }
  auto binary = value.getDefiningOp<riscv::BinaryOp>();
  if (!binary)
    return std::nullopt;

  auto lhs = analyzeRegularIndexRelationImpl(binary.getLhs(), depth + 1);
  auto rhs = analyzeRegularIndexRelationImpl(binary.getRhs(), depth + 1);
  auto lhsConstant = riscv_internal::constantInt(binary.getLhs());
  auto rhsConstant = riscv_internal::constantInt(binary.getRhs());
  auto scalarCoordinate = [](mlir::Value candidate) {
    mlir::Type type = candidate.getType();
    if (type.isIndex())
      return true;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
      return !integer.isSigned();
    auto physical = mlir::dyn_cast<riscv::ValueType>(type);
    if (!physical || physical.getLayout().getCarrier() != "scalar")
      return false;
    auto productIsOne = [](llvm::ArrayRef<int64_t> factors) {
      auto product = riscv_internal::staticProduct(factors);
      return product && *product == 1;
    };
    return productIsOne(
               physical.getLayout().getTimeFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getLaneFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getReplicaFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getFragmentFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getLocalFactors().asArrayRef());
  };
  if (binary.getKind() == "add") {
    if (lhs && rhsConstant)
      return checkedRegularAdd(lhs->base, *rhsConstant, lhs->base)
                 ? lhs
                 : std::nullopt;
    if (rhs && lhsConstant)
      return checkedRegularAdd(rhs->base, *lhsConstant, rhs->base)
                 ? rhs
                 : std::nullopt;
    if (lhs && !lhs->dynamicBase && scalarCoordinate(binary.getRhs())) {
      lhs->dynamicBase = binary.getRhs();
      return lhs;
    }
    if (rhs && !rhs->dynamicBase && scalarCoordinate(binary.getLhs())) {
      rhs->dynamicBase = binary.getLhs();
      return rhs;
    }
  }
  if (binary.getKind() == "sub" && lhs && rhsConstant)
    return checkedRegularSubtract(lhs->base, *rhsConstant, lhs->base)
               ? lhs
               : std::nullopt;
  if (binary.getKind() == "mul") {
    auto scale = [&](riscv_internal::RegularIndexRelation pattern,
                     int64_t factor)
        -> std::optional<riscv_internal::RegularIndexRelation> {
      if (pattern.dynamicBase && factor != 1)
        return std::nullopt;
      if (!checkedRegularScale(pattern.base, factor, pattern.base) ||
          !checkedRegularScale(pattern.stride, factor, pattern.stride))
        return std::nullopt;
      return pattern;
    };
    if (lhs && rhsConstant)
      return scale(*lhs, *rhsConstant);
    if (rhs && lhsConstant)
      return scale(*rhs, *lhsConstant);
  }
  if (binary.getKind() == "div" && lhs && !lhs->dynamicBase &&
      rhsConstant && *rhsConstant > 0 &&
      lhs->base % *rhsConstant == 0 && lhs->stride == 1 &&
      lhs->repeat <= std::numeric_limits<int64_t>::max() / *rhsConstant) {
    lhs->base /= *rhsConstant;
    lhs->repeat *= *rhsConstant;
    return lhs;
  }
  return std::nullopt;
}

} // namespace

mlir::DenseI64ArrayAttr
riscv_internal::integers(mlir::Builder &builder,
                         llvm::ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

std::optional<riscv_internal::RegularIndexRelation>
riscv_internal::analyzeRegularIndexRelation(mlir::Value value,
                                            unsigned depth) {
  return analyzeRegularIndexRelationImpl(value, depth);
}

int64_t riscv_internal::physicalExtent(mlir::Value value, int64_t axis) {
  auto axes = logicalAxes(value.getType());
  auto shape = logicalShape(value.getType());
  auto found = llvm::find(axes, axis);
  if (found == axes.end())
    return 1;
  const int64_t direct = shape[static_cast<size_t>(found - axes.begin())];
  if (direct > 0)
    return direct;
  for (mlir::Operation *parent = value.getParentRegion()->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto loop = mlir::dyn_cast<riscv::LoopOp>(parent)) {
      riscv::DomainType domain = loop.getDomain().getType();
      if (domain.getAxisId() == axis)
        return domainPartition(domain, parent);
    }
  return 0;
}

llvm::StringRef riscv_internal::layeredStorageDecodeInstruction(
    int64_t shiftAmount, int64_t maskValue) {
  if (shiftAmount > 0)
    return maskValue > 0 ? "rvv.layered-storage-decode.shift-mask"
                         : "rvv.layered-storage-decode.shift";
  return maskValue > 0 ? "rvv.layered-storage-decode.mask"
                       : "rvv.layered-storage-decode.identity";
}

std::optional<riscv::StorageWindowPlanAttr>
riscv_internal::storageWindowPlan(
    mlir::Builder &builder, riscv::FieldOp field, int64_t reductionAxis,
    int64_t projectionBase, int64_t projectionStride,
    int64_t projectionRepeat, int64_t projectionExtent,
    int64_t offsetAlignment, bool recordProjection) {
  riscv::LoadOp load = sourceLoad(field.getOwner());
  auto memory = load ? load.getRegion().getType() : riscv::MemDescType();
  auto fieldType =
      mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
  auto integer = fieldType
                     ? mlir::dyn_cast<mlir::IntegerType>(
                           fieldType.getElementType())
                     : mlir::IntegerType();
  auto floating = fieldType
                      ? mlir::dyn_cast<mlir::FloatType>(
                            fieldType.getElementType())
                      : mlir::FloatType();
  const int64_t elementBits =
      integer ? integer.getWidth() : floating ? floating.getWidth() : 0;
  riscv::AccessAttr access = field.getAccess();
  if (!memory || !fieldType || !access || reductionAxis <= 0 ||
      projectionBase < 0 || projectionStride <= 0 || projectionRepeat <= 0 ||
      projectionExtent <= 0 || offsetAlignment <= 0 ||
      memory.getElements() <= 0 || access.getBitOffset() < 0 ||
      access.getBitOffset() % 8 || elementBits <= 0)
    return std::nullopt;

  llvm::StringRef kind;
  int64_t group = 0;
  int64_t layer = 0;
  int64_t physicalLayerBase = 0;
  int64_t physicalLayerStep = 0;
  int64_t shiftBase = 0;
  int64_t shiftStep = 0;
  int64_t maskValue = 0;
  auto ownerElement = logicalElement(field.getOwner().getType());
  auto ownerEncoding = mlir::dyn_cast<kernel::EncodingType>(ownerElement);
  const int64_t recordInterleave =
      ownerEncoding ? interleaveRows(field, ownerEncoding) : 0;
  if (recordInterleave > 1 && access.getFieldChunkBytes() != 1)
    return std::nullopt;
  if (recordProjection && recordInterleave > 0 &&
      access.getMapping() == "natural") {
    kind = "interleaved_natural";
  } else if (recordProjection && recordInterleave > 0 &&
             access.getMapping() == "joined") {
    kind = "interleaved_joined";
  } else if (access.getMapping() == "natural") {
    kind = projectionRepeat > 1
               ? "repeat"
               : projectionStride == 1 ? "unit" : "strided";
  } else if (access.getMapping() == "grouped_layered") {
    group = access.getGroupSize();
    layer = access.getLayerSize();
    const int64_t layers = layer > 0 ? group / layer : 0;
    if (!integer || integer.isSigned() || elementBits >= 8 || group <= 0 ||
        layer <= 0 || group % layer || layers <= 1 ||
        (access.getOrder() != "lo_first" &&
         access.getOrder() != "hi_first"))
      return std::nullopt;
    kind = "layered";
    physicalLayerBase = access.getOrder() == "lo_first" ? 0 : layers - 1;
    physicalLayerStep = access.getOrder() == "lo_first" ? 1 : -1;
    shiftBase = physicalLayerBase * elementBits;
    shiftStep = physicalLayerStep * elementBits;
    maskValue = (int64_t{1} << elementBits) - 1;
  } else {
    return std::nullopt;
  }
  return riscv::StorageWindowPlanAttr::get(
      builder.getContext(), kind, reductionAxis, projectionBase,
      projectionStride, projectionRepeat, projectionExtent, offsetAlignment,
      memory.getElements(), access.getBitOffset() / 8, elementBits, group,
      layer, physicalLayerBase, physicalLayerStep, shiftBase, shiftStep,
      maskValue);
}

std::optional<riscv_internal::WidenDotLaneSlicePlan>
riscv_internal::planWidenDotLaneSlices(
    riscv::ValueType operand, riscv::ValueType result,
    llvm::ArrayRef<int64_t> reductionAxes, riscv::TargetAttr target) {
  auto layout = operand.getLayout();
  auto lanes = staticProduct(layout.getLaneFactors().asArrayRef());
  if (!lanes || *lanes <= 0)
    return std::nullopt;

  int64_t reductionLanes = 1;
  bool sawFreeLane = false;
  for (int64_t position = static_cast<int64_t>(operand.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const int64_t lane = layout.getLaneFactors()[position];
    if (lane <= 1)
      continue;
    const int64_t axis = operand.getAxisIds()[position];
    if (llvm::is_contained(reductionAxes, axis)) {
      if (sawFreeLane ||
          reductionLanes > std::numeric_limits<int64_t>::max() / lane)
        return std::nullopt;
      reductionLanes *= lane;
    } else {
      sawFreeLane = true;
    }
  }
  if (reductionLanes <= 0 || *lanes % reductionLanes)
    return std::nullopt;

  auto sliceLmul =
      riscv::rvvLaneSliceLMULEighths(target, operand, reductionLanes);
  if (!sliceLmul ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), *sliceLmul))
    return std::nullopt;

  int64_t outputParts = 1;
  if (result) {
    if (result.getLayout().getCarrier() != "scalar")
      return std::nullopt;
    outputParts = staticProduct(result.getLayout().getReplicaFactors().asArrayRef())
                      .value_or(-1);
    if (outputParts <= 0)
      return std::nullopt;
  }

  WidenDotLaneSlicePlan plan;
  plan.reductionLanes = reductionLanes;
  plan.sliceLmulEighths = *sliceLmul;
  llvm::SmallVector<int64_t> reductionTimeExtents;
  for (int64_t axis : reductionAxes) {
    auto found = llvm::find(operand.getAxisIds().asArrayRef(), axis);
    if (found == operand.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t position = static_cast<size_t>(
        found - operand.getAxisIds().asArrayRef().begin());
    const int64_t extent = layout.getTimeFactors()[position];
    if (extent <= 0 || layout.getReplicaFactors()[position] != 1)
      return std::nullopt;
    reductionTimeExtents.push_back(extent);
  }
  plan.reductionStreams = staticProduct(reductionTimeExtents).value_or(-1);
  const int64_t totalStreams =
      staticProduct(layout.getTimeFactors().asArrayRef()).value_or(-1);
  const int64_t registerParts =
      staticProduct(layout.getReplicaFactors().asArrayRef()).value_or(-1);
  if (plan.reductionStreams <= 0 || totalStreams <= 0 || registerParts <= 0 ||
      registerParts > std::numeric_limits<int64_t>::max() / totalStreams)
    return std::nullopt;
  const int64_t totalParts = registerParts * totalStreams;

  for (int64_t outputPart = 0; outputPart < outputParts; ++outputPart) {
    llvm::SmallVector<int64_t> resultCoordinates(
        result ? result.getAxisIds().size() : 0, 0);
    int64_t remaining = outputPart;
    if (result) {
      for (int64_t position = static_cast<int64_t>(result.getAxisIds().size()) - 1;
           position >= 0; --position) {
        const int64_t replica = result.getLayout().getReplicaFactors()[position];
        if (replica <= 0)
          return std::nullopt;
        resultCoordinates[position] = remaining % replica;
        remaining /= replica;
      }
      if (remaining)
        return std::nullopt;
    }

    auto resultCoordinateForAxis = [&](int64_t axis) -> std::optional<int64_t> {
      if (!result)
        return std::nullopt;
      auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
      if (found == result.getAxisIds().asArrayRef().end())
        return std::nullopt;
      return resultCoordinates[static_cast<size_t>(
          found - result.getAxisIds().asArrayRef().begin())];
    };

    int64_t laneOffset = 0;
    int64_t laneStride = 1;
    for (int64_t position = static_cast<int64_t>(operand.getAxisIds().size()) - 1;
         position >= 0; --position) {
      const int64_t lane = layout.getLaneFactors()[position];
      if (lane <= 1)
        continue;
      const int64_t axis = operand.getAxisIds()[position];
      int64_t coordinate = 0;
      if (!llvm::is_contained(reductionAxes, axis)) {
        auto logicalCoordinate = resultCoordinateForAxis(axis);
        if (!logicalCoordinate || *logicalCoordinate < 0)
          return std::nullopt;
        coordinate = *logicalCoordinate % lane;
      }
      if (coordinate > 0 &&
          laneOffset > std::numeric_limits<int64_t>::max() -
                           coordinate * laneStride)
        return std::nullopt;
      laneOffset += coordinate * laneStride;
      if (laneStride > std::numeric_limits<int64_t>::max() / lane)
        return std::nullopt;
      laneStride *= lane;
    }
    if (laneOffset < 0 || laneOffset % reductionLanes ||
        laneOffset > *lanes - reductionLanes)
      return std::nullopt;
    plan.offsets.push_back(laneOffset);

    for (int64_t reductionStream = 0;
         reductionStream < plan.reductionStreams; ++reductionStream) {
      llvm::SmallVector<int64_t> reductionCoordinates(reductionAxes.size(), 0);
      int64_t remainingReduction = reductionStream;
      for (int64_t index = static_cast<int64_t>(reductionAxes.size()) - 1;
           index >= 0; --index) {
        reductionCoordinates[index] =
            remainingReduction % reductionTimeExtents[index];
        remainingReduction /= reductionTimeExtents[index];
      }
      if (remainingReduction)
        return std::nullopt;

      int64_t streamPart = 0;
      int64_t registerPart = 0;
      for (size_t position = 0; position < operand.getAxisIds().size();
           ++position) {
        const int64_t axis = operand.getAxisIds()[position];
        const int64_t time = layout.getTimeFactors()[position];
        const int64_t lane = layout.getLaneFactors()[position];
        const int64_t replica = layout.getReplicaFactors()[position];
        int64_t timeCoordinate = 0;
        int64_t replicaCoordinate = 0;
        if (auto reduction = llvm::find(reductionAxes, axis);
            reduction != reductionAxes.end()) {
          timeCoordinate = reductionCoordinates[static_cast<size_t>(
              reduction - reductionAxes.begin())];
        } else {
          auto logicalCoordinate = resultCoordinateForAxis(axis);
          if (!logicalCoordinate)
            return std::nullopt;
          if (lane > 1) {
            if (replica != 1 || *logicalCoordinate >= time * lane)
              return std::nullopt;
            timeCoordinate = *logicalCoordinate / lane;
          } else {
            if (*logicalCoordinate >= time * replica)
              return std::nullopt;
            timeCoordinate = *logicalCoordinate / replica;
            replicaCoordinate = *logicalCoordinate % replica;
          }
        }
        streamPart = streamPart * time + timeCoordinate;
        registerPart = registerPart * replica + replicaCoordinate;
      }
      const int64_t physicalPart = registerPart * totalStreams + streamPart;
      if (physicalPart < 0 || physicalPart >= totalParts)
        return std::nullopt;
      plan.parts.push_back(physicalPart);
    }
  }
  return plan;
}

mlir::ArrayAttr riscv_internal::strings(
    mlir::Builder &builder, llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute> attributes;
  for (const std::string &value : values)
    attributes.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attributes);
}

mlir::DictionaryAttr riscv_internal::dictionary(
    mlir::Builder &builder,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> values) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (const auto &[name, value] : values)
    if (value)
      attributes.emplace_back(builder.getStringAttr(name), value);
  return builder.getDictionaryAttr(attributes);
}

std::optional<int64_t>
riscv_internal::integer(mlir::DictionaryAttr dictionary,
                        llvm::StringRef name) {
  if (!dictionary)
    return std::nullopt;
  if (auto value = dictionary.getAs<mlir::IntegerAttr>(name))
    return value.getInt();
  return std::nullopt;
}

std::optional<llvm::StringRef>
riscv_internal::string(mlir::DictionaryAttr dictionary, llvm::StringRef name) {
  if (!dictionary)
    return std::nullopt;
  if (auto value = dictionary.getAs<mlir::StringAttr>(name))
    return value.getValue();
  return std::nullopt;
}

mlir::ArrayAttr riscv_internal::array(mlir::DictionaryAttr dictionary,
                                      llvm::StringRef name) {
  return dictionary ? dictionary.getAs<mlir::ArrayAttr>(name)
                    : mlir::ArrayAttr();
}

riscv::LayoutAttr
riscv_internal::unassignedLayout(mlir::Builder &builder,
                                 llvm::ArrayRef<int64_t> axes) {
  llvm::SmallVector<int64_t> zero(axes.size(), 0);
  return riscv::LayoutAttr::get(builder.getContext(), "unassigned",
                                integers(builder, axes), integers(builder, zero),
                                integers(builder, zero), integers(builder, zero),
                                integers(builder, zero), integers(builder, zero),
                                0, 0, 0, 0, "unassigned");
}

namespace {

unsigned bitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (type.isIndex())
    return 64;
  return 8;
}

} // namespace

riscv::LayoutAttr riscv_internal::scalarLayout(
    mlir::Builder &builder, mlir::Type element, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> axes, llvm::StringRef validity) {
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  for (int64_t extent : shape)
    time.push_back(extent > 0 ? extent : 1);
  return riscv::LayoutAttr::get(
      builder.getContext(), "scalar", integers(builder, axes),
      integers(builder, time), integers(builder, one), integers(builder, one),
      integers(builder, one), integers(builder, one), bitWidth(element), 0, 1,
      0, validity);
}

riscv::LayoutAttr riscv_internal::projectLayout(
    mlir::Builder &builder, riscv::ValueType source,
    riscv::LayoutAttr reference, riscv::TargetAttr target) {
  auto axes = source.getAxisIds().asArrayRef();
  auto referenceAxes = reference.getAxisIds().asArrayRef();
  auto project = [&](mlir::DenseI64ArrayAttr factors,
                     bool sequentialWhenAbsent) {
    llvm::SmallVector<int64_t> result;
    for (auto [index, axis] : llvm::enumerate(axes)) {
      // A broadcast singleton retains one source coordinate even when its
      // axis identity also names a larger result domain. Copying that result's
      // factors creates an invalid type before a conversion can be selected.
      if (source.getShape()[index] == 1) {
        result.push_back(1);
        continue;
      }
      auto found = llvm::find(referenceAxes, axis);
      result.push_back(
          found == referenceAxes.end()
              ? (sequentialWhenAbsent
                     ? std::max<int64_t>(1, source.getShape()[index])
                     : 1)
              : factors[static_cast<size_t>(found - referenceAxes.begin())]);
    }
    return result;
  };
  auto time = project(reference.getTimeFactors(), true);
  auto lane = project(reference.getLaneFactors(), false);
  auto replica = project(reference.getReplicaFactors(), false);
  auto fragment = project(reference.getFragmentFactors(), false);
  auto local = project(reference.getLocalFactors(), false);
  const int64_t groupedAxis = riscv::replicaGroupAxis(reference);
  const auto groupedSource = groupedAxis >= 0 ? llvm::find(axes, referenceAxes[groupedAxis]) : axes.end();
  const int64_t group = groupedSource != axes.end() &&
                               source.getShape()[groupedSource - axes.begin()] != 1
                            ? reference.getReplicaGroup() : 1;
  int64_t laneCount = group;
  for (int64_t factor : lane)
    laneCount *= factor;
  llvm::StringRef carrier = reference.getCarrier();
  if (carrier == "rvv" && laneCount <= 1)
    carrier = "scalar";
  int64_t sew = std::max<int64_t>(8, logicalBitWidth(source));
  int64_t lmul = 0;
  if (carrier == "rvv") {
    int64_t elen = 0;
    for (int64_t supported : target.getSupportedSEW().asArrayRef())
      elen = std::max(elen, supported);
    if (elen <= 0)
      return {};
    int64_t requested = std::max<int64_t>(
        1, (laneCount * sew * 8 + target.getVlenBits() - 1) /
               target.getVlenBits());
    requested = std::max(requested, (8 * sew + elen - 1) / elen);
    for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
      if (legal >= requested) {
        lmul = legal;
        break;
      }
    if (!lmul)
      return {};
  }
  int64_t replicaCount = 1;
  for (int64_t factor : replica)
    replicaCount *= factor;
  int64_t groups =
      carrier == "rvv" ? ((lmul + 7) / 8) * replicaCount / group : 0;
  return riscv::LayoutAttr::get(
      builder.getContext(), carrier, source.getAxisIds(),
      integers(builder, time), integers(builder, lane),
      integers(builder, replica), integers(builder, fragment),
      integers(builder, local), sew, lmul, carrier == "rvv" ? laneCount : 1,
      groups, reference.getValidity(), group);
}

riscv::AccessAttr
riscv_internal::unassignedAccess(mlir::Builder &builder) {
  return riscv::AccessAttr::get(builder.getContext(), "unassigned", "opaque",
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1);
}

riscv::AccessAttr riscv_internal::denseAccess(mlir::Builder &builder,
                                              riscv::MemDescType memory,
                                              mlir::Type physicalValue) {
  riscv::LayoutAttr layout = layoutOf(physicalValue);
  int64_t mappedAxis = 0;
  if (layout) {
    for (auto [axis, factor] : llvm::zip(
             layout.getAxisIds().asArrayRef(),
             layout.getLaneFactors().asArrayRef()))
      if (factor > 1) {
        mappedAxis = axis;
        break;
      }
    if (!mappedAxis)
      for (auto [axis, factor] : llvm::zip(
               layout.getAxisIds().asArrayRef(),
               layout.getReplicaFactors().asArrayRef()))
        if (factor > 1) {
          mappedAxis = axis;
          break;
        }
  }
  llvm::StringRef form = "unit";
  if (mappedAxis) {
    auto axes = memory.getAxisIds().asArrayRef();
    auto found = llvm::find(axes, mappedAxis);
    if (found == axes.end()) {
      form = "indexed";
    } else {
      const size_t dimension = static_cast<size_t>(found - axes.begin());
      form = dimension < memory.getStrides().size() &&
                     memory.getStrides()[dimension] == 1
                 ? "unit"
                 : "strided";
    }
  }
  return riscv::AccessAttr::get(
      builder.getContext(), form, "dense",
      std::max<int64_t>(1, memory.getAlignment()),
      form == "indexed" ? 32 : 0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1);
}

riscv::ConversionAttr
riscv_internal::conversion(mlir::Builder &builder, llvm::StringRef kind,
                           llvm::StringRef effect,
                           int64_t temporaryGroups) {
  return riscv::ConversionAttr::get(builder.getContext(), kind, effect,
                                    temporaryGroups);
}

riscv::ConversionAttr
riscv_internal::layoutConversion(mlir::Builder &builder,
                                 riscv::LayoutAttr source,
                                 riscv::LayoutAttr target) {
  llvm::StringRef kind = "reshape";
  llvm::StringRef effect = "pure";
  if (source.getCarrier() == target.getCarrier()) {
    bool registerToLane = false;
    bool timeToLane = false;
    bool laneToRegister = false;
    auto sourceAxes = source.getAxisIds().asArrayRef();
    auto targetAxes = target.getAxisIds().asArrayRef();
    for (auto [index, axis] : llvm::enumerate(sourceAxes)) {
      auto found = llvm::find(targetAxes, axis);
      if (found == targetAxes.end())
        continue;
      size_t targetIndex = static_cast<size_t>(found - targetAxes.begin());
      const int64_t sourceLane = source.getLaneFactors()[index];
      const int64_t targetLane = target.getLaneFactors()[targetIndex];
      registerToLane |=
          targetLane > sourceLane &&
          source.getReplicaFactors()[index] >
              target.getReplicaFactors()[targetIndex];
      timeToLane |= targetLane > sourceLane &&
                    source.getTimeFactors()[index] >
                        target.getTimeFactors()[targetIndex];
      laneToRegister |=
          sourceLane > targetLane &&
          target.getReplicaFactors()[targetIndex] >
              source.getReplicaFactors()[index];
    }
    kind = registerToLane ? "register_to_lane"
           : timeToLane   ? "time_to_lane"
           : laneToRegister ? "lane_to_register"
                            : "tuple";
  } else if (source.getCarrier() == "scalar" &&
             target.getCarrier() == "rvv") {
    bool registerToLane = false;
    bool timeToLane = false;
    auto sourceAxes = source.getAxisIds().asArrayRef();
    auto targetAxes = target.getAxisIds().asArrayRef();
    for (auto [index, axis] : llvm::enumerate(sourceAxes)) {
      auto found = llvm::find(targetAxes, axis);
      if (found == targetAxes.end())
        continue;
      size_t targetIndex = static_cast<size_t>(found - targetAxes.begin());
      if (target.getLaneFactors()[targetIndex] <= 1)
        continue;
      registerToLane |= source.getReplicaFactors()[index] >
                        target.getReplicaFactors()[targetIndex];
      timeToLane |= source.getTimeFactors()[index] >
                    target.getTimeFactors()[targetIndex];
    }
    // Moving both issue-time and register factors into one lane axis needs a
    // distinct typed conversion.  Keep it illegal instead of assigning either
    // existing kind and relying on the emitter to reinterpret the layout.
    kind = registerToLane && timeToLane
               ? "reshape"
           : registerToLane ? "register_to_lane"
           : timeToLane     ? "time_to_lane"
                            : "splat";
  } else if (source.getCarrier() == "rvv" &&
             target.getCarrier() == "scalar") {
    kind = "extract";
  } else if (source.getCarrier() == "local") {
    kind = "local_load";
    effect = "read";
  } else if (target.getCarrier() == "local") {
    kind = "local_store";
    effect = "write";
  }
  return conversion(builder, kind, effect);
}

bool riscv_internal::selectLaneRepack(mlir::Builder &builder, riscv::ConvertLayoutOp operation) {
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  auto conversion = operation.getConversion();
  if (!kernel || kernel.getResourcesMaterialized() || conversion.getEffect() != "pure" ||
      (conversion.getKind() != "register_to_lane" && conversion.getKind() != "time_to_lane" &&
       conversion.getKind() != "tuple") ||
      riscv::fieldReadProjection(operation.getInput())) return false;
  auto source = operation.getInput().getType(), target = operation.getResult().getType();
  if (conversion.getKind() == "tuple" &&
      source.getLayout().getReplicaFactors() == target.getLayout().getReplicaFactors())
    return false;
  bool laneMovesOut = false;
  if (source.getAxisIds() == target.getAxisIds())
    for (auto [before, after] : llvm::zip(source.getLayout().getLaneFactors().asArrayRef(),
                                         target.getLayout().getLaneFactors().asArrayRef()))
      laneMovesOut |= before > after;
  if ((!laneMovesOut && riscv::rvvPartToLanePieces(source, target)) ||
      !riscv::rvvLayoutRepackSlices(source, target)) return false;
  // Exchanging lane and replica axes preserves ordered coordinates; concatenating
  // the old vector parts does not. Numeric field reads use this same relation.
  operation.setConversionAttr(riscv_internal::conversion(builder, "lane_repack", "pure",
      riscv::rvvLayoutRepackTemporaryGroups(operation->getParentOfType<riscv::KernelOp>().getTarget(), source, target)));
  operation.setLeafAttr(unselectedLeaf(builder));
  return true;
}

riscv::ScheduleAttr riscv_internal::schedule(mlir::Builder &builder,
                                             int64_t unroll,
                                             int64_t pipelineDepth,
                                             int64_t prefetchDistance) {
  return riscv::ScheduleAttr::get(
      builder.getContext(), pipelineDepth > 1 ? "pipelined" : "sequential",
      unroll, pipelineDepth, pipelineDepth, prefetchDistance);
}

riscv::ImplementationAttr riscv_internal::implementation(
    mlir::Builder &builder, llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef operation, llvm::ArrayRef<int64_t> parameters) {
  return riscv::ImplementationAttr::get(builder.getContext(), engine, family,
                                        operation, integers(builder, parameters));
}

riscv::LeafAttr riscv_internal::unselectedLeaf(mlir::Builder &builder) {
  return riscv::LeafAttr::get(builder.getContext(), "unselected", "", "", "",
                              0, 0, 0, 0, "none", "exact",
                              integers(builder, {}), 0);
}

riscv::LeafAttr riscv_internal::leaf(
    mlir::Builder &builder, llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef instruction, llvm::StringRef spelling,
    int64_t operandGroups, int64_t resultGroups, int64_t temporaryGroups,
    int64_t fragmentGroups, llvm::StringRef mask, llvm::StringRef tail,
    llvm::ArrayRef<int64_t> parameters, int64_t localBytes) {
  return riscv::LeafAttr::get(
      builder.getContext(), engine, family, instruction, spelling, operandGroups,
      resultGroups, temporaryGroups, fragmentGroups, mask, tail,
      integers(builder, parameters), localBytes);
}

riscv::LeafAttr riscv_internal::readSnapshotLeaf(
    mlir::Builder &builder, riscv::TargetAttr target, int64_t bytes) {
  if (target.getHasRVV()) {
    int64_t selectedLMUL = 8;
    int64_t chunkBytes = target.getVlenBits() / 8;
    int64_t transfers = bytes / chunkBytes + (bytes % chunkBytes != 0);
    for (int64_t lmul : {16, 32}) {
      if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul))
        continue;
      const int64_t capacity = target.getVlenBits() * lmul / 64;
      const int64_t count = bytes / capacity + (bytes % capacity != 0);
      if (count < transfers) {
        selectedLMUL = lmul;
        chunkBytes = capacity;
        transfers = count;
      }
    }
    return leaf(builder, "transfer", "load", "rvv.read-snapshot",
                "rvv.read-snapshot", 0, 0, selectedLMUL / 8, 0, "none", "exact",
                {selectedLMUL, chunkBytes});
  }
  return leaf(builder, "transfer", "load", "scalar.read-snapshot",
              "scalar.read-snapshot", 0, 0);
}

riscv::TargetAttr
riscv_internal::target(mlir::Builder &builder,
                       const RISCVTargetProfile &profile) {
  llvm::SmallVector<int64_t> sews;
  llvm::SmallVector<int64_t> lmuls;
  llvm::SmallVector<mlir::Attribute> fragments;
  for (unsigned sew : profile.supportedSEW)
    sews.push_back(sew);
  for (int lmul : profile.legalLMULEighths)
    lmuls.push_back(lmul);
  llvm::StringRef partialCombinePolicy;
  switch (profile.partialCombinePolicy) {
  case RISCVPartialCombinePolicy::IndependentMultilevel:
    partialCombinePolicy = "independent-multilevel";
    break;
  case RISCVPartialCombinePolicy::Sequential:
    partialCombinePolicy = "sequential";
    break;
  }
  llvm::StringRef recordAxisPolicy;
  switch (profile.recordAxisPolicy) {
  case RISCVRecordAxisPolicy::WithinRecord:
    recordAxisPolicy = "within-record";
    break;
  case RISCVRecordAxisPolicy::AcrossRecords:
    recordAxisPolicy = "across-records";
    break;
  }
  for (const RISCVFragmentCapability &fragment : profile.fragmentCapabilities)
    fragments.push_back(riscv::fragmentCapabilityAttr(builder.getContext(), fragment));
  mlir::Attribute cache = builder.getUnitAttr();
  if (profile.dataCache) {
    const auto &geometry = *profile.dataCache;
    cache = riscv::DataCacheAttr::get(builder.getContext(), geometry.sizeBytes,
        geometry.lineBytes, geometry.ways, geometry.sets);
  }
  return riscv::TargetAttr::get(
      builder.getContext(), profile.triple, profile.march, profile.abi,
      profile.hasRVV, profile.hasVectorF16, profile.hasVectorF16Conversion, profile.hasIndexedMemory,
      profile.hasSegmentMemory, profile.hasWideningInteger,
      profile.hasWideningFloat,
      profile.vlenBits, profile.vectorRegisters, profile.maxPrivateStackBytes,
      integers(builder, sews), integers(builder, lmuls),
      profile.maxWideningCombineGroups, partialCombinePolicy, recordAxisPolicy,
      builder.getArrayAttr(fragments), cache);
}

mlir::Type riscv_internal::logicalElement(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getElementType();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getEncoding();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getEncoding();
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getElementType();
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(type))
    return memory.getEncoding();
  if (auto local = mlir::dyn_cast<riscv::LocalType>(type))
    return local.getElementType();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool riscv_internal::requiresFullVectorF16(mlir::Operation *operation) {
  auto halfVector = [](mlir::Type type) {
    auto value = mlir::dyn_cast<riscv::ValueType>(type);
    return value && value.getElementType().isF16() &&
        (value.getLayout().getCarrier() == "rvv" ||
         value.getLayout().getCarrier() == "unassigned");
  };
  if (!llvm::any_of(operation->getOperandTypes(), halfVector) &&
      !llvm::any_of(operation->getResultTypes(), halfVector))
    return false;
  if (mlir::isa<riscv::LoadOp, riscv::StoreOp, riscv::PointerLoadOp, riscv::PointerStoreOp, riscv::RVVTileLoadOp,
                riscv::RVVTileStoreOp, riscv::RVVDenseSegmentLoadOp,
                riscv::RVVDenseSegmentStoreOp, riscv::FieldReadOp,
                riscv::BitcastOp, riscv::RegisterMaterializeOp,
                riscv::SpillOp, riscv::ReloadOp>(operation))
    return false;
  if (auto fresh = mlir::dyn_cast<riscv::NewOp>(operation))
    return fresh.getInitialized() &&
        (!fresh.getInitial() || fresh.getInitial().getType() != fresh.getResult().getType());
  if (mlir::isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp>(operation) &&
      operation->getNumOperands() == 1 && operation->getNumResults() == 1) {
    auto input = logicalElement(operation->getOperand(0).getType());
    auto result = logicalElement(operation->getResult(0).getType());
    return !((input.isF16() && (result.isF32() || result.isF16())) ||
             (input.isF32() && result.isF16()));
  }
  // A source f16 storage contraction with f32 accumulation can use the existing
  // explicit f16->f32 supplies and f32 FMA. A selected native half FMA step is a
  // different operation and still requires full Zvfh.
  if (mlir::isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(operation) &&
      operation->getNumResults() == 1 &&
      logicalElement(operation->getResult(0).getType()).isF32())
    return false;
  return true;
}

llvm::ArrayRef<int64_t> riscv_internal::logicalShape(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getShape().asArrayRef();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getShape().asArrayRef();
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(type))
    return memory.getShape().asArrayRef();
  if (auto local = mlir::dyn_cast<riscv::LocalType>(type))
    return local.getShape().asArrayRef();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> riscv_internal::logicalAxes(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getAxisIds().asArrayRef();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getAxisIds().asArrayRef();
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(type))
    return memory.getAxisIds().asArrayRef();
  if (auto local = mlir::dyn_cast<riscv::LocalType>(type))
    return local.getAxisIds().asArrayRef();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getAxisIds().asArrayRef();
  return {};
}

unsigned riscv_internal::logicalBitWidth(mlir::Type type) {
  return bitWidth(logicalElement(type));
}

riscv::LayoutAttr riscv_internal::layoutOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getLayout();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getLayout();
  return {};
}

int64_t riscv_internal::issueRegisterGroups(mlir::Type type) {
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getResourceGroups();
  if (auto window = mlir::dyn_cast<riscv::WindowType>(type))
    return window.getResourceGroups();
  if (auto window = mlir::dyn_cast<riscv::LayeredWindowType>(type))
    return window.getResourceGroups();
  if (auto partials = mlir::dyn_cast<riscv::PartialSetType>(type))
    return partials.getResourceGroups();
  if (auto layout = layoutOf(type))
    return layout.getRegisterGroups();
  return 0;
}

std::optional<int64_t>
riscv_internal::residentRegisterGroups(mlir::Value value) {
  if (riscv::isPredicateRegister(value))
    return riscv::predicatePartCount(mlir::cast<riscv::ValueType>(value.getType()));
  auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!type)
    return issueRegisterGroups(value.getType());
  // Encoded records and field projections name storage. Only an explicit
  // numeric read (or another numeric producer) owns RVV register parts.
  if (type.getLayout().getCarrier() != "rvv" ||
      mlir::isa<kernel::EncodingType>(type.getElementType()) ||
      riscv::fieldReadProjection(value))
    return 0;
  auto streams = staticProduct(type.getLayout().getTimeFactors().asArrayRef());
  const int64_t groups = type.getLayout().getRegisterGroups();
  if (!streams || groups < 0 ||
      (groups && *streams > std::numeric_limits<int64_t>::max() / groups))
    return std::nullopt;
  // A real issue loop already narrows its SSA window type. A time factor on
  // the remaining numeric value is not, by itself, a register-reuse schedule.
  return groups * *streams;
}

int64_t riscv_internal::issueRegisterGroups(mlir::Value value) {
  if (riscv::isPredicateRegister(value))
    return *riscv::predicateIssueGroups(mlir::cast<riscv::ValueType>(value.getType()));
  return issueRegisterGroups(value.getType());
}

mlir::Value riscv_internal::registerStorageIdentity(mlir::Value value) {
  while (auto source = riscv::registerAliasSource(value))
    value = source;
  return value;
}

void riscv_internal::selectFragmentRegisterReuse(riscv::KernelOp kernel) {
  kernel.walk([&](riscv::IMEFragmentMMAOp mma) {
    const bool reuse = riscv::canReuseFragmentAccumulator(mma);
    mma.setAccumulatorPolicy(reuse ? "reuse" : "copy");
    auto capability = riscv::fragmentCapabilityAttr(kernel.getContext(),
        *findRISCVFragmentCapability(mma.getLeaf().getInstruction()));
    auto old = mma.getLeaf();
    mma.setLeafAttr(riscv::LeafAttr::get(kernel.getContext(), old.getEngine(), old.getFamily(),
        old.getInstruction(), old.getSpelling(), old.getOperandGroups(), old.getResultGroups(),
        old.getTemporaryGroups(), riscv::fragmentMMAAdditionalGroups(capability,
            riscv::fragmentPartCount(mma.getResult().getType()), reuse),
        old.getMask(), old.getTail(), old.getParameters(), old.getLocalBytes()));
  });
  kernel.walk([&](riscv::IMEFragmentPointwiseOp pointwise) {
    const bool reuse = riscv::canReuseFragmentPointwise(pointwise);
    pointwise.setLhsPolicy(reuse ? "reuse" : "copy");
    auto old = pointwise.getLeaf();
    pointwise.setLeafAttr(riscv::LeafAttr::get(kernel.getContext(), old.getEngine(), old.getFamily(),
        old.getInstruction(), old.getSpelling(), old.getOperandGroups(), old.getResultGroups(),
        old.getTemporaryGroups(), reuse ? 0 : pointwise.getResult().getType().getResourceGroups(),
        old.getMask(), old.getTail(), old.getParameters(), old.getLocalBytes()));
  });
}

mlir::Type riscv_internal::withLayout(mlir::Type type,
                                      riscv::LayoutAttr layout) {
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return riscv::ValueType::get(type.getContext(), value.getElementType(),
                                 value.getShape(), value.getAxisIds(), layout);
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return riscv::FragmentType::get(
        type.getContext(), fragment.getFamily(), fragment.getRole(),
        fragment.getPacking(), fragment.getElementType(), fragment.getShape(),
        fragment.getAxisIds(), layout, fragment.getResourceGroups());
  return type;
}

mlir::Value riscv_internal::shapeExtentAt(mlir::Operation *anchor, int64_t id) {
  if (auto local = weft::localShapeExtent<riscv::LocalShapeOp>(anchor, id)) return local;
  auto kernel = anchor->getParentOfType<riscv::KernelOp>();
  if (!kernel || id <= 0 || id > static_cast<int64_t>(kernel.getShapeSymbols().size())) return {};
  auto name = mlir::cast<mlir::StringAttr>(kernel.getShapeSymbols()[id - 1]).getValue();
  mlir::Value value;
  kernel.walk([&](riscv::SymbolOp symbol) { if (!value && symbol.getName() == name) value = symbol.getResult(); });
  return value;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value>> riscv_internal::valueExtentsAt(
    mlir::OpBuilder &builder, mlir::Operation *anchor, riscv::ValueType type) {
  llvm::SmallVector<mlir::Value> result;
  for (int64_t extent : type.getShape().asArrayRef()) {
    mlir::Value value = extent > 0
        ? mlir::Value(mlir::arith::ConstantIndexOp::create(builder, anchor->getLoc(), extent))
        : shapeExtentAt(anchor, -extent);
    if (!value) return mlir::failure();
    result.push_back(value);
  }
  return result;
}

std::optional<int64_t>
riscv_internal::storageFloatCastTemporaryGroups(mlir::Operation *operation) {
  if (!mlir::isa<riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation) ||
      operation->getNumOperands() != 1 || operation->getNumResults() != 1)
    return std::nullopt;
  auto source = mlir::dyn_cast<riscv::ValueType>(operation->getOperand(0).getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
  if (!source && !result) return 0;
  if (!source || !result || source.getShape() != result.getShape() ||
      source.getAxisIds() != result.getAxisIds()) return std::nullopt;
  auto a = source.getLayout(), b = result.getLayout();
  if (a.getCarrier() != b.getCarrier() || a.getLaneFactors() != b.getLaneFactors() ||
      a.getTimeFactors() != b.getTimeFactors() || a.getReplicaFactors() != b.getReplicaFactors() ||
      a.getFragmentFactors() != b.getFragmentFactors() || a.getLocalFactors() != b.getLocalFactors() ||
      a.getReplicaGroup() != b.getReplicaGroup() || a.getValidity() != b.getValidity())
    return std::nullopt;
  for (auto factors : {a.getFragmentFactors(), a.getLocalFactors()})
    if (llvm::any_of(factors.asArrayRef(), [](int64_t factor) { return factor != 1; }))
      return std::nullopt;
  if (a.getCarrier() == "scalar") return 0;
  if (a.getCarrier() != "rvv") return std::nullopt;
  auto wide = source.getElementType().isF32() ? a : b;
  auto narrow = source.getElementType().isF32() ? b : a;
  if ((narrow.getSew() != 8 && narrow.getSew() != 16) ||
      wide.getLmulEighths() != (32 / narrow.getSew()) * narrow.getLmulEighths() ||
      wide.getVl() != narrow.getVl() || wide.getSew() != 32)
    return std::nullopt;
  const int64_t groups = (wide.getLmulEighths() + 7) / 8;
  if (narrow.getSew() == 8) return 3 * groups + 2;
  return source.getElementType().isBF16() ? groups : 3 * groups + 1;
}

bool riscv_internal::isLocalDenseRowExtract(riscv::ExtractOp operation) {
  auto source = operation.getInput().getType();
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (!result || source.getLayout().getCarrier() != "local" ||
      source.getShape().size() != 2 || result.getShape().size() != 2 ||
      source.getAxisIds() != result.getAxisIds() ||
      source.getElementType() != result.getElementType() ||
      operation.getSelectors().size() != 2 || operation.getIndices().size() != 1 ||
      mlir::cast<mlir::StringAttr>(operation.getSelectors()[0]).getValue() != "gather" ||
      mlir::cast<mlir::StringAttr>(operation.getSelectors()[1]).getValue() != "all" ||
      !hasOwnedLocalStorage(operation.getInput()))
    return false;
  auto index = mlir::dyn_cast<riscv::ValueType>(operation.getIndices()[0].getType());
  if (!index || index.getShape().size() != 1 ||
      index.getAxisIds()[0] != result.getAxisIds()[0] ||
      index.getShape()[0] != result.getShape()[0] ||
      !mlir::isa<mlir::IntegerType>(index.getElementType()))
    return false;
  auto a = index.getLayout(), b = result.getLayout();
  if (a.getCarrier() != "scalar" || b.getCarrier() != "rvv" ||
      a.getReplicaGroup() != 1 || b.getReplicaGroup() != 1 ||
      a.getValidity() != "full" || b.getValidity() != "full" ||
      source.getShape()[0] <= 0 || source.getShape()[1] <= 0 || result.getShape()[0] <= 0 ||
      result.getShape()[1] != source.getShape()[1] ||
      a.getTimeFactors()[0] != 1 || a.getLaneFactors()[0] != 1 ||
      a.getReplicaFactors()[0] != result.getShape()[0] ||
      a.getLocalFactors()[0] != 1 || a.getFragmentFactors()[0] != 1 ||
      b.getTimeFactors()[0] != 1 || b.getLaneFactors()[0] != 1 ||
      b.getReplicaFactors()[0] != result.getShape()[0] ||
      b.getReplicaFactors()[1] != 1 || b.getVl() != b.getLaneFactors()[1] ||
      b.getSew() != std::max(8u, logicalBitWidth(result)))
    return false;
  for (auto factors : {b.getLocalFactors(), b.getFragmentFactors()})
    if (llvm::any_of(factors.asArrayRef(), [](int64_t n) { return n != 1; }))
      return false;
  auto columns = staticProduct({b.getTimeFactors()[1], b.getLaneFactors()[1]});
  return columns && *columns == result.getShape()[1];
}

std::optional<int64_t>
riscv_internal::localGatherTemporaryGroups(riscv::ExtractOp operation) {
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (!result || result.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getReplicaGroup() != 1 ||
      operation.getInput().getType().getLayout().getCarrier() != "local")
    return std::nullopt;
  unsigned cursor = 0, vectors = 0;
  int64_t temporaries = 0;
  for (auto [dimension, selector] : llvm::enumerate(operation.getSelectors())) {
    auto kind = mlir::cast<mlir::StringAttr>(selector).getValue();
    if (kind == "all") {
      auto axes = result.getAxisIds().asArrayRef();
      auto found = llvm::find(axes, operation.getInput().getType().getAxisIds()[dimension]);
      if (found == axes.end()) return std::nullopt;
      size_t position = found - axes.begin();
      if (result.getLayout().getLaneFactors()[position] != 1 ||
          result.getLayout().getTimeFactors()[position] != 1)
        return std::nullopt;
      continue;
    }
    if (cursor >= operation.getIndices().size()) return std::nullopt;
    auto index = operation.getIndices()[cursor++];
    if (kind == "index" && index.getType().isIndex()) continue;
    auto type = mlir::dyn_cast<riscv::ValueType>(index.getType());
    if (kind != "gather" || !type) return std::nullopt;
    auto layout = type.getLayout();
    if (layout.getReplicaGroup() != 1 ||
        (layout.getCarrier() != "scalar" && layout.getCarrier() != "rvv"))
      return std::nullopt;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      auto axes = result.getAxisIds().asArrayRef();
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) return std::nullopt;
      size_t target = found - axes.begin();
      if (layout.getTimeFactors()[position] != result.getLayout().getTimeFactors()[target] ||
          layout.getLaneFactors()[position] != result.getLayout().getLaneFactors()[target] ||
          layout.getReplicaFactors()[position] != result.getLayout().getReplicaFactors()[target] ||
          layout.getFragmentFactors()[position] != 1 || layout.getLocalFactors()[position] != 1)
        return std::nullopt;
    }
    if (layout.getCarrier() == "rvv") {
      if (layout.getVl() != result.getLayout().getVl()) return std::nullopt;
      for (auto [position, axis] : llvm::enumerate(result.getAxisIds().asArrayRef()))
        if (result.getLayout().getLaneFactors()[position] > 1 &&
            !llvm::is_contained(type.getAxisIds().asArrayRef(), axis))
          return std::nullopt;
      ++vectors;
      temporaries = (layout.getLmulEighths() + 7) / 8;
    }
  }
  return cursor == operation.getIndices().size() && vectors == 1
      ? std::optional<int64_t>(temporaries) : std::nullopt;
}

std::optional<int64_t>
riscv_internal::booleanComparisonTemporaryGroups(riscv::CompareOp operation) {
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (!result || !result.getElementType().isInteger(1) ||
      result.getLayout().getCarrier() != "rvv" || result.getLayout().getSew() != 8 ||
      (operation.getPredicate() != "eq" && operation.getPredicate() != "ne"))
    return std::nullopt;
  bool vector = false, scalar = false;
  for (auto operand : operation->getOperands()) {
    if (!logicalElement(operand.getType()).isInteger(1)) return std::nullopt;
    auto layout = layoutOf(operand.getType());
    if (!layout || layout.getCarrier() == "scalar") {
      scalar = true;
      continue;
    }
    if (layout.getCarrier() != "rvv" || layout.getSew() != 8 ||
        layout.getVl() != result.getLayout().getVl() ||
        layout.getLmulEighths() != result.getLayout().getLmulEighths())
      return std::nullopt;
    vector = true;
  }
  if (!vector) return std::nullopt;
  return 1 + (scalar ? (result.getLayout().getLmulEighths() + 7) / 8 : 0);
}

std::optional<int64_t> riscv_internal::constantContiguousIndexBase(mlir::Value value, unsigned depth) {
  if (depth > 8) return std::nullopt;
  auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto integer = type ? mlir::dyn_cast<mlir::IntegerType>(type.getElementType()) : mlir::IntegerType();
  if (!integer || !integer.isUnsigned() || integer.getWidth() > 64 ||
      type.getShape().size() != 1 || type.getShape()[0] <= 0)
    return std::nullopt;
  const uint64_t maximum = integer.getWidth() == 64 ? UINT64_MAX
      : (uint64_t{1} << integer.getWidth()) - 1;
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    if (iota.getStart() < 0 || iota.getEnd() <= iota.getStart() ||
        static_cast<uint64_t>(iota.getEnd() - 1) > maximum)
      return std::nullopt;
    return iota.getStart();
  }
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>();
      conversion && conversion.getConversion().getEffect() == "pure" &&
      conversion.getInput().getType().getElementType() == type.getElementType() &&
      conversion.getInput().getType().getShape() == type.getShape() &&
      conversion.getInput().getType().getAxisIds() == type.getAxisIds())
    return constantContiguousIndexBase(conversion.getInput(), depth + 1);
  auto add = value.getDefiningOp<riscv::BinaryOp>();
  if (!add || add.getKind() != "add") return std::nullopt;
  for (unsigned side = 0; side < 2; ++side) {
    auto constant = add->getOperand(side).getDefiningOp<riscv::ConstantOp>();
    auto attr = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()) : mlir::IntegerAttr();
    auto base = constantContiguousIndexBase(add->getOperand(1 - side), depth + 1);
    if (!attr || !base || attr.getValue().getActiveBits() > 63) continue;
    uint64_t offset = attr.getValue().getZExtValue();
    if (static_cast<uint64_t>(type.getShape()[0] - 1) > maximum ||
        offset > INT64_MAX - *base ||
        static_cast<uint64_t>(*base) + offset > maximum - (type.getShape()[0] - 1))
      continue;
    return *base + offset;
  }
  return std::nullopt;
}

bool riscv_internal::isSameWidthIntegerCast(mlir::Operation *operation) {
  if (!mlir::isa<riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation))
    return false;
  auto input = operation->getOperand(0).getType();
  auto output = operation->getResult(0).getType();
  auto source = logicalElement(input), target = logicalElement(output);
  if ((!mlir::isa<mlir::IntegerType>(source) && !source.isIndex()) ||
      (!mlir::isa<mlir::IntegerType>(target) && !target.isIndex()))
    return false;
  if (source.isIndex() || target.isIndex()) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    if (!mlir::isa<riscv::CastOp>(operation) || !kernel ||
        !kernel.getTarget().getMarch().starts_with("rv64") ||
        layoutOf(input) != layoutOf(output))
      return false;
  }
  return logicalBitWidth(source) == logicalBitWidth(target);
}

std::string
riscv_internal::terminalInstruction(mlir::Operation *operation) {
  auto carrierOf = [](mlir::Type type) -> llvm::StringRef {
    if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
      return value.getLayout().getCarrier();
    if (mlir::isa<riscv::FragmentType>(type))
      return "ime";
    return "scalar";
  };
  auto isFloat = [&](mlir::Type type) {
    return mlir::isa<mlir::FloatType>(logicalElement(type));
  };
  auto isUnsigned = [&](mlir::Type type) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(logicalElement(type));
    return integer && integer.isUnsigned();
  };

  if (auto iota = mlir::dyn_cast<riscv::IotaOp>(operation)) {
    auto layout = layoutOf(iota.getResult().getType());
    if (layout && layout.getCarrier() == "rvv")
      return "rvv.iota";
    auto time = layout ? staticProduct(layout.getTimeFactors().asArrayRef())
                       : std::optional<int64_t>();
    auto replicas = layout
                        ? staticProduct(layout.getReplicaFactors().asArrayRef())
                        : std::optional<int64_t>();
    const bool tuple = time && replicas && *time > 0 && *replicas > 0 &&
                       *time <= std::numeric_limits<int64_t>::max() / *replicas &&
                       *time * *replicas > 1;
    return tuple ? "register.iota" : "scalar.iota";
  }
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(operation)) {
    auto element = logicalElement(unary.getResult().getType());
    if (element.isBF16() || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(element))
      return {};
    if (unary.getKind() == "rsqrt")
      return {};
    if (unary.getKind() == "sqrt") {
      auto element = logicalElement(unary.getResult().getType());
      if (!element.isF32() && !element.isF64())
        return {};
      auto carrier = carrierOf(unary.getResult().getType());
      if (carrier == "scalar")
        return "scalar.sqrt";
      return carrier == "rvv" ? "rvv.vfsqrt.v" : std::string();
    }
    if (carrierOf(unary.getResult().getType()) != "rvv" && element.isF32() &&
        (unary.getKind() == "exp" || unary.getKind() == "log"))
      return ("scalar." + unary.getKind() + "-approx-f32").str();
    if (carrierOf(unary.getResult().getType()) != "rvv")
      return ("scalar." + unary.getKind()).str();
    if (unary.getKind() == "exp" &&
        logicalElement(unary.getResult().getType()).isF32())
      return riscv::vectorMathInstruction(unary).str();
    if (unary.getKind() == "log" && element.isF32())
      return riscv::vectorMathInstruction(unary).str();
    if (unary.getKind() == "exp2" &&
        logicalElement(unary.getResult().getType()).isF32()) {
      auto approximate = unary->getAttrOfType<mlir::BoolAttr>("approximate");
      return approximate && approximate.getValue() ? "rvv.exp2-approx-f32" : "";
    }
    if (unary.getKind() == "neg")
      return isFloat(unary.getResult().getType()) ? "rvv.vfneg.v"
                                                  : "rvv.vneg.v";
    if (unary.getKind() == "abs" &&
        isFloat(unary.getResult().getType()))
      return "rvv.vfabs.v";
    return {};
  }
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
    auto element = logicalElement(binary.getResult().getType());
    if (element.isBF16() || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(element))
      return {};
    if (riscv::rvvRowScaleRows(binary))
      return "rvv.mul.row-scale";
    const bool floating = isFloat(binary.getResult().getType());
    const bool unsignedInteger = isUnsigned(binary.getResult().getType());
    std::string stem =
        binary.getKind() == "add" ? (floating ? "vfadd" : "vadd")
      : binary.getKind() == "sub" ? (floating ? "vfsub" : "vsub")
      : binary.getKind() == "mul" ? (floating ? "vfmul" : "vmul")
      : binary.getKind() == "div" ? (floating ? "vfdiv" : unsignedInteger ? "vdivu" : "vdiv")
      : binary.getKind() == "mod" ? (unsignedInteger ? "vremu" : "vrem")
      : binary.getKind() == "and" ? "vand"
      : binary.getKind() == "or"  ? "vor"
      : binary.getKind() == "xor" ? "vxor"
      : binary.getKind() == "shl" ? "vsll"
      : binary.getKind() == "shr" ? (unsignedInteger ? "vsrl" : "vsra")
      : binary.getKind() == "max" ? (floating ? "vfmax" : unsignedInteger ? "vmaxu" : "vmax")
      : binary.getKind() == "min" ? (floating ? "vfmin" : unsignedInteger ? "vminu" : "vmin")
                                     : std::string();
    if (stem.empty())
      return {};
    if (carrierOf(binary.getResult().getType()) != "rvv")
      return ("scalar." + binary.getKind()).str();
    const bool lhsVector = carrierOf(binary.getLhs().getType()) == "rvv";
    const bool rhsVector = carrierOf(binary.getRhs().getType()) == "rvv";
    if (riscv::needsRVVAxisBroadcast(binary.getLhs().getType(), binary.getResult().getType()) ||
        riscv::needsRVVAxisBroadcast(binary.getRhs().getType(), binary.getResult().getType()))
      return riscv::binaryBroadcastTemporaryGroups(binary)
                 ? "rvv." + stem + ".vv.broadcast" : std::string();
    if (lhsVector && rhsVector)
      return "rvv." + stem + ".vv";
    if (lhsVector != rhsVector) {
      const bool scalarOnLeft = !lhsVector;
      const bool commutative =
          binary.getKind() == "add" || binary.getKind() == "mul" ||
          binary.getKind() == "and" || binary.getKind() == "or" ||
          binary.getKind() == "xor" || binary.getKind() == "max" ||
          binary.getKind() == "min";
      if (scalarOnLeft && !commutative)
        return "rvv." + stem + ".vv.splat-lhs";
      return "rvv." + stem + (floating ? ".vf" : ".vx") +
             (scalarOnLeft ? ".swap" : "");
    }
    return {};
  }
  if (auto compare = mlir::dyn_cast<riscv::CompareOp>(operation)) {
    auto element = logicalElement(compare.getLhs().getType());
    if (element.isBF16() || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(element))
      return {};
    if (element.isInteger(1) && carrierOf(compare.getResult().getType()) == "rvv")
      return booleanComparisonTemporaryGroups(compare)
          ? ("rvv.cmp.bool." + compare.getPredicate()).str() : std::string();
    return (carrierOf(compare.getResult().getType()) == "rvv" ? "rvv.cmp."
                                                               : "scalar.cmp.") +
           compare.getPredicate().str();
  }
  if (auto select = mlir::dyn_cast<riscv::SelectOp>(operation)) {
    if (!riscv::selectTemporaryGroups(select)) return {};
    return carrierOf(select.getResult().getType()) == "rvv" ? "rvv.select" : "scalar.select";
  }
  if (auto bitcast = mlir::dyn_cast<riscv::BitcastOp>(operation))
    return carrierOf(bitcast.getResult().getType()) == "rvv" ? "rvv.bitcast" : "scalar.bitcast";
  if (mlir::isa<riscv::CastOp, riscv::NarrowOp,
                riscv::WidenOp>(operation)) {
    mlir::Type source = logicalElement(operation->getOperand(0).getType());
    mlir::Type target = logicalElement(operation->getResult(0).getType());
    if (source != target &&
        (mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(source) ||
         mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(target))) {
      if (!target.isF32() || !storageFloatCastTemporaryGroups(operation)) return {};
      return std::string(carrierOf(operation->getResult(0).getType()) == "rvv" ? "rvv." : "scalar.") +
          (mlir::isa<mlir::Float8E4M3FNType>(source) ? "f8e4m3fn-to-f32" : "f8e5m2-to-f32");
    }
    if (source.isBF16() != target.isBF16()) {
      auto rounding = operation->getAttrOfType<mlir::StringAttr>("rounding");
      auto saturate = operation->getAttrOfType<mlir::BoolAttr>("saturate");
      if ((!source.isF32() && !target.isF32()) ||
          (rounding && rounding.getValue() != "rne") ||
          (saturate && saturate.getValue()) ||
          !storageFloatCastTemporaryGroups(operation))
        return {};
      return std::string(carrierOf(operation->getResult(0).getType()) == "rvv"
                             ? "rvv." : "scalar.") +
             (source.isBF16() ? "bf16-to-f32" : "f32-to-bf16-rne");
    }
    auto rounding = operation->getAttrOfType<mlir::StringAttr>("rounding");
    auto saturate = operation->getAttrOfType<mlir::BoolAttr>("saturate");
    if (carrierOf(operation->getResult(0).getType()) != "rvv") {
      if (rounding && mlir::isa<mlir::FloatType>(source) &&
          mlir::isa<mlir::FloatType>(target) &&
          source.getIntOrFloatBitWidth() > target.getIntOrFloatBitWidth()) {
        if ((saturate && saturate.getValue()) ||
            (rounding.getValue() != "rne" && rounding.getValue() != "rtz" &&
             rounding.getValue() != "rdn" && rounding.getValue() != "rup" &&
             rounding.getValue() != "dynamic")) return {};
        return "scalar.fnarrow.f" + std::to_string(source.getIntOrFloatBitWidth()) +
            "-f" + std::to_string(target.getIntOrFloatBitWidth()) + "." + rounding.getValue().str();
      }
      return mlir::isa<riscv::WidenOp>(operation) ? "scalar.widen"
           : mlir::isa<riscv::NarrowOp>(operation) ? "scalar.narrow"
                                                    : "scalar.cast";
    }
    if (source == target)
      return "rvv.identity";
    auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(source);
    auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
    if (isSameWidthIntegerCast(operation))
      return "rvv.reinterpret";
    if (sourceInteger && targetInteger && (sourceInteger.isUnsigned() || sourceInteger.getWidth() == 1) &&
        sourceInteger.getWidth() < targetInteger.getWidth() &&
        std::max<unsigned>(8, sourceInteger.getWidth()) ==
            std::max<unsigned>(8, targetInteger.getWidth())) {
      if (sourceInteger.getWidth() == 1)
        return targetInteger.isUnsigned() ? "rvv.reinterpret" : "rvv.identity";
      return targetInteger.isSigned() ? "rvv.reinterpret" : "rvv.identity";
    }
    if (source.isF16() && target.isF32())
      return "rvv.fwiden.f16-f32";
    if (source.isF32() && target.isF64())
      return "rvv.fwiden.f32-f64";
    if ((source.isF32() && target.isF16()) || (source.isF64() && target.isF32())) {
      if (saturate && saturate.getValue()) return {};
      std::string instruction = source.isF32() ? "rvv.fnarrow.f32-f16" : "rvv.fnarrow.f64-f32";
      if (rounding) {
        if (rounding.getValue() != "rne" && rounding.getValue() != "rtz" &&
            rounding.getValue() != "rdn" && rounding.getValue() != "rup" &&
            rounding.getValue() != "dynamic") return {};
        instruction += "." + rounding.getValue().str();
      }
      return instruction;
    }
    if (!rounding && targetInteger &&
        ((source.isF16() && targetInteger.getWidth() == 16) ||
         (source.isF32() && targetInteger.getWidth() == 32) ||
         (source.isF64() && targetInteger.getWidth() == 64)))
      return "rvv.convert.f" + std::to_string(source.getIntOrFloatBitWidth()) +
          "-i" + std::to_string(targetInteger.getWidth()) + "-rtz";
    if (!rounding && !(saturate && saturate.getValue()) && sourceInteger &&
        sourceInteger.getWidth() >= 8 &&
        (target.isF16() || target.isF32() || target.isF64())) {
      unsigned sourceBits = sourceInteger.getWidth(), targetBits = target.getIntOrFloatBitWidth();
      if (sourceBits == targetBits || sourceBits * 2 == targetBits || sourceBits == targetBits * 2)
        return "rvv.convert.i" + std::to_string(sourceBits) + "-f" + std::to_string(targetBits);
    }
    if (sourceInteger && targetInteger &&
        targetInteger.getWidth() >
            std::max<unsigned>(8, sourceInteger.getWidth()) &&
        targetInteger.getWidth() %
                std::max<unsigned>(8, sourceInteger.getWidth()) ==
            0) {
      unsigned factor = targetInteger.getWidth() /
                        std::max<unsigned>(8, sourceInteger.getWidth());
      if (factor == 2 || factor == 4 || factor == 8)
        return std::string(sourceInteger.isSigned() ? "rvv.sext.vf"
                                                    : "rvv.zext.vf") +
               std::to_string(factor);
    }
    if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(operation);
        narrow && sourceInteger && targetInteger && !narrow.getSaturate() &&
        sourceInteger.getWidth() == targetInteger.getWidth() * 2 &&
        targetInteger.getWidth() >= 8)
      return "rvv.narrow.int.vf2";
    if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(operation);
        narrow && source.isF32() && targetInteger &&
        targetInteger.getWidth() == 8)
      return ("rvv.f32-i8-" +
              llvm::StringRef(narrow.getSaturate() ? "saturate." : "nonsaturating.") +
              narrow.getRounding())
          .str();
    return {};
  }
  if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
    auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
    if (!input || input.getElementType().isBF16() || reduce.getAxis() < 0 ||
        reduce.getAxis() >= static_cast<int64_t>(input.getAxisIds().size()))
      return {};
    // A fixed register/lane primitive cannot cover an unresolved runtime
    // member domain. Such reductions must first become explicit traversal
    // (for example a stream reduction over its actual memory descriptor).
    if (input.getShape()[reduce.getAxis()] <= 0)
      return {};
    int64_t axis = input.getAxisIds()[reduce.getAxis()];
    bool registerAxis = false;
    for (auto [identity, factor] : llvm::zip(
             input.getAxisIds().asArrayRef(),
             input.getLayout().getReplicaFactors().asArrayRef()))
      registerAxis |= identity == axis && factor > 1;
    llvm::StringRef carrier = input.getLayout().getCarrier();
    if (carrier == "scalar") {
      auto layout = input.getLayout();
      const bool singleton = (reduce.getKind() == "add" ||
          ((reduce.getKind() == "max" || reduce.getKind() == "min") &&
           mlir::isa<mlir::FloatType>(input.getElementType()))) &&
          staticProduct(input.getShape().asArrayRef()) == 1 &&
          llvm::all_of(llvm::ArrayRef<mlir::DenseI64ArrayAttr>{
                        layout.getTimeFactors(), layout.getLaneFactors(),
                        layout.getReplicaFactors(), layout.getFragmentFactors(),
                        layout.getLocalFactors()}, [](mlir::DenseI64ArrayAttr factors) {
            return llvm::all_of(factors.asArrayRef(),
                                [](int64_t factor) { return factor == 1; });
          }) &&
          (mlir::isa<mlir::IntegerType>(input.getElementType()) ||
           input.getElementType().isF16() || input.getElementType().isF32() ||
           input.getElementType().isF64());
      return registerAxis || singleton
                 ? "scalar.register-reduce." + reduce.getKind().str()
                 : std::string();
    }
    if (carrier != "rvv")
      return {};
    if (!registerAxis && llvm::any_of(
            llvm::enumerate(input.getLayout().getLaneFactors().asArrayRef()),
            [&](auto factor) {
              return factor.index() != static_cast<size_t>(reduce.getAxis()) && factor.value() > 1;
            }))
      return {};
    return std::string(registerAxis ? "rvv.register-reduce."
                                    : "rvv.lane-reduce.") +
           reduce.getKind().str();
  }
  if (mlir::isa<riscv::Fold2Op>(operation))
    return "rvv.pair-fold-add";
  return {};
}

std::optional<int64_t>
riscv_internal::staticProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

std::string riscv_internal::printType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream output(result);
  type.print(output);
  return result;
}

riscv::EncodingDeclOp
riscv_internal::findEncoding(mlir::Operation *operation,
                             llvm::StringRef family) {
  riscv::EncodingDeclOp result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](riscv::EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
  return result;
}

llvm::StringRef riscv_internal::baseEncodingFamily(
    mlir::Operation *operation, kernel::EncodingType encoding) {
  if (encoding.getKind() == "base" || encoding.getKind() == "dense")
    return encoding.getFamily();
  llvm::StringRef result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](riscv::DerivedEncodingOp derived) {
      if (result.empty() && derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derived.getSourceFamily();
    });
  return result;
}

int64_t riscv_internal::interleaveRows(mlir::Operation *operation,
                                       kernel::EncodingType encoding) {
  mlir::Value owner;
  if (auto field = mlir::dyn_cast<riscv::FieldOp>(operation))
    owner = field.getOwner();
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (owner) {
    mlir::Operation *definition = owner.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      break;
    if (auto pack = mlir::dyn_cast<riscv::EncodedLocalBindOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto pack = mlir::dyn_cast<riscv::EncodedLocalPackOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto materialize = mlir::dyn_cast<riscv::MaterializeOp>(definition)) {
      if (auto plan = materialize.getLocalPackPlanAttr())
        return plan.getInterleaveRows();
    }
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
      owner = extract.getInput();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition)) {
      owner = materialize.getInput();
      continue;
    }
    if (auto convert = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
      owner = convert.getInput();
      continue;
    }
    break;
  }
  if (encoding.getKind() != "derived_instance")
    return 0;
  int64_t result = 0;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](riscv::DerivedEncodingOp derived) {
      if (derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derived.getInterleaveRows();
    });
  return result;
}

riscv_internal::FieldFacts
riscv_internal::fieldFacts(riscv::FieldOp operation) {
  auto ownerElement = logicalElement(operation.getOwner().getType());
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(ownerElement);
  if (!encoding)
    return {};
  llvm::StringRef base = baseEncodingFamily(operation, encoding);
  riscv::EncodingDeclOp declaration = findEncoding(operation, base);
  if (!declaration)
    return {};
  size_t index = declaration.getFieldNames().size();
  for (auto [position, name] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(name).getValue() == operation.getName()) {
      index = position;
      break;
    }
  if (index == declaration.getFieldNames().size())
    return {};

  FieldFacts result;
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[index]);
  result.scalarPerRecord = fieldShape.empty();
  result.logicalRank = fieldShape.size();
  result.bitOffset = declaration.getFieldBitOffsets()[index];
  result.storageBits = declaration.getFieldStorageBits()[index];
  if (encoding.getKind() == "derived_instance")
    if (auto module = operation->getParentOfType<mlir::ModuleOp>())
      module.walk([&](riscv::DerivedEncodingOp derived) {
        if (derived.getResultFamily() == encoding.getFamily() &&
            derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
            derived.getParameterValues() == encoding.getParameters().asArrayRef())
          result.chunkBytes = derived.getFieldChunks()[index];
      });
  if (result.bitOffset % 8 == 0)
    result.alignment = std::gcd<int64_t>(declaration.getAlignment(),
                                         result.bitOffset / 8);
  auto layouts =
      mlir::cast<mlir::ArrayAttr>(declaration.getFieldLayouts()[index]);
  auto layoutKind = [](mlir::Attribute attribute) -> llvm::StringRef {
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    auto kind = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                           : mlir::StringAttr();
    return kind ? kind.getValue() : llvm::StringRef();
  };
  llvm::StringRef kind =
      layouts.empty() ? llvm::StringRef() : layoutKind(layouts[0]);
  if (kind == "natural") {
    result.mapping = "natural";
  } else if (kind == "joined") {
    result.mapping = "joined";
    auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    result.group = joined.getAs<mlir::IntegerAttr>("size").getInt();
    result.joinFields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
    result.joinLowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
    result.joinRole = joined.getAs<mlir::IntegerAttr>("role").getInt();
    result.order = joined.getAs<mlir::StringAttr>("order").getValue();
  } else if (kind == "grouped" && layouts.size() == 2 &&
             layoutKind(layouts[1]) == "layered") {
    result.mapping = "grouped_layered";
    auto grouped = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    auto layered = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
    result.group = grouped.getAs<mlir::IntegerAttr>("size").getInt();
    result.layer = layered.getAs<mlir::IntegerAttr>("size").getInt();
    result.order = layered.getAs<mlir::StringAttr>("order").getValue();
  }
  return result;
}

std::optional<int64_t> riscv_internal::constantInt(mlir::Value value) {
  struct IntegerConstant { llvm::APInt bits; bool isUnsigned; };
  std::function<std::optional<IntegerConstant>(mlir::Value, unsigned)> evaluate =
      [&](mlir::Value current, unsigned depth) -> std::optional<IntegerConstant> {
    if (depth > 24) return std::nullopt;
    auto element = logicalElement(current.getType());
    auto type = mlir::dyn_cast<mlir::IntegerType>(element);
    if (!type && !element.isIndex()) return std::nullopt;
    const bool isUnsigned = type && type.isUnsigned();
    unsigned width = type ? type.getWidth() : 0;
    if (element.isIndex()) {
      auto scope = current.getParentRegion()->getParentOp();
      auto kernel = mlir::dyn_cast<riscv::KernelOp>(scope);
      if (!kernel) kernel = scope->getParentOfType<riscv::KernelOp>();
      if (!kernel) return std::nullopt;
      auto march = kernel.getTarget().getMarch();
      if (march.starts_with("rv32")) width = 32;
      else if (march.starts_with("rv64")) width = 64;
      else return std::nullopt;
    }
    mlir::IntegerAttr integer;
    if (auto constant = current.getDefiningOp<riscv::ConstantOp>())
      integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    else if (auto constant = current.getDefiningOp<mlir::arith::ConstantOp>())
      integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (integer) return IntegerConstant{isUnsigned ? integer.getValue().zextOrTrunc(width)
                                                 : integer.getValue().sextOrTrunc(width), isUnsigned};
    mlir::Value input;
    std::optional<bool> extension;
    if (auto cast = current.getDefiningOp<riscv::CastOp>()) input = cast.getInput();
    else if (auto cast = current.getDefiningOp<mlir::arith::IndexCastOp>()) {
      input = cast.getIn(); extension = false;
    } else if (auto cast = current.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
      input = cast.getIn(); extension = true;
    }
    if (!input) return std::nullopt;
    auto source = evaluate(input, depth + 1);
    if (!source) return std::nullopt;
    return IntegerConstant{extension.value_or(source->isUnsigned)
        ? source->bits.zextOrTrunc(width) : source->bits.sextOrTrunc(width), isUnsigned};
  };
  auto constant = evaluate(value, 0);
  if (!constant) return std::nullopt;
  if (constant->isUnsigned)
    return constant->bits.isIntN(63) ? std::optional<int64_t>(constant->bits.getZExtValue()) : std::nullopt;
  return constant->bits.isSignedIntN(64) ? std::optional<int64_t>(constant->bits.getSExtValue()) : std::nullopt;
}

mlir::Value riscv_internal::stripRepresentationConversions(
    mlir::Value value,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions) {
  while (true) {
    if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      auto input = mlir::dyn_cast<riscv::ValueType>(conversion.getInput().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      llvm::StringRef effect = conversion.getConversion().getEffect();
      if (!input || !result || input.getShape() != result.getShape() ||
          input.getAxisIds() != result.getAxisIds() ||
          (effect != "pure" && effect != "read"))
        break;
      if (conversions)
        conversions->push_back(conversion);
      value = conversion.getInput();
      continue;
    }
    if (auto materialize =
            value.getDefiningOp<riscv::RegisterMaterializeOp>()) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return value;
}

bool riscv_internal::knownMultipleOf(mlir::Value value, int64_t divisor,
                                     unsigned depth) {
  if (!value || divisor <= 0 || depth > 24)
    return false;
  mlir::Type element = logicalElement(value.getType());
  if (!mlir::isa<mlir::IntegerType, mlir::IndexType>(element))
    return false;
  const bool powerOfTwo = (divisor & (divisor - 1)) == 0;
  const unsigned width = logicalBitWidth(element);
  if (powerOfTwo && width < 63 &&
      static_cast<uint64_t>(divisor) > (uint64_t{1} << width))
    return false;
  if (divisor == 1)
    return true;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    return riscv_internal::knownMultipleOf(conversion.getInput(), divisor,
                                           depth + 1);
  if (auto materialize = value.getDefiningOp<riscv::RegisterMaterializeOp>())
    return knownMultipleOf(materialize.getInput(), divisor, depth + 1);
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    return riscv_internal::knownMultipleOf(cast.getInput(), divisor, depth + 1);
  if (auto widen = value.getDefiningOp<riscv::WidenOp>())
    return riscv_internal::knownMultipleOf(widen.getInput(), divisor,
                                           depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>();
      cast && powerOfTwo)
    return knownMultipleOf(cast.getIn(), divisor, depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>();
      cast && powerOfTwo)
    return knownMultipleOf(cast.getIn(), divisor, depth + 1);
  if (auto constant = riscv_internal::constantInt(value))
    return *constant % divisor == 0;
  if (riscv::knownIndexMultipleOf(value, divisor, depth))
    return true;
  auto binary = value.getDefiningOp<riscv::BinaryOp>();
  if (!binary)
    return false;
  if (binary.getKind() == "shl" && powerOfTwo) {
    auto shift = constantInt(binary.getRhs());
    return shift && *shift >= 0 && *shift < width &&
           (*shift >= 63 || (uint64_t{1} << *shift) >= static_cast<uint64_t>(divisor));
  }
  if (binary.getKind() == "add" || binary.getKind() == "sub")
    return riscv_internal::knownMultipleOf(binary.getLhs(), divisor,
                                           depth + 1) &&
           riscv_internal::knownMultipleOf(binary.getRhs(), divisor,
                                           depth + 1);
  if (binary.getKind() != "mul")
    return false;
  return riscv_internal::knownMultipleOf(binary.getLhs(), divisor, depth + 1) ||
         riscv_internal::knownMultipleOf(binary.getRhs(), divisor, depth + 1);
}

namespace {

mlir::Value stripIndexWidthConversions(mlir::Value value) {
  while (true) {
    if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      value = conversion.getInput();
      continue;
    }
    if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
      value = cast.getInput();
      continue;
    }
    if (auto widen = value.getDefiningOp<riscv::WidenOp>()) {
      value = widen.getInput();
      continue;
    }
    return value;
  }
}

} // namespace

std::optional<riscv_internal::IndexedEntryRelation>
riscv_internal::analyzeIndexedEntryRelation(
    mlir::Value fullIndices, riscv::ValueType result,
    llvm::ArrayRef<int64_t> retainedAxes,
    llvm::ArrayRef<int64_t> retainedShape) {
  auto indices = mlir::dyn_cast<riscv::ValueType>(fullIndices.getType());
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  if (!indices || retainedAxes.size() != retainedShape.size() ||
      resultAxes.size() != retainedAxes.size() + indices.getAxisIds().size() ||
      resultShape.size() != retainedShape.size() + indices.getShape().size() ||
      !llvm::equal(retainedAxes, resultAxes.take_front(retainedAxes.size())) ||
      !llvm::equal(retainedShape, resultShape.take_front(retainedShape.size())) ||
      !llvm::equal(indices.getAxisIds().asArrayRef(),
                   resultAxes.drop_front(retainedAxes.size())) ||
      !llvm::equal(indices.getShape().asArrayRef(),
                   resultShape.drop_front(retainedShape.size())))
    return std::nullopt;

  mlir::Value indexValue = stripRepresentationConversions(fullIndices);
  auto add = indexValue.getDefiningOp<riscv::BinaryOp>();
  if (!add || add.getKind() != "add")
    return std::nullopt;
  auto match = [&](mlir::Value entryTerm,
                   mlir::Value payloadTerm)
      -> std::optional<IndexedEntryRelation> {
    auto stripAxisBroadcast = [](mlir::Value value) {
      while (auto broadcast =
                 value.getDefiningOp<riscv::RVVAxisBroadcastOp>())
        value = broadcast.getInput();
      return value;
    };
    mlir::Value payloadSource =
        stripIndexWidthConversions(stripAxisBroadcast(payloadTerm));
    auto payloadIota = payloadSource.getDefiningOp<riscv::IotaOp>();
    auto payloadType = payloadIota
                           ? mlir::dyn_cast<riscv::ValueType>(
                                 payloadIota.getResult().getType())
                           : riscv::ValueType();
    if (!payloadIota || !payloadType || payloadType.getAxisIds().size() != 1 ||
        payloadIota.getStart() != 0 ||
        payloadIota.getEnd() !=
            static_cast<uint64_t>(payloadType.getShape()[0]) ||
        payloadType.getShape()[0] <= 0)
      return std::nullopt;
    const int64_t payloadAxis = payloadType.getAxisIds()[0];
    const int64_t payloadExtent = payloadType.getShape()[0];
    if (resultAxes.empty() || resultAxes.back() != payloadAxis ||
        resultShape.back() != payloadExtent)
      return std::nullopt;

    mlir::Value scaledEntry =
        stripRepresentationConversions(stripAxisBroadcast(entryTerm));
    auto multiply = scaledEntry.getDefiningOp<riscv::BinaryOp>();
    mlir::Value entryIndices;
    std::optional<int64_t> stride;
    int64_t indexDivisor = 1;
    if (multiply && multiply.getKind() == "mul") {
      if ((stride = constantInt(multiply.getRhs())))
        entryIndices = multiply.getLhs();
      else if ((stride = constantInt(multiply.getLhs())))
        entryIndices = multiply.getRhs();
    }
    if (!entryIndices || !stride || *stride < payloadExtent) {
      if (!riscv_internal::knownMultipleOf(scaledEntry, payloadExtent))
        return std::nullopt;
      entryIndices = scaledEntry;
      stride = payloadExtent;
      indexDivisor = payloadExtent;
    }
    entryIndices = stripRepresentationConversions(entryIndices);
    auto entryType = mlir::dyn_cast<riscv::ValueType>(entryIndices.getType());
    if (entryType) {
      if ((entryType.getLayout().getCarrier() != "unassigned" &&
           entryType.getLayout().getCarrier() != "scalar" &&
           entryType.getLayout().getCarrier() != "rvv") ||
          indices.getAxisIds().size() != entryType.getAxisIds().size() + 1 ||
          !llvm::equal(entryType.getAxisIds().asArrayRef(),
                       indices.getAxisIds().asArrayRef().drop_back()) ||
          !llvm::equal(entryType.getShape().asArrayRef(),
                       indices.getShape().asArrayRef().drop_back()))
        return std::nullopt;
    } else {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(entryIndices.getType());
      if ((!entryIndices.getType().isIndex() &&
           (!integer || integer.isSigned())) ||
          indices.getAxisIds().size() != 1 || indices.getShape().size() != 1)
        return std::nullopt;
    }
    return IndexedEntryRelation{entryIndices, payloadAxis, payloadExtent,
                                *stride, indexDivisor};
  };
  if (auto relation = match(add.getLhs(), add.getRhs()))
    return relation;
  return match(add.getRhs(), add.getLhs());
}

std::optional<riscv_internal::IntegerRange>
riscv_internal::integerRange(mlir::Value value, unsigned depth) {
  if (depth > 16)
    return std::nullopt;
  mlir::Value source = stripRepresentationConversions(value);
  if (source != value)
    return integerRange(source, depth + 1);
  auto type = mlir::dyn_cast<mlir::IntegerType>(logicalElement(value.getType()));
  if (!type || type.getWidth() == 0 || type.getWidth() >= 63)
    return std::nullopt;
  const __int128 typeMinimum =
      type.isSigned() ? -(__int128{1} << (type.getWidth() - 1)) : 0;
  const __int128 typeMaximum =
      type.isSigned() ? (__int128{1} << (type.getWidth() - 1)) - 1
                      : (__int128{1} << type.getWidth()) - 1;
  auto checkedRange = [&](const __int128 minimum,
                          const __int128 maximum)
      -> std::optional<IntegerRange> {
    if (minimum < typeMinimum || maximum > typeMaximum || minimum > maximum)
      return std::nullopt;
    return IntegerRange{static_cast<int64_t>(minimum),
                        static_cast<int64_t>(maximum)};
  };
  auto constantRange = [&](mlir::Attribute attribute)
      -> std::optional<IntegerRange> {
    auto integer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute);
    if (!integer)
      return std::nullopt;
    const int64_t constant = type.isSigned() ? integer.getValue().getSExtValue()
                                             : integer.getValue().getZExtValue();
    return IntegerRange{constant, constant};
  };
  if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
    if (auto range = constantRange(constant.getValue()))
      return range;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto range = constantRange(constant.getValue()))
      return range;
  if (auto widen = value.getDefiningOp<riscv::WidenOp>())
    if (auto range = integerRange(widen.getInput(), depth + 1))
      return range;
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    if (auto range = integerRange(cast.getInput(), depth + 1))
      if (auto checked = checkedRange(range->minimum, range->maximum))
        return checked;
  if (auto narrow = value.getDefiningOp<riscv::NarrowOp>())
    if (!narrow.getSaturate())
      if (auto range = integerRange(narrow.getInput(), depth + 1))
        if (auto checked = checkedRange(range->minimum, range->maximum))
          return checked;
  // A typed logical extract selects elements without changing their numeric
  // interpretation.  Its result range is therefore a subset of the input
  // range, independent of the selected coordinates or physical layout.
  if (auto extract = value.getDefiningOp<riscv::ExtractOp>())
    if (auto range = integerRange(extract.getInput(), depth + 1))
      if (auto checked = checkedRange(range->minimum, range->maximum))
        return checked;
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    auto lhs = integerRange(binary.getLhs(), depth + 1);
    auto rhs = integerRange(binary.getRhs(), depth + 1);
    if (lhs && rhs) {
      if (binary.getKind() == "add")
        if (auto range = checkedRange(
                static_cast<__int128>(lhs->minimum) + rhs->minimum,
                static_cast<__int128>(lhs->maximum) + rhs->maximum))
          return range;
      if (binary.getKind() == "sub")
        if (auto range = checkedRange(
                static_cast<__int128>(lhs->minimum) - rhs->maximum,
                static_cast<__int128>(lhs->maximum) - rhs->minimum))
          return range;
      if (binary.getKind() == "mul") {
        const __int128 candidates[] = {
            static_cast<__int128>(lhs->minimum) * rhs->minimum,
            static_cast<__int128>(lhs->minimum) * rhs->maximum,
            static_cast<__int128>(lhs->maximum) * rhs->minimum,
            static_cast<__int128>(lhs->maximum) * rhs->maximum};
        if (auto range = checkedRange(
                *std::min_element(std::begin(candidates), std::end(candidates)),
                *std::max_element(std::begin(candidates), std::end(candidates))))
          return range;
      }
      if (binary.getKind() == "and") {
        const IntegerRange *mask =
            rhs->minimum == rhs->maximum && rhs->minimum >= 0 ? &*rhs
            : lhs->minimum == lhs->maximum && lhs->minimum >= 0 ? &*lhs
                                                                  : nullptr;
        if (mask)
          if (auto range = checkedRange(0, mask->maximum))
            return range;
      }
      if (binary.getKind() == "or" && lhs->minimum >= 0 &&
          rhs->minimum >= 0) {
        auto coveringMask = [](int64_t maximum) {
          uint64_t value = static_cast<uint64_t>(maximum);
          value |= value >> 1;
          value |= value >> 2;
          value |= value >> 4;
          value |= value >> 8;
          value |= value >> 16;
          value |= value >> 32;
          return value;
        };
        const __int128 maximum =
            static_cast<__int128>(coveringMask(lhs->maximum) |
                                  coveringMask(rhs->maximum));
        if (auto range = checkedRange(0, maximum))
          return range;
      }
      if (binary.getKind() == "shl" && lhs->minimum >= 0 &&
          rhs->minimum == rhs->maximum && rhs->minimum >= 0 &&
          rhs->minimum < static_cast<int64_t>(type.getWidth()))
        if (auto range = checkedRange(
                static_cast<__int128>(lhs->minimum) << rhs->minimum,
                static_cast<__int128>(lhs->maximum) << rhs->minimum))
          return range;
      if (binary.getKind() == "shr" && lhs->minimum >= 0 &&
          rhs->minimum == rhs->maximum && rhs->minimum >= 0 &&
          rhs->minimum < static_cast<int64_t>(type.getWidth()))
        if (auto range = checkedRange(lhs->minimum >> rhs->minimum,
                                      lhs->maximum >> rhs->minimum))
          return range;
    }
  }
  if (type.isSigned()) {
    const int64_t bound = int64_t{1} << (type.getWidth() - 1);
    return IntegerRange{-bound, bound - 1};
  }
  return IntegerRange{0, (int64_t{1} << type.getWidth()) - 1};
}

std::optional<int64_t>
riscv_internal::maximumMagnitude(mlir::Value value) {
  auto range = integerRange(value);
  if (!range)
    return std::nullopt;
  return std::max(range->maximum,
                  range->minimum < 0 ? -range->minimum : range->minimum);
}

riscv::FieldOp riscv_internal::sourceField(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    value = stripRepresentationConversions(value);
    definition = value.getDefiningOp();
    if (auto field = mlir::dyn_cast_or_null<riscv::FieldOp>(definition))
      return field;
    if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    break;
  }
  return {};
}

riscv::LoadOp riscv_internal::sourceLoad(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    value = stripRepresentationConversions(value);
    definition = value.getDefiningOp();
    if (auto load = mlir::dyn_cast_or_null<riscv::LoadOp>(definition))
      return load;
    if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    if (auto pack =
            mlir::dyn_cast_or_null<riscv::EncodedLocalBindOp>(definition)) {
      value = pack.getInput();
      continue;
    }
    if (auto pack =
            mlir::dyn_cast_or_null<riscv::EncodedLocalPackOp>(definition)) {
      value = pack.getInput();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast_or_null<riscv::RegisterMaterializeOp>(definition)) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return {};
}

riscv::AccessAttr riscv_internal::accessOf(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto access = definition->getAttrOfType<riscv::AccessAttr>("access"))
      if (!(access.getForm() == "local" && access.getMapping() == "dense"))
        return access;
    value = stripRepresentationConversions(value);
    definition = value.getDefiningOp();
    if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    break;
  }
  return {};
}

std::optional<riscv::AccessAttr> riscv_internal::contiguousInterleavedReadAccess(
    mlir::Value projection, riscv::ValueType result) {
  llvm::SmallVector<mlir::Value> indices;
  auto field = riscv::fieldReadProjection(projection, &indices);
  if (!field || !result || result.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(logicalElement(field.getOwner().getType()));
  const auto facts = fieldFacts(field);
  if (!encoding || interleaveRows(field, encoding) <= 0 || facts.logicalRank != 1 ||
      (facts.mapping != "natural" && facts.mapping != "grouped_layered"))
    return std::nullopt;
  auto layout = result.getLayout();
  auto axes = result.getAxisIds().asArrayRef();
  if (axes.empty() || layout.getAxisIds() != result.getAxisIds())
    return std::nullopt;
  const int64_t lanes = layout.getLaneFactors().asArrayRef().back();
  const int64_t bits = logicalBitWidth(result.getElementType());
  const int64_t contiguous = facts.mapping == "natural"
                                ? facts.chunkBytes / std::max<int64_t>(1, bits / 8)
                                : std::min(facts.chunkBytes, facts.layer);
  if (lanes <= 1 || lanes > contiguous || contiguous % lanes ||
      (facts.mapping == "natural" && (bits != 8 && bits != 16 && bits != 32)) ||
      (facts.mapping == "grouped_layered" &&
       (bits <= 0 || bits > 8 || facts.group != facts.layer * (8 / bits))))
    return std::nullopt;
  for (size_t dimension = 0; dimension + 1 < axes.size(); ++dimension)
    if (layout.getLaneFactors()[dimension] != 1)
      return std::nullopt;
  for (mlir::Value index : indices) {
    auto point = index.getDefiningOp<riscv::PhysicalPointOp>();
    if (!point)
      return std::nullopt;
    if (point.getResult().getType().getDomain().getAxisId() == axes.back() &&
        !knownMultipleOf(point.getBase(), lanes))
      return std::nullopt;
  }
  auto access = field.getAccess();
  return riscv::AccessAttr::get(result.getContext(), "unit", access.getMapping(),
      access.getAlignment(), 0, 0,
      access.getGroupSize(), access.getLayerSize(), access.getJoinFields(),
      access.getJoinLowBits(), access.getJoinRole(), access.getBitOffset(),
      access.getStorageBits(), access.getOrder(), access.getFieldChunkBytes());
}

riscv::PhysicalPointOp riscv_internal::originPoint(mlir::Value value,
                                                   int64_t axis) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    current = stripRepresentationConversions(current);
    if (auto point = current.getDefiningOp<riscv::PhysicalPointOp>();
        point && point.getResult().getType().getDomain().getAxisId() == axis)
      return point;
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition)
      continue;
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
      for (mlir::Value index : extract.getIndices())
        worklist.push_back(index);
      worklist.push_back(extract.getInput());
      continue;
    }
    if (auto slice = mlir::dyn_cast<riscv::SliceOp>(definition)) {
      for (mlir::Value index : slice.getIndices())
        worklist.push_back(index);
      worklist.push_back(slice.getBase());
      continue;
    }
    if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
      worklist.push_back(field.getOwner());
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition))
      worklist.push_back(materialize.getInput());
  }
  return {};
}

void riscv_internal::copyOrigin(mlir::Operation *source,
                                mlir::Operation *target) {
  if (mlir::Attribute origin = source->getAttr("source_origin")) {
    target->setAttr("source_origin", origin);
    return;
  }
  std::string spelling;
  llvm::raw_string_ostream output(spelling);
  source->getLoc().print(output);
  target->setAttr("source_origin",
                  mlir::StringAttr::get(target->getContext(), output.str()));
}
