#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

bool isBinaryWithConstant(riscv::BinaryOp operation, llvm::StringRef kind,
                          int64_t expected, mlir::Value &other) {
  if (!operation || operation.getKind() != kind)
    return false;
  if (auto value = riscv_internal::constantInt(operation.getRhs());
      value == expected) {
    other = operation.getLhs();
    return true;
  }
  if ((kind == "and" || kind == "or" || kind == "add" || kind == "mul") &&
      riscv_internal::constantInt(operation.getLhs()) == expected) {
    other = operation.getRhs();
    return true;
  }
  return false;
}

struct BitplaneRelation {
  mlir::Value low;
  riscv::FieldOp plane;
  mlir::Value physicalPlane;
  int64_t insertBit = 0;
  llvm::StringRef instruction;
  riscv::PackedPlaneMergePlanAttr packedPlan;
};

struct BitmaskRelation {
  riscv::FieldOp field;
  riscv::PhysicalPointOp origin;
  riscv::PhysicalPointOp point;
};

struct SignedBitmaskReductionRelation {
  mlir::Value data;
  mlir::Value centeredRoot;
  riscv::RVVBitmaskWindowLoadOp mask;
  int64_t reductionAxis = 0;
  int64_t extent = 0;
};

struct MaskedNegateRelation {
  mlir::Value data;
  mlir::Value sign;
  riscv::RVVBitmaskWindowLoadOp mask;
};

mlir::Value stripSingleUsePureCopies(mlir::Value value, mlir::Block *block) {
  for (unsigned depth = 0; depth < 16; ++depth) {
    auto operation = value.getDefiningOp();
    if (!operation || operation->getBlock() != block || !value.hasOneUse())
      return {};
    if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
      if (conversion.getConversion().getEffect() != "pure" ||
          conversion.getInput().getType() != value.getType())
        return {};
      value = conversion.getInput();
    } else if (auto materialize =
                   mlir::dyn_cast<riscv::RegisterMaterializeOp>(operation)) {
      if (materialize.getInput().getType() != value.getType())
        return {};
      value = materialize.getInput();
    } else {
      return value;
    }
  }
  return {};
}

std::optional<MaskedNegateRelation>
matchMaskedNegate(riscv::BinaryOp product, mlir::Value data, mlir::Value sign) {
  auto type = mlir::dyn_cast<riscv::ValueType>(data.getType());
  auto kernel = product->getParentOfType<riscv::KernelOp>();
  if (product.getKind() != "mul" || !type || !kernel ||
      sign.getType() != type || product.getResult().getType() != type)
    return std::nullopt;
  mlir::Block *block = product->getBlock();
  mlir::Value centered = stripSingleUsePureCopies(sign, block);
  auto subtract = centered ? centered.getDefiningOp<riscv::BinaryOp>()
                           : riscv::BinaryOp();
  // This relation is 1 - 2*b, not the reduction relation's 2*b - 1.
  if (!subtract || subtract.getKind() != "sub" ||
      riscv_internal::constantInt(subtract.getLhs()) != 1)
    return std::nullopt;
  mlir::Value doubled = stripSingleUsePureCopies(subtract.getRhs(), block);
  auto multiply = doubled ? doubled.getDefiningOp<riscv::BinaryOp>()
                          : riscv::BinaryOp();
  mlir::Value bit;
  if (!isBinaryWithConstant(multiply, "mul", 2, bit))
    return std::nullopt;
  bit = stripSingleUsePureCopies(bit, block);
  if (!bit || bit.getType() != type)
    return std::nullopt;
  if (auto widen = bit.getDefiningOp<riscv::WidenOp>())
    bit = widen.getInput();
  else if (auto cast = bit.getDefiningOp<riscv::CastOp>())
    bit = cast.getInput();
  else
    return std::nullopt;
  bit = stripSingleUsePureCopies(bit, block);
  auto mask = bit ? bit.getDefiningOp<riscv::RVVBitmaskWindowLoadOp>()
                  : riscv::RVVBitmaskWindowLoadOp();
  if (!mask || mask.getLeaf().getInstruction() != "rvv.bitmask-window-load" ||
      !riscv::supportsRVVMaskedNegate(kernel.getTarget(), type,
                                     mask.getResult().getType()))
    return std::nullopt;
  return MaskedNegateRelation{data, sign, mask};
}

std::optional<BitmaskRelation>
matchBitmaskDecode(riscv::ExtractOp extract, riscv::ValueType result) {
  auto field = extract ? extract.getInput().getDefiningOp<riscv::FieldOp>()
                       : riscv::FieldOp();
  auto element = result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                        : mlir::IntegerType();
  if (!extract || !field || !result || !element || element.isSigned() ||
      element.getWidth() != 1 || result.getLayout().getCarrier() != "rvv" ||
      extract.getIndices().size() != 1)
    return std::nullopt;

  riscv::PhysicalPointOp point;
  size_t cursor = 0;
  for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (cursor >= extract.getIndices().size() || selector != "domain" || point)
      return std::nullopt;
    point = extract.getIndices()[cursor++].getDefiningOp<riscv::PhysicalPointOp>();
  }
  auto origin = point
                    ? point.getParent().getDefiningOp<riscv::PhysicalPointOp>()
                    : riscv::PhysicalPointOp();
  if (!point || !origin || cursor != extract.getIndices().size())
    return std::nullopt;

  const int64_t axis = point.getResult().getType().getDomain().getAxisId();
  bool ownerCoversOriginRecord =
      static_cast<bool>(field.getOwner().getDefiningOp<riscv::LoadOp>());
  if (auto owner = field.getOwner().getDefiningOp<riscv::ExtractOp>()) {
    auto ownerInput = mlir::dyn_cast<riscv::ValueType>(owner.getInput().getType());
    size_t ownerCursor = 0;
    for (auto [position, selectorAttribute] :
         llvm::enumerate(owner.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (ownerCursor >= owner.getIndices().size())
        break;
      mlir::Value index = owner.getIndices()[ownerCursor++];
      if (ownerInput && position < ownerInput.getAxisIds().size() &&
          ownerInput.getAxisIds()[position] == axis && selector == "domain" &&
          index == origin.getResult())
        ownerCoversOriginRecord = true;
    }
  }
  if (!ownerCoversOriginRecord)
    return std::nullopt;

  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (facts.mapping != "grouped_layered" || facts.group <= 0 ||
      facts.layer <= 0 || facts.group != facts.layer * 8 ||
      facts.bitOffset % 8 || facts.order != "lo_first")
    return std::nullopt;
  auto position = llvm::find(result.getAxisIds().asArrayRef(), axis);
  auto partition =
      point.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (position == result.getAxisIds().asArrayRef().end() || !partition ||
      origin.getResult().getType().getDomain().getAxisId() != axis)
    return std::nullopt;
  const size_t axisPosition = static_cast<size_t>(
      position - result.getAxisIds().asArrayRef().begin());
  const int64_t extent = result.getShape()[axisPosition];
  const int64_t lanes = result.getLayout().getLaneFactors()[axisPosition];
  if (extent <= 0 || partition.value() != extent || extent % facts.group ||
      lanes <= 0 || lanes % facts.group || extent % lanes ||
      result.getLayout().getTimeFactors()[axisPosition] != extent / lanes)
    return std::nullopt;
  for (size_t index = 0; index < result.getShape().size(); ++index)
    if (index != axisPosition &&
        (result.getLayout().getTimeFactors()[index] != 1 ||
         result.getLayout().getLaneFactors()[index] != 1))
      return std::nullopt;
  return BitmaskRelation{field, origin, point};
}

std::optional<SignedBitmaskReductionRelation>
matchSignedBitmaskReduction(riscv::RVVWidenDotOp dot, mlir::Value centered,
                            mlir::Value data) {
  if (!dot || dot.getOver().size() != 1)
    return std::nullopt;
  mlir::Value centeredRoot = centered;
  centered = riscv_internal::stripRepresentationConversions(centered);
  auto subtract = centered.getDefiningOp<riscv::BinaryOp>();
  mlir::Value doubled;
  if (!isBinaryWithConstant(subtract, "sub", 1, doubled))
    return std::nullopt;
  doubled = riscv_internal::stripRepresentationConversions(doubled);
  auto multiply = doubled.getDefiningOp<riscv::BinaryOp>();
  mlir::Value bitValue;
  if (!isBinaryWithConstant(multiply, "mul", 2, bitValue))
    return std::nullopt;
  bitValue = riscv_internal::stripRepresentationConversions(bitValue);
  if (auto widen = bitValue.getDefiningOp<riscv::WidenOp>())
    bitValue =
        riscv_internal::stripRepresentationConversions(widen.getInput());
  else if (auto cast = bitValue.getDefiningOp<riscv::CastOp>())
    bitValue = riscv_internal::stripRepresentationConversions(cast.getInput());
  else
    return std::nullopt;
  auto mask = bitValue.getDefiningOp<riscv::RVVBitmaskWindowLoadOp>();
  auto dataType = mlir::dyn_cast<riscv::ValueType>(data.getType());
  auto centeredType =
      mlir::dyn_cast<riscv::ValueType>(centeredRoot.getType());
  auto dataElement = dataType
                         ? mlir::dyn_cast<mlir::IntegerType>(
                               dataType.getElementType())
                         : mlir::IntegerType();
  auto centeredElement =
      centeredType
          ? mlir::dyn_cast<mlir::IntegerType>(centeredType.getElementType())
          : mlir::IntegerType();
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(dot.getResult().getType());
  const int64_t reductionAxis = dot.getOver()[0];
  auto axis = dataType
                  ? llvm::find(dataType.getAxisIds().asArrayRef(), reductionAxis)
                  : llvm::ArrayRef<int64_t>::iterator();
  if (!mask || !dataType || !centeredType || !dataElement ||
      !dataElement.isSigned() || dataElement.getWidth() != 8 ||
      !centeredElement || !centeredElement.isSigned() ||
      centeredElement.getWidth() != 8 || !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      dataType.getShape() != centeredType.getShape() ||
      dataType.getAxisIds() != centeredType.getAxisIds() ||
      dataType.getLayout() != centeredType.getLayout() ||
      axis == dataType.getAxisIds().asArrayRef().end())
    return std::nullopt;
  const size_t position = static_cast<size_t>(
      axis - dataType.getAxisIds().asArrayRef().begin());
  const int64_t extent = dataType.getShape()[position];
  if (extent <= 0 || extent * 128 >= 32768 ||
      dataType.getLayout().getTimeFactors()[position] != 1 ||
      dataType.getLayout().getLaneFactors()[position] != extent)
    return std::nullopt;
  return SignedBitmaskReductionRelation{data, centeredRoot, mask,
                                        reductionAxis, extent};
}

std::optional<BitplaneRelation>
matchLogicalBitplane(riscv::BinaryOp merge, mlir::Value low,
                     riscv::BinaryOp shiftHigh, int64_t insertBit) {
  mlir::Value planeValue =
      riscv_internal::stripRepresentationConversions(shiftHigh.getLhs());
  if (auto widen = planeValue.getDefiningOp<riscv::WidenOp>())
    planeValue = riscv_internal::stripRepresentationConversions(widen.getInput());
  else if (auto cast = planeValue.getDefiningOp<riscv::CastOp>())
    planeValue = riscv_internal::stripRepresentationConversions(cast.getInput());
  else
    return std::nullopt;

  auto window = planeValue.getDefiningOp<riscv::RVVBitmaskWindowLoadOp>();
  auto plane = window
                   ? window.getField().getDefiningOp<riscv::FieldOp>()
                   : riscv_internal::sourceField(planeValue);
  auto lowType = mlir::dyn_cast<riscv::ValueType>(low.getType());
  auto resultType = mlir::dyn_cast<riscv::ValueType>(merge.getResult().getType());
  auto planeType = window
                       ? window.getResult().getType()
                       : (plane ? mlir::dyn_cast<riscv::ValueType>(
                                      plane.getResult().getType())
                                : riscv::ValueType());
  auto lowInteger = lowType ? mlir::dyn_cast<mlir::IntegerType>(
                                  riscv_internal::logicalElement(lowType))
                            : mlir::IntegerType();
  auto planeInteger = planeType ? mlir::dyn_cast<mlir::IntegerType>(
                                      riscv_internal::logicalElement(planeType))
                                : mlir::IntegerType();
  if (!plane || !lowType || !resultType || !planeType || !lowInteger ||
      lowInteger.isSigned() || lowInteger.getWidth() != 8 || !planeInteger ||
      planeInteger.isSigned() || planeInteger.getWidth() != 1 ||
      lowType != resultType || lowType.getLayout().getCarrier() != "rvv" ||
      lowType.getShape() != planeType.getShape() ||
      lowType.getAxisIds() != planeType.getAxisIds())
    return std::nullopt;

  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(plane);
  if (facts.mapping != "grouped_layered" || facts.group != 8 ||
      facts.layer != 1 || facts.logicalRank <= 0 ||
      facts.logicalRank > static_cast<int64_t>(planeType.getAxisIds().size()))
    return std::nullopt;

  int64_t laneAxis = 0;
  int64_t laneAxes = 0;
  for (auto [axis, lane] :
       llvm::zip(lowType.getLayout().getAxisIds().asArrayRef(),
                 lowType.getLayout().getLaneFactors().asArrayRef()))
    if (lane > 1) {
      laneAxis = axis;
      ++laneAxes;
    }
  auto logicalAxes = planeType.getAxisIds().asArrayRef().take_back(
      static_cast<size_t>(facts.logicalRank));
  if (laneAxes != 1 || logicalAxes.size() != 1 ||
      laneAxis != logicalAxes.front() || lowType.getLayout().getVl() <= 0 ||
      lowType.getLayout().getVl() % 8)
    return std::nullopt;
  auto owner = mlir::dyn_cast<riscv::ValueType>(plane.getOwner().getType());
  auto encoding = owner ? mlir::dyn_cast<kernel::EncodingType>(
                              riscv_internal::logicalElement(owner))
                        : kernel::EncodingType();
  if (!encoding || riscv_internal::interleaveRows(plane, encoding) != 0)
    return std::nullopt;
  return BitplaneRelation{low, plane, planeValue, insertBit,
                          "rvv.bitplane-merge.mask", {}};
}

riscv::FieldOp sourceSubByteField(mlir::Value value) {
  value = riscv_internal::stripRepresentationConversions(value);
  if (auto widen = value.getDefiningOp<riscv::WidenOp>())
    value = riscv_internal::stripRepresentationConversions(widen.getInput());
  else if (auto cast = value.getDefiningOp<riscv::CastOp>())
    value = riscv_internal::stripRepresentationConversions(cast.getInput());
  return riscv_internal::sourceField(value);
}

std::optional<BitplaneRelation>
matchPackedPlaneMerge(riscv::BinaryOp merge, mlir::Value low,
                      riscv::BinaryOp shiftHigh, int64_t insertBit) {
  auto lowField = sourceSubByteField(low);
  auto highField = sourceSubByteField(shiftHigh.getLhs());
  auto lowType = lowField
                     ? mlir::dyn_cast<riscv::ValueType>(lowField.getResult().getType())
                     : riscv::ValueType();
  auto highType = highField
                      ? mlir::dyn_cast<riscv::ValueType>(highField.getResult().getType())
                      : riscv::ValueType();
  auto resultType = mlir::dyn_cast<riscv::ValueType>(merge.getResult().getType());
  auto lowInteger = lowType
                        ? mlir::dyn_cast<mlir::IntegerType>(lowType.getElementType())
                        : mlir::IntegerType();
  auto highInteger = highType
                         ? mlir::dyn_cast<mlir::IntegerType>(highType.getElementType())
                         : mlir::IntegerType();
  auto resultInteger =
      resultType
          ? mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType())
          : mlir::IntegerType();
  if (!lowField || !highField || lowField.getOwner() != highField.getOwner() ||
      !lowType || !highType || !resultType || !lowInteger ||
      lowInteger.isSigned() || !highInteger || highInteger.isSigned() ||
      !resultInteger || resultInteger.isSigned() || resultInteger.getWidth() != 8 ||
      lowType.getShape() != resultType.getShape() ||
      highType.getShape() != resultType.getShape() ||
      lowType.getAxisIds() != resultType.getAxisIds() ||
      highType.getAxisIds() != resultType.getAxisIds() ||
      resultType.getLayout().getCarrier() != "scalar")
    return std::nullopt;

  riscv_internal::FieldFacts lowFacts = riscv_internal::fieldFacts(lowField);
  riscv_internal::FieldFacts highFacts = riscv_internal::fieldFacts(highField);
  if (lowFacts.mapping != "grouped_layered" ||
      highFacts.mapping != "grouped_layered" || lowFacts.logicalRank != 1 ||
      highFacts.logicalRank != 1 || lowFacts.group != highFacts.group ||
      lowFacts.group <= 0 || lowFacts.layer <= 0 || highFacts.layer <= 0 ||
      lowFacts.group * lowInteger.getWidth() != lowFacts.layer * 8 ||
      highFacts.group * highInteger.getWidth() != highFacts.layer * 8 ||
      lowFacts.layer % 4 || highFacts.layer % 4 ||
      lowFacts.bitOffset % 8 || highFacts.bitOffset % 8 ||
      lowFacts.order != "lo_first" || highFacts.order != "lo_first" ||
      insertBit < lowInteger.getWidth() ||
      insertBit + highInteger.getWidth() > 8)
    return std::nullopt;
  auto logicalAxes = lowType.getAxisIds().asArrayRef().take_back(1);
  if (logicalAxes.empty())
    return std::nullopt;
  const int64_t logicalAxis = logicalAxes.front();
  auto position = llvm::find(resultType.getAxisIds().asArrayRef(), logicalAxis);
  if (position == resultType.getAxisIds().asArrayRef().end())
    return std::nullopt;
  const size_t ordinal = static_cast<size_t>(
      position - resultType.getAxisIds().asArrayRef().begin());
  if (resultType.getShape()[ordinal] != lowFacts.group ||
      resultType.getLayout().getTimeFactors()[ordinal] != 1 ||
      resultType.getLayout().getLaneFactors()[ordinal] != 1 ||
      resultType.getLayout().getReplicaFactors()[ordinal] != lowFacts.group)
    return std::nullopt;
  auto owner = mlir::dyn_cast<riscv::ValueType>(lowField.getOwner().getType());
  auto encoding = owner ? mlir::dyn_cast<kernel::EncodingType>(
                              riscv_internal::logicalElement(owner))
                        : kernel::EncodingType();
  if (!encoding || riscv_internal::interleaveRows(lowField, encoding) != 0 ||
      riscv_internal::interleaveRows(highField, encoding) != 0)
    return std::nullopt;

  auto plan = riscv::PackedPlaneMergePlanAttr::get(
      merge.getContext(), logicalAxis, lowFacts.group, lowInteger.getWidth(),
      highInteger.getWidth(), lowFacts.group, lowFacts.layer, highFacts.layer,
      lowFacts.bitOffset / 8, highFacts.bitOffset / 8, insertBit);
  return BitplaneRelation{lowField.getResult(), highField,
                          highField.getResult(), insertBit,
                          "scalar.packed-plane-merge.words",
                          plan};
}

std::optional<BitplaneRelation> matchBitplane(riscv::BinaryOp merge,
                                              mlir::Value low,
                                              mlir::Value high) {
  if (!merge || merge.getKind() != "or")
    return std::nullopt;
  auto shiftHigh = riscv_internal::stripRepresentationConversions(high)
                       .getDefiningOp<riscv::BinaryOp>();
  if (!shiftHigh || shiftHigh.getKind() != "shl")
    return std::nullopt;
  auto insertBit = riscv_internal::constantInt(shiftHigh.getRhs());
  if (!insertBit || *insertBit <= 0 || *insertBit >= 8)
    return std::nullopt;

  if (auto packed =
          matchPackedPlaneMerge(merge, low, shiftHigh, *insertBit))
    return packed;

  if (auto logical = matchLogicalBitplane(merge, low, shiftHigh, *insertBit))
    return logical;

  mlir::Value shifted;
  if (!isBinaryWithConstant(
          riscv_internal::stripRepresentationConversions(shiftHigh.getLhs())
              .getDefiningOp<riscv::BinaryOp>(),
                            "and", 1, shifted))
    return std::nullopt;
  auto variableShift = riscv_internal::stripRepresentationConversions(shifted)
                           .getDefiningOp<riscv::BinaryOp>();
  if (!variableShift || variableShift.getKind() != "shr")
    return std::nullopt;
  auto extract =
      riscv_internal::stripRepresentationConversions(variableShift.getLhs())
          .getDefiningOp<riscv::ExtractOp>();
  auto position =
      riscv_internal::stripRepresentationConversions(variableShift.getRhs())
          .getDefiningOp<riscv::BinaryOp>();
  size_t gatherSelectors = 0;
  if (extract)
    for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "gather")
        ++gatherSelectors;
      else if (selector != "all")
        return std::nullopt;
    }
  if (!extract || !position || position.getKind() != "mod" ||
      riscv_internal::constantInt(position.getRhs()) != 8 ||
      gatherSelectors != 1 || extract.getIndices().size() != 1)
    return std::nullopt;

  auto byteIndex =
      riscv_internal::stripRepresentationConversions(extract.getIndices().front())
          .getDefiningOp<riscv::BinaryOp>();
  if (!byteIndex || byteIndex.getKind() != "div" ||
      riscv_internal::constantInt(byteIndex.getRhs()) != 8 ||
      riscv_internal::stripRepresentationConversions(byteIndex.getLhs()) !=
          riscv_internal::stripRepresentationConversions(position.getLhs()))
    return std::nullopt;
  auto coordinates =
      riscv_internal::stripRepresentationConversions(byteIndex.getLhs())
          .getDefiningOp<riscv::IotaOp>();
  if (!coordinates || coordinates.getStart() != 0)
    return std::nullopt;

  auto plane = riscv_internal::sourceField(extract.getInput());
  auto lowType = mlir::dyn_cast<riscv::ValueType>(low.getType());
  auto resultType = mlir::dyn_cast<riscv::ValueType>(merge.getResult().getType());
  auto planeType = plane
                       ? mlir::dyn_cast<riscv::ValueType>(plane.getResult().getType())
                       : riscv::ValueType();
  auto coordinateType =
      mlir::dyn_cast<riscv::ValueType>(coordinates.getResult().getType());
  if (!plane || !lowType || !resultType || !planeType ||
      !coordinateType || coordinateType.getAxisIds().size() != 1 ||
      lowType != resultType || lowType.getLayout().getCarrier() != "rvv" ||
      planeType.getLayout().getCarrier() != "local" ||
      coordinates.getEnd() - coordinates.getStart() !=
          riscv_internal::staticProduct(coordinateType.getShape()).value_or(-1))
    return std::nullopt;
  auto coordinateAxis = llvm::find(resultType.getAxisIds().asArrayRef(),
                                   coordinateType.getAxisIds()[0]);
  if (coordinateAxis == resultType.getAxisIds().asArrayRef().end() ||
      resultType.getShape()[static_cast<size_t>(
          coordinateAxis - resultType.getAxisIds().asArrayRef().begin())] !=
          coordinateType.getShape()[0])
    return std::nullopt;

  auto lowInteger =
      mlir::dyn_cast<mlir::IntegerType>(riscv_internal::logicalElement(lowType));
  auto planeInteger = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(planeType));
  if (!lowInteger || lowInteger.isSigned() || lowInteger.getWidth() != 8 ||
      !planeInteger || planeInteger.isSigned() || planeInteger.getWidth() != 8 ||
      plane.getAccess().getMapping() != "natural" ||
      plane.getAccess().getBitOffset() % 8)
    return std::nullopt;

  if (resultType.getAxisIds() != planeType.getAxisIds() ||
      resultType.getShape().size() != planeType.getShape().size())
    return std::nullopt;
  const size_t packedPosition = static_cast<size_t>(
      coordinateAxis - resultType.getAxisIds().asArrayRef().begin());
  for (size_t index = 0; index < resultType.getShape().size(); ++index) {
    const int64_t lowExtent = resultType.getShape()[index];
    const int64_t planeExtent = planeType.getShape()[index];
    if (index == packedPosition) {
      if (lowExtent <= 0 || planeExtent <= 0 ||
          planeExtent > std::numeric_limits<int64_t>::max() / 8 ||
          planeExtent * 8 != lowExtent)
        return std::nullopt;
    } else if (lowExtent != planeExtent) {
      return std::nullopt;
    }
  }
  auto owner = mlir::dyn_cast<riscv::ValueType>(plane.getOwner().getType());
  auto encoding = owner ? mlir::dyn_cast<kernel::EncodingType>(
                              riscv_internal::logicalElement(owner))
                        : kernel::EncodingType();
  if (!encoding || riscv_internal::interleaveRows(plane, encoding) != 0)
    return std::nullopt;

  int64_t laneAxis = 0;
  int64_t laneAxes = 0;
  for (auto [axis, lane] :
       llvm::zip(lowType.getLayout().getAxisIds().asArrayRef(),
                 lowType.getLayout().getLaneFactors().asArrayRef()))
    if (lane > 1) {
      laneAxis = axis;
      ++laneAxes;
    }
  if (laneAxes != 1)
    return std::nullopt;
  const int64_t packedAxis = coordinateType.getAxisIds()[0];
  llvm::StringRef instruction;
  if (laneAxis == packedAxis && lowType.getLayout().getVl() > 0 &&
      lowType.getLayout().getVl() % 8 == 0) {
    instruction = "rvv.bitplane-merge.mask";
  } else if (laneAxis != packedAxis) {
    auto packed = llvm::find(lowType.getAxisIds().asArrayRef(), packedAxis);
    const size_t position = static_cast<size_t>(
        packed - lowType.getAxisIds().asArrayRef().begin());
    bool otherTime = false;
    for (auto [axis, time] :
         llvm::zip(lowType.getAxisIds().asArrayRef(),
                   lowType.getLayout().getTimeFactors().asArrayRef()))
      otherTime |= axis != packedAxis && time != 1;
    if (packed == lowType.getAxisIds().asArrayRef().end() || otherTime ||
        lowType.getLayout().getLaneFactors()[position] != 1 ||
        lowType.getLayout().getReplicaFactors()[position] != 1 ||
        lowType.getLayout().getFragmentFactors()[position] != 1 ||
        lowType.getLayout().getLocalFactors()[position] != 1 ||
        lowType.getLayout().getTimeFactors()[position] !=
            lowType.getShape()[position])
      return std::nullopt;
    instruction = "rvv.bitplane-merge.strided";
  } else {
    return std::nullopt;
  }

  return BitplaneRelation{low, plane, plane.getResult(), *insertBit,
                          instruction, {}};
}

void eraseDeadTree(mlir::Value value, mlir::IRRewriter &rewriter) {
  mlir::Operation *operation = value.getDefiningOp();
  const bool pureLayoutConversion =
      operation && mlir::isa<riscv::ConvertLayoutOp>(operation) &&
      mlir::cast<riscv::ConvertLayoutOp>(operation)
              .getConversion()
              .getEffect() == "pure";
  if (!operation ||
      (!mlir::isMemoryEffectFree(operation) && !pureLayoutConversion) ||
      llvm::any_of(operation->getResults(),
                   [](mlir::Value result) { return !result.use_empty(); }))
    return;
  llvm::SmallVector<mlir::Value> operands(operation->getOperands());
  rewriter.eraseOp(operation);
  for (mlir::Value operand : operands)
    eraseDeadTree(operand, rewriter);
}

class FuseRISCVBitplanesPass
    : public mlir::PassWrapper<FuseRISCVBitplanesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-fuse-bitplanes";
  }
  llvm::StringRef getDescription() const override {
    return "Fuse typed packed-bit extraction with its RVV value consumer";
  }

  void runOnOperation() override {
    llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
    getOperation().walk([&](riscv::ConvertLayoutOp operation) {
      conversions.push_back(operation);
    });
    mlir::IRRewriter rewriter(&getContext());
    auto materializeDecode = [&](mlir::Operation *root, riscv::ExtractOp extract,
                                 riscv::ValueType result) {
      auto relation = matchBitmaskDecode(extract, result);
      if (!relation)
        return;
      const int64_t resultGroups = result.getLayout().getRegisterGroups();
      const int64_t temporaryGroups = std::max<int64_t>(
          1, (result.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = result.getLayout().getValidity() == "tail";
      rewriter.setInsertionPoint(root);
      auto decoded = rewriter.create<riscv::RVVBitmaskDecodeOp>(
          root->getLoc(), result, relation->field.getResult(),
          relation->origin.getResult(), relation->point.getResult(),
          relation->field.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "bitmask-decode",
                               "rvv.bitmask-decode", "rvv.bitmask-decode", 0,
                               resultGroups, temporaryGroups, 0, "none",
                               tail ? "agnostic" : "exact"));
      riscv_internal::copyOrigin(root, decoded);
      decoded->setAttr("canonical_op",
                       rewriter.getStringAttr("weft_kernel.extract"));
      mlir::Value oldInput = extract.getResult();
      const bool replacesExtract = root == extract.getOperation();
      rewriter.replaceOp(root, decoded.getResult());
      if (!replacesExtract)
        eraseDeadTree(oldInput, rewriter);
    };
    for (riscv::ConvertLayoutOp conversion : conversions) {
      if (conversion.getConversion().getEffect() == "pure" &&
          conversion.getConversion().getKind() == "time_to_lane")
        materializeDecode(
            conversion, conversion.getInput().getDefiningOp<riscv::ExtractOp>(),
            conversion.getResult().getType());
    }
    llvm::SmallVector<riscv::ExtractOp> directBitmasks;
    getOperation().walk([&](riscv::ExtractOp extract) {
      if (extract.getAccess().getLayerSize() == 1)
        directBitmasks.push_back(extract);
    });
    for (riscv::ExtractOp extract : directBitmasks) {
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (result)
        materializeDecode(extract, extract, result);
    }

    llvm::SmallVector<riscv::RVVWidenDotOp> signedMaskDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      signedMaskDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : signedMaskDots) {
      auto relation =
          matchSignedBitmaskReduction(dot, dot.getRhs(), dot.getLhs());
      if (!relation)
        relation =
            matchSignedBitmaskReduction(dot, dot.getLhs(), dot.getRhs());
      if (!relation)
        continue;
      auto dataType =
          mlir::cast<riscv::ValueType>(relation->data.getType());
      const int64_t dataGroups = dataType.getLayout().getRegisterGroups();
      const int64_t temporaryGroups = std::max<int64_t>(1, dataGroups + 1);
      const bool tail = dataType.getLayout().getValidity() == "tail";
      auto maskLeaf = relation->mask.getLeaf();
      relation->mask->setAttr(
          "leaf", riscv_internal::leaf(
                      rewriter, maskLeaf.getEngine(), maskLeaf.getFamily(),
                      "rvv.bitmask-window-mask", "rvv.bitmask-window-mask",
                      maskLeaf.getOperandGroups(), maskLeaf.getResultGroups(),
                      maskLeaf.getTemporaryGroups(),
                      maskLeaf.getFragmentGroups(), maskLeaf.getMask(),
                      maskLeaf.getTail(), maskLeaf.getParameters(),
                      maskLeaf.getLocalBytes()));
      rewriter.setInsertionPoint(dot);
      auto reduction = rewriter.create<riscv::RVVSignedBitmaskReduceOp>(
          dot.getLoc(), dot.getResult().getType(), relation->data,
          relation->mask.getResult(), relation->reductionAxis,
          riscv_internal::leaf(
              rewriter, "rvv", "signed-bitmask-reduce",
              "rvv.signed-bitmask-reduce.i8-i16",
              "rvv.signed-bitmask-reduce.i8-i16", dataGroups, 0,
              temporaryGroups, 0, "none", tail ? "agnostic" : "exact",
              {relation->reductionAxis, relation->extent}));
      riscv_internal::copyOrigin(dot, reduction);
      reduction->setAttr("canonical_op",
                         rewriter.getStringAttr("weft_kernel.contract"));
      mlir::Value centeredRoot = relation->centeredRoot;
      rewriter.replaceOp(dot, reduction.getResult());
      eraseDeadTree(centeredRoot, rewriter);
    }

    llvm::SmallVector<riscv::BinaryOp> signProducts;
    getOperation().walk([&](riscv::BinaryOp product) {
      if (product.getKind() == "mul")
        signProducts.push_back(product);
    });
    llvm::SmallVector<mlir::Value> deadSigns;
    for (riscv::BinaryOp product : signProducts) {
      auto relation = matchMaskedNegate(product, product.getLhs(), product.getRhs());
      if (!relation)
        relation = matchMaskedNegate(product, product.getRhs(), product.getLhs());
      if (!relation)
        continue;
      auto data = mlir::cast<riscv::ValueType>(relation->data.getType());
      auto maskLeaf = relation->mask.getLeaf();
      relation->mask->setAttr(
          "leaf", riscv_internal::leaf(
                      rewriter, "rvv", "bitmask-window-load",
                      "rvv.bitmask-window-mask", "rvv.bitmask-window-mask",
                      maskLeaf.getOperandGroups(), maskLeaf.getResultGroups(),
                      0, 0, "none", maskLeaf.getTail(),
                      maskLeaf.getParameters(), 0));
      rewriter.setInsertionPoint(product);
      auto selected = rewriter.create<riscv::RVVMaskedNegateOp>(
          product.getLoc(), data, relation->data, relation->mask.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "masked-negate", "rvv.masked-negate",
              "rvv.masked-negate", 0, 0,
              std::max<int64_t>(1, (data.getLayout().getLmulEighths() + 7) / 8),
              0, "none", maskLeaf.getTail()));
      riscv_internal::copyOrigin(product, selected);
      selected->setAttr("canonical_op",
                        rewriter.getStringAttr("weft_kernel.binary"));
      rewriter.replaceOp(product, selected.getResult());
      // Keep the data producer (including its widening) and the original mask
      // read in place. Only the now-dead single-use integer sign chain is gone.
      deadSigns.push_back(relation->sign);
    }
    // A sign chain contains multiplication ops from signProducts itself.
    // Defer its deletion until every candidate handle has been inspected.
    for (mlir::Value sign : deadSigns)
      eraseDeadTree(sign, rewriter);

    llvm::SmallVector<riscv::BinaryOp> signValues;
    getOperation().walk([&](riscv::BinaryOp operation) {
      if (operation.getKind() == "sub")
        signValues.push_back(operation);
    });
    for (auto subtract : signValues) {
      auto type = mlir::dyn_cast<riscv::ValueType>(subtract.getResult().getType());
      auto kernel = subtract->getParentOfType<riscv::KernelOp>();
      mlir::Value doubled;
      int64_t zeroBitValue = 0;
      if (riscv_internal::constantInt(subtract.getRhs()) == 1) {
        doubled = subtract.getLhs();
        zeroBitValue = -1;
      } else if (riscv_internal::constantInt(subtract.getLhs()) == 1) {
        doubled = subtract.getRhs();
        zeroBitValue = 1;
      } else {
        continue;
      }
      if (!type || !kernel || doubled.getType() != type)
        continue;
      doubled = stripSingleUsePureCopies(doubled, subtract->getBlock());
      mlir::Value bit;
      if (!doubled ||
          !isBinaryWithConstant(doubled.getDefiningOp<riscv::BinaryOp>(), "mul", 2, bit))
        continue;
      bit = stripSingleUsePureCopies(bit, subtract->getBlock());
      if (!bit || bit.getType() != type)
        continue;
      if (auto widen = bit.getDefiningOp<riscv::WidenOp>())
        bit = widen.getInput();
      else if (auto cast = bit.getDefiningOp<riscv::CastOp>())
        bit = cast.getInput();
      else
        continue;
      bit = stripSingleUsePureCopies(bit, subtract->getBlock());
      auto mask = bit ? bit.getDefiningOp<riscv::RVVBitmaskWindowLoadOp>()
                      : riscv::RVVBitmaskWindowLoadOp();
      auto decode = bit ? bit.getDefiningOp<riscv::RVVBitmaskDecodeOp>()
                        : riscv::RVVBitmaskDecodeOp();
      auto maskLeaf = mask ? mask.getLeaf() : decode ? decode.getLeaf() : riscv::LeafAttr();
      if (!maskLeaf ||
          (mask && maskLeaf.getInstruction() != "rvv.bitmask-window-load") ||
          (decode && maskLeaf.getInstruction() != "rvv.bitmask-decode") ||
          !riscv::supportsRVVMaskedNegate(kernel.getTarget(), type,
                                         mlir::cast<riscv::ValueType>(bit.getType())))
        continue;
      const llvm::StringRef maskInstruction =
          mask ? "rvv.bitmask-window-mask" : "rvv.bitmask-decode-mask";
      bit.getDefiningOp()->setAttr("leaf", riscv_internal::leaf(
          rewriter, "rvv", maskLeaf.getFamily(), maskInstruction,
          maskInstruction, maskLeaf.getOperandGroups(),
          maskLeaf.getResultGroups(), 0, 0, "none", maskLeaf.getTail(),
          maskLeaf.getParameters(), 0));
      rewriter.setInsertionPoint(subtract);
      auto seed = rewriter.create<riscv::ConstantOp>(
          subtract.getLoc(), type.getElementType(),
          rewriter.getIntegerAttr(type.getElementType(), zeroBitValue));
      auto splat = rewriter.create<riscv::RVVSplatOp>(
          subtract.getLoc(), type, seed.getResult(),
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat", "rvv.splat",
                               0, type.getLayout().getRegisterGroups(), 0, 0,
                               "none", "exact"));
      auto selected = rewriter.create<riscv::RVVMaskedNegateOp>(
          subtract.getLoc(), type, splat.getResult(), bit,
          riscv_internal::leaf(rewriter, "rvv", "masked-negate", "rvv.masked-negate",
                               "rvv.masked-negate", 0, 0,
                               std::max<int64_t>(1, (type.getLayout().getLmulEighths() + 7) / 8),
                               0, "none", maskLeaf.getTail()));
      riscv_internal::copyOrigin(subtract, selected);
      if (type.getLayout().getRegisterGroups() > 1)
        if (auto loop = selected->getParentOfType<mlir::scf::ForOp>()) {
          // One typed mask selection expands across a register cohort. Keep
          // the physical unroll binding; downstream scalar-loop heuristics
          // cannot infer that cohort's code size from one intrinsic call.
          loop->setAttr("weft.riscv.system_unroll",
                        rewriter.getStringAttr("disable"));
        }
      llvm::SmallVector<mlir::Value> oldOperands(subtract->getOperands());
      rewriter.replaceOp(subtract, selected.getResult());
      for (mlir::Value operand : oldOperands)
        eraseDeadTree(operand, rewriter);
    }

    llvm::SmallVector<riscv::BinaryOp> merges;
    getOperation().walk([&](riscv::BinaryOp operation) {
      if (operation.getKind() == "or")
        merges.push_back(operation);
    });
    for (riscv::BinaryOp merge : merges) {
      auto relation = matchBitplane(merge, merge.getLhs(), merge.getRhs());
      if (!relation)
        relation = matchBitplane(merge, merge.getRhs(), merge.getLhs());
      if (!relation)
        continue;
      llvm::SmallVector<mlir::Value> oldOperands(merge->getOperands());
      rewriter.setInsertionPoint(merge);
      if (relation->packedPlan) {
        auto packed = rewriter.create<riscv::PackedPlaneMergeOp>(
            merge.getLoc(), merge.getResult().getType(), relation->low,
            relation->plane.getResult(), relation->packedPlan,
            riscv_internal::leaf(
                rewriter, "scalar", "packed-plane-merge",
                relation->instruction, relation->instruction, 0, 0, 0, 0,
                "none", "exact",
                {relation->packedPlan.getLogicalAxis(),
                 relation->packedPlan.getLogicalElements(),
                 relation->packedPlan.getLowBits(),
                 relation->packedPlan.getHighBits(),
                 relation->packedPlan.getInsertBit()}));
        riscv_internal::copyOrigin(merge, packed);
        packed->setAttr("canonical_op",
                        rewriter.getStringAttr("weft_kernel.binary"));
        rewriter.replaceOp(merge, packed.getResult());
      } else {
        if (auto window = relation->physicalPlane
                              .getDefiningOp<riscv::RVVBitmaskWindowLoadOp>()) {
          auto selected = window.getLeaf();
          window->setAttr(
              "leaf",
              riscv_internal::leaf(
                  rewriter, selected.getEngine(), selected.getFamily(),
                  "rvv.bitmask-window-mask", "rvv.bitmask-window-mask",
                  selected.getOperandGroups(), selected.getResultGroups(),
                  selected.getTemporaryGroups(), selected.getFragmentGroups(),
                  selected.getMask(), selected.getTail(),
                  selected.getParameters(), selected.getLocalBytes()));
        }
        auto rvv = rewriter.create<riscv::RVVBitplaneMergeOp>(
            merge.getLoc(), merge.getResult().getType(), relation->low,
            relation->physicalPlane, relation->insertBit,
            riscv_internal::leaf(rewriter, "rvv", "bitplane-merge",
                                 relation->instruction, relation->instruction,
                                 0, 0, 1, 0, "none", "agnostic",
                                 {relation->insertBit}));
        riscv_internal::copyOrigin(merge, rvv);
        rvv->setAttr("canonical_op",
                     rewriter.getStringAttr("weft_kernel.binary"));
        rewriter.replaceOp(merge, rvv.getResult());
      }
      for (mlir::Value operand : oldOperands)
        eraseDeadTree(operand, rewriter);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createFuseRISCVBitplanesPass() {
  return std::make_unique<FuseRISCVBitplanesPass>();
}
