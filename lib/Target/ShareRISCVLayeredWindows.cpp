#include "Weft/Target/RISCVPasses.h"
#include "IME/FragmentSupply.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>

using namespace weft;

namespace {

struct LinearIndex {
  llvm::DenseMap<mlir::Value, int64_t> terms;
  int64_t constant = 0;
  bool valid = true;
};

std::optional<int64_t> constantIndex(mlir::Value value) {
  return riscv_internal::constantInt(value);
}

void decomposeIndex(mlir::Value value, int64_t coefficient,
                    LinearIndex &result) {
  if (!result.valid)
    return;
  if (auto constant = constantIndex(value)) {
    result.constant += coefficient * *constant;
    return;
  }
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    decomposeIndex(add.getLhs(), coefficient, result);
    decomposeIndex(add.getRhs(), coefficient, result);
    return;
  }
  if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
    decomposeIndex(sub.getLhs(), coefficient, result);
    decomposeIndex(sub.getRhs(), -coefficient, result);
    return;
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto lhs = constantIndex(multiply.getLhs())) {
      decomposeIndex(multiply.getRhs(), coefficient * *lhs, result);
      return;
    }
    if (auto rhs = constantIndex(multiply.getRhs())) {
      decomposeIndex(multiply.getLhs(), coefficient * *rhs, result);
      return;
    }
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    decomposeIndex(cast.getIn(), coefficient, result);
    return;
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    decomposeIndex(cast.getIn(), coefficient, result);
    return;
  }
  result.terms[value] += coefficient;
  if (result.terms[value] == 0)
    result.terms.erase(value);
}

bool sameTerms(const LinearIndex &lhs, const LinearIndex &rhs) {
  if (!lhs.valid || !rhs.valid || lhs.terms.size() != rhs.terms.size())
    return false;
  for (const auto &[value, coefficient] : lhs.terms) {
    auto found = rhs.terms.find(value);
    if (found == rhs.terms.end() || found->second != coefficient)
      return false;
  }
  return true;
}

std::optional<int64_t> timeCoordinate(riscv::ValueType type, int64_t stream,
                                      int64_t axis) {
  if (stream < 0)
    return std::nullopt;
  int64_t coordinateForAxis = 0;
  bool found = false;
  for (int64_t position = static_cast<int64_t>(type.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const int64_t factor = type.getLayout().getTimeFactors()[position];
    if (factor <= 0)
      return std::nullopt;
    const int64_t coordinate = stream % factor;
    stream /= factor;
    if (type.getAxisIds()[position] == axis) {
      coordinateForAxis = coordinate;
      found = true;
    }
  }
  if (stream != 0 || !found)
    return std::nullopt;
  return coordinateForAxis;
}

struct LayerSelection {
  int64_t physicalLayer = 0;
  int64_t mode = 0;
  int64_t shift = 0;
  int64_t mask = 0;
};

std::optional<LayerSelection>
selectLayer(riscv::AccessAttr access, int64_t logicalLayer,
            int64_t layerCount, unsigned elementBits,
            bool dynamicLogicalBase = false) {
  if (!access || logicalLayer < 0 || logicalLayer >= layerCount ||
      layerCount <= 1 || elementBits == 0 ||
      layerCount * static_cast<int64_t>(elementBits) > 8 ||
      (access.getOrder() != "lo_first" && access.getOrder() != "hi_first"))
    return std::nullopt;
  LayerSelection selection;
  if (dynamicLogicalBase) {
    selection.mode = access.getOrder() == "lo_first" ? 1 : 2;
    selection.shift = 1;
    selection.mask = 1;
    return selection;
  }
  selection.physicalLayer = access.getOrder() == "lo_first"
                                ? logicalLayer
                                : layerCount - 1 - logicalLayer;
  selection.shift = selection.physicalLayer != 0;
  selection.mask =
      (selection.physicalLayer + 1) * static_cast<int64_t>(elementBits) != 8;
  return selection;
}

std::optional<riscv::LayeredStreamGeometryAttr> buildLayeredStreamGeometry(
    mlir::Builder &builder, riscv::ValueType type, riscv::AccessAttr access,
    int64_t axis, int64_t projectionBase) {
  auto axisIt = llvm::find(type.getAxisIds().asArrayRef(), axis);
  auto streams =
      riscv_internal::staticProduct(type.getLayout().getTimeFactors().asArrayRef());
  auto replicas = riscv_internal::staticProduct(
      type.getLayout().getReplicaFactors().asArrayRef());
  const int64_t group = access.getGroupSize();
  const int64_t layer = access.getLayerSize();
  if (axisIt == type.getAxisIds().asArrayRef().end() || !streams || !replicas ||
      *streams <= 0 || *replicas <= 0 || group <= 0 || layer <= 0 ||
      group % layer)
    return std::nullopt;
  const size_t axisPosition =
      static_cast<size_t>(axisIt - type.getAxisIds().asArrayRef().begin());
  const int64_t lanes = type.getLayout().getLaneFactors()[axisPosition];
  const int64_t layers = group / layer;
  const unsigned elementBits =
      riscv_internal::logicalBitWidth(type.getElementType());
  if (lanes <= 0 || layers <= 1 || elementBits == 0 ||
      layers * static_cast<int64_t>(elementBits) > 8)
    return std::nullopt;

  std::map<std::pair<int64_t, int64_t>, int64_t> windows;
  llvm::SmallVector<int64_t> windowForStream;
  llvm::SmallVector<int64_t> physicalLayerForStream;
  llvm::SmallVector<int64_t> physicalLayerForLogicalLayer(layers, -1);
  llvm::SmallVector<int64_t> shiftAmountForLogicalLayer(layers, -1);
  llvm::SmallVector<int64_t> maskValueForLogicalLayer(layers, -1);
  llvm::SmallVector<int64_t> shiftAmountForStream;
  llvm::SmallVector<int64_t> maskValueForStream;
  llvm::SmallVector<int64_t> groupForWindow;
  llvm::SmallVector<int64_t> withinForWindow;
  for (int64_t stream = 0; stream < *streams; ++stream) {
    auto coordinate = timeCoordinate(type, stream, axis);
    if (!coordinate)
      return std::nullopt;
    const int64_t logicalOffset = projectionBase + *coordinate * lanes;
    const int64_t groupIndex = logicalOffset / group;
    const int64_t withinGroup = logicalOffset % group;
    const int64_t logicalLayer = withinGroup / layer;
    const int64_t withinLayer = withinGroup % layer;
    if (groupIndex < 0 || logicalLayer < 0 || logicalLayer >= layers ||
        withinLayer < 0 || withinLayer + lanes > layer)
      return std::nullopt;
    auto key = std::make_pair(groupIndex, withinLayer);
    auto [found, inserted] =
        windows.emplace(key, static_cast<int64_t>(windows.size()));
    if (inserted) {
      groupForWindow.push_back(groupIndex);
      withinForWindow.push_back(withinLayer);
    }
    windowForStream.push_back(found->second);
    auto selection = selectLayer(access, logicalLayer, layers, elementBits);
    if (!selection)
      return std::nullopt;
    physicalLayerForStream.push_back(selection->physicalLayer);
    shiftAmountForStream.push_back(
        selection->physicalLayer * static_cast<int64_t>(elementBits));
    maskValueForStream.push_back(selection->mask
                                     ? (int64_t(1) << elementBits) - 1
                                     : 0);
    int64_t &logicalMapping =
        physicalLayerForLogicalLayer[static_cast<size_t>(logicalLayer)];
    if (logicalMapping >= 0 && logicalMapping != selection->physicalLayer)
      return std::nullopt;
    logicalMapping = selection->physicalLayer;
    shiftAmountForLogicalLayer[static_cast<size_t>(logicalLayer)] =
        selection->physicalLayer * static_cast<int64_t>(elementBits);
    maskValueForLogicalLayer[static_cast<size_t>(logicalLayer)] =
        selection->mask ? (int64_t(1) << elementBits) - 1 : 0;
  }
  if (llvm::any_of(physicalLayerForLogicalLayer,
                   [](int64_t layer) { return layer < 0; }))
    return std::nullopt;
  llvm::SmallVector<int64_t> representativeStreamForWindow(windows.size(), -1);
  for (auto [stream, window] : llvm::enumerate(windowForStream)) {
    int64_t &representative =
        representativeStreamForWindow[static_cast<size_t>(window)];
    if (representative < 0)
      representative = static_cast<int64_t>(stream);
  }
  if (llvm::any_of(representativeStreamForWindow,
                   [](int64_t stream) { return stream < 0; }))
    return std::nullopt;
  return riscv::LayeredStreamGeometryAttr::get(
      builder.getContext(), axis, group, layer, lanes, *streams, *replicas,
      builder.getDenseI64ArrayAttr(windowForStream),
      builder.getDenseI64ArrayAttr(representativeStreamForWindow),
      builder.getDenseI64ArrayAttr(physicalLayerForStream),
      builder.getDenseI64ArrayAttr(physicalLayerForLogicalLayer),
      builder.getDenseI64ArrayAttr(shiftAmountForLogicalLayer),
      builder.getDenseI64ArrayAttr(maskValueForLogicalLayer),
      builder.getDenseI64ArrayAttr(shiftAmountForStream),
      builder.getDenseI64ArrayAttr(maskValueForStream),
      builder.getDenseI64ArrayAttr(groupForWindow),
      builder.getDenseI64ArrayAttr(withinForWindow));
}

struct ShapedAffineIndex {
  llvm::DenseMap<int64_t, int64_t> axisCoefficients;
  mlir::Value scalarBase;
  int64_t scalarBaseCoefficient = 0;
  int64_t constant = 0;
  bool valid = true;
};

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs == 0 || rhs == 0) {
    result = 0;
    return true;
  }
  if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
      (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()))
    return false;
  if (lhs > 0) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
        (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs))
      return false;
  } else {
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs))
      return false;
  }
  result = lhs * rhs;
  return true;
}

void addConstant(ShapedAffineIndex &result, int64_t value) {
  int64_t next = 0;
  if (!checkedAdd(result.constant, value, next)) {
    result.valid = false;
    return;
  }
  result.constant = next;
}

void addAxisCoefficient(ShapedAffineIndex &result, int64_t axis,
                        int64_t coefficient) {
  int64_t next = 0;
  if (!checkedAdd(result.axisCoefficients.lookup(axis), coefficient, next)) {
    result.valid = false;
    return;
  }
  if (next == 0)
    result.axisCoefficients.erase(axis);
  else
    result.axisCoefficients[axis] = next;
}

void decomposeShapedIndex(mlir::Value value, int64_t coefficient,
                          ShapedAffineIndex &result, unsigned depth = 0) {
  if (!result.valid || depth > 24)
    return result.valid = false, void();
  if (auto constant = constantIndex(value)) {
    int64_t scaled = 0;
    if (!checkedMultiply(coefficient, *constant, scaled))
      return result.valid = false, void();
    addConstant(result, scaled);
    return;
  }

  auto shaped = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!shaped) {
    if (!result.scalarBase) {
      result.scalarBase = value;
      result.scalarBaseCoefficient = coefficient;
      return;
    }
    if (result.scalarBase != value)
      return result.valid = false, void();
    int64_t combined = 0;
    if (!checkedAdd(result.scalarBaseCoefficient, coefficient, combined))
      return result.valid = false, void();
    result.scalarBaseCoefficient = combined;
    return;
  }
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return result.valid = false, void();
    decomposeShapedIndex(conversion.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
    decomposeShapedIndex(cast.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    auto type = iota.getResult().getType();
    if (type.getShape().size() != 1 || type.getAxisIds().size() != 1 ||
        iota.getEnd() - iota.getStart() != type.getShape()[0])
      return result.valid = false, void();
    int64_t start = 0;
    if (!checkedMultiply(coefficient, iota.getStart(), start))
      return result.valid = false, void();
    addConstant(result, start);
    addAxisCoefficient(result, type.getAxisIds()[0], coefficient);
    return;
  }
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    llvm::StringRef kind = binary.getKind();
    if (kind == "add" || kind == "sub") {
      decomposeShapedIndex(binary.getLhs(), coefficient, result, depth + 1);
      int64_t rhsCoefficient = coefficient;
      if (kind == "sub" &&
          !checkedMultiply(coefficient, -1, rhsCoefficient))
        return result.valid = false, void();
      decomposeShapedIndex(binary.getRhs(), rhsCoefficient, result, depth + 1);
      return;
    }
    if (kind == "mul") {
      if (auto lhs = constantIndex(binary.getLhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *lhs, scaled))
          return result.valid = false, void();
        decomposeShapedIndex(binary.getRhs(), scaled, result, depth + 1);
        return;
      }
      if (auto rhs = constantIndex(binary.getRhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *rhs, scaled))
          return result.valid = false, void();
        decomposeShapedIndex(binary.getLhs(), scaled, result, depth + 1);
        return;
      }
    }
  }
  result.valid = false;
}

int64_t knownMultiple(mlir::Value value, unsigned depth = 0) {
  if (depth > 24)
    return 1;
  if (auto constant = constantIndex(value))
    return *constant == std::numeric_limits<int64_t>::min()
               ? 1
               : std::abs(*constant);
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    return knownMultiple(cast.getInput(), depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return knownMultiple(cast.getIn(), depth + 1);
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return knownMultiple(cast.getIn(), depth + 1);
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    if (binary.getKind() == "mul") {
      if (auto lhs = constantIndex(binary.getLhs())) {
        int64_t factor = *lhs == std::numeric_limits<int64_t>::min()
                             ? 1
                             : std::abs(*lhs);
        int64_t result = 0;
        return checkedMultiply(factor, knownMultiple(binary.getRhs(), depth + 1),
                               result)
                   ? result
                   : 1;
      }
      if (auto rhs = constantIndex(binary.getRhs())) {
        int64_t factor = *rhs == std::numeric_limits<int64_t>::min()
                             ? 1
                             : std::abs(*rhs);
        int64_t result = 0;
        return checkedMultiply(factor, knownMultiple(binary.getLhs(), depth + 1),
                               result)
                   ? result
                   : 1;
      }
    }
    if (binary.getKind() == "add" || binary.getKind() == "sub")
      return std::gcd(knownMultiple(binary.getLhs(), depth + 1),
                      knownMultiple(binary.getRhs(), depth + 1));
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto lhs = constantIndex(multiply.getLhs())) {
      int64_t result = 0;
      return checkedMultiply(std::abs(*lhs),
                             knownMultiple(multiply.getRhs(), depth + 1), result)
                 ? result
                 : 1;
    }
    if (auto rhs = constantIndex(multiply.getRhs())) {
      int64_t result = 0;
      return checkedMultiply(std::abs(*rhs),
                             knownMultiple(multiply.getLhs(), depth + 1), result)
                 ? result
                 : 1;
    }
  }
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
    return std::gcd(knownMultiple(add.getLhs(), depth + 1),
                    knownMultiple(add.getRhs(), depth + 1));
  if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>())
    return std::gcd(knownMultiple(sub.getLhs(), depth + 1),
                    knownMultiple(sub.getRhs(), depth + 1));
  return 1;
}

bool isReadOnlyOrPure(mlir::Operation *operation) {
  return riscv::hasOnlyReadEffects(operation);
}

bool canFoldReadConversions(
    riscv::ExtractOp first, riscv::ExtractOp second,
    llvm::ArrayRef<riscv::ConvertLayoutOp> firstConversions,
  llvm::ArrayRef<riscv::ConvertLayoutOp> secondConversions) {
  llvm::SmallVector<mlir::Operation *> reads;
  auto collectReads = [&](llvm::ArrayRef<riscv::ConvertLayoutOp> conversions) {
    for (riscv::ConvertLayoutOp conversion : conversions) {
      if (conversion.getConversion().getEffect() != "read")
        continue;
      if (conversion.getConversion().getKind() != "local_load" ||
          conversion->getBlock() != first->getBlock())
        return false;
      reads.push_back(conversion);
    }
    return true;
  };
  if (!collectReads(firstConversions) || !collectReads(secondConversions))
    return false;
  if (reads.empty())
    return true;

  mlir::Operation *begin = reads.front();
  for (mlir::Operation *read : reads)
    if (read->isBeforeInBlock(begin))
      begin = read;
  if (second->isBeforeInBlock(begin))
    return false;
  for (mlir::Operation *operation = begin; operation;
       operation = operation->getNextNode()) {
    if (!isReadOnlyOrPure(operation))
      return false;
    if (operation == second.getOperation())
      return true;
  }
  return false;
}

bool isLayerExtract(
    riscv::ExtractOp extract, riscv::FieldOp &field,
    riscv::PhysicalPointOp &point,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr) {
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), conversions);
  field = riscv_internal::sourceField(source);
  point = extract.getIndices().size() == 1
              ? riscv_internal::stripRepresentationConversions(
                    extract.getIndices().front())
                    .getDefiningOp<riscv::PhysicalPointOp>()
              : riscv::PhysicalPointOp();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto input = field
                   ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                   : riscv::ValueType();
  int64_t domainSelectors = 0;
  for (mlir::Attribute selector : extract.getSelectors())
    domainSelectors +=
        mlir::cast<mlir::StringAttr>(selector).getValue() == "domain";
  if (!field || !point || !result || !input)
    return false;
  auto axis = llvm::find(result.getAxisIds().asArrayRef(),
                         point.getResult().getType().getDomain().getAxisId());
  const bool hasLayerAxis =
      result && axis != result.getAxisIds().asArrayRef().end() &&
      result.getShape()[static_cast<size_t>(
          axis - result.getAxisIds().asArrayRef().begin())] ==
          extract.getAccess().getLayerSize();
  llvm::StringRef validity = result.getLayout().getValidity();
  return domainSelectors == 1 &&
         extract.getAccess().getForm() == "indexed" &&
         extract.getAccess().getMapping() == "grouped_layered" &&
         extract.getAccess().getGroupSize() ==
             extract.getAccess().getLayerSize() * 2 &&
         result.getShape().size() == input.getShape().size() && hasLayerAxis &&
         result.getLayout().getCarrier() == "rvv" &&
         (validity == "full" || validity == "tail");
}

bool sameFieldEdge(riscv::FieldOp lhs, riscv::FieldOp rhs) {
  if (!lhs || !rhs || lhs.getOwner() != rhs.getOwner() ||
      lhs.getName() != rhs.getName() ||
      lhs.getResult().getType() != rhs.getResult().getType() ||
      lhs.getAccess() != rhs.getAccess())
    return false;
  riscv_internal::FieldFacts left = riscv_internal::fieldFacts(lhs);
  riscv_internal::FieldFacts right = riscv_internal::fieldFacts(rhs);
  return left.mapping == right.mapping &&
         left.scalarPerRecord == right.scalarPerRecord &&
         left.logicalRank == right.logicalRank &&
         left.group == right.group && left.layer == right.layer &&
         left.joinFields == right.joinFields &&
         left.joinLowBits == right.joinLowBits &&
         left.joinRole == right.joinRole && left.bitOffset == right.bitOffset &&
         left.storageBits == right.storageBits &&
         left.alignment == right.alignment && left.order == right.order;
}

bool isLayeredStreamField(riscv::FieldOp field) {
  auto value = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
  auto element = value
                     ? mlir::dyn_cast<mlir::IntegerType>(value.getElementType())
                     : mlir::IntegerType();
  riscv::AccessAttr access = field.getAccess();
  const int64_t group = access ? access.getGroupSize() : 0;
  const int64_t layer = access ? access.getLayerSize() : 0;
  const int64_t layers = layer > 0 ? group / layer : 0;
  if (!value || !element || element.isSigned() || !access ||
      value.getLayout().getCarrier() != "rvv" ||
      access.getForm() != "indexed" ||
      access.getMapping() != "grouped_layered" ||
      group <= 0 || layer <= 0 || group % layer || layers <= 1 ||
      element.getWidth() * layers > 8 || access.getBitOffset() % 8)
    return false;
  bool hasStreamAxis = false;
  for (size_t index = 0; index < value.getShape().size(); ++index) {
    const int64_t extent = value.getShape()[index];
    const int64_t time = value.getLayout().getTimeFactors()[index];
    const int64_t lanes = value.getLayout().getLaneFactors()[index];
    const bool streamAxis = extent > 0 && extent % group == 0 && lanes > 1 &&
                            layer % lanes == 0 && time > 1 &&
                            time * lanes == extent;
    if (streamAxis) {
      if (hasStreamAxis)
        return false;
      hasStreamAxis = true;
    } else if (time != 1 || lanes != 1) {
      return false;
    }
  }
  return hasStreamAxis;
}

bool isProjectedLayeredExtract(
    riscv::ExtractOp extract, riscv::FieldOp &field,
    riscv::PhysicalPointOp &origin, int64_t &axis, int64_t &base,
    int64_t &stride, int64_t &repeat, int64_t &extent,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions = nullptr) {
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), conversions);
  field = source.getDefiningOp<riscv::FieldOp>();
  auto input = field
                   ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                   : riscv::ValueType();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  if (!field || !input || !result ||
      input.getShape().size() != result.getShape().size() ||
      extract.getAccess().getForm() != "indexed" ||
      extract.getAccess().getMapping() != "grouped_layered")
    return false;

  size_t projection = extract.getSelectors().size();
  bool domainProjection = false;
  int64_t domainSelectors = 0;
  for (auto [position, selector] : llvm::enumerate(extract.getSelectors())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
    if (name == "regular" || name == "domain") {
      if (projection != extract.getSelectors().size())
        return false;
      projection = position;
      domainProjection = name == "domain";
      domainSelectors += domainProjection;
    } else if (name != "all") {
      return false;
    }
  }
  if (projection >= result.getShape().size() ||
      projection >= result.getAxisIds().size())
    return false;

  axis = result.getAxisIds()[projection];
  extent = result.getShape()[projection];
  if (domainProjection) {
    origin = extract.getIndices().size() == 1
                 ? riscv_internal::stripRepresentationConversions(
                       extract.getIndices().front())
                       .getDefiningOp<riscv::PhysicalPointOp>()
                 : riscv::PhysicalPointOp();
    if (domainSelectors != 1 || !origin ||
        origin.getResult().getType().getDomain().getAxisId() != axis)
      return false;
    base = 0;
    stride = 1;
    repeat = 1;
  } else {
    auto pattern =
        extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern");
    if (!pattern || pattern.size() != 3 || !extract.getIndices().empty())
      return false;
    base = pattern[0];
    stride = pattern[1];
    repeat = pattern[2];
    origin = riscv_internal::originPoint(field.getOwner(), axis);
  }
  const int64_t sourceExtent = input.getShape()[projection];
  const int64_t group = extract.getAccess().getGroupSize();
  const int64_t layer = extract.getAccess().getLayerSize();
  const int64_t lanes = result.getLayout().getLaneFactors()[projection];
  const int64_t time = result.getLayout().getTimeFactors()[projection];
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto element = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  if (!origin || base < 0 || stride != 1 || repeat != 1 || extent <= 0 ||
      sourceExtent <= 0 || group <= 0 || layer <= 0 || group % layer ||
      layers <= 1 || base % group || extent % group || lanes <= 1 ||
      layer % lanes || time <= 1 || time > std::numeric_limits<int64_t>::max() / lanes ||
      time * lanes != extent || !element || element.isSigned() ||
      element.getWidth() * layers > 8 ||
      base > sourceExtent - extent ||
      origin.getResult().getType().getDomain().getTail() != "exact" ||
      result.getLayout().getCarrier() != "rvv")
    return false;
  if (conversions &&
      !canFoldReadConversions(
          extract, extract, *conversions,
          llvm::ArrayRef<riscv::ConvertLayoutOp>{}))
    return false;
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    if (position == projection)
      continue;
    if (input.getShape()[position] != result.getShape()[position] ||
        result.getLayout().getTimeFactors()[position] != 1 ||
        result.getLayout().getLaneFactors()[position] != 1)
      return false;
  }
  return true;
}

struct LayeredRecordExtract {
  riscv::ExtractOp extract;
  riscv::ExtractOp subview;
  riscv::FieldOp field;
  riscv::PhysicalPointOp origin;
  riscv::ValueType fieldType;
  riscv::ValueType subviewType;
  riscv::ValueType resultType;
  int64_t reductionAxis = 0;
  int64_t laneAxis = 0;
  int64_t projectionExtent = 0;
  llvm::SmallVector<riscv::ConvertLayoutOp> extractConversions;
  llvm::SmallVector<riscv::ConvertLayoutOp> subviewConversions;
};

std::optional<LayeredRecordExtract>
layeredRecordExtract(riscv::ExtractOp extract) {
  LayeredRecordExtract result;
  result.extract = extract;
  mlir::Value subviewValue = riscv_internal::stripRepresentationConversions(
      extract.getInput(), &result.extractConversions);
  result.subview = subviewValue.getDefiningOp<riscv::ExtractOp>();
  if (!result.subview || extract.getIndices().size() != 1 ||
      !extract.getIndices().front().getType().isIndex())
    return std::nullopt;

  mlir::Value fieldValue = riscv_internal::stripRepresentationConversions(
      result.subview.getInput(), &result.subviewConversions);
  result.field = fieldValue.getDefiningOp<riscv::FieldOp>();
  result.fieldType = result.field
                         ? mlir::dyn_cast<riscv::ValueType>(
                               result.field.getResult().getType())
                         : riscv::ValueType();
  result.subviewType =
      mlir::dyn_cast<riscv::ValueType>(result.subview.getResult().getType());
  result.resultType =
      mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  if (!result.field || !result.fieldType || !result.subviewType ||
      !result.resultType ||
      result.fieldType.getShape().size() != result.subviewType.getShape().size() ||
      result.subviewType.getShape().size() !=
          result.subview.getSelectors().size() ||
      result.subviewType.getShape().size() != extract.getSelectors().size() ||
      result.resultType.getShape().size() + 1 !=
          result.subviewType.getShape().size() ||
      extract.getAccess().getMapping() != "grouped_layered" ||
      result.subview.getAccess().getMapping() != "grouped_layered" ||
      (extract.getAccess().getForm() != "strided" &&
       extract.getAccess().getForm() != "indexed") ||
      extract.getAccess().getGroupSize() !=
          result.subview.getAccess().getGroupSize() ||
      extract.getAccess().getLayerSize() !=
          result.subview.getAccess().getLayerSize() ||
      extract.getAccess().getBitOffset() !=
          result.subview.getAccess().getBitOffset() ||
      extract.getAccess().getStorageBits() !=
          result.subview.getAccess().getStorageBits() ||
      extract.getAccess().getOrder() != result.subview.getAccess().getOrder())
    return std::nullopt;

  size_t projection = result.subviewType.getShape().size();
  for (auto [position, subviewSelector, extractSelector] :
       llvm::zip(llvm::seq<size_t>(0, result.subviewType.getShape().size()),
                 result.subview.getSelectors(), extract.getSelectors())) {
    llvm::StringRef subviewName =
        mlir::cast<mlir::StringAttr>(subviewSelector).getValue();
    llvm::StringRef extractName =
        mlir::cast<mlir::StringAttr>(extractSelector).getValue();
    if (subviewName == "domain" && extractName == "index") {
      if (projection != result.subviewType.getShape().size())
        return std::nullopt;
      projection = position;
      continue;
    }
    if (subviewName != "all" || extractName != "all")
      return std::nullopt;
  }
  if (projection == result.subviewType.getShape().size() ||
      result.subview.getIndices().size() != 1)
    return std::nullopt;
  result.origin = riscv_internal::stripRepresentationConversions(
                      result.subview.getIndices().front())
                      .getDefiningOp<riscv::PhysicalPointOp>();
  result.reductionAxis = result.subviewType.getAxisIds()[projection];
  result.projectionExtent = result.subviewType.getShape()[projection];
  if (!result.origin || result.reductionAxis <= 0 ||
      result.projectionExtent <= 0 ||
      result.origin.getResult().getType().getDomain().getAxisId() !=
          result.reductionAxis ||
      result.origin.getResult().getType().getDomain().getTail() != "exact" ||
      result.fieldType.getElementType() != result.resultType.getElementType() ||
      result.resultType.getLayout().getCarrier() != "rvv")
    return std::nullopt;

  size_t resultPosition = 0;
  for (size_t fieldPosition = 0;
       fieldPosition < result.fieldType.getShape().size(); ++fieldPosition) {
    if (fieldPosition == projection)
      continue;
    if (resultPosition >= result.resultType.getShape().size() ||
        result.fieldType.getAxisIds()[fieldPosition] !=
            result.resultType.getAxisIds()[resultPosition] ||
        result.fieldType.getShape()[fieldPosition] !=
            result.resultType.getShape()[resultPosition])
      return std::nullopt;
    if (result.resultType.getLayout().getLaneFactors()[resultPosition] > 1) {
      if (result.laneAxis != 0)
        return std::nullopt;
      result.laneAxis = result.resultType.getAxisIds()[resultPosition];
    }
    ++resultPosition;
  }
  if (resultPosition != result.resultType.getShape().size() ||
      result.laneAxis == 0 || result.laneAxis == result.reductionAxis)
    return std::nullopt;

  if (!canFoldReadConversions(extract, extract, result.extractConversions,
                              llvm::ArrayRef<riscv::ConvertLayoutOp>{}) ||
      !canFoldReadConversions(result.subview, extract,
                              result.subviewConversions,
                              result.extractConversions))
    return std::nullopt;
  return result;
}

struct CompleteLayeredExtract {
  riscv::ExtractOp extract;
  riscv::FieldOp field;
  riscv::PhysicalPointOp point;
  riscv::PhysicalPointOp parent;
  riscv::ValueType resultType;
  LinearIndex relative;
  llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
};

struct CompleteLayeredStorageWindow {
  riscv::RVVStorageWindowOp window;
  riscv::FieldOp field;
  riscv::PhysicalPointOp point;
  riscv::PhysicalPointOp parent;
  riscv::ValueType fieldType;
  riscv::ValueType resultType;
  LinearIndex relative;
};

std::optional<CompleteLayeredStorageWindow>
completeLayeredStorageWindow(riscv::RVVStorageWindowOp window) {
  CompleteLayeredStorageWindow result;
  result.window = window;
  result.field = window.getField().getDefiningOp<riscv::FieldOp>();
  result.point = window.getOrigin().getDefiningOp<riscv::PhysicalPointOp>();
  result.parent = result.point
                      ? result.point.getParent().getDefiningOp<
                            riscv::PhysicalPointOp>()
                      : riscv::PhysicalPointOp();
  result.fieldType =
      result.field
          ? mlir::dyn_cast<riscv::ValueType>(result.field.getResult().getType())
          : riscv::ValueType();
  result.resultType =
      mlir::dyn_cast<riscv::ValueType>(window.getResult().getType());
  auto element = result.fieldType
                     ? mlir::dyn_cast<mlir::IntegerType>(
                           result.fieldType.getElementType())
                     : mlir::IntegerType();
  auto plan = window.getPlan();
  if (!result.field || !result.point || !result.parent || !result.fieldType ||
      !result.resultType || !element || element.isSigned() || !plan ||
      window.getAccess().getForm() != "indexed" ||
      window.getAccess().getMapping() != "grouped_layered" ||
      plan.getKind() != "layered" ||
      plan.getReductionAxis() !=
          result.point.getResult().getType().getDomain().getAxisId() ||
      result.parent.getResult().getType().getDomain().getAxisId() !=
          plan.getReductionAxis() ||
      result.parent.getResult().getType().getDomain().getTail() != "exact")
    return std::nullopt;

  const int64_t group = window.getAccess().getGroupSize();
  const int64_t layer = window.getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto axis = llvm::find(result.resultType.getAxisIds().asArrayRef(),
                         plan.getReductionAxis());
  auto parentPartition = result.parent.getPartition()
                             .getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (group <= 0 || layer <= 0 || group % layer || layers <= 1 ||
      element.getWidth() * layers != 8 ||
      axis == result.resultType.getAxisIds().asArrayRef().end() ||
      !parentPartition || parentPartition.value() % group)
    return std::nullopt;
  const size_t axisPosition = static_cast<size_t>(
      axis - result.resultType.getAxisIds().asArrayRef().begin());
  const int64_t lanes =
      result.resultType.getLayout().getLaneFactors()[axisPosition];
  if (lanes <= 1 || layer % lanes ||
      result.resultType.getShape()[axisPosition] != lanes ||
      result.resultType.getLayout().getTimeFactors()[axisPosition] != 1 ||
      result.resultType.getLayout().getCarrier() != "rvv")
    return std::nullopt;

  decomposeIndex(result.point.getBase(), 1, result.relative);
  decomposeIndex(result.parent.getBase(), -1, result.relative);
  decomposeIndex(window.getLogicalOffset(), 1, result.relative);
  if (!result.relative.valid || result.relative.constant < 0)
    return std::nullopt;
  return result;
}

bool sameCompleteLayeredStorageGroup(CompleteLayeredStorageWindow &lhs,
                                     CompleteLayeredStorageWindow &rhs) {
  return lhs.window->getBlock() == rhs.window->getBlock() &&
         sameFieldEdge(lhs.field, rhs.field) && lhs.parent == rhs.parent &&
         lhs.point.getActive() == rhs.point.getActive() &&
         lhs.point.getPartition() == rhs.point.getPartition() &&
         lhs.point.getResult().getType() == rhs.point.getResult().getType() &&
         lhs.resultType == rhs.resultType &&
         lhs.window.getAccess() == rhs.window.getAccess() &&
         lhs.window.getPlan() == rhs.window.getPlan() &&
         sameTerms(lhs.relative, rhs.relative);
}

std::optional<CompleteLayeredExtract>
completeLayeredExtract(riscv::ExtractOp extract) {
  CompleteLayeredExtract result;
  mlir::Value source = riscv_internal::stripRepresentationConversions(
      extract.getInput(), &result.conversions);
  result.field = source.getDefiningOp<riscv::FieldOp>();
  result.point =
      extract.getIndices().size() == 1
          ? riscv_internal::stripRepresentationConversions(
                extract.getIndices().front())
                .getDefiningOp<riscv::PhysicalPointOp>()
          : riscv::PhysicalPointOp();
  result.parent = result.point
                      ? result.point.getParent().getDefiningOp<
                            riscv::PhysicalPointOp>()
                      : riscv::PhysicalPointOp();
  result.resultType =
      mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto fieldType = result.field
                       ? mlir::dyn_cast<riscv::ValueType>(
                             result.field.getResult().getType())
                       : riscv::ValueType();
  auto element = fieldType
                     ? mlir::dyn_cast<mlir::IntegerType>(
                           fieldType.getElementType())
                     : mlir::IntegerType();
  if (!result.field || !result.point || !result.parent || !result.resultType ||
      !fieldType || !element || element.isSigned() ||
      extract.getAccess().getForm() != "indexed" ||
      extract.getAccess().getMapping() != "grouped_layered" ||
      extract.getIndices().size() != 1 ||
      fieldType.getShape().size() != result.resultType.getShape().size())
    return std::nullopt;

  int64_t domainSelectors = 0;
  for (mlir::Attribute selector : extract.getSelectors()) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
    if (name == "domain")
      ++domainSelectors;
    else if (name != "all")
      return std::nullopt;
  }
  const int64_t axis =
      result.point.getResult().getType().getDomain().getAxisId();
  auto position = llvm::find(result.resultType.getAxisIds().asArrayRef(), axis);
  if (domainSelectors != 1 ||
      position == result.resultType.getAxisIds().asArrayRef().end())
    return std::nullopt;
  const size_t axisPosition = static_cast<size_t>(
      position - result.resultType.getAxisIds().asArrayRef().begin());
  const int64_t group = extract.getAccess().getGroupSize();
  const int64_t layer = extract.getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  const int64_t lanes =
      result.resultType.getLayout().getLaneFactors()[axisPosition];
  auto parentPartition = result.parent.getPartition()
                             .getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (group <= 0 || layer <= 0 || group % layer || layers <= 1 || lanes <= 1 ||
      layer % lanes || element.getWidth() * layers != 8 ||
      extract.getAccess().getBitOffset() % 8 ||
      (extract.getAccess().getOrder() != "lo_first" &&
       extract.getAccess().getOrder() != "hi_first") ||
      result.resultType.getShape()[axisPosition] != lanes ||
      result.resultType.getLayout().getTimeFactors()[axisPosition] != 1 ||
      result.resultType.getLayout().getCarrier() != "rvv" || !parentPartition ||
      parentPartition.value() % group ||
      result.parent.getResult().getType().getDomain().getAxisId() != axis ||
      result.parent.getResult().getType().getDomain().getTail() != "exact")
    return std::nullopt;
  for (size_t index = 0; index < result.resultType.getShape().size(); ++index) {
    if (index == axisPosition)
      continue;
    if (fieldType.getShape()[index] != result.resultType.getShape()[index] ||
        result.resultType.getLayout().getTimeFactors()[index] != 1 ||
        result.resultType.getLayout().getLaneFactors()[index] != 1)
      return std::nullopt;
  }

  decomposeIndex(result.point.getBase(), 1, result.relative);
  decomposeIndex(result.parent.getBase(), -1, result.relative);
  if (!result.relative.valid || result.relative.constant < 0)
    return std::nullopt;
  result.extract = extract;
  return result;
}

bool sameCompleteLayeredGroup(CompleteLayeredExtract &lhs,
                              CompleteLayeredExtract &rhs) {
  return lhs.extract->getBlock() == rhs.extract->getBlock() &&
         sameFieldEdge(lhs.field, rhs.field) && lhs.parent == rhs.parent &&
         lhs.point.getActive() == rhs.point.getActive() &&
         lhs.point.getPartition() == rhs.point.getPartition() &&
         lhs.point.getResult().getType() == rhs.point.getResult().getType() &&
         lhs.resultType == rhs.resultType &&
         lhs.extract.getAccess() == rhs.extract.getAccess() &&
         sameTerms(lhs.relative, rhs.relative);
}

struct InterleavedReplicaGeometry {
  int64_t laneAxis = 0;
  int64_t lanes = 0;
  int64_t parts = 0;
  llvm::SmallVector<int64_t> recordCoordinates;
  llvm::SmallVector<int64_t> windowForPart;
};

std::optional<InterleavedReplicaGeometry>
buildInterleavedReplicaGeometry(riscv::ValueType type, size_t recordRank) {
  if (!type || type.getLayout().getCarrier() != "rvv" || recordRank == 0 ||
      recordRank > type.getAxisIds().size())
    return std::nullopt;
  auto streams = riscv_internal::staticProduct(
      type.getLayout().getTimeFactors().asArrayRef());
  auto replicas = riscv_internal::staticProduct(
      type.getLayout().getReplicaFactors().asArrayRef());
  auto lanes = riscv_internal::staticProduct(
      type.getLayout().getLaneFactors().asArrayRef());
  if (!streams || !replicas || !lanes || *streams <= 0 || *replicas <= 0 ||
      *lanes <= 1 || *streams > std::numeric_limits<int64_t>::max() / *replicas)
    return std::nullopt;

  InterleavedReplicaGeometry result;
  result.lanes = *lanes;
  result.parts = *streams * *replicas;
  for (size_t position = 0; position < type.getShape().size(); ++position) {
    const int64_t time = type.getLayout().getTimeFactors()[position];
    const int64_t lane = type.getLayout().getLaneFactors()[position];
    const int64_t replica = type.getLayout().getReplicaFactors()[position];
    const int64_t fragment = type.getLayout().getFragmentFactors()[position];
    const int64_t local = type.getLayout().getLocalFactors()[position];
    int64_t represented = 0;
    if (time <= 0 || lane <= 0 || replica <= 0 || fragment != 1 || local != 1 ||
        !checkedMultiply(time, lane, represented) ||
        !checkedMultiply(represented, replica, represented) ||
        represented != type.getShape()[position] ||
        (lane > 1 && position >= recordRank))
      return std::nullopt;
    if (lane > 1) {
      if (result.laneAxis)
        return std::nullopt;
      result.laneAxis = type.getAxisIds()[position];
    }
  }
  if (!result.laneAxis)
    return std::nullopt;

  result.recordCoordinates.reserve(
      static_cast<size_t>(result.parts) * recordRank);
  result.windowForPart.reserve(static_cast<size_t>(result.parts));
  for (int64_t part = 0; part < result.parts; ++part) {
    int64_t remainingStream = part % *streams;
    int64_t remainingReplica = part / *streams;
    llvm::SmallVector<int64_t> timeCoordinates(type.getShape().size(), 0);
    llvm::SmallVector<int64_t> replicaCoordinates(type.getShape().size(), 0);
    for (int64_t position = static_cast<int64_t>(type.getShape().size()) - 1;
         position >= 0; --position) {
      const int64_t time =
          type.getLayout().getTimeFactors()[static_cast<size_t>(position)];
      const int64_t replica =
          type.getLayout().getReplicaFactors()[static_cast<size_t>(position)];
      timeCoordinates[static_cast<size_t>(position)] = remainingStream % time;
      remainingStream /= time;
      replicaCoordinates[static_cast<size_t>(position)] =
          remainingReplica % replica;
      remainingReplica /= replica;
    }
    if (remainingStream || remainingReplica)
      return std::nullopt;
    for (size_t position = 0; position < recordRank; ++position) {
      const int64_t lane = type.getLayout().getLaneFactors()[position];
      const int64_t replica = type.getLayout().getReplicaFactors()[position];
      int64_t coordinate = 0;
      if (lane > 1) {
        if (!checkedMultiply(timeCoordinates[position], lane, coordinate) ||
            coordinate + lane > type.getShape()[position])
          return std::nullopt;
      } else if (!checkedMultiply(timeCoordinates[position], replica, coordinate) ||
                 !checkedAdd(coordinate, replicaCoordinates[position], coordinate) ||
                 coordinate >= type.getShape()[position]) {
        return std::nullopt;
      }
      result.recordCoordinates.push_back(coordinate);
    }
    result.windowForPart.push_back(part);
  }
  return result;
}

riscv::ConvertLayoutOp selectedReplicaConsumer(mlir::Value value,
                                               riscv::ValueType sourceType,
                                               riscv::ValueType &selectedType) {
  selectedType = sourceType;
  if (!value.hasOneUse())
    return {};
  auto conversion =
      mlir::dyn_cast<riscv::ConvertLayoutOp>(*value.getUsers().begin());
  auto targetType =
      conversion
          ? mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType())
          : riscv::ValueType();
  if (!conversion || !targetType ||
      conversion.getConversion().getEffect() != "pure" ||
      targetType.getLayout().getCarrier() != "rvv" ||
      targetType.getElementType() != sourceType.getElementType() ||
      targetType.getShape() != sourceType.getShape() ||
      targetType.getAxisIds() != sourceType.getAxisIds())
    return {};
  selectedType = targetType;
  return conversion;
}

bool createInterleavedReplicaLoad(mlir::IRRewriter &rewriter,
                                  mlir::Location location,
                                  riscv::FieldOp field,
                                  riscv::ValueType selectedType,
                                  mlir::Value logicalBase, size_t recordRank,
                                  mlir::Operation *origin,
                                  mlir::Value valueToReplace,
                                  riscv::ConvertLayoutOp consumerConversion) {
  auto ownerEncoding = mlir::dyn_cast<kernel::EncodingType>(
      riscv_internal::logicalElement(field.getOwner().getType()));
  const int64_t interleave =
      ownerEncoding ? riscv_internal::interleaveRows(field, ownerEncoding) : 0;
  auto geometry = buildInterleavedReplicaGeometry(selectedType, recordRank);
  if (interleave <= 0 || !geometry || recordRank != 1 ||
      selectedType.getShape()[0] > interleave ||
      (field.getAccess().getMapping() != "natural" &&
       field.getAccess().getMapping() != "joined"))
    return false;
  const int64_t resultGroups = selectedType.getLayout().getRegisterGroups();
  const int64_t temporaryGroups =
      std::max<int64_t>(1, (selectedType.getLayout().getLmulEighths() + 7) / 8);
  const bool tail = selectedType.getLayout().getValidity() == "tail";
  auto storagePlan = riscv_internal::storageWindowPlan(
      rewriter, field, geometry->laneAxis, 0, 1, 1,
      selectedType.getShape()[0], 1, true);
  if (!storagePlan)
    return false;
  llvm::SmallVector<int64_t> zeros(static_cast<size_t>(geometry->parts), 0);
  llvm::SmallVector<int64_t> windowOffsets(static_cast<size_t>(geometry->parts), 0);
  llvm::StringRef suffix = field.getAccess().getMapping() == "joined"
                              ? "interleaved-joined"
                              : "interleaved-natural";
  const std::string instruction =
      ("rvv.replica-storage-load." + suffix).str();
  if (origin == field.getOperation())
    rewriter.setInsertionPointAfter(field);
  else
    rewriter.setInsertionPoint(origin);
  auto load = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
      location, selectedType, field.getResult(), logicalBase, *storagePlan,
      static_cast<int64_t>(recordRank),
      rewriter.getDenseI64ArrayAttr(windowOffsets),
      rewriter.getDenseI64ArrayAttr(geometry->recordCoordinates),
      rewriter.getDenseI64ArrayAttr(geometry->windowForPart),
      rewriter.getDenseI64ArrayAttr(zeros),
      rewriter.getDenseI64ArrayAttr(zeros),
      rewriter.getDenseI64ArrayAttr(zeros),
      rewriter.getDenseI64ArrayAttr(zeros),
      rewriter.getDenseI64ArrayAttr(zeros), field.getAccess(),
      riscv_internal::leaf(rewriter, "rvv", "replica-storage-load",
                           instruction, instruction, 0, resultGroups,
                           temporaryGroups, 0, "none",
                           tail ? "agnostic" : "exact"));
  riscv_internal::copyOrigin(origin, load);
  if (auto canonical = origin->getAttr("canonical_op"))
    load->setAttr("canonical_op", canonical);
  if (consumerConversion) {
    consumerConversion.getResult().replaceAllUsesWith(load.getResult());
    rewriter.eraseOp(consumerConversion);
  } else {
    valueToReplace.replaceAllUsesExcept(load.getResult(), load.getOperation());
  }
  return true;
}

bool materializeInterleavedReplicaExtract(mlir::IRRewriter &rewriter,
                                          riscv::ExtractOp extract) {
  llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
  mlir::Value input = riscv_internal::stripRepresentationConversions(
      extract.getInput(), &conversions);
  auto field = input.getDefiningOp<riscv::FieldOp>();
  auto fieldType = field
                       ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                       : riscv::ValueType();
  auto sourceType =
      mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  if (!field || !fieldType || !sourceType || extract.getIndices().size() != 1)
    return false;
  const riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (facts.logicalRank <= 0 ||
      facts.logicalRank >= static_cast<int64_t>(fieldType.getAxisIds().size()))
    return false;
  const size_t recordRank =
      fieldType.getAxisIds().size() - static_cast<size_t>(facts.logicalRank);
  if (extract.getSelectors().size() != fieldType.getAxisIds().size())
    return false;
  for (size_t position = 0; position < recordRank; ++position)
    if (mlir::cast<mlir::StringAttr>(extract.getSelectors()[position]).getValue() !=
        "all")
      return false;
  int64_t selectedLogicalAxes = 0;
  for (size_t position = recordRank; position < extract.getSelectors().size();
       ++position) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(extract.getSelectors()[position]).getValue();
    if (selector == "index")
      ++selectedLogicalAxes;
    else if (selector != "all")
      return false;
  }
  mlir::Type baseType = extract.getIndices().front().getType();
  if (selectedLogicalAxes != 1 || sourceType.getAxisIds().size() != recordRank ||
      (!baseType.isIndex() && !mlir::isa<mlir::IntegerType>(baseType)))
    return false;
  riscv::ValueType selectedType;
  riscv::ConvertLayoutOp consumer =
      selectedReplicaConsumer(extract.getResult(), sourceType, selectedType);
  if (!createInterleavedReplicaLoad(
          rewriter, extract.getLoc(), field, selectedType,
          extract.getIndices().front(), recordRank, extract, extract.getResult(),
          consumer))
    return false;
  rewriter.eraseOp(extract);
  for (riscv::ConvertLayoutOp conversion : llvm::reverse(conversions))
    if (conversion.getResult().use_empty())
      rewriter.eraseOp(conversion);
  return true;
}

bool materializeInterleavedReplicaField(mlir::IRRewriter &rewriter,
                                        riscv::FieldOp field) {
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
  const riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (!sourceType || facts.logicalRank != 0 || sourceType.getAxisIds().empty())
    return false;
  riscv::ValueType selectedType;
  riscv::ConvertLayoutOp consumer =
      selectedReplicaConsumer(field.getResult(), sourceType, selectedType);
  rewriter.setInsertionPoint(field);
  auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(field.getLoc(), 0);
  if (!createInterleavedReplicaLoad(
          rewriter, field.getLoc(), field, selectedType, zero,
          sourceType.getAxisIds().size(), field, field.getResult(), consumer)) {
    rewriter.eraseOp(zero);
    return false;
  }
  return true;
}

bool materializeReplicaStorageLoad(mlir::IRRewriter &rewriter,
                                   riscv::ExtractOp extract) {
  llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
  mlir::Value input = riscv_internal::stripRepresentationConversions(
      extract.getInput(), &conversions);
  auto field = input.getDefiningOp<riscv::FieldOp>();
  auto fieldType = field
                       ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                       : riscv::ValueType();
  auto resultType =
      mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  auto kernel = extract->getParentOfType<riscv::KernelOp>();
  if (!field || !fieldType || !resultType || extract.getIndices().size() != 1 ||
      !kernel ||
      extract.getAccess().getForm() != "indexed" ||
      (extract.getAccess().getMapping() != "natural" &&
       extract.getAccess().getMapping() != "grouped_layered"))
    return false;
  int64_t gatherSelectors = 0;
  int64_t domainAxis = -1;
  for (auto [axis, selector] : llvm::enumerate(extract.getSelectors())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
    if (name == "gather")
      ++gatherSelectors;
    else if (name == "domain" && domainAxis == -1)
      domainAxis = axis;
    else if (name != "all")
      return false;
  }
  if (gatherSelectors + (domainAxis >= 0) != 1)
    return false;

  ShapedAffineIndex affine;
  int64_t domainExtent = 0;
  if (domainAxis >= 0) {
    if (domainAxis + 1 != static_cast<int64_t>(fieldType.getShape().size()))
      return false;
    auto point = extract.getIndices().front().getDefiningOp<riscv::PhysicalPointOp>();
    int64_t axis = fieldType.getAxisIds()[domainAxis];
    auto resultAxis = llvm::find(resultType.getAxisIds().asArrayRef(), axis);
    auto partition = point ? constantIndex(point.getPartition()) : std::optional<int64_t>();
    domainExtent = fieldType.getShape()[domainAxis];
    if (!point || point.getResult().getType().getDomain().getAxisId() != axis ||
        point.getResult().getType().getDomain().getTail() != "exact" || !partition || *partition <= 0 ||
        domainExtent <= 0 || domainExtent % *partition || resultAxis == resultType.getAxisIds().asArrayRef().end() ||
        resultType.getShape()[resultAxis - resultType.getAxisIds().asArrayRef().begin()] != *partition)
      return false;
    affine.scalarBase = point.getBase();
    affine.scalarBaseCoefficient = 1;
    affine.axisCoefficients[axis] = 1;
  } else {
    decomposeShapedIndex(extract.getIndices().front(), 1, affine);
  }
  if (!affine.valid)
    return false;

  auto isPreservedFieldAxis = [&](size_t resultPosition) {
    const int64_t axis = resultType.getAxisIds()[resultPosition];
    auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(), axis);
    if (fieldAxis == fieldType.getAxisIds().asArrayRef().end())
      return false;
    const size_t fieldPosition = static_cast<size_t>(
        fieldAxis - fieldType.getAxisIds().asArrayRef().begin());
    return fieldType.getShape()[fieldPosition] ==
           resultType.getShape()[resultPosition];
  };

  auto hasContiguousLaneGeometry = [&](riscv::ValueType type) {
    int64_t expectedStride = 1;
    for (int64_t position = static_cast<int64_t>(type.getShape().size()) - 1;
         position >= 0; --position) {
      const int64_t lane =
          type.getLayout().getLaneFactors()[static_cast<size_t>(position)];
      if (lane <= 1)
        continue;
      const int64_t axis = type.getAxisIds()[static_cast<size_t>(position)];
      if (affine.axisCoefficients.lookup(axis) != expectedStride ||
          !checkedMultiply(expectedStride, lane, expectedStride))
        return false;
    }
    return true;
  };

  riscv::ConvertLayoutOp consumerConversion;
  riscv::ValueType selectedType = resultType;
  if (extract.getResult().hasOneUse()) {
    auto *user = *extract.getResult().getUsers().begin();
    auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(user);
    auto targetType =
        conversion
            ? mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType())
            : riscv::ValueType();
    auto targetLanes =
        targetType
            ? riscv_internal::staticProduct(
                  targetType.getLayout().getLaneFactors().asArrayRef())
            : std::optional<int64_t>();
    const bool targetFitsOneLayer =
        extract.getAccess().getMapping() != "grouped_layered" ||
        (targetLanes && *targetLanes > 0 &&
         extract.getAccess().getLayerSize() % *targetLanes == 0);
    if (conversion && conversion.getConversion().getEffect() == "pure" &&
        targetType &&
        targetType.getLayout().getCarrier() == "rvv" &&
        targetType.getElementType() == resultType.getElementType() &&
        targetType.getShape() == resultType.getShape() &&
        targetType.getAxisIds() == resultType.getAxisIds() &&
        targetFitsOneLayer &&
        (extract.getAccess().getMapping() != "grouped_layered" ||
         hasContiguousLaneGeometry(targetType))) {
      // The typed geometry checks below are sufficient for both natural and
      // grouped/layered storage.  Refusing the latter here forces a narrow raw
      // window followed by register-to-lane packing even when the target lane
      // layout divides the declared group and layer exactly.
      consumerConversion = conversion;
      selectedType = targetType;
    }
  }
  const bool scalarReplicas = resultType.getLayout().getCarrier() == "scalar";
  if (selectedType == resultType && scalarReplicas)
    return false;
  if (selectedType.getLayout().getCarrier() != "rvv") {
    return false;
  }

  const bool layered =
      extract.getAccess().getMapping() == "grouped_layered";
  int64_t laneAxis = 0;
  size_t lanePosition = 0;
  llvm::SmallVector<size_t> lanePositions;
  int64_t lanes = 0;
  int64_t streams = 1;
  int64_t replicas = 1;
  for (size_t position = 0; position < selectedType.getShape().size(); ++position) {
    const int64_t time = selectedType.getLayout().getTimeFactors()[position];
    const int64_t lane = selectedType.getLayout().getLaneFactors()[position];
    const int64_t replica = selectedType.getLayout().getReplicaFactors()[position];
    const int64_t fragment =
        selectedType.getLayout().getFragmentFactors()[position];
    const int64_t local = selectedType.getLayout().getLocalFactors()[position];
    int64_t represented = 0;
    const bool preservedFieldAxis =
        !affine.axisCoefficients.contains(selectedType.getAxisIds()[position]) &&
        isPreservedFieldAxis(position);
    if (time <= 0 || lane <= 0 || replica <= 0 || fragment != 1 || local != 1 ||
        !checkedMultiply(time, lane, represented) ||
        !checkedMultiply(represented, replica, represented) ||
        (selectedType.getShape()[position] > 0 &&
         represented != selectedType.getShape()[position]) ||
        (selectedType.getShape()[position] <= 0 && represented != 1 &&
         !preservedFieldAxis))
      return false;
    if (lane > 1) {
      if (replica != 1)
        return false;
      lanePositions.push_back(position);
      laneAxis = selectedType.getAxisIds()[position];
      lanePosition = position;
    } else {
      if ((time > 1 && replica > 1) ||
          (selectedType.getShape()[position] > 1 &&
           !affine.axisCoefficients.contains(selectedType.getAxisIds()[position]) &&
           !preservedFieldAxis))
        return false;
    }
    if (!checkedMultiply(streams, time, streams) ||
        !checkedMultiply(replicas, replica, replicas))
      return false;
  }
  lanes = 1;
  for (size_t position : lanePositions)
    if (!checkedMultiply(lanes,
                         selectedType.getLayout().getLaneFactors()[position],
                         lanes))
      return false;
  const int64_t laneStride = affine.axisCoefficients.lookup(laneAxis);
  if (!laneAxis || laneStride <= 0 || streams <= 0 || replicas <= 0 ||
      streams > std::numeric_limits<int64_t>::max() / replicas)
    return false;
  const bool profitableStridedWindow =
      laneStride == 1 || lanes >= 4 || kernel.getTarget().getVlenBits() >= 256;
  if ((layered && laneStride != 1) ||
      (laneStride != 1 &&
       (lanePositions.size() != 1 || !profitableStridedWindow)))
    return false;
  for (const auto &[axis, coefficient] : affine.axisCoefficients) {
    auto found = llvm::find(selectedType.getAxisIds().asArrayRef(), axis);
    if (found == selectedType.getAxisIds().asArrayRef().end() || coefficient <= 0)
      return false;
  }

  // Several logical axes may map into one RVV register only when the typed
  // affine storage coordinates prove a contiguous row-major interval.  For a
  // layered field the subsequent group/layer divisibility checks additionally
  // prove that this wider logical interval can be decoded from whole raw
  // windows without crossing an encoded layer boundary.
  if (lanePositions.size() > 1) {
    int64_t expectedStride = 1;
    for (int64_t position =
             static_cast<int64_t>(selectedType.getShape().size()) - 1;
         position >= 0; --position) {
      const int64_t lane =
          selectedType.getLayout().getLaneFactors()[position];
      if (lane <= 1)
        continue;
      const int64_t axis = selectedType.getAxisIds()[position];
      if (affine.axisCoefficients.lookup(axis) != expectedStride ||
          !checkedMultiply(expectedStride, lane, expectedStride))
        return false;
    }
  }

  auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(), laneAxis);
  const bool gatherAxis =
      fieldAxis == fieldType.getAxisIds().asArrayRef().end();
  if (gatherAxis &&
      (fieldType.getShape().size() != 1 ||
       laneStride <= 0 || (layered && laneStride != 1)))
    return false;
  const int64_t fieldExtent =
      gatherAxis
          ? fieldType.getShape()[0]
          : fieldType.getShape()[static_cast<size_t>(
                fieldAxis - fieldType.getAxisIds().asArrayRef().begin())];
  if (fieldExtent <= 0)
    return false;

  llvm::SmallVector<int64_t> partOffsets;
  llvm::SmallVector<llvm::SmallVector<int64_t>> recordCoordinatesForPart;
  const riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (facts.logicalRank < 0 ||
      facts.logicalRank > static_cast<int64_t>(fieldType.getAxisIds().size()))
    return false;
  const size_t recordRank =
      fieldType.getAxisIds().size() - static_cast<size_t>(facts.logicalRank);
  int64_t logicalSpan = 0;
  const int64_t parts = streams * replicas;
  for (int64_t part = 0; part < parts; ++part) {
    int64_t remainingStream = part % streams;
    int64_t remainingReplica = part / streams;
    llvm::SmallVector<int64_t> timeCoordinates(selectedType.getShape().size(), 0);
    llvm::SmallVector<int64_t> replicaCoordinates(selectedType.getShape().size(), 0);
    for (int64_t position = static_cast<int64_t>(selectedType.getShape().size()) - 1;
         position >= 0; --position) {
      const int64_t time =
          selectedType.getLayout().getTimeFactors()[static_cast<size_t>(position)];
      const int64_t replica =
          selectedType.getLayout().getReplicaFactors()[static_cast<size_t>(position)];
      timeCoordinates[static_cast<size_t>(position)] = remainingStream % time;
      remainingStream /= time;
      replicaCoordinates[static_cast<size_t>(position)] =
          remainingReplica % replica;
      remainingReplica /= replica;
    }
    if (remainingStream || remainingReplica)
      return false;
    int64_t logicalOffset = 0;
    for (size_t position = 0; position < selectedType.getShape().size(); ++position) {
      const int64_t time = selectedType.getLayout().getTimeFactors()[position];
      const int64_t replica =
          selectedType.getLayout().getReplicaFactors()[position];
      int64_t coordinate = 0;
      const int64_t lane =
          selectedType.getLayout().getLaneFactors()[position];
      if (lane > 1) {
        if (!checkedMultiply(timeCoordinates[position], lane, coordinate))
          return false;
      } else {
        if (!checkedMultiply(timeCoordinates[position], replica, coordinate) ||
            !checkedAdd(coordinate, replicaCoordinates[position], coordinate))
          return false;
      }
      if (time == 1 && replica == 1 && position != lanePosition)
        continue;
      int64_t contribution = 0;
      if (!checkedMultiply(
              coordinate,
              affine.axisCoefficients.lookup(selectedType.getAxisIds()[position]),
              contribution) ||
          !checkedAdd(logicalOffset, contribution, logicalOffset))
        return false;
    }
    int64_t partAlignment = 0;
    if (!checkedMultiply(lanes, laneStride, partAlignment) ||
        logicalOffset < 0 || logicalOffset % partAlignment)
      return false;
    partOffsets.push_back(logicalOffset);
    llvm::SmallVector<int64_t> recordCoordinates;
    recordCoordinates.reserve(recordRank);
    for (size_t fieldPosition = 0; fieldPosition < recordRank; ++fieldPosition) {
      const int64_t axis = fieldType.getAxisIds()[fieldPosition];
      auto selectedAxis =
          llvm::find(selectedType.getAxisIds().asArrayRef(), axis);
      if (selectedAxis == selectedType.getAxisIds().asArrayRef().end()) {
        if (fieldType.getShape()[fieldPosition] != 1)
          return false;
        recordCoordinates.push_back(0);
        continue;
      }
      const size_t position = static_cast<size_t>(
          selectedAxis - selectedType.getAxisIds().asArrayRef().begin());
      const int64_t lane = selectedType.getLayout().getLaneFactors()[position];
      const int64_t replica =
          selectedType.getLayout().getReplicaFactors()[position];
      int64_t coordinate = 0;
      if (lane > 1) {
        if (!checkedMultiply(timeCoordinates[position], lane, coordinate))
          return false;
      } else if (!checkedMultiply(timeCoordinates[position], replica, coordinate) ||
                 !checkedAdd(coordinate, replicaCoordinates[position], coordinate)) {
        return false;
      }
      recordCoordinates.push_back(coordinate);
    }
    recordCoordinatesForPart.push_back(std::move(recordCoordinates));
    int64_t covered = 0;
    int64_t end = 0;
    if (!checkedMultiply(lanes - 1, laneStride, covered) ||
        !checkedAdd(covered, 1, covered) ||
        !checkedAdd(logicalOffset, covered, end))
      return false;
    logicalSpan = std::max(logicalSpan, end);
  }
  int64_t logicalBaseAlignment = 0;
  if (!checkedMultiply(lanes, laneStride, logicalBaseAlignment))
    return false;
  int64_t scalarBaseMultiple = 1;
  if (affine.scalarBase &&
      (affine.scalarBaseCoefficient <= 0 ||
       !checkedMultiply(knownMultiple(affine.scalarBase),
                        affine.scalarBaseCoefficient, scalarBaseMultiple)))
    return false;
  if (domainExtent) scalarBaseMultiple = std::gcd(scalarBaseMultiple, domainExtent);
  if (logicalSpan <= 0 || logicalSpan > fieldExtent || affine.constant < 0 ||
      (affine.scalarBase && scalarBaseMultiple % logicalBaseAlignment))
    return false;

  const int64_t group = extract.getAccess().getGroupSize();
  const int64_t layer = extract.getAccess().getLayerSize();
  if (layered &&
      (group <= 0 || layer <= 0 || group % layer || layer % lanes ||
       group % logicalSpan ||
       (logicalSpan % layer && layer % logicalSpan) ||
       (affine.scalarBase && scalarBaseMultiple % std::min(logicalSpan, layer)) || affine.constant != 0))
    return false;
  if (!layered && !affine.scalarBase &&
      affine.constant > fieldExtent - logicalSpan)
    return false;
  if (affine.scalarBase && affine.constant != 0)
    return false;

  llvm::SmallVector<int64_t> windowOffsets;
  llvm::SmallVector<int64_t> recordCoordinatesForWindow;
  llvm::SmallVector<int64_t> windowForPart;
  llvm::SmallVector<int64_t> layerForPart;
  for (auto [part, offset] : llvm::enumerate(partOffsets)) {
    const int64_t windowOffset = layered ? offset % layer : offset;
    int64_t windowId = -1;
    for (size_t candidate = 0; candidate < windowOffsets.size(); ++candidate) {
      if (windowOffsets[candidate] != windowOffset)
        continue;
      llvm::ArrayRef<int64_t> coordinates(
          recordCoordinatesForWindow.data() + candidate * recordRank, recordRank);
      if (coordinates ==
          llvm::ArrayRef<int64_t>(recordCoordinatesForPart[part])) {
        windowId = static_cast<int64_t>(candidate);
        break;
      }
    }
    if (windowId < 0) {
      windowId = static_cast<int64_t>(windowOffsets.size());
      windowOffsets.push_back(windowOffset);
      recordCoordinatesForWindow.append(recordCoordinatesForPart[part]);
    }
    windowForPart.push_back(windowId);
    layerForPart.push_back(layered ? offset / layer : 0);
  }

  rewriter.setInsertionPoint(extract);
  mlir::Value scalarBase = affine.scalarBase;
  if (!scalarBase) {
    scalarBase = rewriter.create<mlir::arith::ConstantIndexOp>(extract.getLoc(),
                                                               affine.constant);
  } else if (affine.scalarBaseCoefficient != 1) {
    if (scalarBase.getType().isIndex()) {
      auto coefficient = rewriter.create<mlir::arith::ConstantIndexOp>(
          extract.getLoc(), affine.scalarBaseCoefficient);
      scalarBase = rewriter.create<mlir::arith::MulIOp>(
          extract.getLoc(), scalarBase, coefficient);
    } else if (auto integer =
                   mlir::dyn_cast<mlir::IntegerType>(scalarBase.getType())) {
      auto coefficient = rewriter.create<riscv::ConstantOp>(
          extract.getLoc(), integer,
          rewriter.getIntegerAttr(integer, affine.scalarBaseCoefficient));
      scalarBase = rewriter
                       .create<riscv::BinaryOp>(
                           extract.getLoc(), integer, scalarBase,
                           coefficient.getResult(), "mul",
                           riscv_internal::unselectedLeaf(rewriter))
                       .getResult();
    } else {
      return false;
    }
  }
  if (domainExtent) {
    auto extent = rewriter.create<mlir::arith::ConstantIndexOp>(extract.getLoc(), domainExtent);
    scalarBase = rewriter.create<mlir::arith::RemUIOp>(extract.getLoc(), scalarBase, extent);
  }
  const int64_t logicalBaseMultiple =
      affine.scalarBase
          ? std::max<int64_t>(1, scalarBaseMultiple)
          : (affine.constant == 0
                 ? logicalBaseAlignment
                 : std::max<int64_t>(1, std::abs(affine.constant)));
  llvm::SmallVector<int64_t> physicalLayerForPart;
  llvm::SmallVector<int64_t> shiftOffsetForPart;
  llvm::SmallVector<int64_t> shiftBaseFactorForPart;
  llvm::SmallVector<int64_t> maskValueForPart;
  physicalLayerForPart.reserve(layerForPart.size());
  shiftOffsetForPart.reserve(layerForPart.size());
  shiftBaseFactorForPart.reserve(layerForPart.size());
  maskValueForPart.reserve(layerForPart.size());
  const int64_t layerCount = layered ? group / layer : 1;
  const int64_t elementBits =
      riscv_internal::logicalBitWidth(selectedType.getElementType());
  for (int64_t logicalLayer : layerForPart) {
    if (!layered) {
      physicalLayerForPart.push_back(0);
      shiftOffsetForPart.push_back(0);
      shiftBaseFactorForPart.push_back(0);
      maskValueForPart.push_back(0);
      continue;
    }
    auto selection = selectLayer(
        extract.getAccess(), logicalLayer, layerCount, elementBits,
        logicalBaseMultiple % group);
    if (!selection)
      return false;
    physicalLayerForPart.push_back(selection->physicalLayer);
    if (selection->mode == 0) {
      shiftOffsetForPart.push_back(selection->physicalLayer * elementBits);
      shiftBaseFactorForPart.push_back(0);
    } else if (selection->mode == 1) {
      shiftOffsetForPart.push_back(logicalLayer * elementBits);
      shiftBaseFactorForPart.push_back(elementBits);
    } else if (selection->mode == 2) {
      shiftOffsetForPart.push_back(
          (layerCount - 1 - logicalLayer) * elementBits);
      shiftBaseFactorForPart.push_back(-elementBits);
    } else {
      return false;
    }
    maskValueForPart.push_back(
        selection->mask ? (int64_t{1} << elementBits) - 1 : 0);
  }
  const int64_t resultGroups = selectedType.getLayout().getRegisterGroups();
  const int64_t temporaryGroups = std::max<int64_t>(
      1, (selectedType.getLayout().getLmulEighths() + 7) / 8);
  const bool tail = selectedType.getLayout().getValidity() == "tail";
  const bool strided = !layered && laneStride > 1;
  llvm::StringRef instruction =
      layered ? "rvv.replica-storage-load.layered"
              : strided ? "rvv.replica-storage-load.strided"
                        : "rvv.replica-storage-load.natural";
  auto storagePlan = riscv_internal::storageWindowPlan(
      rewriter, field, laneAxis, 0, laneStride, 1, logicalSpan,
      logicalBaseMultiple);
  if (!storagePlan)
    return false;
  riscv::AccessAttr loadAccess = extract.getAccess();
  if (strided)
    loadAccess = riscv::AccessAttr::get(
        rewriter.getContext(), "strided", loadAccess.getMapping(),
        loadAccess.getAlignment(), 0, 0, loadAccess.getGroupSize(),
        loadAccess.getLayerSize(), loadAccess.getJoinFields(),
        loadAccess.getJoinLowBits(), loadAccess.getJoinRole(),
        loadAccess.getBitOffset(), loadAccess.getStorageBits(),
        loadAccess.getOrder(), loadAccess.getFieldChunkBytes());
  auto load = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
      extract.getLoc(), selectedType, field.getResult(), scalarBase,
      *storagePlan, static_cast<int64_t>(recordRank),
      rewriter.getDenseI64ArrayAttr(windowOffsets),
      rewriter.getDenseI64ArrayAttr(recordCoordinatesForWindow),
      rewriter.getDenseI64ArrayAttr(windowForPart),
      rewriter.getDenseI64ArrayAttr(layerForPart),
      rewriter.getDenseI64ArrayAttr(physicalLayerForPart),
      rewriter.getDenseI64ArrayAttr(shiftOffsetForPart),
      rewriter.getDenseI64ArrayAttr(shiftBaseFactorForPart),
      rewriter.getDenseI64ArrayAttr(maskValueForPart), loadAccess,
      riscv_internal::leaf(rewriter, "rvv", "replica-storage-load",
                           instruction, instruction, 0, resultGroups,
                           temporaryGroups, 0, "none",
                           tail ? "agnostic" : "exact"));
  riscv_internal::copyOrigin(extract, load);
  if (auto canonical = extract->getAttr("canonical_op"))
    load->setAttr("canonical_op", canonical);
  mlir::Value replacement = load.getResult();
  if (consumerConversion) {
    consumerConversion.getResult().replaceAllUsesWith(replacement);
    rewriter.eraseOp(consumerConversion);
  }
  extract.getResult().replaceAllUsesWith(replacement);
  rewriter.eraseOp(extract);
  for (riscv::ConvertLayoutOp conversion : llvm::reverse(conversions))
    if (conversion.getResult().use_empty())
      rewriter.eraseOp(conversion);
  return true;
}

void eraseDeadPureProducers(mlir::IRRewriter &rewriter,
                            llvm::ArrayRef<mlir::Value> roots) {
  llvm::SmallVector<mlir::Operation *> worklist;
  llvm::SmallVector<mlir::Operation *> candidates;
  llvm::DenseSet<mlir::Operation *> seen;
  for (mlir::Value root : roots)
    if (mlir::Operation *definition = root.getDefiningOp())
      worklist.push_back(definition);
  while (!worklist.empty()) {
    mlir::Operation *operation = worklist.pop_back_val();
    if (!operation || !seen.insert(operation).second ||
        operation->getNumRegions() != 0)
      continue;
    if (!mlir::isMemoryEffectFree(operation))
      continue;
    candidates.push_back(operation);
    for (mlir::Value operand : operation->getOperands())
      if (mlir::Operation *definition = operand.getDefiningOp())
        worklist.push_back(definition);
  }

  llvm::DenseSet<mlir::Operation *> erased;
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation *operation : llvm::reverse(candidates)) {
      if (erased.contains(operation) ||
          llvm::any_of(operation->getResults(),
                       [](mlir::Value result) { return !result.use_empty(); }))
        continue;
      rewriter.eraseOp(operation);
      erased.insert(operation);
      changed = true;
    }
  }
}

void materializeReplicaStorageLoads(mlir::IRRewriter &rewriter,
                                    mlir::ModuleOp module) {
  llvm::SmallVector<riscv::ExtractOp> extracts;
  module.walk([&](riscv::ExtractOp extract) { extracts.push_back(extract); });
  llvm::SmallVector<mlir::Value> replacedIndices;
  for (riscv::ExtractOp extract : extracts) {
    llvm::SmallVector<mlir::Value> indices(extract.getIndices().begin(),
                                           extract.getIndices().end());
    if (materializeInterleavedReplicaExtract(rewriter, extract) ||
        materializeReplicaStorageLoad(rewriter, extract))
      replacedIndices.append(indices.begin(), indices.end());
  }
  eraseDeadPureProducers(rewriter, replacedIndices);
}

bool readRangeIsShareable(mlir::Operation *first, mlir::Operation *last) {
  if (!first || !last || first->getBlock() != last->getBlock())
    return false;
  for (mlir::Operation *operation = first; operation;
       operation = operation->getNextNode()) {
    // Physical points are pure SSA coordinates, but deliberately carry no
    // generic fold trait.  A read may cross them when it depends only on the
    // already-dominating parent point used by the shared window.
    if (!mlir::isa<riscv::PhysicalPointOp>(operation) &&
        !isReadOnlyOrPure(operation))
      return false;
    if (operation == last)
      return true;
  }
  return false;
}

std::optional<int64_t>
singleNaturalWindowBase(riscv::RVVReplicaStorageLoadOp load) {
  auto type = load.getResult().getType();
  auto layout = type.getLayout();
  auto plan = load.getPlan();
  auto ones = [](llvm::ArrayRef<int64_t> factors) {
    return llvm::all_of(factors, [](int64_t factor) { return factor == 1; });
  };
  if (plan.getKind() != "unit" || load.getAccess().getMapping() != "natural" ||
      plan.getProjectionBase() != 0 || plan.getProjectionStride() != 1 ||
      plan.getProjectionRepeat() != 1 || layout.getLmulEighths() > 8 ||
      !ones(layout.getTimeFactors().asArrayRef()) ||
      !ones(layout.getReplicaFactors().asArrayRef()) ||
      !ones(layout.getFragmentFactors().asArrayRef()) ||
      !ones(layout.getLocalFactors().asArrayRef()) ||
      load.getWindowOffsets().size() != 1 ||
      load.getWindowForPart() != llvm::ArrayRef<int64_t>({0}))
    return std::nullopt;
  LinearIndex base;
  decomposeIndex(load.getLogicalBase(), 1, base);
  int64_t offset = 0;
  if (!base.valid || !base.terms.empty() ||
      !checkedAdd(base.constant, load.getWindowOffsets()[0], offset) || offset < 0)
    return std::nullopt;
  return offset;
}

bool isContiguousSupplySlice(riscv::ValueType source,
                             riscv::ValueType result) {
  if (source == result)
    return true;
  auto sourceLayout = source.getLayout();
  auto resultLayout = result.getLayout();
  if (source.getElementType() != result.getElementType() ||
      source.getAxisIds() != result.getAxisIds() ||
      sourceLayout.getSew() != resultLayout.getSew() ||
      sourceLayout.getValidity() != resultLayout.getValidity() ||
      sourceLayout.getLmulEighths() < resultLayout.getLmulEighths() ||
      sourceLayout.getLmulEighths() % resultLayout.getLmulEighths())
    return false;
  bool sliced = false;
  bool earlierLane = false;
  for (size_t position = 0; position < source.getShape().size(); ++position) {
    if (source.getShape()[position] <= 0 || result.getShape()[position] <= 0)
      return false;
    const int64_t sourceLanes = sourceLayout.getLaneFactors()[position];
    const int64_t resultLanes = resultLayout.getLaneFactors()[position];
    if (source.getShape()[position] == result.getShape()[position]) {
      if (sourceLanes != resultLanes)
        return false;
    } else {
      if (sliced || earlierLane || source.getShape()[position] != sourceLanes ||
          result.getShape()[position] != resultLanes || resultLanes <= 0 ||
          sourceLanes < resultLanes || sourceLanes % resultLanes)
        return false;
      sliced = true;
    }
    earlierLane |= sourceLanes > 1;
  }
  return true;
}

void reuseNaturalStorageWindows(mlir::IRRewriter &rewriter,
                                mlir::ModuleOp module) {
  struct AvailableWindow {
    riscv::RVVReplicaStorageLoadOp load;
    mlir::Value value;
    int64_t base;
    int64_t lanes;
  };
  module.walk([&](mlir::Block *block) {
    llvm::SmallVector<riscv::RVVReplicaStorageLoadOp> loads;
    for (mlir::Operation &operation : *block)
      if (auto load = mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(operation))
        loads.push_back(load);
    llvm::SmallVector<AvailableWindow> available;
    llvm::SmallVector<riscv::RVVReplicaStorageLoadOp> replaced;
    for (riscv::RVVReplicaStorageLoadOp load : loads) {
      if (load.getResult().use_empty())
        continue;
      auto base = singleNaturalWindowBase(load);
      auto type = load.getResult().getType();
      auto lanes = riscv::rvvLaneCount(type);
      if (!base || !lanes || *lanes <= 0)
        continue;
      mlir::Value replacement;
      unsigned inspected = 0;
      for (AvailableWindow &prior : llvm::reverse(available)) {
        if (++inspected > 32)
          break;
        if (prior.load.getField() != load.getField() ||
            prior.load.getRecordRank() != load.getRecordRank() ||
            prior.load.getRecordCoordinatesForWindow() !=
                load.getRecordCoordinatesForWindow() ||
            prior.load.getPlan().getReductionAxis() !=
                load.getPlan().getReductionAxis() ||
            *base < prior.base || prior.lanes < *lanes)
          continue;
        const int64_t offset = *base - prior.base;
        auto priorType = mlir::cast<riscv::ValueType>(prior.value.getType());
        if (offset > prior.lanes - *lanes || offset % *lanes ||
            !isContiguousSupplySlice(priorType, type) ||
            !readRangeIsShareable(prior.load, load))
          continue;
        replacement = prior.value;
        if (offset || priorType != type) {
          rewriter.setInsertionPoint(load);
          auto slice = rewriter.create<riscv::RVVIssueSliceOp>(
              load.getLoc(), type, replacement,
              rewriter.getDenseI64ArrayAttr({0}),
              rewriter.getDenseI64ArrayAttr({offset}),
              riscv_internal::leaf(
                  rewriter, "transfer", "issue-slice", "rvv.issue-slice",
                  "rvv.issue-slice", 0, type.getLayout().getRegisterGroups(),
                  0, 0, "none", "exact"));
          riscv_internal::copyOrigin(load, slice);
          replacement = slice.getResult();
        }
        load.getResult().replaceAllUsesWith(replacement);
        replaced.push_back(load);
        break;
      }
      available.push_back(
          {load, replacement ? replacement : load.getResult(), *base, *lanes});
    }
    for (riscv::RVVReplicaStorageLoadOp load : replaced)
      rewriter.eraseOp(load);
  });
}

std::optional<int64_t>
segmentPairBase(riscv::RVVReplicaStorageLoadOp load) {
  auto field = load.getField().getType();
  auto type = load.getResult().getType();
  auto layout = type.getLayout();
  auto plan = load.getPlan();
  auto base = constantIndex(load.getLogicalBase());
  if (!base || *base < 0 || field.getShape().size() != 1 ||
      field.getLayout().getCarrier() != "local" ||
      type.getShape().size() != 1 || layout.getCarrier() != "rvv" ||
      layout.getValidity() != "full" || layout.getLmulEighths() <= 0 ||
      layout.getLmulEighths() > 32 || layout.getLaneFactors()[0] <= 1 ||
      layout.getReplicaFactors()[0] != 1 ||
      layout.getFragmentFactors()[0] != 1 ||
      layout.getLocalFactors()[0] != 1 || load.getRecordRank() != 0 ||
      !load.getRecordCoordinatesForWindow().empty() ||
      plan.getKind() != "strided" || plan.getProjectionStride() != 2 ||
      plan.getProjectionBase() != 0 || plan.getProjectionRepeat() != 1 ||
      load.getAccess().getMapping() != "natural" ||
      load.getAccess().getAlignment() < layout.getSew() / 8 ||
      load.getAccess().getAlignment() % (layout.getSew() / 8) ||
      load.getLeaf().getInstruction() != "rvv.replica-storage-load.strided")
    return std::nullopt;
  const int64_t lanes = layout.getLaneFactors()[0];
  const int64_t elements = type.getShape()[0];
  const int64_t streams = layout.getTimeFactors()[0];
  if (elements <= 0 || streams <= 0 || elements % lanes ||
      elements / lanes != streams ||
      load.getWindowOffsets().size() != static_cast<size_t>(streams) ||
      load.getWindowForPart().size() != static_cast<size_t>(streams) ||
      *base >= field.getShape()[0] ||
      elements - 1 > (field.getShape()[0] - *base - 1) / 2)
    return std::nullopt;
  for (int64_t part = 0; part < streams; ++part)
    if (load.getWindowForPart()[part] != part ||
        load.getWindowOffsets()[part] != part * lanes * 2)
      return std::nullopt;
  return base;
}

void materializeSegmentPairs(mlir::IRRewriter &rewriter,
                             mlir::ModuleOp module) {
  module.walk([&](mlir::Block *block) {
    llvm::SmallVector<riscv::RVVReplicaStorageLoadOp> loads;
    for (mlir::Operation &operation : *block)
      if (auto load = mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(operation))
        loads.push_back(load);
    llvm::DenseSet<mlir::Operation *> consumed;
    for (auto [position, first] : llvm::enumerate(loads)) {
      if (consumed.contains(first) || first.getResult().use_empty())
        continue;
      auto base = segmentPairBase(first);
      if (!base)
        continue;
      auto field = first.getField().getDefiningOp<riscv::FieldOp>();
      auto type = first.getResult().getType();
      auto kernel = first->getParentOfType<riscv::KernelOp>();
      const int64_t elements = type.getShape()[0];
      const int64_t tupleGroups =
          2 * std::max<int64_t>(1, (type.getLayout().getLmulEighths() + 7) / 8);
      const int64_t parts = type.getLayout().getTimeFactors()[0];
      if (!kernel || !kernel.getTarget().getHasSegmentMemory() ||
          parts > kernel.getTarget().getVectorRegisters() / tupleGroups ||
          elements > (first.getField().getType().getShape()[0] - *base) / 2)
        continue;
      unsigned inspected = 0;
      for (riscv::RVVReplicaStorageLoadOp second :
           llvm::drop_begin(loads, position + 1)) {
        if (++inspected > 32)
          break;
        if (consumed.contains(second) || second.getResult().use_empty())
          continue;
        auto secondBase = segmentPairBase(second);
        if (!secondBase || *secondBase != *base + 1 ||
            second.getResult().getType() != type ||
            !sameFieldEdge(field,
                           second.getField().getDefiningOp<riscv::FieldOp>()) ||
            !readRangeIsShareable(first, second))
          continue;
        rewriter.setInsertionPoint(first);
        auto sourceAccess = first.getAccess();
        auto access = riscv::AccessAttr::get(
            rewriter.getContext(), "segment", "natural",
            type.getLayout().getSew() / 8, 0, 2, 0, 0, 0, 0, 0,
            sourceAccess.getBitOffset(), sourceAccess.getStorageBits(),
            sourceAccess.getOrder(), sourceAccess.getFieldChunkBytes());
        auto pair = rewriter.create<riscv::RVVSegmentPairLoadOp>(
            first.getLoc(), type, type, first.getField(), *base, access,
            riscv_internal::leaf(
                rewriter, "rvv", "segment-pair-load", "rvv.vlseg2",
                "rvv.vlseg2", 0, 2 * type.getLayout().getRegisterGroups(),
                parts * tupleGroups, 0, "none", "exact"));
        riscv_internal::copyOrigin(first, pair);
        first.getResult().replaceAllUsesWith(pair.getFirst());
        second.getResult().replaceAllUsesWith(pair.getSecond());
        consumed.insert(first);
        consumed.insert(second);
        break;
      }
    }
    for (riscv::RVVReplicaStorageLoadOp load : loads)
      if (consumed.contains(load))
        rewriter.eraseOp(load);
  });
}

void materializeSharedLayeredStorageWindows(mlir::IRRewriter &rewriter,
                                            mlir::ModuleOp module) {
  llvm::SmallVector<riscv::RVVStorageWindowOp> windows;
  module.walk([&](riscv::RVVStorageWindowOp window) {
    windows.push_back(window);
  });
  llvm::DenseSet<mlir::Operation *> consumed;

  for (riscv::RVVStorageWindowOp first : windows) {
    if (consumed.contains(first))
      continue;
    auto firstInfo = completeLayeredStorageWindow(first);
    if (!firstInfo)
      continue;
    const int64_t group = first.getAccess().getGroupSize();
    const int64_t layer = first.getAccess().getLayerSize();
    const int64_t layers = group / layer;
    const int64_t axis = first.getPlan().getReductionAxis();
    auto axisIt = llvm::find(firstInfo->resultType.getAxisIds().asArrayRef(), axis);
    if (axisIt == firstInfo->resultType.getAxisIds().asArrayRef().end())
      continue;
    const size_t axisPosition = static_cast<size_t>(
        axisIt - firstInfo->resultType.getAxisIds().asArrayRef().begin());
    const int64_t lanes =
        firstInfo->resultType.getLayout().getLaneFactors()[axisPosition];
    const int64_t windowsPerLayer = layer / lanes;
    const int64_t slots = layers * windowsPerLayer;
    const int64_t groupBase =
        (firstInfo->relative.constant / group) * group;
    if (slots <= 1)
      continue;

    llvm::SmallVector<
        llvm::SmallVector<CompleteLayeredStorageWindow, 1>, 8>
        bySlot(static_cast<size_t>(slots));
    for (riscv::RVVStorageWindowOp candidate : windows) {
      if (consumed.contains(candidate))
        continue;
      auto info = completeLayeredStorageWindow(candidate);
      if (!info || !sameCompleteLayeredStorageGroup(*firstInfo, *info))
        continue;
      const int64_t delta = info->relative.constant - groupBase;
      if (delta < 0 || delta >= group || delta % lanes)
        continue;
      bySlot[static_cast<size_t>(delta / lanes)].push_back(std::move(*info));
    }
    if (llvm::any_of(bySlot, [](const auto &slot) { return slot.empty(); }))
      continue;

    mlir::Operation *insertion = bySlot.front().front().window;
    mlir::Operation *last = insertion;
    for (auto &slot : bySlot)
      for (CompleteLayeredStorageWindow &info : slot) {
        if (info.window->isBeforeInBlock(insertion))
          insertion = info.window;
        if (last->isBeforeInBlock(info.window))
          last = info.window;
      }
    if (!readRangeIsShareable(insertion, last))
      continue;

    CompleteLayeredStorageWindow &leader = bySlot.front().front();
    auto fieldAxis = llvm::find(leader.fieldType.getAxisIds().asArrayRef(), axis);
    if (fieldAxis == leader.fieldType.getAxisIds().asArrayRef().end())
      continue;
    const int64_t projectionExtent =
        leader.fieldType.getShape()[static_cast<size_t>(
            fieldAxis - leader.fieldType.getAxisIds().asArrayRef().begin())];
    const int64_t rawGroups =
        leader.resultType.getLayout().getRegisterGroups();
    auto storagePlan = riscv_internal::storageWindowPlan(
        rewriter, leader.field, axis, 0, 1, 1, projectionExtent, lanes);
    if (!storagePlan)
      continue;
    llvm::SmallVector<LayerSelection> layerPlan;
    layerPlan.reserve(static_cast<size_t>(layers));
    for (int64_t logicalLayer = 0; logicalLayer < layers; ++logicalLayer) {
      auto selection = selectLayer(
          first.getAccess(), logicalLayer, layers,
          riscv_internal::logicalBitWidth(leader.resultType.getElementType()));
      if (!selection)
        break;
      layerPlan.push_back(*selection);
    }
    if (layerPlan.size() != static_cast<size_t>(layers))
      continue;

    rewriter.setInsertionPoint(insertion);
    auto groupValue = rewriter.create<mlir::arith::ConstantIndexOp>(
        first.getLoc(), group);
    auto relativeBase = rewriter.create<mlir::arith::SubIOp>(
        first.getLoc(), leader.point.getBase(), leader.parent.getBase());
    mlir::Value logicalBase = relativeBase;
    auto constantOffset = constantIndex(leader.window.getLogicalOffset());
    if (!constantOffset || *constantOffset != 0)
      logicalBase = rewriter.create<mlir::arith::AddIOp>(
          first.getLoc(), relativeBase, leader.window.getLogicalOffset());
    auto groupIndex = rewriter.create<mlir::arith::DivUIOp>(
        first.getLoc(), logicalBase, groupValue);
    mlir::Value windowGroup = groupIndex;
    if (windowsPerLayer != 1) {
      auto windowsValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          first.getLoc(), windowsPerLayer);
      windowGroup = rewriter.create<mlir::arith::MulIOp>(
          first.getLoc(), groupIndex, windowsValue);
    }

    auto storageType = riscv::LayeredWindowType::get(
        rewriter.getContext(), leader.fieldType, leader.resultType, axis, layers,
        windowsPerLayer, rawGroups);
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 2> decoded(
        static_cast<size_t>(windowsPerLayer));
    for (int64_t window = 0; window < windowsPerLayer; ++window) {
      mlir::Value windowIndex = windowGroup;
      if (window) {
        auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(
            first.getLoc(), window);
        windowIndex = rewriter.create<mlir::arith::AddIOp>(
            first.getLoc(), windowGroup, offset);
      }
      auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
          first.getLoc(), storageType, leader.field.getResult(),
          leader.parent.getResult(), windowIndex, *storagePlan, first.getAccess(),
          riscv_internal::leaf(
              rewriter, "rvv", "layered-storage-load",
              "rvv.layered-storage-load", "rvv.layered-storage-load",
              leader.fieldType.getLayout().getRegisterGroups(), rawGroups,
              rawGroups, 0, "none",
              leader.resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                                    : "exact"));
      riscv_internal::copyOrigin(first, storage);
      for (int64_t logicalLayer = 0; logicalLayer < layers; ++logicalLayer) {
        const LayerSelection &selection =
            layerPlan[static_cast<size_t>(logicalLayer)];
        const int64_t elementBits =
            riscv_internal::logicalBitWidth(leader.resultType.getElementType());
        const int64_t shiftAmount = selection.physicalLayer * elementBits;
        const int64_t maskValue =
            selection.mask ? (int64_t{1} << elementBits) - 1 : 0;
        llvm::StringRef instruction =
            riscv_internal::layeredStorageDecodeInstruction(shiftAmount,
                                                            maskValue);
        auto value = rewriter.create<riscv::RVVLayeredStorageDecodeOp>(
            first.getLoc(), leader.resultType, storage.getResult(), logicalLayer,
            selection.physicalLayer, shiftAmount, maskValue,
            riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-decode", instruction,
                instruction, rawGroups,
                leader.resultType.getLayout().getRegisterGroups(), 1, 0, "none",
                leader.resultType.getLayout().getValidity() == "tail"
                    ? "agnostic"
                    : "exact"));
        riscv_internal::copyOrigin(first, value);
        decoded[static_cast<size_t>(window)].push_back(value.getResult());
      }
    }

    for (int64_t logicalLayer = 0; logicalLayer < layers; ++logicalLayer)
      for (int64_t window = 0; window < windowsPerLayer; ++window) {
        const size_t slot = static_cast<size_t>(
            logicalLayer * windowsPerLayer + window);
        for (CompleteLayeredStorageWindow &info : bySlot[slot]) {
          info.window.getResult().replaceAllUsesWith(
              decoded[static_cast<size_t>(window)]
                     [static_cast<size_t>(logicalLayer)]);
          consumed.insert(info.window);
        }
      }

    llvm::DenseSet<mlir::Operation *> deadFields;
    for (auto &slot : bySlot)
      for (CompleteLayeredStorageWindow &info : slot) {
        rewriter.eraseOp(info.window);
        if (info.field != leader.field && info.field.getResult().use_empty())
          deadFields.insert(info.field);
      }
    for (mlir::Operation *field : deadFields)
      rewriter.eraseOp(field);
  }
}

bool hasTypedStorageMaterialization(riscv::FieldOp field) {
  return llvm::any_of(field.getResult().getUsers(), [](mlir::Operation *user) {
    return mlir::isa<riscv::RVVRecordStorageLoadOp,
                     riscv::RVVReplicaStorageLoadOp,
                     riscv::RVVLayeredWindowOp,
                     riscv::RVVLayeredStreamOp,
                     riscv::RVVProjectedLayeredStreamOp,
                     riscv::RVVLayeredRecordLoadOp,
                     riscv::RVVLayeredStorageLoadOp,
                     riscv::RVVStorageWindowOp>(user);
  });
}

class ShareRISCVLayeredWindowsPass
    : public mlir::PassWrapper<ShareRISCVLayeredWindowsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-share-layered-windows";
  }
  llvm::StringRef getDescription() const override {
    return "Represent adjacent packed layers as one shared RVV storage window";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    if (mlir::failed(riscv_internal::shareFragmentStorage(getOperation(), rewriter))) {
      signalPassFailure();
      return;
    }
    llvm::SmallVector<riscv::FieldOp> fields;
    getOperation().walk(
        [&](riscv::FieldOp field) { fields.push_back(field); });
    for (riscv::FieldOp field : fields) {
      if (hasTypedStorageMaterialization(field))
        continue;
      if (materializeInterleavedReplicaField(rewriter, field))
        continue;
      if (!isLayeredStreamField(field))
        continue;
      auto type = mlir::cast<riscv::ValueType>(field.getResult().getType());
      const int64_t resultGroups = type.getLayout().getRegisterGroups();
      const int64_t temporaryGroups =
          std::max<int64_t>(1,
                            (type.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = type.getLayout().getValidity() == "tail";
      int64_t streamAxis = 0;
      for (auto [axis, time, lane] :
           llvm::zip(type.getAxisIds().asArrayRef(),
                     type.getLayout().getTimeFactors().asArrayRef(),
                     type.getLayout().getLaneFactors().asArrayRef()))
        if (time > 1 && lane > 1)
          streamAxis = axis;
      auto geometry = buildLayeredStreamGeometry(
          rewriter, type, field.getAccess(), streamAxis, 0);
      if (!geometry)
        continue;
      rewriter.setInsertionPointAfter(field);
      auto stream = rewriter.create<riscv::RVVLayeredStreamOp>(
          field.getLoc(), type, field.getResult(), field.getAccess(), *geometry,
          riscv_internal::leaf(rewriter, "rvv", "layered-stream",
                               "rvv.layered-stream", "rvv.layered-stream", 0,
                               resultGroups, temporaryGroups, 0, "none",
                               tail ? "agnostic" : "exact"));
      if (auto origin = field->getAttr("source_origin"))
        stream->setAttr("source_origin", origin);
      stream->setAttr("canonical_op",
                      rewriter.getStringAttr("weft_kernel.field"));
      field.getResult().replaceAllUsesExcept(stream.getResult(),
                                             stream.getOperation());
    }

    llvm::SmallVector<riscv::ExtractOp> projectedExtracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { projectedExtracts.push_back(extract); });
    for (riscv::ExtractOp extract : projectedExtracts) {
      riscv::FieldOp field;
      riscv::PhysicalPointOp origin;
      llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
      int64_t axis = 0;
      int64_t base = 0;
      int64_t stride = 0;
      int64_t repeat = 0;
      int64_t extent = 0;
      if (!isProjectedLayeredExtract(extract, field, origin, axis, base, stride,
                                     repeat, extent, &conversions))
        continue;
      auto type = mlir::cast<riscv::ValueType>(extract.getResult().getType());
      const int64_t resultGroups = type.getLayout().getRegisterGroups();
      const int64_t temporaryGroups =
          std::max<int64_t>(1, (type.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = type.getLayout().getValidity() == "tail";
      auto geometry = buildLayeredStreamGeometry(rewriter, type,
                                                 extract.getAccess(), axis, base);
      if (!geometry)
        continue;
      rewriter.setInsertionPoint(extract);
      auto stream = rewriter.create<riscv::RVVProjectedLayeredStreamOp>(
          extract.getLoc(), type, field.getResult(), origin.getResult(), axis,
          base, stride, repeat, extent, extract.getAccess(), *geometry,
          riscv_internal::leaf(
              rewriter, "rvv", "projected-layered-stream",
              "rvv.projected-layered-stream", "rvv.projected-layered-stream",
              mlir::cast<riscv::ValueType>(field.getResult().getType())
                  .getLayout()
                  .getRegisterGroups(),
              resultGroups, temporaryGroups, 0, "none",
              tail ? "agnostic" : "exact"));
      riscv_internal::copyOrigin(extract, stream);
      if (auto canonical = extract->getAttr("canonical_op"))
        stream->setAttr("canonical_op", canonical);
      extract.getResult().replaceAllUsesWith(stream.getResult());
      rewriter.eraseOp(extract);
      for (riscv::ConvertLayoutOp conversion : llvm::reverse(conversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
    }

    // A local encoded tile can retain its reduction coordinate in time while a
    // different free axis occupies RVV lanes.  Materialize the child
    // subscript as a typed record-strided load: the point and scalar offset
    // select one logical reduction element, and the result layout selects the
    // records visited by each RVV issue part.
    llvm::SmallVector<riscv::ExtractOp> recordExtracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { recordExtracts.push_back(extract); });
    llvm::SmallVector<riscv::ExtractOp> deadSubviews;
    for (riscv::ExtractOp extract : recordExtracts) {
      auto info = layeredRecordExtract(extract);
      if (!info)
        continue;
      const int64_t lanes = [&]() {
        auto axis = llvm::find(info->resultType.getAxisIds().asArrayRef(),
                               info->laneAxis);
        return axis == info->resultType.getAxisIds().asArrayRef().end()
                   ? int64_t{0}
                   : info->resultType.getLayout().getLaneFactors()[
                         static_cast<size_t>(
                             axis - info->resultType.getAxisIds().asArrayRef().begin())];
      }();
      auto storagePlan = riscv_internal::storageWindowPlan(
          rewriter, info->field, info->reductionAxis, 0, 1, 1,
          info->projectionExtent, 1);
      if (!storagePlan || lanes <= 1)
        continue;
      const int64_t resultGroups =
          info->resultType.getLayout().getRegisterGroups();
      const int64_t temporaryGroups = std::max<int64_t>(
          1, (info->resultType.getLayout().getLmulEighths() + 7) / 8);
      const bool tail = info->resultType.getLayout().getValidity() == "tail";
      rewriter.setInsertionPoint(extract);
      auto load = rewriter.create<riscv::RVVLayeredRecordLoadOp>(
          extract.getLoc(), info->resultType, info->field.getResult(),
          info->origin.getResult(), extract.getIndices().front(), *storagePlan,
          extract.getAccess(),
          riscv_internal::leaf(
              rewriter, "rvv", "layered-record-load",
              "rvv.layered-record-load", "rvv.layered-record-load",
              info->fieldType.getLayout().getRegisterGroups(), resultGroups,
              temporaryGroups, 0, "none", tail ? "agnostic" : "exact"));
      riscv_internal::copyOrigin(extract, load);
      if (auto canonical = extract->getAttr("canonical_op"))
        load->setAttr("canonical_op", canonical);
      extract.getResult().replaceAllUsesWith(load.getResult());
      deadSubviews.push_back(info->subview);
      rewriter.eraseOp(extract);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(info->extractConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(info->subviewConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
    }
    for (riscv::ExtractOp subview : deadSubviews)
      if (subview && subview.getResult().use_empty())
        rewriter.eraseOp(subview);

    materializeReplicaStorageLoads(rewriter, getOperation());
    riscv_internal::formFragmentWindowLoads(getOperation(), rewriter);

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    llvm::DenseSet<mlir::Operation *> consumed;

    // A complete grouped/layered logical group is represented by one raw byte
    // window per lane-width position and one typed decode per storage layer.
    // The relation is derived from Field storage geometry and the child point's
    // affine coordinate relative to its typed parent point.  Equivalent
    // consumers therefore share the same physical load without relying on
    // source operation adjacency or a kernel-specific closure.
    for (riscv::ExtractOp first : extracts) {
      if (consumed.contains(first))
        continue;
      auto firstInfo = completeLayeredExtract(first);
      if (!firstInfo)
        continue;
      const int64_t group = first.getAccess().getGroupSize();
      const int64_t layer = first.getAccess().getLayerSize();
      auto axisIt = llvm::find(
          firstInfo->resultType.getAxisIds().asArrayRef(),
          firstInfo->point.getResult().getType().getDomain().getAxisId());
      if (axisIt == firstInfo->resultType.getAxisIds().asArrayRef().end())
        continue;
      const size_t axisPosition = static_cast<size_t>(
          axisIt - firstInfo->resultType.getAxisIds().asArrayRef().begin());
      const int64_t lanes =
          firstInfo->resultType.getLayout().getLaneFactors()[axisPosition];
      const int64_t layers = group / layer;
      const int64_t windowsPerLayer = layer / lanes;
      const int64_t slots = layers * windowsPerLayer;
      const int64_t groupBase =
          (firstInfo->relative.constant / group) * group;

      llvm::SmallVector<llvm::SmallVector<CompleteLayeredExtract, 1>, 8>
          bySlot(static_cast<size_t>(slots));
      for (riscv::ExtractOp candidate : extracts) {
        if (consumed.contains(candidate))
          continue;
        auto info = completeLayeredExtract(candidate);
        if (!info || !sameCompleteLayeredGroup(*firstInfo, *info))
          continue;
        const int64_t delta = info->relative.constant - groupBase;
        if (delta < 0 || delta >= group || delta % lanes)
          continue;
        bySlot[static_cast<size_t>(delta / lanes)].push_back(std::move(*info));
      }
      if (llvm::any_of(bySlot, [](const auto &slot) { return slot.empty(); }))
        continue;

      mlir::Operation *insertion = bySlot.front().front().extract;
      bool foldable = true;
      for (auto &slot : bySlot) {
        for (CompleteLayeredExtract &info : slot) {
          if (info.extract->isBeforeInBlock(insertion))
            insertion = info.extract;
          foldable &= canFoldReadConversions(
              bySlot.front().front().extract, info.extract,
              bySlot.front().front().conversions, info.conversions);
        }
      }
      if (!foldable)
        continue;

      CompleteLayeredExtract &leader = bySlot.front().front();
      rewriter.setInsertionPoint(insertion);
      auto groupValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          first.getLoc(), group);
      auto relativeBase = rewriter.create<mlir::arith::SubIOp>(
          first.getLoc(), leader.point.getBase(), leader.parent.getBase());
      auto groupIndex = rewriter.create<mlir::arith::DivUIOp>(
          first.getLoc(), relativeBase, groupValue);
      mlir::Value windowGroup = groupIndex;
      if (windowsPerLayer != 1) {
        auto windowsValue = rewriter.create<mlir::arith::ConstantIndexOp>(
            first.getLoc(), windowsPerLayer);
        windowGroup = rewriter.create<mlir::arith::MulIOp>(
            first.getLoc(), groupIndex, windowsValue);
      }

      auto fieldType = mlir::cast<riscv::ValueType>(
          leader.field.getResult().getType());
      auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(),
                                  leader.point.getResult()
                                      .getType()
                                      .getDomain()
                                      .getAxisId());
      if (fieldAxis == fieldType.getAxisIds().asArrayRef().end())
        continue;
      const int64_t projectionExtent =
          fieldType.getShape()[static_cast<size_t>(
              fieldAxis - fieldType.getAxisIds().asArrayRef().begin())];
      const int64_t rawGroups =
          firstInfo->resultType.getLayout().getRegisterGroups();
      llvm::SmallVector<LayerSelection> layerPlan;
      layerPlan.reserve(layers);
      for (int64_t storageLayer = 0; storageLayer < layers; ++storageLayer) {
        auto layerSelection = selectLayer(
            first.getAccess(), storageLayer, layers,
            riscv_internal::logicalBitWidth(
                leader.resultType.getElementType()));
        if (!layerSelection)
          break;
        layerPlan.push_back(*layerSelection);
      }
      if (layerPlan.size() != static_cast<size_t>(layers))
        continue;
      auto storageType = riscv::LayeredWindowType::get(
          rewriter.getContext(), fieldType, leader.resultType,
          leader.point.getResult().getType().getDomain().getAxisId(), layers,
          windowsPerLayer, rawGroups);
      auto storagePlan = riscv_internal::storageWindowPlan(
          rewriter, leader.field,
          leader.point.getResult().getType().getDomain().getAxisId(), 0, 1, 1,
          projectionExtent, lanes);
      if (!storagePlan)
        continue;
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 2> decoded(
          static_cast<size_t>(windowsPerLayer));
      for (int64_t window = 0; window < windowsPerLayer; ++window) {
        mlir::Value windowIndex = windowGroup;
        if (window) {
          auto offset = rewriter.create<mlir::arith::ConstantIndexOp>(
              first.getLoc(), window);
          windowIndex = rewriter.create<mlir::arith::AddIOp>(
              first.getLoc(), windowGroup, offset);
        }
        auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
            first.getLoc(), storageType, leader.field.getResult(),
            leader.parent.getResult(), windowIndex, *storagePlan,
            first.getAccess(),
            riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-load",
                "rvv.layered-storage-load", "rvv.layered-storage-load",
                fieldType.getLayout().getRegisterGroups(), rawGroups, rawGroups,
                0, "none",
                leader.resultType.getLayout().getValidity() == "tail"
                    ? "agnostic"
                    : "exact"));
        riscv_internal::copyOrigin(first, storage);
        for (int64_t storageLayer = 0; storageLayer < layers; ++storageLayer) {
          const LayerSelection &selection =
              layerPlan[static_cast<size_t>(storageLayer)];
          const int64_t elementBits = riscv_internal::logicalBitWidth(
              leader.resultType.getElementType());
          const int64_t shiftAmount = selection.physicalLayer * elementBits;
          const int64_t maskValue =
              selection.mask ? (int64_t(1) << elementBits) - 1 : 0;
          llvm::StringRef instruction =
              riscv_internal::layeredStorageDecodeInstruction(shiftAmount,
                                                              maskValue);
          auto value = rewriter.create<riscv::RVVLayeredStorageDecodeOp>(
              first.getLoc(), leader.resultType, storage.getResult(),
              storageLayer, selection.physicalLayer, shiftAmount, maskValue,
              riscv_internal::leaf(
                  rewriter, "rvv", "layered-storage-decode",
                  instruction, instruction,
                  rawGroups,
                  leader.resultType.getLayout().getRegisterGroups(), 1, 0,
                  "none",
                  leader.resultType.getLayout().getValidity() == "tail"
                      ? "agnostic"
                      : "exact"));
          riscv_internal::copyOrigin(first, value);
          decoded[static_cast<size_t>(window)].push_back(value.getResult());
        }
      }

      for (int64_t storageLayer = 0; storageLayer < layers; ++storageLayer) {
        for (int64_t window = 0; window < windowsPerLayer; ++window) {
          const int64_t slot =
              (storageLayer * layer + window * lanes) / lanes;
          for (CompleteLayeredExtract &info :
               bySlot[static_cast<size_t>(slot)]) {
            info.extract.getResult().replaceAllUsesWith(
                decoded[static_cast<size_t>(window)]
                       [static_cast<size_t>(storageLayer)]);
            consumed.insert(info.extract);
          }
        }
      }
      for (auto &slot : bySlot) {
        for (CompleteLayeredExtract &info : slot) {
          rewriter.eraseOp(info.extract);
          for (riscv::ConvertLayoutOp conversion :
               llvm::reverse(info.conversions))
            if (conversion.getResult().use_empty())
              rewriter.eraseOp(conversion);
          if (info.field != leader.field &&
              info.field.getResult().use_empty())
            rewriter.eraseOp(info.field);
        }
      }
    }

    for (riscv::ExtractOp first : extracts) {
      if (consumed.contains(first))
        continue;
      riscv::FieldOp firstField;
      riscv::PhysicalPointOp firstPoint;
      llvm::SmallVector<riscv::ConvertLayoutOp> firstConversions;
      if (!isLayerExtract(first, firstField, firstPoint, &firstConversions))
        continue;
      const int64_t layer = first.getAccess().getLayerSize();

      LinearIndex firstIndex;
      decomposeIndex(firstPoint.getBase(), 1, firstIndex);
      if (!firstIndex.valid)
        continue;

      riscv::ExtractOp second;
      riscv::FieldOp secondField;
      llvm::SmallVector<riscv::ConvertLayoutOp> secondConversions;
      for (riscv::ExtractOp candidate : extracts) {
        if (candidate == first || consumed.contains(candidate) ||
            candidate->getBlock() != first->getBlock() ||
            first->isBeforeInBlock(candidate) == false)
          continue;
        riscv::PhysicalPointOp candidatePoint;
        riscv::FieldOp candidateField;
        llvm::SmallVector<riscv::ConvertLayoutOp> candidateConversions;
        if (!isLayerExtract(candidate, candidateField, candidatePoint,
                            &candidateConversions) ||
            candidate.getResult().getType() != first.getResult().getType() ||
            candidate.getAccess() != first.getAccess() ||
            !sameFieldEdge(candidateField, firstField) ||
            candidatePoint.getParent() != firstPoint.getParent() ||
            candidatePoint.getActive() != firstPoint.getActive() ||
            candidatePoint.getPartition() != firstPoint.getPartition() ||
            candidatePoint.getResult().getType() !=
                firstPoint.getResult().getType())
          continue;
        LinearIndex candidateIndex;
        decomposeIndex(candidatePoint.getBase(), 1, candidateIndex);
        if (!sameTerms(firstIndex, candidateIndex) ||
            candidateIndex.constant - firstIndex.constant != layer)
          continue;
        second = candidate;
        secondField = candidateField;
        secondConversions = std::move(candidateConversions);
        break;
      }
      if (!second)
        continue;
      if (!canFoldReadConversions(first, second, firstConversions,
                                  secondConversions))
        continue;

      auto firstType = mlir::cast<riscv::ValueType>(first.getResult().getType());
      const int64_t groups = firstType.getLayout().getRegisterGroups();
      const bool tail = firstType.getLayout().getValidity() == "tail";
      auto element =
          mlir::cast<mlir::IntegerType>(firstType.getElementType());
      auto firstSelection =
          selectLayer(first.getAccess(), 0, 2, element.getWidth());
      auto secondSelection =
          selectLayer(first.getAccess(), 1, 2, element.getWidth());
      if (!firstSelection || !secondSelection)
        continue;
      llvm::SmallVector<int64_t> physicalLayers{
          firstSelection->physicalLayer, secondSelection->physicalLayer};
      llvm::SmallVector<int64_t> shifts{
          firstSelection->physicalLayer * static_cast<int64_t>(element.getWidth()),
          secondSelection->physicalLayer * static_cast<int64_t>(element.getWidth())};
      llvm::SmallVector<int64_t> masks{
          firstSelection->mask
              ? (int64_t{1} << static_cast<int64_t>(element.getWidth())) - 1
              : 0,
          secondSelection->mask
              ? (int64_t{1} << static_cast<int64_t>(element.getWidth())) - 1
              : 0};
      const int64_t axis =
          firstPoint.getResult().getType().getDomain().getAxisId();
      auto axisPosition = llvm::find(firstType.getAxisIds().asArrayRef(), axis);
      if (axisPosition == firstType.getAxisIds().asArrayRef().end())
        continue;
      const size_t axisOrdinal = static_cast<size_t>(
          axisPosition - firstType.getAxisIds().asArrayRef().begin());
      auto storagePlan = riscv_internal::storageWindowPlan(
          rewriter, firstField, axis, 0, 1, 1, layer,
          firstType.getLayout().getLaneFactors()[axisOrdinal]);
      if (!storagePlan)
        continue;
      rewriter.setInsertionPoint(first);
      auto shared = rewriter.create<riscv::RVVLayeredWindowOp>(
          first.getLoc(),
          mlir::TypeRange{first.getResult().getType(), second.getResult().getType()},
          firstField.getResult(), firstPoint.getResult(), first.getAccess(),
          *storagePlan,
          rewriter.getDenseI64ArrayAttr(physicalLayers),
          rewriter.getDenseI64ArrayAttr(shifts),
          rewriter.getDenseI64ArrayAttr(masks),
          riscv_internal::leaf(rewriter, "rvv", "layered-window",
                               "rvv.layered-window", "rvv.layered-window", 0,
                               groups * 2, groups, 0, "none",
                               tail ? "agnostic" : "exact"));
      if (auto origin = first->getAttr("source_origin"))
        shared->setAttr("source_origin", origin);
      shared->setAttr("canonical_op",
                      rewriter.getStringAttr("weft_kernel.extract"));
      first.getResult().replaceAllUsesWith(shared.getFirst());
      second.getResult().replaceAllUsesWith(shared.getSecond());
      consumed.insert(first);
      consumed.insert(second);
      rewriter.eraseOp(first);
      rewriter.eraseOp(second);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(firstConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      for (riscv::ConvertLayoutOp conversion :
           llvm::reverse(secondConversions))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      if (secondField.getResult().use_empty())
        rewriter.eraseOp(secondField);
    }

    // Partial materialization introduces typed per-issue storage windows after
    // the first sharing pass.  Once mechanical issue unrolling exposes one
    // complete grouped/layered relation, collapse those logical windows to one
    // raw load per physical byte window and retain each layer as an explicit
    // decode result.  Distinct logical offsets are not CSE-equivalent; their
    // affine (raw-window, layer) decomposition is the proof used here.
    materializeSharedLayeredStorageWindows(rewriter, getOperation());

  }
};

class MaterializeRISCVReplicaStorageLoadsPass
    : public mlir::PassWrapper<MaterializeRISCVReplicaStorageLoadsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-replica-storage-loads";
  }
  llvm::StringRef getDescription() const override {
    return "Materialize typed natural and strided field-to-register loads";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    materializeReplicaStorageLoads(rewriter, getOperation());
    reuseNaturalStorageWindows(rewriter, getOperation());
    materializeSegmentPairs(rewriter, getOperation());
    riscv_internal::formFragmentWindowLoads(getOperation(), rewriter);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createShareRISCVLayeredWindowsPass() {
  return std::make_unique<ShareRISCVLayeredWindowsPass>();
}

std::unique_ptr<mlir::Pass>
weft::createMaterializeRISCVReplicaStorageLoadsPass() {
  return std::make_unique<MaterializeRISCVReplicaStorageLoadsPass>();
}
