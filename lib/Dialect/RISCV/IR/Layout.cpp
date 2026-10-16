#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Numerics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"

using namespace weft::riscv;

std::optional<llvm::SmallVector<int64_t>> weft::riscv::rvvRegularGatherIndices(
    ValueType input, ValueType result, llvm::ArrayRef<int64_t> pattern) {
  if (!input || !result || pattern.size() != 3 || pattern[0] < 0 ||
      pattern[1] <= 0 || pattern[2] <= 0 || input.getShape().size() != 1 ||
      result.getShape().size() != 1 || input.getAxisIds() != result.getAxisIds() ||
      input.getElementType() != result.getElementType() ||
      result.getShape()[0] <= 0 || result.getShape()[0] > 4096)
    return std::nullopt;
  auto source = input.getLayout(), destination = result.getLayout();
  auto from = RegisterCoordinates::get(source), to = RegisterCoordinates::get(destination);
  if (!from || !to || from->parts() != 1 ||
      from->lanes() != input.getShape()[0] ||
      source.getCarrier() != "rvv" || destination.getCarrier() != "rvv" ||
      source.getValidity() != "full" || destination.getValidity() != "full" ||
      source.getSew() != destination.getSew() ||
      source.getLmulEighths() != destination.getLmulEighths() ||
      source.getReplicaGroup() != 1 || destination.getReplicaGroup() != 1 ||
      to->extents()[0] != result.getShape()[0])
    return std::nullopt;
  llvm::SmallVector<int64_t> indices;
  for (int64_t part = 0; part < to->parts(); ++part) {
    auto first = to->logical(part, 0), last = to->logical(part, to->lanes() - 1);
    if (!first || !last || (*first)[0] / pattern[2] != (*last)[0] / pattern[2])
      return std::nullopt;
    const __int128 index = pattern[0] +
        static_cast<__int128>((*first)[0] / pattern[2]) * pattern[1];
    if (index < 0 || index >= input.getShape()[0])
      return std::nullopt;
    indices.push_back(static_cast<int64_t>(index));
  }
  return indices;
}

std::optional<llvm::SmallVector<int64_t>> weft::riscv::scalarExtractParts(ExtractOp op) {
  auto input = op.getInput().getType();
  auto output = mlir::dyn_cast<ValueType>(op.getResult().getType());
  auto coordinates = RegisterCoordinates::get(input.getLayout());
  if (!coordinates || input.getLayout().getCarrier() != "scalar" ||
      coordinates->lanes() != 1 ||
      (output && output.getLayout().getCarrier() != "scalar")) return std::nullopt;
  auto result = output ? RegisterCoordinates::get(output.getLayout()) : std::optional<RegisterCoordinates>();
  if (output && (!result || result->lanes() != 1)) return std::nullopt;
  if (!output && op.getResult().getType() != input.getElementType()) return std::nullopt;
  llvm::SmallVector<int64_t> fixed(input.getShape().size(), -1), kept;
  size_t cursor = 0;
  for (size_t axis = 0; axis < input.getShape().size(); ++axis) {
    llvm::StringRef selector = axis < op.getSelectors().size()
        ? mlir::cast<mlir::StringAttr>(op.getSelectors()[axis]).getValue() : "all";
    if (selector == "all") { kept.push_back(input.getAxisIds()[axis]); continue; }
    if (selector != "index" || cursor >= op.getIndices().size()) return std::nullopt;
    auto value = op.getIndices()[cursor++];
    mlir::IntegerAttr constant;
    if (auto index = value.getDefiningOp<mlir::arith::ConstantOp>())
      constant = mlir::dyn_cast<mlir::IntegerAttr>(index.getValue());
    else if (auto index = value.getDefiningOp<ConstantOp>())
      constant = mlir::dyn_cast<mlir::IntegerAttr>(index.getValue());
    if (!constant || constant.getInt() < 0 || constant.getInt() >= coordinates->extents()[axis])
      return std::nullopt;
    fixed[axis] = constant.getInt();
  }
  if (cursor != op.getIndices().size() ||
      (output ? output.getAxisIds().asArrayRef() != llvm::ArrayRef<int64_t>(kept) : !kept.empty()))
    return std::nullopt;
  llvm::SmallVector<int64_t> parts;
  for (int64_t part = 0; part < (result ? result->parts() : 1); ++part) {
    auto point = result ? result->logical(part) : std::optional<llvm::SmallVector<int64_t>>(llvm::SmallVector<int64_t>{});
    if (!point) return std::nullopt;
    auto source = fixed;
    size_t freeAxis = 0;
    for (int64_t &coordinate : source)
      if (coordinate < 0) coordinate = (*point)[freeAxis++];
    auto location = coordinates->locate(source);
    if (!location || location->lane != 0) return std::nullopt;
    parts.push_back(location->part);
  }
  return parts;
}

ValueType weft::riscv::streamAccumulatorType(ValueType window) {
  auto source = window.getLayout();
  auto layout = LayoutAttr::get(window.getContext(), source.getCarrier(),
      source.getAxisIds(), source.getTimeFactors(), source.getLaneFactors(),
      source.getReplicaFactors(), source.getFragmentFactors(),
      source.getLocalFactors(), source.getSew(), source.getLmulEighths(),
      source.getVl(), source.getRegisterGroups(), "full", source.getReplicaGroup());
  return ValueType::get(window.getContext(), window.getElementType(),
                       window.getShape(), window.getAxisIds(), layout);
}

namespace {
std::optional<int64_t> boundedProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || value > 4096 / result)
      return std::nullopt;
    result *= value;
  }
  return result;
}
}

std::optional<llvm::SmallVector<unsigned>> weft::riscv::reshapeAxisPositions(
    ValueType input, ValueType result, llvm::ArrayRef<int64_t> order) {
  auto axes = input.getAxisIds().asArrayRef();
  if (input.getElementType() != result.getElementType() ||
      order.size() != axes.size() || order.size() != result.getShape().size())
    return std::nullopt;
  llvm::SmallVector<unsigned> positions;
  for (auto [position, axis] : llvm::enumerate(order)) {
    auto found = llvm::find(axes, axis);
    if (found == axes.end()) return std::nullopt;
    unsigned source = found - axes.begin();
    if (llvm::is_contained(positions, source) ||
        input.getShape()[source] != result.getShape()[position])
      return std::nullopt;
    positions.push_back(source);
  }
  return positions;
}

bool weft::riscv::hasIdenticalReshapeCarrier(
    ValueType input, ValueType result, llvm::ArrayRef<int64_t> order) {
  auto positions = reshapeAxisPositions(input, result, order);
  if (!positions) return false;
  auto source = input.getLayout(), target = result.getLayout();
  if (source.getCarrier() != target.getCarrier() ||
      (source.getCarrier() != "rvv" && source.getCarrier() != "scalar") ||
      source.getSew() != target.getSew() ||
      source.getLmulEighths() != target.getLmulEighths() ||
      source.getVl() != target.getVl() ||
      source.getRegisterGroups() != target.getRegisterGroups() ||
      source.getReplicaGroup() != target.getReplicaGroup() ||
      source.getValidity() != target.getValidity())
    return false;
  for (auto layout : {source, target})
    for (auto factors : {layout.getFragmentFactors(), layout.getLocalFactors()})
      if (llvm::any_of(factors.asArrayRef(), [](int64_t factor) { return factor != 1; }))
        return false;
  // RVV/scalar parts enumerate replicas outside time; lanes enumerate their
  // own mixed-radix coordinates (and replicaGroup). Preserve each coordinate
  // and each per-axis radix, not only the total number of registers.
  auto sameCoordinates = [&](mlir::DenseI64ArrayAttr before,
                             mlir::DenseI64ArrayAttr after) {
    llvm::SmallVector<unsigned> sourceOrder, targetOrder;
    for (auto [position, factor] : llvm::enumerate(before.asArrayRef()))
      if (factor != 1) sourceOrder.push_back(position);
    for (auto [position, sourceAxis] : llvm::enumerate(*positions)) {
      if (before[sourceAxis] != after[position]) return false;
      if (after[position] != 1) targetOrder.push_back(sourceAxis);
    }
    return sourceOrder == targetOrder;
  };
  return sameCoordinates(source.getTimeFactors(), target.getTimeFactors()) &&
         sameCoordinates(source.getLaneFactors(), target.getLaneFactors()) &&
         sameCoordinates(source.getReplicaFactors(), target.getReplicaFactors()) &&
         sameCoordinates(source.getFragmentFactors(), target.getFragmentFactors()) &&
         sameCoordinates(source.getLocalFactors(), target.getLocalFactors());
}

int64_t weft::riscv::replicaGroupAxis(LayoutAttr layout) {
  if (layout.getReplicaGroup() == 1)
    return -1;
  for (int64_t axis = layout.getAxisIds().size() - 1; axis >= 0; --axis)
    if (layout.getReplicaFactors()[axis] > 1)
      return axis;
  return -1;
}

std::optional<int64_t> weft::riscv::rvvCarrierCount(LayoutAttr layout) {
  auto coordinates = RegisterCoordinates::get(layout);
  return coordinates && coordinates->parts() <= 4096
      ? std::optional<int64_t>(coordinates->parts()) : std::nullopt;
}

std::optional<llvm::SmallVector<int64_t>> weft::riscv::rvvLaneCoordinates(
    LayoutAttr layout, int64_t part, int64_t lane) {
  auto coordinates = RegisterCoordinates::get(layout);
  if (!coordinates || coordinates->parts() > 4096 || coordinates->lanes() > 4096)
    return std::nullopt;
  return coordinates->logical(part, lane);
}

std::optional<llvm::SmallVector<RVVLayoutSlice>>
weft::riscv::rvvLayoutRepackSlices(ValueType input, ValueType result) {
  auto before = input.getLayout(), after = result.getLayout();
  if (input.getElementType() != result.getElementType() ||
      input.getShape() != result.getShape() || input.getAxisIds() != result.getAxisIds() ||
      before.getCarrier() != "rvv" || after.getCarrier() != "rvv" ||
      before.getValidity() != after.getValidity() || before.getSew() != after.getSew())
    return std::nullopt;
  auto sourceMap = RegisterCoordinates::get(before);
  auto resultMap = RegisterCoordinates::get(after);
  if (!sourceMap || !resultMap || sourceMap->extents() != resultMap->extents() ||
      !boundedProduct({sourceMap->parts(), sourceMap->lanes()}) ||
      !boundedProduct({resultMap->parts(), resultMap->lanes()}))
    return std::nullopt;
  for (auto [extent, represented] : llvm::zip(input.getShape().asArrayRef(), sourceMap->extents()))
    if (extent > 0 && extent != represented)
      return std::nullopt;
  llvm::SmallVector<RVVLayoutSlice> slices;
  for (int64_t part = 0; part < resultMap->parts(); ++part) {
    for (int64_t lane = 0; lane < resultMap->lanes(); ++lane) {
      auto logical = resultMap->logical(part, lane);
      auto source = sourceMap->locate(*logical);
      if (!source) return std::nullopt;
      if (!slices.empty()) {
        auto &previous = slices.back();
        if (previous.resultPart == part && previous.sourcePart == source->part &&
            previous.sourceOffset + previous.count == source->lane) {
          ++previous.count;
          continue;
        }
      }
      slices.push_back({source->part, source->lane, part, lane, 1});
    }
  }
  return slices;
}

bool weft::riscv::isRVVSubregisterRepack(TargetAttr target, ValueType input, ValueType result) {
  auto a = input.getLayout(), b = result.getLayout();
  if (!target || a.getLmulEighths() <= b.getLmulEighths() || b.getLmulEighths() < 8 ||
      a.getValidity() != "full" || b.getValidity() != "full")
    return false;
  auto source = RegisterCoordinates::get(a), destination = RegisterCoordinates::get(b);
  const int64_t capacity = target.getVlenBits() * b.getLmulEighths() / (8 * b.getSew());
  auto slices = rvvLayoutRepackSlices(input, result);
  if (!source || !destination || !slices || capacity <= 0 || destination->lanes() != capacity ||
      source->lanes() != target.getVlenBits() * a.getLmulEighths() / (8 * a.getSew()) ||
      slices->size() != static_cast<size_t>(destination->parts()))
    return false;
  return llvm::all_of(*slices, [&](const RVVLayoutSlice &slice) {
    return slice.resultOffset == 0 && slice.count == capacity && slice.sourceOffset % capacity == 0;
  });
}

int64_t weft::riscv::rvvLayoutRepackTemporaryGroups(TargetAttr target, ValueType input, ValueType result) {
  auto a = input.getLayout(), b = result.getLayout();
  auto streams = boundedProduct(b.getTimeFactors());
  if (isRVVSubregisterRepack(target, input, result) || isRVVSubregisterRepack(target, result, input))
    return streams ? *streams * b.getRegisterGroups() : -1;
  return streams ? *streams * b.getRegisterGroups() +
                       (std::max(a.getLmulEighths(), b.getLmulEighths()) + 7) / 8
                 : -1;
}

std::optional<LayoutAttr> weft::riscv::fragmentRVVHandoffLayout(
    FragmentType input, TargetAttr target, int64_t replicaGroup) {
  auto packing = input.getPacking();
  auto layout = input.getLayout();
  if (!target.getHasRVV() || target.getVlenBits() <= 0 ||
      input.getRole() != "accumulator" || input.getAxisIds().size() != 2 ||
      packing.getElementOrder() != "row_major" || packing.getTileOrder() != "row_major" ||
      packing.getAxisOrder().asArrayRef() != llvm::ArrayRef<int64_t>({0, 1}) ||
      layout.getTimeFactors().asArrayRef() != llvm::ArrayRef<int64_t>({1, 1}))
    return std::nullopt;
  auto lanes = boundedProduct(layout.getFragmentFactors());
  if (!lanes || *lanes != packing.getRowsPerTile() * packing.getColumnsPerTile())
    return std::nullopt;
  const int64_t sew = layout.getSew();
  auto replicas = boundedProduct(layout.getReplicaFactors());
  if (replicaGroup != 1 &&
      (replicaGroup != 2 || input.getShape()[0] != packing.getRowsPerTile() ||
       layout.getValidity() == "mask" ||
       !replicas || *replicas > 4 ||
       layout.getReplicaFactors()[1] % replicaGroup ||
       packing.getRowsPerTile() * packing.getColumnsPerTile() * sew !=
           fragmentPartResourceGroups(input) * target.getVlenBits()))
    return std::nullopt;
  const int64_t requested = (*lanes * replicaGroup * sew * 8 + target.getVlenBits() - 1) / target.getVlenBits();
  int64_t lmul = 0;
  int64_t elen = 0;
  for (int64_t supported : target.getSupportedSEW().asArrayRef())
    elen = std::max(elen, supported);
  for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
    if (legal >= requested && legal * elen >= 8 * sew) {
      lmul = legal;
      break;
    }
  if (!lmul || !replicas)
    return std::nullopt;
  auto one = mlir::DenseI64ArrayAttr::get(input.getContext(), {1, 1});
  auto result = LayoutAttr::get(input.getContext(), "rvv", input.getAxisIds(), one,
      layout.getFragmentFactors(), layout.getReplicaFactors(), one, one,
      sew, lmul, *lanes * replicaGroup, (lmul + 7) / 8 * *replicas / replicaGroup,
      layout.getValidity(), replicaGroup);
  return supportsRVVLayout(target, result) ? std::optional<LayoutAttr>(result) : std::nullopt;
}

bool weft::riscv::isDirectFragmentRVVHandoff(FragmentType input, ValueType result) {
  auto a = input.getLayout(), b = result.getLayout();
  return input.getRole() == "accumulator" && input.getAxisIds() == result.getAxisIds() &&
         input.getElementType() == result.getElementType() &&
         input.getPacking().getAxisOrder().asArrayRef() == llvm::ArrayRef<int64_t>({0, 1}) &&
         input.getPacking().getElementOrder() == "row_major" &&
         input.getPacking().getTileOrder() == "row_major" &&
         b.getCarrier() == "rvv" && b.getReplicaGroup() == 1 && a.getTimeFactors() == b.getTimeFactors() &&
         a.getValidity() == b.getValidity() &&
         a.getReplicaFactors() == b.getReplicaFactors() &&
         a.getFragmentFactors() == b.getLaneFactors() &&
         llvm::all_of(b.getFragmentFactors().asArrayRef(), [](int64_t n) { return n == 1; }) &&
         llvm::all_of(b.getLocalFactors().asArrayRef(), [](int64_t n) { return n == 1; }) &&
         b.getSew() == a.getSew() && b.getLmulEighths() <= fragmentPartResourceGroups(input) * 8;
}

bool weft::riscv::isGroupedFragmentRVVHandoff(FragmentType input, ValueType result) {
  auto b = result.getLayout();
  if (b.getReplicaGroup() != 2)
    return false;
  auto plain = LayoutAttr::get(b.getContext(), b.getCarrier(), b.getAxisIds(),
      b.getTimeFactors(), b.getLaneFactors(), b.getReplicaFactors(), b.getFragmentFactors(),
      b.getLocalFactors(), b.getSew(), b.getLmulEighths() / 2, b.getVl() / 2,
      b.getRegisterGroups(), b.getValidity());
  auto ungrouped = ValueType::get(result.getContext(), result.getElementType(), result.getShape(),
                                 result.getAxisIds(), plain);
  return isDirectFragmentRVVHandoff(input, ungrouped) &&
         plain.getLmulEighths() == fragmentPartResourceGroups(input) * 8;
}

int64_t weft::riscv::fragmentUnpackTemporaryGroups(FragmentType input, ValueType result) {
  if (isDirectFragmentRVVHandoff(input, result))
    return 0;
  if (isGroupedFragmentRVVHandoff(input, result))
    return result.getLayout().getRegisterGroups();
  auto streams = boundedProduct(result.getLayout().getTimeFactors());
  const auto *contract = weft::findRISCVFragmentCapability(input.getFamily());
  return streams && contract ? *streams * result.getLayout().getRegisterGroups() +
                                   contract->unpackTemporaryGroups : -1;
}

bool weft::riscv::needsRVVAxisBroadcast(mlir::Type input, mlir::Type result) {
  auto a = mlir::dyn_cast<ValueType>(input), b = mlir::dyn_cast<ValueType>(result);
  if (!a || !b || a.getLayout().getCarrier() != "rvv" || b.getLayout().getCarrier() != "rvv")
    return false;
  auto al = rvvLaneCount(a);
  auto bl = rvvLaneCount(b);
  return al && bl && *al < *bl;
}

std::optional<int64_t> weft::riscv::rvvBroadcastSourcePart(
    ValueType input, ValueType result, int64_t resultPart) {
  auto a = input.getLayout(), b = result.getLayout();
  auto streams = boundedProduct(b.getTimeFactors());
  auto replicas = boundedProduct(b.getReplicaFactors());
  auto sourceStreams = boundedProduct(a.getTimeFactors());
  if (!streams || !replicas || !sourceStreams || a.getCarrier() != "rvv" || b.getCarrier() != "rvv" ||
      input.getElementType() != result.getElementType() || a.getSew() != b.getSew() ||
      a.getLmulEighths() > b.getLmulEighths() || resultPart < 0 ||
      resultPart >= *streams * *replicas / b.getReplicaGroup())
    return std::nullopt;
  llvm::SmallVector<int64_t> times(b.getAxisIds().size()), copies(times.size());
  int64_t t = resultPart % *streams, r = resultPart / *streams * b.getReplicaGroup();
  for (int64_t i = times.size() - 1; i >= 0; --i) {
    times[i] = t % b.getTimeFactors()[i];
    copies[i] = r % b.getReplicaFactors()[i];
    t /= b.getTimeFactors()[i];
    r /= b.getReplicaFactors()[i];
  }
  int64_t sourceTime = 0, sourceReplica = 0;
  for (auto [i, axis] : llvm::enumerate(a.getAxisIds().asArrayRef())) {
    auto found = llvm::find(b.getAxisIds().asArrayRef(), axis);
    if (found == b.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t j = found - b.getAxisIds().asArrayRef().begin();
    if ((a.getReplicaGroup() > 1 || b.getReplicaGroup() > 1) &&
        rvvLaneCount(input) == rvvLaneCount(result) && a.getLaneFactors()[i] > 1) {
      auto sourceStride = boundedProduct(a.getLaneFactors().asArrayRef().drop_front(i + 1));
      auto resultStride = boundedProduct(b.getLaneFactors().asArrayRef().drop_front(j + 1));
      if (!sourceStride || !resultStride || *sourceStride != *resultStride)
        return std::nullopt;
    }
    if (input.getShape()[i] != result.getShape()[j] || a.getTimeFactors()[i] != b.getTimeFactors()[j] ||
        a.getLaneFactors()[i] != b.getLaneFactors()[j] || a.getReplicaFactors()[i] != b.getReplicaFactors()[j] ||
        a.getFragmentFactors()[i] != 1 || b.getFragmentFactors()[j] != 1 ||
        a.getLocalFactors()[i] != 1 || b.getLocalFactors()[j] != 1)
      return std::nullopt;
    sourceTime = sourceTime * a.getTimeFactors()[i] + times[j];
    sourceReplica = sourceReplica * a.getReplicaFactors()[i] + copies[j];
  }
  const int64_t inputGroupAxis = replicaGroupAxis(a), resultGroupAxis = replicaGroupAxis(b);
  if (inputGroupAxis >= 0 && (resultGroupAxis < 0 ||
      a.getAxisIds()[inputGroupAxis] != b.getAxisIds()[resultGroupAxis] ||
      a.getReplicaGroup() != b.getReplicaGroup()))
    return std::nullopt;
  if (resultGroupAxis >= 0 && llvm::is_contained(a.getAxisIds().asArrayRef(), b.getAxisIds()[resultGroupAxis]) &&
      a.getReplicaGroup() != b.getReplicaGroup())
    return std::nullopt;
  return sourceReplica / a.getReplicaGroup() * *sourceStreams + sourceTime;
}

std::optional<int64_t> weft::riscv::binaryBroadcastTemporaryGroups(BinaryOp op) {
  auto result = mlir::dyn_cast<ValueType>(op.getResult().getType());
  if (!result)
    return std::nullopt;
  auto layout = result.getLayout();
  auto streams = boundedProduct(layout.getTimeFactors());
  auto replicas = boundedProduct(layout.getReplicaFactors());
  if (!streams || !replicas || *replicas > 64 / *streams)
    return std::nullopt;
  const int64_t parts = *streams * *replicas / layout.getReplicaGroup();
  const int64_t resultGroups = (layout.getLmulEighths() + 7) / 8;
  const int64_t indexGroups = (layout.getReplicaGroup() > 1 && layout.getSew() == 32)
      ? (layout.getLmulEighths() / 2 + 7) / 8 : resultGroups;
  using Part = std::pair<mlir::Value, int64_t>;
  llvm::DenseMap<Part, int64_t> lastUse;
  llvm::DenseMap<Part, int64_t> sourceGroups;
  int64_t broadcastOperands = 0;
  int64_t indexCarriers = 1, extensionGroups = 0;
  for (mlir::Value operand : op->getOperands()) {
    auto input = mlir::dyn_cast<ValueType>(operand.getType());
    if (!input || input.getLayout().getCarrier() != "rvv")
      return std::nullopt;
    const bool broadcast = needsRVVAxisBroadcast(input, result);
    broadcastOperands += broadcast;
    if (broadcast) {
      if (input.getLayout().getReplicaGroup() > 1 ||
          llvm::count_if(input.getLayout().getLaneFactors().asArrayRef(), [](int64_t n) { return n > 1; }) > 1)
        indexCarriers = 2;
      extensionGroups = std::max(extensionGroups, resultGroups - (input.getLayout().getLmulEighths() + 7) / 8);
    }
    for (int64_t part = 0; part < parts; ++part) {
      auto projected = rvvBroadcastSourcePart(input, result, part);
      if (!projected)
        return std::nullopt;
      Part key{operand, *projected};
      lastUse[key] = part;
      sourceGroups[key] = (input.getLayout().getLmulEighths() + 7) / 8;
    }
  }
  if (!broadcastOperands)
    return 0;
  int64_t freed = 0, peak = 0;
  for (int64_t part = 0; part < parts; ++part) {
    // The fixed local sequence makes at most two gathered operands and one
    // index at once. Its arithmetic result may reuse a dead gathered operand.
    const int64_t scratch = std::max(
        (broadcastOperands - 1) * resultGroups + indexCarriers * indexGroups,
        broadcastOperands * resultGroups + indexGroups + extensionGroups);
    peak = std::max(peak, part * resultGroups + scratch - freed);
    for (auto [key, last] : lastUse)
      if (last == part && hasUniqueRegisterUse(key.first))
        freed += sourceGroups.lookup(key);
    peak = std::max(peak, (part + 1) * resultGroups - freed);
  }
  return peak;
}

std::optional<int64_t> weft::riscv::unaryTemporaryGroups(llvm::StringRef kind, ValueType result) {
  if (kind != "exp" && kind != "exp2" && kind != "log") return 0;
  auto layout = result.getLayout();
  if (layout.getCarrier() == "scalar") return 0;
  if (!result.getElementType().isF32() || layout.getCarrier() != "rvv" ||
      layout.getReplicaGroup() != 1 || layout.getLmulEighths() <= 0) return std::nullopt;
  // Range reduction and the polynomial/scaling sequence need six additional
  // data carriers and two masks. Input/output residency is accounted by the
  // enclosing numerical issue. A mask occupies one architectural register,
  // while each data carrier grows with LMUL; eight constant groups is only
  // the m1 case, not the contract for every selected vector width.
  int64_t groups = (layout.getLmulEighths() + 7) / 8;
  return kind == "log" ? 8 * groups + 3 : 6 * groups + 2;
}

std::optional<int64_t> weft::riscv::unaryTemporaryGroups(UnaryOp operation, ValueType result,
                                                      llvm::StringRef instruction) {
  auto generic = unaryTemporaryGroups(operation.getKind(), result);
  if (!generic || !*generic) return generic;
  if (instruction.empty()) instruction = vectorMathInstruction(operation);
  if (!vectorMathInstructionIsLegal(operation, instruction)) return std::nullopt;
  int64_t groups = (result.getLayout().getLmulEighths() + 7) / 8;
  if (instruction == "rvv.exp-normal-f32") return 4 * groups + 1;
  if (instruction == "rvv.exp-nonpositive-f32") return 6 * groups + 2;
  if (instruction == "rvv.log-normal-f32") return 6 * groups + 2;
  return generic;
}

std::optional<int64_t> weft::riscv::selectTemporaryGroups(SelectOp op) {
  auto result = mlir::dyn_cast<ValueType>(op.getResult().getType());
  if (!result) {
    if (llvm::any_of(op->getOperandTypes(), [](mlir::Type type) {
          return mlir::isa<ValueType>(type);
        }))
      return std::nullopt;
    return 0;
  }
  auto layout = result.getLayout();
  const bool vector = layout.getCarrier() == "rvv";
  if ((!vector && layout.getCarrier() != "scalar") ||
      layout.getReplicaGroup() != 1)
    return std::nullopt;
  int64_t temporaries = 0;
  for (auto [number, operand] : llvm::enumerate(op->getOperands())) {
    auto input = mlir::dyn_cast<ValueType>(operand.getType());
    bool inputVector = input && input.getLayout().getCarrier() == "rvv";
    if (input) {
      auto before = input.getLayout();
      if ((!inputVector && before.getCarrier() != "scalar") ||
          (inputVector && !vector) || before.getReplicaGroup() != 1)
        return std::nullopt;
      for (auto [i, axis] : llvm::enumerate(input.getAxisIds().asArrayRef())) {
        auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
        if (found == result.getAxisIds().asArrayRef().end())
          return std::nullopt;
        size_t j = found - result.getAxisIds().asArrayRef().begin();
        if (input.getShape()[i] != result.getShape()[j] ||
            before.getTimeFactors()[i] != layout.getTimeFactors()[j] ||
            before.getReplicaFactors()[i] != layout.getReplicaFactors()[j] ||
            before.getLaneFactors()[i] != layout.getLaneFactors()[j] ||
            before.getFragmentFactors()[i] != 1 ||
            before.getLocalFactors()[i] != 1)
          return std::nullopt;
        if (before.getLaneFactors()[i] > 1 &&
            boundedProduct(before.getLaneFactors().asArrayRef().drop_front(i + 1)) !=
                boundedProduct(layout.getLaneFactors().asArrayRef().drop_front(j + 1)))
          return std::nullopt;
      }
      if (inputVector &&
          (before.getVl() != layout.getVl() ||
           before.getSew() * layout.getLmulEighths() !=
               layout.getSew() * before.getLmulEighths() ||
           (number != 0 && before.getSew() != layout.getSew())))
        return std::nullopt;
    }
    if (vector)
      temporaries += number == 0 ? (inputVector ? 1 : 0)
                                : (inputVector ? 0 : (layout.getLmulEighths() + 7) / 8);
  }
  return temporaries;
}

std::optional<int64_t> weft::riscv::rvvRowScaleRows(BinaryOp op) {
  auto row = mlir::dyn_cast<ValueType>(op.getLhs().getType());
  auto column = mlir::dyn_cast<ValueType>(op.getRhs().getType());
  auto result = mlir::dyn_cast<ValueType>(op.getResult().getType());
  if (!row || !column || !result || op.getKind() != "mul" ||
      !row.getElementType().isF32() || row.getElementType() != column.getElementType() ||
      row.getElementType() != result.getElementType() || row.getAxisIds().size() != 1 ||
      column.getAxisIds().size() != 1 || result.getAxisIds().size() != 2 ||
      row.getAxisIds()[0] != result.getAxisIds()[0] ||
      column.getAxisIds()[0] != result.getAxisIds()[1] ||
      row.getShape()[0] != result.getShape()[0] || column.getShape()[0] != result.getShape()[1])
    return std::nullopt;
  auto a = row.getLayout(), b = result.getLayout();
  auto lanes = rvvLaneCount(result), parts = rvvCarrierCount(b);
  const int64_t rows = b.getLaneFactors()[0];
  if (a.getCarrier() != "scalar" || a.getLaneFactors()[0] != 1 ||
      a.getReplicaFactors()[0] != rows || a.getTimeFactors()[0] != 1 ||
      a.getFragmentFactors()[0] != 1 || a.getLocalFactors()[0] != 1 ||
      (rows != 2 && rows != 4) || !lanes || *lanes > 32 || !parts ||
      b.getReplicaGroup() != 2 || b.getReplicaFactors()[0] != 1 ||
      b.getTimeFactors()[0] != 1 || !rvvBroadcastSourcePart(column, result, 0))
    return std::nullopt;
  auto kernel = op->getParentOfType<KernelOp>();
  auto c = column.getLayout();
  const int64_t productLMUL = c.getLmulEighths() / b.getReplicaGroup();
  if (!kernel || !kernel.getTarget().getHasRVV() || kernel.getTarget().getSupportedSEW().empty() ||
      c.getLmulEighths() < 8 || b.getLmulEighths() != rows * c.getLmulEighths() ||
      b.getLaneFactors()[1] != kernel.getTarget().getVlenBits() * productLMUL / (8 * 32) ||
      !llvm::is_contained(kernel.getTarget().getLegalLMULEighths().asArrayRef(), productLMUL))
    return std::nullopt;
  const int64_t elen = *llvm::max_element(kernel.getTarget().getSupportedSEW().asArrayRef());
  if (productLMUL * elen < 8 * 32) return std::nullopt;
  return rows;
}

std::optional<int64_t> weft::riscv::rvvRowScaleTemporaryGroups(BinaryOp op) {
  auto rows = rvvRowScaleRows(op);
  if (!rows) return std::nullopt;
  auto result = mlir::cast<ValueType>(op.getResult().getType());
  auto input = mlir::cast<ValueType>(op.getRhs().getType());
  const int64_t groups = (result.getLayout().getLmulEighths() + 7) / 8;
  const int64_t sourceGroups = (input.getLayout().getLmulEighths() + 7) / 8;
  const int64_t atomGroups = (result.getLayout().getLmulEighths() / 2 + 7) / 8;
  const int64_t productGroups = (input.getLayout().getLmulEighths() / 2 + 7) / 8;
  const int64_t parts = *rvvCarrierCount(result.getLayout());
  const int64_t released = hasUniqueRegisterUse(op.getRhs()) ? sourceGroups : 0;
  // Each atom makes narrow row products, packs pairs, then joins the two atoms.
  const int64_t scratch = std::max({*rows * productGroups + 2 * sourceGroups + atomGroups,
      (*rows / 2) * sourceGroups + 2 * atomGroups + sourceGroups, 2 * groups});
  return (parts - 1) * (groups - released) + scratch;
}

std::optional<int64_t> weft::riscv::axisBroadcastInstructionCount(ValueType input, ValueType result) {
  if (!needsRVVAxisBroadcast(input, result)) return 0;
  auto streams = boundedProduct(result.getLayout().getTimeFactors());
  auto replicas = boundedProduct(result.getLayout().getReplicaFactors());
  auto lanes = rvvLaneCount(result);
  if (!streams || !replicas || !lanes || *replicas > 64 / *streams)
    return std::nullopt;
    int64_t perPart = 1, coordinates = 0, sourceStride = 1;
    for (int64_t i = input.getAxisIds().size() - 1; i >= 0; --i) {
      const int64_t factor = input.getLayout().getLaneFactors()[i];
      if (factor == 1)
        continue;
      auto found = llvm::find(result.getAxisIds().asArrayRef(), input.getAxisIds()[i]);
      if (found == result.getAxisIds().asArrayRef().end())
        return std::nullopt;
      const size_t position = found - result.getAxisIds().asArrayRef().begin();
      auto stride = boundedProduct(result.getLayout().getLaneFactors().asArrayRef().drop_front(position + 1));
      if (!stride)
        return std::nullopt;
      perPart += 1 + (*stride > 1) + (*stride * factor < *lanes) + (sourceStride > 1);
      sourceStride *= factor;
      ++coordinates;
    }
    if (!coordinates)
      return std::nullopt;
    perPart += coordinates - 1;
    if (input.getLayout().getReplicaGroup() > 1)
      perPart += 4; // vid, grouped-replica division, source stride, coordinate add.
    int64_t count = 0;
    for (int64_t part = 0; part < *streams * *replicas / result.getLayout().getReplicaGroup(); ++part) {
      auto source = rvvBroadcastSourcePart(input, result, part);
      if (!source)
        return std::nullopt;
      count += perPart;
    }
  return count;
}

std::optional<int64_t> weft::riscv::binaryBroadcastInstructionCount(BinaryOp op) {
  auto result = mlir::dyn_cast<ValueType>(op.getResult().getType());
  if (!result) return std::nullopt;
  int64_t count = 0;
  for (mlir::Value operand : op->getOperands()) {
    auto input = mlir::dyn_cast<ValueType>(operand.getType());
    if (!input || !needsRVVAxisBroadcast(input, result)) continue;
    auto work = axisBroadcastInstructionCount(input, result);
    if (!work) return std::nullopt;
    count += *work;
  }
  return count;
}

std::optional<int64_t> weft::riscv::axisBroadcastTemporaryGroups(RVVAxisBroadcastOp op) {
  auto input = op.getInput().getType(), result = op.getResult().getType();
  auto streams = boundedProduct(result.getLayout().getTimeFactors());
  auto replicas = boundedProduct(result.getLayout().getReplicaFactors());
  if (!streams || !replicas || *replicas > 64 / *streams ||
      !rvvBroadcastSourcePart(input, result, 0))
    return std::nullopt;
  if (!needsRVVAxisBroadcast(input, result))
    return 0;
  int64_t indexCarriers = input.getLayout().getReplicaGroup() > 1 ||
      llvm::count_if(input.getLayout().getLaneFactors().asArrayRef(),
                    [](int64_t n) { return n > 1; }) > 1 ? 2 : 1;
  const int64_t resultGroups = (result.getLayout().getLmulEighths() + 7) / 8;
  const int64_t inputGroups = (input.getLayout().getLmulEighths() + 7) / 8;
  const int64_t indexGroups = result.getLayout().getReplicaGroup() > 1 && result.getLayout().getSew() == 32
      ? (result.getLayout().getLmulEighths() / 2 + 7) / 8 : resultGroups;
  llvm::DenseMap<int64_t, int64_t> lastUse;
  const int64_t parts = *streams * *replicas / result.getLayout().getReplicaGroup();
  for (int64_t part = 0; part < parts; ++part) {
    auto source = rvvBroadcastSourcePart(input, result, part);
    if (!source) return std::nullopt;
    lastUse[*source] = part;
  }
  int64_t peak = 0, freed = 0;
  for (int64_t part = 0; part < parts; ++part) {
    // vrgather's destination cannot overlap its source/index. Earlier output
    // parts also remain live until this standalone op has finished.
    const int64_t scratch = std::max(indexCarriers * indexGroups,
        resultGroups + indexGroups + std::max<int64_t>(0, resultGroups - inputGroups));
    peak = std::max(peak, part * resultGroups + scratch - freed);
    for (auto [source, last] : lastUse)
      if (last == part && hasUniqueRegisterUse(op.getInput()))
        freed += inputGroups;
  }
  return peak;
}

mlir::Value weft::riscv::registerAliasSource(mlir::Value value) {
  if (auto unpack = value.getDefiningOp<IMEUnpackOp>();
      unpack && isDirectFragmentRVVHandoff(unpack.getInput().getType(), unpack.getResult().getType()) &&
      unpack.getResult().getType().getLayout().getLmulEighths() ==
          fragmentPartResourceGroups(unpack.getInput().getType()) * 8)
    return unpack.getInput();
  if (auto materialize = value.getDefiningOp<RegisterMaterializeOp>();
      materialize && materialize.getRealization() == "share" &&
      materialize.getInput().getType() == value.getType() &&
      !fieldReadProjection(materialize.getInput()))
    return materialize.getInput();
  if (auto slice = value.getDefiningOp<RVVTileSliceOp>())
    return slice.getInput();
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return reshape.getInput();
  if (auto conversion = value.getDefiningOp<ConvertLayoutOp>();
      conversion && conversion.getConversion().getEffect() == "pure" &&
      (conversion.getConversion().getKind() == "tuple" ||
       conversion.getConversion().getKind() == "reshape") &&
      !fieldReadProjection(conversion.getInput())) {
    auto input = conversion.getInput().getType();
    auto result = conversion.getResult().getType();
    auto before = input.getLayout(), after = result.getLayout();
    if (before.getCarrier() != "rvv" || after.getCarrier() != "rvv" ||
        before.getSew() != after.getSew() ||
        before.getLmulEighths() != after.getLmulEighths() ||
        before.getVl() != after.getVl() ||
        before.getValidity() != after.getValidity() ||
        before.getReplicaGroup() != after.getReplicaGroup())
      return {};
    auto parts = rvvCarrierCount(before), resultParts = rvvCarrierCount(after);
    auto lanes = boundedProduct(after.getLaneFactors());
    auto fullLanes = lanes ? boundedProduct({*lanes, after.getReplicaGroup()})
                          : std::nullopt;
    auto slices = rvvLayoutRepackSlices(input, result);
    if (!parts || parts != resultParts || !fullLanes || !slices ||
        slices->size() != static_cast<size_t>(*parts))
      return {};
    for (auto [part, slice] : llvm::enumerate(*slices))
      if (slice.sourcePart != static_cast<int64_t>(part) ||
          slice.resultPart != static_cast<int64_t>(part) ||
          slice.sourceOffset != 0 || slice.resultOffset != 0 ||
          slice.count != *fullLanes)
        return {};
    return conversion.getInput();
  }
  return {};
}

bool weft::riscv::hasUniqueRegisterUse(mlir::Value value) {
  while (value.hasOneUse()) {
    auto source = registerAliasSource(value);
    if (!source)
      return true;
    value = source;
  }
  return false;
}
