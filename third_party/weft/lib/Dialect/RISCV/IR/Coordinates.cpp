#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace weft::riscv;

namespace {

std::optional<int64_t> multiply(int64_t a, int64_t b) {
  if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b)
    return std::nullopt;
  return a * b;
}

std::optional<int64_t> product(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    auto next = multiply(result, value);
    if (!next) return std::nullopt;
    result = *next;
  }
  return result;
}

} // namespace

std::optional<RegisterCoordinates> RegisterCoordinates::get(LayoutAttr layout) {
  if (!layout || (layout.getCarrier() != "scalar" && layout.getCarrier() != "rvv"))
    return std::nullopt;
  const size_t rank = layout.getAxisIds().size();
  for (auto factors : {layout.getTimeFactors(), layout.getLaneFactors(),
                       layout.getReplicaFactors(), layout.getFragmentFactors(),
                       layout.getLocalFactors()})
    if (factors.size() != rank) return std::nullopt;
  for (auto factors : {layout.getFragmentFactors(), layout.getLocalFactors()})
    if (llvm::any_of(factors.asArrayRef(), [](int64_t factor) { return factor != 1; }))
      return std::nullopt;
  llvm::DenseSet<int64_t> axes;
  llvm::SmallVector<int64_t> extents;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (!axes.insert(layout.getAxisIds()[axis]).second) return std::nullopt;
    auto extent = product({layout.getTimeFactors()[axis],
                           layout.getReplicaFactors()[axis],
                           layout.getLaneFactors()[axis]});
    if (!extent) return std::nullopt;
    extents.push_back(*extent);
  }
  auto streams = product(layout.getTimeFactors().asArrayRef());
  auto replicas = product(layout.getReplicaFactors().asArrayRef());
  auto laneTile = product(layout.getLaneFactors().asArrayRef());
  const int64_t group = layout.getReplicaGroup();
  if (!streams || !replicas || !laneTile || (group != 1 && group != 2) ||
      *replicas % group || (layout.getCarrier() == "scalar" && group != 1))
    return std::nullopt;
  if (group > 1) {
    auto axis = replicaGroupAxis(layout);
    if (axis < 0 || layout.getReplicaFactors()[axis] % group || *streams != 1)
      return std::nullopt;
  }
  auto parts = multiply(*streams, *replicas / group);
  auto lanes = multiply(*laneTile, group);
  if (!parts || !lanes || !multiply(*parts, *lanes)) return std::nullopt;
  return RegisterCoordinates(layout, *streams, *parts, *lanes, std::move(extents));
}

std::optional<llvm::SmallVector<int64_t>>
RegisterCoordinates::logical(int64_t part, int64_t lane) const {
  if (part < 0 || part >= partCount || lane < 0 || lane >= laneCount)
    return std::nullopt;
  const int64_t group = layout.getReplicaGroup();
  const int64_t tile = laneCount / group;
  int64_t time = part % streamCount;
  int64_t replica = part / streamCount * group + lane / tile;
  int64_t lanePosition = lane % tile;
  llvm::SmallVector<int64_t> result(axisExtents.size());
  for (size_t axis = result.size(); axis-- > 0;) {
    int64_t t = layout.getTimeFactors()[axis];
    int64_t r = layout.getReplicaFactors()[axis];
    int64_t l = layout.getLaneFactors()[axis];
    result[axis] = ((time % t) * r + replica % r) * l + lanePosition % l;
    time /= t;
    replica /= r;
    lanePosition /= l;
  }
  return result;
}

std::optional<RegisterLocation>
RegisterCoordinates::locate(llvm::ArrayRef<int64_t> coordinates) const {
  if (coordinates.size() != axisExtents.size()) return std::nullopt;
  int64_t time = 0, replica = 0, lane = 0;
  for (size_t axis = 0; axis < coordinates.size(); ++axis) {
    int64_t coordinate = coordinates[axis];
    if (coordinate < 0 || coordinate >= axisExtents[axis]) return std::nullopt;
    int64_t l = layout.getLaneFactors()[axis];
    int64_t r = layout.getReplicaFactors()[axis];
    lane = lane * l + coordinate % l;
    coordinate /= l;
    replica = replica * r + coordinate % r;
    time = time * layout.getTimeFactors()[axis] + coordinate / r;
  }
  int64_t group = layout.getReplicaGroup();
  return RegisterLocation{replica / group * streamCount + time,
                          replica % group * (laneCount / group) + lane};
}

std::optional<int64_t> weft::riscv::projectCarrierPart(
    ValueType source, ValueType result, int64_t resultPart) {
  if (!source || !result) return std::nullopt;
  auto from = RegisterCoordinates::get(source.getLayout());
  auto to = RegisterCoordinates::get(result.getLayout());
  if (!from || !to || resultPart < 0 || resultPart >= to->parts() ||
      to->lanes() > 4096) return std::nullopt;
  llvm::SmallVector<size_t> positions;
  auto resultAxes = result.getAxisIds().asArrayRef();
  for (auto [index, axis] : llvm::enumerate(source.getAxisIds().asArrayRef())) {
    auto found = llvm::find(resultAxes, axis);
    if (found == resultAxes.end()) {
      if (source.getShape()[index] != 1) return std::nullopt;
      positions.push_back(resultAxes.size());
    } else {
      size_t position = found - resultAxes.begin();
      if (source.getShape()[index] != 1 &&
          source.getShape()[index] != result.getShape()[position]) return std::nullopt;
      positions.push_back(position);
    }
  }
  std::optional<int64_t> selected;
  for (int64_t lane = 0; lane < to->lanes(); ++lane) {
    auto coordinates = to->logical(resultPart, lane);
    llvm::SmallVector<int64_t> projected;
    for (auto [index, position] : llvm::enumerate(positions))
      projected.push_back(source.getShape()[index] == 1 ? 0 : (*coordinates)[position]);
    auto location = from->locate(projected);
    if (!location || (selected && *selected != location->part)) return std::nullopt;
    selected = location->part;
  }
  return selected;
}
