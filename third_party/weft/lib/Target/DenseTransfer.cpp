#include "DenseTransfer.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

int64_t riscv_internal::lastLocalShapeIdentity(ModuleOp module) {
  int64_t identity = 0;
  module.walk([&](riscv::KernelOp kernel) {
    identity = std::max(identity, int64_t(kernel.getShapeSymbols().size()));
  });
  module.walk([&](riscv::LocalShapeOp shape) { identity = std::max(identity, int64_t(shape.getId())); });
  return identity;
}

FailureOr<DenseTransferWindow> riscv_internal::denseTransferWindow(
    IRRewriter &b, Value memory, ArrayRef<Value> offsets, ArrayRef<int64_t> capacity,
    ArrayRef<Value> active, int64_t &identity, Operation *origin) {
  auto source = dyn_cast<riscv::MemDescType>(memory.getType());
  if (!source || source.getShape().size() != offsets.size() || capacity.size() != offsets.size() ||
      active.size() != offsets.size() ||
      llvm::any_of(capacity, [](int64_t n) { return n < 1; }) ||
      llvm::any_of(offsets, [](Value value) { return !value.getType().isIndex(); }) ||
      llvm::any_of(active, [](Value value) { return !value.getType().isIndex(); }))
    return origin->emitError("dense transfer window requires one bounded coordinate per memory axis"), failure();
  Location loc = origin->getLoc();
  auto constant = [&](int64_t n) -> Value { return arith::ConstantIndexOp::create(b, loc, n); };
  SmallVector<Value> starts, lengths;
  SmallVector<int64_t> shape, extentMarkers;
  for (size_t axis = 0; axis < capacity.size(); ++axis) {
    Value bound = riscv::ExtentOp::create(b, loc, b.getIndexType(), memory, axis);
    // An inactive replica has an empty window at the end of the object. Never
    // form an out-of-object descriptor, even if the eventual load is guarded.
    Value start = arith::MinUIOp::create(b, loc, offsets[axis], bound);
    Value remaining = arith::SubIOp::create(b, loc, bound, start);
    Value count = arith::MinUIOp::create(b, loc, remaining, constant(capacity[axis]));
    count = arith::MinUIOp::create(b, loc, count, active[axis]);
    starts.push_back(start);
    if (auto size = riscv::denseConstantIndex(count); size && *size > 0) {
      shape.push_back(*size);
      extentMarkers.push_back(*size);
      lengths.push_back(constant(*size));
    } else {
      auto local = riscv::LocalShapeOp::create(b, loc, b.getIndexType(), count, ++identity);
      shape.push_back(-identity);
      extentMarkers.push_back(-1);
      lengths.push_back(local);
    }
  }
  SmallVector<Value> arguments(starts);
  for (auto [axis, marker] : llvm::enumerate(extentMarkers))
    if (marker < 0) arguments.push_back(lengths[axis]);
  auto type = riscv::MemDescType::get(b.getContext(), source.getEncoding(), b.getDenseI64ArrayAttr(shape),
      source.getAxisIds(), source.getStrides(), source.getOrigins(), source.getAlignment(), "slice",
      source.getAccess(), source.getAliasSet(), source.getLayoutIdentity(), source.getStorageBits(),
      source.getElements(), source.getInterleaveRows());
  auto view = riscv::SubviewOp::create(b, loc, type, memory, arguments,
      b.getDenseI64ArrayAttr(SmallVector<int64_t>(capacity.size(), -1)), b.getDenseI64ArrayAttr(extentMarkers));
  copyOrigin(origin, view);
  return DenseTransferWindow{view, std::move(lengths)};
}

namespace {
std::optional<int64_t> laneAxis(riscv::ValueType type) {
  std::optional<int64_t> result;
  for (auto [axis, lanes] : llvm::zip(type.getAxisIds().asArrayRef(), type.getLayout().getLaneFactors().asArrayRef()))
    if (lanes > 1) {
      if (result) return std::nullopt;
      result = axis;
    }
  return result;
}
bool fullScalarWindow(const DenseTransferWindow &window, riscv::ValueType type) {
  if (window.extents.size() != type.getShape().size()) return false;
  for (auto [extent, size] : llvm::zip(window.extents, type.getShape().asArrayRef()))
    if (riscv::denseIndexBounds(extent).lower < uint64_t(size)) return false;
  return true;
}
}

FailureOr<Value> riscv_internal::loadDenseWindow(IRRewriter &b,
    const DenseTransferWindow &window, riscv::ValueType numeric, Operation *origin) {
  auto memory = cast<riscv::MemDescType>(window.memory.getType());
  if (numeric.getAxisIds() != memory.getAxisIds())
    return origin->emitError("dense issue read has inconsistent logical axes"), failure();
  Operation *operation;
  if (numeric.getLayout().getCarrier() == "scalar") {
    operation = riscv::ScalarTileLoadOp::create(b, origin->getLoc(), numeric, window.memory, window.extents,
        !fullScalarWindow(window, numeric),
        leaf(b, "transfer", "scalar-tile-load", "scalar.tile-load", "scalar.tile-load", 0, 0, 0, 0, "none", "exact"), nullptr);
  } else {
    auto axis = laneAxis(numeric);
    if (!axis || !riscv::VectorTileGeometry::get(numeric, *axis))
      return origin->emitError("dense issue read requires a selected single-axis vector carrier"), failure();
    operation = riscv::RVVTileLoadOp::create(b, origin->getLoc(), numeric, window.memory, window.extents, *axis,
        denseAccess(b, memory, numeric), leaf(b, "transfer", "tile-load", "rvv.tile-load", "rvv.tile-load",
            0, numeric.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
  }
  copyOrigin(origin, operation);
  return operation->getResult(0);
}

LogicalResult riscv_internal::storeDenseWindow(IRRewriter &b, Value value,
    const DenseTransferWindow &window, Operation *origin) {
  auto numeric = dyn_cast<riscv::ValueType>(value.getType());
  auto memory = cast<riscv::MemDescType>(window.memory.getType());
  if (!numeric || numeric.getAxisIds() != memory.getAxisIds())
    return origin->emitError("dense issue write requires matching numeric and memory axes");
  Operation *operation;
  if (numeric.getLayout().getCarrier() == "scalar") {
    operation = riscv::ScalarTileStoreOp::create(b, origin->getLoc(), value, window.memory, window.extents,
        !fullScalarWindow(window, numeric),
        leaf(b, "transfer", "scalar-tile-store", "scalar.tile-store", "scalar.tile-store", 0, 0, 0, 0, "none", "exact"));
  } else {
    auto axis = laneAxis(numeric);
    if (!axis || !riscv::VectorTileGeometry::get(numeric, *axis))
      return origin->emitError("dense issue write requires a selected single-axis vector carrier");
    operation = riscv::RVVTileStoreOp::create(b, origin->getLoc(), value, window.memory, window.extents, *axis,
        denseAccess(b, memory, numeric), leaf(b, "transfer", "tile-store", "rvv.tile-store", "rvv.tile-store",
            numeric.getLayout().getRegisterGroups(), 0, 0, 0, "none", "exact"));
  }
  copyOrigin(origin, operation);
  return success();
}
