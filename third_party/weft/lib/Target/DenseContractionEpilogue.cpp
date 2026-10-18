#include "DenseContractionEpilogue.h"
#include "RISCVPhysicalSupport.h"
#include "LocalMemory.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

riscv::ValueType riscv_internal::contractionEpiloguePiece(
    Builder &b, Type type, ArrayRef<int64_t> axes, ArrayRef<int64_t> shape,
    int64_t laneAxis, int64_t lanes, riscv::TargetAttr target) {
  auto numeric = dyn_cast<riscv::ValueType>(type);
  if (!numeric) return {};
  auto piece = valueTileType(b, numeric, axes, shape, laneAxis, target, true, lanes);
  if (!piece) return {};
  auto layout = piece.getLayout();
  SmallVector<int64_t> replicas(layout.getReplicaFactors().asArrayRef());
  for (auto [position, count] : llvm::enumerate(layout.getTimeFactors().asArrayRef()))
    replicas[position] *= count;
  auto groups = staticProduct(replicas);
  if (!groups) return {};
  layout = riscv::LayoutAttr::get(b.getContext(), layout.getCarrier(), layout.getAxisIds(),
      b.getDenseI64ArrayAttr(SmallVector<int64_t>(replicas.size(), 1)), layout.getLaneFactors(),
      b.getDenseI64ArrayAttr(replicas), layout.getFragmentFactors(), layout.getLocalFactors(),
      layout.getSew(), layout.getLmulEighths(), layout.getVl(),
      layout.getCarrier() == "rvv" ? *groups * ((layout.getLmulEighths() + 7) / 8) : 0,
      layout.getValidity());
  return riscv::ValueType::get(b.getContext(), piece.getElementType(), piece.getShape(), piece.getAxisIds(), layout);
}

std::optional<int64_t> riscv_internal::contractionEpiloguePeak(
    Builder &b, const ContractionEpilogueGraph &graph, ArrayRef<int64_t> axes,
    ArrayRef<int64_t> shape, int64_t laneAxis, int64_t lanes, riscv::TargetAttr target) {
  DenseMap<Value, Value> aliases;
  auto root = [&](Value value) {
    while (aliases.count(value)) value = aliases.lookup(value);
    return value;
  };
  DenseMap<Value, int64_t> groups;
  auto classify = [&](Value value) -> bool {
    if (!isa<riscv::ValueType>(value.getType())) { groups[value] = 0; return true; }
    auto piece = contractionEpiloguePiece(b, value.getType(), axes, shape, laneAxis, lanes, target);
    if (!piece) return false;
    Type element = piece.getElementType();
    if (!element.isF32() && !element.isF64() && !element.isInteger(1)) return false;
    auto layout = piece.getLayout();
    if (layout.getCarrier() == "scalar") {
      auto count = staticProduct(piece.getShape().asArrayRef());
      if (!count || *count > 8) return false;
      groups[value] = 0;
    } else groups[value] = element.isInteger(1)
        ? *staticProduct(layout.getReplicaFactors().asArrayRef()) : layout.getRegisterGroups();
    return true;
  };
  if (!classify(graph.root)) return std::nullopt;
  for (Value value : graph.inputs) if (!classify(value)) return std::nullopt;
  for (Operation *op : graph.adapters) {
    if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op)) {
      aliases[conversion.getResult()] = root(conversion.getInput());
      continue;
    }
    if (!classify(op->getResult(0))) return std::nullopt;
  }
  DenseMap<Value, unsigned> uses;
  for (Operation *op : graph.adapters)
    if (!isa<riscv::ConvertLayoutOp>(op))
      for (Value operand : op->getOperands()) ++uses[root(operand)];
  ++uses[root(graph.storedValue())];
  llvm::SmallDenseSet<Value> resident;
  resident.insert(graph.root);
  int64_t live = groups.lookup(graph.root), peak = live;
  for (Operation *op : graph.adapters) {
    if (isa<riscv::ConvertLayoutOp>(op)) continue;
    for (Value operand : op->getOperands()) {
      Value identity = root(operand);
      if (resident.insert(identity).second) live += groups.lookup(identity);
    }
    auto piece = contractionEpiloguePiece(b, op->getResult(0).getType(), axes, shape, laneAxis, lanes, target);
    int64_t scratch = 0;
    if (auto unary = dyn_cast<riscv::UnaryOp>(op)) {
      auto count = riscv::unaryTemporaryGroups(unary, piece);
      if (!count) return std::nullopt;
      scratch = *count;
    }
    if (isa<riscv::SelectOp>(op)) scratch = 2 * ((piece.getLayout().getLmulEighths() + 7) / 8) + 1;
    Value result = op->getResult(0);
    live += groups.lookup(result);
    resident.insert(result);
    peak = std::max(peak, live + scratch);
    for (Value operand : op->getOperands()) {
      Value identity = root(operand);
      if (--uses[identity] == 0) { live -= groups.lookup(identity); resident.erase(identity); }
    }
  }
  return peak;
}

namespace {
class EpilogueProgram {
public:
  EpilogueProgram(IRRewriter &b, Operation *origin, const ContractionEpilogueGraph &graph,
      riscv::ContractionEpilogueAttr selected, Value input, ValueRange offsets, ValueRange active,
      const DenseMap<Value, Value> &readViews, int64_t &shapeIdentity)
      : b(b), origin(origin), graph(graph), selected(selected), input(input),
        offsets(offsets), active(active), readViews(readViews), shapeIdentity(shapeIdentity),
        loc(origin->getLoc()), target(origin->getParentOfType<riscv::KernelOp>().getTarget()),
        axes(logicalAxes(graph.root.getType())) {}

  LogicalResult run() {
    if (!graph.store || offsets.size() != axes.size() || active.size() != axes.size())
      return origin->emitError("contraction epilogue lost its output coordinate boundary");
    for (Value value : graph.inputs)
      if (!readViews.lookup(value)) return origin->emitError("contraction epilogue lost a stable read frontier");
    if (selected.getMode() == "scalar") return scalarProgram();
    auto whole = dyn_cast<riscv::ValueType>(input.getType());
    if (!whole) return origin->emitError("vector contraction epilogue requires a complete numeric tile");
    if (whole.getLayout().getCarrier() == "scalar") {
      auto numeric = contractionEpiloguePiece(b, whole, axes, whole.getShape().asArrayRef(),
          selected.getLaneAxis(), selected.getLanes(), target);
      if (!numeric) return origin->emitError("selected epilogue conversion has no legal numeric representation");
      input = riscv::ConvertLayoutOp::create(b, loc, numeric, input,
          layoutConversion(b, whole.getLayout(), numeric.getLayout()), riscv::AccessAttr(), unselectedLeaf(b));
      copyOrigin(origin, input.getDefiningOp());
      whole = numeric;
    }
    if (selected.getMode() == "direct") {
      writeVector(input, window(graph.destination(), offsets, active));
      return success();
    }
    shape.assign(selected.getShape().asArrayRef().begin(), selected.getShape().asArrayRef().end());
    point.assign(shape.size(), 0);
    return vectorPieces(0, whole);
  }

private:
  IRRewriter &b;
  Operation *origin;
  const ContractionEpilogueGraph &graph;
  riscv::ContractionEpilogueAttr selected;
  Value input;
  SmallVector<Value> offsets, active;
  const DenseMap<Value, Value> &readViews;
  int64_t &shapeIdentity;
  Location loc;
  riscv::TargetAttr target;
  ArrayRef<int64_t> axes;
  SmallVector<int64_t> shape, point;

  Value constant(int64_t value) { return arith::ConstantIndexOp::create(b, loc, value); }

  Value window(Value descriptor, ValueRange starts, ValueRange lengths) {
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    SmallVector<int64_t> sizes(starts.size(), -1), dimensions(starts.size()), markers(starts.size(), -1);
    SmallVector<Value> bounds(starts);
    for (auto [position, length] : llvm::enumerate(lengths)) {
      if (auto size = getConstantIntValue(length); size && *size >= 0) {
        dimensions[position] = sizes[position] = *size;
      } else {
        auto local = riscv::LocalShapeOp::create(b, loc, b.getIndexType(), length, ++shapeIdentity);
        dimensions[position] = -shapeIdentity;
        bounds.push_back(local);
      }
    }
    auto sliced = riscv::MemDescType::get(b.getContext(), type.getEncoding(), b.getDenseI64ArrayAttr(dimensions),
        type.getAxisIds(), type.getStrides(), type.getOrigins(), type.getAlignment(), "slice", type.getAccess(),
        type.getAliasSet(), type.getLayoutIdentity(), type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    return riscv::SubviewOp::create(b, loc, sliced, descriptor, bounds,
        b.getDenseI64ArrayAttr(markers), b.getDenseI64ArrayAttr(sizes));
  }

  Value scalarAddress(Value descriptor, ValueRange positions) {
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    auto empty = b.getDenseI64ArrayAttr({});
    auto address = riscv::MemDescType::get(b.getContext(), type.getEncoding(), empty, empty, empty, empty,
        type.getAlignment(), "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(),
        type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    return riscv::SliceOp::create(b, loc, address, descriptor, positions,
        b.getArrayAttr(SmallVector<Attribute>(positions.size(), b.getStringAttr("index"))));
  }

  SmallVector<Value> bounds(Value descriptor) {
    SmallVector<Value> result;
    auto type = cast<riscv::MemDescType>(descriptor.getType());
    for (size_t position = 0; position < type.getShape().size(); ++position)
      result.push_back(riscv::ExtentOp::create(b, loc, b.getIndexType(), descriptor, position));
    return result;
  }

  FailureOr<Value> read(Value original, ValueRange positions, ValueRange lengths, bool scalar) {
    auto type = cast<riscv::ValueType>(original.getType());
    SmallVector<Value> starts, extents;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) return origin->emitError("epilogue read has an axis outside the selected result"), failure();
      size_t output = found - axes.begin();
      bool broadcast = type.getShape()[position] == 1;
      starts.push_back(broadcast ? constant(0) : positions[output]);
      if (!scalar) extents.push_back(broadcast ? constant(1) : lengths[output]);
    }
    Value source = readViews.lookup(original);
    if (scalar) {
      source = scalarAddress(source, starts);
      auto read = riscv::LoadOp::create(b, loc, type.getElementType(), source, Value{},
          denseAccess(b, cast<riscv::MemDescType>(source.getType()), type.getElementType()), unselectedLeaf(b));
      copyOrigin(graph.store, read);
      return read.getResult();
    }
    source = window(source, starts, extents);
    auto numeric = contractionEpiloguePiece(b, type, axes, shape, selected.getLaneAxis(), selected.getLanes(), target);
    if (!numeric) return origin->emitError("selected epilogue read has no exact issue type"), failure();
    Value value;
    if (numeric.getLayout().getCarrier() == "rvv") {
      value = riscv::RVVTileLoadOp::create(b, loc, numeric, source, bounds(source), selected.getLaneAxis(),
          denseAccess(b, cast<riscv::MemDescType>(source.getType()), numeric),
          leaf(b, "transfer", "tile-load", "rvv.tile-load", "rvv.tile-load", 0,
               numeric.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
    } else {
      value = riscv::ScalarTileLoadOp::create(b, loc, numeric, source, bounds(source), true,
          leaf(b, "transfer", "scalar-tile-load", "scalar.tile-load", "scalar.tile-load", 0, 0, 0, 0, "none", "exact"), nullptr);
    }
    copyOrigin(graph.store, value.getDefiningOp());
    return value;
  }

  FailureOr<Value> compute(Value root, ValueRange positions, ValueRange lengths, bool scalar) {
    IRMapping mapping;
    mapping.map(graph.root, root);
    auto supply = [&](Value value) -> LogicalResult {
      if (mapping.contains(value) || !llvm::is_contained(graph.inputs, value)) return success();
      auto loaded = read(value, positions, lengths, scalar);
      if (failed(loaded)) return failure();
      mapping.map(value, *loaded);
      return success();
    };
    for (Operation *adapter : graph.adapters) {
      for (Value operand : adapter->getOperands()) if (failed(supply(operand))) return failure();
      if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(adapter)) {
        mapping.map(conversion.getResult(), mapping.lookup(conversion.getInput()));
        continue;
      }
      Type result = scalar ? logicalElement(adapter->getResult(0).getType())
          : Type(contractionEpiloguePiece(b, adapter->getResult(0).getType(), axes, shape,
                     selected.getLaneAxis(), selected.getLanes(), target));
      if (!result) return origin->emitError("selected epilogue operation has no exact result type"), failure();
      Operation *computed = b.clone(*adapter, mapping);
      computed->getResult(0).setType(result);
      computed->setAttr("leaf", unselectedLeaf(b));
      computed->removeAttr("implementation");
      mapping.map(adapter->getResult(0), computed->getResult(0));
    }
    if (failed(supply(graph.storedValue()))) return failure();
    return mapping.lookup(graph.storedValue());
  }

  void writeVector(Value value, Value destination) {
    auto type = cast<riscv::ValueType>(value.getType());
    auto store = riscv::RVVTileStoreOp::create(b, loc, value, destination, bounds(destination),
        selected.getLaneAxis(), denseAccess(b, cast<riscv::MemDescType>(destination.getType()), type),
        leaf(b, "transfer", "tile-store", "rvv.tile-store", "rvv.tile-store",
             type.getLayout().getRegisterGroups(), 0, 0, 0, "none", "exact"));
    copyOrigin(origin, store);
  }

  LogicalResult vectorPieces(size_t axis, riscv::ValueType whole) {
    if (axis < shape.size()) {
      if (whole.getShape()[axis] % shape[axis])
        return origin->emitError("selected epilogue issue does not partition the complete numeric tile");
      for (int64_t index = 0; index < whole.getShape()[axis]; index += shape[axis]) {
        point[axis] = index;
        if (failed(vectorPieces(axis + 1, whole))) return failure();
      }
      return success();
    }
    SmallVector<Value> starts, lengths;
    for (auto [position, index] : llvm::enumerate(point)) {
      starts.push_back(arith::AddIOp::create(b, loc, offsets[position], constant(index)));
      Value bounded = arith::MinUIOp::create(b, loc, active[position], constant(index));
      Value remaining = arith::SubIOp::create(b, loc, active[position], bounded);
      lengths.push_back(arith::MinUIOp::create(b, loc, remaining, constant(shape[position])));
    }
    auto numeric = contractionEpiloguePiece(b, whole, axes, shape, selected.getLaneAxis(), selected.getLanes(), target);
    if (!numeric) return origin->emitError("selected epilogue lost its numeric piece representation");
    StringRef instruction = whole.getLayout().getLmulEighths() == numeric.getLayout().getLmulEighths()
        ? "rvv.tile-slice" : "rvv.tile-slice.split";
    auto slice = riscv::RVVTileSliceOp::create(b, loc, numeric, input, b.getDenseI64ArrayAttr(point), selected.getLaneAxis(),
        leaf(b, "transfer", "tile-slice", instruction, instruction,
             whole.getLayout().getRegisterGroups(), numeric.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
    copyOrigin(origin, slice);
    auto result = compute(slice, starts, lengths, false);
    if (failed(result)) return failure();
    writeVector(*result, window(graph.destination(), starts, lengths));
    return success();
  }

  LogicalResult scalarProgram() {
    auto numeric = dyn_cast<riscv::ValueType>(input.getType());
    if (numeric && numeric.getLayout().getCarrier() != "scalar") {
      auto type = riscv::ValueType::get(b.getContext(), numeric.getElementType(), numeric.getShape(), numeric.getAxisIds(),
          scalarLayout(b, numeric.getElementType(), numeric.getShape().asArrayRef(), numeric.getAxisIds().asArrayRef()));
      input = riscv::ConvertLayoutOp::create(b, loc, type, input,
          layoutConversion(b, numeric.getLayout(), type.getLayout()), riscv::AccessAttr(), unselectedLeaf(b));
      copyOrigin(origin, input.getDefiningOp());
      numeric = type;
    }
    auto coordinates = numeric ? riscv::RegisterCoordinates::get(numeric.getLayout())
                               : std::optional<riscv::RegisterCoordinates>{};
    if (numeric && !coordinates) return origin->emitError("scalar epilogue has no complete register coordinate map");
    int64_t count = coordinates ? coordinates->parts() : 1;
    for (int64_t part = 0; part < count; ++part) {
      SmallVector<Value> positions, indices;
      SmallVector<int64_t> point = coordinates ? *coordinates->logical(part) : SmallVector<int64_t>{};
      Value valid;
      for (auto [position, coordinate] : llvm::enumerate(point)) {
        indices.push_back(constant(coordinate));
        positions.push_back(arith::AddIOp::create(b, loc, offsets[position], indices.back()));
        Value bounded = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, indices.back(), active[position]);
        valid = valid ? Value(arith::AndIOp::create(b, loc, valid, bounded)) : bounded;
      }
      OpBuilder::InsertionGuard guard(b);
      if (valid) {
        auto branch = scf::IfOp::create(b, loc, valid, false);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
      }
      Value scalar = input;
      if (numeric) {
        scalar = riscv::ExtractOp::create(b, loc, numeric.getElementType(), input, indices,
            b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))),
            unassignedAccess(b), unselectedLeaf(b));
        copyOrigin(origin, scalar.getDefiningOp());
      }
      auto computed = compute(scalar, positions, {}, true);
      if (failed(computed)) return failure();
      auto store = riscv::StoreOp::create(b, loc, *computed, scalarAddress(graph.destination(), positions),
          unassignedAccess(b), unselectedLeaf(b), riscv::FieldStorePlanAttr());
      copyOrigin(origin, store);
    }
    return success();
  }
};
}

LogicalResult riscv_internal::materializeContractionEpilogue(
    IRRewriter &builder, Operation *origin, const ContractionEpilogueGraph &graph,
    riscv::ContractionEpilogueAttr selected, Value input, ValueRange offsets, ValueRange active,
    const DenseMap<Value, Value> &readViews, int64_t &shapeIdentity) {
  return EpilogueProgram(builder, origin, graph, selected, input, offsets, active, readViews, shapeIdentity).run();
}
