#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include <functional>
#include <limits>

using namespace mlir;
using namespace weft;

std::optional<int64_t> riscv_internal::completePointExtent(riscv::PhysicalPointOp point) {
  if (!point) return std::nullopt;
  if (auto active = getConstantIntValue(point.getActive())) return *active;
  auto partition = getConstantIntValue(point.getPartition());
  if (!partition || *partition <= 0) return std::nullopt;
  Value base = point.getBase();
  if (auto add = base.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getConstantIntValue(add.getLhs()), rhs = getConstantIntValue(add.getRhs());
    if (lhs && !*lhs) base = add.getRhs();
    else if (rhs && !*rhs) base = add.getLhs();
  }
  auto argument = dyn_cast<BlockArgument>(base);
  auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
  if (!loop || loop.getInductionVar() != argument || !loop->hasAttr("weft.riscv.level")) return std::nullopt;
  auto lower = getConstantIntValue(loop.getLowerBound()), upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *lower < 0 || *upper < *lower || *step != *partition ||
      (*upper - *lower) % *partition) return std::nullopt;
  auto minimum = point.getActive().getDefiningOp<arith::MinUIOp>();
  if (!minimum) return std::nullopt;
  Value remaining;
  if (minimum.getLhs() == point.getPartition()) remaining = minimum.getRhs();
  else if (minimum.getRhs() == point.getPartition()) remaining = minimum.getLhs();
  auto subtract = remaining ? remaining.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
  auto limit = subtract ? getConstantIntValue(subtract.getLhs()) : std::nullopt;
  if (!limit || *limit != *upper || subtract.getRhs() != base) return std::nullopt;
  return *partition;
}

int64_t riscv_internal::lastLocalIdentity(ModuleOp module) {
  int64_t identity = 0;
  module.walk([&](Operation *operation) {
    if (auto birth = operation->getAttrOfType<IntegerAttr>("birth_id"))
      identity = std::max(identity, birth.getInt());
    for (Value value : operation->getResults())
      if (auto type = dyn_cast<riscv::LocalType>(value.getType()))
        identity = std::max({identity, type.getAliasSet(), type.getBirthId()});
  });
  return identity;
}

riscv::ValueType riscv_internal::localValueType(Builder &builder, riscv::ValueType type) {
  SmallVector<int64_t> ones(type.getShape().size(), 1), factors;
  for (int64_t extent : type.getShape().asArrayRef()) factors.push_back(std::max<int64_t>(1, extent));
  auto unit = builder.getDenseI64ArrayAttr(ones);
  auto layout = riscv::LayoutAttr::get(builder.getContext(), "local", type.getAxisIds(),
      unit, unit, unit, unit, builder.getDenseI64ArrayAttr(factors),
      std::max<int64_t>(8, logicalBitWidth(type)), 0, 1, 0, "full");
  return cast<riscv::ValueType>(withLayout(type, layout));
}

FailureOr<Value> riscv_internal::allocateLocalValue(
    IRRewriter &b, riscv::ValueType type, Operation *anchor,
    int64_t &identity, bool preserveBirth, StringRef schema) {
  if (auto result = anchor->getNumResults() == 1
          ? dyn_cast<riscv::ValueType>(anchor->getResult(0).getType()) : riscv::ValueType();
      result && result.getLayout().getCarrier() == "local" &&
      !riscv::storageAliasesRemainInScope(anchor->getResult(0), anchor->getBlock()))
    return anchor->emitError("numeric storage requires a closed lexical value lifetime"), failure();
  auto extents = valueExtentsAt(b, anchor, type);
  if (failed(extents)) return anchor->emitError("numeric storage has no actual extent binding"), failure();
  const int64_t elementBytes = std::max<int64_t>(1, (logicalBitWidth(type) + 7) / 8);
  auto count = staticProduct(type.getShape().asArrayRef());
  if (!count && llvm::all_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent > 0; }))
    return anchor->emitError("numeric storage element count overflows"), failure();
  int64_t bytes = -1;
  if (count && *count > 0) {
    if (*count > std::numeric_limits<int64_t>::max() / elementBytes)
      return anchor->emitError("numeric storage size overflows"), failure();
    bytes = *count * elementBytes;
  }
  if (identity == std::numeric_limits<int64_t>::max())
    return anchor->emitError("numeric storage identity overflows"), failure();
  int64_t owner = 0;
  for (Operation *parent = anchor; parent; parent = parent->getParentOp()) {
    if (auto level = parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level")) {
      owner = level.getDomainId(); break;
    }
    if (auto level = dyn_cast<riscv::LoopOp>(parent)) {
      owner = level.getDomain().getType().getDomainId(); break;
    }
  }
  int64_t allocation = ++identity;
  int64_t birth = allocation, lifetime = owner;
  StringRef purpose = "pack";
  if (preserveBirth) {
    if (auto state = dyn_cast<riscv::NewOp>(anchor)) {
      owner = state.getOwnerDomainId(); birth = state.getBirthId();
      lifetime = state.getLifetimeEndDomainId(); purpose = "handoff"; schema = "local-state-array";
    } else if (auto empty = dyn_cast<riscv::EmptyOp>(anchor)) {
      owner = empty.getOwnerDomainId(); birth = empty.getBirthId();
      lifetime = empty.getLifetimeEndDomainId(); purpose = "handoff"; schema = "local-state-array";
    } else if (auto staged = dyn_cast<riscv::MaterializeOp>(anchor)) {
      owner = staged.getOwnerDomainId(); birth = staged.getBirthId();
      lifetime = staged.getLifetimeEndDomainId(); schema = "staged-value";
    }
  }
  if (auto staged = dyn_cast<riscv::RegisterMaterializeOp>(anchor); staged && preserveBirth) {
    if (staged.getRealization() == "physical-share") schema = "physical-share";
    else {
      owner = staged.getOwnerDomainId();
      birth = staged.getBirthId();
      lifetime = staged.getLifetimeEndDomainId();
      schema = "staged-value";
    }
  }
  auto storage = riscv::LocalType::get(b.getContext(), type.getElementType(), type.getShape(),
      type.getAxisIds(), bytes, std::max<int64_t>(8, elementBytes), allocation,
      purpose, owner, birth, lifetime, schema);
  Location loc = anchor->getLoc();
  Value size = bytes >= 0 ? Value(arith::ConstantIndexOp::create(b, loc, bytes))
      : Value(riscv::LocalSizeOp::create(b, loc, b.getIndexType(), *extents, elementBytes,
            leaf(b, "scalar", "local-size", "local.size.checked", "local.size.checked", 0, 0)));
  Value elements = count && *count >= 0 ? Value(arith::ConstantIndexOp::create(b, loc, *count))
      : Value(arith::DivUIOp::create(b, loc, size, arith::ConstantIndexOp::create(b, loc, elementBytes)));
  Value capacity = bytes >= 0 ? size : Value(arith::MaxUIOp::create(b, loc, size, arith::ConstantIndexOp::create(b, loc, 1)));
  auto object = riscv::LocalAllocOp::create(b, loc, storage, capacity);
  auto binding = riscv::LocalBindOp::create(b, loc, localValueType(b, type), object, elements, *extents);
  copyOrigin(anchor, object);
  copyOrigin(anchor, binding);
  return binding.getResult();
}

LogicalResult riscv_internal::initializeLocalValue(IRRewriter &b, Value destination,
                                                  Value initial, Operation *anchor) {
  auto type = dyn_cast<riscv::ValueType>(destination.getType());
  if (!type || type.getLayout().getCarrier() != "local")
    return anchor->emitError("state initialization requires one explicit private value");
  auto memory = localView(b, destination, anchor);
  if (failed(memory)) return anchor->emitError("state initialization has no actual extent descriptor");
  auto target = anchor->getParentOfType<riscv::KernelOp>().getTarget();
  auto initialType = dyn_cast<riscv::ValueType>(initial.getType());
  if (initialType && (initialType.getElementType() != type.getElementType() ||
      initialType.getShape() != type.getShape() || initialType.getAxisIds() != type.getAxisIds()))
    return anchor->emitError("state initializer changes its declared numerical domain");
  if (initialType && initialType.getLayout().getCarrier() != "local") {
    auto store = riscv::StoreOp::create(b, anchor->getLoc(), initial, completeLocalSlice(b, *memory),
        unassignedAccess(b), unselectedLeaf(b), riscv::FieldStorePlanAttr());
    copyOrigin(anchor, store);
    return success();
  }
  Value input;
  if (initialType) {
    auto view = localView(b, initial, anchor);
    if (failed(view)) return anchor->emitError("state initializer has no complete private source");
    input = *view;
  } else if (initial.getType() != type.getElementType()) {
    return anchor->emitError("scalar initializer must preserve the state element type");
  }
  Type element = type.getElementType();
  bool vector = element.isF32() || element.isF64() ||
      (isa<IntegerType>(element) && !element.isInteger(1));
  int64_t width = vector ? target.getVlenBits() / std::max<int64_t>(8, logicalBitWidth(type)) : 1;
  std::optional<std::pair<int64_t, int64_t>> issue;
  if (!type.getShape().empty() && width > 1) issue = {{type.getAxisIds().asArrayRef().back(), width}};
  auto scalarAddress = [&](Value region) -> Value {
    auto full = cast<riscv::MemDescType>(region.getType());
    auto empty = b.getDenseI64ArrayAttr({});
    auto scalar = riscv::MemDescType::get(b.getContext(), full.getEncoding(), empty, empty, empty, empty,
        full.getAlignment(), "slice", full.getAccess(), full.getAliasSet(), full.getLayoutIdentity(),
        full.getStorageBits(), full.getElements(), full.getInterleaveRows());
    SmallVector<Value> indices(full.getShape().size(), arith::ConstantIndexOp::create(b, anchor->getLoc(), 0));
    return riscv::SliceOp::create(b, anchor->getLoc(), scalar, region, indices,
        b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))));
  };
  return forEachValuePiece(b, type, anchor,
      [&](ArrayRef<Value> coordinates, int64_t axis, int64_t span) -> LogicalResult {
    auto piece = valuePieceType(b, type, axis, span, target);
    if (!piece) return failure();
    auto output = valueMemoryWindow(b, *memory, type, coordinates, piece, anchor);
    if (failed(output)) return failure();
    Value value = initial;
    if (input) {
      auto source = valueMemoryWindow(b, input, type, coordinates, piece, anchor);
      if (failed(source)) return failure();
      value = riscv::LoadOp::create(b, anchor->getLoc(), piece, *source, Value(),
          unassignedAccess(b), unselectedLeaf(b));
    } else if (piece.getLayout().getCarrier() == "rvv") {
      value = riscv::RVVSplatOp::create(b, anchor->getLoc(), piece, initial,
          leaf(b, "rvv", "splat", "rvv.splat", "rvv.splat", 0, piece.getLayout().getRegisterGroups()));
    } else {
      output = scalarAddress(*output);
    }
    auto stored = riscv::StoreOp::create(b, anchor->getLoc(), value, *output,
        unassignedAccess(b), unselectedLeaf(b), riscv::FieldStorePlanAttr());
    copyOrigin(anchor, stored);
    return success();
  }, issue);
}

riscv::ValueType riscv_internal::valuePieceType(Builder &b, riscv::ValueType type,
                                                int64_t axis, int64_t width,
                                                riscv::TargetAttr target, bool tail) {
  return valueTileType(b, type, {axis}, {width}, axis, target, tail);
}

riscv::ValueType riscv_internal::valueTileType(Builder &b, riscv::ValueType type,
    ArrayRef<int64_t> axes, ArrayRef<int64_t> sizes, int64_t axis,
    riscv::TargetAttr target, bool tail, int64_t maximumLaneSpan) {
  if (axes.size() != sizes.size() || llvm::any_of(sizes, [](int64_t n) { return n <= 0; })) return {};
  SmallVector<int64_t> shape(type.getShape().size(), 1), lanes(shape), ones(shape), replicas(shape), times(shape);
  int64_t span = 1;
  for (size_t position = 0; position < shape.size(); ++position) {
    auto found = llvm::find(axes, type.getAxisIds()[position]);
    if (found == axes.end() || type.getShape()[position] == 1) continue;
    int64_t width = sizes[found - axes.begin()];
    shape[position] = width;
    if (type.getAxisIds()[position] == axis) {
      span = maximumLaneSpan > 0 ? std::min(width, maximumLaneSpan) : width;
      if (width % span) return {};
      lanes[position] = span;
      times[position] = width / span;
    }
    else replicas[position] = width;
  }
  int64_t sew = std::max<int64_t>(8, logicalBitWidth(type));
  if (type.getElementType().isInteger(1)) sew = type.getLayout().getSew();
  if (sew <= 0) return {};
  int64_t lmul = 0;
  int64_t elen = 0;
  for (int64_t width : target.getSupportedSEW().asArrayRef()) elen = std::max(elen, width);
  if (elen <= 0) return {};
  if (span > 1)
    for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
      if (legal >= (8 * sew + elen - 1) / elen &&
          target.getVlenBits() * legal / (8 * sew) >= span) { lmul = legal; break; }
  if (span > 1 && !lmul) return {};
  auto replicaCount = staticProduct(replicas);
  if (!replicaCount || *replicaCount > 1024) return {};
  auto unit = b.getDenseI64ArrayAttr(ones);
  auto layout = riscv::LayoutAttr::get(b.getContext(), span > 1 ? "rvv" : "scalar", type.getAxisIds(),
      b.getDenseI64ArrayAttr(times), b.getDenseI64ArrayAttr(lanes), b.getDenseI64ArrayAttr(replicas), unit, unit,
      sew, lmul, span, span > 1 ? *replicaCount * ((lmul + 7) / 8) : 0, tail ? "tail" : "full");
  if (span > 1 && !riscv::supportsRVVLayout(target, layout)) return {};
  return riscv::ValueType::get(b.getContext(), type.getElementType(),
      b.getDenseI64ArrayAttr(shape), type.getAxisIds(), layout);
}

FailureOr<Value> riscv_internal::valueMemoryWindow(
    IRRewriter &b, Value memory, riscv::ValueType domain,
    ArrayRef<Value> coordinates, riscv::ValueType piece, Operation *anchor) {
  auto type = dyn_cast<riscv::MemDescType>(memory.getType());
  if (!type || coordinates.size() != domain.getShape().size() ||
      type.getAxisIds() != piece.getAxisIds() || type.getShape().size() != piece.getShape().size()) return failure();
  SmallVector<Value> offsets;
  for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
    auto found = llvm::find(domain.getAxisIds().asArrayRef(), axis);
    if (found == domain.getAxisIds().asArrayRef().end()) return failure();
    size_t coordinate = found - domain.getAxisIds().asArrayRef().begin();
    if (type.getShape()[position] != 1 && type.getShape()[position] != domain.getShape()[coordinate]) return failure();
    offsets.push_back(type.getShape()[position] == 1
        ? Value(arith::ConstantIndexOp::create(b, anchor->getLoc(), 0)) : coordinates[coordinate]);
  }
  auto region = riscv::MemDescType::get(b.getContext(), type.getEncoding(), piece.getShape(), type.getAxisIds(),
      type.getStrides(), type.getOrigins(), type.getAlignment(), "slice", type.getAccess(),
      type.getAliasSet(), type.getLayoutIdentity(), type.getStorageBits(), type.getElements(), type.getInterleaveRows());
  return riscv::SubviewOp::create(b, anchor->getLoc(), region, memory, offsets,
      b.getDenseI64ArrayAttr(SmallVector<int64_t>(offsets.size(), -1)), piece.getShape()).getResult();
}

LogicalResult riscv_internal::forEachValuePiece(IRRewriter &b, riscv::ValueType type, Operation *anchor,
    function_ref<LogicalResult(ArrayRef<Value>, int64_t, int64_t)> body,
    std::optional<std::pair<int64_t, int64_t>> issue, ArrayRef<Value> suppliedExtents) {
  FailureOr<SmallVector<Value>> extents = suppliedExtents.empty()
      ? valueExtentsAt(b, anchor, type) : FailureOr<SmallVector<Value>>(llvm::to_vector(suppliedExtents));
  if (failed(extents)) return failure();
  if (extents->size() != type.getShape().size() ||
      llvm::any_of(*extents, [](Value value) { return !value.getType().isIndex(); })) return failure();
  int64_t lanePosition = -1, width = 1;
  for (size_t position = 0; position < type.getShape().size(); ++position)
    if (type.getLayout().getLaneFactors()[position] > 1) {
      lanePosition = position;
      width = type.getLayout().getLaneFactors()[position];
    }
  if (lanePosition < 0 && !type.getShape().empty()) lanePosition = type.getShape().size() - 1;
  if (issue) {
    auto axis = llvm::find(type.getAxisIds().asArrayRef(), issue->first);
    if (axis == type.getAxisIds().asArrayRef().end() || issue->second <= 0) return failure();
    lanePosition = axis - type.getAxisIds().asArrayRef().begin();
    width = issue->second;
  }
  SmallVector<Value> coordinates(type.getShape().size());
  SmallVector<size_t> order;
  for (size_t position = 0; position < coordinates.size(); ++position)
    if (int64_t(position) != lanePosition) order.push_back(position);
  if (lanePosition >= 0) order.push_back(lanePosition);
  auto zero = arith::ConstantIndexOp::create(b, anchor->getLoc(), 0);
  auto one = arith::ConstantIndexOp::create(b, anchor->getLoc(), 1);
  std::function<LogicalResult(size_t, int64_t)> emit = [&](size_t depth, int64_t span) -> LogicalResult {
    if (depth == coordinates.size())
      return body(coordinates, lanePosition < 0 ? 0 : type.getAxisIds()[lanePosition], span);
    size_t dimension = order[depth];
    auto issue = [&](Value begin, Value end, Value step, int64_t issueWidth) -> LogicalResult {
      auto loop = scf::ForOp::create(b, anchor->getLoc(), begin, end, step);
      loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
      copyOrigin(anchor, loop);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      coordinates[dimension] = loop.getInductionVar();
      return emit(depth + 1, issueWidth);
    };
    if (int64_t(dimension) != lanePosition || width == 1)
      return issue(zero, (*extents)[dimension], one, span);
    auto step = arith::ConstantIndexOp::create(b, anchor->getLoc(), width);
    auto quotient = arith::DivUIOp::create(b, anchor->getLoc(), (*extents)[dimension], step);
    auto full = arith::MulIOp::create(b, anchor->getLoc(), quotient, step);
    if (failed(issue(zero, full, step, width))) return failure();
    return issue(full, (*extents)[dimension], one, 1);
  };
  return emit(0, 1);
}

FailureOr<Value> riscv_internal::localView(IRRewriter &b, Value source,
                                          Operation *anchor) {
  auto type = dyn_cast<riscv::ValueType>(source.getType());
  if (!type || type.getLayout().getCarrier() != "local" ||
      !isa<IntegerType, FloatType, IndexType>(type.getElementType()))
    return failure();
  FailureOr<SmallVector<Value>> extents = failure();
  if (auto binding = source.getDefiningOp<riscv::LocalBindOp>())
    extents = llvm::to_vector(binding.getExtents());
  else extents = valueExtentsAt(b, anchor, type);
  if (failed(extents) || extents->size() != type.getShape().size()) return failure();
  SmallVector<Value> strides(extents->size());
  SmallVector<int64_t> statics(extents->size()), origins(extents->size(), 0);
  Location loc = anchor->getLoc();
  Value stride = arith::ConstantIndexOp::create(b, loc, 1);
  int64_t staticStride = 1;
  for (size_t axis = extents->size(); axis-- > 0;) {
    strides[axis] = stride;
    statics[axis] = staticStride;
    int64_t extent = type.getShape()[axis];
    if (staticStride && extent > 0) {
      auto next = staticProduct({staticStride, extent});
      if (!next) return failure();
      staticStride = *next;
    } else staticStride = 0;
    stride = staticStride ? Value(arith::ConstantIndexOp::create(b, loc, staticStride))
                          : Value(arith::MulIOp::create(b, loc, stride, (*extents)[axis]));
  }
  Type element = type.getElementType();
  std::string family;
  if (element.isIndex()) family = "i64";
  else if (auto integer = dyn_cast<IntegerType>(element))
    family = (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
  else { llvm::raw_string_ostream out(family); element.print(out); }
  int64_t bits = std::max(8u, logicalBitWidth(type));
  auto encoding = kernel::EncodingType::get(b.getContext(), family, "dense",
      "dense." + family, b.getDenseI64ArrayAttr({}));
  auto descriptor = riscv::MemDescType::get(b.getContext(), encoding, type.getShape(),
      type.getAxisIds(), b.getDenseI64ArrayAttr(statics), b.getDenseI64ArrayAttr(origins),
      bits / 8, "local", "readwrite", -1, "dense." + family, bits, 1, 0);
  auto view = riscv::LocalViewOp::create(b, loc, descriptor, source, *extents, strides);
  copyOrigin(anchor, view);
  return view.getResult();
}

Value riscv_internal::completeLocalSlice(IRRewriter &b, Value view) {
  auto base = cast<riscv::MemDescType>(view.getType());
  auto region = riscv::MemDescType::get(b.getContext(), base.getEncoding(),
      base.getShape(), base.getAxisIds(), base.getStrides(), base.getOrigins(),
      base.getAlignment(), "slice", base.getAccess(), base.getAliasSet(),
      base.getLayoutIdentity(), base.getStorageBits(), base.getElements(), base.getInterleaveRows());
  SmallVector<Attribute> selectors(base.getShape().size(), b.getStringAttr("all"));
  return riscv::SliceOp::create(b, view.getLoc(), region, view, ValueRange{}, b.getArrayAttr(selectors));
}

bool riscv_internal::isCompleteLocalRead(riscv::ConvertLayoutOp conversion) {
  auto input = conversion.getInput().getType(), result = conversion.getResult().getType();
  if (conversion.getConversion().getKind() != "local_load" ||
      input.getLayout().getCarrier() != "local" ||
      !riscv::storageOrigins(conversion.getInput()).isOwnedLocal() ||
      input.getShape() != result.getShape() || input.getAxisIds() != result.getAxisIds() ||
      input.getElementType() != result.getElementType() ||
      result.getLayout().getReplicaGroup() != 1) return false;
  auto coordinates = riscv::RegisterCoordinates::get(result.getLayout());
  if (!coordinates || !isa<IntegerType, FloatType, IndexType>(input.getElementType())) return false;
  for (auto [extent, represented] : llvm::zip(result.getShape().asArrayRef(), coordinates->extents()))
    if (extent == 0 || (extent > 0 && extent != represented)) return false;
  return true;
}
