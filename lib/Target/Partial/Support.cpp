#include "Support.h"

namespace weft::riscv_partial {

int64_t product(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values)
    result *= value;
  return result;
}

std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

bool reachesReductionThroughPureOps(mlir::Value root) {
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (mlir::Operation *consumer : value.getUsers()) {
      if (mlir::isa<riscv::ReduceOp>(consumer))
        return true;
      if (consumer->getNumRegions() == 0 && consumer->getNumResults() == 1 &&
          mlir::isMemoryEffectFree(consumer))
        worklist.push_back(consumer->getResult(0));
    }
  }
  return false;
}

bool checkedSubtract(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedScale(int64_t value, int64_t factor, int64_t &result) {
  if (factor < 0 ||
      (factor > 0 &&
       (value > std::numeric_limits<int64_t>::max() / factor ||
        value < std::numeric_limits<int64_t>::min() / factor)))
    return false;
  result = value * factor;
  return true;
}

bool hasAxis(riscv::ValueType value, int64_t axis) {
  return llvm::is_contained(value.getAxisIds().asArrayRef(), axis);
}

bool hasTopology(riscv::RVVWidenDotOp dot, llvm::StringRef kind) {
  return dot.getPartialTopology().getKind() == kind;
}

int64_t ownerDomain(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation; parent;
       parent = parent->getParentOp()) {
    if (auto level =
            parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      return level.getDomainId();
  }
  return 0;
}

bool mayMoveReadAcross(mlir::Operation *operation) {
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &instance) {
    return mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
  });
}

void sinkReplicaSupplies(riscv::RVVPartialSetOp partial) {
  llvm::SmallVector<riscv::RVVReplicaStorageLoadOp> supplies;
  for (mlir::Value operand : partial->getOperands()) {
    auto supply = operand.getDefiningOp<riscv::RVVReplicaStorageLoadOp>();
    if (!supply || !supply.getResult().hasOneUse() ||
        supply->getBlock() != partial->getBlock() ||
        llvm::is_contained(supplies, supply))
      continue;
    bool movable = true;
    for (mlir::Operation *cursor = supply->getNextNode();
         cursor && cursor != partial.getOperation(); cursor = cursor->getNextNode())
      if (!mayMoveReadAcross(cursor)) {
        movable = false;
        break;
      }
    if (movable)
      supplies.push_back(supply);
  }
  for (riscv::RVVReplicaStorageLoadOp supply : supplies)
    supply->moveBefore(partial);
}

void placePureSliceAtFirstPostLoopUse(mlir::scf::ForOp loop,
                                      mlir::Value root) {
  mlir::Block *block = loop->getBlock();
  if (!block)
    return;
  llvm::SmallPtrSet<mlir::Operation *, 32> slice;
  llvm::SmallVector<mlir::Value> worklist{root};
  while (!worklist.empty()) {
    mlir::Operation *definition = worklist.pop_back_val().getDefiningOp();
    if (!definition || definition->getBlock() != block ||
        !definition->isBeforeInBlock(loop) || definition->getNumRegions() != 0 ||
        definition->getNumResults() != 1 ||
        !mlir::isMemoryEffectFree(definition) ||
        !mlir::isa<riscv::ValueType>(definition->getResult(0).getType()) ||
        !slice.insert(definition).second)
      continue;
    for (mlir::Value operand : definition->getOperands())
      if (mlir::isa<riscv::ValueType>(operand.getType()))
        worklist.push_back(operand);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> rejected;
    for (mlir::Operation *operation : slice) {
      bool movable = !operation->getResult(0).use_empty();
      for (mlir::Operation *user : operation->getResult(0).getUsers())
        movable &= slice.contains(user) ||
                   (user->getBlock() == block && loop->isBeforeInBlock(user));
      if (!movable)
        rejected.push_back(operation);
    }
    for (mlir::Operation *operation : rejected)
      changed |= slice.erase(operation);
  }
  if (slice.empty())
    return;

  mlir::Operation *firstExternalUser = nullptr;
  for (mlir::Operation *operation : slice)
    for (mlir::Operation *user : operation->getResult(0).getUsers())
      if (!slice.contains(user) &&
          (!firstExternalUser || user->isBeforeInBlock(firstExternalUser)))
        firstExternalUser = user;
  if (!firstExternalUser)
    return;

  llvm::SmallVector<mlir::Operation *> ordered;
  for (mlir::Operation &operation : *block) {
    if (&operation == loop.getOperation())
      break;
    if (slice.contains(&operation))
      ordered.push_back(&operation);
  }
  for (mlir::Operation *operation : ordered)
    operation->moveBefore(firstExternalUser);
}

std::optional<size_t> axisPosition(riscv::ValueType value, int64_t axis) {
  auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
  if (found == value.getAxisIds().asArrayRef().end())
    return std::nullopt;
  return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
}

std::optional<int64_t>
laneProductForAxes(riscv::ValueType value, llvm::ArrayRef<int64_t> axes) {
  int64_t result = 1;
  for (int64_t axis : axes) {
    auto position = axisPosition(value, axis);
    if (!position)
      return std::nullopt;
    const int64_t lanes = value.getLayout().getLaneFactors()[*position];
    if (lanes <= 0 || result > std::numeric_limits<int64_t>::max() / lanes)
      return std::nullopt;
    result *= lanes;
  }
  return result;
}

std::optional<int64_t>
primaryReductionLaneAxis(riscv::ValueType partial,
                         llvm::ArrayRef<int64_t> reductionAxes) {
  if (!partial || reductionAxes.empty())
    return std::nullopt;
  std::optional<int64_t> primary;
  for (int64_t axis : reductionAxes) {
    auto position = axisPosition(partial, axis);
    if (!position)
      return std::nullopt;
    if (partial.getLayout().getLaneFactors()[*position] > 1)
      primary = axis;
  }
  // Layout propagation coalesces reduction coordinates in logical order and
  // makes the last lane-bearing axis the physical anchor. A scalar product has
  // no non-unit lane, but still needs one deterministic retained identity.
  return primary ? primary
                 : std::optional<int64_t>(reductionAxes.back());
}

riscv::ValueType projectOneWindow(mlir::Builder &builder, riscv::ValueType value,
                                  int64_t axis) {
  auto position = axisPosition(value, axis);
  if (!position)
    return {};
  llvm::SmallVector<int64_t> shape(value.getShape().asArrayRef());
  llvm::SmallVector<int64_t> time(value.getLayout().getTimeFactors().asArrayRef());
  const int64_t lanes = value.getLayout().getLaneFactors()[*position];
  if (lanes <= 1 || time[*position] <= 1)
    return {};
  shape[*position] = lanes;
  time[*position] = 1;
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), value.getLayout().getCarrier(),
      value.getLayout().getAxisIds(), riscv_internal::integers(builder, time),
      value.getLayout().getLaneFactors(), value.getLayout().getReplicaFactors(),
      value.getLayout().getFragmentFactors(), value.getLayout().getLocalFactors(),
      value.getLayout().getSew(), value.getLayout().getLmulEighths(),
      value.getLayout().getVl(), value.getLayout().getRegisterGroups(),
      value.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), value.getElementType(),
                               riscv_internal::integers(builder, shape),
                               value.getAxisIds(), layout);
}

std::optional<ProjectedRoot> projectedRoot(mlir::Value value, int64_t axis) {
  if (auto stream =
          value.getDefiningOp<riscv::RVVProjectedLayeredStreamOp>()) {
    return ProjectedRoot{stream.getResult(),
                         stream.getField().getDefiningOp<riscv::FieldOp>(),
                         stream.getOrigin().getDefiningOp<riscv::PhysicalPointOp>(),
                         stream.getResult().getType(),
                         stream.getAccess(),
                         static_cast<int64_t>(stream.getProjectionBase()),
                         static_cast<int64_t>(stream.getProjectionStride()),
                         static_cast<int64_t>(stream.getProjectionRepeat()),
                         static_cast<int64_t>(stream.getProjectionExtent()),
                         true,
                         stream.getGeometry()};
  }
  if (auto stream = value.getDefiningOp<riscv::RVVLayeredStreamOp>()) {
    auto field = stream.getField().getDefiningOp<riscv::FieldOp>();
    auto type = stream.getResult().getType();
    auto position = axisPosition(type, axis);
    if (!field || !position)
      return std::nullopt;
    return ProjectedRoot{stream.getResult(),
                         field,
                         riscv_internal::originPoint(field.getOwner(), axis),
                         type,
                         stream.getAccess(),
                         0,
                         1,
                         1,
                         type.getShape()[*position],
                         true,
                         stream.getGeometry()};
  }
  if (auto extract = value.getDefiningOp<riscv::ExtractOp>()) {
    auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
    auto type = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
    auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
    if (!field || !type || !position)
      return std::nullopt;
    if (extract.getIndices().size() == 1) {
      auto point = riscv_internal::stripRepresentationConversions(
                       extract.getIndices().front())
                       .getDefiningOp<riscv::PhysicalPointOp>();
      int64_t domainSelectors = 0;
      bool selectorsLegal = true;
      for (mlir::Attribute selector : extract.getSelectors()) {
        llvm::StringRef name =
            mlir::cast<mlir::StringAttr>(selector).getValue();
        if (name == "domain")
          ++domainSelectors;
        else if (name != "all")
          selectorsLegal = false;
      }
      if (point && selectorsLegal && domainSelectors == 1 &&
          point.getResult().getType().getDomain().getAxisId() == axis)
        return ProjectedRoot{
            extract.getResult(), field, point, type, extract.getAccess(), 0, 1,
            1, type.getShape()[*position],
            extract.getAccess().getMapping() == "grouped_layered"};
    }
    auto pattern =
        extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern");
    if (!pattern || pattern.size() != 3)
      return std::nullopt;
    size_t regular = extract.getSelectors().size();
    bool hasRegular = false;
    for (auto [index, selector] : llvm::enumerate(extract.getSelectors())) {
      llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
      if (name == "regular") {
        if (regular != extract.getSelectors().size())
          return std::nullopt;
        regular = index;
        hasRegular = true;
      } else if (name != "all") {
        return std::nullopt;
      }
    }
    if (!hasRegular || regular != *position || !extract.getIndices().empty())
      return std::nullopt;
    return ProjectedRoot{extract.getResult(),
                         field,
                         riscv_internal::originPoint(field.getOwner(), axis),
                         type,
                         extract.getAccess(),
                         pattern[0],
                         pattern[1],
                         pattern[2],
                         type.getShape()[*position],
                         extract.getAccess().getMapping() == "grouped_layered"};
  }
  if (auto field = value.getDefiningOp<riscv::FieldOp>()) {
    auto type = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
    auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
    if (!type || !position)
      return std::nullopt;
    return ProjectedRoot{field.getResult(),
                         field,
                         riscv_internal::originPoint(field.getOwner(), axis),
                         type,
                         field.getAccess(),
                         0,
                         1,
                         1,
                         type.getShape()[*position],
                         field.getAccess().getMapping() == "grouped_layered"};
  }
  return std::nullopt;
}

void collectProjectedRoots(mlir::Value value, int64_t axis,
                           llvm::DenseSet<mlir::Operation *> &visited,
                           llvm::SmallVectorImpl<ProjectedRoot> &roots) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  if (auto root = projectedRoot(value, axis)) {
    roots.push_back(*root);
    return;
  }
  if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                 riscv::LookupOp,
                 riscv::RVVWidenAddOp,
                 riscv::RVVWidenMultiplyOp,
                 riscv::RVVWidenScalarMultiplyOp,
                 riscv::RegisterMaterializeOp>(definition))
    return;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      collectProjectedRoots(operand, axis, visited, roots);
}

bool dependsOnAny(mlir::Value value,
                  const llvm::DenseMap<mlir::Value, mlir::Value> &replacements,
                  llvm::DenseMap<mlir::Value, bool> &cache) {
  if (replacements.contains(value))
    return true;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  bool dependent = false;
  if (mlir::Operation *definition = value.getDefiningOp())
    if (mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                  riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                  riscv::LookupOp,
                  riscv::RVVWidenAddOp,
                  riscv::RVVWidenMultiplyOp,
                  riscv::RVVWidenScalarMultiplyOp,
                  riscv::RegisterMaterializeOp>(definition))
      for (mlir::Value operand : definition->getOperands())
        if (mlir::isa<riscv::ValueType>(operand.getType()))
          dependent |= dependsOnAny(operand, replacements, cache);
  cache[value] = dependent;
  return dependent;
}

bool canProjectWindowSlice(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Value> &replacements, int64_t axis,
    mlir::Builder &builder, llvm::DenseMap<mlir::Value, bool> &dependence,
    llvm::DenseMap<mlir::Value, bool> &cache) {
  if (replacements.contains(value))
    return true;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
  if (!type || !position) {
    cache[value] = true;
    return true;
  }
  if (!dependsOnAny(value, replacements, dependence)) {
    const bool alreadyOneWindow =
        type.getLayout().getTimeFactors()[*position] == 1;
    cache[value] = alreadyOneWindow;
    return alreadyOneWindow;
  }
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition ||
      !mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                 riscv::LookupOp, riscv::RVVWidenAddOp,
                 riscv::RVVWidenMultiplyOp,
                 riscv::RVVWidenScalarMultiplyOp,
                 riscv::RegisterMaterializeOp>(definition) ||
      !projectOneWindow(builder, type, axis)) {
    cache[value] = false;
    return false;
  }
  bool projectable = true;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      projectable &= canProjectWindowSlice(operand, replacements, axis, builder,
                                           dependence, cache);
  cache[value] = projectable;
  return projectable;
}

mlir::FailureOr<mlir::Value> cloneWindowSlice(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Value> &replacements, int64_t axis,
    mlir::IRRewriter &rewriter, llvm::DenseMap<mlir::Value, mlir::Value> &clones,
    llvm::DenseMap<mlir::Value, bool> &dependence) {
  if (auto found = replacements.find(value); found != replacements.end())
    return found->second;
  if (!dependsOnAny(value, replacements, dependence))
    return value;
  if (auto found = clones.find(value); found != clones.end())
    return found->second;
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto resultType = sourceType ? projectOneWindow(rewriter, sourceType, axis)
                               : riscv::ValueType();
  mlir::Operation *definition = value.getDefiningOp();
  if (!resultType || !definition)
    return mlir::failure();

  auto cloneOperand = [&](mlir::Value operand) -> mlir::FailureOr<mlir::Value> {
    if (!mlir::isa<riscv::ValueType>(operand.getType()))
      return operand;
    return cloneWindowSlice(operand, replacements, axis, rewriter, clones,
                            dependence);
  };
  mlir::Value cloned;
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(definition)) {
    auto input = cloneOperand(unary.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::UnaryOp>(unary.getLoc(), resultType, *input,
                                         unary.getKind(), unary.getLeaf())
                 .getResult();
  } else if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(definition)) {
    auto lhs = cloneOperand(binary.getLhs());
    auto rhs = cloneOperand(binary.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::BinaryOp>(binary.getLoc(), resultType, *lhs, *rhs,
                                          binary.getKind(), binary.getLeaf())
                 .getResult();
  } else if (auto cast = mlir::dyn_cast<riscv::CastOp>(definition)) {
    auto input = cloneOperand(cast.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    auto inputType = mlir::dyn_cast<riscv::ValueType>((*input).getType());
    const bool sameWidth =
        inputType &&
        riscv_internal::logicalBitWidth(inputType.getElementType()) ==
            riscv_internal::logicalBitWidth(resultType.getElementType());
    if (sameWidth && inputType.getLayout() != resultType.getLayout()) {
      auto castType = riscv::ValueType::get(
          rewriter.getContext(), resultType.getElementType(),
          inputType.getShape(), inputType.getAxisIds(), inputType.getLayout());
      auto localCast = rewriter.create<riscv::CastOp>(
          cast.getLoc(), castType, *input,
          riscv_internal::unselectedLeaf(rewriter));
      auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
          cast.getLoc(), resultType, localCast.getResult(),
          riscv_internal::layoutConversion(rewriter, inputType.getLayout(),
                                           resultType.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(cast, localCast);
      riscv_internal::copyOrigin(cast, conversion);
      cloned = conversion.getResult();
    } else {
      cloned = rewriter
                   .create<riscv::CastOp>(cast.getLoc(), resultType, *input,
                                          cast.getLeaf())
                   .getResult();
    }
  } else if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(definition)) {
    auto input = cloneOperand(narrow.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::NarrowOp>(narrow.getLoc(), resultType, *input,
                                          narrow.getRounding(),
                                          narrow.getSaturate(), narrow.getLeaf())
                 .getResult();
  } else if (auto widen = mlir::dyn_cast<riscv::WidenOp>(definition)) {
    auto input = cloneOperand(widen.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::WidenOp>(widen.getLoc(), resultType, *input,
                                         widen.getLeaf())
                 .getResult();
  } else if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(definition)) {
    auto indices = cloneOperand(lookup.getIndices());
    if (mlir::failed(indices)) {
      lookup.emitError("issue-window projection could not project lookup indices");
      return mlir::failure();
    }
    cloned = rewriter
                 .create<riscv::LookupOp>(
                     lookup.getLoc(), resultType, lookup.getTable(), *indices,
                     lookup.getBounds(), lookup.getAccess(), lookup.getLeaf())
                 .getResult();
  } else if (auto add = mlir::dyn_cast<riscv::RVVWidenAddOp>(definition)) {
    auto lhs = cloneOperand(add.getLhs());
    auto rhs = cloneOperand(add.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVWidenAddOp>(add.getLoc(), resultType,
                                              *lhs, *rhs, add.getLeaf())
                 .getResult();
  } else if (auto multiply =
                 mlir::dyn_cast<riscv::RVVWidenMultiplyOp>(definition)) {
    auto lhs = cloneOperand(multiply.getLhs());
    auto rhs = cloneOperand(multiply.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVWidenMultiplyOp>(
                     multiply.getLoc(), resultType, *lhs, *rhs,
                     multiply.getLeaf())
                 .getResult();
  } else if (auto multiply =
                 mlir::dyn_cast<riscv::RVVWidenScalarMultiplyOp>(definition)) {
    auto lhs = cloneOperand(multiply.getLhs());
    if (mlir::failed(lhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVWidenScalarMultiplyOp>(
                     multiply.getLoc(), resultType, *lhs, multiply.getRhs(),
                     multiply.getLeaf())
                 .getResult();
  } else if (auto conversion =
                 mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
    auto input = cloneOperand(conversion.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::ConvertLayoutOp>(
                     conversion.getLoc(), resultType, *input,
                     conversion.getConversion(), conversion.getSourceAccessAttr(),
                     conversion.getLeaf())
                 .getResult();
  } else if (auto materialize =
                 mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition)) {
    auto input = cloneOperand(materialize.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RegisterMaterializeOp>(
                     materialize.getLoc(), resultType, *input,
                     materialize.getOwnerDomainId(), materialize.getBirthId(),
                     materialize.getLifetimeEndDomainId(),
                     materialize.getRealization())
                 .getResult();
  } else {
    return mlir::failure();
  }
  riscv_internal::copyOrigin(definition, cloned.getDefiningOp());
  if (auto implementation =
          definition->getAttrOfType<riscv::ImplementationAttr>("implementation"))
    cloned.getDefiningOp()->setAttr("implementation", implementation);
  clones[value] = cloned;
  return cloned;
}

// Materialize one issue-time window of a shaped axis.  This differs from
// projectOneWindow: an upstream producer may carry a wider lane group than its
// downstream consumer and therefore needs to be sliced to the consumer's
// selected window extent, not merely to its own current lane factor.
riscv::ValueType issueWindowType(mlir::Builder &builder,
                                 riscv::TargetAttr target,
                                 riscv::ValueType source, int64_t axis,
                                 int64_t windowExtent) {
  auto position = axisPosition(source, axis);
  if (!position || windowExtent <= 0 ||
      source.getShape()[*position] < windowExtent ||
      source.getShape()[*position] % windowExtent)
    return {};

  llvm::SmallVector<int64_t> shape(source.getShape().asArrayRef());
  llvm::SmallVector<int64_t> time(
      source.getLayout().getTimeFactors().asArrayRef());
  llvm::SmallVector<int64_t> lane(
      source.getLayout().getLaneFactors().asArrayRef());
  llvm::SmallVector<int64_t> replica(
      source.getLayout().getReplicaFactors().asArrayRef());
  llvm::SmallVector<int64_t> fragment(
      source.getLayout().getFragmentFactors().asArrayRef());
  llvm::SmallVector<int64_t> local(
      source.getLayout().getLocalFactors().asArrayRef());
  if (fragment[*position] != 1 || local[*position] != 1)
    return {};
  shape[*position] = windowExtent;
  time[*position] = 1;

  int64_t lmul = source.getLayout().getLmulEighths();
  int64_t vl = source.getLayout().getVl();
  int64_t groups = source.getLayout().getRegisterGroups();
  if (source.getLayout().getCarrier() == "scalar") {
    lane[*position] = 1;
    replica[*position] = windowExtent;
    lmul = 0;
    vl = 1;
    groups = 0;
  } else if (source.getLayout().getCarrier() == "rvv") {
    const int64_t oldLanes = product(source.getLayout().getLaneFactors());
    if (replica[*position] > 1) {
      if (time[*position] != 1 || lane[*position] != 1 ||
          source.getShape()[*position] != replica[*position] ||
          replica[*position] % windowExtent)
        return {};
      replica[*position] = windowExtent;
    } else {
      lane[*position] = windowExtent;
    }
    const int64_t newLanes = product(lane);
    const int64_t replicas = product(replica);
    if (oldLanes <= 0 || newLanes <= 0 || replicas <= 0 || lmul <= 0 ||
        lmul > std::numeric_limits<int64_t>::max() / newLanes)
      return {};
    const int64_t scaledLmul = lmul * newLanes;
    const int64_t requestedLmul =
        scaledLmul / oldLanes + (scaledLmul % oldLanes != 0);
    int64_t elen = 0;
    for (int64_t supported : target.getSupportedSEW().asArrayRef())
      elen = std::max(elen, supported);
    lmul = 0;
    for (int64_t legal : target.getLegalLMULEighths().asArrayRef()) {
      if (legal >= requestedLmul && elen > 0 &&
          legal * elen >= 8 * source.getLayout().getSew() &&
          (lmul == 0 || legal < lmul))
        lmul = legal;
    }
    if (lmul <= 0)
      return {};
    vl = newLanes;
    groups = product({std::max<int64_t>(1, (lmul + 7) / 8), replicas});
  } else {
    return {};
  }

  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), source.getLayout().getCarrier(),
      source.getLayout().getAxisIds(), riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local), source.getLayout().getSew(),
      lmul, vl, groups, source.getLayout().getValidity());
  if (layout.getCarrier() == "rvv" && !riscv::supportsRVVLayout(target, layout))
    return {};
  return riscv::ValueType::get(
      builder.getContext(), source.getElementType(),
      riscv_internal::integers(builder, shape), source.getAxisIds(), layout);
}

std::optional<IssueStorageWindowCandidate>
analyzeIssueStorageWindowCandidate(mlir::Builder &builder,
                                   riscv::ExtractOp extract) {
  auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  if (!field || !result ||
      extract.getAccess().getMapping() != "grouped_layered" ||
      extract.getIndices().size() != 1)
    return std::nullopt;
  auto point =
      extract.getIndices().front().getDefiningOp<riscv::PhysicalPointOp>();
  if (!point || point.getResult().getType().getDomain().getTail() != "exact")
    return std::nullopt;
  const int64_t axis = point.getResult().getType().getDomain().getAxisId();
  auto position = axisPosition(result, axis);
  if (!position)
    return std::nullopt;
  const int64_t lanes = result.getLayout().getLaneFactors()[*position];
  const int64_t layer = field.getAccess().getLayerSize();
  if (lanes <= 1 || layer <= 0 || layer % lanes ||
      result.getShape()[*position] !=
          lanes * result.getLayout().getTimeFactors()[*position])
    return std::nullopt;
  auto plan = riscv_internal::storageWindowPlan(
      builder, field, axis, 0, 1, 1, result.getShape()[*position],
      result.getLayout().getLaneFactors()[*position]);
  if (!plan)
    return std::nullopt;
  return IssueStorageWindowCandidate{field, point, result, *plan, axis};
}

IssueStorageSupplyFacts analyzeIssueStorageSupply(
    mlir::Builder &builder, mlir::Value value, int64_t axis,
    int64_t windowExtent, riscv::TargetAttr target,
    llvm::DenseMap<mlir::Value, IssueStorageSupplyFacts> &memo) {
  if (auto found = memo.find(value); found != memo.end())
    return found->second;
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto position = sourceType ? axisPosition(sourceType, axis)
                             : std::optional<size_t>();
  if (!sourceType || !position)
    return memo[value] = {true, false};
  if (!issueWindowType(builder, target, sourceType, axis, windowExtent))
    return memo[value] = {};
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return memo[value] = {};

  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    auto candidate = analyzeIssueStorageWindowCandidate(builder, extract);
    return memo[value] = {
               static_cast<bool>(candidate) && candidate->axis == axis,
               static_cast<bool>(candidate) && candidate->axis == axis};
  }
  if (auto window = mlir::dyn_cast<riscv::RVVStorageWindowOp>(definition)) {
    auto plan = window.getPlan();
    const int64_t lanes =
        sourceType.getLayout().getLaneFactors()[*position];
    const bool closed = plan && plan.getReductionAxis() == axis && lanes > 0 &&
                        sourceType.getShape()[*position] ==
                            sourceType.getLayout().getTimeFactors()[*position] *
                                lanes;
    return memo[value] = {closed, closed};
  }

  auto analyzeOperand = [&](mlir::Value operand) {
    return analyzeIssueStorageSupply(builder, operand, axis, windowExtent,
                                     target, memo);
  };
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(definition))
    return memo[value] = analyzeOperand(unary.getInput());
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(definition)) {
    IssueStorageSupplyFacts lhs = analyzeOperand(binary.getLhs());
    IssueStorageSupplyFacts rhs = analyzeOperand(binary.getRhs());
    return memo[value] = {lhs.closed && rhs.closed,
                          lhs.hasStorageWindow || rhs.hasStorageWindow};
  }
  if (auto cast = mlir::dyn_cast<riscv::CastOp>(definition))
    return memo[value] = analyzeOperand(cast.getInput());
  if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(definition))
    return memo[value] = analyzeOperand(narrow.getInput());
  if (auto widen = mlir::dyn_cast<riscv::WidenOp>(definition))
    return memo[value] = analyzeOperand(widen.getInput());
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition))
    return memo[value] = analyzeOperand(conversion.getInput());
  return memo[value] = {};
}

bool supportsIssueStorageRematerialization(mlir::Builder &builder,
                                           mlir::Value value, int64_t axis,
                                           int64_t windowExtent,
                                           riscv::TargetAttr target) {
  llvm::DenseMap<mlir::Value, IssueStorageSupplyFacts> memo;
  IssueStorageSupplyFacts facts = analyzeIssueStorageSupply(
      builder, value, axis, windowExtent, target, memo);
  return facts.closed && facts.hasStorageWindow;
}

mlir::FailureOr<mlir::Value> cloneIssueWindow(
    mlir::Value value, int64_t axis, int64_t windowExtent,
    mlir::Value windowIndex, riscv::TargetAttr target,
    mlir::IRRewriter &rewriter,
    llvm::DenseMap<mlir::Value, mlir::Value> &clones) {
  if (auto found = clones.find(value); found != clones.end())
    return found->second;
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto position = sourceType ? axisPosition(sourceType, axis)
                             : std::optional<size_t>();
  if (!sourceType || !position)
    return value;
  auto resultType =
      issueWindowType(rewriter, target, sourceType, axis, windowExtent);
  if (resultType == sourceType) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(windowIndex);
    auto loop = argument
                    ? mlir::dyn_cast<mlir::scf::ForOp>(
                          argument.getOwner()->getParentOp())
                    : mlir::scf::ForOp();
    if (loop && loop.getInductionVar() == windowIndex) {
      auto lower =
          loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto upper =
          loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
      // A single complete window is an identity, including its read value.
      if (lower && upper && step && lower.value() == 0 && upper.value() == 1 &&
          step.value() == 1)
        return value;
    }
  }
  mlir::Operation *definition = value.getDefiningOp();
  if (!resultType || !definition) {
    if (definition)
      definition->emitError("issue-window projection has no legal projected type; source=")
          << sourceType << ", axis=" << axis
          << ", window_extent=" << windowExtent;
    return mlir::failure();
  }

  auto cloneOperand = [&](mlir::Value operand) -> mlir::FailureOr<mlir::Value> {
    if (!mlir::isa<riscv::ValueType>(operand.getType()))
      return operand;
    return cloneIssueWindow(operand, axis, windowExtent, windowIndex, target,
                            rewriter, clones);
  };
  auto remember = [&](mlir::Value cloned) -> mlir::Value {
    riscv_internal::copyOrigin(definition, cloned.getDefiningOp());
    clones[value] = cloned;
    return cloned;
  };

  if (auto iota = mlir::dyn_cast<riscv::IotaOp>(definition)) {
    if (iota.getStart() != 0 ||
        iota.getEnd() - iota.getStart() != sourceType.getShape()[*position])
      return mlir::failure();
    auto local = rewriter.create<riscv::IotaOp>(
        iota.getLoc(), resultType, 0, windowExtent,
        riscv_internal::unselectedLeaf(rewriter));
    auto element = mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType());
    if (!element)
      return mlir::failure();
    auto castIndex = rewriter.create<riscv::CastOp>(
        iota.getLoc(), element, windowIndex,
        riscv_internal::unselectedLeaf(rewriter));
    auto extent = rewriter.create<riscv::ConstantOp>(
        iota.getLoc(), element, rewriter.getIntegerAttr(element, windowExtent));
    auto offset = rewriter.create<riscv::BinaryOp>(
        iota.getLoc(), element, castIndex.getResult(), extent.getResult(), "mul",
        riscv_internal::unselectedLeaf(rewriter));
    auto shifted = rewriter.create<riscv::BinaryOp>(
        iota.getLoc(), resultType, local.getResult(), offset.getResult(), "add",
        riscv_internal::unselectedLeaf(rewriter));
    return remember(shifted.getResult());
  }

  if (auto load =
          mlir::dyn_cast<riscv::RVVUnitEntryWindowLoadOp>(definition)) {
    llvm::SmallVector<int64_t> axes(load.getEntryAxes());
    llvm::SmallVector<int64_t> extents(load.getEntryExtents());
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || axes.size() != extents.size())
      return mlir::failure();
    const size_t entryPosition = static_cast<size_t>(found - axes.begin());
    int64_t suffix = 1;
    for (size_t index = entryPosition + 1; index < extents.size(); ++index) {
      if (extents[index] <= 0 ||
          suffix > std::numeric_limits<int64_t>::max() / extents[index])
        return mlir::failure();
      suffix *= extents[index];
    }
    if (windowExtent > std::numeric_limits<int64_t>::max() / suffix)
      return mlir::failure();
    const int64_t coordinateSpan = windowExtent * suffix;
    extents[entryPosition] = windowExtent;
    mlir::Value base = load.getEntryBase();
    mlir::Value windowBase;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(base.getType())) {
      auto typedIndex = rewriter.create<riscv::CastOp>(
          load.getLoc(), integer, windowIndex,
          riscv_internal::unselectedLeaf(rewriter));
      auto span = rewriter.create<riscv::ConstantOp>(
          load.getLoc(), integer,
          rewriter.getIntegerAttr(integer, coordinateSpan));
      auto offset = rewriter.create<riscv::BinaryOp>(
          load.getLoc(), integer, typedIndex.getResult(), span.getResult(), "mul",
          riscv_internal::unselectedLeaf(rewriter));
      windowBase = rewriter
                       .create<riscv::BinaryOp>(
                           load.getLoc(), integer, base, offset.getResult(),
                           "add", riscv_internal::unselectedLeaf(rewriter))
                       .getResult();
    } else if (base.getType().isIndex()) {
      auto span = rewriter.create<mlir::arith::ConstantIndexOp>(
          load.getLoc(), coordinateSpan);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          load.getLoc(), windowIndex, span.getResult());
      windowBase = rewriter
                       .create<mlir::arith::AddIOp>(load.getLoc(), base,
                                                   offset.getResult())
                       .getResult();
    } else {
      return mlir::failure();
    }
    auto cloned = rewriter.create<riscv::RVVUnitEntryWindowLoadOp>(
        load.getLoc(), resultType, load.getSource(), windowBase,
        load.getSourceAxis(), rewriter.getDenseI64ArrayAttr(axes),
        rewriter.getDenseI64ArrayAttr(extents), load.getPayloadAxis(),
        load.getPayloadExtent(), load.getEntryStride(), load.getAccess(),
        riscv_internal::leaf(
            rewriter, "rvv", "unit-entry-window-load",
            "rvv.unit-entry-window-load", "rvv.unit-entry-window-load", 0,
            resultType.getLayout().getRegisterGroups()));
    return remember(cloned.getResult());
  }

  if (auto load =
          mlir::dyn_cast<riscv::RVVBitmaskWindowLoadOp>(definition)) {
    llvm::SmallVector<int64_t> axes(load.getWindowAxes());
    llvm::SmallVector<int64_t> extents(load.getWindowExtents());
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || axes.size() != extents.size())
      return mlir::failure();
    const size_t windowPosition = static_cast<size_t>(found - axes.begin());
    int64_t suffix = 1;
    for (size_t index = windowPosition + 1; index < extents.size(); ++index) {
      if (extents[index] <= 0 ||
          suffix > std::numeric_limits<int64_t>::max() / extents[index])
        return mlir::failure();
      suffix *= extents[index];
    }
    if (windowExtent > std::numeric_limits<int64_t>::max() / suffix)
      return mlir::failure();
    const int64_t windowBits = windowExtent * suffix;
    if (windowBits <= 0 || windowBits % 8)
      return mlir::failure();
    extents[windowPosition] = windowExtent;

    mlir::Value base = load.getByteBase();
    mlir::Value windowBase;
    const int64_t windowBytes = windowBits / 8;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(base.getType())) {
      auto typedIndex = rewriter.create<riscv::CastOp>(
          load.getLoc(), integer, windowIndex,
          riscv_internal::unselectedLeaf(rewriter));
      auto span = rewriter.create<riscv::ConstantOp>(
          load.getLoc(), integer,
          rewriter.getIntegerAttr(integer, windowBytes));
      auto offset = rewriter.create<riscv::BinaryOp>(
          load.getLoc(), integer, typedIndex.getResult(), span.getResult(),
          "mul", riscv_internal::unselectedLeaf(rewriter));
      windowBase = rewriter
                       .create<riscv::BinaryOp>(
                           load.getLoc(), integer, base, offset.getResult(),
                           "add", riscv_internal::unselectedLeaf(rewriter))
                       .getResult();
    } else if (base.getType().isIndex()) {
      auto span = rewriter.create<mlir::arith::ConstantIndexOp>(
          load.getLoc(), windowBytes);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          load.getLoc(), windowIndex, span.getResult());
      windowBase = rewriter
                       .create<mlir::arith::AddIOp>(load.getLoc(), base,
                                                   offset.getResult())
                       .getResult();
    } else {
      return mlir::failure();
    }

    auto partBitOffsets =
        riscv::bitmaskWindowPartOffsets(resultType, axes, extents);
    if (!partBitOffsets)
      return mlir::failure();
    auto selected = load.getLeaf();
    const int64_t temporaryGroups = std::max<int64_t>(
        1, (resultType.getLayout().getLmulEighths() + 7) / 8);
    auto cloned = rewriter.create<riscv::RVVBitmaskWindowLoadOp>(
        load.getLoc(), resultType, load.getField(), windowBase,
        load.getSourceAxis(), rewriter.getDenseI64ArrayAttr(axes),
        rewriter.getDenseI64ArrayAttr(extents),
        rewriter.getDenseI64ArrayAttr(*partBitOffsets), load.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(),
            selected.getOperandGroups(),
            resultType.getLayout().getRegisterGroups(), temporaryGroups,
            selected.getFragmentGroups(), selected.getMask(),
            selected.getTail(), selected.getParameters(),
            selected.getLocalBytes()));
    return remember(cloned.getResult());
  }

  if (auto window = mlir::dyn_cast<riscv::RVVStorageWindowOp>(definition)) {
    auto sourcePlan = window.getPlan();
    auto resultPosition = axisPosition(resultType, axis);
    if (!resultPosition)
      return mlir::failure();
    const int64_t lanes =
        resultType.getLayout().getLaneFactors()[*resultPosition];
    if (!sourcePlan || sourcePlan.getReductionAxis() != axis || lanes <= 0 ||
        windowExtent != lanes ||
        resultType.getLayout().getTimeFactors()[*resultPosition] != 1)
      return mlir::failure();

    auto extent = rewriter.create<mlir::arith::ConstantIndexOp>(
        window.getLoc(), windowExtent);
    auto dynamicOffset = rewriter.create<mlir::arith::MulIOp>(
        window.getLoc(), windowIndex, extent.getResult());
    mlir::Value logicalOffset = rewriter.create<mlir::arith::AddIOp>(
        window.getLoc(), window.getLogicalOffset(), dynamicOffset.getResult());
    auto projectedPlan = riscv::StorageWindowPlanAttr::get(
        rewriter.getContext(), sourcePlan.getKind(),
        sourcePlan.getReductionAxis(), sourcePlan.getProjectionBase(),
        sourcePlan.getProjectionStride(), sourcePlan.getProjectionRepeat(),
        windowExtent, lanes, sourcePlan.getRecordElements(),
        sourcePlan.getByteOffset(), sourcePlan.getElementBits(),
        sourcePlan.getGroupSize(), sourcePlan.getLayerSize(),
        sourcePlan.getPhysicalLayerBase(), sourcePlan.getPhysicalLayerStep(),
        sourcePlan.getShiftBase(), sourcePlan.getShiftStep(),
        sourcePlan.getMaskValue());
    auto selected = window.getLeaf();
    auto projected = rewriter.create<riscv::RVVStorageWindowOp>(
        window.getLoc(), resultType, window.getField(), window.getOrigin(),
        logicalOffset, projectedPlan, window.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(),
            selected.getOperandGroups(),
            resultType.getLayout().getRegisterGroups(),
            selected.getTemporaryGroups(), selected.getFragmentGroups(),
            selected.getMask(),
            resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                            : "exact",
            selected.getParameters(), selected.getLocalBytes()));
    return remember(projected.getResult());
  }

  if (auto load =
          mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(definition)) {
    auto field = load.getField().getDefiningOp<riscv::FieldOp>();
    auto sourcePlan = load.getPlan();
    auto issueTimes =
        checkedProduct(resultType.getLayout().getTimeFactors().asArrayRef());
    auto issueReplicas =
        checkedProduct(resultType.getLayout().getReplicaFactors().asArrayRef());
    int64_t issueParts = 0;
    if (!issueTimes || !issueReplicas ||
        !checkedScale(*issueTimes, *issueReplicas, issueParts))
      return mlir::failure();
    if (!field || !sourcePlan || sourcePlan.getKind() != "unit") {
      load.emitError("issue-window storage projection requires a concrete unit field; plan=")
          << sourcePlan << ", field=" << load.getField();
      return mlir::failure();
    }

    // A preplanned storage value may be split in issue time along an axis
    // orthogonal to its load-lane axis.  When that split is a translation of
    // one otherwise identical physical window, project it by advancing the
    // scalar base and retaining the selected one-part load.  The storage form
    // and all decode facts remain owned by the original plan.
    if (sourcePlan.getReductionAxis() != axis) {
      auto issuePosition = axisPosition(sourceType, axis);
      auto sourceTimes =
          checkedProduct(sourceType.getLayout().getTimeFactors().asArrayRef());
      auto sourceReplicas = checkedProduct(
          sourceType.getLayout().getReplicaFactors().asArrayRef());
      int64_t sourceParts = 0;
      int64_t representedIssueExtent = 0;
      const int64_t issueCount =
          issuePosition
              ? sourceType.getLayout().getTimeFactors()[*issuePosition]
              : 0;
      const bool oneTimeAxis =
          issuePosition && sourceTimes && sourceReplicas &&
          checkedScale(*sourceTimes, *sourceReplicas, sourceParts) &&
          checkedScale(issueCount, windowExtent, representedIssueExtent) &&
          issueCount > 1 && issueParts == 1 &&
          sourceParts == issueCount &&
          sourceType.getLayout().getLaneFactors()[*issuePosition] ==
              windowExtent &&
          sourceType.getShape()[*issuePosition] == representedIssueExtent &&
          llvm::all_of(sourceType.getLayout().getReplicaFactors().asArrayRef(),
                       [](int64_t factor) { return factor == 1; });
      if (!oneTimeAxis ||
          load.getWindowForPart().size() != static_cast<size_t>(issueCount) ||
          load.getLayerForPart().size() != static_cast<size_t>(issueCount) ||
          load.getPhysicalLayerForPart().size() !=
              static_cast<size_t>(issueCount) ||
          load.getShiftOffsetForPart().size() !=
              static_cast<size_t>(issueCount) ||
          load.getShiftBaseFactorForPart().size() !=
              static_cast<size_t>(issueCount) ||
          load.getMaskValueForPart().size() !=
              static_cast<size_t>(issueCount))
        return mlir::failure();

      const int64_t firstWindow = load.getWindowForPart()[0];
      if (firstWindow < 0 ||
          firstWindow >= static_cast<int64_t>(load.getWindowOffsets().size()))
        return mlir::failure();
      const int64_t firstOffset = load.getWindowOffsets()[firstWindow];
      int64_t issueStride = 0;
      if (issueCount > 1) {
        const int64_t secondWindow = load.getWindowForPart()[1];
        if (secondWindow < 0 ||
            secondWindow >= static_cast<int64_t>(load.getWindowOffsets().size()) ||
            !checkedSubtract(load.getWindowOffsets()[secondWindow], firstOffset,
                             issueStride) ||
            issueStride <= 0)
          return mlir::failure();
      }
      const int64_t recordRank = load.getRecordRank();
      for (int64_t issue = 0; issue < issueCount; ++issue) {
        const int64_t window = load.getWindowForPart()[issue];
        int64_t expectedDelta = 0;
        int64_t actualDelta = 0;
        if (window < 0 ||
            window >= static_cast<int64_t>(load.getWindowOffsets().size()) ||
            !checkedScale(issue, issueStride, expectedDelta) ||
            !checkedSubtract(load.getWindowOffsets()[window], firstOffset,
                             actualDelta) ||
            actualDelta != expectedDelta ||
            load.getLayerForPart()[issue] != load.getLayerForPart()[0] ||
            load.getPhysicalLayerForPart()[issue] !=
                load.getPhysicalLayerForPart()[0] ||
            load.getShiftOffsetForPart()[issue] !=
                load.getShiftOffsetForPart()[0] ||
            load.getShiftBaseFactorForPart()[issue] !=
                load.getShiftBaseFactorForPart()[0] ||
            load.getMaskValueForPart()[issue] !=
                load.getMaskValueForPart()[0])
          return mlir::failure();
        for (int64_t coordinate = 0; coordinate < recordRank; ++coordinate)
          if (load.getRecordCoordinatesForWindow()[static_cast<size_t>(window) *
                                                       recordRank +
                                                   coordinate] !=
              load.getRecordCoordinatesForWindow()[
                  static_cast<size_t>(firstWindow) * recordRank + coordinate])
            return mlir::failure();
      }

      mlir::Value issueBase = load.getLogicalBase();
      const int64_t constantOffset = firstOffset;
      if (auto integer =
              mlir::dyn_cast<mlir::IntegerType>(issueBase.getType())) {
        auto typedIndex = rewriter.create<riscv::CastOp>(
            load.getLoc(), integer, windowIndex,
            riscv_internal::unselectedLeaf(rewriter));
        auto stride = rewriter.create<riscv::ConstantOp>(
            load.getLoc(), integer,
            rewriter.getIntegerAttr(integer, issueStride));
        auto dynamicOffset = rewriter.create<riscv::BinaryOp>(
            load.getLoc(), integer, typedIndex.getResult(), stride.getResult(),
            "mul", riscv_internal::unselectedLeaf(rewriter));
        mlir::Value totalOffset = dynamicOffset.getResult();
        if (constantOffset) {
          auto constant = rewriter.create<riscv::ConstantOp>(
              load.getLoc(), integer,
              rewriter.getIntegerAttr(integer, constantOffset));
          totalOffset = rewriter
                            .create<riscv::BinaryOp>(
                                load.getLoc(), integer, totalOffset,
                                constant.getResult(), "add",
                                riscv_internal::unselectedLeaf(rewriter))
                            .getResult();
        }
        issueBase = rewriter
                        .create<riscv::BinaryOp>(
                            load.getLoc(), integer, issueBase, totalOffset,
                            "add", riscv_internal::unselectedLeaf(rewriter))
                        .getResult();
      } else if (issueBase.getType().isIndex()) {
        auto stride = rewriter.create<mlir::arith::ConstantIndexOp>(
            load.getLoc(), issueStride);
        mlir::Value totalOffset =
            rewriter
                .create<mlir::arith::MulIOp>(load.getLoc(), windowIndex,
                                             stride.getResult())
                .getResult();
        if (constantOffset) {
          auto constant = rewriter.create<mlir::arith::ConstantIndexOp>(
              load.getLoc(), constantOffset);
          totalOffset =
              rewriter
                  .create<mlir::arith::AddIOp>(load.getLoc(), totalOffset,
                                              constant.getResult())
                  .getResult();
        }
        issueBase =
            rewriter
                .create<mlir::arith::AddIOp>(load.getLoc(), issueBase,
                                            totalOffset)
                .getResult();
      } else {
        return mlir::failure();
      }

      int64_t projectedAlignment = sourcePlan.getOffsetAlignment();
      projectedAlignment = std::gcd(projectedAlignment, issueStride);
      if (constantOffset)
        projectedAlignment = std::gcd(projectedAlignment, constantOffset);
      projectedAlignment = std::max<int64_t>(1, projectedAlignment);
      auto projectedPlan = riscv::StorageWindowPlanAttr::get(
          rewriter.getContext(), sourcePlan.getKind(),
          sourcePlan.getReductionAxis(), sourcePlan.getProjectionBase(),
          sourcePlan.getProjectionStride(), sourcePlan.getProjectionRepeat(),
          sourcePlan.getProjectionExtent(), projectedAlignment,
          sourcePlan.getRecordElements(), sourcePlan.getByteOffset(),
          sourcePlan.getElementBits(), sourcePlan.getGroupSize(),
          sourcePlan.getLayerSize(), sourcePlan.getPhysicalLayerBase(),
          sourcePlan.getPhysicalLayerStep(), sourcePlan.getShiftBase(),
          sourcePlan.getShiftStep(), sourcePlan.getMaskValue());
      llvm::SmallVector<int64_t> recordCoordinates;
      for (int64_t coordinate = 0; coordinate < recordRank; ++coordinate)
        recordCoordinates.push_back(load.getRecordCoordinatesForWindow()[
            static_cast<size_t>(firstWindow) * recordRank + coordinate]);
      auto singleton = [&](int64_t value) {
        return rewriter.getDenseI64ArrayAttr({value});
      };
      auto selected = load.getLeaf();
      const int64_t temporaryGroups = std::max<int64_t>(
          1, (resultType.getLayout().getLmulEighths() + 7) / 8);
      auto cloned = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
          load.getLoc(), resultType, load.getField(), issueBase, projectedPlan,
          recordRank, singleton(0),
          rewriter.getDenseI64ArrayAttr(recordCoordinates), singleton(0),
          singleton(load.getLayerForPart()[0]),
          singleton(load.getPhysicalLayerForPart()[0]),
          singleton(load.getShiftOffsetForPart()[0]),
          singleton(load.getShiftBaseFactorForPart()[0]),
          singleton(load.getMaskValueForPart()[0]), load.getAccess(),
          riscv_internal::leaf(
              rewriter, selected.getEngine(), selected.getFamily(),
              selected.getInstruction(), selected.getSpelling(),
              selected.getOperandGroups(),
              resultType.getLayout().getRegisterGroups(), temporaryGroups,
              selected.getFragmentGroups(), selected.getMask(),
              resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                              : "exact",
              selected.getParameters(), selected.getLocalBytes()));
      return remember(cloned.getResult());
    }

    if (load.getRecordRank() != 0 || issueParts != 1)
      return mlir::failure();

    mlir::Value base = load.getLogicalBase();
    mlir::Value issueBase;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(base.getType())) {
      auto typedIndex = rewriter.create<riscv::CastOp>(
          load.getLoc(), integer, windowIndex,
          riscv_internal::unselectedLeaf(rewriter));
      auto extent = rewriter.create<riscv::ConstantOp>(
          load.getLoc(), integer,
          rewriter.getIntegerAttr(integer, windowExtent));
      auto offset = rewriter.create<riscv::BinaryOp>(
          load.getLoc(), integer, typedIndex.getResult(), extent.getResult(),
          "mul", riscv_internal::unselectedLeaf(rewriter));
      issueBase = rewriter
                      .create<riscv::BinaryOp>(
                          load.getLoc(), integer, base, offset.getResult(),
                          "add", riscv_internal::unselectedLeaf(rewriter))
                      .getResult();
    } else if (base.getType().isIndex()) {
      auto extent = rewriter.create<mlir::arith::ConstantIndexOp>(
          load.getLoc(), windowExtent);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          load.getLoc(), windowIndex, extent.getResult());
      issueBase = rewriter
                      .create<mlir::arith::AddIOp>(
                          load.getLoc(), base, offset.getResult())
                      .getResult();
    } else {
      return mlir::failure();
    }

    // The storage-load owner has already selected the physical memory form.
    // Issue-window materialization only projects that closed unit-load plan to
    // one smaller carrier; it must not re-run storage-form selection.
    const int64_t projectedAlignment =
        std::gcd(sourcePlan.getOffsetAlignment(), windowExtent);
    auto projectedPlan = riscv::StorageWindowPlanAttr::get(
        rewriter.getContext(), sourcePlan.getKind(),
        sourcePlan.getReductionAxis(), sourcePlan.getProjectionBase(),
        sourcePlan.getProjectionStride(), sourcePlan.getProjectionRepeat(),
        windowExtent, std::max<int64_t>(1, projectedAlignment),
        sourcePlan.getRecordElements(), sourcePlan.getByteOffset(),
        sourcePlan.getElementBits(), sourcePlan.getGroupSize(),
        sourcePlan.getLayerSize(), sourcePlan.getPhysicalLayerBase(),
        sourcePlan.getPhysicalLayerStep(), sourcePlan.getShiftBase(),
        sourcePlan.getShiftStep(), sourcePlan.getMaskValue());
    auto zero = rewriter.getDenseI64ArrayAttr({0});
    auto empty = rewriter.getDenseI64ArrayAttr({});
    auto selected = load.getLeaf();
    const int64_t temporaryGroups = std::max<int64_t>(
        1, (resultType.getLayout().getLmulEighths() + 7) / 8);
    auto cloned = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
        load.getLoc(), resultType, load.getField(), issueBase, projectedPlan, 0,
        zero, empty, zero, zero, zero, zero, zero, zero, load.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(),
            selected.getOperandGroups(),
            resultType.getLayout().getRegisterGroups(), temporaryGroups,
            selected.getFragmentGroups(), selected.getMask(),
            resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                           : "exact",
            selected.getParameters(), selected.getLocalBytes()));
    return remember(cloned.getResult());
  }

  if (auto load = mlir::dyn_cast<riscv::RVVIndexedEntryLoadOp>(definition)) {
    auto offsets = cloneOperand(load.getEntryOffsets());
    if (mlir::failed(offsets)) {
      load.emitError(
          "issue-window projection could not project indexed-entry byte offsets; offset_type=")
          << load.getEntryOffsets().getType() << ", offset_def="
          << (load.getEntryOffsets().getDefiningOp()
                  ? load.getEntryOffsets()
                        .getDefiningOp()
                        ->getName()
                        .getStringRef()
                  : llvm::StringRef("<block-argument>"));
      return mlir::failure();
    }
    auto offsetType = mlir::dyn_cast<riscv::ValueType>((*offsets).getType());
    auto selected = load.getLeaf();
    const int64_t operandGroups =
        offsetType && offsetType.getLayout().getCarrier() == "rvv"
            ? offsetType.getLayout().getRegisterGroups()
            : 0;
    auto cloned = rewriter.create<riscv::RVVIndexedEntryLoadOp>(
        load.getLoc(), resultType, load.getSource(), *offsets,
        load.getSourceAxis(), load.getPayloadAxis(), load.getPayloadExtent(),
        load.getEntryByteStride(), load.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(), operandGroups,
            resultType.getLayout().getRegisterGroups(),
            selected.getTemporaryGroups(), selected.getFragmentGroups(),
            selected.getMask(), selected.getTail(), selected.getParameters(),
            selected.getLocalBytes()));
    return remember(cloned.getResult());
  }

  if (auto index =
          mlir::dyn_cast<riscv::RVVRegularRepeatIndexOp>(definition)) {
    auto selected = llvm::find(index.getResults(), value);
    if (selected == index.getResults().end())
      return mlir::failure();
    auto bases = rewriter.getDenseI64ArrayAttr({0});
    auto cloned = rewriter.create<riscv::RVVRegularRepeatIndexOp>(
        index.getLoc(), mlir::TypeRange{resultType}, index.getReductionAxis(),
        index.getRepeat(), rewriter.getArrayAttr({bases}),
        riscv_internal::leaf(
            rewriter, "rvv", "regular-repeat-index",
            (index.getRepeat() & (index.getRepeat() - 1)) == 0
                ? "rvv.regular-repeat-index.pow2"
                : "rvv.regular-repeat-index.div",
            (index.getRepeat() & (index.getRepeat() - 1)) == 0
                ? "rvv.regular-repeat-index.pow2"
                : "rvv.regular-repeat-index.div",
            0, resultType.getLayout().getRegisterGroups(), 1, 0, "none",
            "exact", {static_cast<int64_t>(index.getReductionAxis()),
                       static_cast<int64_t>(index.getRepeat())}));
    return remember(cloned.getResults().front());
  }

  if (auto gather =
          mlir::dyn_cast<riscv::RVVRegularRepeatGatherOp>(definition)) {
    auto selected = llvm::find(gather.getResults(), value);
    if (selected == gather.getResults().end() ||
        !gather.getSourceBase().getType().isIndex())
      return mlir::failure();
    const size_t resultNumber =
        static_cast<size_t>(selected - gather.getResults().begin());
    auto axis = axisPosition(resultType, gather.getReductionAxis());
    const int64_t lanes =
        axis ? resultType.getLayout().getLaneFactors()[*axis] : 0;
    const int64_t repeat = gather.getRepeat();
    if (lanes <= 1 || repeat <= 1 ||
        (lanes % repeat != 0 && repeat % lanes != 0))
      return mlir::failure();
    const int64_t sourceCount = lanes <= repeat ? 1 : lanes / repeat;
    auto stride = rewriter.create<mlir::arith::ConstantIndexOp>(
        gather.getLoc(), sourceCount);
    auto offset = rewriter.create<mlir::arith::MulIOp>(
        gather.getLoc(), windowIndex, stride.getResult());
    auto sourceBase = rewriter.create<mlir::arith::AddIOp>(
        gather.getLoc(), gather.getSourceBase(), offset.getResult());

    llvm::SmallVector<mlir::Value> indices;
    if (lanes > repeat) {
      if (resultNumber >= gather.getIndices().size())
        return mlir::failure();
      auto projectedIndex = cloneOperand(gather.getIndices()[resultNumber]);
      if (mlir::failed(projectedIndex))
        return mlir::failure();
      indices.push_back(*projectedIndex);
    }
    auto bases = rewriter.getDenseI64ArrayAttr({0});
    const bool powerOfTwo = (repeat & (repeat - 1)) == 0;
    llvm::StringRef instruction =
        lanes <= repeat ? "rvv.regular-repeat-broadcast"
                        : powerOfTwo ? "rvv.regular-repeat-gather.pow2"
                                     : "rvv.regular-repeat-gather.div";
    const int64_t temporaryGroups =
        2 * resultType.getLayout().getRegisterGroups();
    auto cloned = rewriter.create<riscv::RVVRegularRepeatGatherOp>(
        gather.getLoc(), mlir::TypeRange{resultType}, gather.getField(), indices,
        gather.getSourceAxis(), gather.getReductionAxis(),
        sourceBase.getResult(), sourceCount, repeat,
        rewriter.getArrayAttr({bases}), gather.getAccess(),
        riscv_internal::leaf(
            rewriter, "rvv", "regular-repeat-gather", instruction,
            instruction,
            gather.getField().getType().getLayout().getRegisterGroups(), 0,
            temporaryGroups, 0, "none", "exact",
            {static_cast<int64_t>(gather.getSourceAxis()),
             static_cast<int64_t>(gather.getReductionAxis()), sourceCount,
             repeat}));
    return remember(cloned.getResults().front());
  }

  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    llvm::SmallVector<mlir::Value> indices;
    for (mlir::Value index : extract.getIndices()) {
      auto cloned = cloneOperand(index);
      if (mlir::failed(cloned)) {
        auto diagnostic = extract.emitError(
            "issue-window projection could not project extract coordinates; index=");
        diagnostic << index;
        if (mlir::Operation *producer = index.getDefiningOp())
          diagnostic << ", producer=" << producer->getName().getStringRef();
        return mlir::failure();
      }
      indices.push_back(*cloned);
    }
    auto cloned = rewriter.create<riscv::ExtractOp>(
        extract.getLoc(), resultType, extract.getInput(), indices,
        extract.getSelectors(), extract.getAccess(),
        riscv_internal::unselectedLeaf(rewriter));
    if (auto pattern =
            extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern"))
      cloned->setAttr("index_pattern", pattern);
    return remember(cloned.getResult());
  }

  mlir::Value cloned;
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(definition)) {
    auto input = cloneOperand(unary.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::UnaryOp>(unary.getLoc(), resultType, *input,
                                         unary.getKind(),
                                         riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(definition)) {
    auto lhs = cloneOperand(binary.getLhs());
    auto rhs = cloneOperand(binary.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::BinaryOp>(
                     binary.getLoc(), resultType, *lhs, *rhs, binary.getKind(),
                     riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto cast = mlir::dyn_cast<riscv::CastOp>(definition)) {
    auto input = cloneOperand(cast.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    auto inputType = mlir::dyn_cast<riscv::ValueType>((*input).getType());
    const bool sameWidth =
        inputType &&
        riscv_internal::logicalBitWidth(inputType.getElementType()) ==
            riscv_internal::logicalBitWidth(resultType.getElementType());
    if (sameWidth && inputType.getLayout() != resultType.getLayout()) {
      auto castType = riscv::ValueType::get(
          rewriter.getContext(), resultType.getElementType(),
          inputType.getShape(), inputType.getAxisIds(), inputType.getLayout());
      auto localCast = rewriter.create<riscv::CastOp>(
          cast.getLoc(), castType, *input,
          riscv_internal::unselectedLeaf(rewriter));
      auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
          cast.getLoc(), resultType, localCast.getResult(),
          riscv_internal::layoutConversion(rewriter, inputType.getLayout(),
                                           resultType.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(cast, localCast);
      riscv_internal::copyOrigin(cast, conversion);
      cloned = conversion.getResult();
    } else {
      cloned = rewriter
                   .create<riscv::CastOp>(
                       cast.getLoc(), resultType, *input,
                       riscv_internal::unselectedLeaf(rewriter))
                   .getResult();
    }
  } else if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(definition)) {
    auto input = cloneOperand(narrow.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::NarrowOp>(
                     narrow.getLoc(), resultType, *input, narrow.getRounding(),
                     narrow.getSaturate(),
                     riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto widen = mlir::dyn_cast<riscv::WidenOp>(definition)) {
    auto input = cloneOperand(widen.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::WidenOp>(widen.getLoc(), resultType, *input,
                                         riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto conversion =
                 mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
    auto input = cloneOperand(conversion.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    if ((*input).getType() == resultType) {
      cloned = *input;
    } else {
      auto inputType = mlir::cast<riscv::ValueType>((*input).getType());
      cloned = rewriter
                   .create<riscv::ConvertLayoutOp>(
                       conversion.getLoc(), resultType, *input,
                       riscv_internal::layoutConversion(
                           rewriter, inputType.getLayout(), resultType.getLayout()),
                       conversion.getSourceAccessAttr(),
                       riscv_internal::unselectedLeaf(rewriter))
                   .getResult();
    }
  } else if (auto add = mlir::dyn_cast<riscv::RVVWidenAddOp>(definition)) {
    auto lhs = cloneOperand(add.getLhs());
    auto rhs = cloneOperand(add.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVWidenAddOp>(add.getLoc(), resultType,
                                              *lhs, *rhs, add.getLeaf())
                 .getResult();
  } else if (auto broadcast =
                 mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(definition)) {
    auto input = cloneOperand(broadcast.getInput());
    auto inputType = mlir::succeeded(input)
                         ? mlir::dyn_cast<riscv::ValueType>((*input).getType())
                         : riscv::ValueType();
    const int64_t inputLanes =
        inputType ? product(inputType.getLayout().getLaneFactors()) : 0;
    const int64_t resultLanes = product(resultType.getLayout().getLaneFactors());
    if (mlir::failed(input) || !inputType || inputLanes <= 0 ||
        resultLanes < inputLanes)
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVAxisBroadcastOp>(
                     broadcast.getLoc(), resultType, *input,
                     riscv_internal::leaf(
                         rewriter, "rvv", "axis-broadcast",
                         "rvv.axis-broadcast", "rvv.axis-broadcast",
                         inputType.getLayout().getRegisterGroups(),
                         resultType.getLayout().getRegisterGroups(), 1, 0,
                         "none", "exact", {inputLanes, resultLanes}))
                 .getResult();
  } else if (auto materialize =
                 mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition)) {
    auto input = cloneOperand(materialize.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RegisterMaterializeOp>(
                     materialize.getLoc(), resultType, *input,
                     materialize.getOwnerDomainId(), materialize.getBirthId(),
                     materialize.getLifetimeEndDomainId(),
                     materialize.getRealization())
                 .getResult();
  } else {
    definition->emitError("issue-window projection does not support producer ")
        << definition->getName().getStringRef() << "; source=" << sourceType;
    return mlir::failure();
  }
  return remember(cloned);
}

riscv::ValueType partialType(mlir::Builder &builder, riscv::ValueType lhs,
                             mlir::Type resultType, int64_t reductionAxis) {
  auto result = mlir::dyn_cast<riscv::ValueType>(resultType);
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> replicas;
  llvm::StringRef validity = lhs.getLayout().getValidity();
  if (result) {
    if (result.getLayout().getCarrier() != "scalar")
      return {};
    axes.assign(result.getAxisIds().asArrayRef().begin(),
                result.getAxisIds().asArrayRef().end());
    shape.assign(result.getShape().asArrayRef().begin(),
                 result.getShape().asArrayRef().end());
    replicas.assign(result.getLayout().getReplicaFactors().asArrayRef().begin(),
                    result.getLayout().getReplicaFactors().asArrayRef().end());
    for (size_t position = 0; position < result.getShape().size(); ++position)
      if (result.getLayout().getTimeFactors()[position] != 1 ||
          result.getLayout().getLaneFactors()[position] != 1 ||
          (result.getShape()[position] > 0
               ? result.getLayout().getReplicaFactors()[position] !=
                     result.getShape()[position]
               : result.getLayout().getReplicaFactors()[position] <= 0) ||
          result.getLayout().getFragmentFactors()[position] != 1 ||
          result.getLayout().getLocalFactors()[position] != 1)
        return {};
    validity = result.getLayout().getValidity();
  }
  auto position = axisPosition(lhs, reductionAxis);
  if (!position)
    return {};
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  if (!inputElement || inputElement.isSignless() || inputElement.getWidth() > 16)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t lanes = lhs.getLayout().getLaneFactors()[*position];
  const int64_t partialLMUL = lhs.getLayout().getLmulEighths() * 2;
  axes.push_back(reductionAxis);
  shape.push_back(lanes);
  replicas.push_back(1);
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  lane.back() = lanes;
  const int64_t groupsPerVector = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  const int64_t groups = groupsPerVector * product(replicas);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", riscv_internal::integers(builder, axes),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), partialWidth, partialLMUL,
      lhs.getLayout().getVl(), groups, validity);
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape),
                               riscv_internal::integers(builder, axes), layout);
}

riscv::ValueType projectReplicaReductionOperandType(
    mlir::Builder &builder, riscv::ValueType input, int64_t removedAxis) {
  auto removed = axisPosition(input, removedAxis);
  if (!removed || input.getLayout().getCarrier() != "rvv" ||
      input.getLayout().getTimeFactors()[*removed] <= 1 ||
      input.getLayout().getLaneFactors()[*removed] != 1 ||
      input.getLayout().getReplicaFactors()[*removed] != 1 ||
      input.getLayout().getFragmentFactors()[*removed] != 1 ||
      input.getLayout().getLocalFactors()[*removed] != 1)
    return {};

  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane;
  llvm::SmallVector<int64_t> replica;
  llvm::SmallVector<int64_t> fragment;
  llvm::SmallVector<int64_t> local;
  auto source = input.getLayout();
  for (size_t position = 0; position < input.getAxisIds().size(); ++position) {
    if (position == *removed)
      continue;
    shape.push_back(input.getShape()[position]);
    axes.push_back(input.getAxisIds()[position]);
    time.push_back(source.getTimeFactors()[position]);
    lane.push_back(source.getLaneFactors()[position]);
    replica.push_back(source.getReplicaFactors()[position]);
    fragment.push_back(source.getFragmentFactors()[position]);
    local.push_back(source.getLocalFactors()[position]);
  }
  const int64_t laneCount = product(lane);
  const int64_t replicaCount = product(replica);
  if (laneCount <= 1 || replicaCount <= 0 || product(time) != 1 ||
      product(fragment) != 1 || product(local) != 1)
    return {};
  const int64_t groupsPerVector =
      std::max<int64_t>(1, (source.getLmulEighths() + 7) / 8);
  auto axisIds = riscv_internal::integers(builder, axes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds,
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local), source.getSew(),
      source.getLmulEighths(), laneCount, groupsPerVector * replicaCount,
      source.getValidity());
  return riscv::ValueType::get(builder.getContext(), input.getElementType(),
                               riscv_internal::integers(builder, shape), axisIds,
                               layout);
}

riscv::ValueType replicaReducedAccumulatorType(
    mlir::Builder &builder, riscv::ValueType lhs, riscv::ValueType rhs,
    llvm::ArrayRef<int64_t> reductionAxes) {
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  if (reductionAxes.empty() || !lhsElement || !rhsElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      lhs.getLayout().getCarrier() != "rvv" ||
      rhs.getLayout().getCarrier() != "rvv" ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      lhs.getLayout().getLmulEighths() != rhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl())
    return {};

  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> replicas;
  auto appendFree = [&](riscv::ValueType operand) -> bool {
    for (size_t position = 0; position < operand.getAxisIds().size(); ++position) {
      const int64_t axis = operand.getAxisIds()[position];
      if (llvm::is_contained(reductionAxes, axis))
        continue;
      if (operand.getLayout().getTimeFactors()[position] != 1 ||
          operand.getLayout().getLaneFactors()[position] != 1 ||
          operand.getLayout().getReplicaFactors()[position] <= 0 ||
          operand.getLayout().getFragmentFactors()[position] != 1 ||
          operand.getLayout().getLocalFactors()[position] != 1)
        return false;
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) {
        axes.push_back(axis);
        shape.push_back(operand.getShape()[position]);
        replicas.push_back(operand.getLayout().getReplicaFactors()[position]);
        continue;
      }
      const size_t target = static_cast<size_t>(found - axes.begin());
      if (shape[target] != operand.getShape()[position] ||
          replicas[target] != operand.getLayout().getReplicaFactors()[position])
        return false;
    }
    return true;
  };
  if (!appendFree(lhs) || !appendFree(rhs))
    return {};

  llvm::SmallVector<int64_t> reductionLanes;
  int64_t totalReductionLanes = 1;
  for (int64_t reductionAxis : reductionAxes) {
    auto lhsReduction = axisPosition(lhs, reductionAxis);
    auto rhsReduction = axisPosition(rhs, reductionAxis);
    if (!lhsReduction || !rhsReduction)
      return {};
    const int64_t lanes =
        lhs.getLayout().getLaneFactors()[*lhsReduction];
    if (lanes <= 0 ||
        lanes != rhs.getLayout().getLaneFactors()[*rhsReduction] ||
        lhs.getShape()[*lhsReduction] != rhs.getShape()[*rhsReduction] ||
        lhs.getLayout().getTimeFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getTimeFactors()[*rhsReduction] != 1 ||
        lhs.getLayout().getReplicaFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getReplicaFactors()[*rhsReduction] != 1 ||
        lhs.getLayout().getFragmentFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getFragmentFactors()[*rhsReduction] != 1 ||
        lhs.getLayout().getLocalFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getLocalFactors()[*rhsReduction] != 1 ||
        totalReductionLanes > std::numeric_limits<int64_t>::max() / lanes)
      return {};
    axes.push_back(reductionAxis);
    shape.push_back(lhs.getShape()[*lhsReduction]);
    replicas.push_back(1);
    reductionLanes.push_back(lanes);
    totalReductionLanes *= lanes;
  }
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  for (size_t position = 0; position < reductionLanes.size(); ++position)
    lane[lane.size() - reductionLanes.size() + position] =
        reductionLanes[position];
  const int64_t partialWidth =
      2 * std::max<unsigned>(8, lhsElement.getWidth());
  const int64_t partialLMUL = 2 * lhs.getLayout().getLmulEighths();
  const int64_t groupsPerVector =
      std::max<int64_t>(1, (partialLMUL + 7) / 8);
  const int64_t replicaCount = product(replicas);
  if (partialLMUL <= 0 || replicaCount <= 0)
    return {};
  auto axisIds = riscv_internal::integers(builder, axes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds,
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), partialWidth, partialLMUL,
      totalReductionLanes, groupsPerVector * replicaCount,
      lhs.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape), axisIds,
                               layout);
}

riscv::ValueType replicaReducedAccumulatorType(
    mlir::Builder &builder, riscv::ValueType lhs, riscv::ValueType rhs,
    int64_t reductionAxis) {
  return replicaReducedAccumulatorType(
      builder, lhs, rhs, llvm::ArrayRef<int64_t>(reductionAxis));
}

riscv::ValueType partialSlotType(mlir::Builder &builder, riscv::ValueType operand,
                                 int64_t reductionAxis) {
  auto position = axisPosition(operand, reductionAxis);
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(operand.getElementType());
  if (!position || !inputElement || inputElement.isSignless() ||
      inputElement.getWidth() > 16)
    return {};
  const int64_t lanes = operand.getLayout().getLaneFactors()[*position];
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = operand.getLayout().getLmulEighths() * 2;
  const int64_t groups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      partialWidth, partialLMUL, operand.getLayout().getVl(), groups,
      operand.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element, lane, axisIds,
                               layout);
}

riscv::ValueType partialSlotType(mlir::Builder &builder,
                                 riscv::RVVWidenDotOp dot) {
  if (dot.getOver().empty() || dot.getReductionLanes() <= 0 ||
      dot.getSliceLmulEighths() <= 0)
    return {};
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(
      dot.getLhs().getType().getElementType());
  if (!inputElement || inputElement.isSignless() ||
      inputElement.getWidth() > 16)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = 2 * dot.getSliceLmulEighths();
  const int64_t groups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> lanes;
  int64_t laneProduct = 1;
  for (int64_t axis : dot.getOver()) {
    auto position = axisPosition(dot.getLhs().getType(), axis);
    if (!position)
      return {};
    const int64_t lane =
        dot.getLhs().getType().getLayout().getLaneFactors()[*position];
    if (lane <= 0 || laneProduct > dot.getReductionLanes() / lane)
      return {};
    axes.push_back(axis);
    lanes.push_back(lane);
    laneProduct *= lane;
  }
  if (laneProduct != dot.getReductionLanes())
    return {};
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  auto axisIds = riscv_internal::integers(builder, axes);
  auto oneAttr = riscv_internal::integers(builder, one);
  auto lanesAttr = riscv_internal::integers(builder, lanes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, oneAttr, lanesAttr, oneAttr,
      oneAttr, oneAttr,
      partialWidth, partialLMUL, dot.getReductionLanes(), groups,
      dot.getLhs().getType().getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element, lanesAttr, axisIds,
                               layout);
}

riscv::ValueType issueSliceType(mlir::Builder &builder,
                                riscv::ValueType source,
                                llvm::ArrayRef<int64_t> reductionAxes) {
  if (reductionAxes.empty() || source.getLayout().getCarrier() != "rvv")
    return {};
  llvm::SmallVector<int64_t> shape(source.getShape().asArrayRef());
  llvm::SmallVector<int64_t> timeFactors(
      source.getLayout().getTimeFactors().asArrayRef());
  bool sliced = false;
  for (int64_t reductionAxis : reductionAxes) {
    auto position = axisPosition(source, reductionAxis);
    if (!position)
      return {};
    const int64_t time = source.getLayout().getTimeFactors()[*position];
    const int64_t lane = source.getLayout().getLaneFactors()[*position];
    const int64_t replica = source.getLayout().getReplicaFactors()[*position];
    if (time <= 0 || lane <= 0 || replica != 1 ||
        source.getShape()[*position] != time * lane ||
        source.getLayout().getFragmentFactors()[*position] != 1 ||
        source.getLayout().getLocalFactors()[*position] != 1)
      return {};
    if (time > 1) {
      shape[*position] = lane;
      timeFactors[*position] = 1;
      sliced = true;
    }
  }
  if (!sliced)
    return {};
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      builder.getDenseI64ArrayAttr(timeFactors),
      source.getLayout().getLaneFactors(),
      source.getLayout().getReplicaFactors(),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), source.getLayout().getSew(),
      source.getLayout().getLmulEighths(), source.getLayout().getVl(),
      source.getLayout().getRegisterGroups(),
      source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                               builder.getDenseI64ArrayAttr(shape),
                               source.getAxisIds(), layout);
}

riscv::ValueType sequentialIssueType(mlir::Builder &builder,
                                     riscv::RVVWidenDotOp dot,
                                     riscv::ValueType source) {
  if (dot.getOver().empty() || dot.getReductionLanes() <= 0 ||
      dot.getSliceLmulEighths() <= 0 ||
      source.getLayout().getCarrier() != "rvv" ||
      source.getLayout().getLmulEighths() < dot.getSliceLmulEighths() ||
      source.getLayout().getLmulEighths() % dot.getSliceLmulEighths())
    return {};
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> lanes;
  int64_t laneProduct = 1;
  for (int64_t reductionAxis : dot.getOver()) {
    auto position = axisPosition(source, reductionAxis);
    if (!position)
      return {};
    const int64_t time = source.getLayout().getTimeFactors()[*position];
    const int64_t lane = source.getLayout().getLaneFactors()[*position];
    if (time <= 0 || lane <= 0 ||
        source.getShape()[*position] != time * lane ||
        source.getLayout().getReplicaFactors()[*position] != 1 ||
        source.getLayout().getFragmentFactors()[*position] != 1 ||
        source.getLayout().getLocalFactors()[*position] != 1 ||
        laneProduct > dot.getReductionLanes() / lane)
      return {};
    shape.push_back(lane);
    lanes.push_back(lane);
    laneProduct *= lane;
  }
  for (size_t position = 0; position < source.getAxisIds().size(); ++position)
    if (!llvm::is_contained(dot.getOver(), source.getAxisIds()[position]) &&
        (source.getLayout().getFragmentFactors()[position] != 1 ||
         source.getLayout().getLocalFactors()[position] != 1))
      return {};
  if (laneProduct != dot.getReductionLanes())
    return {};
  llvm::SmallVector<int64_t> one(dot.getOver().size(), 1);
  auto axes = builder.getDenseI64ArrayAttr(dot.getOver());
  auto oneAttr = builder.getDenseI64ArrayAttr(one);
  auto laneAttr = builder.getDenseI64ArrayAttr(lanes);
  const int64_t groups =
      std::max<int64_t>(1, (dot.getSliceLmulEighths() + 7) / 8);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axes, oneAttr, laneAttr, oneAttr, oneAttr,
      oneAttr, source.getLayout().getSew(), dot.getSliceLmulEighths(),
      dot.getReductionLanes(), groups, source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                               builder.getDenseI64ArrayAttr(shape), axes, layout);
}

riscv::ValueType groupedLanePartialSlotType(mlir::Builder &builder,
                                            riscv::TargetAttr target,
                                            riscv::ValueType lhs,
                                            riscv::ValueType rhs,
                                            int64_t reductionAxis,
                                            int64_t lanes) {
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(
      lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(
      rhs.getElementType());
  if (lanes <= 0 || !lhsElement || !rhsElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      lhsElement.getWidth() != rhsElement.getWidth() ||
      lhsElement.getWidth() > 16)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, lhsElement.getWidth());
  auto lhsSliceLMUL = riscv::rvvLaneSliceLMULEighths(target, lhs, lanes);
  auto rhsSliceLMUL = riscv::rvvLaneSliceLMULEighths(target, rhs, lanes);
  if (!lhsSliceLMUL || !rhsSliceLMUL || *lhsSliceLMUL != *rhsSliceLMUL ||
      *lhsSliceLMUL > std::numeric_limits<int64_t>::max() / 2)
    return {};
  const int64_t partialLMUL = 2 * *lhsSliceLMUL;
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      partialWidth, partialLMUL, lanes,
      std::max<int64_t>(1, (partialLMUL + 7) / 8),
      lhs.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element, lane, axisIds,
                               layout);
}

riscv::ValueType splitPartialSlotType(mlir::Builder &builder,
                                      riscv::ValueType source,
                                      int64_t reductionAxis, int64_t split) {
  auto position = axisPosition(source, reductionAxis);
  if (!position || split <= 1 ||
      source.getLayout().getLaneFactors()[*position] % split ||
      source.getLayout().getLmulEighths() % split ||
      source.getLayout().getVl() % split)
    return {};
  const int64_t lanes =
      source.getLayout().getLaneFactors()[*position] / split;
  const int64_t lmul = source.getLayout().getLmulEighths() / split;
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      source.getLayout().getSew(), lmul, source.getLayout().getVl() / split,
      std::max<int64_t>(1, (lmul + 7) / 8),
      source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                               lane, axisIds, layout);
}

std::optional<riscv::LeafAttr> partialRepackLeaf(
    mlir::Builder &builder, riscv::TargetAttr target,
    riscv::PartialSetType source, riscv::PartialSetType result, int64_t split) {
  if (!source || !result)
    return std::nullopt;
  auto carrier = riscv::rvvPartialRepackCarrierLMULEighths(
      target, source.getPartialType(), result.getPartialType(), split);
  if (!carrier)
    return std::nullopt;
  auto layout = result.getPartialType().getLayout();
  const int64_t capacity = target.getVlenBits() * *carrier /
                           (8 * layout.getSew());
  const bool registerView = *carrier == layout.getLmulEighths() &&
                            layout.getLmulEighths() >= 8 &&
                            layout.getVl() == capacity;
  const llvm::StringRef instruction = registerView
      ? "rvv.partial-repack.split" : "rvv.partial-repack.slice";
  llvm::SmallVector<int64_t> parameters{source.getReductionAxis(), split};
  if (!registerView)
    parameters.push_back(*carrier);
  const int64_t temporaries = registerView ? 0 :
      result.getResourceGroups() + std::max<int64_t>(1, *carrier / 8);
  return riscv_internal::leaf(
      builder, "rvv", "partial-repack", instruction, instruction,
      source.getResourceGroups(), result.getResourceGroups(), temporaries, 0,
      "none", "exact", parameters);
}

riscv::ValueType widenPartialSlotType(mlir::Builder &builder,
                                      riscv::ValueType source) {
  auto element = mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
  if (!element || !element.isSigned() || element.getWidth() != 16 ||
      source.getLayout().getCarrier() != "rvv")
    return {};
  const int64_t lmul = source.getLayout().getLmulEighths() * 2;
  const int64_t groups = std::max<int64_t>(1, (lmul + 7) / 8);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getLayout().getAxisIds(),
      source.getLayout().getTimeFactors(), source.getLayout().getLaneFactors(),
      source.getLayout().getReplicaFactors(),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), 32, lmul,
      source.getLayout().getVl(), groups, source.getLayout().getValidity());
  auto widened = mlir::IntegerType::get(builder.getContext(), 32,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), widened, source.getShape(),
                               source.getAxisIds(), layout);
}

riscv::ValueType vectorPartialSlotType(
    mlir::Builder &builder, riscv::ValueType operand,
    riscv::ValueType result, int64_t reductionAxis) {
  auto reductionPosition = axisPosition(operand, reductionAxis);
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(operand.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  if (!reductionPosition || !inputElement || inputElement.isSignless() ||
      inputElement.getWidth() > 16 || !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      result.getLayout().getCarrier() != "rvv")
    return {};

  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane;
  llvm::SmallVector<int64_t> replicas;
  llvm::SmallVector<int64_t> one(operand.getShape().size(), 1);
  for (size_t position = 0; position < operand.getShape().size(); ++position) {
    const int64_t axis = operand.getAxisIds()[position];
    if (axis == reductionAxis) {
      const int64_t lanes =
          operand.getLayout().getLaneFactors()[position];
      if (lanes <= 0)
        return {};
      shape.push_back(lanes);
      time.push_back(1);
      lane.push_back(lanes);
      replicas.push_back(1);
      continue;
    }
    auto resultPosition = axisPosition(result, axis);
    if (!resultPosition || result.getShape()[*resultPosition] !=
                               operand.getShape()[position] ||
        result.getLayout().getTimeFactors()[*resultPosition] != 1 ||
        result.getLayout().getReplicaFactors()[*resultPosition] != 1 ||
        result.getLayout().getFragmentFactors()[*resultPosition] != 1 ||
        result.getLayout().getLocalFactors()[*resultPosition] != 1 ||
        result.getLayout().getLaneFactors()[*resultPosition] !=
            operand.getLayout().getLaneFactors()[position])
      return {};
    shape.push_back(result.getShape()[*resultPosition]);
    time.push_back(1);
    lane.push_back(result.getLayout().getLaneFactors()[*resultPosition]);
    replicas.push_back(1);
  }
  if (result.getAxisIds().size() + 1 != operand.getAxisIds().size())
    return {};

  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = operand.getLayout().getLmulEighths() * 2;
  if (result.getLayout().getSew() != partialWidth * 2 ||
      result.getLayout().getLmulEighths() != partialLMUL * 2)
    return {};
  const int64_t groups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", operand.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), partialWidth, partialLMUL,
      operand.getLayout().getVl(), groups,
      operand.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape),
                               operand.getAxisIds(), layout);
}

std::optional<std::string>
partialMultiplyInstruction(riscv::ValueType lhs, riscv::ValueType rhs) {
  auto lhsElement =
      mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement =
      mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
      rhsElement.isSignless())
    return std::nullopt;
  if (lhsElement.isSigned() && rhsElement.isSigned())
    return std::string("rvv.vwmul.vv");
  if (lhsElement.isSigned() && rhsElement.isUnsigned())
    return rhsElement.getWidth() < rhs.getLayout().getSew()
               ? std::string("rvv.vwmul.vv.reinterpret-rhs")
               : std::string("rvv.vwmulsu.vv");
  if (lhsElement.isUnsigned() && rhsElement.isSigned())
    return lhsElement.getWidth() < lhs.getLayout().getSew()
               ? std::string("rvv.vwmul.vv.reinterpret-lhs")
               : std::string("rvv.vwmulsu.vv.swap");
  return std::nullopt;
}

std::optional<std::string>
sequentialMultiplyInstruction(riscv::ValueType lhs, riscv::ValueType rhs) {
  if (!lhs || !rhs)
    return std::nullopt;
  auto lhsElement =
      mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement =
      mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
      rhsElement.isSignless())
    return std::nullopt;
  if (lhsElement.isSigned() && rhsElement.isSigned())
    return std::string("rvv.vwmul.vv");
  if (lhsElement.isSigned())
    return std::string("rvv.vwmulsu.vv");
  if (rhsElement.isSigned())
    return std::string("rvv.vwmulsu.vv.swap");
  return std::nullopt;
}

riscv::ValueType reducedPartialSlotType(mlir::Builder &builder,
                                        riscv::ValueType partial,
                                        int64_t reductionAxis) {
  auto position = axisPosition(partial, reductionAxis);
  auto element = mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  if (!position || !element || !element.isSigned() ||
      (element.getWidth() != 16 && element.getWidth() != 32))
    return {};
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, one, one, one, one, 32, 8, 1,
      1, "full");
  auto resultElement = mlir::IntegerType::get(
      builder.getContext(), 32, mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), resultElement, one, axisIds,
                               layout);
}

std::optional<llvm::StringRef>
partialFinalizeInstruction(riscv::PartialSetType input,
                           int64_t reductionAxis) {
  auto partial = input.getPartialType();
  auto axis = llvm::find(partial.getAxisIds().asArrayRef(), reductionAxis);
  if (axis == partial.getAxisIds().asArrayRef().end())
    return std::nullopt;
  auto lanes = riscv::rvvLaneCount(partial);
  if (!lanes)
    return std::nullopt;
  return *lanes == 1
             ? llvm::StringRef("rvv.partial-finalize.extract")
             : llvm::StringRef("rvv.partial-finalize.reduce");
}

bool isIntegerZero(mlir::Value value) {
  auto constant = value.getDefiningOp<riscv::ConstantOp>();
  if (!constant)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return integer && integer.getValue().isZero();
}

bool collectPartialAddTree(
    mlir::Value value, riscv::BinaryOp root,
    llvm::SmallVectorImpl<riscv::BinaryOp> &adds,
    llvm::SmallVectorImpl<PartialAddLeaf> &leaves) {
  // A local selection never searches an unbounded arithmetic graph.
  if (adds.size() >= 32 || leaves.size() >= 32)
    return false;
  if (auto add = value.getDefiningOp<riscv::BinaryOp>()) {
    if (add.getKind() != "add" ||
        (add != root && !add.getResult().hasOneUse()))
      return false;
    adds.push_back(add);
    return collectPartialAddTree(add.getLhs(), root, adds, leaves) &&
           collectPartialAddTree(add.getRhs(), root, adds, leaves);
  }
  if (isIntegerZero(value))
    return true;
  auto finalize = value.getDefiningOp<riscv::RVVPartialFinalizeOp>();
  if (finalize) {
    auto input = finalize.getInput().getType();
    if (!finalize.getResult().hasOneUse() ||
        finalize.getResult().getType() != root.getResult().getType() ||
        input.getSlots() != 1 ||
        input.getReductionAxis() != finalize.getReductionAxis())
      return false;
    leaves.push_back(PartialAddLeaf{finalize, {}});
    return true;
  }
  auto dot = value.getDefiningOp<riscv::RVVWidenDotOp>();
  if (!dot || !dot.getResult().hasOneUse() ||
      dot.getResult().getType() != root.getResult().getType() ||
      dot.getOver().size() != 1)
    return false;
  leaves.push_back(PartialAddLeaf{{}, dot});
  return true;
}

std::optional<riscv::PartialSetType>
partialTypeForAddLeaf(mlir::Builder &builder, PartialAddLeaf leaf) {
  if (leaf.finalize)
    return leaf.finalize.getInput().getType();
  if (!leaf.dot)
    return std::nullopt;
  const int64_t reductionAxis = leaf.dot.getOver()[0];
  riscv::ValueType lhs = leaf.dot.getLhs().getType();
  riscv::ValueType rhs = leaf.dot.getRhs().getType();
  auto lhsStreams = riscv_internal::staticProduct(
      lhs.getLayout().getTimeFactors().asArrayRef());
  auto rhsStreams = riscv_internal::staticProduct(
      rhs.getLayout().getTimeFactors().asArrayRef());
  auto layoutPlan = leaf.dot.getPartialLayoutPlanAttr();
  auto slot = layoutPlan
                  ? mlir::dyn_cast<riscv::ValueType>(
                        layoutPlan.getPartialSlotType())
                  : riscv::ValueType();
  auto result = mlir::dyn_cast<mlir::IntegerType>(leaf.dot.getResult().getType());
  if (!lhsStreams || !rhsStreams || *lhsStreams != 1 || *rhsStreams != 1 ||
      !slot || !result || !result.isSigned() ||
      result.getWidth() != 32)
    return std::nullopt;
  return riscv::PartialSetType::get(
      builder.getContext(), slot, reductionAxis, 1,
      slot.getShape()[*axisPosition(slot, reductionAxis)],
      slot.getLayout().getRegisterGroups());
}

std::optional<int64_t> partialSplitFactor(riscv::PartialSetType source,
                                          riscv::PartialSetType target) {
  if (source.getSlots() != 1 || target.getSlots() != 1 ||
      source.getReductionAxis() != target.getReductionAxis() ||
      source.getPartialType().getElementType() !=
          target.getPartialType().getElementType() ||
      source.getPartialType().getAxisIds() !=
          target.getPartialType().getAxisIds())
    return std::nullopt;
  riscv::ValueType sourcePartial = source.getPartialType();
  riscv::ValueType targetPartial = target.getPartialType();
  auto sourcePosition =
      axisPosition(sourcePartial, source.getReductionAxis());
  auto targetPosition =
      axisPosition(targetPartial, target.getReductionAxis());
  if (!sourcePosition || !targetPosition ||
      *sourcePosition != *targetPosition ||
      sourcePartial.getShape().size() != targetPartial.getShape().size() ||
      sourcePartial.getLayout().getCarrier() != "rvv" ||
      targetPartial.getLayout().getCarrier() != "rvv" ||
      sourcePartial.getLayout().getSew() !=
          targetPartial.getLayout().getSew())
    return std::nullopt;
  const int64_t sourceLanes =
      sourcePartial.getLayout().getLaneFactors()[*sourcePosition];
  const int64_t targetLanes =
      targetPartial.getLayout().getLaneFactors()[*targetPosition];
  if (sourceLanes <= 0 || targetLanes <= 0 || sourceLanes % targetLanes)
    return std::nullopt;
  const int64_t split = sourceLanes / targetLanes;
  if (split <= 0 || source.getTermsPerSlot() % split ||
      sourcePartial.getLayout().getLmulEighths() !=
          targetPartial.getLayout().getLmulEighths() * split ||
      sourcePartial.getLayout().getVl() !=
          targetPartial.getLayout().getVl() * split)
    return std::nullopt;
  for (size_t position = 0; position < sourcePartial.getShape().size();
       ++position) {
    if (position == *sourcePosition) {
      if (sourcePartial.getShape()[position] !=
              targetPartial.getShape()[position] * split ||
          sourcePartial.getLayout().getTimeFactors()[position] != 1 ||
          targetPartial.getLayout().getTimeFactors()[position] != 1 ||
          sourcePartial.getLayout().getReplicaFactors()[position] != 1 ||
          targetPartial.getLayout().getReplicaFactors()[position] != 1)
        return std::nullopt;
      continue;
    }
    if (sourcePartial.getShape()[position] !=
            targetPartial.getShape()[position] ||
        sourcePartial.getLayout().getTimeFactors()[position] !=
            targetPartial.getLayout().getTimeFactors()[position] ||
        sourcePartial.getLayout().getLaneFactors()[position] !=
            targetPartial.getLayout().getLaneFactors()[position] ||
        sourcePartial.getLayout().getReplicaFactors()[position] !=
            targetPartial.getLayout().getReplicaFactors()[position] ||
        sourcePartial.getLayout().getFragmentFactors()[position] !=
            targetPartial.getLayout().getFragmentFactors()[position] ||
        sourcePartial.getLayout().getLocalFactors()[position] !=
            targetPartial.getLayout().getLocalFactors()[position])
      return std::nullopt;
  }
  return split;
}

std::optional<int64_t> projectReplica(riscv::ValueType source,
                                      riscv::ValueType result,
                                      int64_t resultPart) {
  auto resultParts = riscv_internal::staticProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  auto sourceParts = riscv_internal::staticProduct(
      source.getLayout().getReplicaFactors().asArrayRef());
  if (!resultParts || !sourceParts || resultPart < 0 ||
      resultPart >= *resultParts)
    return std::nullopt;
  llvm::DenseMap<int64_t, int64_t> coordinates;
  int64_t remaining = resultPart;
  for (int64_t position = static_cast<int64_t>(result.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const int64_t factor = result.getLayout().getReplicaFactors()[position];
    if (factor <= 0)
      return std::nullopt;
    coordinates[result.getAxisIds()[position]] = remaining % factor;
    remaining /= factor;
  }
  int64_t sourcePart = 0;
  for (auto [axis, factor] :
       llvm::zip(source.getAxisIds().asArrayRef(),
                 source.getLayout().getReplicaFactors().asArrayRef())) {
    if (factor <= 0)
      return std::nullopt;
    int64_t coordinate = 0;
    if (factor > 1) {
      auto found = coordinates.find(axis);
      if (found == coordinates.end() || found->second >= factor)
        return std::nullopt;
      coordinate = found->second;
    }
    sourcePart = sourcePart * factor + coordinate;
  }
  return sourcePart < *sourceParts ? std::optional<int64_t>(sourcePart)
                                   : std::nullopt;
}

std::optional<int64_t> composeReplicaPart(
    riscv::ValueType value, llvm::ArrayRef<int64_t> outerAxes,
    int64_t outerPart, llvm::ArrayRef<int64_t> innerAxes, int64_t innerPart) {
  llvm::DenseMap<int64_t, int64_t> coordinates;
  auto decode = [&](llvm::ArrayRef<int64_t> axes,
                    int64_t part) -> mlir::LogicalResult {
    for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
         position >= 0; --position) {
      auto axis = axisPosition(value, axes[position]);
      if (!axis)
        return mlir::failure();
      const int64_t factor = value.getLayout().getReplicaFactors()[*axis];
      if (factor <= 0)
        return mlir::failure();
      coordinates[axes[position]] = part % factor;
      part /= factor;
    }
    return part == 0 ? mlir::success() : mlir::failure();
  };
  if (mlir::failed(decode(outerAxes, outerPart)) ||
      mlir::failed(decode(innerAxes, innerPart)))
    return std::nullopt;
  int64_t result = 0;
  for (auto [axis, factor] :
       llvm::zip(value.getAxisIds().asArrayRef(),
                 value.getLayout().getReplicaFactors().asArrayRef())) {
    auto found = coordinates.find(axis);
    const int64_t coordinate = found == coordinates.end() ? 0 : found->second;
    if (factor <= 0 || coordinate < 0 || coordinate >= factor)
      return std::nullopt;
    result = result * factor + coordinate;
  }
  return result;
}

void eraseDeadChain(mlir::Value value,
                    const llvm::DenseSet<mlir::Value> &stops,
                    llvm::DenseSet<mlir::Operation *> &visited,
                    mlir::IRRewriter &rewriter) {
  if (stops.contains(value))
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  llvm::SmallVector<mlir::Value> operands(definition->getOperands());
  if (llvm::all_of(definition->getResults(), [](mlir::Value result) {
        return result.use_empty();
      }) &&
      mlir::isa<riscv::RVVLayeredStreamOp,
                riscv::RVVProjectedLayeredStreamOp, riscv::ExtractOp,
                riscv::RVVUnitEntryWindowLoadOp,
                riscv::RVVIndexedEntryLoadOp, riscv::IotaOp,
                riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                riscv::NarrowOp, riscv::WidenOp,
                riscv::RVVWidenAddOp,
                riscv::RVVWidenMultiplyOp,
                riscv::RVVWidenScalarMultiplyOp, riscv::RVVWidenDotOp,
                riscv::RVVRegularRepeatIndexOp,
                riscv::RVVRegularRepeatGatherOp,
                riscv::ConvertLayoutOp,
                riscv::RegisterMaterializeOp>(definition)) {
    rewriter.eraseOp(definition);
    for (mlir::Value operand : operands)
      eraseDeadChain(operand, stops, visited, rewriter);
  }
}

bool isDeadChainCandidate(mlir::Operation *operation) {
  return mlir::isa<riscv::RVVLayeredStreamOp,
                   riscv::RVVProjectedLayeredStreamOp, riscv::ExtractOp,
                   riscv::RVVUnitEntryWindowLoadOp,
                   riscv::RVVIndexedEntryLoadOp, riscv::IotaOp,
                   riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                   riscv::NarrowOp, riscv::WidenOp,
                   riscv::ReduceOp,
                   riscv::RVVWidenAddOp,
                   riscv::RVVWidenMultiplyOp,
                   riscv::RVVWidenScalarMultiplyOp, riscv::RVVWidenDotOp,
                   riscv::ConvertLayoutOp, riscv::RVVAxisBroadcastOp,
                   riscv::RVVRegularRepeatIndexOp,
                   riscv::RVVRegularRepeatGatherOp,
                   riscv::RegisterMaterializeOp>(operation);
}

void collectDeadChainCandidates(
    mlir::Value value, const llvm::DenseSet<mlir::Value> &stops,
    llvm::DenseSet<mlir::Operation *> &candidates) {
  if (stops.contains(value))
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !isDeadChainCandidate(definition) ||
      !candidates.insert(definition).second)
    return;
  for (mlir::Value operand : definition->getOperands())
    collectDeadChainCandidates(operand, stops, candidates);
}

void sweepDeadChainCandidates(
    llvm::DenseSet<mlir::Operation *> &candidates,
    mlir::IRRewriter &rewriter) {
  while (true) {
    mlir::Operation *dead = nullptr;
    for (mlir::Operation *candidate : candidates) {
      if (llvm::all_of(candidate->getResults(),
                       [](mlir::Value result) { return result.use_empty(); })) {
        dead = candidate;
        break;
      }
    }
    if (!dead)
      return;
    candidates.erase(dead);
    rewriter.eraseOp(dead);
  }
}

riscv::ValueType scalarReplicaType(mlir::Builder &builder,
                                   riscv::ValueType source) {
  llvm::SmallVector<int64_t> one(source.getShape().size(), 1);
  llvm::SmallVector<int64_t> replicas;
  for (auto [position, extent] :
       llvm::enumerate(source.getShape().asArrayRef())) {
    const int64_t replicasForAxis =
        extent > 0 ? extent
                   : source.getLayout().getReplicaFactors()[position];
    if (replicasForAxis <= 0)
      return {};
    replicas.push_back(replicasForAxis);
  }
  mlir::Type element = source.getElementType();
  unsigned width = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
    width = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(element))
    width = floating.getWidth();
  if (!width)
    return {};
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "scalar", source.getAxisIds(),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), std::max<unsigned>(8, width), 0, 1,
      0, source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), element, source.getShape(),
                               source.getAxisIds(), layout);
}

mlir::FailureOr<mlir::Value> rematerializeScalarReplicas(
    mlir::Value value, mlir::IRRewriter &rewriter,
    llvm::DenseMap<mlir::Value, mlir::Value> &memo) {
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!sourceType)
    return value;
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;
  bool alreadyScalar = sourceType.getLayout().getCarrier() == "scalar";
  for (int64_t factor :
       sourceType.getLayout().getTimeFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  for (int64_t factor :
       sourceType.getLayout().getLaneFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  if (alreadyScalar) {
    memo.try_emplace(value, value);
    return value;
  }

  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return mlir::failure();
    auto rematerialized = rematerializeScalarReplicas(conversion.getInput(),
                                                      rewriter, memo);
    if (mlir::failed(rematerialized))
      return mlir::failure();
    memo.try_emplace(value, *rematerialized);
    return *rematerialized;
  }

  mlir::Operation *producer = value.getDefiningOp();
  if (auto gather =
          mlir::dyn_cast_or_null<riscv::RVVRegularRepeatGatherOp>(producer)) {
    auto targetType = scalarReplicaType(rewriter, sourceType);
    auto replicas =
        targetType
            ? riscv_internal::staticProduct(
                  targetType.getLayout().getReplicaFactors().asArrayRef())
            : std::optional<int64_t>();
    auto bases = gather.getPartBases().size() == 1
                     ? mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                           gather.getPartBases()[0])
                     : mlir::DenseI64ArrayAttr();
    if (!targetType || !replicas || gather.getResults().size() != 1 ||
        gather.getIndices().size() != 1 || !bases || bases.size() != 1 ||
        bases[0] != 0 || *replicas != gather.getSourceCount() * gather.getRepeat())
      return mlir::failure();
    llvm::SmallVector<int64_t> partBases;
    partBases.reserve(*replicas);
    for (int64_t replica = 0; replica < *replicas; ++replica)
      partBases.push_back(replica / gather.getRepeat());
    auto scalarLoad = rewriter.create<riscv::RVVRegularRepeatScalarLoadOp>(
        gather.getLoc(), targetType, gather.getField(), gather.getSourceAxis(),
        gather.getReductionAxis(), gather.getSourceBase(),
        gather.getSourceCount(), gather.getRepeat(),
        rewriter.getDenseI64ArrayAttr(partBases), gather.getAccess(),
        riscv_internal::leaf(
            rewriter, "scalar", "regular-repeat-scalar-load",
            "scalar.regular-repeat-load", "scalar.regular-repeat-load", 0, 0,
            0, 0, "none", "exact",
            {static_cast<int64_t>(gather.getSourceAxis()),
             static_cast<int64_t>(gather.getReductionAxis()),
             static_cast<int64_t>(gather.getSourceCount()),
             static_cast<int64_t>(gather.getRepeat())}));
    riscv_internal::copyOrigin(gather, scalarLoad);
    memo.try_emplace(value, scalarLoad.getResult());
    return scalarLoad.getResult();
  }
  if (!producer || !mlir::isa<riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp,
                              riscv::CompareOp, riscv::CastOp,
                              riscv::NarrowOp, riscv::WidenOp>(producer) ||
      producer->getNumResults() != 1)
    return mlir::failure();
  riscv::ValueType targetType = scalarReplicaType(rewriter, sourceType);
  if (!targetType)
    return mlir::failure();
  llvm::SmallVector<mlir::Value> operands;
  for (mlir::Value operand : producer->getOperands()) {
    auto rematerialized = rematerializeScalarReplicas(operand, rewriter, memo);
    if (mlir::failed(rematerialized))
      return mlir::failure();
    operands.push_back(*rematerialized);
  }
  mlir::OperationState state(producer->getLoc(), producer->getName());
  state.addOperands(operands);
  state.addTypes(targetType);
  for (mlir::NamedAttribute attribute : producer->getAttrs())
    if (attribute.getName() != "leaf")
      state.addAttribute(attribute.getName(), attribute.getValue());
  state.addAttribute("leaf", riscv_internal::unselectedLeaf(rewriter));
  mlir::Operation *clone = rewriter.create(state);
  mlir::Value result = clone->getResult(0);
  memo.try_emplace(value, result);
  return result;
}

bool supportsScalarReplicaRematerialization(
    mlir::Value value, llvm::DenseSet<mlir::Operation *> &visited) {
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!sourceType)
    return true;
  bool alreadyScalar = sourceType.getLayout().getCarrier() == "scalar";
  for (int64_t factor :
       sourceType.getLayout().getTimeFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  for (int64_t factor :
       sourceType.getLayout().getLaneFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  if (alreadyScalar)
    return true;

  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || visited.size() >= 32 || !visited.insert(producer).second)
    return false;
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(producer))
    return conversion.getConversion().getEffect() == "pure" &&
           supportsScalarReplicaRematerialization(conversion.getInput(),
                                                  visited);
  if (auto gather = mlir::dyn_cast<riscv::RVVRegularRepeatGatherOp>(producer)) {
    auto bases = gather.getPartBases().size() == 1
                     ? mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                           gather.getPartBases()[0])
                     : mlir::DenseI64ArrayAttr();
    return gather.getResults().size() == 1 && gather.getIndices().size() == 1 &&
           bases && bases.size() == 1 && bases[0] == 0;
  }
  if (!mlir::isa<riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                 riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(producer) ||
      producer->getNumResults() != 1)
    return false;
  return llvm::all_of(producer->getOperands(), [&](mlir::Value operand) {
    return supportsScalarReplicaRematerialization(operand, visited);
  });
}

std::optional<mlir::Value> narrowScaleSource(mlir::Value value) {
  mlir::Value cursor = value;
  while (auto conversion = cursor.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return std::nullopt;
    cursor = conversion.getInput();
  }
  auto widen = cursor.getDefiningOp<riscv::WidenOp>();
  if (!widen)
    return std::nullopt;
  auto source = mlir::dyn_cast<riscv::ValueType>(widen.getInput().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(widen.getResult().getType());
  auto sourceElement =
      source ? mlir::dyn_cast<mlir::IntegerType>(source.getElementType())
             : mlir::IntegerType();
  auto resultElement =
      result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
             : mlir::IntegerType();
  if (!source || !result || !sourceElement || !sourceElement.isSigned() ||
      sourceElement.getWidth() <= 0 || sourceElement.getWidth() > 16 ||
      !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 ||
      source.getLayout().getCarrier() != "scalar" ||
      result.getLayout().getCarrier() != "scalar" ||
      source.getShape() != result.getShape() ||
      source.getAxisIds() != result.getAxisIds() ||
      source.getLayout().getTimeFactors() !=
          result.getLayout().getTimeFactors() ||
      source.getLayout().getLaneFactors() !=
          result.getLayout().getLaneFactors() ||
      source.getLayout().getReplicaFactors() !=
          result.getLayout().getReplicaFactors())
    return std::nullopt;
  return widen.getInput();
}

riscv::ValueType plannedNarrowScaleType(mlir::Builder &builder,
                                        mlir::Value value,
                                        riscv::ValueType scalarScaleType) {
  if (!scalarScaleType ||
      scalarScaleType.getLayout().getCarrier() != "scalar")
    return {};
  unsigned width = 0;
  if (auto existing = narrowScaleSource(value)) {
    auto existingType = mlir::dyn_cast<riscv::ValueType>(existing->getType());
    auto element = existingType
                       ? mlir::dyn_cast<mlir::IntegerType>(
                             existingType.getElementType())
                       : mlir::IntegerType();
    if (element && element.isSigned() && element.getWidth() <= 16)
      width = element.getWidth();
  } else if (auto range = riscv_internal::integerRange(value);
             range && range->minimum >= std::numeric_limits<int16_t>::min() &&
             range->maximum <= std::numeric_limits<int16_t>::max()) {
    width = 16;
  }
  if (!width)
    return {};
  auto element = mlir::IntegerType::get(builder.getContext(), width,
                                        mlir::IntegerType::Signed);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "scalar", scalarScaleType.getAxisIds(),
      scalarScaleType.getLayout().getTimeFactors(),
      scalarScaleType.getLayout().getLaneFactors(),
      scalarScaleType.getLayout().getReplicaFactors(),
      scalarScaleType.getLayout().getFragmentFactors(),
      scalarScaleType.getLayout().getLocalFactors(),
      std::max<unsigned>(8, width), 0, 1, 0,
      scalarScaleType.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), element,
                               scalarScaleType.getShape(),
                               scalarScaleType.getAxisIds(), layout);
}

std::optional<mlir::Value>
materializeExactNarrowScale(mlir::Value value, mlir::IRRewriter &rewriter) {
  if (auto existing = narrowScaleSource(value))
    return existing;
  auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto range = riscv_internal::integerRange(value);
  if (!source || source.getLayout().getCarrier() != "scalar" || !range ||
      range->minimum < std::numeric_limits<int16_t>::min() ||
      range->maximum > std::numeric_limits<int16_t>::max())
    return std::nullopt;
  auto element = mlir::IntegerType::get(rewriter.getContext(), 16,
                                        mlir::IntegerType::Signed);
  auto layout = riscv::LayoutAttr::get(
      rewriter.getContext(), "scalar", source.getLayout().getAxisIds(),
      source.getLayout().getTimeFactors(), source.getLayout().getLaneFactors(),
      source.getLayout().getReplicaFactors(),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), 16, 0, 1, 0,
      source.getLayout().getValidity());
  auto target = riscv::ValueType::get(rewriter.getContext(), element,
                                      source.getShape(), source.getAxisIds(),
                                      layout);
  auto narrow = rewriter.create<riscv::NarrowOp>(
      value.getLoc(), target, value, "none", false,
      riscv_internal::unselectedLeaf(rewriter));
  if (mlir::Operation *definition = value.getDefiningOp())
    riscv_internal::copyOrigin(definition, narrow);
  return narrow.getResult();
}

std::optional<ScaledPartialContribution>
matchScaledPartialContribution(mlir::Value value) {
  auto add = value.getDefiningOp<riscv::BinaryOp>();
  if (!add || add.getKind() != "add" || !add.getResult().hasOneUse())
    return std::nullopt;

  auto matchSide = [&](mlir::Value contribution,
                       mlir::Value previous)
      -> std::optional<ScaledPartialContribution> {
    auto multiply = contribution.getDefiningOp<riscv::BinaryOp>();
    if (!multiply || multiply.getKind() != "mul")
      return std::nullopt;
    auto lhsDot = multiply.getLhs().getDefiningOp<riscv::RVVWidenDotOp>();
    auto rhsDot = multiply.getRhs().getDefiningOp<riscv::RVVWidenDotOp>();
    if (static_cast<bool>(lhsDot) == static_cast<bool>(rhsDot))
      return std::nullopt;
    riscv::RVVWidenDotOp dot = lhsDot ? lhsDot : rhsDot;
    mlir::Value scale = lhsDot ? multiply.getRhs() : multiply.getLhs();
    auto scaleType = mlir::dyn_cast<riscv::ValueType>(scale.getType());
    auto scalarInteger = mlir::dyn_cast<mlir::IntegerType>(scale.getType());
    auto scaleParts = scaleType
                          ? riscv_internal::staticProduct(
                                scaleType.getLayout()
                                    .getReplicaFactors()
                                    .asArrayRef())
                          : scalarInteger && scalarInteger.isSigned() &&
                                    scalarInteger.getWidth() == 32
                                ? std::optional<int64_t>(1)
                                : std::optional<int64_t>();
    auto scaleElement =
        scaleType
            ? mlir::dyn_cast<mlir::IntegerType>(scaleType.getElementType())
            : scalarInteger;
    const bool scalarCarrier =
        scaleType ? scaleType.getLayout().getCarrier() == "scalar"
                  : static_cast<bool>(scalarInteger);
    if (!scalarCarrier ||
        !scaleParts || *scaleParts <= 0 || !scaleElement ||
        !scaleElement.isSigned() || scaleElement.getWidth() != 32 ||
        dot.getPartialUnroll() <= 1 || dot.getOver().size() != 1 ||
        !dot.getResult().hasOneUse() || !multiply.getResult().hasOneUse())
      return std::nullopt;
    return ScaledPartialContribution{dot, multiply, add, scale, previous};
  };

  auto lhs = matchSide(add.getLhs(), add.getRhs());
  auto rhs = matchSide(add.getRhs(), add.getLhs());
  if (static_cast<bool>(lhs) == static_cast<bool>(rhs))
    return std::nullopt;
  return lhs ? lhs : rhs;
}

std::optional<llvm::SmallVector<ScaledPartialContribution>>
matchLevelScaledLoop(mlir::scf::ForOp loop) {
  if (loop.getInitArgs().size() != 1 || loop.getNumResults() != 1)
    return std::nullopt;
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  mlir::Value cursor = yield.getOperand(0);
  auto first = matchScaledPartialContribution(cursor);
  if (!first)
    return std::nullopt;
  const int64_t slots = first->dot.getPartialUnroll();
  if (slots <= 1)
    return std::nullopt;
  llvm::SmallVector<ScaledPartialContribution> contributions;
  while (static_cast<int64_t>(contributions.size()) < slots) {
    auto contribution = matchScaledPartialContribution(cursor);
    if (!contribution)
      break;
    contributions.push_back(*contribution);
    cursor = contribution->previous;
  }
  if (static_cast<int64_t>(contributions.size()) != slots ||
      cursor != loop.getRegionIterArg(0))
    return std::nullopt;
  std::reverse(contributions.begin(), contributions.end());
  return contributions;
}

std::optional<ScaledPartialContribution>
matchLevelScaledLoopSeed(mlir::scf::ForOp loop) {
  if (loop.getInitArgs().size() != 1 || loop.getNumResults() != 1)
    return std::nullopt;
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto contribution = matchScaledPartialContribution(yield.getOperand(0));
  if (!contribution || contribution->previous != loop.getRegionIterArg(0))
    return std::nullopt;
  auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
      "weft.riscv.unroll_factor");
  if (!factor || factor.getInt() != contribution->dot.getPartialUnroll() ||
      factor.getInt() <= 1)
    return std::nullopt;
  return contribution;
}

mlir::FailureOr<mlir::scf::ForOp>
expandPlannedLevelScaledIssueLoop(mlir::scf::ForOp loop, int64_t factor,
                                  mlir::IRRewriter &rewriter) {
  auto lower = loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto upper = loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!lower || !upper || !step || factor <= 1 || step.value() <= 0 ||
      upper.value() < lower.value())
    return mlir::failure();
  const int64_t distance = upper.value() - lower.value();
  if (distance % step.value())
    return mlir::failure();
  const int64_t iterations = distance / step.value();
  if (iterations <= 0 || iterations % factor)
    return mlir::failure();

  loop->removeAttr("weft.riscv.unroll_factor");
  loop->removeAttr("weft.riscv.unroll_order");

  // The generic SCF unroller erases the loop when the selected factor equals
  // its exact trip count.  A level-scaled partial plan still needs one region
  // to own the cloned issue cohort until the typed PartialSet program replaces
  // it.  Keep that owner as a single-iteration loop and clone the remaining
  // issues into its body, threading the original iter_args mechanically.
  if (iterations == factor) {
    auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
        loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getInitArgs().size())
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> templateOperations;
    for (mlir::Operation &operation : loop.getBody()->without_terminator())
      templateOperations.push_back(&operation);
    llvm::SmallVector<mlir::Value> current(yield.getOperands().begin(),
                                           yield.getOperands().end());
    rewriter.setInsertionPoint(yield);
    for (int64_t iteration = 1; iteration < factor; ++iteration) {
      mlir::IRMapping mapping;
      auto induction = rewriter.create<mlir::arith::ConstantIndexOp>(
          loop.getLoc(), lower.value() + iteration * step.value());
      mapping.map(loop.getInductionVar(), induction.getResult());
      for (auto [regionArgument, value] :
           llvm::zip(loop.getRegionIterArgs(), current))
        mapping.map(regionArgument, value);
      for (mlir::Operation *operation : templateOperations)
        rewriter.clone(*operation, mapping);

      llvm::SmallVector<mlir::Value> next;
      next.reserve(yield.getNumOperands());
      for (mlir::Value value : yield.getOperands())
        next.push_back(mapping.lookupOrDefault(value));
      current = std::move(next);
    }
    rewriter.setInsertionPointToStart(loop.getBody());
    auto firstInduction = rewriter.create<mlir::arith::ConstantIndexOp>(
        loop.getLoc(), lower.value());
    loop.getInductionVar().replaceAllUsesWith(firstInduction.getResult());
    rewriter.modifyOpInPlace(yield, [&] { yield->setOperands(current); });
    rewriter.setInsertionPoint(loop);
    auto singleIterationUpper = rewriter.create<mlir::arith::ConstantIndexOp>(
        loop.getLoc(), lower.value() + step.value());
    rewriter.modifyOpInPlace(loop, [&] {
      loop.getUpperBoundMutable().assign(singleIterationUpper.getResult());
    });
    return loop;
  }

  auto unrolled = mlir::loopUnrollByFactor(loop, factor);
  if (mlir::failed(unrolled) || !(*unrolled).mainLoopOp)
    return mlir::failure();
  return *(*unrolled).mainLoopOp;
}

std::optional<ReplicaScaledDotReduction>
matchReplicaScaledDotReduction(riscv::ReduceOp reduce) {
  llvm::SmallVector<riscv::ReduceOp> reductions;
  llvm::SmallVector<riscv::ConvertLayoutOp> reductionConversions;
  llvm::SmallVector<int64_t> reducedAxes;
  auto stripPureConversions = [&](mlir::Value value) {
    while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      if (conversion.getConversion().getEffect() != "pure" ||
          !conversion.getResult().hasOneUse())
        break;
      reductionConversions.push_back(conversion);
      value = conversion.getInput();
    }
    return value;
  };
  riscv::ReduceOp current = reduce;
  mlir::Value multipliedInput;
  while (current) {
    if (current.getKind() != "add")
      return std::nullopt;
    auto inputType = mlir::dyn_cast<riscv::ValueType>(current.getInput().getType());
    if (!inputType || current.getAxis() < 0 ||
        static_cast<size_t>(current.getAxis()) >= inputType.getAxisIds().size())
      return std::nullopt;
    reductions.push_back(current);
    reducedAxes.push_back(inputType.getAxisIds()[current.getAxis()]);
    mlir::Value reducedInput = stripPureConversions(current.getInput());
    auto inner = reducedInput.getDefiningOp<riscv::ReduceOp>();
    if (!inner) {
      multipliedInput = reducedInput;
      break;
    }
    if (!reducedInput.hasOneUse())
      return std::nullopt;
    current = inner;
  }
  auto multiply = multipliedInput.getDefiningOp<riscv::BinaryOp>();
  if (!multiply || multiply.getKind() != "mul" ||
      !multiply.getResult().hasOneUse())
    return std::nullopt;
  auto findDot = [](mlir::Value value) -> riscv::RVVWidenDotOp {
    while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      if (conversion.getConversion().getEffect() != "pure")
        return {};
      value = conversion.getInput();
    }
    return value.getDefiningOp<riscv::RVVWidenDotOp>();
  };
  auto lhsDot = findDot(multiply.getLhs());
  auto rhsDot = findDot(multiply.getRhs());
  if (static_cast<bool>(lhsDot) == static_cast<bool>(rhsDot))
    return std::nullopt;
  riscv::RVVWidenDotOp dot = lhsDot ? lhsDot : rhsDot;
  mlir::Value dotSide = lhsDot ? multiply.getLhs() : multiply.getRhs();
  mlir::Value scale = lhsDot ? multiply.getRhs() : multiply.getLhs();
  return ReplicaScaledDotReduction{std::move(reductions),
                                   std::move(reductionConversions),
                                   std::move(reducedAxes), multiply, dot, dotSide,
                                   scale};
}

int64_t replicaProductForAxes(riscv::ValueType value,
                              llvm::ArrayRef<int64_t> selectedAxes) {
  int64_t result = 1;
  for (auto [axis, factor] :
       llvm::zip(value.getAxisIds().asArrayRef(),
                 value.getLayout().getReplicaFactors().asArrayRef())) {
    if (!llvm::is_contained(selectedAxes, axis))
      continue;
    if (factor <= 0 || result > std::numeric_limits<int64_t>::max() / factor)
      return -1;
    result *= factor;
  }
  return result;
}

int64_t replicaProduct(riscv::ValueType value) {
  return replicaProductForAxes(value, value.getAxisIds().asArrayRef());
}

llvm::SmallVector<int64_t>
remainingAxes(riscv::ValueType value, llvm::ArrayRef<int64_t> removed) {
  llvm::SmallVector<int64_t> result;
  for (int64_t axis : value.getAxisIds().asArrayRef())
    if (!llvm::is_contained(removed, axis))
      result.push_back(axis);
  return result;
}

riscv::PartialTopologyAttr makePartialTopology(
    mlir::Builder &builder, riscv::RVVWidenDotOp dot, llvm::StringRef kind,
    int64_t rootOperand, llvm::ArrayRef<int64_t> partialAxes,
    llvm::ArrayRef<int64_t> outputAxes, int64_t sourceSlots,
    int64_t partialSlots, int64_t outputReplicas, int64_t laneSplit,
    int64_t combineArity, int64_t resourceGroups) {
  llvm::SmallVector<int64_t> slotOrder;
  const bool chainOrdered =
      kind == "reduced_scaled" || (kind == "scaled" && laneSplit > 1);
  if (chainOrdered && laneSplit > 1 &&
      sourceSlots * laneSplit == partialSlots &&
      combineArity == sourceSlots)
    for (int64_t source = 0; source < sourceSlots; ++source)
      for (int64_t lane = 0; lane < laneSplit; ++lane)
        slotOrder.push_back(source * laneSplit + lane);
  else if (chainOrdered)
    for (int64_t chain = 0; chain < combineArity; ++chain)
      for (int64_t slot = chain; slot < partialSlots; slot += combineArity)
        slotOrder.push_back(slot);
  else
    for (int64_t slot = 0; slot < partialSlots; ++slot)
      slotOrder.push_back(slot);
  return riscv::PartialTopologyAttr::get(
      builder.getContext(), kind, rootOperand,
      builder.getDenseI64ArrayAttr(partialAxes),
      builder.getDenseI64ArrayAttr(outputAxes), sourceSlots, partialSlots,
      outputReplicas, laneSplit, combineArity,
      builder.getDenseI64ArrayAttr(slotOrder), resourceGroups);
}

std::optional<LayeredTopologyFacts>
analyzeLayeredTopology(mlir::Builder &builder, riscv::RVVWidenDotOp dot) {
  if (dot.getOver().size() != 1)
    return std::nullopt;
  LayeredTopologyFacts facts;
  facts.reductionAxis = dot.getOver()[0];
  facts.lhsType = dot.getLhs().getType();
  facts.rhsType = dot.getRhs().getType();
  auto lhsPosition = axisPosition(facts.lhsType, facts.reductionAxis);
  auto rhsPosition = axisPosition(facts.rhsType, facts.reductionAxis);
  if (!lhsPosition || !rhsPosition)
    return std::nullopt;
  facts.streams =
      facts.lhsType.getLayout().getTimeFactors()[*lhsPosition];
  if (facts.streams <= 1 ||
      facts.streams !=
          facts.rhsType.getLayout().getTimeFactors()[*rhsPosition])
    return std::nullopt;

  llvm::DenseSet<mlir::Operation *> lhsVisited;
  llvm::DenseSet<mlir::Operation *> rhsVisited;
  llvm::SmallVector<ProjectedRoot> lhsRoots;
  llvm::SmallVector<ProjectedRoot> rhsRoots;
  collectProjectedRoots(dot.getLhs(), facts.reductionAxis, lhsVisited, lhsRoots);
  collectProjectedRoots(dot.getRhs(), facts.reductionAxis, rhsVisited, rhsRoots);
  const int64_t lhsLayered = llvm::count_if(
      lhsRoots, [](const ProjectedRoot &root) { return root.layered; });
  const int64_t rhsLayered = llvm::count_if(
      rhsRoots, [](const ProjectedRoot &root) { return root.layered; });
  if (lhsLayered + rhsLayered != 1)
    return std::nullopt;
  facts.rootOperand = lhsLayered == 1 ? 0 : 1;
  facts.roots.assign(lhsRoots.begin(), lhsRoots.end());
  facts.roots.append(rhsRoots.begin(), rhsRoots.end());
  auto layeredIt = llvm::find_if(
      facts.roots, [](const ProjectedRoot &root) { return root.layered; });
  if (layeredIt == facts.roots.end())
    return std::nullopt;
  facts.layered = *layeredIt;
  if (!facts.layered.field || !facts.layered.origin)
    return std::nullopt;

  riscv::AccessAttr access = facts.layered.access;
  riscv::LayeredStreamGeometryAttr geometry = facts.layered.geometry;
  if (!geometry || geometry.getAxis() != facts.reductionAxis ||
      geometry.getStreamCount() != facts.streams)
    return std::nullopt;
  facts.group = geometry.getGroupSize();
  facts.layerExtent = geometry.getLayerSize();
  facts.layers = facts.layerExtent > 0 ? facts.group / facts.layerExtent : 0;
  auto layeredInteger = mlir::dyn_cast<mlir::IntegerType>(
      mlir::cast<riscv::ValueType>(facts.layered.field.getResult().getType())
          .getElementType());
  auto layeredPosition = axisPosition(facts.layered.type, facts.reductionAxis);
  if (!layeredPosition)
    return std::nullopt;
  facts.lanes = geometry.getLaneCount();
  if (!layeredInteger || layeredInteger.isSigned() || facts.group <= 0 ||
      facts.layerExtent <= 0 || facts.group % facts.layerExtent ||
      facts.layers <= 1 || layeredInteger.getWidth() * facts.layers != 8 ||
      access.getBitOffset() % 8 ||
      (access.getOrder() != "lo_first" && access.getOrder() != "hi_first") ||
      facts.layered.origin.getResult().getType().getDomain().getTail() !=
          "exact" ||
      facts.lanes <= 1 ||
      facts.layered.type.getLayout().getLaneFactors()[*layeredPosition] !=
          facts.lanes ||
      facts.layerExtent % facts.lanes ||
      facts.streams % facts.layers || facts.streams < facts.layers)
    return std::nullopt;
  facts.windowsPerLayer = facts.layerExtent / facts.lanes;
  facts.windowCount = geometry.getGroupForWindow().size();
  if (facts.windowCount <= 0 ||
      facts.windowCount != facts.streams / facts.layers)
    return std::nullopt;
  facts.layeredWindowType =
      projectOneWindow(builder, facts.layered.type, facts.reductionAxis);
  bool completeRoots = static_cast<bool>(facts.layeredWindowType);
  for (ProjectedRoot &root : facts.roots) {
    auto position = axisPosition(root.type, facts.reductionAxis);
    auto fieldType =
        root.field
            ? mlir::dyn_cast<riscv::ValueType>(root.field.getResult().getType())
            : riscv::ValueType();
    auto fieldPosition = fieldType
                             ? axisPosition(fieldType, facts.reductionAxis)
                             : std::optional<size_t>();
    auto integer = fieldType
                       ? mlir::dyn_cast<mlir::IntegerType>(
                             fieldType.getElementType())
                       : mlir::IntegerType();
    auto window = projectOneWindow(builder, root.type, facts.reductionAxis);
    const bool bounded =
        position && fieldPosition && root.base >= 0 && root.stride > 0 &&
        root.repeat > 0 && root.extent == root.type.getShape()[*position] &&
        root.extent > 0 && fieldType.getShape()[*fieldPosition] > 0 &&
        root.base <= fieldType.getShape()[*fieldPosition] - 1 &&
        (root.extent - 1) / root.repeat <=
            (fieldType.getShape()[*fieldPosition] - 1 - root.base) /
                root.stride;
    completeRoots &=
        root.field && root.origin && integer && window && bounded &&
        root.origin.getResult().getType().getDomain().getTail() == "exact" &&
        root.type.getLayout().getTimeFactors()[*position] == facts.streams &&
        root.type.getLayout().getLaneFactors()[*position] == facts.lanes &&
        root.extent == facts.layered.extent &&
        (root.repeat % facts.lanes == 0 || facts.lanes % root.repeat == 0) &&
        (root.layered ||
         (root.access.getMapping() == "natural" &&
          root.access.getBitOffset() % 8 == 0 && integer.getWidth() >= 8 &&
          integer.getWidth() <= 32 && integer.getWidth() % 8 == 0));
    if (window)
      facts.windowTypes[root.value] = window;
  }
  facts.lhsIssueType =
      projectOneWindow(builder, facts.lhsType, facts.reductionAxis);
  facts.rhsIssueType =
      projectOneWindow(builder, facts.rhsType, facts.reductionAxis);
  facts.accumulatorType = partialType(builder, facts.lhsType,
                                      dot.getResult().getType(),
                                      facts.reductionAxis);
  auto expectedAccumulator =
      facts.lhsIssueType && facts.rhsIssueType
          ? replicaReducedAccumulatorType(builder, facts.lhsIssueType,
                                          facts.rhsIssueType,
                                          facts.reductionAxis)
          : riscv::ValueType();
  facts.partialInteger =
      facts.accumulatorType
          ? mlir::dyn_cast<mlir::IntegerType>(
                facts.accumulatorType.getElementType())
          : mlir::IntegerType();
  llvm::DenseMap<mlir::Value, mlir::Value> projectedRoots;
  for (const ProjectedRoot &root : facts.roots)
    projectedRoots[root.value] = root.value;
  llvm::DenseMap<mlir::Value, bool> dependence;
  llvm::DenseMap<mlir::Value, bool> projectable;
  const bool completeSlice =
      canProjectWindowSlice(dot.getLhs(), projectedRoots, facts.reductionAxis,
                            builder, dependence, projectable) &&
      canProjectWindowSlice(dot.getRhs(), projectedRoots, facts.reductionAxis,
                            builder, dependence, projectable);
  if (!completeRoots || !completeSlice || !facts.lhsIssueType ||
      !facts.rhsIssueType || !facts.accumulatorType ||
      facts.accumulatorType != expectedAccumulator ||
      !facts.partialInteger ||
      (facts.partialInteger.getWidth() != 16 &&
       facts.partialInteger.getWidth() != 32))
    return std::nullopt;
  return facts;
}

bool hasGappedIndexedWindowRoot(mlir::Value value, int64_t windowAxis,
                                llvm::DenseSet<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  if (auto load = mlir::dyn_cast<riscv::RVVIndexedEntryLoadOp>(definition)) {
    auto result = load.getResult().getType();
    auto position = axisPosition(result, windowAxis);
    const unsigned elementBits =
        riscv_internal::logicalBitWidth(result.getElementType());
    const int64_t payloadBytes =
        elementBits && elementBits % 8 == 0
            ? load.getPayloadExtent() * static_cast<int64_t>(elementBits / 8)
            : 0;
    if (position && payloadBytes > 0 &&
        load.getEntryByteStride() > payloadBytes &&
        result.getLayout().getTimeFactors()[*position] > 0 &&
        result.getLayout().getLaneFactors()[*position] > 1 &&
        result.getLayout().getReplicaFactors()[*position] == 1 &&
        result.getLayout().getFragmentFactors()[*position] == 1 &&
        result.getLayout().getLocalFactors()[*position] == 1)
      return true;
  }
  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
    if (result && extract.getIndices().size() == 1 &&
        riscv_internal::sourceField(extract.getInput())) {
      auto relation = riscv_internal::analyzeIndexedEntryRelation(
          extract.getIndices()[0], result, {}, {});
      auto position = axisPosition(result, windowAxis);
      if (relation && position &&
          relation->entryStride > relation->payloadExtent &&
          result.getLayout().getTimeFactors()[*position] > 0 &&
          result.getLayout().getLaneFactors()[*position] > 1 &&
          result.getLayout().getReplicaFactors()[*position] == 1 &&
          result.getLayout().getFragmentFactors()[*position] == 1 &&
          result.getLayout().getLocalFactors()[*position] == 1)
        return true;
    }
  }
  if (!mayMoveReadAcross(definition))
    return false;
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return hasGappedIndexedWindowRoot(operand, windowAxis, visited);
  });
}

bool hasGappedIndexedWindowRoot(mlir::Value value, int64_t windowAxis) {
  llvm::DenseSet<mlir::Operation *> visited;
  return hasGappedIndexedWindowRoot(value, windowAxis, visited);
}

bool hasIndexedLookupSupply(mlir::Value value,
                            llvm::DenseSet<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  if (mlir::isa<riscv::LookupOp, riscv::RVVIndexedEntryLoadOp>(definition))
    return true;
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return mlir::isa<riscv::ValueType>(operand.getType()) &&
           hasIndexedLookupSupply(operand, visited);
  });
}

bool hasIndexedLookupSupply(mlir::Value value) {
  llvm::DenseSet<mlir::Operation *> visited;
  return hasIndexedLookupSupply(value, visited);
}

void collectConcreteFieldSupplies(
    mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &supplies,
    llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  if (mlir::isa<riscv::FieldOp>(definition)) {
    supplies.insert(definition);
    return;
  }
  if (definition->getNumRegions() != 0)
    return;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      collectConcreteFieldSupplies(operand, supplies, visited);
}

bool sharesConcreteFieldSupply(mlir::Value lhs, mlir::Value rhs) {
  llvm::SmallPtrSet<mlir::Operation *, 8> lhsSupplies;
  llvm::SmallPtrSet<mlir::Operation *, 8> rhsSupplies;
  llvm::SmallPtrSet<mlir::Operation *, 32> lhsVisited;
  llvm::SmallPtrSet<mlir::Operation *, 32> rhsVisited;
  collectConcreteFieldSupplies(lhs, lhsSupplies, lhsVisited);
  collectConcreteFieldSupplies(rhs, rhsSupplies, rhsVisited);
  return llvm::any_of(lhsSupplies, [&](mlir::Operation *supply) {
    return rhsSupplies.contains(supply);
  });
}

std::optional<riscv::ValueType>
coalescePartialOperand(mlir::Builder &builder, riscv::ValueType source,
                       llvm::ArrayRef<int64_t> partialAxes,
                       int64_t requiredLanes, riscv::TargetAttr target) {
  auto currentLanes = riscv_internal::staticProduct(
      source.getLayout().getLaneFactors().asArrayRef());
  if (!currentLanes || *currentLanes <= 0 || requiredLanes <= 0)
    return std::nullopt;
  if (*currentLanes >= requiredLanes)
    return source;
  if (requiredLanes % *currentLanes)
    return std::nullopt;
  const int64_t factor = requiredLanes / *currentLanes;
  if (factor <= 1 || (factor & (factor - 1)))
    return std::nullopt;

  llvm::SmallVector<int64_t> time(
      source.getLayout().getTimeFactors().asArrayRef());
  llvm::SmallVector<int64_t> lane(
      source.getLayout().getLaneFactors().asArrayRef());
  llvm::SmallVector<int64_t> replica(
      source.getLayout().getReplicaFactors().asArrayRef());
  std::optional<size_t> movedPosition;
  for (int64_t axis : partialAxes) {
    auto position = axisPosition(source, axis);
    if (!position || replica[*position] < factor ||
        replica[*position] % factor)
      continue;
    if (movedPosition)
      return std::nullopt;
    movedPosition = *position;
  }
  if (!movedPosition)
    return std::nullopt;
  const size_t position = *movedPosition;
  const int64_t remainingReplica = replica[position] / factor;
  if (remainingReplica <= 0 ||
      time[position] > std::numeric_limits<int64_t>::max() /
                           remainingReplica ||
      lane[position] > std::numeric_limits<int64_t>::max() / factor ||
      source.getLayout().getLmulEighths() >
          std::numeric_limits<int64_t>::max() / factor ||
      source.getLayout().getVl() >
          std::numeric_limits<int64_t>::max() / factor)
    return std::nullopt;
  time[position] *= remainingReplica;
  lane[position] *= factor;
  replica[position] = 1;
  const int64_t lmul = source.getLayout().getLmulEighths() * factor;
  const int64_t vl = source.getLayout().getVl() * factor;
  if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul))
    return std::nullopt;
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), source.getLayout().getSew(), lmul,
      vl, std::max<int64_t>(1, (lmul + 7) / 8),
      source.getLayout().getValidity());
  auto result = riscv::ValueType::get(builder.getContext(),
                                      source.getElementType(), source.getShape(),
                                      source.getAxisIds(), layout);
  if (!riscv::supportsRVVLayout(target, layout))
    return std::nullopt;
  return result;
}

std::optional<ScaledSourceLaneGeometry>
deriveScaledSourceLaneGeometry(mlir::Builder &builder,
                               riscv::RVVWidenDotOp dot,
                               riscv::ValueType lhsSource,
                               riscv::ValueType rhsSource,
                               int64_t partialSlots, int64_t sourceSlots,
                               int64_t laneSplit) {
  if (dot.getOver().size() != 1 || partialSlots <= 0 || sourceSlots <= 0 ||
      laneSplit <= 1 || sourceSlots * laneSplit != partialSlots)
    return std::nullopt;
  const int64_t reductionAxis = dot.getOver()[0];
  const int64_t sourceLanes = dot.getReductionLanes() * laneSplit;
  auto sourceSlot = groupedLanePartialSlotType(
      builder, dot->getParentOfType<riscv::KernelOp>().getTarget(),
      lhsSource, rhsSource, reductionAxis, sourceLanes);
  auto splitSourceSlot = sourceSlot
                             ? splitPartialSlotType(builder, sourceSlot,
                                                    reductionAxis, laneSplit)
                             : riscv::ValueType();
  auto partialSlot = splitSourceSlot;
  auto reducedSlot =
      partialSlot
          ? reducedPartialSlotType(builder, partialSlot, reductionAxis)
          : riscv::ValueType();
  auto kernel = dot->getParentOfType<riscv::KernelOp>();
  if (!kernel || !sourceSlot || !splitSourceSlot || !partialSlot || !reducedSlot)
    return std::nullopt;
  const auto legalLMUL =
      kernel.getTarget().getLegalLMULEighths().asArrayRef();
  if (!llvm::is_contained(legalLMUL,
                          sourceSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legalLMUL,
                          splitSourceSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legalLMUL,
                          reducedSlot.getLayout().getLmulEighths()))
    return std::nullopt;
  const int64_t sourceSetGroups =
      sourceSlots * sourceSlot.getLayout().getRegisterGroups();
  const int64_t reducedGroups =
      partialSlots * reducedSlot.getLayout().getRegisterGroups();
  const int64_t repackedGroups =
      partialSlots * splitSourceSlot.getLayout().getRegisterGroups();
  if (std::max({sourceSetGroups, repackedGroups, reducedGroups}) >
      kernel.getTarget().getVectorRegisters())
    return std::nullopt;
  return ScaledSourceLaneGeometry{sourceSlot, splitSourceSlot, partialSlot,
                                  reducedSlot, sourceSetGroups, reducedGroups};
}

std::optional<riscv::NestedPartialPlanAttr> planNestedPartialCarrier(
    mlir::Builder &builder, riscv::RVVWidenDotOp dot,
    const ReplicaScaledDotReduction &match, int64_t windowAxis,
    int64_t issueStreams, int64_t windowExtent) {
  auto kernel = dot->getParentOfType<riscv::KernelOp>();
  auto dotResult = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
  mlir::Value finalResult = match.reductions.empty()
                                ? mlir::Value()
                                : match.reductions.front()->getResult(0);
  mlir::Type finalType = finalResult ? finalResult.getType() : mlir::Type();
  auto finalValue = mlir::dyn_cast<riscv::ValueType>(finalType);
  llvm::SmallVector<int64_t> outputAxes =
      finalValue
          ? llvm::SmallVector<int64_t>(finalValue.getAxisIds().asArrayRef())
          : llvm::SmallVector<int64_t>();
  auto outputReplicaCount =
      finalValue ? riscv_internal::staticProduct(
                       finalValue.getLayout().getReplicaFactors().asArrayRef())
                 : std::optional<int64_t>(1);
  const bool terminalOutputCohort =
      finalResult && outputReplicaCount && *outputReplicaCount > 1 &&
      !reachesReductionThroughPureOps(finalResult);
  auto scaleType = mlir::dyn_cast<riscv::ValueType>(match.scale.getType());
  const bool finalCarrierClosed =
      !finalValue ||
      (finalValue.getLayout().getCarrier() == "scalar" &&
       llvm::all_of(finalValue.getLayout().getTimeFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) &&
       llvm::all_of(finalValue.getLayout().getLaneFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) &&
       llvm::all_of(finalValue.getLayout().getFragmentFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) &&
       llvm::all_of(finalValue.getLayout().getLocalFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }));
  if (!kernel || !dotResult || dot.getOver().empty() || issueStreams <= 0 ||
      windowExtent <= 0 || !scaleType || !outputReplicaCount ||
      *outputReplicaCount <= 0 || !finalCarrierClosed)
    return std::nullopt;

  auto coalesceReductionTime = [&](riscv::ValueType source) {
    if (!source || source.getLayout().getCarrier() != "rvv")
      return riscv::ValueType();
    llvm::SmallVector<int64_t> time(
        source.getLayout().getTimeFactors().asArrayRef());
    llvm::SmallVector<int64_t> lane(
        source.getLayout().getLaneFactors().asArrayRef());
    auto replica = source.getLayout().getReplicaFactors();
    auto fragment = source.getLayout().getFragmentFactors();
    auto local = source.getLayout().getLocalFactors();
    auto oldLanes = riscv_internal::staticProduct(lane);
    if (!oldLanes || *oldLanes <= 0)
      return riscv::ValueType();
    for (int64_t axis : dot.getOver()) {
      auto position = axisPosition(source, axis);
      if (!position || time[*position] <= 0 || lane[*position] <= 0 ||
          replica[*position] != 1 || fragment[*position] != 1 ||
          local[*position] != 1 ||
          lane[*position] >
              std::numeric_limits<int64_t>::max() / time[*position])
        return riscv::ValueType();
      lane[*position] *= time[*position];
      time[*position] = 1;
    }
    auto newLanes = riscv_internal::staticProduct(lane);
    auto replicas = riscv_internal::staticProduct(replica.asArrayRef());
    if (!newLanes || !replicas || *newLanes <= 0 || *replicas <= 0 ||
        source.getLayout().getLmulEighths() <= 0 ||
        source.getLayout().getLmulEighths() >
            std::numeric_limits<int64_t>::max() / *newLanes)
      return riscv::ValueType();
    const int64_t scaled =
        source.getLayout().getLmulEighths() * *newLanes;
    const int64_t requested = scaled / *oldLanes + (scaled % *oldLanes != 0);
    int64_t selectedLMUL = 0;
    for (int64_t legal : kernel.getTarget().getLegalLMULEighths().asArrayRef())
      if (legal >= requested &&
          (selectedLMUL == 0 || legal < selectedLMUL))
        selectedLMUL = legal;
    if (selectedLMUL <= 0)
      return riscv::ValueType();
    auto layout = riscv::LayoutAttr::get(
        builder.getContext(), "rvv", source.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane), replica, fragment, local,
        source.getLayout().getSew(), selectedLMUL, *newLanes,
        std::max<int64_t>(1, (selectedLMUL + 7) / 8) * *replicas,
        source.getLayout().getValidity());
    if (!riscv::supportsRVVLayout(kernel.getTarget(), layout))
      return riscv::ValueType();
    return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                                 source.getShape(), source.getAxisIds(), layout);
  };

  auto issueLhs = issueWindowType(builder, kernel.getTarget(),
                                  dot.getLhs().getType(), windowAxis,
                                  windowExtent);
  auto issueRhs = issueWindowType(builder, kernel.getTarget(),
                                  dot.getRhs().getType(), windowAxis,
                                  windowExtent);
  // A multi-output contraction can amortize one complete reduction carrier
  // across its output replicas.  For a scalar result the existing nested
  // topology is already closed; widening it here would rewrite unrelated
  // index/storage producers without reducing consumer work.
  if (terminalOutputCohort) {
    issueLhs = coalesceReductionTime(issueLhs);
    issueRhs = coalesceReductionTime(issueRhs);
  }
  auto issueScale = issueWindowType(builder, kernel.getTarget(), scaleType,
                                    windowAxis, windowExtent);
  auto scaleReplicaType =
      issueScale ? scalarReplicaType(builder, issueScale) : riscv::ValueType();
  auto scaleElement =
      scaleReplicaType
          ? mlir::dyn_cast<mlir::IntegerType>(scaleReplicaType.getElementType())
          : mlir::IntegerType();
  auto lhsLanes = issueLhs ? riscv_internal::staticProduct(
                                 issueLhs.getLayout().getLaneFactors().asArrayRef())
                           : std::optional<int64_t>();
  auto rhsLanes = issueRhs ? riscv_internal::staticProduct(
                                 issueRhs.getLayout().getLaneFactors().asArrayRef())
                           : std::optional<int64_t>();
  auto lhsTime = issueLhs ? riscv_internal::staticProduct(
                                issueLhs.getLayout().getTimeFactors().asArrayRef())
                          : std::optional<int64_t>();
  auto rhsTime = issueRhs ? riscv_internal::staticProduct(
                                issueRhs.getLayout().getTimeFactors().asArrayRef())
                          : std::optional<int64_t>();
  auto lhsReplicas =
      issueLhs ? riscv_internal::staticProduct(
                     issueLhs.getLayout().getReplicaFactors().asArrayRef())
               : std::optional<int64_t>();
  auto rhsReplicas =
      issueRhs ? riscv_internal::staticProduct(
                     issueRhs.getLayout().getReplicaFactors().asArrayRef())
               : std::optional<int64_t>();
  llvm::SmallVector<int64_t> partialAxes(match.reducedAxes);
  llvm::SmallVector<int64_t> partialAxisExtents;
  int64_t partialSlots = 1;
  for (int64_t axis : partialAxes) {
    auto lhsExtent = laneProductForAxes(issueLhs, {axis});
    auto rhsExtent = laneProductForAxes(issueRhs, {axis});
    if (!lhsExtent || !rhsExtent || *lhsExtent <= 0 ||
        *lhsExtent != *rhsExtent ||
        partialSlots > std::numeric_limits<int64_t>::max() / *lhsExtent) {
      partialSlots = 0;
      break;
    }
    partialAxisExtents.push_back(*lhsExtent);
    partialSlots *= *lhsExtent;
  }
  const int64_t sourceLanes =
      dot.getReductionLanes() * partialSlots;
  if (!issueLhs || !issueRhs || !issueScale || !scaleReplicaType || !scaleElement ||
      !scaleElement.isSigned() || scaleElement.getWidth() != 32 || !lhsLanes ||
      !rhsLanes || !lhsTime || !rhsTime || !lhsReplicas || !rhsReplicas ||
      partialSlots < windowExtent || partialSlots % windowExtent ||
      *lhsLanes != sourceLanes || *rhsLanes != sourceLanes || *lhsTime != 1 ||
      *rhsTime != 1 || *lhsReplicas <= 0 || *rhsReplicas <= 0) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t> lhsSourceParts;
  llvm::SmallVector<int64_t> rhsSourceParts;
  llvm::SmallVector<int64_t> lhsLaneOffsets;
  llvm::SmallVector<int64_t> rhsLaneOffsets;
  llvm::SmallVector<int64_t> scaleReplicaParts;
  llvm::SmallVector<int64_t> windowAxes(partialAxes);
  if (!terminalOutputCohort) {
    auto availableScales = riscv_internal::staticProduct(
        scaleReplicaType.getLayout().getReplicaFactors().asArrayRef());
    if (*outputReplicaCount != 1 || !availableScales ||
        *availableScales < partialSlots)
      return std::nullopt;
    lhsSourceParts.push_back(0);
    rhsSourceParts.push_back(0);
    lhsLaneOffsets.push_back(0);
    rhsLaneOffsets.push_back(0);
    for (int64_t window = 0; window < partialSlots; ++window)
      scaleReplicaParts.push_back(window);
  }
  for (int64_t output = 0; terminalOutputCohort &&
                                   output < *outputReplicaCount;
       ++output) {
    auto firstResultPart = composeReplicaPart(
        dotResult, outputAxes, output, windowAxes, 0);
    auto lhsPart = firstResultPart
                       ? projectReplica(issueLhs, dotResult, *firstResultPart)
                       : std::optional<int64_t>();
    auto rhsPart = firstResultPart
                       ? projectReplica(issueRhs, dotResult, *firstResultPart)
                       : std::optional<int64_t>();
    if (!lhsPart || !rhsPart)
      return std::nullopt;
    lhsSourceParts.push_back(*lhsPart);
    rhsSourceParts.push_back(*rhsPart);
    lhsLaneOffsets.push_back(0);
    rhsLaneOffsets.push_back(0);
    for (int64_t window = 0; window < partialSlots; ++window) {
      auto resultPart = composeReplicaPart(dotResult, outputAxes, output,
                                           windowAxes, window);
      auto scalePart = resultPart
                           ? projectReplica(scaleReplicaType, dotResult,
                                            *resultPart)
                           : std::optional<int64_t>();
      if (!scalePart)
        return std::nullopt;
      scaleReplicaParts.push_back(*scalePart);
    }
  }

  const int64_t reductionAxis = dot.getOver()[0];
  auto sourceSlot = groupedLanePartialSlotType(
      builder, kernel.getTarget(), issueLhs, issueRhs, reductionAxis, sourceLanes);
  auto splitSlot =
      partialSlots == 1
          ? sourceSlot
          : sourceSlot
                ? splitPartialSlotType(builder, sourceSlot, reductionAxis,
                                       partialSlots)
                : riscv::ValueType();
  auto reducedSlot = splitSlot
                         ? reducedPartialSlotType(builder, splitSlot,
                                                  reductionAxis)
                         : riscv::ValueType();
  auto multiplyInstruction = partialMultiplyInstruction(issueLhs, issueRhs);
  const auto legal = kernel.getTarget().getLegalLMULEighths().asArrayRef();
  if (!sourceSlot || !splitSlot || !reducedSlot || !multiplyInstruction ||
      !llvm::is_contained(legal, sourceSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legal, splitSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legal, reducedSlot.getLayout().getLmulEighths())) {
    return std::nullopt;
  }
  auto sourceSet = riscv::PartialSetType::get(
      builder.getContext(), sourceSlot, reductionAxis, 1,
      sourceSlot.getShape()[0], sourceSlot.getLayout().getRegisterGroups());
  auto repackedSet =
      partialSlots == 1
          ? sourceSet
          : riscv::PartialSetType::get(
                builder.getContext(), splitSlot, reductionAxis, partialSlots,
                sourceSet.getTermsPerSlot() / partialSlots,
                partialSlots * splitSlot.getLayout().getRegisterGroups());
  auto reducedSet = riscv::PartialSetType::get(
      builder.getContext(), reducedSlot, reductionAxis, partialSlots,
      repackedSet.getTermsPerSlot(),
      partialSlots * reducedSlot.getLayout().getRegisterGroups());
  const int64_t scaleCombineArity =
      partialAxisExtents.empty() ? 0 : partialAxisExtents.back();
  if (scaleCombineArity <= 0 || partialSlots % scaleCombineArity)
    return std::nullopt;
  int64_t remainingSlots = partialSlots / scaleCombineArity;
  int64_t termsPerSlot =
      repackedSet.getTermsPerSlot() * scaleCombineArity;
  auto scaleCombinedSet = riscv::PartialSetType::get(
      builder.getContext(), reducedSlot, reductionAxis, remainingSlots,
      termsPerSlot,
      remainingSlots * reducedSlot.getLayout().getRegisterGroups());
  llvm::SmallVector<mlir::Attribute> combineSetTypes;
  llvm::SmallVector<int64_t> combineAxes;
  llvm::SmallVector<int64_t> combineArities;
  for (int64_t stage = static_cast<int64_t>(partialAxes.size()) - 2;
       stage >= 0; --stage) {
    const int64_t arity = partialAxisExtents[stage];
    if (arity <= 0 || remainingSlots % arity)
      return std::nullopt;
    remainingSlots /= arity;
    termsPerSlot *= arity;
    auto stageSet = riscv::PartialSetType::get(
        builder.getContext(), reducedSlot, reductionAxis, remainingSlots,
        termsPerSlot,
        remainingSlots * reducedSlot.getLayout().getRegisterGroups());
    combineSetTypes.push_back(mlir::TypeAttr::get(stageSet));
    combineAxes.push_back(partialAxes[stage]);
    combineArities.push_back(arity);
  }
  if (remainingSlots != 1)
    return std::nullopt;
  auto finalSet = combineSetTypes.empty()
                      ? scaleCombinedSet
                      : mlir::cast<riscv::PartialSetType>(
                            mlir::cast<mlir::TypeAttr>(combineSetTypes.back())
                                .getValue());
  const bool scaleSharesLhs =
      hasIndexedLookupSupply(dot.getLhs()) &&
      sharesConcreteFieldSupply(match.scale, dot.getLhs());
  const bool scaleSharesRhs =
      hasIndexedLookupSupply(dot.getRhs()) &&
      sharesConcreteFieldSupply(match.scale, dot.getRhs());
  auto scaleLayout = scaleType.getLayout();
  const bool pairedScaleWindows =
      windowExtent >= 2 && windowExtent <= kernel.getTarget().getVlenBits() / 64 &&
      issueStreams % 2 == 0 && dot.getPartialUnroll() >= issueStreams &&
      scaleLayout.getRegisterGroups() <= 2;
  const bool sharedScale =
      !match.scale.hasOneUse() && issueStreams > 1 &&
      ((windowExtent >= 4 && scaleLayout.getRegisterGroups() == 1) ||
       pairedScaleWindows) &&
      *outputReplicaCount == 1 && scaleType.getShape().size() == 1 &&
      scaleType.getAxisIds()[0] == windowAxis &&
      scaleType.getShape()[0] == issueStreams * windowExtent &&
      scaleLayout.getCarrier() == "rvv" && scaleLayout.getValidity() == "full" &&
      scaleLayout.getTimeFactors()[0] == 1 &&
      scaleLayout.getLaneFactors()[0] == scaleType.getShape()[0] &&
      scaleLayout.getReplicaFactors()[0] == 1 &&
      scaleLayout.getFragmentFactors()[0] == 1 &&
      scaleLayout.getLocalFactors()[0] == 1 &&
      riscv::supportsRVVLayout(kernel.getTarget(), scaleLayout);
  const llvm::StringRef scaleSupplyStage =
      sharedScale ? "shared-before-issue"
      : *outputReplicaCount == 1 && (scaleSharesLhs || scaleSharesRhs)
          ? "after-partial-reduce"
          : "before-product";
  const int64_t operandAndProduct =
      issueLhs.getLayout().getRegisterGroups() +
      issueRhs.getLayout().getRegisterGroups() + sourceSet.getResourceGroups() +
      (scaleSupplyStage != "after-partial-reduce"
           ? issueScale.getLayout().getRegisterGroups()
           : 0);
  int64_t repackAndReduce =
      std::max(repackedSet.getResourceGroups(), reducedSet.getResourceGroups());
  mlir::Attribute selectedRepackLeaf = builder.getUnitAttr();
  if (partialSlots > 1) {
    auto repack = partialRepackLeaf(builder, kernel.getTarget(), sourceSet,
                                    repackedSet, partialSlots);
    if (!repack)
      return std::nullopt;
    selectedRepackLeaf = *repack;
    repackAndReduce = std::max(repackAndReduce,
        sourceSet.getResourceGroups() + repack->getTemporaryGroups());
  }
  const int64_t reduceAndScale =
      reducedSet.getResourceGroups() +
      issueScale.getLayout().getRegisterGroups() +
      scaleCombinedSet.getResourceGroups();
  int64_t combineResources = scaleCombinedSet.getResourceGroups();
  riscv::PartialSetType previousSet = scaleCombinedSet;
  for (mlir::Attribute typeAttr : combineSetTypes) {
    auto nextSet = mlir::cast<riscv::PartialSetType>(
        mlir::cast<mlir::TypeAttr>(typeAttr).getValue());
    combineResources =
        std::max(combineResources, previousSet.getResourceGroups() +
                                       nextSet.getResourceGroups());
    previousSet = nextSet;
  }
  const int64_t resources =
      1 + std::max({operandAndProduct, repackAndReduce, reduceAndScale,
                    combineResources}) +
      (sharedScale ? 3 * scaleLayout.getRegisterGroups() : 0);
  if (resources > kernel.getTarget().getVectorRegisters())
    return std::nullopt;
  const int64_t issueUnroll =
      std::min<int64_t>(dot.getPartialUnroll(), issueStreams);
  llvm::SmallVector<int64_t> issueMaterializationOrder{0, 1, 2};
  if (scaleSharesRhs && !scaleSharesLhs)
    issueMaterializationOrder =
        *outputReplicaCount == 1 ? llvm::SmallVector<int64_t>{1, 0, 2}
                                 : llvm::SmallVector<int64_t>{1, 2, 0};
  else if (scaleSharesLhs && !scaleSharesRhs)
    issueMaterializationOrder =
        *outputReplicaCount == 1 ? llvm::SmallVector<int64_t>{0, 1, 2}
                                 : llvm::SmallVector<int64_t>{0, 2, 1};
  auto scaleParts = riscv_internal::staticProduct(
      scaleReplicaType.getLayout().getReplicaFactors().asArrayRef());
  llvm::DenseSet<mlir::Operation *> visited;
  const llvm::StringRef scaleSupply =
      !sharedScale && scaleParts && *scaleParts >= 8 &&
              !scaleSharesLhs && !scaleSharesRhs &&
              supportsScalarReplicaRematerialization(match.scale, visited)
          ? "scalar-rematerialize"
          : "vector-convert";
  auto finalizeInstruction = partialFinalizeInstruction(finalSet, reductionAxis);
  if (!finalizeInstruction)
    return std::nullopt;
  return riscv::NestedPartialPlanAttr::get(
      builder.getContext(), windowAxis, issueStreams, windowExtent,
      riscv_internal::integers(builder, partialAxes),
      riscv_internal::integers(builder, partialAxisExtents), partialSlots,
      issueUnroll,
      riscv_internal::integers(builder, outputAxes), *outputReplicaCount,
      riscv_internal::integers(builder, issueMaterializationOrder),
      riscv_internal::integers(builder, lhsSourceParts),
      riscv_internal::integers(builder, rhsSourceParts),
      riscv_internal::integers(builder, lhsLaneOffsets),
      riscv_internal::integers(builder, rhsLaneOffsets),
      riscv_internal::integers(builder, scaleReplicaParts), issueLhs, issueRhs,
      sourceSlot, splitSlot, reducedSlot, scaleReplicaType,
      sourceSet, repackedSet, reducedSet,
      scaleCombinedSet, builder.getArrayAttr(combineSetTypes),
      riscv_internal::integers(builder, combineAxes),
      riscv_internal::integers(builder, combineArities), scaleSupply,
      scaleSupplyStage,
      *multiplyInstruction, "rvv.partial-reduce.widen", *finalizeInstruction,
      resources, selectedRepackLeaf);
}

std::optional<SourceLanePlan>
planScaledSourceLanes(mlir::Builder &builder, riscv::RVVWidenDotOp dot,
                      const ScaledTopologyFacts &facts) {
  auto result = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
  auto kernel = dot->getParentOfType<riscv::KernelOp>();
  if (!result || !kernel || facts.partialSlots <= 1 ||
      facts.outputReplicas <= 0 || dot.getReductionLanes() <= 0 ||
      dot.getSliceLmulEighths() <= 0)
    return std::nullopt;

  for (int64_t split = facts.partialSlots; split > 1; --split) {
    if (facts.partialSlots % split)
      continue;
    const int64_t sourceLanes = split * dot.getReductionLanes();
    auto lhsSource = coalescePartialOperand(
        builder, dot.getLhs().getType(), facts.partialAxes, sourceLanes,
        kernel.getTarget());
    auto rhsSource = coalescePartialOperand(
        builder, dot.getRhs().getType(), facts.partialAxes, sourceLanes,
        kernel.getTarget());
    auto lhsSlices =
        lhsSource ? riscv_internal::planWidenDotLaneSlices(
                        *lhsSource, result, dot.getOver(), kernel.getTarget())
                  : std::optional<riscv_internal::WidenDotLaneSlicePlan>();
    auto rhsSlices =
        rhsSource ? riscv_internal::planWidenDotLaneSlices(
                        *rhsSource, result, dot.getOver(), kernel.getTarget())
                  : std::optional<riscv_internal::WidenDotLaneSlicePlan>();
    if (!lhsSource || !rhsSource || !lhsSlices || !rhsSlices ||
        sourceLanes > lhsSource->getLayout().getVl() ||
        sourceLanes > rhsSource->getLayout().getVl() ||
        lhsSlices->reductionLanes != dot.getReductionLanes() ||
        rhsSlices->reductionLanes != dot.getReductionLanes() ||
        lhsSlices->reductionStreams != dot.getReductionStreams() ||
        rhsSlices->reductionStreams != dot.getReductionStreams() ||
        lhsSlices->parts.size() != rhsSlices->parts.size() ||
        lhsSlices->offsets.size() != rhsSlices->offsets.size())
      continue;
    bool complete = true;
    llvm::SmallVector<int64_t> sourceLhsParts;
    llvm::SmallVector<int64_t> sourceRhsParts;
    llvm::SmallVector<int64_t> sourceLhsOffsets;
    llvm::SmallVector<int64_t> sourceRhsOffsets;
    for (int64_t outputPart = 0; outputPart < facts.outputReplicas && complete;
         ++outputPart) {
      for (int64_t first = 0; first < facts.partialSlots && complete;
           first += split) {
        std::optional<int64_t> firstLhsPart;
        std::optional<int64_t> firstRhsPart;
        std::optional<int64_t> firstLhsOffset;
        std::optional<int64_t> firstRhsOffset;
        for (int64_t inner = 0; inner < split; ++inner) {
          auto resultPart = composeReplicaPart(
              result, facts.outputAxes, outputPart, facts.partialAxes,
              first + inner);
          const int64_t planned =
              resultPart ? *resultPart * dot.getReductionStreams() : -1;
          if (!resultPart || planned < 0 ||
              planned >= static_cast<int64_t>(lhsSlices->parts.size()) ||
              planned >= static_cast<int64_t>(rhsSlices->parts.size()) ||
              *resultPart >=
                  static_cast<int64_t>(lhsSlices->offsets.size()) ||
              *resultPart >=
                  static_cast<int64_t>(rhsSlices->offsets.size())) {
            complete = false;
            break;
          }
          const int64_t lhsPart = lhsSlices->parts[planned];
          const int64_t rhsPart = rhsSlices->parts[planned];
          const int64_t lhsOffset = lhsSlices->offsets[*resultPart];
          const int64_t rhsOffset = rhsSlices->offsets[*resultPart];
          if (!firstLhsPart) {
            firstLhsPart = lhsPart;
            firstRhsPart = rhsPart;
            firstLhsOffset = lhsOffset;
            firstRhsOffset = rhsOffset;
          }
          const int64_t expectedLhs =
              *firstLhsOffset + inner * dot.getReductionLanes();
          const int64_t expectedRhs =
              *firstRhsOffset + inner * dot.getReductionLanes();
          if (lhsPart != *firstLhsPart || rhsPart != *firstRhsPart ||
              lhsOffset != expectedLhs || rhsOffset != expectedRhs) {
            complete = false;
            break;
          }
        }
        complete &= firstLhsOffset && firstRhsOffset && sourceLanes > 0 &&
                    *firstLhsOffset % sourceLanes == 0 &&
                    *firstRhsOffset % sourceLanes == 0;
        if (complete) {
          sourceLhsParts.push_back(*firstLhsPart);
          sourceRhsParts.push_back(*firstRhsPart);
          sourceLhsOffsets.push_back(*firstLhsOffset);
          sourceRhsOffsets.push_back(*firstRhsOffset);
        }
      }
    }
    const int64_t sourceSlots = facts.partialSlots / split;
    complete &= sourceLhsParts.size() ==
                    static_cast<size_t>(facts.outputReplicas * sourceSlots) &&
                sourceRhsParts.size() == sourceLhsParts.size() &&
                sourceLhsOffsets.size() == sourceLhsParts.size() &&
                sourceRhsOffsets.size() == sourceLhsParts.size();
    auto geometry = complete
                        ? deriveScaledSourceLaneGeometry(
                              builder, dot, *lhsSource, *rhsSource,
                              facts.partialSlots, sourceSlots, split)
                        : std::optional<ScaledSourceLaneGeometry>();
    if (geometry)
      return SourceLanePlan{sourceSlots,
                            split,
                            *lhsSource,
                            *rhsSource,
                            *geometry,
                            std::move(sourceLhsParts),
                            std::move(sourceRhsParts),
                            std::move(sourceLhsOffsets),
                            std::move(sourceRhsOffsets)};
  }
  return std::nullopt;
}

} // namespace weft::riscv_partial
