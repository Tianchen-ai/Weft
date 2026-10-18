#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace {

namespace riscv = weft::riscv;

struct StorageRelation {
  int64_t storageIndex = 0;
  int64_t storageWidth = 0;
  int64_t shift = 0;
  int64_t mask = 0;
  mlir::Type rawElement;
  llvm::SmallVector<int64_t> logicalIndices;
};

std::optional<int64_t> product(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

std::optional<int64_t> logicalElementCount(mlir::Type type) {
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return product(value.getShape().asArrayRef());
  if (type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
    return int64_t{1};
  return std::nullopt;
}

std::optional<StorageRelation> storageRelation(riscv::FieldOp field,
                                               int64_t logicalIndex) {
  riscv::AccessAttr access = field.getAccess();
  mlir::Type element = weft::riscv_internal::logicalElement(
      field.getResult().getType());
  const int64_t elementBits = weft::riscv_internal::logicalBitWidth(element);
  auto elements = logicalElementCount(field.getResult().getType());
  auto bit = riscv::recordFieldRelativeBit(access, elementBits, logicalIndex);
  if (!elements || logicalIndex < 0 || logicalIndex >= *elements || !bit ||
      access.getBitOffset() % 8)
    return std::nullopt;

  StorageRelation result;
  if (access.getMapping() == "natural") {
    if (elementBits != 8 && elementBits != 16 && elementBits != 32 &&
        elementBits != 64)
      return std::nullopt;
    result.storageWidth = elementBits;
    result.rawElement = element;
  } else if (access.getMapping() == "grouped_layered") {
    if (elementBits <= 0 || elementBits >= 8 || *bit % 8 + elementBits > 8)
      return std::nullopt;
    result.storageWidth = 8;
    result.rawElement = mlir::IntegerType::get(
        field.getContext(), 8, mlir::IntegerType::Unsigned);
    result.shift = *bit % 8;
    result.mask = (int64_t{1} << elementBits) - 1;
  } else {
    return std::nullopt;
  }
  result.storageIndex = *bit / 8;
  for (int64_t candidate = 0; candidate < *elements; ++candidate) {
    auto candidateBit =
        riscv::recordFieldRelativeBit(access, elementBits, candidate);
    if (candidateBit && *candidateBit / 8 == result.storageIndex)
      result.logicalIndices.push_back(candidate);
  }
  if (result.logicalIndices.empty() ||
      result.storageIndex >
          (access.getStorageBits() - result.storageWidth) / 8)
    return std::nullopt;
  return result;
}

std::optional<int64_t> recordByteStride(riscv::MemDescType memory,
                                        int64_t axis, int64_t partition) {
  if (!memory || memory.getShape().size() != 1 ||
      memory.getAxisIds().size() != 1 || memory.getStrides().size() != 1 ||
      memory.getAxisIds()[0] != axis || memory.getStrides()[0] != 1 ||
      memory.getElements() <= 0 || memory.getStorageBits() <= 0 ||
      memory.getStorageBits() % 8 || partition <= 0 ||
      partition % memory.getElements())
    return std::nullopt;
  const int64_t recordBytes = memory.getStorageBits() / 8;
  const int64_t recordsPerPoint = partition / memory.getElements();
  if (recordsPerPoint <= 0 ||
      recordBytes > std::numeric_limits<int64_t>::max() / recordsPerPoint)
    return std::nullopt;
  return recordBytes * recordsPerPoint;
}

riscv::AccessAttr stridedAccess(mlir::Builder &builder,
                                riscv::AccessAttr source) {
  return riscv::AccessAttr::get(
      builder.getContext(), "strided", source.getMapping(),
      source.getAlignment(), 0, 0, source.getGroupSize(), source.getLayerSize(),
      source.getJoinFields(), source.getJoinLowBits(), source.getJoinRole(),
      source.getBitOffset(), source.getStorageBits(), source.getOrder(),
      source.getFieldChunkBytes());
}

riscv::AccessAttr denseStridedAccess(mlir::Builder &builder,
                                     riscv::MemDescType destination) {
  return riscv::AccessAttr::get(
      builder.getContext(), "strided", "dense",
      std::max<int64_t>(1, destination.getAlignment()), 0, 0, 0, 0, 0, 0, 0,
      0, 0, "none", 1);
}

std::optional<riscv::ValueType>
cohortValueType(mlir::Builder &builder, mlir::Type element, int64_t axis,
                int64_t width, riscv::TargetAttr target) {
  int64_t sew = weft::riscv_internal::logicalBitWidth(element);
  if (sew <= 0)
    return std::nullopt;
  sew = std::max<int64_t>(8, sew);
  if (!llvm::is_contained(target.getSupportedSEW().asArrayRef(), sew) ||
      width <= 1 || target.getVlenBits() <= 0 ||
      width > std::numeric_limits<int64_t>::max() / sew ||
      width * sew > std::numeric_limits<int64_t>::max() / 8)
    return std::nullopt;
  const int64_t requiredBits = width * sew * 8;
  const int64_t required =
      requiredBits / target.getVlenBits() +
      (requiredBits % target.getVlenBits() != 0 ? 1 : 0);
  int64_t lmulEighths = 0;
  for (int64_t candidate : target.getLegalLMULEighths().asArrayRef())
    if (candidate >= required &&
        (lmulEighths == 0 || candidate < lmulEighths))
      lmulEighths = candidate;
  if (lmulEighths <= 0)
    return std::nullopt;
  const int64_t groups = std::max<int64_t>(1, (lmulEighths + 7) / 8);
  auto axes = builder.getDenseI64ArrayAttr({axis});
  auto ones = builder.getDenseI64ArrayAttr({1});
  auto lanes = builder.getDenseI64ArrayAttr({width});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axes, ones, lanes, ones, ones, ones, sew,
      lmulEighths, width, groups, "full");
  if (!riscv::supportsRVVLayout(target, layout))
    return std::nullopt;
  return riscv::ValueType::get(builder.getContext(), element,
                               builder.getDenseI64ArrayAttr({width}), axes,
                               layout);
}

bool isSupportedPointwiseValue(mlir::Value value, mlir::scf::ForOp loop,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
        argument && argument.getOwner() == loop.getBody())
      return false;
    return value.getType().isIndex() ||
           mlir::isa<mlir::IntegerType, mlir::FloatType>(value.getType());
  }
  if (definition->getParentOp() != loop.getOperation()) {
    return value.getType().isIndex() ||
           mlir::isa<mlir::IntegerType, mlir::FloatType>(value.getType());
  }
  if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition))
    return static_cast<bool>(storageRelation(field, 0));
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition))
    return isSupportedPointwiseValue(conversion.getInput(), loop, visited) &&
           conversion.getConversion().getEffect() == "pure";
  if (mlir::isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp,
                riscv::UnaryOp, riscv::BinaryOp, riscv::ConstantOp,
                mlir::arith::ConstantOp>(definition))
    return llvm::all_of(definition->getOperands(), [&](mlir::Value operand) {
      return isSupportedPointwiseValue(operand, loop, visited);
    });
  return false;
}

void collectLoopProducers(mlir::Value value, mlir::scf::ForOp loop,
                          llvm::DenseSet<mlir::Operation *> &required) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || definition->getParentOp() != loop.getOperation() ||
      !required.insert(definition).second)
    return;
  for (mlir::Value operand : definition->getOperands())
    collectLoopProducers(operand, loop, required);
}

bool isKnownNonNegativeExtent(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value() >= 0;
  return mlir::isa_and_nonnull<mlir::arith::DivUIOp,
                               mlir::arith::CeilDivUIOp>(
      value.getDefiningOp());
}

struct RecordLoopMatch {
  mlir::scf::ForOp loop;
  riscv::LevelAttr level;
  riscv::PhysicalPointOp point;
  riscv::StoreOp store;
  riscv::SliceOp destinationSlice;
  int64_t partition = 0;
  int64_t cohortWidth = 0;
  int64_t sourceRecordByteStride = 0;
  int64_t destinationRecordByteStride = 0;
  riscv::TargetAttr target;
};

std::optional<RecordLoopMatch> matchRecordLoop(mlir::scf::ForOp loop) {
  auto reject = [](llvm::StringRef) -> std::optional<RecordLoopMatch> {
    return std::nullopt;
  };
  auto level = loop->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
  auto lower =
      loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!level || level.getDirection() != "ascending" || !lower ||
      lower.value() != 0 || !step ||
      step.value() != 1 || !loop.getInitArgs().empty() ||
      !loop.getResults().empty() || loop->hasAttr("weft.riscv.schedule") ||
      loop->hasAttr("weft.riscv.record_vectorized") ||
      !isKnownNonNegativeExtent(loop.getUpperBound()))
    return reject("loop contract");

  riscv::PhysicalPointOp point;
  riscv::StoreOp store;
  int64_t stores = 0;
  bool unsupportedEffect = false;
  loop.getBody()->walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::scf::YieldOp>(operation))
      return;
    if (auto candidate = mlir::dyn_cast<riscv::PhysicalPointOp>(operation)) {
      if (point)
        unsupportedEffect = true;
      point = candidate;
      return;
    }
    if (auto candidate = mlir::dyn_cast<riscv::StoreOp>(operation)) {
      store = candidate;
      ++stores;
      return;
    }
    if (mlir::isa<riscv::LoadOp>(operation))
      return;
    if (!mlir::isMemoryEffectFree(operation)) {
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
      if (!effects) {
        unsupportedEffect = true;
        return;
      }
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      unsupportedEffect |= llvm::any_of(instances, [](const auto &instance) {
        return !mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
      });
    }
  });
  auto partition = point
                       ? point.getPartition().getDefiningOp<
                             mlir::arith::ConstantIndexOp>()
                       : mlir::arith::ConstantIndexOp();
  auto active = point
                    ? point.getActive().getDefiningOp<
                          mlir::arith::ConstantIndexOp>()
                    : mlir::arith::ConstantIndexOp();
  auto destinationSlice =
      store ? store.getRegion().getDefiningOp<riscv::SliceOp>()
            : riscv::SliceOp();
  auto stored = store ? mlir::dyn_cast<riscv::ValueType>(
                            store.getValue().getType())
                      : riscv::ValueType();
  if (!point || !partition || !active || partition.value() <= 1 ||
      active.value() != partition.value() || stores != 1 ||
      unsupportedEffect || !destinationSlice || !stored ||
      destinationSlice.getIndices().size() != 1 ||
      destinationSlice.getIndices().front() != point.getResult() ||
      destinationSlice.getSelectors().size() != 1 ||
      mlir::cast<mlir::StringAttr>(destinationSlice.getSelectors()[0])
              .getValue() != "domain" ||
      stored.getShape().size() != 1 || stored.getAxisIds().size() != 1 ||
      stored.getShape()[0] != partition.value() ||
      stored.getAxisIds()[0] != level.getAxisId() ||
      point.getResult().getType().getDomain().getAxisId() !=
          level.getAxisId() ||
      stored.getLayout().getCarrier() != "rvv")
    return reject("point/store/output contract");
  const int64_t cohortWidth = stored.getLayout().getVl();
  if (cohortWidth <= 1)
    return reject("pointwise closure");

  llvm::DenseSet<mlir::Value> visited;
  if (!isSupportedPointwiseValue(store.getValue(), loop, visited))
    return reject("pointwise closure");
  llvm::SmallVector<riscv::FieldOp> fields;
  for (mlir::Value value : visited)
    if (auto field = value.getDefiningOp<riscv::FieldOp>())
      fields.push_back(field);
  int64_t shapedFields = 0;
  riscv::LoadOp commonLoad;
  for (riscv::FieldOp field : fields) {
    if (auto type = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType()))
      if (!type.getShape().empty())
        ++shapedFields;
    auto load = weft::riscv_internal::sourceLoad(field.getOwner());
    auto slice = load ? load.getRegion().getDefiningOp<riscv::SliceOp>()
                      : riscv::SliceOp();
    if (!slice || slice.getIndices().size() != 1 ||
        slice.getIndices().front() != point.getResult() ||
        slice.getSelectors().size() != 1 ||
        mlir::cast<mlir::StringAttr>(slice.getSelectors()[0]).getValue() !=
            "domain")
      return reject("source point contract");
    if (!load || (commonLoad && commonLoad != load))
      return reject("multiple encoded loads");
    commonLoad = load;
  }
  if (fields.empty() || shapedFields != 1 || !commonLoad)
    return reject("field contract");

  auto sourceEncoding = mlir::dyn_cast<weft::kernel::EncodingType>(
      commonLoad.getRegion().getType().getEncoding());
  if (!sourceEncoding || sourceEncoding.getKind() == "dense")
    return reject("encoded source contract");

  llvm::DenseSet<mlir::Operation *> required;
  collectLoopProducers(store.getValue(), loop, required);
  collectLoopProducers(store.getRegion(), loop, required);
  collectLoopProducers(point.getResult(), loop, required);
  required.insert(store.getOperation());
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (!required.contains(&operation) &&
        !mlir::isMemoryEffectFree(&operation))
      return reject("exact loop-body closure");
  }

  auto sourceStride = recordByteStride(commonLoad.getRegion().getType(),
                                       level.getAxisId(), partition.value());
  auto destinationStride = recordByteStride(
      destinationSlice.getBase().getType(), level.getAxisId(),
      partition.value());
  if (!sourceStride || !destinationStride)
    return reject("record stride contract");

  auto kernel = loop->getParentOfType<riscv::KernelOp>();
  if (!kernel || !kernel.getTarget().getHasRVV() ||
      kernel.getTarget().getRecordAxisPolicy() != "across-records")
    return reject("target contract");
  return RecordLoopMatch{loop,
                         level,
                         point,
                         store,
                         destinationSlice,
                         partition.value(),
                         cohortWidth,
                         *sourceStride,
                         *destinationStride,
                         kernel.getTarget()};
}

class RecordLoopMaterializer {
public:
  RecordLoopMaterializer(RecordLoopMatch &match, mlir::IRRewriter &rewriter,
                         riscv::PhysicalPointOp point,
                         riscv::RecordCohortOp cohort)
      : match(match), rewriter(rewriter), point(point), cohort(cohort) {}

  mlir::FailureOr<mlir::Value> materialize(mlir::Value value,
                                           int64_t logicalIndex) {
    const int64_t keyIndex =
        weft::riscv_internal::logicalShape(value.getType()).empty()
            ? -1
            : logicalIndex;
    ValueKey key{value, keyIndex};
    if (auto found = values.find(key); found != values.end())
      return found->second;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || definition->getParentOp() != match.loop.getOperation())
      return value;

    if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
      auto result = materializeField(field, keyIndex < 0 ? 0 : keyIndex);
      if (mlir::succeeded(result))
        values.try_emplace(key, *result);
      return result;
    }
    if (auto conversion =
            mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
      auto result = materialize(conversion.getInput(), logicalIndex);
      if (mlir::succeeded(result))
        values.try_emplace(key, *result);
      return result;
    }
    if (auto constant = mlir::dyn_cast<riscv::ConstantOp>(definition)) {
      auto cloned = riscv::ConstantOp::create(rewriter,
          constant.getLoc(), constant.getResult().getType(), constant.getValue());
      weft::riscv_internal::copyOrigin(constant, cloned);
      values.try_emplace(key, cloned.getResult());
      return cloned.getResult();
    }
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(definition)) {
      auto cloned = mlir::arith::ConstantOp::create(rewriter,
          constant.getLoc(), constant.getValue());
      values.try_emplace(key, cloned.getResult());
      return cloned.getResult();
    }

    llvm::SmallVector<mlir::Value> operands;
    for (mlir::Value operand : definition->getOperands()) {
      auto cloned = materialize(operand, logicalIndex);
      if (mlir::failed(cloned))
        return mlir::failure();
      operands.push_back(*cloned);
    }
    if (!dependsOnField(value)) {
      if (definition->getNumResults() != 1)
        return mlir::failure();
      mlir::IRMapping mapping;
      for (auto [source, target] : llvm::zip(definition->getOperands(), operands))
        mapping.map(source, target);
      mlir::Operation *cloned = rewriter.clone(*definition, mapping);
      mlir::Value result = cloned->getResult(0);
      values.try_emplace(key, result);
      return result;
    }
    mlir::Type element =
        weft::riscv_internal::logicalElement(value.getType());
    auto resultType = cohortValueType(rewriter, element, match.level.getAxisId(),
                                      match.cohortWidth, match.target);
    if (!resultType)
      return mlir::failure();
    mlir::Value cloned;
    auto leaf = weft::riscv_internal::unselectedLeaf(rewriter);
    if (auto operation = mlir::dyn_cast<riscv::CastOp>(definition))
      cloned = rewriter
                   .create<riscv::CastOp>(operation.getLoc(), *resultType,
                                          operands.front(), leaf)
                   .getResult();
    else if (auto operation = mlir::dyn_cast<riscv::WidenOp>(definition))
      cloned = rewriter
                   .create<riscv::WidenOp>(operation.getLoc(), *resultType,
                                           operands.front(), leaf)
                   .getResult();
    else if (auto operation = mlir::dyn_cast<riscv::NarrowOp>(definition))
      cloned = rewriter
                   .create<riscv::NarrowOp>(
                       operation.getLoc(), *resultType, operands.front(),
                       operation.getRounding(), operation.getSaturate(), leaf)
                   .getResult();
    else if (auto operation = mlir::dyn_cast<riscv::UnaryOp>(definition))
      cloned = rewriter
                   .create<riscv::UnaryOp>(operation.getLoc(), *resultType,
                                           operands.front(), operation.getKind(),
                                           leaf)
                   .getResult();
    else if (auto operation = mlir::dyn_cast<riscv::BinaryOp>(definition))
      cloned = rewriter
                   .create<riscv::BinaryOp>(
                       operation.getLoc(), *resultType, operands[0], operands[1],
                       operation.getKind(), leaf)
                   .getResult();
    else
      return mlir::failure();
    weft::riscv_internal::copyOrigin(definition, cloned.getDefiningOp());
    values.try_emplace(key, cloned);
    return cloned;
  }

private:
  struct ValueKey {
    mlir::Value value;
    int64_t logicalIndex;
    bool operator==(const ValueKey &other) const {
      return value == other.value && logicalIndex == other.logicalIndex;
    }
  };

  struct ValueKeyInfo {
    static ValueKey getEmptyKey() {
      return {llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), 0};
    }
    static ValueKey getTombstoneKey() {
      return {llvm::DenseMapInfo<mlir::Value>::getTombstoneKey(), 0};
    }
    static unsigned getHashValue(const ValueKey &key) {
      return llvm::hash_combine(key.value, key.logicalIndex);
    }
    static bool isEqual(const ValueKey &lhs, const ValueKey &rhs) {
      return lhs == rhs;
    }
  };

  struct RawKey {
    mlir::Value field;
    int64_t storageIndex;
    bool operator==(const RawKey &other) const {
      return field == other.field && storageIndex == other.storageIndex;
    }
  };

  struct RawKeyInfo {
    static RawKey getEmptyKey() {
      return {llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), 0};
    }
    static RawKey getTombstoneKey() {
      return {llvm::DenseMapInfo<mlir::Value>::getTombstoneKey(), 0};
    }
    static unsigned getHashValue(const RawKey &key) {
      return llvm::hash_combine(key.field, key.storageIndex);
    }
    static bool isEqual(const RawKey &lhs, const RawKey &rhs) {
      return lhs == rhs;
    }
  };

  bool dependsOnField(mlir::Value value) {
    if (auto found = fieldDependence.find(value); found != fieldDependence.end())
      return found->second;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || definition->getParentOp() != match.loop.getOperation()) {
      fieldDependence.try_emplace(value, false);
      return false;
    }
    if (mlir::isa<riscv::FieldOp>(definition)) {
      fieldDependence.try_emplace(value, true);
      return true;
    }
    bool dependent = llvm::any_of(definition->getOperands(),
                                  [&](mlir::Value operand) {
                                    return dependsOnField(operand);
                                  });
    fieldDependence.try_emplace(value, dependent);
    return dependent;
  }

  mlir::FailureOr<mlir::Value> cloneField(riscv::FieldOp field) {
    if (auto found = fields.find(field.getResult()); found != fields.end())
      return found->second;
    auto load = weft::riscv_internal::sourceLoad(field.getOwner());
    auto slice = load ? load.getRegion().getDefiningOp<riscv::SliceOp>()
                      : riscv::SliceOp();
    if (!load || !slice || slice.getIndices().size() != 1 ||
        slice.getIndices().front() != match.point.getResult())
      return mlir::failure();
    riscv::LoadOp clonedLoad;
    if (auto found = loads.find(load.getResult()); found != loads.end()) {
      clonedLoad = found->second.getDefiningOp<riscv::LoadOp>();
    } else {
      auto clonedSlice = riscv::SliceOp::create(rewriter,
          slice.getLoc(), slice.getResult().getType(), slice.getBase(),
          mlir::ValueRange{point.getResult()}, slice.getSelectors());
      clonedLoad = riscv::LoadOp::create(rewriter,
          load.getLoc(), load.getResult().getType(), clonedSlice.getResult(),
          mlir::Value(), load.getAccess(), load.getLeaf());
      weft::riscv_internal::copyOrigin(slice, clonedSlice);
      weft::riscv_internal::copyOrigin(load, clonedLoad);
      loads.try_emplace(load.getResult(), clonedLoad.getResult());
    }
    auto clonedField = riscv::FieldOp::create(rewriter,
        field.getLoc(), field.getResult().getType(), clonedLoad.getResult(),
        field.getName(), field.getAccess(), field.getLeaf());
    weft::riscv_internal::copyOrigin(field, clonedField);
    fields.try_emplace(field.getResult(), clonedField.getResult());
    return clonedField.getResult();
  }

  mlir::FailureOr<mlir::Value> materializeField(riscv::FieldOp field,
                                                int64_t logicalIndex) {
    auto relation = storageRelation(field, logicalIndex);
    auto clonedField = cloneField(field);
    if (!relation || mlir::failed(clonedField))
      return mlir::failure();
    RawKey rawKey{field.getResult(), relation->storageIndex};
    mlir::Value raw;
    if (auto found = rawValues.find(rawKey); found != rawValues.end()) {
      raw = found->second;
    } else {
      auto rawType = cohortValueType(
          rewriter, relation->rawElement, match.level.getAxisId(),
          match.cohortWidth, match.target);
      if (!rawType)
        return mlir::failure();
      auto access = stridedAccess(rewriter, field.getAccess());
      auto load = riscv::RVVRecordStorageLoadOp::create(rewriter,
          field.getLoc(), *rawType, *clonedField, cohort.getResult(),
          relation->storageIndex, relation->storageWidth,
          match.sourceRecordByteStride,
          rewriter.getDenseI64ArrayAttr(relation->logicalIndices), access,
          weft::riscv_internal::leaf(
              rewriter, "rvv", "record-storage-load",
              "rvv.record-storage-load.strided",
              "rvv.record-storage-load.strided", 0,
              rawType->getLayout().getRegisterGroups()));
      weft::riscv_internal::copyOrigin(field, load);
      raw = load.getResult();
      rawValues.try_emplace(rawKey, raw);
    }
    if (relation->storageWidth ==
            weft::riscv_internal::logicalBitWidth(
                weft::riscv_internal::logicalElement(
                    field.getResult().getType())) &&
        relation->shift == 0 && relation->mask == 0)
      return raw;
    auto resultType = riscv::ValueType::get(
        rewriter.getContext(),
        weft::riscv_internal::logicalElement(field.getResult().getType()),
        mlir::cast<riscv::ValueType>(raw.getType()).getShape(),
        mlir::cast<riscv::ValueType>(raw.getType()).getAxisIds(),
        mlir::cast<riscv::ValueType>(raw.getType()).getLayout());
    llvm::StringRef instruction =
        relation->shift > 0
            ? (relation->mask > 0
                   ? "rvv.record-storage-decode.shift-mask"
                   : "rvv.record-storage-decode.shift")
            : (relation->mask > 0 ? "rvv.record-storage-decode.mask"
                                  : "rvv.record-storage-decode.identity");
    auto decoded = riscv::RVVRecordStorageDecodeOp::create(rewriter,
        field.getLoc(), resultType, raw, logicalIndex, relation->shift,
        relation->mask,
        weft::riscv_internal::leaf(
            rewriter, "rvv", "record-storage-decode", instruction,
            instruction,
            mlir::cast<riscv::ValueType>(raw.getType())
                .getLayout()
                .getRegisterGroups(),
            resultType.getLayout().getRegisterGroups()));
    weft::riscv_internal::copyOrigin(field, decoded);
    return decoded.getResult();
  }

  RecordLoopMatch &match;
  mlir::IRRewriter &rewriter;
  riscv::PhysicalPointOp point;
  riscv::RecordCohortOp cohort;
  llvm::DenseMap<ValueKey, mlir::Value, ValueKeyInfo> values;
  llvm::DenseMap<RawKey, mlir::Value, RawKeyInfo> rawValues;
  llvm::DenseMap<mlir::Value, mlir::Value> fields;
  llvm::DenseMap<mlir::Value, mlir::Value> loads;
  llvm::DenseMap<mlir::Value, bool> fieldDependence;
};

mlir::FailureOr<mlir::Value>
cloneIndexValue(mlir::Value value, mlir::scf::ForOp source,
                mlir::Value targetInduction, mlir::IRRewriter &rewriter,
                mlir::IRMapping &mapping) {
  if (value == source.getInductionVar())
    return targetInduction;
  if (mlir::Value mapped = mapping.lookupOrNull(value))
    return mapped;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || definition->getParentOp() != source.getOperation())
    return value;
  if (!definition->getDialect() ||
      definition->getDialect()->getNamespace() != "arith" ||
      !mlir::isMemoryEffectFree(definition) || definition->getNumRegions() != 0)
    return mlir::failure();
  for (mlir::Value operand : definition->getOperands()) {
    auto cloned = cloneIndexValue(operand, source, targetInduction, rewriter,
                                  mapping);
    if (mlir::failed(cloned))
      return mlir::failure();
    mapping.map(operand, *cloned);
  }
  mlir::Operation *cloned = rewriter.clone(*definition, mapping);
  for (auto [oldResult, newResult] :
       llvm::zip(definition->getResults(), cloned->getResults()))
    mapping.map(oldResult, newResult);
  return mapping.lookup(value);
}

mlir::LogicalResult vectorizeRecordLoop(RecordLoopMatch &match,
                                        mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(match.loop);
  auto width = mlir::arith::ConstantIndexOp::create(rewriter,
      match.loop.getLoc(), match.cohortWidth);
  auto groups = mlir::arith::DivUIOp::create(rewriter,
      match.loop.getLoc(), match.loop.getUpperBound(), width);
  auto mainUpper = mlir::arith::MulIOp::create(rewriter,
      match.loop.getLoc(), groups, width);
  auto mainLoop = mlir::scf::ForOp::create(rewriter,
      match.loop.getLoc(), match.loop.getLowerBound(), mainUpper, width);
  mainLoop->setAttr("weft.riscv.level", match.level);
  mainLoop->setAttr("weft.riscv.record_vectorized", rewriter.getUnitAttr());
  if (auto origin = match.loop->getAttr("source_origin"))
    mainLoop->setAttr("source_origin", origin);

  rewriter.setInsertionPoint(mainLoop.getBody()->getTerminator());
  mlir::IRMapping indexMapping;
  indexMapping.map(match.loop.getInductionVar(), mainLoop.getInductionVar());
  auto base = cloneIndexValue(match.point.getBase(), match.loop,
                              mainLoop.getInductionVar(), rewriter,
                              indexMapping);
  if (mlir::failed(base)) {
    rewriter.eraseOp(mainLoop);
    rewriter.eraseOp(mainUpper);
    rewriter.eraseOp(groups);
    rewriter.eraseOp(width);
    return mlir::failure();
  }
  auto point = riscv::PhysicalPointOp::create(rewriter,
      match.point.getLoc(), match.point.getResult().getType(),
      match.point.getParent(), *base, match.point.getActive(),
      match.point.getPartition());
  weft::riscv_internal::copyOrigin(match.point, point);
  auto cohortType = riscv::RecordCohortType::get(
      rewriter.getContext(), match.point.getResult().getType().getDomain(),
      match.cohortWidth, match.partition);
  auto cohort = riscv::RecordCohortOp::create(rewriter,
      match.point.getLoc(), cohortType, point.getResult());
  weft::riscv_internal::copyOrigin(match.point, cohort);

  RecordLoopMaterializer materializer(match, rewriter, point, cohort);
  riscv::MemDescType destination = match.destinationSlice.getBase().getType();
  auto access = denseStridedAccess(rewriter, destination);
  for (int64_t logicalIndex = 0; logicalIndex < match.partition;
       ++logicalIndex) {
    auto value = materializer.materialize(match.store.getValue(), logicalIndex);
    if (mlir::failed(value)) {
      rewriter.eraseOp(mainLoop);
      return mlir::failure();
    }
    auto store = riscv::RVVRecordStoreOp::create(rewriter,
        match.store.getLoc(), *value, match.destinationSlice.getBase(),
        cohort.getResult(), logicalIndex, match.destinationRecordByteStride,
        access,
        weft::riscv_internal::leaf(
            rewriter, "transfer", "record-store",
            "rvv.record-store.strided", "rvv.record-store.strided",
            mlir::cast<riscv::ValueType>((*value).getType())
                .getLayout()
                .getRegisterGroups(),
            0));
    weft::riscv_internal::copyOrigin(match.store, store);
  }
  match.loop.getLowerBoundMutable().assign(mainUpper.getResult());
  return mlir::success();
}

class VectorizeRISCVRecordLoopsPass
    : public mlir::PassWrapper<VectorizeRISCVRecordLoopsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeRISCVRecordLoopsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-vectorize-record-loops";
  }
  llvm::StringRef getDescription() const override {
    return "Map independent exact Level instances to an explicit RVV record cohort";
  }

  void runOnOperation() override {
    llvm::SmallVector<RecordLoopMatch> matches;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (auto match = matchRecordLoop(loop))
        matches.push_back(*match);
    });
    mlir::IRRewriter rewriter(&getContext());
    for (RecordLoopMatch &match : matches)
      if (mlir::failed(vectorizeRecordLoop(match, rewriter))) {
        match.loop.emitError(
            "failed to materialize a proven record-cohort vectorization");
        signalPassFailure();
        return;
      }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createVectorizeRISCVRecordLoopsPass() {
  return std::make_unique<VectorizeRISCVRecordLoopsPass>();
}
