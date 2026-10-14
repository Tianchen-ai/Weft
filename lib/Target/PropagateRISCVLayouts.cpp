#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "ContractionSupply.h"
#include "RISCVResidencyPlanning.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

struct Roles {
  int64_t laneAxis = 0;
  // Some target operations consume a cartesian logical tile in one RVV
  // register group.  laneAxis is the instruction's primary lane coordinate
  // (for example a contraction reduction axis); coalescedLaneAxes are
  // surviving logical coordinates that are linearized beside it.  Keeping the
  // axes distinct here lets later typed operations recover each logical
  // coordinate while the physical lane count is their product.
  llvm::SmallSet<int64_t, 4> coalescedLaneAxes;
  llvm::SmallSet<int64_t, 4> replicaAxes;
  llvm::SmallSet<int64_t, 4> sequentialAxes;
  bool local = false;
  bool anchored = false;
  bool fullLaneExtent = false;
  bool registerTuple = false;
  // A typed affine extract may have a storage-contiguous axis that conflicts
  // with a downstream compute layout.  Keep that memory anchor intact and let
  // insertUseConversions materialize the representation change on the SSA use
  // edge.  Propagating the compute layout back into the extract would turn a
  // provably contiguous window into an indexed gather.
  bool memoryAnchored = false;
};

struct AffineAxes {
  llvm::DenseMap<int64_t, int64_t> coefficients;
  bool valid = true;
};

bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs == 0 || rhs == 0) {
    result = 0;
    return true;
  }
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

void addAxisCoefficient(AffineAxes &result, int64_t axis,
                        int64_t coefficient) {
  if (!result.valid)
    return;
  int64_t next = result.coefficients.lookup(axis);
  if ((coefficient > 0 &&
       next > std::numeric_limits<int64_t>::max() - coefficient) ||
      (coefficient < 0 &&
       next < std::numeric_limits<int64_t>::min() - coefficient)) {
    result.valid = false;
    return;
  }
  next += coefficient;
  if (next)
    result.coefficients[axis] = next;
  else
    result.coefficients.erase(axis);
}

void decomposeAffineAxes(mlir::Value value, int64_t coefficient,
                         AffineAxes &result, unsigned depth = 0) {
  if (!result.valid || depth > 24)
    return result.valid = false, void();
  if (riscv_internal::constantInt(value))
    return;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return result.valid = false, void();
    decomposeAffineAxes(conversion.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
    decomposeAffineAxes(cast.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    auto type = iota.getResult().getType();
    if (type.getShape().size() != 1 || type.getAxisIds().size() != 1 ||
        iota.getEnd() - iota.getStart() != type.getShape()[0])
      return result.valid = false, void();
    addAxisCoefficient(result, type.getAxisIds()[0], coefficient);
    return;
  }
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    llvm::StringRef kind = binary.getKind();
    if (kind == "add" || kind == "sub") {
      decomposeAffineAxes(binary.getLhs(), coefficient, result, depth + 1);
      int64_t rhsCoefficient = coefficient;
      if (kind == "sub" &&
          !checkedMultiply(coefficient, -1, rhsCoefficient))
        return result.valid = false, void();
      decomposeAffineAxes(binary.getRhs(), rhsCoefficient, result, depth + 1);
      return;
    }
    if (kind == "mul") {
      if (auto lhs = riscv_internal::constantInt(binary.getLhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *lhs, scaled))
          return result.valid = false, void();
        decomposeAffineAxes(binary.getRhs(), scaled, result, depth + 1);
        return;
      }
      if (auto rhs = riscv_internal::constantInt(binary.getRhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *rhs, scaled))
          return result.valid = false, void();
        decomposeAffineAxes(binary.getLhs(), scaled, result, depth + 1);
        return;
      }
    }
  }
  // Scalar loop/Level bases do not affect which shaped axis is contiguous.
  if (!mlir::isa<riscv::ValueType>(value.getType()))
    return;
  result.valid = false;
}

std::optional<Roles> storageRolesFor(mlir::Value value,
                                     bool requireWideningConsumer = true) {
  auto extract = value.getDefiningOp<riscv::ExtractOp>();
  auto result = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!extract || !result)
    return std::nullopt;
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  bool feedsWideningContraction = !requireWideningConsumer;
  while (!worklist.empty() && !feedsWideningContraction) {
    mlir::Value current = worklist.pop_back_val();
    for (mlir::Operation *user : current.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (mlir::isa<riscv::DotOp, riscv::ContractOp,
                    riscv::OuterContractOp>(user)) {
        auto implementation =
            user->getAttrOfType<riscv::ImplementationAttr>("implementation");
        feedsWideningContraction =
            implementation && implementation.getFamily() == "widen-dot" &&
            (user->getOperand(0) == current || user->getOperand(1) == current);
        continue;
      }
      if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp, riscv::SelectOp,
                     riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp, riscv::WidenOp,
                     riscv::ConvertLayoutOp>(user) ||
          user->getNumResults() != 1)
        continue;
      auto next = mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (next && next.getShape() == result.getShape() &&
          next.getAxisIds() == result.getAxisIds())
        worklist.push_back(user->getResult(0));
    }
  }
  if (!feedsWideningContraction)
    return std::nullopt;
  auto field = riscv_internal::sourceField(extract.getInput());
  if (!field)
    return std::nullopt;
  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (facts.mapping != "natural" && facts.mapping != "grouped_layered")
    return std::nullopt;

  size_t indexCursor = 0;
  mlir::Value gatherIndex;
  for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (indexCursor >= extract.getIndices().size())
      return std::nullopt;
    mlir::Value index = extract.getIndices()[indexCursor++];
    if (selector == "gather") {
      if (gatherIndex)
        return std::nullopt;
      gatherIndex = index;
    }
  }
  if (!gatherIndex || indexCursor != extract.getIndices().size())
    return std::nullopt;

  AffineAxes affine;
  decomposeAffineAxes(gatherIndex, 1, affine);
  if (!affine.valid)
    return std::nullopt;
  int64_t contiguousAxis = 0;
  for (auto [axis, extent] :
       llvm::zip(result.getAxisIds().asArrayRef(), result.getShape().asArrayRef())) {
    if (affine.coefficients.lookup(axis) != 1 || extent <= 1)
      continue;
    if (facts.mapping == "grouped_layered" &&
        (facts.layer <= 0 || extent > facts.layer))
      continue;
    if (contiguousAxis)
      return std::nullopt;
    contiguousAxis = axis;
  }
  if (!contiguousAxis)
    return std::nullopt;

  Roles roles;
  roles.laneAxis = contiguousAxis;
  if (facts.mapping == "natural" || facts.mapping == "grouped_layered") {
    int64_t span = 0;
    for (auto [axis, extent] : llvm::zip(result.getAxisIds().asArrayRef(),
                                         result.getShape().asArrayRef()))
      if (axis == contiguousAxis)
        span = extent;
    if (span <= 0)
      return std::nullopt;
    llvm::SmallSet<int64_t, 4> consumed{contiguousAxis};
    while (true) {
      int64_t nextAxis = 0;
      int64_t nextExtent = 0;
      for (auto [axis, extent] : llvm::zip(
               result.getAxisIds().asArrayRef(), result.getShape().asArrayRef())) {
        if (consumed.contains(axis) || extent <= 1 ||
            affine.coefficients.lookup(axis) != span)
          continue;
        if (nextAxis)
          return std::nullopt;
        nextAxis = axis;
        nextExtent = extent;
      }
      if (!nextAxis)
        break;
      int64_t nextSpan = 0;
      if (!checkedMultiply(span, nextExtent, nextSpan))
        return std::nullopt;
      // A grouped/layered field is contiguous only inside one physical layer.
      // Crossing the layer boundary changes which packed bits supply the
      // logical value, so that axis must remain a replica/time coordinate.
      if (facts.mapping == "grouped_layered" &&
          (facts.layer <= 0 || nextSpan > facts.layer))
        break;
      roles.coalescedLaneAxes.insert(nextAxis);
      consumed.insert(nextAxis);
      span = nextSpan;
    }
  }
  roles.anchored = true;
  roles.fullLaneExtent = true;
  roles.memoryAnchored = true;
  for (int64_t axis : riscv_internal::logicalAxes(value.getType())) {
    const int64_t extent = riscv_internal::physicalExtent(value, axis);
    if (axis != contiguousAxis &&
        !roles.coalescedLaneAxes.contains(axis) && extent > 0 && extent <= 16)
      roles.replicaAxes.insert(axis);
  }
  return roles;
}

bool containsAxis(mlir::Type type, int64_t axis) {
  return llvm::is_contained(riscv_internal::logicalAxes(type), axis);
}

void addSmallReplicas(mlir::Value value, Roles &roles, int64_t except = 0) {
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (int64_t extent = riscv_internal::physicalExtent(value, axis);
        axis != except && !roles.coalescedLaneAxes.contains(axis) &&
        !roles.sequentialAxes.contains(axis) && extent > 0 &&
        extent <= 16)
      roles.replicaAxes.insert(axis);
}

bool layoutPreservingPointwise(mlir::Operation *operation) {
  return mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp, riscv::SelectOp,
                   riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp, riscv::WidenOp,
                   riscv::ConvertLayoutOp, riscv::MaterializeOp>(operation);
}

bool preservesExistingAxes(mlir::Type sourceType, mlir::Type resultType) {
  auto source = mlir::dyn_cast<riscv::ValueType>(sourceType);
  auto result = mlir::dyn_cast<riscv::ValueType>(resultType);
  if (!source || !result)
    return false;
  for (auto [position, axis] :
       llvm::enumerate(source.getAxisIds().asArrayRef())) {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (found == result.getAxisIds().asArrayRef().end() ||
        source.getShape()[position] !=
            result.getShape()[static_cast<size_t>(
                found - result.getAxisIds().asArrayRef().begin())])
      return false;
  }
  return true;
}

bool feedsExpandedIndexedLookup(mlir::Value root) {
  auto rootType = mlir::dyn_cast<riscv::ValueType>(root.getType());
  if (!rootType)
    return false;
  int64_t nonSingletonAxes = 0;
  for (int64_t axis : rootType.getAxisIds().asArrayRef()) {
    const int64_t extent = riscv_internal::physicalExtent(root, axis);
    if (extent <= 0)
      return false;
    nonSingletonAxes += extent > 1;
  }
  if (nonSingletonAxes != 1)
    return false;
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(user)) {
        auto indices = mlir::dyn_cast<riscv::ValueType>(
            lookup.getIndices().getType());
        if (use.getOperandNumber() == 1 && indices &&
            indices.getAxisIds().size() > rootType.getAxisIds().size() &&
            preservesExistingAxes(root.getType(), indices))
          return true;
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1 ||
          !visited.insert(user).second ||
          !preservesExistingAxes(value.getType(), user->getResult(0).getType()))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return false;
}

std::optional<int64_t>
downstreamImmediateReducedFreeAxis(mlir::Operation *contraction,
                                   llvm::ArrayRef<int64_t> reductionAxes) {
  if (!contraction || contraction->getNumResults() != 1)
    return std::nullopt;
  llvm::SmallVector<mlir::Value> worklist{contraction->getResult(0)};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  std::optional<int64_t> reducedAxis;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!source)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(user)) {
        if (reduce.getAxis() < 0 ||
            reduce.getAxis() >= static_cast<int64_t>(source.getAxisIds().size()))
          continue;
        const int64_t axis = source.getAxisIds()[reduce.getAxis()];
        if (llvm::is_contained(reductionAxes, axis))
          continue;
        if (reducedAxis && *reducedAxis != axis)
          return std::nullopt;
        reducedAxis = axis;
        // Preserve the explicit nested reduction tree.  Only the first
        // reduction outside the contraction may join its lane coordinate;
        // outer reductions remain independent partial/register coordinates.
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result = mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result ||
          !preservesExistingAxes(value.getType(), user->getResult(0).getType()))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return reducedAxis;
}

llvm::SmallVector<int64_t>
downstreamReducedFreeAxes(mlir::Operation *contraction,
                          llvm::ArrayRef<int64_t> contractionAxes) {
  llvm::SmallVector<int64_t> reducedAxes;
  if (!contraction || contraction->getNumResults() != 1)
    return reducedAxes;

  llvm::SmallVector<mlir::Value> worklist{contraction->getResult(0)};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!source)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(user)) {
        if (reduce.getAxis() < 0 ||
            reduce.getAxis() >= static_cast<int64_t>(source.getAxisIds().size()))
          continue;
        const int64_t axis = source.getAxisIds()[reduce.getAxis()];
        if (!llvm::is_contained(contractionAxes, axis) &&
            !llvm::is_contained(reducedAxes, axis))
          reducedAxes.push_back(axis);
        worklist.push_back(reduce.getResult());
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result ||
          !preservesExistingAxes(value.getType(), user->getResult(0).getType()))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return reducedAxes;
}

llvm::SmallVector<int64_t> downstreamContractionAxes(mlir::Value root) {
  llvm::SmallVector<int64_t> axes;
  using WorkItem = std::pair<mlir::Value, llvm::SmallVector<int64_t>>;
  llvm::SmallVector<WorkItem> worklist;
  worklist.emplace_back(root, riscv_internal::logicalAxes(root.getType()));
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    auto [value, preservedAxes] = worklist.pop_back_val();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!source)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (auto contract = mlir::dyn_cast<riscv::ContractOp>(user)) {
        auto implementation =
            contract->getAttrOfType<riscv::ImplementationAttr>("implementation");
        if (!implementation || implementation.getFamily() != "widen-dot" ||
            (contract.getLhs() != value && contract.getRhs() != value))
          continue;
        for (int64_t axis : contract.getOver())
          if (llvm::is_contained(preservedAxes, axis) &&
              !llvm::is_contained(axes, axis))
            axes.push_back(axis);
        for (int64_t axis : downstreamReducedFreeAxes(
                 contract.getOperation(), contract.getOver()))
          if (llvm::is_contained(preservedAxes, axis) &&
              !llvm::is_contained(axes, axis))
            axes.push_back(axis);
        continue;
      }
      if (auto reshape = mlir::dyn_cast<riscv::ReshapeOp>(user)) {
        auto result = reshape.getResult().getType();
        if (reshape.getInput() != value ||
            !llvm::equal(reshape.getOrder(), source.getAxisIds().asArrayRef()))
          continue;
        // A suffix reshape changes only its mixed-radix coordinates.  The
        // identical prefix still names the same entries supplied to a later
        // contraction; new/reused suffix axis names are not evidence of that.
        llvm::SmallVector<int64_t> prefix;
        for (size_t position = 0;
             position < source.getShape().size() &&
             position < result.getShape().size(); ++position) {
          const int64_t axis = source.getAxisIds()[position];
          if (axis != result.getAxisIds()[position] ||
              source.getShape()[position] != result.getShape()[position])
            break;
          if (llvm::is_contained(preservedAxes, axis))
            prefix.push_back(axis);
        }
        if (!prefix.empty())
          worklist.emplace_back(reshape.getResult(), std::move(prefix));
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result || result.getShape() != source.getShape() ||
          result.getAxisIds() != source.getAxisIds())
        continue;
      worklist.emplace_back(user->getResult(0), preservedAxes);
    }
  }
  return axes;
}

bool supportsPackedIndexedEntryLookup(riscv::LookupOp lookup,
                                      const riscv_internal::IndexedEntryRelation &relation,
                                      riscv::ValueType result) {
  const int64_t packedBits =
      riscv_internal::logicalBitWidth(result.getElementType()) *
      relation.payloadExtent;
  if (packedBits != 32 && packedBits != 64)
    return false;
  mlir::Value table =
      riscv_internal::stripRepresentationConversions(lookup.getTable());
  mlir::Value descriptor = table;
  if (auto load = table.getDefiningOp<riscv::LoadOp>())
    descriptor = load.getRegion();
  int64_t alignment = 1;
  int64_t bitOffset = 0;
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(descriptor.getType()))
    alignment = memory.getAlignment();
  if (auto field = descriptor.getDefiningOp<riscv::FieldOp>()) {
    auto facts = riscv_internal::fieldFacts(field);
    alignment = facts.alignment;
    bitOffset = facts.bitOffset;
  }
  return alignment >= packedBits / 8 && bitOffset % packedBits == 0;
}

bool constrainIndexedEntryLookup(riscv::LookupOp lookup, mlir::Value value,
                                 Roles &roles) {
  auto result = mlir::dyn_cast<riscv::ValueType>(lookup.getResult().getType());
  if (!result)
    return false;
  auto relation = riscv_internal::analyzeIndexedEntryRelation(
      lookup.getIndices(), result, {}, {});
  if (!relation || !containsAxis(value.getType(), relation->payloadAxis))
    return false;
  // The payload inside one table entry owns the primary memory-facing lane
  // coordinate. Entry coordinates consumed by the same contraction can join
  // that lane domain; surviving axes remain independent replicas.
  roles.anchored = true;
  roles.fullLaneExtent = true;
  roles.memoryAnchored = true;
  roles.laneAxis = relation->payloadAxis;
  if (supportsPackedIndexedEntryLookup(lookup, *relation, result)) {
    llvm::SmallVector<int64_t> contractionAxes =
        downstreamContractionAxes(lookup.getResult());
    auto entryType =
        mlir::dyn_cast<riscv::ValueType>(relation->entryIndices.getType());
    if (entryType)
      for (int64_t axis : entryType.getAxisIds().asArrayRef())
        if (llvm::is_contained(contractionAxes, axis))
          roles.coalescedLaneAxes.insert(axis);
  }
  addSmallReplicas(value, roles, roles.laneAxis);
  return true;
}

bool storageSupportsLaneAxes(mlir::Value root,
                             llvm::ArrayRef<int64_t> requestedAxes) {
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      continue;
    if (mlir::isa<riscv::ExtractOp>(definition)) {
      auto storage = storageRolesFor(value);
      if (!storage || !storage->memoryAnchored)
        continue;
      for (int64_t axis : requestedAxes)
        if (containsAxis(value.getType(), axis) &&
            axis != storage->laneAxis &&
            !storage->coalescedLaneAxes.contains(axis))
          return false;
      continue;
    }
    if (!layoutPreservingPointwise(definition))
      continue;
    for (mlir::Value operand : definition->getOperands()) {
      auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
      auto resultType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      if (operandType && resultType &&
          operandType.getShape() == resultType.getShape() &&
          operandType.getAxisIds() == resultType.getAxisIds())
        worklist.push_back(operand);
    }
  }
  return true;
}

int64_t freeAxisSharedByOneOperand(mlir::Value lhs, mlir::Value rhs) {
  auto lhsAxes = riscv_internal::logicalAxes(lhs.getType());
  auto rhsAxes = riscv_internal::logicalAxes(rhs.getType());
  for (int64_t axis : lhsAxes)
    if (!llvm::is_contained(rhsAxes, axis))
      return axis;
  for (int64_t axis : rhsAxes)
    if (!llvm::is_contained(lhsAxes, axis))
      return axis;
  return 0;
}

bool hasOperandLocalFreeAxis(mlir::Value value, mlir::Value other,
                             llvm::ArrayRef<int64_t> reductionAxes) {
  auto otherAxes = riscv_internal::logicalAxes(other.getType());
  return llvm::any_of(riscv_internal::logicalAxes(value.getType()),
                      [&](int64_t axis) {
                        return !llvm::is_contained(reductionAxes, axis) &&
                               !llvm::is_contained(otherAxes, axis);
                      });
}

void constrainGroupedMac(riscv::MacGroupsOp operation, mlir::Value value,
                         Roles &roles) {
  roles.anchored = true;
  roles.fullLaneExtent = true;
  // A grouped MAC consumes one shared reduction coordinate.  If one operand
  // also carries an output/cohort axis, that free axis is the reusable vector
  // direction.  Falling back to the final reduction axis is only valid for a
  // scalar-output vec-dot.  This is an axis relation, not a source-shape
  // matcher: canonical and derived encodings use the same rule.
  int64_t groupedAxis = 0;
  auto lhsAxes = riscv_internal::logicalAxes(operation.getLhs().getType());
  auto rhsAxes = riscv_internal::logicalAxes(operation.getRhs().getType());
  if (!lhsAxes.empty() && !rhsAxes.empty() && lhsAxes.back() == rhsAxes.back())
    groupedAxis = lhsAxes.back();
  auto integerOf = [](mlir::Value operand) {
    return mlir::dyn_cast<mlir::IntegerType>(
        riscv_internal::logicalElement(operand.getType()));
  };
  auto lhsInteger = integerOf(operation.getLhs());
  auto rhsInteger = integerOf(operation.getRhs());
  mlir::Value packedOperand;
  if (lhsInteger && rhsInteger) {
    if (lhsInteger.isUnsigned() && rhsInteger.isSigned())
      packedOperand = operation.getLhs();
    if (rhsInteger.isUnsigned() && lhsInteger.isSigned())
      packedOperand = operation.getRhs();
  }
  int64_t lane = 0;
  if (packedOperand)
    for (int64_t axis : riscv_internal::logicalAxes(packedOperand.getType()))
      if (axis != groupedAxis) {
        lane = axis;
        break;
      }
  if (!lane)
    lane = freeAxisSharedByOneOperand(operation.getLhs(), operation.getRhs());
  if (!lane) {
    lane = groupedAxis;
  }
  // When a free output axis occupies lanes, the final common axis remains the
  // grouped reduction coordinate.  It is issued in time; treating its small
  // extent as register replicas would keep every term live simultaneously and
  // multiply the partial resource count by the group size.
  if (groupedAxis && lane != groupedAxis &&
      containsAxis(value.getType(), groupedAxis))
    roles.sequentialAxes.insert(groupedAxis);
  if (lane && containsAxis(value.getType(), lane))
    roles.laneAxis = lane;
  addSmallReplicas(value, roles, roles.laneAxis);
}

int64_t rhsFreeLane(mlir::Value lhs, mlir::Value rhs,
                    llvm::ArrayRef<int64_t> reduction) {
  for (auto it = riscv_internal::logicalAxes(rhs.getType()).rbegin();
       it != riscv_internal::logicalAxes(rhs.getType()).rend(); ++it)
    if (!llvm::is_contained(reduction, *it) &&
        !llvm::is_contained(riscv_internal::logicalAxes(lhs.getType()), *it))
      return *it;
  for (auto it = riscv_internal::logicalAxes(lhs.getType()).rbegin();
       it != riscv_internal::logicalAxes(lhs.getType()).rend(); ++it)
    if (!llvm::is_contained(reduction, *it) &&
        !llvm::is_contained(riscv_internal::logicalAxes(rhs.getType()), *it))
      return *it;
  return 0;
}

std::optional<Roles>
reductionSupplyRoles(mlir::Value value,
                     llvm::ArrayRef<int64_t> allowedAxes,
                     bool projectFreeAxes = false) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  std::optional<Roles> result;
  auto mergeCandidate = [&](Roles candidate) {
    if (!candidate.laneAxis ||
        !llvm::is_contained(allowedAxes, candidate.laneAxis))
      return true;
    for (int64_t axis :
         llvm::SmallVector<int64_t>(candidate.coalescedLaneAxes.begin(),
                                    candidate.coalescedLaneAxes.end()))
      if (!llvm::is_contained(allowedAxes, axis)) {
        if (projectFreeAxes) {
          candidate.coalescedLaneAxes.erase(axis);
          continue;
        }
        return true;
      }
    if (!result) {
      result = candidate;
      return true;
    }
    if (result->laneAxis != candidate.laneAxis)
      return false;
    for (int64_t axis : candidate.coalescedLaneAxes)
      result->coalescedLaneAxes.insert(axis);
    result->fullLaneExtent |= candidate.fullLaneExtent;
    result->memoryAnchored |= candidate.memoryAnchored;
    return true;
  };

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      continue;
    if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(definition)) {
      Roles candidate;
      if (constrainIndexedEntryLookup(lookup, current, candidate) &&
          !mergeCandidate(candidate))
        return std::nullopt;
      continue;
    }
    if (mlir::isa<riscv::ExtractOp>(definition)) {
      auto candidate = storageRolesFor(current, false);
      if (candidate && !mergeCandidate(*candidate))
        return std::nullopt;
      continue;
    }
    if (!layoutPreservingPointwise(definition) ||
        definition->getNumResults() != 1)
      continue;
    auto currentType = mlir::dyn_cast<riscv::ValueType>(current.getType());
    if (!currentType)
      continue;
    for (mlir::Value operand : definition->getOperands()) {
      auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
      if (operandType && operandType.getShape() == currentType.getShape() &&
          operandType.getAxisIds() == currentType.getAxisIds())
        worklist.push_back(operand);
    }
  }
  return result;
}

riscv::FragmentCapabilityAttr selectedContractFragment(mlir::Operation *operation) {
  auto implementation = operation->getAttrOfType<riscv::ImplementationAttr>(
      "implementation");
  if (!implementation || implementation.getEngine() != "ime" ||
      implementation.getFamily() != "fragment-mma")
    return {};
  for (mlir::Attribute candidate :
       operation->getParentOfType<riscv::KernelOp>().getTarget().getFragments()) {
    auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(candidate);
    if (fragment.getInstruction() == implementation.getOperation())
      return fragment;
  }
  return {};
}

bool hasContiguousFragmentSupply(mlir::Value value, int64_t axis) {
  return llvm::is_contained(
      riscv_internal::queryContractionSupply(value).contiguousAxes, axis);
}

template <typename Contract>
bool constrainFragmentContractOperand(Contract operation, mlir::Value value,
                                      Roles &roles) {
  const bool lhs = value == operation.getLhs();
  if (!lhs && value != operation.getRhs()) return false;
  auto fragment = selectedContractFragment(operation);
  if (!fragment) return false;
  auto result = mlir::cast<riscv::ValueType>(operation->getResult(0).getType());
  auto reduction = operation.getOver();
  auto packing = lhs ? fragment.getLhsPacking() : fragment.getRhsPacking();
  if (result.getAxisIds().size() != 2 || reduction.size() != 1 ||
      packing.getAxisOrder().size() != 2 ||
      packing.getElementOrder() != "row_major")
    return false;
  int64_t axes[] = {lhs ? result.getAxisIds()[0] : reduction[0],
                    lhs ? reduction[0] : result.getAxisIds()[1]};
  const int64_t inner = axes[packing.getAxisOrder()[1]];
  if (!hasContiguousFragmentSupply(value, inner)) return false;
  // A unit-stride or already formed inner-axis carrier can feed this packing
  // directly. Otherwise preserve the existing memory-facing lane choice and
  // let the explicit fragment pack perform its required register conversion.
  roles.anchored = true;
  roles.fullLaneExtent = true;
  roles.laneAxis = inner;
  addSmallReplicas(value, roles, inner);
  return true;
}

template <typename Contract>
void constrainOuterContractOperand(Contract operation, mlir::Value value,
                                   Roles &roles) {
  if (constrainFragmentContractOperand(operation, value, roles)) return;
  roles.anchored = true;
  auto reduction = operation.getOver();
  int64_t lane = rhsFreeLane(operation.getLhs(), operation.getRhs(), reduction);
  auto implementation = operation->template getAttrOfType<
      riscv::ImplementationAttr>("implementation");
  bool registerFloat = false;
  if (implementation && (implementation.getOperation() == "rvv.vfmacc" ||
                         implementation.getOperation() == "rvv.vfwmacc")) {
    auto lhsSupply = riscv_internal::queryContractionSupply(operation.getLhs());
    auto rhsSupply = riscv_internal::queryContractionSupply(operation.getRhs());
    const bool lhsMemory = lhsSupply.hasMemorySupply();
    const bool rhsMemory = rhsSupply.hasMemorySupply();
    if (!lhsMemory || !rhsMemory) {
      auto freeLane = [&](mlir::Value input, mlir::Value other) {
        for (int64_t axis :
             llvm::reverse(riscv_internal::logicalAxes(input.getType())))
          if (!llvm::is_contained(reduction, axis) &&
              !containsAxis(other.getType(), axis) &&
              riscv_internal::physicalExtent(input, axis) > 1)
            return axis;
        return int64_t{0};
      };
      int64_t lhsLane = freeLane(operation.getLhs(), operation.getRhs());
      int64_t rhsLane = freeLane(operation.getRhs(), operation.getLhs());
      if (rhsLane && llvm::is_contained(rhsSupply.birthUnitAxes, rhsLane))
        lane = rhsLane;
      else if (lhsLane && llvm::is_contained(lhsSupply.birthUnitAxes, lhsLane))
        lane = lhsLane;
      else if (rhsMemory && rhsLane)
        lane = rhsLane;
      else if (lhsMemory && lhsLane)
        lane = lhsLane;
      else
        lane = rhsLane ? rhsLane : lhsLane;
      registerFloat = lane != 0;
    }
  }
  if (implementation &&
      ((implementation.getFamily() == "widen-float-contract" && !registerFloat) ||
       implementation.getFamily() == "widen-dot" ||
       (implementation.getOperation() == "rvv.vfmacc" && !registerFloat))) {
    roles.fullLaneExtent = true;
    for (int64_t axis : reduction)
      if (containsAxis(value.getType(), axis)) {
        roles.laneAxis = axis;
        break;
      }
    if (!roles.laneAxis)
      roles.registerTuple = true;
    addSmallReplicas(value, roles, roles.laneAxis);
    return;
  }
  if (lane && containsAxis(value.getType(), lane))
    roles.laneAxis = lane;
  else
    for (int64_t axis : reduction)
      if (containsAxis(value.getType(), axis)) {
        roles.laneAxis = axis;
        break;
      }
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (!llvm::is_contained(reduction, axis) && axis != roles.laneAxis &&
        riscv_internal::physicalExtent(value, axis) > 0 && riscv_internal::physicalExtent(value, axis) <= 16)
      roles.replicaAxes.insert(axis);
}

template <typename Contract>
void constrainReductionContractOperand(Contract operation, mlir::Value value,
                                       Roles &roles) {
  if (constrainFragmentContractOperand(operation, value, roles)) return;
  auto implementation = operation->template getAttrOfType<
      riscv::ImplementationAttr>("implementation");
  if (value == operation->getResult(0)) {
    if (auto fragment = selectedContractFragment(operation)) {
      auto packing = fragment.getAccumulatorPacking();
      if (packing.getAxisOrder().size() == 2 &&
          packing.getAxisOrder()[0] == 0 && packing.getAxisOrder()[1] == 1 &&
          packing.getElementOrder() == "row_major") {
        // This selected fragment hands off complete output rows to RVV. Its
        // result is not a tuple of scalar reduction finalizers; operand supply
        // still follows the independent reduction/storage constraints below.
        constrainOuterContractOperand(operation, value, roles);
        return;
      }
    }
  }
  if (implementation && (implementation.getOperation() == "rvv.vfmacc" ||
                         implementation.getOperation() == "rvv.vfwmacc") &&
      hasOperandLocalFreeAxis(operation.getLhs(), operation.getRhs(),
                              operation.getOver()) &&
      hasOperandLocalFreeAxis(operation.getRhs(), operation.getLhs(),
                              operation.getOver())) {
    // A contract whose two operands contribute independent output axes is the
    // same physical outer-product relation regardless of which canonical
    // spelling produced it. Its operand supplies determine the lane carrier.
    constrainOuterContractOperand(operation, value, roles);
    return;
  }
  roles.anchored = true;
  roles.fullLaneExtent = true;
  auto reduction = operation.getOver();
  int64_t reductionElements = 1;
  bool completeReduction = !reduction.empty();
  for (int64_t axis : reduction) {
    if (!containsAxis(value.getType(), axis)) {
      completeReduction = false;
      break;
    }
    const int64_t extent = riscv_internal::physicalExtent(value, axis);
    if (extent <= 0) {
      completeReduction = false;
      break;
    }
    reductionElements *= extent;
  }
  auto kernel = value.getParentRegion()->getParentOfType<riscv::KernelOp>();
  const int64_t sew =
      std::max<int64_t>(8, riscv_internal::logicalBitWidth(value.getType()));
  const int64_t baseLanes =
      kernel && sew > 0 ? kernel.getTarget().getVlenBits() / sew : 0;
  // A contraction shorter than one base RVV vector whose operand already
  // carries a free axis at least as wide as the reduction is an outer-product
  // issue: keep that free axis in lanes and advance the reduction in time.
  // A smaller free axis does not amortize serializing the larger reduction;
  // keep the reduction in lanes and preserve the free axis as replicas instead.
  // This is derived from the two axis extents and the target base vector, not
  // from an operator or encoding family.
  if (completeReduction && baseLanes > 1 && reductionElements < baseLanes)
    for (int64_t axis : riscv_internal::logicalAxes(value.getType())) {
      const int64_t extent = riscv_internal::physicalExtent(value, axis);
      if (!llvm::is_contained(reduction, axis) &&
          extent > reductionElements && extent <= baseLanes) {
        roles.laneAxis = axis;
        for (int64_t reductionAxis : reduction)
          if (containsAxis(value.getType(), reductionAxis))
            roles.sequentialAxes.insert(reductionAxis);
        return;
      }
    }
  const bool isOperand =
      value == operation.getLhs() || value == operation.getRhs();
  const bool cartesianOutput =
      hasOperandLocalFreeAxis(operation.getLhs(), operation.getRhs(),
                              reduction) &&
      hasOperandLocalFreeAxis(operation.getRhs(), operation.getLhs(), reduction);
  const auto lhsAxes = riscv_internal::logicalAxes(operation.getLhs().getType());
  const auto rhsAxes = riscv_internal::logicalAxes(operation.getRhs().getType());
  const bool sharedFreeAxis = llvm::any_of(lhsAxes, [&](int64_t axis) {
    return !llvm::is_contained(reduction, axis) &&
           llvm::is_contained(rhsAxes, axis);
  });
  if (!isOperand && cartesianOutput && !sharedFreeAxis) {
    // Each operand contributes an independently surviving output coordinate.
    // With no shared free coordinate, the reduction carrier feeds their final
    // Cartesian product, so neither output axis is a SIMD reduction lane.
    // Preserve every output coordinate as a register-replica slot.
    roles.registerTuple = true;
    addSmallReplicas(value, roles);
    return;
  }
  if (isOperand)
    if (auto supply = reductionSupplyRoles(value, reduction)) {
      roles.laneAxis = supply->laneAxis;
      roles.coalescedLaneAxes = supply->coalescedLaneAxes;
      roles.memoryAnchored = supply->memoryAnchored;
    }
  if (!isOperand && implementation &&
      implementation.getFamily() == "widen-dot") {
    // A widening reduction finalizes one scalar for each surviving coordinate.
    // Keep those coordinates as replicas; vector consumers convert explicitly.
    roles.registerTuple = true;
    roles.laneAxis = 0;
    roles.coalescedLaneAxes.clear();
    roles.replicaAxes.clear();
    roles.sequentialAxes.clear();
    addSmallReplicas(value, roles);
    return;
  }
  if (!roles.laneAxis)
    for (int64_t axis : reduction)
      if (containsAxis(value.getType(), axis)) {
        if (!roles.laneAxis)
          roles.laneAxis = axis;
        else
          roles.coalescedLaneAxes.insert(axis);
      }
  // A contraction result and its operands deliberately have different physical
  // layouts.  The result preserves every later reduction coordinate as an
  // independent register partial.  An operand may instead carry those same
  // coordinates beside K in one wide lane window, from which the typed partial
  // program takes narrow slices.  This is the RVV analogue of a Triton
  // DotOperandEncoding whose parent is the accumulator encoding: it changes the
  // operand representation, not the accumulator topology.
  auto reducedFreeAxes =
      downstreamReducedFreeAxes(operation.getOperation(), reduction);
  if (roles.laneAxis && isOperand)
    for (int64_t freeAxis : reducedFreeAxes)
      if (freeAxis != roles.laneAxis && containsAxis(value.getType(), freeAxis) &&
          storageSupportsLaneAxes(value, llvm::ArrayRef<int64_t>(freeAxis))) {
        roles.coalescedLaneAxes.insert(freeAxis);
        roles.replicaAxes.erase(freeAxis);
      }
  // A shaped contraction result no longer contains the eliminated reduction
  // axis.  When exactly one operand contributes a surviving free axis, that
  // axis is the contraction's reusable output cohort and preserves the same
  // lane organization already chosen on that operand.  A scalar-output dot has
  // no such axis and remains a register tuple.
  if (!roles.laneAxis) {
    if (auto reducedFreeAxis = downstreamImmediateReducedFreeAxis(
            operation.getOperation(), reduction);
        reducedFreeAxis && containsAxis(value.getType(), *reducedFreeAxis)) {
      roles.registerTuple = true;
      addSmallReplicas(value, roles);
      return;
    }
    const int64_t freeLane =
        freeAxisSharedByOneOperand(operation.getLhs(), operation.getRhs());
    if (freeLane && containsAxis(value.getType(), freeLane)) {
      roles.laneAxis = freeLane;
      addSmallReplicas(value, roles, freeLane);
      return;
    }
    roles.registerTuple = true;
    addSmallReplicas(value, roles);
    return;
  }
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (!llvm::is_contained(reduction, axis) && axis != roles.laneAxis &&
        !roles.coalescedLaneAxes.contains(axis) &&
        riscv_internal::physicalExtent(value, axis) > 0 &&
        riscv_internal::physicalExtent(value, axis) <= 16)
      roles.replicaAxes.insert(axis);
}

std::optional<Roles> fragmentConversionRoles(mlir::Value value) {
  if (!mlir::isa_and_nonnull<riscv::WidenOp, riscv::NarrowOp, riscv::CastOp,
                           riscv::MaterializeOp, riscv::ExtractOp>(value.getDefiningOp()) ||
      value.use_empty()) return std::nullopt;
  Roles roles;
  llvm::SmallVector<mlir::OpOperand *> consumers;
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  std::function<bool(mlir::Value)> collect = [&](mlir::Value current) {
    for (mlir::OpOperand &use : current.getUses()) {
      auto *op = use.getOwner();
      if (mlir::isa<riscv::MaterializeOp, riscv::ReshapeOp>(op) && use.getOperandNumber() == 0) {
        if (visited.insert(op).second && !collect(op->getResult(0))) return false;
      } else if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(op);
                 extract && use.getOperandNumber() == 0 &&
                 llvm::all_of(extract.getSelectors(), [](mlir::Attribute attr) {
                   auto selector = mlir::cast<mlir::StringAttr>(attr).getValue();
                   return selector == "all" || selector == "domain";
                 })) {
        if (visited.insert(op).second && !collect(op->getResult(0))) return false;
      } else consumers.push_back(&use);
    }
    return true;
  };
  if (!collect(value) || consumers.empty()) return std::nullopt;
  for (mlir::OpOperand *selectedUse : consumers) {
    auto &use = *selectedUse;
    auto *operation = use.getOwner();
    auto selected = operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
    if (!mlir::isa<riscv::ContractOp, riscv::OuterContractOp>(operation) ||
        !selected || selected.getFamily() != "fragment-mma" || use.getOperandNumber() > 1)
      return std::nullopt;
    auto result = mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
    auto reduction = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
    if (!result || result.getAxisIds().size() != 2 || !reduction || reduction.size() != 1)
      return std::nullopt;
    riscv::FragmentCapabilityAttr capability;
    for (auto candidate : operation->getParentOfType<riscv::KernelOp>().getTarget().getFragments()) {
      auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(candidate);
      if (fragment.getInstruction() == selected.getOperation()) { capability = fragment; break; }
    }
    if (!capability) return std::nullopt;
    bool lhs = use.getOperandNumber() == 0;
    auto packing = lhs ? capability.getLhsPacking() : capability.getRhsPacking();
    int64_t axes[] = {lhs ? result.getAxisIds()[0] : reduction[0],
                      lhs ? reduction[0] : result.getAxisIds()[1]};
    int64_t inner = axes[packing.getAxisOrder()[1]];
    int64_t extent = riscv_internal::physicalExtent(value, inner);
    if (extent <= 0 || extent % packing.getColumnsPerTile() ||
        (roles.laneAxis && roles.laneAxis != inner)) return std::nullopt;
    roles.laneAxis = inner;
  }
  // Converted numeric values follow the selected fragment's packing relation;
  // storage-facing operands keep their independent access/layout decisions.
  roles.anchored = true;
  roles.fullLaneExtent = true;
  addSmallReplicas(value, roles, roles.laneAxis);
  return roles;
}

bool canPlaceStateLocally(mlir::Value value) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    return mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp, riscv::LoopOp>(
        argument.getOwner()->getParentOp());
  mlir::Operation *definition = value.getDefiningOp();
  if (mlir::isa_and_nonnull<riscv::EmptyOp, riscv::NewOp, riscv::MaterializeOp,
                            riscv::LocalBindOp, mlir::scf::ForOp,
                            mlir::scf::IfOp, mlir::scf::WhileOp,
                            riscv::LoopOp>(definition))
    return true;
  if (auto update = mlir::dyn_cast_or_null<riscv::UpdateOp>(definition))
    return canPlaceStateLocally(update.getInput());
  if (auto reshape = mlir::dyn_cast_or_null<riscv::ReshapeOp>(definition))
    return llvm::equal(reshape.getOrder(),
                       reshape.getInput().getType().getAxisIds().asArrayRef()) &&
           canPlaceStateLocally(reshape.getInput());
  return false;
}

Roles rolesFor(mlir::Value value) {
  Roles roles;
  // An uninitialized state denotes addressable storage, including rank-zero
  // state.  It cannot acquire a numerical register value before its writes.
  if (value.getDefiningOp<riscv::EmptyOp>()) {
    roles.local = true;
    return roles;
  }
  if (auto unary = value.getDefiningOp<riscv::UnaryOp>()) {
    auto selected = unary->getAttrOfType<riscv::ImplementationAttr>("implementation");
    if (selected && selected.getEngine() == "scalar") {
      roles.anchored = true;
      roles.registerTuple = true;
      addSmallReplicas(value, roles);
      return roles;
    }
  }
  auto element = riscv_internal::logicalElement(value.getType());
  if (mlir::isa<kernel::EncodingType>(element))
    return roles;
  if (auto update = value.getDefiningOp<riscv::UpdateOp>(); update && !update.coversWholeState()) {
    roles.local = true;
    return roles;
  }
  if (auto converted = fragmentConversionRoles(value)) return *converted;

  // A storage-facing carrier is authoritative when this exact shaped value
  // supplies either a widening contraction or an indexed lookup.  The latter
  // may add broadcast axes while preserving every source axis: those existing
  // axes still have to arrive in lanes so the lookup index is not rebuilt from
  // a scalar tuple at every issue.  This remains narrower than treating every
  // affine scale/index value as a contraction anchor.
  if (auto storage =
          storageRolesFor(value, !feedsExpandedIndexedLookup(value)))
    return *storage;

  if (value.getDefiningOp<riscv::FieldOp>()) {
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *owner = use.getOwner();
      auto implementation = owner->getAttrOfType<riscv::ImplementationAttr>(
          "implementation");
      if (!implementation || implementation.getFamily() != "widen-dot")
        continue;
      if (auto contract = mlir::dyn_cast<riscv::DotOp>(owner))
        constrainReductionContractOperand(contract, value, roles);
      else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(owner))
        constrainReductionContractOperand(contract, value, roles);
      else if (auto contract = mlir::dyn_cast<riscv::OuterContractOp>(owner))
        constrainOuterContractOperand(contract, value, roles);
      if (roles.anchored)
        return roles;
    }
    const bool addressableField =
        !value.use_empty() && llvm::all_of(value.getUsers(), [](mlir::Operation *user) {
          return mlir::isa<riscv::ExtractOp, riscv::SliceOp,
                           riscv::Fold2Op>(user);
        });
    if (addressableField) {
      roles.local = true;
      return roles;
    }
  }

  if (auto state = value.getDefiningOp<riscv::NewOp>()) {
    if (state.getOwnerDomainId() == 0 &&
         llvm::any_of(riscv_internal::logicalShape(value.getType()),
                      [](int64_t extent) { return extent < 0; })) {
      roles.local = true;
      return roles;
    }
  }

  if (auto materialize = value.getDefiningOp<riscv::MaterializeOp>()) {
    bool groupedProjection = !value.use_empty();
    for (mlir::Operation *user : value.getUsers()) {
      auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
      bool hasGroupIndex = false;
      if (extract && extract.getInput() == value)
        hasGroupIndex = llvm::any_of(
            extract.getSelectors(), [](mlir::Attribute selector) {
              return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                     "group_index";
            });
      groupedProjection &= static_cast<bool>(extract) && hasGroupIndex;
    }
    bool dynamicState = llvm::any_of(riscv_internal::logicalShape(value.getType()),
                                    [](int64_t extent) { return extent < 0; });
    if (materialize.getPlacement() == "local" || groupedProjection || dynamicState) {
      roles.local = true;
      return roles;
    }
  }
  if (auto reshape = value.getDefiningOp<riscv::ReshapeOp>()) {
    if (!riscv::reshapeAxisPositions(reshape.getInput().getType(),
                                    reshape.getResult().getType(),
                                    reshape.getOrder())) {
      auto axes = reshape.getResult().getType().getAxisIds().asArrayRef();
      if (!axes.empty()) {
        roles.laneAxis = axes.back();
        roles.anchored = true;
        roles.fullLaneExtent = true;
      }
      return roles;
    }
  }
  if (value.getDefiningOp<riscv::ReduceOp>()) {
    // A reduction result preserves every free logical axis.  Small free axes
    // are register replicas; the eliminated axis is anchored on the producer,
    // not manufactured on the result.
    roles.anchored = true;
    addSmallReplicas(value, roles);
    return roles;
  }
  if (value.getDefiningOp<riscv::Fold2Op>()) {
    roles.anchored = true;
    roles.fullLaneExtent = true;
    auto axes = riscv_internal::logicalAxes(value.getType());
    if (!axes.empty())
      roles.laneAxis = axes.back();
    addSmallReplicas(value, roles, roles.laneAxis);
    return roles;
  }
  if (auto lookup = value.getDefiningOp<riscv::LookupOp>();
      lookup && constrainIndexedEntryLookup(lookup, value, roles))
    return roles;
  if (auto operation = value.getDefiningOp<riscv::MacGroupsOp>()) {
    constrainGroupedMac(operation, value, roles);
  } else if (auto operation = value.getDefiningOp<riscv::DotOp>()) {
    constrainReductionContractOperand(operation, value, roles);
  } else if (auto operation = value.getDefiningOp<riscv::ContractOp>()) {
    constrainReductionContractOperand(operation, value, roles);
  } else if (auto operation = value.getDefiningOp<riscv::OuterContractOp>()) {
    constrainOuterContractOperand(operation, value, roles);
  }

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(owner)) {
      auto axes = riscv_internal::logicalAxes(value.getType());
      if (reduce.getAxis() >= 0 &&
          reduce.getAxis() < static_cast<int64_t>(axes.size())) {
        // The eliminated axis is only the fallback lane anchor.  A producer or
        // loop-carried value may already have a surviving lane axis, in which
        // case this is a register-axis reduction.  Defer that choice until the
        // strong producer/control constraints have propagated.
        roles.anchored = true;
      }
    } else if (auto mac = mlir::dyn_cast<riscv::MacGroupsOp>(owner)) {
      constrainGroupedMac(mac, value, roles);
    } else if (auto contract = mlir::dyn_cast<riscv::DotOp>(owner)) {
      constrainReductionContractOperand(contract, value, roles);
    } else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(owner)) {
      constrainReductionContractOperand(contract, value, roles);
    } else if (auto contract = mlir::dyn_cast<riscv::OuterContractOp>(owner)) {
      constrainOuterContractOperand(contract, value, roles);
    } else if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(owner)) {
      if (!constrainIndexedEntryLookup(lookup, value, roles)) {
        auto axes = riscv_internal::logicalAxes(lookup.getIndices().getType());
        if (!axes.empty() && containsAxis(value.getType(), axes.back()))
          roles.laneAxis = axes.back();
        addSmallReplicas(value, roles, roles.laneAxis);
      }
    } else if (auto reshape = mlir::dyn_cast<riscv::ReshapeOp>(owner)) {
      if (reshape.getInput() != value || reshape.getOrder().empty())
        continue;
      if (riscv::reshapeAxisPositions(reshape.getInput().getType(),
                                     reshape.getResult().getType(),
                                     reshape.getOrder()))
        continue;
      roles.laneAxis = reshape.getOrder().back();
      roles.coalescedLaneAxes.clear();
      roles.replicaAxes.clear();
      roles.sequentialAxes.clear();
      for (int64_t axis : reshape.getOrder().drop_back())
        roles.coalescedLaneAxes.insert(axis);
      roles.anchored = true;
      roles.fullLaneExtent = true;
    }
  }

  return roles;
}

bool mergeRoles(const Roles &source, mlir::Value target, Roles &destination,
                bool preserveCarrier = false) {
  bool changed = false;
  const bool destinationWasAnchored = destination.anchored;
  auto axes = riscv_internal::logicalAxes(target.getType());
  if (preserveCarrier && source.local && !destination.local) {
    destination.local = true;
    changed = true;
  }
  if (source.registerTuple && !destination.registerTuple &&
      !(destination.anchored && destination.laneAxis)) {
    destination.registerTuple = true;
    destination.laneAxis = 0;
    addSmallReplicas(target, destination);
    changed = true;
  }
  if (source.laneAxis && llvm::is_contained(axes, source.laneAxis) &&
      destination.laneAxis == 0 && !destination.registerTuple) {
    destination.laneAxis = source.laneAxis;
    destination.replicaAxes.erase(source.laneAxis);
    changed = true;
  }
  for (int64_t axis : source.replicaAxes)
    if (axis != destination.laneAxis && llvm::is_contained(axes, axis) &&
        !destination.coalescedLaneAxes.contains(axis) &&
        !destination.sequentialAxes.contains(axis) &&
        destination.replicaAxes.insert(axis).second)
      changed = true;
  for (int64_t axis : source.coalescedLaneAxes)
    if (axis != destination.laneAxis && llvm::is_contained(axes, axis) &&
        !destination.memoryAnchored &&
        !destination.sequentialAxes.contains(axis) &&
        destination.coalescedLaneAxes.insert(axis).second) {
      destination.replicaAxes.erase(axis);
      changed = true;
    }
  for (int64_t axis : source.sequentialAxes)
    if (llvm::is_contained(axes, axis) &&
        destination.sequentialAxes.insert(axis).second) {
      destination.replicaAxes.erase(axis);
      changed = true;
    }
  if (source.anchored && !destination.anchored) {
    destination.anchored = true;
    changed = true;
  }
  if (source.fullLaneExtent && !destination.fullLaneExtent) {
    destination.fullLaneExtent = true;
    changed = true;
  }
  if (source.memoryAnchored && !destinationWasAnchored &&
      !destination.memoryAnchored) {
    destination.memoryAnchored = true;
    changed = true;
  }
  return changed;
}

Roles projectRoles(const Roles &roles, mlir::Value from, mlir::Value to) {
  Roles projected = roles;
  auto source = mlir::cast<riscv::ValueType>(from.getType());
  auto target = mlir::cast<riscv::ValueType>(to.getType());
  auto retained = [&](int64_t axis) {
    auto input = llvm::find(source.getAxisIds().asArrayRef(), axis);
    auto output = llvm::find(target.getAxisIds().asArrayRef(), axis);
    if (input == source.getAxisIds().asArrayRef().end() || output == target.getAxisIds().asArrayRef().end()) return false;
    int64_t before = source.getShape()[input - source.getAxisIds().asArrayRef().begin()];
    int64_t after = target.getShape()[output - target.getAxisIds().asArrayRef().begin()];
    return before == after;
  };
  // A scalar tuple constrains only the coordinates actually represented by
  // that value. A singleton channel coefficient must not force the added
  // state/reduction dimension of its consumer into scalar replicas.
  bool complete = llvm::all_of(target.getAxisIds().asArrayRef(), retained);
  bool lane = projected.laneAxis && retained(projected.laneAxis);
  if (!lane) projected.laneAxis = 0;
  auto filter = [&](auto &axes) {
    auto before = axes;
    for (int64_t axis : before) if (!retained(axis)) axes.erase(axis);
  };
  filter(projected.replicaAxes);
  filter(projected.coalescedLaneAxes);
  filter(projected.sequentialAxes);
  if (!complete) {
    projected.registerTuple = false;
    // Anchored-without-lanes denotes the source's scalar coordinate domain,
    // not a prohibition on vectorizing newly introduced logical coordinates.
    if (!lane && projected.coalescedLaneAxes.empty()) projected.anchored = false;
  }
  if (!lane && projected.coalescedLaneAxes.empty()) {
    projected.memoryAnchored = false;
    projected.fullLaneExtent = false;
  }
  return projected;
}

void completeRoles(mlir::Value value, Roles &roles) {
  if (roles.local)
    return;
  if (roles.registerTuple) {
    roles.laneAxis = 0;
    roles.coalescedLaneAxes.clear();
    addSmallReplicas(value, roles);
    return;
  }
  auto axes = riscv_internal::logicalAxes(value.getType());
  // A coalesced lane coordinate cannot exist without a primary lane carrier.
  // Pointwise propagation can legitimately retain only a surviving coalesced
  // axis after its original primary axis was projected away.  Promote the
  // innermost such axis so the resulting type never claims a scalar carrier
  // while still containing non-unit lane factors.
  if (!roles.laneAxis && !roles.coalescedLaneAxes.empty())
    for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis)
      if (roles.coalescedLaneAxes.erase(*axis)) {
        roles.laneAxis = *axis;
        roles.anchored = true;
        break;
      }
  if (!roles.anchored && roles.laneAxis == 0 && !axes.empty()) {
    // A shaped Value already carries an explicit logical domain.  Unlike an
    // ordinary scalar for/while iteration, that domain is legal input to the
    // target mapping.  The target's base structural rule maps the innermost
    // logical axis to RVV; memory and consumer anchors above may replace this
    // with a more constrained axis.  Root dynamic state is handled earlier and
    // deliberately remains local (Top-K is the canonical example).
    roles.laneAxis = axes.back();
    roles.anchored = true;
  }
  addSmallReplicas(value, roles, roles.laneAxis);
  if (roles.laneAxis)
    roles.replicaAxes.erase(roles.laneAxis);
  for (int64_t axis : roles.coalescedLaneAxes)
    roles.replicaAxes.erase(axis);
}

int64_t roundLegalLMUL(riscv::TargetAttr target, int64_t requested,
                       int64_t sew) {
  int64_t elen = 0;
  for (int64_t supported : target.getSupportedSEW().asArrayRef())
    elen = std::max(elen, supported);
  if (elen <= 0)
    return 0;
  requested = std::max(requested, (8 * sew + elen - 1) / elen);
  for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
    if (legal >= requested)
      return legal;
  return 0;
}

riscv::LayoutAttr buildLayout(mlir::Builder &builder, mlir::Value value,
                              Roles roles, int64_t connectedLaneSpan,
                              int64_t lmulEighths) {
  auto type = mlir::cast<riscv::ValueType>(value.getType());
  auto axes = type.getAxisIds().asArrayRef();
  auto element = type.getElementType();
  auto kernel = value.getParentRegion()->getParentOfType<riscv::KernelOp>();
  riscv::TargetAttr target = kernel.getTarget();
  const int64_t sew = std::max<int64_t>(8, riscv_internal::logicalBitWidth(type));

  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> replica(axes.size(), 1);
  llvm::SmallVector<int64_t> fragment(axes.size(), 1);
  llvm::SmallVector<int64_t> local(axes.size(), 1);

  if (mlir::isa<kernel::EncodingType>(element)) {
    for (int64_t axis : axes)
      time.push_back(std::max<int64_t>(1, riscv_internal::physicalExtent(value, axis)));
    return riscv::LayoutAttr::get(
        builder.getContext(), "local", type.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane),
        riscv_internal::integers(builder, replica),
        riscv_internal::integers(builder, fragment),
        riscv_internal::integers(builder, local), 8, 0, 1, 0, "full");
  }

  if (roles.local) {
    for (size_t index = 0; index < axes.size(); ++index) {
      int64_t extent =
          std::max<int64_t>(1, riscv_internal::physicalExtent(value, axes[index]));
      time.push_back(1);
      local[index] = extent;
    }
    return riscv::LayoutAttr::get(
        builder.getContext(), "local", type.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane),
        riscv_internal::integers(builder, replica),
        riscv_internal::integers(builder, fragment),
        riscv_internal::integers(builder, local), sew, 0, 1, 0, "full");
  }

  int64_t desiredLanes = 1;
  const int64_t laneCapacity =
      std::max<int64_t>(1, target.getVlenBits() * lmulEighths / (8 * sew));
  if (roles.laneAxis) {
    int64_t logicalExtent = riscv_internal::physicalExtent(value, roles.laneAxis);
    int64_t representationExtent =
        logicalExtent > 0 ? logicalExtent
                          : std::max<int64_t>(1, connectedLaneSpan);
    // The instantiated LMUL binding is a maximum lane capacity, not a new
    // logical axis.  Smaller values still use the smallest legal LMUL that
    // covers their extent; larger values become explicit issue-time strips.
    desiredLanes = std::min(
        representationExtent,
        std::max<int64_t>(1, std::min(laneCapacity, connectedLaneSpan)));
    if (logicalExtent > 0 && logicalExtent % desiredLanes != 0) {
      int64_t divisor = 1;
      while (divisor <= desiredLanes / 2)
        divisor *= 2;
      while (divisor > 1 && logicalExtent % divisor != 0)
        divisor /= 2;
      desiredLanes = divisor;
    }
  }
  int64_t totalDesiredLanes = desiredLanes;
  for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
       position >= 0; --position) {
    const size_t index = static_cast<size_t>(position);
    const int64_t axis = axes[index];
    if (axis == roles.laneAxis || !roles.coalescedLaneAxes.contains(axis))
      continue;
    const int64_t extent =
        std::max<int64_t>(1, riscv_internal::physicalExtent(value, axis));
    int64_t available = std::max<int64_t>(1, laneCapacity / totalDesiredLanes);
    int64_t factor = std::min(extent, available);
    while (factor > 1 && extent % factor)
      --factor;
    lane[index] = std::max<int64_t>(1, factor);
    totalDesiredLanes *= lane[index];
  }
  int64_t requestedLMUL =
      roles.laneAxis
          ? std::max<int64_t>(1, (totalDesiredLanes * sew * 8 +
                                  target.getVlenBits() - 1) /
                                     target.getVlenBits())
          : 0;
  int64_t lmul =
      roles.laneAxis ? roundLegalLMUL(target, requestedLMUL, sew) : 0;
  if (auto lookup = value.getDefiningOp<riscv::LookupOp>()) {
    auto relation = riscv_internal::analyzeIndexedEntryRelation(
        lookup.getIndices(), type, {}, {});
    if (relation && roles.laneAxis == relation->payloadAxis &&
        desiredLanes == relation->payloadExtent &&
        totalDesiredLanes > relation->payloadExtent &&
        supportsPackedIndexedEntryLookup(lookup, *relation, type)) {
      // Packed indexed memory reinterprets this same register as a complete
      // storage word. Its LMUL must be legal at that word's SEW as well as at
      // the payload SEW, even when only part of a wide VLEN is active.
      const int64_t packedSEW =
          riscv_internal::logicalBitWidth(type) * relation->payloadExtent;
      lmul = roundLegalLMUL(target, requestedLMUL, packedSEW);
    }
  }
  if (roles.laneAxis && lmul == 0)
    return {};
  int64_t actualLanes = roles.laneAxis ? totalDesiredLanes : 1;

  int64_t replicas = 1;
  for (size_t index = 0; index < axes.size(); ++index) {
    int64_t logicalExtent = riscv_internal::physicalExtent(value, axes[index]);
    int64_t extent = logicalExtent > 0 ? logicalExtent
                     : axes[index] == roles.laneAxis
                         ? std::max<int64_t>(1, connectedLaneSpan)
                         : 1;
    if (axes[index] == roles.laneAxis)
      lane[index] = std::min(extent, desiredLanes);
    if (roles.replicaAxes.contains(axes[index])) {
      replica[index] = extent;
      replicas *= extent;
    }
    int64_t mapped = lane[index] * replica[index];
    time.push_back((extent + mapped - 1) / mapped);
  }
  // A logical axis of extent one does not constitute an RVV representation.
  // Keep the axis identity in the type, but use the scalar carrier so later
  // memory and operation passes do not invent a one-lane vector operation.
  const bool usesRVV = roles.laneAxis && actualLanes > 1;
  llvm::StringRef carrier = usesRVV ? "rvv" : "scalar";
  if (!usesRVV)
    lmul = 0;
  int64_t groups = usesRVV ? ((lmul + 7) / 8) * replicas : 0;
  llvm::StringRef validity = "full";
  if (roles.laneAxis && riscv_internal::physicalExtent(value, roles.laneAxis) <= 0)
    validity = "tail";
  for (mlir::Operation *parent = value.getParentRegion()->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto loop = mlir::dyn_cast<riscv::LoopOp>(parent))
      if (loop.getDomain().getType().getTail() == "tail" &&
          llvm::is_contained(axes,
                             loop.getDomain().getType().getAxisId()))
        validity = "tail";
  return riscv::LayoutAttr::get(
      builder.getContext(), carrier, type.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local), sew, lmul,
      usesRVV ? actualLanes : 1, groups, validity);
}

bool pointwiseLike(mlir::Operation *operation) {
  return mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp, riscv::SelectOp,
                   riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp, riscv::WidenOp,
                   riscv::UpdateOp, riscv::MaterializeOp>(operation);
}

riscv::LayoutAttr reshapeCarrierLayout(mlir::Builder &builder,
                                       riscv::ReshapeOp reshape) {
  auto input = reshape.getInput().getType();
  auto result = reshape.getResult().getType();
  auto layout = input.getLayout();
  if (!llvm::equal(reshape.getOrder(), input.getAxisIds().asArrayRef())) {
    auto positions = riscv::reshapeAxisPositions(input, result, reshape.getOrder());
    if (!positions) return {};
    auto map = [&](mlir::DenseI64ArrayAttr factors) {
      llvm::SmallVector<int64_t> mapped;
      for (unsigned position : *positions) mapped.push_back(factors[position]);
      return riscv_internal::integers(builder, mapped);
    };
    auto permuted = riscv::LayoutAttr::get(
        builder.getContext(), layout.getCarrier(), result.getAxisIds(),
        map(layout.getTimeFactors()), map(layout.getLaneFactors()),
        map(layout.getReplicaFactors()), map(layout.getFragmentFactors()),
        map(layout.getLocalFactors()), layout.getSew(), layout.getLmulEighths(),
        layout.getVl(), layout.getRegisterGroups(), layout.getValidity(),
        layout.getReplicaGroup());
    auto mappedType = riscv::ValueType::get(builder.getContext(), result.getElementType(),
        result.getShape(), result.getAxisIds(), permuted);
    return riscv::hasIdenticalReshapeCarrier(input, mappedType, reshape.getOrder())
               ? permuted : riscv::LayoutAttr();
  }
  if (input.getShape() == result.getShape())
    return riscv::LayoutAttr::get(builder.getContext(), layout.getCarrier(), result.getAxisIds(),
        layout.getTimeFactors(), layout.getLaneFactors(), layout.getReplicaFactors(),
        layout.getFragmentFactors(), layout.getLocalFactors(), layout.getSew(),
        layout.getLmulEighths(), layout.getVl(), layout.getRegisterGroups(),
        layout.getValidity(), layout.getReplicaGroup());
  // Inserting/removing unit axes may also rename static non-unit axes.  Keep
  // their ordered physical factors; dynamic dimensions still require the
  // original axis identity, as in the canonical unit-projection contract.
  llvm::SmallVector<int64_t> positions(result.getShape().size(), -1);
  llvm::SmallVector<int64_t> projectedAxes(result.getAxisIds().asArrayRef());
  size_t sourcePosition = 0;
  bool unitProjection = true;
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    if (result.getShape()[position] == 1) continue;
    while (sourcePosition < input.getShape().size() &&
           input.getShape()[sourcePosition] == 1)
      ++sourcePosition;
    if (sourcePosition == input.getShape().size() ||
        input.getShape()[sourcePosition] != result.getShape()[position] ||
        (result.getShape()[position] < 0 &&
         input.getAxisIds()[sourcePosition] != result.getAxisIds()[position])) {
      unitProjection = false;
      break;
    }
    positions[position] = sourcePosition;
    projectedAxes[position] = input.getAxisIds()[sourcePosition++];
  }
  unitProjection &= kernel::isUnitAxisProjection(
      input.getShape(), input.getAxisIds(), result.getShape(), projectedAxes,
      reshape.getOrder());
  for (size_t position = 0; unitProjection && position < input.getShape().size();
       ++position)
    if (input.getShape()[position] == 1)
      for (auto factors : {layout.getTimeFactors(), layout.getLaneFactors(),
                           layout.getReplicaFactors(), layout.getFragmentFactors(),
                           layout.getLocalFactors()})
        unitProjection &= factors[position] == 1;
  if (unitProjection) {
    auto project = [&](mlir::DenseI64ArrayAttr factors) {
      llvm::SmallVector<int64_t> mapped;
      for (int64_t position : positions)
        mapped.push_back(position < 0 ? 1 : factors[position]);
      return riscv_internal::integers(builder, mapped);
    };
    return riscv::LayoutAttr::get(builder.getContext(), layout.getCarrier(), result.getAxisIds(),
        project(layout.getTimeFactors()), project(layout.getLaneFactors()), project(layout.getReplicaFactors()),
        project(layout.getFragmentFactors()), project(layout.getLocalFactors()), layout.getSew(),
        layout.getLmulEighths(), layout.getVl(), layout.getRegisterGroups(),
        layout.getValidity(), layout.getReplicaGroup());
  }
  if (layout.getCarrier() == "local") {
    auto inputCount = riscv_internal::staticProduct(input.getShape().asArrayRef());
    auto resultCount = riscv_internal::staticProduct(result.getShape().asArrayRef());
    if (!inputCount || !resultCount || *inputCount <= 0 ||
        *inputCount != *resultCount ||
        layout.getLocalFactors() != input.getShape())
      return {};
    for (auto factors : {layout.getTimeFactors(), layout.getLaneFactors(),
                         layout.getReplicaFactors(), layout.getFragmentFactors()})
      if (!llvm::all_of(factors.asArrayRef(), [](int64_t factor) { return factor == 1; }))
        return {};
    auto ones = riscv_internal::integers(
        builder, llvm::SmallVector<int64_t>(result.getShape().size(), 1));
    return riscv::LayoutAttr::get(
        builder.getContext(), "local", result.getAxisIds(), ones, ones, ones,
        ones, result.getShape(), layout.getSew(), layout.getLmulEighths(),
        layout.getVl(), layout.getRegisterGroups(), layout.getValidity(),
        layout.getReplicaGroup());
  }
  if (layout.getCarrier() != "rvv") return {};
  size_t prefix = 0;
  while (prefix < input.getShape().size() &&
         prefix < result.getShape().size() &&
         input.getAxisIds()[prefix] == result.getAxisIds()[prefix] &&
         input.getShape()[prefix] == result.getShape()[prefix])
    ++prefix;
  if (prefix == input.getShape().size() ||
      prefix == result.getShape().size())
    return {};
  int64_t inputLanes = 1;
  for (size_t position = prefix; position < input.getShape().size(); ++position) {
    const int64_t extent = input.getShape()[position];
    if (extent <= 0 || layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != extent ||
        layout.getReplicaFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1 ||
        !checkedMultiply(inputLanes, extent, inputLanes))
      return {};
  }
  int64_t resultLanes = 1;
  for (int64_t extent : result.getShape().asArrayRef().drop_front(prefix))
    if (extent <= 0 || !checkedMultiply(resultLanes, extent, resultLanes))
      return {};
  if (inputLanes != resultLanes)
    return {};
  auto reshapeFactors = [&](mlir::DenseI64ArrayAttr factors, bool lane) {
    llvm::SmallVector<int64_t> mapped(factors.asArrayRef().take_front(prefix));
    for (int64_t extent : result.getShape().asArrayRef().drop_front(prefix))
      mapped.push_back(lane ? extent : 1);
    return riscv_internal::integers(builder, mapped);
  };
  return riscv::LayoutAttr::get(
      builder.getContext(), layout.getCarrier(), result.getAxisIds(),
      reshapeFactors(layout.getTimeFactors(), false),
      reshapeFactors(layout.getLaneFactors(), true),
      reshapeFactors(layout.getReplicaFactors(), false),
      reshapeFactors(layout.getFragmentFactors(), false),
      reshapeFactors(layout.getLocalFactors(), false), layout.getSew(),
      layout.getLmulEighths(), layout.getVl(), layout.getRegisterGroups(),
      layout.getValidity(), layout.getReplicaGroup());
}

std::optional<int64_t> encodedLaneLimit(mlir::Value value, int64_t laneAxis) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  std::function<std::optional<int64_t>(mlir::Value)> visit =
      [&](mlir::Value current) -> std::optional<int64_t> {
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return std::nullopt;
    if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
      riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
      auto type = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
      if (!type || facts.logicalRank <= 0 ||
          facts.logicalRank > static_cast<int64_t>(type.getAxisIds().size()))
        return std::nullopt;
      auto logicalFieldAxes = type.getAxisIds().asArrayRef().take_back(
          static_cast<size_t>(facts.logicalRank));
      if (facts.mapping == "grouped_layered" && facts.layer > 0 &&
          llvm::is_contained(logicalFieldAxes, laneAxis))
        return facts.layer;
      return std::nullopt;
    }
    if (auto conversion =
            mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
      auto input =
          mlir::dyn_cast<riscv::ValueType>(conversion.getInput().getType());
      auto result =
          mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      llvm::StringRef effect = conversion.getConversion().getEffect();
      if (!input || !result || input.getShape() != result.getShape() ||
          input.getAxisIds() != result.getAxisIds() ||
          (effect != "pure" && effect != "read"))
        return std::nullopt;
      return visit(conversion.getInput());
    }
    if (auto materialize = mlir::dyn_cast<riscv::MaterializeOp>(definition))
      return visit(materialize.getInput());
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition))
      return visit(extract.getInput());
    // Lookup computes a new value; its indices' storage layer does not bound
    // that value's carrier. Operand mismatches remain explicit conversions.
    if (mlir::isa<riscv::LookupOp>(definition))
      return std::nullopt;
    // Storage-layer width constrains the raw field and representation-only
    // casts, not a newly computed logical value.  In particular, combining
    // low/high bit layers creates a full logical vector that may be
    // re-partitioned through an explicit convert_layout.
    if (!mlir::isa<riscv::UnaryOp, riscv::CastOp, riscv::NarrowOp,
                   riscv::WidenOp>(definition))
      return std::nullopt;
    std::optional<int64_t> limit;
    for (mlir::Value operand : definition->getOperands())
      if (auto operandLimit = visit(operand))
        limit = limit ? std::min(*limit, *operandLimit) : operandLimit;
    return limit;
  };
  return visit(value);
}

void setValueLayout(mlir::Value value, riscv::LayoutAttr layout) {
  if (mlir::isa<riscv::ValueType>(value.getType()))
    value.setType(riscv_internal::withLayout(value.getType(), layout));
}

void materializeScalarIndexedSources(mlir::ModuleOp module,
                                     mlir::OpBuilder &builder) {
  llvm::SmallVector<riscv::ExtractOp> reads;
  module.walk([&](riscv::ExtractOp extract) {
    auto input = extract.getInput().getType();
    auto layout = input.getLayout();
    if (mlir::isa<riscv::ValueType>(extract.getResult().getType()) ||
        !mlir::isa<mlir::IntegerType, mlir::FloatType>(input.getElementType()) ||
        (layout.getCarrier() != "rvv" && layout.getCarrier() != "scalar") ||
        layout.getValidity() != "full" ||
        riscv_internal::sourceField(extract.getInput()) ||
        extract.getSelectors().size() != input.getShape().size() ||
        extract.getIndices().size() != input.getShape().size() ||
        llvm::any_of(extract.getSelectors(), [](mlir::Attribute selector) {
          return mlir::cast<mlir::StringAttr>(selector).getValue() != "index";
        }))
      return;
    auto elements = riscv_internal::staticProduct(input.getShape().asArrayRef());
    if (!elements || *elements == 1) return;
    for (size_t axis = 0; axis < input.getShape().size(); ++axis) {
      auto extent = riscv_internal::staticProduct(
          {layout.getTimeFactors()[axis], layout.getLaneFactors()[axis],
           layout.getReplicaFactors()[axis]});
      if (!extent || *extent != input.getShape()[axis] ||
          layout.getFragmentFactors()[axis] != 1 ||
          layout.getLocalFactors()[axis] != 1 ||
          (layout.getCarrier() == "scalar" && layout.getTimeFactors()[axis] != 1))
        return;
    }
    reads.push_back(extract);
  });
  llvm::DenseMap<mlir::Value, mlir::Value> snapshots;
  for (auto extract : reads) {
    mlir::Value source = extract.getInput();
    if (auto found = snapshots.find(source); found != snapshots.end()) {
      extract.getInputMutable().assign(found->second);
      continue;
    }
    auto input = extract.getInput().getType();
    auto units = builder.getDenseI64ArrayAttr(
        llvm::SmallVector<int64_t>(input.getShape().size(), 1));
    auto layout = riscv::LayoutAttr::get(builder.getContext(), "local",
        input.getAxisIds(), units, units, units, units, input.getShape(),
        input.getLayout().getSew(), 0, 1, 0, "full");
    auto type = riscv_internal::withLayout(input, layout);
    mlir::OpBuilder::InsertionGuard guard(builder);
    // Capture the completed numerical value once, without moving or replaying
    // its memory producer. Only scalar projections use the addressable copy.
    if (auto definition = source.getDefiningOp())
      builder.setInsertionPointAfter(definition);
    else
      builder.setInsertionPointToStart(mlir::cast<mlir::BlockArgument>(source).getOwner());
    auto snapshot = builder.create<riscv::ConvertLayoutOp>(extract.getLoc(), type,
        source, riscv_internal::layoutConversion(builder, input.getLayout(), layout),
        riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
    extract.getInputMutable().assign(snapshot.getResult());
    if (!riscv::storageAliasesRemainInScope(
            snapshot.getResult(), snapshot->getBlock())) {
      extract.getInputMutable().assign(source);
      snapshot.erase();
      continue;
    }
    riscv_internal::copyOrigin(extract, snapshot);
    snapshots.try_emplace(source, snapshot.getResult());
  }
}

riscv::LayoutAttr withRegisterWidth(mlir::Builder &builder,
                                    riscv::LayoutAttr layout,
                                    int64_t lmulEighths) {
  int64_t replicas = 1;
  for (int64_t factor : layout.getReplicaFactors().asArrayRef())
    replicas *= factor;
  return riscv::LayoutAttr::get(
      builder.getContext(), layout.getCarrier(), layout.getAxisIds(),
      layout.getTimeFactors(), layout.getLaneFactors(),
      layout.getReplicaFactors(), layout.getFragmentFactors(),
      layout.getLocalFactors(), layout.getSew(), lmulEighths, layout.getVl(),
      ((lmulEighths + 7) / 8) * replicas, layout.getValidity());
}

riscv::LayoutAttr registerGatherSourceLayout(mlir::Builder &builder,
                                             riscv::ValueType source,
                                             riscv::LayoutAttr resultLayout,
                                             riscv::TargetAttr target) {
  auto shape = source.getShape().asArrayRef();
  if (shape.size() != 1 || shape.front() <= 0 ||
      resultLayout.getCarrier() != "rvv" ||
      resultLayout.getSew() !=
          std::max<int64_t>(8, riscv_internal::logicalBitWidth(source)))
    return {};
  const int64_t capacity = target.getVlenBits() *
                           resultLayout.getLmulEighths() /
                           (8 * resultLayout.getSew());
  if (shape.front() > capacity)
    return {};
  llvm::SmallVector<int64_t> one{1};
  llvm::SmallVector<int64_t> lane{shape.front()};
  return riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), resultLayout.getSew(),
      resultLayout.getLmulEighths(), shape.front(),
      (resultLayout.getLmulEighths() + 7) / 8, "full");
}

class PropagateRISCVLayoutsPass
    : public mlir::PassWrapper<PropagateRISCVLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit PropagateRISCVLayoutsPass(int64_t value = 8) { lmulEighths = value; }
  PropagateRISCVLayoutsPass(const PropagateRISCVLayoutsPass &other)
      : PassWrapper(other) { copyOptionValuesFrom(&other); }
  llvm::StringRef getArgument() const override {
    return "weft-riscv-propagate-layouts";
  }
  llvm::StringRef getDescription() const override {
    return "Propagate logical-axis preserving physical layouts and insert conversions";
  }

  void runOnOperation() override {
    bool invalidBinding = lmulEighths < 1 || lmulEighths > 64 ||
                          (lmulEighths & (lmulEighths - 1));
    getOperation().walk([&](riscv::KernelOp kernel) {
      invalidBinding |= !llvm::is_contained(
          kernel.getTarget().getLegalLMULEighths().asArrayRef(),
          lmulEighths.getValue());
    });
    if (invalidBinding) {
      getOperation().emitError("layout propagation requires a legal target LMUL binding");
      signalPassFailure();
      return;
    }
    mlir::OpBuilder builder(&getContext());
    bool failed = false;
    llvm::SmallVector<mlir::Value> values;
    getOperation().walk([&](mlir::Operation *operation) {
      for (mlir::Value result : operation->getResults())
        if (mlir::isa<riscv::ValueType>(result.getType()))
          values.push_back(result);
      for (mlir::Region &region : operation->getRegions())
        for (mlir::Block &block : region)
          for (mlir::BlockArgument argument : block.getArguments())
            if (mlir::isa<riscv::ValueType>(argument.getType()))
              values.push_back(argument);
    });

    llvm::DenseMap<mlir::Value, Roles> roles;
    for (mlir::Value value : values)
      roles.try_emplace(value, rolesFor(value));

    // Pointwise values and loop-carried state share one logical distribution.
    // Anchors come from selected target operations; propagation never creates a
    // new logical axis and never crosses a local-storage boundary.
    auto propagateRoles = [&]() {
      bool changed = true;
      while (changed) {
        changed = false;
      getOperation().walk([&](riscv::ReshapeOp reshape) {
        auto input = reshape.getInput(), result = reshape.getResult();
        if (llvm::equal(reshape.getOrder(),
                        input.getType().getAxisIds().asArrayRef())) {
          Roles local;
          local.local = true;
          if (roles[input].local)
            changed |= mergeRoles(local, result, roles[result], true);
          if (roles[result].local && canPlaceStateLocally(input))
            changed |= mergeRoles(local, input, roles[input], true);
        }
        auto positions = riscv::reshapeAxisPositions(input.getType(), result.getType(),
                                                     reshape.getOrder());
        if (!positions) return;
        auto rename = [&](mlir::Value from, mlir::Value to, bool forward) {
          Roles mapped = roles[from];
          auto fromAxes = riscv_internal::logicalAxes(from.getType());
          auto toAxes = riscv_internal::logicalAxes(to.getType());
          auto axis = [&](int64_t id) {
            if (!id) return int64_t(0);
            auto position = llvm::find(fromAxes, id);
            if (position == fromAxes.end()) return int64_t(0);
            unsigned slot = position - fromAxes.begin();
            if (forward) {
              auto mapped = llvm::find(*positions, slot);
              return mapped == positions->end() ? int64_t(0)
                                                 : toAxes[mapped - positions->begin()];
            }
            return toAxes[(*positions)[slot]];
          };
          mapped.laneAxis = axis(mapped.laneAxis);
          auto remap = [&](auto &set) {
            auto original = set;
            set.clear();
            for (int64_t id : original) if (int64_t target = axis(id)) set.insert(target);
          };
          remap(mapped.replicaAxes); remap(mapped.coalescedLaneAxes); remap(mapped.sequentialAxes);
          return mergeRoles(mapped, to, roles[to], true);
        };
        changed |= rename(input, result, true);
        changed |= rename(result, input, false);
      });
      getOperation().walk([&](mlir::Operation *operation) {
        if (!pointwiseLike(operation) || operation->getNumResults() != 1)
          return;
        mlir::Value result = operation->getResult(0);
        if (!mlir::isa<riscv::ValueType>(result.getType()))
          return;
        for (mlir::Value operand : operation->getOperands()) {
          if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation);
              update && operand != update.getInput() && !update.coversWholeState())
            continue;
          if (!mlir::isa<riscv::ValueType>(operand.getType()))
            continue;
          if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation);
              update && operand == update.getInput() &&
              (roles[result].local || roles[operand].local)) {
            Roles local;
            local.local = true;
            changed |= mergeRoles(local, result, roles[result], true);
            changed |= mergeRoles(local, operand, roles[operand], true);
            continue;
          }
          // A local-storage value used by ordinary pointwise code requires a
          // typed local_load conversion; it does not force the consumer's
          // entire value chain into local storage.  Conversely, a vector
          // consumer does not rewrite the source object's placement.
          if (roles[result].local || roles[operand].local)
            continue;
          changed |= mergeRoles(projectRoles(roles[operand], operand, result), result, roles[result]);
          changed |= mergeRoles(projectRoles(roles[result], result, operand), operand, roles[operand]);
        }
      });
      getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
        mlir::Value input = conversion.getInput();
        mlir::Value result = conversion.getResult();
        auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
        auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
        llvm::StringRef effect = conversion.getConversion().getEffect();
        if (!inputType || !resultType ||
            inputType.getElementType() != resultType.getElementType() ||
            inputType.getShape() != resultType.getShape() ||
            inputType.getAxisIds() != resultType.getAxisIds() ||
            (effect != "pure" && effect != "read"))
          return;
        changed |= mergeRoles(roles[input], result, roles[result]);
        changed |= mergeRoles(roles[result], input, roles[input]);
      });
      auto mergeControlSlot = [&](mlir::ValueRange incoming,
                                  mlir::ValueRange boundaries) {
        llvm::SmallVector<mlir::Value> related(incoming);
        llvm::append_range(related, boundaries);
        for (mlir::Value source : related)
          for (mlir::Value target : related) {
            auto sourceType = mlir::dyn_cast<riscv::ValueType>(source.getType());
            auto targetType = mlir::dyn_cast<riscv::ValueType>(target.getType());
            if (!sourceType || !targetType ||
                sourceType.getElementType() != targetType.getElementType() ||
                sourceType.getShape() != targetType.getShape() ||
                sourceType.getAxisIds() != targetType.getAxisIds())
              continue;
            // State births and their placement-preserving aliases can realize
            // a local carry directly. Numerical producers still compute in
            // their own carrier; a use conversion realizes that boundary
            // without rewriting the producer or conflating distinct slots.
            changed |= mergeRoles(roles[source], target, roles[target],
                                  llvm::is_contained(boundaries, target) ||
                                      canPlaceStateLocally(target));
          }
      };
      getOperation().walk([&](riscv::LoopOp loop) {
        auto yield = mlir::cast<riscv::YieldOp>(loop.getBody().front().getTerminator());
        for (auto [index, initial] : llvm::enumerate(loop.getCarried())) {
          mlir::Value argument = loop.getBody().front().getArgument(index + 1);
          mlir::Value next = yield.getValues()[index];
          mlir::Value result = loop.getResult(index);
          mergeControlSlot({initial, next}, {argument, result});
        }
      });
      getOperation().walk([&](mlir::scf::ForOp loop) {
        auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        for (auto [index, initial] : llvm::enumerate(loop.getInitArgs()))
          mergeControlSlot({initial, yield.getOperand(index)},
                           {loop.getRegionIterArg(index), loop.getResult(index)});
      });
      getOperation().walk([&](mlir::scf::IfOp branch) {
        for (auto [index, result] : llvm::enumerate(branch.getResults()))
          mergeControlSlot({branch.thenYield().getOperand(index),
                            branch.elseYield().getOperand(index)},
                           {result});
      });
      getOperation().walk([&](mlir::scf::WhileOp loop) {
        auto condition = mlir::cast<mlir::scf::ConditionOp>(
            loop.getBefore().front().getTerminator());
        auto yield = mlir::cast<mlir::scf::YieldOp>(
            loop.getAfter().front().getTerminator());
        for (auto [index, initial] : llvm::enumerate(loop.getInits()))
          mergeControlSlot({initial, yield.getOperand(index)},
                           {loop.getBefore().front().getArgument(index)});
        for (auto [index, result] : llvm::enumerate(loop.getResults()))
          mergeControlSlot({condition.getArgs()[index]},
                           {loop.getAfter().front().getArgument(index), result});
      });
      // Reduction is a projection, not a layout-equivalence edge.  Its result
      // is projected from the selected input representation after all values
      // receive layouts; consumer conflicts are represented by convert_layout.
      getOperation().walk([&](riscv::ExtractOp extract) {
        mlir::Value input = extract.getInput();
        mlir::Value result = extract.getResult();
        if (!mlir::isa<riscv::ValueType>(input.getType()) ||
            !mlir::isa<riscv::ValueType>(result.getType()) ||
            roles[input].local || roles[result].local)
          return;
        // Extract removes axes selected by a point/index and preserves every
        // remaining logical axis.  Representation roles on those surviving
        // axes therefore propagate in both directions; consumed axes are
        // filtered by mergeRoles rather than rediscovered from source shape.
        changed |= mergeRoles(roles[input], result, roles[result]);
        Roles projectedResult = roles[result];
        const bool shapedGather = llvm::any_of(
            extract.getSelectors(), [](mlir::Attribute selector) {
              return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                     "gather";
            });
        if (projectedResult.registerTuple && !shapedGather) {
          auto inputAxes = riscv_internal::logicalAxes(input.getType());
          auto resultAxes = riscv_internal::logicalAxes(result.getType());
          if (llvm::any_of(inputAxes, [&](int64_t axis) {
                return !llvm::is_contained(resultAxes, axis);
              })) {
            projectedResult.registerTuple = false;
            projectedResult.anchored = false;
          }
        }
        changed |= mergeRoles(projectedResult, input, roles[input]);
      });
      }
    };
    // First propagate operation and consumer anchors.  Only values that remain
    // unanchored then receive the target's base innermost-lane mapping.  That
    // default is itself a physical fact needed by downstream reductions and
    // control-flow handoffs, so run the same propagation to a second fixed
    // point instead of leaving it stranded on the defining value.
    propagateRoles();
    getOperation().walk([&](riscv::ReduceOp reduce) {
      mlir::Value input = reduce.getInput();
      auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
      auto axes = riscv_internal::logicalAxes(input.getType());
      if (!inputType ||
          reduce.getAxis() < 0 ||
          reduce.getAxis() >= static_cast<int64_t>(axes.size()))
        return;
      const int64_t eliminated = axes[reduce.getAxis()];
      Roles &inputRoles = roles[input];
      const int64_t eliminatedExtent = inputType.getShape()[reduce.getAxis()];
      const bool registerReductionKind = reduce.getKind() == "add" ||
          (mlir::isa<mlir::FloatType>(inputType.getElementType()) &&
           (reduce.getKind() == "max" || reduce.getKind() == "min"));
      // Preserve a consistent free-axis memory supply when the eliminated
      // coordinate already has a bounded register-reduction realization.
      if (registerReductionKind && !inputRoles.registerTuple &&
          !inputRoles.local && eliminatedExtent > 1 &&
          eliminatedExtent <= 16) {
        llvm::SmallVector<int64_t> freeAxes;
        for (int64_t axis : axes)
          if (axis != eliminated)
            freeAxes.push_back(axis);
        if (auto supply = reductionSupplyRoles(input, freeAxes, true)) {
          inputRoles.laneAxis = supply->laneAxis;
          inputRoles.coalescedLaneAxes = supply->coalescedLaneAxes;
          inputRoles.fullLaneExtent = supply->fullLaneExtent;
        }
      }
      // A structured-product result deliberately remains a scalar register
      // tuple over its surviving free axes.  A following reduction requires a
      // different lane representation of that same logical value; preserve the
      // producer mapping here and insert the typed use-edge conversion below.
      // Ordinary shaped producers still anchor the eliminated axis directly.
      if (!inputRoles.laneAxis && !inputRoles.registerTuple)
        inputRoles.laneAxis = eliminated;
      inputRoles.anchored = true;
      if (inputRoles.laneAxis) {
        inputRoles.replicaAxes.erase(inputRoles.laneAxis);
        addSmallReplicas(input, inputRoles, inputRoles.laneAxis);
        if (inputRoles.laneAxis == eliminated ||
            inputRoles.coalescedLaneAxes.contains(eliminated))
          inputRoles.replicaAxes.erase(eliminated);
      }
    });
    propagateRoles();
    for (mlir::Value value : values)
      completeRoles(value, roles[value]);

    // A widening contraction has one physical issue topology over its reduction
    // coordinates.  Operand-local storage propagation is only a proposal for
    // how to supply that common carrier: it cannot give the two operands
    // different entry/payload decompositions and leave the terminal operation
    // to reconcile them.  When both storage proposals agree, preserve them.  If
    // they disagree, use the canonical logical reduction order: the last
    // declared reduction axis is primary and the preceding reduction axes
    // coalesce outside it.  Memory planning may then realize either operand with
    // a direct load or an explicit representation conversion.  Unique output
    // axes remain owned by their respective operand.
    auto normalizeWidenDotReductionCarrier =
        [&](mlir::Operation *operation) -> bool {
      llvm::SmallVector<int64_t> reductionAxes;
      if (auto dot = mlir::dyn_cast<riscv::DotOp>(operation))
        reductionAxes.assign(dot.getOver().begin(), dot.getOver().end());
      else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(operation))
        reductionAxes.assign(contract.getOver().begin(), contract.getOver().end());
      if (reductionAxes.empty())
        return false;
      auto implementation = operation->getAttrOfType<
          riscv::ImplementationAttr>("implementation");
      if (!implementation || implementation.getFamily() != "widen-dot")
        return false;
      mlir::Value lhs = operation->getOperand(0);
      mlir::Value rhs = operation->getOperand(1);
      // A common carrier is required when both operands contribute an
      // independently surviving free-axis domain, as in a blocked outer
      // contraction.  A scalar-output dot has no such relation: forcing its
      // storage/index producers through this anchor can only create an
      // unrelated lane/time repartition before the lookup or decode leaf.
      if (!hasOperandLocalFreeAxis(lhs, rhs, reductionAxes) ||
          !hasOperandLocalFreeAxis(rhs, lhs, reductionAxes))
        return false;
      auto lhsSupply = reductionSupplyRoles(lhs, reductionAxes);
      auto rhsSupply = reductionSupplyRoles(rhs, reductionAxes);
      Roles commonReductionCarrier;
      const Roles *anchor = nullptr;
      if (lhsSupply && rhsSupply &&
          lhsSupply->laneAxis == rhsSupply->laneAxis &&
          lhsSupply->coalescedLaneAxes == rhsSupply->coalescedLaneAxes)
        anchor = &*lhsSupply;
      else if (lhsSupply && !rhsSupply)
        anchor = &*lhsSupply;
      else if (!lhsSupply && rhsSupply)
        anchor = &*rhsSupply;
      if (!anchor) {
        auto lhsType = mlir::dyn_cast<riscv::ValueType>(lhs.getType());
        auto rhsType = mlir::dyn_cast<riscv::ValueType>(rhs.getType());
        bool commonDomain = lhsType && rhsType;
        for (int64_t axis : reductionAxes) {
          auto lhsAxis = lhsType
                             ? llvm::find(lhsType.getAxisIds().asArrayRef(), axis)
                             : llvm::ArrayRef<int64_t>::iterator();
          auto rhsAxis = rhsType
                             ? llvm::find(rhsType.getAxisIds().asArrayRef(), axis)
                             : llvm::ArrayRef<int64_t>::iterator();
          commonDomain &= lhsType && rhsType &&
                          lhsAxis != lhsType.getAxisIds().asArrayRef().end() &&
                          rhsAxis != rhsType.getAxisIds().asArrayRef().end();
          if (!commonDomain)
            break;
          const size_t lhsPosition = static_cast<size_t>(
              lhsAxis - lhsType.getAxisIds().asArrayRef().begin());
          const size_t rhsPosition = static_cast<size_t>(
              rhsAxis - rhsType.getAxisIds().asArrayRef().begin());
          commonDomain &= lhsType.getShape()[lhsPosition] > 0 &&
                          lhsType.getShape()[lhsPosition] ==
                              rhsType.getShape()[rhsPosition];
        }
        if (!commonDomain)
          return true;
        commonReductionCarrier.anchored = true;
        commonReductionCarrier.fullLaneExtent = true;
        commonReductionCarrier.laneAxis = reductionAxes.back();
        for (int64_t axis :
             llvm::ArrayRef<int64_t>(reductionAxes).drop_back())
          commonReductionCarrier.coalescedLaneAxes.insert(axis);
        anchor = &commonReductionCarrier;
      }
      for (mlir::Value operand : {lhs, rhs}) {
        Roles &operandRoles = roles[operand];
        for (int64_t axis : reductionAxes) {
          operandRoles.replicaAxes.erase(axis);
          operandRoles.coalescedLaneAxes.erase(axis);
          operandRoles.sequentialAxes.erase(axis);
        }
        operandRoles.laneAxis = anchor->laneAxis;
        for (int64_t axis : anchor->coalescedLaneAxes)
          operandRoles.coalescedLaneAxes.insert(axis);
        operandRoles.registerTuple = false;
        operandRoles.anchored = true;
        operandRoles.fullLaneExtent = anchor->fullLaneExtent;
        operandRoles.memoryAnchored = anchor->memoryAnchored;
      }
      return true;
    };
    getOperation().walk([&](mlir::Operation *operation) {
      if (normalizeWidenDotReductionCarrier(operation))
        return;
      // Only an outer contraction has two independently surviving free-axis
      // domains whose shared partial coordinates require one issue mapping.
      // Dot/contract and grouped-MAC reductions already anchor their single
      // surviving domain through the reduction result and must retain it.
      if (!mlir::isa<riscv::OuterContractOp>(operation))
        return;
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      if (!implementation || implementation.getFamily() != "widen-dot")
        return;
      mlir::Value lhs = operation->getOperand(0);
      mlir::Value rhs = operation->getOperand(1);
      auto lhsType = mlir::dyn_cast<riscv::ValueType>(lhs.getType());
      auto rhsType = mlir::dyn_cast<riscv::ValueType>(rhs.getType());
      if (!lhsType || !rhsType)
        return;
      llvm::SmallVector<int64_t> sharedAxes;
      for (auto [lhsPosition, axis] :
           llvm::enumerate(lhsType.getAxisIds().asArrayRef())) {
        auto rhsAxis = llvm::find(rhsType.getAxisIds().asArrayRef(), axis);
        if (rhsAxis == rhsType.getAxisIds().asArrayRef().end())
          continue;
        const size_t rhsPosition = static_cast<size_t>(
            rhsAxis - rhsType.getAxisIds().asArrayRef().begin());
        if (lhsType.getShape()[lhsPosition] != rhsType.getShape()[rhsPosition])
          return;
        sharedAxes.push_back(axis);
      }
      const bool lhsHasUniqueFreeAxis = llvm::any_of(
          lhsType.getAxisIds().asArrayRef(),
          [&](int64_t axis) { return !llvm::is_contained(sharedAxes, axis); });
      const bool rhsHasUniqueFreeAxis = llvm::any_of(
          rhsType.getAxisIds().asArrayRef(),
          [&](int64_t axis) { return !llvm::is_contained(sharedAxes, axis); });
      if (sharedAxes.empty() || !lhsHasUniqueFreeAxis ||
          !rhsHasUniqueFreeAxis || roles[lhs].laneAxis == 0 ||
          roles[lhs].laneAxis != roles[rhs].laneAxis ||
          !llvm::is_contained(sharedAxes, roles[lhs].laneAxis))
        return;
      auto sharedReplicaProduct = [&](mlir::Value value) {
        int64_t result = 1;
        for (int64_t axis : sharedAxes)
          if (roles[value].replicaAxes.contains(axis)) {
            const int64_t extent = riscv_internal::physicalExtent(value, axis);
            if (extent <= 0 || !checkedMultiply(result, extent, result))
              return int64_t{-1};
          }
        return result;
      };
      const int64_t lhsReplicas = sharedReplicaProduct(lhs);
      const int64_t rhsReplicas = sharedReplicaProduct(rhs);
      if (lhsReplicas <= 0 || rhsReplicas <= 0 || lhsReplicas == rhsReplicas)
        return;
      mlir::Value anchor = lhsReplicas < rhsReplicas ? lhs : rhs;
      mlir::Value target = lhsReplicas < rhsReplicas ? rhs : lhs;
      Roles &anchorRoles = roles[anchor];
      Roles &targetRoles = roles[target];
      for (int64_t axis : sharedAxes) {
        targetRoles.replicaAxes.erase(axis);
        targetRoles.coalescedLaneAxes.erase(axis);
        targetRoles.sequentialAxes.erase(axis);
        if (anchorRoles.replicaAxes.contains(axis))
          targetRoles.replicaAxes.insert(axis);
        if (anchorRoles.coalescedLaneAxes.contains(axis))
          targetRoles.coalescedLaneAxes.insert(axis);
        if (anchorRoles.sequentialAxes.contains(axis))
          targetRoles.sequentialAxes.insert(axis);
      }
      targetRoles.laneAxis = anchorRoles.laneAxis;
      targetRoles.registerTuple = false;
      targetRoles.anchored = true;
      targetRoles.fullLaneExtent = anchorRoles.fullLaneExtent;
    });
    propagateRoles();
    // The fixed-point propagation above carries the selected operand relation
    // through producers, but storage-facing proposals may also reintroduce a
    // reduction axis as issue time or a replica.  Reapply the same operation
    // anchor once, after propagation, so the final operand types consumed by
    // LowerRISCVComposites have one closed lane/time decomposition.  This is the
    // same owner and rule, not a second topology decision.
    getOperation().walk([&](mlir::Operation *operation) {
      normalizeWidenDotReductionCarrier(operation);
    });

    // The command-line LMUL binding fixes the base physical parameter, but a
    // width-changing use-def chain has one additional legality relation: a
    // widening value that preserves its logical lane span needs proportionally
    // more register width.  Freeze that derived width per SSA value and carry it
    // through the selected widening contraction.  This is not a second LMUL
    // candidate; it is the unique width required to keep the same logical lanes.
    llvm::DenseMap<mlir::Value, int64_t> valueLMULBounds;
    for (mlir::Value value : values)
      valueLMULBounds[value] = lmulEighths;
    llvm::DenseSet<mlir::Value> pointwiseDerivedWidths;
    bool widthChanged = true;
    while (widthChanged) {
      widthChanged = false;
      getOperation().walk([&](mlir::Operation *operation) {
        if (mlir::isa<riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp, riscv::WidenOp>(operation) &&
            operation->getNumOperands() == 1 && operation->getNumResults() == 1) {
          mlir::Value input = operation->getOperand(0);
          mlir::Value result = operation->getResult(0);
          auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
          auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
          if (!inputType || !resultType ||
              inputType.getShape() != resultType.getShape() ||
              inputType.getAxisIds() != resultType.getAxisIds() ||
              roles[input].laneAxis == 0 ||
              roles[input].laneAxis != roles[result].laneAxis ||
              roles[input].local || roles[result].local)
            return;
          auto kernel = operation->getParentOfType<riscv::KernelOp>();
          if (!kernel)
            return;
          const int64_t inputSEW = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(inputType));
          const int64_t resultSEW = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(resultType));
          auto propagateWidth = [&](mlir::Value from, int64_t fromSEW,
                                    mlir::Value to, int64_t toSEW) {
            const int64_t fromWidth = valueLMULBounds.lookup(from);
            const int64_t requested =
                (fromWidth * toSEW + fromSEW - 1) / fromSEW;
            const int64_t legal =
                roundLegalLMUL(kernel.getTarget(), requested, toSEW);
            if (legal && legal > valueLMULBounds.lookup(to)) {
              valueLMULBounds[to] = legal;
              widthChanged = true;
            }
          };
          propagateWidth(input, inputSEW, result, resultSEW);
          propagateWidth(result, resultSEW, input, inputSEW);
          return;
        }

        // A pointwise consumer preserves the lane span already derived from
        // widening. Resetting its result to the base width would immediately
        // strip-mine that same numerical chain back into smaller issues.
        if (mlir::isa<riscv::BinaryOp, riscv::UnaryOp>(operation)) {
          mlir::Value result = operation->getResult(0);
          auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
          if (!resultType ||
              !mlir::isa<mlir::IntegerType>(resultType.getElementType()) ||
              !roles[result].laneAxis || roles[result].local)
            return;
          for (mlir::Value input : operation->getOperands()) {
            auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
            if (!inputType ||
                inputType.getElementType() != resultType.getElementType() ||
                inputType.getShape() != resultType.getShape() ||
                inputType.getAxisIds() != resultType.getAxisIds() ||
                roles[input].laneAxis != roles[result].laneAxis ||
                roles[input].local)
              continue;
            const int64_t width = valueLMULBounds.lookup(input);
            if (width > valueLMULBounds.lookup(result)) {
              valueLMULBounds[result] = width;
              pointwiseDerivedWidths.insert(result);
              widthChanged = true;
            }
          }
          return;
        }

        if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                       riscv::OuterContractOp>(operation))
          return;
        auto implementation = operation->getAttrOfType<
            riscv::ImplementationAttr>("implementation");
        if (!implementation || implementation.getFamily() != "widen-dot")
          return;
        mlir::Value lhs = operation->getOperand(0);
        mlir::Value rhs = operation->getOperand(1);
        if (!valueLMULBounds.count(lhs) || !valueLMULBounds.count(rhs) ||
            roles[lhs].laneAxis == 0 ||
            roles[lhs].laneAxis != roles[rhs].laneAxis)
          return;
        auto kernel = operation->getParentOfType<riscv::KernelOp>();
        if (!kernel)
          return;
        const int64_t axis = roles[lhs].laneAxis;
        const int64_t lhsExtent = riscv_internal::physicalExtent(lhs, axis);
        const int64_t rhsExtent = riscv_internal::physicalExtent(rhs, axis);
        if (lhsExtent <= 0 || lhsExtent != rhsExtent)
          return;
        auto laneCapacity = [&](mlir::Value value) {
          const int64_t sew = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(value.getType()));
          return std::min(
              lhsExtent,
              std::max<int64_t>(
                  1, kernel.getTarget().getVlenBits() *
                         valueLMULBounds.lookup(value) / (8 * sew)));
        };
        const int64_t desired = std::max(laneCapacity(lhs), laneCapacity(rhs));
        for (mlir::Value value : {lhs, rhs}) {
          const int64_t sew = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(value.getType()));
          const int64_t requested =
              std::max<int64_t>(1, (desired * sew * 8 +
                                    kernel.getTarget().getVlenBits() - 1) /
                                       kernel.getTarget().getVlenBits());
          const int64_t legal =
              roundLegalLMUL(kernel.getTarget(), requested, sew);
          if (legal && legal > valueLMULBounds.lookup(value)) {
            valueLMULBounds[value] = legal;
            widthChanged = true;
          }
        }
      });
    }

    if (!pointwiseDerivedWidths.empty())
      llvm::errs() << "weft-pointwise-width: propagated-values="
                   << pointwiseDerivedWidths.size() << '\n';

    // Each physical value receives the largest lane span legal for its own
    // type and the instantiated LMUL bound.  Structured-product operands and
    // a same-axis extract remain coupled where the target instruction requires
    // it. A domain projection does not shrink its parent's lane span;
    // ordinary pointwise mismatches become explicit convert_layout edges
    // instead of conservatively shrinking the whole use-def chain.
    llvm::DenseMap<mlir::Value, int64_t> connectedLaneSpans;
    for (mlir::Value value : values) {
      int64_t axis = roles[value].laneAxis;
      if (!axis || roles[value].local)
        continue;
      auto kernel = value.getParentRegion()->getParentOfType<riscv::KernelOp>();
      if (!kernel)
        continue;
      int64_t sew = std::max<int64_t>(
          8, riscv_internal::logicalBitWidth(value.getType()));
      int64_t capacity = std::max<int64_t>(
          1, kernel.getTarget().getVlenBits() * valueLMULBounds.lookup(value) /
                 (8 * sew));
      // Indexed memory makes a storage relation executable; it does not erase
      // the number of logical elements represented by one encoded layer.
      // Preserve that per-use storage width on every target, then express any
      // wider VLEN as additional issue-time strips.
      if (auto storageLimit = encodedLaneLimit(value, axis))
        capacity = std::min(capacity, *storageLimit);
      int64_t extent = riscv_internal::physicalExtent(value, axis);
      connectedLaneSpans[value] =
          extent > 0 ? std::min(extent, capacity) : capacity;
    }
    bool laneSpanChanged = true;
    while (laneSpanChanged) {
      laneSpanChanged = false;
      getOperation().walk([&](mlir::Operation *operation) {
        auto singleAxis = [&](mlir::Value value) {
          auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
          return type && llvm::count_if(type.getShape().asArrayRef(), [](int64_t n) { return n != 1; }) == 1;
        };
        auto capSpan = [&](mlir::Value value, int64_t capacity) {
          if (capacity > 0 && connectedLaneSpans.count(value) && connectedLaneSpans[value] > capacity) {
            connectedLaneSpans[value] = capacity;
            laneSpanChanged = true;
          }
        };
        if (auto select = mlir::dyn_cast<riscv::SelectOp>(operation)) {
          auto condition = mlir::dyn_cast<riscv::ValueType>(select.getCondition().getType());
          auto output = mlir::dyn_cast<riscv::ValueType>(select.getResult().getType());
          if (condition && output && singleAxis(select.getCondition()) &&
              condition.getShape() == output.getShape() && condition.getAxisIds() == output.getAxisIds() &&
              roles[select.getCondition()].laneAxis == roles[select.getResult()].laneAxis)
            capSpan(select.getCondition(), connectedLaneSpans.lookup(select.getResult()));
        }
        if (auto logic = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
          auto type = mlir::dyn_cast<riscv::ValueType>(logic.getResult().getType());
          if (type && type.getElementType().isInteger(1) && singleAxis(logic.getResult())) {
            llvm::SmallVector<mlir::Value> connected{logic.getResult()};
            int64_t span = connectedLaneSpans.lookup(logic.getResult());
            for (mlir::Value operand : logic->getOperands()) {
              auto input = mlir::dyn_cast<riscv::ValueType>(operand.getType());
              if (!input || input.getShape() != type.getShape() || input.getAxisIds() != type.getAxisIds() ||
                  !connectedLaneSpans.count(operand) || roles[operand].laneAxis != roles[logic.getResult()].laneAxis) continue;
              connected.push_back(operand);
              span = std::min(span, connectedLaneSpans[operand]);
            }
            for (auto value : connected) capSpan(value, span);
          }
        }
        if (auto comparison = mlir::dyn_cast<riscv::CompareOp>(operation)) {
          mlir::Value result = comparison.getResult();
          if (connectedLaneSpans.count(result)) {
            auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
            int64_t maximumLMUL = 0;
            for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
              maximumLMUL = std::max(maximumLMUL, legal);
            for (mlir::Value operand : comparison->getOperands()) {
              auto type = mlir::dyn_cast<riscv::ValueType>(operand.getType());
              int64_t axis = roles[result].laneAxis;
              if (!type || !axis || !containsAxis(type, axis) ||
                  riscv_internal::physicalExtent(operand, axis) == 1) continue;
              int64_t sew = std::max<int64_t>(8, riscv_internal::logicalBitWidth(type));
              int64_t capacity = target.getVlenBits() * maximumLMUL / (8 * sew);
              if (singleAxis(result) && type.getShape() == mlir::cast<riscv::ValueType>(result.getType()).getShape() &&
                  connectedLaneSpans.count(operand) && roles[operand].laneAxis == axis)
                capacity = std::min(capacity, connectedLaneSpans[operand]);
              // A byte predicate is not a hardware mask capable of comparing
              // four times as many f32 coordinates in one instruction. Bound
              // its mapping by the data operand's legal width first; unusual
              // extents can then use the common bounded value realization.
              if (capacity > 0 && connectedLaneSpans[result] > capacity) {
                connectedLaneSpans[result] = capacity;
                laneSpanChanged = true;
              }
            }
          }
        }
        if (mlir::isa<riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp, riscv::WidenOp>(operation)) {
          mlir::Value input = operation->getOperand(0);
          mlir::Value result = operation->getResult(0);
          auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
          auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
          if (inputType && resultType && inputType.getShape().size() == 1 &&
              inputType.getShape() == resultType.getShape() &&
              inputType.getAxisIds() == resultType.getAxisIds() &&
              connectedLaneSpans.count(result) && roles[input].laneAxis &&
              roles[input].laneAxis == roles[result].laneAxis) {
            auto target =
                operation->getParentOfType<riscv::KernelOp>().getTarget();
            int64_t maximumLMUL = 0;
            for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
              maximumLMUL = std::max(maximumLMUL, legal);
            const int64_t inputSEW = std::max<int64_t>(
                8, riscv_internal::logicalBitWidth(inputType));
            const int64_t capacity =
                target.getVlenBits() * maximumLMUL / (8 * inputSEW);
            // The operand must realize the result's lanes at its own SEW.
            // A narrower result cannot demand an unsupported wide operand.
            if (capacity > 0 && connectedLaneSpans[result] > capacity) {
              connectedLaneSpans[result] = capacity;
              laneSpanChanged = true;
            }
          }
        }
        if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                       riscv::OuterContractOp>(operation)) {
          if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation)) {
            mlir::Value input = extract.getInput();
            mlir::Value result = extract.getResult();
            if (!connectedLaneSpans.count(input) ||
                !connectedLaneSpans.count(result) ||
                roles[input].laneAxis == 0 ||
                roles[input].laneAxis != roles[result].laneAxis)
              return;
            int64_t span =
                std::min(connectedLaneSpans[input], connectedLaneSpans[result]);
            const int64_t axis = roles[input].laneAxis;
            const int64_t inputExtent =
                riscv_internal::physicalExtent(input, axis);
            const int64_t resultExtent =
                riscv_internal::physicalExtent(result, axis);
            unsigned domains = 0;
            bool alignedDomain = true;
            auto inputType = mlir::cast<riscv::ValueType>(input.getType());
            auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
            for (auto [position, attribute] : llvm::enumerate(extract.getSelectors())) {
              auto selector = mlir::cast<mlir::StringAttr>(attribute).getValue();
              if (selector == "domain") {
                ++domains;
                alignedDomain &= inputType.getAxisIds()[position] == axis;
              } else alignedDomain &= selector == "all";
            }
            const bool domainProjection = alignedDomain && domains == 1 && resultType &&
                inputType.getAxisIds() == resultType.getAxisIds() &&
                inputExtent > resultExtent && resultExtent > 0 && inputExtent % resultExtent == 0;
            for (mlir::Value value : {input, result})
              if ((!domainProjection || value == result) &&
                  connectedLaneSpans[value] != span) {
                connectedLaneSpans[value] = span;
                laneSpanChanged = true;
              }
            return;
          } else {
            return;
          }
        }
        auto implementation = operation->getAttrOfType<riscv::ImplementationAttr>(
            "implementation");
        if (!implementation || implementation.getFamily() != "widen-dot")
          return;
        mlir::Value lhs = operation->getOperand(0);
        mlir::Value rhs = operation->getOperand(1);
        if (!connectedLaneSpans.count(lhs) || !connectedLaneSpans.count(rhs) ||
            roles[lhs].laneAxis == 0 ||
            roles[lhs].laneAxis != roles[rhs].laneAxis)
          return;
        int64_t span =
            std::min(connectedLaneSpans[lhs], connectedLaneSpans[rhs]);
        for (mlir::Value value : {lhs, rhs})
          if (connectedLaneSpans[value] != span) {
            connectedLaneSpans[value] = span;
            laneSpanChanged = true;
          }
      });
    }

    // State residency depends on the actual selected decomposition. Fractional
    // LMUL parts each occupy one architectural register: byte capacity alone
    // cannot prove that a complete loop-carried value fits.
    bool localState = false;
    for (mlir::Value value : values) {
      auto state = value.getDefiningOp<riscv::NewOp>();
      if (!state || roles[value].local || !state.getInitialized() ||
          !mlir::isa<mlir::IntegerType, mlir::FloatType>(state.getInitial().getType())) continue;
      auto layout = buildLayout(builder, value, roles[value],
          std::max<int64_t>(1, connectedLaneSpans.lookup(value)), valueLMULBounds.lookup(value));
      if (!layout || layout.getCarrier() != "rvv") continue;
      auto time = riscv_internal::staticProduct(layout.getTimeFactors().asArrayRef());
      auto resident = time ? riscv_internal::staticProduct({*time, layout.getRegisterGroups()}) : std::nullopt;
      auto target = state->getParentOfType<riscv::KernelOp>().getTarget();
      // Keep room for one selected update issue. A state occupying the entire
      // bank cannot also receive its next operand, even if its bytes alone fit.
      int64_t available = target.getVectorRegisters() - layout.getRegisterGroups();
      if (!resident || *resident > available) {
        roles[value].local = true;
        localState = true;
      }
    }
    if (localState) propagateRoles();

    for (mlir::Value value : values) {
      int64_t connectedLaneSpan =
          std::max<int64_t>(1, connectedLaneSpans.lookup(value));
      riscv::LayoutAttr layout =
          buildLayout(builder, value, roles[value], connectedLaneSpan,
                      valueLMULBounds.lookup(value));
      if (!layout) {
        value.getParentRegion()->getParentOp()->emitError(
            "no legal LMUL can preserve the inferred logical lane mapping");
        failed = true;
        continue;
      }
      setValueLayout(value, layout);
    }
    // A reshape preserves its producer's linear storage/register partition.
    // Consumer layout conflicts remain explicit use conversions.
    getOperation().walk<mlir::WalkOrder::PreOrder>([&](riscv::ReshapeOp reshape) {
      if (auto layout = reshapeCarrierLayout(builder, reshape))
        setValueLayout(reshape.getResult(), layout);
    });
    // A register lookup uses the result's RVV register group as a table source.
    // The table may contain fewer active lanes than the result (for example a
    // 16-entry codebook feeding a 32-lane result on VLEN256), but vrgather.vv
    // still requires the same SEW/LMUL register type.  Propagate that physical
    // width through the explicit materialize edge; do not manufacture or merge
    // either logical axis.
    getOperation().walk([&](riscv::LookupOp lookup) {
      auto materialize = riscv_internal::stripRepresentationConversions(
                             lookup.getTable())
                             .getDefiningOp<riscv::MaterializeOp>();
      auto directLoad = lookup.getTable().getDefiningOp<riscv::LoadOp>();
      auto table = mlir::dyn_cast<riscv::ValueType>(lookup.getTable().getType());
      auto indices =
          mlir::dyn_cast<riscv::ValueType>(lookup.getIndices().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(lookup.getResult().getType());
      const bool sharedTable = materialize && materialize.getPlacement() == "shared";
      const bool singleUseRead = directLoad && directLoad.getResult().hasOneUse() &&
          table && indices && result &&
          table.getLayout().getSew() == result.getLayout().getSew() &&
          indices.getLayout().getSew() == result.getLayout().getSew();
      if ((!sharedTable && !singleUseRead) || !table ||
          !indices || !result || table.getLayout().getCarrier() != "rvv" ||
          indices.getLayout().getCarrier() != "rvv" ||
          result.getLayout().getCarrier() != "rvv")
        return;
      if (table.getLayout().getSew() != result.getLayout().getSew() ||
          indices.getLayout().getSew() != result.getLayout().getSew()) {
        lookup.emitError(
            "register lookup operands have no common RVV SEW representation");
        failed = true;
        return;
      }
      const int64_t requiredLMUL = result.getLayout().getLmulEighths();
      auto tableElements =
          riscv_internal::staticProduct(table.getShape().asArrayRef());
      const int64_t tableCapacity =
          lookup->getParentOfType<riscv::KernelOp>().getTarget().getVlenBits() *
          requiredLMUL / (8 * result.getLayout().getSew());
      // A table larger than the selected result register group remains an
      // addressable table.  PlanRISCVMemory will choose an indexed-memory leaf
      // for this physical instance instead of inventing a partial register
      // table or changing the logical lookup.
      if (!tableElements || *tableElements > tableCapacity)
        return;
      if (table.getLayout().getLmulEighths() == requiredLMUL)
        return;
      auto update = [&](mlir::Value value) {
        auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
        if (!type || type.getLayout().getCarrier() != "rvv" ||
            type.getLayout().getSew() != result.getLayout().getSew())
          return;
        setValueLayout(value,
                       withRegisterWidth(builder, type.getLayout(), requiredLMUL));
      };
      if (sharedTable) {
        update(materialize.getInput());
        update(materialize.getResult());
      } else {
        // Preserve this read point and its logical extent. Only its unused
        // physical register capacity changes to match the sole lookup use.
        update(directLoad.getResult());
      }
    });
    if (mlir::failed(weft::planRISCVResidency(getOperation()))) {
      signalPassFailure();
      return;
    }
    getOperation().walk([&](riscv::NewOp state) {
      if (auto value = mlir::dyn_cast<riscv::ValueType>(state.getResult().getType()))
        state.setPlacementAttr(builder.getStringAttr(
            value.getLayout().getCarrier() == "local" ? "local" :
            value.getLayout().getCarrier() == "scalar" ? "scalar" : "register"));
    });
    if (failed) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<riscv::ConvertLayoutOp> redundantConversions;
    getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
      if (conversion.getConversion().getEffect() == "pure" &&
          conversion.getInput().getType() == conversion.getResult().getType())
        redundantConversions.push_back(conversion);
    });
    for (riscv::ConvertLayoutOp conversion : redundantConversions) {
      conversion.getResult().replaceAllUsesWith(conversion.getInput());
      conversion.erase();
    }

    insertUseConversions(builder);
  }

private:
  static mlir::LogicalResult convertControlOperand(mlir::OpBuilder &builder,
                                                   mlir::Operation *owner,
                                                   unsigned index,
                                                   mlir::Type required) {
    mlir::Value value = owner->getOperand(index);
    if (value.getType() == required)
      return mlir::success();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    auto target = mlir::dyn_cast<riscv::ValueType>(required);
    if (!source || !target || source.getElementType() != target.getElementType() ||
        source.getShape() != target.getShape() ||
        source.getAxisIds() != target.getAxisIds())
      return owner->emitError(
          "control-flow handoff changes the logical value domain");
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(owner);
    auto conversion = builder.create<riscv::ConvertLayoutOp>(
        owner->getLoc(), target, value,
        riscv_internal::layoutConversion(builder, source.getLayout(),
                                         target.getLayout()),
        riscv::AccessAttr(),
        riscv_internal::unselectedLeaf(builder));
    riscv_internal::copyOrigin(owner, conversion);
    owner->setOperand(index, conversion.getResult());
    return mlir::success();
  }

  bool reconcileControlCarries(mlir::OpBuilder &builder) {
    bool failed = false;
    getOperation().walk([&](riscv::LoopOp loop) {
      mlir::Block &body = loop.getBody().front();
      auto yield = mlir::cast<riscv::YieldOp>(body.getTerminator());
      for (auto [index, carried] : llvm::enumerate(loop.getCarried())) {
        mlir::Type type = body.getArgument(index + 1).getType();
        loop.getResult(index).setType(type);
        failed |= mlir::failed(convertControlOperand(
            builder, loop, loop.getCarried().getBeginOperandIndex() + index, type));
        failed |= mlir::failed(convertControlOperand(builder, yield, index, type));
      }
    });
    getOperation().walk([&](mlir::scf::ForOp loop) {
      for (auto [index, initial] : llvm::enumerate(loop.getInitArgs())) {
        mlir::Type type = loop.getRegionIterArg(index).getType();
        loop.getResult(index).setType(type);
        failed |= mlir::failed(convertControlOperand(
            builder, loop, loop.getInitArgs().getBeginOperandIndex() + index, type));
        failed |= mlir::failed(convertControlOperand(
            builder, loop.getBody()->getTerminator(), index, type));
      }
    });
    getOperation().walk([&](mlir::scf::IfOp branch) {
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        failed |= mlir::failed(convertControlOperand(
            builder, branch.thenYield(), index, result.getType()));
        failed |= mlir::failed(convertControlOperand(
            builder, branch.elseYield(), index, result.getType()));
      }
    });
    getOperation().walk([&](mlir::scf::WhileOp loop) {
      auto condition = mlir::cast<mlir::scf::ConditionOp>(
          loop.getBefore().front().getTerminator());
      auto yield = mlir::cast<mlir::scf::YieldOp>(
          loop.getAfter().front().getTerminator());
      for (auto [index, initial] : llvm::enumerate(loop.getInits())) {
        mlir::Type type = loop.getBefore().front().getArgument(index).getType();
        failed |= mlir::failed(convertControlOperand(builder, loop, index, type));
        failed |= mlir::failed(convertControlOperand(builder, yield, index, type));
      }
      for (auto [index, argument] : llvm::enumerate(loop.getAfter().front().getArguments())) {
        mlir::Type type = argument.getType();
        loop.getResult(index).setType(type);
        failed |= mlir::failed(
            convertControlOperand(builder, condition, index + 1, type));
      }
    });
    return failed;
  }

  bool reconcileStateInitializers(mlir::OpBuilder &builder) {
    bool failed = false;
    llvm::SmallVector<riscv::NewOp> states;
    getOperation().walk([&](riscv::NewOp state) {
      if (state.getInitialized())
        states.push_back(state);
    });
    for (riscv::NewOp state : states) {
      mlir::Value initial = state.getInitial();
      auto source = mlir::dyn_cast<riscv::ValueType>(initial.getType());
      auto target = mlir::dyn_cast<riscv::ValueType>(state.getResult().getType());
      // Canonical `new` permits a scalar initializer to broadcast to a shaped
      // state.  That is a numerical rule and does not require a layout edge.
      if (!source || !target || initial.getType() == state.getResult().getType())
        continue;
      if (source.getElementType() != target.getElementType() ||
          source.getShape() != target.getShape() ||
          source.getAxisIds() != target.getAxisIds()) {
        state.emitError(
            "physical state initializer changes the canonical logical domain");
        failed = true;
        continue;
      }
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(state);
      auto conversion = builder.create<riscv::ConvertLayoutOp>(
          state.getLoc(), target, initial,
          riscv_internal::layoutConversion(builder, source.getLayout(),
                                           target.getLayout()),
          riscv::AccessAttr(),
          riscv_internal::unselectedLeaf(builder));
      state.getInitialMutable().assign(conversion.getResult());
    }
    return failed;
  }

  bool projectReductionResult(mlir::OpBuilder &builder,
                              riscv::ReduceOp reduce) {
    auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
    auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
    if (!input || reduce.getAxis() < 0 ||
        reduce.getAxis() >= static_cast<int64_t>(input.getAxisIds().size())) {
      reduce.emitError(
          "cannot project a physical reduction result from its input layout");
      return false;
    }
    const size_t eliminated = static_cast<size_t>(reduce.getAxis());
    if (!result) {
      if (input.getAxisIds().size() == 1)
        return true;
      reduce.emitError(
          "a shaped reduction result is required when free axes survive");
      return false;
    }
    llvm::SmallVector<int64_t> expectedShape;
    llvm::SmallVector<int64_t> expectedAxes;
    for (size_t position = 0; position < input.getAxisIds().size(); ++position) {
      if (position == eliminated)
        continue;
      expectedShape.push_back(input.getShape()[position]);
      expectedAxes.push_back(input.getAxisIds()[position]);
    }
    if (result.getShape().asArrayRef() !=
            llvm::ArrayRef<int64_t>(expectedShape) ||
        result.getAxisIds().asArrayRef() !=
            llvm::ArrayRef<int64_t>(expectedAxes)) {
      reduce.emitError(
          "physical reduction result does not equal its input domain with one axis removed");
      return false;
    }
    auto kernel = reduce->getParentOfType<riscv::KernelOp>();
    riscv::LayoutAttr projected = riscv_internal::projectLayout(
        builder, result, input.getLayout(), kernel.getTarget());
    if (!projected) {
      reduce.emitError(
          "no legal target layout preserves every reduction free axis");
      return false;
    }
    setValueLayout(reduce.getResult(), projected);
    return true;
  }

  void reconcileReshapeCarriers(mlir::OpBuilder &builder) {
    // Reduction result projection can change a producer after the initial
    // layout propagation.  Keep the reshape itself representation-preserving
    // and realize the already-selected consumer carrier on an explicit edge.
    getOperation().walk<mlir::WalkOrder::PreOrder>([&](riscv::ReshapeOp reshape) {
      auto required = reshape.getResult().getType();
      auto layout = reshapeCarrierLayout(builder, reshape);
      if (!layout || layout == required.getLayout()) return;
      setValueLayout(reshape.getResult(), layout);
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(reshape);
      auto conversion = builder.create<riscv::ConvertLayoutOp>(
          reshape.getLoc(), required, reshape.getResult(),
          riscv_internal::layoutConversion(builder, layout, required.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
      reshape.getResult().replaceAllUsesExcept(conversion.getResult(), conversion);
    });
  }

  void insertUseConversions(mlir::OpBuilder &builder) {
    // A contraction may produce one scalar register tuple over its surviving
    // free axes while a downstream reduction consumes one of those axes in
    // RVV lanes.  Keep both operation anchors intact and make their handoff a
    // first-class physical conversion instead of forcing one layout onto the
    // shared SSA value.
    bool inputFailure = false;
    getOperation().walk([&](mlir::Operation *operation) {
      if (!mlir::isa<riscv::DotOp, riscv::ContractOp, riscv::OuterContractOp>(operation)) return;
      for (mlir::OpOperand &operand : operation->getOpOperands()) {
        auto type = mlir::dyn_cast<riscv::ValueType>(operand.get().getType());
        if (!type || type.getLayout().getCarrier() != "local" ||
            mlir::isa<kernel::EncodingType>(type.getElementType())) continue;
        Roles consumer;
        if (auto contract = mlir::dyn_cast<riscv::DotOp>(operation))
          constrainReductionContractOperand(contract, operand.get(), consumer);
        else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(operation))
          constrainReductionContractOperand(contract, operand.get(), consumer);
        else constrainOuterContractOperand(mlir::cast<riscv::OuterContractOp>(operation), operand.get(), consumer);
        int64_t span = consumer.laneAxis ? riscv_internal::physicalExtent(operand.get(), consumer.laneAxis) : 1;
        auto layout = buildLayout(builder, operand.get(), consumer, std::max<int64_t>(1, span), lmulEighths);
        if (!layout || layout.getCarrier() == "local") {
          operation->emitError("private contraction input has no bounded compute carrier");
          inputFailure = true;
          continue;
        }
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(operation);
        auto conversion = builder.create<riscv::ConvertLayoutOp>(operation->getLoc(),
            riscv_internal::withLayout(type, layout), operand.get(),
            riscv_internal::layoutConversion(builder, type.getLayout(), layout),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
        operand.set(conversion);
      }
    });
    if (inputFailure) { signalPassFailure(); return; }
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp reduce) { reductions.push_back(reduce); });
    for (riscv::ReduceOp reduce : reductions) {
      if (!projectReductionResult(builder, reduce)) {
        signalPassFailure();
        continue;
      }
      mlir::Value input = reduce.getInput();
      auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
      if (!inputType || reduce.getAxis() < 0 ||
          reduce.getAxis() >=
              static_cast<int64_t>(inputType.getAxisIds().size()))
        continue;
      // A completed local vector is a whole addressable snapshot, including
      // runtime extents.  Composite lowering consumes that storage through
      // the explicit stream-reduction view; loading one register here would
      // truncate its domain to the physical vector width.
      if (inputType.getLayout().getCarrier() == "local" &&
          inputType.getShape().size() == 1 && reduce.getAxis() == 0 &&
          inputType.getElementType().isF32() &&
          reduce.getResult().getType().isF32() &&
          (reduce.getKind() == "add" || reduce.getKind() == "max" ||
           reduce.getKind() == "min"))
        continue;
      const int64_t eliminated = inputType.getAxisIds()[reduce.getAxis()];
      if (input.hasOneUse() &&
          mlir::isa_and_nonnull<riscv::DotOp, riscv::ContractOp,
                                riscv::OuterContractOp>(
              input.getDefiningOp()))
        continue;
      const auto axes = inputType.getAxisIds().asArrayRef();
      const auto time = inputType.getLayout().getTimeFactors().asArrayRef();
      const auto lane = inputType.getLayout().getLaneFactors().asArrayRef();
      const auto replica =
          inputType.getLayout().getReplicaFactors().asArrayRef();
      auto position = llvm::find(axes, eliminated);
      if (position == axes.end())
        continue;
      const size_t ordinal = static_cast<size_t>(position - axes.begin());
      if (ordinal < lane.size() && lane[ordinal] > 1)
        continue;
      auto projectedResult =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      const bool registerReductionKind =
          reduce.getKind() == "add" ||
          (mlir::isa<mlir::FloatType>(inputType.getElementType()) &&
           (reduce.getKind() == "max" || reduce.getKind() == "min"));
      if (registerReductionKind && projectedResult &&
          inputType.getLayout().getCarrier() == "rvv" &&
          projectedResult.getLayout().getCarrier() == "rvv" &&
          time[ordinal] == 1 && replica[ordinal] > 1 &&
          inputType.getLayout().getSew() ==
              projectedResult.getLayout().getSew() &&
          inputType.getLayout().getLmulEighths() ==
              projectedResult.getLayout().getLmulEighths() &&
          inputType.getLayout().getVl() == projectedResult.getLayout().getVl())
        continue;
      // A reduction may consume an issue-time coordinate while preserving an
      // independent SIMD coordinate carried by a surviving free axis.  This is
      // already a closed parent representation: the reduction removes only the
      // selected time factor and does not require a lane transpose.  Treating
      // every non-lane reduction as a layout conflict used to reject grouped
      // reductions whose target lowering explicitly accumulates their streams.
      if (inputType.getLayout().getCarrier() == "rvv" &&
          ordinal < time.size() && time[ordinal] > 1 &&
          ordinal < replica.size() && replica[ordinal] == 1 &&
          llvm::any_of(llvm::enumerate(lane), [&](auto entry) {
            return entry.index() != ordinal && entry.value() > 1;
          }))
        continue;
      if (inputType.getLayout().getCarrier() != "scalar" &&
          inputType.getLayout().getCarrier() != "rvv" &&
          !(inputType.getLayout().getCarrier() == "local" &&
            llvm::all_of(inputType.getShape().asArrayRef(), [](int64_t extent) { return extent > 0; }))) {
        auto diagnostic = reduce.emitError(
            "reduction layout conflict has neither a lane nor a time-axis realization");
        if (mlir::Operation *definition = input.getDefiningOp())
          diagnostic << "; producer=" << definition->getName();
        diagnostic << "; layout=" << inputType.getLayout();
        signalPassFailure();
        continue;
      }
      Roles consumerRoles;
      consumerRoles.anchored = true;
      consumerRoles.fullLaneExtent = true;
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      int64_t connectedLaneSpan =
          std::max<int64_t>(1,
                            riscv_internal::physicalExtent(input, eliminated));
      if (resultType) {
        for (auto [axis, resultTime, resultLane, resultReplica] : llvm::zip(
                 resultType.getAxisIds().asArrayRef(),
                 resultType.getLayout().getTimeFactors().asArrayRef(),
                 resultType.getLayout().getLaneFactors().asArrayRef(),
                 resultType.getLayout().getReplicaFactors().asArrayRef())) {
          if (resultLane > 1) {
            if (!consumerRoles.laneAxis)
              consumerRoles.laneAxis = axis;
            else
              consumerRoles.coalescedLaneAxes.insert(axis);
            if (connectedLaneSpan <=
                std::numeric_limits<int64_t>::max() / resultLane)
              connectedLaneSpan *= resultLane;
          }
          if (resultReplica > 1)
            consumerRoles.replicaAxes.insert(axis);
          if (resultTime > 1)
            consumerRoles.sequentialAxes.insert(axis);
        }
      }
      if (consumerRoles.laneAxis)
        consumerRoles.coalescedLaneAxes.insert(eliminated);
      else
        consumerRoles.laneAxis = eliminated;
      addSmallReplicas(input, consumerRoles, consumerRoles.laneAxis);
      consumerRoles.replicaAxes.erase(eliminated);
      riscv::LayoutAttr required =
          buildLayout(builder, input, consumerRoles, connectedLaneSpan,
                      std::max<int64_t>(
                          lmulEighths,
                          inputType.getLayout().getLmulEighths()));
      if (!required) {
        reduce.emitError(
            "no legal RVV layout can consume the register-axis reduction");
        signalPassFailure();
        continue;
      }
      mlir::Type requiredType =
          riscv_internal::withLayout(input.getType(), required);
      if (requiredType == input.getType())
        continue;
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(reduce);
      auto conversion = builder.create<riscv::ConvertLayoutOp>(
          reduce.getLoc(), requiredType, input,
          riscv_internal::layoutConversion(builder, inputType.getLayout(),
                                           required),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
      reduce.getInputMutable().assign(conversion.getResult());
      if (!projectReductionResult(builder, reduce))
        signalPassFailure();
    }
    reconcileReshapeCarriers(builder);
    if (reconcileControlCarries(builder) ||
        reconcileStateInitializers(builder)) {
      signalPassFailure();
      return;
    }
    llvm::SmallVector<mlir::Operation *> operations;
    getOperation().walk([&](mlir::Operation *operation) {
      // Pointwise operations require one shared element mapping.  Structured
      // products do not: free and reduction operands deliberately retain
      // different lane/register decompositions and are reconciled by their
      // typed RVV step or IME pack operations during composite lowering.
      if (auto materialize = mlir::dyn_cast<riscv::MaterializeOp>(operation);
          materialize && (materialize.getPlacement() == "reload" ||
                          materialize.getPlacement() == "local"))
        return;
      if (pointwiseLike(operation) || mlir::isa<riscv::LookupOp>(operation))
        operations.push_back(operation);
    });
    for (mlir::Operation *operation : operations) {
      if (operation->getNumResults() != 1 ||
          !mlir::isa<riscv::ValueType>(operation->getResult(0).getType()))
        continue;
      auto result = mlir::cast<riscv::ValueType>(operation->getResult(0).getType());
      for (mlir::OpOperand &operand : operation->getOpOperands()) {
        if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation);
            update && operand.getOperandNumber() != 0 &&
                (!update.coversWholeState() || result.getLayout().getCarrier() == "local"))
          continue;
        if (!mlir::isa<riscv::ValueType>(operand.get().getType()))
          continue;
        if (mlir::isa<riscv::LookupOp>(operation) &&
            operand.getOperandNumber() == 0)
          continue;
        auto kernel = operation->getParentOfType<riscv::KernelOp>();
        mlir::Value sourceValue = operand.get();
        auto operandType = mlir::cast<riscv::ValueType>(sourceValue.getType());
        auto sourceInteger =
            mlir::dyn_cast<mlir::IntegerType>(operandType.getElementType());
        auto resultInteger =
            mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
        const bool integerReinterpret =
            mlir::isa<riscv::CastOp>(operation) && sourceInteger &&
            resultInteger &&
            sourceInteger.getWidth() == resultInteger.getWidth();
        riscv::LayoutAttr required =
            (integerReinterpret || mlir::isa<riscv::BitcastOp>(operation) ||
             (mlir::isa<riscv::MaterializeOp>(operation) &&
              operandType.getElementType() == result.getElementType())) &&
                    operandType.getShape() == result.getShape() &&
                    operandType.getAxisIds() == result.getAxisIds()
                ? result.getLayout()
                : riscv_internal::projectLayout(builder, operandType,
                                                result.getLayout(),
                                                kernel.getTarget());
        if (!required) {
          if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(operation);
              lookup && operand.getOperandNumber() == 1 &&
              riscv_internal::analyzeIndexedEntryRelation(
                  sourceValue, result, {}, {}))
            continue;
          operation->emitError()
              << "no legal target layout projects pointwise result "
              << result << " to operand " << operandType;
          signalPassFailure();
          continue;
        }
        mlir::Type requiredType =
            riscv_internal::withLayout(sourceValue.getType(), required);
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(operation);
        if (sourceValue.getType() != requiredType) {
          auto conversion = builder.create<riscv::ConvertLayoutOp>(
              operation->getLoc(), requiredType, sourceValue,
              riscv_internal::layoutConversion(builder,
                                               operandType.getLayout(), required),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(builder));
          riscv_internal::copyOrigin(operation, conversion);
          sourceValue = conversion.getResult();
          operandType = mlir::cast<riscv::ValueType>(sourceValue.getType());
        }

        if (operandType.getShape() == result.getShape() &&
            operandType.getAxisIds() == result.getAxisIds()) {
          operand.set(sourceValue);
          continue;
        }
        bool properSubdomain = operandType.getAxisIds().size() <
                                   result.getAxisIds().size() &&
                               operandType.getElementType() ==
                                   result.getElementType() &&
                               operandType.getLayout().getCarrier() == "rvv" &&
                               result.getLayout().getCarrier() == "rvv" &&
                               operandType.getLayout().getSew() ==
                                   result.getLayout().getSew();
        for (auto [position, axis] :
             llvm::enumerate(operandType.getAxisIds().asArrayRef())) {
          auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
          properSubdomain &=
              found != result.getAxisIds().asArrayRef().end() &&
              operandType.getShape()[position] ==
                  result.getShape()[static_cast<size_t>(
                      found - result.getAxisIds().asArrayRef().begin())];
        }
        auto inputLanes = riscv_internal::staticProduct(
            operandType.getLayout().getLaneFactors().asArrayRef());
        auto resultLanes = riscv_internal::staticProduct(
            result.getLayout().getLaneFactors().asArrayRef());
        // Missing logical axes that remain scalar time/register coordinates are
        // handled by the ordinary pointwise projection.  A physical broadcast
        // is needed only when the result introduces additional SIMD lanes.
        if (inputLanes && resultLanes && *resultLanes <= *inputLanes) {
          operand.set(sourceValue);
          continue;
        }
        if (operandType.getLayout().getCarrier() == "scalar") {
          operand.set(sourceValue);
          continue;
        }
        if (!properSubdomain || !inputLanes || !resultLanes ||
            *inputLanes <= 0 || *resultLanes <= *inputLanes) {
          operation->emitError(
              "pointwise broadcast has no typed RVV sub-domain realization; operand=")
              << operandType << ", result=" << result
              << ", proper_subdomain=" << properSubdomain
              << ", input_lanes=" << inputLanes.value_or(-1)
              << ", result_lanes=" << resultLanes.value_or(-1);
          signalPassFailure();
          continue;
        }
        auto broadcast = builder.create<riscv::RVVAxisBroadcastOp>(
            operation->getLoc(), operation->getResult(0).getType(), sourceValue,
            riscv_internal::leaf(
                builder, "rvv", "axis-broadcast", "rvv.axis-broadcast",
                "rvv.axis-broadcast",
                operandType.getLayout().getRegisterGroups(),
                result.getLayout().getRegisterGroups(), 1, 0, "none", "exact",
                {*inputLanes, *resultLanes}));
        riscv_internal::copyOrigin(operation, broadcast);
        operand.set(broadcast.getResult());
      }
    }

    // Structured products do not require one common full operand layout.  Each
    // operand owns its storage-compatible DotOperand-style representation; the
    // selected widening operation requires only equal reduction slices.  The
    // later partial plan records the exact part and lane offset taken from each
    // operand.  Forcing the wider operand through the narrower layout (or vice
    // versa) duplicates loads and recreates a second layout owner here.

    llvm::SmallVector<riscv::ExtractOp> gathers;
    getOperation().walk([&](riscv::ExtractOp extract) {
      if (llvm::any_of(extract.getSelectors(), [](mlir::Attribute selector) {
            return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                   "gather";
          }))
        gathers.push_back(extract);
    });
    for (riscv::ExtractOp extract : gathers) {
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (!result)
        continue;
      auto input = mlir::dyn_cast<riscv::ValueType>(extract.getInput().getType());
      auto encodedField = riscv_internal::sourceField(extract.getInput());
      const bool hasGatherSelector =
          llvm::any_of(extract.getSelectors(), [](mlir::Attribute selector) {
            return mlir::cast<mlir::StringAttr>(selector).getValue() == "gather";
          });
      const riscv_internal::FieldFacts encodedFacts =
          encodedField ? riscv_internal::fieldFacts(encodedField)
                       : riscv_internal::FieldFacts();
      const bool typedRegularJoined =
          encodedField && hasGatherSelector && encodedFacts.mapping == "joined" &&
          encodedFacts.bitOffset % 8 == 0 &&
          llvm::any_of(extract.getIndices(), [](mlir::Value index) {
            auto relation =
                riscv_internal::analyzeRegularIndexRelation(index);
            return relation && relation->stride == 1 && relation->repeat > 1;
          });
      const bool typedEncodedEntry =
          encodedField && hasGatherSelector &&
          (encodedFacts.mapping == "natural" || typedRegularJoined) &&
          encodedFacts.bitOffset % 8 == 0 &&
          !extract.getIndices().empty() &&
          llvm::all_of(extract.getIndices(), [](mlir::Value index) {
            mlir::Type element = riscv_internal::logicalElement(index.getType());
            auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
            return element.isIndex() || (integer && integer.isUnsigned());
          });
      if (input && input.getShape().size() == 1 && input.getShape()[0] > 0 &&
          input.getLayout().getCarrier() == "rvv" && result.getLayout().getCarrier() == "rvv" &&
          input.getLayout().getSew() == result.getLayout().getSew() && !typedEncodedEntry &&
          input.getLayout().getLmulEighths() > result.getLayout().getLmulEighths() &&
          riscv_internal::staticProduct(input.getLayout().getTimeFactors().asArrayRef()) == 1 &&
          riscv_internal::staticProduct(input.getLayout().getReplicaFactors().asArrayRef()) == 1) {
        // vrgather addresses the complete source VLMAX even when the result
        // contains fewer active lanes. Keep that register width at the gather,
        // then make the consumer's narrower representation an explicit edge.
        auto wide = withRegisterWidth(builder, result.getLayout(), input.getLayout().getLmulEighths());
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Type original = extract.getResult().getType();
        extract.getResult().setType(riscv_internal::withLayout(original, wide));
        builder.setInsertionPointAfter(extract);
        auto conversion = builder.create<riscv::ConvertLayoutOp>(extract.getLoc(), original,
            extract.getResult(), riscv_internal::layoutConversion(builder, wide, result.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
        extract.getResult().replaceAllUsesExcept(conversion.getResult(), conversion);
        result = mlir::cast<riscv::ValueType>(extract.getResult().getType());
      }
      if (input && input.getLayout().getCarrier() == "local" &&
          result.getLayout().getCarrier() == "rvv" &&
          input.getElementType() == result.getElementType() &&
          !typedEncodedEntry) {
        auto kernel = extract->getParentOfType<riscv::KernelOp>();
        auto required = registerGatherSourceLayout(
            builder, input, result.getLayout(), kernel.getTarget());
        if (required) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(extract);
          auto conversion = builder.create<riscv::ConvertLayoutOp>(
              extract.getLoc(),
              riscv_internal::withLayout(input, required), extract.getInput(),
              riscv_internal::layoutConversion(builder, input.getLayout(),
                                               required),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(builder));
          extract.getInputMutable().assign(conversion.getResult());
        }
      }
      size_t indexCursor = 0;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(extract.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (indexCursor >= extract.getIndices().size()) {
          extract.emitError("gather selector has no physical index operand");
          signalPassFailure();
          break;
        }
        mlir::OpOperand &operand =
            extract->getOpOperand(1 + indexCursor++);
        if (selector != "gather" ||
            !mlir::isa<riscv::ValueType>(operand.get().getType()))
          continue;
        auto kernel = extract->getParentOfType<riscv::KernelOp>();
        riscv::LayoutAttr required = riscv_internal::projectLayout(
            builder, mlir::cast<riscv::ValueType>(operand.get().getType()),
            result.getLayout(), kernel.getTarget());
        if (required && required.getCarrier() == "rvv" && result.getLayout().getCarrier() == "rvv") {
          int64_t indexWidth = result.getLayout().getLmulEighths() * required.getSew();
          if (indexWidth % result.getLayout().getSew() == 0) {
            indexWidth /= result.getLayout().getSew();
            if (llvm::is_contained(kernel.getTarget().getLegalLMULEighths().asArrayRef(), indexWidth))
              required = withRegisterWidth(builder, required, indexWidth);
          }
        }
        if (!required) {
          llvm::SmallVector<int64_t> retainedAxes;
          llvm::SmallVector<int64_t> retainedShape;
          if (input)
            for (size_t position = 0; position < input.getAxisIds().size();
                 ++position) {
              if (position == dimension)
                continue;
              retainedAxes.push_back(input.getAxisIds()[position]);
              retainedShape.push_back(input.getShape()[position]);
            }
          if (riscv_internal::analyzeIndexedEntryRelation(
                  operand.get(), result, retainedAxes, retainedShape))
            continue;
          if (typedRegularJoined) {
            auto relation =
                riscv_internal::analyzeRegularIndexRelation(operand.get());
            if (relation && relation->stride == 1 && relation->repeat > 1)
              continue;
          }
          extract.emitError(
              "no legal target layout projects the gather result mapping to its index");
          signalPassFailure();
          continue;
        }
        mlir::Type requiredType =
            riscv_internal::withLayout(operand.get().getType(), required);
        if (operand.get().getType() == requiredType)
          continue;
        auto source =
            mlir::cast<riscv::ValueType>(operand.get().getType()).getLayout();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(extract);
        auto conversion = builder.create<riscv::ConvertLayoutOp>(
            extract.getLoc(), requiredType, operand.get(),
            riscv_internal::layoutConversion(builder, source, required),
            riscv::AccessAttr(),
            riscv_internal::unselectedLeaf(builder));
        operand.set(conversion.getResult());
      }
    }
    materializeScalarIndexedSources(getOperation(), builder);
  }

private:
  Option<int64_t> lmulEighths{*this, "lmul-eighths", llvm::cl::init(8),
      llvm::cl::desc("Instantiated base LMUL binding in eighths")};
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createPropagateRISCVLayoutsPass(int64_t lmulEighths) {
  return std::make_unique<PropagateRISCVLayoutsPass>(lmulEighths);
}
