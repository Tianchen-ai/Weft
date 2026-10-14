#include "PipelineShapeBindings.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
#include <limits>

using namespace mlir;
using namespace weft;

namespace {
constexpr llvm::StringLiteral stageAttribute = "weft.riscv.pipeline_stage";

ArrayRef<int64_t> shapeOf(Type type) {
  if (auto value = dyn_cast<riscv::ValueType>(type)) return value.getShape().asArrayRef();
  if (auto memory = dyn_cast<riscv::MemDescType>(type)) return memory.getShape().asArrayRef();
  if (auto local = dyn_cast<riscv::LocalType>(type)) return local.getShape().asArrayRef();
  if (auto fragment = dyn_cast<riscv::FragmentType>(type)) return fragment.getShape().asArrayRef();
  return {};
}

SmallVector<int64_t> shapeReferences(Operation *operation) {
  SmallVector<int64_t> result;
  llvm::DenseSet<Type> seenTypes;
  llvm::DenseSet<Attribute> seenAttributes;
  std::function<void(Type)> type;
  std::function<void(Attribute)> attribute;
  type = [&](Type value) {
    if (!seenTypes.insert(value).second) return;
    for (int64_t extent : shapeOf(value))
      if (extent < 0 && !llvm::is_contained(result, -extent)) result.push_back(-extent);
    value.walkImmediateSubElements(attribute, type);
  };
  attribute = [&](Attribute value) {
    if (seenAttributes.insert(value).second) value.walkImmediateSubElements(attribute, type);
  };
  for (Type value : operation->getOperandTypes()) type(value);
  for (Type value : operation->getResultTypes()) type(value);
  for (auto value : operation->getAttrs()) attribute(value.getValue());
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (BlockArgument argument : block.getArguments()) type(argument.getType());
  return result;
}

Operation *ownerInBody(Operation *operation, Block *body) {
  while (operation && operation->getBlock() != body) operation = operation->getParentOp();
  return operation;
}

DenseI64ArrayAttr renameShape(DenseI64ArrayAttr shape, const llvm::DenseMap<int64_t, int64_t> &versions) {
  SmallVector<int64_t> result(shape.asArrayRef());
  bool changed = false;
  for (int64_t &extent : result) {
    if (extent >= 0) continue;
    auto replacement = versions.find(-extent);
    if (replacement == versions.end()) continue;
    extent = -replacement->second;
    changed = true;
  }
  return changed ? DenseI64ArrayAttr::get(shape.getContext(), result) : shape;
}

void replaceShapeTypes(Operation *operation, const llvm::DenseMap<int64_t, int64_t> &versions) {
  if (versions.empty()) return;
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](riscv::MemDescType type) -> Type {
    return riscv::MemDescType::get(type.getContext(), type.getEncoding(), renameShape(type.getShape(), versions),
        type.getAxisIds(), type.getStrides(), type.getOrigins(), type.getAlignment(), type.getAddressClass(),
        type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(), type.getStorageBits(),
        type.getElements(), type.getInterleaveRows());
  });
  replacer.addReplacement([&](riscv::ValueType type) -> Type {
    return riscv::ValueType::get(type.getContext(), type.getElementType(), renameShape(type.getShape(), versions),
        type.getAxisIds(), type.getLayout());
  });
  replacer.addReplacement([&](riscv::LocalType type) -> Type {
    return riscv::LocalType::get(type.getContext(), type.getElementType(), renameShape(type.getShape(), versions),
        type.getAxisIds(), type.getSizeBytes(), type.getAlignment(), type.getAliasSet(), type.getPurpose(),
        type.getOwnerDomainId(), type.getBirthId(), type.getLifetimeEndDomainId(), type.getSchema());
  });
  replacer.addReplacement([&](riscv::FragmentType type) -> Type {
    return riscv::FragmentType::get(type.getContext(), type.getFamily(), type.getRole(), type.getPacking(),
        type.getElementType(), renameShape(type.getShape(), versions), type.getAxisIds(), type.getLayout(), type.getResourceGroups());
  });
  replacer.addReplacement([&](riscv::LayoutAttr layout) -> Attribute {
    return riscv::LayoutAttr::get(layout.getContext(), layout.getCarrier(), layout.getAxisIds(),
        layout.getTimeFactors(), layout.getLaneFactors(), layout.getReplicaFactors(), layout.getFragmentFactors(),
        renameShape(layout.getLocalFactors(), versions), layout.getSew(), layout.getLmulEighths(), layout.getVl(),
        layout.getRegisterGroups(), layout.getValidity(), layout.getReplicaGroup());
  });
  replacer.replaceElementsIn(operation, true, false, true);
}

void renameTypes(Operation *operation, Operation *source,
                 const llvm::DenseMap<Value, int64_t> &bindings) {
  llvm::DenseMap<int64_t, int64_t> versions;
  for (int64_t id : shapeReferences(source)) {
    auto binding = weft::localShapeExtent<riscv::LocalShapeOp>(source, id);
    auto version = bindings.find(binding);
    if (version != bindings.end()) versions[id] = version->second;
  }
  replaceShapeTypes(operation, versions);
}
} // namespace

FailureOr<SmallVector<Value>> riscv_internal::physicalShapeDependencies(Operation *operation) {
  auto kernel = dyn_cast<riscv::KernelOp>(operation);
  if (!kernel) kernel = operation->getParentOfType<riscv::KernelOp>();
  if (!kernel) return operation->emitError("physical shape query has no owning kernel"), failure();
  SmallVector<Value> result;
  for (int64_t id : shapeReferences(operation)) {
    if (id <= int64_t(kernel.getShapeSymbols().size())) continue;
    Value binding = weft::localShapeExtent<riscv::LocalShapeOp>(operation, id);
    if (!binding)
      return operation->emitError("physical type has no dominating declaration for local shape ") << id, failure();
    if (!llvm::is_contained(result, binding)) result.push_back(binding);
  }
  return result;
}

bool riscv_internal::isPipelineShapeBinding(Operation *operation) {
  return isa<riscv::LocalShapeOp>(operation);
}

LogicalResult riscv_internal::verifyPipelineShapeStages(scf::ForOp loop) {
  bool valid = true;
  int64_t publicSymbols = loop->getParentOfType<riscv::KernelOp>().getShapeSymbols().size();
  loop.getBody()->walk([&](Operation *operation) {
    Operation *owner = ownerInBody(operation, loop.getBody());
    if (!owner || owner == loop.getBody()->getTerminator()) return;
    auto stage = owner->getAttrOfType<IntegerAttr>(stageAttribute);
    if (!stage) {
      operation->emitError("shape-binding verification requires a scheduled stage");
      valid = false;
      return;
    }
    for (int64_t id : shapeReferences(operation)) {
      if (id <= publicSymbols) continue;
      Value binding = weft::localShapeExtent<riscv::LocalShapeOp>(operation, id);
      if (!binding) {
        operation->emitError("pipelined type has no dominating local shape declaration");
        valid = false;
        continue;
      }
      Operation *definition = ownerInBody(binding.getDefiningOp(), loop.getBody());
      if (!definition) continue; // A dominating outer binding is iteration-invariant.
      auto sourceStage = definition->getAttrOfType<IntegerAttr>(stageAttribute);
      if (!sourceStage || (sourceStage.getInt() != stage.getInt() &&
                           !definition->hasAttr("weft.riscv.pipeline_invariant"))) {
        operation->emitError("type-bound local shape cannot cross pipeline stages without its declaration");
        valid = false;
      }
    }
  });
  return success(valid);
}

int64_t riscv_internal::nextPipelineShapeIdentity(Operation *root) {
  int64_t identity = 0;
  root->walk([&](riscv::KernelOp kernel) {
    identity = std::max(identity, int64_t(kernel.getShapeSymbols().size()));
  });
  root->walk([&](riscv::LocalShapeOp shape) { identity = std::max(identity, int64_t(shape.getId())); });
  return identity;
}

Operation *riscv_internal::clonePipelineOperation(IRRewriter &rewriter, Operation *source,
    IRMapping &values, llvm::DenseMap<Value, int64_t> &shapeVersions, int64_t &nextIdentity) {
  Operation *clone = rewriter.clone(*source, values);
  SmallVector<Operation *> originalOperations, clonedOperations;
  source->walk<WalkOrder::PreOrder>([&](Operation *operation) { originalOperations.push_back(operation); });
  clone->walk<WalkOrder::PreOrder>([&](Operation *operation) { clonedOperations.push_back(operation); });
  for (auto [original, operation] : llvm::zip_equal(originalOperations, clonedOperations)) {
    if (auto shape = dyn_cast<riscv::LocalShapeOp>(original)) {
      int64_t id = ++nextIdentity;
      shapeVersions[shape.getResult()] = id;
      cast<riscv::LocalShapeOp>(operation).setIdAttr(rewriter.getI64IntegerAttr(id));
    }
    // Resolve the source declaration lexically. A nested declaration may use
    // the same spelling as an outer one; its version must not escape its block.
    renameTypes(operation, original, shapeVersions);
  }
  clone->removeAttr("weft.riscv.pipeline_stage");
  clone->removeAttr("weft.riscv.pipeline_order");
  clone->removeAttr("weft.riscv.pipeline_invariant");
  return clone;
}
