#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/DenseRelations.h"
#include "Weft/Dialect/RISCV/IR/MemoryEffects.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/StrideABI.h"
#include "Verification.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace weft::riscv;

namespace {

bool addressable(Type type) {
  if (isa<LocalType, MemDescType>(type))
    return true;
  auto value = dyn_cast<ValueType>(type);
  return value && (value.getLayout().getCarrier() == "local" ||
                   isa<weft::kernel::EncodingType>(value.getElementType()));
}

void appendUnique(llvm::SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

// Both the initial edge and the back edge contribute to a loop carry's
// provenance. The visited set closes cycles without assuming loop invariance.
bool incomingStorage(Value value, llvm::SmallVectorImpl<Value> &pending) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    unsigned position = argument.getArgNumber();
    if (auto loop = dyn_cast<scf::ForOp>(owner)) {
      if (position == 0) return false;
      --position;
      pending.push_back(loop.getInitArgs()[position]);
      pending.push_back(loop.getBody()->getTerminator()->getOperand(position));
      return true;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
      if (argument.getOwner() == loop.getBeforeBody()) {
        pending.push_back(loop.getInits()[position]);
        pending.push_back(loop.getAfterBody()->getTerminator()->getOperand(position));
      } else if (argument.getOwner() == loop.getAfterBody()) {
        pending.push_back(loop.getConditionOp().getArgs()[position]);
      } else return false;
      return true;
    }
    if (auto loop = dyn_cast<LoopOp>(owner)) {
      if (position == 0) return false;
      --position;
      pending.push_back(loop.getCarried()[position]);
      pending.push_back(loop.getBody().front().getTerminator()->getOperand(position));
      return true;
    }
    return false;
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result) return false;
  Operation *owner = result.getOwner();
  unsigned position = result.getResultNumber();
  if (auto loop = dyn_cast<scf::ForOp>(owner)) {
    pending.push_back(loop.getInitArgs()[position]);
    pending.push_back(loop.getBody()->getTerminator()->getOperand(position));
  } else if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
    pending.push_back(loop.getConditionOp().getArgs()[position]);
  } else if (auto branch = dyn_cast<scf::IfOp>(owner)) {
    pending.push_back(branch.thenYield().getOperand(position));
    pending.push_back(branch.elseYield().getOperand(position));
  } else if (auto select = dyn_cast<arith::SelectOp>(owner)) {
    pending.push_back(select.getTrueValue());
    pending.push_back(select.getFalseValue());
  } else if (auto loop = dyn_cast<LoopOp>(owner)) {
    pending.push_back(loop.getCarried()[position]);
    pending.push_back(loop.getBody().front().getTerminator()->getOperand(position));
  } else return false;
  return true;
}

// These edges transport addresses, or project an encoded storage value. A
// numeric load/conversion is deliberately absent: it creates a new snapshot.
Value borrowedStorage(Value value) {
  if (auto op = value.getDefiningOp<PointerCastOp>()) return op.getInput();
  if (auto op = value.getDefiningOp<MemoryViewOp>()) return op.getBase();
  if (auto op = value.getDefiningOp<StagedViewOp>()) return op.getSource();
  if (auto op = value.getDefiningOp<FieldViewOp>()) return op.getField();
  if (auto op = value.getDefiningOp<SliceOp>()) return op.getBase();
  if (auto op = value.getDefiningOp<SubviewOp>()) return op.getBase();
  if (auto op = value.getDefiningOp<ReaxisOp>()) return op.getBase();
  if (auto op = value.getDefiningOp<LinearViewOp>()) return op.getBase();
  if (auto op = value.getDefiningOp<LocalViewOp>()) return op.getSource();
  if (auto op = value.getDefiningOp<LocalBindOp>()) return op.getStorage();
  if (auto op = value.getDefiningOp<LocalReuseOp>()) return op.getPrevious();
  if (auto op = value.getDefiningOp<EncodedLocalBindOp>()) return op.getStorage();
  if (auto op = value.getDefiningOp<EncodedLocalPackOp>()) return op.getStorage();
  if (auto op = value.getDefiningOp<DenseSnapshotOp>()) return op.getStorage();
  if (auto op = value.getDefiningOp<FieldOp>()) return op.getOwner();
  if (auto op = value.getDefiningOp<ExtractOp>()) return op.getInput();
  if (!addressable(value.getType())) return {};
  if (auto op = value.getDefiningOp<ReshapeOp>()) return op.getInput();
  if (auto op = value.getDefiningOp<UpdateOp>()) return op.getInput();
  if (auto op = value.getDefiningOp<LoadOp>()) {
    auto encoding = dyn_cast<weft::kernel::EncodingType>(
        op.getRegion().getType().getEncoding());
    if (encoding && encoding.getKind() != "dense")
      return op.getSnapshotStorage() ? Value(op.getSnapshotStorage()) : Value(op.getRegion());
  }
  if (auto op = value.getDefiningOp<ConvertLayoutOp>())
    if (op.getConversion().getEffect() == "pure" &&
        addressable(op.getInput().getType())) return op.getInput();
  if (auto op = value.getDefiningOp<NewOp>())
    if (op.getInitial()) return op.getInitial();
  return {};
}

bool privateResource(SideEffects::Resource *resource) {
  return resource == PrimitiveStorageResource::get() ||
         resource == weft::kernel::LocalShapeResource::get();
}

bool coordinate(Operation *operation) {
  return isa<RootPointOp, PhysicalPointOp, RecordCohortOp>(operation);
}

bool storageBarrier(Operation *operation) {
  return isa<LocalCapacityGuardOp, IndexMultipleGuardOp>(operation);
}

bool originsMayAlias(const StorageOrigins &a, const StorageOrigins &b) {
  if (a.unknown || b.unknown) return true;
  for (Value allocation : a.allocations)
    if (llvm::is_contained(b.allocations, allocation)) return true;
  for (Value left : a.arguments) {
    auto l = cast<MemDescType>(left.getType());
    auto la = cast<BlockArgument>(left);
    for (Value right : b.arguments) {
      auto ra = cast<BlockArgument>(right);
      // Alias-set numbers only have meaning inside their owning kernel.
      if (la.getOwner() != ra.getOwner() ||
          l.getAliasSet() == cast<MemDescType>(right.getType()).getAliasSet())
        return true;
    }
  }
  return false;
}

bool writesOrigins(Operation *operation, const StorageOrigins &origins) {
  if (coordinate(operation) || storageBarrier(operation)) return false;
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const auto &effect : instances) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect()) ||
          privateResource(effect.getResource())) continue;
      Value value = effect.getValue();
      if (!value || originsMayAlias(origins, storageOrigins(value))) return true;
    }
  } else if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
             !isMemoryEffectFree(operation)) {
    return true;
  }
  if (operation->getNumRegions()) {
    if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return true;
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (writesOrigins(&nested, origins)) return true;
  }
  return false;
}

bool onlyReads(Operation *operation, bool recursive) {
  if (coordinate(operation)) return true;
  if (storageBarrier(operation)) return false;
  if (operation->getNumRegions() && !recursive) return false;
  if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    if (llvm::any_of(effects, [](const auto &effect) {
          return !isa<MemoryEffects::Read>(effect.getEffect());
        })) return false;
  } else if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
             !isMemoryEffectFree(operation)) {
    return false;
  }
  if (operation->getNumRegions()) {
    if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return false;
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!onlyReads(&nested, true)) return false;
  }
  return true;
}

bool prefixWrites(Block &block, Operation *consumer,
                  const StorageOrigins &origins) {
  Operation *anchor = block.findAncestorOpInBlock(*consumer);
  if (!anchor) return true;
  for (Operation &operation : block) {
    if (&operation == anchor) break;
    if (writesOrigins(&operation, origins)) return true;
  }
  if (anchor == consumer) return false;
  if (isa<scf::ForOp, scf::WhileOp, LoopOp>(anchor))
    return writesOrigins(anchor, origins);
  if (auto branch = dyn_cast<scf::IfOp>(anchor)) {
    for (Region &region : branch->getRegions())
      for (Block &nested : region)
        if (nested.findAncestorOpInBlock(*consumer))
          return prefixWrites(nested, consumer, origins);
  }
  // Unknown control may execute a prefix repeatedly or have its own effects.
  return true;
}

} // namespace

bool weft::riscv::isAddressableDenseDescriptor(Value source) {
  for (unsigned depth = 0; depth < 32; ++depth) {
    if (source.getDefiningOp<MemoryViewOp>() || source.getDefiningOp<LocalViewOp>() ||
        source.getDefiningOp<FieldViewOp>() || source.getDefiningOp<LinearViewOp>() ||
        source.getDefiningOp<DenseSnapshotOp>())
      return true;
    if (auto slice = source.getDefiningOp<SliceOp>()) source = slice.getBase();
    else if (auto subview = source.getDefiningOp<SubviewOp>()) source = subview.getBase();
    else if (auto rename = source.getDefiningOp<ReaxisOp>()) source = rename.getBase();
    else if (auto stage = source.getDefiningOp<StagedViewOp>()) source = stage.getSource();
    else return false;
  }
  return false;
}

std::optional<int64_t> weft::riscv::contiguousLocalElements(Value value) {
  auto type = dyn_cast<MemDescType>(value.getType());
  auto encoding = type ? dyn_cast<kernel::EncodingType>(type.getEncoding()) : kernel::EncodingType();
  if (!type || !encoding || encoding.getKind() != "dense" || type.getShape().empty() ||
      type.getElements() != 1 || type.getInterleaveRows() ||
      !isAddressableDenseDescriptor(value) || !storageOrigins(value).isOwnedLocal()) return std::nullopt;
  int64_t count = 1;
  for (size_t axis = type.getShape().size(); axis-- > 0;) {
    int64_t extent = type.getShape()[axis];
    if (extent <= 0 || type.getStrides()[axis] != count ||
        count > std::numeric_limits<int64_t>::max() / extent) return std::nullopt;
    count *= extent;
  }
  return count;
}

LogicalResult weft::riscv::LinearViewOp::verify() {
  auto source = getBase().getType(), result = getResult().getType();
  auto count = contiguousLocalElements(getBase());
  const size_t rank = result.getShape().size();
  if (!count || rank == 0 || rank >= source.getShape().size())
    return emitOpError("linear view must merge a nonempty contiguous suffix of private memory");
  int64_t suffix = 1;
  for (int64_t extent : source.getShape().asArrayRef().drop_front(rank - 1)) suffix *= extent;
  SmallVector<int64_t> shape(source.getShape().asArrayRef().take_front(rank - 1));
  shape.push_back(suffix);
  SmallVector<int64_t> strides(rank, 1), origins(rank, 0);
  for (size_t i = rank - 1; i > 0; --i) strides[i - 1] = strides[i] * shape[i];
  if (!count || result.getEncoding() != source.getEncoding() ||
      result.getShape().asArrayRef() != ArrayRef<int64_t>(shape) ||
      result.getAxisIds().asArrayRef() != source.getAxisIds().asArrayRef().take_front(rank) ||
      result.getStrides().asArrayRef() != ArrayRef<int64_t>(strides) ||
      result.getOrigins().asArrayRef() != ArrayRef<int64_t>(origins) ||
      result.getAlignment() != source.getAlignment() || result.getAddressClass() != "slice" ||
      result.getAccess() != source.getAccess() || result.getAliasSet() != source.getAliasSet() ||
      result.getLayoutIdentity() != source.getLayoutIdentity() || result.getElements() != 1 ||
      result.getStorageBits() != source.getStorageBits() || result.getInterleaveRows())
    return emitOpError("linear view requires an unchanged contiguous private byte range and a complete linear descriptor");
  return success();
}

LogicalResult weft::riscv::ReloadSliceOp::verify() {
  auto source = dyn_cast<ValueType>(getSourceType());
  auto result = getResult().getType();
  auto slot = getSlot().getType();
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto before = source ? RegisterCoordinates::get(source.getLayout()) : std::nullopt;
  auto after = RegisterCoordinates::get(result.getLayout());
  if (!source || !kernel || !before || !after || source.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "rvv" || source.getElementType() != result.getElementType() ||
      source.getLayout().getSew() != result.getLayout().getSew() || source.getLayout().getSew() < 8 ||
      slot.getSchema() != "rvv-register-spill" || slot.getElementType() != source.getElementType() ||
      slot.getSizeBytes() <= 0 || slot.getSizeBytes() % before->parts() ||
      getSourceParts().size() != static_cast<size_t>(after->parts()) ||
      getLaneOffsets().size() != getSourceParts().size() || after->lanes() > 4096 ||
      !supportsRVVLayout(kernel.getTarget(), result.getLayout()) ||
      !detail::exactLeaf(getLeaf(), "transfer", "reload-slice", "rvv.reload-slice", "none", "exact") ||
      getLeaf().getTemporaryGroups() || getLeaf().getFragmentGroups() || getLeaf().getLocalBytes())
    return emitOpError("reload slice requires a typed RVV spill slot and exact bounded projection");
  SpillOp writer;
  for (Operation *user : getSlot().getUsers()) {
    if (auto spill = dyn_cast<SpillOp>(user)) {
      auto stored = dyn_cast<ValueType>(spill.getInput().getType());
      if (writer || !stored || slot.getShape() != stored.getShape() || slot.getAxisIds() != stored.getAxisIds() ||
          (stored != source && !hasIdenticalLinearCarrier(stored, source)))
        return emitOpError("reload slice requires one immutable matching spill");
      writer = spill;
    } else if (!isa<ReloadOp, ReloadSliceOp>(user)) {
      return emitOpError("reload slice slot has an unproven storage observer");
    }
  }
  DominanceInfo dominance(kernel);
  if (!writer || !dominance.properlyDominates(writer.getOperation(), getOperation()))
    return emitOpError("reload slice must follow its unique spill on every path");
  const int64_t partBytes = slot.getSizeBytes() / before->parts();
  for (auto [part, sourcePart] : llvm::enumerate(getSourceParts())) {
    int64_t offset = getLaneOffsets()[part];
    if (sourcePart < 0 || sourcePart >= before->parts() || offset < 0 ||
        offset > before->lanes() - after->lanes() ||
        (offset + after->lanes()) * (source.getLayout().getSew() / 8) > partBytes)
      return emitOpError("reload slice exceeds its selected source part");
    auto base = before->logical(sourcePart, offset);
    auto origin = after->logical(part, 0);
    if (!base || !origin) return emitOpError("reload slice has no source coordinates");
    for (int64_t axis : result.getAxisIds().asArrayRef())
      if (!llvm::is_contained(source.getAxisIds().asArrayRef(), axis))
        return emitOpError("reload slice introduces a logical axis");
    for (int64_t lane = 0; lane < after->lanes(); ++lane) {
      auto actual = before->logical(sourcePart, offset + lane);
      auto expected = after->logical(part, lane);
      if (!actual || !expected) return emitOpError("reload slice has incomplete lane coordinates");
      for (auto [position, axis] : llvm::enumerate(source.getAxisIds().asArrayRef())) {
        auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
        int64_t delta = found == result.getAxisIds().asArrayRef().end() ? 0 :
            (*expected)[found - result.getAxisIds().asArrayRef().begin()] -
            (*origin)[found - result.getAxisIds().asArrayRef().begin()];
        if ((*actual)[position] - (*base)[position] != delta)
          return emitOpError("reload slice changes the selected lane order");
      }
    }
  }
  return detail::verifyLeafOperation(*this);
}

StorageOrigins weft::riscv::storageOrigins(Value root) {
  StorageOrigins result;
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> visited;
  bool sawAddress = false;
  for (size_t i = 0; i < pending.size(); ++i) {
    Value value = pending[i];
    if (!value) { result.unknown = true; break; }
    if (!visited.insert(value).second) continue;
    if (visited.size() > 512) { result.unknown = true; break; }
    if (auto allocation = value.getDefiningOp<LocalAllocOp>()) {
      appendUnique(result.allocations, value);
      continue;
    }
    if (Value source = borrowedStorage(value)) {
      pending.push_back(source);
      continue;
    }
    if (!addressable(value.getType())) continue;
    sawAddress = true;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (isa<KernelOp, ArtifactPackOp>(argument.getOwner()->getParentOp()) &&
          isa<MemDescType>(value.getType())) {
        appendUnique(result.arguments, value);
        continue;
      }
    }
    if (!incomingStorage(value, pending)) result.unknown = true;
  }
  // A cycle with no allocation/entry source is not a proof of empty storage.
  if (sawAddress && result.arguments.empty() && result.allocations.empty())
    result.unknown = true;
  return result;
}

bool weft::riscv::hasDynamicDenseStride(MemDescType memory, Type value) {
  auto type = dyn_cast<ValueType>(value);
  auto encoding = dyn_cast<weft::kernel::EncodingType>(memory.getEncoding());
  if (!type || !encoding || encoding.getKind() != "dense" ||
      type.getLayout().getCarrier() != "rvv" || type.getLayout().getReplicaGroup() != 1 ||
      memory.getStorageBits() != type.getLayout().getSew()) return false;
  int64_t laneAxis = 0;
  for (auto [axis, lanes] : llvm::zip(type.getAxisIds().asArrayRef(), type.getLayout().getLaneFactors().asArrayRef())) {
    if (lanes <= 1) continue;
    if (laneAxis) return false;
    laneAxis = axis;
  }
  auto found = llvm::find(memory.getAxisIds().asArrayRef(), laneAxis);
  return found != memory.getAxisIds().asArrayRef().end() &&
      memory.getStrides()[found - memory.getAxisIds().asArrayRef().begin()] == -1;
}

bool weft::riscv::isInnermostDenseAxis(Value memory, int64_t axis) {
  for (unsigned depth = 0; depth < 64; ++depth) {
    auto type = dyn_cast<MemDescType>(memory.getType());
    if (!type) return false;
    auto found = llvm::find(type.getAxisIds().asArrayRef(), axis);
    if (found == type.getAxisIds().asArrayRef().end()) return false;
    size_t position = found - type.getAxisIds().asArrayRef().begin();
    for (size_t after = position + 1; after < type.getShape().size(); ++after)
      if (type.getShape()[after] != 1) return false;
    if (auto slice = memory.getDefiningOp<SliceOp>()) memory = slice.getBase();
    else if (auto subview = memory.getDefiningOp<SubviewOp>()) memory = subview.getBase();
    else if (auto rename = memory.getDefiningOp<ReaxisOp>()) {
      axis = rename.getBase().getType().getAxisIds()[position];
      memory = rename.getBase();
    } else return isa<BlockArgument>(memory) || memory.getDefiningOp<MemoryViewOp>() ||
        memory.getDefiningOp<LocalViewOp>();
  }
  return false;
}

std::optional<std::pair<Value, int64_t>> weft::riscv::denseArgumentStride(Value memory, int64_t axis) {
  // Slice/subview preserve retained strides; reaxis transports the same axis
  // by position. A logical rename must not lose an ABI stride proof.
  for (unsigned depth = 0; depth < 64; ++depth) {
    auto type = dyn_cast<MemDescType>(memory.getType());
    if (!type) return std::nullopt;
    auto encoding = dyn_cast<weft::kernel::EncodingType>(type.getEncoding());
    auto position = llvm::find(type.getAxisIds().asArrayRef(), axis);
    if (!encoding || encoding.getKind() != "dense" ||
        position == type.getAxisIds().asArrayRef().end()) return std::nullopt;
    int64_t index = position - type.getAxisIds().asArrayRef().begin();
    if (type.getStrides()[index] != -1) return std::nullopt;
    if (auto argument = dyn_cast<BlockArgument>(memory)) {
      auto kernel = dyn_cast<KernelOp>(argument.getOwner()->getParentOp());
      if (!kernel || argument.getOwner() != &kernel.getBody().front() ||
          !weft::hasStrideArguments(kernel.getStridedArgsAttr(), kernel.getArgNames()[argument.getArgNumber()]))
        return std::nullopt;
      return std::pair<Value, int64_t>{memory, index};
    }
    if (auto slice = memory.getDefiningOp<SliceOp>()) memory = slice.getBase();
    else if (auto view = memory.getDefiningOp<SubviewOp>()) memory = view.getBase();
    else if (auto rename = memory.getDefiningOp<ReaxisOp>()) {
      auto source = rename.getBase().getType();
      if (source.getShape() != type.getShape() || source.getStrides() != type.getStrides() ||
          source.getAxisIds().size() != type.getAxisIds().size()) return std::nullopt;
      axis = source.getAxisIds()[index];
      memory = rename.getBase();
    }
    else if (auto view = memory.getDefiningOp<MemoryViewOp>()) {
      auto stride = view.getStrides()[index].getDefiningOp<StrideOp>();
      if (!stride || stride.getView() != view.getBase() || stride.getAxis() != index)
        return std::nullopt;
      memory = view.getBase();
    }
    else return std::nullopt;
  }
  return std::nullopt;
}

bool weft::riscv::hasDenseStrideProof(Operation *operation, Value memory,
                                     int64_t axis, int64_t expected) {
  auto type = dyn_cast<MemDescType>(memory.getType());
  if (!type) return false;
  auto position = llvm::find(type.getAxisIds().asArrayRef(), axis);
  if (position == type.getAxisIds().asArrayRef().end()) return false;
  auto declared = type.getStrides()[position - type.getAxisIds().asArrayRef().begin()];
  if (declared != -1) return declared == expected;
  auto source = denseArgumentStride(memory, axis);
  if (!source) return false;
  for (Operation *child = operation, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    if (!branch || child->getParentRegion() != &branch.getThenRegion()) continue;
    SmallVector<Value> pending{branch.getCondition()};
    for (unsigned visited = 0; !pending.empty() && visited < 64; ++visited) {
      Value condition = pending.pop_back_val();
      if (auto both = condition.getDefiningOp<arith::AndIOp>()) {
        pending.append({both.getLhs(), both.getRhs()});
        continue;
      }
      auto compare = condition.getDefiningOp<arith::CmpIOp>();
      if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq) continue;
      for (unsigned side = 0; side < 2; ++side) {
        auto stride = compare->getOperand(side).getDefiningOp<StrideOp>();
        auto unit = getConstantIntValue(compare->getOperand(1 - side));
        if (stride && stride.getView() == source->first && stride.getAxis() == source->second &&
            unit && *unit == expected) return true;
      }
    }
  }
  return false;
}

bool weft::riscv::hasUnitStrideProof(Operation *operation, Value memory, int64_t axis) {
  return hasDenseStrideProof(operation, memory, axis, 1);
}

bool weft::riscv::isStorageMetadataUse(OpOperand &use) {
  // Extents belong to the immutable descriptor, not its current byte occupant.
  return isa<ExtentOp>(use.getOwner()) && use.getOperandNumber() == 0;
}

bool weft::riscv::mayAliasStorage(Value lhs, Value rhs) {
  return originsMayAlias(storageOrigins(lhs), storageOrigins(rhs));
}

Value weft::riscv::fullDenseArgumentView(Value storage) {
  auto origins = storageOrigins(storage);
  if (origins.unknown || !origins.allocations.empty() || origins.arguments.size() != 1) return {};
  Value argument = origins.arguments.front();
  auto type = cast<MemDescType>(argument.getType());
  auto encoding = dyn_cast<weft::kernel::EncodingType>(type.getEncoding());
  if (!encoding || encoding.getKind() != "dense" || type.getStorageBits() < 8 ||
      type.getStorageBits() % 8 || type.getShape().size() > 8) return {};
  auto kernel = dyn_cast<KernelOp>(cast<BlockArgument>(argument).getOwner()->getParentOp());
  if (!kernel) return {};
  MemoryViewOp full;
  for (Operation *user : argument.getUsers()) {
    auto view = dyn_cast<MemoryViewOp>(user);
    if (!view || view.getBase() != argument || view->getBlock() != argument.getParentBlock()) continue;
    // A descriptor prefix or a second physical view of the same ABI argument
    // cannot serve as proof of that argument's complete byte footprint.
    if (full) return {};
    if (!llvm::all_of(view.getOrigins(), [](Value origin) {
          auto value = getConstantIntValue(origin);
          return value && *value == 0;
        })) return {};
    for (auto [extent, expected] : llvm::zip(view.getExtents(), type.getShape().asArrayRef())) {
      if (expected > 0) {
        auto constant = getConstantIntValue(extent);
        if (!constant || *constant != expected) return {};
      } else {
        auto symbol = extent.getDefiningOp<SymbolOp>();
        if (expected >= 0 || -expected > int64_t(kernel.getShapeSymbols().size()) || !symbol ||
            symbol.getNameAttr() != kernel.getShapeSymbols()[-expected - 1]) return {};
      }
    }
    full = view;
  }
  return full ? full.getResult() : Value();
}

LogicalResult DenseDisjointOp::verify() {
  for (Value value : {getLhs(), getRhs()}) {
    auto view = value.getDefiningOp<MemoryViewOp>();
    auto argument = view ? dyn_cast<BlockArgument>(view.getBase()) : BlockArgument();
    auto kernel = argument ? dyn_cast<KernelOp>(argument.getOwner()->getParentOp()) : KernelOp();
    if (!kernel || argument.getOwner() != &kernel.getBody().front() ||
        view->getBlock() != argument.getOwner() || value != fullDenseArgumentView(value))
      return emitOpError("requires complete pinned dense views with explicit ABI metadata");
    if (kernel != (*this)->getParentOfType<KernelOp>())
      return emitOpError("dense ranges must belong to the containing kernel");
  }
  return success();
}

LogicalResult DenseElementwiseAliasOp::verify() {
  for (Value value : {getLhs(), getRhs()}) {
    if (value != fullDenseArgumentView(value))
      return emitOpError("requires full dense ABI views with zero origins");
    auto view = value.getDefiningOp<MemoryViewOp>();
    if (!view || view->getParentOfType<KernelOp>() != (*this)->getParentOfType<KernelOp>())
      return emitOpError("requires views of the containing kernel");
  }
  auto lhs = getLhs().getType(), rhs = getRhs().getType();
  if (lhs.getEncoding() != rhs.getEncoding() || lhs.getShape().size() != rhs.getShape().size())
    return emitOpError("requires equal element storage and ranks");
  return success();
}

bool weft::riscv::mayAliasStorageAt(Value lhs, Value rhs, Operation *at) {
  auto a = storageOrigins(lhs), b = storageOrigins(rhs);
  if (!originsMayAlias(a, b)) return false;
  if (!at || a.unknown || b.unknown) return true;
  if (haveDisjointDenseProjectionsAt(lhs, rhs, at)) return false;
  for (Value allocation : a.allocations)
    if (llvm::is_contained(b.allocations, allocation)) return true;
  SmallVector<std::pair<Value, Value>> proven;
  for (Operation *child = at, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    if (!branch || child->getParentRegion() != &branch.getThenRegion()) continue;
    SmallVector<Value> pending{branch.getCondition()};
    for (unsigned visited = 0; !pending.empty() && visited < 64; ++visited) {
      Value condition = pending.pop_back_val();
      if (auto both = condition.getDefiningOp<arith::AndIOp>()) {
        pending.append({both.getLhs(), both.getRhs()});
      } else if (auto disjoint = condition.getDefiningOp<DenseDisjointOp>()) {
        auto left = disjoint.getLhs().getDefiningOp<MemoryViewOp>();
        auto right = disjoint.getRhs().getDefiningOp<MemoryViewOp>();
        if (left && right) proven.emplace_back(left.getBase(), right.getBase());
      }
    }
  }
  for (Value left : a.arguments)
    for (Value right : b.arguments) {
      auto l = cast<BlockArgument>(left), r = cast<BlockArgument>(right);
      if (l.getOwner() != r.getOwner()) return true;
      if (cast<MemDescType>(left.getType()).getAliasSet() != cast<MemDescType>(right.getType()).getAliasSet()) continue;
      if (!llvm::any_of(proven, [&](auto pair) {
            return (pair.first == left && pair.second == right) ||
                   (pair.first == right && pair.second == left);
          })) return true;
    }
  return false;
}

bool weft::riscv::haveDisjointLocalStorage(Value lhs, Value rhs) {
  auto a = storageOrigins(lhs), b = storageOrigins(rhs);
  return a.isOwnedLocal() && b.isOwnedLocal() && !originsMayAlias(a, b);
}

void weft::riscv::appendStorageAliases(OpOperand &use,
                                     SmallVectorImpl<Value> &aliases) {
  Operation *user = use.getOwner();
  auto append = [&](Value value) {
    if (addressable(value.getType())) appendUnique(aliases, value);
  };
  if (auto loop = dyn_cast<scf::ForOp>(user)) {
    unsigned start = loop.getInitArgs().getBeginOperandIndex();
    if (use.getOperandNumber() >= start) {
      unsigned index = use.getOperandNumber() - start;
      append(loop.getRegionIterArgs()[index]);
      append(loop.getResult(index));
    }
  } else if (auto loop = dyn_cast<scf::WhileOp>(user)) {
    append(loop.getBeforeBody()->getArgument(use.getOperandNumber()));
  } else if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
    if (use.getOperandNumber()) {
      auto loop = cast<scf::WhileOp>(condition->getParentOp());
      unsigned index = use.getOperandNumber() - 1;
      append(loop.getAfterBody()->getArgument(index));
      append(loop.getResult(index));
    }
  } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
    auto *parent = yield->getParentOp();
    unsigned index = use.getOperandNumber();
    if (auto loop = dyn_cast<scf::WhileOp>(parent))
      append(loop.getBeforeBody()->getArgument(index));
    else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      append(loop.getRegionIterArgs()[index]);
      append(loop.getResult(index));
    } else if (isa<scf::IfOp>(parent)) append(parent->getResult(index));
  } else if (auto loop = dyn_cast<LoopOp>(user)) {
    unsigned start = loop.getCarried().getBeginOperandIndex();
    if (use.getOperandNumber() >= start) {
      unsigned index = use.getOperandNumber() - start;
      append(loop.getBody().front().getArgument(index + 1));
      append(loop.getResult(index));
    }
  } else if (auto yield = dyn_cast<YieldOp>(user)) {
    if (auto loop = dyn_cast<LoopOp>(yield->getParentOp())) {
      unsigned index = use.getOperandNumber();
      append(loop.getBody().front().getArgument(index + 1));
      append(loop.getResult(index));
    }
  } else {
    for (Value result : user->getResults()) {
      if (borrowedStorage(result) == use.get()) append(result);
      else if (isa<arith::SelectOp>(user) && use.getOperandNumber() > 0)
        append(result);
      else if (addressable(result.getType()) &&
               mayAliasStorage(result, use.get())) append(result);
    }
  }
}

bool weft::riscv::storageAliasesRemainInScope(Value root, Block *scope) {
  auto whileScope = dyn_cast<scf::WhileOp>(scope->getParentOp());
  if (whileScope && scope != whileScope.getBeforeBody()) whileScope = {};
  auto inside = [&](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument && argument.getOwner() == scope) return true;
    Operation *owner = argument ? argument.getOwner()->getParentOp() : value.getDefiningOp();
    if (whileScope) {
      if (argument && argument.getOwner() == whileScope.getAfterBody()) return true;
      return owner != whileScope && whileScope->isAncestor(owner);
    }
    return owner && scope->findAncestorOpInBlock(*owner);
  };
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> visited;
  for (size_t i = 0; i < pending.size(); ++i) {
    Value value = pending[i];
    if (!visited.insert(value).second || value.use_empty()) continue;
    if (!inside(value)) return false;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      const bool control = isa<scf::ForOp, scf::WhileOp, scf::IfOp,
          scf::ConditionOp, scf::YieldOp, LoopOp, YieldOp>(user);
      if (!control && (user->getNumRegions() ||
          (!isMemoryEffectFree(user) && !isa<MemoryEffectOpInterface>(user))))
        return false;
      appendStorageAliases(use, pending);
    }
  }
  return true;
}

bool weft::riscv::hasOnlyReadEffects(Operation *operation, bool recursive) {
  return onlyReads(operation, recursive);
}

bool weft::riscv::mayWriteStorage(Operation *operation, Value storage) {
  return writesOrigins(operation, storageOrigins(storage));
}

bool weft::riscv::storageWrittenBetween(Value storage, Operation *read,
                                      Operation *consumer) {
  if (read == consumer) return false;
  Block *block = read->getBlock();
  Operation *anchor = block->findAncestorOpInBlock(*consumer);
  if (!anchor || anchor == read || !read->isBeforeInBlock(anchor)) return true;
  auto origins = storageOrigins(storage);
  // Structural region traversal is finite, including the body of a loop.
  // Its instruction count is not a write to an unrelated private snapshot.
  for (Operation *cursor = read->getNextNode(); cursor != anchor;
       cursor = cursor->getNextNode())
    if (writesOrigins(cursor, origins)) return true;
  if (anchor == consumer) return false;
  if (isa<scf::ForOp, scf::WhileOp, LoopOp>(anchor))
    return writesOrigins(anchor, origins);
  if (auto branch = dyn_cast<scf::IfOp>(anchor))
    for (Region &region : branch->getRegions())
      for (Block &nested : region)
        if (nested.findAncestorOpInBlock(*consumer))
          return prefixWrites(nested, consumer, origins);
  return true;
}

bool weft::riscv::readCrossesWrite(Operation *read, Operation *consumer) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(read);
  if (!interface) return true;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  for (const auto &effect : effects) {
    if (privateResource(effect.getResource()) ||
        !isa<MemoryEffects::Read>(effect.getEffect())) continue;
    if (!effect.getValue() || storageWrittenBetween(effect.getValue(), read, consumer))
      return true;
  }
  return false;
}

LogicalResult weft::riscv::verifyMemoryEffects(Operation *operation) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface) return success();
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  for (const auto &effect : effects) {
    // Exiting a helper is a control effect, not an access to an unnamed
    // storage object. It remains a barrier to speculative reads/motion.
    if (effect.getResource() == weft::kernel::FunctionControlResource::get()) continue;
    if (privateResource(effect.getResource())) continue;
    Value value = effect.getValue();
    if (!value)
      return operation->emitOpError("memory effect must name its storage operand or result");
    if (!llvm::is_contained(operation->getOperands(), value) &&
        !llvm::is_contained(operation->getResults(), value))
      return operation->emitOpError("memory effect must refer to this operation's SSA interface");
  }
  return success();
}
