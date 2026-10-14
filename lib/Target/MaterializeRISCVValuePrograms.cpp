#include "Weft/Target/RISCVPasses.h"
#include "RISCVPhysicalSupport.h"
#include "LocalMemory.h"
#include "DenseValuePrograms.h"
#include "DenseIssue.h"
#include "LocalLifetime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>
#include <limits>

using namespace mlir;
using namespace weft;

namespace {

bool pointwise(Operation *op) {
  return isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
             riscv::SelectOp, riscv::CastOp, riscv::BitcastOp,
             riscv::NarrowOp, riscv::WidenOp>(op);
}

class ValuePrograms {
public:
  ValuePrograms(ModuleOp module) : module(module), b(module.getContext()) {}

  LogicalResult run() {
    if (failed(riscv_internal::materializeDenseValuePrograms(module))) return failure();
    nextIdentity = riscv_internal::lastLocalIdentity(module);
    SmallVector<Operation *> operations;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) { operations.push_back(op); });
    collectPointwisePrograms(operations);
    for (Operation *op : operations) {
      if (delayed.contains(op)) continue;
      if (auto reshape = dyn_cast<riscv::ReshapeOp>(op)) {
        Value source = local(reshape.getInput());
        if (source && llvm::equal(reshape.getOrder(),
                                 cast<riscv::ValueType>(source.getType()).getAxisIds().asArrayRef())) {
          auto old = reshape.getResult().getType();
          b.setInsertionPoint(reshape);
          auto replacement = b.create<riscv::ReshapeOp>(reshape.getLoc(),
              localType(old), source, reshape.getOrderAttr());
          riscv_internal::copyOrigin(reshape, replacement);
          replace(reshape, replacement);
        }
        continue;
      }
      if (auto reduce = dyn_cast<riscv::ReduceOp>(op)) {
        if (Value source = local(reduce.getInput()); source &&
            (cast<riscv::ValueType>(source.getType()).getElementType().isF32() ||
             cast<riscv::ValueType>(source.getType()).getElementType().isF64()))
          if (failed(riscv_internal::materializeDenseValueProgram(reduce))) return failure();
        continue;
      }
      if (auto extract = dyn_cast<riscv::ExtractOp>(op)) {
        if (Value source = local(extract.getInput()); source && needsProgram(op))
          if (failed(extraction(extract, source))) return failure();
        continue;
      }
      if (auto projection = dyn_cast<riscv::ProjectReductionOperandOp>(op)) {
        if (Value source = local(projection.getInput())) {
          b.setInsertionPoint(op);
          auto memory = view(source, op);
          if (failed(memory)) return failure();
          auto full = cast<riscv::MemDescType>(memory->getType());
          SmallVector<Attribute> selectors;
          SmallVector<int64_t> strides, origins;
          for (auto [position, axis] : llvm::enumerate(full.getAxisIds().asArrayRef())) {
            bool reduced = axis == projection.getReductionAxis();
            selectors.push_back(b.getStringAttr(reduced ? "index" : "all"));
            if (!reduced) { strides.push_back(full.getStrides()[position]); origins.push_back(full.getOrigins()[position]); }
          }
          auto result = projection.getResult().getType();
          auto sliced = riscv::MemDescType::get(b.getContext(), full.getEncoding(), result.getShape(),
              result.getAxisIds(), b.getDenseI64ArrayAttr(strides), b.getDenseI64ArrayAttr(origins),
              full.getAlignment(), "slice", full.getAccess(), full.getAliasSet(), full.getLayoutIdentity(),
              full.getStorageBits(), full.getElements(), full.getInterleaveRows());
          auto slice = b.create<riscv::SliceOp>(op->getLoc(), sliced, *memory,
              ValueRange{projection.getReductionIndex()}, b.getArrayAttr(selectors));
          auto loaded = b.create<riscv::LoadOp>(op->getLoc(), result, slice, Value(),
              riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
          riscv_internal::copyOrigin(op, loaded);
          b.replaceOp(op, loaded.getResult());
        }
        continue;
      }
      if (auto store = dyn_cast<riscv::StoreOp>(op)) {
        auto valueType = dyn_cast<riscv::ValueType>(store.getValue().getType());
        if (Value source = local(store.getValue()); source && valueType &&
            valueType.getShape() == store.getRegion().getType().getShape() &&
            valueType.getAxisIds() == store.getRegion().getType().getAxisIds())
          if (failed(riscv_internal::materializeDenseValueProgram(store))) return failure();
        continue;
      }
      if (op->getNumResults() != 1 ||
          !isa<riscv::ValueType>(op->getResult(0).getType())) continue;
      auto type = cast<riscv::ValueType>(op->getResult(0).getType());
      if (!isa<IntegerType, FloatType>(type.getElementType())) continue;
      if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op)) {
        if (Value source = local(conversion.getInput())) {
          // A local read is a snapshot at this exact boundary. Pure register
          // rearrangements of a snapshot can use its existing local carrier.
          if (conversion.getConversion().getEffect() == "pure") {
            b.setInsertionPoint(conversion);
            replace(conversion, source);
          } else if (needsProgram(op) || type.getLayout().getCarrier() == "local") {
            b.setInsertionPoint(op);
            auto copied = copyLocal(source, type, op);
            if (failed(copied)) return failure();
            replace(op, *copied);
          }
        }
        continue;
      }
      if (!needsProgram(op)) continue;
      b.setInsertionPoint(op);
      if (auto materialize = dyn_cast<riscv::RegisterMaterializeOp>(op)) {
        auto memory = sourceView(materialize.getInput(), op);
        if (failed(memory)) return failure();
        auto copied = copyView(*memory, type, op);
        if (failed(copied)) return failure();
        replace(op, *copied);
        continue;
      }
      if (auto load = dyn_cast<riscv::LoadOp>(op)) {
        auto memory = load.getRegion().getType();
        auto encoding = dyn_cast<kernel::EncodingType>(memory.getEncoding());
        bool completeLocal = bool(localMemory(load.getRegion()));
        bool completeDynamic = llvm::any_of(type.getShape().asArrayRef(), [](int64_t n) { return n < 0; }) &&
            llvm::none_of(type.getAxisIds().asArrayRef(), [&](int64_t axis) {
              return bool(riscv_internal::originPoint(load.getRegion(), axis));
            });
        if (!encoding || encoding.getKind() != "dense" || load.getSnapshotStorage() ||
            memory.getShape() != type.getShape() || memory.getAxisIds() != type.getAxisIds() ||
            (!completeLocal && !completeDynamic && type.getLayout().getValidity() != "full" &&
             !completePhysicalDomain(load.getRegion(), type)) ||
            (llvm::any_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent < 0; }) &&
             !completeLocal && !completeDynamic))
          continue;
        auto copied = copyView(load.getRegion(), type, op);
        if (failed(copied)) return failure();
        replace(op, *copied);
      } else if (pointwise(op)) {
        if (failed(riscv_internal::materializeDenseValueProgram(op))) return failure();
      }
    }
    module.walk([](riscv::IotaOp iota) {
      if (iota.getResult().use_empty()) iota.erase();
    });
    return riscv_internal::forwardLocalCopies(module, b);
  }

private:
  ModuleOp module;
  IRRewriter b;
  DenseMap<Value, Value> numericStorage;
  llvm::SmallPtrSet<Operation *, 32> delayed;
  int64_t nextIdentity = 0;

  // Local numerical state can enter a block through a control argument, so it
  // need not have an in-block load producer. Once the complete source version
  // exists, a chain of pointwise consumers still needs only one bounded walk.
  // Keep its typed operations and rounding boundaries; only their temporary
  // representation changes from whole arrays to one issue's values.
  void collectPointwisePrograms(ArrayRef<Operation *> operations) {
    for (Operation *root : llvm::reverse(operations)) {
      if (delayed.contains(root) || !pointwise(root) || !needsProgram(root)) continue;
      auto domain = dyn_cast<riscv::ValueType>(root->getResult(0).getType());
      if (!domain || (!domain.getElementType().isF32() && !domain.getElementType().isF64()) ||
          domain.getShape().empty()) continue;
      SmallVector<Operation *> order;
      llvm::SmallPtrSet<Operation *, 32> members;
      std::function<void(Operation *)> collect = [&](Operation *op) {
        if (!members.insert(op).second) return;
        for (Value value : op->getOperands()) {
          Operation *producer = value.getDefiningOp();
          auto type = dyn_cast<riscv::ValueType>(value.getType());
          if (!producer || !type || producer->getBlock() != root->getBlock() ||
              !pointwise(producer) || delayed.contains(producer) || members.size() >= 64 ||
              type.getElementType() != domain.getElementType() ||
              type.getShape() != domain.getShape() || type.getAxisIds() != domain.getAxisIds()) continue;
          // All uses must have one consumer. Repeated operands of that same
          // consumer are fine; a shared result observed elsewhere stays a
          // complete snapshot, and is not recomputed in this program.
          if (llvm::any_of(value.getUsers(), [&](Operation *user) { return user != op; })) continue;
          bool movable = true;
          unsigned distance = 0;
          for (Operation *next = producer->getNextNode(); next != root; next = next->getNextNode()) {
            if (!next || ++distance > 256 || next->getNumRegions() || !riscv::hasOnlyReadEffects(next)) {
              movable = false;
              break;
            }
          }
          if (movable) collect(producer);
        }
        order.push_back(op);
      };
      collect(root);
      if (order.size() < 2) continue;
      for (Operation *op : order) if (op != root) delayed.insert(op);
    }
  }

  bool completePhysicalDomain(Value source, riscv::ValueType type) const {
    for (auto [extent, axis] : llvm::zip(type.getShape().asArrayRef(), type.getAxisIds().asArrayRef())) {
      auto point = riscv_internal::originPoint(source, axis);
      auto partition = point ? point.getPartition().getDefiningOp<arith::ConstantIndexOp>() : arith::ConstantIndexOp();
      auto active = riscv_internal::completePointExtent(point);
      if (extent <= 0 || !partition || partition.value() != extent || !active || *active != extent) return false;
    }
    return true;
  }

  Value local(Value value) const {
    if (auto type = dyn_cast<riscv::ValueType>(value.getType());
        type && type.getLayout().getCarrier() == "local") return value;
    if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>();
        conversion && riscv_internal::isCompleteLocalRead(conversion))
      return conversion.getInput();
    return {};
  }

  Value localMemory(Value descriptor) const {
    // Only addressable projections preserve the actual complete descriptor
    // domain. A domain/point projection can denote one active physical issue.
    while (descriptor) {
      if (auto view = descriptor.getDefiningOp<riscv::LocalViewOp>())
        return view.getSource();
      if (auto subview = descriptor.getDefiningOp<riscv::SubviewOp>()) {
        descriptor = subview.getBase();
      } else if (auto reaxis = descriptor.getDefiningOp<riscv::ReaxisOp>()) {
        descriptor = reaxis.getBase();
      } else if (auto slice = descriptor.getDefiningOp<riscv::SliceOp>()) {
        for (Attribute selector : slice.getSelectors()) {
          auto kind = cast<StringAttr>(selector).getValue();
          if (kind != "index" && kind != "all") return {};
        }
        descriptor = slice.getBase();
      } else return {};
    }
    return {};
  }

  bool exceedsRegisterBudget(Operation *op, int64_t budget) const {
    int64_t groups = 0;
    llvm::SmallPtrSet<void *, 8> seen;
    auto account = [&](Value value) {
      if (!isa<riscv::ValueType>(value.getType())) return;
      Value identity = riscv_internal::registerStorageIdentity(value);
      if (!seen.insert(identity.getAsOpaquePointer()).second) return;
      auto count = riscv_internal::residentRegisterGroups(identity);
      if (!count || *count > std::numeric_limits<int64_t>::max() - groups)
        groups = std::numeric_limits<int64_t>::max();
      else groups += *count;
    };
    for (Value value : op->getOperands()) account(value);
    for (Value value : op->getResults()) account(value);
    if (auto leaf = op->getAttrOfType<riscv::LeafAttr>("leaf"))
      for (int64_t temporary : {leaf.getTemporaryGroups(), leaf.getFragmentGroups()}) {
        if (temporary > std::numeric_limits<int64_t>::max() - groups)
          groups = std::numeric_limits<int64_t>::max();
        else groups += temporary;
      }
    return groups > budget;
  }

  bool consumerExceedsRegisterBudget(Value source, int64_t budget) const {
    SmallVector<Value> pending{source};
    llvm::SmallPtrSet<void *, 8> seen;
    Value identity = riscv_internal::registerStorageIdentity(source);
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!seen.insert(value.getAsOpaquePointer()).second) continue;
      for (Operation *user : value.getUsers()) {
        // Region operands/results are not one simultaneous numeric issue.
        // Keep the existing region-aware resource owner responsible for them.
        if (user->getNumRegions()) continue;
        if (exceedsRegisterBudget(user, budget)) return true;
        for (Value result : user->getResults())
          if (isa<riscv::ValueType>(result.getType()) &&
              riscv_internal::registerStorageIdentity(result) == identity)
            pending.push_back(result);
      }
    }
    return false;
  }

  bool needsProgram(Operation *op) const {
    auto kernel = op->getParentOfType<riscv::KernelOp>();
    if (!kernel) return false;
    if (auto load = dyn_cast<riscv::LoadOp>(op))
      if (localMemory(load.getRegion()))
        if (auto type = dyn_cast<riscv::ValueType>(load.getResult().getType()); type &&
            type.getShape() == load.getRegion().getType().getShape() &&
            type.getAxisIds() == load.getRegion().getType().getAxisIds() &&
            llvm::any_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent < 0; }))
          return true;
    llvm::SmallPtrSet<void *, 8> seen;
    bool dynamicLocal = false, scalarTraversal = false, dynamicNumeric = false;
    auto account = [&](Value value) {
      auto type = dyn_cast<riscv::ValueType>(value.getType());
      if (!type || !seen.insert(value.getAsOpaquePointer()).second) return;
      dynamicLocal |= bool(local(value)) && llvm::any_of(type.getShape().asArrayRef(),
          [](int64_t extent) { return extent < 0; });
      dynamicNumeric |= type.getLayout().getCarrier() != "local" &&
          llvm::any_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent < 0; });
      scalarTraversal |= type.getLayout().getCarrier() == "scalar" &&
          llvm::any_of(type.getLayout().getTimeFactors().asArrayRef(),
              [](int64_t factor) { return factor > 1; });
    };
    for (Value value : op->getOperands()) account(value);
    for (Value value : op->getResults()) account(value);
    const int64_t budget = kernel.getTarget().getVectorRegisters();
    if (dynamicLocal || dynamicNumeric || scalarTraversal || exceedsRegisterBudget(op, budget)) return true;
    for (Value result : op->getResults())
      if (isa<riscv::ValueType>(result.getType()) &&
          consumerExceedsRegisterBudget(result, budget)) return true;
    return false;
  }

  riscv::ValueType localType(riscv::ValueType type) {
    return riscv_internal::localValueType(b, type);
  }

  FailureOr<Value> allocate(riscv::ValueType type, Operation *anchor, bool preserveBirth = false) {
    return riscv_internal::allocateLocalValue(b, type, anchor, nextIdentity, preserveBirth);
  }

  FailureOr<Value> view(Value source, Operation *anchor) {
    return riscv_internal::localView(b, source, anchor);
  }

  riscv::ValueType fragment(riscv::ValueType type, int64_t axis, int64_t width, riscv::TargetAttr target) {
    return riscv_internal::valuePieceType(b, type, axis, width, target);
  }

  LogicalResult traverse(riscv::ValueType type, Operation *anchor,
      function_ref<LogicalResult(ArrayRef<Value>, int64_t, int64_t)> body,
      bool numericMemory = false, Value region = {},
      std::optional<std::pair<int64_t, int64_t>> selectedIssue = std::nullopt) {
    std::optional<std::pair<int64_t, int64_t>> issue;
    auto byteElements = [](riscv::ValueType value) {
      unsigned bits = riscv_internal::logicalBitWidth(value);
      return bits >= 8 && bits % 8 == 0;
    };
    // Boolean local objects are byte storage, whereas RVV predicates are bit
    // masks. They need an explicit byte-to-mask transfer before vector issues.
    bool ordinaryStorage = byteElements(type) && llvm::all_of(anchor->getOperands(), [&](Value value) {
      auto operand = dyn_cast<riscv::ValueType>(value.getType());
      return !operand || byteElements(operand);
    });
    if (numericMemory && !type.getShape().empty() &&
        type.getLayout().getCarrier() != "rvv" && ordinaryStorage) {
      auto target = anchor->getParentOfType<riscv::KernelOp>().getTarget();
      int64_t bits = riscv_internal::logicalBitWidth(type);
      for (Value operand : anchor->getOperands())
        if (auto input = dyn_cast<riscv::ValueType>(operand.getType()))
          bits = std::max<int64_t>(bits, riscv_internal::logicalBitWidth(input));
      bits = std::max<int64_t>(8, bits);
      int64_t width = target.getVlenBits() / bits;
      if (width > 1) issue = {{type.getAxisIds().asArrayRef().back(), width}};
    }
    SmallVector<Value> extents;
    if (region) {
      auto memory = dyn_cast<riscv::MemDescType>(region.getType());
      if (!memory || memory.getShape() != type.getShape() || memory.getAxisIds() != type.getAxisIds()) return failure();
      for (size_t axis = 0; axis < type.getShape().size(); ++axis)
        extents.push_back(b.create<riscv::ExtentOp>(anchor->getLoc(), b.getIndexType(), region, axis));
    }
    return riscv_internal::forEachValuePiece(b, type, anchor, body,
        selectedIssue ? selectedIssue : issue, extents);
  }

  FailureOr<Value> load(Value memory, riscv::ValueType domain, ArrayRef<Value> coordinates,
             riscv::ValueType piece, Operation *anchor) {
    auto region = riscv_internal::valueMemoryWindow(b, memory, domain, coordinates, piece, anchor);
    if (failed(region)) return failure();
    return b.create<riscv::LoadOp>(anchor->getLoc(), piece, *region, Value(),
        riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b)).getResult();
  }

  LogicalResult store(Value value, Value memory, riscv::ValueType domain,
             ArrayRef<Value> coordinates, riscv::ValueType piece, Operation *anchor) {
    auto region = riscv_internal::valueMemoryWindow(b, memory, domain, coordinates, piece, anchor);
    if (failed(region)) return failure();
    b.create<riscv::StoreOp>(anchor->getLoc(), value, *region,
        riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
    return success();
  }

  FailureOr<Value> copyView(Value input, riscv::ValueType type, Operation *anchor) {
    auto destination = allocate(type, anchor, true);
    if (failed(destination)) return failure();
    auto output = view(*destination, anchor);
    if (failed(output)) return failure();
    auto target = anchor->getParentOfType<riscv::KernelOp>().getTarget();
    if (failed(traverse(type, anchor, [&](ArrayRef<Value> coordinates, int64_t axis, int64_t width) {
          auto piece = fragment(type, axis, width, target);
          if (!piece) return failure();
          auto value = load(input, type, coordinates, piece, anchor);
          if (failed(value)) return failure();
          return store(*value, *output, type, coordinates, piece, anchor);
        }, true, input))) return anchor->emitError("whole-value copy has no bounded physical fragment"), failure();
    return *destination;
  }

  FailureOr<Value> copyLocal(Value input, riscv::ValueType type, Operation *anchor) {
    auto memory = view(input, anchor);
    if (failed(memory)) return failure();
    return copyView(*memory, type, anchor);
  }

  FailureOr<Value> sourceView(Value value, Operation *anchor) {
    if (Value source = local(value)) return view(source, anchor);
    if (auto found = numericStorage.find(value); found != numericStorage.end()) {
      DominanceInfo dominance(module);
      if (dominance.properlyDominates(found->second.getDefiningOp(), anchor)) return found->second;
    }
    auto type = cast<riscv::ValueType>(value.getType());
    auto groups = riscv_internal::residentRegisterGroups(value);
    if (!groups || llvm::any_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent < 0; }) ||
        *groups > anchor->getParentOfType<riscv::KernelOp>().getTarget().getVectorRegisters()) {
      auto diagnostic = anchor->emitError("whole-value operand has no completed bounded source; type=");
      diagnostic << type << "; groups=" << groups.value_or(-1) << "; producer=";
      if (auto *definition = value.getDefiningOp()) diagnostic << definition->getName();
      else diagnostic << "block argument";
      return failure();
    }
    auto storage = allocate(type, anchor);
    if (failed(storage)) return failure();
    auto memory = view(*storage, anchor);
    if (failed(memory)) return failure();
    auto m = cast<riscv::MemDescType>(memory->getType());
    auto sliceType = riscv::MemDescType::get(b.getContext(), m.getEncoding(), m.getShape(), m.getAxisIds(),
        m.getStrides(), m.getOrigins(), m.getAlignment(), "slice", m.getAccess(), m.getAliasSet(),
        m.getLayoutIdentity(), m.getStorageBits(), m.getElements(), m.getInterleaveRows());
    auto slice = b.create<riscv::SliceOp>(anchor->getLoc(), sliceType, *memory, ValueRange{},
        b.getArrayAttr(SmallVector<Attribute>(type.getShape().size(), b.getStringAttr("all"))));
    b.create<riscv::StoreOp>(anchor->getLoc(), value, slice, riscv_internal::unassignedAccess(b),
        riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
    numericStorage[value] = *memory;
    return *memory;
  }

  Value iotaElement(riscv::IotaOp iota, Value coordinate, Operation *anchor) {
    Type element = iota.getResult().getType().getElementType();
    Value base = coordinate;
    if (base.getType() != element)
      base = b.create<riscv::CastOp>(anchor->getLoc(), element, base,
          riscv_internal::unselectedLeaf(b));
    if (iota.getStart() != 0) {
      Value start = b.create<arith::ConstantOp>(anchor->getLoc(), element,
          b.getIntegerAttr(element, iota.getStart()));
      base = b.create<riscv::BinaryOp>(anchor->getLoc(), element, base, start,
          "add", riscv_internal::unselectedLeaf(b));
    }
    return base;
  }

  Value address(Value memory, ArrayRef<Value> coordinates, Operation *anchor,
                int64_t retained = -1) {
    auto type = cast<riscv::MemDescType>(memory.getType());
    SmallVector<int64_t> shape, axes, strides, origins;
    SmallVector<Attribute> selectors;
    SmallVector<Value> indices;
    for (size_t position = 0; position < type.getShape().size(); ++position) {
      selectors.push_back(b.getStringAttr(int64_t(position) == retained ? "all" : "index"));
      if (int64_t(position) == retained) {
        shape.push_back(type.getShape()[position]); axes.push_back(type.getAxisIds()[position]);
        strides.push_back(type.getStrides()[position]); origins.push_back(type.getOrigins()[position]);
      } else indices.push_back(coordinates[position]);
    }
    auto selected = riscv::MemDescType::get(b.getContext(), type.getEncoding(),
        b.getDenseI64ArrayAttr(shape), b.getDenseI64ArrayAttr(axes),
        b.getDenseI64ArrayAttr(strides), b.getDenseI64ArrayAttr(origins),
        type.getAlignment(), "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(),
        type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    return b.create<riscv::SliceOp>(anchor->getLoc(), selected, memory, indices, b.getArrayAttr(selectors));
  }

  LogicalResult extraction(riscv::ExtractOp op, Value source) {
    auto type = dyn_cast<riscv::ValueType>(op.getResult().getType());
    if (!type || op->hasAttr("index_pattern")) return success();
    for (Attribute attribute : op.getSelectors()) {
      auto kind = cast<StringAttr>(attribute).getValue();
      if (kind != "all" && kind != "index" && kind != "gather") return success();
    }
    b.setInsertionPoint(op);
    auto input = view(source, op);
    if (failed(input)) return failure();
    SmallVector<Value> gatherViews(op.getIndices().size());
    for (auto [position, index] : llvm::enumerate(op.getIndices()))
      if (isa<riscv::ValueType>(index.getType()) && !index.getDefiningOp<riscv::IotaOp>()) {
        auto memory = sourceView(index, op);
        if (failed(memory)) return failure();
        gatherViews[position] = *memory;
      }
    auto destination = allocate(type, op);
    if (failed(destination)) return failure();
    auto output = view(*destination, op);
    if (failed(output)) return failure();
    auto sourceType = cast<riscv::ValueType>(source.getType());
    auto target = op->getParentOfType<riscv::KernelOp>().getTarget();
    // A retained all-axis is a dense strip. Gathered axes are traversed one
    // actual result coordinate at a time, without a full index-register tuple.
    auto domainLayout = type.getLayout();
    for (size_t position = 0; position < type.getShape().size(); ++position)
      if (domainLayout.getLaneFactors()[position] > 1) {
        auto found = llvm::find(sourceType.getAxisIds().asArrayRef(), type.getAxisIds()[position]);
        size_t sourcePosition = found - sourceType.getAxisIds().asArrayRef().begin();
        bool all = found != sourceType.getAxisIds().asArrayRef().end() &&
            (sourcePosition >= op.getSelectors().size() ||
             cast<StringAttr>(op.getSelectors()[sourcePosition]).getValue() == "all");
        // A gather may share an output axis with a preceding all-selector.
        // That gather then changes for every lane, so it cannot become one
        // scalar address followed by a contiguous vector load.
        for (Value index : op.getIndices())
          if (auto indexType = dyn_cast<riscv::ValueType>(index.getType())) {
            auto shared = llvm::find(indexType.getAxisIds().asArrayRef(), type.getAxisIds()[position]);
            if (shared != indexType.getAxisIds().asArrayRef().end() &&
                indexType.getShape()[shared - indexType.getAxisIds().asArrayRef().begin()] != 1)
              all = false;
          }
        if (!all) domainLayout = fragment(type, 0, 1, target).getLayout();
      }
    auto domain = cast<riscv::ValueType>(riscv_internal::withLayout(type, domainLayout));
    if (failed(traverse(domain, op, [&](ArrayRef<Value> coordinates, int64_t axis, int64_t width) {
          SmallVector<Value> inputCoordinates;
          int64_t retained = -1;
          size_t cursor = 0;
          for (size_t position = 0; position < sourceType.getShape().size(); ++position) {
            StringRef kind = position < op.getSelectors().size()
                ? cast<StringAttr>(op.getSelectors()[position]).getValue() : "all";
            Value coordinate;
            if (kind == "all") {
              auto found = llvm::find(type.getAxisIds().asArrayRef(), sourceType.getAxisIds()[position]);
              coordinate = coordinates[found - type.getAxisIds().asArrayRef().begin()];
              if (width > 1 && sourceType.getAxisIds()[position] == axis) retained = position;
            } else {
              Value index = op.getIndices()[cursor];
              if (kind == "index") coordinate = index;
              else {
                auto indexType = cast<riscv::ValueType>(index.getType());
                SmallVector<Value> indexCoordinates;
                for (int64_t indexAxis : indexType.getAxisIds().asArrayRef()) {
                  auto found = llvm::find(type.getAxisIds().asArrayRef(), indexAxis);
                  indexCoordinates.push_back(coordinates[found - type.getAxisIds().asArrayRef().begin()]);
                }
                if (auto iota = index.getDefiningOp<riscv::IotaOp>())
                  coordinate = iotaElement(iota, indexCoordinates.front(), op);
                else coordinate = b.create<riscv::LoadOp>(op.getLoc(), indexType.getElementType(),
                    address(gatherViews[cursor], indexCoordinates, op), Value(),
                    riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
                if (!coordinate.getType().isIndex())
                  coordinate = b.create<riscv::CastOp>(op.getLoc(), b.getIndexType(), coordinate,
                      riscv_internal::unselectedLeaf(b));
              }
              ++cursor;
            }
            inputCoordinates.push_back(coordinate);
          }
          Value region = address(*input, inputCoordinates, op, retained);
          if (retained < 0) {
            Value value = b.create<riscv::LoadOp>(op.getLoc(), type.getElementType(), region,
                Value(), riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
            b.create<riscv::StoreOp>(op.getLoc(), value, address(*output, coordinates, op),
                riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
          } else {
            auto piece = fragment(type, axis, width, target);
            if (!piece) return failure();
            auto rankOne = riscv::ValueType::get(b.getContext(), type.getElementType(),
                b.getDenseI64ArrayAttr({width}), b.getDenseI64ArrayAttr({axis}),
                riscv_internal::scalarLayout(b, type.getElementType(), {width}, {axis}));
            auto layout = riscv_internal::projectLayout(b, rankOne, piece.getLayout(), target);
            if (!layout) return failure();
            rankOne = cast<riscv::ValueType>(riscv_internal::withLayout(rankOne, layout));
            auto original = cast<riscv::MemDescType>(region.getType());
            auto strip = riscv::MemDescType::get(b.getContext(), original.getEncoding(), rankOne.getShape(),
                original.getAxisIds(), original.getStrides(), original.getOrigins(), original.getAlignment(),
                "slice", original.getAccess(), original.getAliasSet(), original.getLayoutIdentity(),
                original.getStorageBits(), original.getElements(), original.getInterleaveRows());
            region = b.create<riscv::SubviewOp>(op.getLoc(), strip, region,
                ValueRange{inputCoordinates[retained]}, b.getDenseI64ArrayAttr({-1}), rankOne.getShape());
            Value loaded = b.create<riscv::LoadOp>(op.getLoc(), rankOne, region, Value(),
                riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
            if (rankOne != piece)
              loaded = b.create<riscv::ReshapeOp>(op.getLoc(), piece, loaded, rankOne.getAxisIds());
            if (failed(store(loaded, *output, type, coordinates, piece, op))) return failure();
          }
          return success();
        }))) return op.emitError("whole-value extraction has no bounded source projection");
    replace(op, *destination);
    return success();
  }

  void replace(Operation *op, Value source) {
    auto expected = cast<riscv::ValueType>(op->getResult(0).getType());
    Value replacement = source;
    if (source.getType() != expected) {
      auto actual = cast<riscv::ValueType>(source.getType());
      auto bridge = b.create<riscv::ConvertLayoutOp>(op->getLoc(), expected, source,
          riscv_internal::layoutConversion(b, actual.getLayout(), expected.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(b));
      replacement = bridge;
    }
    b.replaceOp(op, replacement);
  }
};

class MaterializeRISCVValueProgramsPass
    : public PassWrapper<MaterializeRISCVValueProgramsPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "weft-riscv-materialize-value-programs"; }
  StringRef getDescription() const override { return "Realize complete numeric snapshots through bounded physical value programs"; }
  void runOnOperation() override {
    if (failed(ValuePrograms(getOperation()).run())) signalPassFailure();
  }
};
}

std::unique_ptr<Pass> weft::createMaterializeRISCVValueProgramsPass() {
  return std::make_unique<MaterializeRISCVValueProgramsPass>();
}
