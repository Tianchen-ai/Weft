#include "Weft/Target/RISCVPasses.h"
#include "RISCVPhysicalSupport.h"
#include "LocalMemory.h"
#include "LocalLifetime.h"
#include "ControlStorage.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>

using namespace mlir;
using namespace weft;

namespace {

struct Window {
  SmallVector<Value> origins;
  SmallVector<int64_t> sizes;
  SmallVector<bool> drop;
};

FailureOr<Window> windowFor(Operation *operation, riscv::ValueType source,
                            ValueRange indices, ArrayAttr selectors,
                            IRRewriter &b) {
  Window result;
  size_t cursor = 0;
  for (auto [axis, attribute] : llvm::enumerate(selectors)) {
    StringRef selector = cast<StringAttr>(attribute).getValue();
    Value origin;
    int64_t size = source.getShape()[axis];
    bool drop = selector == "index";
    if (selector == "all") {
      if (size == 0) return failure();
      origin = arith::ConstantIndexOp::create(b, operation->getLoc(), 0);
    } else if (selector == "index") {
      origin = indices[cursor++];
      size = 1;
    } else if (selector == "gather") {
      Value index = indices[cursor++];
      auto type = dyn_cast<riscv::ValueType>(index.getType());
      auto relation = riscv_internal::analyzeRegularIndexRelation(index);
      if (!type || type.getShape().size() != 1 || type.getShape()[0] <= 0 ||
          type.getAxisIds()[0] != source.getAxisIds()[axis] || !relation ||
          relation->stride != 1 || relation->repeat != 1)
        return failure();
      size = type.getShape()[0];
      origin = arith::ConstantIndexOp::create(b, operation->getLoc(), relation->base);
      if (Value base = relation->dynamicBase) {
        if (!riscv_internal::logicalShape(base.getType()).empty()) return failure();
        if (!base.getType().isIndex())
          base = riscv::CastOp::create(b, operation->getLoc(), b.getIndexType(), base,
                                        riscv_internal::unselectedLeaf(b));
        origin = arith::AddIOp::create(b, operation->getLoc(), origin, base);
      }
    } else return failure();
    result.origins.push_back(origin);
    result.sizes.push_back(size);
    result.drop.push_back(drop);
  }
  if (selectors.size() != source.getShape().size() || cursor != indices.size())
    return failure();
  return result;
}

LogicalResult materializeLocalStores(ModuleOp module, IRRewriter &b) {
  SmallVector<riscv::ConvertLayoutOp> stores;
  int64_t nextIdentity = riscv_internal::lastLocalIdentity(module);
  module.walk([&](Operation *operation) {
    if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(operation);
        conversion && conversion.getConversion().getKind() == "local_store")
      stores.push_back(conversion);
  });
  for (riscv::ConvertLayoutOp conversion : stores) {
    auto source = conversion.getInput().getType();
    auto target = conversion.getResult().getType();
    if (!isa<IntegerType, FloatType>(target.getElementType()) ||
        (source.getLayout().getCarrier() != "scalar" &&
         source.getLayout().getCarrier() != "rvv") ||
        target.getLayout().getCarrier() != "local" ||
        source.getElementType() != target.getElementType() ||
        source.getShape() != target.getShape() ||
        source.getAxisIds() != target.getAxisIds())
      return conversion.emitOpError(
          "local-store conversion requires one identical complete numeric domain");
    if (!riscv::storageAliasesRemainInScope(
            conversion.getResult(), conversion->getBlock()))
      return conversion.emitOpError(
          "local-store conversion requires storage outside its lexical scope");

    b.setInsertionPoint(conversion);
    Location loc = conversion.getLoc();
    // Dynamic extents use the same checked allocation and scope accounting as
    // all other whole values. Each conversion still owns a separate snapshot.
    auto binding = riscv_internal::allocateLocalValue(b, target, conversion,
        nextIdentity, false, "physical-share");
    if (failed(binding)) return failure();
    auto view = riscv_internal::localView(b, *binding, conversion);
    if (failed(view))
      return conversion.emitOpError(
          "local-store conversion has no explicit storage descriptor");
    auto store = riscv::StoreOp::create(b,
        loc, conversion.getInput(), riscv_internal::completeLocalSlice(b, *view),
        riscv_internal::unassignedAccess(b),
        riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
    riscv_internal::copyOrigin(conversion, store);
    b.replaceOp(conversion, *binding);
  }
  return success();
}

Value project(Value base, const Window &window, IRRewriter &b) {
  auto type = cast<riscv::MemDescType>(base.getType());
  auto make = [&](ArrayRef<int64_t> shape, ArrayRef<int64_t> axes,
                  ArrayRef<int64_t> strides, ArrayRef<int64_t> origins) {
    return riscv::MemDescType::get(b.getContext(), type.getEncoding(),
        b.getDenseI64ArrayAttr(shape), b.getDenseI64ArrayAttr(axes),
        b.getDenseI64ArrayAttr(strides), b.getDenseI64ArrayAttr(origins),
        type.getAlignment(), "slice", type.getAccess(), type.getAliasSet(),
        type.getLayoutIdentity(), type.getStorageBits(), 1, 0);
  };
  SmallVector<Value> bounds(window.origins);
  SmallVector<int64_t> extents;
  for (auto [axis, size] : llvm::enumerate(window.sizes)) {
    extents.push_back(size < 0 ? -1 : size);
    if (size < 0)
      bounds.push_back(riscv::ExtentOp::create(b, base.getLoc(), b.getIndexType(), base, axis));
  }
  Value view = riscv::SubviewOp::create(b, base.getLoc(),
      make(window.sizes, type.getAxisIds().asArrayRef(), type.getStrides().asArrayRef(), type.getOrigins().asArrayRef()),
      base, bounds, b.getDenseI64ArrayAttr(SmallVector<int64_t>(window.sizes.size(), -1)),
      b.getDenseI64ArrayAttr(extents));
  if (llvm::none_of(window.drop, [](bool drop) { return drop; })) return view;
  SmallVector<int64_t> shape, axes, strides, origins;
  SmallVector<Attribute> selectors;
  SmallVector<Value> indices;
  for (size_t axis = 0; axis < window.sizes.size(); ++axis) {
    selectors.push_back(b.getStringAttr(window.drop[axis] ? "index" : "all"));
    if (window.drop[axis]) indices.push_back(arith::ConstantIndexOp::create(b, base.getLoc(), 0));
    else {
      shape.push_back(window.sizes[axis]); axes.push_back(type.getAxisIds()[axis]);
      strides.push_back(type.getStrides()[axis]); origins.push_back(type.getOrigins()[axis]);
    }
  }
  return riscv::SliceOp::create(b, base.getLoc(), make(shape, axes, strides, origins), view, indices, b.getArrayAttr(selectors));
}

class MaterializeRISCVLocalWindowsPass
    : public PassWrapper<MaterializeRISCVLocalWindowsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVLocalWindowsPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-local-windows"; }
  StringRef getDescription() const override { return "Expose bounded private state windows as ordinary descriptor memory edges"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    if (failed(riscv_internal::materializeControlStorage(getOperation(), b))) {
      signalPassFailure();
      return;
    }
    if (failed(materializeLocalStores(getOperation(), b))) {
      signalPassFailure();
      return;
    }
    SmallVector<Operation *> operations;
    getOperation().walk([&](Operation *op) {
      if (isa<riscv::ExtractOp, riscv::UpdateOp>(op)) operations.push_back(op);
      else if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op);
          conversion && conversion.getConversion().getKind() == "local_load") operations.push_back(op);
    });
    DenseMap<Operation *, bool> exclusiveVersions;
    DenseMap<Operation *, Location> unprovenUses;
    DominanceInfo dominance(getOperation());
    for (Operation *op : operations)
      if (auto update = dyn_cast<riscv::UpdateOp>(op)) {
        Operation *unproven = nullptr;
        exclusiveVersions[op] = riscv_internal::hasExclusiveLocalVersion(update.getInput(), op, dominance, unproven);
        if (unproven) unprovenUses.try_emplace(op, unproven->getLoc());
      }
    for (Operation *op : operations) {
      Value source = op->getOperand(0);
      auto type = cast<riscv::ValueType>(source.getType());
      if (type.getLayout().getCarrier() != "local" || isa<kernel::EncodingType>(type.getElementType())) continue;
      if (!riscv_internal::isPrivateLocalState(source)) continue;
      b.setInsertionPoint(op);
      if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op)) {
        auto view = riscv_internal::localView(b, source, op);
        if (failed(view)) {
          op->emitError("private compute input has no explicit extent binding");
          signalPassFailure(); return;
        }
        auto load = riscv::LoadOp::create(b, op->getLoc(), conversion.getResult().getType(), *view, Value(),
            riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
        riscv_internal::copyOrigin(op, load);
        b.replaceOp(op, load.getResult());
        continue;
      }
      auto extract = dyn_cast<riscv::ExtractOp>(op);
      auto update = dyn_cast<riscv::UpdateOp>(op);
      if (update && update.getValue().getType() == source.getType() &&
          update.getIndices().empty() && llvm::all_of(update.getSelectors(), [](Attribute selector) {
            return cast<StringAttr>(selector).getValue() == "all";
          })) {
        if (!exclusiveVersions.lookup(op)) {
          auto diagnostic = op->emitError("complete private update keeps an older state live; materialize an independent destination first");
          if (auto use = unprovenUses.find(op); use != unprovenUses.end())
            diagnostic << "; proof blocked at " << use->second;
          signalPassFailure(); return;
        }
        auto view = riscv_internal::localView(b, source, op);
        if (failed(view)) {
          op->emitError("complete private update has no explicit extent binding");
          signalPassFailure(); return;
        }
        auto store = riscv::StoreOp::create(b, op->getLoc(), update.getValue(), riscv_internal::completeLocalSlice(b, *view),
            riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
        riscv_internal::copyOrigin(op, store);
        b.replaceOp(op, source);
        continue;
      }
      auto window = windowFor(op, type, extract ? extract.getIndices() : update.getIndices(),
          extract ? extract.getSelectors() : update.getSelectors(), b);
      if (failed(window)) continue;
      if (update) {
        if (!exclusiveVersions.lookup(op)) {
          auto diagnostic = op->emitError("private window update requires an exclusive current state version; materialize a separate state before keeping an older version live");
          if (auto use = unprovenUses.find(op); use != unprovenUses.end())
            diagnostic << "; proof blocked at " << use->second;
          signalPassFailure(); return;
        }
      }
      auto view = riscv_internal::localView(b, source, op);
      if (failed(view)) {
        op->emitError("private window has no explicit source extent binding");
        signalPassFailure(); return;
      }
      Value region = project(*view, *window, b);
      if (extract) {
        auto load = riscv::LoadOp::create(b, op->getLoc(), extract.getResult().getType(), region, Value(),
            riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
        riscv_internal::copyOrigin(op, load);
        b.replaceOp(op, load.getResult());
      } else {
        auto store = riscv::StoreOp::create(b, op->getLoc(), update.getValue(), region,
            riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
        riscv_internal::copyOrigin(op, store);
        b.replaceOp(op, source);
      }
    }
  }
};

}

std::unique_ptr<Pass> weft::createMaterializeRISCVLocalWindowsPass() {
  return std::make_unique<MaterializeRISCVLocalWindowsPass>();
}
