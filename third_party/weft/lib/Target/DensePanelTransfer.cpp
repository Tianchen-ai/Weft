#include "DensePanelTransfer.h"
#include "DenseIssue.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/PanelTransfer.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
struct TransferShape {
  riscv::ValueType type;
  int64_t fields;
  int64_t lanes;
};

class PanelTransfer {
public:
  PanelTransfer(IRRewriter &b, riscv::DensePanelCopyOp operation, int64_t &shapeIdentity)
      : b(b), source(operation.getSource()), destination(operation.getDestination()),
        shapeIdentity(shapeIdentity), anchor(operation), loc(operation.getLoc()),
        operation(operation) {}

  LogicalResult run() {
    if (failed(riscv::verifyPanelTransferPlan(operation))) return failure();
    geometry = *riscv::PanelTransferGeometry::get(operation);
    b.setInsertionPoint(operation);
    auto plan = *operation.getPlan();
    auto cases = plan.getCases();
    if (!plan.getUnitAxis()) return issue(cast<riscv::PanelTransferCaseAttr>(cases[0]));
    auto alternate = riscv::denseArgumentStride(source, plan.getUnitAxis());
    auto stride = riscv::StrideOp::create(b, loc, b.getIndexType(), alternate->first, alternate->second);
    Value contiguous = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, stride, constant(1));
    if (plan.getNonunitAxis()) {
      auto preferred = riscv::denseArgumentStride(source, plan.getNonunitAxis());
      auto primary = riscv::StrideOp::create(b, loc, b.getIndexType(), preferred->first, preferred->second);
      auto nonunit = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, primary, constant(1));
      contiguous = arith::AndIOp::create(b, loc, contiguous, nonunit);
    }
    auto branch = scf::IfOp::create(b, loc, contiguous, true);
    copyOrigin(anchor, branch);
    b.setInsertionPointToStart(&branch.getThenRegion().front());
    if (failed(issue(cast<riscv::PanelTransferCaseAttr>(cases[0])))) return failure();
    b.setInsertionPointToStart(&branch.getElseRegion().front());
    if (failed(issue(cast<riscv::PanelTransferCaseAttr>(cases[1])))) return failure();
    b.setInsertionPointAfter(branch);
    return success();
  }

private:
  LogicalResult issue(riscv::PanelTransferCaseAttr selected) {
    copyAxis = selected.getMain().getAxis();
    fieldAxis = geometry.fieldAxis(copyAxis);
    Value outer = extent(source, fieldAxis), inner = extent(source, copyAxis);
    auto shape = [&](riscv::PanelTransferTileAttr tile) {
      return TransferShape{geometry.numeric(b, tile), tile.getFields(), tile.getLanes()};
    };
    auto main = shape(selected.getMain());
    auto single = shape(selected.getTail());
    Value begin = constant(0);
    if (main.fields > 1) {
      Value width = constant(main.fields);
      Value complete = arith::SubIOp::create(b, loc, outer, arith::RemUIOp::create(b, loc, outer, width));
      if (failed(rows(begin, complete, inner, main))) return failure();
      begin = complete;
    }
    return rows(begin, outer, inner, single);
  }

  IRRewriter &b;
  Value source, destination;
  int64_t copyAxis = 0;
  int64_t &shapeIdentity;
  Operation *anchor;
  Location loc;
  riscv::DensePanelCopyOp operation;
  riscv::PanelTransferGeometry geometry;
  int64_t fieldAxis = 0;

  Value constant(int64_t value) { return arith::ConstantIndexOp::create(b, loc, value); }
  Value extent(Value memory, int64_t axis) {
    auto axes = cast<riscv::MemDescType>(memory.getType()).getAxisIds().asArrayRef();
    return riscv::ExtentOp::create(b, loc, b.getIndexType(), memory, llvm::find(axes, axis) - axes.begin());
  }

  Value window(Value memory, Value row, Value column, Value length, int64_t fields) {
    auto type = cast<riscv::MemDescType>(memory.getType());
    SmallVector<Value> bounds;
    SmallVector<int64_t> sizes, shape;
    for (int64_t axis : type.getAxisIds().asArrayRef()) bounds.push_back(axis == copyAxis ? column : row);
    for (int64_t axis : type.getAxisIds().asArrayRef()) {
      if (axis == copyAxis) {
        if (auto size = riscv::denseConstantIndex(length)) {
          sizes.push_back(*size);
          shape.push_back(*size);
          continue;
        }
        auto local = riscv::LocalShapeOp::create(b, loc, b.getIndexType(), length, ++shapeIdentity);
        bounds.push_back(local);
        sizes.push_back(-1);
        shape.push_back(-shapeIdentity);
      } else {
        sizes.push_back(fields);
        shape.push_back(fields);
      }
    }
    auto result = riscv::MemDescType::get(b.getContext(), type.getEncoding(), b.getDenseI64ArrayAttr(shape),
        type.getAxisIds(), type.getStrides(), type.getOrigins(), type.getAlignment(), "slice", type.getAccess(),
        type.getAliasSet(), type.getLayoutIdentity(), type.getStorageBits(), type.getElements(), type.getInterleaveRows());
    return riscv::SubviewOp::create(b, loc, result, memory, bounds,
        b.getDenseI64ArrayAttr({-1, -1}), b.getDenseI64ArrayAttr(sizes));
  }

  SmallVector<Value> bounds(Value memory, riscv::ValueType type) {
    SmallVector<Value> result;
    for (int64_t axis : type.getAxisIds().asArrayRef()) result.push_back(extent(memory, axis));
    return result;
  }

  LogicalResult rows(Value begin, Value end, Value inner, const TransferShape &shape) {
    auto first = riscv::denseConstantIndex(begin), last = riscv::denseConstantIndex(end);
    if (begin == end || (first && last && *first == *last)) return success();
    auto row = scf::ForOp::create(b, loc, begin, end, constant(shape.fields));
    row->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    row->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(anchor, row);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(row.getBody());
    auto result = emitDenseIssues(b, anchor, inner, shape.lanes, {},
        [&](Value column, Value length, bool full, ValueRange) -> FailureOr<SmallVector<Value>> {
          if (failed(copy(row.getInductionVar(), column, length, shape, full))) return failure();
          return SmallVector<Value>{};
        }, 1, true, false, true);
    return failed(result) ? failure() : success();
  }

  LogicalResult copy(Value row, Value column, Value length, const TransferShape &shape, bool full) {
    Value from = window(source, row, column, length, shape.fields);
    Value to = window(destination, row, column, length, shape.fields);
    auto numeric = shape.type;
    auto load = riscv::RVVTileLoadOp::create(b, loc, numeric, from, bounds(from, numeric), copyAxis,
        denseAccess(b, cast<riscv::MemDescType>(from.getType()), numeric),
        leaf(b, "transfer", "tile-load", "rvv.tile-load", "rvv.tile-load", 0,
             numeric.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
    auto access = denseAccess(b, cast<riscv::MemDescType>(to.getType()), numeric);
    StringRef instruction = "rvv.tile-store";
    SmallVector<int64_t> parameters;
    if (shape.fields > 1) {
      instruction = full ? "rvv.tile-store.segment.full" : "rvv.tile-store.segment";
      access = riscv::AccessAttr::get(b.getContext(), "segment", "dense", access.getAlignment(),
          0, shape.fields, 0, 0, 0, 0, 0, 0, 0, "none", 1);
      parameters = {fieldAxis, shape.fields};
    }
    auto store = riscv::RVVTileStoreOp::create(b, loc, load, to, bounds(to, numeric), copyAxis, access,
        leaf(b, "transfer", "tile-store", instruction, instruction, numeric.getLayout().getRegisterGroups(),
             0, 0, 0, "none", "exact", parameters));
    copyOrigin(anchor, load);
    copyOrigin(anchor, store);
    return success();
  }
};

class MaterializeRISCVPanelTransfersPass
    : public PassWrapper<MaterializeRISCVPanelTransfersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVPanelTransfersPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-panel-transfers"; }
  StringRef getDescription() const override { return "Expand selected panel transfer cases into explicit physical loops"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    int64_t shapeIdentity = 0;
    getOperation().walk([&](riscv::LocalShapeOp shape) { shapeIdentity = std::max<int64_t>(shapeIdentity, shape.getId()); });
    SmallVector<riscv::DensePanelCopyOp> operations;
    getOperation().walk([&](riscv::DensePanelCopyOp operation) { operations.push_back(operation); });
    IRRewriter b(&getContext());
    for (auto operation : operations) {
      if (operation->getParentOfType<riscv::KernelOp>().getResourcesMaterialized() ||
          failed(PanelTransfer(b, operation, shapeIdentity).run())) {
        operation.emitError("could not materialize the selected panel transfer at an open physical boundary");
        signalPassFailure();
        return;
      }
      b.eraseOp(operation);
    }
  }
};
} // namespace

LogicalResult weft::riscv_internal::transferDensePanel(IRRewriter &builder,
    Value source, Value destination, int64_t copyAxis, int64_t maximumLMUL,
    int64_t maximumFields, Operation *anchor) {
  auto operation = riscv::DensePanelCopyOp::create(builder, anchor->getLoc(), source, destination,
      copyAxis, maximumLMUL, maximumFields, nullptr);
  copyOrigin(anchor, operation);
  return success();
}

std::unique_ptr<Pass> weft::createMaterializeRISCVPanelTransfersPass() {
  return std::make_unique<MaterializeRISCVPanelTransfersPass>();
}
