#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
class MaterializeRISCVAtomicsPass
    : public PassWrapper<MaterializeRISCVAtomicsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVAtomicsPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-atomics"; }
  StringRef getDescription() const override { return "Materialize pointer atomic lanes and their old-value results"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    SmallVector<riscv::PointerAtomicOp> operations;
    int64_t nextBirth = 0;
    getOperation().walk([&](Operation *op) {
      if (auto birth = op->getAttrOfType<IntegerAttr>("birth_id")) nextBirth = std::max(nextBirth, birth.getInt() + 1);
    });
    getOperation().walk([&](riscv::PointerAtomicOp op) { operations.push_back(op); });
    IRRewriter b(&getContext());
    for (auto op : operations) {
      b.setInsertionPoint(op);
      auto selected = leaf(b, "scalar", "pointer-atomic", "scalar.pointer-atomic." + op.getKind().str(),
          "scalar.pointer-atomic." + op.getKind().str(), 0, 0, 0, 0, "explicit", "exact");
      auto shaped = dyn_cast<riscv::ValueType>(op.getResult().getType());
      if (!shaped) { op.setLeafAttr(selected); continue; }
      int64_t elements = 1;
      for (int64_t size : shaped.getShape().asArrayRef()) {
        if (size <= 0 || elements > INT64_MAX / size) {
          op.emitOpError("atomic tensor requires a bounded static domain"); signalPassFailure(); return;
        }
        elements *= size;
      }
      auto loc = op.getLoc();
      auto element = shaped.getElementType();
      int64_t owner = 0;
      if (auto loop = op->getParentOfType<riscv::LoopOp>()) owner = loop.getDomain().getType().getDomainId();
      SmallVector<Value> initial;
      const bool keepOldValues = !op.getResult().use_empty();
      if (keepOldValues) {
        SmallVector<Value> extents;
        for (int64_t size : shaped.getShape().asArrayRef()) extents.push_back(arith::ConstantIndexOp::create(b, loc, size));
        initial.push_back(riscv::EmptyOp::create(b, loc, shaped, extents, owner, nextBirth++, owner));
      }
      auto zero = arith::ConstantIndexOp::create(b, loc, 0);
      auto end = arith::ConstantIndexOp::create(b, loc, elements);
      auto one = arith::ConstantIndexOp::create(b, loc, 1);
      auto loop = scf::ForOp::create(b, loc, zero, end, one, initial);
      loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      b.setInsertionPointToStart(loop.getBody());
      SmallVector<Value> coordinates;
      int64_t stride = elements;
      for (int64_t size : shaped.getShape().asArrayRef()) {
        stride /= size;
        Value coordinate = loop.getInductionVar();
        if (stride != 1) coordinate = arith::DivUIOp::create(b, loc, coordinate,
            arith::ConstantIndexOp::create(b, loc, stride));
        coordinate = arith::RemUIOp::create(b, loc, coordinate, arith::ConstantIndexOp::create(b, loc, size));
        coordinates.push_back(coordinate);
      }
      auto selectors = b.getArrayAttr(SmallVector<Attribute>(coordinates.size(), b.getStringAttr("index")));
      auto scalar = [&](Value value) -> Value {
        auto type = dyn_cast<riscv::ValueType>(value.getType());
        if (!type) return value;
        return riscv::ExtractOp::create(b, loc, type.getElementType(), value, coordinates,
            selectors, unassignedAccess(b), unselectedLeaf(b));
      };
      auto atomic = riscv::PointerAtomicOp::create(b, loc, element, op.getBase(), scalar(op.getOffsets()),
          scalar(op.getMask()), scalar(op.getValue()), scalar(op.getCompare()), op.getKind(),
          op.getOrdering(), op.getScope(), selected);
      if (keepOldValues) {
        auto update = riscv::UpdateOp::create(b, loc, shaped, loop.getRegionIterArgs()[0], atomic,
            coordinates, selectors, unselectedLeaf(b));
        scf::YieldOp::create(b, loc, ValueRange{update});
      }
      copyOrigin(op, loop);
      copyOrigin(op, atomic);
      if (keepOldValues) b.replaceOp(op, loop.getResults());
      else b.eraseOp(op);
    }
  }
};
}

std::unique_ptr<Pass> weft::createMaterializeRISCVAtomicsPass() {
  return std::make_unique<MaterializeRISCVAtomicsPass>();
}
