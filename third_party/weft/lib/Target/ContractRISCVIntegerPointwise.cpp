#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
std::optional<int64_t> sharedConstant(Value value) {
  for (unsigned depth = 0; depth < 8; ++depth) {
    if (auto shared = value.getDefiningOp<riscv::RegisterMaterializeOp>();
        shared && shared.getRealization() == "physical-share") {
      value = shared.getInput();
      continue;
    }
    if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
      auto source = dyn_cast<IntegerType>(cast.getInput().getType());
      auto result = dyn_cast<IntegerType>(cast.getResult().getType());
      if (!source || !result || source.getWidth() != result.getWidth()) return std::nullopt;
      value = cast.getInput();
      continue;
    }
    if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
      if (isa<IntegerAttr>(constant.getValue())) return constantInt(value);
    if (auto constant = value.getDefiningOp<arith::ConstantOp>())
      if (isa<IntegerAttr>(constant.getValue())) return constantInt(value);
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<unsigned> unsignedBits(Value value, unsigned depth = 0) {
  if (depth >= 8) return std::nullopt;
  auto type = dyn_cast<IntegerType>(logicalElement(value.getType()));
  if (!type) return std::nullopt;
  if (type.isUnsigned()) return type.getWidth();
  if (auto widen = value.getDefiningOp<riscv::WidenOp>()) {
    auto bits = unsignedBits(widen.getInput(), depth + 1);
    if (bits && *bits < type.getWidth()) return bits;
  }
  return std::nullopt;
}

class ContractRISCVIntegerPointwisePass
    : public PassWrapper<ContractRISCVIntegerPointwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractRISCVIntegerPointwisePass)
  StringRef getArgument() const override { return "weft-riscv-contract-integer-pointwise"; }
  StringRef getDescription() const override {
    return "Contract disjoint integer bit assembly and bias into modular RVV MAC";
  }
  void runOnOperation() override {
    SmallVector<riscv::BinaryOp> roots;
    getOperation().walk([&](riscv::BinaryOp op) {
      if (op.getKind() == "sub" || op.getKind() == "add") roots.push_back(op);
    });
    IRRewriter b(&getContext());
    unsigned selected = 0;
    for (auto root : roots) {
      auto kernel = root->getParentOfType<riscv::KernelOp>();
      auto type = dyn_cast<riscv::ValueType>(root.getResult().getType());
      auto element = type ? dyn_cast<IntegerType>(type.getElementType()) : IntegerType();
      if (!kernel || kernel.getResourcesMaterialized() || !element || element.isSignless() ||
          element.getWidth() < 8 || element.getWidth() > 64 ||
          root.getRhs().getType() != element || type.getLayout().getCarrier() != "rvv" ||
          type.getLayout().getSew() != element.getWidth() ||
          (type.getLayout().getValidity() != "full" && type.getLayout().getValidity() != "tail")) continue;
      auto joined = root.getLhs().getDefiningOp<riscv::BinaryOp>();
      if (!joined || joined.getKind() != "or" || !joined.getResult().hasOneUse() ||
          joined->getBlock() != root->getBlock() || joined.getResult().getType() != type) continue;
      for (unsigned side = 0; side < 2; ++side) {
        auto shift = joined->getOperand(side).getDefiningOp<riscv::BinaryOp>();
        Value low = joined->getOperand(1 - side);
        if (!shift || shift.getKind() != "shl" || !shift.getResult().hasOneUse() ||
            shift->getBlock() != root->getBlock() || shift.getLhs().getType() != type ||
            shift.getResult().getType() != type || low.getType() != type) continue;
        auto amount = sharedConstant(shift.getRhs());
        auto bits = unsignedBits(low);
        if (!amount || *amount <= 0 || *amount >= element.getWidth() || !bits || *bits > *amount) continue;
        bool ordered = true;
        for (Operation *op = shift->getNextNode(); op != root; op = op->getNextNode()) {
          if (op->getNumRegions() || !riscv::hasOnlyReadEffects(op)) { ordered = false; break; }
        }
        if (!ordered) continue;
        b.setInsertionPoint(root);
        auto factor = riscv::ConstantOp::create(b, root.getLoc(), element,
            b.getIntegerAttr(element, llvm::APInt(element.getWidth(), 1).shl(*amount)));
        auto biased = riscv::BinaryOp::create(b, root.getLoc(), type, low, root.getRhs(),
            root.getKind(), root.getLeaf());
        auto mac = riscv::RVVIntegerMACOp::create(b, root.getLoc(), type, shift.getLhs(),
            factor.getResult(), biased.getResult(),
            leaf(b, "rvv", "integer-mac", "rvv.vmacc.vx", "rvv.vmacc.vx",
                 2 * type.getLayout().getRegisterGroups(), type.getLayout().getRegisterGroups(),
                 0, 0, "none", "exact"));
        copyOrigin(root, biased);
        copyOrigin(root, mac);
        b.replaceOp(root, mac.getResult());
        b.eraseOp(joined);
        b.eraseOp(shift);
        ++selected;
        break;
      }
    }
    if (selected) llvm::errs() << "weft-integer-bit-assembly: selected=" << selected << '\n';
  }
};
} // namespace

std::unique_ptr<Pass> weft::createContractRISCVIntegerPointwisePass() {
  return std::make_unique<ContractRISCVIntegerPointwisePass>();
}
