#include "PointerMemory.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace weft::riscv_internal {
namespace {
// An affine relation in the original integer format. Widening is legal only
// when the whole source interval cannot cross a wrap/sign boundary. The base
// multiple is integer divisibility, not an assumption about storage alignment.
struct OffsetRelation {
  int64_t stride = 0;
  uint64_t multiple = 1;
  std::optional<std::pair<int64_t, int64_t>> range;
};
class PointerOffsets {
  mlir::IRRewriter &b;
  int64_t extent;
  unsigned remaining = 64;
  llvm::DenseMap<mlir::Value, std::optional<OffsetRelation>> facts;
  llvm::DenseMap<mlir::Value, mlir::Value> bases;
  static mlir::IntegerType integer(mlir::Type type) {
    return mlir::dyn_cast<mlir::IntegerType>(logicalElement(type));
  }
  static uint64_t multiple(int64_t value) {
    return value ? uint64_t(1) << std::min<unsigned>(62, llvm::countr_zero(uint64_t(value))) : uint64_t(1) << 62;
  }
  static bool fits(__int128 low, __int128 high, mlir::IntegerType type) {
    if (!type || type.getWidth() > 64) return false;
    unsigned width = type.getWidth();
    __int128 lower = type.isUnsigned() ? 0 : -(__int128(1) << (width - 1));
    __int128 upper = (__int128(1) << (type.isUnsigned() ? width : width - 1)) - 1;
    return low >= lower && high <= upper && low >= INT64_MIN && high <= INT64_MAX;
  }
  std::optional<OffsetRelation> compute(mlir::Value value) {
    if (!remaining) return std::nullopt;
    --remaining;
    auto type = integer(value.getType());
    auto shaped = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!type) return !shaped && value.getType().isIndex() ? std::optional<OffsetRelation>(OffsetRelation{}) : std::nullopt;
    if (shaped && (shaped.getShape().size() != 1 || shaped.getShape()[0] != extent)) return std::nullopt;
    auto op = value.getDefiningOp();
    mlir::IntegerAttr literal;
    if (auto c = mlir::dyn_cast_or_null<riscv::ConstantOp>(op)) literal = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue());
    if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(op)) literal = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue());
    if (literal && literal.getValue().getBitWidth() <= 64) {
      if (type.isUnsigned() && literal.getValue().getZExtValue() > INT64_MAX) return OffsetRelation{};
      int64_t n = type.isUnsigned() ? literal.getValue().getZExtValue() : literal.getValue().getSExtValue();
      return OffsetRelation{0, multiple(n), {{n, n}}};
    }
    if (auto iota = mlir::dyn_cast_or_null<riscv::IotaOp>(op))
      return OffsetRelation{1, multiple(iota.getStart()), {{iota.getStart(), iota.getEnd() - 1}}};
    if (auto initial = mlir::dyn_cast_or_null<riscv::NewOp>(op); initial && initial.getInitial()) return analyze(initial.getInitial());
    if (auto layout = mlir::dyn_cast_or_null<riscv::ConvertLayoutOp>(op); layout && layout.getConversion().getEffect() == "pure")
      return analyze(layout.getInput());
    if (auto reshape = mlir::dyn_cast_or_null<riscv::ReshapeOp>(op); reshape && reshape.getInput().getType().getShape() == shaped.getShape())
      return analyze(reshape.getInput());
    if (mlir::isa_and_nonnull<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp>(op)) {
      if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(op); narrow && narrow.getSaturate()) return std::nullopt;
      auto input = analyze(op->getOperand(0));
      auto before = integer(op->getOperand(0).getType());
      if (!input || !before) return !shaped ? std::optional<OffsetRelation>(OffsetRelation{}) : std::nullopt;
      bool known = input->range && fits(input->range->first, input->range->second, before) &&
                   fits(input->range->first, input->range->second, type);
      uint64_t boundary = uint64_t(1) << std::min<unsigned>(62, std::min(before.getWidth(), type.getWidth()) - 1);
      bool aligned = input->stride >= 0 && __int128(input->stride) * (extent - 1) < std::min(boundary, input->multiple);
      if (input->stride && !known && !aligned) return std::nullopt;
      if (!known) input->range.reset();
      input->multiple = std::min(input->multiple, boundary);
      return input;
    }
    if (auto binary = mlir::dyn_cast_or_null<riscv::BinaryOp>(op)) {
      auto lhs = analyze(binary.getLhs()), rhs = analyze(binary.getRhs());
      if (!lhs || !rhs) return std::nullopt;
      OffsetRelation result;
      __int128 delta;
      auto kind = binary.getKind();
      if (kind == "add" || kind == "sub") {
        delta = __int128(lhs->stride) + (kind == "add" ? __int128(rhs->stride) : -__int128(rhs->stride));
        result.multiple = std::min(lhs->multiple, rhs->multiple);
      } else if (kind == "mul") {
        auto constant = [](const OffsetRelation &x) { return x.range && x.range->first == x.range->second; };
        if (!constant(*rhs)) std::swap(lhs, rhs);
        if (!constant(*rhs)) return !shaped ? std::optional<OffsetRelation>(OffsetRelation{}) : std::nullopt;
        delta = __int128(lhs->stride) * rhs->range->first;
        result.multiple = uint64_t(std::min<__int128>(__int128(lhs->multiple) * rhs->multiple, __int128(1) << 62));
      } else return !shaped ? std::optional<OffsetRelation>(OffsetRelation{}) : std::nullopt;
      if (delta < INT64_MIN || delta > INT64_MAX) return std::nullopt;
      result.stride = int64_t(delta);
      if (lhs->range && rhs->range) {
        auto [a, z] = *lhs->range; auto [c, d] = *rhs->range;
        __int128 low, high;
        if (kind == "add") { low = __int128(a) + c; high = __int128(z) + d; }
        else if (kind == "sub") { low = __int128(a) - d; high = __int128(z) - c; }
        else { low = std::min(__int128(a) * c, __int128(z) * c); high = std::max(__int128(a) * c, __int128(z) * c); }
        if (fits(low, high, type)) result.range = {{int64_t(low), int64_t(high)}};
      }
      return result;
    }
    return !shaped ? std::optional<OffsetRelation>(OffsetRelation{}) : std::nullopt;
  }
public:
  PointerOffsets(mlir::IRRewriter &b, int64_t extent) : b(b), extent(extent) {}
  std::optional<OffsetRelation> analyze(mlir::Value value) {
    if (auto found = facts.find(value); found != facts.end()) return found->second;
    auto result = compute(value); facts[value] = result; return result;
  }
  mlir::Value scalarBase(mlir::Value value) {
    if (!mlir::isa<riscv::ValueType>(value.getType())) return value;
    if (auto found = bases.find(value); found != bases.end()) return found->second;
    auto op = value.getDefiningOp();
    mlir::Value result;
    if (auto iota = mlir::dyn_cast<riscv::IotaOp>(op)) {
      result = riscv::ConstantOp::create(b, op->getLoc(), integer(value.getType()), b.getIntegerAttr(integer(value.getType()), iota.getStart()));
      copyOrigin(op, result.getDefiningOp());
    } else if (mlir::isa<riscv::NewOp, riscv::ConvertLayoutOp, riscv::ReshapeOp>(op)) result = scalarBase(op->getOperand(0));
    else {
      mlir::IRMapping mapping;
      for (auto operand : op->getOperands()) mapping.map(operand, scalarBase(operand));
      auto clone = b.clone(*op, mapping);
      clone->getResult(0).setType(integer(value.getType()));
      if (clone->hasAttr("leaf")) clone->setAttr("leaf", unselectedLeaf(b));
      result = clone->getResult(0);
    }
    bases[value] = result; return result;
  }
};

class SpecializePointerOffsetsPass : public mlir::PassWrapper<SpecializePointerOffsetsPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializePointerOffsetsPass)
  llvm::StringRef getArgument() const override { return "weft-riscv-specialize-pointer-offsets"; }
  llvm::StringRef getDescription() const override { return "Prove affine pointer domains before selecting numeric storage"; }
  void runOnOperation() override {
    mlir::IRRewriter b(&getContext());
    bool specialized = false;
    getOperation().walk([&](mlir::Operation *op) {
      if (!mlir::isa<riscv::PointerLoadOp, riscv::PointerStoreOp>(op) || op->hasAttr("element_stride")) return;
      auto payload = mlir::dyn_cast<riscv::ValueType>((op->getNumResults() ? op->getResult(0) : op->getOperand(3)).getType());
      if (!payload || payload.getShape().size() != 1 || payload.getShape()[0] <= 0) return;
      PointerOffsets analysis(b, payload.getShape()[0]);
      auto relation = analysis.analyze(op->getOperand(1));
      int64_t bytes = std::max<int64_t>(1, logicalBitWidth(payload.getElementType()) / 8);
      if (!relation || !relation->stride || relation->stride > INT64_MAX / bytes || relation->stride < INT64_MIN / bytes) return;
      b.setInsertionPoint(op);
      op->setOperand(1, analysis.scalarBase(op->getOperand(1)));
      op->setAttr("element_stride", b.getI64IntegerAttr(relation->stride));
      specialized = true;
    });
    if (!specialized) return;
    // Dead shaped address chains must disappear before layout/resource choices
    // can turn their scalar broadcasts into unnecessary local arrays.
    bool changed;
    do {
      changed = false;
      getOperation().walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *op) {
        if (mlir::isOpTriviallyDead(op)) { b.eraseOp(op); changed = true; }
      });
    } while (changed);
  }
};
}
mlir::LogicalResult selectPointerMemory(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  bool invalid = false;
  module.walk([&](mlir::Operation *op) {
    if (!mlir::isa<riscv::PointerLoadOp, riscv::PointerStoreOp>(op) || invalid) return;
    if (op->getAttrOfType<riscv::LeafAttr>("leaf").getEngine() != "unselected") return;
    auto kernel = op->getParentOfType<riscv::KernelOp>();
    // Cache/eviction modifiers are advisory source hints. The RISC-V memory
    // instructions selected here have ordinary coherent-cache behavior.
    op->removeAttr("cache_hint");
    op->removeAttr("eviction_hint");
    auto uniform = [](mlir::Value value) -> mlir::Value {
      while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
        if (conversion.getConversion().getEffect() != "pure") return {};
        value = conversion.getInput();
      }
      if (auto initial = value.getDefiningOp<riscv::NewOp>(); initial &&
          initial.getInitial() && !mlir::isa<riscv::ValueType>(initial.getInitial().getType()))
        return initial.getInitial();
      return {};
    };
    if (auto scalar = uniform(op->getOperand(2))) op->setOperand(2, scalar);
    if (op->getNumResults())
      if (auto scalar = uniform(op->getOperand(3))) op->setOperand(3, scalar);
    rewriter.setInsertionPoint(op);
    auto payload = mlir::dyn_cast<riscv::ValueType>((op->getNumResults() ? op->getResult(0) : op->getOperand(3)).getType());
    auto stride = op->getAttrOfType<mlir::IntegerAttr>("element_stride");
    auto offsets = stride ? payload : mlir::dyn_cast<riscv::ValueType>(op->getOperand(1).getType());
    // The value-program owner materializes local-backed pointer operations
    // into bounded issues. Select their memory leaves after that rewrite.
    if ((payload && payload.getLayout().getCarrier() == "local") ||
        (offsets && offsets.getLayout().getCarrier() == "local")) return;
    bool vector = offsets && offsets.getLayout().getCarrier() == "rvv";
    if (offsets && !vector && offsets.getLayout().getCarrier() != "scalar") {
      op->emitError("shaped pointer offsets require an RVV memory representation");
      invalid = true;
      return;
    }
    rewriter.setInsertionPoint(op);
    auto convert = [&](mlir::Value input) -> mlir::Value {
      auto type = mlir::dyn_cast<riscv::ValueType>(input.getType());
      if (!type) return input;
      auto layout = projectLayout(rewriter, type, offsets.getLayout(), kernel.getTarget());
      if (!layout) return {};
      auto required = mlir::cast<riscv::ValueType>(withLayout(type, layout));
      if (required == type) return input;
      auto conversionOp = riscv::ConvertLayoutOp::create(rewriter, op->getLoc(), required, input,
          layoutConversion(rewriter, type.getLayout(), layout), riscv::AccessAttr(), unselectedLeaf(rewriter));
      copyOrigin(op, conversionOp);
      return conversionOp.getResult();
    };
    if (offsets) {
      for (unsigned operand : {2u, 3u}) {
        auto value = convert(op->getOperand(operand));
        if (!value) { invalid = true; op->emitError("pointer operand layout cannot preserve active coordinates"); return; }
        op->setOperand(operand, value);
      }
      if (op->getNumResults()) {
        auto result = op->getResult(0);
        auto original = mlir::cast<riscv::ValueType>(result.getType());
        auto layout = projectLayout(rewriter, original, offsets.getLayout(), kernel.getTarget());
        if (!layout) { invalid = true; op->emitError("pointer result has no coordinate-preserving layout"); return; }
        auto required = withLayout(original, layout);
        if (original != required) {
          result.setType(required);
          rewriter.setInsertionPointAfter(op);
          auto converted = riscv::ConvertLayoutOp::create(rewriter, op->getLoc(), original, result,
              layoutConversion(rewriter, mlir::cast<riscv::ValueType>(required).getLayout(), original.getLayout()),
              riscv::AccessAttr(), unselectedLeaf(rewriter));
          copyOrigin(op, converted);
          result.replaceAllUsesExcept(converted.getResult(), converted);
        }
      }
    }
    std::string form = vector ? (stride ? (stride.getInt() == 1 ? "rvv.pointer-unit." : "rvv.pointer-strided.") : "rvv.pointer-indexed.") : "scalar.pointer-masked.";
    form += op->getNumResults() ? "load" : "store";
    int64_t temporary = vector ? 1 + (stride ? 0 : (offsets.getLayout().getLmulEighths() + 7) / 8) : 0;
    if (vector && op->getNumResults() && !mlir::isa<riscv::ValueType>(op->getOperand(3).getType()))
      temporary += (mlir::cast<riscv::ValueType>(op->getResult(0).getType()).getLayout().getLmulEighths() + 7) / 8;
    int64_t operands = 0, results = 0;
    auto groups = [](mlir::Type type) {
      auto value = mlir::dyn_cast<riscv::ValueType>(type);
      return value && value.getLayout().getCarrier() == "rvv" ? value.getLayout().getRegisterGroups() : 0;
    };
    for (auto type : op->getOperandTypes()) operands += groups(type);
    for (auto type : op->getResultTypes()) results += groups(type);
    op->setAttr("leaf", leaf(rewriter, vector ? "rvv" : "scalar", "pointer-memory", form, form,
                             operands, results, temporary, 0, "explicit", vector && op->getNumResults() ? "undisturbed" : "exact"));
  });
  return mlir::failure(invalid);
}
}

std::unique_ptr<mlir::Pass> weft::createSpecializeRISCVPointerOffsetsPass() {
  return std::make_unique<riscv_internal::SpecializePointerOffsetsPass>();
}
