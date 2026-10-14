#include "Weft/Dialect/RISCV/IR/Numerics.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cmath>
#include <optional>

using namespace mlir;
using namespace weft::riscv;

namespace {
Type elementOf(Value value) {
  if (auto type = dyn_cast<ValueType>(value.getType())) return type.getElementType();
  return value.getType();
}

std::optional<double> constant(Value value) {
  Attribute attribute;
  if (auto op = value.getDefiningOp<ConstantOp>()) attribute = op.getValue();
  else if (auto op = value.getDefiningOp<arith::ConstantOp>()) attribute = op.getValue();
  auto number = dyn_cast_or_null<FloatAttr>(attribute);
  if (!number || number.getValue().isNaN()) return std::nullopt;
  return number.getValue().convertToDouble();
}

FloatingBounds unite(FloatingBounds a, FloatingBounds b) {
  return {std::min(a.lower, b.lower), std::max(a.upper, b.upper), a.mayNaN || b.mayNaN};
}

FloatingBounds intersect(FloatingBounds a, FloatingBounds b) {
  double lower = std::max(a.lower, b.lower), upper = std::min(a.upper, b.upper);
  // An impossible ordered arm can still carry an unordered value. Keeping an
  // unknown range avoids treating an empty interval as a numeric proof.
  if (lower > upper) return {};
  return {lower, upper, a.mayNaN && b.mayNaN};
}

Value numericIdentity(Value value) {
  for (unsigned depth = 0; depth < 32; ++depth) {
    auto conversion = value.getDefiningOp<ConvertLayoutOp>();
    if (!conversion || conversion.getConversion().getEffect() != "pure") return value;
    auto source = dyn_cast<ValueType>(conversion.getInput().getType());
    auto result = dyn_cast<ValueType>(conversion.getResult().getType());
    if (!source || !result || source.getElementType() != result.getElementType() ||
        source.getShape() != result.getShape() || source.getAxisIds() != result.getAxisIds()) return value;
    value = conversion.getInput();
  }
  return value;
}

// IEEE directed rounding encloses the declared RNE result, including overflow
// and gradual underflow. Host double arithmetic is not a proof for f32 edges.
double rounded(double a, double b, StringRef operation, bool upward) {
  llvm::APFloat lhs(static_cast<float>(a)), rhs(static_cast<float>(b));
  auto direction = upward ? llvm::APFloat::rmTowardPositive : llvm::APFloat::rmTowardNegative;
  if (operation == "add") lhs.add(rhs, direction);
  else if (operation == "sub") lhs.subtract(rhs, direction);
  else if (operation == "mul") lhs.multiply(rhs, direction);
  else if (operation == "div") lhs.divide(rhs, direction);
  else return std::numeric_limits<double>::quiet_NaN();
  return lhs.convertToDouble();
}

class BoundsQuery {
public:
  FloatingBounds get(Value value) {
    if (!elementOf(value).isF32()) return {};
    if (auto found = known.find(value); found != known.end()) return found->second;
    if (!remaining) return {};
    --remaining;
    // Unknown until the producer has been closed. Cyclic loop carries do not
    // inherit a bound from only their initial value.
    known.try_emplace(value, FloatingBounds());
    auto result = calculate(value);
    known[value] = result;
    return result;
  }

private:
  unsigned remaining = 128;
  unsigned predicateRemaining = 128;
  llvm::DenseMap<Value, FloatingBounds> known;

  FloatingBounds selected(Value value, Value tested, Value threshold, StringRef predicate, bool truth) {
    auto result = get(value);
    if (numericIdentity(tested) != numericIdentity(value)) {
      if (numericIdentity(threshold) != numericIdentity(value)) return result;
      std::swap(tested, threshold);
      predicate = predicate == "lt" ? "gt" : predicate == "le" ? "ge" :
                  predicate == "gt" ? "lt" : predicate == "ge" ? "le" : predicate;
    }
    auto bound = constant(threshold);
    if (!bound || !std::isfinite(*bound)) return result;
    if (!truth)
      predicate = predicate == "lt" ? "ge" : predicate == "le" ? "gt" :
                  predicate == "gt" ? "le" : predicate == "ge" ? "lt" :
                  predicate == "eq" ? "ne" : predicate == "ne" ? "eq" : predicate;
    // Inclusive limits are sufficient even for strict comparisons. Unordered
    // inputs may select the false arm; their NaN possibility remains explicit.
    if (predicate == "gt" || predicate == "ge") result.lower = std::max(result.lower, *bound);
    if (predicate == "lt" || predicate == "le") result.upper = std::min(result.upper, *bound);
    if (predicate == "eq") result.lower = result.upper = *bound;
    if (result.lower > result.upper) return {};
    return result;
  }

  FloatingBounds selected(Value value, Value condition, bool truth, unsigned depth = 16) {
    if (!depth || !predicateRemaining) return get(value);
    --predicateRemaining;
    auto compare = condition.getDefiningOp<CompareOp>();
    if (compare) return selected(value, compare.getLhs(), compare.getRhs(), compare.getPredicate(), truth);
    if (auto mask = condition.getDefiningOp<RVVPredicateCompareOp>())
      return selected(value, mask.getLhs(), mask.getRhs(), mask.getPredicate(), truth);
    if (auto conversion = condition.getDefiningOp<ConvertLayoutOp>();
        conversion && conversion.getConversion().getEffect() == "pure")
      return selected(value, conversion.getInput(), truth, depth - 1);
    if (auto ingress = condition.getDefiningOp<RVVPredicateFromValueOp>())
      return selected(value, ingress.getInput(), truth, depth - 1);
    if (auto egress = condition.getDefiningOp<RVVPredicateToValueOp>())
      return selected(value, egress.getInput(), truth, depth - 1);
    StringRef kind;
    Value lhs, rhs;
    if (auto logic = condition.getDefiningOp<RVVPredicateLogicOp>()) {
      if (logic.getKind() == "not")
        return selected(value, logic.getInputs()[0], !truth, depth - 1);
      kind = logic.getKind();
      lhs = logic.getInputs()[0];
      rhs = logic.getInputs()[1];
    } else if (auto logic = condition.getDefiningOp<BinaryOp>()) {
      kind = logic.getKind();
      lhs = logic.getLhs();
      rhs = logic.getRhs();
    } else return get(value);
    if (kind == "and" || kind == "or") {
      auto a = selected(value, lhs, truth, depth - 1);
      auto b = selected(value, rhs, truth, depth - 1);
      // AND true / OR false constrain both predicates. The other arms can
      // satisfy either one and therefore require a union, never intersection.
      return (kind == "and") == truth ? intersect(a, b) : unite(a, b);
    }
    if (kind == "xor") {
      auto boolean = [](Value input) -> std::optional<bool> {
        Attribute attribute;
        if (auto op = input.getDefiningOp<ConstantOp>()) attribute = op.getValue();
        else if (auto op = input.getDefiningOp<arith::ConstantOp>()) attribute = op.getValue();
        auto integer = dyn_cast_or_null<IntegerAttr>(attribute);
        if (!integer || !integer.getType().isInteger(1)) return std::nullopt;
        return !integer.getValue().isZero();
      };
      if (auto scalar = boolean(rhs)) return selected(value, lhs, truth != *scalar, depth - 1);
      if (auto scalar = boolean(lhs)) return selected(value, rhs, truth != *scalar, depth - 1);
    }
    return get(value);
  }

  FloatingBounds binary(BinaryOp operation) {
    auto a = get(operation.getLhs()), b = get(operation.getRhs());
    StringRef kind = operation.getKind();
    if (kind == "min" || kind == "max") {
      // minimumNumber/maximumNumber can return the other operand when one is
      // NaN. Include that operand's complete ordered range in the result.
      FloatingBounds result = kind == "min"
          ? FloatingBounds{std::min(a.lower, b.lower), std::min(a.upper, b.upper), a.mayNaN && b.mayNaN}
          : FloatingBounds{std::max(a.lower, b.lower), std::max(a.upper, b.upper), a.mayNaN && b.mayNaN};
      if (a.mayNaN) { result.lower = std::min(result.lower, b.lower); result.upper = std::max(result.upper, b.upper); }
      if (b.mayNaN) { result.lower = std::min(result.lower, a.lower); result.upper = std::max(result.upper, a.upper); }
      return result;
    }
    if (kind == "maximum" || kind == "minimum") {
      return kind == "minimum"
          ? FloatingBounds{std::min(a.lower, b.lower), std::min(a.upper, b.upper), a.mayNaN || b.mayNaN}
          : FloatingBounds{std::max(a.lower, b.lower), std::max(a.upper, b.upper), a.mayNaN || b.mayNaN};
    }
    if (kind != "add" && kind != "sub" && kind != "mul" && kind != "div") return {};
    if (kind == "div" && b.lower <= 0 && b.upper >= 0) return {};
    if (kind == "mul" && operation.getLhs() == operation.getRhs()) {
      double bound = std::max(std::abs(a.lower), std::abs(a.upper));
      return {0, rounded(bound, bound, "mul", true), a.mayNaN};
    }
    double lower = std::numeric_limits<double>::infinity(), upper = -lower;
    for (double lhs : {a.lower, a.upper})
      for (double rhs : {b.lower, b.upper}) {
        double lo = rounded(lhs, rhs, kind, false), hi = rounded(lhs, rhs, kind, true);
        if (std::isnan(lo) || std::isnan(hi)) return {};
        lower = std::min(lower, lo); upper = std::max(upper, hi);
      }
    return {lower, upper, a.mayNaN || b.mayNaN};
  }

  FloatingBounds calculate(Value value) {
    if (auto number = constant(value)) return {*number, *number, false};
    if (auto unary = value.getDefiningOp<UnaryOp>()) {
      auto input = get(unary.getInput());
      if (unary.getKind() == "neg") return {-input.upper, -input.lower, input.mayNaN};
      if (unary.getKind() == "abs") {
        double lower = input.lower >= 0 ? input.lower : input.upper <= 0 ? -input.upper : 0;
        return {lower, std::max(std::abs(input.lower), std::abs(input.upper)), input.mayNaN};
      }
      // Use a loose enclosure for approximate exp, not an exact transcendental
      // endpoint. Nonpositive finite inputs yield [0,1] up to its ULP bound;
      // [0,2] encloses that error without relying on sampled monotonicity.
      if (unary.getKind() == "exp" || unary.getKind() == "exp2")
        return {0, input.nonpositive() ? 2 : std::numeric_limits<double>::infinity(), input.mayNaN};
      if (unary.getKind() == "sqrt") return {0, std::numeric_limits<double>::infinity(), input.mayNaN || input.lower < 0};
      return {};
    }
    if (auto op = value.getDefiningOp<BinaryOp>()) return binary(op);
    if (auto op = value.getDefiningOp<SelectOp>())
      return unite(selected(op.getTrueValue(), op.getCondition(), true), selected(op.getFalseValue(), op.getCondition(), false));
    if (auto op = value.getDefiningOp<RVVPredicateSelectOp>())
      return unite(selected(op.getTrueValue(), op.getCondition(), true), selected(op.getFalseValue(), op.getCondition(), false));
    if (auto op = value.getDefiningOp<RVVCompareSelectOp>())
      return unite(selected(op.getTrueValue(), op.getLhs(), op.getRhs(), op.getPredicate(), true),
                   selected(op.getFalseValue(), op.getLhs(), op.getRhs(), op.getPredicate(), false));
    if (auto op = value.getDefiningOp<RVVSplatOp>()) return get(op.getScalar());
    if (auto op = value.getDefiningOp<ConvertLayoutOp>(); op && op.getConversion().getEffect() == "pure")
      return get(op.getInput());
    if (auto op = value.getDefiningOp<ReshapeOp>()) return get(op.getInput());
    if (auto op = value.getDefiningOp<RVVIssueSliceOp>()) return get(op.getInput());
    return {};
  }
};
} // namespace

bool FloatingBounds::positiveNormalF32() const {
  return lower >= std::numeric_limits<float>::min() && upper <= std::numeric_limits<float>::max();
}

FloatingBounds weft::riscv::floatingBounds(Value value) { return BoundsQuery().get(value); }

llvm::StringRef weft::riscv::vectorMathInstruction(UnaryOp operation) {
  if (!elementOf(operation.getResult()).isF32()) return {};
  auto bounds = floatingBounds(operation.getInput());
  if (operation.getKind() == "exp") {
    if (bounds.lower >= -87 && bounds.upper <= 87) return "rvv.exp-normal-f32";
    if (bounds.nonpositive()) return "rvv.exp-nonpositive-f32";
    return "rvv.exp-approx-f32";
  }
  if (operation.getKind() == "log")
    return bounds.positiveNormalF32() ? "rvv.log-normal-f32" : "rvv.log-approx-f32";
  if (operation.getKind() == "exp2") return "rvv.exp2-approx-f32";
  return {};
}

bool weft::riscv::isVectorMathInstruction(llvm::StringRef instruction) {
  return instruction == "rvv.exp-approx-f32" || instruction == "rvv.exp2-approx-f32" ||
      instruction == "rvv.log-approx-f32" || instruction == "rvv.exp-normal-f32" ||
      instruction == "rvv.exp-nonpositive-f32" || instruction == "rvv.log-normal-f32";
}

bool weft::riscv::vectorMathInstructionIsLegal(UnaryOp operation, llvm::StringRef instruction) {
  if (!elementOf(operation.getResult()).isF32()) return false;
  if (instruction == "rvv.exp-approx-f32") return operation.getKind() == "exp";
  if (instruction == "rvv.exp2-approx-f32") return operation.getKind() == "exp2";
  if (instruction == "rvv.log-approx-f32") return operation.getKind() == "log";
  auto bounds = floatingBounds(operation.getInput());
  if (instruction == "rvv.exp-normal-f32") return operation.getKind() == "exp" && bounds.lower >= -87 && bounds.upper <= 87;
  if (instruction == "rvv.exp-nonpositive-f32") return operation.getKind() == "exp" && bounds.nonpositive();
  if (instruction == "rvv.log-normal-f32") return operation.getKind() == "log" && bounds.positiveNormalF32();
  return false;
}
