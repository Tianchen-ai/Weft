#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace weft::riscv;

namespace {
arith::CmpIPredicate reversed(arith::CmpIPredicate predicate) {
  using P = arith::CmpIPredicate;
  switch (predicate) {
  case P::ult: return P::ugt;
  case P::ule: return P::uge;
  case P::ugt: return P::ult;
  case P::uge: return P::ule;
  case P::slt: return P::sgt;
  case P::sle: return P::sge;
  case P::sgt: return P::slt;
  case P::sge: return P::sle;
  default: return predicate;
  }
}
class ExtentQuery {
public:
  OpFoldResult get(Value memory, unsigned axis, unsigned depth = 0) {
    if (depth > 32) return {};
    auto type = dyn_cast<MemDescType>(memory.getType());
    if (!type || axis >= type.getShape().size()) return {};
    auto constant = [&](int64_t value) -> OpFoldResult {
      return IntegerAttr::get(IndexType::get(memory.getContext()), value);
    };
    if (auto view = memory.getDefiningOp<MemoryViewOp>()) return view.getExtents()[axis];
    if (auto view = memory.getDefiningOp<LocalViewOp>()) return view.getExtents()[axis];
    if (auto subview = memory.getDefiningOp<SubviewOp>()) {
      int64_t size = subview.getExtents()[axis];
      if (size >= 0) return constant(size);
      unsigned dynamic = llvm::count(subview.getOffsets(), int64_t(-1));
      for (unsigned dimension = 0; dimension < axis; ++dimension)
        dynamic += subview.getExtents()[dimension] < 0;
      return subview.getDynamicBounds()[dynamic];
    }
    if (auto rename = memory.getDefiningOp<ReaxisOp>()) return get(rename.getBase(), axis, depth + 1);
    if (auto stage = memory.getDefiningOp<StagedViewOp>()) return get(stage.getSource(), axis, depth + 1);
    if (auto slice = memory.getDefiningOp<SliceOp>()) {
      unsigned resultAxis = 0, operand = 0;
      for (unsigned sourceAxis = 0; sourceAxis < slice.getBase().getType().getShape().size(); ++sourceAxis) {
        StringRef kind = sourceAxis < slice.getSelectors().size()
            ? cast<StringAttr>(slice.getSelectors()[sourceAxis]).getValue() : StringRef("all");
        if (kind == "index" || kind == "group_index") { ++operand; continue; }
        if (kind != "all" && kind != "domain") return {};
        if (resultAxis++ != axis) {
          if (kind != "all") ++operand;
          continue;
        }
        if (kind == "all") return get(slice.getBase(), sourceAxis, depth + 1);
        if (auto point = slice.getIndices()[operand].getDefiningOp<PhysicalPointOp>())
          return point.getActive();
        return {};
      }
      return {};
    }
    int64_t size = type.getShape()[axis];
    return size >= 0 ? constant(size) : OpFoldResult{};
  }

  OpFoldResult index(OpFoldResult input, unsigned depth = 0) {
    if (!input || depth > 32) return {};
    if (auto constant = getConstantIntValue(input))
      return IntegerAttr::get(IndexType::get(input.getContext()), *constant);
    auto value = dyn_cast<Value>(input);
    if (!value) return {};
    if (auto shape = value.getDefiningOp<LocalShapeOp>()) return index(shape.getExtent(), depth + 1);
    if (auto extent = value.getDefiningOp<ExtentOp>()) {
      auto resolved = get(extent.getBase(), extent.getAxis(), depth + 1);
      return resolved ? index(resolved, depth + 1) : input;
    }
    return input;
  }
};

class BoundsQuery {
public:
  explicit BoundsQuery(Value context, Operation *use = nullptr) : use(use) {
    Operation *owner = context && context.getParentRegion() ? context.getParentRegion()->getParentOp() : nullptr;
    auto kernel = owner ? dyn_cast<KernelOp>(owner) : KernelOp{};
    if (!kernel && owner) kernel = owner->getParentOfType<KernelOp>();
    knownWidth = bool(kernel);
    if (kernel && kernel.getTarget().getMarch().starts_with("rv32"))
      maximum = std::numeric_limits<uint32_t>::max();
    arithmeticMaximum = knownWidth ? maximum : std::numeric_limits<uint32_t>::max();
    unknown = {0, maximum};
    if (use) collectConditions();
  }

  DenseIndexBounds get(Value value, unsigned depth = 0) {
    if (!value || !value.getType().isIndex() || depth > 32 || ++visits > 128) return unknown;
    auto cached = memo.find(value);
    if (cached != memo.end()) return cached->second;
    // Insert before recursion. Descriptor/control cycles cannot manufacture
    // a constant, and repeated DAG operands do not consume unbounded work.
    memo.try_emplace(value, unknown);
    DenseIndexBounds result = compute(value, depth + 1);
    if (use) result = refine(value, result, depth + 1);
    memo[value] = result;
    return result;
  }

  std::optional<bool> compare(arith::CmpIPredicate predicate, Value lhs,
                              Value rhs, unsigned depth = 0) {
    if (!lhs.getType().isIndex() || lhs.getType() != rhs.getType() || depth > 32 || ++visits > 512)
      return std::nullopt;
    lhs = resolve(lhs); rhs = resolve(rhs);
    using P = arith::CmpIPredicate;
    if (lhs == rhs) {
      switch (predicate) {
      case P::eq: case P::ule: case P::uge: case P::sle: case P::sge: return true;
      default: return false;
      }
    }
    for (auto [condition, truth] : conditions)
      if (auto comparison = condition.getDefiningOp<arith::CmpIOp>()) {
        auto p = comparison.getPredicate();
        Value a = resolve(comparison.getLhs()), b = resolve(comparison.getRhs());
        if (a == rhs && b == lhs) { std::swap(a, b); p = reversed(p); }
        if (a == lhs && b == rhs) {
          if (!truth) p = arith::invertPredicate(p);
          if (p == predicate) return true;
          if (arith::invertPredicate(p) == predicate) return false;
          if (p == P::eq) return predicate == P::ule || predicate == P::uge ||
              predicate == P::sle || predicate == P::sge;
        }
      }
    // A loop's current coordinate is below its (possibly rounded) upper
    // extent. Independent intervals lose that relation for runtime sizes;
    // retaining min(iv, extent) would also destroy a simple address recurrence.
    auto below = inductionDistance(rhs, lhs, depth + 1);
    auto above = below ? std::optional<uint64_t>{} : inductionDistance(lhs, rhs, depth + 1);
    if (below || above) {
      switch (predicate) {
      case P::eq: return false;
      case P::ne: return true;
      case P::ult: case P::ule: return bool(below);
      case P::ugt: case P::uge: return bool(above);
      default: break;
      }
    }
    auto a = get(lhs, depth + 1), b = get(rhs, depth + 1);
    // Signed comparisons are only interval comparisons when neither interval
    // straddles the sign bit. Exact bit patterns also work for negative indices.
    if (predicate == P::slt || predicate == P::sle || predicate == P::sgt || predicate == P::sge) {
      uint64_t sign = (maximum >> 1) + 1;
      if (!knownWidth || (a.lower < sign && a.upper >= sign) || (b.lower < sign && b.upper >= sign))
        return std::nullopt;
      a.lower ^= sign; a.upper ^= sign; b.lower ^= sign; b.upper ^= sign;
    }
    switch (predicate) {
    case P::eq:
      if (a.lower == a.upper && b.lower == b.upper) return a.lower == b.lower;
      if (a.upper < b.lower || b.upper < a.lower) return false;
      break;
    case P::ne:
      if (a.lower == a.upper && b.lower == b.upper) return a.lower != b.lower;
      if (a.upper < b.lower || b.upper < a.lower) return true;
      break;
    case P::ult: case P::slt:
      if (a.upper < b.lower) return true;
      if (a.lower >= b.upper) return false;
      break;
    case P::ule: case P::sle:
      if (a.upper <= b.lower) return true;
      if (a.lower > b.upper) return false;
      break;
    case P::ugt: case P::sgt:
      if (a.lower > b.upper) return true;
      if (a.upper <= b.lower) return false;
      break;
    case P::uge: case P::sge:
      if (a.lower >= b.upper) return true;
      if (a.upper < b.lower) return false;
      break;
    }
    return std::nullopt;
  }

  std::optional<bool> condition(Value value, unsigned depth = 0) {
    if (!value.getType().isInteger(1) || depth > 32 || ++visits > 512) return std::nullopt;
    if (auto fixed = getConstantIntValue(value)) return *fixed != 0;
    for (auto [fact, truth] : conditions) if (fact == value) return truth;
    if (auto comparison = value.getDefiningOp<arith::CmpIOp>())
      return compare(comparison.getPredicate(), comparison.getLhs(), comparison.getRhs(), depth + 1);
    if (auto conjunction = value.getDefiningOp<arith::AndIOp>()) {
      auto a = condition(conjunction.getLhs(), depth + 1), b = condition(conjunction.getRhs(), depth + 1);
      if ((a && !*a) || (b && !*b)) return false;
      if (a && b) return *a && *b;
    }
    if (auto disjunction = value.getDefiningOp<arith::OrIOp>()) {
      auto a = condition(disjunction.getLhs(), depth + 1), b = condition(disjunction.getRhs(), depth + 1);
      if ((a && *a) || (b && *b)) return true;
      if (a && b) return *a || *b;
    }
    if (auto negation = value.getDefiningOp<arith::XOrIOp>()) {
      auto a = condition(negation.getLhs(), depth + 1), b = condition(negation.getRhs(), depth + 1);
      if (a && b) return *a != *b;
    }
    return std::nullopt;
  }

private:
  uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t arithmeticMaximum;
  bool knownWidth;
  DenseIndexBounds unknown;
  unsigned visits = 0;
  llvm::DenseMap<Value, DenseIndexBounds> memo;
  Operation *use;
  SmallVector<std::pair<Value, bool>> conditions;

  Value resolve(Value value) {
    auto result = ExtentQuery().index(value);
    return dyn_cast_if_present<Value>(result) ? cast<Value>(result) : value;
  }

  void collectConditions() {
    Operation *child = use;
    for (Operation *parent = child->getParentOp(); parent && conditions.size() < 64;
         child = parent, parent = parent->getParentOp()) {
      auto branch = dyn_cast<scf::IfOp>(parent);
      if (!branch) continue;
      bool truth = child->getParentRegion() == &branch.getThenRegion();
      SmallVector<std::pair<Value, bool>> pending{{branch.getCondition(), truth}};
      while (!pending.empty() && conditions.size() < 64) {
        auto fact = pending.pop_back_val();
        conditions.push_back(fact);
        if (auto andOp = fact.first.getDefiningOp<arith::AndIOp>(); andOp && fact.second) {
          pending.emplace_back(andOp.getLhs(), true);
          pending.emplace_back(andOp.getRhs(), true);
        } else if (auto orOp = fact.first.getDefiningOp<arith::OrIOp>(); orOp && !fact.second) {
          pending.emplace_back(orOp.getLhs(), false);
          pending.emplace_back(orOp.getRhs(), false);
        }
      }
    }
  }

  DenseIndexBounds refine(Value value, DenseIndexBounds bounds, unsigned depth) {
    Value identity = resolve(value);
    for (auto [condition, truth] : conditions) {
      auto comparison = condition.getDefiningOp<arith::CmpIOp>();
      if (!comparison || !comparison.getLhs().getType().isIndex()) continue;
      Value lhs = resolve(comparison.getLhs()), rhs = resolve(comparison.getRhs());
      auto predicate = comparison.getPredicate();
      if (!truth) predicate = arith::invertPredicate(predicate);
      if (rhs == identity) { std::swap(lhs, rhs); predicate = reversed(predicate); }
      if (lhs != identity || rhs == identity) continue;
      auto other = get(rhs, depth + 1);
      auto narrowed = bounds;
      switch (predicate) {
      case arith::CmpIPredicate::eq:
        narrowed.lower = std::max(bounds.lower, other.lower);
        narrowed.upper = std::min(bounds.upper, other.upper);
        break;
      case arith::CmpIPredicate::uge: narrowed.lower = std::max(bounds.lower, other.lower); break;
      case arith::CmpIPredicate::ugt:
        if (other.lower != maximum) narrowed.lower = std::max(bounds.lower, other.lower + 1);
        break;
      case arith::CmpIPredicate::ule: narrowed.upper = std::min(bounds.upper, other.upper); break;
      case arith::CmpIPredicate::ult:
        if (other.upper) narrowed.upper = std::min(bounds.upper, other.upper - 1);
        break;
      default: break;
      }
      // Do not exploit inconsistent paths to invent arbitrary values. Empty
      // branches are removed by the control owner from their own predicate.
      if (narrowed.lower <= narrowed.upper) bounds = narrowed;
    }
    return bounds;
  }

  DenseIndexBounds induction(BlockArgument argument, unsigned depth) {
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument != loop.getInductionVar() ||
        (use && !loop->isProperAncestor(use))) return unknown;
    auto direction = loop->getAttrOfType<StringAttr>("weft.riscv.direction");
    auto level = loop->getAttrOfType<LevelAttr>("weft.riscv.level");
    if (level ? level.getDirection() != "ascending" : !direction || direction.getValue() != "ascending")
      return unknown;
    auto lower = get(loop.getLowerBound(), depth), upper = get(loop.getUpperBound(), depth);
    auto step = get(loop.getStep(), depth);
    if (!step.lower || step.upper > (maximum >> 1) || !upper.upper ||
        lower.upper > (maximum >> 1) || lower.lower >= upper.upper) return unknown;
    uint64_t last = upper.upper - 1;
    uint64_t first = lower.lower;
    if (lower.lower == lower.upper && upper.lower == upper.upper && step.lower == step.upper) {
      uint64_t alignedLast = first + ((last - first) / step.lower) * step.lower;
      if (step.upper <= maximum - alignedLast) return {first, alignedLast};
    } else if (step.upper <= maximum - last) return {first, last};
    // An unsigned physical induction can wrap before its next exit check.
    // Only the loop condition still bounds such later iterations; neither the
    // initial lower bound nor its original congruence is an invariant then.
    return {0, last};
  }

  std::optional<uint64_t> inductionDistance(Value bound, Value coordinate, unsigned depth) {
    coordinate = resolve(coordinate); bound = resolve(bound);
    auto argument = dyn_cast<BlockArgument>(coordinate);
    auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
    if (!loop || argument != loop.getInductionVar()) return std::nullopt;
    auto direction = loop->getAttrOfType<StringAttr>("weft.riscv.direction");
    auto level = loop->getAttrOfType<LevelAttr>("weft.riscv.level");
    if (level ? level.getDirection() != "ascending" : !direction || direction.getValue() != "ascending")
      return std::nullopt;
    Value upper = resolve(loop.getUpperBound());
    if (upper == bound) return 1;
    // A full-issue loop starts at a step multiple and ends at floor(E/S)*S.
    // Its current issue is entirely within E even when E is dynamic. This
    // proves active length, not the safety of moving any memory read.
    auto product = upper.getDefiningOp<arith::MulIOp>();
    if (!product) return std::nullopt;
    for (unsigned side : {0u, 1u}) {
      auto factor = get(product->getOperand(side), depth);
      auto quotient = resolve(product->getOperand(1 - side)).getDefiningOp<arith::DivUIOp>();
      if (!factor.lower || factor.lower != factor.upper || !quotient ||
          resolve(quotient.getLhs()) != bound) continue;
      auto divisor = get(quotient.getRhs(), depth);
      if (divisor.lower != factor.lower || divisor.upper != factor.upper) continue;
      auto lower = get(loop.getLowerBound(), depth), step = get(loop.getStep(), depth);
      if (lower.lower == lower.upper && lower.upper <= (maximum >> 1) &&
          step.lower == factor.lower && step.upper == factor.upper &&
          step.upper <= (maximum >> 1) && lower.lower % step.lower == 0) return step.lower;
      return 1;
    }
    return std::nullopt;
  }

  DenseIndexBounds constant(int64_t value) const {
    uint64_t bits = uint64_t(value);
    if (!knownWidth && bits > arithmeticMaximum) return unknown;
    bits &= maximum;
    return {bits, bits};
  }

  DenseIndexBounds extent(Value memory, unsigned axis, unsigned depth) {
    auto resolved = ExtentQuery().get(memory, axis, depth);
    if (!resolved) return unknown;
    if (auto value = getConstantIntValue(resolved)) return constant(*value);
    auto value = dyn_cast<Value>(resolved);
    return value ? get(value, depth + 1) : unknown;
  }

  DenseIndexBounds compute(Value value, unsigned depth) {
    if (auto fixed = getConstantIntValue(value)) return constant(*fixed);
    if (auto argument = dyn_cast<BlockArgument>(value)) return induction(argument, depth);
    if (auto shape = value.getDefiningOp<LocalShapeOp>()) return get(shape.getExtent(), depth);
    if (auto size = value.getDefiningOp<ExtentOp>()) return extent(size.getBase(), size.getAxis(), depth);
    if (auto op = value.getDefiningOp<arith::MinUIOp>()) {
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      return {std::min(a.lower, b.lower), std::min(a.upper, b.upper)};
    }
    if (auto op = value.getDefiningOp<arith::MaxUIOp>()) {
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      return {std::max(a.lower, b.lower), std::max(a.upper, b.upper)};
    }
    if (auto op = value.getDefiningOp<arith::AddIOp>()) {
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      if (a.upper > arithmeticMaximum || b.upper > arithmeticMaximum - a.upper) return unknown;
      return {a.lower + b.lower, a.upper + b.upper};
    }
    if (auto op = value.getDefiningOp<arith::MulIOp>()) {
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      if (!a.upper || !b.upper) return {0, 0};
      if (b.upper > arithmeticMaximum / a.upper) return unknown;
      return {a.lower * b.lower, a.upper * b.upper};
    }
    if (auto op = value.getDefiningOp<arith::SubIOp>()) {
      if (op.getLhs() == op.getRhs()) return {0, 0};
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      if (a.lower >= b.upper) return {a.lower - b.upper, a.upper - b.lower};
      if (auto distance = inductionDistance(op.getLhs(), op.getRhs(), depth); distance && *distance <= a.upper)
        return {*distance, a.upper};
      // The clamped subtraction used for an actual rectangular remainder is
      // nonnegative even though independent intervals cannot establish it.
      if (auto minimum = op.getRhs().getDefiningOp<arith::MinUIOp>(); minimum &&
          (resolve(minimum.getLhs()) == resolve(op.getLhs()) || resolve(minimum.getRhs()) == resolve(op.getLhs()))) {
        Value other = resolve(minimum.getLhs()) == resolve(op.getLhs()) ? minimum.getRhs() : minimum.getLhs();
        if (auto distance = inductionDistance(op.getLhs(), other, depth); distance && *distance <= a.upper)
          return {*distance, a.upper};
        return {0, a.upper};
      }
      return unknown;
    }
    if (auto op = value.getDefiningOp<arith::DivUIOp>()) {
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      if (!b.lower) return unknown;
      return {a.lower / b.upper, a.upper / b.lower};
    }
    if (auto op = value.getDefiningOp<arith::RemUIOp>()) {
      auto a = get(op.getLhs(), depth), b = get(op.getRhs(), depth);
      if (!b.lower) return unknown;
      if (a.upper < b.lower) return a;
      if (a.lower == a.upper && b.lower == b.upper) return {a.lower % b.lower, a.lower % b.lower};
      return {0, std::min(a.upper, b.upper - 1)};
    }
    if (auto op = value.getDefiningOp<arith::SelectOp>()) {
      if (auto truth = condition(op.getCondition(), depth + 1))
        return get(*truth ? op.getTrueValue() : op.getFalseValue(), depth);
      auto a = get(op.getTrueValue(), depth), b = get(op.getFalseValue(), depth);
      return {std::min(a.lower, b.lower), std::max(a.upper, b.upper)};
    }
    return unknown;
  }
};
} // namespace

std::optional<int64_t> DenseIndexBounds::constant() const {
  if (lower != upper || upper > uint64_t(std::numeric_limits<int64_t>::max())) return std::nullopt;
  return int64_t(upper);
}

DenseIndexBounds weft::riscv::denseIndexBounds(Value value) {
  return BoundsQuery(value).get(value);
}

DenseIndexBounds weft::riscv::denseIndexBoundsAt(Value value, Operation *use) {
  return BoundsQuery(value, use).get(value);
}

std::optional<bool> weft::riscv::denseIndexCompareAt(arith::CmpIPredicate predicate,
    Value lhs, Value rhs, Operation *use) {
  return BoundsQuery(lhs, use).compare(predicate, lhs, rhs);
}

std::optional<bool> weft::riscv::denseConditionAt(Value condition, Operation *use) {
  return BoundsQuery(condition, use).condition(condition);
}

std::optional<int64_t> weft::riscv::denseConstantIndex(Value value) {
  return denseIndexBounds(value).constant();
}

OpFoldResult weft::riscv::denseDescriptorExtent(Value memory, unsigned axis) {
  ExtentQuery query;
  return query.index(query.get(memory, axis));
}

bool weft::riscv::sameDenseExtent(Value extent, Value memory, unsigned axis) {
  if (auto original = extent.getDefiningOp<ExtentOp>(); original &&
      original.getBase() == memory && original.getAxis() == axis) return true;
  ExtentQuery query;
  auto actual = query.index(extent), expected = denseDescriptorExtent(memory, axis);
  if (!actual || !expected) return false;
  if (actual == expected) return true;
  auto lhs = dyn_cast<Value>(actual), rhs = dyn_cast<Value>(expected);
  return lhs && rhs && sameDenseIndex(lhs, rhs);
}

bool weft::riscv::sameDenseIndex(Value first, Value second) {
  ExtentQuery query;
  unsigned budget = 64;
  auto equal = [&](auto &&self, Value lhs, Value rhs, unsigned depth) -> bool {
    if (lhs == rhs) return true;
    if (!budget || depth > 16) return false;
    --budget;
    auto a = query.index(lhs), b = query.index(rhs);
    if (!a || !b) return false;
    if (a == b) return true;
    auto x = dyn_cast<Value>(a), y = dyn_cast<Value>(b);
    if (!x || !y || x.getType() != y.getType()) return false;
    Operation *left = x.getDefiningOp(), *right = y.getDefiningOp();
    if (!left || !right || left->getName() != right->getName() ||
        left->getAttrs() != right->getAttrs() || left->getNumOperands() != right->getNumOperands() ||
        !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::MinUIOp, arith::MaxUIOp>(left)) return false;
    // Compare identical bounded index expressions, not arbitrary pure graphs.
    // In particular this never follows a load or assumes integer no-overflow.
    for (auto [a, b] : llvm::zip(left->getOperands(), right->getOperands()))
      if (!self(self, a, b, depth + 1)) return false;
    return true;
  };
  return equal(equal, first, second, 0);
}

OpFoldResult ExtentOp::fold(FoldAdaptor) {
  // This preserves the descriptor's actual bound, including cropped views.
  // No address calculation, memory read or Level identity is moved.
  return denseDescriptorExtent(getBase(), getAxis());
}
