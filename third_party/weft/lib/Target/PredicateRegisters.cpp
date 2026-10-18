#include "PredicateRegisters.h"
#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/Predicate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace weft;

namespace {
std::optional<bool> booleanConstant(Value value) {
  Attribute attr;
  if (auto op = value.getDefiningOp<riscv::ConstantOp>()) attr = op.getValue();
  else if (auto op = value.getDefiningOp<arith::ConstantOp>()) attr = op.getValue();
  auto integer = dyn_cast_or_null<IntegerAttr>(attr);
  if (!integer || !integer.getType().isInteger(1)) return std::nullopt;
  return !integer.getValue().isZero();
}

class PredicateRegion {
public:
  explicit PredicateRegion(riscv::SelectOp first)
      : first(first), block(first->getBlock()), type(dyn_cast<riscv::ValueType>(first.getCondition().getType())) {}

  bool analyze() {
    auto parts = type ? riscv::predicatePartCount(type) : std::nullopt;
    if (!parts || *parts > 8 || !accept(first) || !collect(first.getCondition())) return false;
    selects.insert(first);
    // Loaded Booleans retain their ordinary byte representation for other
    // observers. Only the selected consumers share the ingress mask.
    for (Value boundary : boundaries)
      for (OpOperand &use : boundary.getUses())
        if (auto select = dyn_cast<riscv::SelectOp>(use.getOwner());
            select && select->getBlock() == block && use.getOperandNumber() == 0 && accept(select))
          selects.insert(select);
    for (size_t position = 0; position < nodes.size(); ++position) {
      Operation *node = nodes[position];
      for (OpOperand &use : node->getResult(0).getUses()) {
        Operation *user = use.getOwner();
        if (user->getBlock() != block) return false;
        if (auto select = dyn_cast<riscv::SelectOp>(user)) {
          if (use.getOperandNumber() != 0 || !accept(select)) return false;
          selects.insert(select);
        } else if (auto logic = dyn_cast<riscv::BinaryOp>(user)) {
          if (!collect(logic.getResult())) return false;
        } else if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(user)) {
          if (!collect(conversion.getResult())) return false;
        } else if (auto store = dyn_cast<riscv::StoreOp>(user)) {
          if (store.getValue() != node->getResult(0) || node->getResult(0).getType() != type) return false;
          observations[node].push_back(&use);
        } else return false;
      }
    }
    if (selects.empty() || selects.size() > 16 || (nodes.size() + selects.size()) * *parts > 256) return false;
    bool simple = nodes.size() == 1 && isa<riscv::CompareOp>(nodes.front()) && selects.size() == 1 &&
                  observations.empty() && *parts == 1;
    if (simple) {
      auto compare = cast<riscv::CompareOp>(nodes.front());
      auto data = dyn_cast<riscv::ValueType>(compare.getLhs().getType());
      if (!data) data = dyn_cast<riscv::ValueType>(compare.getRhs().getType());
      // The existing single compare/merge leaf already avoids Boolean bytes.
      // Retain it when both data and result use that same complete type.
      if (data && data == first.getResult().getType()) return false;
    }
    if (hasUniformLogic) {
      auto peak = riscv_internal::registerPeak(*block);
      auto target = first->getParentOfType<riscv::KernelOp>().getTarget();
      if (!peak || peak->total() + 2 * *parts > target.getVectorRegisters()) return false;
    }
    llvm::sort(nodes, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    SmallVector<riscv::SelectOp> orderedSelects(selects.begin(), selects.end());
    llvm::sort(orderedSelects, [](riscv::SelectOp a, riscv::SelectOp b) { return a->isBeforeInBlock(b); });
    selects.clear();
    selects.insert(orderedSelects.begin(), orderedSelects.end());
    return true;
  }

  LogicalResult rewrite(IRRewriter &b) {
    // Reuse an already selected pure numerical conversion when the predicate
    // consumers need its narrower coordinates. Moving that representation
    // earlier does not repeat a load, arithmetic producer or storage birth.
    for (auto [conversion, anchor] : moves)
      if (anchor->isBeforeInBlock(conversion)) conversion->moveBefore(anchor);
    DenseMap<Value, Value> masks;
    DenseMap<Value, Value> uniforms;
    DenseMap<Value, Value> complements;
    auto mask = [&](Value value, Location loc) -> Value {
      if (auto found = masks.find(value); found != masks.end()) return found->second;
      if (boundaries.contains(value)) {
        Value result = riscv::RVVPredicateFromValueOp::create(b, loc, type, value,
            leaf(b, "predicate-from-value", "rvv.predicate-from-value"));
        masks[value] = result;
        return result;
      }
      if (!riscv::uniformPredicate(value.getType())) return {};
      if (auto found = uniforms.find(value); found != uniforms.end()) return found->second;
      auto selected = riscv::RVVPredicateSplatOp::create(b, loc, type, value,
          leaf(b, "predicate-splat", "rvv.predicate-splat"));
      uniforms[value] = selected;
      return selected;
    };
    for (Operation *node : nodes) {
      b.setInsertionPoint(node);
      Value replacement;
      if (auto compare = dyn_cast<riscv::CompareOp>(node)) {
        auto operands = comparisonOperands.lookup(node);
        Value lhs = operands.first, rhs = operands.second;
        auto data = dyn_cast<riscv::ValueType>(lhs.getType());
        std::string predicate = compare.getPredicate().str();
        if (!data || data.getLayout().getCarrier() != "rvv") {
          std::swap(lhs, rhs);
          data = cast<riscv::ValueType>(lhs.getType());
          predicate = predicate == "lt" ? "gt" : predicate == "le" ? "ge" :
                      predicate == "gt" ? "lt" : predicate == "ge" ? "le" : predicate;
        }
        bool vector = rhs.getType() == data;
        std::string instruction = std::string("rvv.predicate-compare.") +
            (vector ? "vv" : isa<FloatType>(data.getElementType()) ? "vf" : "vx");
        replacement = riscv::RVVPredicateCompareOp::create(b, node->getLoc(), type, lhs, rhs, predicate,
            leaf(b, "predicate-compare", instruction));
      } else if (auto splat = dyn_cast<riscv::RVVSplatOp>(node)) {
        replacement = mask(splat.getScalar(), node->getLoc());
      } else if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(node)) {
        replacement = mask(conversion.getInput(), node->getLoc());
      } else {
        auto logic = cast<riscv::BinaryOp>(node);
        Value lhs = logic.getLhs(), rhs = logic.getRhs();
        auto right = booleanConstant(rhs), left = booleanConstant(lhs);
        if (left && !right) { std::swap(lhs, rhs); std::swap(left, right); }
        if (right) {
          if ((logic.getKind() == "and" && *right) || (logic.getKind() == "or" && !*right) ||
              (logic.getKind() == "xor" && !*right)) replacement = mask(lhs, node->getLoc());
          else if (logic.getKind() != "xor") replacement = mask(rhs, node->getLoc());
          else {
            Value input = mask(lhs, node->getLoc());
            if (!input) return node->emitError("predicate complement lost its planned input");
            replacement = riscv::RVVPredicateLogicOp::create(b, node->getLoc(), type, ValueRange{input}, "not",
                leaf(b, "predicate-logic", "rvv.predicate-not"));
          }
        } else {
          Value a = mask(lhs, node->getLoc()), c = mask(rhs, node->getLoc());
          if (!a || !c) return node->emitError("predicate logic lost its planned inputs");
          replacement = riscv::RVVPredicateLogicOp::create(b, node->getLoc(), type, ValueRange{a, c}, logic.getKind(),
              leaf(b, "predicate-logic", ("rvv.predicate-" + logic.getKind()).str()));
        }
      }
      if (!replacement) return node->emitError("predicate has no planned representation");
      if (replacement.getDefiningOp()->getLoc() == node->getLoc())
        riscv_internal::copyOrigin(node, replacement.getDefiningOp());
      masks[node->getResult(0)] = replacement;
      if (auto found = observations.find(node); found != observations.end()) {
        auto bytes = riscv::RVVPredicateToValueOp::create(b, node->getLoc(), type, replacement,
            leaf(b, "predicate-to-value", "rvv.predicate-to-value"));
        riscv_internal::copyOrigin(node, bytes);
        for (OpOperand *use : found->second) use->set(bytes.getResult());
      }
    }
    for (auto select : selects) {
      b.setInsertionPoint(select);
      auto output = cast<riscv::ValueType>(select.getResult().getType());
      Value condition = mask(select.getCondition(), select.getLoc());
      if (!condition) return select.emitError("predicate merge has no selected mask");
      Value chosen = select.getTrueValue(), otherwise = select.getFalseValue();
      // RVV merge has a vector false operand and a vector/scalar true operand.
      // Swapping a vector/scalar pair with the complemented mask saves the
      // complete data splat. The inversion is a selected mask operation and
      // stays shared when several consumers use the same scalar-false form.
      if (chosen.getType() == output && otherwise.getType() == output.getElementType()) {
        Value inverse = complements.lookup(condition);
        if (!inverse) {
          inverse = riscv::RVVPredicateLogicOp::create(b, select.getLoc(), type,
              ValueRange{condition}, "not", leaf(b, "predicate-logic", "rvv.predicate-not"));
          riscv_internal::copyOrigin(select, inverse.getDefiningOp());
          complements[condition] = inverse;
        }
        condition = inverse;
        std::swap(chosen, otherwise);
      }
      if (otherwise.getType() != output) {
        auto splat = riscv::RVVSplatOp::create(b, select.getLoc(), output, otherwise,
            riscv_internal::leaf(b, "rvv", "splat", "rvv.splat", "rvv.splat", 0, output.getLayout().getRegisterGroups()));
        riscv_internal::copyOrigin(select, splat);
        otherwise = splat;
      }
      std::string instruction = std::string("rvv.predicate-select.") +
          (chosen.getType() == output ? "vvm" : isa<FloatType>(output.getElementType()) ? "vfm" : "vxm");
      auto result = riscv::RVVPredicateSelectOp::create(b, select.getLoc(), output, condition,
          chosen, otherwise, leaf(b, "predicate-select", instruction));
      riscv_internal::copyOrigin(select, result);
      b.replaceOp(select, result.getResult());
    }
    for (Operation *node : llvm::reverse(nodes)) {
      if (!node->use_empty()) return node->emitError("closed predicate retained an ordinary Boolean consumer");
      b.eraseOp(node);
    }
    return success();
  }

  llvm::SetVector<riscv::SelectOp> selects;

private:
  riscv::SelectOp first;
  Block *block;
  riscv::ValueType type;
  SmallVector<Operation *> nodes;
  llvm::SmallPtrSet<Operation *, 32> members;
  bool hasUniformLogic = false;
  llvm::DenseSet<Value> boundaries;
  DenseMap<Operation *, std::pair<Value, Value>> comparisonOperands;
  DenseMap<Operation *, SmallVector<OpOperand *>> observations;
  SmallVector<std::pair<Operation *, Operation *>> moves;

  riscv::LeafAttr leaf(Builder &b, StringRef family, StringRef instruction) {
    return riscv_internal::leaf(b, "rvv", family, instruction, instruction, 0, 0, 0, 0,
                               "none", riscv::predicateTail(type));
  }

  bool accept(riscv::SelectOp select) {
    auto result = dyn_cast<riscv::ValueType>(select.getResult().getType());
    auto predicate = dyn_cast<riscv::ValueType>(select.getCondition().getType());
    if (!predicate || predicate.getShape() != type.getShape() || predicate.getAxisIds() != type.getAxisIds() ||
        !result || !riscv::predicateDataType(result) ||
        !riscv::predicateGeometry(type, result) || select.getLeaf().getInstruction() != "rvv.select") return false;
    for (Value branch : {select.getTrueValue(), select.getFalseValue()})
      if (branch.getType() != result && branch.getType() != result.getElementType()) return false;
    return true;
  }

  Value numericRepresentation(Value value, Operation *before) {
    auto data = dyn_cast<riscv::ValueType>(value.getType());
    if (!data) return value;
    if (data.getShape() != type.getShape() || data.getAxisIds() != type.getAxisIds()) return {};
    Builder b(value.getContext());
    auto target = before->getParentOfType<riscv::KernelOp>().getTarget();
    auto layout = riscv_internal::projectLayout(b, data, type.getLayout(), target);
    if (!layout) return {};
    auto wanted = riscv_internal::withLayout(data, layout);
    SmallVector<Value> existing{value};
    for (unsigned depth = 0; depth < existing.size() && depth < 16; ++depth) {
      Value candidate = existing[depth];
      if (candidate.getType() == wanted) return candidate;
      if (auto conversion = candidate.getDefiningOp<riscv::ConvertLayoutOp>();
          conversion && conversion.getConversion().getEffect() == "pure" && !conversion->hasAttr("birth_id"))
        existing.push_back(conversion.getInput());
      for (Operation *user : candidate.getUsers()) {
        auto conversion = dyn_cast<riscv::ConvertLayoutOp>(user);
        if (!conversion || conversion->getBlock() != block || conversion.getConversion().getEffect() != "pure" ||
            conversion.getResult().getType() != wanted || conversion->hasAttr("birth_id")) continue;
        moves.emplace_back(conversion, before);
        return conversion.getResult();
      }
    }
    return {};
  }

  bool collect(Value value) {
    if (riscv::uniformPredicate(value.getType())) { hasUniformLogic = true; return true; }
    auto numeric = dyn_cast<riscv::ValueType>(value.getType());
    if (!numeric || !numeric.getElementType().isInteger(1) || numeric.getShape() != type.getShape() ||
        numeric.getAxisIds() != type.getAxisIds()) return false;
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != block || isa<riscv::LoadOp>(op)) {
      if (value.getType() != type) return false;
      boundaries.insert(value);
      return true;
    }
    if (op->getNumRegions() || op->hasAttr("birth_id")) return false;
    if (members.contains(op)) return true;
    if (members.size() >= 32) return false;
    members.insert(op);
    if (auto compare = dyn_cast<riscv::CompareOp>(op)) {
      Value left = numericRepresentation(compare.getLhs(), compare);
      Value right = numericRepresentation(compare.getRhs(), compare);
      if (!left || !right) return false;
      comparisonOperands[op] = {left, right};
      auto lhs = dyn_cast<riscv::ValueType>(left.getType());
      Value other = right;
      if (!lhs || lhs.getLayout().getCarrier() != "rvv") {
        lhs = dyn_cast<riscv::ValueType>(right.getType()); other = left;
      }
      if (!lhs || !riscv::predicateDataType(lhs) || !riscv::predicateGeometry(type, lhs) ||
          (other.getType() != lhs && other.getType() != lhs.getElementType())) return false;
    } else if (auto logic = dyn_cast<riscv::BinaryOp>(op)) {
      if ((logic.getKind() != "and" && logic.getKind() != "or" && logic.getKind() != "xor") ||
          !collect(logic.getLhs()) || !collect(logic.getRhs())) return false;
    } else if (auto splat = dyn_cast<riscv::RVVSplatOp>(op)) {
      if (!riscv::uniformPredicate(splat.getScalar().getType())) return false;
      hasUniformLogic = true;
    } else if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op)) {
      if (conversion.getConversion().getEffect() != "pure" || !collect(conversion.getInput())) return false;
    } else return false;
    nodes.push_back(op);
    return true;
  }
};
} // namespace

LogicalResult riscv_internal::selectPredicateRegisters(ModuleOp module) {
  SmallVector<riscv::SelectOp> candidates;
  module.walk([&](riscv::SelectOp op) { candidates.push_back(op); });
  llvm::SmallPtrSet<Operation *, 32> consumed;
  DenseMap<Operation *, unsigned> attempts;
  IRRewriter b(module.getContext());
  for (auto select : candidates) {
    if (consumed.contains(select)) continue;
    PredicateRegion region(select);
    if (!region.analyze()) continue;
    auto kernel = select->getParentOfType<riscv::KernelOp>();
    if (++attempts[kernel] > 32) continue;
    {
      // A narrow numeric consumer can need several independently live masks
      // where one byte-vector predicate previously packed all coordinates.
      // Check the actual selected SSA graph, not an assumption that masks are
      // always smaller. The original byte representation remains a legal
      // candidate when the packed-mask alternative exceeds the target budget.
      OpBuilder::InsertionGuard insertion(b);
      IRMapping mapping;
      OwningOpRef<riscv::KernelOp> trial(cast<riscv::KernelOp>(kernel->clone(mapping)));
      auto copied = cast<riscv::SelectOp>(mapping.lookup(select.getResult()).getDefiningOp());
      PredicateRegion candidate(copied);
      if (!candidate.analyze()) return select.emitError("predicate candidate lost its closed region during resource evaluation");
      if (failed(candidate.rewrite(b))) return failure();
      auto peak = registerPeak(trial->getBody().front());
      if (!peak || peak->total() > kernel.getTarget().getVectorRegisters()) continue;
    }
    for (auto sink : region.selects) consumed.insert(sink);
    if (failed(region.rewrite(b))) return failure();
  }
  return success();
}
