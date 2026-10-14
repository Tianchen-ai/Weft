#include "Weft/Dialect/RISCV/IR/Predicate.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Verification.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace weft::riscv;

bool weft::riscv::isPredicateRegister(Value value) {
  auto *producer = value.getDefiningOp();
  return producer && isa<RVVPredicateCompareOp, RVVPredicateLogicOp, RVVPredicateSplatOp,
                         RVVPredicateFromValueOp>(producer);
}

bool weft::riscv::uniformPredicate(Type type) {
  if (type.isInteger(1)) return true;
  auto value = dyn_cast<ValueType>(type);
  return value && value.getElementType().isInteger(1) && value.getLayout().getCarrier() == "scalar" &&
      llvm::all_of(value.getShape().asArrayRef(), [](int64_t n) { return n == 1; });
}

std::optional<int64_t> weft::riscv::predicatePartCount(ValueType type) {
  if (!type.getElementType().isInteger(1)) return std::nullopt;
  auto layout = type.getLayout();
  auto coordinates = RegisterCoordinates::get(layout);
  if (layout.getCarrier() != "rvv" || layout.getReplicaGroup() != 1 || layout.getSew() != 8 ||
      layout.getLmulEighths() <= 0 || !coordinates || coordinates->parts() > 64 ||
      coordinates->lanes() != layout.getVl() ||
      (layout.getValidity() != "full" && layout.getValidity() != "tail")) return std::nullopt;
  int64_t ratio = layout.getSew() * 8 / layout.getLmulEighths();
  if (ratio < 1 || ratio > 64 || (ratio & (ratio - 1))) return std::nullopt;
  return coordinates->parts();
}

std::optional<int64_t> weft::riscv::predicateIssueGroups(ValueType type) {
  auto parts = predicatePartCount(type);
  auto coordinates = RegisterCoordinates::get(type.getLayout());
  if (!parts || !coordinates || !coordinates->streams() || *parts % coordinates->streams()) return std::nullopt;
  return *parts / coordinates->streams();
}

bool weft::riscv::predicateGeometry(ValueType predicate, ValueType data) {
  if (!predicatePartCount(predicate) || predicate.getShape() != data.getShape() ||
      predicate.getAxisIds() != data.getAxisIds()) return false;
  auto p = predicate.getLayout(), d = data.getLayout();
  if (d.getCarrier() != "rvv" || d.getReplicaGroup() != 1 || d.getLmulEighths() <= 0) return false;
  return p.getTimeFactors() == d.getTimeFactors() && p.getLaneFactors() == d.getLaneFactors() &&
      p.getReplicaFactors() == d.getReplicaFactors() && p.getFragmentFactors() == d.getFragmentFactors() &&
      p.getLocalFactors() == d.getLocalFactors() && p.getVl() == d.getVl() && p.getValidity() == d.getValidity() &&
      p.getSew() * d.getLmulEighths() == d.getSew() * p.getLmulEighths();
}

llvm::StringRef weft::riscv::predicateTail(ValueType type) {
  return type.getLayout().getValidity() == "tail" ? "agnostic" : "exact";
}

bool weft::riscv::predicateDataType(ValueType type) {
  auto integer = dyn_cast<IntegerType>(type.getElementType());
  return type.getElementType().isF32() || type.getElementType().isF64() ||
      (integer && !integer.isSignless() && (integer.getWidth() == 8 || integer.getWidth() == 16 ||
                                          integer.getWidth() == 32 || integer.getWidth() == 64));
}

namespace {

LogicalResult maskResult(Operation *operation, Value result) {
  auto kernel = operation->getParentOfType<KernelOp>();
  auto type = cast<ValueType>(result.getType());
  if (!kernel || !predicatePartCount(type) || !supportsRVVLayout(kernel.getTarget(), type.getLayout()))
    return operation->emitOpError("predicate requires a legal complete RVV coordinate mapping");
  for (OpOperand &use : result.getUses()) {
    auto *user = use.getOwner();
    bool accepted = isa<RVVPredicateLogicOp, RVVPredicateToValueOp>(user) ||
        (isa<RVVPredicateSelectOp>(user) && use.getOperandNumber() == 0);
    if (!accepted || user->getBlock() != operation->getBlock())
      return operation->emitOpError("packed predicate can only feed selected mask consumers in the same block");
  }
  return success();
}

bool exact(LeafAttr leaf, StringRef family, StringRef instruction, ValueType type) {
  return weft::riscv::detail::exactLeaf(leaf, "rvv", family, instruction, "none", predicateTail(type)) &&
      leaf.getTemporaryGroups() == 0 && leaf.getFragmentGroups() == 0 &&
      leaf.getLocalBytes() == 0 && leaf.getParameters().empty();
}

bool maskInput(Operation *operation, Value value, ValueType expected) {
  return value.getType() == expected && isPredicateRegister(value) &&
      value.getDefiningOp()->getBlock() == operation->getBlock();
}
} // namespace

LogicalResult RVVPredicateCompareOp::verify() {
  auto type = getLhs().getType(), mask = getResult().getType();
  bool vectorRHS = getRhs().getType() == type;
  bool floating = isa<FloatType>(type.getElementType());
  StringRef predicates[] = {"eq", "ne", "lt", "le", "gt", "ge"};
  std::string instruction = std::string("rvv.predicate-compare.") + (vectorRHS ? "vv" : floating ? "vf" : "vx");
  if (!predicateDataType(type) || !predicateGeometry(mask, type) ||
      (!vectorRHS && getRhs().getType() != type.getElementType()) ||
      !llvm::is_contained(predicates, getPredicate()) || !exact(getLeaf(), "predicate-compare", instruction, mask))
    return emitOpError("predicate comparison requires exact data geometry, predicate and operand form");
  return maskResult(*this, getResult());
}

LogicalResult RVVPredicateSplatOp::verify() {
  if (!uniformPredicate(getScalar().getType()) ||
      !exact(getLeaf(), "predicate-splat", "rvv.predicate-splat", getResult().getType()))
    return emitOpError("predicate splat requires one scalar Boolean and the exact mask leaf");
  return maskResult(*this, getResult());
}

LogicalResult RVVPredicateFromValueOp::verify() {
  if (getInput().getType() != getResult().getType() || isPredicateRegister(getInput()) ||
      !exact(getLeaf(), "predicate-from-value", "rvv.predicate-from-value", getResult().getType()))
    return emitOpError("mask ingress requires an existing Boolean byte value with identical coordinates");
  return maskResult(*this, getResult());
}

LogicalResult RVVPredicateToValueOp::verify() {
  auto type = getResult().getType();
  if (!predicatePartCount(type) || !maskInput(*this, getInput(), type) ||
      !exact(getLeaf(), "predicate-to-value", "rvv.predicate-to-value", type))
    return emitOpError("mask egress requires a matching selected mask and Boolean byte coordinates");
  return success();
}

LogicalResult RVVPredicateLogicOp::verify() {
  bool unary = getKind() == "not";
  if ((!unary && getKind() != "and" && getKind() != "or" && getKind() != "xor") ||
      getInputs().size() != (unary ? 1u : 2u) ||
      !llvm::all_of(getInputs(), [&](Value value) { return maskInput(*this, value, getResult().getType()); }) ||
      !exact(getLeaf(), "predicate-logic", ("rvv.predicate-" + getKind()).str(), getResult().getType()))
    return emitOpError("predicate logic requires identical packed-mask inputs and a closed logical instruction");
  return maskResult(*this, getResult());
}

LogicalResult RVVPredicateSelectOp::verify() {
  auto type = getResult().getType(), condition = getCondition().getType();
  bool vectorTrue = getTrueValue().getType() == type;
  std::string instruction = std::string("rvv.predicate-select.") +
      (vectorTrue ? "vvm" : isa<FloatType>(type.getElementType()) ? "vfm" : "vxm");
  if (!predicateDataType(type) || !maskInput(*this, getCondition(), condition) ||
      !predicateGeometry(condition, type) || getFalseValue().getType() != type ||
      (!vectorTrue && getTrueValue().getType() != type.getElementType()) ||
      !exact(getLeaf(), "predicate-select", instruction, type))
    return emitOpError("predicate selection requires one matching mask and exact numeric branch forms");
  return success();
}
