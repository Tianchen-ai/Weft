#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Predicate.h"

namespace weft::riscv_emission {
namespace {
std::string maskRatio(riscv::ValueType type) {
  auto layout = type.getLayout();
  return std::to_string(layout.getSew() * 8 / layout.getLmulEighths());
}
}

mlir::LogicalResult Emitter::compilePredicateCompare(riscv::RVVPredicateCompareOp operation) {
  llvm::StringRef form = operation.getLeaf().getInstruction();
  if (!form.consume_front("rvv.predicate-compare.") || (form != "vv" && form != "vf" && form != "vx"))
    return fail(operation, "predicate comparison has no selected operand form");
  bool vectorRHS = form == "vv";
  auto lhs = materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  int64_t parts = vectorPartCount(operation.getResult());
  if (mlir::failed(lhs) || mlir::failed(rhs)) return mlir::failure();
  if (lhs->kind != Binding::Kind::Vector || lhs->parts.size() != size_t(parts) ||
      (vectorRHS ? rhs->kind != Binding::Kind::Vector || rhs->parts.size() != size_t(parts)
                 : rhs->kind != Binding::Kind::Scalar))
    return fail(operation, "predicate comparison lost its selected numeric operands");
  auto element = operation.getLhs().getType().getElementType();
  bool floating = mlir::isa<mlir::FloatType>(element);
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  llvm::StringRef predicate = operation.getPredicate();
  bool reverse = vectorRHS && (predicate == "gt" || predicate == "ge");
  if (reverse) predicate = predicate == "gt" ? "lt" : "le";
  bool complement = !floating && !vectorRHS && predicate == "ge";
  if (complement) predicate = "lt";
  std::string instruction = floating ? "vmf" + predicate.str() : "vms" + predicate.str() +
      ((integer.isUnsigned() && predicate != "eq" && predicate != "ne") ? "u" : "");
  const std::string ratio = maskRatio(operation.getResult().getType());
  const std::string suffix = vectorSuffix(operation.getLhs());
  Binding result;
  result.kind = Binding::Kind::Mask;
  for (int64_t part = 0; part < parts; ++part) {
    const std::string vl = partVL(operation.getResult(), part);
    std::string a = lhs->parts[part], b = vectorRHS ? rhs->parts[part] : rhs->scalar;
    if (reverse) std::swap(a, b);
    std::string value = fresh("predicate");
    line("vbool" + ratio + "_t " + value + " = __riscv_" + instruction + "_" + form.str() + "_" + suffix +
         "_b" + ratio + "(" + a + ", " + b + ", " + vl + ");");
    if (complement) line(value + " = __riscv_vmnot_m_b" + ratio + "(" + value + ", " + vl + ");");
    result.parts.push_back(std::move(value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compilePredicateSplat(riscv::RVVPredicateSplatOp operation) {
  if (operation.getLeaf().getInstruction() != "rvv.predicate-splat")
    return fail(operation, "predicate splat has no selected mask instruction");
  auto input = materializeNumeric(operation.getScalar(), bindings.lookup(operation.getScalar()));
  if (mlir::failed(input)) return mlir::failure();
  if (input->kind != Binding::Kind::Scalar) return fail(operation, "predicate splat requires one scalar Boolean");
  std::string ratio = maskRatio(operation.getResult().getType());
  Binding result;
  result.kind = Binding::Kind::Mask;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    std::string name = fresh("predicate_uniform"), vl = partVL(operation.getResult(), part);
    line("vbool" + ratio + "_t " + name + " = (" + input->scalar + ") ? __riscv_vmset_m_b" + ratio +
         "(" + vl + ") : __riscv_vmclr_m_b" + ratio + "(" + vl + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compilePredicateLogic(riscv::RVVPredicateLogicOp operation) {
  llvm::StringRef kind = operation.getKind();
  if (operation.getLeaf().getInstruction() != ("rvv.predicate-" + kind).str())
    return fail(operation, "predicate logic has no selected mask instruction");
  std::string instruction = kind == "not" ? "vmnot_m" : kind == "and" ? "vmand_mm" :
                            kind == "or" ? "vmor_mm" : kind == "xor" ? "vmxor_mm" : "";
  if (instruction.empty()) return fail(operation, "unsupported predicate logic");
  llvm::SmallVector<Binding, 2> inputs;
  int64_t parts = vectorPartCount(operation.getResult());
  for (auto value : operation.getInputs()) {
    auto input = bindings.lookup(value);
    if (input.kind != Binding::Kind::Mask || input.parts.size() != size_t(parts))
      return fail(operation, "predicate logic requires complete mask parts");
    inputs.push_back(std::move(input));
  }
  std::string ratio = maskRatio(operation.getResult().getType());
  Binding result;
  result.kind = Binding::Kind::Mask;
  for (int64_t part = 0; part < parts; ++part) {
    std::string name = fresh("predicate_logic"), arguments = inputs[0].parts[part];
    if (inputs.size() == 2) arguments += ", " + inputs[1].parts[part];
    line("vbool" + ratio + "_t " + name + " = __riscv_" + instruction + "_b" + ratio +
         "(" + arguments + ", " + partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compilePredicateFromValue(riscv::RVVPredicateFromValueOp operation) {
  if (operation.getLeaf().getInstruction() != "rvv.predicate-from-value")
    return fail(operation, "predicate ingress has no selected instruction");
  auto input = materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input)) return mlir::failure();
  int64_t parts = vectorPartCount(operation.getResult());
  if (input->kind != Binding::Kind::Vector || input->parts.size() != size_t(parts))
    return fail(operation, "predicate ingress requires complete numeric byte parts");
  Binding result;
  result.kind = Binding::Kind::Mask;
  std::string ratio = maskRatio(operation.getResult().getType()), suffix = vectorSuffix(operation.getInput());
  for (int64_t part = 0; part < parts; ++part) {
    std::string name = fresh("predicate_loaded");
    line("vbool" + ratio + "_t " + name + " = __riscv_vmsne_vx_" + suffix + "_b" + ratio +
         "(" + input->parts[part] + ", 0, " + partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compilePredicateToValue(riscv::RVVPredicateToValueOp operation) {
  if (operation.getLeaf().getInstruction() != "rvv.predicate-to-value")
    return fail(operation, "predicate egress has no selected instruction");
  auto mask = bindings.lookup(operation.getInput());
  int64_t parts = vectorPartCount(operation.getResult());
  if (mask.kind != Binding::Kind::Mask || mask.parts.size() != size_t(parts))
    return fail(operation, "predicate egress requires complete mask parts");
  Binding result;
  result.kind = Binding::Kind::Vector;
  std::string suffix = vectorSuffix(operation.getResult()), type = vectorType(operation.getResult());
  for (int64_t part = 0; part < parts; ++part) {
    std::string name = fresh("predicate_bytes"), vl = partVL(operation.getResult(), part);
    line(type + " " + name + " = __riscv_vmerge_vxm_" + suffix + "(__riscv_vmv_v_x_" + suffix +
         "(0, " + vl + "), 1, " + mask.parts[part] + ", " + vl + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compilePredicateSelect(riscv::RVVPredicateSelectOp operation) {
  llvm::StringRef form = operation.getLeaf().getInstruction();
  if (!form.consume_front("rvv.predicate-select.") || (form != "vvm" && form != "vfm" && form != "vxm"))
    return fail(operation, "predicate selection has no selected merge form");
  bool vectorTrue = form == "vvm";
  auto condition = bindings.lookup(operation.getCondition());
  auto yes = materializeNumeric(operation.getTrueValue(), bindings.lookup(operation.getTrueValue()));
  auto no = materializeNumeric(operation.getFalseValue(), bindings.lookup(operation.getFalseValue()));
  int64_t parts = vectorPartCount(operation.getResult());
  if (mlir::failed(yes) || mlir::failed(no)) return mlir::failure();
  if (condition.kind != Binding::Kind::Mask || condition.parts.size() != size_t(parts) ||
      no->kind != Binding::Kind::Vector || no->parts.size() != size_t(parts) ||
      (vectorTrue ? yes->kind != Binding::Kind::Vector || yes->parts.size() != size_t(parts)
                  : yes->kind != Binding::Kind::Scalar))
    return fail(operation, "predicate selection lost its packed mask or numeric branches");
  std::string suffix = vectorSuffix(operation.getResult()), type = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    std::string name = fresh("predicate_selected");
    line(type + " " + name + " = __riscv_" + (form == "vfm" ? "vfmerge_" : "vmerge_") + form.str() + "_" + suffix +
         "(" + no->parts[part] + ", " + (vectorTrue ? yes->parts[part] : yes->scalar) + ", " + condition.parts[part] +
         ", " + partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}
} // namespace weft::riscv_emission
