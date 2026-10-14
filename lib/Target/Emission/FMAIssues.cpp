#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/FMAIssue.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileRVVFMAIssue(riscv::RVVFMAIssueOp operation) {
  auto geometry = riscv::FMAIssueGeometry::get(operation.getVector().getType(),
      operation.getMultiplier().getType(), operation.getResult().getType());
  if (!geometry) return fail(operation, "FMA issue lost its verified numeric mapping");
  const auto vector = bindings.lookup(operation.getVector());
  const auto multiplier = bindings.lookup(operation.getMultiplier());
  const auto accumulator = bindings.lookup(operation.getAccumulator());
  if (vector.kind != Binding::Kind::Vector || accumulator.kind != Binding::Kind::Vector ||
      accumulator.parts.size() != size_t(geometry->parts()) ||
      (geometry->scalar && multiplier.kind != Binding::Kind::Scalar && multiplier.kind != Binding::Kind::ScalarTuple) ||
      (!geometry->scalar && multiplier.kind != Binding::Kind::Vector))
    return fail(operation, "FMA issue requires already materialized numeric operands");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (const std::string &part : accumulator.parts) {
    if (operation.getAccumulatorPolicy() == "reuse") result.parts.push_back(part);
    else {
      result.parts.push_back(fresh("fma_acc"));
      line(vectorType(operation.getResult()) + " " + result.parts.back() + " = " + part + ";");
    }
  }
  std::string instruction = geometry->widening ? "vfwmacc" : "vfmacc";
  instruction += geometry->scalar ? "_vf_" : "_vv_";
  instruction += vectorSuffix(operation.getResult());
  instruction += operation.getTail() == "agnostic" ? "(" : "_tu(";
  int64_t begin = 0;
  for (int64_t end : operation.getGroupEnds()) {
    std::string outputs;
    for (int64_t part = begin; part < end; ++part) {
      int64_t v = geometry->vectorParts[part], m = geometry->multiplierParts[part];
      if (v >= int64_t(vector.parts.size()) ||
          (multiplier.kind != Binding::Kind::Scalar && m >= int64_t(multiplier.parts.size())))
        return fail(operation, "FMA issue part mapping exceeds its numeric binding");
      mlir::Value length = operation.getLengths()[part];
      auto vl = bindings.lookup(length);
      if (vl.kind != Binding::Kind::Scalar)
        return fail(operation, "FMA issue VL requires an explicit scalar binding");
      std::string value = multiplier.kind == Binding::Kind::Scalar ? multiplier.scalar : multiplier.parts[m];
      line(result.parts[part] + " = __riscv_" + instruction + result.parts[part] +
           ", " + value + ", " + vector.parts[v] + ", " + vl.scalar + ");");
      if (!outputs.empty()) outputs += ", ";
      outputs += "\"+vr\"(" + result.parts[part] + ")";
    }
    // Finish these numeric results before subsequent independent loads can
    // enter the issue frontier. This is a compiler scheduling boundary, not
    // a hardware memory fence or a memory read/write. Leaving arithmetic as
    // typed intrinsics preserves the system compiler's exact VL/VTYPE state.
    line("__asm__ volatile(\"\" : " + outputs + " : : \"memory\");");
    begin = end;
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
