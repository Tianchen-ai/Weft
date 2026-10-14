#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileIota(riscv::IotaOp operation) {
  llvm::StringRef realization = instructionOf(operation.getOperation());
  mlir::Type element =
      riscv_internal::logicalElement(operation.getResult().getType());
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  if (!element.isIndex() && (!integer || integer.isSigned()))
    return fail(operation, "iota requires an unsigned integer or index result");
  if (realization == "scalar.iota") {
    auto type = scalarCType(element);
    if (!type || operation.getEnd() - operation.getStart() != 1)
      return fail(operation,
                  "scalar iota requires one selected logical element");
    bindings[operation.getResult()] =
        scalar("((" + *type + ")" + std::to_string(operation.getStart()) + ")");
    return mlir::success();
  }
  if (realization == "register.iota") {
    auto type = scalarCType(element);
    const int64_t parts = scalarPartCount(operation.getResult());
    if (!type || parts <= 1)
      return fail(operation,
                  "register iota requires a non-trivial scalar tuple mapping");
    Binding result;
    result.kind = Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < parts; ++part)
      result.parts.push_back("((" + *type + ")" +
                             std::to_string(operation.getStart() + part) + ")");
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (realization != "rvv.iota")
    return fail(operation, "iota has no selected local realization");
  const int64_t parts = vectorPartCount(operation.getResult());
  if (parts <= 0)
    return fail(operation, "RVV iota has no physical vector parts");
  const std::string suffix = vectorSuffix(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string type = vectorType(operation.getResult());
  for (int64_t part = 0; part < parts; ++part) {
    const std::string vl = partVL(operation.getResult(), part);
    std::string expression = "__riscv_vid_v_" + suffix + "(" + vl + ")";
    const std::string offset = "(" + std::to_string(operation.getStart()) +
                               " + " + partOffset(operation.getResult(), part) + ")";
    expression = "__riscv_vadd_vx_" + suffix + "(" + expression + ", " +
                 offset + ", " + vl + ")";
    std::string name = fresh("iota");
    line(type + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::FailureOr<std::string> Emitter::axisBroadcastPart(
    mlir::Operation *operation, mlir::Value inputValue, mlir::Value resultValue,
    int64_t part, const Binding &input) {
  auto inputType = mlir::cast<riscv::ValueType>(inputValue.getType());
  auto resultType = mlir::cast<riscv::ValueType>(resultValue.getType());
  auto sourcePart = riscv::rvvBroadcastSourcePart(inputType, resultType, part);
  if (!sourcePart || input.kind != Binding::Kind::Vector ||
      *sourcePart >= static_cast<int64_t>(input.parts.size())) {
    (void)fail(operation, "axis broadcast has no closed source-part projection");
    return mlir::failure();
  }
  const std::string from = vectorSuffix(inputValue), to = vectorSuffix(resultValue);
  std::string source = input.parts[*sourcePart];
  if (from != to)
    source = "__riscv_vlmul_ext_v_" + from + "_" + to + "(" + source + ")";
  if (physicalLanes(inputValue) == physicalLanes(resultValue))
    return source;
  const std::string vl = partVL(resultValue, part);
  const bool narrowIndex = resultType.getLayout().getReplicaGroup() > 1 &&
                           resultType.getLayout().getSew() == 32;
  const int64_t indexSEW = narrowIndex ? 16 : resultType.getLayout().getSew();
  const int64_t indexLMUL = narrowIndex ? resultType.getLayout().getLmulEighths() / 2
                                      : resultType.getLayout().getLmulEighths();
  const std::string indexSuffix = "u" + std::to_string(indexSEW) + lmulSpelling(indexLMUL);
  auto log2 = [](int64_t value) {
    int64_t result = 0;
    while (value > 1) { value >>= 1; ++result; }
    return result;
  };
  int64_t resultStride = 1;
  llvm::DenseMap<int64_t, int64_t> resultStrides;
  for (int64_t i = resultType.getAxisIds().size() - 1; i >= 0; --i) {
    resultStrides[resultType.getAxisIds()[i]] = resultStride;
    resultStride *= resultType.getLayout().getLaneFactors()[i];
  }
  llvm::SmallVector<int64_t> sourceStrides(inputType.getAxisIds().size(), 1);
  int64_t sourceStride = 1;
  for (int64_t i = sourceStrides.size() - 1; i >= 0; --i) {
    sourceStrides[i] = sourceStride;
    sourceStride *= inputType.getLayout().getLaneFactors()[i];
  }
  std::string index;
  const std::string indexType = "vuint" + std::to_string(indexSEW) + lmulSpelling(indexLMUL) + "_t";
  for (auto [i, axis] : llvm::enumerate(inputType.getAxisIds().asArrayRef())) {
    const int64_t factor = inputType.getLayout().getLaneFactors()[i];
    if (factor == 1)
      continue;
    const int64_t stride = resultStrides.lookup(axis);
    std::string expression = "__riscv_vid_v_" + indexSuffix + "(" + vl + ")";
    if (stride > 1)
      expression = !(stride & (stride - 1))
          ? "__riscv_vsrl_vx_" + indexSuffix + "(" + expression + ", " + std::to_string(log2(stride)) + ", " + vl + ")"
          : "__riscv_vdivu_vx_" + indexSuffix + "(" + expression + ", " + std::to_string(stride) + ", " + vl + ")";
    if (stride * factor < physicalLanes(resultValue))
      expression = !(factor & (factor - 1))
          ? "__riscv_vand_vx_" + indexSuffix + "(" + expression + ", " + std::to_string(factor - 1) + ", " + vl + ")"
          : "__riscv_vremu_vx_" + indexSuffix + "(" + expression + ", " + std::to_string(factor) + ", " + vl + ")";
    if (sourceStrides[i] > 1)
      expression = "__riscv_vmul_vx_" + indexSuffix + "(" + expression + ", " + std::to_string(sourceStrides[i]) + ", " + vl + ")";
    auto coordinate = fresh("broadcast_index");
    line(indexType + " " + coordinate + " = " + expression + ";");
    if (index.empty())
      index = coordinate;
    else
      line(index + " = __riscv_vadd_vv_" + indexSuffix + "(" + index + ", " + coordinate + ", " + vl + ");");
  }
  if (inputType.getLayout().getReplicaGroup() > 1) {
    const int64_t resultTileLanes = physicalLanes(resultValue) / resultType.getLayout().getReplicaGroup();
    const int64_t sourceTileLanes = physicalLanes(inputValue) / inputType.getLayout().getReplicaGroup();
    const std::string outer = fresh("broadcast_replica");
    const bool divideByPowerOfTwo = !(resultTileLanes & (resultTileLanes - 1));
    const bool multiplyByPowerOfTwo = !(sourceTileLanes & (sourceTileLanes - 1));
    line(indexType + " " + outer + (divideByPowerOfTwo ? " = __riscv_vsrl_vx_" : " = __riscv_vdivu_vx_") + indexSuffix +
         "(__riscv_vid_v_" + indexSuffix + "(" + vl + "), " +
         std::to_string(divideByPowerOfTwo ? log2(resultTileLanes) : resultTileLanes) + ", " + vl + ");");
    line(index + " = __riscv_vadd_vv_" + indexSuffix + "(" + index +
         (multiplyByPowerOfTwo ? ", __riscv_vsll_vx_" : ", __riscv_vmul_vx_") + indexSuffix +
         "(" + outer + ", " + std::to_string(multiplyByPowerOfTwo ? log2(sourceTileLanes) : sourceTileLanes) + ", " + vl + "), " + vl + ");");
  }
  if (index.empty()) {
    (void)fail(operation, "axis broadcast has no source lane coordinate");
    return mlir::failure();
  }
  auto broadcast = fresh("axis_broadcast");
  line(vectorType(resultValue) + " " + broadcast + (narrowIndex ? " = __riscv_vrgatherei16_vv_" : " = __riscv_vrgather_vv_") +
       to + "(" + source + ", " + index + ", " + vl + ");");
  return broadcast;
}

mlir::LogicalResult
Emitter::compileRVVAxisBroadcast(riscv::RVVAxisBroadcastOp operation) {
  if (instructionOf(operation) != "rvv.axis-broadcast")
    return fail(operation, "axis broadcast has no exact selected leaf");
  auto input = materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto output = axisBroadcastPart(operation, operation.getInput(), operation.getResult(), part, *input);
    if (mlir::failed(output))
      return mlir::failure();
    result.parts.push_back(*output);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileUnary(riscv::UnaryOp operation) {
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  mlir::Type element =
      riscv_internal::logicalElement(operation.getResult().getType());
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  auto scalarExpression = [&](llvm::StringRef source)
      -> std::optional<std::string> {
    std::string expression;
    if (selected == "scalar.exp-approx-f32" && element.isF32())
      expression = "weft_exp_f32(" + source.str() + ")";
    else if (selected == "scalar.log-approx-f32" && element.isF32())
      expression = "weft_log_f32(" + source.str() + ")";
    else if (selected == "scalar.neg")
      expression = "(-(" + source.str() + "))";
    else if (selected == "scalar.abs" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "fabs(" + source.str() + ")"
                                   : "fabsf(" + source.str() + ")";
    else if (selected == "scalar.abs")
      expression = "((" + source.str() + ") < 0 ? -(" + source.str() +
                   ") : (" + source.str() + "))";
    else if (selected == "scalar.exp" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "exp(" + source.str() + ")"
                                   : "expf(" + source.str() + ")";
    else if (selected == "scalar.exp2" && mlir::isa<mlir::FloatType>(element))
      expression = element.isF64() ? "exp2(" + source.str() + ")"
                                   : "exp2f(" + source.str() + ")";
    else if (selected == "scalar.sqrt" && (element.isF32() || element.isF64()))
      expression = element.isF64() ? "sqrt(" + source.str() + ")"
                                   : "sqrtf(" + source.str() + ")";
    else if (selected == "scalar.log" && (element.isF32() || element.isF64()))
      expression = element.isF64() ? "log(" + source.str() + ")"
                                   : "logf(" + source.str() + ")";
    else
      return std::nullopt;
    return expression;
  };
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "unary result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : scalarPartCount(operation.getResult());
    Binding result;
    result.kind = *targetBindingKind;
    for (int64_t part = 0; part < parts; ++part) {
      std::optional<std::string> source;
      if (input->kind == Binding::Kind::Scalar)
        source = input->scalar;
      else if (input->kind == Binding::Kind::ScalarTuple) {
        auto sourcePart = mappedPart(operation.getOperation(), 0, part);
        if (sourcePart && *sourcePart < input->parts.size())
          source = input->parts[*sourcePart];
      }
      auto expression = source ? scalarExpression(*source) : std::nullopt;
      if (!expression)
        return fail(operation,
                    "selected scalar unary relation or conversion has no C spelling");
      auto scalarType = scalarCType(element);
      if (!scalarType) return fail(operation, "scalar unary has no selected C type");
      // A numeric SSA is evaluated here once. Substituting a libm expression
      // into every use changes its execution count and can move it into a
      // descendant loop even though the author computed it before that loop.
      std::string value = materializeScalarRead(*scalarType, *expression, "unary");
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(value);
      else
        result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (input->kind != Binding::Kind::Vector)
    return fail(operation, "unary operation has no selected numeric handoff");
  if (selected == "rvv.log-approx-f32" || selected == "rvv.log-normal-f32") {
    if (!element.isF32()) return fail(operation, "RVV logarithm requires f32");
    const bool normal = selected == "rvv.log-normal-f32";
    auto layout = mlir::cast<riscv::ValueType>(operation.getResult().getType()).getLayout();
    const std::string lmul = lmulSpelling(layout.getLmulEighths());
    const std::string suffix = "f32" + lmul, type = "vfloat32" + lmul + "_t";
    const std::string usuffix = "u32" + lmul, isuffix = "i32" + lmul;
    const std::string mask = std::to_string(256 / layout.getLmulEighths());
    const std::string maskType = "vbool" + mask + "_t";
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto source = mappedPart(operation.getOperation(), 0, part);
      if (!source || *source >= input->parts.size())
        return fail(operation, "RVV logarithm has no exact numeric part relation");
      const std::string x = input->parts[*source];
      const std::string vl = partVL(operation.getResult(), part);
      std::string small, normalized = x;
      if (!normal) {
        small = fresh("log_subnormal"); normalized = fresh("log_normalized");
        line(maskType + " " + small + " = __riscv_vmflt_vf_" + suffix + "_b" + mask +
             "(" + x + ", 0x1p-126f, " + vl + ");");
        line(type + " " + normalized + " = __riscv_vmerge_vvm_" + suffix + "(" + x +
             ", __riscv_vfmul_vf_" + suffix + "(" + x + ", 0x1p23f, " + vl + "), " + small + ", " + vl + ");");
      }
      std::string bits = fresh("log_bits"), exponent = fresh("log_exponent"), mantissa = fresh("log_mantissa");
      line("vuint32" + lmul + "_t " + bits + " = __riscv_vreinterpret_v_" + suffix + "_" + usuffix + "(" + normalized + ");");
      line("vint32" + lmul + "_t " + exponent + " = __riscv_vsub_vx_" + isuffix +
           "(__riscv_vreinterpret_v_" + usuffix + "_" + isuffix + "(__riscv_vsrl_vx_" + usuffix +
           "(" + bits + ", 23, " + vl + ")), 127, " + vl + ");");
      if (!normal)
        line(exponent + " = __riscv_vmerge_vvm_" + isuffix + "(" + exponent + ", __riscv_vsub_vx_" + isuffix +
             "(" + exponent + ", 23, " + vl + "), " + small + ", " + vl + ");");
      line(type + " " + mantissa + " = __riscv_vreinterpret_v_" + usuffix + "_" + suffix +
           "(__riscv_vor_vx_" + usuffix + "(__riscv_vand_vx_" + usuffix + "(" + bits +
           ", 0x007fffff, " + vl + "), 0x3f800000, " + vl + "));");
      std::string upper = fresh("log_upper");
      line(maskType + " " + upper + " = __riscv_vmfgt_vf_" + suffix + "_b" + mask +
           "(" + mantissa + ", 0x1.6a09e6p+0f, " + vl + ");");
      line(mantissa + " = __riscv_vmerge_vvm_" + suffix + "(" + mantissa + ", __riscv_vfmul_vf_" + suffix +
           "(" + mantissa + ", 0.5f, " + vl + "), " + upper + ", " + vl + ");");
      line(exponent + " = __riscv_vmerge_vvm_" + isuffix + "(" + exponent + ", __riscv_vadd_vx_" + isuffix +
           "(" + exponent + ", 1, " + vl + "), " + upper + ", " + vl + ");");
      // log(m) = 2 * (z + z^3/3 + ...), z=(m-1)/(m+1).
      // m in [sqrt(1/2),sqrt(2)] bounds |z| by 0.171573. The
      // omitted degree-13 tail is below 2*|z|^13/(13*(1-z*z)).
      // This is DLMF 4.6.4; coefficients are the analytic reciprocals.
      std::string z = fresh("log_z"), square = fresh("log_square"), polynomial = fresh("log_polynomial");
      line(type + " " + z + " = __riscv_vfdiv_vv_" + suffix + "(__riscv_vfsub_vf_" + suffix +
           "(" + mantissa + ", 1.0f, " + vl + "), __riscv_vfadd_vf_" + suffix +
           "(" + mantissa + ", 1.0f, " + vl + "), " + vl + ");");
      line(type + " " + square + " = __riscv_vfmul_vv_" + suffix + "(" + z + ", " + z + ", " + vl + ");");
      line(type + " " + polynomial + " = __riscv_vfmv_v_f_" + suffix + "(0.0909090909090909f, " + vl + ");");
      for (const char *coefficient : {"0.1111111111111111f", "0.1428571428571429f",
                                      "0.2f", "0.3333333333333333f", "1.0f"})
        line(polynomial + " = __riscv_vfmadd_vv_" + suffix + "(" + polynomial + ", " + square +
             ", __riscv_vfmv_v_f_" + suffix + "(" + coefficient + ", " + vl + "), " + vl + ");");
      std::string power = fresh("log_power"), out = fresh("log_value");
      line(type + " " + power + " = __riscv_vfcvt_f_x_v_" + suffix + "(" + exponent + ", " + vl + ");");
      line(type + " " + out + " = __riscv_vfmul_vv_" + suffix + "(__riscv_vfmul_vf_" + suffix +
           "(" + z + ", 2.0f, " + vl + "), " + polynomial + ", " + vl + ");");
      line(out + " = __riscv_vfmacc_vf_" + suffix + "(" + out + ", 0x1.7f7d1cp-20f, " + power + ", " + vl + ");");
      line(out + " = __riscv_vfmacc_vf_" + suffix + "(" + out + ", 0x1.62e4p-1f, " + power + ", " + vl + ");");
      auto special = [&](llvm::StringRef predicate, const char *bound, const char *value) {
        line(out + " = __riscv_vfmerge_vfm_" + suffix + "(" + out + ", " + value +
             ", __riscv_vmf" + predicate.str() + "_vf_" + suffix + "_b" + mask +
             "(" + x + ", " + bound + ", " + vl + "), " + vl + ");");
      };
      if (!normal) {
        special("eq", "0.0f", "(-INFINITY)");
        special("eq", "INFINITY", "INFINITY");
        special("lt", "0.0f", "NAN");
      }
      line(out + " = __riscv_vfmerge_vfm_" + suffix + "(" + out + ", NAN, __riscv_vmfne_vv_" + suffix +
           "_b" + mask + "(" + x + ", " + x + ", " + vl + "), " + vl + ");");
      result.parts.push_back(std::move(out));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.exp2-approx-f32") {
    if (!element.isF32()) return fail(operation, "RVV exp2 requires f32");
    Binding result;
    result.kind = Binding::Kind::Vector;
    auto layout = mlir::cast<riscv::ValueType>(operation.getResult().getType()).getLayout();
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    const std::string integer = "i32" + lmulSpelling(layout.getLmulEighths());
    const std::string integerType = "vint32" + lmulSpelling(layout.getLmulEighths()) + "_t";
    const std::string mask = std::to_string(32 * 8 / layout.getLmulEighths());
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation, "RVV exp2 requires an explicit value-use relation");
      const std::string &x = input->parts[*sourcePart];
      const std::string vl = partVL(operation.getResult(), part);
      std::string bounded = fresh("exp2_x"), q = fresh("exp2_q"), residual = fresh("exp2_r");
      line(type + " " + bounded + " = __riscv_vfmax_vf_" + suffix +
           "(__riscv_vfmin_vf_" + suffix + "(" + x + ", 128.0f, " + vl + "), -150.0f, " + vl + ");");
      line(integerType + " " + q + " = __riscv_vfcvt_x_f_v_" + integer + "(" + bounded + ", " + vl + ");");
      line(type + " " + residual + " = __riscv_vfsub_vv_" + suffix + "(" + bounded +
           ", __riscv_vfcvt_f_x_v_" + suffix + "(" + q + ", " + vl + "), " + vl + ");");
      // SLEEF xexp2f's degree-six minimax coefficients on [-1/2,1/2].
      // Fused Horner evaluation retains its u10 numerical organization.
      std::string polynomial = fresh("exp2_p");
      line(type + " " + polynomial + " = __riscv_vfmv_v_f_" + suffix + "(0.1535920892e-3f, " + vl + ");");
      for (const char *coefficient : {"0.1339262701e-2f", "0.9618384764e-2f", "0.5550347269e-1f",
                                      "0.2402264476f", "0.6931471825f", "1.0f"})
        line(polynomial + " = __riscv_vfmadd_vv_" + suffix + "(" + polynomial + ", " + residual +
             ", __riscv_vfmv_v_f_" + suffix + "(" + coefficient + ", " + vl + "), " + vl + ");");
      std::string high = fresh("exp2_hi"), low = fresh("exp2_lo");
      line(integerType + " " + high + " = __riscv_vsra_vx_" + integer + "(" + q + ", 1, " + vl + ");");
      line(integerType + " " + low + " = __riscv_vsub_vv_" + integer + "(" + q + ", " + high + ", " + vl + ");");
      auto scale = [&](const std::string &exponent) {
        return "__riscv_vreinterpret_v_" + integer + "_" + suffix + "(__riscv_vsll_vx_" + integer +
            "(__riscv_vadd_vx_" + integer + "(" + exponent + ", 127, " + vl + "), 23, " + vl + "))";
      };
      std::string out = fresh("exp2");
      line(type + " " + out + " = __riscv_vfmul_vv_" + suffix + "(__riscv_vfmul_vv_" + suffix +
           "(" + polynomial + ", " + scale(high) + ", " + vl + "), " + scale(low) + ", " + vl + ");");
      line(out + " = __riscv_vmerge_vvm_" + suffix + "(" + out + ", " + x +
           ", __riscv_vmfne_vv_" + suffix + "_b" + mask + "(" + x + ", " + x + ", " + vl + "), " + vl + ");");
      result.parts.push_back(std::move(out));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.exp-normal-f32" || selected == "rvv.exp-nonpositive-f32") {
    if (!element.isF32()) return fail(operation, "bounded RVV exponential requires f32");
    const bool nonpositive = selected == "rvv.exp-nonpositive-f32";
    auto layout = mlir::cast<riscv::ValueType>(operation.getResult().getType()).getLayout();
    const std::string suffix = vectorSuffix(operation.getResult()), type = vectorType(operation.getResult());
    const std::string lmul = lmulSpelling(layout.getLmulEighths());
    const std::string integer = "i32" + lmul, integerType = "vint32" + lmul + "_t";
    const std::string mask = std::to_string(256 / layout.getLmulEighths());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto source = mappedPart(operation.getOperation(), 0, part);
      if (!source || *source >= input->parts.size())
        return fail(operation, "bounded exponential has no exact numeric part relation");
      const std::string x = input->parts[*source], vl = partVL(operation.getResult(), part);
      std::string bounded = x;
      std::string bias = fresh("exp_bias"), z = fresh("exp_z"), n = fresh("exp_n"), r = fresh("exp_r");
      line(type + " " + bias + " = __riscv_vfmv_v_f_" + suffix + "(0x1.8p23f, " + vl + ");");
      line(type + " " + z + " = __riscv_vfmacc_vf_" + suffix + "(" + bias + ", 0x1.715476p+0f, " + bounded + ", " + vl + ");");
      line(type + " " + n + " = __riscv_vfsub_vv_" + suffix + "(" + z + ", " + bias + ", " + vl + ");");
      line(type + " " + r + " = __riscv_vfnmsac_vf_" + suffix + "(__riscv_vfnmsac_vf_" + suffix + "(" + bounded +
           ", 0x1.62e4p-1f, " + n + ", " + vl + "), 0x1.7f7d1cp-20f, " + n + ", " + vl + ");");
      std::string square = fresh("exp_square"), first = fresh("exp_first"), second = fresh("exp_second"), polynomial = fresh("exp_polynomial");
      line(type + " " + square + " = __riscv_vfmul_vv_" + suffix + "(" + r + ", " + r + ", " + vl + ");");
      line(type + " " + first + " = __riscv_vfmacc_vf_" + suffix + "(__riscv_vfmv_v_f_" + suffix +
           "(0x1.fffdb6p-2f, " + vl + "), 0x1.555e66p-3f, " + r + ", " + vl + ");");
      line(type + " " + second + " = __riscv_vfmacc_vf_" + suffix + "(__riscv_vfmv_v_f_" + suffix +
           "(0x1.573e2ep-5f, " + vl + "), 0x1.0e4020p-7f, " + r + ", " + vl + ");");
      line(first + " = __riscv_vfmacc_vv_" + suffix + "(" + first + ", " + second + ", " + square + ", " + vl + ");");
      line(type + " " + polynomial + " = __riscv_vfmacc_vv_" + suffix + "(__riscv_vfmul_vf_" + suffix +
           "(" + r + ", 0x1.ffffecp-1f, " + vl + "), " + first + ", " + square + ", " + vl + ");");
      std::string out = fresh("exp_value");
      if (nonpositive) {
        // The proven domain excludes positive overflow. Normal outputs retain
        // one scale/FMA; only subnormal lanes need the second normal power.
        // Unordered lanes keep their NaN polynomial, including in mixed issues.
        std::string bits = fresh("exp_bits"), scale = fresh("exp_scale"), small = fresh("exp_small");
        line(integerType + " " + bits + " = __riscv_vsll_vx_" + integer + "(__riscv_vreinterpret_v_" + suffix + "_" + integer +
             "(" + z + "), 23, " + vl + ");");
        line(type + " " + scale + " = __riscv_vreinterpret_v_" + integer + "_" + suffix + "(__riscv_vadd_vx_" + integer +
             "(" + bits + ", 0x3f800000, " + vl + "));");
        line("vbool" + mask + "_t " + small + " = __riscv_vmflt_vf_" + suffix + "_b" + mask + "(" + n + ", -126.0f, " + vl + ");");
        line(type + " " + out + " = __riscv_vfmacc_vv_" + suffix + "(" + scale + ", " + polynomial + ", " + scale + ", " + vl + ");");
        line("if (__riscv_vcpop_m_b" + mask + "(" + small + ", " + vl + ")) {");
        ++indent;
        std::string power = fresh("exp_subnormal_power"), alternate = fresh("exp_subnormal");
        line(type + " " + power + " = __riscv_vreinterpret_v_" + integer + "_" + suffix + "(__riscv_vadd_vx_" + integer +
             "(" + bits + ", 0x7e000000, " + vl + "));");
        line(type + " " + alternate + " = __riscv_vfmul_vf_" + suffix + "(__riscv_vfmacc_vv_" + suffix +
             "(" + power + ", " + polynomial + ", " + power + ", " + vl + "), 0x1p-125f, " + vl + ");");
        line(out + " = __riscv_vmerge_vvm_" + suffix + "(" + out + ", " + alternate + ", " + small + ", " + vl + ");");
        line(out + " = __riscv_vfmerge_vfm_" + suffix + "(" + out + ", 0.0f, __riscv_vmflt_vf_" + suffix +
             "_b" + mask + "(" + n + ", -150.0f, " + vl + "), " + vl + ");");
        --indent;
        line("}");
      } else {
        std::string scale = fresh("exp_scale");
        line(type + " " + scale + " = __riscv_vreinterpret_v_" + integer + "_" + suffix + "(__riscv_vadd_vx_" + integer +
             "(__riscv_vsll_vx_" + integer + "(__riscv_vreinterpret_v_" + suffix + "_" + integer +
             "(" + z + "), 23, " + vl + "), 0x3f800000, " + vl + "));");
        line(type + " " + out + " = __riscv_vfmacc_vv_" + suffix + "(" + scale + ", " + polynomial + ", " + scale + ", " + vl + ");");
      }
      result.parts.push_back(std::move(out));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.exp-approx-f32") {
    if (!element.isF32())
      return fail(operation, "selected RVV exp leaf requires f32 elements");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    auto layout = mlir::cast<riscv::ValueType>(operation.getResult().getType())
                      .getLayout();
    const int64_t maskBits = 32 * 8 / layout.getLmulEighths();
    if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
        maskBits != 16 && maskBits != 32 && maskBits != 64)
      return fail(operation, "selected RVV exp leaf has no legal mask ratio");
    const std::string unsignedSuffix =
        "u32" + lmulSpelling(layout.getLmulEighths());
    const std::string unsignedType =
        "vuint32" + lmulSpelling(layout.getLmulEighths()) + "_t";
    const std::string maskType = "vbool" + std::to_string(maskBits) + "_t";
    for (int64_t partIndex = 0;
         partIndex < vectorPartCount(operation.getResult()); ++partIndex) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, partIndex);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "RVV exp relation requires its typed value-use conversion");
      const std::string vl = partVL(operation.getResult(), partIndex);
      const std::string &x = input->parts[*sourcePart];
      std::string r = fresh("exp_r");
      std::string z = fresh("exp_z");
      std::string n = fresh("exp_n");
      std::string b = fresh("exp_b");
      std::string e = fresh("exp_e");
      std::string k = fresh("exp_k");
      std::string c = fresh("exp_c");
      std::string u = fresh("exp_u");
      std::string j = fresh("exp_j");
      std::string out = fresh("exp");
      line(type + " " + r + " = __riscv_vfmv_v_f_" + suffix +
           "(0x1.8p23f, " + vl + ");");
      line(type + " " + z + " = __riscv_vfmacc_vf_" + suffix + "(" + r +
           ", 0x1.715476p+0f, " + x + ", " + vl + ");");
      line(type + " " + n + " = __riscv_vfsub_vv_" + suffix + "(" + z +
           ", " + r + ", " + vl + ");");
      line(type + " " + b + " = __riscv_vfnmsac_vf_" + suffix +
           "(__riscv_vfnmsac_vf_" + suffix + "(" + x +
           ", 0x1.62e4p-1f, " + n + ", " + vl +
           "), 0x1.7f7d1cp-20f, " + n + ", " + vl + ");");
      line(unsignedType + " " + e + " = __riscv_vsll_vx_" + unsignedSuffix +
           "(__riscv_vreinterpret_v_" + suffix + "_" + unsignedSuffix + "(" +
           z + "), 23, " + vl + ");");
      line(type + " " + k + " = __riscv_vreinterpret_v_" + unsignedSuffix +
           "_" + suffix + "(__riscv_vadd_vx_" + unsignedSuffix + "(" + e +
           ", 0x3f800000, " + vl + "));" );
      line(maskType + " " + c + " = __riscv_vmfgt_vf_" + suffix + "_b" +
           std::to_string(maskBits) + "(__riscv_vfabs_v_" + suffix + "(" + n +
           ", " + vl + "), 126.0f, " + vl + ");");
      line(type + " " + u + " = __riscv_vfmul_vv_" + suffix + "(" + b +
           ", " + b + ", " + vl + ");");
      line(type + " " + j + " = __riscv_vfmacc_vv_" + suffix +
           "(__riscv_vfmul_vf_" + suffix + "(" + b +
           ", 0x1.ffffecp-1f, " + vl + "), __riscv_vfmacc_vv_" + suffix +
           "(__riscv_vfmacc_vf_" + suffix + "(__riscv_vfmv_v_f_" + suffix +
           "(0x1.fffdb6p-2f, " + vl + "), 0x1.555e66p-3f, " + b + ", " + vl +
           "), __riscv_vfmacc_vf_" + suffix + "(__riscv_vfmv_v_f_" + suffix +
           "(0x1.573e2ep-5f, " + vl + "), 0x1.0e4020p-7f, " + b + ", " + vl +
           "), " + u + ", " + vl + "), " + u + ", " + vl + ");");
      line(type + " " + out + ";");
      line("if (!__riscv_vcpop_m_b" + std::to_string(maskBits) + "(" + c +
           ", " + vl + ")) {");
      ++indent;
      line(out + " = __riscv_vfmacc_vv_" + suffix + "(" + k + ", " + j +
           ", " + k + ", " + vl + ");");
      --indent;
      line("} else {");
      ++indent;
      std::string dm = fresh("exp_dm");
      std::string d = fresh("exp_d");
      std::string s1 = fresh("exp_s1");
      std::string s2 = fresh("exp_s2");
      std::string r1 = fresh("exp_r1");
      line(maskType + " " + dm + " = __riscv_vmfle_vf_" + suffix + "_b" +
           std::to_string(maskBits) + "(" + n + ", 0.0f, " + vl + ");");
      line(unsignedType + " " + d + " = __riscv_vmerge_vxm_" +
           unsignedSuffix + "(__riscv_vmv_v_x_" + unsignedSuffix + "(0, " +
           vl + "), 0x82000000, " + dm + ", " + vl + ");");
      line(type + " " + s1 + " = __riscv_vreinterpret_v_" + unsignedSuffix +
           "_" + suffix + "(__riscv_vadd_vx_" + unsignedSuffix + "(" + d +
           ", 0x7f000000, " + vl + "));" );
      line(type + " " + s2 + " = __riscv_vreinterpret_v_" + unsignedSuffix +
           "_" + suffix + "(__riscv_vsub_vv_" + unsignedSuffix + "(" + e +
           ", " + d + ", " + vl + "));" );
      line(type + " " + r1 + " = __riscv_vmerge_vvm_" + suffix +
           "(__riscv_vfmacc_vv_" + suffix + "(" + k + ", " + k + ", " + j +
           ", " + vl + "), __riscv_vfmul_vv_" + suffix +
           "(__riscv_vfmacc_vv_" + suffix + "(" + s2 + ", " + s2 + ", " + j +
           ", " + vl + "), " + s1 + ", " + vl + "), " + c + ", " + vl +
           ");");
      line(out + " = __riscv_vmerge_vvm_" + suffix + "(" + r1 +
           ", __riscv_vfmul_vv_" + suffix + "(" + s1 + ", " + s1 + ", " +
           vl + "), __riscv_vmfgt_vf_" + suffix + "_b" +
           std::to_string(maskBits) + "(__riscv_vfabs_v_" + suffix + "(" + n +
           ", " + vl + "), 192.0f, " + vl + "), " + vl + ");");
      --indent;
      line("}");
      result.parts.push_back(std::move(out));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const bool floating = usesFloatingRegisters(element);
  for (int64_t index = 0; index < vectorPartCount(operation.getResult());
       ++index) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, index);
    if (!sourcePart || *sourcePart >= input->parts.size())
      return fail(operation,
                  "vector unary relation requires its typed value-use conversion");
    const std::string &part = input->parts[*sourcePart];
    std::string expression;
    if (selected == "rvv.vfneg.v" || selected == "rvv.vneg.v")
      expression = "__riscv_" + std::string(floating ? "vfneg_v_" : "vneg_v_") +
                   suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else if (selected == "rvv.vfabs.v" && floating)
      expression = "__riscv_vfabs_v_" + suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else if (selected == "rvv.vfsqrt.v" && (element.isF32() || element.isF64()))
      expression = "__riscv_vfsqrt_v_" + suffix + "(" + part + ", " +
                   partVL(operation.getResult(), index) + ")";
    else
      return fail(operation,
                  "selected integer vector absolute value is not implemented");
    if (streamPartCount(operation.getResult()) > 1) {
      result.parts.push_back(std::move(expression));
    } else {
      std::string name = fresh("unary");
      line(type + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCompare(riscv::CompareOp operation) {
  mlir::FailureOr<Binding> lhsOr =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhsOr =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhsOr) || mlir::failed(rhsOr))
    return mlir::failure();
  Binding lhs = std::move(*lhsOr);
  Binding rhs = std::move(*rhsOr);
  llvm::StringRef predicate = operation.getLeaf().getInstruction();
  auto scalarSpelling = [](llvm::StringRef kind) -> llvm::StringRef {
    return kind == "eq" ? "=="
         : kind == "ne" ? "!="
         : kind == "lt" ? "<"
         : kind == "le" ? "<="
         : kind == "gt" ? ">"
         : kind == "ge" ? ">="
                          : "";
  };
  auto selectedKind = selectedBindingKind(operation.getResult());
  if (!selectedKind)
    return fail(operation, "comparison result has no selected physical carrier");
  if (*selectedKind == Binding::Kind::Scalar ||
      *selectedKind == Binding::Kind::ScalarTuple) {
    if (!predicate.consume_front("scalar.cmp."))
      return fail(operation,
                  "scalar comparison has no selected scalar predicate spelling");
    llvm::StringRef spelling = scalarSpelling(predicate);
    if (spelling.empty())
      return fail(operation, "unknown selected scalar comparison predicate");
    const int64_t parts = *selectedKind == Binding::Kind::Scalar
                              ? 1
                              : registerPartCount(operation.getResult());
    Binding result;
    result.kind = *selectedKind;
    for (int64_t part = 0; part < parts; ++part) {
      auto expressionFor = [&](size_t operand, const Binding &binding)
          -> std::optional<std::string> {
        if (binding.kind == Binding::Kind::Scalar)
          return binding.scalar;
        if (binding.kind != Binding::Kind::ScalarTuple)
          return std::nullopt;
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (!projected || *projected >= binding.parts.size())
          return std::nullopt;
        return binding.parts[*projected];
      };
      auto left = expressionFor(0, lhs);
      auto right = expressionFor(1, rhs);
      if (!left || !right)
        return fail(operation,
                    "scalar comparison operands do not project to the result tuple");
      std::string expression =
          "(" + *left + " " + spelling.str() + " " + *right + ")";
      if (parts == 1)
        result.scalar = std::move(expression);
      else
        result.parts.push_back(std::move(expression));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (*selectedKind != Binding::Kind::Vector ||
      !predicate.consume_front("rvv.cmp."))
    return fail(operation,
                "comparison leaf disagrees with the selected result carrier");
  const bool booleanComparison = predicate.consume_front("bool.");
  if (booleanComparison &&
      !riscv_internal::booleanComparisonTemporaryGroups(operation))
    return fail(operation, "Boolean comparison has no selected numeric 0/1 representation");
  if (lhs.kind != Binding::Kind::Vector && rhs.kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV comparison requires at least one selected vector operand");
  mlir::Value referenceValue = lhs.kind == Binding::Kind::Vector
                                   ? operation.getLhs()
                                   : operation.getRhs();
  auto referenceLayout = layoutOf(referenceValue);
  if (!referenceLayout || referenceLayout.getCarrier() != "rvv" ||
      referenceLayout.getLmulEighths() <= 0 ||
      (referenceLayout.getSew() * 8) % referenceLayout.getLmulEighths())
    return fail(operation, "RVV comparison has no legal mask ratio");
  const int64_t maskBits =
      referenceLayout.getSew() * 8 / referenceLayout.getLmulEighths();
  if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
      maskBits != 16 && maskBits != 32 && maskBits != 64)
    return fail(operation, "RVV comparison mask type is outside the ISA domain");
  mlir::Type operandElement =
      riscv_internal::logicalElement(operation.getLhs().getType());
  const bool floating = usesFloatingRegisters(operandElement);
  auto integer = mlir::dyn_cast<mlir::IntegerType>(operandElement);
  if (!floating && (!integer || (integer.isSignless() && !booleanComparison)))
    return fail(operation,
                "RVV integer comparison requires explicit signedness");
  const bool reverse = predicate == "gt" || predicate == "ge";
  llvm::StringRef normalized = predicate == "gt" ? "lt"
                              : predicate == "ge" ? "le"
                                                   : predicate;
  std::string stem;
  if (floating)
    stem = normalized == "eq" ? "vmfeq"
         : normalized == "ne" ? "vmfne"
         : normalized == "lt" ? "vmflt"
         : normalized == "le" ? "vmfle"
                                : std::string();
  else
    stem = normalized == "eq" ? "vmseq"
         : normalized == "ne" ? "vmsne"
         : normalized == "lt" ? (integer.isUnsigned() ? "vmsltu" : "vmslt")
         : normalized == "le" ? (integer.isUnsigned() ? "vmsleu" : "vmsle")
                                : std::string();
  if (stem.empty())
    return fail(operation, "RVV comparison predicate has no exact instruction");

  const int64_t resultParts = vectorPartCount(operation.getResult());
  const std::string operandSuffix = vectorSuffix(referenceValue);
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < resultParts; ++part) {
    auto vectorExpression = [&](size_t operand, mlir::Value value,
                                const Binding &binding)
        -> std::optional<std::string> {
      if (binding.kind == Binding::Kind::Vector) {
        if (vectorSuffix(value) != operandSuffix)
          return std::nullopt;
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (!projected || *projected >= binding.parts.size())
          return std::nullopt;
        return binding.parts[*projected];
      }
      std::optional<std::string> scalarValue;
      if (binding.kind == Binding::Kind::Scalar)
        scalarValue = binding.scalar;
      else if (binding.kind == Binding::Kind::ScalarTuple) {
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (projected && *projected < binding.parts.size())
          scalarValue = binding.parts[*projected];
      }
      if (!scalarValue)
        return std::nullopt;
      const std::string vl = partVL(operation.getResult(), part);
      return "__riscv_" +
             std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
             operandSuffix + "(" + *scalarValue + ", " + vl + ")";
    };
    auto left = vectorExpression(0, operation.getLhs(), lhs);
    auto right = vectorExpression(1, operation.getRhs(), rhs);
    if (!left || !right)
      return fail(operation,
                  "RVV comparison operands do not project to one physical part");
    if (reverse)
      std::swap(left, right);
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = "__riscv_" + stem + "_vv_" + operandSuffix + "_b" +
                       std::to_string(maskBits) + "(" + *left + ", " +
                       *right + ", " + vl + ")";
    std::string zero = "__riscv_vmv_v_x_" + resultSuffix + "(0, " + vl + ")";
    std::string expression = "__riscv_vmerge_vxm_" + resultSuffix + "(" +
                             zero + ", 1, " + mask + ", " + vl + ")";
    std::string name = fresh("compare");
    line(resultType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVCompareSelect(riscv::RVVCompareSelectOp operation) {
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  if (!selected.consume_front("rvv.compare-select."))
    return fail(operation, "direct compare/select has no selected leaf");
  size_t separator = selected.find('.');
  if (separator == llvm::StringRef::npos)
    return fail(operation, "direct compare/select has no selected operand forms");
  llvm::StringRef comparison = selected.take_front(separator), merge = selected.drop_front(separator + 1);
  bool vectorComparison = comparison == "vv", scalarTrue = merge != "vvm";
  if ((!vectorComparison && comparison != "vf" && comparison != "vx") ||
      (scalarTrue && merge != "vfm" && merge != "vxm"))
    return fail(operation, "direct compare/select operand form is unsupported");
  auto type = operation.getResult().getType();
  auto layout = type.getLayout();
  const int64_t parts = vectorPartCount(operation.getResult());
  llvm::SmallVector<Binding, 4> inputs;
  bool vectorOperand[] = {true, vectorComparison, !scalarTrue, true};
  for (auto [position, value] : llvm::enumerate(operation.getOperands())) {
    auto input = materializeNumeric(value, bindings.lookup(value));
    if (mlir::failed(input)) return mlir::failure();
    if (vectorOperand[position] ? (input->kind != Binding::Kind::Vector || input->parts.size() != static_cast<size_t>(parts))
                                : input->kind != Binding::Kind::Scalar)
      return fail(operation, "direct compare/select lost its selected operand representation");
    inputs.push_back(std::move(*input));
  }
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  const bool floating = mlir::isa<mlir::FloatType>(type.getElementType());
  llvm::StringRef predicate = operation.getPredicate();
  const bool reverse = vectorComparison && (predicate == "gt" || predicate == "ge");
  if (reverse) predicate = predicate == "gt" ? "lt" : "le";
  // Integer GE with a scalar has no native vmsge.vx instruction. Its selected
  // local spelling complements LT in the same mask; floating GE uses vmfge.vf
  // directly so unordered comparisons retain their declared false result.
  bool complement = !floating && !vectorComparison && predicate == "ge";
  if (complement) predicate = "lt";
  std::string stem;
  if (floating) stem = "vmf" + predicate.str();
  else stem = "vms" + predicate.str() +
      ((integer.isUnsigned() && predicate != "eq" && predicate != "ne") ? "u" : "");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string ratio = std::to_string(layout.getSew() * 8 / layout.getLmulEighths());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto operand = [&](size_t position) {
      return vectorOperand[position] ? inputs[position].parts[part] : inputs[position].scalar;
    };
    std::string vl = partVL(operation.getResult(), part);
    std::string mask = fresh("compare_mask"), selected = fresh("selected");
    line("vbool" + ratio + "_t " + mask + " = __riscv_" + stem + "_" + comparison.str() + "_" + suffix + "_b" + ratio +
         "(" + operand(reverse ? 1 : 0) + ", " + operand(reverse ? 0 : 1) + ", " + vl + ");");
    if (complement) line(mask + " = __riscv_vmnot_m_b" + ratio + "(" + mask + ", " + vl + ");");
    line(vectorType(operation.getResult()) + " " + selected + " = __riscv_" +
         std::string(merge == "vfm" ? "vfmerge_" : "vmerge_") + merge.str() + "_" + suffix +
         "(" + operand(3) + ", " + operand(2) + ", " + mask + ", " + vl + ");");
    result.parts.push_back(std::move(selected));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSelect(riscv::SelectOp operation) {
  llvm::SmallVector<Binding, 3> operands;
  for (mlir::Value operand : operation->getOperands()) {
    auto binding = materializeNumeric(operand, bindings.lookup(operand));
    if (mlir::failed(binding))
      return mlir::failure();
    operands.push_back(std::move(*binding));
  }
  auto kind = selectedBindingKind(operation.getResult());
  if (!kind)
    return fail(operation, "select result has no selected carrier");
  const bool vector = *kind == Binding::Kind::Vector;
  if (operation.getLeaf().getInstruction() !=
      (vector ? "rvv.select" : "scalar.select"))
    return fail(operation, "select leaf disagrees with result carrier");
  const int64_t parts = vector ? vectorPartCount(operation.getResult())
                              : registerPartCount(operation.getResult());
  Binding result;
  result.kind = *kind;
  for (int64_t part = 0; part < parts; ++part) {
    llvm::SmallVector<std::string> expressions;
    for (auto [number, binding] : llvm::enumerate(operands)) {
      if (binding.kind == Binding::Kind::Scalar) {
        expressions.push_back(binding.scalar);
      } else {
        auto projected = mappedPart(operation, number, part);
        if (!projected || *projected >= binding.parts.size())
          return fail(operation, "select operand does not project to result part");
        expressions.push_back(binding.parts[*projected]);
      }
    }
    if (vector) {
      auto suffix = vectorSuffix(operation.getResult());
      auto vl = partVL(operation.getResult(), part);
      const bool floating = usesFloatingRegisters(
          riscv_internal::logicalElement(operation.getResult().getType()));
      for (size_t branch : {1, 2})
        if (operands[branch].kind != Binding::Kind::Vector)
          expressions[branch] = "__riscv_" +
              std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix +
              "(" + expressions[branch] + ", " + vl + ")";
      std::string expression;
      if (operands[0].kind == Binding::Kind::Vector) {
        auto predicate = operation.getCondition();
        auto layout = layoutOf(predicate);
        auto maskBits = layout.getSew() * 8 / layout.getLmulEighths();
        auto mask = "__riscv_vmsne_vx_" + vectorSuffix(predicate) + "_b" +
                    std::to_string(maskBits) + "(" + expressions[0] +
                    ", 0, " + vl + ")";
        expression = "__riscv_vmerge_vvm_" + suffix + "(" + expressions[2] +
                     ", " + expressions[1] + ", " + mask + ", " + vl + ")";
      } else {
        expression = "(" + expressions[0] + " ? " + expressions[1] +
                     " : " + expressions[2] + ")";
      }
      auto name = fresh("select");
      line(vectorType(operation.getResult()) + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    } else {
      auto expression = "(" + expressions[0] + " ? " + expressions[1] +
                        " : " + expressions[2] + ")";
      if (*kind == Binding::Kind::Scalar)
        result.scalar = std::move(expression);
      else
        result.parts.push_back(std::move(expression));
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

template <typename ConversionOp>
mlir::LogicalResult Emitter::compileCastImpl(ConversionOp operation) {
  mlir::Operation *rawOperation = operation.getOperation();
  mlir::Value inputValue = operation.getInput();
  mlir::Value resultValue = operation.getResult();
  mlir::FailureOr<Binding> input = materializeNumeric(
      inputValue, bindings.lookup(inputValue));
  if (mlir::failed(input))
    return mlir::failure();
  mlir::Type source =
      riscv_internal::logicalElement(inputValue.getType());
  mlir::Type target =
      riscv_internal::logicalElement(resultValue.getType());
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  auto targetCType = scalarCType(target);
  if (!targetCType)
    return fail(operation, "cast target has no intrinsic-C type");
  auto rounding = rawOperation->getAttrOfType<mlir::StringAttr>("rounding");
  auto saturate = rawOperation->getAttrOfType<mlir::BoolAttr>("saturate");
  auto scalarExpression = [&](llvm::StringRef inputExpression)
      -> mlir::FailureOr<std::string> {
    std::string expression = inputExpression.str();
    if (selected == "scalar.bitcast") {
      auto sourceCType = scalarCType(source);
      if (!sourceCType) return mlir::failure();
      std::string stored = fresh("bitcast_source"), result = fresh("bitcast_result");
      line(*sourceCType + " " + stored + " = " + expression + ";");
      line(*targetCType + " " + result + ";");
      line("memcpy(&" + result + ", &" + stored + ", sizeof(" + result + "));");
      return result;
    }
    if (selected == "scalar.f8e4m3fn-to-f32")
      return "weft_f8e4m3fn_to_f32(" + expression + ")";
    if (selected == "scalar.f8e5m2-to-f32")
      return "weft_f8e5m2_to_f32(" + expression + ")";
    if (selected == "scalar.bf16-to-f32")
      return "weft_bf16_to_f32(" + expression + ")";
    if (selected == "scalar.f32-to-bf16-rne")
      return "weft_f32_to_bf16(" + expression + ")";
    if (rounding) {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(target);
      if (!integer) {
        if (!mlir::isa<mlir::FloatType>(target) ||
            (saturate && saturate.getValue()))
          return mlir::failure();
        return "((" + *targetCType + ")(" + expression + "))";
      }
      const unsigned width = integer.getWidth();
      if (width == 0 || width >= 64)
        return mlir::failure();
      const bool isUnsigned = integer.isUnsigned();
      const double minimum = isUnsigned ? 0.0 : -static_cast<double>(uint64_t{1} << (width - 1));
      const double maximum = isUnsigned
                                 ? static_cast<double>((uint64_t{1} << width) - 1)
                                 : static_cast<double>((uint64_t{1} << (width - 1)) - 1);
      if (saturate && saturate.getValue())
        expression = "fmax(" + std::to_string(minimum) + ", fmin(" +
                     std::to_string(maximum) + ", " + expression + "))";
      llvm::StringRef roundFunction =
                                      rounding.getValue() == "dynamic" ? "nearbyint"
                                      : rounding.getValue() == "rne" ? "nearbyint"
                                      : rounding.getValue() == "rtz" ? "trunc"
                                      : rounding.getValue() == "rdn" ? "floor"
                                                                        : "ceil";
      expression = roundFunction.str() + "(" + expression + ")";
    }
    return "((" + *targetCType + ")(" + expression + "))";
  };
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "cast result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    if (!selected.starts_with("scalar."))
      return fail(operation,
                  "scalar cast has no exact selected scalar instruction");
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : scalarPartCount(operation.getResult());
    Binding result;
    result.kind = *targetBindingKind;
    for (int64_t part = 0; part < parts; ++part) {
      std::optional<std::string> sourceExpression;
      if (input->kind == Binding::Kind::Scalar)
        sourceExpression = input->scalar;
      else if (input->kind == Binding::Kind::ScalarTuple) {
        auto sourcePart = mappedPart(operation.getOperation(), 0, part);
        if (sourcePart && *sourcePart < input->parts.size())
          sourceExpression = input->parts[*sourcePart];
      }
      if (!sourceExpression)
        return fail(operation,
                    "cast scalar result requires its typed value-use conversion");
      mlir::FailureOr<std::string> expression =
          scalarExpression(*sourceExpression);
      if (mlir::failed(expression))
        return fail(operation,
                    "rounded scalar or tuple narrowing requires an integer target");
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(*expression);
      else
        result.parts.push_back(std::move(*expression));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (input->kind != Binding::Kind::Vector)
    return fail(operation, "cast has no selected numeric handoff");
  if (selected == "rvv.f8e4m3fn-to-f32" || selected == "rvv.f8e5m2-to-f32") {
    if (!riscv_internal::storageFloatCastTemporaryGroups(rawOperation))
      return fail(operation, "FP8 conversion has incompatible physical coordinates");
    const bool e4m3 = mlir::isa<mlir::Float8E4M3FNType>(source);
    const std::string wide = "u32" + lmulSpelling(layoutOf(resultValue).getLmulEighths());
    const std::string wideType = "vuint32" + lmulSpelling(layoutOf(resultValue).getLmulEighths()) + "_t";
    const std::string out = vectorSuffix(resultValue);
    const std::string maskSuffix = std::to_string(32 * 8 / layoutOf(resultValue).getLmulEighths());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(resultValue); ++part) {
      auto sourcePart = mappedPart(rawOperation, 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation, "FP8 conversion requires an explicit operand-part mapping");
      const std::string vl = partVL(resultValue, part);
      const std::string bits = fresh("fp8_bits"), sign = fresh("fp8_sign");
      const std::string mag = fresh("fp8_magnitude"), sub = fresh("fp8_subnormal");
      const std::string mask = fresh("fp8_mask"), converted = fresh("fp8_cast");
      line(wideType + " " + bits + " = __riscv_vzext_vf4_" + wide + "(" + input->parts[*sourcePart] + ", " + vl + ");");
      line(wideType + " " + sign + " = __riscv_vand_vx_" + wide + "(" + bits + ", 128, " + vl + ");");
      line(sign + " = __riscv_vsll_vx_" + wide + "(" + sign + ", 24, " + vl + ");");
      line(wideType + " " + mag + " = __riscv_vand_vx_" + wide + "(" + bits + ", 127, " + vl + ");");
      line(bits + " = __riscv_vsll_vx_" + wide + "(" + mag + ", " + (e4m3 ? "20" : "21") + ", " + vl + ");");
      line(bits + " = __riscv_vadd_vx_" + wide + "(" + bits + ", UINT32_C(" + (e4m3 ? "1006632960" : "939524096") + "), " + vl + ");");
      line(vectorType(resultValue) + " " + sub + " = __riscv_vfcvt_f_xu_v_" + out + "(" + mag + ", " + vl + ");");
      line(sub + " = __riscv_vfmul_vf_" + out + "(" + sub + ", " + (e4m3 ? "0x1p-9f" : "0x1p-16f") + ", " + vl + ");");
      line("vbool" + maskSuffix + "_t " + mask + " = __riscv_vmsltu_vx_" + wide + "_b" + maskSuffix + "(" + mag + ", " + (e4m3 ? "8" : "4") + ", " + vl + ");");
      line(bits + " = __riscv_vmerge_vvm_" + wide + "(" + bits + ", __riscv_vreinterpret_v_" + out + "_" + wide + "(" + sub + "), " + mask + ", " + vl + ");");
      line(mask + " = __riscv_vmsgtu_vx_" + wide + "_b" + maskSuffix + "(" + mag + ", " + (e4m3 ? "126" : "124") + ", " + vl + ");");
      line(bits + " = __riscv_vmerge_vxm_" + wide + "(" + bits + ", UINT32_C(0x7fc00000), " + mask + ", " + vl + ");");
      if (!e4m3) {
        line(mask + " = __riscv_vmseq_vx_" + wide + "_b" + maskSuffix + "(" + mag + ", 124, " + vl + ");");
        line(bits + " = __riscv_vmerge_vxm_" + wide + "(" + bits + ", UINT32_C(0x7f800000), " + mask + ", " + vl + ");");
      }
      line(bits + " = __riscv_vor_vv_" + wide + "(" + bits + ", " + sign + ", " + vl + ");");
      line(vectorType(resultValue) + " " + converted + " = __riscv_vreinterpret_v_" + wide + "_" + out + "(" + bits + ");");
      result.parts.push_back(converted);
    }
    bindings[resultValue] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.bf16-to-f32" || selected == "rvv.f32-to-bf16-rne") {
    if (!riscv_internal::storageFloatCastTemporaryGroups(rawOperation))
      return fail(operation, "BF16 conversion has incompatible physical coordinates");
    const bool widen = source.isBF16();
    auto wideLayout = layoutOf(widen ? resultValue : inputValue);
    const std::string wide = "u32" + lmulSpelling(wideLayout.getLmulEighths());
    const std::string wideType = "vuint32" + lmulSpelling(wideLayout.getLmulEighths()) + "_t";
    const std::string targetSuffix = vectorSuffix(resultValue);
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(resultValue); ++part) {
      auto sourcePart = mappedPart(rawOperation, 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation, "BF16 conversion requires an explicit operand-part mapping");
      const std::string vl = partVL(resultValue, part);
      const std::string bits = fresh("bf16_bits");
      const std::string converted = fresh("bf16_cast");
      if (widen) {
        line(wideType + " " + bits + " = __riscv_vzext_vf2_" + wide + "(" + input->parts[*sourcePart] + ", " + vl + ");");
        line(bits + " = __riscv_vsll_vx_" + wide + "(" + bits + ", 16, " + vl + ");");
        line(vectorType(resultValue) + " " + converted + " = __riscv_vreinterpret_v_" + wide + "_" + targetSuffix + "(" + bits + ");");
      } else {
        line(wideType + " " + bits + " = __riscv_vreinterpret_v_" + vectorSuffix(inputValue) + "_" + wide + "(" + input->parts[*sourcePart] + ");");
        const std::string work = fresh("bf16_round");
        line(wideType + " " + work + " = __riscv_vsrl_vx_" + wide + "(" + bits + ", 16, " + vl + ");");
        line(work + " = __riscv_vand_vx_" + wide + "(" + work + ", 1, " + vl + ");");
        line(work + " = __riscv_vadd_vx_" + wide + "(" + work + ", 32767, " + vl + ");");
        line(work + " = __riscv_vadd_vv_" + wide + "(" + work + ", " + bits + ", " + vl + ");");
        line(work + " = __riscv_vsrl_vx_" + wide + "(" + work + ", 16, " + vl + ");");
        const std::string nanBits = fresh("bf16_nan_bits"), nan = fresh("bf16_nan");
        const std::string mask = std::to_string(32 * 8 / wideLayout.getLmulEighths());
        line(wideType + " " + nanBits + " = __riscv_vand_vx_" + wide + "(" + bits + ", UINT32_C(0x7fffffff), " + vl + ");");
        line("vbool" + mask + "_t " + nan + " = __riscv_vmsgtu_vx_" + wide + "_b" + mask + "(" + nanBits + ", UINT32_C(0x7f800000), " + vl + ");");
        line(nanBits + " = __riscv_vsrl_vx_" + wide + "(" + bits + ", 16, " + vl + ");");
        line(nanBits + " = __riscv_vor_vx_" + wide + "(" + nanBits + ", 64, " + vl + ");");
        line(work + " = __riscv_vmerge_vvm_" + wide + "(" + work + ", " + nanBits + ", " + nan + ", " + vl + ");");
        line(vectorType(resultValue) + " " + converted + " = __riscv_vnsrl_wx_" + targetSuffix + "(" + work + ", 0, " + vl + ");");
      }
      result.parts.push_back(converted);
    }
    bindings[resultValue] = std::move(result);
    return mlir::success();
  }
  const bool identity = source == target;
  auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(source);
  const bool integerReinterpret =
      riscv_internal::isSameWidthIntegerCast(rawOperation);
  if (selected == "rvv.identity" && !rounding && identity) {
    Binding projected;
    projected.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "identity cast requires its typed value-use conversion");
      projected.parts.push_back(input->parts[*sourcePart]);
    }
    bindings[operation.getResult()] = std::move(projected);
    return mlir::success();
  }
  if ((selected == "rvv.reinterpret" && !rounding && integerReinterpret) ||
      selected == "rvv.bitcast") {
    const std::string sourceSuffix = vectorSuffix(operation.getInput());
    const std::string targetSuffix = vectorSuffix(operation.getResult());
    const std::string targetType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "integer reinterpret requires an explicit physical layout conversion");
      std::string value = fresh("reinterpret");
      if (sourceSuffix == targetSuffix) {
        result.parts.push_back(input->parts[*sourcePart]);
        continue;
      }
      line(targetType + " " + value + " = __riscv_vreinterpret_v_" +
           sourceSuffix + "_" + targetSuffix + "(" +
           input->parts[*sourcePart] + ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if ((selected == "rvv.fnarrow.f32-f16" ||
       selected == "rvv.fwiden.f16-f32" ||
       selected == "rvv.fwiden.f32-f64") &&
      (!rounding || (saturate && !saturate.getValue())) &&
      ((source.isF32() && target.isF16()) ||
       (source.isF16() && target.isF32()) ||
       (source.isF32() && target.isF64()))) {
    const bool narrowing =
        source.getIntOrFloatBitWidth() > target.getIntOrFloatBitWidth();
    int64_t sourceLMUL = layoutOf(inputValue).getLmulEighths();
    int64_t targetLMUL = layoutOf(resultValue).getLmulEighths();
    if ((narrowing && sourceLMUL != targetLMUL * 2) ||
        (!narrowing && targetLMUL != sourceLMUL * 2))
      return fail(operation,
                  "selected float cast LMUL relation cannot be emitted mechanically");
    const std::string targetSuffix = vectorSuffix(operation.getResult());
    const std::string targetType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "float cast requires an explicit physical layout conversion");
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = fresh(narrowing ? "narrow_float" : "widen_float");
      std::string intrinsic = narrowing ? "__riscv_vfncvt_f_f_w_"
                                        : "__riscv_vfwcvt_f_f_v_";
      line(targetType + " " + value + " = " + intrinsic + targetSuffix +
           "(" + input->parts[*sourcePart] + ", " + vl + ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.narrow.int.vf2") {
    auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(source);
    auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
    if (!sourceInteger || !targetInteger ||
        sourceInteger.getWidth() != targetInteger.getWidth() * 2 ||
        targetInteger.getWidth() < 8 || (saturate && saturate.getValue()))
      return fail(operation,
                  "selected integer narrowing has incompatible typed operands");
    const int64_t sourceLMUL = layoutOf(inputValue).getLmulEighths();
    const int64_t targetLMUL = layoutOf(resultValue).getLmulEighths();
    if (sourceLMUL != targetLMUL * 2)
      return fail(operation,
                  "selected integer narrowing has an incompatible LMUL relation");
    const std::string targetSuffix = vectorSuffix(operation.getResult());
    const std::string targetType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation,
                    "integer narrowing requires its typed value-use conversion");
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = fresh("narrow_int");
      line(targetType + " " + value + " = __riscv_vncvt_x_x_w_" +
           targetSuffix + "(" + input->parts[*sourcePart] + ", " + vl +
           ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selected == "rvv.convert.i32-f32" && !rounding && sourceInteger &&
      sourceInteger.getWidth() == 32 && target.isF32()) {
    if (layoutOf(inputValue).getLmulEighths() != layoutOf(resultValue).getLmulEighths())
      return fail(operation, "selected i32/f32 cast requires one unchanged register mapping");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string intrinsic = sourceInteger.isSigned() ? "__riscv_vfcvt_f_x_v_" : "__riscv_vfcvt_f_xu_v_";
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = mappedPart(operation.getOperation(), 0, part);
      if (!sourcePart || *sourcePart >= input->parts.size())
        return fail(operation, "integer-to-float cast requires its typed value-use conversion");
      std::string value = fresh("convert_f32");
      line(vectorType(operation.getResult()) + " " + value + " = " + intrinsic +
           vectorSuffix(operation.getResult()) + "(" + input->parts[*sourcePart] + ", " +
           partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const bool saturatedF32I8 = selected.starts_with("rvv.f32-i8-saturate.");
  const bool nonsaturatingF32I8 =
      selected.starts_with("rvv.f32-i8-nonsaturating.");
  if ((!saturatedF32I8 && !nonsaturatingF32I8) || !rounding || !saturate ||
      saturate.getValue() != saturatedF32I8 || !source.isF32())
    return fail(operation,
                "current vector cast implements explicit saturated f32 narrowing only");
  auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
  if (!targetInteger || targetInteger.getWidth() != 8)
    return fail(operation,
                "current vector narrowing implements an eight-bit integer target");
  int64_t sourceLMUL = layoutOf(inputValue).getLmulEighths();
  int64_t targetLMUL = layoutOf(resultValue).getLmulEighths();
  auto lmul = [](int64_t eighths) {
    if (eighths < 8)
      return std::string("mf") + std::to_string(8 / eighths);
    return std::string("m") + std::to_string(eighths / 8);
  };
  if (sourceLMUL < 4 || targetLMUL * 4 != sourceLMUL)
    return fail(operation,
                "selected f32-to-i8 LMUL relation cannot be narrowed mechanically");
  const std::string i16Suffix =
      std::string(targetInteger.isUnsigned() ? "u16" : "i16") +
      lmul(sourceLMUL / 2);
  const std::string i16Type =
      std::string(targetInteger.isUnsigned() ? "vuint16" : "vint16") +
      lmul(sourceLMUL / 2) + "_t";
  const std::string i32Suffix =
      std::string(targetInteger.isUnsigned() ? "u32" : "i32") +
      lmul(sourceLMUL);
  const std::string i32Type =
      std::string(targetInteger.isUnsigned() ? "vuint32" : "vint32") +
      lmul(sourceLMUL) + "_t";
  const std::string targetSuffix = vectorSuffix(operation.getResult());
  const std::string targetType = vectorType(operation.getResult());
  llvm::StringRef frm = rounding.getValue() == "rne" ? "__RISCV_FRM_RNE"
                       : rounding.getValue() == "rtz" ? "__RISCV_FRM_RTZ"
                       : rounding.getValue() == "rdn" ? "__RISCV_FRM_RDN"
                       : rounding.getValue() == "rup" ? "__RISCV_FRM_RUP"
                                                       : llvm::StringRef();
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, part);
    if (!sourcePart || *sourcePart >= input->parts.size())
      return fail(operation,
                  "rounded narrowing requires an explicit physical layout conversion");
    const std::string vl = partVL(operation.getResult(), part);
    std::string i16Value = fresh("narrow_i16");
    if (saturatedF32I8) {
      std::string i32Value = fresh("narrow_i32");
      const std::string conversion =
          "__riscv_vfcvt_" +
          std::string(targetInteger.isUnsigned() ? "xu" : "x") + "_f_v_" +
          i32Suffix;
      line(i32Type + " " + i32Value + " = " + conversion +
           (frm.empty() ? "(" + input->parts[*sourcePart] + ", " + vl + ");"
                        : "_rm(" + input->parts[*sourcePart] + ", " + frm.str() +
                              ", " + vl + ");"));
      line(i16Type + " " + i16Value + " = __riscv_" +
           std::string(targetInteger.isUnsigned() ? "vnclipu" : "vnclip") +
           "_wx_" + i16Suffix + "(" + i32Value +
           ", 0, __RISCV_VXRM_RNE, " + vl + ");");
    } else {
      const std::string conversion =
          "__riscv_vfncvt_" +
          std::string(targetInteger.isUnsigned() ? "xu" : "x") + "_f_w_" +
          i16Suffix;
      line(i16Type + " " + i16Value + " = " + conversion +
           (frm.empty() ? "(" + input->parts[*sourcePart] + ", " + vl + ");"
                        : "_rm(" + input->parts[*sourcePart] + ", " + frm.str() +
                              ", " + vl + ");"));
    }
    std::string i8Value = fresh("narrow_i8");
    if (saturatedF32I8)
      line(targetType + " " + i8Value + " = __riscv_" +
           std::string(targetInteger.isUnsigned() ? "vnclipu" : "vnclip") +
           "_wx_" + targetSuffix + "(" + i16Value +
           ", 0, __RISCV_VXRM_RNE, " + vl + ");");
    else
      line(targetType + " " + i8Value + " = __riscv_vncvt_x_x_w_" +
           targetSuffix + "(" + i16Value + ", " + vl + ");");
    result.parts.push_back(std::move(i8Value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileBitcast(riscv::BitcastOp operation) {
  return compileCastImpl(operation);
}

mlir::LogicalResult Emitter::compileCast(riscv::CastOp operation) {
  return compileCastImpl(operation);
}

mlir::LogicalResult Emitter::compileNarrow(riscv::NarrowOp operation) {
  return compileCastImpl(operation);
}

mlir::LogicalResult Emitter::compileWiden(riscv::WidenOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  mlir::FailureOr<Binding> materialized =
      materializeNumeric(operation.getInput(), std::move(input));
  if (mlir::failed(materialized))
    return mlir::failure();
  input = std::move(*materialized);
  mlir::Type source =
      riscv_internal::logicalElement(operation.getInput().getType());
  mlir::Type target =
      riscv_internal::logicalElement(operation.getResult().getType());
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  auto targetCType = scalarCType(target);
  auto targetBindingKind = selectedBindingKind(operation.getResult());
  if (!targetBindingKind)
    return fail(operation, "widen result has no selected numeric representation");
  if (*targetBindingKind == Binding::Kind::Scalar ||
      *targetBindingKind == Binding::Kind::ScalarTuple) {
    if (selected != "scalar.widen")
      return fail(operation,
                  "scalar widening has no exact selected scalar instruction");
    if (!targetCType)
      return fail(operation, "scalar or tuple widening target has no C type");
    const int64_t parts = *targetBindingKind == Binding::Kind::Scalar
                              ? 1
                              : scalarPartCount(operation.getResult());
    Binding tuple;
    tuple.kind = *targetBindingKind;
    for (int64_t part = 0; part < parts; ++part) {
      std::optional<std::string> sourceExpression;
      if (input.kind == Binding::Kind::Scalar)
        sourceExpression = input.scalar;
      else if (input.kind == Binding::Kind::ScalarTuple) {
        auto sourcePart = mappedPart(operation.getOperation(), 0, part);
        if (sourcePart && *sourcePart < input.parts.size())
          sourceExpression = input.parts[*sourcePart];
      }
      if (!sourceExpression)
        return fail(operation,
                    "scalar or tuple widening requires its typed value-use conversion");
      std::string expression = "((" + *targetCType + ")(" +
                               *sourceExpression + "))";
      if (tuple.kind == Binding::Kind::Scalar)
        tuple.scalar = std::move(expression);
      else
        tuple.parts.push_back(std::move(expression));
    }
    bindings[operation.getResult()] = std::move(tuple);
    return mlir::success();
  }
  if (input.kind != Binding::Kind::Vector)
    return fail(operation, "widen requires a selected vector input");
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string targetType = vectorType(operation.getResult());
  const std::string targetSuffix = vectorSuffix(operation.getResult());
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto sourcePart = mappedPart(operation.getOperation(), 0, part);
    if (!sourcePart || *sourcePart >= input.parts.size())
      return fail(operation,
                  "widen requires an explicit physical layout conversion");
    const std::string &sourceExpression = input.parts[*sourcePart];
    const std::string vl = partVL(operation.getResult(), part);
    std::string expression;
    if ((selected == "rvv.fwiden.f16-f32" && source.isF16() && target.isF32()) ||
        (selected == "rvv.fwiden.f32-f64" && source.isF32() && target.isF64()))
      expression = "__riscv_vfwcvt_f_f_v_" + targetSuffix + "(" +
                   sourceExpression + ", " + vl + ")";
    else if (selected == "rvv.convert.i32-f32" && source.isInteger(32) &&
             target.isF32()) {
      auto integer = mlir::cast<mlir::IntegerType>(source);
      expression = std::string(integer.isSigned() ? "__riscv_vfcvt_f_x_v_"
                                                   : "__riscv_vfcvt_f_xu_v_") +
                   targetSuffix + "(" + sourceExpression + ", " + vl + ")";
    }
    else if (selected == "rvv.identity")
      expression = sourceExpression;
    else if (selected == "rvv.reinterpret") {
      const std::string sourceSuffix = vectorSuffix(operation.getInput());
      expression = "__riscv_vreinterpret_v_" + sourceSuffix + "_" +
                   targetSuffix + "(" + sourceExpression + ")";
    }
    else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(source)) {
      auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
      unsigned sourceWidth = std::max(8u, integer.getWidth());
      unsigned targetWidth = targetInteger ? targetInteger.getWidth() : 0;
      if (!targetInteger || targetWidth <= sourceWidth ||
          targetWidth % sourceWidth)
        return fail(operation, "unsupported integer widening width relation");
      unsigned factor = targetWidth / sourceWidth;
      if (factor != 2 && factor != 4 && factor != 8)
        return fail(operation, "unsupported RVV integer widening factor");
      std::string unsignedSuffix = targetSuffix;
      unsignedSuffix[0] = 'u';
      const std::string expected =
          std::string(integer.isSigned() ? "rvv.sext.vf" : "rvv.zext.vf") +
          std::to_string(factor);
      if (selected != expected)
        return fail(operation,
                    "selected integer widening leaf disagrees with width/sign");
      if (integer.isSigned()) {
        expression = "__riscv_vsext_vf" + std::to_string(factor) + "_" +
                     targetSuffix + "(" + sourceExpression + ", " + vl +
                     ")";
      } else {
        std::string widened =
            "__riscv_vzext_vf" + std::to_string(factor) + "_" + unsignedSuffix +
            "(" + sourceExpression + ", " + vl + ")";
        expression = targetSuffix[0] == 'i'
                         ? "__riscv_vreinterpret_v_" + unsignedSuffix + "_" +
                               targetSuffix + "(" + widened + ")"
                         : widened;
      }
    } else {
      return fail(operation, "unsupported vector widening relation");
    }
    std::string name = fresh("widen");
    line(targetType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileReduce(riscv::ReduceOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
  int64_t eliminatedAxis =
      inputType && operation.getAxis() >= 0 &&
              operation.getAxis() < static_cast<int64_t>(inputType.getAxisIds().size())
          ? inputType.getAxisIds()[operation.getAxis()]
          : 0;
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  llvm::StringRef selectedKind = selected;
  const bool selectedScalarRegister =
      selectedKind.consume_front("scalar.register-reduce.");
  bool selectedRVVRegister = false;
  bool selectedRVVLane = false;
  if (!selectedScalarRegister) {
    selectedRVVRegister =
        selectedKind.consume_front("rvv.register-reduce.");
    if (!selectedRVVRegister)
      selectedRVVLane = selectedKind.consume_front("rvv.lane-reduce.");
  }
  if (!selectedScalarRegister && !selectedRVVRegister && !selectedRVVLane)
    return fail(operation, "reduction has no exact selected reduction form");
  const bool registerAxis =
      llvm::is_contained(registerAxesFor(operation.getInput()), eliminatedAxis);
  if (selectedScalarRegister && !registerAxis &&
      riscv_internal::terminalInstruction(operation) == selected) {
    auto materialized = materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    auto scalarType = scalarCType(inputType.getElementType());
    if (!scalarType ||
        (materialized->kind != Binding::Kind::Scalar &&
         (materialized->kind != Binding::Kind::ScalarTuple ||
          materialized->parts.size() != 1)))
      return fail(operation,
                  "singleton reduction requires its one complete scalar value");
    const std::string value = materialized->kind == Binding::Kind::Scalar
                                  ? materialized->scalar
                                  : materialized->parts.front();
    const std::string name = fresh("reduce_scalar");
    const std::string identity = selectedKind == "add" ? "0"
        : selectedKind == "max" ? "-INFINITY" : "INFINITY";
    std::string expression;
    if (selectedKind == "add")
      expression = "(" + *scalarType + ")0 + " + value;
    else {
      const std::string function = std::string(selectedKind == "max" ? "fmax" : "fmin") +
          (inputType.getElementType().isF64() ? "" : "f");
      expression = function + "(" + identity + ", " + value + ")";
    }
    auto active = partActive(operation.getInput(), 0);
    if (!active)
      return fail(operation, "singleton reduction has no active member domain");
    // Preserve both the empty identity and floating signed-zero addition.
    if (inputType.getLayout().getValidity() == "tail")
      expression = "(" + *active + ") ? (" + expression + ") : (" + identity + ")";
    line(*scalarType + " " + name + " = (" + *scalarType + ")(" + expression + ");");
    bindings[operation.getResult()] = scalar(name);
    return mlir::success();
  }
  if ((selectedScalarRegister || selectedRVVRegister) != registerAxis)
    return fail(operation,
                "selected reduction form disagrees with the input axis mapping");
  if (selectedScalarRegister) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    auto resultType =
        mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    if (materialized->kind != Binding::Kind::ScalarTuple || !inputType ||
        !resultType || inputType.getLayout().getCarrier() != "scalar" ||
        resultType.getLayout().getCarrier() != "scalar" ||
        streamPartCount(operation.getInput()) != 1 ||
        streamPartCount(operation.getResult()) != 1)
      return fail(operation,
                  "scalar register reduction requires one typed scalar tuple without time parts");

    llvm::SmallVector<int64_t, 4> inputAxes =
        registerAxesFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> inputExtents =
        registerExtentsFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> resultAxes =
        registerAxesFor(operation.getResult());
    auto eliminatedPosition = llvm::find(inputAxes, eliminatedAxis);
    if (inputAxes.size() != inputExtents.size() ||
        eliminatedPosition == inputAxes.end())
      return fail(operation,
                  "scalar register reduction input mapping is incomplete");
    const int64_t eliminatedExtent =
        inputExtents[eliminatedPosition - inputAxes.begin()];
    auto scalarType = scalarCType(inputType.getElementType());
    if (eliminatedExtent <= 0 || !scalarType)
      return fail(operation,
                  "scalar register reduction has no complete scalar representation");

    llvm::StringRef kind = selectedKind;
    const bool arithmetic = kind == "add";
    if (!arithmetic && kind != "max" && kind != "min")
      return fail(operation,
                  "scalar register reduction implements add, max, and min");
    const int64_t resultRegisters = registerPartCount(operation.getResult());
    Binding result;
    result.kind = resultRegisters == 1 ? Binding::Kind::Scalar
                                       : Binding::Kind::ScalarTuple;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto resultCoordinates =
          registerCoordinates(operation.getResult(), resultRegister);
      if (!resultCoordinates || resultCoordinates->size() != resultAxes.size())
        return fail(operation,
                    "scalar register reduction result mapping is incomplete");
      std::string accumulator;
      for (int64_t reduced = 0; reduced < eliminatedExtent; ++reduced) {
        int64_t inputRegister = 0;
        for (auto [axis, extent] : llvm::zip(inputAxes, inputExtents)) {
          if (extent <= 0)
            return fail(operation,
                        "scalar register reduction has an invalid input extent");
          int64_t coordinate = reduced;
          if (axis != eliminatedAxis) {
            auto found = llvm::find(resultAxes, axis);
            if (found == resultAxes.end())
              return fail(operation,
                          "scalar register reduction cannot project a free axis");
            coordinate =
                (*resultCoordinates)[found - resultAxes.begin()];
          }
          if (coordinate < 0 || coordinate >= extent)
            return fail(operation,
                        "scalar register reduction coordinate is out of range");
          inputRegister = inputRegister * extent + coordinate;
        }
        if (inputRegister < 0 ||
            inputRegister >= static_cast<int64_t>(materialized->parts.size()))
          return fail(operation,
                      "scalar register reduction source mapping is incomplete");
        if (accumulator.empty()) {
          accumulator = fresh("register_reduce");
          line(*scalarType + " " + accumulator + " = " +
               materialized->parts[inputRegister] + ";");
          continue;
        }
        if (arithmetic)
          line(accumulator + " = (" + *scalarType + ")(" + accumulator +
               " + " + materialized->parts[inputRegister] + ");");
        else
          line(accumulator + " = (" + accumulator + ") " +
               std::string(kind == "max" ? ">" : "<") + " (" +
               materialized->parts[inputRegister] + ") ? (" + accumulator +
               ") : (" + materialized->parts[inputRegister] + ");");
      }
      if (result.kind == Binding::Kind::Scalar)
        result.scalar = std::move(accumulator);
      else
        result.parts.push_back(std::move(accumulator));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (selectedRVVRegister) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    if (materialized->kind != Binding::Kind::Vector)
      return fail(operation,
                  "register-axis reduction requires a selected vector input");
    llvm::SmallVector<int64_t, 4> inputAxes =
        registerAxesFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> inputExtents =
        registerExtentsFor(operation.getInput());
    llvm::SmallVector<int64_t, 4> resultAxes =
        registerAxesFor(operation.getResult());
    auto eliminatedPosition = llvm::find(inputAxes, eliminatedAxis);
    if (inputAxes.size() != inputExtents.size() ||
        eliminatedPosition == inputAxes.end())
      return fail(operation,
                  "register-axis reduction input mapping is incomplete");
    const size_t eliminatedOrdinal = eliminatedPosition - inputAxes.begin();
    const int64_t eliminatedExtent = inputExtents[eliminatedOrdinal];
    const int64_t inputStreams = streamPartCount(operation.getInput());
    const int64_t resultStreams = streamPartCount(operation.getResult());
    if (eliminatedExtent <= 0 || inputStreams <= 0 ||
        inputStreams != resultStreams ||
        laneAxisFor(operation.getInput()) != laneAxisFor(operation.getResult()))
      return fail(operation,
                  "register-axis reduction does not preserve its SIMD mapping");

    mlir::Type element =
        riscv_internal::logicalElement(operation.getInput().getType());
    const bool floating = mlir::isa<mlir::FloatType>(element);
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    llvm::StringRef kind = selectedKind;
    std::string instruction;
    if (kind == "add")
      instruction = floating ? "vfadd" : "vadd";
    else if ((kind == "max" || kind == "min") && floating)
      instruction = kind == "max" ? "vfmax" : "vfmin";
    else
      return fail(operation,
                  "register-axis reduction implements add and floating max/min");
    if (!floating && !integer)
      return fail(operation,
                  "register-axis reduction has no numeric vector element type");

    const std::string type = vectorType(operation.getResult());
    const std::string suffix = vectorSuffix(operation.getResult());
    const int64_t resultRegisters = registerPartCount(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto resultCoordinates =
          registerCoordinates(operation.getResult(), resultRegister);
      if (!resultCoordinates || resultCoordinates->size() != resultAxes.size())
        return fail(operation,
                    "register-axis reduction result mapping is incomplete");
      for (int64_t stream = 0; stream < resultStreams; ++stream) {
        std::string accumulator;
        for (int64_t reduced = 0; reduced < eliminatedExtent; ++reduced) {
          int64_t inputRegister = 0;
          for (auto [axis, extent] : llvm::zip(inputAxes, inputExtents)) {
            if (extent <= 0)
              return fail(operation,
                          "register-axis reduction has an invalid input extent");
            int64_t coordinate = reduced;
            if (axis != eliminatedAxis) {
              auto found = llvm::find(resultAxes, axis);
              if (found == resultAxes.end())
                return fail(operation,
                            "register-axis reduction cannot project a free axis");
              coordinate =
                  (*resultCoordinates)[found - resultAxes.begin()];
            }
            if (coordinate < 0 || coordinate >= extent)
              return fail(operation,
                          "register-axis reduction coordinate is out of range");
            inputRegister = inputRegister * extent + coordinate;
          }
          const int64_t inputPart = inputRegister * inputStreams + stream;
          if (inputPart < 0 ||
              inputPart >= static_cast<int64_t>(materialized->parts.size()))
            return fail(operation,
                        "register-axis reduction source mapping is incomplete");
          if (accumulator.empty()) {
            accumulator = fresh("register_reduce");
            line(type + " " + accumulator + " = " +
                 materialized->parts[inputPart] + ";");
            continue;
          }
          const int64_t resultPart = resultRegister * resultStreams + stream;
          line(accumulator + " = __riscv_" + instruction + "_vv_" + suffix +
               "(" + accumulator + ", " + materialized->parts[inputPart] +
               ", " + partVL(operation.getResult(), resultPart) + ");");
        }
        result.parts.push_back(std::move(accumulator));
      }
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  mlir::FailureOr<Binding> materialized =
      materializeNumeric(operation.getInput(), std::move(input));
  if (mlir::failed(materialized))
    return mlir::failure();
  if (materialized->kind != Binding::Kind::Vector)
    return fail(operation, "wide reduce requires a selected vector input");
  mlir::Type element =
      riscv_internal::logicalElement(operation.getInput().getType());
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "reduce element has no intrinsic-C scalar type");
  const std::string sourceSuffix = vectorSuffix(operation.getInput());
  unsigned sew = riscv_internal::logicalBitWidth(operation.getInput().getType());
  const bool floating = mlir::isa<mlir::FloatType>(element);
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  const bool unsignedInteger = integer && integer.isUnsigned();
  const std::string baseSuffix =
      std::string(floating ? "f" : unsignedInteger ? "u" : "i") +
      std::to_string(sew) + "m1";
  const std::string baseType =
      std::string(floating ? "vfloat" : unsignedInteger ? "vuint" : "vint") +
      std::to_string(sew) + "m1_t";
  llvm::StringRef kind = selectedKind;
  std::string initial;
  if (kind == "add")
    initial = "0";
  else if (kind == "max" && floating)
    initial = "-INFINITY";
  else if (kind == "min" && floating)
    initial = "INFINITY";
  else
    return fail(operation,
                "current wide reduce implements add and floating max/min");
  const int64_t resultParts = registerPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getInput());
  if (resultParts <= 0 || streams <= 0)
    return fail(operation,
                "reduce input/result mappings do not form an eliminated-axis slice");
  Binding result;
  result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                 : Binding::Kind::ScalarTuple;
  const bool combineFullStreams =
      streams > 1 && inputType.getLayout().getValidity() == "full";
  std::string horizontalStem;
  if (operation.getLeaf().getTail() != "exact")
    return fail(operation, "lane reduction requires its selected seed-preserving destination policy");
  if (floating)
    horizontalStem = kind == "add"   ? "vfredusum"
                     : kind == "max" ? "vfredmax"
                                       : "vfredmin";
  else
    horizontalStem = "vredsum";
  for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
    auto sourceRegister = projectRegisterPart(
        operation.getInput(), operation.getResult(), resultPart);
    if (!sourceRegister)
      return fail(operation,
                  "reduce free-axis mapping is not projectable");
    std::string accumulator = fresh("reduce_scalar");
    line(*scalarType + " " + accumulator + " = (" + *scalarType + ")(" +
         initial + ");");
    auto emitHorizontal = [&](llvm::StringRef source, size_t index) {
      const std::string seed = fresh("reduce_seed");
      line(baseType + " " + seed + " = __riscv_" +
           std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + baseSuffix +
           "(" + accumulator + ", 1);");
      const std::string reduced = fresh("reduce_vector");
      line(baseType + " " + reduced + " = __riscv_" + horizontalStem +
           "_vs_" + sourceSuffix + "_" + baseSuffix + "_tu(" + seed + ", " + source.str() +
           ", " + seed + ", " + partVL(operation.getInput(), index) + ");");
      if (floating)
        line(accumulator + " = __riscv_vfmv_f_s_" + baseSuffix + "_f" +
             std::to_string(sew) + "(" + reduced + ");");
      else
        line(accumulator + " = __riscv_vmv_x_s_" + baseSuffix + "_" +
             std::string(unsignedInteger ? "u" : "i") +
             std::to_string(sew) + "(" + reduced + ");");
    };
    if (combineFullStreams) {
      const size_t first = *sourceRegister * streams;
      if (first + streams > materialized->parts.size())
        return fail(operation, "reduce source mapping is incomplete");
      const std::string vectorTypeName = vectorType(operation.getInput());
      const std::string combineStem =
          floating ? (kind == "add" ? "vfadd" : kind == "max" ? "vfmax"
                                                              : "vfmin")
                   : "vadd";
      std::string combined = fresh("reduce_streams");
      line(vectorTypeName + " " + combined + " = " +
           materialized->parts[first] + ";");
      for (int64_t stream = 1; stream < streams; ++stream)
        line(combined + " = __riscv_" + combineStem + "_vv_" + sourceSuffix +
             "(" + combined + ", " + materialized->parts[first + stream] +
             ", " + std::to_string(physicalLanes(operation.getInput())) +
             ");");
      emitHorizontal(combined, first);
    } else for (int64_t stream = 0; stream < streams; ++stream) {
      const size_t index = *sourceRegister * streams + stream;
      if (index >= materialized->parts.size())
        return fail(operation, "reduce source mapping is incomplete");
      emitHorizontal(materialized->parts[index], index);
    }
    if (resultParts == 1)
      result.scalar = accumulator;
    else
      result.parts.push_back(std::move(accumulator));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFold2(riscv::Fold2Op operation) {
  if (operation.getLeaf().getInstruction() != "rvv.pair-fold-add")
    return fail(operation, "fold2 has no exact selected RVV pair-fold leaf");
  Binding fieldBinding = bindings.lookup(operation.getInput());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation,
                "pair-fold leaf requires one explicit encoded-field operand");
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto input = mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  auto inputInteger = input
                          ? mlir::dyn_cast<mlir::IntegerType>(input.getElementType())
                          : mlir::IntegerType();
  auto resultInteger = result
                           ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                           : mlir::IntegerType();
  auto resultLayout = result ? result.getLayout() : riscv::LayoutAttr();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !field || field->access != operation.getAccess() ||
      operation.getAccess().getMapping() != "natural" ||
      operation.getAccess().getBitOffset() % 8 ||
      !inputInteger || inputInteger.getWidth() != 16 ||
      !inputInteger.isSigned() || !resultInteger ||
      resultInteger.getWidth() != 32 || !resultInteger.isSigned() ||
      !resultLayout || resultLayout.getCarrier() != "rvv" ||
      resultLayout.getLmulEighths() <= 0 ||
      resultLayout.getLmulEighths() % 2)
    return fail(operation,
                "pair-fold leaf has no closed natural i16-to-i32 RVV representation");

  const int64_t inputLMUL = resultLayout.getLmulEighths() / 2;
  const std::string inputSuffix = "i16" + lmulSpelling(inputLMUL);
  const std::string inputType = "vint16" + lmulSpelling(inputLMUL) + "_t";
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  Binding folded;
  folded.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || !coordinates ||
        coordinates->size() != registerAxes.size())
      return fail(operation,
                  "pair-fold result has no complete register coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates))
      for (const auto &[recordAxis, stride] : owner.recordByteStrides)
        if (recordAxis == axis)
          record += " + " + std::to_string(coordinate) + " * " + stride;
    record += " + " + std::to_string(field->bitOffset / 8) + ")";
    const std::string outputOffset = partOffset(operation.getResult(), part);
    const std::string evenAddress =
        record + " + 4 * (" + outputOffset + ")";
    const std::string oddAddress = evenAddress + " + 2";
    const std::string vl = partVL(operation.getResult(), part);
    std::string even = fresh("pair_even");
    std::string odd = fresh("pair_odd");
    line(inputType + " " + even + " = __riscv_vlse16_v_" + inputSuffix +
         "((const int16_t *)(" + evenAddress + "), (ptrdiff_t)4, " + vl +
         ");");
    line(inputType + " " + odd + " = __riscv_vlse16_v_" + inputSuffix +
         "((const int16_t *)(" + oddAddress + "), (ptrdiff_t)4, " + vl +
         ");");
    std::string sum = fresh("pair_fold");
    line(resultType + " " + sum + " = __riscv_vwadd_vv_" + resultSuffix +
         "(" + even + ", " + odd + ", " + vl + ");");
    folded.parts.push_back(std::move(sum));
  }
  bindings[operation.getResult()] = std::move(folded);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileBinary(riscv::BinaryOp operation) {
  mlir::FailureOr<Binding> lhsOr =
      materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  mlir::FailureOr<Binding> rhsOr =
      materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhsOr) || mlir::failed(rhsOr))
    return mlir::failure();
  Binding lhs = std::move(*lhsOr);
  Binding rhs = std::move(*rhsOr);
  llvm::StringRef selected = operation.getLeaf().getInstruction();
  if (selected == "rvv.mul.row-scale") {
    auto rows = riscv::rvvRowScaleRows(operation);
    if (!rows || lhs.kind != Binding::Kind::ScalarTuple || lhs.parts.size() != static_cast<size_t>(*rows) ||
        rhs.kind != Binding::Kind::Vector)
      return fail(operation, "row scale requires its selected scalar-row/vector-column relation");
    auto inputType = mlir::cast<riscv::ValueType>(operation.getRhs().getType());
    auto outputType = mlir::cast<riscv::ValueType>(operation.getResult().getType());
    const int64_t group = outputType.getLayout().getReplicaGroup();
    const std::string columns = std::to_string(outputType.getLayout().getLaneFactors()[1]);
    const std::string pairSuffix = vectorSuffix(operation.getRhs());
    const std::string productSuffix = "f32" + lmulSpelling(inputType.getLayout().getLmulEighths() / group);
    const std::string atomSuffix = "f32" + lmulSpelling(outputType.getLayout().getLmulEighths() / group);
    const std::string outputSuffix = vectorSuffix(operation.getResult());
    auto join = [&](llvm::ArrayRef<std::string> parts, llvm::StringRef from, llvm::StringRef to) {
      if (parts.size() == 1) return parts.front();
      auto name = fresh("scale_group");
      line("vfloat" + to.substr(1).str() + "_t " + name + " = __riscv_vlmul_ext_v_" +
           from.str() + "_" + to.str() + "(" + parts.front() + ");");
      for (size_t index = 1; index < parts.size(); ++index)
        line(name + " = __riscv_vset_v_" + from.str() + "_" + to.str() + "(" + name + ", " +
             std::to_string(index) + ", " + parts[index] + ");");
      return name;
    };
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto sourcePart = riscv::rvvBroadcastSourcePart(inputType, outputType, part);
      if (!sourcePart || *sourcePart >= static_cast<int64_t>(rhs.parts.size()))
        return fail(operation, "row scale has no complete column carrier projection");
      llvm::SmallVector<std::string> atoms;
      for (int64_t atom = 0; atom < group; ++atom) {
        std::string input = rhs.parts[*sourcePart];
        if (atom)
          input = "__riscv_vslidedown_vx_" + pairSuffix + "(" + input + ", " +
                  std::to_string(atom * outputType.getLayout().getLaneFactors()[1]) + ", " + columns + ")";
        input = "__riscv_vlmul_trunc_v_" + pairSuffix + "_" + productSuffix + "(" + input + ")";
        llvm::SmallVector<std::string> products, pairs;
        for (int64_t row = 0; row < *rows; ++row) {
          auto name = fresh("row_product");
          line("vfloat" + productSuffix.substr(1) + "_t " + name + " = __riscv_vfmul_vf_" +
               productSuffix + "(" + input + ", " + lhs.parts[row] + ", " + columns + ");");
          products.push_back(std::move(name));
        }
        for (int64_t row = 0; row < *rows; row += 2) {
          auto name = fresh("row_pair");
          line("vfloat" + pairSuffix.substr(1) + "_t " + name + " = __riscv_vlmul_ext_v_" +
               productSuffix + "_" + pairSuffix + "(" + products[row] + ");");
          line(name + " = __riscv_vslideup_vx_" + pairSuffix + "_tu(" + name +
               ", __riscv_vlmul_ext_v_" + productSuffix + "_" + pairSuffix + "(" + products[row + 1] +
               "), " + columns + ", " + std::to_string(2 * outputType.getLayout().getLaneFactors()[1]) + ");");
          pairs.push_back(std::move(name));
        }
        atoms.push_back(join(pairs, pairSuffix, atomSuffix));
      }
      result.parts.push_back(join(atoms, atomSuffix, outputSuffix));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  auto scalarTupleCompatible = [](const Binding &binding) {
    return binding.kind == Binding::Kind::Scalar ||
           binding.kind == Binding::Kind::ScalarTuple;
  };
  if (scalarTupleCompatible(lhs) && scalarTupleCompatible(rhs)) {
    llvm::StringRef kind = selected;
    if (!kind.consume_front("scalar."))
      return fail(operation,
                  "scalar-tuple binary has no exact selected scalar instruction");
    llvm::StringRef spelling = kind == "add" ? "+"
                              : kind == "sub" ? "-"
                              : kind == "mul" ? "*"
                              : kind == "div" ? "/"
                              : kind == "mod" ? "%"
                              : kind == "and" ? "&"
                              : kind == "or"  ? "|"
                              : kind == "xor" ? "^"
                              : kind == "shl" ? "<<"
                              : kind == "shr" ? ">>"
                                                : "";
    if (spelling.empty() && kind != "max" && kind != "min")
      return fail(operation,
                  "unsupported selected scalar-tuple pointwise binary kind");
    const int64_t parts = mlir::isa<riscv::ValueType>(operation.getResult().getType())
        ? scalarPartCount(operation.getResult()) : 1;
    auto scalarResultType = scalarCType(
        riscv_internal::logicalElement(operation.getResult().getType()));
    if (!scalarResultType)
      return fail(operation,
                  "scalar-tuple binary result has no exact intrinsic-C type");
    Binding result;
    result.kind = parts == 1 ? Binding::Kind::Scalar
                             : Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < parts; ++part) {
      auto expressionFor = [&](size_t operand, const Binding &binding)
          -> std::optional<std::string> {
        if (binding.kind == Binding::Kind::Scalar)
          return binding.scalar;
        auto projected = mappedPart(operation.getOperation(), operand, part);
        if (!projected || *projected >= binding.parts.size())
          return std::nullopt;
        return binding.parts[*projected];
      };
      auto lhsExpression =
          expressionFor(0, lhs);
      auto rhsExpression =
          expressionFor(1, rhs);
      if (!lhsExpression || !rhsExpression)
        return fail(operation,
                    "scalar-tuple operand projection is not representable");
      std::string expression;
      if (!spelling.empty())
        expression = "(" + *lhsExpression + " " + spelling.str() + " " +
                     *rhsExpression + ")";
      else
        expression = "((" + *lhsExpression + ") " +
                     std::string(kind == "max" ? ">" : "<") + " (" +
                     *rhsExpression + ") ? (" + *lhsExpression + ") : (" +
                     *rhsExpression + "))";
      std::string name = materializeScalarRead(*scalarResultType, expression, "pointwise");
      if (parts == 1) result.scalar = std::move(name);
      else result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const bool lhsIsVectorOperand =
      lhs.kind == Binding::Kind::Vector;
  const Binding &vector = lhsIsVectorOperand ? lhs : rhs;
  const Binding &other = lhsIsVectorOperand ? rhs : lhs;
  if (vector.kind != Binding::Kind::Vector ||
      (other.kind != Binding::Kind::Vector &&
       other.kind != Binding::Kind::Scalar &&
       other.kind != Binding::Kind::ScalarTuple))
    return fail(operation, "pointwise binary has no selected vector/scalar handoff; lhs=" +
        riscv_internal::printType(operation.getLhs().getType()) + ", binding=" +
        std::to_string(static_cast<int>(lhs.kind)) + "; rhs=" +
        riscv_internal::printType(operation.getRhs().getType()) + ", binding=" +
        std::to_string(static_cast<int>(rhs.kind)) + "; result=" +
        riscv_internal::printType(operation.getResult().getType()));
  Binding result;
  result.kind = Binding::Kind::Vector;
  const std::string type = vectorType(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  bool floating = suffix.front() == 'f';
  llvm::StringRef leaf = selected;
  if (!leaf.consume_front("rvv."))
    return fail(operation,
                ("vector binary has selected instruction '" + selected +
                 "' for operand types " +
                 riscv_internal::printType(operation.getLhs().getType()) +
                 " and " +
                 riscv_internal::printType(operation.getRhs().getType()) +
                 ", result " +
                 riscv_internal::printType(operation.getResult().getType()))
                    .str());
  size_t separator = leaf.find('.');
  if (separator == llvm::StringRef::npos)
    return fail(operation, "selected RVV binary instruction has no operand form");
  std::string stem = leaf.take_front(separator).str();
  llvm::StringRef form = leaf.drop_front(separator + 1);
  const bool leafVV = form == "vv";
  const bool leafBroadcast = form == "vv.broadcast";
  const bool leafSplatLhs = form == "vv.splat-lhs";
  const bool leafScalar = form == (floating ? "vf" : "vx") ||
                          form == (floating ? "vf.swap" : "vx.swap");
  const bool leafSwap = form.ends_with(".swap");
  const int64_t resultParts = vectorPartCount(operation.getResult());
  if (leafBroadcast) {
    if (lhs.kind != Binding::Kind::Vector || rhs.kind != Binding::Kind::Vector ||
        !riscv::binaryBroadcastTemporaryGroups(operation))
      return fail(operation, "binary lane broadcast has no closed two-operand relation");
    std::optional<int64_t> previousPart[2];
    std::string previousValue[2];
    for (int64_t part = 0; part < resultParts; ++part) {
      std::string operands[2];
      for (unsigned side = 0; side < 2; ++side) {
        mlir::Value value = operation->getOperand(side);
        const Binding &input = side == 0 ? lhs : rhs;
        auto inputType = mlir::cast<riscv::ValueType>(value.getType());
        auto resultType = mlir::cast<riscv::ValueType>(operation.getResult().getType());
        auto sourcePart = riscv::rvvBroadcastSourcePart(inputType, resultType, part);
        if (!sourcePart)
          return fail(operation, "binary lane broadcast lost its source part");
        if (!riscv::needsRVVAxisBroadcast(inputType, resultType)) {
          operands[side] = input.parts[*sourcePart];
        } else if (previousPart[side] == sourcePart) {
          operands[side] = previousValue[side];
        } else {
          auto broadcast = axisBroadcastPart(operation, value, operation.getResult(), part, input);
          if (mlir::failed(broadcast))
            return mlir::failure();
          operands[side] = *broadcast;
          previousPart[side] = sourcePart;
          previousValue[side] = *broadcast;
        }
      }
      auto name = fresh("pointwise");
      line(type + " " + name + " = __riscv_" + stem + "_vv_" + suffix + "(" +
           operands[0] + ", " + operands[1] + ", " + partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  std::string incompatibleMapping;
  auto compatibleParts = [&](size_t operand, const Binding &binding) {
    if (binding.kind == Binding::Kind::Scalar)
      return true;
    if (binding.kind != Binding::Kind::Vector &&
        binding.kind != Binding::Kind::ScalarTuple) {
      incompatibleMapping = "operand " + std::to_string(operand) +
                            " has binding kind " +
                            std::to_string(static_cast<int>(binding.kind));
      return false;
    }
    for (int64_t part = 0; part < resultParts; ++part) {
      auto projected = mappedPart(operation.getOperation(), operand, part);
      if (!projected || *projected >= binding.parts.size()) {
        incompatibleMapping =
            "operand " + std::to_string(operand) + ", result part " +
            std::to_string(part) + ", binding parts " +
            std::to_string(binding.parts.size()) + ", operand type " +
            riscv_internal::printType(operation->getOperand(operand).getType());
        return false;
      }
    }
    return true;
  };
  if (resultParts <= 0 ||
      !compatibleParts(0, lhs) || !compatibleParts(1, rhs))
    return fail(operation,
                "pointwise operand mappings cannot broadcast to the selected result mapping: " +
                    incompatibleMapping);
  for (int64_t index = 0; index < resultParts; ++index) {
    std::string expression;
    if (other.kind == Binding::Kind::Scalar ||
        other.kind == Binding::Kind::ScalarTuple) {
      const size_t vectorOperand = lhsIsVectorOperand ? 0 : 1;
      const size_t otherOperand = lhsIsVectorOperand ? 1 : 0;
      auto vectorPartIndex =
          mappedPart(operation.getOperation(), vectorOperand, index);
      auto scalarPartIndex =
          other.kind == Binding::Kind::Scalar
              ? std::optional<size_t>(0)
              : mappedPart(operation.getOperation(), otherOperand, index);
      if (!vectorPartIndex || !scalarPartIndex)
        return fail(operation,
                    "pointwise operand projection is not representable");
      std::string vectorPart = vector.parts[*vectorPartIndex];
      const std::string scalarPart =
          other.kind == Binding::Kind::Scalar
              ? other.scalar
              : other.parts[*scalarPartIndex];
      bool scalarOnLeft = !lhsIsVectorOperand;
      if (leafSplatLhs) {
        if (!scalarOnLeft)
          return fail(operation,
                      "selected scalar-left splat form disagrees with operands");
        std::string splat = "__riscv_" +
                            std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
                            suffix + "(" + scalarPart + ", " +
                            partVL(operation.getResult(), index) + ")";
        expression = "__riscv_" + stem + "_vv_" + suffix + "(" + splat +
                     ", " + vectorPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      } else {
        if (!leafScalar || leafSwap != scalarOnLeft)
          return fail(operation,
                      "selected vector-scalar form disagrees with operands");
        expression = "__riscv_" + stem + (floating ? "_vf_" : "_vx_") +
                     suffix + "(" + vectorPart + ", " + scalarPart + ", " +
                     partVL(operation.getResult(), index) + ")";
      }
    } else {
      if (!leafVV)
        return fail(operation,
                    "selected vector-vector form disagrees with operands");
      auto lhsPartIndex = mappedPart(operation.getOperation(), 0, index);
      auto rhsPartIndex = mappedPart(operation.getOperation(), 1, index);
      if (!lhsPartIndex || !rhsPartIndex)
        return fail(operation,
                    "pointwise vector operand projection is not representable");
      const std::string &lhsPart = lhs.parts[*lhsPartIndex];
      const std::string &rhsPart = rhs.parts[*rhsPartIndex];
      expression = "__riscv_" + stem + "_vv_" + suffix + "(" + lhsPart +
                   ", " + rhsPart + ", " +
                   partVL(operation.getResult(), index) + ")";
    }
    // Preserve the physical SSA operation as one C value.  Inlining the
    // expression into each downstream use would duplicate loop-invariant
    // coordinate work after LICM and would no longer spell the selected
    // physical program mechanically.
    std::string name = fresh("pointwise");
    line(type + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenAdd(riscv::RVVWidenAddOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs) ||
      lhs->kind != Binding::Kind::Vector || rhs->kind != Binding::Kind::Vector)
    return fail(operation, "RVV widening add requires two selected vectors");
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.vwadd.vv" && instruction != "rvv.vwaddu.vv")
    return fail(operation, "RVV widening add has no exact selected instruction");
  const std::string intrinsic =
      std::string(instruction == "rvv.vwadd.vv" ? "__riscv_vwadd_vv_"
                                               : "__riscv_vwaddu_vv_") +
      vectorSuffix(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto left = mappedPart(operation.getOperation(), 0, part);
    auto right = mappedPart(operation.getOperation(), 1, part);
    if (!left || !right || *left >= lhs->parts.size() || *right >= rhs->parts.size())
      return fail(operation, "RVV widening add has no closed operand-part mapping");
    std::string name = fresh("widen_add");
    line(vectorType(operation.getResult()) + " " + name + " = " + intrinsic +
         "(" + lhs->parts[*left] + ", " + rhs->parts[*right] + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVNarrowShiftRight(
    riscv::RVVNarrowShiftRightOp operation) {
  auto input = materializeNumeric(operation.getInput(),
                                  bindings.lookup(operation.getInput()));
  auto amount = materializeNumeric(operation.getAmount(),
                                   bindings.lookup(operation.getAmount()));
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool vectorAmount = instruction == "rvv.vnsrl.wv";
  if (mlir::failed(input) || mlir::failed(amount) ||
      input->kind != Binding::Kind::Vector ||
      (instruction != "rvv.vnsrl.wv" && instruction != "rvv.vnsrl.wx") ||
      amount->kind != (vectorAmount ? Binding::Kind::Vector : Binding::Kind::Scalar))
    return fail(operation, "RVV narrowing shift operands disagree with selected instruction");
  const std::string intrinsic =
      std::string(vectorAmount ? "__riscv_vnsrl_wv_" : "__riscv_vnsrl_wx_") +
      vectorSuffix(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto source = mappedPart(operation.getOperation(), 0, part);
    if (!source || *source >= input->parts.size())
      return fail(operation, "RVV narrowing shift has no closed input-part mapping");
    std::string shift = amount->scalar;
    if (vectorAmount) {
      auto index = mappedPart(operation.getOperation(), 1, part);
      if (!index || *index >= amount->parts.size())
        return fail(operation, "RVV narrowing shift has no closed amount-part mapping");
      shift = amount->parts[*index];
    }
    std::string name = fresh("narrow_shift");
    line(vectorType(operation.getResult()) + " " + name + " = " + intrinsic +
         "(" + input->parts[*source] + ", " + shift + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenMultiply(
    riscv::RVVWidenMultiplyOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs) ||
      lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV widening multiply requires two selected vector operands");
  const std::string instruction = instructionOf(operation.getOperation()).str();
  if (instruction != "rvv.vwmulu.vv" && instruction != "rvv.vwmul.vv" &&
      instruction != "rvv.vwmulsu.vv" &&
      instruction != "rvv.vwmulsu.vv.swap")
    return fail(operation,
                "RVV widening multiply has no exact selected instruction");
  Binding result;
  result.kind = Binding::Kind::Vector;
  std::string suffix = vectorSuffix(operation.getResult());
  std::string unsignedSuffix = suffix;
  unsignedSuffix[0] = 'u';
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto lhsPart = mappedPart(operation.getOperation(), 0, part);
    auto rhsPart = mappedPart(operation.getOperation(), 1, part);
    if (!lhsPart || !rhsPart || *lhsPart >= lhs->parts.size() ||
        *rhsPart >= rhs->parts.size())
      return fail(operation,
                  "RVV widening multiply has no closed operand-part mapping");
    std::string intrinsic;
    std::string left = lhs->parts[*lhsPart];
    std::string right = rhs->parts[*rhsPart];
    if (instruction == "rvv.vwmulu.vv")
      intrinsic = "__riscv_vwmulu_vv_" + unsignedSuffix;
    else {
      intrinsic = "__riscv_" +
                  std::string(instruction == "rvv.vwmul.vv"
                                  ? "vwmul_vv_"
                                  : "vwmulsu_vv_") +
                  suffix;
      if (instruction == "rvv.vwmulsu.vv.swap")
        std::swap(left, right);
    }
    std::string expression = intrinsic + "(" + left + ", " + right + ", " +
                             partVL(operation.getResult(), part) + ")";
    if (instruction == "rvv.vwmulu.vv" && suffix[0] == 'i')
      expression = "__riscv_vreinterpret_v_" + unsignedSuffix + "_" + suffix +
                   "(" + expression + ")";
    std::string name = fresh("widen_multiply");
    line(vectorType(operation.getResult()) + " " + name + " = " + expression +
         ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenScalarMultiply(
    riscv::RVVWidenScalarMultiplyOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs) ||
      lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV widening scalar multiply requires one vector and one scalar operand");
  const std::string instruction = instructionOf(operation.getOperation()).str();
  if (instruction != "rvv.vwmulu.vx" && instruction != "rvv.vwmul.vx" &&
      instruction != "rvv.vwmulsu.vx")
    return fail(operation,
                "RVV widening scalar multiply has no exact selected instruction");
  Binding result;
  result.kind = Binding::Kind::Vector;
  std::string suffix = vectorSuffix(operation.getResult());
  std::string unsignedSuffix = suffix;
  unsignedSuffix[0] = 'u';
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto lhsPart = mappedPart(operation.getOperation(), 0, part);
    if (!lhsPart || *lhsPart >= lhs->parts.size())
      return fail(operation,
                  "RVV widening scalar multiply has no closed operand-part mapping");
    std::string intrinsic;
    if (instruction == "rvv.vwmulu.vx")
      intrinsic = "__riscv_vwmulu_vx_" + unsignedSuffix;
    else
      intrinsic = "__riscv_" +
                  std::string(instruction == "rvv.vwmul.vx"
                                  ? "vwmul_vx_"
                                  : "vwmulsu_vx_") +
                  suffix;
    std::string expression =
        intrinsic + "(" + lhs->parts[*lhsPart] + ", " + rhs->scalar + ", " +
        partVL(operation.getResult(), part) + ")";
    if (instruction == "rvv.vwmulu.vx" && suffix[0] == 'i')
      expression = "__riscv_vreinterpret_v_" + unsignedSuffix + "_" + suffix +
                   "(" + expression + ")";
    std::string name = fresh("widen_scalar_multiply");
    line(vectorType(operation.getResult()) + " " + name + " = " + expression +
         ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVMultiplyHighScalar(
    riscv::RVVMultiplyHighScalarOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  if (mlir::failed(lhs) || mlir::failed(rhs) ||
      lhs->kind != Binding::Kind::Vector || rhs->kind != Binding::Kind::Scalar ||
      instructionOf(operation.getOperation()) != "rvv.vmulhu.vx")
    return fail(operation,
                "RVV multiply-high requires its selected vector-scalar leaf");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto source = mappedPart(operation.getOperation(), 0, part);
    if (!source || *source >= lhs->parts.size())
      return fail(operation,
                  "RVV multiply-high has no closed operand-part mapping");
    std::string name = fresh("multiply_high");
    line(vectorType(operation.getResult()) + " " + name +
         " = __riscv_vmulhu_vx_" + vectorSuffix(operation.getResult()) + "(" +
         lhs->parts[*source] + ", " + rhs->scalar + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVSplat(riscv::RVVSplatOp operation) {
  const bool local = instructionOf(operation.getOperation()) == "rvv.splat.local";
  if (!local && instructionOf(operation.getOperation()) != "rvv.splat")
    return fail(operation, "RVV splat has no exact selected leaf");
  Binding scalarValue = bindings.lookup(operation.getScalar());
  if (scalarValue.kind != Binding::Kind::Scalar)
    return fail(operation, "RVV splat input has no scalar representation");
  if (local) {
    Binding result;
    result.kind = Binding::Kind::Vector;
    const bool floating = usesFloatingRegisters(operation.getResult().getType().getElementType());
    auto layout = layoutOf(operation.getResult());
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto name = fresh("local_splat");
      line(vectorType(operation.getResult()) + " " + name + ";");
      line("__asm__ volatile(\"vsetvli zero, %1, e" + std::to_string(layout.getSew()) + ", " +
           lmulSpelling(layout.getLmulEighths()) + ", ta, ma\\n\\t" +
           (floating ? "vfmv.v.f" : "vmv.v.x") + " %0, %2\" : \"=vr\"(" + name +
           ") : \"r\"((size_t)" + partVL(operation.getResult(), part) + "), \"" +
           (floating ? "f" : "r") + "\"(" + scalarValue.scalar + "));");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  mlir::FailureOr<Binding> result = makeVector(
      operation.getResult(), "splat", scalarValue, operation.getScalar());
  if (mlir::failed(result))
    return mlir::failure();
  bindings[operation.getResult()] = std::move(*result);
  return mlir::success();
}

} // namespace weft::riscv_emission
