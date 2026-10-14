#include "Emitter.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileRVVAssembleValue(riscv::RVVAssembleValueOp operation) {
  if (instructionOf(operation) != "rvv.assemble-value")
    return fail(operation, "value assembly has no selected handoff");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (mlir::Value piece : operation.getPieces()) {
    auto input = materializeNumeric(piece, bindings.lookup(piece));
    if (mlir::failed(input) || input->kind != Binding::Kind::Vector || input->parts.size() != 1)
      return fail(operation, "value assembly requires one register carrier per issue");
    result.parts.push_back(input->parts.front());
  }
  if (result.parts.size() != static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation, "value assembly lost its exact issue-time partition");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVIssueSlice(riscv::RVVIssueSliceOp operation) {
  const bool registerSlice =
      instructionOf(operation.getOperation()) == "rvv.issue-slice.split";
  if (!registerSlice && instructionOf(operation.getOperation()) != "rvv.issue-slice")
    return fail(operation, "RVV issue slice has no exact selected leaf");
  auto input = materializeNumeric(operation.getInput(),
                                  bindings.lookup(operation.getInput()));
  const int64_t resultParts = vectorPartCount(operation.getResult());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  const int64_t sourceLMUL = inputType.getLayout().getLmulEighths();
  const int64_t resultLMUL = resultType.getLayout().getLmulEighths();
  const int64_t sourceLanes = physicalLanesFor(inputType);
  const std::string sourceSuffix = vectorSuffix(operation.getInput());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
      resultParts <= 0 || sourceLMUL <= 0 || resultLMUL <= 0 ||
      sourceLanes <= 0 || sourceSuffix.empty() || resultSuffix.empty() ||
      operation.getSourceParts().size() != static_cast<size_t>(resultParts) ||
      operation.getLaneOffsets().size() != static_cast<size_t>(resultParts))
    return fail(operation,
                "RVV issue slice requires one materialized source vector per result part");
  Binding result;
  result.kind = Binding::Kind::Vector;
  result.parts.reserve(static_cast<size_t>(resultParts));
  for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
    const int64_t sourcePart = operation.getSourceParts()[resultPart];
    const int64_t laneOffset = operation.getLaneOffsets()[resultPart];
    if (sourcePart < 0 || sourcePart >= static_cast<int64_t>(input->parts.size()))
      return fail(operation,
                  "RVV issue slice references an absent source vector part");
    std::string sliced = input->parts[static_cast<size_t>(sourcePart)];
    if (registerSlice) {
      std::string selected = fresh("issue_slice_part");
      line(vectorType(operation.getResult()) + " " + selected +
           " = __riscv_vget_v_" + sourceSuffix + "_" + resultSuffix + "(" + sliced +
           ", " + std::to_string(laneOffset / physicalLanesFor(resultType)) + ");");
      sliced = std::move(selected);
    } else if (sourceLMUL != resultLMUL) {
      if (sourceLMUL < resultLMUL || sourceLMUL % resultLMUL)
        return fail(operation,
                    "RVV issue slice requires an integral LMUL truncation");
      if (laneOffset != 0) {
        std::string shifted = fresh("issue_slice_lane");
        line(vectorType(operation.getInput()) + " " + shifted +
             " = __riscv_vslidedown_vx_" + sourceSuffix + "(" + sliced + ", " +
             std::to_string(laneOffset) + ", " +
             std::to_string(sourceLanes) + ");");
        sliced = std::move(shifted);
      }
      std::string truncated = fresh("issue_slice_part");
      line(vectorType(operation.getResult()) + " " + truncated +
           " = __riscv_vlmul_trunc_v_" + sourceSuffix + "_" + resultSuffix +
           "(" + sliced + ");");
      sliced = std::move(truncated);
    } else if (laneOffset != 0) {
      std::string shifted = fresh("issue_slice_lane");
      line(vectorType(operation.getResult()) + " " + shifted +
           " = __riscv_vslidedown_vx_" + resultSuffix + "(" + sliced + ", " +
           std::to_string(laneOffset) + ", " +
           partVL(operation.getResult(), resultPart) + ");");
      sliced = std::move(shifted);
    }
    result.parts.push_back(std::move(sliced));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVWidenAccumulate(
    riscv::RVVWidenAccumulateOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.vwmacc.partial")
    return fail(operation,
                "RVV widened partial accumulation has no exact selected leaf");
  auto lhs = materializeNumeric(operation.getLhs(),
                                bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(),
                                bindings.lookup(operation.getRhs()));
  auto accumulator = materializeNumeric(
      operation.getAccumulator(), bindings.lookup(operation.getAccumulator()));
  if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(accumulator) ||
      lhs->kind != Binding::Kind::Vector ||
      rhs->kind != Binding::Kind::Vector ||
      accumulator->kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV widened partial accumulation requires materialized vectors");
  auto lhsElement = operation.getLhs().getType().getElementType();
  auto rhsElement = operation.getRhs().getType().getElementType();
  const bool mixed = lhsElement.isSignedInteger() != rhsElement.isSignedInteger();
  const std::string suffix = vectorSuffix(operation.getAccumulator());
  const std::string type = vectorType(operation.getAccumulator());
  Binding result;
  result.kind = Binding::Kind::Vector;
  const int64_t parts = vectorPartCount(operation.getResult());
  if (accumulator->parts.size() != static_cast<size_t>(parts))
    return fail(operation,
                "RVV widened partial accumulator part count is incomplete");
  for (int64_t part = 0; part < parts; ++part) {
    auto lhsPart = projectRegisterPart(operation.getLhs(), operation.getResult(), part);
    auto rhsPart = projectRegisterPart(operation.getRhs(), operation.getResult(), part);
    if (!lhsPart || !rhsPart || *lhsPart >= lhs->parts.size() ||
        *rhsPart >= rhs->parts.size())
      return fail(operation,
                  "RVV widened partial operands do not project to accumulator replicas");
    const std::string &signedOperand =
        lhsElement.isSignedInteger() ? lhs->parts[*lhsPart] : rhs->parts[*rhsPart];
    const std::string &unsignedOperand =
        lhsElement.isSignedInteger() ? rhs->parts[*rhsPart] : lhs->parts[*lhsPart];
    const std::string operands =
        mixed ? signedOperand + ", " + unsignedOperand
              : lhs->parts[*lhsPart] + ", " + rhs->parts[*rhsPart];
    std::string name = fresh("widen_partial");
    line(type + " " + name + " = __riscv_" +
         std::string(mixed ? "vwmaccsu" : "vwmacc") + "_vv_" + suffix + "(" +
         accumulator->parts[part] + ", " + operands + ", " +
         partVL(operation.getLhs(), *lhsPart) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVFinalizeWidenDot(
    riscv::RVVFinalizeWidenDotOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.vwredsum.partial" &&
      instruction != "rvv.vredsum.partial")
    return fail(operation,
                "RVV final widened dot has no exact selected leaf");
  auto partial = materializeNumeric(operation.getPartial(),
                                    bindings.lookup(operation.getPartial()));
  if (mlir::failed(partial) || partial->kind != Binding::Kind::Vector ||
      partial->parts.empty())
    return fail(operation,
                "RVV final widened dot requires a materialized partial tuple");
  auto resultValue = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  const int64_t resultParts =
      resultValue ? registerPartCount(operation.getResult()) : 1;
  if (partial->parts.size() != static_cast<size_t>(resultParts))
    return fail(operation,
                "RVV final widened dot cannot project partials to result replicas");
  std::string seed = fresh("widen_partial_seed");
  line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
  Binding result;
  result.kind = resultValue ? Binding::Kind::ScalarTuple : Binding::Kind::Scalar;
  const std::string partialSuffix = vectorSuffix(operation.getPartial());
  auto partialElement = mlir::dyn_cast<mlir::IntegerType>(
      operation.getPartial().getType().getElementType());
  if (!partialElement ||
      (partialElement.getWidth() != 16 && partialElement.getWidth() != 32))
    return fail(operation,
                "RVV final widened dot has no supported partial element width");
  for (int64_t part = 0; part < resultParts; ++part) {
    std::string reduced = fresh("widen_partial_sum");
    line("vint32m1_t " + reduced + " = __riscv_" +
         std::string(partialElement.getWidth() == 16 ? "vwredsum" : "vredsum") +
         "_vs_" +
         partialSuffix + "_i32m1(" + partial->parts[part] + ", " + seed + ", " +
         partVL(operation.getPartial(), part) + ");");
    std::string scalarName = fresh("widen_partial_scalar");
    line("int32_t " + scalarName + " = __riscv_vmv_x_s_i32m1_i32(" +
         reduced + ");");
    if (resultValue)
      result.parts.push_back(std::move(scalarName));
    else
      result.scalar = std::move(scalarName);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialSet(riscv::RVVPartialSetOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-set")
    return fail(operation, "RVV partial set has no exact selected leaf");
  auto setType = operation.getResult().getType();
  auto partialType = setType.getPartialType();
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  const int64_t sliceLMUL = partialType.getLayout().getLmulEighths() / 2;
  const int64_t sliceLanes =
      riscv::rvvLaneCount(partialType).value_or(int64_t{0});
  if (!kernel || sliceLMUL <= 0 || sliceLanes <= 0 ||
      operation.getLhsLaneOffsets().size() !=
          static_cast<size_t>(setType.getSlots()) ||
      operation.getRhsLaneOffsets().size() !=
          static_cast<size_t>(setType.getSlots()))
    return fail(operation,
                "RVV partial set is missing its typed lane-slice plan");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  const llvm::StringRef instruction = operation.getMultiplyInstruction();
  for (int64_t slot = 0; slot < setType.getSlots(); ++slot) {
    const int64_t lhsOperand = operation.getLhsOperands()[slot];
    const int64_t rhsOperand = operation.getRhsOperands()[slot];
    if (lhsOperand < 0 ||
        lhsOperand >= static_cast<int64_t>(operation.getLhs().size()) ||
        rhsOperand < 0 ||
        rhsOperand >= static_cast<int64_t>(operation.getRhs().size()))
      return fail(operation,
                  "RVV partial set slot references an absent operand");
    mlir::Value lhsValue = operation.getLhs()[lhsOperand];
    mlir::Value rhsValue = operation.getRhs()[rhsOperand];
    auto lhs = materializeNumeric(lhsValue, bindings.lookup(lhsValue));
    auto rhs = materializeNumeric(rhsValue, bindings.lookup(rhsValue));
    if (mlir::failed(lhs) || mlir::failed(rhs) ||
        lhs->kind != Binding::Kind::Vector ||
        rhs->kind != Binding::Kind::Vector)
      return fail(operation,
                  "RVV partial set requires materialized vector operands");
    auto lhsElement =
        mlir::cast<riscv::ValueType>(lhsValue.getType()).getElementType();
    auto rhsElement =
        mlir::cast<riscv::ValueType>(rhsValue.getType()).getElementType();
    const int64_t lhsPart = operation.getLhsParts()[slot];
    const int64_t rhsPart = operation.getRhsParts()[slot];
    if (lhsPart < 0 || rhsPart < 0 ||
        lhsPart >= static_cast<int64_t>(lhs->parts.size()) ||
        rhsPart >= static_cast<int64_t>(rhs->parts.size()) ||
        lhs->parts[lhsPart].empty() || rhs->parts[rhsPart].empty())
      return fail(operation,
                  "RVV partial set slot has no selected physical operand part");
    struct OperandSlice {
      std::string value;
      std::string suffix;
    };
    auto sliceOperand = [&](mlir::Value value, llvm::StringRef source,
                            int64_t laneOffset)
        -> std::optional<OperandSlice> {
      auto valueType = mlir::cast<riscv::ValueType>(value.getType());
      const int64_t sourceLMUL = valueType.getLayout().getLmulEighths();
      const int64_t inputSEW = valueType.getLayout().getSew();
      if (sourceLMUL < sliceLMUL || sourceLMUL % sliceLMUL || inputSEW <= 0 ||
          laneOffset < 0 || laneOffset % sliceLanes)
        return std::nullopt;
      const std::string sourceSuffix = vectorSuffix(value);
      if (sourceSuffix.empty())
        return std::nullopt;
      const char category = sourceSuffix.front();
      const std::string sliceSuffix =
          std::string(1, category) + std::to_string(inputSEW) +
          lmulSpelling(sliceLMUL);
      const std::string sliceType =
          std::string(category == 'u' ? "vuint" : "vint") +
          sliceSuffix.substr(1) + "_t";
      const int64_t sliceSpan =
          kernel.getTarget().getVlenBits() * sliceLMUL / (8 * inputSEW);
      if (sliceSpan <= 0)
        return std::nullopt;
      const int64_t group = laneOffset / sliceSpan;
      const int64_t intra = laneOffset % sliceSpan;
      std::string sliced = source.str();
      if (sourceLMUL != sliceLMUL) {
        sliced = fresh("partial_operand_slice");
        line(sliceType + " " + sliced + " = __riscv_vget_v_" +
             sourceSuffix + "_" + sliceSuffix + "(" + source.str() + ", " +
             std::to_string(group) + ");");
      } else if (group != 0) {
        return std::nullopt;
      }
      if (intra != 0) {
        std::string shifted = fresh("partial_operand_slide");
        line(sliceType + " " + shifted + " = __riscv_vslidedown_vx_" +
             sliceSuffix + "(" + sliced + ", " + std::to_string(intra) +
             ", " + std::to_string(sliceLanes) + ");");
        sliced = std::move(shifted);
      }
      return OperandSlice{std::move(sliced), sliceSuffix};
    };
    auto lhsSlice = sliceOperand(lhsValue, lhs->parts[lhsPart],
                                 operation.getLhsLaneOffsets()[slot]);
    auto rhsSlice = sliceOperand(rhsValue, rhs->parts[rhsPart],
                                 operation.getRhsLaneOffsets()[slot]);
    if (!lhsSlice || !rhsSlice)
      return fail(operation,
                  "RVV partial set cannot materialize its selected lane slice");
    std::string left = lhsSlice->value;
    std::string right = rhsSlice->value;
    std::string mnemonic;
    if (instruction == "rvv.vwmul.vv") {
      mnemonic = "vwmul";
    } else if (instruction == "rvv.vwmul.vv.reinterpret-rhs" ||
               instruction == "rvv.vwmul.vv.reinterpret-lhs") {
      mnemonic = "vwmul";
      std::string sourceSuffix =
          instruction == "rvv.vwmul.vv.reinterpret-rhs" ? rhsSlice->suffix
                                                          : lhsSlice->suffix;
      std::string targetSuffix = sourceSuffix;
      if (sourceSuffix.empty() || sourceSuffix.front() != 'u')
        return fail(operation,
                    "narrow unsigned partial operand has no unsigned RVV spelling");
      targetSuffix.front() = 'i';
      std::string &operand =
          instruction == "rvv.vwmul.vv.reinterpret-rhs" ? right : left;
      operand = "__riscv_vreinterpret_v_" + sourceSuffix + "_" +
                targetSuffix + "(" + operand + ")";
    } else if (instruction == "rvv.vwmulsu.vv") {
      mnemonic = "vwmulsu";
    } else if (instruction == "rvv.vwmulsu.vv.swap") {
      mnemonic = "vwmulsu";
      std::swap(left, right);
    } else {
      return fail(operation,
                  "RVV partial set has no exact widening multiply instruction");
    }
    std::string product = fresh("partial_slot");
    line(vectorTypeFor(partialType) + " " + product + " = __riscv_" +
         mnemonic + "_vv_" + vectorSuffixFor(partialType) + "(" + left +
         ", " + right + ", " + std::to_string(sliceLanes) + ");");
    result.parts.push_back(std::move(product));
  }
  if (result.parts.size() != static_cast<size_t>(setType.getSlots()))
    return fail(operation,
                "RVV partial set did not materialize every declared slot");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialCapture(riscv::RVVPartialCaptureOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-capture")
    return fail(operation, "RVV partial capture has no exact selected leaf");
  auto input = materializeNumeric(operation.getInput(),
                                  bindings.lookup(operation.getInput()));
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
      input->parts.size() != 1 || operation.getResult().getType().getSlots() != 1)
    return fail(operation,
                "RVV partial capture requires one materialized vector partial");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  result.parts.push_back(input->parts.front());
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialCollect(riscv::RVVPartialCollectOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-collect")
    return fail(operation, "RVV partial collect has no exact selected leaf");
  auto resultType = operation.getResult().getType();
  if (operation.getInputs().size() !=
      static_cast<size_t>(resultType.getSlots()))
    return fail(operation,
                "RVV partial collect has no value for every selected slot");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  for (mlir::Value inputValue : operation.getInputs()) {
    auto input = materializeNumeric(inputValue, bindings.lookup(inputValue));
    if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
        input->parts.size() != 1 || input->parts.front().empty())
      return fail(operation,
                  "RVV partial collect requires one materialized vector per slot");
    result.parts.push_back(input->parts.front());
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialRepack(riscv::RVVPartialRepackOp operation) {
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.partial-repack.split" &&
      instruction != "rvv.partial-repack.slice")
    return fail(operation, "RVV partial repack has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  const int64_t split = operation.getSplitFactor();
  if (input.kind != Binding::Kind::PartialSet || split <= 1 ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      resultType.getSlots() != inputType.getSlots() * split)
    return fail(operation,
                "RVV partial repack has no complete typed split input");
  const std::string sourceSuffix =
      vectorSuffixFor(inputType.getPartialType());
  const std::string targetSuffix =
      vectorSuffixFor(resultType.getPartialType());
  const std::string targetType = vectorTypeFor(resultType.getPartialType());
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  if (instruction == "rvv.partial-repack.slice") {
    auto parameters = operation.getLeaf().getParameters().asArrayRef();
    if (parameters.size() != 3)
      return fail(operation, "partial repack slice lacks its selected carrier");
    const int64_t carrierLMUL = parameters[2];
    const int64_t sourceLMUL =
        inputType.getPartialType().getLayout().getLmulEighths();
    const int64_t targetLMUL =
        resultType.getPartialType().getLayout().getLmulEighths();
    const int64_t sew = resultType.getPartialType().getLayout().getSew();
    const int64_t window = resultType.getPartialType().getLayout().getVl();
    const int64_t capacity = kernel.getTarget().getVlenBits() * carrierLMUL /
                             (8 * sew);
    const std::string carrierSuffix = sourceSuffix.substr(0, 1) +
        std::to_string(sew) + lmulSpelling(carrierLMUL);
    const std::string carrierType =
        std::string(sourceSuffix.front() == 'u' ? "vuint" : "vint") +
        carrierSuffix.substr(1) + "_t";
    for (llvm::StringRef part : input.parts) {
      llvm::DenseMap<int64_t, std::string> groups;
      for (int64_t chunk = 0; chunk < split; ++chunk) {
        const int64_t group = chunk * window / capacity;
        const int64_t offset = chunk * window % capacity;
        auto [entry, inserted] = groups.try_emplace(group, part.str());
        if (inserted && sourceLMUL != carrierLMUL) {
          entry->second = fresh("partial_group");
          line(carrierType + " " + entry->second + " = __riscv_vget_v_" +
               sourceSuffix + "_" + carrierSuffix + "(" + part.str() + ", " +
               std::to_string(group) + ");");
        }
        std::string value = entry->second;
        if (offset != 0) {
          std::string shifted = fresh("partial_window");
          line(carrierType + " " + shifted + " = __riscv_vslidedown_vx_" +
               carrierSuffix + "(" + value + ", " + std::to_string(offset) +
               ", " + std::to_string(window) + ");");
          value = std::move(shifted);
        }
        if (carrierLMUL != targetLMUL) {
          std::string narrowed = fresh("partial_slice");
          line(targetType + " " + narrowed + " = __riscv_vlmul_trunc_v_" +
               carrierSuffix + "_" + targetSuffix + "(" + value + ");");
          value = std::move(narrowed);
        }
        result.parts.push_back(std::move(value));
      }
    }
  } else for (llvm::StringRef part : input.parts)
    for (int64_t chunk = 0; chunk < split; ++chunk) {
      std::string splitPart = fresh("partial_split");
      line(targetType + " " + splitPart + " = __riscv_vget_v_" +
           sourceSuffix + "_" + targetSuffix + "(" + part.str() + ", " +
           std::to_string(chunk) + ");");
      result.parts.push_back(std::move(splitPart));
    }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialMerge(riscv::RVVPartialMergeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-merge" ||
      operation.getTopology() != "pairwise")
    return fail(operation, "RVV partial merge has no exact selected topology");
  llvm::SmallVector<std::string> level;
  for (mlir::Value inputValue : operation.getInputs()) {
    Binding input = bindings.lookup(inputValue);
    auto inputType = mlir::cast<riscv::PartialSetType>(inputValue.getType());
    if (input.kind != Binding::Kind::PartialSet ||
        input.parts.size() != static_cast<size_t>(inputType.getSlots()))
      return fail(operation,
                  "RVV partial merge input has no complete typed slot set");
    level.append(input.parts.begin(), input.parts.end());
  }
  if (level.size() < 2 || operation.getResult().getType().getSlots() != 1)
    return fail(operation,
                "RVV partial merge requires multiple slots and one result");
  const std::string type =
      vectorTypeFor(operation.getResult().getType().getPartialType());
  const std::string suffix =
      vectorSuffixFor(operation.getResult().getType().getPartialType());
  const std::string vl = std::to_string(
      physicalLanesFor(operation.getResult().getType().getPartialType()));
  while (level.size() > 1) {
    llvm::SmallVector<std::string> next;
    for (size_t part = 0; part < level.size(); part += 2) {
      if (part + 1 == level.size()) {
        next.push_back(std::move(level[part]));
        continue;
      }
      std::string combined = fresh("partial_merge");
      line(type + " " + combined + " = __riscv_vadd_vv_" + suffix + "(" +
           level[part] + ", " + level[part + 1] + ", " + vl + ");");
      next.push_back(std::move(combined));
    }
    level = std::move(next);
  }
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  result.parts.push_back(std::move(level.front()));
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialReduce(riscv::RVVPartialReduceOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.partial-reduce.widen" &&
      instruction != "rvv.partial-reduce")
    return fail(operation, "RVV partial reduction has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  auto inputPartial = inputType.getPartialType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      resultType.getSlots() != inputType.getSlots())
    return fail(operation,
                "RVV partial reduction has no complete typed input set");
  std::string seed = fresh("partial_reduce_seed");
  line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
  const std::string reduction = instruction == "rvv.partial-reduce.widen"
                                    ? "vwredsum"
                                    : "vredsum";
  const std::string suffix = vectorSuffixFor(inputPartial);
  const std::string vl = std::to_string(physicalLanesFor(inputPartial));
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  for (llvm::StringRef partial : input.parts) {
    std::string reduced = fresh("partial_reduced");
    line("vint32m1_t " + reduced + " = __riscv_" + reduction + "_vs_" +
         suffix + "_i32m1(" + partial.str() + ", " + seed + ", " + vl +
         ");");
    result.parts.push_back(std::move(reduced));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartialScaleCombine(
    riscv::RVVPartialScaleCombineOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.partial-scale-combine")
    return fail(operation,
                "RVV scaled partial combine has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      operation.getScales().size() != input.parts.size() ||
      operation.getScaleReplicas().size() != input.parts.size() ||
      operation.getSlotOrder().size() != input.parts.size() ||
      resultType.getSlots() <= 0 || inputType.getSlots() % resultType.getSlots())
    return fail(operation,
                "RVV scaled partial combine has no complete typed slot set");
  llvm::SmallVector<std::string> scales;
  for (auto [scaleValue, replica] :
       llvm::zip(operation.getScales(), operation.getScaleReplicas())) {
    auto scale =
        materializeNumeric(scaleValue, bindings.lookup(scaleValue));
    if (mlir::failed(scale))
      return mlir::failure();
    if (scale->kind == Binding::Kind::Scalar && replica == 0) {
      scales.push_back(scale->scalar);
      continue;
    }
    if (scale->kind == Binding::Kind::ScalarTuple && replica >= 0 &&
        static_cast<size_t>(replica) < scale->parts.size()) {
      scales.push_back(scale->parts[replica]);
      continue;
    }
    return fail(operation,
                "RVV scaled partial combine scale replica has no scalar binding");
  }
  const int64_t fanout = inputType.getSlots() / resultType.getSlots();
  const std::string suffix = vectorSuffixFor(inputType.getPartialType());
  const std::string type = vectorTypeFor(inputType.getPartialType());
  const std::string vl =
      std::to_string(physicalLanesFor(inputType.getPartialType()));
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  llvm::SmallVector<std::string> combined;
  combined.reserve(resultType.getSlots());
  for (int64_t resultSlot = 0; resultSlot < resultType.getSlots(); ++resultSlot) {
    const int64_t first = operation.getSlotOrder()[resultSlot * fanout];
    std::string seed = fresh("scaled_partial");
    line(type + " " + seed + " = __riscv_vmul_vx_" + suffix + "(" +
         input.parts[first] + ", " + scales[first] + ", " + vl + ");");
    combined.push_back(std::move(seed));
  }
  // Independent chains are already fixed by the typed result slots.  Spell
  // one dependency level from every chain before advancing to the next level
  // so the C compiler can preserve their instruction-level parallelism.
  for (int64_t term = 1; term < fanout; ++term) {
    for (int64_t resultSlot = 0; resultSlot < resultType.getSlots();
         ++resultSlot) {
      const int64_t slot =
          operation.getSlotOrder()[resultSlot * fanout + term];
      std::string next = fresh("scaled_partial_chain");
      line(type + " " + next + " = __riscv_vmacc_vx_" + suffix + "(" +
           combined[resultSlot] + ", " + scales[slot] + ", " +
           input.parts[slot] + ", " + vl + ");");
      combined[resultSlot] = std::move(next);
    }
  }
  result.parts = std::move(combined);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartialPackedScale(
    riscv::RVVPartialPackedScaleOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-packed-scale")
    return fail(operation, "packed partial scale has no exact selected leaf");
  llvm::SmallVector<std::string> partials;
  for (mlir::Value value : operation.getInputs()) {
    Binding input = bindings.lookup(value);
    const int64_t slots = mlir::cast<riscv::PartialSetType>(value.getType()).getSlots();
    if (input.kind != Binding::Kind::PartialSet ||
        input.parts.size() != static_cast<size_t>(slots))
      return fail(operation, "packed partial scale has incomplete slot bindings");
    llvm::append_range(partials, input.parts);
  }
  std::string scales;
  int64_t scaleLanes = 0;
  for (mlir::Value value : operation.getScales()) {
    Binding scale = bindings.lookup(value);
    auto type = mlir::cast<riscv::ValueType>(value.getType());
    if (scale.kind != Binding::Kind::Vector || scale.parts.size() != 1)
      return fail(operation, "packed partial scale has incomplete scale bindings");
    if (!scaleLanes) {
      scales = scale.parts[0];
    } else {
      std::string next = fresh("partial_scale_pack");
      line("vint32m1_t " + next + " = __riscv_vslideup_vx_i32m1_tu(" +
           scales + ", " + scale.parts[0] + ", " + std::to_string(scaleLanes) +
           ", " + std::to_string(scaleLanes + type.getShape()[0]) + ");");
      scales = std::move(next);
    }
    scaleLanes += type.getShape()[0];
  }
  const int64_t slots = partials.size();
  if (slots < 2 || scaleLanes != slots)
    return fail(operation, "packed partial scale slot and scale counts disagree");
  std::string packed = partials[operation.getSlotOrder()[0]];
  // Only the constructed prefix is observed by a later slide. Every other
  // lane is overwritten before multiply/reduce, so a single final VL is valid
  // for the whole pack even though each input defines only lane zero.
  for (int64_t lane = 1; lane < slots; ++lane) {
    std::string next = fresh("partial_lane_pack");
    line("vint32m1_t " + next + " = __riscv_vslideup_vx_i32m1_tu(" +
         packed + ", " + partials[operation.getSlotOrder()[lane]] + ", " +
         std::to_string(lane) + ", " + std::to_string(slots) + ");");
    packed = std::move(next);
  }
  std::string product = fresh("partial_lane_scale");
  line("vint32m1_t " + product + " = __riscv_vmul_vv_i32m1(" + packed +
       ", " + scales + ", " + std::to_string(slots) + ");");
  std::string seed = fresh("partial_lane_seed");
  line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
  std::string reduced = fresh("partial_lane_sum");
  line("vint32m1_t " + reduced + " = __riscv_vredsum_vs_i32m1_i32m1(" +
       product + ", " + seed + ", " + std::to_string(slots) + ");");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  result.parts.push_back(std::move(reduced));
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartialWidenScale(
    riscv::RVVPartialWidenScaleOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.partial-widen-scale")
    return fail(operation,
                "RVV partial widen-scale has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      operation.getCombineArity() <= 0 ||
      resultType.getSlots() * operation.getCombineArity() !=
          inputType.getSlots() ||
      operation.getScales().size() != input.parts.size() ||
      operation.getScaleReplicas().size() != input.parts.size())
    return fail(operation,
                "RVV partial widen-scale has no complete typed slot set");
  llvm::SmallVector<std::string> scales;
  for (auto [scaleValue, replica] :
       llvm::zip(operation.getScales(), operation.getScaleReplicas())) {
    auto scale =
        materializeNumeric(scaleValue, bindings.lookup(scaleValue));
    if (mlir::failed(scale))
      return mlir::failure();
    if (scale->kind == Binding::Kind::Scalar && replica == 0) {
      scales.push_back(scale->scalar);
      continue;
    }
    if (scale->kind == Binding::Kind::ScalarTuple && replica >= 0 &&
        static_cast<size_t>(replica) < scale->parts.size()) {
      scales.push_back(scale->parts[replica]);
      continue;
    }
    return fail(
        operation,
        "RVV partial widen-scale scale replica has no scalar binding");
  }
  const std::string type = vectorTypeFor(resultType.getPartialType());
  const std::string suffix = vectorSuffixFor(resultType.getPartialType());
  const std::string vl =
      std::to_string(physicalLanesFor(inputType.getPartialType()));
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  for (int64_t resultSlot = 0; resultSlot < resultType.getSlots();
       ++resultSlot) {
    const size_t first =
        static_cast<size_t>(resultSlot * operation.getCombineArity());
    std::string accumulated = fresh("partial_widen_scale");
    line(type + " " + accumulated + " = __riscv_vwmul_vx_" + suffix + "(" +
         input.parts[first] + ", " + scales[first] + ", " + vl + ");");
    for (int64_t term = 1; term < operation.getCombineArity(); ++term) {
      const size_t slot = first + static_cast<size_t>(term);
      std::string next = fresh("partial_widen_scale_chain");
      line(type + " " + next + " = __riscv_vwmacc_vx_" + suffix + "(" +
           accumulated + ", " + scales[slot] + ", " + input.parts[slot] +
           ", " + vl + ");");
      accumulated = std::move(next);
    }
    result.parts.push_back(std::move(accumulated));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialCombine(riscv::RVVPartialCombineOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.partial-combine" &&
      instruction != "rvv.partial-combine.widen")
    return fail(operation, "RVV partial combine has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  if (input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(inputType.getSlots()) ||
      resultType.getSlots() * operation.getArity() != inputType.getSlots() ||
      operation.getTopology() != "pairwise")
    return fail(operation,
                "RVV partial combine has no complete input slot set");
  Binding result;
  result.kind = Binding::Kind::PartialSet;
  const std::string type = vectorTypeFor(resultType.getPartialType());
  const std::string suffix = vectorSuffixFor(resultType.getPartialType());
  const std::string vl = std::to_string(
      physicalLanesFor(resultType.getPartialType()));
  if (instruction == "rvv.partial-combine.widen" &&
      operation.getArity() != 2)
    return fail(operation,
                "RVV widening partial combine requires one exact pair");
  for (int64_t slot = 0; slot < resultType.getSlots(); ++slot) {
    if (instruction == "rvv.partial-combine.widen") {
      const size_t first = static_cast<size_t>(slot * 2);
      std::string next = fresh("partial_widen_tree");
      line(type + " " + next + " = __riscv_vwadd_vv_" + suffix + "(" +
           input.parts[first] + ", " + input.parts[first + 1] + ", " + vl +
           ");");
      result.parts.push_back(std::move(next));
      continue;
    }
    llvm::SmallVector<std::string> level;
    for (int64_t term = 0; term < operation.getArity(); ++term)
      level.push_back(input.parts[slot * operation.getArity() + term]);
    while (level.size() > 1) {
      llvm::SmallVector<std::string> nextLevel;
      for (size_t term = 0; term < level.size(); term += 2) {
        if (term + 1 == level.size()) {
          nextLevel.push_back(std::move(level[term]));
          continue;
        }
        std::string next = fresh("partial_tree");
        line(type + " " + next + " = __riscv_vadd_vv_" + suffix + "(" +
             level[term] + ", " + level[term + 1] + ", " + vl + ");");
        nextLevel.push_back(std::move(next));
      }
      level = std::move(nextLevel);
    }
    result.parts.push_back(std::move(level.front()));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVPartialFinalize(riscv::RVVPartialFinalizeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.partial-finalize.reduce" &&
      instruction != "rvv.partial-finalize.extract" &&
      instruction != "rvv.partial-finalize.widen")
    return fail(operation, "RVV partial finalize has no exact selected leaf");
  auto setType = operation.getInput().getType();
  auto partialType = setType.getPartialType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partialType.getElementType());
  Binding input = bindings.lookup(operation.getInput());
  if (!partialElement || input.kind != Binding::Kind::PartialSet ||
      input.parts.size() != static_cast<size_t>(setType.getSlots()) ||
      input.parts.empty())
    return fail(operation,
                "RVV partial finalize has no complete typed partial set");
  if (instruction == "rvv.partial-finalize.widen") {
    auto resultType = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    if (!resultType || input.parts.size() != 1 ||
        vectorPartCount(operation.getResult()) != 1)
      return fail(operation,
                  "RVV vector partial finalize requires one typed input and one output part");
    Binding result;
    result.kind = Binding::Kind::Vector;
    std::string widened = fresh("partial_vector");
    line(vectorType(operation.getResult()) + " " + widened +
         " = __riscv_vwcvt_x_x_v_" + vectorSuffix(operation.getResult()) +
         "(" + input.parts.front() + ", " +
         partVL(operation.getResult(), 0) + ");");
    result.parts.push_back(std::move(widened));
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  llvm::SmallVector<std::string> scalars;
  if (instruction == "rvv.partial-finalize.extract") {
    const std::string suffix = vectorSuffixFor(partialType);
    const std::string scalarSuffix =
        partialElement.getWidth() == 16 ? "i16" : "i32";
    for (llvm::StringRef part : input.parts) {
      std::string extracted = fresh("partial_scalar_part");
      line("int32_t " + extracted + " = (int32_t)__riscv_vmv_x_s_" +
           suffix + "_" + scalarSuffix + "(" + part.str() + ");");
      scalars.push_back(std::move(extracted));
    }
  } else {
    std::string seed = fresh("partial_seed");
    line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
    const std::string reduction =
        partialElement.getWidth() == 16 ? "vwredsum" : "vredsum";
    const std::string suffix = vectorSuffixFor(partialType);
    const std::string vl = std::to_string(physicalLanesFor(partialType));
    for (llvm::StringRef part : input.parts) {
      std::string reduced = fresh("partial_reduction");
      line("vint32m1_t " + reduced + " = __riscv_" + reduction + "_vs_" +
           suffix + "_i32m1(" + part.str() + ", " + seed + ", " + vl +
           ");");
      seed = std::move(reduced);
    }
    std::string extracted = fresh("partial_scalar_part");
    line("int32_t " + extracted + " = __riscv_vmv_x_s_i32m1_i32(" + seed +
         ");");
    scalars.push_back(std::move(extracted));
  }
  std::string scalarName = fresh("partial_scalar");
  std::string expression = scalars.front();
  for (size_t index = 1; index < scalars.size(); ++index)
    expression = "(" + expression + " + " + scalars[index] + ")";
  line("int32_t " + scalarName + " = " + expression + ";");
  bindings[operation.getResult()] = scalar(std::move(scalarName));
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVAssembleReplicas(
    riscv::RVVAssembleReplicasOp operation) {
  if (instructionOf(operation.getOperation()) != "scalar.assemble-replicas")
    return fail(operation,
                "scalar replica assembly has no exact selected leaf");
  Binding result;
  const bool singleton = operation.getValues().size() == 1;
  result.kind = singleton ? Binding::Kind::Scalar
                          : Binding::Kind::ScalarTuple;
  for (mlir::Value value : operation.getValues()) {
    Binding part = bindings.lookup(value);
    if (part.kind != Binding::Kind::Scalar)
      return fail(operation,
                  "scalar replica assembly operand has no scalar binding");
    if (singleton)
      result.scalar = part.scalar;
    else
      result.parts.push_back(part.scalar);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVWidenReduce(riscv::RVVWidenReduceOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.vwredsum")
    return fail(operation, "RVV widening reduction has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  if (input->kind != Binding::Kind::Vector || input->parts.size() != 1)
    return fail(operation,
                "RVV widening reduction requires one materialized vector part");
  auto inputElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getInput().getType()));
  auto resultElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getResult().getType()));
  const bool isUnsigned = inputElement.isUnsigned();
  const std::string inputSuffix = vectorSuffix(operation.getInput());
  const std::string resultStem =
      std::string(isUnsigned ? "u" : "i") +
      std::to_string(resultElement.getWidth());
  const std::string resultSuffix = resultStem + "m1";
  const std::string resultVectorType =
      std::string(isUnsigned ? "vuint" : "vint") +
      std::to_string(resultElement.getWidth()) + "m1_t";
  auto scalarType = scalarCType(resultElement);
  if (!scalarType)
    return fail(operation, "RVV widening reduction result has no C scalar type");
  const std::string vl = partVL(operation.getInput(), 0);
  std::string seed = fresh("widen_reduce_seed");
  line(resultVectorType + " " + seed + " = __riscv_vmv_v_x_" +
       resultSuffix + "(0, 1);");
  std::string reduced = fresh("widen_reduce");
  line(resultVectorType + " " + reduced + " = __riscv_" +
       std::string(isUnsigned ? "vwredsumu" : "vwredsum") + "_vs_" +
       inputSuffix + "_" + resultSuffix + "(" + input->parts.front() + ", " +
       seed + ", " + vl + ");");
  std::string scalarName = fresh("widen_reduce_scalar");
  line(*scalarType + " " + scalarName + " = __riscv_vmv_x_s_" +
       resultSuffix + "_" + resultStem + "(" + reduced + ");");
  bindings[operation.getResult()] = scalar(scalarName);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVPartitionedWidenReduceStore(
    riscv::RVVPartitionedWidenReduceStoreOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.partitioned-vwredsum-store")
    return fail(operation,
                "partitioned widening reduction store has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  if (input->kind != Binding::Kind::Vector)
    return fail(operation,
                "partitioned widening reduction store requires an RVV input");
  Binding destination = bindings.lookup(operation.getDestination());
  if (destination.kind != Binding::Kind::Field)
    return fail(operation,
                "partitioned widening reduction store requires one encoded field destination");
  auto field = fieldFor(destination);
  Binding record = bindings.lookup(destination.field.owner);
  if (record.kind == Binding::Kind::Slice) {
    auto materialized = recordForSlice(destination.field.owner);
    if (mlir::failed(materialized))
      return mlir::failure();
    record = std::move(*materialized);
  }
  auto inputElement = mlir::cast<mlir::IntegerType>(
      riscv_internal::logicalElement(operation.getInput().getType()));
  auto fieldInteger = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                            : mlir::IntegerType();
  if (!field || record.kind != Binding::Kind::Record ||
      field->access.getMapping() != "natural" || field->bitOffset % 8 ||
      !fieldInteger || !fieldInteger.isSigned() ||
      fieldInteger.getWidth() != 16 ||
      !inputElement.isSigned() || inputElement.getWidth() != 8 ||
      operation.getPartition() != 16)
    return fail(operation,
                "partitioned widening reduction store has no closed i8-to-i16 field realization");
  Binding baseIndex = bindings.lookup(operation.getBaseIndex());
  if (baseIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "partitioned widening reduction store base index is not scalar");
  const int64_t count = operation.getCount();
  const int64_t inputParts = static_cast<int64_t>(input->parts.size());
  if (inputParts <= 0 || count % inputParts)
    return fail(operation,
                "partitioned widening reduction store cannot project its input parts");
  const int64_t partitionsPerInput = count / inputParts;
  const std::string inputSuffix = vectorSuffix(operation.getInput());
  const std::string inputType = vectorType(operation.getInput());
  for (int64_t part = 0; part < count; ++part) {
    const int64_t sourcePart = part / partitionsPerInput;
    const int64_t withinPart = part % partitionsPerInput;
    std::string chunk = input->parts[sourcePart];
    std::string reductionSuffix = inputSuffix;
    if (withinPart != 0) {
      std::string projected = fresh("partition_chunk");
      line(inputType + " " + projected + " = __riscv_vslidedown_vx_" +
           inputSuffix + "(" + chunk + ", " +
           std::to_string(withinPart * operation.getPartition()) + ", " +
           std::to_string(operation.getInput().getType().getLayout().getVl()) +
           ");");
      chunk = std::move(projected);
    }
    std::string seed = fresh("partition_seed");
    line("vint16m1_t " + seed + " = __riscv_vmv_v_x_i16m1(0, 1);");
    std::string reduced = fresh("partition_sum");
    line("vint16m1_t " + reduced + " = __riscv_vwredsum_vs_" +
         reductionSuffix +
         "_i16m1(" + chunk + ", " + seed + ", " +
         std::to_string(operation.getPartition()) + ");");
    std::string scalarName = fresh("partition_scalar");
    line("int16_t " + scalarName + " = __riscv_vmv_x_s_i16m1_i16(" +
         reduced + ");");
    const std::string address =
        record.recordPointer + " + " + std::to_string(field->bitOffset / 8) +
        " + (" + baseIndex.scalar + " + " + std::to_string(part) + ") * 2";
    line(field->access.getAlignment() >= 2
             ? "*(int16_t *)(" + address + ") = " + scalarName + ";"
             : "weft_store_i16_le(" + address + ", " + scalarName + ");");
  }
  return mlir::success();
}

} // namespace weft::riscv_emission
