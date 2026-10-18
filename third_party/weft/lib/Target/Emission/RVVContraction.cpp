#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Stream.h"
#include "Weft/Dialect/RISCV/IR/Contraction.h"

namespace weft::riscv_emission {

mlir::LogicalResult
Emitter::compileGroupedMacLoad(riscv::RVVGroupedMacLoadOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.grouped-mac-load.u8-s8")
    return fail(operation, "grouped MAC load has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding groupIndex = bindings.lookup(operation.getGroupIndex());
  Binding activeTerms = bindings.lookup(operation.getActiveTerms());
  if (lhs.kind != Binding::Kind::Field || rhs.kind != Binding::Kind::Field ||
      groupIndex.kind != Binding::Kind::Scalar ||
      activeTerms.kind != Binding::Kind::Scalar)
    return fail(operation,
                "grouped MAC load requires two fields and explicit scalar window bounds");
  auto lhsField = fieldFor(lhs);
  auto rhsField = fieldFor(rhs);
  Binding lhsOwner = bindings.lookup(lhs.field.owner);
  Binding rhsOwner = bindings.lookup(rhs.field.owner);
  if (lhsOwner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(lhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    lhsOwner = std::move(*record);
  }
  if (rhsOwner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(rhs.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    rhsOwner = std::move(*record);
  }
  auto lhsInteger = lhsField ? mlir::dyn_cast<mlir::IntegerType>(lhsField->type)
                             : mlir::IntegerType();
  auto rhsInteger = rhsField ? mlir::dyn_cast<mlir::IntegerType>(rhsField->type)
                             : mlir::IntegerType();
  if (!lhsField || !rhsField || lhsOwner.kind != Binding::Kind::Record ||
      rhsOwner.kind != Binding::Kind::Record || !lhsInteger ||
      !lhsInteger.isUnsigned() || lhsInteger.getWidth() >= 8 || !rhsInteger ||
      !rhsInteger.isSigned() || rhsInteger.getWidth() != 8)
    return fail(operation,
                "grouped MAC load fields do not match the selected narrow/signed window");

  auto lhsType = mlir::dyn_cast<riscv::ValueType>(operation.getLhs().getType());
  auto rhsType = mlir::dyn_cast<riscv::ValueType>(operation.getRhs().getType());
  auto windowType = operation.getResult().getType();
  auto resultType = mlir::dyn_cast<riscv::ValueType>(windowType.getResultType());
  const int64_t outputParts = windowType.getResultParts();
  const int64_t outputStreams =
      product(resultType.getLayout().getTimeFactors());
  const int64_t outputRegisters =
      product(resultType.getLayout().getReplicaFactors());
  auto supplyForResult = operation.getPackedSupplyForResult();
  const int64_t packedSupplies =
      supplyForResult.empty()
          ? 0
          : *llvm::max_element(supplyForResult) + int64_t{1};
  if (!lhsType || !rhsType || !resultType ||
      lhsType.getLayout().getCarrier() != "rvv" ||
      outputParts <= 0 || outputStreams <= 0 || outputRegisters <= 0 ||
      outputStreams * outputRegisters != outputParts ||
      supplyForResult.size() != static_cast<size_t>(outputParts) ||
      packedSupplies <= 0)
    return fail(operation,
                "grouped MAC load has no closed result-replica or packed-supply mapping");
  const std::string rawSuffix =
      "u8" + lmulSpelling(operation.getLoadLayout().getLmulEighths());
  const std::string rawType =
      "vuint8" + lmulSpelling(operation.getLoadLayout().getLmulEighths()) +
      "_t";
  if (rawSuffix == "u8" || rawType == "vuint8_t")
    return fail(operation, "grouped MAC load has no legal raw RVV shape");

  Binding point = lhs.field.index ? bindings.lookup(*lhs.field.index) : Binding();
  Binding rhsPoint = rhs.field.index ? bindings.lookup(*rhs.field.index) : Binding();
  if (point.kind != Binding::Kind::Point || rhsPoint.kind != Binding::Kind::Point ||
      point.point.axis != rhsPoint.point.axis ||
      point.point.base != rhsPoint.point.base)
    return fail(operation,
                "grouped MAC load operands do not share one reduction point");
  riscv::GroupedMacPlanAttr plan = operation.getPlan();
  const std::string logicalBase = "((" + point.point.base + " % " +
                                  std::to_string(plan.getRecordElements()) +
                                  ") + " + groupIndex.scalar + " * " +
                                  std::to_string(plan.getGroup()) + ")";
  std::string rowStride;
  for (const auto &[axis, stride] : lhsOwner.recordByteStrides)
    if (axis == plan.getRowStrideAxis())
      rowStride = stride;
  if ((plan.getLoadForm() == "unit" && plan.getInterleaveRows() <= 0) ||
      (plan.getLoadForm() == "strided" && rowStride.empty()))
    return fail(operation, "grouped MAC load plan has no bound load form");

  llvm::SmallVector<int64_t, 4> resultAxes;
  llvm::SmallVector<int64_t, 4> resultExtents;
  for (auto [axis, factor] :
       llvm::zip(resultType.getAxisIds().asArrayRef(),
                 resultType.getLayout().getReplicaFactors().asArrayRef()))
    if (factor > 1) {
      resultAxes.push_back(axis);
      resultExtents.push_back(factor);
    }
  auto coordinatesForPart = [&](int64_t part)
      -> std::optional<llvm::SmallVector<int64_t, 4>> {
    part /= outputStreams;
    llvm::SmallVector<int64_t, 4> coordinates(resultExtents.size(), 0);
    for (int64_t index = static_cast<int64_t>(resultExtents.size()) - 1;
         index >= 0; --index) {
      const int64_t extent = resultExtents[static_cast<size_t>(index)];
      if (extent <= 0)
        return std::nullopt;
      coordinates[static_cast<size_t>(index)] = part % extent;
      part /= extent;
    }
    if (part != 0)
      return std::nullopt;
    return coordinates;
  };
  int64_t laneAxis = 0;
  for (auto [axis, factor] :
       llvm::zip(resultType.getAxisIds().asArrayRef(),
                 resultType.getLayout().getLaneFactors().asArrayRef()))
    if (factor > 1) {
      laneAxis = axis;
      break;
    }
  const int64_t physicalLanes =
      product(resultType.getLayout().getLaneFactors());
  auto partOffset = [&](int64_t part) -> std::optional<int64_t> {
    if (laneAxis <= 0 || physicalLanes <= 0)
      return std::nullopt;
    int64_t stream = part % outputStreams;
    int64_t laneCoordinate = 0;
    for (int64_t position =
             static_cast<int64_t>(resultType.getAxisIds().size()) - 1;
         position >= 0; --position) {
      const int64_t extent =
          resultType.getLayout().getTimeFactors()[position];
      if (extent <= 0)
        return std::nullopt;
      const int64_t coordinate = stream % extent;
      stream /= extent;
      if (resultType.getAxisIds()[position] == laneAxis)
        laneCoordinate = coordinate;
    }
    if (stream != 0)
      return std::nullopt;
    return laneCoordinate * physicalLanes;
  };
  auto resultPartVL = [&](int64_t part) {
    if (resultType.getLayout().getValidity() == "full")
      return std::to_string(physicalLanes);
    auto scope = axisScopes.find(laneAxis);
    auto offset = partOffset(part);
    if (scope == axisScopes.end() || scope->second.empty() || !offset)
      return std::to_string(physicalLanes);
    const std::string active = scope->second.back().active;
    const std::string lanes = std::to_string(physicalLanes);
    const std::string position = std::to_string(*offset);
    return "((" + active + " > " + position + ") ? ((" + active + " - " +
           position + " < " + lanes + ") ? " + active + " - " + position +
           " : " + lanes + ") : 0)";
  };
  auto recordForPart = [&](const Binding &owner, riscv::ValueType operand,
                           int64_t part) -> std::optional<std::string> {
    auto coordinates = coordinatesForPart(part);
    if (!coordinates || coordinates->size() != resultAxes.size())
      return std::nullopt;
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(resultAxes, *coordinates)) {
      if (!llvm::is_contained(operand.getAxisIds().asArrayRef(), axis))
        continue;
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return std::nullopt;
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    if (llvm::is_contained(operand.getAxisIds().asArrayRef(), laneAxis)) {
      auto issueOffset = partOffset(part);
      if (!issueOffset)
        return std::nullopt;
      if (plan.getLoadForm() == "unit")
        record += " + " + std::to_string(*issueOffset);
      else
        record += " + " + std::to_string(*issueOffset) + " * (" + rowStride + ")";
    }
    record += ")";
    return record;
  };

  llvm::SmallVector<std::string> lhsRecords;
  llvm::SmallVector<std::string> rhsRecords;
  for (int64_t part = 0; part < outputParts; ++part) {
    auto lhsRecord = recordForPart(lhsOwner, lhsType, part);
    auto rhsRecord = recordForPart(rhsOwner, rhsType, part);
    if (!lhsRecord || !rhsRecord)
      return fail(operation,
                  "grouped MAC window output replica has no record-coordinate mapping");
    lhsRecords.push_back(std::move(*lhsRecord));
    rhsRecords.push_back(std::move(*rhsRecord));
  }

  llvm::SmallVector<int64_t> representative(
      static_cast<size_t>(packedSupplies), int64_t{-1});
  for (int64_t part = 0; part < outputParts; ++part) {
    const int64_t supply = supplyForResult[static_cast<size_t>(part)];
    if (supply < 0 || supply >= packedSupplies)
      return fail(operation,
                  "grouped MAC window packed-supply index is out of range");
    if (representative[static_cast<size_t>(supply)] < 0)
      representative[static_cast<size_t>(supply)] = part;
  }
  if (llvm::any_of(representative,
                   [](int64_t part) { return part < 0; }))
    return fail(operation,
                "grouped MAC window packed-supply mapping is not dense");

  const int64_t storageGroup = plan.getStorageGroup();
  const int64_t storageLayer = plan.getStorageLayer();
  const bool compactWindow = plan.getKind() == "compact";
  llvm::SmallVector<std::string> compactQBases;
  llvm::SmallVector<std::string> compactXBases;
  std::string compactShift;
  if (compactWindow) {
    std::string logical = fresh("group_logical_base");
    std::string within = fresh("group_layer_within");
    std::string byte = fresh("group_storage_byte");
    compactShift = fresh("group_shift");
    line("const size_t " + logical + " = " + logicalBase + ";");
    line("const size_t " + within + " = " + logical + " % " +
         std::to_string(storageGroup) + ";");
    line("const size_t " + byte + " = " +
         std::to_string(plan.getLhsBitOffsetBytes()) + " + (" + logical + " / " +
         std::to_string(storageGroup) + ") * " +
         std::to_string(storageLayer) + " + " + within + " % " +
         std::to_string(storageLayer) + ";");
    line("const ptrdiff_t " + compactShift + " = " +
         std::to_string(plan.getShiftBase()) + " + (" + within + " / " +
         std::to_string(storageLayer) + ") * " +
         std::to_string(plan.getShiftStep()) + ";");
    const std::string qScale = std::to_string(plan.getTermByteStride());
    for (int64_t supply = 0; supply < packedSupplies; ++supply) {
      std::string qBase = fresh("group_q_window");
      line("const uint8_t *" + qBase + " = " +
           lhsRecords[static_cast<size_t>(
               representative[static_cast<size_t>(supply)])] +
           " + " + byte + " * " + qScale + ";");
      compactQBases.push_back(std::move(qBase));
    }
    for (int64_t part = 0; part < outputParts; ++part) {
      std::string xBase = fresh("group_x_window");
      line("const int8_t *" + xBase + " = (const int8_t *)(" +
           rhsRecords[static_cast<size_t>(part)] + " + " +
           std::to_string(plan.getRhsBitOffsetBytes()) + " + " + logical +
           ");");
      compactXBases.push_back(std::move(xBase));
    }
  }

  Binding window;
  window.kind = Binding::Kind::Window;
  window.windowFamily = "grouped-mac";
  const bool exactWindow = operation.getLeaf().getTail() == "exact";
  const int64_t terms = plan.getUnroll() * plan.getGroup();
  llvm::SmallVector<std::string> packedValues(
      static_cast<size_t>(packedSupplies * terms));
  for (int64_t slot = 0; slot < plan.getUnroll(); ++slot) {
    for (int64_t term = 0; term < plan.getGroup(); ++term) {
      const int64_t ordinal = slot * plan.getGroup() + term;
      const int64_t linear = plan.getTermOrder()[ordinal];
      const std::string logical = "(" + logicalBase + " + " +
                                  std::to_string(linear) + ")";
      const std::string valid =
          exactWindow ? "1"
                      : "(" + groupIndex.scalar + " * " +
                            std::to_string(plan.getGroup()) + " + " +
                            std::to_string(linear) + " < " +
                            activeTerms.scalar + ")";
      const std::string within = "((" + logical + ") % " +
                                 std::to_string(storageGroup) + ")";
      const std::string byte =
          "(" + std::to_string(plan.getLhsBitOffsetBytes()) + " + ((" +
          logical + ") / " + std::to_string(storageGroup) + ") * " +
          std::to_string(storageLayer) + " + (" + within + ") % " +
          std::to_string(storageLayer) + ")";
      const std::string dynamicShift =
          "(" + std::to_string(plan.getShiftBase()) + " + ((" + within +
          ") / " + std::to_string(storageLayer) + ") * " +
          std::to_string(plan.getShiftStep()) + ")";
      auto loadExpression = [&](llvm::StringRef qAddress,
                                llvm::StringRef qStride, llvm::StringRef vl) {
        if (qStride.empty())
          return "__riscv_vle8_v_" + rawSuffix +
                 "((const uint8_t *)(" + qAddress.str() + "), " + vl.str() +
                 ")";
        return "__riscv_vlse8_v_" + rawSuffix +
               "((const uint8_t *)(" + qAddress.str() + "), (ptrdiff_t)(" +
               qStride.str() + "), " + vl.str() + ")";
      };
      for (int64_t supply = 0; supply < packedSupplies; ++supply) {
        std::string qAddress;
        std::string qStride;
        std::string shift;
        if (compactWindow) {
          qAddress = compactQBases[static_cast<size_t>(supply)] + " + " +
                     std::to_string(linear * plan.getTermByteStride());
          if (plan.getLoadForm() == "strided")
            qStride = rowStride;
          shift = compactShift;
        } else {
          const std::string &record = lhsRecords[static_cast<size_t>(
              representative[static_cast<size_t>(supply)])];
          if (plan.getLoadForm() == "unit")
            qAddress = record + " + (" + byte + ") * " +
                       std::to_string(plan.getInterleaveRows());
          else {
            qAddress = record + " + (" + byte + ")";
            qStride = rowStride;
          }
          shift = dynamicShift;
        }
        std::string loaded = fresh("group_q_raw");
        std::string q = fresh("group_q");
        const std::string vl = resultPartVL(
            representative[static_cast<size_t>(supply)]);
        if (exactWindow) {
          line(rawType + " " + loaded + " = " +
               loadExpression(qAddress, qStride, vl) + ";");
        } else {
          line(rawType + " " + q + " = __riscv_vmv_v_x_" + rawSuffix +
               "(0, " + vl + ");");
          line("if (" + valid + ") {");
          ++indent;
          line(rawType + " " + loaded + " = " +
               loadExpression(qAddress, qStride, vl) + ";");
        }
        std::string decoded = loaded;
        if (shift != "0")
          decoded = "__riscv_vsrl_vx_" + rawSuffix + "(" + decoded + ", " +
                    shift + ", " + vl + ")";
        if (plan.getLogicalWidth() < 8)
          decoded = "__riscv_vand_vx_" + rawSuffix + "(" + decoded + ", " +
                    std::to_string(plan.getMaskValue()) + ", " + vl +
                    ")";
        if (exactWindow) {
          line(rawType + " " + q + " = " + decoded + ";");
        } else {
          line(q + " = " + decoded + ";");
          --indent;
          line("}");
        }
        packedValues[static_cast<size_t>(supply * terms + ordinal)] =
            std::move(q);
      }
      for (int64_t part = 0; part < outputParts; ++part) {
        const std::string xAddress =
            (compactWindow
                 ? compactXBases[static_cast<size_t>(part)] + " + " +
                       std::to_string(linear)
                 : rhsRecords[static_cast<size_t>(part)] + " + " +
                       std::to_string(plan.getRhsBitOffsetBytes()) + " + " +
                       logical);
        std::string x = fresh("group_x");
        if (exactWindow)
          line("int8_t " + x + " = *(const int8_t *)(" + xAddress + ");");
        else {
          line("int8_t " + x + " = 0;");
          line("if (" + valid + ") " + x + " = *(const int8_t *)(" +
               xAddress + ");");
        }
        const int64_t supply = supplyForResult[static_cast<size_t>(part)];
        window.windowLhs.push_back(
            packedValues[static_cast<size_t>(supply * terms + ordinal)]);
        window.windowRhs.push_back(std::move(x));
      }
    }
  }
  bindings[operation.getResult()] = std::move(window);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileGroupedMacStep(riscv::RVVGroupedMacStepOp operation) {
  if (operation.getLeaf().getInstruction() !=
      "rvv.vwmaccsu.vx.grouped")
    return fail(operation, "grouped MAC step has no exact selected leaf");
  Binding window = bindings.lookup(operation.getWindow());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  auto windowType = operation.getWindow().getType();
  const int64_t outputParts = windowType.getResultParts();
  const int64_t terms = windowType.getSlots() * windowType.getTermsPerSlot();
  if (window.kind != Binding::Kind::Window ||
      window.windowFamily != "grouped-mac" ||
      accumulator.kind != Binding::Kind::Vector || outputParts <= 0 ||
      accumulator.parts.size() != static_cast<size_t>(outputParts) ||
      window.windowLhs.size() != window.windowRhs.size() ||
      window.windowLhs.size() !=
          static_cast<size_t>(outputParts * terms))
    return fail(operation, "grouped MAC step has an incomplete physical window");
  const std::string partialSuffix =
      "i16" + lmulSpelling(windowType.getPartialLayout().getLmulEighths());
  const std::string partialType =
      "vint16" + lmulSpelling(windowType.getPartialLayout().getLmulEighths()) +
      "_t";
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  Binding output;
  output.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < outputParts; ++part) {
    const std::string vl = partVL(operation.getResult(), part);
    std::string result = fresh("grouped_mac");
    line(vectorType(operation.getResult()) + " " + result + " = " +
         accumulator.parts[static_cast<size_t>(part)] + ";");
    for (int64_t slot = 0; slot < windowType.getSlots(); ++slot) {
      std::string partial = fresh("group_partial");
      line(partialType + " " + partial + " = __riscv_vmv_v_x_" +
           partialSuffix + "(0, " + vl + ");");
      for (int64_t term = 0; term < windowType.getTermsPerSlot(); ++term) {
        const size_t index = static_cast<size_t>(
            (slot * windowType.getTermsPerSlot() + term) * outputParts + part);
        line(partial + " = __riscv_vwmaccsu_vx_" + partialSuffix + "(" +
             partial + ", " + window.windowRhs[index] + ", " +
             window.windowLhs[index] + ", " + vl + ");");
      }
      std::string widened = fresh("group_partial_wide");
      line(vectorType(operation.getResult()) + " " + widened +
           " = __riscv_vwcvt_x_x_v_" + resultSuffix + "(" + partial + ", " +
           vl + ");");
      line(result + " = __riscv_vadd_vv_" + resultSuffix + "(" + result +
           ", " + widened + ", " + vl + ");");
    }
    output.parts.push_back(std::move(result));
  }
  bindings[operation.getResult()] = std::move(output);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRegularRepeatIndex(
    riscv::RVVRegularRepeatIndexOp operation) {
  const int64_t repeat = operation.getRepeat();
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool powerOfTwo = instruction == "rvv.regular-repeat-index.pow2";
  if ((!powerOfTwo && instruction != "rvv.regular-repeat-index.div") ||
      operation.getResults().size() != operation.getPartBases().size())
    return fail(operation,
                "regular-repeat index has no exact selected physical spelling");
  mlir::Value firstResult = operation.getResults().front();
  const std::string indexSuffix = vectorSuffix(firstResult);
  const int64_t lanes = physicalLanes(firstResult);
  std::string baseExpression = "__riscv_vid_v_" + indexSuffix + "(" +
                               std::to_string(lanes) + ")";
  if (powerOfTwo) {
    int64_t shift = 0;
    for (int64_t value = repeat; value > 1; value >>= 1)
      ++shift;
    baseExpression = "__riscv_vsrl_vx_" + indexSuffix + "(" +
                     baseExpression + ", " + std::to_string(shift) + ", " +
                     std::to_string(lanes) + ")";
  } else {
    baseExpression = "__riscv_vdivu_vx_" + indexSuffix + "(" +
                     baseExpression + ", " + std::to_string(repeat) + ", " +
                     std::to_string(lanes) + ")";
  }
  std::string baseIndex = fresh("repeat_index_base");
  line(vectorType(firstResult) + " " + baseIndex + " = " + baseExpression +
       ";");
  for (auto [resultValue, basesAttribute] :
       llvm::zip(operation.getResults(), operation.getPartBases())) {
    auto bases = mlir::cast<mlir::DenseI64ArrayAttr>(basesAttribute);
    const int64_t streams = streamPartCount(resultValue);
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(resultValue); ++part) {
      const int64_t stream = part % streams;
      if (stream < 0 || stream >= static_cast<int64_t>(bases.size()))
        return fail(operation,
                    "regular-repeat index result part exceeds its typed bases");
      const std::string suffix = vectorSuffix(resultValue);
      std::string expression = baseIndex;
      if (bases[stream] != 0)
        expression = "__riscv_vadd_vx_" + suffix + "(" + expression + ", " +
                     std::to_string(bases[stream]) + ", " +
                     partVL(resultValue, part) + ")";
      std::string name = fresh("repeat_index");
      line(vectorType(resultValue) + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[resultValue] = std::move(result);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRegularRepeatGather(
    riscv::RVVRegularRepeatGatherOp operation) {
  mlir::Value firstValue = operation.getResults().front();
  const int64_t lanes = physicalLanes(firstValue);
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool joinedBroadcast =
      instruction == "rvv.regular-repeat-joined-broadcast";
  const bool joinedGatherPowerOfTwo =
      instruction == "rvv.regular-repeat-joined-gather.pow2";
  const bool joinedGatherDivision =
      instruction == "rvv.regular-repeat-joined-gather.div";
  const bool joinedUnit = instruction == "rvv.regular-repeat-joined-unit";
  const bool joinedGather = joinedGatherPowerOfTwo || joinedGatherDivision || joinedUnit;
  const bool broadcastOnly =
      joinedBroadcast || instruction == "rvv.regular-repeat-broadcast";
  const bool powerOfTwo = instruction == "rvv.regular-repeat-gather.pow2";
  if (!broadcastOnly && !joinedGather && !powerOfTwo &&
      instruction != "rvv.regular-repeat-gather.div")
    return fail(operation,
                "regular-repeat gather has no exact selected index spelling");
  Binding field = bindings.lookup(operation.getField());
  if (field.kind != Binding::Kind::Field || field.field.index ||
      operation.getResults().empty())
    return fail(operation,
                "regular-repeat gather requires one unprojected encoded field");
  Binding owner = bindings.lookup(field.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(field.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto storage = fieldFor(field);
  auto elementType = storage ? scalarCType(storage->type) : std::nullopt;
  auto integer = storage
                     ? mlir::dyn_cast<mlir::IntegerType>(storage->type)
                     : mlir::IntegerType();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !storage || !elementType || !integer || storage->bitOffset % 8)
    return fail(operation,
                "regular-repeat gather has incomplete record storage geometry");
  const std::string suffix = vectorSuffix(firstValue);
  const std::string type = vectorType(firstValue);
  auto layout = layoutOf(firstValue);
  if (!layout)
    return fail(operation, "regular-repeat gather result has no RVV layout");
  if (lanes <= 1)
    return fail(operation, "regular-repeat gather has no lane extent");

  const int64_t registerParts = registerPartCount(firstValue);
  auto sourceBase = materializeNumeric(operation.getSourceBase(),
                                       bindings.lookup(operation.getSourceBase()));
  if (mlir::failed(sourceBase) || sourceBase->kind != Binding::Kind::Scalar)
    return fail(operation,
                "regular-repeat gather requires one selected scalar source base");
  llvm::SmallVector<std::string> sourceWindows;
  llvm::SmallVector<std::string> sourceRecords;
  llvm::SmallVector<int64_t, 4> axes = registerAxesFor(firstValue);
  for (int64_t replica = 0; replica < registerParts; ++replica) {
    auto coordinates = registerCoordinates(firstValue, replica);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "regular-repeat gather has no free-axis register coordinates");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto byteStride = llvm::find_if(owner.recordByteStrides,
                                      [&](const auto &entry) {
                                        return entry.first == axis;
                                      });
      if (byteStride == owner.recordByteStrides.end())
        return fail(operation,
                    "regular-repeat gather free axis has no record stride");
      record += " + " + std::to_string(coordinate) + " * " +
                byteStride->second;
    }
    record += ")";
    sourceRecords.push_back(record);
    if (joinedUnit) {
      std::string raw = fresh("joined_unit_bytes");
      line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
           "((const uint8_t *)(" + record + ") + " +
           std::to_string(storage->bitOffset / 8 +
                          operation.getAccess().getJoinRole() *
                              operation.getAccess().getGroupSize()) +
           ", " + std::to_string(operation.getSourceCount()) + ");");
      sourceWindows.push_back(std::move(raw));
      continue;
    }
    if (broadcastOnly || joinedGather)
      continue;
    if (operation.getAccess().getMapping() != "natural")
      return fail(operation,
                  "non-broadcast regular-repeat gather requires natural storage");
    const std::string pointer =
        "((const " + *elementType + " *)((const uint8_t *)(" + record +
        ") + " + std::to_string(storage->bitOffset / 8) + ") + " +
        sourceBase->scalar + ")";
    std::string loaded = fresh("repeat_source");
    line(type + " " + loaded + " = __riscv_vle" +
         std::to_string(integer.getWidth()) + "_v_" + suffix + "(" + pointer +
         ", " + std::to_string(operation.getSourceCount()) + ");");
    sourceWindows.push_back(std::move(loaded));
  }

  for (auto [resultNumber, values] : llvm::enumerate(
           llvm::zip(operation.getResults(), operation.getPartBases()))) {
    auto [resultValue, basesAttribute] = values;
    auto bases = mlir::cast<mlir::DenseI64ArrayAttr>(basesAttribute);
    const int64_t streams = streamPartCount(resultValue);
    Binding indexBinding;
    if (!broadcastOnly) {
      if (resultNumber >= operation.getIndices().size())
        return fail(operation,
                    "regular-repeat gather lost its typed index operand");
      indexBinding = bindings.lookup(operation.getIndices()[resultNumber]);
      if (indexBinding.kind != Binding::Kind::Vector ||
          indexBinding.parts.size() !=
              static_cast<size_t>(vectorPartCount(resultValue)))
        return fail(operation,
                    "regular-repeat gather typed index is not materialized");
    }
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(resultValue); ++part) {
      const int64_t replica = part / streams;
      const int64_t stream = part % streams;
      const int64_t availableSources = static_cast<int64_t>(
          (broadcastOnly || joinedGather) ? sourceRecords.size()
                                          : sourceWindows.size());
      if (replica < 0 || replica >= availableSources ||
          stream < 0 || stream >= static_cast<int64_t>(bases.size()))
        return fail(operation,
                    "regular-repeat gather result parts exceed its selected window");
      if (broadcastOnly) {
        const std::string sourceIndex = "(" + sourceBase->scalar + " + " +
                                        std::to_string(bases[stream]) + ")";
        Binding sourceRecord = owner;
        sourceRecord.recordPointer = sourceRecords[replica];
        auto scalarValue = scalarFieldValue(sourceRecord, *storage, sourceIndex);
        if (!scalarValue)
          return fail(operation,
                      "regular-repeat broadcast has no closed scalar storage spelling");
        std::string gathered = fresh("regular_broadcast");
        const bool floating = mlir::isa<mlir::FloatType>(storage->type);
        line(vectorType(resultValue) + " " + gathered + " = __riscv_" +
             std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
             vectorSuffix(resultValue) + "(" + *scalarValue + ", " +
             partVL(resultValue, part) + ");");
        result.parts.push_back(std::move(gathered));
        continue;
      }
      if (joinedGather) {
        mlir::Value indexValue = operation.getIndices()[resultNumber];
        auto indexLayout = layoutOf(indexValue);
        auto resultLayout = layoutOf(resultValue);
        auto logicalInteger =
            mlir::dyn_cast<mlir::IntegerType>(storage->type);
        const int64_t group = operation.getAccess().getGroupSize();
        const int64_t fields = operation.getAccess().getJoinFields();
        const int64_t lowBits = operation.getAccess().getJoinLowBits();
        const int64_t role = operation.getAccess().getJoinRole();
        const int64_t physicalRole =
            operation.getAccess().getOrder() == "lo_first"
                ? role
                : fields - 1 - role;
        const unsigned logicalWidth =
            logicalInteger ? logicalInteger.getWidth() : 0;
        const int64_t maskBits =
            resultLayout
                ? resultLayout.getSew() * 8 /
                      resultLayout.getLmulEighths()
                : 0;
        if (!indexLayout || !resultLayout || !logicalInteger ||
            !logicalInteger.isUnsigned() || logicalWidth == 0 ||
            logicalWidth > 8 || group <= 0 || fields <= 1 || lowBits <= 0 ||
            lowBits >= static_cast<int64_t>(logicalWidth) || role < 0 ||
            role >= fields ||
            (maskBits != 1 && maskBits != 2 && maskBits != 4 &&
             maskBits != 8 && maskBits != 16 && maskBits != 32 &&
             maskBits != 64) ||
            indexLayout.getSew() != resultLayout.getSew() ||
            indexLayout.getLmulEighths() !=
                resultLayout.getLmulEighths())
          return fail(operation,
                      "joined regular-repeat gather has incomplete typed byte geometry");
        const std::string indexSuffix = vectorSuffix(indexValue);
        const std::string indexType = vectorType(indexValue);
        const std::string resultSuffix = vectorSuffix(resultValue);
        const std::string resultType = vectorType(resultValue);
        const std::string vl = partVL(resultValue, part);
        const uint64_t logicalMask = (uint64_t(1) << logicalWidth) - 1;
        const uint64_t lowMask = (uint64_t(1) << lowBits) - 1;
        const uint64_t highMask =
            (uint64_t(1) << (logicalWidth - lowBits)) - 1;

        std::string logicalIndex = indexBinding.parts[part];
        if (sourceBase->scalar != "0") {
          std::string adjusted = fresh("joined_index");
          line(indexType + " " + adjusted + " = __riscv_vadd_vx_" +
               indexSuffix + "(" + logicalIndex + ", " + sourceBase->scalar +
               ", " + vl + ");");
          logicalIndex = std::move(adjusted);
        }
        std::string within = fresh("joined_within");
        line(indexType + " " + within + " = __riscv_" +
             std::string((joinedGatherPowerOfTwo || joinedUnit) ? "vand_vx_" : "vremu_vx_") +
             indexSuffix + "(" + logicalIndex + ", " +
             std::to_string((joinedGatherPowerOfTwo || joinedUnit) ? group - 1 : group) +
             ", " + vl + ");");
        auto addByteOffset = [&](int64_t offset,
                                 llvm::StringRef stem) -> std::string {
          if (offset == 0)
            return within;
          std::string value = fresh(stem);
          line(indexType + " " + value + " = __riscv_vadd_vx_" +
               indexSuffix + "(" + within + ", " +
               std::to_string(offset) + ", " + vl + ");");
          return value;
        };
        const std::string pointer =
            "((const uint8_t *)(" + sourceRecords[replica] + ") + " +
            std::to_string(storage->bitOffset / 8) + ")";
        std::string roleBytes = fresh("joined_role_bytes");
        std::string lowBytes = fresh("joined_low_bytes");
        if (joinedUnit) {
          const std::string lowOffsets = addByteOffset(group, "joined_low_offsets");
          line(resultType + " " + roleBytes + " = __riscv_vrgather_vv_" + resultSuffix +
               "(" + sourceWindows[replica] + ", " + within + ", " + vl + ");");
          line(resultType + " " + lowBytes + " = __riscv_vrgather_vv_" + resultSuffix +
               "(" + sourceWindows[replica] + ", " + lowOffsets + ", " + vl + ");");
        } else {
          const std::string roleOffsets = addByteOffset(role * group, "joined_role_offsets");
          const std::string lowOffsets = addByteOffset(fields * group, "joined_low_offsets");
          line(resultType + " " + roleBytes + " = __riscv_vluxei" +
               std::to_string(indexLayout.getSew()) + "_v_" + resultSuffix +
               "(" + pointer + ", " + roleOffsets + ", " + vl + ");");
          line(resultType + " " + lowBytes + " = __riscv_vluxei" +
               std::to_string(indexLayout.getSew()) + "_v_" + resultSuffix +
               "(" + pointer + ", " + lowOffsets + ", " + vl + ");");
        }
        const std::string head =
            "__riscv_vand_vx_" + resultSuffix + "(" + roleBytes + ", " +
            std::to_string(logicalMask) + ", " + vl + ")";
        const uint64_t selectedLowMask = joinedUnit ? operation.getLeaf().getParameters()[4] : lowMask;
        const uint64_t selectedHighMask = joinedUnit ? operation.getLeaf().getParameters()[5] : highMask;
        std::string low = "__riscv_vsrl_vx_" + resultSuffix + "(" + lowBytes + ", " +
                          std::to_string(physicalRole * lowBits) + ", " + vl + ")";
        std::string high = "__riscv_vsrl_vx_" + resultSuffix + "(" + roleBytes + ", " +
                           std::to_string(logicalWidth) + ", " + vl + ")";
        if (selectedLowMask)
          low = "__riscv_vand_vx_" + resultSuffix + "(" + low + ", " +
                std::to_string(selectedLowMask) + ", " + vl + ")";
        if (selectedHighMask)
          high = "__riscv_vand_vx_" + resultSuffix + "(" + high + ", " +
                 std::to_string(selectedHighMask) + ", " + vl + ")";
        const std::string assembled =
            operation.getAccess().getOrder() == "lo_first"
                ? "__riscv_vor_vv_" + resultSuffix + "(" + low +
                      ", __riscv_vsll_vx_" + resultSuffix + "(" + high +
                      ", " + std::to_string(lowBits) + ", " + vl + "), " +
                      vl + ")"
                : "__riscv_vor_vv_" + resultSuffix + "(" + high +
                      ", __riscv_vsll_vx_" + resultSuffix + "(" + low +
                      ", " + std::to_string(logicalWidth - lowBits) + ", " +
                      vl + "), " + vl + ")";
        std::string firstHalf = fresh("joined_first_half");
        line("vbool" + std::to_string(maskBits) + "_t " + firstHalf +
             " = __riscv_vmsltu_vx_" + indexSuffix + "_b" +
             std::to_string(maskBits) + "(" + logicalIndex + ", " +
             std::to_string(group) + ", " + vl + ");");
        std::string decoded = fresh("joined_gather");
        line(resultType + " " + decoded + " = __riscv_vmerge_vvm_" +
             resultSuffix + "(" + assembled + ", " + head + ", " +
             firstHalf + ", " + vl + ");");
        result.parts.push_back(std::move(decoded));
        continue;
      }
      std::string gathered = fresh("regular_gather");
      line(vectorType(resultValue) + " " + gathered +
           " = __riscv_vrgather_vv_" + vectorSuffix(resultValue) + "(" +
           sourceWindows[replica] + ", " + indexBinding.parts[part] + ", " +
           partVL(resultValue, part) + ");");
      result.parts.push_back(std::move(gathered));
    }
    bindings[resultValue] = std::move(result);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRegularRepeatScalarLoad(
    riscv::RVVRegularRepeatScalarLoadOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "scalar.regular-repeat-load")
    return fail(operation,
                "regular-repeat scalar load has no exact selected spelling");
  Binding field = bindings.lookup(operation.getField());
  if (field.kind != Binding::Kind::Field || field.field.index)
    return fail(operation,
                "regular-repeat scalar load requires one unprojected encoded field");
  Binding owner = bindings.lookup(field.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(field.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto storage = fieldFor(field);
  auto elementType = storage ? scalarCType(storage->type) : std::nullopt;
  auto sourceBase = materializeNumeric(operation.getSourceBase(),
                                       bindings.lookup(operation.getSourceBase()));
  const int64_t sourceCount = operation.getSourceCount();
  const int64_t repeat = operation.getRepeat();
  const int64_t scalarParts = scalarPartCount(operation.getResult());
  auto partBases = operation.getPartBases();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !storage || !elementType || storage->bitOffset % 8 ||
      mlir::failed(sourceBase) || sourceBase->kind != Binding::Kind::Scalar ||
      sourceCount <= 0 || repeat <= 0 || scalarParts <= 0 ||
      partBases.size() != static_cast<size_t>(scalarParts))
    return fail(operation,
                "regular-repeat scalar load has incomplete typed storage geometry");

  llvm::SmallVector<std::string> sources;
  sources.reserve(sourceCount);
  for (int64_t source = 0; source < sourceCount; ++source) {
    const std::string sourceIndex = "(" + sourceBase->scalar + " + " +
                                    std::to_string(source) + ")";
    auto expression = scalarFieldValue(owner, *storage, sourceIndex);
    if (!expression)
      return fail(operation,
                  "regular-repeat scalar load has no closed encoded-field spelling");
    std::string loaded = fresh("repeat_scalar");
    line(*elementType + " " + loaded + " = " + *expression + ";");
    sources.push_back(std::move(loaded));
  }
  Binding result;
  result.kind = scalarParts == 1 ? Binding::Kind::Scalar
                                 : Binding::Kind::ScalarTuple;
  result.parts.reserve(scalarParts);
  for (int64_t base : partBases) {
    if (base < 0 || base >= sourceCount)
      return fail(operation,
                  "regular-repeat scalar load part exceeds its source window");
    result.parts.push_back(sources[base]);
  }
  if (result.kind == Binding::Kind::Scalar) {
    result.scalar = result.parts.front();
    result.parts.clear();
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVLayeredStream(riscv::RVVLayeredStreamOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.layered-stream")
    return fail(operation, "RVV layered stream has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation, "RVV layered stream requires one encoded field edge");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto geometry = operation.getGeometry();
  const int64_t layer = geometry.getLayerSize();
  const int64_t streams = geometry.getStreamCount();
  const int64_t replicas = geometry.getReplicaCount();
  const int64_t lanes = geometry.getLaneCount();
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  if (owner.kind != Binding::Kind::Record || !field || !integer ||
      integer.isSigned() || field->bitOffset % 8 || layer <= 0 || lanes <= 0 ||
      streams <= 1 || replicas <= 0 ||
      geometry.getWindowForStream().size() != static_cast<size_t>(streams) ||
      geometry.getRepresentativeStreamForWindow().size() !=
          geometry.getGroupForWindow().size() ||
      geometry.getPhysicalLayerForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getShiftAmountForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getMaskValueForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getGroupForWindow().size() !=
          geometry.getWithinForWindow().size())
    return fail(operation,
                "RVV layered stream has incomplete field, record, or layout geometry");

  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t replica = 0; replica < replicas; ++replica) {
    auto coordinates = registerCoordinates(operation.getResult(), replica);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "RVV layered stream has no register-coordinate mapping");
    Binding record = owner;
    std::string pointer = "(" + owner.recordPointer;
    std::string rowOffset = "0";
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      if (owner.interleaveRows > 0 && axis == owner.interleaveRowAxis) {
        rowOffset = std::to_string(coordinate);
        continue;
      }
      bool foundStride = false;
      for (const auto &[recordAxis, stride] : owner.recordByteStrides)
        if (recordAxis == axis) {
          pointer += " + " + std::to_string(coordinate) + " * " + stride;
          foundStride = true;
        }
      if (!foundStride)
        return fail(operation,
                    "RVV layered stream register axis has no record-byte stride");
    }
    record.recordPointer = pointer + ")";
    llvm::SmallVector<std::string> loadedWindows;
    loadedWindows.reserve(geometry.getRepresentativeStreamForWindow().size());
    for (auto [window, firstStream] :
         llvm::enumerate(
             geometry.getRepresentativeStreamForWindow().asArrayRef())) {
      if (firstStream < 0)
        return fail(operation,
                    "RVV layered stream geometry contains an unused raw window");
      const int64_t byteOffset =
          field->bitOffset / 8 + geometry.getGroupForWindow()[window] * layer +
          geometry.getWithinForWindow()[window];
      const int64_t part = replica * streams + firstStream;
      const std::string address = fieldStorageAddress(
          record, *field, std::to_string(byteOffset), rowOffset);
      std::string raw = fresh("layered_stream_raw");
      line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
           "((const uint8_t *)(" + address + "), " +
           partVL(operation.getResult(), part) + ");");
      loadedWindows.push_back(std::move(raw));
    }
    for (int64_t stream = 0; stream < streams; ++stream) {
      const int64_t part = replica * streams + stream;
      const int64_t window = geometry.getWindowForStream()[stream];
      const int64_t shiftAmount = geometry.getShiftAmountForStream()[stream];
      const int64_t maskValue = geometry.getMaskValueForStream()[stream];
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = loadedWindows[static_cast<size_t>(window)];
      if (shiftAmount != 0) {
        std::string shifted = fresh("layered_stream_shift");
        line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
             value + ", " + std::to_string(shiftAmount) + ", " + vl +
             ");");
        value = std::move(shifted);
      }
      if (maskValue != 0) {
        std::string masked = fresh("layered_stream_value");
        line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
             value + ", " + std::to_string(maskValue) + ", " + vl + ");");
        value = std::move(masked);
      }
      result.parts.push_back(std::move(value));
    }
  }
  if (result.parts.size() !=
      static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV layered stream did not produce every selected physical part");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVProjectedLayeredStream(
    riscv::RVVProjectedLayeredStreamOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.projected-layered-stream")
    return fail(operation,
                "projected RVV layered stream has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  if (fieldBinding.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point)
    return fail(operation,
                "projected RVV layered stream requires one encoded field and typed point");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto geometry = operation.getGeometry();
  const int64_t group = geometry.getGroupSize();
  const int64_t layer = geometry.getLayerSize();
  const int64_t streams = geometry.getStreamCount();
  const int64_t replicas = geometry.getReplicaCount();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !field || !integer || integer.isSigned() || field->bitOffset % 8 ||
      group <= 0 || layer <= 0 || group % layer || streams <= 1 || replicas <= 0 ||
      operation.getProjectionStride() != 1 ||
      operation.getProjectionRepeat() != 1 ||
      geometry.getWindowForStream().size() != static_cast<size_t>(streams) ||
      geometry.getRepresentativeStreamForWindow().size() !=
          geometry.getGroupForWindow().size() ||
      geometry.getPhysicalLayerForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getShiftAmountForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getMaskValueForStream().size() !=
          static_cast<size_t>(streams) ||
      geometry.getGroupForWindow().size() !=
          geometry.getWithinForWindow().size())
    return fail(operation,
                "projected RVV layered stream has incomplete field or projection geometry");

  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t replica = 0; replica < replicas; ++replica) {
    auto coordinates = registerCoordinates(operation.getResult(), replica);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "projected RVV layered stream has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "projected RVV layered stream free axis has no record stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += ")";
    const std::string recordGroup =
        "(((" + point.point.base + ") % " +
        std::to_string(owner.recordElements) + ") / " +
        std::to_string(group) + ")";
    llvm::SmallVector<std::string> loadedWindows;
    loadedWindows.reserve(geometry.getRepresentativeStreamForWindow().size());
    for (auto [window, firstStream] :
         llvm::enumerate(
             geometry.getRepresentativeStreamForWindow().asArrayRef())) {
      if (firstStream < 0)
        return fail(operation,
                    "projected RVV layered stream geometry contains an unused raw window");
      const std::string byte =
          "(" + std::to_string(field->bitOffset / 8) + " + (" + recordGroup +
          " + " + std::to_string(geometry.getGroupForWindow()[window]) +
          ") * " + std::to_string(layer) + " + " +
          std::to_string(geometry.getWithinForWindow()[window]) + ")";
      const int64_t part = replica * streams + firstStream;
      std::string raw = fresh("projected_layered_raw");
      line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
           "((const uint8_t *)(" + record + " + " + byte + "), " +
           partVL(operation.getResult(), part) + ");");
      loadedWindows.push_back(std::move(raw));
    }
    for (int64_t stream = 0; stream < streams; ++stream) {
      const int64_t part = replica * streams + stream;
      const int64_t window = geometry.getWindowForStream()[stream];
      const int64_t shiftAmount = geometry.getShiftAmountForStream()[stream];
      const int64_t maskValue = geometry.getMaskValueForStream()[stream];
      const std::string vl = partVL(operation.getResult(), part);
      std::string value = loadedWindows[static_cast<size_t>(window)];
      if (shiftAmount != 0) {
        std::string shifted = fresh("projected_layered_shift");
        line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
             value + ", " + std::to_string(shiftAmount) + ", " + vl +
             ");");
        value = std::move(shifted);
      }
      if (maskValue != 0) {
        std::string masked = fresh("projected_layered_value");
        line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
             value + ", " + std::to_string(maskValue) + ", " + vl + ");");
        value = std::move(masked);
      }
      result.parts.push_back(std::move(value));
    }
  }
  if (result.parts.size() !=
      static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "projected RVV layered stream did not produce every physical part");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVLayeredWindow(riscv::RVVLayeredWindowOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.layered-window")
    return fail(operation, "layered window has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation, "layered window requires one encoded field edge");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  Binding point = bindings.lookup(operation.getPoint());
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  auto plan = operation.getPlan();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  if (owner.kind != Binding::Kind::Record || !field ||
      point.kind != Binding::Kind::Point || !integer || integer.isSigned() ||
      field->bitOffset % 8 || plan.getKind() != "layered" ||
      group != layer * 2 || owner.recordElements != plan.getRecordElements() ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      operation.getPhysicalLayerForResult().size() != 2 ||
      operation.getShiftForResult().size() != 2 ||
      operation.getMaskForResult().size() != 2 ||
      vectorPartCount(operation.getFirst()) !=
          vectorPartCount(operation.getSecond()))
    return fail(operation,
                "layered window has no closed two-layer encoded realization");

  const std::string suffix = vectorSuffix(operation.getFirst());
  const std::string type = vectorType(operation.getFirst());
  const int64_t streams = streamPartCount(operation.getFirst());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getFirst());
  const std::string logical = "(" + point.point.base + " % " +
                              std::to_string(plan.getRecordElements()) + ")";
  Binding first;
  Binding second;
  first.kind = Binding::Kind::Vector;
  second.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getFirst()); ++part) {
    auto coordinates = registerCoordinates(operation.getFirst(), part / streams);
    if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "layered window has no complete register coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates))
      for (const auto &[recordAxis, stride] : owner.recordByteStrides)
        if (recordAxis == axis)
          record += " + " + std::to_string(coordinate) + " * " + stride;
    record += ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + (" + logical +
        " / " + std::to_string(group) + ") * " + std::to_string(layer) +
        " + (" + logical + " % " + std::to_string(layer) + ") + " +
        partOffset(operation.getFirst(), part) + ")";
    const std::string vl = partVL(operation.getFirst(), part);
    std::string raw = fresh("layered_raw");
    line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte + "), " + vl +
         ");");
    auto decode = [&](size_t resultIndex, llvm::StringRef prefix) {
      const int64_t shiftAmount = operation.getShiftForResult()[resultIndex];
      const int64_t maskValue = operation.getMaskForResult()[resultIndex];
      std::string value = raw;
      if (shiftAmount != 0) {
        std::string shifted = fresh("layered_shift");
        line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
             value + ", " + std::to_string(shiftAmount) + ", " + vl +
             ");");
        value = std::move(shifted);
      }
      if (maskValue == 0)
        return value;
      std::string masked = fresh(prefix);
      line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
           value + ", " + std::to_string(maskValue) + ", " + vl + ");");
      return masked;
    };
    first.parts.push_back(decode(0, "layered_first"));
    second.parts.push_back(decode(1, "layered_second"));
  }
  bindings[operation.getFirst()] = std::move(first);
  bindings[operation.getSecond()] = std::move(second);
  return mlir::success();
}


mlir::LogicalResult Emitter::compileScalarTileLoad(riscv::ScalarTileLoadOp operation) {
  auto instruction = instructionOf(operation);
  const bool unit = instruction == "scalar.tile-load.unit";
  if (instruction != "scalar.tile-load" && instruction != "scalar.tile-load.strided" && !unit)
    return fail(operation, "scalar tile load lost its selected transfer leaf");
  Binding source = bindings.lookup(operation.getSource());
  if (source.kind == Binding::Kind::Memory) {
    source.kind = Binding::Kind::Slice;
    source.slice.base = operation.getSource();
  }
  auto type = operation.getResult().getType();
  auto geometry = riscv::RegisterCoordinates::get(type.getLayout());
  auto ctype = scalarCType(type.getElementType());
  if (source.kind != Binding::Kind::Slice || !geometry || !ctype)
    return fail(operation, "scalar tile load has no typed descriptor or scalar coordinates");
  Binding result;
  result.kind = geometry->parts() == 1 ? Binding::Kind::Scalar : Binding::Kind::ScalarTuple;
  for (int64_t part = 0; part < geometry->parts(); ++part) {
    auto point = geometry->logical(part);
    if (!point) return fail(operation, "scalar tile load lost a physical coordinate");
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    std::string valid = "1";
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      auto bound = bindings.lookup(operation.getExtents()[position]);
      if (bound.kind != Binding::Kind::Scalar)
        return fail(operation, "scalar tile load bound has no scalar binding");
      offsets.emplace_back(axis, unit && axis == operation.getLeaf().getParameters()[0]
          ? "0" : std::to_string((*point)[position]));
      if (operation.getGuarded()) valid += " && " + std::to_string((*point)[position]) + " < " + bound.scalar;
    }
    auto address = denseAddress(source.slice, offsets);
    if (!address) return fail(operation, "scalar tile load has no exact dense address");
    if (unit) {
      auto axes = type.getAxisIds().asArrayRef();
      size_t position = llvm::find(axes, operation.getLeaf().getParameters()[0]) - axes.begin();
      *address = "(" + *address + ") + " + std::to_string((*point)[position]);
    }
    auto value = fresh("tile_scalar");
    if (operation.getGuarded()) {
      line(*ctype + " " + value + " = 0;");
      line("if (" + valid + ") " + value + " = *(" + *address + ");");
    } else line(*ctype + " " + value + " = *(" + *address + ");");
    if (result.kind == Binding::Kind::Scalar) result.scalar = value;
    else result.parts.push_back(value);
  }
  if (auto packet = operation.getPacket()) {
    if (result.kind != Binding::Kind::ScalarTuple || result.parts.size() != size_t(packet->getParts()))
      return fail(operation, "closed scalar load packet lost its exact resident parts");
    std::string boundary = "__asm__ volatile(\"\" : ";
    for (auto [index, part] : llvm::enumerate(result.parts)) {
      if (index) boundary += ", ";
      boundary += "\"+f\"(" + part + ")";
    }
    line(boundary + " : : \"memory\");");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileScalarTileStore(riscv::ScalarTileStoreOp operation) {
  if (instructionOf(operation) != "scalar.tile-store")
    return fail(operation, "scalar tile store lost its selected transfer leaf");
  Binding destination = bindings.lookup(operation.getDestination());
  if (destination.kind == Binding::Kind::Memory) {
    destination.kind = Binding::Kind::Slice;
    destination.slice.base = operation.getDestination();
  }
  auto type = operation.getInput().getType();
  auto geometry = riscv::RegisterCoordinates::get(type.getLayout());
  Binding input = bindings.lookup(operation.getInput());
  if (destination.kind != Binding::Kind::Slice || !geometry ||
      (input.kind != Binding::Kind::Scalar && input.kind != Binding::Kind::ScalarTuple) ||
      (input.kind == Binding::Kind::Scalar ? geometry->parts() != 1 : input.parts.size() != size_t(geometry->parts())))
    return fail(operation, "scalar tile store lost its descriptor or scalar parts");
  for (int64_t part = 0; part < geometry->parts(); ++part) {
    auto point = geometry->logical(part);
    if (!point) return fail(operation, "scalar tile store lost a physical coordinate");
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    std::string valid = "1";
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      const auto &bound = bindings.lookup(operation.getExtents()[position]);
      if (bound.kind != Binding::Kind::Scalar)
        return fail(operation, "scalar tile store bound has no scalar binding");
      offsets.emplace_back(axis, std::to_string((*point)[position]));
      if (operation.getGuarded()) valid += " && " + std::to_string((*point)[position]) + " < " + bound.scalar;
    }
    auto address = denseAddress(destination.slice, offsets);
    if (!address) return fail(operation, "scalar tile store has no exact dense address");
    std::string value = input.kind == Binding::Kind::Scalar ? input.scalar : input.parts[part];
    line((operation.getGuarded() ? "if (" + valid + ") " : "") + "*(" + *address + ") = " + value + ";");
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVStreamReduceStep(
    riscv::RVVStreamReduceStepOp operation) {
  bool full = instructionOf(operation.getOperation()) == "rvv.stream-reduce-full-step";
  if (instructionOf(operation.getOperation()) != "rvv.stream-reduce-step" && !full)
    return fail(operation, "RVV stream reduce step has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  Binding active = bindings.lookup(operation.getActive());
  if (input.kind != Binding::Kind::Vector ||
      accumulator.kind != Binding::Kind::Vector ||
      active.kind != Binding::Kind::Scalar ||
      input.parts.size() != accumulator.parts.size())
    return fail(operation,
                "RVV stream reduce step requires matching vector parts and active lanes");
  llvm::StringRef stem = operation.getResult().getType().getElementType().isInteger(32) ? "vadd"
                         : operation.getKind() == "max" ? "vfmax"
                         : operation.getKind() == "min" ? "vfmin"
                                                        : "vfadd";
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < input.parts.size(); ++part) {
    std::string value = fresh("stream_reduce_step");
    line(vectorType(operation.getResult()) + " " + value + " = __riscv_" +
         stem.str() + "_vv_" + vectorSuffix(operation.getResult()) + (full ? "(" : "_tu(") +
         (full ? "" : accumulator.parts[part] + ", ") + accumulator.parts[part] + ", " + input.parts[part] + ", " +
         active.scalar + ");");
    result.parts.push_back(std::move(value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVStreamContractStep(
    riscv::RVVStreamContractStepOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool full = instruction == "rvv.stream-contract-full-step" ||
                    instruction == "rvv.stream-widen-contract-full-step";
  const bool widening = instruction == "rvv.stream-widen-contract-step" ||
                        instruction == "rvv.stream-widen-contract-full-step";
  if (instruction != "rvv.stream-contract-step" && !full && !widening)
    return fail(operation,
                "RVV stream contract step has no exact selected leaf");
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  Binding active = bindings.lookup(operation.getActive());
  if (lhs.kind != Binding::Kind::Vector || rhs.kind != Binding::Kind::Vector ||
      accumulator.kind != Binding::Kind::Vector ||
      active.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV stream contract step requires typed vector operands and active lanes");
  Binding result;
  result.kind = Binding::Kind::Vector;
  auto geometry = riscv::StreamContractGeometry::get(operation.getLhs().getType(),
      operation.getRhs().getType(), operation.getAccumulator().getType(), operation.getReductionAxis());
  if (!geometry || geometry->lhsParts.size() != accumulator.parts.size())
    return fail(operation, "RVV stream contract lost its verified carrier projection");
  for (size_t part = 0; part < accumulator.parts.size(); ++part) {
    int64_t lhsPart = geometry->lhsParts[part], rhsPart = geometry->rhsParts[part];
    if (lhsPart >= int64_t(lhs.parts.size()) || rhsPart >= int64_t(rhs.parts.size()))
      return fail(operation,
                  "RVV stream contract step has no operand-to-accumulator replica mapping");
    std::string value = fresh("stream_contract_step");
    line(vectorType(operation.getResult()) + " " + value + " = __riscv_" +
         std::string(widening ? "vfwmacc_vv_" : "vfmacc_vv_") +
         vectorSuffix(operation.getResult()) + (full ? "(" : "_tu(") + accumulator.parts[part] +
         ", " + lhs.parts[lhsPart] + ", " + rhs.parts[rhsPart] + ", " +
         active.scalar + ");");
    result.parts.push_back(std::move(value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVStreamFinalize(riscv::RVVStreamFinalizeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.stream-finalize")
    return fail(operation, "RVV stream finalize has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  if (input.kind != Binding::Kind::Vector || input.parts.empty())
    return fail(operation,
                "RVV stream finalize requires one materialized vector accumulator");
  auto element = operation.getInput().getType().getElementType();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  const bool floating = !integer;
  llvm::StringRef stem = integer ? "vredsum"
                         : operation.getKind() == "max" ? "vfredmax"
                         : operation.getKind() == "min" ? "vfredmin"
                                                        : "vfredusum";
  const std::string initial = operation.getKind() == "max" ? "(-INFINITY)"
                              : operation.getKind() == "min" ? "INFINITY"
                                                               : "0";
  Binding result;
  result.kind = input.parts.size() == 1 ? Binding::Kind::Scalar
                                        : Binding::Kind::ScalarTuple;
  const std::string suffix = std::string(floating ? "f" : integer.isUnsigned() ? "u" : "i") +
                            std::to_string(element.getIntOrFloatBitWidth());
  const std::string scalarType = *scalarCType(element);
  const std::string reductionType =
      std::string(floating ? "vfloat" : integer.isUnsigned() ? "vuint" : "vint") +
      std::to_string(element.getIntOrFloatBitWidth()) + "m1_t";
  auto projection = riscv::streamFinalizeParts(operation.getInput().getType(),
      operation.getResult().getType(), operation.getAxis());
  if (!projection || projection->size() != input.parts.size())
    return fail(operation, "RVV stream finalize lost its verified scalar projection");
  for (int64_t index : *projection) {
    llvm::StringRef part = input.parts[index];
    std::string seed = fresh("stream_seed");
    line(reductionType + " " + seed +
         " = __riscv_" + std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
         suffix + "m1(" + initial + ", 1);");
    std::string reduced = fresh("stream_reduced");
    line(reductionType + " " + reduced + " = __riscv_" + stem.str() + "_vs_" +
         vectorSuffix(operation.getInput()) + "_" + suffix + "m1(" + part.str() + ", " +
         seed + ", " +
         std::to_string(operation.getInput().getType().getLayout().getVl()) +
         ");");
    std::string scalarValue = fresh("stream_value");
    line(scalarType + " " + scalarValue +
         " = __riscv_" + std::string(floating ? "vfmv_f_s_" : "vmv_x_s_") +
         suffix + "m1_" + suffix + "(" + reduced + ");");
    if (result.kind == Binding::Kind::Scalar)
      result.scalar = std::move(scalarValue);
    else
      result.parts.push_back(std::move(scalarValue));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVContractStep(
    riscv::RVVContractStepOp operation) {
  auto geometry = riscv::ContractStepGeometry::get(operation.getLhs().getType(),
      operation.getRhs().getType(), operation.getResult().getType(),
      operation.getReductionAxis(), operation.getLaneOperand(),
      operation.getLeaf().getInstruction());
  if (!geometry) return fail(operation, "contract step has no typed scalar/vector mapping");
  const bool rhsLane = operation.getLaneOperand() == "rhs";
  const Binding &lane = bindings.lookup(rhsLane ? operation.getRhs() : operation.getLhs());
  const Binding &scalar = bindings.lookup(rhsLane ? operation.getLhs() : operation.getRhs());
  const Binding &accumulator = bindings.lookup(operation.getAccumulator());
  if (lane.kind != Binding::Kind::Vector || accumulator.kind != Binding::Kind::Vector ||
      (scalar.kind != Binding::Kind::Scalar && scalar.kind != Binding::Kind::ScalarTuple) ||
      accumulator.parts.size() != geometry->laneParts.size())
    return fail(operation, "contract step requires explicit numeric bindings");
  Binding result;
  result.kind = Binding::Kind::Vector;
  auto output = operation.getResult();
  Binding active;
  if (operation.getActive()) {
    active = bindings.lookup(operation.getActive());
    if (active.kind != Binding::Kind::Scalar)
      return fail(operation, "contract step lost its explicit active length");
  }
  const std::string intrinsic = geometry->widening ? (geometry->floating ? "vfwmacc_vf_" : "vwmacc_vx_") :
      (geometry->floating ? "vfmacc_vf_" : "vmacc_vx_");
  for (size_t part = 0; part < accumulator.parts.size(); ++part) {
    const int64_t lanePart = geometry->laneParts[part];
    const int64_t scalarPart = geometry->scalarParts[part];
    if (lanePart >= int64_t(lane.parts.size()) ||
        (scalar.kind == Binding::Kind::ScalarTuple && scalarPart >= int64_t(scalar.parts.size())))
      return fail(operation, "contract step binding disagrees with typed part count");
    const auto &repeated = scalar.kind == Binding::Kind::Scalar ? scalar.scalar : scalar.parts[scalarPart];
    auto name = fresh("contract_step");
    line(vectorType(output) + " " + name + " = __riscv_" + intrinsic +
         vectorSuffix(output) + (operation.getActive() ? "_tu(" : "(") + accumulator.parts[part] + ", " + repeated +
         ", " + lane.parts[lanePart] + ", " +
         (operation.getActive() ? active.scalar : partVL(output, part)) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[output] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
