#include "Emitter.h"

namespace weft::riscv_emission {

mlir::LogicalResult
Emitter::compilePackedPlaneMerge(riscv::PackedPlaneMergeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "scalar.packed-plane-merge.words")
    return fail(operation, "packed plane merge has no exact selected leaf");
  Binding low = bindings.lookup(operation.getLowField());
  Binding high = bindings.lookup(operation.getHighField());
  if (low.kind != Binding::Kind::Field || high.kind != Binding::Kind::Field ||
      low.field.owner != high.field.owner)
    return fail(operation,
                "packed plane merge requires two fields from one encoded record");
  Binding owner = bindings.lookup(low.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(low.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto lowField = fieldFor(low);
  auto highField = fieldFor(high);
  auto plan = operation.getPlan();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !lowField || !highField || lowField->bitOffset % 8 ||
      highField->bitOffset % 8 ||
      lowField->bitOffset / 8 != plan.getLowByteOffset() ||
      highField->bitOffset / 8 != plan.getHighByteOffset() ||
      streamPartCount(operation.getResult()) != 1)
    return fail(operation,
                "packed plane merge has no closed record and scalar-tuple mapping");

  llvm::StringMap<std::string> loadedWords;
  const int64_t wordAlignment =
      std::min(lowField->access.getAlignment(), highField->access.getAlignment());
  auto loadWord = [&](llvm::StringRef record, int64_t byteOffset) -> std::string {
    std::string key = (record + "#" + llvm::Twine(byteOffset)).str();
    auto found = loadedWords.find(key);
    if (found != loadedWords.end())
      return found->second;
    std::string name = fresh("packed_plane_word");
    if (wordAlignment >= 2 && byteOffset % 2 == 0) {
      line("uint32_t " + name + " = (uint32_t)*(const uint16_t *)(" +
           record.str() + " + " + std::to_string(byteOffset) +
           ") | ((uint32_t)*(const uint16_t *)(" + record.str() + " + " +
           std::to_string(byteOffset + 2) + ") << 16);");
    } else {
      line("uint32_t " + name + " = weft_load_u32_le((const uint8_t *)(" +
           record.str() + " + " + std::to_string(byteOffset) + "));");
    }
    loadedWords[key] = name;
    return name;
  };
  auto repeatedByteMask = [](int64_t bits) -> uint32_t {
    const uint32_t byte = (uint32_t{1} << bits) - 1;
    return byte * uint32_t{0x01010101};
  };

  Binding result;
  result.kind = Binding::Kind::ScalarTuple;
  const int64_t parts = registerPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  auto scaleAxis = llvm::find(axes, plan.getLogicalAxis());
  if (parts <= 0 || scaleAxis == axes.end())
    return fail(operation,
                "packed plane merge has no complete register coordinate mapping");
  const size_t scalePosition =
      static_cast<size_t>(scaleAxis - axes.begin());
  for (int64_t part = 0; part < parts; ++part) {
    auto coordinates = registerCoordinates(operation.getResult(), part);
    if (!coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "packed plane merge cannot decode one result register coordinate");
    const int64_t logical = (*coordinates)[scalePosition];
    if (logical < 0 || logical >= plan.getLogicalElements())
      return fail(operation,
                  "packed plane merge scale coordinate is outside the planned field");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      if (axis == plan.getLogicalAxis() || coordinate == 0)
        continue;
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "packed plane merge replica has no encoded-record stride");
      record += " + " + std::to_string(coordinate) + " * (" +
                stride->second + ")";
    }
    record += ")";
    const int64_t word = logical / 4;
    const int64_t byte = logical % 4;
    const int64_t wordLogical = word * 4;
    const int64_t lowLayer = wordLogical / plan.getLowLayerBytes();
    const int64_t highLayer = wordLogical / plan.getHighLayerBytes();
    const int64_t lowOffset =
        plan.getLowByteOffset() + wordLogical % plan.getLowLayerBytes();
    const int64_t highOffset =
        plan.getHighByteOffset() + wordLogical % plan.getHighLayerBytes();
    const std::string lowWord = loadWord(record, lowOffset);
    const std::string highWord = loadWord(record, highOffset);
    const uint32_t lowMask = repeatedByteMask(plan.getLowBits());
    const uint32_t highMask = repeatedByteMask(plan.getHighBits());
    std::string merged = fresh("packed_plane_merge");
    line("uint32_t " + merged + " = ((" + lowWord + " >> " +
         std::to_string(lowLayer * plan.getLowBits()) + ") & " +
         std::to_string(lowMask) + "u) | (((" + highWord + " >> " +
         std::to_string(highLayer * plan.getHighBits()) + ") & " +
         std::to_string(highMask) + "u) << " +
         std::to_string(plan.getInsertBit()) + ");");
    result.parts.push_back("((uint8_t)((" + merged + " >> " +
                           std::to_string(byte * 8) + ") & 255u))");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVBitplaneMerge(riscv::RVVBitplaneMergeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.bitplane-merge.mask" &&
      instruction != "rvv.bitplane-merge.strided")
    return fail(operation, "RVV bitplane merge has no exact selected leaf");
  mlir::FailureOr<Binding> low = materializeNumeric(
      operation.getLow(), bindings.lookup(operation.getLow()));
  Binding plane = bindings.lookup(operation.getPlane());
  if (mlir::failed(low) || low->kind != Binding::Kind::Vector ||
      (plane.kind != Binding::Kind::Field &&
       plane.kind != Binding::Kind::Mask))
    return fail(operation,
                "RVV bitplane merge requires one vector value and one typed bitplane supply");
  if (plane.kind == Binding::Kind::Mask) {
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    const int64_t parts = vectorPartCount(operation.getResult());
    const int64_t streams = streamPartCount(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < parts; ++part) {
      auto lowPart = mappedPart(operation.getOperation(), 0, part);
      if (!lowPart || *lowPart >= low->parts.size() ||
          part >= static_cast<int64_t>(plane.parts.size()) || streams <= 0)
        return fail(operation,
                    "RVV bitplane merge mask has no typed part mapping");
      const std::string vl = partVL(operation.getResult(), part);
      std::string merged = fresh("bitplane_merge");
      line(type + " " + merged + " = __riscv_vor_vx_" + suffix + "_mu(" +
           plane.parts[part] + ", " + low->parts[*lowPart] + ", " +
           low->parts[*lowPart] + ", " +
           std::to_string(int64_t(1) << operation.getInsertBit()) + ", " + vl +
           ");");
      result.parts.push_back(std::move(merged));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  Binding owner = bindings.lookup(plane.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(plane.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(plane);
  auto planeInteger = field
                          ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                          : mlir::IntegerType();
  const bool logicalPlane =
      planeInteger && planeInteger.isUnsigned() && planeInteger.getWidth() == 1 &&
      field->access.getForm() == "indexed" &&
      field->access.getMapping() == "grouped_layered" &&
      field->access.getGroupSize() == 8 && field->access.getLayerSize() == 1;
  const bool bytePlane =
      planeInteger && planeInteger.isUnsigned() && planeInteger.getWidth() == 8 &&
      field->access.getMapping() == "natural";
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !field || (!logicalPlane && !bytePlane) || field->bitOffset % 8)
    return fail(operation,
                "RVV bitplane merge field has no direct byte-plane address");

  auto layout = operation.getResult().getType().getLayout();
  if (instruction == "rvv.bitplane-merge.strided") {
    if (!bytePlane)
      return fail(operation,
                  "strided RVV bitplane merge requires a byte-plane field");
    auto lowType = operation.getResult().getType();
    auto planeType = operation.getPlane().getType();
    int64_t bitplaneAxis = 0;
    const int64_t laneAxis = laneAxisFor(operation.getResult());
    for (auto [axis, lowExtent, planeExtent] :
         llvm::zip(lowType.getAxisIds().asArrayRef(),
                   lowType.getShape().asArrayRef(),
                   planeType.getShape().asArrayRef()))
      if (lowExtent != planeExtent) {
        if (bitplaneAxis || planeExtent <= 0 || lowExtent != planeExtent * 8)
          return fail(operation,
                      "strided bitplane merge has no unique packed logical axis");
        bitplaneAxis = axis;
      }
    auto laneStride = llvm::find_if(owner.recordByteStrides, [&](const auto &entry) {
      return entry.first == laneAxis;
    });
    auto packed = llvm::find(lowType.getAxisIds().asArrayRef(), bitplaneAxis);
    if (!bitplaneAxis || laneAxis <= 0 || laneAxis == bitplaneAxis ||
        laneStride == owner.recordByteStrides.end() ||
        packed == lowType.getAxisIds().asArrayRef().end())
      return fail(operation,
                  "strided bitplane merge has no lane stride or packed-axis mapping");
    const size_t packedPosition = static_cast<size_t>(
        packed - lowType.getAxisIds().asArrayRef().begin());
    const int64_t streams = streamPartCount(operation.getResult());
    const int64_t parts = vectorPartCount(operation.getResult());
    llvm::SmallVector<int64_t, 4> registerAxes =
        registerAxesFor(operation.getResult());
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < parts; ++part) {
      auto lowPart = mappedPart(operation.getOperation(), 0, part);
      auto coordinates =
          registerCoordinates(operation.getResult(), part / streams);
      if (!lowPart || *lowPart >= low->parts.size() || streams <= 0 ||
          !coordinates || coordinates->size() != registerAxes.size())
        return fail(operation,
                    "strided bitplane merge lacks a typed value/register mapping");
      int64_t stream = part % streams;
      int64_t bitCoordinate = 0;
      for (int64_t dimension =
               static_cast<int64_t>(layout.getAxisIds().size()) - 1;
           dimension >= 0; --dimension) {
        const int64_t factor = layout.getTimeFactors()[dimension];
        if (factor <= 0)
          return fail(operation,
                      "strided bitplane merge has an invalid time factor");
        const int64_t coordinate = stream % factor;
        stream /= factor;
        if (static_cast<size_t>(dimension) == packedPosition)
          bitCoordinate = coordinate;
      }
      if (stream != 0)
        return fail(operation,
                    "strided bitplane merge time coordinates are incomplete");
      std::string record = "(" + owner.recordPointer;
      for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
        auto stride = llvm::find_if(owner.recordByteStrides,
                                    [&](const auto &entry) {
                                      return entry.first == axis;
                                    });
        if (stride == owner.recordByteStrides.end())
          return fail(operation,
                      "strided bitplane register axis has no record-byte stride");
        record += " + " + std::to_string(coordinate) + " * " + stride->second;
      }
      record += " + (" + partOffset(operation.getResult(), part) + ") * " +
                laneStride->second + " + " +
                std::to_string(field->bitOffset / 8 + bitCoordinate / 8) + ")";
      const std::string vl = partVL(operation.getResult(), part);
      std::string loaded = fresh("bitplane_bytes");
      line(type + " " + loaded + " = __riscv_vlse8_v_" + suffix +
           "((const uint8_t *)(" + record + "), (ptrdiff_t)(" +
           laneStride->second + "), " + vl + ");");
      std::string bits = "__riscv_vsrl_vx_" + suffix + "(" + loaded + ", " +
                         std::to_string(bitCoordinate % 8) + ", " + vl + ")";
      bits = "__riscv_vand_vx_" + suffix + "(" + bits + ", 1, " + vl + ")";
      bits = "__riscv_vsll_vx_" + suffix + "(" + bits + ", " +
             std::to_string(operation.getInsertBit()) + ", " + vl + ")";
      std::string merged = fresh("bitplane_merge");
      line(type + " " + merged + " = __riscv_vor_vv_" + suffix + "(" +
           low->parts[*lowPart] + ", " + bits + ", " + vl + ");");
      result.parts.push_back(std::move(merged));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const int64_t maskBits = layout.getSew() * 8 / layout.getLmulEighths();
  if ((maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
       maskBits != 16 && maskBits != 32 && maskBits != 64) ||
      layout.getVl() % 8)
    return fail(operation,
                "RVV bitplane merge selected an illegal mask ratio or byte boundary");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const std::string maskType =
      "vbool" + std::to_string(maskBits) + "_t";
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto lowPart = mappedPart(operation.getOperation(), 0, part);
    auto coordinates = registerCoordinates(operation.getResult(), part / streams);
    auto axes = registerAxesFor(operation.getResult());
    if (!lowPart || *lowPart >= low->parts.size() || !coordinates ||
        coordinates->size() != axes.size())
      return fail(operation,
                  "RVV bitplane merge lacks one typed value-use or register mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides, [&](const auto &entry) {
        return entry.first == axis;
      });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV bitplane merge register axis has no record-byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += " + " + std::to_string(field->bitOffset / 8) + " + (" +
              partOffset(operation.getResult(), part) + " / 8))";
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = fresh("bitplane_mask");
    line(maskType + " " + mask + " = __riscv_vlm_v_b" +
         std::to_string(maskBits) + "((const uint8_t *)(" + record + "), " +
         vl + ");");
    std::string merged = fresh("bitplane_merge");
    line(type + " " + merged + " = __riscv_vor_vx_" + suffix + "_mu(" +
         mask + ", " + low->parts[*lowPart] + ", " + low->parts[*lowPart] +
         ", " + std::to_string(int64_t(1) << operation.getInsertBit()) + ", " +
         vl + ");");
    result.parts.push_back(std::move(merged));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVBitmaskDecode(riscv::RVVBitmaskDecodeOp operation) {
  const bool maskOnly =
      instructionOf(operation.getOperation()) == "rvv.bitmask-decode-mask";
  if (!maskOnly && instructionOf(operation.getOperation()) != "rvv.bitmask-decode")
    return fail(operation, "RVV bitmask decode has no exact selected leaf");
  Binding packed = bindings.lookup(operation.getField());
  Binding origin = bindings.lookup(operation.getOrigin());
  Binding point = bindings.lookup(operation.getPoint());
  if (packed.kind != Binding::Kind::Field ||
      origin.kind != Binding::Kind::Point || point.kind != Binding::Kind::Point)
    return fail(operation,
                "RVV bitmask decode requires one encoded field and two typed points");
  Binding owner = bindings.lookup(packed.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    mlir::FailureOr<Binding> record = recordForSlice(packed.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(packed);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      !field || !integer || !integer.isUnsigned() || integer.getWidth() != 1 ||
      field->access.getMapping() != "grouped_layered" ||
      field->access.getOrder() != "lo_first" || field->bitOffset % 8 ||
      field->access.getGroupSize() != field->access.getLayerSize() * 8)
    return fail(operation,
                "RVV bitmask decode field has no direct contiguous mask address");

  auto layout = operation.getResult().getType().getLayout();
  const int64_t maskBits = layout.getSew() * 8 / layout.getLmulEighths();
  if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
      maskBits != 16 && maskBits != 32 && maskBits != 64)
    return fail(operation, "RVV bitmask decode selected an illegal mask ratio");
  const int64_t group = field->access.getGroupSize();
  const int64_t layer = field->access.getLayerSize();
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const std::string maskType =
      "vbool" + std::to_string(maskBits) + "_t";
  Binding result;
  result.kind = maskOnly ? Binding::Kind::Mask : Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (!coordinates || coordinates->size() != registerAxes.size())
      return fail(operation,
                  "RVV bitmask decode lacks a typed register mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV bitmask decode register axis has no record-byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    const std::string relative =
        "((" + point.point.base + ") - (" + origin.point.base + ") + " +
        partOffset(operation.getResult(), part) + ")";
    record += " + " + std::to_string(field->bitOffset / 8) + " + ((" +
              relative + ") / " + std::to_string(group) + ") * " +
              std::to_string(layer) + ")";
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = fresh("bitmask");
    line(maskType + " " + mask + " = __riscv_vlm_v_b" +
         std::to_string(maskBits) + "((const uint8_t *)(" + record + "), " +
         vl + ");");
    if (maskOnly) {
      result.parts.push_back(std::move(mask));
      continue;
    }
    std::string decoded = fresh("bitmask_decode");
    line(type + " " + decoded + " = __riscv_vmerge_vxm_" + suffix +
         "(__riscv_vmv_v_x_" + suffix + "(0, " + vl + "), 1, " + mask +
         ", " + vl + ");");
    result.parts.push_back(std::move(decoded));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVSignedBitmaskReduce(
    riscv::RVVSignedBitmaskReduceOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.signed-bitmask-reduce.i8-i16")
    return fail(operation,
                "RVV signed-bitmask reduction has no exact selected leaf");
  Binding data = bindings.lookup(operation.getData());
  Binding mask = bindings.lookup(operation.getMask());
  auto materializedData = materializeNumeric(operation.getData(), data);
  if (mlir::failed(materializedData))
    return mlir::failure();
  data = std::move(*materializedData);
  if (data.kind != Binding::Kind::Vector || mask.kind != Binding::Kind::Mask ||
      data.parts.size() != 1 || mask.parts.size() != 1)
    return fail(operation,
                "RVV signed-bitmask reduction requires one vector data part and one materialized mask part");

  auto dataType = operation.getData().getType();
  auto layout = dataType.getLayout();
  const int64_t maskBits = layout.getSew() * 8 / layout.getLmulEighths();
  if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
      maskBits != 16 && maskBits != 32 && maskBits != 64)
    return fail(operation,
                "RVV signed-bitmask reduction selected an illegal mask ratio");
  const std::string vl = partVL(operation.getData(), 0);
  const std::string suffix = vectorSuffix(operation.getData());
  const std::string vectorTypeName = vectorType(operation.getData());
  std::string negated = fresh("signed_bitmask_negated");
  line(vectorTypeName + " " + negated + " = __riscv_vneg_v_" + suffix +
       "(" + data.parts.front() + ", " + vl + ");");
  std::string selected = fresh("signed_bitmask_selected");
  line(vectorTypeName + " " + selected + " = __riscv_vmerge_vvm_" + suffix +
       "(" + negated + ", " + data.parts.front() + ", " +
       mask.parts.front() + ", " + vl + ");");
  std::string zero = fresh("signed_bitmask_zero");
  line("vint16m1_t " + zero +
       " = __riscv_vmv_v_x_i16m1(0, 1);");
  std::string reduced = fresh("signed_bitmask_sum");
  line("vint16m1_t " + reduced + " = __riscv_vwredsum_vs_" + suffix +
       "_i16m1(" + selected + ", " + zero + ", " + vl + ");");
  Binding result;
  result.kind = Binding::Kind::Scalar;
  result.scalar = "((int32_t)__riscv_vmv_x_s_i16m1_i16(" + reduced + "))";
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVMaskedNegate(
    riscv::RVVMaskedNegateOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.masked-negate")
    return fail(operation, "RVV masked negate has no exact selected leaf");
  Binding data = bindings.lookup(operation.getData());
  Binding mask = bindings.lookup(operation.getMask());
  if (data.kind != Binding::Kind::Vector || mask.kind != Binding::Kind::Mask ||
      data.parts.size() != mask.parts.size() ||
      data.parts.size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV masked negate requires matching materialized data and mask parts");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < data.parts.size(); ++part) {
    const std::string vl = partVL(operation.getResult(), part);
    std::string negated = fresh("masked_negate");
    line(type + " " + negated + " = __riscv_vneg_v_" + suffix + "(" +
         data.parts[part] + ", " + vl + ");");
    std::string selected = fresh("masked_negate_selected");
    line(type + " " + selected + " = __riscv_vmerge_vvm_" + suffix + "(" +
         data.parts[part] + ", " + negated + ", " + mask.parts[part] + ", " +
         vl + ");");
    result.parts.push_back(std::move(selected));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVBitmaskWindowLoad(
    riscv::RVVBitmaskWindowLoadOp operation) {
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool directMask = instruction == "rvv.bitmask-window-mask";
  if (instruction != "rvv.bitmask-window-load" && !directMask)
    return fail(operation,
                "RVV bitmask window load has no exact selected leaf");
  Binding byteBase = bindings.lookup(operation.getByteBase());
  auto materializedBase = materializeNumeric(operation.getByteBase(), byteBase);
  if (mlir::failed(materializedBase))
    return mlir::failure();
  byteBase = std::move(*materializedBase);
  if (byteBase.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV bitmask window load requires one scalar byte base");

  Binding packed = bindings.lookup(operation.getField());
  if (packed.kind != Binding::Kind::Field)
    return fail(operation,
                "RVV bitmask window load requires one encoded field");
  packed.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(packed.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto materializedRecord = recordForSlice(packed.field.owner);
    if (mlir::failed(materializedRecord))
      return mlir::failure();
    owner = std::move(*materializedRecord);
  }
  auto field = fieldFor(packed);
  auto integer = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                       : mlir::IntegerType();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      owner.recordAxis != operation.getSourceAxis() || !field || !integer ||
      !integer.isUnsigned() || integer.getWidth() != 1 ||
      field->access.getMapping() != "grouped_layered" ||
      field->access.getOrder() != "lo_first" || field->bitOffset % 8 ||
      field->access.getGroupSize() != field->access.getLayerSize() * 8)
    return fail(operation,
                "RVV bitmask window field has no direct contiguous mask address");

  auto layout = operation.getResult().getType().getLayout();
  const int64_t maskBits = layout.getSew() * 8 / layout.getLmulEighths();
  if (maskBits != 1 && maskBits != 2 && maskBits != 4 && maskBits != 8 &&
      maskBits != 16 && maskBits != 32 && maskBits != 64)
    return fail(operation,
                "RVV bitmask window load selected an illegal mask ratio");
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  auto partBitOffsets = operation.getPartBitOffsets();
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const std::string maskType =
      "vbool" + std::to_string(maskBits) + "_t";
  auto fieldType = operation.getField().getType();
  Binding result;
  result.kind = directMask ? Binding::Kind::Mask : Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || partBitOffsets.size() != static_cast<size_t>(parts) ||
        !coordinates ||
        coordinates->size() != registerAxes.size())
      return fail(operation,
                  "RVV bitmask window load has no register-coordinate mapping");
    std::string recordPointer = "(" + owner.recordPointer;
    for (int64_t axis : fieldType.getAxisIds().asArrayRef()) {
      if (axis == operation.getSourceAxis())
        continue;
      auto resultAxis = llvm::find(registerAxes, axis);
      int64_t coordinate = 0;
      if (resultAxis != registerAxes.end())
        coordinate = (*coordinates)[resultAxis - registerAxes.begin()];
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV bitmask window retained source axis has no byte stride");
      recordPointer += " + " + std::to_string(coordinate) + " * " +
                       stride->second;
    }
    recordPointer += " + " + std::to_string(field->bitOffset / 8) + " + " +
                     byteBase.scalar + " + " +
                     std::to_string(partBitOffsets[part] / 8) + ")";
    const std::string vl = partVL(operation.getResult(), part);
    std::string mask = fresh("bitmask_window");
    line(maskType + " " + mask + " = __riscv_vlm_v_b" +
         std::to_string(maskBits) + "((const uint8_t *)(" + recordPointer +
         "), " + vl + ");");
    if (directMask) {
      result.parts.push_back(std::move(mask));
      continue;
    }
    std::string decoded = fresh("bitmask_window_decode");
    line(type + " " + decoded + " = __riscv_vmerge_vxm_" + suffix +
         "(__riscv_vmv_v_x_" + suffix + "(0, " + vl + "), 1, " + mask +
         ", " + vl + ");");
    result.parts.push_back(std::move(decoded));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
