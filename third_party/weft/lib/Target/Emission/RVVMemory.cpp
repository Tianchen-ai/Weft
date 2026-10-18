#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compilePointerAtomic(riscv::PointerAtomicOp op) {
  auto base = bindings.lookup(op.getBase());
  auto index = bindings.lookup(op.getOffsets()), mask = bindings.lookup(op.getMask());
  auto value = bindings.lookup(op.getValue()), compare = bindings.lookup(op.getCompare());
  auto type = scalarCType(op.getResult().getType());
  if (!type || base.kind != Binding::Kind::Memory || index.kind != Binding::Kind::Scalar ||
      mask.kind != Binding::Kind::Scalar || value.kind != Binding::Kind::Scalar || compare.kind != Binding::Kind::Scalar ||
      instructionOf(op) != "scalar.pointer-atomic." + op.getKind().str())
    return fail(op, "pointer atomic lost its selected scalar operands");
  std::string order = op.getOrdering() == "relaxed" ? "__ATOMIC_RELAXED" : op.getOrdering() == "release" ? "__ATOMIC_RELEASE"
      : op.getOrdering() == "acquire" ? "__ATOMIC_ACQUIRE" : "__ATOMIC_ACQ_REL";
  std::string failure = op.getOrdering() == "release" || op.getOrdering() == "relaxed" ? "__ATOMIC_RELAXED" : "__ATOMIC_ACQUIRE";
  auto scalar = op.getResult().getType();
  bool floating = mlir::isa<mlir::FloatType>(scalar);
  std::string bits = "uint" + std::to_string(scalar.getIntOrFloatBitWidth()) + "_t";
  std::string storage = fresh("atomic_storage"), pointer = fresh("atomic_address"), old = fresh("atomic_old");
  line("typedef " + (floating ? bits : *type) + " " + storage + " __attribute__((__may_alias__));");
  line(*type + " " + old + " = 0;");
  line("if (" + mask.scalar + ") {");
  line(storage + " *" + pointer + " = ((" + storage + " *)(" + base.memory.name + ")) + (ptrdiff_t)(" + index.scalar + ");");
  std::string argument = value.scalar, expected = compare.scalar;
  if (floating) {
    argument = fresh("atomic_value_bits"); expected = fresh("atomic_compare_bits");
    std::string v = fresh("atomic_value"), c = fresh("atomic_compare");
    line(*type + " " + v + " = " + value.scalar + ", " + c + " = " + compare.scalar + ";");
    line(bits + " " + argument + ", " + expected + ";");
    line("__builtin_memcpy(&" + argument + ", &" + v + ", sizeof(" + v + ")); ");
    line("__builtin_memcpy(&" + expected + ", &" + c + ", sizeof(" + c + ")); ");
  }
  std::string result = floating ? fresh("atomic_old_bits") : old;
  if (floating) line(bits + " " + result + ";");
  auto kind = op.getKind();
  if (kind == "cas") {
    line(result + " = " + expected + ";");
    line("__atomic_compare_exchange_n(" + pointer + ", &" + result + ", " + argument + ", 0, " + order + ", " + failure + ");");
  } else if (kind == "xchg") {
    line(result + " = __atomic_exchange_n(" + pointer + ", " + argument + ", " + order + ");");
  } else if (kind == "add" || kind == "and" || kind == "or" || kind == "xor") {
    line(result + " = __atomic_fetch_" + kind.str() + "(" + pointer + ", " + argument + ", " + order + ");");
  } else {
    line(result + " = __atomic_load_n(" + pointer + ", " + failure + ");");
    line("for (;;) {");
    std::string desired = fresh("atomic_desired");
    if (kind == "fadd") {
      std::string prior = fresh("atomic_prior"), sum = fresh("atomic_sum");
      line(*type + " " + prior + "; __builtin_memcpy(&" + prior + ", &" + result + ", sizeof(" + prior + "));");
      line(*type + " " + sum + " = (" + *type + ")(" + prior + " + " + value.scalar + ");");
      line(bits + " " + desired + "; __builtin_memcpy(&" + desired + ", &" + sum + ", sizeof(" + sum + "));");
    } else {
      bool minimum = kind == "min" || kind == "umin";
      line(*type + " " + desired + " = (" + result + (minimum ? " < " : " > ") + argument + ") ? " + result + " : " + argument + ";");
    }
    line("if (__atomic_compare_exchange_n(" + pointer + ", &" + result + ", " + desired + ", 0, " + order + ", " + failure + ")) break;");
    line("}");
  }
  if (floating) line("__builtin_memcpy(&" + old + ", &" + result + ", sizeof(" + old + "));");
  line("}");
  Binding bound;
  bound.kind = Binding::Kind::Scalar;
  bound.scalar = old;
  bindings[op.getResult()] = std::move(bound);
  return mlir::success();
}

mlir::LogicalResult Emitter::compilePointerMemory(mlir::Operation *operation) {
  bool load = operation->getNumResults() != 0;
  mlir::Value address = operation->getOperand(0), offset = operation->getOperand(1),
              predicate = operation->getOperand(2), data = operation->getOperand(3);
  mlir::Value carrier = load ? operation->getResult(0) : data;
  auto stride = operation->getAttrOfType<mlir::IntegerAttr>("element_stride");
  Binding base = bindings.lookup(address);
  if (base.kind != Binding::Kind::Memory)
    return fail(operation, "pointer memory requires a bound storage origin");
  auto indices = materializeNumeric(offset, bindings.lookup(offset));
  auto masks = materializeNumeric(predicate, bindings.lookup(predicate));
  auto values = materializeNumeric(data, bindings.lookup(data));
  if (mlir::failed(indices) || mlir::failed(masks) || mlir::failed(values)) return mlir::failure();
  auto element = riscv_internal::logicalElement(data.getType());
  auto ctype = scalarCType(element);
  if (!ctype) return fail(operation, "pointer memory has no storage C type");
  // Reinterpreted origins can be observed through both element types in the
  // same program. Preserve the IR alias contract through the C compiler's TBAA.
  std::string storageType = fresh("pointer_storage");
  line("typedef " + *ctype + " " + storageType + " __attribute__((__may_alias__));");
  std::string pointer = "((" + std::string(load ? "const " : "") + storageType + " *)(" + base.memory.name + "))";
  auto vector = mlir::dyn_cast<riscv::ValueType>(carrier.getType());
  if (!vector || vector.getLayout().getCarrier() == "scalar") {
    int64_t parts = vector ? scalarPartCount(carrier) : 1;
    auto part = [&](const Binding &binding, int64_t position) -> std::optional<std::string> {
      if (binding.kind == Binding::Kind::Scalar) return binding.scalar;
      if (binding.kind == Binding::Kind::ScalarTuple && position < int64_t(binding.parts.size())) return binding.parts[position];
      return std::nullopt;
    };
    Binding result;
    result.kind = vector ? Binding::Kind::ScalarTuple : Binding::Kind::Scalar;
    for (int64_t i = 0; i < parts; ++i) {
      auto index = part(*indices, i), mask = part(*masks, i), value = part(*values, i);
      if (!index || !mask || !value) return fail(operation, "scalar pointer memory lost its part mapping");
      if (stride) {
        auto coordinates = scalarCoordinates(carrier, i);
        if (!coordinates || coordinates->size() != 1) return fail(operation, "affine pointer memory lost its scalar coordinates");
        *index = "(int64_t)((uint64_t)(" + *index + ") + " + std::to_string(uint64_t(coordinates->front()) * uint64_t(stride.getInt())) + "ULL)";
      }
      std::string access = pointer + "[(ptrdiff_t)(" + *index + ")]";
      if (load) {
        std::string name = fresh("pointer_load");
        line(*ctype + " " + name + " = (" + *mask + ") ? " + access + " : " + *value + ";");
        if (vector) result.parts.push_back(name); else result.scalar = name;
      } else line("if (" + *mask + ") " + access + " = " + *value + ";");
    }
    if (load) bindings[operation->getResult(0)] = std::move(result);
    return mlir::success();
  }
  if ((stride ? indices->kind != Binding::Kind::Scalar : indices->kind != Binding::Kind::Vector) ||
      (masks->kind != Binding::Kind::Vector && masks->kind != Binding::Kind::Scalar) ||
      (values->kind != Binding::Kind::Vector && values->kind != Binding::Kind::Scalar))
    return fail(operation, "indexed pointer memory requires selected vector operands");
  const int64_t parts = vectorPartCount(carrier);
  if ((!stride && indices->parts.size() != size_t(parts)) ||
      (masks->kind == Binding::Kind::Vector && masks->parts.size() != size_t(parts)) ||
      (values->kind == Binding::Kind::Vector && values->parts.size() != size_t(parts)))
    return fail(operation, "indexed pointer memory lost its part mapping");
  auto offsetLayout = stride ? vector.getLayout() : mlir::cast<riscv::ValueType>(offset.getType()).getLayout();
  std::string unsignedIndex = "u64" + lmulSpelling(offsetLayout.getLmulEighths());
  std::string ratio = std::to_string(offsetLayout.getSew() * 8 / offsetLayout.getLmulEighths());
  unsigned bytes = std::max<unsigned>(8, riscv_internal::logicalBitWidth(element)) / 8;
  unsigned shift = llvm::Log2_32(bytes);
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    std::string vl = partVL(carrier, part);
    std::string mask = fresh("pointer_mask");
    if (masks->kind == Binding::Kind::Scalar)
      line("vbool" + ratio + "_t " + mask + " = (" + masks->scalar + ") ? __riscv_vmset_m_b" + ratio +
           "(" + vl + ") : __riscv_vmclr_m_b" + ratio + "(" + vl + ");");
    else
      line("vbool" + ratio + "_t " + mask + " = __riscv_vmsne_vx_" + vectorSuffix(predicate) + "_b" + ratio +
           "(" + masks->parts[part] + ", 0, " + vl + ");");
    std::string index, memory = pointer, instruction;
    if (stride) {
      auto geometry = riscv::RegisterCoordinates::get(vector.getLayout());
      auto coordinates = geometry ? geometry->logical(part) : std::nullopt;
      if (!coordinates || coordinates->size() != 1) return fail(operation, "affine pointer memory lost its vector coordinates");
      std::string start = "((uint64_t)(" + indices->scalar + ") + " + std::to_string(uint64_t(coordinates->front()) * uint64_t(stride.getInt())) + "ULL)";
      memory = "((" + std::string(load ? "const " : "") + storageType + " *)((uintptr_t)(" + base.memory.name + ") + " + start + " * " + std::to_string(bytes) + "ULL))";
      instruction = stride.getInt() == 1 ? (load ? "vle" : "vse") : (load ? "vlse" : "vsse");
      instruction += std::to_string(bytes * 8);
      if (stride.getInt() != 1) index = ", (ptrdiff_t)(" + std::to_string(stride.getInt() * int64_t(bytes)) + ")";
    } else {
      std::string name = fresh("pointer_bytes");
      std::string expression = "__riscv_vreinterpret_v_" + vectorSuffix(offset) + "_" + unsignedIndex + "(" + indices->parts[part] + ")";
      if (shift) expression = "__riscv_vsll_vx_" + unsignedIndex + "(" + expression + ", " + std::to_string(shift) + ", " + vl + ")";
      line("vuint64" + lmulSpelling(offsetLayout.getLmulEighths()) + "_t " + name + " = " + expression + ";");
      index = ", " + name;
      instruction = load ? "vluxei64" : "vsuxei64";
    }
    if (load) {
      std::string loaded = fresh("pointer_loaded");
      std::string fill = values->kind == Binding::Kind::Scalar
          ? "__riscv_" + std::string(usesFloatingRegisters(element) ? "vfmv_v_f_" : "vmv_v_x_") + vectorSuffix(carrier) +
            "(" + values->scalar + ", " + vl + ")" : values->parts[part];
      line(vectorType(carrier) + " " + loaded + " = __riscv_" + instruction + "_v_" + vectorSuffix(carrier) + "_tumu(" + mask + ", " +
           fill + ", " + memory + index + ", " + vl + ");");
      result.parts.push_back(loaded);
    } else {
      line("__riscv_" + instruction + "_v_" + vectorSuffix(data) + "_m(" + mask + ", " + memory + index + ", " +
           values->parts[part] + ", " + vl + ");");
    }
  }
  if (load) bindings[operation->getResult(0)] = std::move(result);
  return mlir::success();
}

std::string Emitter::emitBytePayloadLoad(llvm::StringRef lmul,
                                        llvm::StringRef address,
                                        llvm::StringRef activeBytes,
                                        llvm::StringRef stem) {
  const std::string bytes = fresh(stem);
  line("vuint8" + lmul.str() + "_t " + bytes + " = __riscv_vle8_v_u8" +
       lmul.str() + "((const uint8_t *)(" + address.str() + "), " +
       activeBytes.str() + ");");
  // Preserve the selected byte-count read before its same-register wide
  // reinterpretation. A later numeric predicate cannot become a byte mask.
  line("__asm__(\"\" : \"+vr\"(" + bytes + "));");
  return bytes;
}

mlir::LogicalResult Emitter::compileRVVByteGather(riscv::RVVByteGatherOp operation) {
  if (instructionOf(operation) != "rvv.byte-gather")
    return fail(operation, "byte gather has no selected byte-load instruction");
  auto address = byteReadAddress(operation.getSource());
  auto offsets = materializeNumeric(operation.getByteOffsets(),
                                    bindings.lookup(operation.getByteOffsets()));
  if (!address || mlir::failed(offsets) || offsets->kind != Binding::Kind::Vector)
    return fail(operation, "byte gather requires its selected contiguous storage and RVV offsets");
  const unsigned width = riscv_internal::logicalBitWidth(
      operation.getByteOffsets().getType().getElementType());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto indexPart = projectPart(operation.getByteOffsets(), operation.getResult(), part);
    if (!indexPart || *indexPart >= offsets->parts.size())
      return fail(operation, "byte gather has no exact index-part projection");
    std::string value = fresh("byte_gather");
    line(vectorType(operation.getResult()) + " " + value + " = __riscv_vluxei" +
         std::to_string(width) + "_v_" + vectorSuffix(operation.getResult()) +
         "((const uint8_t *)(" + *address + "), " + offsets->parts[*indexPart] +
         ", " + partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVByteWindowsLoad(riscv::RVVByteWindowsLoadOp operation) {
  if (instructionOf(operation) != "rvv.byte-windows-pack")
    return fail(operation, "byte windows have no exact load-and-pack instruction");
  auto address = byteReadAddress(operation.getSource());
  auto elementType = scalarCType(operation.getResult().getType().getElementType());
  if (!address || !elementType || vectorPartCount(operation.getResult()) != 1)
    return fail(operation, "byte windows require their selected storage and single RVV result");
  const std::string type = vectorType(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  const int64_t width = operation.getWindowBytesAttr().getInt();
  std::string packed;
  for (auto [ordinal, byteBase] : llvm::enumerate(operation.getByteBases())) {
    auto base = materializeNumeric(byteBase, bindings.lookup(byteBase));
    if (mlir::failed(base) || base->kind != Binding::Kind::Scalar)
      return fail(operation, "byte window base is not a materialized scalar value");
    std::string loaded = fresh("byte_window");
    line(type + " " + loaded + " = __riscv_vle8_v_" + suffix +
         "(((const " + *elementType + " *)(" + *address + ")) + " +
         base->scalar + ", " + std::to_string(width) + ");");
    if (ordinal == 0) {
      packed = std::move(loaded);
    } else {
      std::string next = fresh("byte_window_pack");
      line(type + " " + next + " = __riscv_vslideup_vx_" + suffix + "_tu(" +
           packed + ", " + loaded + ", " + std::to_string(ordinal * width) +
           ", " + std::to_string((ordinal + 1) * width) + ");");
      packed = std::move(next);
    }
  }
  Binding result;
  result.kind = Binding::Kind::Vector;
  result.parts.push_back(std::move(packed));
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVIndexedEntryLoad(
    riscv::RVVIndexedEntryLoadOp operation) {
  const llvm::StringRef realization = instructionOf(operation.getOperation());
  Binding entries = bindings.lookup(operation.getEntryOffsets());
  auto materializedEntries =
      materializeNumeric(operation.getEntryOffsets(), entries);
  if (mlir::failed(materializedEntries))
    return mlir::failure();
  entries = std::move(*materializedEntries);
  if (entries.kind != Binding::Kind::Scalar &&
      entries.kind != Binding::Kind::ScalarTuple &&
      entries.kind != Binding::Kind::Vector)
    return fail(operation,
                "indexed entry load requires scalar, replica-tuple, or RVV byte offsets");

  mlir::Type element = operation.getResult().getType().getElementType();
  auto cType = scalarCType(element);
  const unsigned elementBits = riscv_internal::logicalBitWidth(element);
  if (!cType || !elementBits || elementBits % 8)
    return fail(operation,
                "indexed entry load requires a byte-addressable payload element");

  Binding source = bindings.lookup(operation.getSource());
  std::optional<std::string> descriptorByteBase;
  Binding record;
  std::optional<EncodingField> field;
  if (source.kind == Binding::Kind::Slice) {
    auto address = denseAddress(source.slice, {});
    if (address)
      descriptorByteBase = "((const uint8_t *)(" + *address + "))";
  } else if (source.kind == Binding::Kind::Memory) {
    descriptorByteBase =
        "((const uint8_t *)(" + source.memory.name + "))";
  } else if (source.kind == Binding::Kind::Field) {
    source.field.useAccess = operation.getAccess();
    record = bindings.lookup(source.field.owner);
    if (record.kind == Binding::Kind::Slice) {
      auto materializedRecord = recordForSlice(source.field.owner);
      if (mlir::failed(materializedRecord))
        return mlir::failure();
      record = std::move(*materializedRecord);
    }
    field = fieldFor(source);
    if (record.kind == Binding::Kind::Memory && field &&
        field->access.getMapping() == "natural" && !(field->bitOffset % 8))
      descriptorByteBase =
          "((const uint8_t *)(" + record.memory.name + " + " +
          std::to_string(field->bitOffset / 8) + "))";
  } else {
    return fail(operation,
                "indexed entry load source has no dense descriptor or encoded field binding");
  }
  if (!descriptorByteBase &&
      (record.kind != Binding::Kind::Record || !field ||
       field->access.getMapping() != "natural" || field->bitOffset % 8 ||
       record.recordAxis != operation.getSourceAxis()))
    return fail(operation,
                "indexed entry load field has no byte-aligned record-axis mapping");

  const std::string resultType = vectorType(operation.getResult());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    std::optional<std::string> entry;
    if (entries.kind == Binding::Kind::Scalar) {
      entry = entries.scalar;
    } else if (entries.kind == Binding::Kind::ScalarTuple) {
      auto entryPart = mappedPart(operation.getOperation(), 1, part);
      if (entryPart && *entryPart < entries.parts.size())
        entry = entries.parts[*entryPart];
    }
    if (!entry && entries.kind != Binding::Kind::Vector)
      return fail(operation,
                  "indexed entry load outer axes have no typed byte-offset projection");
    std::string byteBase;
    if (descriptorByteBase) {
      byteBase = *descriptorByteBase;
    } else {
      auto coordinates = registerCoordinates(operation.getResult(), part / streams);
      auto sourceType =
          mlir::dyn_cast<riscv::ValueType>(operation.getSource().getType());
      if (streams <= 0 || !coordinates ||
          coordinates->size() != registerAxes.size() || !sourceType)
        return fail(operation,
                    "indexed entry load field has no register-coordinate mapping");
      std::string recordPointer = "(" + record.recordPointer;
      for (int64_t axis : sourceType.getAxisIds().asArrayRef()) {
        if (axis == operation.getSourceAxis())
          continue;
        auto resultAxis = llvm::find(registerAxes, axis);
        int64_t coordinate = 0;
        if (resultAxis != registerAxes.end())
          coordinate = (*coordinates)[resultAxis - registerAxes.begin()];
        auto stride = llvm::find_if(record.recordByteStrides,
                                    [&](const auto &entry) {
                                      return entry.first == axis;
                                    });
        if (stride == record.recordByteStrides.end())
          return fail(operation,
                      "indexed entry load retained source axis has no byte stride");
        recordPointer += " + " + std::to_string(coordinate) + " * " +
                         stride->second;
      }
      recordPointer += " + " + std::to_string(field->bitOffset / 8) + ")";
      byteBase = "((const uint8_t *)(" + recordPointer + "))";
    }
    const std::string base =
        "((const " + *cType + " *)(" + byteBase + "))";
    if (entries.kind == Binding::Kind::Vector) {
      if (streams <= 0)
        return fail(operation,
                    "indexed entry gather has no result stream mapping");
      auto entryPart = projectPart(operation.getEntryOffsets(),
                                   operation.getResult(), part);
      if (!entryPart || *entryPart >= entries.parts.size())
        return fail(operation,
                    "indexed entry gather free axes have no typed entry projection");
      auto entryType = mlir::dyn_cast<riscv::ValueType>(
          operation.getEntryOffsets().getType());
      if (!entryType || entryType.getLayout().getCarrier() != "rvv")
        return fail(operation,
                    "indexed entry gather requires typed RVV byte offsets");
      auto entryInteger =
          mlir::dyn_cast<mlir::IntegerType>(entryType.getElementType());
      if (!entryInteger || entryInteger.isSigned() ||
          (entryInteger.getWidth() != 16 && entryInteger.getWidth() != 32 &&
           entryInteger.getWidth() != 64))
        return fail(operation,
                    "indexed entry gather requires u16, u32, or u64 byte offsets");
      const std::string entryVL =
          partVL(operation.getEntryOffsets(), *entryPart);

      const unsigned packedBits =
          elementBits * static_cast<unsigned>(operation.getPayloadExtent());
      if (packedBits != 32 && packedBits != 64)
        return fail(operation,
                    "indexed entry gather has no exact packed word width");
      const std::string lmul =
          lmulSpelling(operation.getResult().getType().getLayout().getLmulEighths());
      const std::string packedSuffix =
          "u" + std::to_string(packedBits) + lmul;
      const std::string packedType =
          "vuint" + std::to_string(packedBits) + lmul + "_t";
      std::string packed = fresh("entry_gather");
      line(packedType + " " + packed + " = __riscv_vluxei" +
           std::to_string(entryInteger.getWidth()) + "_v_" + packedSuffix +
           "((const uint" + std::to_string(packedBits) + "_t *)(" + base +
           "), " + entries.parts[*entryPart] + ", " + entryVL + ");");
      const std::string unsignedResultSuffix =
          "u" + std::to_string(elementBits) + lmul;
      std::string unpacked = "__riscv_vreinterpret_v_" + packedSuffix + "_" +
                             unsignedResultSuffix + "(" + packed + ")";
      if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
          integer && integer.isSigned())
        unpacked = "__riscv_vreinterpret_v_" + unsignedResultSuffix + "_" +
                   resultSuffix + "(" + unpacked + ")";
      std::string loaded = fresh("entry_payload");
      line(resultType + " " + loaded + " = " + unpacked + ";");
      result.parts.push_back(std::move(loaded));
      continue;
    }
    const std::string bytePointer =
        "((const uint8_t *)(" + byteBase + ") + (size_t)(" + *entry + "))";
    const std::string pointer =
        "((const " + *cType + " *)(" + bytePointer + "))";
    const std::string vl = partVL(operation.getResult(), part);
    std::string loaded = fresh("entry_payload");
    if (realization == "rvv.indexed-entry-byte-load") {
      const int64_t lmulEighths =
          operation.getResult().getType().getLayout().getLmulEighths();
      const std::string rawLMUL = lmulSpelling(lmulEighths);
      if (rawLMUL.empty() || elementBits <= 8)
        return fail(operation,
                    "indexed entry byte load has no legal raw-byte carrier");
      const std::string rawSuffix = "u8" + rawLMUL;
      // Reinterpretation preserves one physical register group.  The byte
      // load therefore keeps the selected LMUL and scales VL by the payload
      // width instead of selecting a second carrier here.
      const std::string bytes = emitBytePayloadLoad(
          rawLMUL, bytePointer, "(" + vl + ") * " +
              std::to_string(elementBits / 8), "entry_payload_bytes");
      const std::string storageSuffix =
          "u" + std::to_string(elementBits) + rawLMUL;
      std::string payload = "__riscv_vreinterpret_v_" + rawSuffix + "_" +
                            storageSuffix + "(" + bytes + ")";
      if (storageSuffix != resultSuffix)
        payload = "__riscv_vreinterpret_v_" + storageSuffix + "_" +
                  resultSuffix + "(" + payload + ")";
      line(resultType + " " + loaded + " = " + payload + ";");
    } else if (realization == "rvv.indexed-entry-load") {
      line(resultType + " " + loaded + " = __riscv_vle" +
           std::to_string(elementBits) + "_v_" + resultSuffix + "(" +
           pointer + ", " + vl + ");");
    } else {
      return fail(operation,
                  "scalar indexed entry load has no exact selected unit leaf");
    }
    result.parts.push_back(std::move(loaded));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVUnitEntryWindowLoad(
    riscv::RVVUnitEntryWindowLoadOp operation) {
  Binding baseIndex = bindings.lookup(operation.getEntryBase());
  auto materializedBase =
      materializeNumeric(operation.getEntryBase(), baseIndex);
  if (mlir::failed(materializedBase))
    return mlir::failure();
  baseIndex = std::move(*materializedBase);
  if (baseIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "unit entry window requires one selected scalar base index");

  mlir::Type element = operation.getResult().getType().getElementType();
  auto cType = scalarCType(element);
  const unsigned elementBits = riscv_internal::logicalBitWidth(element);
  if (!cType || !elementBits || elementBits % 8)
    return fail(operation,
                "unit entry window requires a byte-addressable payload element");

  Binding source = bindings.lookup(operation.getSource());
  if (source.kind != Binding::Kind::Field)
    return fail(operation,
                "unit entry window source has no encoded field binding");
  source.field.useAccess = operation.getAccess();
  Binding record = bindings.lookup(source.field.owner);
  if (record.kind == Binding::Kind::Slice) {
    auto materializedRecord = recordForSlice(source.field.owner);
    if (mlir::failed(materializedRecord))
      return mlir::failure();
    record = std::move(*materializedRecord);
  }
  auto field = fieldFor(source);
  if (record.kind != Binding::Kind::Record || !field ||
      field->access.getMapping() != "natural" || field->bitOffset % 8 ||
      record.recordAxis != operation.getSourceAxis())
    return fail(operation,
                "unit entry window field has no byte-aligned record-axis mapping");

  const std::string resultType = vectorType(operation.getResult());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const int64_t parts = vectorPartCount(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  auto sourceType = operation.getSource().getType();
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || !coordinates ||
        coordinates->size() != registerAxes.size())
      return fail(operation,
                  "unit entry window has no register-coordinate mapping");
    std::string recordPointer = "(" + record.recordPointer;
    for (int64_t axis : sourceType.getAxisIds().asArrayRef()) {
      if (axis == operation.getSourceAxis())
        continue;
      auto resultAxis = llvm::find(registerAxes, axis);
      int64_t coordinate = 0;
      if (resultAxis != registerAxes.end())
        coordinate = (*coordinates)[resultAxis - registerAxes.begin()];
      auto stride = llvm::find_if(record.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == record.recordByteStrides.end())
        return fail(operation,
                    "unit entry window retained source axis has no byte stride");
      recordPointer += " + " + std::to_string(coordinate) + " * " +
                       stride->second;
    }
    recordPointer += " + " + std::to_string(field->bitOffset / 8) + ")";
    const std::string pointer =
        "((const " + *cType + " *)(" + recordPointer + ")) + ((size_t)(" +
        baseIndex.scalar + ") * " + std::to_string(operation.getEntryStride()) +
        ") + " + partOffset(operation.getResult(), part);
    const std::string vl = partVL(operation.getResult(), part);
    std::string loaded = fresh("entry_window");
    line(resultType + " " + loaded + " = __riscv_vle" +
         std::to_string(elementBits) + "_v_" + resultSuffix + "(" + pointer +
         ", " + vl + ");");
    result.parts.push_back(std::move(loaded));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLookup(riscv::LookupOp operation) {
  llvm::StringRef realization = instructionOf(operation.getOperation());
  Binding indices = bindings.lookup(operation.getIndices());
  mlir::FailureOr<Binding> materializedIndices =
      materializeNumeric(operation.getIndices(), indices);
  if (mlir::failed(materializedIndices))
    return mlir::failure();
  indices = std::move(*materializedIndices);

  if (realization == "rvv.vrgather") {
    Binding table = bindings.lookup(operation.getTable());
    mlir::FailureOr<Binding> materializedTable =
        materializeNumeric(operation.getTable(), table);
    if (mlir::failed(materializedTable))
      return mlir::failure();
    table = std::move(*materializedTable);
    if (table.kind != Binding::Kind::Vector || table.parts.size() != 1 ||
        indices.kind != Binding::Kind::Vector)
      return fail(operation,
                  "RVV register lookup requires one complete table vector and vector indices");
    const int64_t parts = vectorPartCount(operation.getResult());
    const std::string resultSuffix = vectorSuffix(operation.getResult());
    const std::string resultType = vectorType(operation.getResult());
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < parts; ++part) {
      auto indexPart = mappedPart(operation.getOperation(), 1, part);
      if (!indexPart || *indexPart >= indices.parts.size())
        return fail(operation,
                    "register lookup index mapping requires an explicit layout conversion");
      const std::string vl = partVL(operation.getResult(), part);
      std::string expression = "__riscv_vrgather_vv_" + resultSuffix + "(" +
                               table.parts.front() + ", " +
                               indices.parts[*indexPart] + ", " + vl + ")";
      std::string name = fresh("lookup");
      line(resultType + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }

  Binding table = bindings.lookup(operation.getTable());
  if (table.kind != Binding::Kind::Slice && table.kind != Binding::Kind::Memory)
    return fail(operation,
                "selected lookup requires one explicit dense memory descriptor");
  std::optional<std::string> address;
  if (table.kind == Binding::Kind::Slice)
    address = denseAddress(table.slice, {});
  else
    address = table.memory.name;
  if (!address)
    return fail(operation, "selected lookup table has no dense base address");

  mlir::Type tableElement =
      riscv_internal::logicalElement(operation.getTable().getType());
  if (auto encoding = mlir::dyn_cast<kernel::EncodingType>(tableElement);
      encoding && encoding.getKind() == "dense")
    tableElement = riscv_internal::logicalElement(operation.getResult().getType());
  auto tableCType = scalarCType(tableElement);
  const unsigned tableBits = riscv_internal::logicalBitWidth(tableElement);
  if (!tableCType || !tableBits || tableBits % 8)
    return fail(operation,
                "selected lookup requires a byte-addressable numeric table");
  const unsigned tableBytes = tableBits / 8;
  const std::string base = "((const " + *tableCType + " *)(" + *address + "))";

  if (realization == "scalar.lookup") {
    if (indices.kind == Binding::Kind::Scalar) {
      std::string loaded = fresh("lookup_scalar");
      line(*tableCType + " " + loaded + " = " + base + "[(size_t)(" +
           indices.scalar + ")];");
      bindings[operation.getResult()] = scalar(std::move(loaded));
      return mlir::success();
    }
    const int64_t resultParts = scalarPartCount(operation.getResult());
    if (indices.kind != Binding::Kind::ScalarTuple || resultParts <= 1 ||
        static_cast<int64_t>(indices.parts.size()) != resultParts)
      return fail(operation,
                  "scalar lookup requires scalar or replica-tuple unsigned indices");
    Binding result;
    result.kind = Binding::Kind::ScalarTuple;
    for (llvm::StringRef part : indices.parts) {
      std::string loaded = fresh("lookup_scalar");
      line(*tableCType + " " + loaded + " = " + base + "[(size_t)(" +
           part.str() + ")];");
      result.parts.push_back(std::move(loaded));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (realization != "rvv.vluxei")
    return fail(operation, "lookup has an unknown selected realization");
  if (indices.kind != Binding::Kind::Vector)
    return fail(operation,
                "RVV indexed lookup requires vector unsigned indices");

  const int64_t parts = vectorPartCount(operation.getResult());
  auto indexLayout = layoutOf(operation.getIndices());
  if (!indexLayout)
    return fail(operation,
                "RVV indexed lookup has no selected index SEW/LMUL");
  const std::string indexSuffix =
      "u" + std::to_string(indexLayout.getSew()) +
      lmulSpelling(indexLayout.getLmulEighths());
  const std::string resultSuffix = vectorSuffix(operation.getResult());
  const std::string resultType = vectorType(operation.getResult());

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    auto indexPart = mappedPart(operation.getOperation(), 1, part);
    if (!indexPart || *indexPart >= indices.parts.size())
      return fail(operation,
                  "lookup index mapping requires an explicit layout conversion");
    const std::string vl = partVL(operation.getResult(), part);
    std::string byteOffsets = indices.parts[*indexPart];
    if (tableBytes != 1)
      byteOffsets = "__riscv_vmul_vx_" + indexSuffix + "(" + byteOffsets +
                    ", " + std::to_string(tableBytes) + ", " + vl + ")";
    std::string expression =
        "__riscv_vluxei" + std::to_string(indexLayout.getSew()) + "_v_" +
        resultSuffix + "(" + base + ", " + byteOffsets + ", " + vl + ")";
    std::string name = fresh("lookup");
    line(resultType + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVStorageWindow(riscv::RVVStorageWindowOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  if (instruction != "rvv.storage-window.natural" &&
      instruction != "rvv.storage-window.layered")
    return fail(operation, "RVV storage window has no exact selected leaf");
  Binding field = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  Binding offset = bindings.lookup(operation.getLogicalOffset());
  if (field.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point ||
      offset.kind != Binding::Kind::Scalar || field.field.index)
    return fail(operation,
                "RVV storage window requires one unprojected field, point, and scalar offset");
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  field.field.useAccess = operation.getAccess();
  if (instruction == "rvv.storage-window.natural") {
    Binding owner = bindings.lookup(field.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      auto record = recordForSlice(field.field.owner);
      if (mlir::failed(record))
        return mlir::failure();
      owner = std::move(*record);
    }
    auto storage = fieldFor(field);
    auto resultType = operation.getResult().getType();
    auto elementType = storage ? scalarCType(storage->type) : std::nullopt;
    auto axis = llvm::find(resultType.getAxisIds().asArrayRef(),
                           plan.getReductionAxis());
    const int64_t lanes =
        axis == resultType.getAxisIds().asArrayRef().end()
            ? 0
            : resultType.getLayout().getLaneFactors()[static_cast<size_t>(
                  axis - resultType.getAxisIds().asArrayRef().begin())];
    unsigned width = 0;
    if (storage) {
      if (auto integer = mlir::dyn_cast<mlir::IntegerType>(storage->type))
        width = integer.getWidth();
      else if (auto floating = mlir::dyn_cast<mlir::FloatType>(storage->type))
        width = floating.getWidth();
    }
    const int64_t bytes = width / 8;
    const int64_t repeat = plan.getProjectionRepeat();
    const int64_t stride = plan.getProjectionStride();
    if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
        !storage || !elementType ||
        owner.recordElements != plan.getRecordElements() ||
        storage->bitOffset / 8 != plan.getByteOffset() ||
        width != static_cast<unsigned>(plan.getElementBits()) ||
        (plan.getKind() != "unit" && plan.getKind() != "strided" &&
         plan.getKind() != "repeat") ||
        (width != 8 && width != 16 && width != 32) ||
        bytes <= 0 || lanes <= 0 || repeat <= 0 || stride <= 0 ||
        (repeat % lanes && lanes % repeat))
      return fail(operation,
                  "RVV projected storage window has incomplete natural-field geometry");
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    const bool floating = mlir::isa<mlir::FloatType>(storage->type);
    const std::string broadcast =
        "__riscv_" + std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix;
    const int64_t streams = streamPartCount(operation.getResult());
    llvm::SmallVector<int64_t, 4> axes =
        registerAxesFor(operation.getResult());
    Binding loaded;
    loaded.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      auto coordinates =
          registerCoordinates(operation.getResult(), part / streams);
      if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
        return fail(operation,
                    "RVV projected storage window has no register-coordinate mapping");
      std::string record = "(" + owner.recordPointer;
      for (auto [freeAxis, coordinate] : llvm::zip(axes, *coordinates)) {
        auto byteStride = llvm::find_if(owner.recordByteStrides,
                                        [&](const auto &entry) {
                                          return entry.first == freeAxis;
                                        });
        if (byteStride == owner.recordByteStrides.end())
          return fail(operation,
                      "RVV projected storage window free axis has no record stride");
        record += " + " + std::to_string(coordinate) + " * " + byteStride->second;
      }
      record += ")";
      const std::string logical =
          "((" + point.point.base + ") % " +
          std::to_string(plan.getRecordElements()) + " + (" + offset.scalar + "))";
      auto address = [&](llvm::StringRef sourceIndex) {
        return "((const " + *elementType + " *)((const uint8_t *)(" + record +
               ") + " + std::to_string(plan.getByteOffset()) + "))[(" +
               sourceIndex.str() + ")]";
      };
      const std::string vl = partVL(operation.getResult(), part);
      std::string value;
      if (plan.getKind() == "unit" || plan.getKind() == "strided") {
        const std::string sourceIndex =
            std::to_string(plan.getProjectionBase()) + " + (" + logical +
            ") * " + std::to_string(stride);
        std::string name = fresh("projected_window");
        const std::string pointer = "&(" + address(sourceIndex) + ")";
        const std::string expression =
            plan.getKind() == "unit"
                ? "__riscv_vle" + std::to_string(width) + "_v_" + suffix +
                      "(" + pointer + ", " + vl + ")"
                : "__riscv_vlse" + std::to_string(width) + "_v_" + suffix +
                      "(" + pointer + ", (ptrdiff_t)" +
                      std::to_string(stride * bytes) + ", " + vl + ")";
        line(type + " " + name + " = " + expression + ";");
        value = std::move(name);
      } else {
        const int64_t pieceWidth = std::min<int64_t>(repeat, lanes);
        const int64_t pieces = lanes / pieceWidth;
        for (int64_t piece = 0; piece < pieces; ++piece) {
          const std::string sourceIndex =
              std::to_string(plan.getProjectionBase()) + " + ((" + logical +
              " + " + std::to_string(piece * pieceWidth) + ") / " +
              std::to_string(repeat) + ") * " + std::to_string(stride);
          const std::string next =
              broadcast + "(" + address(sourceIndex) + ", " + vl + ")";
          if (piece == 0) {
            value = next;
          } else {
            value = "__riscv_vslideup_vx_" + suffix + "(" + value + ", " +
                    next + ", " + std::to_string(piece * pieceWidth) + ", " + vl +
                    ")";
          }
        }
        std::string name = fresh("projected_repeat");
        line(type + " " + name + " = " + value + ";");
        value = std::move(name);
      }
      loaded.parts.push_back(std::move(value));
    }
    bindings[operation.getResult()] = std::move(loaded);
    return mlir::success();
  }
  Binding owner = bindings.lookup(field.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(field.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto storage = fieldFor(field);
  auto integer = storage
                     ? mlir::dyn_cast<mlir::IntegerType>(storage->type)
                     : mlir::IntegerType();
  auto resultType = operation.getResult().getType();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  const int64_t width = plan.getElementBits();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      owner.recordElements != plan.getRecordElements() || !storage || !integer ||
      integer.isSigned() || integer.getWidth() != width ||
      storage->bitOffset / 8 != plan.getByteOffset() ||
      plan.getKind() != "layered" ||
      width <= 0 || width >= 8 || group <= 0 || layer <= 0 || group % layer ||
      plan.getProjectionBase() != 0 || plan.getProjectionStride() != 1 ||
      plan.getProjectionRepeat() != 1)
    return fail(operation,
                "layered RVV storage window has incomplete typed record geometry");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> axes =
      registerAxesFor(operation.getResult());
  Binding loaded;
  loaded.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || !coordinates || coordinates->size() != axes.size())
      return fail(operation,
                  "layered RVV storage window has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
      auto byteStride = llvm::find_if(owner.recordByteStrides,
                                      [&](const auto &entry) {
                                        return entry.first == axis;
                                      });
      if (byteStride == owner.recordByteStrides.end())
        return fail(operation,
                    "layered RVV storage window free axis has no record stride");
      record += " + " + std::to_string(coordinate) + " * " + byteStride->second;
    }
    record += ")";
    const std::string logical =
        "((" + point.point.base + ") % " +
        std::to_string(plan.getRecordElements()) + " + (" + offset.scalar +
        ") + " + std::to_string(plan.getProjectionBase()) + " + " +
        partOffset(operation.getResult(), part) + ")";
    const std::string within = "((" + logical + ") % " +
                               std::to_string(group) + ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + ((" + logical +
        ") / " + std::to_string(group) + ") * " + std::to_string(layer) +
        " + (" + within + ") % " + std::to_string(layer) + ")";
    const std::string logicalLayer =
        "((" + within + ") / " + std::to_string(layer) + ")";
    const std::string shift =
        "(" + std::to_string(plan.getShiftBase()) + " + (" + logicalLayer +
        ") * " + std::to_string(plan.getShiftStep()) + ")";
    const std::string vl = partVL(operation.getResult(), part);
    std::string raw = fresh("layered_window_raw");
    line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte + "), " + vl + ");");
    std::string decoded = "__riscv_vsrl_vx_" + suffix + "(" + raw + ", " +
                          shift + ", " + vl + ")";
    decoded = "__riscv_vand_vx_" + suffix + "(" + decoded + ", " +
              std::to_string(plan.getMaskValue()) + ", " + vl + ")";
    std::string name = fresh("layered_window");
    line(type + " " + name + " = " + decoded + ";");
    loaded.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(loaded);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLayeredRecordLoad(
    riscv::RVVLayeredRecordLoadOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.layered-record-load")
    return fail(operation,
                "RVV layered record load has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  Binding offset = bindings.lookup(operation.getLogicalOffset());
  if (fieldBinding.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point ||
      offset.kind != Binding::Kind::Scalar || fieldBinding.field.index)
    return fail(operation,
                "RVV layered record load requires one field, point, and scalar offset");
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
  auto resultType = operation.getResult().getType();
  auto layout = resultType.getLayout();
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  const int64_t laneAxis = laneAxisFor(operation.getResult());
  auto laneStride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == laneAxis;
                                  });
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0 ||
      owner.recordElements != plan.getRecordElements() || !field || !integer ||
      integer.isSigned() || field->bitOffset % 8 ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      integer.getWidth() != plan.getElementBits() ||
      plan.getKind() != "layered" || laneAxis <= 0 ||
      laneStride == owner.recordByteStrides.end() || group <= 0 || layer <= 0 ||
      group % layer || layout.getSew() != 8 || plan.getMaskValue() <= 0)
    return fail(operation,
                "RVV layered record load has incomplete record-stride geometry");

  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  const int64_t streams = streamPartCount(operation.getResult());
  llvm::SmallVector<int64_t, 4> registerAxes =
      registerAxesFor(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    auto coordinates =
        registerCoordinates(operation.getResult(), part / streams);
    if (streams <= 0 || !coordinates ||
        coordinates->size() != registerAxes.size())
      return fail(operation,
                  "RVV layered record load has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(registerAxes, *coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV layered record load register axis has no byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += " + (" + partOffset(operation.getResult(), part) + ") * (" +
              laneStride->second + "))";
    const std::string logical =
        "(((" + point.point.base + ") % " +
        std::to_string(plan.getRecordElements()) + ") + (" + offset.scalar +
        ") + " + std::to_string(plan.getProjectionBase()) + ")";
    const std::string within =
        "((" + logical + ") % " + std::to_string(group) + ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + ((" + logical +
        ") / " + std::to_string(group) + ") * " + std::to_string(layer) +
        " + (" + within + ") % " + std::to_string(layer) + ")";
    const std::string shift =
        "(" + std::to_string(plan.getShiftBase()) + " + ((" + within +
        ") / " + std::to_string(layer) + ") * " +
        std::to_string(plan.getShiftStep()) + ")";
    const std::string vl = partVL(operation.getResult(), part);
    std::string raw = fresh("layered_record_raw");
    line(type + " " + raw + " = __riscv_vlse8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte +
         "), (ptrdiff_t)(" + laneStride->second + "), " + vl + ");");
    std::string shifted = fresh("layered_record_shift");
    line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" + raw +
         ", " + shift + ", " + vl + ");");
    std::string decoded = fresh("layered_record_value");
    line(type + " " + decoded + " = __riscv_vand_vx_" + suffix + "(" +
         shifted + ", " + std::to_string(plan.getMaskValue()) + ", " + vl +
         ");");
    result.parts.push_back(std::move(decoded));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLayeredStorageLoad(
    riscv::RVVLayeredStorageLoadOp operation) {
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool scalarPrime =
      instruction == "rvv.layered-storage-load.scalar-prime";
  if (instruction != "rvv.layered-storage-load" && !scalarPrime)
    return fail(operation,
                "RVV layered storage load has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding point = bindings.lookup(operation.getOrigin());
  Binding windowIndex = bindings.lookup(operation.getWindowIndex());
  if (fieldBinding.kind != Binding::Kind::Field ||
      point.kind != Binding::Kind::Point ||
      windowIndex.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV layered storage load requires a field, point, and scalar window index");
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
  auto windowType = operation.getResult().getType();
  auto valueType = windowType.getResultType();
  auto layout = valueType.getLayout();
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  auto reduction = llvm::find(valueType.getAxisIds().asArrayRef(),
                              plan.getReductionAxis());
  const int64_t lanes =
      reduction == valueType.getAxisIds().asArrayRef().end()
          ? 0
          : layout.getLaneFactors()[static_cast<size_t>(
                reduction - valueType.getAxisIds().asArrayRef().begin())];
  const int64_t windowsPerLayer = windowType.getWindowsPerLayer();
  const int64_t replicas = product(layout.getReplicaFactors());
  if (owner.kind != Binding::Kind::Record || !field || !integer ||
      integer.isSigned() || field->bitOffset % 8 || owner.recordElements <= 0 ||
      plan.getKind() != "layered" ||
      owner.recordElements != plan.getRecordElements() ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      integer.getWidth() != plan.getElementBits() ||
      group <= 0 || layer <= 0 || lanes <= 0 || windowsPerLayer <= 0 ||
      replicas <= 0 || layout.getSew() != 8)
    return fail(operation,
                "RVV layered storage load has incomplete record or vector geometry");

  const std::string suffix =
      "u8" + lmulSpelling(layout.getLmulEighths());
  const std::string type =
      "vuint8" + lmulSpelling(layout.getLmulEighths()) + "_t";
  llvm::SmallVector<int64_t, 4> axes;
  for (auto [axis, factor] :
       llvm::zip(valueType.getAxisIds().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef()))
    if (factor > 1)
      axes.push_back(axis);
  const int64_t sourceStreams = streamPartCount(operation.getField());
  Binding result;
  result.kind = Binding::Kind::Window;
  result.windowFamily = "layered-storage";
  for (int64_t replica = 0; replica < replicas; ++replica) {
    llvm::SmallVector<int64_t, 4> allCoordinates(
        layout.getReplicaFactors().size(), 0);
    int64_t remaining = replica;
    for (int64_t position =
             static_cast<int64_t>(layout.getReplicaFactors().size()) - 1;
         position >= 0; --position) {
      const int64_t factor = layout.getReplicaFactors()[position];
      if (factor <= 0)
        return fail(operation,
                    "RVV layered storage load has an invalid replica factor");
      allCoordinates[static_cast<size_t>(position)] = remaining % factor;
      remaining /= factor;
    }
    llvm::SmallVector<int64_t, 4> coordinates;
    for (auto [coordinate, factor] :
         llvm::zip(allCoordinates, layout.getReplicaFactors().asArrayRef()))
      if (factor > 1)
        coordinates.push_back(coordinate);
    if (sourceStreams <= 0 || coordinates.size() != axes.size())
      return fail(operation,
                  "RVV layered storage load has no register-coordinate mapping");
    std::string record = "(" + owner.recordPointer;
    for (auto [axis, coordinate] : llvm::zip(axes, coordinates)) {
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end())
        return fail(operation,
                    "RVV layered storage load register axis has no byte stride");
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += ")";
    const std::string logicalBase =
        "(((" + point.point.base + ") % " +
        std::to_string(plan.getRecordElements()) + ") + " +
        std::to_string(plan.getProjectionBase()) + ")";
    const std::string groupIndex =
        "((" + logicalBase + " / " + std::to_string(group) + ") + (" +
        windowIndex.scalar + " / " + std::to_string(windowsPerLayer) + "))";
    const std::string within =
        "((" + logicalBase + " % " + std::to_string(layer) + ") + (" +
        windowIndex.scalar + " % " + std::to_string(windowsPerLayer) + ") * " +
        std::to_string(lanes) + ")";
    const std::string byte =
        "(" + std::to_string(plan.getByteOffset()) + " + " + groupIndex +
        " * " + std::to_string(layer) + " + " + within + ")";
    if (scalarPrime)
      line("__asm__ volatile(\"lb zero, " + std::to_string(lanes - 1) +
           "(%0)\" : : \"r\"((const uint8_t *)(" + record + " + " +
           byte + ")) : \"memory\");");
    std::string raw = fresh("layered_window_raw");
    line(type + " " + raw + " = __riscv_vle8_v_" + suffix +
         "((const uint8_t *)(" + record + " + " + byte + "), " +
         std::to_string(lanes) + ");");
    result.windowLhs.push_back(std::move(raw));
    result.windowValidity.push_back(std::to_string(lanes));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLayeredStorageDecode(
    riscv::RVVLayeredStorageDecodeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool emitsShift =
      instruction == "rvv.layered-storage-decode.shift" ||
      instruction == "rvv.layered-storage-decode.shift-mask";
  const bool emitsMask =
      instruction == "rvv.layered-storage-decode.mask" ||
      instruction == "rvv.layered-storage-decode.shift-mask";
  if (instruction != "rvv.layered-storage-decode.identity" &&
      instruction != "rvv.layered-storage-decode.mask" &&
      instruction != "rvv.layered-storage-decode.shift" &&
      instruction != "rvv.layered-storage-decode.shift-mask")
    return fail(operation,
                "RVV layered storage decode has no exact selected leaf");
  Binding window = bindings.lookup(operation.getWindow());
  auto load =
      operation.getWindow().getDefiningOp<riscv::RVVLayeredStorageLoadOp>();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      operation.getResult().getType().getElementType());
  if (window.kind != Binding::Kind::Window ||
      window.windowFamily != "layered-storage" || !load || !integer ||
      integer.isSigned() || window.windowLhs.empty())
    return fail(operation,
                "RVV layered storage decode has no typed raw storage window");
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string type = vectorType(operation.getResult());
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (auto [part, raw] : llvm::enumerate(window.windowLhs)) {
    std::string value = raw;
    const std::string vl = part < window.windowValidity.size()
                               ? window.windowValidity[part]
                               : std::to_string(physicalLanes(operation.getResult()));
    if (emitsShift) {
      std::string shifted = fresh("layered_window_shift");
      line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
           value + ", " + std::to_string(operation.getShiftAmount()) + ", " + vl +
           ");");
      value = std::move(shifted);
    }
    if (emitsMask) {
      std::string masked = fresh("layered_window_value");
      line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
           value + ", " + std::to_string(operation.getMaskValue()) + ", " + vl +
           ");");
      value = std::move(masked);
    }
    result.parts.push_back(std::move(value));
  }
  if (result.parts.size() !=
      static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV layered storage decode did not produce every register replica");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVReplicaStorageLoad(
    riscv::RVVReplicaStorageLoadOp operation) {
  riscv::StorageWindowPlanAttr plan = operation.getPlan();
  const bool layered = plan.getKind() == "layered";
  const bool natural = plan.getKind() == "unit";
  const bool strided = plan.getKind() == "strided";
  const bool interleavedNatural = plan.getKind() == "interleaved_natural";
  const bool interleavedJoined = plan.getKind() == "interleaved_joined";
  const bool interleaved = interleavedNatural || interleavedJoined;
  if (!natural && !strided && !layered && !interleaved)
    return fail(operation,
                "RVV replica storage load has no selected storage-window form");
  const std::string expected =
      layered ? "rvv.replica-storage-load.layered"
      : natural ? "rvv.replica-storage-load.natural"
      : strided ? "rvv.replica-storage-load.strided"
      : interleavedNatural
          ? "rvv.replica-storage-load.interleaved-natural"
          : "rvv.replica-storage-load.interleaved-joined";
  const llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool scalarPrime = instruction == expected + ".scalar-prime";
  if (instruction != expected && !scalarPrime)
    return fail(operation,
                "RVV replica storage load has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding logicalBase = bindings.lookup(operation.getLogicalBase());
  if (fieldBinding.kind != Binding::Kind::Field ||
      logicalBase.kind != Binding::Kind::Scalar)
    return fail(operation,
                "RVV replica storage load requires a field and scalar logical base");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto fieldInteger = field ? mlir::dyn_cast<mlir::IntegerType>(field->type)
                            : mlir::IntegerType();
  auto fieldFloat = field ? mlir::dyn_cast<mlir::FloatType>(field->type)
                          : mlir::FloatType();
  const unsigned fieldWidth =
      fieldInteger ? fieldInteger.getWidth() : fieldFloat ? fieldFloat.getWidth() : 0;
  auto resultType = operation.getResult().getType();
  auto layout = resultType.getLayout();
  const int64_t lanes = physicalLanes(operation.getResult());
  const int64_t recordRank = operation.getRecordRank();
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0 ||
      !field || !fieldWidth || field->bitOffset % 8 || lanes <= 0 ||
      owner.recordElements != plan.getRecordElements() ||
      field->bitOffset / 8 != plan.getByteOffset() ||
      static_cast<int64_t>(fieldWidth) != plan.getElementBits() ||
      operation.getWindowOffsets().empty() ||
      recordRank < 0 ||
      recordRank > static_cast<int64_t>(operation.getField().getType().getShape().size()) ||
      operation.getRecordCoordinatesForWindow().size() !=
          operation.getWindowOffsets().size() * static_cast<size_t>(recordRank) ||
      operation.getWindowForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getLayerForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getPhysicalLayerForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getShiftOffsetForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getShiftBaseFactorForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())) ||
      operation.getMaskValueForPart().size() !=
          static_cast<size_t>(vectorPartCount(operation.getResult())))
    return fail(operation,
                "RVV replica storage load has incomplete record or part geometry");

  const std::string vectorSuffixValue = vectorSuffix(operation.getResult());
  const std::string vectorTypeValue = vectorType(operation.getResult());
  const std::string vl = std::to_string(lanes);
  const char category = vectorSuffixValue.empty() ? '\0' : vectorSuffixValue.front();
  const std::string loadSuffix =
      category == '\0'
          ? std::string()
          : std::string(1, category) + std::to_string(layout.getSew()) +
                lmulSpelling(layout.getLmulEighths());
  const std::string loadType =
      category == 'u' ? "vuint" + loadSuffix.substr(1) + "_t"
      : category == 'i' ? "vint" + loadSuffix.substr(1) + "_t"
      : category == 'f' ? "vfloat" + loadSuffix.substr(1) + "_t"
                        : std::string();
  const std::string loadVL = std::to_string(lanes);
  auto recordForWindow = [&](size_t window) -> std::optional<std::string> {
    if (window >= operation.getWindowOffsets().size())
      return std::nullopt;
    std::string record = "(" + owner.recordPointer;
    auto fieldType = operation.getField().getType();
    for (int64_t position = 0; position < recordRank; ++position) {
      const int64_t axis = fieldType.getAxisIds()[static_cast<size_t>(position)];
      const int64_t coordinate = operation.getRecordCoordinatesForWindow()[
          window * static_cast<size_t>(recordRank) + static_cast<size_t>(position)];
      auto stride = llvm::find_if(owner.recordByteStrides,
                                  [&](const auto &entry) {
                                    return entry.first == axis;
                                  });
      if (stride == owner.recordByteStrides.end()) {
        if (coordinate != 0)
          return std::nullopt;
        continue;
      }
      record += " + " + std::to_string(coordinate) + " * " + stride->second;
    }
    record += ")";
    return record;
  };
  llvm::SmallVector<std::string> windows;
  windows.reserve(operation.getWindowOffsets().size());
  if (interleaved) {
    if (owner.interleaveRows <= 0 || layout.getSew() <= 0 ||
        layout.getLmulEighths() <= 0)
      return fail(operation,
                  "interleaved replica storage load has no selected byte-plane geometry");
    const int64_t rawNumerator = layout.getLmulEighths() * 8;
    if (rawNumerator % layout.getSew())
      return fail(operation,
                  "interleaved replica storage load cannot preserve the selected lane count");
    const int64_t rawLMUL = rawNumerator / layout.getSew();
    const std::string rawLMULName = lmulSpelling(rawLMUL);
    if (rawLMULName.empty())
      return fail(operation,
                  "interleaved replica storage load has an illegal raw-byte LMUL");
    const std::string rawSuffix = "u8" + rawLMULName;
    const std::string rawType = "vuint8" + rawLMULName + "_t";
    auto byteLoad = [&](llvm::StringRef record, const std::string &byte,
                        llvm::StringRef stem) {
      std::string loaded = fresh(stem);
      line(rawType + " " + loaded + " = __riscv_vle8_v_" + rawSuffix +
           "((const uint8_t *)(" + record.str() + " + (" + byte + ") * " +
           std::to_string(owner.interleaveRows) + "), " + vl + ");");
      return loaded;
    };
    for (auto [window, offset] :
         llvm::enumerate(operation.getWindowOffsets())) {
      auto record = recordForWindow(window);
      if (!record)
        return fail(operation,
                    "interleaved replica storage window has no typed record coordinate");
      const std::string logical = "((" + logicalBase.scalar + ") + " +
                                  std::to_string(offset) + ")";
      if (interleavedJoined) {
        if (!fieldInteger || !fieldInteger.isUnsigned())
          return fail(operation,
                      "joined interleaved storage requires an unsigned logical field");
        const int64_t group = operation.getAccess().getGroupSize();
        const int64_t fields = operation.getAccess().getJoinFields();
        const int64_t lowBits = operation.getAccess().getJoinLowBits();
        const int64_t role = operation.getAccess().getJoinRole();
        const int64_t physicalRole = operation.getAccess().getOrder() == "lo_first"
                                         ? role
                                         : fields - 1 - role;
        const uint64_t logicalMask = (uint64_t{1} << fieldWidth) - 1;
        const uint64_t lowMask = (uint64_t{1} << lowBits) - 1;
        const uint64_t highMask =
            (uint64_t{1} << (fieldWidth - lowBits)) - 1;
        const std::string tail = "(" + logical + " - " +
                                 std::to_string(group) + ")";
        const std::string headByte =
            std::to_string(plan.getByteOffset() + role * group) + " + " + logical;
        const std::string lowByte =
            std::to_string(plan.getByteOffset() + fields * group) + " + " + tail;
        const std::string highByte =
            std::to_string(plan.getByteOffset() + role * group) + " + " + tail;
        std::string head = byteLoad(*record, headByte, "replica_joined_head");
        head = "__riscv_vand_vx_" + rawSuffix + "(" + head + ", " +
               std::to_string(logicalMask) + ", " + vl + ")";
        std::string low = byteLoad(*record, lowByte, "replica_joined_low");
        low = "__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" +
              rawSuffix + "(" + low + ", " +
              std::to_string(physicalRole * lowBits) + ", " + vl + "), " +
              std::to_string(lowMask) + ", " + vl + ")";
        std::string high = byteLoad(*record, highByte, "replica_joined_high");
        high = "__riscv_vand_vx_" + rawSuffix + "(__riscv_vsrl_vx_" +
               rawSuffix + "(" + high + ", " + std::to_string(fieldWidth) +
               ", " + vl + "), " + std::to_string(highMask) + ", " + vl +
               ")";
        const std::string assembled =
            operation.getAccess().getOrder() == "lo_first"
                ? "__riscv_vor_vv_" + rawSuffix + "(" + low +
                      ", __riscv_vsll_vx_" + rawSuffix + "(" + high + ", " +
                      std::to_string(lowBits) + ", " + vl + "), " + vl + ")"
                : "__riscv_vor_vv_" + rawSuffix + "(" + high +
                      ", __riscv_vsll_vx_" + rawSuffix + "(" + low + ", " +
                      std::to_string(fieldWidth - lowBits) + ", " + vl +
                      "), " + vl + ")";
        std::string value = fresh("replica_joined_value");
        line(rawType + " " + value + " = (" + logical + " < " +
             std::to_string(group) + " ? " + head + " : " + assembled +
             ");");
        windows.push_back(std::move(value));
        continue;
      }

      const unsigned bytes = fieldWidth / 8;
      if (!bytes || (fieldWidth != 8 && fieldWidth != 16 && fieldWidth != 32))
        return fail(operation,
                    "natural interleaved storage requires one byte-addressable logical element");
      const std::string unsignedSuffix =
          "u" + std::to_string(fieldWidth) +
          lmulSpelling(layout.getLmulEighths());
      const std::string unsignedType =
          "vuint" + std::to_string(fieldWidth) +
          lmulSpelling(layout.getLmulEighths()) + "_t";
      std::string assembled;
      for (unsigned byte = 0; byte < bytes; ++byte) {
        const std::string byteIndex =
            std::to_string(plan.getByteOffset() + byte) + " + (" + logical +
            ") * " + std::to_string(bytes);
        std::string loaded =
            byteLoad(*record, byteIndex, "replica_natural_byte");
        std::string widened = loaded;
        if (bytes > 1) {
          widened = "__riscv_vzext_vf" + std::to_string(bytes) + "_" +
                    unsignedSuffix + "(" + loaded + ", " + vl + ")";
          if (byte)
            widened = "__riscv_vsll_vx_" + unsignedSuffix + "(" + widened +
                      ", " + std::to_string(byte * 8) + ", " + vl + ")";
        }
        assembled = assembled.empty()
                        ? widened
                        : "__riscv_vor_vv_" + unsignedSuffix + "(" +
                              assembled + ", " + widened + ", " + vl + ")";
      }
      std::string value = fresh("replica_natural_value");
      if (vectorSuffixValue == unsignedSuffix) {
        line(vectorTypeValue + " " + value + " = " + assembled + ";");
      } else {
        line(vectorTypeValue + " " + value + " = __riscv_vreinterpret_v_" +
             unsignedSuffix + "_" + vectorSuffixValue + "(" + assembled +
             ");");
      }
      windows.push_back(std::move(value));
    }
  } else if (!layered) {
    const unsigned width = fieldInteger.getWidth();
    auto scalarType = scalarCType(field->type);
    if (!scalarType || (width != 8 && width != 16 && width != 32) ||
        loadSuffix.empty() || loadType.empty())
      return fail(operation,
                  "natural replica storage load requires a byte-addressable integer field");
    for (auto [window, offset] :
         llvm::enumerate(operation.getWindowOffsets())) {
      auto record = recordForWindow(window);
      if (!record)
        return fail(operation,
                    "natural replica storage window has no register-coordinate record");
      const std::string index = "((" + logicalBase.scalar + ") + " +
                                std::to_string(offset) + ")";
      const std::string pointer =
          "((const " + *scalarType + " *)((const uint8_t *)(" +
          *record + ") + " + std::to_string(plan.getByteOffset()) +
          ")) + " + index;
      std::string raw = fresh("replica_window");
      if (strided) {
        const int64_t strideBytes =
            plan.getProjectionStride() * static_cast<int64_t>(width / 8);
        line(loadType + " " + raw + " = __riscv_vlse" +
             std::to_string(width) + "_v_" + loadSuffix + "(" + pointer +
             ", " + std::to_string(strideBytes) + ", " + loadVL + ");");
      } else {
        line(loadType + " " + raw + " = __riscv_vle" +
             std::to_string(width) + "_v_" + loadSuffix + "(" + pointer +
             ", " + loadVL + ");");
      }
      windows.push_back(std::move(raw));
    }
  } else {
    const int64_t group = plan.getGroupSize();
    const int64_t layer = plan.getLayerSize();
    const int64_t layers = group / layer;
    if (fieldInteger.isSigned() || group <= 0 || layer <= 0 || group % layer ||
        fieldInteger.getWidth() * layers > 8 || layout.getSew() != 8)
      return fail(operation,
                  "layered replica storage load has invalid packed storage geometry");
    const std::string groupIndex = "((" + logicalBase.scalar + ") / " +
                                   std::to_string(group) + ")";
    const std::string withinLayer = "((" + logicalBase.scalar + ") % " +
                                    std::to_string(layer) + ")";
    if (scalarPrime) {
      auto record = recordForWindow(0);
      if (!record)
        return fail(operation,
                    "scalar-prime replica storage window has no record coordinate");
      const int64_t offset = operation.getWindowOffsets()[0];
      const std::string byte =
          "(" + std::to_string(plan.getByteOffset()) + " + " + groupIndex +
          " * " + std::to_string(layer) + " + " + withinLayer + " + " +
          std::to_string(offset) + ")";
      line("__asm__ volatile(\"lb zero, " + std::to_string(lanes - 1) +
           "(%0)\" : : \"r\"((const uint8_t *)(" + *record + " + " +
           byte + ")) : \"memory\");");
    }
    for (auto [window, offset] :
         llvm::enumerate(operation.getWindowOffsets())) {
      auto record = recordForWindow(window);
      if (!record)
        return fail(operation,
                    "layered replica storage window has no register-coordinate record");
      const std::string byte =
          "(" + std::to_string(plan.getByteOffset()) + " + " + groupIndex +
          " * " + std::to_string(layer) + " + " + withinLayer + " + " +
          std::to_string(offset) + ")";
      std::string raw = fresh("replica_layered_raw");
      line(vectorTypeValue + " " + raw + " = __riscv_vle8_v_" +
           vectorSuffixValue + "((const uint8_t *)(" + *record +
           " + " + byte + "), " + vl + ");");
      windows.push_back(std::move(raw));
    }
  }

  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < operation.getWindowForPart().size(); ++part) {
    const int64_t window = operation.getWindowForPart()[part];
    if (window < 0 || window >= static_cast<int64_t>(windows.size()))
      return fail(operation,
                  "RVV replica storage load part references an absent source window");
    std::string value = windows[static_cast<size_t>(window)];
    if (layered) {
      const int64_t group = plan.getGroupSize();
      const int64_t layer = plan.getLayerSize();
      const int64_t shiftOffset = operation.getShiftOffsetForPart()[part];
      const int64_t shiftBaseFactor =
          operation.getShiftBaseFactorForPart()[part];
      if (shiftOffset != 0 || shiftBaseFactor != 0) {
        std::string shift = std::to_string(shiftOffset);
        if (shiftBaseFactor != 0)
          shift = "(" + shift + " + " + std::to_string(shiftBaseFactor) +
                  " * (((" + logicalBase.scalar + ") % " +
                  std::to_string(group) + ") / " + std::to_string(layer) +
                  "))";
        std::string shifted = fresh("replica_layered_shift");
        line(vectorTypeValue + " " + shifted + " = __riscv_vsrl_vx_" +
             vectorSuffixValue + "(" + value + ", (size_t)(" +
             shift + "), " + vl + ");");
        value = std::move(shifted);
      }
      const int64_t maskValue = operation.getMaskValueForPart()[part];
      if (maskValue != 0) {
        std::string decoded = fresh("replica_layered_value");
        line(vectorTypeValue + " " + decoded + " = __riscv_vand_vx_" +
             vectorSuffixValue + "(" + value + ", " +
             std::to_string(maskValue) + ", " + vl + ");");
        value = std::move(decoded);
      }
    }
    result.parts.push_back(std::move(value));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVSegmentPairLoad(
    riscv::RVVSegmentPairLoadOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.vlseg2")
    return fail(operation, "segment pair load has no exact selected leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  if (fieldBinding.kind != Binding::Kind::Field)
    return fail(operation, "segment pair load requires a field binding");
  fieldBinding.field.useAccess = operation.getAccess();
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  const auto type = operation.getFirst().getType();
  const int64_t lanes = physicalLanes(operation.getFirst());
  const int64_t parts = vectorPartCount(operation.getFirst());
  auto elementType = scalarCType(type.getElementType());
  if (owner.kind != Binding::Kind::Record || !field || !elementType ||
      field->bitOffset % 8 || owner.interleaveRows != 0 || lanes <= 0 ||
      parts <= 0 || type != operation.getSecond().getType() ||
      type.getShape().size() != 1 || type.getShape()[0] / lanes != parts)
    return fail(operation, "segment pair load has incomplete record or part geometry");
  const std::string suffix = vectorSuffix(operation.getFirst());
  const std::string vectorTypeValue = vectorType(operation.getFirst());
  const std::string tupleType =
      vectorTypeValue.substr(0, vectorTypeValue.size() - 2) + "x2_t";
  const std::string tupleSuffix = suffix + "x2";
  Binding first;
  Binding second;
  first.kind = second.kind = Binding::Kind::Vector;
  for (int64_t part = 0; part < parts; ++part) {
    const int64_t offset = operation.getLogicalBase() + part * lanes * 2;
    const std::string pointer =
        "((const " + *elementType + " *)((const uint8_t *)(" +
        owner.recordPointer + ") + " + std::to_string(field->bitOffset / 8) +
        ")) + " + std::to_string(offset);
    const std::string loaded = fresh("segment_pair");
    line(tupleType + " " + loaded + " = __riscv_vlseg2e" +
         std::to_string(type.getLayout().getSew()) + "_v_" + tupleSuffix +
         "(" + pointer + ", " + std::to_string(lanes) + ");");
    for (int64_t segment = 0; segment < 2; ++segment) {
      const std::string value = fresh("segment_value");
      line(vectorTypeValue + " " + value + " = __riscv_vget_v_" +
           tupleSuffix + "_" + suffix + "(" + loaded + ", " +
           std::to_string(segment) + ");");
      (segment == 0 ? first : second).parts.push_back(value);
    }
  }
  bindings[operation.getFirst()] = std::move(first);
  bindings[operation.getSecond()] = std::move(second);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRecordStorageLoad(
    riscv::RVVRecordStorageLoadOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.record-storage-load.strided")
    return fail(operation,
                "record storage load has no exact selected strided leaf");
  Binding fieldBinding = bindings.lookup(operation.getField());
  Binding cohort = bindings.lookup(operation.getCohort());
  if (fieldBinding.kind != Binding::Kind::Field ||
      cohort.kind != Binding::Kind::RecordCohort)
    return fail(operation,
                "record storage load requires one field and record cohort");
  Binding owner = bindings.lookup(fieldBinding.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(fieldBinding.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(fieldBinding);
  auto scalarType = scalarCType(operation.getResult().getType().getElementType());
  const int64_t parts = vectorPartCount(operation.getResult());
  if (owner.kind != Binding::Kind::Record || !field || !scalarType ||
      field->bitOffset % 8 || operation.getRecordByteStride() <= 0 ||
      cohort.cohortWidth != operation.getResult().getType().getLayout().getVl() ||
      parts != 1)
    return fail(operation,
                "record storage load has incomplete field/stride/vector geometry");
  const std::string type = vectorType(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string name = fresh("record_storage");
  const std::string pointer =
      "((const " + *scalarType + " *)(" + owner.recordPointer + " + " +
      std::to_string(field->bitOffset / 8 + operation.getStorageIndex()) + "))";
  const std::string vl = partVL(operation.getResult(), 0);
  line(type + " " + name + " = __riscv_vlse" +
       std::to_string(operation.getStorageWidth()) + "_v_" + suffix + "(" +
       pointer + ", (ptrdiff_t)" +
       std::to_string(operation.getRecordByteStride()) + ", " + vl + ");");
  Binding result;
  result.kind = Binding::Kind::Vector;
  result.parts.push_back(name);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVRecordStorageDecode(
    riscv::RVVRecordStorageDecodeOp operation) {
  llvm::StringRef instruction = instructionOf(operation.getOperation());
  const bool emitsShift =
      instruction == "rvv.record-storage-decode.shift" ||
      instruction == "rvv.record-storage-decode.shift-mask";
  const bool emitsMask =
      instruction == "rvv.record-storage-decode.mask" ||
      instruction == "rvv.record-storage-decode.shift-mask";
  if (instruction != "rvv.record-storage-decode.identity" && !emitsShift &&
      !emitsMask)
    return fail(operation,
                "record storage decode has no exact selected leaf");
  Binding input = bindings.lookup(operation.getInput());
  if (input.kind != Binding::Kind::Vector || input.parts.size() != 1 ||
      vectorPartCount(operation.getResult()) != 1)
    return fail(operation,
                "record storage decode requires one complete raw vector");
  const std::string type = vectorType(operation.getResult());
  const std::string suffix = vectorSuffix(operation.getResult());
  const std::string vl = partVL(operation.getResult(), 0);
  std::string value = input.parts.front();
  if (emitsShift) {
    std::string shifted = fresh("record_shift");
    line(type + " " + shifted + " = __riscv_vsrl_vx_" + suffix + "(" +
         value + ", " + std::to_string(operation.getShiftAmount()) + ", " +
         vl + ");");
    value = std::move(shifted);
  }
  if (emitsMask) {
    std::string masked = fresh("record_mask");
    line(type + " " + masked + " = __riscv_vand_vx_" + suffix + "(" +
         value + ", " + std::to_string(operation.getMaskValue()) + ", " + vl +
         ");");
    value = std::move(masked);
  }
  Binding result;
  result.kind = Binding::Kind::Vector;
  result.parts.push_back(std::move(value));
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRVVRecordStore(riscv::RVVRecordStoreOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.record-store.strided")
    return fail(operation, "record store has no exact selected strided leaf");
  Binding value = bindings.lookup(operation.getValue());
  Binding destination = bindings.lookup(operation.getDestination());
  Binding cohort = bindings.lookup(operation.getCohort());
  auto valueType = operation.getValue().getType();
  auto scalarType = scalarCType(valueType.getElementType());
  const int64_t storageBytes = operation.getDestination().getType().getStorageBits() / 8;
  if (value.kind != Binding::Kind::Vector || value.parts.size() != 1 ||
      destination.kind != Binding::Kind::Memory ||
      cohort.kind != Binding::Kind::RecordCohort || !scalarType ||
      destination.memory.axes.size() != 1 ||
      destination.memory.strides.size() != 1 ||
      destination.memory.origins.size() != 1 || storageBytes <= 0 ||
      vectorPartCount(operation.getValue()) != 1)
    return fail(operation,
                "record store has incomplete dense destination geometry");
  const std::string logical =
      "(" + cohort.point.base + " + " +
      std::to_string(operation.getElementOffset()) + ")";
  const std::string pointer =
      destination.memory.name + " + (" + destination.memory.origins.front() +
      " + " + logical + " * " + destination.memory.strides.front() + ")";
  if (operation.getRecordByteStride() <= 0)
    return fail(operation, "record store requires one positive selected stride");
  line("__riscv_vsse" + std::to_string(valueType.getLayout().getSew()) + "_v_" +
       vectorSuffix(operation.getValue()) + "(" + pointer +
       ", (ptrdiff_t)" + std::to_string(operation.getRecordByteStride()) +
       ", " + value.parts.front() + ", " +
       partVL(operation.getValue(), 0) + ");");
  return mlir::success();
}

} // namespace weft::riscv_emission
