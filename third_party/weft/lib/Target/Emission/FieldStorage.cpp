#include "Emitter.h"

#include <limits>
#include <numeric>

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileFieldView(riscv::FieldViewOp operation) {
  Binding projection = bindings.lookup(operation.getField());
  if (projection.kind != Binding::Kind::Field || projection.field.index)
    return fail(operation, "field view requires its complete selected field binding");
  auto field = fieldFor(projection);
  if (riscv::isContiguousRecordField(operation.getField())) {
    Binding owner = bindings.lookup(projection.field.owner);
    auto type = scalarCType(projection.field.elementType);
    if (!field || !type || owner.kind != Binding::Kind::Memory)
      return fail(operation, "complete record field view requires its emitted memory owner");
    auto descriptor = operation.getResult().getType();
    owner.memory.name = "((" + std::string(descriptor.getAccess() == "read" ? "const " : "") +
                        *type + " *)(" + owner.memory.name + "))";
    owner.memory.encoding = descriptor.getEncoding();
    owner.memory.elementType = projection.field.elementType;
    owner.memory.recordBytes = descriptor.getStorageBits() / 8;
    owner.memory.logicalElements = 1;
    bindings[operation.getResult()] = std::move(owner);
    return mlir::success();
  }
  auto record = recordForSlice(projection.field.owner);
  auto origin = bindings.lookup(operation.getOrigin());
  if (!field || mlir::failed(record) || origin.kind != Binding::Kind::Scalar)
    return fail(operation, "field view has no record address or explicit scalar origin");
  auto type = scalarCType(projection.field.elementType);
  if (!type) return fail(operation, "field view element has no C spelling");
  auto descriptor = operation.getResult().getType();
  Binding result;
  result.kind = Binding::Kind::Memory;
  result.memory.name = "((" + *type + " *)(" + record->recordPointer + " + " +
      std::to_string(field->bitOffset / 8) + "))";
  result.memory.encoding = descriptor.getEncoding();
  result.memory.elementType = projection.field.elementType;
  result.memory.axes.assign(descriptor.getAxisIds().asArrayRef().begin(), descriptor.getAxisIds().asArrayRef().end());
  result.memory.extents.push_back(std::to_string(descriptor.getShape()[0]));
  result.memory.strides.push_back("1");
  result.memory.origins.push_back(origin.scalar);
  result.memory.isConst = descriptor.getAccess() == "read";
  result.memory.recordBytes = descriptor.getStorageBits() / 8;
  result.memory.logicalElements = 1;
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

std::string fieldStorageAddress(const Binding &record, const EncodingField &field,
                               llvm::StringRef byte, llvm::StringRef rowOffset) {
  if (record.interleaveRows == 0)
    return "(" + record.recordPointer + " + (" + byte.str() + "))";
  const int64_t rows = record.interleaveRows;
  const int64_t chunk = field.access.getFieldChunkBytes();
  const int64_t begin = field.bitOffset / 8;
  const std::string relative = "((" + byte.str() + ") - " + std::to_string(begin) + ")";
  const std::string row = "((" + record.interleaveRowOffset + ") + (" + rowOffset.str() + "))";
  return "(" + record.recordPointer + " + (" + rowOffset.str() + ") + " + row +
         " * " + std::to_string(chunk - 1) + " + (" + row + " / " +
         std::to_string(rows) + ") * (" + record.interleaveCohortStrideBytes +
         " - " + std::to_string(rows * chunk) + ") + " + std::to_string(begin * rows) +
         " + (" + relative + " / " + std::to_string(chunk) + ") * " +
         std::to_string(chunk * rows) + " + (" + relative + " % " +
         std::to_string(chunk) + "))";
}

std::optional<std::string> Emitter::scalarFieldValue(
    const Binding &record, const EncodingField &field,
    llvm::StringRef logicalIndex, llvm::StringRef rowOffset) {
  const int64_t width = riscv_internal::logicalBitWidth(field.type);
  if (field.access.getMapping() == "joined") {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(field.type);
    const int64_t group = field.access.getGroupSize();
    const int64_t fields = field.access.getJoinFields();
    const int64_t lowBits = field.access.getJoinLowBits();
    const int64_t role = field.access.getJoinRole();
    const auto order = field.access.getOrder();
    if (!integer || !integer.isUnsigned() || width <= lowBits || width > 8 ||
        lowBits <= 0 || group <= 0 || fields <= 1 || role < 0 || role >= fields ||
        (order != "lo_first" && order != "hi_first")) return std::nullopt;
    const int64_t physicalRole = order == "lo_first" ? role : fields - 1 - role;
    const std::string index = "(" + logicalIndex.str() + ")";
    const std::string tail = "(" + index + " - " + std::to_string(group) + ")";
    auto read = [&](int64_t offset, const std::string &element) {
      return "*(const uint8_t *)" + fieldStorageAddress(record, field,
          "(" + std::to_string(field.bitOffset / 8 + offset) + " + " + element + ")", rowOffset);
    };
    const std::string headByte = read(role * group, index);
    const std::string lowByte = read(fields * group, tail);
    const std::string highByte = read(role * group, tail);
    const std::string head = "(" + headByte + " & " + std::to_string((uint64_t(1) << width) - 1) + ")";
    const std::string low = "((" + lowByte + " >> " + std::to_string(physicalRole * lowBits) +
        ") & " + std::to_string((uint64_t(1) << lowBits) - 1) + ")";
    const std::string high = "((" + highByte + " >> " + std::to_string(width) +
        ") & " + std::to_string((uint64_t(1) << (width - lowBits)) - 1) + ")";
    return "(" + index + " < " + std::to_string(group) + " ? " + head + " : (" +
        low + " | (" + high + " << " + std::to_string(lowBits) + ")))";
  }
  auto fragment = singleStorageFragment(field, logicalIndex, width);
  if (!fragment)
    return std::nullopt;
  const std::string address = fieldStorageAddress(record, field, fragment->byte, rowOffset);
  if (width <= 8) {
    const uint64_t mask = (uint64_t(1) << width) - 1;
    std::string bits = "((*(const uint8_t *)" + address + " >> (" + fragment->shift +
                       ")) & " + std::to_string(mask) + ")";
    auto integer = mlir::dyn_cast<mlir::IntegerType>(field.type);
    if (usesRawFloatStorage(field.type)) return bits;
    if (!integer)
      return std::nullopt;
    if (integer.isSigned()) {
      const uint64_t sign = uint64_t(1) << (width - 1);
      bits = "((int8_t)((" + bits + " ^ " + std::to_string(sign) + ") - " +
             std::to_string(sign) + "))";
    }
    return bits;
  }
  if ((width != 16 && width != 32 && width != 64) || field.access.getMapping() != "natural")
    return std::nullopt;
  auto type = scalarCType(field.type);
  if (!type)
    return std::nullopt;
  const std::string result = fresh("field_scalar");
  line(*type + " " + result + ";");
  if (record.interleaveRows <= 1 || field.access.getFieldChunkBytes() % (width / 8) == 0) {
    int64_t alignment = std::gcd(field.access.getAlignment(), width / 8);
    if (record.recordStrideBytes > 0) alignment = std::gcd(alignment, record.recordStrideBytes);
    if (record.interleaveRows > 1) alignment = std::gcd(alignment, field.access.getFieldChunkBytes());
    if (alignment > 1)
      line("memcpy(&" + result + ", __builtin_assume_aligned(" + address + ", " +
           std::to_string(alignment) + "), sizeof(" + result + "));");
    else
      line("memcpy(&" + result + ", " + address + ", sizeof(" + result + "));");
  } else {
    const std::string bits = fresh("field_bits");
    std::string expression = "0";
    for (int64_t byte = 0; byte < width / 8; ++byte)
      expression += " | ((uint" + std::to_string(width) + "_t)*(const uint8_t *)" +
                    fieldStorageAddress(record, field, "(" + fragment->byte + ") + " +
                                        std::to_string(byte), rowOffset) +
                    " << " + std::to_string(byte * 8) + ")";
    line("uint" + std::to_string(width) + "_t " + bits + " = " + expression + ";");
    line("memcpy(&" + result + ", &" + bits + ", sizeof(" + result + "));");
  }
  return result;
}

mlir::FailureOr<Binding> Emitter::loadScalarField(mlir::Value value,
                                                const Binding &projection) {
  auto reject = [&](llvm::StringRef reason) -> mlir::FailureOr<Binding> {
    value.getDefiningOp()->emitError(reason);
    return mlir::failure();
  };
  Binding owner = bindings.lookup(projection.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(projection.field.owner);
    if (mlir::failed(record)) return mlir::failure();
    owner = std::move(*record);
  }
  auto field = fieldFor(projection);
  auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
  auto shaped = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto kind = selectedBindingKind(value);
  if (!kind || (*kind != Binding::Kind::Scalar && *kind != Binding::Kind::ScalarTuple) ||
      owner.kind != Binding::Kind::Record || !field || !type || field->bitOffset % 8)
    return reject("scalar field read requires a selected carrier and byte-addressable record");
  const int64_t count = shaped ? scalarPartCount(value) : 1;
  if (count <= 0) return reject("scalar field read has no physical parts");
  const bool regular = projection.field.regularRepeat > 0;
  if (regular && projection.field.index)
    return reject("scalar field read has conflicting regular and indexed address contracts");
  llvm::SmallVector<int64_t> axes, fieldAxes, indexAxes;
  if (shaped) axes.assign(shaped.getAxisIds().asArrayRef().begin(),
                          shaped.getAxisIds().asArrayRef().end());
  const int64_t rank = projection.field.logicalRank;
  if (!projection.field.index && shaped && rank > 0) {
    if (rank > int64_t(axes.size()) || rank > int64_t(projection.field.shape.size()))
      return reject("scalar field read lost its logical field axes");
    fieldAxes.append(axes.end() - rank, axes.end());
  }
  auto addIndexAxes = [&](mlir::Value index) {
    if (auto t = mlir::dyn_cast<riscv::ValueType>(index.getType()))
      for (int64_t axis : t.getAxisIds().asArrayRef())
        if (!llvm::is_contained(indexAxes, axis)) indexAxes.push_back(axis);
  };
  if (projection.field.index) addIndexAxes(*projection.field.index);
  for (mlir::Value relative : projection.field.relativeIndices) addIndexAxes(relative);
  auto indexFor = [&](mlir::Value indexValue, int64_t part) -> std::optional<std::string> {
    const Binding &index = bindings.lookup(indexValue);
    if (index.kind == Binding::Kind::Scalar) return index.scalar;
    if (index.kind != Binding::Kind::ScalarTuple || !shaped) return std::nullopt;
    auto selected = projectPart(indexValue, value, part);
    if (!selected || *selected >= index.parts.size()) return std::nullopt;
    return index.parts[*selected];
  };
  Binding result;
  result.kind = *kind;
  for (int64_t part = 0; part < count; ++part) {
    auto coordinates = shaped ? scalarCoordinates(value, part)
                              : std::optional<llvm::SmallVector<int64_t, 4>>(llvm::SmallVector<int64_t, 4>());
    if (!coordinates || coordinates->size() != axes.size())
      return reject("scalar field read has no complete time/replica coordinates");
    std::string logicalIndex = regular ? std::to_string(projection.field.regularBase) : "0";
    int64_t pointAxis = 0;
    if (projection.field.index) {
      const Binding &index = bindings.lookup(*projection.field.index);
      if (index.kind == Binding::Kind::Point) {
        if (owner.recordElements <= 0) return reject("field point requires complete record geometry");
        logicalIndex = "(" + index.point.base + " % " + std::to_string(owner.recordElements) + ")";
        if (projection.field.selector == "group_index")
          logicalIndex = "(" + logicalIndex + " / " + std::to_string(index.point.physicalExtent) + ")";
        pointAxis = index.point.axis;
      } else {
        auto selected = indexFor(*projection.field.index, part);
        if (!selected) return reject("field address index does not project onto its physical part");
        logicalIndex = *selected;
      }
    }
    for (mlir::Value relative : projection.field.relativeIndices) {
      auto selected = indexFor(relative, part);
      if (!selected) return reject("relative field index does not project onto its physical part");
      logicalIndex = "(" + logicalIndex + " + " + *selected + ")";
    }
    Binding record = owner;
    int64_t rowOffset = 0, fieldIndex = 0;
    for (size_t position = 0; position < axes.size(); ++position) {
      int64_t axis = axes[position], coordinate = (*coordinates)[position];
      auto fieldAxis = llvm::find(fieldAxes, axis);
      if (fieldAxis != fieldAxes.end()) {
        size_t dimension = fieldAxis - fieldAxes.begin();
        int64_t extent = projection.field.shape[projection.field.shape.size() - rank + dimension];
        if (extent <= 0 || coordinate < 0 || coordinate >= extent ||
            fieldIndex > (std::numeric_limits<int64_t>::max() - coordinate) / extent)
          return reject("scalar field coordinates exceed the declared field shape");
        fieldIndex = fieldIndex * extent + coordinate;
        continue;
      }
      if (coordinate == 0 || llvm::is_contained(indexAxes, axis)) continue;
      if (owner.interleaveRows > 0 && axis == owner.interleaveRowAxis) {
        rowOffset += coordinate;
        continue;
      }
      auto stride = llvm::find_if(owner.recordByteStrides,
          [&](const auto &entry) { return entry.first == axis; });
      if (stride != owner.recordByteStrides.end()) {
        record.recordPointer = "(" + record.recordPointer + " + " +
            std::to_string(coordinate) + " * (" + stride->second + "))";
      } else if (axis == pointAxis) {
        logicalIndex = "(" + logicalIndex + " + " + std::to_string(coordinate) + ")";
      } else return reject("scalar field axis has no record stride or selected index mapping");
    }
    if (!fieldAxes.empty()) {
      if (regular) {
        __int128 offset = __int128(fieldIndex / projection.field.regularRepeat) *
                          projection.field.regularStride;
        if (offset < std::numeric_limits<int64_t>::min() ||
            offset > std::numeric_limits<int64_t>::max())
          return reject("regular field coordinate overflows its selected stride");
        fieldIndex = static_cast<int64_t>(offset);
      }
      logicalIndex = "(" + logicalIndex + " + " + std::to_string(fieldIndex) + ")";
    }
    auto active = shaped ? partActive(value, part) : std::optional<std::string>("1");
    if (!active) return reject("scalar field has no physical validity mapping");
    const bool guarded = *active != "1";
    std::string name = fresh("field_read");
    if (guarded) {
      line(*type + " " + name + " = (" + *type + ")0;");
      line("if (" + *active + ") {");
      ++indent;
    }
    auto expression = scalarFieldValue(record, *field, logicalIndex, std::to_string(rowOffset));
    if (!expression) {
      if (guarded) --indent;
      return reject("selected scalar field has no closed storage spelling");
    }
    line((guarded ? "" : *type + " ") + name + " = " + *expression + ";");
    if (guarded) { --indent; line("}"); }
    if (result.kind == Binding::Kind::Scalar) result.scalar = std::move(name);
    else result.parts.push_back(std::move(name));
  }
  bindings[value] = result;
  return result;
}

mlir::LogicalResult Emitter::compileFieldCommit(
    riscv::StoreOp operation, const Binding &value, const Binding &region) {
  auto plan = operation.getFieldStorePlanAttr();
  auto field = fieldFor(region);
  if (!plan || !field)
    return fail(operation, "encoded field commit has no selected packing plan");
  Binding record = bindings.lookup(region.field.owner);
  if (record.kind == Binding::Kind::Slice) {
    auto owner = recordForSlice(region.field.owner);
    if (mlir::failed(owner))
      return mlir::failure();
    record = std::move(*owner);
  }
  if (record.kind != Binding::Kind::Record || region.field.index)
    return fail(operation, "field store requires its complete selected record-local field");
  auto sourceType = scalarCType(riscv_internal::logicalElement(operation.getValue().getType()));
  if (!sourceType)
    return fail(operation, "field store source has no C element type");
  std::string scalarBits;
  if (plan.getKind() == "scalar_bytes") {
    const std::string scalar = fresh("store_scalar");
    scalarBits = fresh("store_bits");
    line(*sourceType + " " + scalar + " = " + value.scalar + ";");
    line("uint" + std::to_string(plan.getLogicalBits()) + "_t " + scalarBits + ";");
    line("memcpy(&" + scalarBits + ", &" + scalar + ", sizeof(" + scalarBits + "));");
  }
  for (auto [i, offset] : llvm::enumerate(plan.getByteOffsets().asArrayRef())) {
    const std::string address = fieldStorageAddress(record, *field,
        std::to_string(field->bitOffset / 8 + offset));
    if (plan.getKind() == "scalar") {
      if (value.kind != Binding::Kind::Scalar)
        return fail(operation, "scalar field store has no materialized scalar value");
      const std::string scalar = fresh("store_scalar");
      line(*sourceType + " " + scalar + " = " + value.scalar + ";");
      line("memcpy(" + address + ", &" + scalar + ", sizeof(" + scalar + "));");
    } else if (plan.getKind() == "scalar_bytes") {
      line("*(uint8_t *)" + address + " = (uint8_t)(" + scalarBits + " >> " +
           std::to_string(plan.getBitShifts()[i]) + ");");
    } else {
      if (value.kind != Binding::Kind::Vector)
        return fail(operation, "vector field store has no selected RVV values");
      std::string packed;
      std::string suffix = vectorSuffix(operation.getValue());
      const std::string vl = std::to_string(plan.getLanes());
      if (plan.getKind() == "natural") {
        packed = value.parts[plan.getSourceParts()[i]];
      } else {
        auto layout = mlir::cast<riscv::ValueType>(operation.getValue().getType()).getLayout();
        const std::string rawSuffix = "u8" + lmulSpelling(layout.getLmulEighths());
        const std::string rawType = "vuint8" + lmulSpelling(layout.getLmulEighths()) + "_t";
        packed = fresh("packed_field");
        line(rawType + " " + packed + ";");
        for (int64_t layer = 0; layer < plan.getSourcesPerStore(); ++layer) {
          const size_t index = i * plan.getSourcesPerStore() + layer;
          std::string source = value.parts[plan.getSourceParts()[index]];
          if (suffix != rawSuffix)
            source = "__riscv_vreinterpret_v_" + suffix + "_" + rawSuffix + "(" + source + ")";
          source = "__riscv_vand_vx_" + rawSuffix + "(" + source + ", " +
                   std::to_string((1u << plan.getLogicalBits()) - 1) + ", " + vl + ")";
          if (plan.getBitShifts()[index])
            source = "__riscv_vsll_vx_" + rawSuffix + "(" + source + ", " +
                     std::to_string(plan.getBitShifts()[index]) + ", " + vl + ")";
          line(packed + " = " + (layer == 0 ? source : "__riscv_vor_vv_" + rawSuffix +
               "(" + packed + ", " + source + ", " + vl + ")") + ";");
        }
        suffix = rawSuffix;
      }
      const std::string type = plan.getKind() == "layered" ? "uint8_t" : *sourceType;
      line("__riscv_vse" + std::to_string(plan.getStoreBits()) + "_v_" + suffix +
           "((" + type + " *)" + address + ", " + packed + ", " + vl + ");");
    }
  }
  return mlir::success();
}

} // namespace weft::riscv_emission
