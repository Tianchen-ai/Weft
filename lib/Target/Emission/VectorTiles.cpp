#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Contraction.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "Weft/Dialect/RISCV/IR/DenseSegments.h"

namespace weft::riscv_emission {
namespace {
std::string selectedFullLength(riscv::ValueType type) {
  // The physical program already fixes VL. A fresh hardware VLMAX query is
  // opaque to the system compiler and needlessly changes its VL state between
  // these transfers/FMA leaves and adjacent fixed-width arithmetic.
  return std::to_string(type.getLayout().getVl());
}

std::string tileLength(const riscv::VectorTileGeometry &tile,
                       llvm::ArrayRef<std::string> extents, size_t part) {
  const auto &bound = extents[tile.lanePosition];
  auto start = std::to_string(tile.starts[part][tile.lanePosition]);
  auto width = std::to_string(tile.lanes);
  auto remaining = "(" + bound + " - " + start + ")";
  return "(" + bound + " > " + start + " ? (" + remaining + " < " + width +
      " ? " + remaining + " : " + width + ") : 0)";
}

std::string tileValid(const riscv::VectorTileGeometry &tile,
                      llvm::ArrayRef<std::string> extents, size_t part) {
  std::string valid = "1";
  for (size_t dimension = 0; dimension < extents.size(); ++dimension)
    if (dimension != tile.lanePosition)
      valid += " && " + std::to_string(tile.starts[part][dimension]) +
          " < " + extents[dimension];
  return valid;
}
} // namespace

mlir::FailureOr<llvm::SmallVector<std::string>>
Emitter::tileExtents(mlir::Operation *operation, mlir::ValueRange extents) {
  llvm::SmallVector<std::string> result;
  for (auto extent : extents) {
    const auto &binding = bindings.lookup(extent);
    if (binding.kind != Binding::Kind::Scalar) {
      (void)fail(operation, "vector tile extent has no explicit scalar binding");
      return mlir::failure();
    }
    result.push_back(binding.scalar);
  }
  return result;
}

mlir::LogicalResult Emitter::compileRVVTileLoad(riscv::RVVTileLoadOp operation) {
  if (instructionOf(operation) == "rvv.tile-load.segment" ||
      instructionOf(operation) == "rvv.tile-load.segment.full")
    return compileRVVTileSegmentLoad(operation);
  bool full = instructionOf(operation) == "rvv.tile-load.full";
  if (instructionOf(operation) != "rvv.tile-load" && !full)
    return fail(operation, "vector tile load has no exact selected transfer leaf");
  Binding source = bindings.lookup(operation.getSource());
  if (source.kind == Binding::Kind::Memory) {
    source.kind = Binding::Kind::Slice;
    source.slice.base = operation.getSource();
  }
  auto type = operation.getResult().getType();
  auto tile = riscv::VectorTileGeometry::get(type, operation.getAxis());
  auto extents = tileExtents(operation, operation.getExtents());
  auto storage = scalarCType(type.getElementType());
  if (source.kind != Binding::Kind::Slice || !tile || mlir::failed(extents) || !storage)
    return fail(operation, "vector tile load lost its typed descriptor and part bounds");
  const auto &base = bindings.lookup(source.slice.base);
  auto axis = llvm::find(base.memory.axes, operation.getAxis());
  if (base.kind != Binding::Kind::Memory || base.memory.elementType != type.getElementType() ||
      axis == base.memory.axes.end())
    return fail(operation, "vector tile load lost its exact dense storage stride");
  const auto dimension = axis - base.memory.axes.begin();
  const auto output = operation.getResult();
  const auto suffix = vectorSuffix(output);
  const auto bits = std::to_string(type.getLayout().getSew());
  const bool floating = usesFloatingRegisters(type.getElementType());
  const bool unit = operation.getAccess().getForm() == "unit";
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < tile->starts.size(); ++part) {
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    for (auto [position, id] : llvm::enumerate(type.getAxisIds().asArrayRef()))
      offsets.emplace_back(id, std::to_string(tile->starts[part][position]));
    auto address = denseAddress(source.slice, offsets);
    if (!address) return fail(operation, "vector tile load has no exact dense address");
    auto length = full ? selectedFullLength(type) : fresh("tile_vl");
    if (!full) line("size_t " + length + " = " + tileLength(*tile, *extents, part) + ";");
    auto name = fresh("tile_load");
    if (!full) {
      auto zero = "__riscv_" + std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix +
          "(" + (floating ? "0.0f" : "0") + ", " + length + ")";
      if (type.getElementType().isF16()) {
        auto integer = "u16" + suffix.substr(3);
        zero = "__riscv_vreinterpret_v_" + integer + "_" + suffix +
            "(__riscv_vmv_v_x_" + integer + "(0, " + length + "))";
      }
      line(vectorType(output) + " " + name + " = " + zero + ";");
    }
    std::string expression = "__riscv_" + std::string(unit ? "vle" : "vlse") +
        bits + "_v_" + suffix + "(" + *address + ", ";
    if (!unit) expression += "(ptrdiff_t)(" + base.memory.strides[dimension] +
        " * (ptrdiff_t)sizeof(" + *storage + ")), ";
    expression += length + ")";
    if (full) line(vectorType(output) + " " + name + " = " + expression + ";");
    else line("if (" + tileValid(*tile, *extents, part) + " && " + length + ") " +
              name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  bindings[output] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVTileSegmentLoad(riscv::RVVTileLoadOp operation) {
  const bool full = instructionOf(operation) == "rvv.tile-load.segment.full";
  const auto type = operation.getResult().getType();
  auto parameters = operation.getLeaf().getParameters().asArrayRef();
  if (parameters.size() != 2)
    return fail(operation, "segmented tile load lost its field-axis selection");
  auto segments = riscv::VectorTileSegments::get(type, operation.getAxis(), parameters[0], parameters[1]);
  auto extents = tileExtents(operation, operation.getExtents());
  auto storage = scalarCType(type.getElementType());
  Binding source = bindings.lookup(operation.getSource());
  if (source.kind == Binding::Kind::Memory) {
    source.kind = Binding::Kind::Slice;
    source.slice.base = operation.getSource();
  }
  if (!segments || mlir::failed(extents) || !storage || source.kind != Binding::Kind::Slice)
    return fail(operation, "segmented tile load has no exact dense field binding");
  const auto &base = bindings.lookup(source.slice.base);
  auto lane = llvm::find(base.memory.axes, operation.getAxis());
  if (base.kind != Binding::Kind::Memory || base.memory.elementType != type.getElementType() ||
      lane == base.memory.axes.end())
    return fail(operation, "segmented tile load lost its dense byte stride");
  const auto output = operation.getResult();
  auto suffix = vectorSuffix(output);
  // Segment memory transfers have no floating arithmetic. Spell half tuples
  // with integer carriers: Clang's half tuple extract/create APIs require full
  // Zvfh even though the e16 memory instruction and bit reinterpret need only
  // the already-admitted half storage capability.
  const bool half = type.getElementType().isF16();
  auto transferSuffix = half ? "u16" + suffix.substr(3) : suffix;
  auto tupleSuffix = transferSuffix + "x" + std::to_string(segments->fields);
  auto tupleType = half ? "vuint16" + suffix.substr(3) + "_t" : vectorType(output);
  tupleType.resize(tupleType.size() - 2);
  tupleType += "x" + std::to_string(segments->fields) + "_t";
  Binding result;
  result.kind = Binding::Kind::Vector;
  result.parts.resize(segments->tile.parts());
  for (const auto &group : segments->groups) {
    const auto first = group.front();
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef()))
      offsets.emplace_back(axis, std::to_string(segments->tile.starts[first][position]));
    auto address = denseAddress(source.slice, offsets);
    if (!address) return fail(operation, "segmented tile load has no complete address mapping");
    auto length = full ? selectedFullLength(type) : fresh("segment_vl");
    if (!full) line("size_t " + length + " = " + tileLength(segments->tile, *extents, first) + ";");
    for (int64_t part : group) {
      result.parts[part] = fresh("segment_field");
      if (full) continue;
      const bool floating = usesFloatingRegisters(type.getElementType());
      std::string zero = "__riscv_" + std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") +
          suffix + "(" + (floating ? "0.0f" : "0") + ", " + length + ")";
      if (type.getElementType().isF16()) {
        auto integer = "u16" + suffix.substr(3);
        zero = "__riscv_vreinterpret_v_" + integer + "_" + suffix +
            "(__riscv_vmv_v_x_" + integer + "(0, " + length + "))";
      }
      line(vectorType(output) + " " + result.parts[part] + " = " + zero + ";");
    }
    if (!full) {
      line("if (" + tileValid(segments->tile, *extents, first) + " && " + length + ") {");
      ++indent;
    }
    const auto tuple = fresh("segment_load");
    line(tupleType + " " + tuple + " = __riscv_vlsseg" + std::to_string(segments->fields) + "e" +
        std::to_string(type.getLayout().getSew()) + "_v_" + tupleSuffix + "(" +
        (half ? "(const uint16_t *)(" + *address + ")" : *address) +
        ", (ptrdiff_t)(" + base.memory.strides[lane - base.memory.axes.begin()] +
        " * (ptrdiff_t)sizeof(" + *storage + ")), " + length + ");");
    for (auto [field, part] : llvm::enumerate(group)) {
      auto value = "__riscv_vget_v_" + tupleSuffix + "_" + transferSuffix + "(" + tuple +
          ", " + std::to_string(field) + ")";
      if (half) value = "__riscv_vreinterpret_v_" + transferSuffix + "_" + suffix + "(" + value + ")";
      line((full ? vectorType(output) + " " : "") + result.parts[part] + " = " + value + ";");
    }
    if (!full) {
      --indent;
      line("}");
    }
  }
  bindings[output] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVTileSlice(riscv::RVVTileSliceOp operation) {
  const bool split = instructionOf(operation) == "rvv.tile-slice.split";
  if (!split && instructionOf(operation) != "rvv.tile-slice")
    return fail(operation, "vector tile slice has no exact register projection leaf");
  auto inputType = operation.getInput().getType(), resultType = operation.getResult().getType();
  auto projection = riscv::VectorTileProjection::get(inputType, resultType,
      operation.getAxis(), operation.getOffsets());
  const auto &input = bindings.lookup(operation.getInput());
  if (!projection || input.kind != Binding::Kind::Vector)
    return fail(operation, "vector tile slice has no typed vector binding");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (auto [part, offset] : llvm::zip(projection->sourceParts, projection->laneOffsets)) {
    if (part >= int64_t(input.parts.size()))
      return fail(operation, "vector tile slice source part is not bound");
    if (!split) {
      result.parts.push_back(input.parts[part]);
      continue;
    }
    auto name = fresh("tile_slice");
    line(vectorType(operation.getResult()) + " " + name + " = __riscv_vget_v_" +
         vectorSuffix(operation.getInput()) + "_" + vectorSuffix(operation.getResult()) +
         "(" + input.parts[part] + ", " + std::to_string(offset / resultType.getLayout().getVl()) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVTileStore(riscv::RVVTileStoreOp operation) {
  if (instructionOf(operation) == "rvv.tile-store.segment" || instructionOf(operation) == "rvv.tile-store.segment.full")
    return compileRVVTileSegmentStore(operation);
  bool full = instructionOf(operation) == "rvv.tile-store.full";
  if (instructionOf(operation) != "rvv.tile-store" && !full)
    return fail(operation, "vector tile store has no exact selected transfer leaf");
  Binding destination = bindings.lookup(operation.getDestination());
  if (destination.kind == Binding::Kind::Memory) {
    destination.kind = Binding::Kind::Slice;
    destination.slice.base = operation.getDestination();
  }
  const auto &input = bindings.lookup(operation.getInput());
  auto type = operation.getInput().getType();
  auto tile = riscv::VectorTileGeometry::get(type, operation.getAxis());
  auto extents = tileExtents(operation, operation.getExtents());
  auto storage = scalarCType(type.getElementType());
  if (destination.kind != Binding::Kind::Slice || !tile || mlir::failed(extents) || !storage ||
      input.kind != Binding::Kind::Vector || input.parts.size() != tile->starts.size())
    return fail(operation, "vector tile store lost its typed descriptor and part bounds");
  const auto &base = bindings.lookup(destination.slice.base);
  auto axis = llvm::find(base.memory.axes, operation.getAxis());
  if (base.kind != Binding::Kind::Memory || base.memory.elementType != type.getElementType() ||
      axis == base.memory.axes.end())
    return fail(operation, "vector tile store lost its exact dense storage stride");
  const auto dimension = axis - base.memory.axes.begin();
  const bool unit = operation.getAccess().getForm() == "unit";
  for (size_t part = 0; part < tile->starts.size(); ++part) {
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    for (auto [position, id] : llvm::enumerate(type.getAxisIds().asArrayRef()))
      offsets.emplace_back(id, std::to_string(tile->starts[part][position]));
    auto address = denseAddress(destination.slice, offsets);
    if (!address) return fail(operation, "vector tile store has no exact dense address");
    auto length = full ? selectedFullLength(type) : fresh("tile_vl");
    if (!full) line("size_t " + length + " = " + tileLength(*tile, *extents, part) + ";");
    std::string expression = "__riscv_" + std::string(unit ? "vse" : "vsse") +
        std::to_string(type.getLayout().getSew()) + "_v_" + vectorSuffix(operation.getInput()) +
        "(" + *address + ", ";
    if (!unit) expression += "(ptrdiff_t)(" + base.memory.strides[dimension] +
        " * (ptrdiff_t)sizeof(" + *storage + ")), ";
    expression += input.parts[part] + ", " + length + ");";
    if (full) line(expression);
    else line("if (" + tileValid(*tile, *extents, part) + " && " + length + ") " + expression);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVTileSegmentStore(riscv::RVVTileStoreOp operation) {
  bool full = instructionOf(operation) == "rvv.tile-store.segment.full";
  auto type = operation.getInput().getType();
  auto parameters = operation.getLeaf().getParameters().asArrayRef();
  if (parameters.size() != 2)
    return fail(operation, "segmented tile store lost its field-axis selection");
  auto segments = riscv::VectorTileSegments::get(type, operation.getAxis(), parameters[0], parameters[1]);
  auto extents = tileExtents(operation, operation.getExtents());
  auto storage = scalarCType(type.getElementType());
  Binding destination = bindings.lookup(operation.getDestination());
  if (destination.kind == Binding::Kind::Memory) {
    destination.kind = Binding::Kind::Slice;
    destination.slice.base = operation.getDestination();
  }
  const auto &input = bindings.lookup(operation.getInput());
  if (!segments || mlir::failed(extents) || !storage || destination.kind != Binding::Kind::Slice ||
      input.kind != Binding::Kind::Vector || input.parts.size() != segments->tile.starts.size())
    return fail(operation, "segmented tile store has no exact tuple binding");
  const auto &base = bindings.lookup(destination.slice.base);
  auto lane = llvm::find(base.memory.axes, operation.getAxis());
  if (base.kind != Binding::Kind::Memory || base.memory.elementType != type.getElementType() ||
      lane == base.memory.axes.end())
    return fail(operation, "segmented tile store lost its dense byte stride");
  auto suffix = vectorSuffix(operation.getInput());
  const bool half = type.getElementType().isF16();
  auto transferSuffix = half ? "u16" + suffix.substr(3) : suffix;
  auto tupleSuffix = transferSuffix + "x" + std::to_string(segments->fields);
  auto tupleType = half ? "vuint16" + suffix.substr(3) + "_t" : vectorType(operation.getInput());
  tupleType.resize(tupleType.size() - 2);
  tupleType += "x" + std::to_string(segments->fields) + "_t";
  for (const auto &group : segments->groups) {
    const auto first = group.front();
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef()))
      offsets.emplace_back(axis, std::to_string(segments->tile.starts[first][position]));
    auto address = denseAddress(destination.slice, offsets);
    if (!address) return fail(operation, "segmented tile store has no complete address mapping");
    auto length = full ? selectedFullLength(type) : fresh("segment_vl");
    if (!full) line("size_t " + length + " = " + tileLength(segments->tile, *extents, first) + ";");
    auto tuple = fresh("segment_tuple");
    std::string create = "__riscv_vcreate_v_" + tupleSuffix + "(";
    for (auto [position, part] : llvm::enumerate(group)) {
      if (position) create += ", ";
      create += half ? "__riscv_vreinterpret_v_" + suffix + "_" + transferSuffix +
          "(" + input.parts[part] + ")" : input.parts[part];
    }
    line(tupleType + " " + tuple + " = " + create + ");");
    std::string store = "__riscv_vssseg" + std::to_string(segments->fields) + "e" +
        std::to_string(type.getLayout().getSew()) + "_v_" + tupleSuffix + "(" +
        (half ? "(uint16_t *)(" + *address + ")" : *address) +
        ", (ptrdiff_t)(" + base.memory.strides[lane - base.memory.axes.begin()] +
        " * (ptrdiff_t)sizeof(" + *storage + ")), " + tuple + ", " + length + ");";
    if (full) line(store);
    else line("if (" + tileValid(segments->tile, *extents, first) + " && " + length + ") " + store);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVDenseSegmentLoad(riscv::RVVDenseSegmentLoadOp operation) {
  return compileDenseSegmentTransfer(operation, operation.getSources(), operation.getValues(),
                                      operation.getExtents(), operation.getAxis(), false);
}

mlir::LogicalResult Emitter::compileRVVDenseSegmentStore(riscv::RVVDenseSegmentStoreOp operation) {
  return compileDenseSegmentTransfer(operation, operation.getDestinations(), operation.getValues(),
                                      operation.getExtents(), operation.getAxis(), true);
}

mlir::LogicalResult Emitter::compileDenseSegmentTransfer(mlir::Operation *operation,
    mlir::ValueRange memory, mlir::ValueRange values, mlir::ValueRange bounds,
    int64_t axis, bool store) {
  auto instruction = instructionOf(operation);
  if (instruction != (store ? "rvv.dense-segment-store" : "rvv.dense-segment-load") || values.empty())
    return fail(operation, "dense field transfer lost its exact selected instruction");
  auto type = mlir::cast<riscv::ValueType>(values.front().getType());
  auto geometry = riscv::DenseSegmentFields::get(memory, type, axis);
  auto extents = tileExtents(operation, bounds);
  auto storage = scalarCType(type.getElementType());
  auto source = bindings.lookup(memory.front());
  if (!geometry || mlir::failed(extents) || !storage || source.kind != Binding::Kind::Slice)
    return fail(operation, "dense field transfer lost its proven sibling descriptor geometry");
  const auto &base = bindings.lookup(source.slice.base);
  auto lane = llvm::find(base.memory.axes, axis);
  if (base.kind != Binding::Kind::Memory || base.memory.elementType != type.getElementType() ||
      lane == base.memory.axes.end())
    return fail(operation, "dense field transfer lost its element storage and lane stride");
  auto suffix = vectorSuffix(values.front());
  const bool half = type.getElementType().isF16();
  auto transferSuffix = half ? "u16" + suffix.substr(3) : suffix;
  auto tupleSuffix = transferSuffix + "x" + std::to_string(values.size());
  auto tupleType = half ? "vuint16" + suffix.substr(3) + "_t" : vectorType(values.front());
  tupleType.resize(tupleType.size() - 2);
  tupleType += "x" + std::to_string(values.size()) + "_t";
  llvm::SmallVector<Binding, 0> fields;
  for (mlir::Value value : values) {
    if (store) {
      auto input = bindings.lookup(value);
      if (input.kind != Binding::Kind::Vector || input.parts.size() != geometry->tile.starts.size())
        return fail(operation, "dense field store lost one complete resident input");
      fields.push_back(std::move(input));
    } else {
      Binding output;
      output.kind = Binding::Kind::Vector;
      output.parts.resize(geometry->tile.parts());
      fields.push_back(std::move(output));
    }
  }
  for (size_t part = 0; part < geometry->tile.starts.size(); ++part) {
    llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
    for (auto [position, id] : llvm::enumerate(type.getAxisIds().asArrayRef()))
      offsets.emplace_back(id, std::to_string(geometry->tile.starts[part][position]));
    auto address = denseAddress(source.slice, offsets);
    if (!address) return fail(operation, "dense field transfer has no exact first-field address");
    const auto length = fresh("field_vl");
    line("size_t " + length + " = " + tileLength(geometry->tile, *extents, part) + ";");
    if (!store) {
      for (auto &field : fields) {
        field.parts[part] = fresh("field_load");
        const bool floating = usesFloatingRegisters(type.getElementType());
        auto zero = "__riscv_" + std::string(floating ? "vfmv_v_f_" : "vmv_v_x_") + suffix +
            "(" + (floating ? "0.0f" : "0") + ", " + length + ")";
        if (half) zero = "__riscv_vreinterpret_v_" + transferSuffix + "_" + suffix +
            "(__riscv_vmv_v_x_" + transferSuffix + "(0, " + length + "))";
        line(vectorType(values.front()) + " " + field.parts[part] + " = " + zero + ";");
      }
    }
    line("if (" + tileValid(geometry->tile, *extents, part) + " && " + length + ") {");
    ++indent;
    const auto tuple = fresh("field_tuple");
    if (store) {
      std::string create = "__riscv_vcreate_v_" + tupleSuffix + "(";
      for (auto [index, field] : llvm::enumerate(fields)) {
        if (index) create += ", ";
        create += half ? "__riscv_vreinterpret_v_" + suffix + "_" + transferSuffix +
            "(" + field.parts[part] + ")" : field.parts[part];
      }
      line(tupleType + " " + tuple + " = " + create + ");");
      // The selected store contract proves complete, non-overlapping records.
      line("__riscv_vsseg" + std::to_string(fields.size()) + "e" +
          std::to_string(type.getLayout().getSew()) + "_v_" + tupleSuffix + "(" +
          (half ? "(uint16_t *)(" + *address + ")" : *address) + ", " + tuple + ", " + length + ");");
    } else {
      line(tupleType + " " + tuple + " = __riscv_vlsseg" + std::to_string(fields.size()) + "e" +
          std::to_string(type.getLayout().getSew()) + "_v_" + tupleSuffix + "(" +
          (half ? "(const uint16_t *)(" + *address + ")" : *address) +
          ", (ptrdiff_t)(" + base.memory.strides[lane - base.memory.axes.begin()] +
          " * (ptrdiff_t)sizeof(" + *storage + ")), " + length + ");");
      for (auto [index, field] : llvm::enumerate(fields)) {
        auto value = "__riscv_vget_v_" + tupleSuffix + "_" + transferSuffix + "(" + tuple +
            ", " + std::to_string(index) + ")";
        if (half) value = "__riscv_vreinterpret_v_" + transferSuffix + "_" + suffix + "(" + value + ")";
        line(field.parts[part] + " = " + value + ";");
      }
    }
    --indent;
    line("}");
  }
  if (!store)
    for (auto [value, field] : llvm::zip(values, fields)) bindings[value] = std::move(field);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVTileContract(riscv::RVVTileContractOp operation) {
  auto type = operation.getResult().getType();
  auto tile = riscv::VectorTileGeometry::get(type, operation.getAxis());
  auto geometry = riscv::ContractStepGeometry::get(operation.getLhs().getType(),
      operation.getRhs().getType(), type, operation.getReductionAxis(), operation.getLaneOperand(),
      riscv::tileContractInstruction(operation.getLeaf().getInstruction()));
  auto extents = tileExtents(operation, operation.getExtents());
  if (!tile || !geometry || mlir::failed(extents))
    return fail(operation, "vector tile contraction lost its selected numeric mapping");
  const bool rhsLane = operation.getLaneOperand() == "rhs";
  const auto &lane = bindings.lookup(rhsLane ? operation.getRhs() : operation.getLhs());
  const auto &scalar = bindings.lookup(rhsLane ? operation.getLhs() : operation.getRhs());
  const auto &accumulator = bindings.lookup(operation.getAccumulator());
  if (lane.kind != Binding::Kind::Vector || accumulator.kind != Binding::Kind::Vector ||
      (scalar.kind != Binding::Kind::Scalar && scalar.kind != Binding::Kind::ScalarTuple) ||
      accumulator.parts.size() != tile->starts.size())
    return fail(operation, "vector tile contraction requires explicit numeric bindings");
  Binding result;
  result.kind = Binding::Kind::Vector;
  auto output = operation.getResult();
  bool full = instructionOf(operation) == "rvv.tile-vfmacc-full.vf" ||
              instructionOf(operation) == "rvv.tile-vfwmacc-full.vf";
  for (size_t part = 0; part < tile->starts.size(); ++part) {
    int64_t lanePart = geometry->laneParts[part];
    int64_t scalarPart = geometry->scalarParts[part];
    if (lanePart >= int64_t(lane.parts.size()) ||
        (scalar.kind == Binding::Kind::ScalarTuple && scalarPart >= int64_t(scalar.parts.size())))
      return fail(operation, "vector tile contraction disagrees with its typed part mapping");
    const auto &repeated = scalar.kind == Binding::Kind::Scalar ? scalar.scalar : scalar.parts[scalarPart];
    auto length = selectedFullLength(mlir::cast<riscv::ValueType>(
        (rhsLane ? operation.getRhs() : operation.getLhs()).getType()));
    if (!full) {
      length = fresh("tile_vl");
      line("size_t " + length + " = (" + tileValid(*tile, *extents, part) + ") ? " +
           tileLength(*tile, *extents, part) + " : 0;");
    }
    auto name = fresh("tile_contract");
    line(vectorType(output) + " " + name + " = __riscv_" +
         std::string(geometry->widening ? "vfwmacc_vf_" : "vfmacc_vf_") +
         vectorSuffix(output) + (full ? "(" : "_tu(") + accumulator.parts[part] + ", " + repeated +
         ", " + lane.parts[lanePart] + ", " + length + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[output] = std::move(result);
  return mlir::success();
}
} // namespace weft::riscv_emission
