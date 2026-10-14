#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include <numeric>

namespace weft::riscv_emission {

std::optional<llvm::SmallVector<std::pair<int64_t, std::string>>>
Emitter::partOffsets(mlir::Value value, size_t part) const {
  auto layout = layoutOf(value);
  auto geometry = riscv::RegisterCoordinates::get(layout);
  if (!geometry || part >= static_cast<size_t>(geometry->parts())) return std::nullopt;
  auto coordinates = geometry->logical(part);
  llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
  for (auto [axis, coordinate] : llvm::zip(layout.getAxisIds().asArrayRef(), *coordinates))
    offsets.emplace_back(axis, std::to_string(coordinate));
  return offsets;
}

std::optional<std::string> Emitter::partActive(mlir::Value value, size_t part) const {
  auto offsets = partOffsets(value, part);
  if (!offsets) return std::nullopt;
  std::string active = "1";
  for (const auto &[axis, coordinate] : *offsets) {
    auto scope = axisScopes.find(axis);
    if (scope != axisScopes.end() && !scope->second.empty())
      active += " && " + coordinate + " < (" + scope->second.back().active + ")";
  }
  return active;
}

mlir::LogicalResult Emitter::compileFieldRead(riscv::FieldReadOp operation) {
  auto projection = fieldProjections.find(operation.getInput());
  if (projection == fieldProjections.end())
    return fail(operation, "field read input is not its selected storage projection");
  Binding input;
  input.kind = Binding::Kind::Field;
  input.field = projection->second;
  llvm::SmallVector<mlir::Value> captured;
  riscv::fieldReadProjection(operation.getInput(), &captured);
  auto bindIndex = [&](mlir::Value index) -> std::optional<mlir::Value> {
    auto found = llvm::find(captured, index);
    if (found == captured.end())
      return std::nullopt;
    return operation.getIndices()[static_cast<size_t>(found - captured.begin())];
  };
  if (input.field.index) {
    auto index = bindIndex(*input.field.index);
    if (!index)
      return fail(operation, "field read has no explicit primary address operand");
    input.field.index = *index;
  }
  for (mlir::Value &relative : input.field.relativeIndices) {
    auto index = bindIndex(relative);
    if (!index)
      return fail(operation, "field read has no explicit relative address operand");
    relative = *index;
  }
  input.field.useAccess = operation.getAccess();
  if (operation.getLeaf().getInstruction() == "encoded.field-read.record-element") {
    auto field = fieldFor(input);
    if (!field || !input.field.index || !input.field.relativeIndices.empty())
      return fail(operation, "record element read requires one explicit scalar field coordinate");
    auto index = bindings.lookup(*input.field.index);
    if (index.kind != Binding::Kind::Scalar)
      return fail(operation, "record element read index is not scalar");
    Binding owner = bindings.lookup(input.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      auto record = recordForSlice(input.field.owner);
      if (mlir::failed(record)) return mlir::failure();
      owner = std::move(*record);
    }
    if (owner.kind != Binding::Kind::Record || owner.interleaveRows != 0)
      return fail(operation, "record element read requires a non-interleaved record descriptor");
    auto output = operation.getResult();
    const int64_t laneAxis = laneAxisFor(output);
    std::string stride;
    for (const auto &[axis, bytes] : owner.recordByteStrides)
      if (axis == laneAxis) stride = bytes;
    if (stride.empty()) return fail(operation, "record element read has no selected lane stride");
    const unsigned bits = mlir::cast<mlir::IntegerType>(output.getType().getElementType()).getWidth();
    auto fragment = singleStorageFragment(*field, index.scalar, bits);
    if (!fragment) return fail(operation, "record element read does not fit one storage byte");
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(output); ++part) {
      auto offsets = partOffsets(output, part);
      auto active = partActive(output, part);
      if (!offsets || !active) return fail(operation, "record element read has no typed part coordinates");
      std::string address = owner.recordPointer;
      for (const auto &[axis, offset] : *offsets)
        for (const auto &[recordAxis, bytes] : owner.recordByteStrides)
          if (axis == recordAxis) address += " + (" + offset + ") * (" + bytes + ")";
      address += " + (" + fragment->byte + ")";
      const std::string suffix = vectorSuffix(output), vl = partVL(output, part);
      auto name = fresh("record_element");
      line(vectorType(output) + " " + name + " = __riscv_vmv_v_x_" + suffix + "(0, " + vl + ");");
      line("if (" + *active + ") {");
      ++indent;
      line(name + " = __riscv_vlse8_v_" + suffix + "((const uint8_t *)(" + address + "), (ptrdiff_t)(" + stride + "), " + vl + ");");
      if (fragment->shift != "0")
        line(name + " = __riscv_vsrl_vx_" + suffix + "(" + name + ", " + fragment->shift + ", " + vl + ");");
      if (bits < 8)
        line(name + " = __riscv_vand_vx_" + suffix + "(" + name + ", " + std::to_string((1u << bits) - 1) + ", " + vl + ");");
      --indent;
      line("}");
      result.parts.push_back(std::move(name));
    }
    bindings[output] = std::move(result);
    return mlir::success();
  }
  auto loaded = materializeNumeric(operation.getResult(), std::move(input));
  if (mlir::failed(loaded))
    return mlir::failure();
  bindings[operation.getResult()] = std::move(*loaded);
  return mlir::success();
}

std::optional<std::string> Emitter::denseAddress(
    const SliceInfo &slice,
    llvm::ArrayRef<std::pair<int64_t, std::string>> offsets) const {
  auto memoryIt = bindings.find(slice.base);
  if (memoryIt == bindings.end() || memoryIt->second.kind != Binding::Kind::Memory)
    return std::nullopt;
  const MemoryInfo &memory = memoryIt->second.memory;
  llvm::StringMap<std::string> offsetByAxis;
  for (auto [axis, expression] : offsets)
    offsetByAxis[std::to_string(axis)] = expression;
  std::string linear = "0";
  size_t domainCursor = 0;
  for (size_t dimension = 0; dimension < memory.axes.size(); ++dimension) {
    std::string coordinate = "0";
    if (dimension < slice.selectors.size() &&
        slice.selectors[dimension] != "all") {
      const std::string &selector = slice.selectors[dimension];
      if (domainCursor >= slice.indices.size())
        return std::nullopt;
      const Binding &index = bindings.lookup(slice.indices[domainCursor++]);
      if (selector == "domain")
        coordinate = index.point.base;
      else if (selector == "group_index")
        coordinate = "(" + index.point.base + " / " +
                     std::to_string(index.point.physicalExtent) + ")";
      else if (selector == "index")
        coordinate = index.scalar;
      else
        return std::nullopt;
    }
    for (auto [axis, value] : slice.localOffsets) {
      if (axis != memory.axes[dimension])
        continue;
      const Binding &offset = bindings.lookup(value);
      std::string expression;
      if (offset.kind == Binding::Kind::Scalar)
        expression = offset.scalar;
      else if (offset.kind == Binding::Kind::Point)
        expression = offset.point.base;
      else
        return std::nullopt;
      coordinate = "(" + coordinate + " + " + expression + ")";
    }
    for (auto [axis, offset] : slice.staticOffsets)
      if (axis == memory.axes[dimension] && offset != 0)
        coordinate = "(" + coordinate + " + " + std::to_string(offset) + ")";
    auto extra = offsetByAxis.find(std::to_string(memory.axes[dimension]));
    if (extra != offsetByAxis.end())
      coordinate = "(" + coordinate + " + " + extra->second + ")";
    if (dimension < memory.origins.size() && memory.origins[dimension] != "0")
      coordinate = "(" + coordinate + " - " + memory.origins[dimension] + ")";
    linear = "(" + linear + " + (" + coordinate + ") * (" +
             memory.strides[dimension] + "))";
  }
  return memory.name + " + " + linear;
}

mlir::FailureOr<Binding>
Emitter::loadDenseBlock(mlir::Value result, const Binding &memoryBlock) {
  Binding loaded;
  loaded.kind = Binding::Kind::Vector;
  auto slice = memoryBlock.slice;
  const Binding &base = bindings.lookup(slice.base);
  if (base.kind != Binding::Kind::Memory) {
    result.getDefiningOp()->emitError("selected dense load has no memory-backed slice");
    return mlir::failure();
  }
  const MemoryInfo &memory = base.memory;
  mlir::Type resultElement = riscv_internal::logicalElement(result.getType());
  if (!memory.elementType || memory.elementType != resultElement) {
    result.getDefiningOp()->emitError(
        "dense load element type does not match its selected local value");
    return mlir::failure();
  }
  auto elementType = scalarCType(resultElement);
  unsigned sew = memory.recordBytes * 8;
  if (!elementType || !sew)
    return mlir::failure();
  const int64_t laneAxis = laneAxisFor(result);
  size_t laneDimension = llvm::find(memory.axes, laneAxis) - memory.axes.begin();
  if (laneDimension >= memory.strides.size()) {
    result.getDefiningOp()->emitError("selected dense load has no physical lane axis");
    return mlir::failure();
  }
  const std::string suffix = vectorSuffix(result);
  const std::string type = vectorType(result);
  mlir::Operation *transfer = result.getDefiningOp();
  while (transfer) {
    if (auto fresh = mlir::dyn_cast<riscv::NewOp>(transfer);
        fresh && fresh.getInitialized()) {
      transfer = fresh.getInitial().getDefiningOp();
      continue;
    }
    if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(transfer)) {
      transfer = conversion.getInput().getDefiningOp();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(transfer)) {
      transfer = materialize.getInput().getDefiningOp();
      continue;
    }
    break;
  }
  riscv::AccessAttr edge = accessOf(transfer);
  llvm::StringRef memoryForm = edge ? edge.getForm() : llvm::StringRef();
  if (memoryForm != "unit" && memoryForm != "strided") {
    result.getDefiningOp()->emitError(
        "dense load has no pass-selected unit or strided memory form");
    return mlir::failure();
  }
  const int64_t totalParts = vectorPartCount(result);
  for (int64_t part = 0; part < totalParts; ++part) {
    auto offsets = partOffsets(result, part);
    auto active = partActive(result, part);
    if (!offsets || !active) {
      result.getDefiningOp()->emitError(
          "selected dense load has no complete physical coordinates");
      return mlir::failure();
    }
    auto address = denseAddress(slice, *offsets);
    if (!address) {
      result.getDefiningOp()->emitError("selected dense load has no address relation");
      return mlir::failure();
    }
    std::string expression;
    if (memoryForm == "unit")
      expression = "__riscv_vle" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + partVL(result, part) + ")";
    else
      expression = "__riscv_vlse" + std::to_string(sew) + "_v_" + suffix +
                   "(" + *address + ", " + memory.strides[laneDimension] +
                   " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                   partVL(result, part) + ")";
    std::string name = fresh("load");
    if (*active != "1") {
      std::string zero = usesFloatingRegisters(resultElement)
                             ? "__riscv_vfmv_v_f_" + suffix + "(0.0, " +
                                   partVL(result, part) + ")"
                             : "__riscv_vmv_v_x_" + suffix + "(0, " +
                                   partVL(result, part) + ")";
      if (resultElement.isF16()) {
        const std::string integerSuffix = "u16" + suffix.substr(3);
        zero = "__riscv_vreinterpret_v_" + integerSuffix + "_" + suffix +
            "(__riscv_vmv_v_x_" + integerSuffix + "(0, " + partVL(result, part) + "))";
      }
      line(type + " " + name + " = " + zero + ";");
      line("if (" + *active + ") " + name + " = " + expression + ";");
    } else {
      line(type + " " + name + " = " + expression + ";");
    }
    loaded.parts.push_back(std::move(name));
  }
  return loaded;
}

mlir::FailureOr<Binding> Emitter::materializeNumeric(mlir::Value value,
                                                     Binding binding) {
  if (binding.kind == Binding::Kind::Slice) {
    auto layout = layoutOf(value);
    llvm::StringRef carrier = layout ? layout.getCarrier() : llvm::StringRef("scalar");
    if (carrier == "scalar") {
      const int64_t parts = scalarPartCount(value);
      if (parts > 1) {
        const Binding &base = bindings.lookup(binding.slice.base);
        mlir::Type element = riscv_internal::logicalElement(value.getType());
        auto type = scalarCType(element);
        if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
            base.memory.elementType != element || !type) {
          value.getDefiningOp()->emitError(
              "register-replicated dense load has no matching scalar memory type");
          return mlir::failure();
        }
        Binding tuple;
        tuple.kind = Binding::Kind::ScalarTuple;
        for (int64_t part = 0; part < parts; ++part) {
          auto offsets = partOffsets(value, part);
          auto active = partActive(value, part);
          if (!offsets || !active) {
            value.getDefiningOp()->emitError(
                "register-replicated dense load has no typed coordinate mapping");
            return mlir::failure();
          }
          auto address = denseAddress(binding.slice, *offsets);
          if (!address)
            return mlir::failure();
          tuple.parts.push_back(materializeScalarRead(
              *type, "((" + *active + ") ? *(const " + *type + " *)(" + *address + ") : (" + *type + ")0)"));
        }
        bindings[value] = tuple;
        return tuple;
      }
      auto valueType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      auto layout = valueType ? valueType.getLayout() : riscv::LayoutAttr();
      const bool scalarElement = value.getType().isIntOrFloat() ||
                                 value.getType().isIndex();
      if ((!valueType && !scalarElement) ||
          (valueType &&
           (!layout || product(layout.getTimeFactors()) != 1 ||
            product(layout.getLaneFactors()) != 1 ||
            product(layout.getReplicaFactors()) != 1 ||
            product(layout.getFragmentFactors()) != 1 ||
            product(layout.getLocalFactors()) != 1))) {
        value.getDefiningOp()->emitError(
            "sequential dense load requires one selected physical element") << "; value=" << value.getType();
        return mlir::failure();
      }
      const Binding &base = bindings.lookup(binding.slice.base);
      mlir::Type element = riscv_internal::logicalElement(value.getType());
      if (base.kind != Binding::Kind::Memory || !base.memory.elementType ||
          base.memory.elementType != element) {
        value.getDefiningOp()->emitError(
            "sequential dense load has no matching memory element");
        return mlir::failure();
      }
      auto address = denseAddress(binding.slice, {});
      auto type = scalarCType(element);
      if (!address || !type) {
        value.getDefiningOp()->emitError(
            "sequential dense load has no address or C type");
        return mlir::failure();
      }
      Binding loaded = scalar(materializeScalarRead(
          *type, "*(const " + *type + " *)(" + *address + ")"));
      bindings[value] = loaded;
      return loaded;
    }
    if (carrier != "rvv") {
      value.getDefiningOp()->emitError(
          "dense slice has no selected scalar or vector load realization");
      return mlir::failure();
    }
    mlir::FailureOr<Binding> loaded = loadDenseBlock(value, binding);
    if (mlir::failed(loaded))
      return mlir::failure();
    // A selected physical SSA load is evaluated once.  Any reload or
    // rematerialization must already be an explicit physical operation; the
    // terminal emitter must not silently duplicate the producer per consumer.
    bindings[value] = *loaded;
    return loaded;
  }
  if (binding.kind == Binding::Kind::Field) {
    auto selectedKind = selectedBindingKind(value);
    if (selectedKind && (*selectedKind == Binding::Kind::ScalarTuple ||
                         *selectedKind == Binding::Kind::Scalar))
      return loadScalarField(value, binding);
    auto expected = selectedBindingKind(value);
    const int64_t expectedParts = expected && *expected == Binding::Kind::Vector
                                     ? vectorPartCount(value) : 1;
    Binding owner = bindings.lookup(binding.field.owner);
    if (owner.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> record = recordForSlice(binding.field.owner);
      if (mlir::failed(record))
        return mlir::failure();
      owner = std::move(*record);
    }
    const bool materializeInterleavedParts =
        expected && *expected == Binding::Kind::Vector &&
        owner.kind == Binding::Kind::Record && owner.interleaveRows > 0;
    if (materializeInterleavedParts) {
      Binding streamed;
      streamed.kind = Binding::Kind::Vector;
      for (int64_t partIndex = 0; partIndex < expectedParts; ++partIndex) {
        Binding part =
            emitInterleavedField(value, binding, "0", partIndex);
        if (part.kind != Binding::Kind::Vector || part.parts.size() != 1) {
          value.getDefiningOp()->emitError(
              "encoded field part has no single selected RVV value");
          return mlir::failure();
        }
        streamed.parts.push_back(std::move(part.parts.front()));
      }
      bindings[value] = streamed;
      return streamed;
    }
    Binding loaded = emitInterleavedField(value, binding, "0", 0);
    if (loaded.kind == Binding::Kind::None) {
      auto diagnostic = value.getDefiningOp()->emitError(
          "encoded field has no selected numeric load realization");
      diagnostic << " (field=" << binding.field.name;
      if (binding.field.storageAccess)
        diagnostic << ", mapping="
                   << binding.field.storageAccess.getMapping();
      if (binding.field.useAccess)
        diagnostic << ", form=" << binding.field.useAccess.getForm();
      diagnostic << ", value_type=" << value.getType()
                 << ", owner_type=" << binding.field.owner.getType();
      diagnostic << ", selected-carrier="
                 << static_cast<int>(expected.value_or(Binding::Kind::None))
                 << ")";
      return mlir::failure();
    }
    if (expected && loaded.kind != *expected) {
      value.getDefiningOp()->emitError(
          "encoded field load does not implement its selected carrier");
      return mlir::failure();
    }
    if ((loaded.kind == Binding::Kind::Vector ||
         loaded.kind == Binding::Kind::ScalarTuple) &&
        static_cast<int64_t>(loaded.parts.size()) != expectedParts) {
      value.getDefiningOp()->emitError(
          "encoded field load does not materialize every selected physical part");
      return mlir::failure();
    }
    if (loaded.kind == Binding::Kind::Scalar ||
        loaded.kind == Binding::Kind::ScalarTuple) {
      auto type = scalarCType(riscv_internal::logicalElement(value.getType()));
      if (!type)
        return value.getDefiningOp()->emitError(
                   "selected scalar field has no intrinsic-C value type"),
               mlir::failure();
      if (loaded.kind == Binding::Kind::Scalar)
        loaded.scalar = materializeScalarRead(*type, loaded.scalar);
      else
        for (std::string &part : loaded.parts)
          part = materializeScalarRead(*type, part);
    }
    bindings[value] = loaded;
    return loaded;
  }
  return binding;
}

mlir::LogicalResult Emitter::compileMemoryView(riscv::MemoryViewOp operation) {
  Binding base = bindings.lookup(operation.getBase());
  if (base.kind != Binding::Kind::Memory)
    return fail(operation, "memory view base has no physical pointer binding");
  const size_t rank = operation.getResult().getType().getShape().size();
  if (operation.getExtents().size() != rank ||
      operation.getStrides().size() != rank ||
      operation.getOrigins().size() != rank)
    return fail(operation, "memory view runtime metadata rank is incomplete");
  base.memory.extents.clear();
  base.memory.strides.clear();
  base.memory.origins.clear();
  auto appendScalars = [&](mlir::ValueRange values,
                           llvm::SmallVectorImpl<std::string> &target) {
    for (mlir::Value value : values) {
      Binding scalarValue = bindings.lookup(value);
      if (scalarValue.kind != Binding::Kind::Scalar)
        return false;
      target.push_back(scalarValue.scalar);
    }
    return true;
  };
  if (!appendScalars(operation.getExtents(), base.memory.extents) ||
      !appendScalars(operation.getStrides(), base.memory.strides) ||
      !appendScalars(operation.getOrigins(), base.memory.origins))
    return fail(operation,
                "memory view extent, stride, and origin must be explicit index SSA values");
  auto descriptor = operation.getResult().getType();
  base.memory.recordBytes = descriptor.getStorageBits() / 8;
  base.memory.logicalElements = descriptor.getElements();
  base.memory.interleaveRows = descriptor.getInterleaveRows();
  bindings[operation.getResult()] = std::move(base);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileDenseDisjoint(riscv::DenseDisjointOp operation) {
  struct Range { std::string address, bytes, valid, empty; };
  auto range = [&](mlir::Value value) -> mlir::FailureOr<Range> {
    auto source = bindings.lookup(value);
    auto type = mlir::cast<riscv::MemDescType>(value.getType());
    if (source.kind != Binding::Kind::Memory || source.memory.extents.size() != type.getShape().size() ||
        source.memory.strides.size() != type.getShape().size()) {
      (void)fail(operation, "dense disjointness has no complete extent/stride binding");
      return mlir::failure();
    }
    Range result{fresh("range_address"), fresh("range_bytes"), fresh("range_valid"), fresh("range_empty")};
    line("const uintptr_t " + result.address + " = (uintptr_t)" + source.memory.name + ";");
    line("size_t " + result.bytes + " = " + std::to_string(type.getStorageBits() / 8) + ";");
    line("bool " + result.valid + " = true;");
    std::string empty = "false";
    for (const auto &extent : source.memory.extents) empty += " || (" + extent + ") == 0";
    line("const bool " + result.empty + " = " + empty + ";");
    // The rank is static and bounded by the op contract. Arithmetic checks do
    // not touch the data, nor form an out-of-bounds C pointer. An unrepresentable
    // bounding interval cannot establish disjointness.
    line("if (!" + result.empty + ") {");
    ++indent;
    for (size_t axis = 0; axis < source.memory.extents.size(); ++axis) {
      std::string count = fresh("range_count"), stride = fresh("range_stride"), term = fresh("range_term");
      line("const size_t " + count + " = (" + source.memory.extents[axis] + ") - 1;");
      line("const size_t " + stride + " = " + source.memory.strides[axis] + ";");
      line("if (" + stride + " > SIZE_MAX / " + std::to_string(type.getStorageBits() / 8) + ") " + result.valid + " = false;");
      line("else {");
      ++indent;
      line("const size_t " + term + " = " + stride + " * " + std::to_string(type.getStorageBits() / 8) + ";");
      line("if (" + term + " != 0 && " + count + " > (SIZE_MAX - " + result.bytes + ") / " + term + ") " + result.valid + " = false;");
      line("else " + result.bytes + " += " + count + " * " + term + ";");
      --indent;
      line("}");
    }
    --indent;
    line("}");
    return result;
  };
  auto lhs = range(operation.getLhs()), rhs = range(operation.getRhs());
  if (mlir::failed(lhs) || mlir::failed(rhs)) return mlir::failure();
  std::string result = fresh("dense_disjoint");
  line("const bool " + result + " = " + lhs->empty + " || " + rhs->empty + " || (" + lhs->valid + " && " + rhs->valid + " && (" +
       lhs->address + " <= " + rhs->address + " ? " + rhs->address + " - " + lhs->address + " >= " + lhs->bytes + " : " +
       lhs->address + " - " + rhs->address + " >= " + rhs->bytes + "));");
  bindings[operation.getResult()] = scalar(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileDenseElementwiseAlias(riscv::DenseElementwiseAliasOp operation) {
  auto a = bindings.lookup(operation.getLhs()), b = bindings.lookup(operation.getRhs());
  const size_t rank = operation.getLhs().getType().getShape().size();
  if (a.kind != Binding::Kind::Memory || b.kind != Binding::Kind::Memory ||
      a.memory.extents.size() != rank || b.memory.extents.size() != rank ||
      a.memory.strides.size() != rank || b.memory.strides.size() != rank)
    return fail(operation, "elementwise alias proof requires complete dense descriptors");
  auto injective = denseInjectivity(operation.getLhs(), operation);
  if (mlir::failed(injective)) return mlir::failure();
  std::string same = fresh("elementwise_alias");
  std::string condition = a.memory.name == b.memory.name ? "true" :
      "(uintptr_t)" + a.memory.name + " == (uintptr_t)" + b.memory.name;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (a.memory.extents[axis] != b.memory.extents[axis])
      condition += " && (" + a.memory.extents[axis] + ") == (" + b.memory.extents[axis] + ")";
    if (a.memory.strides[axis] != b.memory.strides[axis])
      condition += " && (" + a.memory.strides[axis] + ") == (" + b.memory.strides[axis] + ")";
  }
  line("const bool " + same + " = (" + condition + ") && " + *injective + ";");
  bindings[operation.getResult()] = scalar(same);
  return mlir::success();
}

mlir::FailureOr<std::string> Emitter::denseInjectivity(mlir::Value view,
                                                     mlir::Operation *anchor) {
  auto source = bindings.lookup(view);
  const size_t rank = mlir::cast<riscv::MemDescType>(view.getType()).getShape().size();
  if (source.kind != Binding::Kind::Memory || rank > 8 ||
      source.memory.extents.size() != rank || source.memory.strides.size() != rank) {
    (void)fail(anchor, "dense injectivity requires complete bounded ABI metadata");
    return mlir::failure();
  }
  if (!rank) return std::string("true");
  std::string result = fresh("dense_injective"), span = fresh("dense_span");
  std::string sizes = fresh("dense_sizes"), steps = fresh("dense_steps");
  line("size_t " + sizes + "[" + std::to_string(rank) + "] = {" +
       llvm::join(source.memory.extents, ", ") + "};");
  line("size_t " + steps + "[" + std::to_string(rank) + "] = {" +
       llvm::join(source.memory.strides, ", ") + "};");
  std::string empty = "false";
  for (const auto &extent : source.memory.extents) empty += " || (" + extent + ") == 0";
  line("bool " + result + " = true;");
  line("if (!(" + empty + ")) {");
  ++indent;
  // Sort the at-most-eight physical strides, not the data. A permuted or
  // padded layout is injective when every new axis starts beyond the complete
  // footprint of all faster axes. Singleton strides do not constrain it.
  line("for (size_t i = 1; i < " + std::to_string(rank) + "; ++i) {");
  ++indent;
  line("for (size_t j = i; j > 0 && " + steps + "[j] < " + steps + "[j - 1]; --j) {");
  ++indent;
  line("size_t step = " + steps + "[j], size = " + sizes + "[j];");
  line(steps + "[j] = " + steps + "[j - 1]; " + sizes + "[j] = " + sizes + "[j - 1];");
  line(steps + "[j - 1] = step; " + sizes + "[j - 1] = size;");
  --indent;
  line("}");
  --indent;
  line("}");
  line("size_t " + span + " = 1;");
  line("for (size_t i = 0; i < " + std::to_string(rank) + "; ++i) {");
  ++indent;
  line("if (" + sizes + "[i] > 1) {");
  ++indent;
  line("if (" + steps + "[i] < " + span + " || (" + steps + "[i] != 0 && " + sizes +
       "[i] - 1 > (SIZE_MAX - " + span + ") / " + steps + "[i])) " + result + " = false;");
  line("else " + span + " += (" + sizes + "[i] - 1) * " + steps + "[i];");
  --indent;
  line("}");
  --indent;
  line("}");
  --indent;
  line("}");
  return result;
}

mlir::LogicalResult Emitter::compileDenseInjective(riscv::DenseInjectiveOp operation) {
  auto result = denseInjectivity(operation.getView(), operation);
  if (mlir::failed(result)) return mlir::failure();
  bindings[operation.getResult()] = scalar(*result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStorageLoad(
    riscv::StorageLoadOp operation) {
  if (operation.getLeaf().getInstruction() != "scalar.storage.load.u8")
    return fail(operation, "artifact storage load has no closed byte-load leaf");
  Binding source = bindings.lookup(operation.getSource());
  Binding index = bindings.lookup(operation.getByteIndex());
  if (source.kind != Binding::Kind::Memory ||
      index.kind != Binding::Kind::Scalar)
    return fail(operation,
                "artifact storage load requires explicit memory and byte-index bindings");
  bindings[operation.getResult()] =
      scalar(materializeScalarRead(
          "uint8_t", source.memory.name + "[(size_t)(" + index.scalar + ")]"));
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStorageStore(
    riscv::StorageStoreOp operation) {
  if (operation.getLeaf().getInstruction() != "scalar.storage.store.u8")
    return fail(operation, "artifact storage store has no closed byte-store leaf");
  Binding value = bindings.lookup(operation.getValue());
  Binding target = bindings.lookup(operation.getDestination());
  Binding index = bindings.lookup(operation.getByteIndex());
  if (value.kind != Binding::Kind::Scalar ||
      target.kind != Binding::Kind::Memory ||
      index.kind != Binding::Kind::Scalar)
    return fail(operation,
                "artifact storage store requires explicit value, memory, and byte index");
  line(target.memory.name + "[(size_t)(" + index.scalar + ")] = " +
       value.scalar + ";");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSlice(riscv::SliceOp slice) {
  Binding base = bindings.lookup(slice.getBase());
  if (base.kind == Binding::Kind::Field) {
    auto inputType = mlir::dyn_cast<riscv::ValueType>(slice.getBase().getType());
    const int64_t fieldStart =
        inputType ? static_cast<int64_t>(inputType.getShape().size()) -
                        base.field.logicalRank
                  : 0;
    size_t cursor = 0;
    for (auto [position, selectorAttribute] :
         llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size())
        return fail(slice, "field slice selector has no corresponding index");
      if (base.field.index)
        return fail(slice,
                    "intrinsic-C field projection supports one logical field axis");
      base.field.index = slice.getIndices()[cursor++];
      base.field.selector = selector.str();
      // A domain projection narrows an axis to the current Level partition but
      // keeps that logical axis in the result.  Only selectors that eliminate
      // the source axis reduce the encoded field rank.
      if (inputType && static_cast<int64_t>(position) >= fieldStart &&
          selector != "domain")
        --base.field.logicalRank;
    }
    if (auto resultType =
            mlir::dyn_cast<riscv::ValueType>(slice.getResult().getType()))
      base.field.shape.assign(resultType.getShape().asArrayRef().begin(),
                              resultType.getShape().asArrayRef().end());
    bindings[slice.getResult()] = std::move(base);
    return mlir::success();
  }
  Binding binding;
  binding.kind = Binding::Kind::Slice;
  if (base.kind == Binding::Kind::Slice) {
    binding.slice = base.slice;
    auto baseType = mlir::cast<riscv::MemDescType>(slice.getBase().getType());
    const Binding &root = bindings.lookup(binding.slice.base);
    if (root.kind != Binding::Kind::Memory)
      return fail(slice, "nested slice has no root memory descriptor");
    if (binding.slice.selectors.size() > root.memory.axes.size())
      return fail(slice, "nested slice root selector rank exceeds its memory rank");
    binding.slice.selectors.resize(root.memory.axes.size(), "all");
    size_t cursor = 0;
    for (auto [position, selectorAttribute] : llvm::enumerate(slice.getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= slice.getIndices().size() ||
          position >= baseType.getAxisIds().size())
        return fail(slice, "nested slice selector has no logical axis or index");
      int64_t axis = baseType.getAxisIds()[position];
      mlir::Value index = slice.getIndices()[cursor++];
      llvm::erase_if(binding.slice.extents, [axis](const auto &item) { return item.first == axis; });
      if (selector == "index") {
        // Plain scalar indices are offsets relative to the selected block.
        binding.slice.localOffsets.emplace_back(axis, index);
        continue;
      }
      if (selector != "domain" && selector != "group_index")
        return fail(slice, "nested slice has an unknown selector");
      if (selector == "domain")
        binding.slice.extents.emplace_back(axis, bindings.lookup(index).point.active);
      auto found = llvm::find(root.memory.axes, axis);
      if (found == root.memory.axes.end())
        return fail(slice, "nested slice logical axis is absent from root memory");
      size_t rootDimension = static_cast<size_t>(
          std::distance(root.memory.axes.begin(), found));
      size_t rootCursor = 0;
      for (size_t dimension = 0; dimension < rootDimension; ++dimension)
        rootCursor += binding.slice.selectors[dimension] != "all";
      if (binding.slice.selectors[rootDimension] == "all")
        binding.slice.indices.insert(binding.slice.indices.begin() + rootCursor,
                                     index);
      else if (rootCursor < binding.slice.indices.size())
        binding.slice.indices[rootCursor] = index;
      else
        return fail(slice, "nested slice root selector has no matching index");
      // Physical points carry absolute coordinates in their canonical domain.
      // Domain and group-index projections therefore replace the root selector
      // for that axis; adding the child point to its parent would count the
      // parent base twice.
      binding.slice.selectors[rootDimension] = selector.str();
    }
  } else {
    binding.slice.base = slice.getBase();
    binding.slice.indices.assign(slice.getIndices().begin(), slice.getIndices().end());
    for (mlir::Attribute selector : slice.getSelectors())
      binding.slice.selectors.push_back(
          mlir::cast<mlir::StringAttr>(selector).getValue().str());
  }
  bindings[slice.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSubview(riscv::SubviewOp subview) {
  Binding base = bindings.lookup(subview.getBase());
  Binding result;
  result.kind = Binding::Kind::Slice;
  if (base.kind == Binding::Kind::Memory) {
    result.slice.base = subview.getBase();
  } else if (base.kind == Binding::Kind::Slice) {
    result.slice = base.slice;
  } else {
    return fail(subview, "subview base has no memory-backed slice binding");
  }
  auto baseType = subview.getBase().getType();
  if (subview.getOffsets().size() != baseType.getAxisIds().size())
    return fail(subview, "subview offset rank disagrees with its descriptor");
  size_t dynamic = 0;
  for (auto [axis, offset] :
       llvm::zip(baseType.getAxisIds().asArrayRef(),
                 subview.getOffsets()))
    if (offset == -1)
      result.slice.localOffsets.emplace_back(axis, subview.getDynamicBounds()[dynamic++]);
    else if (offset != 0)
      result.slice.staticOffsets.emplace_back(axis, offset);
  for (auto [axis, extent] : llvm::zip(baseType.getAxisIds().asArrayRef(), subview.getExtents())) {
    std::string expression;
    if (extent == -1) {
      const Binding &bound = bindings.lookup(subview.getDynamicBounds()[dynamic++]);
      if (bound.kind != Binding::Kind::Scalar)
        return fail(subview, "subview extent has no scalar SSA binding");
      expression = bound.scalar;
    } else expression = std::to_string(extent);
    llvm::erase_if(result.slice.extents, [axis](const auto &item) { return item.first == axis; });
    result.slice.extents.emplace_back(axis, expression);
  }
  bindings[subview.getResult()] = std::move(result);
  return mlir::success();
}

std::optional<std::string> Emitter::descriptorExtent(
    mlir::Value value, int64_t axis) const {
  Binding base = bindings.lookup(value);
  std::string expression;
  if (base.kind == Binding::Kind::Memory) {
    auto position = llvm::find(base.memory.axes, axis);
    if (position == base.memory.axes.end()) return std::nullopt;
    expression = base.memory.extents[position - base.memory.axes.begin()];
  } else if (base.kind == Binding::Kind::Slice) {
    for (const auto &bound : base.slice.extents)
      if (bound.first == axis) expression = bound.second;
    if (expression.empty()) {
      const Binding &root = bindings.lookup(base.slice.base);
      if (root.kind != Binding::Kind::Memory)
        return std::nullopt;
      size_t cursor = 0;
      for (size_t dimension = 0; dimension < root.memory.axes.size(); ++dimension) {
        llvm::StringRef selector = dimension < base.slice.selectors.size()
            ? base.slice.selectors[dimension] : llvm::StringRef("all");
        if (root.memory.axes[dimension] == axis) {
          if (selector == "all") expression = root.memory.extents[dimension];
          else if (selector == "domain") expression = bindings.lookup(base.slice.indices[cursor]).point.active;
          else return std::nullopt;
          break;
        }
        cursor += selector != "all";
      }
    }
  }
  if (expression.empty()) return std::nullopt;
  return expression;
}

std::optional<std::string> Emitter::descriptorStride(
    mlir::Value value, int64_t axis) const {
  Binding source = bindings.lookup(value);
  if (source.kind == Binding::Kind::Slice)
    source = bindings.lookup(source.slice.base);
  if (source.kind != Binding::Kind::Memory)
    return std::nullopt;
  auto position = llvm::find(source.memory.axes, axis);
  if (position == source.memory.axes.end())
    return std::nullopt;
  size_t index = position - source.memory.axes.begin();
  if (index >= source.memory.strides.size())
    return std::nullopt;
  return source.memory.strides[index];
}

mlir::LogicalResult Emitter::compileDenseRowWindowGuard(
    riscv::DenseRowWindowGuardOp operation) {
  auto memory = operation.getSource().getType();
  int64_t rowAxis = memory.getAxisIds()[operation.getRowAxis()];
  int64_t innerAxis = memory.getAxisIds()[operation.getInnerAxis()];
  auto rows = descriptorExtent(operation.getSource(), rowAxis);
  auto columns = descriptorExtent(operation.getSource(), innerAxis);
  auto stride = descriptorStride(operation.getSource(), rowAxis);
  auto address = byteReadAddress(operation.getSource());
  if (!rows || !columns || !stride || !address)
    return fail(operation, "dense row-window guard has no complete descriptor binding");
  const std::string result = fresh("dense_row_window");
  // The extent predicates come first: an incomplete/empty window never forms
  // the address used by the whole-row transfer. This query reads no data.
  line("const bool " + result + " = (" + *rows + ") >= " +
       std::to_string(operation.getRows()) + " && (" + *columns + ") >= " +
       std::to_string(operation.getColumns()) + " && (((uintptr_t)(" +
       *address + ") | (uintptr_t)(" + *stride + ")) & " +
       std::to_string(operation.getAlignment() - 1) + "u) == 0;");
  bindings[operation.getResult()] = scalar(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileExtent(riscv::ExtentOp operation) {
  int64_t axis = operation.getBase().getType().getAxisIds()[operation.getAxis()];
  auto expression = descriptorExtent(operation.getBase(), axis);
  if (!expression) return fail(operation, "extent has no physical descriptor binding");
  Binding result;
  result.kind = Binding::Kind::Scalar;
  result.scalar = *expression;
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileReshape(riscv::ReshapeOp reshape) {
  Binding input = bindings.lookup(reshape.getInput());
  if (input.kind == Binding::Kind::LocalArray &&
      reshape.getInput().getType().getShape() != reshape.getResult().getType().getShape()) {
    llvm::SmallVector<std::string> extents;
    auto sourceShape = reshape.getInput().getType().getShape();
    for (int64_t extent : reshape.getResult().getType().getShape().asArrayRef()) {
      if (extent > 0) { extents.push_back(std::to_string(extent)); continue; }
      auto position = llvm::find(sourceShape.asArrayRef(), extent);
      if (position == sourceShape.asArrayRef().end() ||
          static_cast<size_t>(position - sourceShape.asArrayRef().begin()) >= input.localExtents.size())
        return fail(reshape, "local reshape has no preserved actual dimension");
      extents.push_back(input.localExtents[position - sourceShape.asArrayRef().begin()]);
    }
    input.localExtents = std::move(extents);
  }
  const int64_t inputParts =
      input.kind == Binding::Kind::Vector
          ? vectorPartCount(reshape.getInput())
          : input.kind == Binding::Kind::ScalarTuple
                ? scalarPartCount(reshape.getInput())
                : 1;
  const int64_t resultParts =
      input.kind == Binding::Kind::Vector
          ? vectorPartCount(reshape.getResult())
          : input.kind == Binding::Kind::ScalarTuple
                ? scalarPartCount(reshape.getResult())
                : 1;
  if (input.kind == Binding::Kind::None || inputParts <= 0 ||
      inputParts != resultParts ||
      ((input.kind == Binding::Kind::Vector ||
        input.kind == Binding::Kind::ScalarTuple) &&
       static_cast<int64_t>(input.parts.size()) != inputParts))
    return fail(reshape,
                "reshape input does not match its selected linear carrier partition");
  bindings[reshape.getResult()] = std::move(input);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileReaxis(riscv::ReaxisOp operation) {
  Binding base = bindings.lookup(operation.getBase());
  Binding result;
  result.kind = Binding::Kind::Memory;
  if (base.kind == Binding::Kind::Memory) result = base;
  else if (base.kind == Binding::Kind::Slice) {
    const auto &root = bindings.lookup(base.slice.base);
    if (root.kind != Binding::Kind::Memory)
      return fail(operation, "descriptor reaxis has no root pointer binding");
    result.memory = root.memory;
    result.memory.extents.clear(); result.memory.strides.clear(); result.memory.origins.clear();
    llvm::SmallVector<std::pair<int64_t, std::string>> origins;
    for (int64_t axis : operation.getBase().getType().getAxisIds().asArrayRef()) {
      auto position = llvm::find(root.memory.axes, axis);
      if (position == root.memory.axes.end()) return fail(operation, "reaxis lost a source descriptor axis");
      size_t index = position - root.memory.axes.begin();
      auto extent = descriptorExtent(operation.getBase(), axis);
      if (!extent) return fail(operation, "reaxis lost a source descriptor extent");
      result.memory.extents.push_back(*extent);
      result.memory.strides.push_back(root.memory.strides[index]);
      result.memory.origins.push_back(root.memory.origins[index]);
      origins.emplace_back(axis, root.memory.origins[index]);
    }
    auto address = denseAddress(base.slice, origins);
    if (!address) return fail(operation, "reaxis has no closed source address");
    result.memory.name = "(" + *address + ")";
  } else return fail(operation, "reaxis requires an existing dense descriptor");
  auto axes = operation.getResult().getType().getAxisIds().asArrayRef();
  result.memory.axes.assign(axes.begin(), axes.end());
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::FailureOr<Binding> Emitter::recordForSlice(mlir::Value value) {
  Binding region = bindings.lookup(value);
  if (region.kind != Binding::Kind::Slice) {
    value.getDefiningOp()->emitError("encoded record requires a canonical slice region");
    return mlir::failure();
  }
  const Binding &base = bindings.lookup(region.slice.base);
  auto sliceType = mlir::dyn_cast<riscv::MemDescType>(value.getType());
  auto encoding = sliceType
                      ? mlir::dyn_cast<kernel::EncodingType>(sliceType.getEncoding())
                      : kernel::EncodingType();
  if (!encoding || encoding.getKind() == "dense") {
    value.getDefiningOp()->emitError("record slice must carry a concrete encoding");
    return mlir::failure();
  }
  if (base.kind != Binding::Kind::Memory) {
    value.getDefiningOp()->emitError("encoded record requires a memory-backed view");
    return mlir::failure();
  }
  std::string family = encoding.getFamily().str();
  if (base.memory.recordBytes <= 0 || base.memory.logicalElements <= 0 ||
      base.memory.strides.size() != base.memory.axes.size() ||
      base.memory.origins.size() != base.memory.axes.size())
    return value.getDefiningOp()->emitError(
               "encoded record descriptor has no explicit storage geometry"),
           mlir::failure();

  llvm::DenseMap<int64_t, PointInfo> points;
  llvm::SmallVector<std::string> coordinates(base.memory.origins.begin(),
                                             base.memory.origins.end());
  size_t cursor = 0;
  for (auto [dimension, selector] : llvm::enumerate(region.slice.selectors)) {
    if (selector == "all")
      continue;
    if (cursor >= region.slice.indices.size())
      return value.getDefiningOp()->emitError(
                 "encoded record selector has no index operand"),
             mlir::failure();
    const Binding &index = bindings.lookup(region.slice.indices[cursor++]);
    if (selector == "domain") {
      coordinates[dimension] = index.point.base;
      points[base.memory.axes[dimension]] = index.point;
    } else if (selector == "group_index") {
      coordinates[dimension] = "(" + index.point.base + " / " +
                               std::to_string(index.point.physicalExtent) + ")";
      points[base.memory.axes[dimension]] = index.point;
    } else if (selector == "index") {
      coordinates[dimension] = index.scalar;
    } else {
      return value.getDefiningOp()->emitError(
                 "encoded record has an unknown slice selector"),
             mlir::failure();
    }
  }
  for (auto [axis, offsetValue] : region.slice.localOffsets) {
    auto dimension = llvm::find(base.memory.axes, axis);
    if (dimension == base.memory.axes.end())
      return value.getDefiningOp()->emitError(
                 "encoded record local offset has no matching memory axis"),
             mlir::failure();
    const size_t position =
        static_cast<size_t>(std::distance(base.memory.axes.begin(), dimension));
    const Binding &offset = bindings.lookup(offsetValue);
    std::string expression;
    if (offset.kind == Binding::Kind::Scalar) {
      expression = offset.scalar;
    } else if (offset.kind == Binding::Kind::Point) {
      expression = offset.point.base;
      PointInfo point = offset.point;
      point.base = "(" + coordinates[position] + " + " + expression + ")";
      points[axis] = std::move(point);
    } else {
      return value.getDefiningOp()->emitError(
                 "encoded record local offset is neither scalar nor domain point"),
             mlir::failure();
    }
    coordinates[position] =
        "(" + coordinates[position] + " + " + expression + ")";
    if (auto point = points.find(axis); point != points.end())
      point->second.base = coordinates[position];
  }
  Binding result;
  result.kind = Binding::Kind::Record;
  const int64_t recordBytes = base.memory.recordBytes;
  const int64_t elements = base.memory.logicalElements;
  result.recordAxis = base.memory.axes.empty() ? 0 : base.memory.axes.back();
  result.recordElements = elements;
  result.recordStrideBytes = recordBytes;
  for (auto [dimension, axis] : llvm::enumerate(base.memory.axes))
    if (llvm::is_contained(sliceType.getAxisIds().asArrayRef(), axis))
      result.recordOrigins.emplace_back(axis, coordinates[dimension]);
  if (encoding.getKind() == "derived_instance" ||
      base.memory.interleaveRows > 0) {
    int64_t rows = base.memory.interleaveRows > 0
                       ? base.memory.interleaveRows
                       : sliceType.getInterleaveRows();
    if (rows <= 0 || base.memory.axes.size() != 2)
      return value.getDefiningOp()->emitError(
                 "derived interleave requires a two-axis view and selected rows"),
             mlir::failure();
    const std::string rowBase =
        "(" + coordinates[0] + " - " + base.memory.origins[0] + ")";
    const std::string blockBase =
        "(" + coordinates[1] + " - " + base.memory.origins[1] + ")";
    const std::string blocks = "(" + base.memory.extents[1] + " / " +
                               std::to_string(elements) + ")";
    result.interleaveRows = rows;
    result.interleaveRowAxis = base.memory.axes[0];
    result.interleaveRowOffset = "(" + rowBase + " % " + std::to_string(rows) + ")";
    result.interleaveCohortStrideBytes = "(" + blocks + " * " +
                                         std::to_string(recordBytes * rows) + ")";
    result.recordStrideBytes = recordBytes * rows;
    result.recordByteStrides.emplace_back(base.memory.axes[0], "1");
    result.recordPointer = base.memory.name + " + (((" + rowBase + " / " +
                           std::to_string(rows) + ") * " + blocks +
                           " + (" + blockBase + " / " +
                           std::to_string(elements) + ")) * " +
                           std::to_string(recordBytes * rows) + ") + " +
                           result.interleaveRowOffset;
  } else {
    if (base.memory.axes.empty())
      return value.getDefiningOp()->emitError(
                 "base encoded view has no logical axis"),
             mlir::failure();
    const size_t recordDimension = base.memory.axes.size() - 1;
    for (size_t dimension = 0; dimension < recordDimension; ++dimension) {
      result.recordByteStrides.emplace_back(
          base.memory.axes[dimension],
          "((" + base.memory.strides[dimension] + " / " +
              std::to_string(elements) + ") * " +
              std::to_string(recordBytes) + ")");
    }
    // Every dimension before the encoded element axis advances by whole
    // records. Keep that divisibility visible in C instead of flattening all
    // logical coordinates and dividing the sum: the latter hides the affine
    // record induction from the system compiler when extents are dynamic.
    std::string record = "0";
    for (size_t dimension = 0; dimension < recordDimension; ++dimension) {
      record = "(" + record + " + ((" + coordinates[dimension] + ") - (" +
               base.memory.origins[dimension] + ")) * ((" +
               base.memory.strides[dimension] + ") / " +
               std::to_string(elements) + "))";
    }
    const size_t elementDimension = recordDimension;
    record = "(" + record + " + (((" + coordinates[elementDimension] +
             ") - (" + base.memory.origins[elementDimension] + ")) * (" +
             base.memory.strides[elementDimension] + ")) / " +
             std::to_string(elements) + ")";
    result.recordPointer = base.memory.name + " + (" + record + ") * " +
                           std::to_string(recordBytes);
  }
  return result;
}

mlir::LogicalResult Emitter::compileAdmit(riscv::LoadOp admit) {
  Binding region = bindings.lookup(admit.getRegion());
  if (region.kind == Binding::Kind::Memory) {
    auto view = mlir::dyn_cast<riscv::MemDescType>(admit.getRegion().getType());
    auto encoding = view ? mlir::dyn_cast<kernel::EncodingType>(view.getEncoding())
                         : kernel::EncodingType();
    if (!encoding || encoding.getKind() != "dense")
      return fail(admit,
                  "whole-view admit is only defined for a dense read-only value");
    Binding slice;
    slice.kind = Binding::Kind::Slice;
    slice.slice.base = admit.getRegion();
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(admit.getResult(), std::move(slice));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[admit.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(admit, "admit currently requires a canonical slice region");
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(
      mlir::cast<riscv::MemDescType>(admit.getRegion().getType()).getEncoding());
  if (!encoding)
    return fail(admit, "admit slice has no encoding");
  if (encoding.getKind() == "dense") {
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(admit.getResult(), std::move(region));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[admit.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  mlir::FailureOr<Binding> result = recordForSlice(admit.getRegion());
  if (mlir::failed(result))
    return mlir::failure();
  if (mlir::Value allocation = admit.getSnapshotStorage()) {
    Binding storage = bindings.lookup(allocation);
    auto storageType = mlir::cast<riscv::LocalType>(allocation.getType());
    if (storage.kind != Binding::Kind::LocalArray)
      return fail(admit, "record snapshot has no selected local storage binding");
    if (mlir::failed(emitReadSnapshot(admit, admit.getLeaf(), storage.scalar,
                                     result->recordPointer,
                                     storageType.getSizeBytes())))
      return mlir::failure();
    result->recordPointer = storage.scalar;
  }
  bindings[admit.getResult()] = std::move(*result);
  return mlir::success();
}

std::optional<std::string> Emitter::byteReadAddress(mlir::Value value) {
  Binding source = bindings.lookup(value);
  std::optional<std::string> address;
  if (source.kind == Binding::Kind::Slice) {
    address = denseAddress(source.slice, {});
  } else if (source.kind == Binding::Kind::Memory) {
    address = source.memory.name;
  } else if (source.kind == Binding::Kind::Field) {
    Binding owner = bindings.lookup(source.field.owner);
    auto field = fieldFor(source);
    if (field && field->bitOffset % 8 == 0) {
      if (owner.kind == Binding::Kind::Record)
        address = "(" + owner.recordPointer + " + " +
                  std::to_string(field->bitOffset / 8) + ")";
      else if (owner.kind == Binding::Kind::Memory)
        address = "(" + owner.memory.name + " + " +
                  std::to_string(field->bitOffset / 8) + ")";
    }
  }
  return address;
}

mlir::LogicalResult Emitter::compileRegisterMaterialize(
    riscv::RegisterMaterializeOp materialize) {
  Binding input = bindings.lookup(materialize.getInput());
  if (input.kind == Binding::Kind::None)
    return fail(materialize, "register materialize input has no emitted value");
  if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
    mlir::FailureOr<Binding> value =
        materializeNumeric(materialize.getResult(), std::move(input));
    if (mlir::failed(value))
      return mlir::failure();
    input = std::move(*value);
  }
  const bool scalarShare =
      materialize.getRealization() == "physical-share" ||
      (materialize.getRealization() == "share" &&
       (input.kind == Binding::Kind::Scalar ||
        input.kind == Binding::Kind::ScalarTuple));
  if (scalarShare) {
    auto type = scalarCType(
        riscv_internal::logicalElement(materialize.getResult().getType()));
    if (!type || (input.kind != Binding::Kind::Scalar &&
                  input.kind != Binding::Kind::ScalarTuple))
      return fail(materialize,
                  "physical scalar materialization has no closed intrinsic-C value");
    if (input.kind == Binding::Kind::Scalar) {
      std::string name = fresh("physical_share");
      line(*type + " " + name + " = " + input.scalar + ";");
      input.scalar = std::move(name);
    } else {
      for (std::string &part : input.parts) {
        std::string name = fresh("physical_share");
        line(*type + " " + name + " = " + part + ";");
        part = std::move(name);
      }
    }
  }
  bindings[materialize.getResult()] = std::move(input);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileStagedView(riscv::StagedViewOp staged) {
  Binding source = bindings.lookup(staged.getSource());
  if (source.kind != Binding::Kind::Memory &&
      source.kind != Binding::Kind::Slice)
    return fail(staged, "reload staged view requires one emitted memory descriptor");
  bindings[staged.getResult()] = std::move(source);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalCapacityGuard(
    riscv::LocalCapacityGuardOp operation) {
  Binding bytes = bindings.lookup(operation.getSizeBytes());
  Binding live = bindings.lookup(operation.getLiveBytes());
  if (bytes.kind != Binding::Kind::Scalar || live.kind != Binding::Kind::Scalar ||
      operation.getLeaf().getInstruction() != "scalar.local-capacity-guard")
    return fail(operation, "dynamic local allocation has no closed capacity guard");
  line("if ((uint64_t)(" + bytes.scalar + ") > " +
       std::to_string(operation.getLimitBytes()) +
       "ULL || (uint64_t)(" + live.scalar + ") > " + std::to_string(operation.getLimitBytes()) +
       "ULL - (uint64_t)(" + bytes.scalar + ")) __builtin_trap();");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalSize(riscv::LocalSizeOp operation) {
  llvm::SmallVector<std::string> extents;
  std::string nonempty = "1";
  for (auto extent : operation.getExtents()) {
    auto binding = bindings.lookup(extent);
    if (binding.kind != Binding::Kind::Scalar)
      return fail(operation, "local storage size requires actual scalar dimensions");
    extents.push_back(binding.scalar);
    line("if ((ptrdiff_t)(" + binding.scalar + ") < 0) __builtin_trap();");
    nonempty += " && (" + binding.scalar + ") != 0";
  }
  std::string bytes = fresh("local_bytes");
  line("size_t " + bytes + " = 0;");
  line("if (" + nonempty + ") {");
  ++indent;
  line(bytes + " = " + std::to_string(operation.getElementBytes()) + ";");
  for (const auto &extent : extents) {
    line("if ((size_t)(" + extent + ") > SIZE_MAX / " + bytes + ") __builtin_trap();");
    line(bytes + " *= (size_t)(" + extent + ");");
  }
  --indent;
  line("}");
  bindings[operation.getResult()] = scalar(bytes);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalAlloc(riscv::LocalAllocOp operation) {
  riscv::LocalType type = operation.getResult().getType();
  Binding size = bindings.lookup(operation.getSizeBytes());
  if (size.kind != Binding::Kind::Scalar || type.getAlignment() <= 0)
    return fail(operation, "local allocation has no closed byte size/alignment");
  std::string name = fresh(type.getPurpose() == "spill" ? "spill_slot"
                                                         : "local_object");
  line("_Alignas(" + std::to_string(type.getAlignment()) + ") uint8_t " + name +
       "[" + size.scalar + "];");
  Binding binding;
  binding.kind = Binding::Kind::LocalArray;
  binding.scalar = name;
  binding.localElements = size.scalar;
  bindings[operation.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::emitReadSnapshot(
    mlir::Operation *operation, riscv::LeafAttr leaf,
    llvm::StringRef destination, llvm::StringRef source, int64_t bytes) {
  if (leaf.getInstruction() == "scalar.read-snapshot") {
    line("memcpy(" + destination.str() + ", " + source.str() + ", " +
         std::to_string(bytes) + ");");
    return mlir::success();
  }
  auto parameters = leaf.getParameters().asArrayRef();
  if (leaf.getInstruction() != "rvv.read-snapshot" ||
      parameters.size() != 2 || parameters[1] <= 0 || bytes <= 0)
    return fail(operation, "read snapshot has no exact byte-transfer leaf");
  const std::string group = "m" + std::to_string(parameters[0] / 8);
  for (int64_t offset = 0; offset < bytes;) {
    const int64_t width = std::min(parameters[1], bytes - offset);
    const std::string count = std::to_string(width);
    const std::string at = std::to_string(offset);
    const std::string value = fresh("snapshot_bytes");
    line("vuint8" + group + "_t " + value + " = __riscv_vle8_v_u8" + group + "((const uint8_t *)(" +
         source.str() + ") + " + at + ", " + count + ");");
    line("__riscv_vse8_v_u8" + group + "((uint8_t *)(" + destination.str() + ") + " + at +
         ", " + value + ", " + count + ");");
    offset += width;
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileDenseSnapshot(riscv::DenseSnapshotOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  Binding storage = bindings.lookup(operation.getStorage());
  std::optional<std::string> address;
  if (source.kind == Binding::Kind::Slice)
    address = denseAddress(source.slice, {});
  else if (source.kind == Binding::Kind::Memory)
    address = source.memory.name;
  auto descriptor = operation.getResult().getType();
  auto element = denseElementType(descriptor.getEncoding());
  if (!address || !element || storage.kind != Binding::Kind::LocalArray ||
      storage.scalar.empty())
    return fail(operation, "dense snapshot has no complete memory/storage binding");
  if (mlir::failed(emitReadSnapshot(
          operation, operation.getLeaf(), storage.scalar, *address,
          operation.getStorage().getType().getSizeBytes())))
    return mlir::failure();
  Binding result;
  result.kind = Binding::Kind::Memory;
  result.memory.name = storage.scalar;
  result.memory.encoding = descriptor.getEncoding();
  result.memory.elementType = *element;
  result.memory.axes.assign(descriptor.getAxisIds().asArrayRef().begin(),
                             descriptor.getAxisIds().asArrayRef().end());
  result.memory.extents.push_back(std::to_string(descriptor.getShape()[0]));
  result.memory.strides.push_back("1");
  result.memory.origins.push_back("0");
  result.memory.recordBytes = descriptor.getStorageBits() / 8;
  result.memory.logicalElements = 1;
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalCopy(riscv::LocalCopyOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  Binding destination = bindings.lookup(operation.getDestination());
  Binding offset = bindings.lookup(operation.getByteOffset()), active = bindings.lookup(operation.getActiveBytes());
  if (source.kind != Binding::Kind::Memory || destination.kind != Binding::Kind::Memory ||
      offset.kind != Binding::Kind::Scalar || active.kind != Binding::Kind::Scalar ||
      operation.getLeaf().getInstruction() != "rvv.local-copy")
    return fail(operation, "local copy lost its selected memory, offset or byte-count binding");
  auto parameters = operation.getLeaf().getParameters().asArrayRef();
  const std::string group = "m" + std::to_string(parameters[0] / 8);
  const std::string value = fresh("local_copy_bytes");
  line("vuint8" + group + "_t " + value + " = __riscv_vle8_v_u8" + group +
       "((const uint8_t *)(" + source.memory.name + ") + (" + offset.scalar + "), " + active.scalar + ");");
  line("__riscv_vse8_v_u8" + group + "((uint8_t *)(" + destination.memory.name +
       ") + (" + offset.scalar + "), " + value + ", " + active.scalar + ");");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalBind(riscv::LocalBindOp operation) {
  Binding storage = bindings.lookup(operation.getStorage());
  Binding elements = bindings.lookup(operation.getElements());
  if (storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      elements.kind != Binding::Kind::Scalar)
    return fail(operation,
                "local bind requires one allocation and one explicit element extent");
  auto element = scalarCType(operation.getResult().getType().getElementType());
  if (!element)
    return fail(operation, "local bind element has no intrinsic-C type");
  storage.scalar = "((" + *element + " *)" + storage.scalar + ")";
  storage.localElements = elements.scalar;
  storage.localExtents.clear();
  for (auto extent : operation.getExtents()) {
    auto dimension = bindings.lookup(extent);
    if (dimension.kind != Binding::Kind::Scalar)
      return fail(operation, "local bind has an unavailable dimension");
    storage.localExtents.push_back(dimension.scalar);
  }
  bindings[operation.getResult()] = std::move(storage);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalView(riscv::LocalViewOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  if (source.kind != Binding::Kind::LocalArray || source.scalar.empty())
    return fail(operation, "local view requires its existing allocation binding");
  auto descriptor = operation.getResult().getType();
  Binding result;
  result.kind = Binding::Kind::Memory;
  result.memory.name = source.scalar;
  result.memory.encoding = descriptor.getEncoding();
  result.memory.elementType = operation.getSource().getType().getElementType();
  result.memory.axes.assign(descriptor.getAxisIds().asArrayRef().begin(), descriptor.getAxisIds().asArrayRef().end());
  result.memory.recordBytes = descriptor.getStorageBits() / 8;
  result.memory.logicalElements = 1;
  for (auto [extent, stride] : llvm::zip(operation.getExtents(), operation.getStrides())) {
    auto e = bindings.lookup(extent), s = bindings.lookup(stride);
    if (e.kind != Binding::Kind::Scalar || s.kind != Binding::Kind::Scalar)
      return fail(operation, "local view extent and stride require explicit scalar bindings");
    result.memory.extents.push_back(e.scalar);
    result.memory.strides.push_back(s.scalar);
    result.memory.origins.push_back("0");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalLoad(riscv::LocalLoadOp operation) {
  Binding source = bindings.lookup(operation.getSource());
  Binding index = bindings.lookup(operation.getIndex());
  auto element = scalarCType(operation.getResult().getType());
  if (source.kind != Binding::Kind::LocalArray || source.scalar.empty() ||
      index.kind != Binding::Kind::Scalar || !element ||
      operation.getLeaf().getInstruction() != "local.load.element")
    return fail(operation, "local element load has no closed physical binding");
  bindings[operation.getResult()] =
      scalar(materializeScalarRead(
          *element, "((const " + *element + " *)" + source.scalar + ")[" +
                        index.scalar + "]",
          "local_load"));
  return mlir::success();
}

mlir::LogicalResult Emitter::compileLocalStore(riscv::LocalStoreOp operation) {
  Binding value = bindings.lookup(operation.getValue());
  Binding destination = bindings.lookup(operation.getDestination());
  Binding index = bindings.lookup(operation.getIndex());
  auto element = scalarCType(operation.getValue().getType());
  if (value.kind != Binding::Kind::Scalar ||
      destination.kind != Binding::Kind::LocalArray ||
      destination.scalar.empty() || index.kind != Binding::Kind::Scalar ||
      !element || operation.getLeaf().getInstruction() != "local.store.element")
    return fail(operation, "local element store has no closed physical binding");
  line("((" + *element + " *)" + destination.scalar + ")[" + index.scalar +
       "] = " + value.scalar + ";");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVLocalMaterialize(
    riscv::RVVLocalMaterializeOp operation) {
  if (instructionOf(operation.getOperation()) != "rvv.local-materialize")
    return fail(operation,
                "RVV local materialize has no exact selected leaf");
  mlir::FailureOr<Binding> input = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  Binding destination = bindings.lookup(operation.getDestination());
  auto element = scalarCType(operation.getInput().getType().getElementType());
  const int64_t parts = vectorPartCount(operation.getInput());
  const int64_t lanes = physicalLanes(operation.getInput());
  const unsigned bits = operation.getInput().getType().getLayout().getSew();
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector ||
      destination.kind != Binding::Kind::LocalArray ||
      destination.scalar.empty() || !element || parts <= 0 || lanes <= 0 ||
      input->parts.size() != static_cast<size_t>(parts))
    return fail(operation,
                "RVV local materialize has no complete vector/local binding");
  const std::string suffix = vectorSuffix(operation.getInput());
  for (int64_t part = 0; part < parts; ++part)
    line("__riscv_vse" + std::to_string(bits) + "_v_" + suffix + "((" +
         *element + " *)" + destination.scalar + " + " +
         std::to_string(part * lanes) + ", " + input->parts[part] + ", " +
         partVL(operation.getInput(), part) + ");");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIndexMultipleGuard(
    riscv::IndexMultipleGuardOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "scalar.index-multiple-guard")
    return fail(operation, "index multiple guard has no exact selected leaf");
  Binding value = bindings.lookup(operation.getValue());
  if (value.kind != Binding::Kind::Scalar || operation.getMultiple() <= 0)
    return fail(operation,
                "index multiple guard requires one scalar value and positive divisor");
  line("if (((size_t)(" + value.scalar + ") % " +
       std::to_string(operation.getMultiple()) + ") != 0) __builtin_trap();");
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileEncodedLocalBind(riscv::EncodedLocalBindOp operation) {
  Binding source = bindings.lookup(operation.getInput());
  Binding storage = bindings.lookup(operation.getStorage());
  Binding blocks = bindings.lookup(operation.getBlockCount());
  riscv::LocalPackPlanAttr plan = operation.getPlan();
  if (source.kind != Binding::Kind::Record || source.interleaveRows != 0 ||
      source.recordElements != plan.getRecordElements() ||
      source.recordStrideBytes != plan.getRecordBytes() ||
      storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      blocks.kind != Binding::Kind::Scalar)
    return fail(operation,
                "encoded local bind has no complete source, storage, and block-count bindings");

  Binding result;
  result.kind = Binding::Kind::Record;
  result.recordPointer = storage.scalar;
  result.interleaveRows = plan.getInterleaveRows();
  result.interleaveRowAxis = plan.getRowAxis();
  result.interleaveCohortStrideBytes = "((" + blocks.scalar + ") * " +
      std::to_string(plan.getRecordBytes() * plan.getInterleaveRows()) + ")";
  result.recordAxis = plan.getRecordAxis();
  result.recordElements = plan.getRecordElements();
  result.recordStrideBytes =
      plan.getRecordBytes() * plan.getInterleaveRows();
  result.recordGroupAxis = plan.getRowAxis();
  result.recordByteStrides.emplace_back(
      plan.getRowAxis(), "((" + blocks.scalar + ") * " +
                             std::to_string(plan.getRecordBytes()) + ")");
  result.recordOrigins = source.recordOrigins;
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVEncodedLocalPackTransfer(
    riscv::RVVEncodedLocalPackTransferOp operation) {
  if (instructionOf(operation.getOperation()) !=
      "rvv.local-pack.transfer.interleave")
    return fail(operation,
                "encoded local pack transfer has no exact selected leaf");
  Binding source = bindings.lookup(operation.getInput());
  Binding storage = bindings.lookup(operation.getStorage());
  Binding rowGroup = bindings.lookup(operation.getRowGroup());
  Binding rowBase = bindings.lookup(operation.getRowBase());
  Binding block = bindings.lookup(operation.getBlock());
  Binding blocks = bindings.lookup(operation.getBlockCount());
  Binding byte = bindings.lookup(operation.getByte());
  Binding transferVL = bindings.lookup(operation.getTransferVl());
  riscv::LocalPackPlanAttr plan = operation.getPlan();
  auto rowStride = llvm::find_if(
      source.recordByteStrides,
      [&](const auto &entry) { return entry.first == plan.getRowAxis(); });
  if (source.kind != Binding::Kind::Record || source.interleaveRows != 0 ||
      source.recordElements != plan.getRecordElements() ||
      source.recordStrideBytes != plan.getRecordBytes() ||
      rowStride == source.recordByteStrides.end() ||
      storage.kind != Binding::Kind::LocalArray || storage.scalar.empty() ||
      rowGroup.kind != Binding::Kind::Scalar ||
      rowBase.kind != Binding::Kind::Scalar ||
      block.kind != Binding::Kind::Scalar ||
      blocks.kind != Binding::Kind::Scalar ||
      byte.kind != Binding::Kind::Scalar ||
      transferVL.kind != Binding::Kind::Scalar)
    return fail(operation,
                "encoded local pack transfer has incomplete typed coordinates");

  const std::string transferSuffix =
      "u8" + lmulSpelling(operation.getTransferLayout().getLmulEighths());
  const std::string transferType =
      "vuint8" + lmulSpelling(operation.getTransferLayout().getLmulEighths()) +
      "_t";
  const std::string transferVLName = fresh("pack_vl");
  line("const size_t " + transferVLName + " = (size_t)(" +
       transferVL.scalar + ");");
  const std::string sourceAddress =
      source.recordPointer + " + (" + rowBase.scalar + ") * (" +
      rowStride->second + ") + (" + block.scalar + ") * " +
      std::to_string(plan.getRecordBytes()) + " + (" + byte.scalar + ")";
  const std::string targetAddress =
      "((((" + rowGroup.scalar + ") * (" + blocks.scalar + ") + (" +
      block.scalar + ")) * " + std::to_string(plan.getRecordBytes()) +
      " + (" + byte.scalar + ")) * " +
      std::to_string(plan.getInterleaveRows()) + ")";
  const std::string packed = fresh("pack_bytes");
  line(transferType + " " + packed + " = __riscv_vlse8_v_" +
       transferSuffix + "((const uint8_t *)(" + sourceAddress +
       "), (ptrdiff_t)(" + rowStride->second + "), " + transferVLName +
       ");");
  line("__riscv_vse8_v_" + transferSuffix + "((uint8_t *)(" + storage.scalar +
       " + " + targetAddress + "), " + packed + ", " + transferVLName +
       ");");
  line("if (" + transferVLName + " < " +
       std::to_string(plan.getInterleaveRows()) + ") {");
  ++indent;
  const std::string tail = fresh("pack_tail");
  const std::string zeros = fresh("pack_zeros");
  line("const size_t " + tail + " = " +
       std::to_string(plan.getInterleaveRows()) + " - " + transferVLName +
       ";");
  line(transferType + " " + zeros + " = __riscv_vmv_v_x_" + transferSuffix +
       "(0, " + tail + ");");
  line("__riscv_vse8_v_" + transferSuffix + "((uint8_t *)(" + storage.scalar +
       " + " + targetAddress + " + " + transferVLName + "), " + zeros +
       ", " + tail + ");");
  --indent;
  line("}");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileSpill(riscv::SpillOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  Binding slot = bindings.lookup(operation.getSlot());
  const bool rvvSpill =
      operation.getLeaf().getEngine() == "transfer" &&
      operation.getLeaf().getFamily() == "spill" &&
      operation.getLeaf().getInstruction() == "rvv.spill";
  const bool scalarTupleSpill =
      operation.getLeaf().getEngine() == "transfer" &&
      operation.getLeaf().getFamily() == "spill" &&
      operation.getLeaf().getInstruction() == "scalar.tuple-spill";
  if ((!rvvSpill && !scalarTupleSpill) ||
      slot.kind != Binding::Kind::LocalArray || slot.scalar.empty())
    return fail(operation, "spill destination has no typed local allocation");
  riscv::LocalType slotType = operation.getSlot().getType();
  mlir::Type element = riscv_internal::logicalElement(operation.getInput().getType());
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "spill value has no C element type");
  if (input.kind == Binding::Kind::Slice || input.kind == Binding::Kind::Field) {
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(operation.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if (input.kind == Binding::Kind::Vector) {
    if (!rvvSpill)
      return fail(operation, "vector spill requires the selected RVV spill leaf");
    auto inputType =
        mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
    const int64_t parts = vectorPartCount(operation.getInput());
    if (!inputType || parts <= 0 ||
        input.parts.size() != static_cast<size_t>(parts) ||
        slotType.getSizeBytes() % parts)
      return fail(operation, "vector spill byte partition is incomplete");
    const int64_t bytes = slotType.getSizeBytes() / parts;
    const unsigned bits = inputType.getLayout().getSew();
    const std::string suffix = vectorSuffix(operation.getInput());
    for (int64_t part = 0; part < parts; ++part)
      line("__riscv_vse" + std::to_string(bits) + "_v_" + suffix +
           "((" + *scalarType + " *)(" + slot.scalar + " + " +
           std::to_string(part * bytes) + "), " + input.parts[part] + ", " +
           partVL(operation.getInput(), part) + ");");
    return mlir::success();
  }
  if (input.kind == Binding::Kind::Scalar ||
      input.kind == Binding::Kind::ScalarTuple) {
    if (!scalarTupleSpill)
      return fail(operation,
                  "scalar register spill requires the selected tuple-spill leaf");
    llvm::SmallVector<std::string> values;
    if (input.kind == Binding::Kind::Scalar)
      values.push_back(input.scalar);
    else
      values.assign(input.parts.begin(), input.parts.end());
    for (auto [index, value] : llvm::enumerate(values))
      line("((" + *scalarType + " *)" + slot.scalar + ")[" +
           std::to_string(index) + "] = " + value + ";");
    return mlir::success();
  }
  return fail(operation, "spill supports only typed RVV/scalar register values");
}

mlir::LogicalResult Emitter::compileReload(riscv::ReloadOp operation) {
  Binding slot = bindings.lookup(operation.getSlot());
  if (operation.getLeaf().getEngine() != "transfer" ||
      operation.getLeaf().getFamily() != "reload" ||
      operation.getLeaf().getInstruction() != "rvv.reload" ||
      slot.kind != Binding::Kind::LocalArray || slot.scalar.empty())
    return fail(operation, "reload source has no typed local allocation");
  auto resultType = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (!resultType)
    return fail(operation, "reload result has no typed physical value");
  auto kind = selectedBindingKind(operation.getResult());
  auto scalarType = scalarCType(resultType.getElementType());
  if (!kind || !scalarType)
    return fail(operation, "reload result has no terminal representation");
  Binding result;
  result.kind = *kind;
  if (*kind == Binding::Kind::Vector) {
    riscv::LocalType slotType = operation.getSlot().getType();
    const int64_t parts = vectorPartCount(operation.getResult());
    if (parts <= 0 || slotType.getSizeBytes() % parts)
      return fail(operation, "vector reload byte partition is incomplete");
    const int64_t bytes = slotType.getSizeBytes() / parts;
    const unsigned bits = resultType.getLayout().getSew();
    const std::string suffix = vectorSuffix(operation.getResult());
    const std::string type = vectorType(operation.getResult());
    auto partIsUsed = [&](int64_t part) {
      for (mlir::OpOperand &use : operation.getResult().getUses()) {
        if (auto slice = mlir::dyn_cast<riscv::RVVIssueSliceOp>(use.getOwner())) {
          if (llvm::is_contained(slice.getSourceParts(), part))
            return true;
          continue;
        }
        auto set = mlir::dyn_cast<riscv::RVVPartialSetOp>(use.getOwner());
        if (!set)
          return true;
        const size_t operand = use.getOperandNumber();
        const size_t lhsCount = set.getLhs().size();
        if (operand >= lhsCount + set.getRhs().size())
          return true;
        for (int64_t slot = 0; slot < set.getResult().getType().getSlots(); ++slot) {
          if (operand < lhsCount) {
            if (set.getLhsOperands()[slot] == static_cast<int64_t>(operand) &&
                set.getLhsParts()[slot] == part)
              return true;
          } else if (set.getRhsOperands()[slot] ==
                         static_cast<int64_t>(operand - lhsCount) &&
                     set.getRhsParts()[slot] == part) {
            return true;
          }
        }
      }
      return false;
    };
    for (int64_t part = 0; part < parts; ++part) {
      if (!partIsUsed(part)) {
        result.parts.emplace_back();
        continue;
      }
      std::string loaded = fresh("reload");
      line(type + " " + loaded + " = __riscv_vle" + std::to_string(bits) +
           "_v_" + suffix + "((const " + *scalarType + " *)(" + slot.scalar +
           " + " + std::to_string(part * bytes) + "), " +
           partVL(operation.getResult(), part) + ");");
      result.parts.push_back(std::move(loaded));
    }
  } else if (*kind == Binding::Kind::Scalar) {
    result.scalar = materializeScalarRead(
        *scalarType, "((const " + *scalarType + " *)" + slot.scalar + ")[0]",
        "reload");
  } else if (*kind == Binding::Kind::ScalarTuple) {
    for (int64_t part = 0; part < scalarPartCount(operation.getResult()); ++part)
      result.parts.push_back(materializeScalarRead(
          *scalarType, "((const " + *scalarType + " *)" + slot.scalar +
                           ")[" + std::to_string(part) + "]",
          "reload"));
  } else {
    return fail(operation, "reload supports only typed RVV/scalar values");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileAtomicRMW(riscv::AtomicRMWOp operation) {
  Binding value = bindings.lookup(operation.getValue());
  Binding region = bindings.lookup(operation.getRegion());
  if (instructionOf(operation) != "scalar.atomic-add" ||
      value.kind != Binding::Kind::Scalar || region.kind != Binding::Kind::Slice)
    return fail(operation, "atomic update requires its selected scalar value and address");
  auto address = denseAddress(region.slice, {});
  auto type = scalarCType(operation.getResult().getType());
  if (!address || !type)
    return fail(operation, "atomic update has no dense address or scalar C type");
  llvm::StringRef ordering = operation.getOrdering();
  llvm::StringRef nativeOrder = ordering == "relaxed" ? "__ATOMIC_RELAXED"
      : ordering == "acquire" ? "__ATOMIC_ACQUIRE"
      : ordering == "release" ? "__ATOMIC_RELEASE"
      : ordering == "acq_rel" ? "__ATOMIC_ACQ_REL" : "__ATOMIC_SEQ_CST";
  std::string old = fresh("atomic_old");
  line(*type + " " + old + " = __atomic_fetch_add((" + *type + " *)(" +
       *address + "), " + value.scalar + ", " + nativeOrder.str() + ");");
  Binding result;
  result.kind = Binding::Kind::Scalar;
  result.scalar = std::move(old);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileCommit(riscv::StoreOp operation) {
  auto materialized = materializeNumeric(
      operation.getValue(), bindings.lookup(operation.getValue()));
  if (mlir::failed(materialized))
    return mlir::failure();
  Binding value = std::move(*materialized);
  Binding region = bindings.lookup(operation.getRegion());
  riscv::AccessAttr selectedEdge = operation.getAccess();
  if (region.kind == Binding::Kind::Field) {
    return compileFieldCommit(operation, value, region);
  }
  if (region.kind != Binding::Kind::Slice)
    return fail(operation, "commit requires an explicit View slice");
  const Binding &memory = bindings.lookup(region.slice.base);
  if (memory.kind != Binding::Kind::Memory)
    return fail(operation, "commit slice is not memory-backed");
  mlir::Type element =
      riscv_internal::logicalElement(operation.getValue().getType());
  if (!memory.memory.elementType || memory.memory.elementType != element)
    return fail(operation, "commit value and dense memory element types disagree");
  if (value.kind == Binding::Kind::Scalar) {
    auto address = denseAddress(region.slice, {});
    auto type = scalarCType(element);
    if (!address || !type)
      return fail(operation, "scalar commit has no dense address or C type");
    std::string statement = "*(" + *type + " *)(" + *address + ") = " + value.scalar + ";";
    if (auto shaped = mlir::dyn_cast<riscv::ValueType>(operation.getValue().getType());
        shaped && !shaped.getShape().empty()) {
      std::string active = "1";
      for (int64_t axis : shaped.getAxisIds().asArrayRef()) {
        auto extent = descriptorExtent(operation.getRegion(), axis);
        if (!extent)
          return fail(operation, "singleton commit destination has no explicit extent");
        active += " && (" + *extent + ") > 0";
      }
      statement = "if (" + active + ") " + statement;
    }
    line(statement);
    return mlir::success();
  }
  if (value.kind == Binding::Kind::ScalarTuple) {
    auto type = scalarCType(element);
    llvm::SmallVector<int64_t, 4> axes(layoutOf(operation.getValue()).getAxisIds().asArrayRef());
    if (!type || value.parts.size() !=
                     static_cast<size_t>(scalarPartCount(operation.getValue())))
      return fail(operation,
                  "dense scalar tuple commit has no complete physical mapping");
    for (size_t part = 0; part < value.parts.size(); ++part) {
      auto coordinates = scalarCoordinates(operation.getValue(), part);
      if (!coordinates || coordinates->size() != axes.size())
        return fail(operation,
                    "dense scalar tuple commit has no coordinate projection");
      llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
      std::string active = "1";
      for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
        offsets.push_back({axis, std::to_string(coordinate)});
        auto extent = descriptorExtent(operation.getRegion(), axis);
        if (!extent)
          return fail(operation, "scalar commit destination has no explicit extent");
        active += " && " + std::to_string(coordinate) + " < (" + *extent + ")";
        auto scope = axisScopes.find(axis);
        if (scope != axisScopes.end() && !scope->second.empty())
          active += " && " + std::to_string(coordinate) + " < " +
                    scope->second.back().active;
      }
      auto address = denseAddress(region.slice, offsets);
      if (!address)
        return fail(operation,
                    "dense scalar tuple commit has no address relation");
      line("if (" + active + ") *(" + *type + " *)(" + *address + ") = " +
           value.parts[part] + ";");
    }
    return mlir::success();
  }
  if (value.kind != Binding::Kind::Vector)
    return fail(operation, "dense commit has no selected numeric handoff");
  size_t laneDimension =
      llvm::find(memory.memory.axes, laneAxisFor(operation.getValue())) -
      memory.memory.axes.begin();
  if (laneDimension >= memory.memory.axes.size())
    return fail(operation, "commit destination has no selected lane axis");
  llvm::StringRef memoryForm =
      selectedEdge ? selectedEdge.getForm() : llvm::StringRef();
  if (memoryForm != "unit" && memoryForm != "strided")
    return fail(operation,
                "dense commit has no pass-selected unit or strided form");
  for (size_t part = 0; part < value.parts.size(); ++part) {
    auto offsets = partOffsets(operation.getValue(), part);
    auto active = partActive(operation.getValue(), part);
    if (!offsets || !active)
      return fail(operation,
                  "dense vector commit has no complete physical coordinates");
    std::string vl = partVL(operation.getValue(), part);
    for (const auto &[axis, coordinate] : *offsets) {
      auto extent = descriptorExtent(operation.getRegion(), axis);
      if (!extent)
        return fail(operation, "vector commit destination has no explicit extent");
      auto scope = axisScopes.find(axis);
      if (layoutOf(operation.getValue()).getValidity() == "tail" &&
          scope != axisScopes.end() && !scope->second.empty() &&
          *extent == scope->second.back().active)
        continue; // partActive/partVL already spell this same bound.
      *active += " && (" + coordinate + ") < (" + *extent + ")";
      if (axis == laneAxisFor(operation.getValue())) {
        // Numeric validity cannot enlarge the destination memory domain.
        const std::string remaining = "((" + *extent + ") - (" + coordinate + "))";
        vl = "((" + vl + ") < " + remaining + " ? (" + vl + ") : " + remaining + ")";
      }
    }
    auto address = denseAddress(region.slice, *offsets);
    if (!address)
      return fail(operation, "commit destination has no address relation");
    const std::string suffix = vectorSuffix(operation.getValue());
    const unsigned width = operation.getRegion().getType().getStorageBits();
    auto elementType = scalarCType(element);
    if (!width || !elementType)
      return fail(operation, "vector commit element has no fixed C representation");
    std::string statement;
    if (memoryForm == "unit")
      statement = "__riscv_vse" + std::to_string(width) + "_v_" + suffix +
                  "(" + *address + ", " +
                  value.parts[part] + ", " +
                  vl + ");";
    else
      statement = "__riscv_vsse" + std::to_string(width) + "_v_" + suffix +
                  "(" + *address + ", " +
                  memory.memory.strides[laneDimension] +
                  " * (ptrdiff_t)sizeof(" + *elementType + "), " +
                  value.parts[part] + ", " +
                  vl + ");";
    if (*active != "1")
      line("if (" + *active + ") " + statement);
    else
      line(statement);
  }
  return mlir::success();
}

} // namespace weft::riscv_emission
