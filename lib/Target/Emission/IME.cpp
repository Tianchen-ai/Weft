#include "Emitter.h"
#include "../IME/FragmentEmission.h"
#include "Weft/Target/RISCVFragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace weft::riscv_emission {

std::optional<std::string> fragmentVectorType(riscv::FragmentType fragment) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(fragment.getElementType());
  if (!integer || !integer.isSigned() || !findRISCVFragmentCapability(fragment.getFamily()))
    return std::nullopt;
  const int64_t groups = riscv::fragmentPartResourceGroups(fragment);
  if (groups != 1 && groups != 2 && groups != 4 && groups != 8)
    return std::nullopt;
  return "vint" + std::to_string(integer.getWidth()) + "m" + std::to_string(groups) + "_t";
}

mlir::LogicalResult Emitter::compileIMEZero(riscv::IMEZeroOp operation) {
  auto fragment = operation.getResult().getType();
  auto type = fragmentVectorType(fragment);
  if (!type)
    return fail(operation, "fragment zero has no selected register carrier");
  const int64_t bits = riscv_internal::logicalBitWidth(fragment.getElementType());
  const int64_t groups = riscv::fragmentPartResourceGroups(fragment);
  const int64_t elements = fragment.getPacking().getRowsPerTile() *
                           fragment.getPacking().getColumnsPerTile();
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  const bool prefix = operation.getLeaf().getInstruction() == "ime.zero.local-prefix";
  const bool local = prefix || operation.getLeaf().getInstruction() == "ime.zero.local";
  if (!local && operation.getLeaf().getInstruction() != "ime.zero")
    return fail(operation, "fragment zero has no exact selected local realization");
  if (local) {
    for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
      result.parts.push_back(fresh("ime_zero"));
      line(*type + " " + result.parts.back() + ";");
    }
    line("__asm__ volatile(");
    const std::string lmul = prefix ? lmulSpelling(operation.getLeaf().getParameters()[0])
                                    : "m" + std::to_string(groups);
    line("  \"vsetvli zero, %[vl], e" + std::to_string(bits) + ", " + lmul + ", ta, ma\\n\\t\"");
    std::string outputs = "  : ";
    for (auto [part, name] : llvm::enumerate(result.parts)) {
      line("  \"vmv.v.i %[part" + std::to_string(part) + "], 0\\n\\t\"");
      if (part) outputs += ", ";
      outputs += "[part" + std::to_string(part) + "] \"=vr\"(" + name + ")";
    }
    line(outputs);
    line("  : [vl] \"r\"((size_t)" + std::to_string(elements) + ") : \"vl\", \"vtype\");");
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    result.parts.push_back(fresh("ime_zero"));
    line(*type + " " + result.parts.back() + " = __riscv_vmv_v_x_i" +
         std::to_string(bits) + "m" + std::to_string(groups) +
         "(0, " + std::to_string(elements) + ");");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEPack(riscv::IMEPackOp operation) {
  auto input = materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  auto inputType = mlir::cast<riscv::ValueType>(operation.getInput().getType());
  auto fragment = operation.getResult().getType();
  auto type = fragmentVectorType(fragment);
  const int64_t rows = operation.getPacking().getRowsPerTile();
  const int64_t columns = operation.getPacking().getColumnsPerTile();
  const int64_t parts = riscv::fragmentPartCount(fragment);
  if (!type || input->kind != Binding::Kind::Vector ||
      static_cast<int64_t>(input->parts.size()) != rows * parts ||
      riscv::fragmentPartResourceGroups(fragment) != 1)
    return fail(operation, "fragment pack has no complete selected register slices");
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  const std::string suffix = vectorSuffix(operation.getInput());
  const bool full = inputType.getLayout().getValidity() == "full";
  for (int64_t atom = 0; atom < parts; ++atom) {
    const std::string packed = fresh("ime_pack");
    if (full) {
      std::string first = input->parts[atom * rows];
      if (suffix != "i8m1")
        first = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + first + ")";
      line(*type + " " + packed + " = " + first + ";");
    } else {
      line(*type + " " + packed + " = __riscv_vmv_v_x_i8m1(0, " +
           std::to_string(kernel.getTarget().getVlenBits() / 8) + ");");
    }
    for (int64_t row = full ? 1 : 0; row < rows; ++row) {
      const int64_t sourcePart = atom * rows + row;
      std::string part = input->parts[sourcePart];
      if (suffix != "i8m1")
        part = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + part + ")";
      const std::string offset = std::to_string(row * columns);
      // Full slices overwrite every logical cell by the final insert. Tail
      // slices keep zero padding outside their explicitly active prefix.
      const std::string vl = full ? std::to_string(rows * columns)
          : offset + " + " + partVL(operation.getInput(), sourcePart);
      line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + part +
           ", " + offset + ", " + vl + ");");
    }
    result.parts.push_back(packed);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentStorageLoad(riscv::IMEFragmentStorageLoadOp operation) {
  Binding projection = bindings.lookup(operation.getInput());
  auto field = fieldFor(projection);
  if (projection.kind != Binding::Kind::Field || !field || !projection.field.relativeIndices.empty())
    return fail(operation, "fragment load requires its selected field projection binding");
  Binding owner = bindings.lookup(projection.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(projection.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0)
    return fail(operation, "fragment field load has no declared record binding");
  std::string logicalIndex = "0";
  if (projection.field.index) {
    Binding point = bindings.lookup(*projection.field.index);
    if (point.kind != Binding::Kind::Point)
      return fail(operation, "fragment field load has no selected inner point");
    logicalIndex = "(" + point.point.base + " % " + std::to_string(owner.recordElements) + ")";
  }
  const int64_t bits = riscv_internal::logicalBitWidth(field->type);
  auto storage = singleStorageFragment(*field, logicalIndex, bits);
  if (!storage)
    return fail(operation, "fragment field load has no closed byte projection");
  auto fragment = operation.getResult().getType();
  const int64_t cells = fragment.getPacking().getRowsPerTile() * fragment.getPacking().getColumnsPerTile();
  const auto instruction = operation.getLeaf().getInstruction();
  const bool grouped = instruction == "ime.field-raw-load.grouped" ||
                       instruction == "ime.field-raw-load.grouped.scalar-prime";
  if (instruction == "ime.field-raw-load.scalar-prime" ||
      instruction == "ime.field-raw-load.grouped.scalar-prime") {
    const int64_t finalByte = cells * riscv::fragmentPartCount(fragment) - 1;
    line("__asm__ volatile(\"lb zero, " + std::to_string(finalByte) +
         "(%0)\" : : \"r\"((const uint8_t *)" +
         fieldStorageAddress(owner, *field, storage->byte) + ") : \"memory\");");
  }
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  if (grouped) {
    const int64_t parts = riscv::fragmentPartCount(fragment);
    const std::string suffix = "i8m" + std::to_string(parts);
    const std::string raw = fresh("ime_field_group");
    line("vint8m" + std::to_string(parts) + "_t " + raw + " = __riscv_vle8_v_" + suffix +
         "((const int8_t *)" + fieldStorageAddress(owner, *field, storage->byte) + ", " +
         std::to_string(cells * parts) + ");");
    for (int64_t part = 0; part < parts; ++part) {
      result.parts.push_back(fresh("ime_field"));
      line("vint8m1_t " + result.parts.back() + " = __riscv_vget_v_" + suffix + "_i8m1(" +
           raw + ", " + std::to_string(part) + ");");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
  const std::string raw = fresh("ime_field_bytes");
  const std::string address = "((const uint8_t *)" + fieldStorageAddress(owner, *field, storage->byte) +
                              " + " + std::to_string(part * cells) + ")";
  line("vuint8m1_t " + raw + " = __riscv_vle8_v_u8m1((const uint8_t *)" + address + ", " + std::to_string(cells) + ");");
  const std::string expression = "__riscv_vreinterpret_v_u8m1_i8m1(" + raw + ")";
  result.parts.push_back(fresh("ime_field"));
  line("vint8m1_t " + result.parts.back() + " = " + expression + ";");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentWindowLoad(riscv::IMEFragmentWindowLoadOp operation) {
  Binding projection = bindings.lookup(operation.getField());
  Binding base = bindings.lookup(operation.getLogicalBase());
  auto field = fieldFor(projection);
  if (projection.kind != Binding::Kind::Field || !field || base.kind != Binding::Kind::Scalar)
    return fail(operation, "fragment byte windows require their selected field and logical base");
  Binding owner = bindings.lookup(projection.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(projection.field.owner);
    if (mlir::failed(record)) return mlir::failure();
    owner = std::move(*record);
  }
  auto fragment = operation.getResult().getType();
  auto packing = fragment.getPacking();
  const int64_t rowAxis = fragment.getAxisIds()[packing.getAxisOrder()[0]];
  auto stride = llvm::find_if(owner.recordByteStrides,
      [&](const auto &entry) { return entry.first == rowAxis; });
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows ||
      stride == owner.recordByteStrides.end())
    return fail(operation, "fragment byte windows lost their independent record stride");
  const int64_t rows = packing.getRowsPerTile(), columns = packing.getColumnsPerTile();
  const std::string lmul = lmulSpelling(columns * 64 / kernel.getTarget().getVlenBits());
  const int64_t group = operation.getAccess().getGroupSize(), layer = operation.getAccess().getLayerSize();
  const std::string byte = "(" + std::to_string(operation.getAccess().getBitOffset() / 8) +
      " + (" + base.scalar + ") / " + std::to_string(group) + " * " + std::to_string(layer) +
      " + (" + base.scalar + ") % " + std::to_string(layer) + ")";
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    const std::string packed = fresh("ime_byte_windows");
    bool words = operation.getLeaf().getInstruction() == "ime.field-byte-windows.guarded-vlse64";
    if (words) {
      const std::string address = "(" + owner.recordPointer + " + " +
          std::to_string(part * rows) + " * " + stride->second + " + " + byte + ")";
      line("vint8m1_t " + packed + ";");
      line("if ((((uintptr_t)" + address + " | (uintptr_t)(" + stride->second + ")) & 7u) == 0) {");
      const std::string loaded = fresh("ime_word_windows");
      line("  vuint64m1_t " + loaded + " = __riscv_vlse64_v_u64m1((const uint64_t *)" + address +
          ", (ptrdiff_t)(" + stride->second + "), " + std::to_string(rows) + ");");
      line("  " + packed + " = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(" + loaded + "));");
      line("} else {");
    }
    llvm::SmallVector<std::string> windows;
    for (int64_t row = 0; row < rows; ++row) {
      const std::string address = "((const int8_t *)(" + owner.recordPointer + " + " +
          std::to_string(part * rows + row) + " * " + stride->second + " + " + byte + "))";
      const std::string window = fresh("ime_byte_window");
      line("vint8" + lmul + "_t " + window + " = __riscv_vle8_v_i8" + lmul +
          "(" + address + ", " + std::to_string(columns) + ");");
      windows.push_back("__riscv_vlmul_ext_v_i8" + lmul + "_i8m1(" + window + ")");
    }
    line((words ? "" : "vint8m1_t ") + packed + " = " + windows.front() + ";");
    for (int64_t row = 1; row < rows; ++row)
      line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + windows[row] + ", " +
          std::to_string(row * columns) + ", " + std::to_string(rows * columns) + ");");
    if (words) line("}");
    result.parts.push_back(packed);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentDenseLoad(
    riscv::IMEFragmentDenseLoadOp operation) {
  auto fragment = operation.getResult().getType();
  auto packing = fragment.getPacking();
  int64_t rowAxis = fragment.getAxisIds()[packing.getAxisOrder()[0]];
  auto address = byteReadAddress(operation.getSource());
  auto stride = descriptorStride(operation.getSource(), rowAxis);
  if (!address || !stride ||
      operation.getLeaf().getInstruction() != "ime.dense-row-words")
    return fail(operation, "dense fragment transfer lost its exact selected memory binding");
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  const int64_t rows = packing.getRowsPerTile();
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    const std::string words = fresh("ime_dense_words");
    const std::string at = "((const uint8_t *)(" + *address + ") + " +
        std::to_string(part * rows) + " * (" + *stride + "))";
    line("vuint64m1_t " + words + " = __riscv_vlse64_v_u64m1((const uint64_t *)" +
         at + ", (ptrdiff_t)(" + *stride + "), " + std::to_string(rows) + ");");
    result.parts.push_back(fresh("ime_dense_fragment"));
    line("vint8m1_t " + result.parts.back() +
         " = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(" +
         words + "));");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentDecode(riscv::IMEFragmentDecodeOp operation) {
  const Binding input = bindings.lookup(operation.getInput());
  const Binding shift = bindings.lookup(operation.getShift());
  auto fragment = operation.getResult().getType();
  if (input.kind != Binding::Kind::Fragment || shift.kind != Binding::Kind::Scalar ||
      input.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)))
    return fail(operation, "fragment decode requires its raw-byte SSA and bounded shift");
  const int64_t bits = operation.getElementBits();
  auto constant = operation.getShift().getDefiningOp<mlir::arith::ConstantIndexOp>();
  const std::optional<int64_t> knownShift = constant
      ? std::optional<int64_t>(constant.value()) : std::nullopt;
  const bool grouped = operation.getLeaf().getInstruction() == "ime.decode.grouped-bits";
  const int64_t parts = riscv::fragmentPartCount(fragment);
  const std::string group = "m" + std::to_string(grouped ? parts : 1);
  const std::string vl = std::to_string(fragment.getPacking().getRowsPerTile() *
      fragment.getPacking().getColumnsPerTile() * (grouped ? parts : 1));
  llvm::SmallVector<std::string> operands(input.parts.begin(), input.parts.end());
  if (grouped) {
    const std::string packed = fresh("ime_decode_input");
    line("vint8" + group + "_t " + packed + " = __riscv_vundefined_i8" + group + "();");
    for (auto [part, value] : llvm::enumerate(input.parts))
      line(packed + " = __riscv_vset_v_i8m1_i8" + group + "(" + packed + ", " +
           std::to_string(part) + ", " + value + ");");
    operands = {packed};
  }
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  for (const std::string &part : operands) {
    std::string expression = "__riscv_vreinterpret_v_i8" + group + "_u8" + group + "(" + part + ")";
    if (knownShift != 0)
      expression = "__riscv_vsrl_vx_u8" + group + "(" + expression + ", " + shift.scalar + ", " + vl + ")";
    if (knownShift != 8 - bits)
      expression = "__riscv_vand_vx_u8" + group + "(" + expression + ", " +
                   std::to_string((1u << bits) - 1) + ", " + vl + ")";
    expression = "__riscv_vreinterpret_v_u8" + group + "_i8" + group + "(" + expression + ")";
    if (operation.getSignedElements())
      expression = "__riscv_vsra_vx_i8" + group + "(__riscv_vsll_vx_i8" + group + "(" + expression + ", " +
                   std::to_string(8 - bits) + ", " + vl + "), " + std::to_string(8 - bits) + ", " + vl + ")";
    const std::string decoded = fresh("ime_decode");
    line("vint8" + group + "_t " + decoded + " = " + expression + ";");
    if (grouped) {
      for (int64_t index = 0; index < parts; ++index) {
        result.parts.push_back(fresh("ime_decode_part"));
        line("vint8m1_t " + result.parts.back() + " = __riscv_vget_v_i8" + group +
             "_i8m1(" + decoded + ", " + std::to_string(index) + ");");
      }
    } else {
      result.parts.push_back(decoded);
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentRowBroadcast(riscv::IMEFragmentRowBroadcastOp operation) {
  auto input = materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector || input->parts.size() != 1)
    return fail(operation, "fragment broadcast has no selected row input or local index sequence");
  auto fragment = operation.getResult().getType();
  const int64_t rows = fragment.getPacking().getRowsPerTile();
  const int64_t parts = riscv::fragmentPartCount(fragment);
  const std::string suffix = vectorSuffix(operation.getInput());
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  if (operation.getLeaf().getInstruction() == "ime.rows.byte-repeat") {
    const std::string byteSuffix = "u8" + lmulSpelling(operation.getInput().getType().getLayout().getLmulEighths());
    const std::string wordSuffix = "u64" + lmulSpelling(parts * 8);
    const std::string wideByteSuffix = "u8" + lmulSpelling(parts * 8);
    const std::string repeatedSuffix = "i8" + lmulSpelling(parts * 8);
    const std::string vl = std::to_string(rows * parts);
    const std::string words = fresh("row_words");
    line("vuint64" + lmulSpelling(parts * 8) + "_t " + words + " = __riscv_vzext_vf8_" +
         wordSuffix + "(__riscv_vreinterpret_v_" + suffix + "_" + byteSuffix + "(" +
         input->parts.front() + "), " + vl + ");");
    const std::string repeated = fresh("row_bytes");
    line("vint8" + lmulSpelling(parts * 8) + "_t " + repeated + " = __riscv_vreinterpret_v_" +
         wideByteSuffix + "_" + repeatedSuffix + "(__riscv_vreinterpret_v_" + wordSuffix + "_" +
         wideByteSuffix + "(__riscv_vmul_vx_" + wordSuffix + "(" + words +
         ", UINT64_C(0x0101010101010101), " + vl + ")));");
    for (int64_t part = 0; part < parts; ++part) {
      result.parts.push_back(fresh("fragment_rows"));
      const std::string value = parts == 1 ? repeated : "__riscv_vget_v_" + repeatedSuffix +
          "_i8m1(" + repeated + ", " + std::to_string(part) + ")";
      line("vint8m1_t " + result.parts.back() + " = " + value + ";");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (operation.getLeaf().getInstruction() == "ime.rows.indices") {
    std::string source = input->parts.front();
    if (suffix != "i8m1")
      source = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + source + ")";
    const std::string vl = std::to_string(rows * 8);
    const std::string base = fresh("row_index");
    line("vuint8m1_t " + base + " = __riscv_vsrl_vx_u8m1(__riscv_vid_v_u8m1(" +
         vl + "), 3, " + vl + ");");
    for (int64_t part = 0; part < parts; ++part) {
      std::string index = base;
      if (part) {
        index = fresh("row_index");
        line("vuint8m1_t " + index + " = __riscv_vadd_vx_u8m1(" + base + ", " +
             std::to_string(part * rows) + ", " + vl + ");");
      }
      result.parts.push_back(fresh("fragment_rows"));
      line("vint8m1_t " + result.parts.back() + " = __riscv_vrgather_vv_i8m1(" +
           source + ", " + index + ", " + vl + ");");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (operation.getLeaf().getInstruction() != "ime.rows.local")
    return fail(operation, "fragment broadcast has no selected local realization");
  std::string source = input->parts.front();
  if (suffix != "i8m1") source = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + source + ")";
  for (int64_t part = 0; part < parts; ++part) {
    result.parts.push_back(fresh("fragment_rows"));
    line("vint8m1_t " + result.parts.back() + ";");
  }
  auto index = fresh("row_index");
  line("vuint8m1_t " + index + ";");
  line("__asm__ volatile(");
  line("  \"vsetvli zero, %[vl], e8, m1, ta, ma\\n\\tvid.v %[index]\\n\\tvsrl.vi %[index], %[index], 3\\n\\t\"");
  std::string outputs = "  : [index] \"=&vr\"(" + index + ")";
  for (int64_t part = 0; part < parts; ++part) {
    if (part)
      line("  \"vadd.vi %[index], %[index], " + std::to_string(rows) + "\\n\\t\"");
    line("  \"vrgather.vv %[part" + std::to_string(part) + "], %[source], %[index]\\n\\t\"");
    outputs += ", [part" + std::to_string(part) + "] \"=&vr\"(" + result.parts[part] + ")";
  }
  line(outputs);
  line("  : [source] \"vr\"(" + source + "), [vl] \"r\"((size_t)" + std::to_string(rows * 8) + ") : \"vl\", \"vtype\");");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentPointwise(riscv::IMEFragmentPointwiseOp operation) {
  auto lhs = bindings.lookup(operation.getLhs());
  auto rhs = materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (lhs.kind != Binding::Kind::Fragment || mlir::failed(rhs))
    return fail(operation, "fragment pointwise requires its selected numeric operands");
  auto fragment = operation.getResult().getType();
  const int64_t columns = fragment.getPacking().getColumnsPerTile();
  const std::string vl = std::to_string(fragment.getPacking().getRowsPerTile() * columns);
  if (lhs.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)))
    return fail(operation, "fragment pointwise lost its atom replicas");
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  if (operation.getLeaf().getInstruction() == ("ime.fragment." + operation.getKind() + ".grouped").str()) {
    if (rhs->kind != Binding::Kind::Fragment || rhs->parts.size() != lhs.parts.size())
      return fail(operation, "grouped fragment pointwise lost its matching fragment operand");
    const int64_t parts = riscv::fragmentPartCount(fragment);
    const std::string group = "m" + std::to_string(parts);
    auto pack = [&](llvm::ArrayRef<std::string> values) {
      auto packed = fresh("ime_pointwise_input");
      line("vint8" + group + "_t " + packed + " = __riscv_vundefined_i8" + group + "();");
      for (auto [part, value] : llvm::enumerate(values))
        line(packed + " = __riscv_vset_v_i8m1_i8" + group + "(" + packed + ", " +
             std::to_string(part) + ", " + value + ");");
      return packed;
    };
    const std::string left = pack(lhs.parts), right = pack(rhs->parts);
    line(left + " = __riscv_v" + operation.getKind().str() + "_vv_i8" + group + "(" +
         left + ", " + right + ", " + std::to_string(fragment.getPacking().getRowsPerTile() * columns * parts) + ");");
    for (int64_t part = 0; part < parts; ++part) {
      const bool reuse = operation.getLhsPolicy() == "reuse";
      result.parts.push_back(reuse ? lhs.parts[part] : fresh("ime_pointwise"));
      line((reuse ? "" : "vint8m1_t ") + result.parts.back() + " = __riscv_vget_v_i8" + group +
           "_i8m1(" + left + ", " + std::to_string(part) + ");");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
  std::string expression;
  if (rhs->kind == Binding::Kind::Scalar) {
    expression = "__riscv_v" + operation.getKind().str() + "_vx_i8m1(" + lhs.parts[part] + ", " + rhs->scalar + ", " + vl + ")";
  } else if (rhs->kind == Binding::Kind::Fragment && rhs->fragmentFamily == lhs.fragmentFamily &&
             rhs->parts.size() == lhs.parts.size()) {
    expression = "__riscv_v" + operation.getKind().str() + "_vv_i8m1(" + lhs.parts[part] + ", " + rhs->parts[part] + ", " + vl + ")";
  } else if (rhs->kind == Binding::Kind::Vector && rhs->parts.size() == 1) {
    const std::string index = fresh("ime_row_index");
    const std::string row = fresh("ime_rows");
    line("vuint8m1_t " + index + " = __riscv_vdivu_vx_u8m1(__riscv_vid_v_u8m1(" + vl + "), " + std::to_string(columns) + ", " + vl + ");");
    if (part)
      line(index + " = __riscv_vadd_vx_u8m1(" + index + ", " +
           std::to_string(part * fragment.getPacking().getRowsPerTile()) + ", " + vl + ");");
    const std::string suffix = vectorSuffix(operation.getRhs());
    const std::string values = suffix == "i8m1" ? rhs->parts.front()
        : "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + rhs->parts.front() + ")";
    line("vint8m1_t " + row + " = __riscv_vrgather_vv_i8m1(" + values + ", " + index + ", " + vl + ");");
    expression = "__riscv_v" + operation.getKind().str() + "_vv_i8m1(" + lhs.parts[part] + ", " + row + ", " + vl + ")";
  } else {
    return fail(operation, "fragment pointwise has no selected scalar/row broadcast binding");
  }
  if (operation.getLhsPolicy() == "reuse") {
    result.parts.push_back(lhs.parts[part]);
    line(result.parts.back() + " = " + expression + ";");
  } else if (operation.getLhsPolicy() == "copy") {
    result.parts.push_back(fresh("ime_pointwise"));
    line("vint8m1_t " + result.parts.back() + " = " + expression + ";");
  } else {
    return fail(operation, "fragment pointwise requires a selected lhs reuse policy");
  }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentMMA(riscv::IMEFragmentMMAOp operation) {
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  auto family = operation.getResult().getType().getFamily();
  auto valid = [&](const Binding &binding, riscv::FragmentType type) {
    return binding.kind == Binding::Kind::Fragment &&
           binding.parts.size() == static_cast<size_t>(riscv::fragmentPartCount(type)) &&
           binding.fragmentFamily == family;
  };
  if (!valid(lhs, operation.getLhs().getType()) || !valid(rhs, operation.getRhs().getType()) ||
      !valid(accumulator, operation.getAccumulator().getType()))
    return fail(operation, "IME MMA operands do not match the selected fragment leaf");
  auto freshName = [&](llvm::StringRef prefix) { return fresh(prefix); };
  auto emitLine = [&](llvm::StringRef text) { line(text.str()); };
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = family.str();
  auto output = emitFragmentMMA(operation, lhs.parts, rhs.parts,
                                accumulator.parts, {freshName, emitLine});
  if (mlir::failed(output))
    return mlir::failure();
  llvm::append_range(result.parts, *output);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEUnpack(riscv::IMEUnpackOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  auto fragment = operation.getInput().getType();
  if (input.kind != Binding::Kind::Fragment ||
      input.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)) ||
      input.fragmentFamily != fragment.getFamily())
    return fail(operation, "IME unpack input has no typed fragment binding");
  Binding result;
  result.kind = Binding::Kind::Vector;
  auto layout = operation.getResult().getType().getLayout();
  const std::string sourceSuffix = "i32m" + std::to_string(riscv::fragmentPartResourceGroups(fragment));
  const std::string suffix = vectorSuffix(operation.getResult());
  if (riscv::isDirectFragmentRVVHandoff(fragment, operation.getResult().getType())) {
    for (const std::string &part : input.parts)
      result.parts.push_back(sourceSuffix == suffix ? part
          : "__riscv_vlmul_trunc_v_" + sourceSuffix + "_" + suffix + "(" + part + ")");
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (riscv::isGroupedFragmentRVVHandoff(fragment, operation.getResult().getType())) {
    const int64_t group = layout.getReplicaGroup();
    for (size_t part = 0; part < input.parts.size(); part += group) {
      const std::string name = fresh("ime_group");
      line(vectorType(operation.getResult()) + " " + name +
           " = __riscv_vlmul_ext_v_" + sourceSuffix + "_" + suffix + "(" + input.parts[part] + ");");
      for (int64_t offset = 1; offset < group; ++offset)
        line(name + " = __riscv_vset_v_" + sourceSuffix + "_" + suffix + "(" + name + ", " +
             std::to_string(offset) + ", " + input.parts[part + offset] + ");");
      result.parts.push_back(name);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const int64_t streams = streamPartCount(operation.getResult());
  const int64_t atomRows = fragment.getPacking().getRowsPerTile();
  const int64_t atomColumns = fragment.getPacking().getColumnsPerTile();
  const int64_t columnAtoms = fragment.getShape()[1] / atomColumns;
  const int64_t lanes = layout.getLaneFactors()[1];
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    int64_t time = part % streams;
    int64_t replica = part / streams;
    llvm::SmallVector<int64_t, 2> coordinates(2);
    for (int64_t dimension = fragment.getShape().size() - 1; dimension >= 0; --dimension) {
      coordinates[dimension] = (time % layout.getTimeFactors()[dimension] *
          layout.getReplicaFactors()[dimension] + replica % layout.getReplicaFactors()[dimension]) *
          layout.getLaneFactors()[dimension];
      time /= layout.getTimeFactors()[dimension];
      replica /= layout.getReplicaFactors()[dimension];
    }
    const std::string name = fresh("ime_unpack");
    const std::string vl = partVL(operation.getResult(), part);
    for (int64_t inserted = 0; inserted < lanes;) {
      const int64_t column = coordinates[1] + inserted;
      const int64_t sourcePart = coordinates[0] / atomRows * columnAtoms + column / atomColumns;
      const int64_t offset = coordinates[0] % atomRows * atomColumns + column % atomColumns;
      const int64_t count = std::min(lanes - inserted, atomColumns - column % atomColumns);
      std::string expression = input.parts[sourcePart];
      if (offset)
        expression = "__riscv_vslidedown_vx_" + sourceSuffix + "(" + expression + ", " +
                     std::to_string(offset) + ", " + std::to_string(count) + ")";
      if (sourceSuffix != suffix)
        expression = "__riscv_vlmul_trunc_v_" + sourceSuffix + "_" + suffix + "(" + expression + ")";
      if (!inserted)
        line(vectorType(operation.getResult()) + " " + name + " = " + expression + ";");
      else
        line(name + " = __riscv_vslideup_vx_" + suffix + "_tu(" + name + ", " + expression +
             ", " + std::to_string(inserted) + ", " + vl + ");");
      inserted += count;
    }
    result.parts.push_back(name);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
