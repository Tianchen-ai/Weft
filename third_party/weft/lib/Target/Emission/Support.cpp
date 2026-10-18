#include "Emitter.h"

namespace weft::riscv_emission {

std::string identifier(llvm::StringRef source) {
  std::string result;
  result.reserve(source.size() + 1);
  for (char character : source)
    result.push_back(std::isalnum(static_cast<unsigned char>(character))
                         ? character
                         : '_');
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result[0])))
    result.insert(result.begin(), '_');
  return result;
}

std::string arrayString(mlir::ArrayAttr values, llvm::StringRef separator) {
  std::string result;
  for (auto [index, attribute] : llvm::enumerate(values)) {
    if (index)
      result += separator;
    result += mlir::cast<mlir::StringAttr>(attribute).getValue().str();
  }
  return result;
}

std::optional<StorageFragment>
singleStorageFragment(const EncodingField &field, llvm::StringRef logicalIndex,
                      unsigned logicalWidth) {
  if (!field.access)
    return std::nullopt;
  llvm::StringRef kind = field.access.getMapping();
  if (kind == "natural") {
    std::string bit = "(" + std::to_string(field.bitOffset) + " + (" +
                      logicalIndex.str() + ") * " +
                      std::to_string(logicalWidth) + ")";
    return StorageFragment{"(" + bit + " / 8)", "(" + bit + " % 8)",
                           logicalWidth};
  }
  if (kind != "grouped_layered")
    return std::nullopt;
  int64_t group = field.access.getGroupSize();
  int64_t layer = field.access.getLayerSize();
  llvm::StringRef order = field.access.getOrder();
  int64_t layers = group / layer;
  std::string within = "((" + logicalIndex.str() + ") % " +
                       std::to_string(group) + ")";
  std::string layerIndex = "(" + within + " / " + std::to_string(layer) + ")";
  if (order == "hi_first")
    layerIndex = "(" + std::to_string(layers - 1) + " - " + layerIndex + ")";
  std::string byte = "(" + std::to_string(field.bitOffset / 8) + " + ((" +
                     logicalIndex.str() + ") / " + std::to_string(group) + ") * " +
                     std::to_string(layer) + " + " + within + " % " +
                     std::to_string(layer) + ")";
  std::string shift = "(" + layerIndex + " * " +
                      std::to_string(logicalWidth) + ")";
  return StorageFragment{byte, shift, logicalWidth};
}

std::string lmulSpelling(int64_t eighths) {
  switch (eighths) {
  case 1:
    return "mf8";
  case 2:
    return "mf4";
  case 4:
    return "mf2";
  case 8:
    return "m1";
  case 16:
    return "m2";
  case 32:
    return "m4";
  case 64:
    return "m8";
  default:
    return {};
  }
}

std::string vectorSuffixFor(riscv::ValueType value) {
  char category = 'i';
  if (usesRawFloatStorage(value.getElementType()))
    category = 'u';
  else if (mlir::isa<mlir::FloatType>(value.getElementType()))
    category = 'f';
  else if (auto integer =
               mlir::dyn_cast<mlir::IntegerType>(value.getElementType()))
    category = integer.isUnsigned() ? 'u' : 'i';
  return std::string(1, category) +
         std::to_string(value.getLayout().getSew()) +
         lmulSpelling(value.getLayout().getLmulEighths());
}

std::string vectorTypeFor(riscv::ValueType value) {
  std::string suffix = vectorSuffixFor(value);
  const std::string prefix = suffix.front() == 'f'
                                 ? "vfloat"
                                 : suffix.front() == 'u' ? "vuint" : "vint";
  return prefix + suffix.substr(1) + "_t";
}

int64_t physicalLanesFor(riscv::ValueType value) {
  int64_t lanes = value.getLayout().getReplicaGroup();
  for (int64_t factor : value.getLayout().getLaneFactors().asArrayRef())
    lanes *= factor;
  return std::max<int64_t>(1, lanes);
}

int64_t product(mlir::DenseI64ArrayAttr values) {
  return riscv_internal::staticProduct(values.asArrayRef()).value_or(0);
}

riscv::AccessAttr accessOf(mlir::Operation *operation) {
  return operation ? operation->getAttrOfType<riscv::AccessAttr>("access")
                   : riscv::AccessAttr();
}

riscv::LeafAttr leafOf(mlir::Operation *operation) {
  return operation ? operation->getAttrOfType<riscv::LeafAttr>("leaf")
                   : riscv::LeafAttr();
}

} // namespace weft::riscv_emission
