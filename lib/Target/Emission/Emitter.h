#ifndef WEFT_TARGET_EMISSION_EMITTER_H
#define WEFT_TARGET_EMISSION_EMITTER_H

#include "../RISCVIntrinsicC.h"
#include "../PhysicalDiagnostics.h"

#include "../RISCVPhysicalSupport.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include "Weft/Dialect/Kernel/IR/StrideABI.h"
#include "Weft/Target/RISCVCompiler.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace weft::riscv_emission {

inline bool usesRawFloatStorage(mlir::Type type) {
  return type.isBF16() || mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(type);
}
inline bool usesFloatingRegisters(mlir::Type type) {
  return mlir::isa<mlir::FloatType>(type) && !usesRawFloatStorage(type);
}

std::string identifier(llvm::StringRef source);

std::string arrayString(mlir::ArrayAttr values, llvm::StringRef separator);

struct EncodingField {
  std::string name;
  mlir::Type type;
  llvm::SmallVector<int64_t> shape;
  int64_t logicalRank = 0;
  riscv::AccessAttr access;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
};

struct EncodingInfo {
  std::string family;
  std::string kind;
  int64_t storageBits = 0;
  int64_t alignment = 1;
  int64_t logicalElements = 0;
  llvm::SmallVector<EncodingField> fields;
};

struct StorageFragment {
  std::string byte;
  std::string shift;
  unsigned width = 0;
};

std::optional<StorageFragment>
singleStorageFragment(const EncodingField &field, llvm::StringRef logicalIndex,
                      unsigned logicalWidth);

struct PointInfo {
  int64_t axis = 0;
  std::string base;
  std::string active;
  int64_t physicalExtent = 1;
};

struct MemoryInfo {
  std::string name;
  mlir::Type encoding;
  mlir::Type elementType;
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<std::string> extents;
  llvm::SmallVector<std::string> strides;
  llvm::SmallVector<std::string> origins;
  bool isConst = true;
  int64_t interleaveRows = 0;
  int64_t recordBytes = 0;
  int64_t logicalElements = 0;
};

struct SliceInfo {
  mlir::Value base;
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<std::string> selectors;
  llvm::SmallVector<std::pair<int64_t, mlir::Value>> localOffsets;
  llvm::SmallVector<std::pair<int64_t, int64_t>> staticOffsets;
  llvm::SmallVector<std::pair<int64_t, std::string>> extents;
};

struct FieldInfo {
  mlir::Value owner;
  std::string name;
  std::optional<mlir::Value> index;
  std::string selector;
  llvm::SmallVector<mlir::Value, 2> relativeIndices;
  riscv::AccessAttr storageAccess;
  riscv::AccessAttr useAccess;
  mlir::Type elementType;
  llvm::SmallVector<int64_t> shape;
  int64_t logicalRank = 0;
  int64_t regularBase = 0;
  int64_t regularStride = 0;
  int64_t regularRepeat = 0;
};

struct Binding {
  enum class Kind {
    None,
    Scalar,
    ScalarTuple,
    Vector,
    Mask,
    Memory,
    Slice,
    Record,
    Field,
    LocalArray,
    Fragment,
    Window,
    PartialSet,
    Point,
    RecordCohort,
    Domain,
  } kind = Kind::None;

  std::string scalar;
  std::string localElements;
  llvm::SmallVector<std::string> localExtents;
  llvm::SmallVector<std::string> parts;
  MemoryInfo memory;
  SliceInfo slice;
  FieldInfo field;
  PointInfo point;
  mlir::Value parentDomain;
  riscv::DomainType domainType;
  std::string domainExtent;
  std::string domainPartition;
  std::string domainMultiplicity;
  std::string recordPointer;
  std::string fragmentFamily;
  std::string windowFamily;
  llvm::SmallVector<std::string> windowLhs;
  llvm::SmallVector<std::string> windowRhs;
  llvm::SmallVector<std::string> windowValidity;
  int64_t interleaveRows = 0;
  int64_t recordGroupAxis = 0;
  int64_t interleaveRowAxis = 0;
  std::string interleaveRowOffset = "0";
  std::string interleaveCohortStrideBytes = "0";
  int64_t recordAxis = 0;
  int64_t recordElements = 0;
  int64_t recordStrideBytes = 0;
  int64_t cohortWidth = 0;
  int64_t cohortPartition = 0;
  llvm::SmallVector<std::pair<int64_t, std::string>> recordByteStrides;
  llvm::SmallVector<std::pair<int64_t, std::string>> recordOrigins;
};

std::string lmulSpelling(int64_t eighths);

std::string fieldStorageAddress(const Binding &record,
                               const EncodingField &field,
                               llvm::StringRef byte,
                               llvm::StringRef rowOffset = "0");

std::string vectorSuffixFor(riscv::ValueType value);

std::string vectorTypeFor(riscv::ValueType value);
std::optional<std::string> fragmentVectorType(riscv::FragmentType value);

int64_t physicalLanesFor(riscv::ValueType value);

int64_t product(mlir::DenseI64ArrayAttr values);

riscv::AccessAttr accessOf(mlir::Operation *operation);

riscv::LeafAttr leafOf(mlir::Operation *operation);

class Emitter {
public:
  Emitter(mlir::ModuleOp module, riscv::KernelOp kernel,
          llvm::raw_ostream &output)
      : module(module), kernel(kernel), output(output) {
    collectEncodings();
    for (auto [index, symbol] : llvm::enumerate(kernel.getShapeSymbols()))
      axisSymbolIds[mlir::cast<mlir::StringAttr>(symbol).getValue()] = index + 1;
    for (auto [name, value] :
         llvm::zip(kernel.getParameterNames(), kernel.getParameterValues()))
      autoBindings[mlir::cast<mlir::StringAttr>(name).getValue()] = value;
  }

  Emitter(mlir::ModuleOp module, riscv::ArtifactPackOp artifact,
          llvm::raw_ostream &output)
      : module(module), artifact(artifact), output(output) {}

  mlir::LogicalResult emit() {
    if (!kernel.getBody().hasOneBlock())
      return fail(kernel, "intrinsic-C emission requires one kernel entry block");
    if (mlir::failed(validatePhysicalProgram()))
      return mlir::failure();
    mlir::FailureOr<std::string> arguments = functionArguments();
    if (mlir::failed(arguments))
      return mlir::failure();
    line("void " + identifier(kernel.getSymName()) + "(" + *arguments + ") {");
    ++indent;
    for (auto [index, arg] : llvm::enumerate(kernel.getBody().front().getArguments()))
      if (hasStrideArguments(kernel.getStridedArgsAttr(), kernel.getArgNames()[index]))
        for (unsigned axis = 0; axis < mlir::cast<riscv::MemDescType>(arg.getType()).getShape().size(); ++axis)
          line("(void)" + strideArgumentName(index, axis) + ";");
    for (mlir::Attribute symbol : kernel.getShapeSymbols()) {
      llvm::StringRef name =
          mlir::cast<mlir::StringAttr>(symbol).getValue();
      if (!autoBindings.contains(name))
        line("(void)" + identifier(name) + ";");
    }
    if (mlir::failed(initializeKernelArguments()))
      return mlir::failure();
    if (mlir::failed(compileBlock(kernel.getBody().front(), {})))
      return mlir::failure();
    --indent;
    line("}");
    line("");
    return mlir::success();
  }

  mlir::LogicalResult emitArtifact() {
    if (!artifact || !artifact.getSizeBody().hasOneBlock() ||
        !artifact.getPackBody().hasOneBlock())
      return fail(artifact, "artifact emission requires two closed physical regions");
    llvm::StringRef march = artifact.getTarget().getMarch();
    unsigned indexBits = march.starts_with("rv32") ? 32 :
                         march.starts_with("rv64") ? 64 : 0;
    if (!indexBits)
      return fail(artifact, "artifact emission requires its explicit target index width");
    line("_Static_assert(sizeof(size_t) * 8 == " + std::to_string(indexBits) +
         " && sizeof(ptrdiff_t) * 8 == " + std::to_string(indexBits) +
         ", \"artifact target index width must match the compiled C ABI\");");
    const std::string prefix = identifier(artifact.getSymName());
    bindings.clear();
    line("size_t " + prefix + "_packed_size(size_t M, size_t K) {");
    ++indent;
    if (mlir::failed(compileBlock(artifact.getSizeBody().front(),
                                  {scalar("M"), scalar("K")})))
      return mlir::failure();
    auto sizeYield = mlir::cast<riscv::ArtifactSizeYieldOp>(
        artifact.getSizeBody().front().getTerminator());
    Binding size = bindings.lookup(sizeYield.getValue());
    if (size.kind != Binding::Kind::Scalar)
      return fail(sizeYield, "artifact size region did not produce one scalar index");
    line("return " + size.scalar + ";");
    --indent;
    line("}");
    line("");

    bindings.clear();
    Binding source;
    source.kind = Binding::Kind::Memory;
    source.memory.name = "source";
    source.memory.isConst = true;
    Binding target;
    target.kind = Binding::Kind::Memory;
    target.memory.name = "target";
    target.memory.isConst = false;
    line("void " + prefix +
         "_pack(const uint8_t *restrict source, uint8_t *restrict target, size_t M, size_t K) {");
    ++indent;
    if (mlir::failed(compileBlock(
            artifact.getPackBody().front(),
            {source, target, scalar("M"), scalar("K")})))
      return mlir::failure();
    --indent;
    line("}");
    line("");
    return mlir::success();
  }

  mlir::FailureOr<RISCVKernelABI> kernelABI() {
    RISCVKernelABI result;
    result.symbol = identifier(kernel.getSymName());
    auto arguments = functionArguments();
    if (mlir::failed(arguments))
      return mlir::failure();
    result.declaration = "void " + result.symbol + "(" + *arguments + ");";
    result.march = kernel.getTarget().getMarch().str();
    result.abi = kernel.getTarget().getAbi().str();
    result.vlenBits = kernel.getTarget().getVlenBits();
    std::string isaError;
    if (auto cache = mlir::dyn_cast<riscv::DataCacheAttr>(kernel.getTarget().getDataCache()))
      result.dataCache = RISCVDataCache{cache.getSizeBytes(), cache.getLineBytes(), cache.getWays(), cache.getSets()};
    if (!parseRISCVISAExtensions(result.march, result.vlenBits, result.isaExtensions, isaError)) {
      kernel.emitError("cannot export the selected ISA contract: ") << isaError;
      return mlir::failure();
    }
    result.localTileStrategy = kernel.getLocalTileStrategy().str();
    if (!kernel.getResourcesMaterialized()) {
      kernel.emitError("cannot export resources before physical resource closure");
      return mlir::failure();
    }
    result.physicalResources = {
        {"vector_register_peak", kernel.getVectorRegisterPeak()},
        {"fragment_register_peak", kernel.getFragmentRegisterPeak()},
        {"vector_register_budget", kernel.getTarget().getVectorRegisters()},
        {"local_storage_bytes", kernel.getLocalStorageBytes()},
        {"local_storage_budget_bytes", kernel.getTarget().getMaxPrivateStackBytes()}};
    result.physicalOperations = riscv_internal::collectPhysicalOperations(kernel);
    int64_t grouped = 0, partitioned = 0;
    kernel.walk([&](riscv::IMEUnpackOp op) {
      if (op.getResult().getType().getLayout().getReplicaGroup() > 1)
        ++grouped;
      else
        ++partitioned;
    });
    if (grouped) result.localTileRealizations.emplace_back("register-grouped", grouped);
    if (partitioned) result.localTileRealizations.emplace_back("atom-partitioned", partitioned);
    for (mlir::Attribute attribute : kernel.getRequiredExtensions())
      result.requiredExtensions.push_back(mlir::cast<mlir::StringAttr>(attribute).getValue().str());
    auto used = riscv::usedInstructionExtensions(kernel);
    if (mlir::failed(used))
      return mlir::failure();
    for (llvm::StringRef extension : *used)
      result.usedExtensions.push_back(extension.str());
    for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
      auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(attribute);
      const auto *contract = findRISCVFragmentCapability(fragment.getInstruction());
      if (!contract) {
        kernel.emitError("kernel ABI has an unregistered matrix capability");
        return mlir::failure();
      }
      if (!llvm::is_contained(result.matrixExtensions, contract->extension))
        result.matrixExtensions.push_back(contract->extension);
    }
    for (auto [index, argument] :
         llvm::enumerate(kernel.getBody().front().getArguments())) {
      auto view = mlir::dyn_cast<riscv::MemDescType>(argument.getType());
      auto cType = argumentCType(argument);
      if (!cType) {
        kernel.emitError("kernel ABI requires a supported scalar or memory parameter");
        return mlir::failure();
      }
      RISCVArgumentABI parameter;
      parameter.name = mlir::cast<mlir::StringAttr>(
          kernel.getArgNames()[index]).getValue().str();
      parameter.cType = *cType;
      parameter.access = mlir::cast<mlir::StringAttr>(kernel.getArgAccess()[index]).getValue().str();
      if (!view) {
        parameter.kind = "scalar";
        parameter.dtype = scalarABIName(argument.getType());
        parameter.aliasSet = -1;
        result.arguments.push_back(std::move(parameter));
        continue;
      }
      auto encoding = mlir::cast<kernel::EncodingType>(view.getEncoding());
      parameter.kind = "view";
      parameter.strided = hasStrideArguments(kernel.getStridedArgsAttr(), kernel.getArgNames()[index]);
      parameter.dtype = encoding.getKind() == "dense" ? encoding.getFamily().str() : "";
      parameter.encoding = encoding.getLayoutIdentity().str();
      parameter.interleaveRows = view.getInterleaveRows();
      parameter.aliasSet = kernel.getArgAliasSets()[index];
      auto access = mlir::cast<mlir::StringAttr>(
          kernel.getArgAccess()[index]).getValue();
      parameter.writable = access == "write" || access == "readwrite";
      for (int64_t extent : view.getShape().asArrayRef()) {
        if (extent > 0) {
          parameter.shape.push_back(std::to_string(extent));
        } else if (extent < 0 &&
                   extent >= -static_cast<int64_t>(kernel.getShapeSymbols().size())) {
          parameter.shape.push_back(mlir::cast<mlir::StringAttr>(
              kernel.getShapeSymbols()[-extent - 1]).getValue().str());
        } else {
          kernel.emitError("kernel ABI has an unresolved logical extent");
          return mlir::failure();
        }
      }
      if (encoding.getKind() == "dense") {
        auto element = denseElementType(encoding);
        if (!element || !element->isIntOrFloat()) {
          kernel.emitError("kernel ABI has an unsupported dense element");
          return mlir::failure();
        }
        parameter.storageBytes = std::max<unsigned>(8, element->getIntOrFloatBitWidth()) / 8;
        parameter.recordElements = 1;
        parameter.alignment = parameter.storageBytes;
      } else {
        auto base = riscv_internal::baseEncodingFamily(kernel, encoding);
        auto found = encodings.find(base);
        if (found == encodings.end() || found->second.storageBits % 8 ||
            found->second.logicalElements <= 0) {
          kernel.emitError("kernel ABI has no closed storage record");
          return mlir::failure();
        }
        parameter.storageBytes = found->second.storageBits / 8;
        parameter.recordElements = found->second.logicalElements;
        parameter.alignment = found->second.alignment;
      }
      result.arguments.push_back(std::move(parameter));
    }
    for (mlir::Attribute symbol : kernel.getShapeSymbols()) {
      auto name = mlir::cast<mlir::StringAttr>(symbol).getValue();
      if (!autoBindings.contains(name))
        result.shapeParameters.push_back(name.str());
    }
    for (auto [name, value] :
         llvm::zip(kernel.getParameterNames(), kernel.getParameterValues()))
      result.bindings.emplace_back(
          mlir::cast<mlir::StringAttr>(name).getValue().str(), value);
    return result;
  }

private:
  mlir::LogicalResult validatePhysicalProgram();

  mlir::LogicalResult fail(mlir::Operation *operation, llvm::Twine message) {
    operation->emitError(message);
    return mlir::failure();
  }

  template <typename Op>
  mlir::LogicalResult fail(Op operation, llvm::Twine message) {
    operation.emitError(message);
    return mlir::failure();
  }

  void line(llvm::StringRef text = "") {
    output.indent(indent * 2) << text << '\n';
  }

  std::string fresh(llvm::StringRef prefix) {
    return identifier(prefix) + "_" + std::to_string(nextName++);
  }

  std::string materializeScalarRead(llvm::StringRef type,
                                    llvm::StringRef expression,
                                    llvm::StringRef prefix = "load") {
    std::string name = fresh(prefix);
    line(type.str() + " " + name + " = " + expression.str() + ";");
    return name;
  }

  riscv::LayoutAttr layoutOf(mlir::Value value) const {
    return riscv_internal::layoutOf(value.getType());
  }
  llvm::StringRef instructionOf(mlir::Operation *operation) const {
    if (auto leaf = operation
                        ? operation->getAttrOfType<riscv::LeafAttr>("leaf")
                        : riscv::LeafAttr())
      return leaf.getInstruction();
    return {};
  }
  bool isSharedMaterialization(mlir::Value value) const {
    auto materialize = value.getDefiningOp<riscv::RegisterMaterializeOp>();
    return materialize && materialize.getRealization() == "share";
  }

  void collectEncodings() {
    for (riscv::EncodingDeclOp declaration :
         module.getOps<riscv::EncodingDeclOp>()) {
      EncodingInfo info;
      info.family = declaration.getSymName().str();
      info.kind = declaration.getKind().str();
      info.storageBits = declaration.getStorageBits();
      info.alignment = declaration.getAlignment();
      info.logicalElements = declaration.getElements();
      auto names = declaration.getFieldNames();
      auto types = declaration.getFieldTypes();
      auto shapes = declaration.getFieldShapes();
      auto offsets = declaration.getFieldBitOffsets();
      auto storage = declaration.getFieldStorageBits();
      for (size_t index = 0; index < names.size(); ++index) {
        EncodingField field;
        field.name = mlir::cast<mlir::StringAttr>(names[index]).getValue().str();
        field.type = mlir::cast<mlir::TypeAttr>(types[index]).getValue();
        auto shape = mlir::cast<mlir::DenseI64ArrayAttr>(shapes[index]);
        field.shape.assign(shape.asArrayRef().begin(), shape.asArrayRef().end());
        field.bitOffset = offsets[index];
        field.storageBits = storage[index];
        info.fields.push_back(std::move(field));
      }
      encodings[info.family] = std::move(info);
    }
  }

  std::optional<std::string> scalarCType(mlir::Type type) const {
    if (mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(type))
      return "uint8_t";
    if (type.isBF16())
      return "uint16_t";
    if (type.isIndex())
      return "size_t";
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
      std::string prefix = integer.isUnsigned() ? "uint" : "int";
      return prefix + std::to_string(std::max(8u, integer.getWidth())) + "_t";
    }
    if (type.isF16())
      return "_Float16";
    if (type.isF32())
      return "float";
    if (type.isF64())
      return "double";
    return std::nullopt;
  }

  std::optional<std::string> storageCType(mlir::Type type) const {
    if (type.isBF16())
      return "uint16_t";
    return scalarCType(type);
  }

  std::optional<std::string> descriptorExtent(mlir::Value value, int64_t axis) const;
  std::optional<std::string> descriptorStride(mlir::Value value, int64_t axis) const;

  std::optional<mlir::Type> denseElementType(mlir::Type type) const {
    auto encoding = mlir::dyn_cast<kernel::EncodingType>(type);
    if (!encoding || encoding.getKind() != "dense")
      return std::nullopt;
    llvm::StringRef family = encoding.getFamily();
    if (family == "f8E4M3FN") return mlir::Float8E4M3FNType::get(type.getContext());
    if (family == "f8E5M2") return mlir::Float8E5M2Type::get(type.getContext());
    if (family == "f16")
      return mlir::Float16Type::get(type.getContext());
    if (family == "bf16")
      return mlir::BFloat16Type::get(type.getContext());
    if (family == "f32")
      return mlir::Float32Type::get(type.getContext());
    if (family == "f64")
      return mlir::Float64Type::get(type.getContext());
    llvm::StringRef widthSpelling = family;
    mlir::IntegerType::SignednessSemantics signedness;
    if (widthSpelling.consume_front("i"))
      signedness = mlir::IntegerType::Signed;
    else if (widthSpelling.consume_front("u"))
      signedness = mlir::IntegerType::Unsigned;
    else
      widthSpelling = {};
    unsigned width = 0;
    if (!widthSpelling.empty() &&
        !widthSpelling.getAsInteger(10, width) && width > 0) {
      if (width == 1 && signedness == mlir::IntegerType::Signed)
        return mlir::IntegerType::get(type.getContext(), 1);
      return mlir::IntegerType::get(type.getContext(), width, signedness);
    }
    auto info = encodings.find(encoding.getFamily());
    if (info != encodings.end() && info->second.fields.size() == 1)
      return info->second.fields.front().type;
    return std::nullopt;
  }

  std::string scalarABIName(mlir::Type type) {
    if (type.isIndex())
      return "index";
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
      return std::string(integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
    return riscv_internal::printType(type);
  }

  std::optional<std::string> argumentCType(mlir::BlockArgument argument) {
    auto view = mlir::dyn_cast<riscv::MemDescType>(argument.getType());
    if (!view)
      return scalarCType(argument.getType());
    auto encoding = mlir::cast<kernel::EncodingType>(view.getEncoding());
    llvm::StringRef access =
        mlir::cast<mlir::StringAttr>(
            kernel.getArgAccess()[argument.getArgNumber()])
            .getValue();
    bool writable = access == "write" || access == "readwrite";
    std::string element = "uint8_t";
    if (encoding.getKind() == "dense") {
      auto denseType = denseElementType(encoding);
      if (!denseType)
        return std::nullopt;
      auto fieldType = storageCType(*denseType);
      if (!fieldType)
        return std::nullopt;
      element = *fieldType;
    }
    int64_t aliasSet = kernel.getArgAliasSets()[argument.getArgNumber()];
    size_t aliases = 0;
    for (int64_t candidate : kernel.getArgAliasSets())
      aliases += candidate == aliasSet;
    std::string qualifier = aliasSet >= 0 && aliases == 1 ? " *restrict" : " *";
    return (writable ? "" : "const ") + element + qualifier;
  }

  mlir::FailureOr<std::string> functionArguments() {
    llvm::SmallVector<std::string> arguments;
    auto names = kernel.getArgNames();
    for (auto [index, argument] :
         llvm::enumerate(kernel.getBody().front().getArguments())) {
      auto type = argumentCType(argument);
      if (!type) {
        kernel.emitError("intrinsic-C emission does not support a kernel argument type");
        return mlir::failure();
      }
      arguments.push_back(*type + " " +
                          identifier(mlir::cast<mlir::StringAttr>(names[index])
                                         .getValue()));
    }
    for (mlir::Attribute symbol : kernel.getShapeSymbols())
      if (!autoBindings.contains(
              mlir::cast<mlir::StringAttr>(symbol).getValue()))
        arguments.push_back("size_t " +
                            identifier(mlir::cast<mlir::StringAttr>(symbol)
                                           .getValue()));
    for (auto [index, arg] : llvm::enumerate(kernel.getBody().front().getArguments()))
      if (hasStrideArguments(kernel.getStridedArgsAttr(), kernel.getArgNames()[index]))
        for (unsigned axis = 0; axis < mlir::cast<riscv::MemDescType>(arg.getType()).getShape().size(); ++axis)
          arguments.push_back("size_t " + strideArgumentName(index, axis));
    return llvm::join(arguments, ", ");
  }

  std::string resolve(llvm::StringRef spelling) const {
    if (spelling.consume_front("auto:")) {
      auto direct = autoBindings.find(spelling);
      if (direct != autoBindings.end())
        return std::to_string(direct->second);
      llvm::SmallVector<llvm::StringRef> choices;
      spelling.split(choices, '|', -1, false);
      for (llvm::StringRef choice : choices) {
        auto found = autoBindings.find(choice);
        if (found != autoBindings.end())
          return std::to_string(found->second);
      }
    }
    int64_t integer = 0;
    if (!spelling.getAsInteger(10, integer))
      return std::to_string(integer);
    return identifier(spelling);
  }

  std::optional<std::string> extentForShape(int64_t extent, int64_t axis) {
    if (extent > 0)
      return std::to_string(extent);
    if (extent < 0 && currentOperation)
      if (auto local = weft::localShapeExtent<riscv::LocalShapeOp>(currentOperation, -extent)) {
        auto found = bindings.find(local);
        if (found != bindings.end() && found->second.kind == Binding::Kind::Scalar)
          return found->second.scalar;
        return std::nullopt;
      }
    if (extent < 0 && extent >= -int64_t(kernel.getShapeSymbols().size()))
      return identifier(mlir::cast<mlir::StringAttr>(kernel.getShapeSymbols()[-extent - 1]).getValue());
    return std::nullopt;
  }

  mlir::LogicalResult initializeKernelArguments() {
    mlir::Block &entry = kernel.getBody().front();
    auto names = kernel.getArgNames();
    for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
      auto view = mlir::dyn_cast<riscv::MemDescType>(argument.getType());
      if (!view) {
        auto name = identifier(mlir::cast<mlir::StringAttr>(names[index]).getValue());
        bindings[argument] = scalar(name);
        line("(void)" + name + ";");
        continue;
      }
      MemoryInfo memory;
      memory.name = identifier(
          mlir::cast<mlir::StringAttr>(names[index]).getValue());
      memory.encoding = view.getEncoding();
      if (auto elementType = denseElementType(view.getEncoding()))
        memory.elementType = *elementType;
      memory.axes.assign(view.getAxisIds().asArrayRef().begin(),
                         view.getAxisIds().asArrayRef().end());
      llvm::StringRef access = mlir::cast<mlir::StringAttr>(
                                   kernel.getArgAccess()[index])
                                   .getValue();
      memory.isConst = access != "write" && access != "readwrite";
      Binding binding;
      binding.kind = Binding::Kind::Memory;
      binding.memory = std::move(memory);
      bindings[argument] = std::move(binding);
    }
    return mlir::success();
  }

  // The remaining operation emitters consume typed physical operation
  // semantics and closed local leaves; they never inspect kernel names or a
  // side assignment.

  mlir::LogicalResult compileBlock(mlir::Block &block,
                                   llvm::ArrayRef<Binding> arguments) {
    llvm::SmallVector<int64_t> physicalPointAxes;
    auto leavePhysicalPoints = [&]() {
      for (int64_t axis : llvm::reverse(physicalPointAxes))
        axisScopes[axis].pop_back();
    };
    if (!arguments.empty()) {
      if (arguments.size() != block.getNumArguments())
        return fail(block.getParentOp(), "internal intrinsic-C block binding mismatch");
      for (auto [argument, binding] : llvm::zip(block.getArguments(), arguments))
        bindings[argument] = binding;
    }
    for (mlir::Operation &operation : block) {
      if (mlir::isa<riscv::ReturnOp, riscv::YieldOp,
                    riscv::ArtifactSizeYieldOp, riscv::ArtifactReturnOp,
                    mlir::scf::YieldOp,
                    mlir::scf::ConditionOp>(operation))
        continue;
      if (mlir::failed(compileOperation(operation))) {
        leavePhysicalPoints();
        return mlir::failure();
      }
      if (auto physicalPoint =
              mlir::dyn_cast<riscv::PhysicalPointOp>(operation)) {
        const Binding &binding = bindings.lookup(physicalPoint.getResult());
        axisScopes[binding.point.axis].push_back(binding.point);
        physicalPointAxes.push_back(binding.point.axis);
      }
    }
    leavePhysicalPoints();
    return mlir::success();
  }

  mlir::LogicalResult compileOperation(mlir::Operation &operation);
  mlir::LogicalResult compileScalarControl(mlir::Operation &operation);
  mlir::LogicalResult compileFieldRead(riscv::FieldReadOp operation);
  mlir::LogicalResult compilePhysicalPoint(riscv::PhysicalPointOp point);
  mlir::LogicalResult compileRecordCohort(riscv::RecordCohortOp cohort);
  mlir::LogicalResult compileFor(mlir::scf::ForOp operation);
  mlir::LogicalResult compileIf(mlir::scf::IfOp operation);
  mlir::LogicalResult compileWhile(mlir::scf::WhileOp operation);
  mlir::LogicalResult compileMemoryView(riscv::MemoryViewOp operation);
  mlir::LogicalResult compileDenseDisjoint(riscv::DenseDisjointOp operation);
  mlir::LogicalResult compileDenseElementwiseAlias(riscv::DenseElementwiseAliasOp operation);
  mlir::LogicalResult compileDenseInjective(riscv::DenseInjectiveOp operation);
  mlir::FailureOr<std::string> denseInjectivity(mlir::Value view, mlir::Operation *anchor);
  mlir::LogicalResult compileDenseRowWindowGuard(riscv::DenseRowWindowGuardOp operation);
  mlir::LogicalResult compilePredicateCompare(riscv::RVVPredicateCompareOp operation);
  mlir::LogicalResult compilePredicateLogic(riscv::RVVPredicateLogicOp operation);
  mlir::LogicalResult compilePredicateSplat(riscv::RVVPredicateSplatOp operation);
  mlir::LogicalResult compilePredicateFromValue(riscv::RVVPredicateFromValueOp operation);
  mlir::LogicalResult compilePredicateToValue(riscv::RVVPredicateToValueOp operation);
  mlir::LogicalResult compilePredicateSelect(riscv::RVVPredicateSelectOp operation);
  mlir::LogicalResult compileStorageLoad(riscv::StorageLoadOp operation);
  mlir::LogicalResult compileStorageStore(riscv::StorageStoreOp operation);
  mlir::LogicalResult compileSlice(riscv::SliceOp slice);
  mlir::LogicalResult compileSubview(riscv::SubviewOp subview);
  mlir::LogicalResult compileExtent(riscv::ExtentOp extent);
  mlir::LogicalResult compileReshape(riscv::ReshapeOp reshape);
  mlir::LogicalResult compileReaxis(riscv::ReaxisOp operation);
  mlir::LogicalResult compileAdmit(riscv::LoadOp admit);
  mlir::LogicalResult compileStagedView(riscv::StagedViewOp staged);
  mlir::LogicalResult
  compileRegisterMaterialize(riscv::RegisterMaterializeOp materialize);
  mlir::LogicalResult
  compileLocalCapacityGuard(riscv::LocalCapacityGuardOp operation);
  mlir::LogicalResult compileLocalAlloc(riscv::LocalAllocOp operation);
  mlir::LogicalResult compileLocalSize(riscv::LocalSizeOp operation);
  mlir::LogicalResult compileLocalBind(riscv::LocalBindOp operation);
  mlir::LogicalResult compileLocalView(riscv::LocalViewOp operation);
  mlir::LogicalResult compileDenseSnapshot(riscv::DenseSnapshotOp operation);
  mlir::LogicalResult compileLocalCopy(riscv::LocalCopyOp operation);
  mlir::LogicalResult emitReadSnapshot(mlir::Operation *operation,
                                        riscv::LeafAttr leaf,
                                        llvm::StringRef destination,
                                        llvm::StringRef source, int64_t bytes);
  std::string emitBytePayloadLoad(llvm::StringRef lmul,
                                 llvm::StringRef address,
                                 llvm::StringRef activeBytes,
                                 llvm::StringRef stem);
  mlir::LogicalResult compileLocalLoad(riscv::LocalLoadOp operation);
  mlir::LogicalResult compileLocalStore(riscv::LocalStoreOp operation);
  mlir::LogicalResult
  compileRVVLocalMaterialize(riscv::RVVLocalMaterializeOp operation);
  mlir::LogicalResult
  compileIndexMultipleGuard(riscv::IndexMultipleGuardOp operation);
  mlir::LogicalResult
  compileEncodedLocalBind(riscv::EncodedLocalBindOp operation);
  mlir::LogicalResult compileRVVEncodedLocalPackTransfer(
      riscv::RVVEncodedLocalPackTransferOp operation);
  mlir::LogicalResult compileSpill(riscv::SpillOp operation);
  mlir::LogicalResult compileReload(riscv::ReloadOp operation);
  mlir::LogicalResult compileIMEPack(riscv::IMEPackOp operation);
  mlir::LogicalResult compileIMEZero(riscv::IMEZeroOp operation);
  mlir::LogicalResult compileIMEFragmentStorageLoad(riscv::IMEFragmentStorageLoadOp operation);
  mlir::LogicalResult compileIMEFragmentWindowLoad(riscv::IMEFragmentWindowLoadOp operation);
  mlir::LogicalResult compileIMEFragmentDenseLoad(riscv::IMEFragmentDenseLoadOp operation);
  mlir::LogicalResult compileIMEFragmentDecode(riscv::IMEFragmentDecodeOp operation);
  mlir::LogicalResult compileIMEFragmentPointwise(riscv::IMEFragmentPointwiseOp operation);
  mlir::LogicalResult compileIMEFragmentRowBroadcast(riscv::IMEFragmentRowBroadcastOp operation);
  mlir::LogicalResult compileIMEFragmentMMA(
      riscv::IMEFragmentMMAOp operation);
  mlir::LogicalResult compileIMEUnpack(riscv::IMEUnpackOp operation);
  mlir::FailureOr<std::string> axisBroadcastPart(
      mlir::Operation *operation, mlir::Value input, mlir::Value result,
      int64_t part, const Binding &binding);
  mlir::LogicalResult compileConvertLayout(riscv::ConvertLayoutOp conversion);
  mlir::LogicalResult compileIota(riscv::IotaOp operation);
  mlir::LogicalResult
  compileRVVAxisBroadcast(riscv::RVVAxisBroadcastOp operation);
  mlir::LogicalResult compileNew(riscv::NewOp operation);
  mlir::LogicalResult compileField(riscv::FieldOp operation);
  mlir::LogicalResult compileFieldView(riscv::FieldViewOp operation);
  mlir::LogicalResult compileExtract(riscv::ExtractOp operation);
  mlir::LogicalResult compileUpdate(riscv::UpdateOp operation);
  mlir::LogicalResult compileUnary(riscv::UnaryOp operation);
  mlir::LogicalResult compileCompare(riscv::CompareOp operation);
  mlir::LogicalResult compileSelect(riscv::SelectOp operation);
  mlir::LogicalResult compileRVVCompareSelect(riscv::RVVCompareSelectOp operation);
  mlir::LogicalResult compileRVVAssembleValue(riscv::RVVAssembleValueOp operation);
  template <typename ConversionOp>
  mlir::LogicalResult compileCastImpl(ConversionOp operation);
  mlir::LogicalResult compileCast(riscv::CastOp operation);
  mlir::LogicalResult compileNarrow(riscv::NarrowOp operation);
  mlir::LogicalResult compileWiden(riscv::WidenOp operation);
  mlir::LogicalResult compileReduce(riscv::ReduceOp operation);
  mlir::LogicalResult compileFold2(riscv::Fold2Op operation);
  mlir::LogicalResult compileLookup(riscv::LookupOp operation);
  mlir::LogicalResult compileRVVByteGather(riscv::RVVByteGatherOp operation);
  mlir::LogicalResult compileRVVByteWindowsLoad(riscv::RVVByteWindowsLoadOp operation);
  std::optional<std::string> byteReadAddress(mlir::Value value);
  mlir::LogicalResult compileRVVIndexedEntryLoad(
      riscv::RVVIndexedEntryLoadOp operation);
  mlir::LogicalResult compileRVVUnitEntryWindowLoad(
      riscv::RVVUnitEntryWindowLoadOp operation);
  mlir::LogicalResult compileBinary(riscv::BinaryOp operation);
  mlir::LogicalResult compileCommit(riscv::StoreOp operation);
  mlir::LogicalResult compileFieldCommit(riscv::StoreOp operation,
                                         const Binding &value,
                                         const Binding &region);
  mlir::LogicalResult compileRVVSplat(riscv::RVVSplatOp operation);
  mlir::LogicalResult compileProjectReductionOperand(
      riscv::ProjectReductionOperandOp operation);
  mlir::LogicalResult
  compileRVVContractStep(riscv::RVVContractStepOp operation);
  mlir::LogicalResult
  compileRVVBitplaneMerge(riscv::RVVBitplaneMergeOp operation);
  mlir::LogicalResult
  compilePackedPlaneMerge(riscv::PackedPlaneMergeOp operation);
  mlir::LogicalResult
  compileRVVBitmaskDecode(riscv::RVVBitmaskDecodeOp operation);
  mlir::LogicalResult compileRVVSignedBitmaskReduce(
      riscv::RVVSignedBitmaskReduceOp operation);
  mlir::LogicalResult compileRVVMaskedNegate(riscv::RVVMaskedNegateOp operation);
  mlir::LogicalResult compileRVVBitmaskWindowLoad(
      riscv::RVVBitmaskWindowLoadOp operation);
  mlir::LogicalResult
  compileGroupedMacLoad(riscv::RVVGroupedMacLoadOp operation);
  mlir::LogicalResult
  compileGroupedMacStep(riscv::RVVGroupedMacStepOp operation);
  mlir::LogicalResult
  compileRVVWidenMultiply(riscv::RVVWidenMultiplyOp operation);
  mlir::LogicalResult compileRVVWidenAdd(riscv::RVVWidenAddOp operation);
  mlir::LogicalResult compileRVVNarrowShiftRight(
      riscv::RVVNarrowShiftRightOp operation);
  mlir::LogicalResult compileRVVWidenScalarMultiply(
      riscv::RVVWidenScalarMultiplyOp operation);
  mlir::LogicalResult compileRVVMultiplyHighScalar(
      riscv::RVVMultiplyHighScalarOp operation);
  mlir::LogicalResult compileRVVRegularRepeatIndex(
      riscv::RVVRegularRepeatIndexOp operation);
  mlir::LogicalResult compileRVVRegularRepeatGather(
      riscv::RVVRegularRepeatGatherOp operation);
  mlir::LogicalResult compileRVVRegularRepeatScalarLoad(
      riscv::RVVRegularRepeatScalarLoadOp operation);
  mlir::LogicalResult
  compileRVVStorageWindow(riscv::RVVStorageWindowOp operation);
  mlir::LogicalResult
  compileRVVLayeredRecordLoad(riscv::RVVLayeredRecordLoadOp operation);
  mlir::LogicalResult
  compileRVVLayeredStorageLoad(riscv::RVVLayeredStorageLoadOp operation);
  mlir::LogicalResult
  compileRVVLayeredStorageDecode(riscv::RVVLayeredStorageDecodeOp operation);
  mlir::LogicalResult
  compileRVVReplicaStorageLoad(riscv::RVVReplicaStorageLoadOp operation);
  mlir::LogicalResult
  compileRVVSegmentPairLoad(riscv::RVVSegmentPairLoadOp operation);
  mlir::LogicalResult compileRVVRecordStorageLoad(
      riscv::RVVRecordStorageLoadOp operation);
  mlir::LogicalResult compileRVVRecordStorageDecode(
      riscv::RVVRecordStorageDecodeOp operation);
  mlir::LogicalResult
  compileRVVRecordStore(riscv::RVVRecordStoreOp operation);
  mlir::LogicalResult compileRVVIssueSlice(riscv::RVVIssueSliceOp operation);
  mlir::LogicalResult
  compileRVVWidenAccumulate(riscv::RVVWidenAccumulateOp operation);
  mlir::LogicalResult
  compileRVVFinalizeWidenDot(riscv::RVVFinalizeWidenDotOp operation);
  mlir::LogicalResult compileRVVPartialSet(riscv::RVVPartialSetOp operation);
  mlir::LogicalResult
  compileRVVPartialCapture(riscv::RVVPartialCaptureOp operation);
  mlir::LogicalResult
  compileRVVPartialCollect(riscv::RVVPartialCollectOp operation);
  mlir::LogicalResult
  compileRVVPartialRepack(riscv::RVVPartialRepackOp operation);
  mlir::LogicalResult
  compileRVVPartialMerge(riscv::RVVPartialMergeOp operation);
  mlir::LogicalResult
  compileRVVPartialReduce(riscv::RVVPartialReduceOp operation);
  mlir::LogicalResult compileRVVPartialScaleCombine(
      riscv::RVVPartialScaleCombineOp operation);
  mlir::LogicalResult compileRVVPartialPackedScale(
      riscv::RVVPartialPackedScaleOp operation);
  mlir::LogicalResult compileRVVPartialWidenScale(
      riscv::RVVPartialWidenScaleOp operation);
  mlir::LogicalResult
  compileRVVPartialCombine(riscv::RVVPartialCombineOp operation);
  mlir::LogicalResult
  compileRVVPartialFinalize(riscv::RVVPartialFinalizeOp operation);
  mlir::LogicalResult
  compileRVVAssembleReplicas(riscv::RVVAssembleReplicasOp operation);
  mlir::LogicalResult
  compileRVVWidenReduce(riscv::RVVWidenReduceOp operation);
  mlir::LogicalResult compileRVVPartitionedWidenReduceStore(
      riscv::RVVPartitionedWidenReduceStoreOp operation);
  mlir::LogicalResult
  compileRVVLayeredWindow(riscv::RVVLayeredWindowOp operation);
  mlir::LogicalResult
  compileRVVLayeredStream(riscv::RVVLayeredStreamOp operation);
  mlir::LogicalResult compileRVVProjectedLayeredStream(
      riscv::RVVProjectedLayeredStreamOp operation);
  mlir::LogicalResult compileScalarTileLoad(riscv::ScalarTileLoadOp operation);
  mlir::LogicalResult compileScalarTileStore(riscv::ScalarTileStoreOp operation);
  mlir::FailureOr<llvm::SmallVector<std::string>>
  tileExtents(mlir::Operation *operation, mlir::ValueRange extents);
  mlir::LogicalResult compileRVVTileLoad(riscv::RVVTileLoadOp operation);
  mlir::LogicalResult compileRVVTileSegmentLoad(riscv::RVVTileLoadOp operation);
  mlir::LogicalResult compileRVVDenseSegmentLoad(riscv::RVVDenseSegmentLoadOp operation);
  mlir::LogicalResult compileRVVDenseSegmentStore(riscv::RVVDenseSegmentStoreOp operation);
  mlir::LogicalResult compileDenseSegmentTransfer(mlir::Operation *operation,
      mlir::ValueRange memory, mlir::ValueRange values, mlir::ValueRange extents,
      int64_t axis, bool store);
  mlir::LogicalResult compileRVVTileSlice(riscv::RVVTileSliceOp operation);
  mlir::LogicalResult compileRVVTileStore(riscv::RVVTileStoreOp operation);
  mlir::LogicalResult compileRVVTileSegmentStore(riscv::RVVTileStoreOp operation);
  mlir::LogicalResult compileRVVTileContract(riscv::RVVTileContractOp operation);
  mlir::LogicalResult compileRVVFMAIssue(riscv::RVVFMAIssueOp operation);
  mlir::LogicalResult compileAtomicRMW(riscv::AtomicRMWOp operation);
  mlir::LogicalResult compileBitcast(riscv::BitcastOp operation);
  mlir::LogicalResult compileRVVStreamReduceStep(
      riscv::RVVStreamReduceStepOp operation);
  mlir::LogicalResult compileRVVStreamContractStep(
      riscv::RVVStreamContractStepOp operation);
  mlir::LogicalResult
  compileRVVStreamFinalize(riscv::RVVStreamFinalizeOp operation);
  Binding scalar(llvm::StringRef expression) const {
    Binding binding;
    binding.kind = Binding::Kind::Scalar;
    binding.scalar = expression.str();
    return binding;
  }

  std::string vectorType(mlir::Value value) const;
  std::string vectorSuffix(mlir::Value value) const;
  int64_t vectorPartCount(mlir::Value value) const;
  int64_t streamPartCount(mlir::Value value) const;
  int64_t registerPartCount(mlir::Value value) const;
  int64_t scalarPartCount(mlir::Value value) const;
  int64_t physicalLanes(mlir::Value value) const;
  int64_t laneAxisFor(mlir::Value value) const;
  std::optional<Binding::Kind> selectedBindingKind(mlir::Value value) const;
  llvm::SmallVector<int64_t, 4> registerAxesFor(mlir::Value value) const;
  llvm::SmallVector<int64_t, 4> registerExtentsFor(mlir::Value value) const;
  std::optional<llvm::SmallVector<int64_t, 4>>
  registerCoordinates(mlir::Value value, size_t registerPart) const;
  std::optional<llvm::SmallVector<int64_t, 4>>
  scalarCoordinates(mlir::Value value, size_t part) const;
  std::optional<size_t> projectPart(mlir::Value source, mlir::Value result,
                                    size_t resultPart) const;
  std::optional<size_t> mappedPart(mlir::Operation *operation, size_t operand,
                                   size_t resultPart) const;
  std::optional<size_t> projectRegisterPart(mlir::Value source,
                                            mlir::Value result,
                                            size_t resultRegisterPart) const;
  using LaneExtractionCache =
      llvm::DenseMap<std::pair<size_t, int64_t>, std::string>;
  std::optional<std::string>
  extractVectorLane(mlir::Value source, const Binding &binding,
                    size_t sourcePart, int64_t laneOffset,
                    LaneExtractionCache &localExtractions,
                    std::string *failureReason = nullptr);
  std::optional<std::string>
  extractLaneForRegisterBroadcast(mlir::Value source, mlir::Value result,
                                  size_t resultPart, const Binding &binding,
                                  LaneExtractionCache &localExtractions,
                                  std::string *failureReason = nullptr);
  mlir::FailureOr<Binding> projectBinding(mlir::Value source,
                                          mlir::Value result,
                                          const Binding &binding);
  mlir::LogicalResult assignBinding(mlir::Operation *operation,
                                    mlir::Value targetValue, Binding &target,
                                    mlir::Value sourceValue,
                                    const Binding &source,
                                    llvm::StringRef mismatch);
  std::string partOffset(mlir::Value value, size_t part) const;
  std::optional<int64_t> timeCoordinate(mlir::Value value, size_t part,
                                        int64_t axis) const;
  std::string partVL(mlir::Value value, size_t part) const;
  mlir::FailureOr<Binding>
  makeVector(mlir::Value value, llvm::StringRef prefix,
             std::optional<Binding> initial = std::nullopt,
             mlir::Value initialValue = {});
  mlir::FailureOr<Binding> declareMutableBinding(mlir::Value value,
                                                 llvm::StringRef prefix);
  mlir::FailureOr<Binding> loadDenseBlock(mlir::Value result,
                                          const Binding &memoryBlock);
  mlir::FailureOr<Binding> loadScalarField(mlir::Value result,
                                          const Binding &field);
  std::optional<std::string> partActive(mlir::Value value, size_t part) const;
  std::optional<llvm::SmallVector<std::pair<int64_t, std::string>>>
  partOffsets(mlir::Value value, size_t part) const;
  mlir::FailureOr<Binding> materializeNumeric(mlir::Value value,
                                              Binding binding);
  mlir::FailureOr<Binding> recordForSlice(mlir::Value value);
  std::optional<std::string>
  denseAddress(const SliceInfo &slice,
               llvm::ArrayRef<std::pair<int64_t, std::string>> offsets) const;
  std::optional<EncodingField> fieldFor(const Binding &fieldBinding) const;
  std::optional<std::string> scalarFieldValue(
      const Binding &record, const EncodingField &field,
      llvm::StringRef logicalIndex, llvm::StringRef rowOffset = "0");
  Binding emitInterleavedField(mlir::Value result, const Binding &fieldBinding,
                               llvm::StringRef logicalIndex, size_t resultPart);

  mlir::ModuleOp module;
  riscv::KernelOp kernel;
  riscv::ArtifactPackOp artifact;
  llvm::raw_ostream &output;
  unsigned indent = 0;
  unsigned nextName = 0;
  llvm::DenseMap<mlir::Value, Binding> bindings;
  mlir::Operation *currentOperation = nullptr;
  llvm::DenseMap<mlir::Value, FieldInfo> fieldProjections;
  llvm::StringMap<EncodingInfo> encodings;
  llvm::StringMap<int64_t> autoBindings;
  llvm::StringMap<int64_t> axisSymbolIds;
  llvm::DenseMap<int64_t, llvm::SmallVector<PointInfo>> axisScopes;
};

} // namespace weft::riscv_emission

#endif
