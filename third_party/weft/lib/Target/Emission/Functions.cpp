#include "Emitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileHelper(riscv::HelperOp op) {
  auto abi = op->getAttrOfType<mlir::StringAttr>("native_abi");
  if (!abi || abi.getValue() != "selected-parts") return fail(op, "helper has no closed native ABI");
  std::string body;
  llvm::raw_string_ostream stream(body);
  Emitter function(module, kernel, stream);
  function.axisScopes = axisScopes;
  function.fieldProjections = fieldProjections;
  std::string name = fresh(identifier(kernel.getSymName()) + "_" + identifier(op.getName()));
  llvm::SmallVector<std::string> declarations, arguments;
  auto parameter = [&](std::string type, std::string value, bool output = false) {
    std::string arg = "arg_" + std::to_string(declarations.size());
    declarations.push_back(type + (output ? " *" : " ") + arg);
    arguments.push_back(output ? "&(" + value + ")" : value);
    return output ? "(*" + arg + ")" : arg;
  };
  auto scalarList = [&](auto &values) {
    for (std::string &value : values) value = parameter("size_t", value);
  };
  auto bind = [&](mlir::Value value, Binding binding, bool result) -> mlir::FailureOr<Binding> {
    auto element = riscv_internal::logicalElement(value.getType());
    auto type = scalarCType(element);
    switch (binding.kind) {
    case Binding::Kind::Scalar:
      if (!type) return mlir::failure();
      if (!result && mlir::isa_and_nonnull<riscv::ConstantOp, mlir::arith::ConstantOp>(value.getDefiningOp()))
        break;
      binding.scalar = parameter(*type, binding.scalar, result);
      break;
    case Binding::Kind::Vector:
    case Binding::Kind::Fragment:
    case Binding::Kind::ScalarTuple: {
      std::string partType;
      if (binding.kind == Binding::Kind::Vector) partType = vectorType(value);
      else if (binding.kind == Binding::Kind::Fragment) {
        auto t = fragmentVectorType(mlir::cast<riscv::FragmentType>(value.getType()));
        if (!t) return mlir::failure();
        partType = *t;
      } else { if (!type) return mlir::failure(); partType = *type; }
      for (std::string &part : binding.parts) part = parameter(partType, part, result);
      break;
    }
    case Binding::Kind::Memory: {
      auto storage = binding.memory.elementType ? storageCType(binding.memory.elementType) : std::optional<std::string>("uint8_t");
      if (!storage) return mlir::failure();
      binding.memory.name = parameter(std::string(binding.memory.isConst ? "const " : "") + *storage + " *", binding.memory.name, result);
      scalarList(binding.memory.extents);
      scalarList(binding.memory.strides);
      scalarList(binding.memory.origins);
      break;
    }
    case Binding::Kind::LocalArray:
      if (result || !type) return mlir::failure();
      binding.scalar = parameter(*type + " *", binding.scalar);
      binding.localElements = parameter("size_t", binding.localElements);
      scalarList(binding.localExtents);
      break;
    case Binding::Kind::Point:
    case Binding::Kind::RecordCohort:
      binding.point.base = parameter("size_t", binding.point.base);
      binding.point.active = parameter("size_t", binding.point.active);
      break;
    case Binding::Kind::Domain:
      if (value.getDefiningOp<riscv::RootDomainOp>()) break;
      binding.domainExtent = parameter("size_t", binding.domainExtent);
      binding.domainPartition = parameter("size_t", binding.domainPartition);
      binding.domainMultiplicity = parameter("size_t", binding.domainMultiplicity);
      break;
    case Binding::Kind::Slice:
      for (auto &extent : binding.slice.extents) extent.second = parameter("size_t", extent.second);
      break;
    case Binding::Kind::Field:
      break;
    default:
      return mlir::failure();
    }
    return binding;
  };
  for (mlir::Value capture : op.getCaptures()) {
    auto bound = bind(capture, bindings.lookup(capture), false);
    if (mlir::failed(bound)) return fail(op, "helper capture has no selected native representation");
    function.bindings[capture] = std::move(*bound);
  }
  for (auto &[axis, scopes] : function.axisScopes)
    for (auto &scope : scopes) {
      scope.base = parameter("size_t", scope.base);
      scope.active = parameter("size_t", scope.active);
    }
  for (auto attribute : kernel.getShapeSymbols()) {
    auto symbol = mlir::cast<mlir::StringAttr>(attribute).getValue();
    if (!autoBindings.contains(symbol)) {
      declarations.push_back("size_t " + identifier(symbol));
      arguments.push_back(identifier(symbol));
    }
  }
  for (mlir::Value result : op.getResults()) {
    auto storage = declareMutableBinding(result, "helper_result");
    if (mlir::failed(storage)) return fail(op, "helper result has no native return representation");
    bindings[result] = *storage;
    auto bound = bind(result, *storage, true);
    if (mlir::failed(bound)) return fail(op, "helper return ABI is incomplete");
    function.helperResults.push_back({result, std::move(*bound)});
  }
  function.nextName = nextName;
  function.indent = 1;
  if (mlir::failed(function.compileBlock(op.getBody().front(), {}))) return mlir::failure();
  nextName = function.nextName;
  helperDefinitions += function.helperDefinitions;
  helperDefinitions += std::string(op.getNoinline() ? "static __attribute__((noinline)) void " : "static inline __attribute__((always_inline)) void ") +
      name + "(" + llvm::join(declarations, ", ") + ") {\n" + body + "}\n";
  line(name + "(" + llvm::join(arguments, ", ") + ");");
  return mlir::success();
}

mlir::LogicalResult Emitter::compileHelperReturn(riscv::HelperReturnOp op) {
  if (helperResults.size() != op.getValues().size()) return fail(op, "helper return has no matching native signature");
  for (auto [i, value] : llvm::enumerate(op.getValues())) {
    auto &[destination, storage] = helperResults[i];
    if (mlir::failed(assignBinding(op, destination, storage, value, bindings.lookup(value),
                                   "helper return requires an explicit physical conversion"))) return mlir::failure();
  }
  line("return;");
  return mlir::success();
}
} // namespace weft::riscv_emission
