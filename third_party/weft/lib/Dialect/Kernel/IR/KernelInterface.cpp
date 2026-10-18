#include "Weft/Dialect/Kernel/IR/KernelInterface.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace weft::kernel;

namespace {

FailureOr<std::string> scalarName(Type type, Operation *owner) {
  if (type.isIndex()) return std::string("index");
  if (type.isF16()) return std::string("f16");
  if (type.isBF16()) return std::string("bf16");
  if (type.isF32()) return std::string("f32");
  if (type.isF64()) return std::string("f64");
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (llvm::is_contained(ArrayRef<unsigned>({1, 8, 16, 32, 64}), integer.getWidth()))
      return (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
  }
  owner->emitError("canonical invocation interface does not support scalar type ") << type;
  return failure();
}

// Multiple occurrences denote one author parameter. Intersect their declared
// finite domains exactly, treating an empty list as caller-bound, never as a
// target-supplied default. The ordinary specialization pass remains responsible
// for verifying the selected value at every occurrence.
LogicalResult sourceDomains(KernelOp kernel, llvm::json::Object &domains,
                            llvm::StringSet<> &automatic) {
  llvm::StringMap<SmallVector<int64_t>> choices;
  auto status = kernel.walk([&](SymbolOp symbol) {
    if (symbol.getKind() != "source_auto") return WalkResult::advance();
    automatic.insert(symbol.getName());
    auto incoming = symbol.getChoices();
    auto [position, inserted] = choices.try_emplace(symbol.getName(), incoming.begin(), incoming.end());
    if (inserted || incoming.empty()) return WalkResult::advance();
    auto &current = position->second;
    if (current.empty()) {
      current.assign(incoming.begin(), incoming.end());
      return WalkResult::advance();
    }
    llvm::erase_if(current, [&](int64_t value) { return !llvm::is_contained(incoming, value); });
    if (current.empty()) {
      symbol.emitError("source parameter occurrences have disjoint finite domains: ") << symbol.getName();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) return failure();
  for (auto &entry : choices) {
    llvm::json::Array values;
    for (int64_t value : entry.second) values.push_back(value);
    domains[entry.getKey()] = std::move(values);
  }
  return success();
}

FailureOr<llvm::json::Array> argumentShape(KernelOp kernel, ViewType view) {
  llvm::json::Array shape;
  auto symbols = kernel.getShapeSymbols();
  for (int64_t extent : view.getShape().asArrayRef()) {
    if (extent >= 0) {
      shape.push_back(std::to_string(extent));
      continue;
    }
    // Avoid negating INT64_MIN, and reject invocation dimensions that refer to
    // a local SSA shape born after entry. Neither can be supplied by the caller.
    if (extent < -static_cast<int64_t>(symbols.size())) {
      kernel.emitError("invocation View shape must refer to a declared entry shape symbol");
      return failure();
    }
    shape.push_back(cast<StringAttr>(symbols[-extent - 1]).getValue());
  }
  return std::move(shape);
}

} // namespace

FailureOr<std::string> weft::kernel::describeKernelInterface(ModuleOp module) {
  SmallVector<KernelOp> kernels(module.getOps<KernelOp>());
  if (kernels.size() != 1) {
    module.emitError("a callable source interface requires exactly one canonical kernel");
    return failure();
  }
  auto kernel = kernels.front();
  llvm::json::Object domains;
  llvm::StringSet<> automatic;
  if (failed(sourceDomains(kernel, domains, automatic))) return failure();
  llvm::json::Array shapes;
  llvm::StringSet<> seen;
  for (Attribute value : kernel.getShapeSymbols()) {
    auto name = cast<StringAttr>(value).getValue();
    if (name.empty() || !seen.insert(name).second) {
      kernel.emitError("invocation shape symbols must be nonempty and distinct");
      return failure();
    }
    if (!automatic.contains(name)) shapes.push_back(name);
  }
  llvm::json::Array arguments;
  for (auto [index, argument] : llvm::enumerate(kernel.getBody().front().getArguments())) {
    auto name = cast<StringAttr>(kernel.getArgNames()[index]).getValue();
    auto access = cast<StringAttr>(kernel.getArgAccess()[index]).getValue();
    bool strided = kernel.getStridedArgsAttr() &&
        llvm::is_contained(kernel.getStridedArgsAttr(), kernel.getArgNames()[index]);
    if (auto pointer = dyn_cast<PointerType>(argument.getType())) {
      auto dtype = scalarName(pointer.getElementType(), kernel);
      if (failed(dtype)) return failure();
      arguments.push_back(llvm::json::Object{
          {"name", name}, {"kind", "pointer"}, {"dtype", *dtype},
          {"encoding", "dense." + *dtype}, {"shape", llvm::json::Array{}},
          {"access", access}, {"alias_set", kernel.getArgAliasSets()[index]}, {"strided", false}});
    } else if (auto view = dyn_cast<ViewType>(argument.getType())) {
      auto encoding = dyn_cast<EncodingType>(view.getEncoding());
      if (!encoding || encoding.getLayoutIdentity().empty()) {
        kernel.emitError("invocation View requires an explicit Encoding layout identity");
        return failure();
      }
      auto shape = argumentShape(kernel, view);
      if (failed(shape)) return failure();
      arguments.push_back(llvm::json::Object{
          {"name", name}, {"kind", "view"},
          {"dtype", encoding.getKind() == "dense" ? encoding.getFamily() : ""},
          {"encoding", encoding.getLayoutIdentity()}, {"shape", std::move(*shape)},
          {"access", access}, {"alias_set", kernel.getArgAliasSets()[index]}, {"strided", strided}});
    } else {
      auto dtype = scalarName(argument.getType(), kernel);
      if (failed(dtype)) return failure();
      arguments.push_back(llvm::json::Object{
          {"name", name}, {"kind", "scalar"}, {"dtype", *dtype}, {"encoding", ""},
          {"shape", llvm::json::Array{}}, {"access", access},
          {"alias_set", kernel.getArgAliasSets()[index]}, {"strided", false}});
    }
  }
  llvm::json::Object interface{
      {"arguments", std::move(arguments)}, {"shape_parameters", std::move(shapes)},
      {"source_parameters", std::move(domains)}};
  llvm::json::Object result{{"kind", "weft-kernel-interface"},
      {"symbol", kernel.getSymName()}, {"interface", std::move(interface)}};
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result))).str();
}
