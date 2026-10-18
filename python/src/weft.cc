#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelInterface.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVCompilation.h"
#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVExtensions.h"
#include "Weft/Target/RISCVTargetCatalog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>

namespace py = pybind11;
using namespace mlir;

namespace {
std::string jsonText(const llvm::json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  return text;
}
py::object json(llvm::json::Value value) {
  return py::module_::import("json").attr("loads")(
      jsonText(value));
}

[[noreturn]] void compilationFailure(const weft::RISCVCompilationReport &report) {
  auto type = py::module_::import("triton.compiler.errors").attr("BackendCompilationError");
  auto error = type(json(report.failureJSON()));
  PyErr_SetObject(type.ptr(), error.ptr());
  throw py::error_already_set();
}

void loadDialects(MLIRContext &context) {
  context.loadDialect<weft::kernel::WEFTKernelDialect,
                      weft::riscv::WEFTRISCVDialect,
                      arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
}

// Construct canonical operations from the frontend's typed construction tree.
// Values cross the binding as SSA identities, never as serialized MLIR modules.
class CanonicalBuilder {
  MLIRContext &context;
  std::map<int64_t, Value> values;

  Type type(py::handle value) {
    auto spelling = py::cast<std::string>(value);
    auto result = parseType(spelling, &context);
    if (!result) throw py::value_error("invalid canonical type: " + spelling);
    return result;
  }

  Location location(py::handle node) {
    auto loc = node.attr("location");
    return FileLineColLoc::get(&context, py::cast<std::string>(loc.attr("filename")),
                              py::cast<unsigned>(loc.attr("line")),
                              py::cast<unsigned>(loc.attr("column")) + 1);
  }

  void addValue(py::handle description, Value value) {
    auto id = py::cast<int64_t>(description.attr("id"));
    if (!values.emplace(id, value).second)
      throw py::value_error("duplicate canonical SSA value");
  }

  void populate(Region &region, py::handle description, Location loc) {
    auto *block = new Block();
    region.push_back(block);
    for (auto arg : description.attr("arguments")) {
      auto spelling = py::module_::import("triton.compiler.weft.types")
                          .attr("emit_type")(arg.attr("type"));
      addValue(arg, block->addArgument(type(spelling), loc));
    }
    for (auto item : description.attr("operations"))
      block->push_back(operation(item));
  }

  Operation *operation(py::handle node) {
    auto loc = location(node);
    OperationState state(loc, py::cast<std::string>(node.attr("name")));
    for (auto operand : node.attr("operands")) {
      auto id = py::cast<int64_t>(operand.attr("id"));
      auto found = values.find(id);
      if (found == values.end()) {
        std::string message;
        llvm::raw_string_ostream output(message);
        output << "undefined Canonical SSA operand %v" << id << " of " << state.name << " at " << loc;
        if (!operand.attr("producer").is_none()) {
          auto producer = operand.attr("producer");
          output << "; produced by " << py::cast<std::string>(producer.attr("name"))
                 << " at " << location(producer);
        }
        throw py::value_error(output.str());
      }
      state.addOperands(found->second);
    }
    auto emitType = py::module_::import("triton.compiler.weft.types").attr("emit_type");
    for (auto result : node.attr("results"))
      state.addTypes(type(emitType(result.attr("type"))));
    for (auto item : py::cast<py::dict>(node.attr("attributes"))) {
      auto name = py::cast<std::string>(item.first);
      auto spelling = py::cast<std::string>(item.second);
      auto attr = parseAttribute(spelling, &context);
      if (!attr) throw py::value_error("invalid canonical attribute: " + name);
      state.addAttribute(name, attr);
    }
    for (auto region : node.attr("regions"))
      populate(*state.addRegion(), region, loc);
    auto *op = Operation::create(state);
    unsigned index = 0;
    for (auto result : node.attr("results"))
      addValue(result, op->getResult(index++));
    return op;
  }

public:
  explicit CanonicalBuilder(MLIRContext &context) : context(context) {}
  ModuleOp build(const std::string &name, py::iterable items) {
    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
    module->getOperation()->setAttr("weft.source_module", StringAttr::get(&context, name));
    for (auto item : items) module->push_back(operation(item));
    if (failed(verify(*module))) throw py::value_error("invalid Weft canonical program");
    return module.release();
  }
};

weft::RISCVCompilerOptions options(const py::dict &values) {
  weft::RISCVCompilerOptions result;
  std::string error;
  auto text = [&](const char *key) { return py::cast<std::string>(values[key]); };
  auto integer = [&](const char *key) { return py::cast<int64_t>(values[key]); };
  if (!weft::parseRISCVTargetProfile(text("march"), text("abi"), integer("vlen_bits"),
      text("matrix_extension"), integer("max_widening_combine_groups"),
      text("partial_combine_policy"), text("record_axis_policy"), result.target, error))
    throw py::value_error(error);
  result.target.maxPrivateStackBytes = integer("private_stack_bytes");
  if (!values["data_cache"].is_none()) {
    auto cache = py::cast<py::dict>(values["data_cache"]);
    result.target.dataCache = weft::RISCVDataCache{
      py::cast<int64_t>(cache["size_bytes"]), py::cast<int64_t>(cache["line_bytes"]),
      py::cast<int64_t>(cache["ways"]), py::cast<int64_t>(cache["sets"])};
    if (!result.target.dataCache->validate(error)) throw py::value_error(error);
  }
  result.requiredExtensions = py::cast<std::vector<std::string>>(values["required_extensions"]);
  result.localTileStrategy = text("local_tile_strategy");
  result.lmulEighths = integer("lmul_eighths");
  result.unroll = integer("unroll");
  result.pipelineDepth = integer("pipeline_depth");
  result.scalarLoadPrime = integer("scalar_load_prime");
  result.pointwiseIssueCopies = integer("pointwise_issue_copies");
  result.contractionColumnCopies = integer("contraction_column_copies");
  result.panelTransferFields = integer("panel_transfer_fields");
  result.panelTransferStrategy = text("panel_transfer_strategy");
  result.contractionPanelK = integer("contraction_panel_k");
  result.contractionKSteps = integer("contraction_k_steps");
  for (auto item : py::cast<py::dict>(values["meta"]))
    result.metaBindings[py::cast<std::string>(item.first)] = py::cast<int64_t>(item.second);
  if (!result.validateBindings(error)) throw py::value_error(error);
  return result;
}

template <typename T> llvm::json::Array diagnosticArray(const std::vector<T> &values) {
  llvm::json::Array result;
  for (const auto &value : values) result.push_back(value);
  return result;
}

llvm::json::Object layoutDiagnostic(const weft::RISCVLayoutDiagnostic &layout) {
  return llvm::json::Object{
      {"role", layout.role}, {"index", layout.index}, {"element_type", layout.elementType},
      {"carrier", layout.carrier}, {"validity", layout.validity},
      {"shape", diagnosticArray(layout.shape)}, {"axis_ids", diagnosticArray(layout.axes)},
      {"layout_axis_ids", diagnosticArray(layout.layoutAxes)},
      {"time_factors", diagnosticArray(layout.timeFactors)},
      {"lane_factors", diagnosticArray(layout.laneFactors)},
      {"replica_factors", diagnosticArray(layout.replicaFactors)},
      {"fragment_factors", diagnosticArray(layout.fragmentFactors)},
      {"local_factors", diagnosticArray(layout.localFactors)},
      {"sew", layout.sew}, {"lmul_eighths", layout.lmulEighths}, {"vl", layout.vl},
      {"register_groups", layout.registerGroups}, {"replica_group", layout.replicaGroup}};
}

llvm::json::Value accessDiagnostic(const std::optional<weft::RISCVAccessDiagnostic> &access) {
  if (!access) return nullptr;
  return llvm::json::Object{
      {"form", access->form}, {"mapping", access->mapping},
      {"alignment", access->alignment}, {"index_sew", access->indexSEW},
      {"segment_fields", access->segmentFields}, {"group_size", access->groupSize},
      {"layer_size", access->layerSize}, {"join_fields", access->joinFields},
      {"join_low_bits", access->joinLowBits}, {"join_role", access->joinRole},
      {"bit_offset", access->bitOffset}, {"storage_bits", access->storageBits},
      {"order", access->order}, {"field_chunk_bytes", access->fieldChunkBytes}};
}

llvm::json::Object physicalDiagnostics(const weft::RISCVKernelABI &kernel) {
  llvm::json::Array operations;
  int64_t count = 0;
  for (const auto &operation : kernel.physicalOperations) {
    llvm::json::Array layouts;
    for (const auto &layout : operation.layouts) layouts.push_back(layoutDiagnostic(layout));
    count += operation.occurrences;
    llvm::json::Value scalarPacket(nullptr);
    if (operation.scalarPacket) {
      const auto &packet = *operation.scalarPacket;
      scalarPacket = llvm::json::Object{{"axis", packet.axis}, {"steps", packet.steps},
          {"parts", packet.parts}, {"registers", packet.registers}, {"live_registers", packet.liveRegisters}};
    }
    operations.push_back(llvm::json::Object{
        {"operation", operation.operation}, {"engine", operation.engine},
        {"family", operation.family}, {"instruction", operation.instruction},
        {"mask", operation.mask}, {"tail", operation.tail},
        {"access", accessDiagnostic(operation.access)},
        {"scalar_packet", std::move(scalarPacket)},
        {"occurrences", operation.occurrences}, {"operand_count", operation.operandCount},
        {"result_count", operation.resultCount}, {"operand_groups", operation.operandGroups},
        {"result_groups", operation.resultGroups}, {"temporary_groups", operation.temporaryGroups},
        {"fragment_groups", operation.fragmentGroups}, {"local_bytes", operation.localBytes},
        {"parameters", diagnosticArray(operation.parameters)},
        {"locations", diagnosticArray(operation.locations)}, {"layouts", std::move(layouts)}});
  }
  return llvm::json::Object{{"scope", "final-physical-ir/static-selected-operations"},
      {"selected_operation_count", count}, {"operations", std::move(operations)}};
}

py::dict kernelABI(const weft::RISCVKernelABI &kernel) {
  py::dict result;
  result["symbol"] = kernel.symbol;
  result["declaration"] = kernel.declaration;
  result["march"] = kernel.march;
  result["abi"] = kernel.abi;
  result["vlen_bits"] = kernel.vlenBits;
  result["isa_extensions"] = kernel.isaExtensions;
  result["matrix_extensions"] = kernel.matrixExtensions;
  result["required_extensions"] = kernel.requiredExtensions;
  result["used_extensions"] = kernel.usedExtensions;
  result["shape_parameters"] = kernel.shapeParameters;
  py::dict bindings;
  for (const auto &[name, value] : kernel.bindings) bindings[py::str(name)] = value;
  result["bindings"] = bindings;
  result["physical_resources"] = kernel.physicalResources;
  result["physical_operations"] = json(physicalDiagnostics(kernel));
  result["local_tile_strategy"] = kernel.localTileStrategy;
  py::dict realizations;
  for (const auto &[name, count] : kernel.localTileRealizations) realizations[py::str(name)] = count;
  result["local_tile_realizations"] = realizations;
  if (kernel.dataCache) {
    py::dict cache;
    cache["size_bytes"] = kernel.dataCache->sizeBytes;
    cache["line_bytes"] = kernel.dataCache->lineBytes;
    cache["ways"] = kernel.dataCache->ways;
    cache["sets"] = kernel.dataCache->sets;
    result["data_cache"] = cache;
  } else result["data_cache"] = py::none();
  py::list arguments;
  for (const auto &arg : kernel.arguments) {
    py::dict item;
    item["name"] = arg.name;
    item["kind"] = arg.kind;
    item["dtype"] = arg.dtype;
    item["access"] = arg.access;
    item["strided"] = arg.strided;
    item["c_type"] = arg.cType;
    item["encoding"] = arg.encoding;
    item["shape"] = arg.shape;
    item["storage_bytes"] = arg.storageBytes;
    item["record_elements"] = arg.recordElements;
    item["interleave_rows"] = arg.interleaveRows;
    item["alignment"] = arg.alignment;
    item["alias_set"] = arg.aliasSet;
    item["writable"] = arg.writable;
    arguments.append(item);
  }
  result["arguments"] = arguments;
  return result;
}
} // namespace

void init_weft_runtime(py::module_ &m);

void init_weft(py::module_ &&m) {
  init_weft_runtime(m);
  m.def("load_dialects", loadDialects);
  m.def("create_module", [](MLIRContext &context, const std::string &name, py::iterable items) {
    loadDialects(context);
    return CanonicalBuilder(context).build(name, items);
  }, py::return_value_policy::take_ownership);
  m.def("interface", [](ModuleOp module) {
    auto description = weft::kernel::describeKernelInterface(module);
    if (failed(description)) throw py::value_error("invalid canonical kernel interface");
    return py::module_::import("json").attr("loads")(*description);
  });
  m.def("lower", [](ModuleOp module, const py::dict &binding) {
    weft::RISCVCompilationReport report;
    auto lowered = weft::lowerRISCVModule(module, options(binding), &report);
    if (failed(lowered))
      compilationFailure(report);
    return py::make_tuple(lowered->release(), json(report.compilationJSON()));
  });
  m.def("emit", [](ModuleOp module) {
    weft::RISCVCompilationReport report;
    auto compiled = weft::translateRISCVModule(module, &report);
    if (failed(compiled))
      compilationFailure(report);
    py::dict result;
    result["intrinsic_c"] = compiled->intrinsicC;
    py::list kernels;
    for (const auto &kernel : compiled->kernels) kernels.append(kernelABI(kernel));
    result["kernels"] = kernels;
    result["compilation"] = json(report.compilationJSON());
    return result;
  });
  m.def("native_target", [] {
    weft::RISCVNativeTarget native;
    std::string error;
    if (!weft::queryNativeRISCVTarget(native, error)) throw py::value_error(error);
    llvm::json::Object facts;
    if (!weft::riscvTargetCapabilitiesJSON(native.profile, facts, error)) throw py::value_error(error);
    auto result = py::cast<py::dict>(json(std::move(facts)));
    result["cpus"] = native.cpus;
    result["native_matrix_extensions"] = native.matrixExtensions;
    return result;
  });
  m.def("extension_catalog", [] {
    std::string error;
    if (!weft::validateRISCVExtensionRegistry(error)) throw py::value_error(error);
    return json(weft::riscvExtensionCatalogJSON());
  });
  m.def("target_capabilities", [](const py::dict &binding) {
    auto configuration = options(binding);
    llvm::json::Object facts;
    std::string error;
    if (!weft::riscvTargetCapabilitiesJSON(configuration.target, facts, error))
      throw py::value_error(error);
    return json(std::move(facts));
  });
}
