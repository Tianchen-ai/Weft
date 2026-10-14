#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelInterface.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVCompilation.h"
#include "Weft/Target/RISCVTargetProfile.h"
#include "Weft/Target/RISCVExtensions.h"
#include "Weft/Target/RISCVTargetCatalog.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <utility>

namespace {

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<Weft MLIR>"),
    llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output path"), llvm::cl::init("-"));
llvm::cl::opt<std::string> emitKind(
    "emit", llvm::cl::desc("kernel-ir, kernel-interface, riscv-contraction-input, contraction-selections, riscv-panel-transfer-input, panel-transfer-selections, riscv-layout-input, riscv-ir, intrinsic-c, or artifact"),
    llvm::cl::init("riscv-ir"));
llvm::cl::opt<bool> queryNativeTarget(
    "query-native-target", llvm::cl::desc("Discover the local RISC-V execution target as JSON"),
    llvm::cl::init(false));
llvm::cl::opt<bool> queryTarget(
    "query-target", llvm::cl::desc("Describe an explicit RISC-V target without compiling or probing hardware"),
    llvm::cl::init(false));
llvm::cl::opt<bool> listExtensions(
    "list-extensions", llvm::cl::desc("List registered local extension and instruction contracts as JSON"),
    llvm::cl::init(false));
llvm::cl::opt<bool> resumeLayoutInput(
    "resume-layout-input",
    llvm::cl::desc("Finalize a parsed pre-resource layout input checkpoint"),
    llvm::cl::init(false));
llvm::cl::opt<bool> resumeContractionInput(
    "resume-contraction-input",
    llvm::cl::desc("Materialize a parsed contraction selection checkpoint through the normal main suffix"),
    llvm::cl::init(false));
llvm::cl::opt<bool> resumePanelTransferInput(
    "resume-panel-transfer-input",
    llvm::cl::desc("Materialize selected panel transfers through the normal main suffix"),
    llvm::cl::init(false));
llvm::cl::opt<std::string> march("march", llvm::cl::desc("RISC-V ISA string"));
llvm::cl::opt<std::string> abi("abi", llvm::cl::desc("RISC-V ABI"));
llvm::cl::opt<std::string> dataCache(
    "data-cache", llvm::cl::desc("Known L1 data cache: SIZE_BYTES:LINE_BYTES:WAYS:SETS"));
llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits", llvm::cl::desc("Explicit fixed VLEN in bits"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> privateStackBytes(
    "private-stack-bytes", llvm::cl::desc("Explicit per-invocation private stack budget"),
    llvm::cl::init(65536));
llvm::cl::opt<std::string> matrixExtension(
    "matrix-extension",
    llvm::cl::desc("Registered local matrix extension, or none; see --list-extensions"),
    llvm::cl::init("none"));
llvm::cl::list<std::string> requiredExtensions(
    "require-extension", llvm::cl::desc("Require actual selected instruction use of an extension"),
    llvm::cl::ZeroOrMore);
llvm::cl::opt<std::string> localTileStrategy(
    "local-tile-strategy", llvm::cl::desc("target-default, atom-partitioned, or register-grouped"),
    llvm::cl::init("target-default"));
llvm::cl::opt<int64_t> maxWideningCombineGroups(
    "max-widening-combine-groups",
    llvm::cl::desc(
        "Target structural limit for one widened partial-combine result"),
    llvm::cl::init(2));
llvm::cl::opt<std::string> partialCombinePolicy(
    "partial-combine-policy",
    llvm::cl::desc(
        "Target partial combine priority: independent-multilevel or sequential"),
    llvm::cl::init("independent-multilevel"));
llvm::cl::opt<std::string> recordAxisPolicy(
    "record-axis-policy",
    llvm::cl::desc(
        "Target record-axis placement priority: within-record or across-records"),
    llvm::cl::init("within-record"));
llvm::cl::list<std::string> metaBindings(
    "meta", llvm::cl::desc("Auto specialization choices NAME=INTEGER[,INTEGER...]"),
    llvm::cl::ZeroOrMore);
llvm::cl::opt<int64_t> autoUnroll(
    "auto-unroll", llvm::cl::desc("Instantiated Level-local unroll binding"),
    llvm::cl::init(1));
llvm::cl::opt<int64_t> autoLMULEighths(
    "auto-lmul-eighths",
    llvm::cl::desc("Instantiated maximum RVV LMUL binding in eighths"),
    llvm::cl::init(8));
llvm::cl::opt<int64_t> autoPipelineDepth(
    "auto-pipeline-depth",
    llvm::cl::desc("Instantiated Level-local pipeline-depth binding"),
    llvm::cl::init(1));
llvm::cl::opt<int64_t> autoScalarLoadPrime(
    "auto-scalar-load-prime",
    llvm::cl::desc(
        "Instantiated target-local scalar-prime memory-leaf binding (0 or 1)"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> pointwiseIssueCopies(
    "pointwise-issue-copies",
    llvm::cl::desc("Maximum independent pointwise issues retained in one bounded value (1, 2 or 4)"),
    llvm::cl::init(1));
llvm::cl::opt<int64_t> contractionColumnCopies(
    "contraction-column-copies",
    llvm::cl::desc("Maximum output-column vector replicas in one contraction tile (1, 2 or 4)"),
    llvm::cl::init(4));
llvm::cl::opt<int64_t> panelTransferFields(
    "panel-transfer-fields",
    llvm::cl::desc("Maximum contiguous fields in one dense panel transfer (1, 2, 4 or 8)"),
    llvm::cl::init(4));
llvm::cl::opt<std::string> panelTransferStrategy(
    "panel-transfer-strategy", llvm::cl::desc("Finite transfer selection: issue-work, max-fields or single-field"),
    llvm::cl::init("issue-work"));
llvm::cl::opt<int64_t> contractionPanelK(
    "contraction-panel-k",
    llvm::cl::desc("Maximum K extent in one private contraction panel (32, 64, 128 or 256)"),
    llvm::cl::init(128));
llvm::cl::opt<int64_t> contractionKSteps(
    "contraction-k-steps", llvm::cl::desc("Maximum adjacent K scalars supplied together (1, 2 or 4)"),
    llvm::cl::init(4));

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

llvm::json::Value cacheDiagnostic(const std::optional<weft::RISCVDataCache> &cache) {
  if (!cache) return nullptr;
  return llvm::json::Object{{"size_bytes", cache->sizeBytes}, {"line_bytes", cache->lineBytes},
      {"ways", cache->ways}, {"sets", cache->sets}};
}

bool parseDataCache(std::optional<weft::RISCVDataCache> &cache, std::string &error) {
  if (!dataCache.getNumOccurrences()) return true;
  llvm::SmallVector<llvm::StringRef> fields;
  llvm::StringRef(dataCache).split(fields, ':');
  int64_t values[4];
  if (fields.size() != 4) {
    error = "--data-cache requires SIZE_BYTES:LINE_BYTES:WAYS:SETS";
    return false;
  }
  for (size_t i = 0; i < 4; ++i)
    if (fields[i].getAsInteger(10, values[i])) {
      error = "--data-cache requires decimal integer geometry";
      return false;
    }
  weft::RISCVDataCache geometry{values[0], values[1], values[2], values[3]};
  if (!geometry.validate(error)) return false;
  cache = geometry;
  return true;
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

bool parseMetaBindings(
    llvm::StringMap<int64_t> &result, std::string &error) {
  for (llvm::StringRef spelling : metaBindings) {
    auto [name, valueSpelling] = spelling.split('=');
    if (name.empty() || valueSpelling.empty()) {
      error = "invalid --meta binding: " + spelling.str();
      return false;
    }
    int64_t value = 0;
    if (valueSpelling.contains(',') || valueSpelling.getAsInteger(0, value) ||
        value <= 0) {
      error = "invalid --meta binding: " + spelling.str();
      return false;
    }
    if (!result.try_emplace(name, value).second) {
      error = "duplicate --meta binding: " + name.str();
      return false;
    }
  }
  return true;
}

bool readTargetProfile(weft::RISCVTargetProfile &target, std::string &error) {
  if (!weft::parseRISCVTargetProfile(march, abi, vlenBits, matrixExtension,
          maxWideningCombineGroups, partialCombinePolicy, recordAxisPolicy, target, error))
    return false;
  if (privateStackBytes <= 0) {
    error = "--private-stack-bytes must be positive";
    return false;
  }
  target.maxPrivateStackBytes = privateStackBytes;
  return parseDataCache(target.dataCache, error);
}

void writeCompilationResult(weft::RISCVCompilationResult &result,
                            llvm::raw_ostream &output,
                            const weft::RISCVCompilationReport &report) {
  if (emitKind == "riscv-ir") {
    output << result.riscvIR;
    return;
  }
  if (emitKind == "intrinsic-c") {
    output << result.intrinsicC;
    return;
  }
  llvm::json::Array kernels;
  for (const auto &kernel : result.kernels) {
    llvm::json::Array arguments;
    for (const auto &parameter : kernel.arguments) {
      llvm::json::Array shape;
      for (const auto &dimension : parameter.shape)
        shape.push_back(dimension);
      arguments.push_back(llvm::json::Object{
          {"name", parameter.name}, {"c_type", parameter.cType},
          {"kind", parameter.kind}, {"dtype", parameter.dtype}, {"access", parameter.access},
          {"strided", parameter.strided},
          {"encoding", parameter.encoding}, {"shape", std::move(shape)},
          {"storage_bytes", parameter.storageBytes},
          {"record_elements", parameter.recordElements},
          {"interleave_rows", parameter.interleaveRows},
          {"alignment", parameter.alignment}, {"alias_set", parameter.aliasSet},
          {"writable", parameter.writable}});
    }
    llvm::json::Array shapeParameters;
    for (const auto &symbol : kernel.shapeParameters)
      shapeParameters.push_back(symbol);
    llvm::json::Object bindings;
    for (const auto &[name, value] : kernel.bindings)
      bindings[name] = value;
    llvm::json::Array matrixExtensions;
    for (const auto &extension : kernel.matrixExtensions)
      matrixExtensions.push_back(extension);
    llvm::json::Array requirements;
    for (const auto &extension : kernel.requiredExtensions)
      requirements.push_back(extension);
    llvm::json::Array usedExtensions;
    for (const auto &extension : kernel.usedExtensions)
      usedExtensions.push_back(extension);
    llvm::json::Object realizations;
    for (const auto &[strategy, count] : kernel.localTileRealizations)
      realizations[strategy] = count;
    llvm::json::Object isaExtensions;
    for (const auto &[name, version] : kernel.isaExtensions)
      isaExtensions[name] = llvm::json::Array{version.first, version.second};
    llvm::json::Object resources;
    for (const auto &[name, value] : kernel.physicalResources)
      resources[name] = value;
    kernels.push_back(llvm::json::Object{
        {"symbol", kernel.symbol}, {"march", kernel.march},
        {"declaration", kernel.declaration},
        {"abi", kernel.abi}, {"vlen_bits", kernel.vlenBits},
        {"data_cache", cacheDiagnostic(kernel.dataCache)},
        {"isa_extensions", std::move(isaExtensions)},
        {"matrix_extensions", std::move(matrixExtensions)},
        {"required_extensions", std::move(requirements)},
        {"used_extensions", std::move(usedExtensions)},
        {"local_tile_strategy", kernel.localTileStrategy},
        {"local_tile_realizations", std::move(realizations)},
        {"physical_resources", std::move(resources)},
        {"physical_operations", physicalDiagnostics(kernel)},
        {"arguments", std::move(arguments)},
        {"shape_parameters", std::move(shapeParameters)},
        {"bindings", std::move(bindings)}});
  }
  llvm::json::Object artifact{
      {"kind", "weft-riscv-artifact"}, {"riscv_ir", result.riscvIR},
      {"compilation", report.compilationJSON()},
      {"intrinsic_c", result.intrinsicC}, {"kernels", std::move(kernels)}};
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(artifact))) << '\n';
}

// Every explicit build binding belongs to canonical physicalization. The same
// inventory governs discovery, canonical-only printing and both replay entries;
// adding an option here cannot leave a second partial rejection list behind.
llvm::SmallVector<llvm::cl::Option *> buildBindings() {
  return {&march, &abi, &vlenBits, &matrixExtension, &privateStackBytes, &dataCache,
      &maxWideningCombineGroups, &partialCombinePolicy, &recordAxisPolicy,
      &localTileStrategy, std::addressof(metaBindings), std::addressof(requiredExtensions), &autoUnroll,
      &autoLMULEighths, &autoPipelineDepth, &autoScalarLoadPrime,
      &pointwiseIssueCopies, &contractionColumnCopies, &panelTransferFields, &panelTransferStrategy,
      &contractionPanelK, &contractionKSteps};
}

std::string explicitBuildBindings() {
  std::string names;
  for (const auto *option : buildBindings()) {
    if (!option->getNumOccurrences()) continue;
    if (!names.empty()) names += ", ";
    names += "--" + option->ArgStr.str();
  }
  return names;
}

std::string explicitKernelBindings() {
  const llvm::SmallVector<llvm::cl::Option *> targetOnly = {
      &march, &abi, &vlenBits, &matrixExtension, &privateStackBytes,
      &maxWideningCombineGroups, &partialCombinePolicy, &recordAxisPolicy};
  std::string names;
  for (const auto *option : buildBindings()) {
    if (!option->getNumOccurrences() || llvm::is_contained(targetOnly, option)) continue;
    if (!names.empty()) names += ", ";
    names += "--" + option->ArgStr.str();
  }
  return names;
}

class CommandResult {
public:
  CommandResult() { report.begin("canonical"); }
  ~CommandResult() {
    if (successful || emitKind != "artifact") return;
    report.ensureFailure();
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(report.failureJSON())) << '\n';
  }
  int fail(llvm::StringRef stage, llvm::Twine message,
           weft::RISCVFailureCategory category = weft::RISCVFailureCategory::InvalidInput) {
    report.enter(stage);
    auto text = message.str();
    report.fail(category, text);
    llvm::errs() << text << '\n';
    return 1;
  }
  int complete() { successful = true; return 0; }
  weft::RISCVCompilationReport report;
private:
  bool successful = false;
};

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Weft RISC-V representation compiler\n");
  CommandResult command;
  std::string registrationError;
  if (!weft::validateRISCVExtensionRegistry(registrationError))
    return command.fail("extension-registry", registrationError,
                         weft::RISCVFailureCategory::CompilerError);
  const std::string explicitBindings = explicitBuildBindings();
  const unsigned queries = static_cast<unsigned>(queryNativeTarget) +
                           static_cast<unsigned>(queryTarget) +
                           static_cast<unsigned>(listExtensions);
  if (queries > 1)
    return command.fail("arguments", "select exactly one target discovery, target description or extension catalog query");
  if (queries && (inputFilename.getNumOccurrences() || emitKind.getNumOccurrences() ||
                  outputFilename.getNumOccurrences() || resumeLayoutInput.getNumOccurrences() || resumeContractionInput.getNumOccurrences() || resumePanelTransferInput.getNumOccurrences()))
    return command.fail("arguments", "target queries do not accept a source, output artifact or replay mode");
  if (listExtensions) {
    if (!explicitBindings.empty())
      return command.fail("arguments", "--list-extensions describes the compiler registry and cannot consume " + explicitBindings);
    std::string error;
    if (!weft::validateRISCVExtensionRegistry(error))
      return command.fail("extension-registry", error);
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(weft::riscvExtensionCatalogJSON())) << '\n';
    return command.complete();
  }
  if (queryTarget) {
    const auto kernelBindings = explicitKernelBindings();
    if (!kernelBindings.empty())
      return command.fail("arguments", "--query-target cannot consume kernel or cache bindings: " + kernelBindings);
    weft::RISCVTargetProfile target;
    std::string error;
    llvm::json::Object facts;
    if (!readTargetProfile(target, error) || !weft::riscvTargetCapabilitiesJSON(target, facts, error))
      return command.fail("target-description", error);
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(facts))) << '\n';
    return command.complete();
  }
  if (queryNativeTarget) {
    if (inputFilename.getNumOccurrences() || emitKind.getNumOccurrences() ||
        outputFilename.getNumOccurrences() || resumeLayoutInput.getNumOccurrences() || resumeContractionInput.getNumOccurrences() || resumePanelTransferInput.getNumOccurrences() || !explicitBindings.empty())
      return command.fail("arguments", "--query-native-target is a standalone discovery operation");
    weft::RISCVNativeTarget native;
    std::string error;
    if (!weft::queryNativeRISCVTarget(native, error)) {
      return command.fail("target-discovery", error);
    }
    llvm::json::Array cpus;
    for (int cpu : native.cpus)
      cpus.push_back(cpu);
    llvm::json::Object matrixExtensions;
    for (const auto &[extension, supportedCPUs] : native.matrixExtensions) {
      llvm::json::Array matrixCPUs;
      for (int cpu : supportedCPUs)
        matrixCPUs.push_back(cpu);
      matrixExtensions[extension] = std::move(matrixCPUs);
    }
    llvm::json::Object isaExtensions;
    for (const auto &[name, version] : native.isaExtensions)
      isaExtensions[name] = llvm::json::Array{version.first, version.second};
    llvm::json::Array dataCaches;
    for (const auto &observation : native.dataCaches)
      dataCaches.push_back(llvm::json::Object{{"cpu", observation.cpu}, {"cache", cacheDiagnostic(observation.cache)}});
    llvm::json::Object target{
        {"march", native.profile.march}, {"abi", native.profile.abi},
        {"xlen", native.profile.xlen}, {"vlen_bits", native.profile.vlenBits},
        {"data_cache", cacheDiagnostic(native.profile.dataCache)},
        {"cpu_data_caches", std::move(dataCaches)},
        {"isa_extensions", std::move(isaExtensions)},
        {"matrix_extensions", std::move(matrixExtensions)},
        {"cpus", std::move(cpus)}};
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(target))) << '\n';
    return command.complete();
  }
  if (emitKind != "kernel-ir" && emitKind != "kernel-interface" && emitKind != "riscv-layout-input" && emitKind != "riscv-contraction-input" &&
      emitKind != "riscv-panel-transfer-input" && emitKind != "panel-transfer-selections" &&
      emitKind != "riscv-ir" &&
      emitKind != "intrinsic-c" && emitKind != "artifact" && emitKind != "contraction-selections") {
    return command.fail("arguments", "unsupported --emit value: " + emitKind);
  }
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!buffer) {
    return command.fail("input", "cannot read " + inputFilename + ": " + buffer.getError().message());
  }
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  mlir::DialectRegistry registry;
  registry.insert<weft::kernel::WEFTKernelDialect,
                  weft::riscv::WEFTRISCVDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<weft::riscv::WEFTRISCVDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    command.report.enter("parse");
    mlir::ScopedDiagnosticHandler capture(&context, [&](mlir::Diagnostic &diagnostic) {
      command.report.capture(diagnostic, weft::RISCVFailureCategory::InvalidInput);
      return mlir::failure();
    });
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
    if (!module || mlir::failed(mlir::verify(*module))) return 1;
  }
  bool hasPhysicalProgram = false;
  bool hasCanonicalProgram = false;
  module->walk([&](mlir::Operation *operation) {
    auto dialect = operation->getName().getDialectNamespace();
    hasPhysicalProgram |= dialect == "weft_riscv";
    hasCanonicalProgram |= dialect == "weft_kernel";
  });
  if (hasPhysicalProgram == hasCanonicalProgram)
    return command.fail("input-validation", "input must contain one unmixed canonical or physical Weft program");
  unsigned resumeCount = unsigned(resumeLayoutInput) + unsigned(resumeContractionInput) + unsigned(resumePanelTransferInput);
  if (resumeCount > 1)
    return command.fail("arguments", "select exactly one physical checkpoint to resume");
  if (resumeCount &&
      (!hasPhysicalProgram ||
       (emitKind != "riscv-ir" && emitKind != "intrinsic-c" && emitKind != "artifact")))
    return command.fail("arguments", "checkpoint resumption requires physical input and final output");
  if (hasPhysicalProgram && (emitKind == "riscv-layout-input" || emitKind == "riscv-contraction-input" || emitKind == "riscv-panel-transfer-input"))
    return command.fail("arguments", "physical checkpoints must be produced from Canonical Kernel IR");
  if ((hasPhysicalProgram || emitKind == "kernel-ir" || emitKind == "kernel-interface") && !explicitBindings.empty())
    return command.fail("arguments", "build bindings apply only to canonical physicalization; this entry cannot consume " + explicitBindings);
  if (hasPhysicalProgram && emitKind == "kernel-ir")
    return command.fail("arguments", "kernel-ir output requires canonical input");
  if (hasPhysicalProgram && emitKind == "kernel-interface")
    return command.fail("arguments", "kernel-interface output requires canonical input");
  if (!hasPhysicalProgram && emitKind == "contraction-selections")
    return command.fail("arguments", "contraction-selections requires a prepared contraction input");
  if (!hasPhysicalProgram && emitKind == "panel-transfer-selections")
    return command.fail("arguments", "panel-transfer-selections requires a prepared panel transfer input");

  std::error_code errorCode;
  llvm::ToolOutputFile output(outputFilename, errorCode,
                              llvm::sys::fs::OF_Text);
  if (errorCode) {
    return command.fail("output", "cannot open output: " + errorCode.message());
  }
  if (emitKind == "kernel-ir") {
    if (hasPhysicalProgram) {
      llvm::errs()
          << "kernel-ir input cannot contain compiler-owned weft_riscv operations\n";
      return 1;
    }
    module->print(output.os());
    output.os() << '\n';
  } else if (emitKind == "kernel-interface") {
    auto result = weft::kernel::describeKernelInterface(*module);
    if (mlir::failed(result)) return 1;
    output.os() << *result;
  } else if (emitKind == "contraction-selections") {
    auto result = weft::inspectRISCVContractions(*module);
    if (mlir::failed(result)) return 1;
    output.os() << *result;
  } else if (emitKind == "panel-transfer-selections") {
    auto result = weft::inspectRISCVPanelTransfers(*module);
    if (mlir::failed(result)) return 1;
    output.os() << *result;
  } else if (hasPhysicalProgram) {
    mlir::FailureOr<weft::RISCVCompilationResult> result =
        resumeCount ? weft::completeRISCVModule(*module,
            resumeLayoutInput ? weft::RISCVCheckpoint::Layout : resumePanelTransferInput ?
            weft::RISCVCheckpoint::PanelTransfers : weft::RISCVCheckpoint::Contractions, &command.report)
                          : weft::translateRISCVModule(*module, &command.report);
    if (mlir::failed(result))
      return 1;
    writeCompilationResult(*result, output.os(), command.report);
  } else {
    weft::RISCVCompilerOptions options;
    options.requiredExtensions.assign(requiredExtensions.begin(), requiredExtensions.end());
    options.localTileStrategy = localTileStrategy;
    std::string error;
    if (!readTargetProfile(options.target, error)) {
      return command.fail("bindings", error);
    }
    if (!options.target.supportsLMULEighths(autoLMULEighths)) {
      return command.fail("bindings", "--auto-lmul-eighths is not legal for this target: " + std::to_string(autoLMULEighths));
    }
    if (!parseMetaBindings(options.metaBindings, error))
      return command.fail("bindings", error);
    if (autoUnroll <= 0 || autoPipelineDepth <= 0 || autoPipelineDepth > 2 ||
        (autoScalarLoadPrime != 0 && autoScalarLoadPrime != 1)) {
      return command.fail("bindings", "--auto-unroll must be positive and --auto-pipeline-depth must be 1 or 2; --auto-scalar-load-prime must be 0 or 1");
    }
    options.lmulEighths = autoLMULEighths;
    options.unroll = autoUnroll;
    options.pipelineDepth = autoPipelineDepth;
    options.scalarLoadPrime = autoScalarLoadPrime;
    if (pointwiseIssueCopies != 1 && pointwiseIssueCopies != 2 && pointwiseIssueCopies != 4) {
      return command.fail("bindings", "--pointwise-issue-copies must be 1, 2 or 4");
    }
    options.pointwiseIssueCopies = pointwiseIssueCopies;
    options.contractionColumnCopies = contractionColumnCopies;
    options.panelTransferFields = panelTransferFields;
    options.panelTransferStrategy = panelTransferStrategy;
    options.contractionPanelK = contractionPanelK;
    options.contractionKSteps = contractionKSteps;
    if (!options.validateBindings(error)) return command.fail("bindings", error);
    if (emitKind == "riscv-ir" || emitKind == "riscv-layout-input" || emitKind == "riscv-contraction-input" || emitKind == "riscv-panel-transfer-input") {
      mlir::FailureOr<weft::RISCVPhysicalizationResult> result =
          emitKind != "riscv-ir"
              ? weft::prepareRISCVModule(*module, std::move(options),
                  emitKind == "riscv-layout-input" ? weft::RISCVCheckpoint::Layout : emitKind == "riscv-panel-transfer-input" ?
                  weft::RISCVCheckpoint::PanelTransfers : weft::RISCVCheckpoint::Contractions,
                  &command.report)
              : weft::physicalizeRISCVModule(*module, std::move(options), &command.report);
      if (mlir::failed(result))
        return 1;
      output.os() << result->riscvIR;
    } else {
      mlir::FailureOr<weft::RISCVCompilationResult> result =
          weft::compileRISCVModule(*module, std::move(options), &command.report);
      if (mlir::failed(result))
        return 1;
      writeCompilationResult(*result, output.os(), command.report);
    }
  }
  output.keep();
  return command.complete();
}
