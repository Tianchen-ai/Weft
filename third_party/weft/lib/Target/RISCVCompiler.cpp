#include "Weft/Target/RISCVCompiler.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVPasses.h"
#include "RISCVIntrinsicC.h"
#include "Weft/Target/RISCVCompilation.h"
#include "Weft/Target/RISCVPipeline.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

mlir::LogicalResult beginCompilation(mlir::ModuleOp module, llvm::StringRef entry,
                                     weft::RISCVCompilationReport *report) {
  if (report) { report->begin(entry); report->enter("input-validation"); }
  mlir::ScopedDiagnosticHandler capture(module.getContext(), [&](mlir::Diagnostic &diagnostic) {
    if (report) report->capture(diagnostic, weft::RISCVFailureCategory::InvalidInput);
    return mlir::failure();
  });
  if (mlir::failed(mlir::verify(module))) return mlir::failure();
  bool canonical = false, physical = false;
  module.walk([&](mlir::Operation *operation) {
    auto dialect = operation->getName().getDialectNamespace();
    canonical |= dialect == "weft_kernel";
    physical |= dialect == "weft_riscv";
  });
  if (canonical == physical || (entry == "canonical") != canonical)
    return module.emitError("compilation entry requires one unmixed canonical or physical program of the selected kind");
  bool checkpoint = module->hasAttr("weft.riscv.layout_input") || module->hasAttr("weft.riscv.contraction_input") ||
                    module->hasAttr("weft.riscv.panel_transfer_input");
  if ((entry == "layout-input" || entry == "contraction-input" || entry == "panel-transfer-input") != checkpoint)
    return module.emitError("physical checkpoint must be resumed through its matching input entry");
  return mlir::success();
}

mlir::LogicalResult emitResult(mlir::ModuleOp module,
    weft::RISCVCompilationResult &result, weft::RISCVCompilationReport *report) {
  if (report) report->enter("emission");
  mlir::ScopedDiagnosticHandler capture(module.getContext(), [&](mlir::Diagnostic &diagnostic) {
    if (report) report->capture(diagnostic);
    return mlir::failure();
  });
  auto status = weft::emitSelectedRISCVIntrinsicC(module, result.intrinsicC, result.kernels);
  if (mlir::failed(status) && report) {
    report->ensureFailure();
    report->captureFailedIR(module);
  }
  return status;
}

// Observe the actual manager, including repeated invocations of one pass. The
// descriptor sequence comes from the same owner that installs those passes.
class ExecutionObserver final : public mlir::PassInstrumentation {
public:
  ExecutionObserver(weft::RISCVCompilationReport &report,
                    llvm::ArrayRef<weft::RISCVPipelinePass> sequence)
      : report(report), sequence(sequence) {}
  void runBeforePass(mlir::Pass *pass, mlir::Operation *) override {
    if (position >= sequence.size() || pass->getArgument() != sequence[position].argument) {
      report.enter("pipeline", pass->getArgument());
      report.fail(weft::RISCVFailureCategory::CompilerError,
                  "executed pass sequence disagrees with the compiler pipeline");
      report.startPass("pipeline", pass->getArgument(), pass->getArgument());
      return;
    }
    std::string invocation;
    llvm::raw_string_ostream stream(invocation);
    pass->printAsTextualPipeline(stream);
    report.startPass(sequence[position].stage, pass->getArgument(), invocation);
  }
  void runAfterPass(mlir::Pass *, mlir::Operation *) override {
    report.finishPass(true);
    ++position;
  }
  void runAfterPassFailed(mlir::Pass *, mlir::Operation *operation) override {
    report.finishPass(false);
    report.captureFailedIR(operation);
    ++position;
  }
private:
  weft::RISCVCompilationReport &report;
  llvm::ArrayRef<weft::RISCVPipelinePass> sequence;
  size_t position = 0;
};

mlir::LogicalResult executePipeline(mlir::ModuleOp module, mlir::PassManager &manager,
    llvm::ArrayRef<weft::RISCVPipelinePass> sequence,
    weft::RISCVCompilationReport *report) {
  manager.enableVerifier(true);
  if (!report) return manager.run(module);
  manager.addInstrumentation(std::make_unique<ExecutionObserver>(*report, sequence));
  mlir::ScopedDiagnosticHandler capture(module.getContext(), [&](mlir::Diagnostic &diagnostic) {
    report->capture(diagnostic);
    return mlir::failure(); // Preserve the ordinary MLIR/source diagnostic too.
  });
  auto result = manager.run(module);
  if (mlir::failed(result)) report->ensureFailure();
  return mlir::failure(mlir::failed(result) || report->failed());
}

mlir::LogicalResult runPhysicalization(mlir::ModuleOp module,
                                       weft::RISCVCompilerOptions options,
                                       std::optional<weft::RISCVCheckpoint> checkpoint,
                                       weft::RISCVCompilationReport *report) {
  std::string error;
  if (!options.validateBindings(error)) {
    if (report) {
      report->enter("bindings");
      report->fail(weft::RISCVFailureCategory::InvalidInput, error);
    }
    return module.emitError(error);
  }
  module.getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
  module.getContext()->getOrLoadDialect<mlir::scf::SCFDialect>();
  module.getContext()->getOrLoadDialect<weft::riscv::WEFTRISCVDialect>();
  mlir::PassManager manager(module.getContext());
  if (checkpoint) weft::buildRISCVPreparePipeline(manager, options, *checkpoint);
  else weft::buildRISCVPhysicalPipeline(manager, options);
  if (mlir::failed(executePipeline(module, manager,
      checkpoint ? weft::getRISCVPreparePipeline(*checkpoint) : weft::getRISCVPhysicalPipeline(), report)))
    return mlir::failure();
  return mlir::success();
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream output(text);
  module.print(output);
  output << '\n';
  output.flush();
  return text;
}

} // namespace

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
weft::lowerRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options,
                       RISCVCompilationReport *report) {
  if (mlir::failed(beginCompilation(module, "canonical", report))) return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options), std::nullopt, report)))
    return mlir::failure();
  return std::move(working);
}

mlir::FailureOr<weft::RISCVPhysicalizationResult>
weft::physicalizeRISCVModule(mlir::ModuleOp module,
                            RISCVCompilerOptions options, RISCVCompilationReport *report) {
  auto working = lowerRISCVModule(module, std::move(options), report);
  if (mlir::failed(working)) return mlir::failure();
  RISCVPhysicalizationResult result;
  result.riscvIR = printModule(**working);
  return result;
}

mlir::FailureOr<weft::RISCVPhysicalizationResult>
weft::prepareRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options,
                         RISCVCheckpoint checkpoint, RISCVCompilationReport *report) {
  if (mlir::failed(beginCompilation(module, "canonical", report))) return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options), checkpoint, report)))
    return mlir::failure();
  return RISCVPhysicalizationResult{printModule(*working)};
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::completeRISCVModule(mlir::ModuleOp module, RISCVCheckpoint checkpoint, RISCVCompilationReport *report) {
  if (mlir::failed(beginCompilation(module, checkpointEntry(checkpoint), report))) return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  auto binding = [&] {
    mlir::ScopedDiagnosticHandler capture(module.getContext(), [&](mlir::Diagnostic &diagnostic) {
      if (report) report->capture(diagnostic, RISCVFailureCategory::InvalidInput);
      return mlir::failure();
    });
    return readRISCVPipelineInput(*working, checkpoint);
  }();
  if (mlir::failed(binding)) return mlir::failure();
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  buildRISCVResumePipeline(manager, *binding);
  if (mlir::failed(executePipeline(*working, manager, getRISCVResumePipeline(checkpoint), report)))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitResult(*working, result, report)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options,
                        RISCVCompilationReport *report) {
  if (mlir::failed(beginCompilation(module, "canonical", report))) return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options), std::nullopt, report)))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitResult(*working, result, report)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::translateRISCVModule(mlir::ModuleOp module, RISCVCompilationReport *report) {
  if (mlir::failed(beginCompilation(module, "physical", report))) return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(weft::createVerifyFinalRISCVPass());
  const RISCVPipelinePass verification[] = {{"weft-riscv-verify-final", "final-verification"}};
  if (mlir::failed(executePipeline(*working, manager, verification, report)))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitResult(*working, result, report)))
    return mlir::failure();
  return result;
}
