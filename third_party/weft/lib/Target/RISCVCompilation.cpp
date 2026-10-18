#include "Weft/Target/RISCVCompilation.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace weft;

namespace {
constexpr llvm::StringLiteral resourceMarker = "weft.candidate.resource-limit";
constexpr llvm::StringLiteral unsupportedMarker = "weft.candidate.unsupported";
constexpr llvm::StringLiteral inputMarker = "weft.invalid-input";

std::string locationText(Location location) {
  std::string text;
  llvm::raw_string_ostream output(text);
  location.print(output);
  return text;
}

std::string severityName(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Error: return "error";
  case DiagnosticSeverity::Warning: return "warning";
  case DiagnosticSeverity::Remark: return "remark";
  case DiagnosticSeverity::Note: return "note";
  }
  llvm_unreachable("unknown MLIR diagnostic severity");
}

RISCVFailureCategory diagnosticCategory(Diagnostic &diagnostic, RISCVFailureCategory unclassified) {
  for (const DiagnosticArgument &argument : diagnostic.getMetadata()) {
    if (argument.getKind() != DiagnosticArgument::DiagnosticArgumentKind::String) continue;
    auto value = argument.getAsString();
    if (value == resourceMarker) return RISCVFailureCategory::ResourceLimit;
    if (value == unsupportedMarker) return RISCVFailureCategory::Unsupported;
    if (value == inputMarker) return RISCVFailureCategory::InvalidInput;
  }
  return unclassified;
}
} // namespace

llvm::StringRef weft::riscvFailureCategoryName(RISCVFailureCategory category) {
  switch (category) {
  case RISCVFailureCategory::InvalidInput: return "invalid-input";
  case RISCVFailureCategory::Unsupported: return "unsupported";
  case RISCVFailureCategory::ResourceLimit: return "resource-limit";
  case RISCVFailureCategory::CompilerError: return "compiler-error";
  }
  llvm_unreachable("unknown compilation failure category");
}

InFlightDiagnostic weft::rejectRISCVCandidate(Operation *operation,
    RISCVFailureCategory category, llvm::Twine message) {
  assert((category == RISCVFailureCategory::Unsupported ||
          category == RISCVFailureCategory::ResourceLimit) &&
         "candidate rejection must name an expected legality/resource failure");
  auto diagnostic = operation->emitError(message);
  diagnostic.getUnderlyingDiagnostic()->getMetadata().emplace_back(
      category == RISCVFailureCategory::ResourceLimit ? llvm::StringRef(resourceMarker) : llvm::StringRef(unsupportedMarker));
  return diagnostic;
}

InFlightDiagnostic weft::invalidRISCVInput(Operation *operation, llvm::Twine message) {
  auto diagnostic = operation->emitError(message);
  diagnostic.getUnderlyingDiagnostic()->getMetadata().emplace_back(llvm::StringRef(inputMarker));
  return diagnostic;
}

void RISCVCompilationReport::begin(llvm::StringRef input) {
  entry = input.str();
  stage = "input";
  pass.clear();
  failedIR.clear();
  category = RISCVFailureCategory::CompilerError;
  hasError = false;
  diagnostics.clear();
  pipeline.clear();
}

void RISCVCompilationReport::enter(llvm::StringRef nextStage, llvm::StringRef nextPass) {
  stage = nextStage.str();
  pass = nextPass.str();
}

void RISCVCompilationReport::startPass(llvm::StringRef nextStage, llvm::StringRef nextPass,
                                       llvm::StringRef invocation) {
  enter(nextStage, nextPass);
  // A running status is never published in a completed artifact. It is useful
  // internally so failed verification can identify the actual pass invocation.
  pipeline.push_back({stage, pass, "running", invocation.str()});
}

void RISCVCompilationReport::finishPass(bool successful) {
  assert(!pipeline.empty() && pipeline.back().status == "running");
  pipeline.back().status = successful ? "completed" : "failed";
  if (!successful) ensureFailure();
}

void RISCVCompilationReport::captureFailedIR(Operation *operation) {
  while (operation->getParentOp()) operation = operation->getParentOp();
  llvm::raw_string_ostream output(failedIR);
  operation->print(output, OpPrintingFlags().printGenericOpForm().enableDebugInfo());
}

void RISCVCompilationReport::classify(RISCVFailureCategory next) {
  if (!hasError) category = next;
  else if (next == RISCVFailureCategory::CompilerError || category == RISCVFailureCategory::CompilerError)
    category = RISCVFailureCategory::CompilerError;
  else if (next == RISCVFailureCategory::InvalidInput || category == RISCVFailureCategory::InvalidInput)
    category = RISCVFailureCategory::InvalidInput;
  // Multiple expected candidate rejections retain the first explicit reason.
  // A fatal diagnostic above always dominates a tagged candidate rejection.
  hasError = true;
}

void RISCVCompilationReport::capture(Diagnostic &diagnostic, RISCVFailureCategory unclassified) {
  RISCVCompilationDiagnostic record{severityName(diagnostic.getSeverity()),
      diagnostic.str(), locationText(diagnostic.getLocation()), {}};
  for (const Diagnostic &note : diagnostic.getNotes())
    record.notes.push_back({note.str(), locationText(note.getLocation())});
  diagnostics.push_back(std::move(record));
  if (diagnostic.getSeverity() == DiagnosticSeverity::Error)
    classify(diagnosticCategory(diagnostic, unclassified));
}

void RISCVCompilationReport::fail(RISCVFailureCategory next, llvm::StringRef message,
                                  llvm::StringRef location) {
  classify(next);
  diagnostics.push_back({"error", message.str(), location.str(), {}});
}

void RISCVCompilationReport::ensureFailure() {
  if (!hasError) fail(RISCVFailureCategory::CompilerError,
      "compilation failed without an emitted diagnostic");
}

llvm::json::Array RISCVCompilationReport::pipelineJSON() const {
  llvm::json::Array steps;
  for (const auto &step : pipeline)
    steps.push_back(llvm::json::Object{{"stage", step.stage}, {"pass", step.pass},
        {"status", step.status}, {"invocation", step.invocation}});
  return steps;
}

llvm::json::Object RISCVCompilationReport::compilationJSON() const {
  return llvm::json::Object{{"entry", entry}, {"pipeline", pipelineJSON()}};
}

llvm::json::Object RISCVCompilationReport::failureJSON() const {
  llvm::json::Array messages;
  for (const auto &diagnostic : diagnostics) {
    llvm::json::Array notes;
    for (const auto &note : diagnostic.notes)
      notes.push_back(llvm::json::Object{{"message", note.message}, {"location", note.location}});
    messages.push_back(llvm::json::Object{{"severity", diagnostic.severity},
        {"message", diagnostic.message}, {"location", diagnostic.location}, {"notes", std::move(notes)}});
  }
  return llvm::json::Object{{"kind", "weft-compilation-failure"},
      {"category", riscvFailureCategoryName(category)}, {"stage", stage}, {"pass", pass},
      {"diagnostics", std::move(messages)}, {"pipeline", pipelineJSON()}, {"failed_ir", failedIR}};
}
