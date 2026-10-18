#ifndef WEFT_TARGET_RISCVCOMPILATION_H
#define WEFT_TARGET_RISCVCOMPILATION_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

namespace mlir { class Operation; }

namespace weft {

enum class RISCVFailureCategory { InvalidInput, Unsupported, ResourceLimit, CompilerError };

llvm::StringRef riscvFailureCategoryName(RISCVFailureCategory category);

// Only the origin of a known unsupported realization or exhausted resource may
// declare a candidate rejection. Ordinary verification and implementation errors
// retain the fatal default, even when they occur inside a resource pass.
mlir::InFlightDiagnostic rejectRISCVCandidate(
    mlir::Operation *operation, RISCVFailureCategory category,
    llvm::Twine message = {});

mlir::InFlightDiagnostic invalidRISCVInput(mlir::Operation *operation,
                                          llvm::Twine message);

struct RISCVCompilationNote {
  std::string message;
  std::string location;
};

struct RISCVCompilationDiagnostic {
  std::string severity;
  std::string message;
  std::string location;
  std::vector<RISCVCompilationNote> notes;
};

struct RISCVPassExecution {
  std::string stage;
  std::string pass;
  std::string status;
  std::string invocation;
};

// A per-invocation observation of the real pass manager. This is not a program
// representation, an alternative schedule, or permission to retry a lowering.
// A failed invocation retains its completed prefix and the responsible pass.
class RISCVCompilationReport {
public:
  void begin(llvm::StringRef input);
  void enter(llvm::StringRef stage, llvm::StringRef pass = {});
  void startPass(llvm::StringRef stage, llvm::StringRef pass, llvm::StringRef invocation);
  void finishPass(bool successful);
  void captureFailedIR(mlir::Operation *operation);
  void capture(mlir::Diagnostic &diagnostic,
               RISCVFailureCategory unclassified = RISCVFailureCategory::CompilerError);
  void fail(RISCVFailureCategory category, llvm::StringRef message,
            llvm::StringRef location = {});
  void ensureFailure();

  llvm::StringRef currentStage() const { return stage; }
  llvm::StringRef currentPass() const { return pass; }
  bool failed() const { return hasError; }
  llvm::json::Object compilationJSON() const;
  llvm::json::Object failureJSON() const;

private:
  void classify(RISCVFailureCategory category);
  llvm::json::Array pipelineJSON() const;
  std::string entry;
  std::string stage;
  std::string pass;
  std::string failedIR;
  RISCVFailureCategory category = RISCVFailureCategory::CompilerError;
  bool hasError = false;
  std::vector<RISCVCompilationDiagnostic> diagnostics;
  std::vector<RISCVPassExecution> pipeline;
};

} // namespace weft
#endif
