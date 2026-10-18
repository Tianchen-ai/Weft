#ifndef WEFT_TARGET_RISCVPIPELINE_H
#define WEFT_TARGET_RISCVPIPELINE_H

#include "Weft/Target/RISCVCompiler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
class OpPassManager;
class Pass;
} // namespace mlir

namespace weft {

/// MLIR's ordinary string option printer does not protect a typed attribute's
/// embedded spaces and commas. This option retains the exact text and prints
/// the standard braced option value, so a printed pipeline is parseable.
struct RISCVPipelineTarget {
  std::string text;
  bool operator==(const RISCVPipelineTarget &other) const { return text == other.text; }
  bool operator!=(const RISCVPipelineTarget &other) const { return !(*this == other); }
};

/// One execution occurrence, including repeated instances of the same pass.
/// The descriptors and builders use the same ordered registry.
struct RISCVPipelinePass {
  llvm::StringRef argument;
  llvm::StringRef stage;
};

llvm::ArrayRef<RISCVPipelinePass>
getRISCVPhysicalPipeline(bool stopBeforeLayout = false);
llvm::ArrayRef<RISCVPipelinePass> getRISCVLayoutPipeline();
llvm::ArrayRef<RISCVPipelinePass> getRISCVPreparePipeline(RISCVCheckpoint checkpoint);
llvm::ArrayRef<RISCVPipelinePass> getRISCVResumePipeline(RISCVCheckpoint checkpoint);

void buildRISCVPhysicalPipeline(mlir::OpPassManager &manager,
                               const RISCVCompilerOptions &options,
                               bool stopBeforeLayout = false);
void buildRISCVLayoutPipeline(mlir::OpPassManager &manager,
                             bool scalarLoadPrime);
void buildRISCVPreparePipeline(mlir::OpPassManager &manager,
                              const RISCVCompilerOptions &options, RISCVCheckpoint checkpoint);
struct RISCVRemainingBinding {
  RISCVCheckpoint checkpoint;
  int64_t lmulEighths;
  bool scalarLoadPrime;
};
void buildRISCVResumePipeline(mlir::OpPassManager &manager,
                             RISCVRemainingBinding binding);

/// Checkpoints own their remaining binding; resumption cannot replace it.
llvm::StringRef checkpointName(RISCVCheckpoint checkpoint);
llvm::StringRef checkpointEntry(RISCVCheckpoint checkpoint);
mlir::LogicalResult markRISCVPipelineInput(mlir::ModuleOp module, RISCVRemainingBinding binding);
mlir::FailureOr<RISCVRemainingBinding> readRISCVPipelineInput(
    mlir::ModuleOp module, RISCVCheckpoint checkpoint);

/// Register the complete individual pass surface and the shared main pipelines.
void registerRISCVPasses();
void registerRISCVPipelines();

/// Preserve the existing typed target attribute in printed pass pipelines.
/// Target capability fields are not re-inferred from a shortened ISA spelling.
std::string serializeRISCVPipelineTarget(const RISCVTargetProfile &target);

/// Constructor used by the registered pipeline after parsing its bindings.
/// The serialized target is verified by the conversion pass before rewriting.
std::unique_ptr<mlir::Pass>
createConvertWeftToRISCVPass(llvm::StringRef serializedTarget,
                           const RISCVCompilerOptions &options);

} // namespace weft

namespace llvm::cl {

template <>
class parser<weft::RISCVPipelineTarget>
    : public basic_parser<weft::RISCVPipelineTarget> {
public:
  explicit parser(Option &option) : basic_parser(option) {}
  bool parse(Option &, StringRef, StringRef text,
             weft::RISCVPipelineTarget &value) {
    value.text = text.str();
    return false;
  }
  static void print(raw_ostream &output, const weft::RISCVPipelineTarget &value) {
    output << '{' << value.text << '}';
  }
  StringRef getValueName() const override { return "target-attribute"; }
  void printOptionDiff(const Option &option,
                       const weft::RISCVPipelineTarget &,
                       const OptionValue<weft::RISCVPipelineTarget> &,
                       size_t width) const {
    printOptionNoValue(option, width);
  }
};

} // namespace llvm::cl

#endif // WEFT_TARGET_RISCVPIPELINE_H
