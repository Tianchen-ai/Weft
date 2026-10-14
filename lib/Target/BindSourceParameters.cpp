#include "Weft/Target/SourceSpecialization.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Target/RISCVCompilation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace weft;

namespace {

class BindSourceParametersPass
    : public PassWrapper<BindSourceParametersPass, OperationPass<ModuleOp>> {
public:
  BindSourceParametersPass() = default;
  BindSourceParametersPass(const BindSourceParametersPass &other)
      : PassWrapper(other) {
    copyOptionValuesFrom(&other);
  }
  explicit BindSourceParametersPass(const llvm::StringMap<int64_t> &bindings) {
    llvm::SmallVector<std::string> values;
    for (const auto &binding : bindings)
      values.push_back((binding.getKey() + "=" + llvm::Twine(binding.second)).str());
    llvm::sort(values);
    meta = values;
  }

  llvm::StringRef getArgument() const override {
    return "weft-bind-source-parameters";
  }
  llvm::StringRef getDescription() const override {
    return "Specialize declared canonical source-auto domains to concrete singleton bindings";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<kernel::WEFTKernelDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<int64_t> requested;
    for (llvm::StringRef entry : meta) {
      auto [name, spelling] = entry.split('=');
      int64_t value = 0;
      if (name.empty() || spelling.getAsInteger(10, value) || value <= 0) {
        invalidRISCVInput(module, "source meta binding must be NAME=POSITIVE_INTEGER: ") << entry;
        signalPassFailure();
        return;
      }
      if (!requested.try_emplace(name, value).second) {
        invalidRISCVInput(module, "duplicate source meta binding: ") << name;
        signalPassFailure();
        return;
      }
    }

    bool physical = false;
    module.walk([&](Operation *operation) {
      physical |= operation->getName().getDialectNamespace() == "weft_riscv";
    });
    if (physical) {
      invalidRISCVInput(module,
          "source parameter specialization accepts Canonical Kernel IR before target lowering");
      signalPassFailure();
      return;
    }
    llvm::SmallVector<kernel::KernelOp> kernels(module.getOps<kernel::KernelOp>());
    if (kernels.empty()) {
      invalidRISCVInput(module, "source parameter specialization requires canonical kernels");
      signalPassFailure();
      return;
    }

    llvm::StringSet<> declared;
    auto symbols = module.walk([&](kernel::SymbolOp symbol) {
      if (symbol.getKind() != "source_auto")
        return WalkResult::advance();
      if (!symbol->getParentOfType<kernel::KernelOp>()) {
        invalidRISCVInput(symbol, "source auto parameter must belong to a canonical kernel");
        return WalkResult::interrupt();
      }
      declared.insert(symbol.getName());
      return WalkResult::advance();
    });
    if (symbols.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    for (const auto &binding : requested) {
      if (declared.contains(binding.getKey()))
        continue;
      invalidRISCVInput(module, "unknown source auto binding: ") << binding.getKey();
      signalPassFailure();
      return;
    }

    // Complete validation precedes mutation, including declarations in later
    // kernels. One bad declaration must not leave a half-specialized candidate.
    llvm::SmallVector<std::pair<kernel::SymbolOp, int64_t>> selected;
    for (kernel::KernelOp kernel : kernels) {
      llvm::StringMap<int64_t> kernelBindings;
      auto checked = kernel.getBody().walk([&](kernel::SymbolOp symbol) {
        if (symbol.getKind() != "source_auto")
          return WalkResult::advance();
        int64_t value;
        auto supplied = requested.find(symbol.getName());
        if (supplied != requested.end()) {
          value = supplied->second;
        } else if (symbol.getChoices().size() == 1) {
          value = symbol.getChoices()[0];
        } else {
          invalidRISCVInput(symbol, "source auto parameter '") << symbol.getName()
              << "' requires one concrete meta binding";
          return WalkResult::interrupt();
        }
        if (!symbol.getChoices().empty() &&
            !llvm::is_contained(symbol.getChoices(), value)) {
          invalidRISCVInput(symbol, "source auto binding is outside its declared finite domain: ")
              << symbol.getName() << '=' << value;
          return WalkResult::interrupt();
        }
        auto [position, inserted] = kernelBindings.try_emplace(symbol.getName(), value);
        if (!inserted && position->second != value) {
          invalidRISCVInput(symbol,
              "source auto declarations require conflicting values within one kernel: ")
              << symbol.getName();
          return WalkResult::interrupt();
        }
        selected.emplace_back(symbol, value);
        return WalkResult::advance();
      });
      if (checked.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }
    Builder builder(&getContext());
    for (auto [symbol, value] : selected)
      symbol->setAttr("choices", builder.getDenseI64ArrayAttr({value}));
  }

private:
  ListOption<std::string> meta{*this, "meta",
      llvm::cl::desc("Explicit source-auto bindings NAME=POSITIVE_INTEGER")};
};

} // namespace

std::unique_ptr<Pass> weft::createBindSourceParametersPass(
    const llvm::StringMap<int64_t> &bindings) {
  return std::make_unique<BindSourceParametersPass>(bindings);
}
