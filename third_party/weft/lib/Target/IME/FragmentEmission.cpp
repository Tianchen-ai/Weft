#include "FragmentEmission.h"
#include "Weft/Target/RISCVExtensions.h"

namespace weft {
#define WEFT_RISCV_EXTENSION(module) const RISCVExtension &get##module##Extension();
#include "WeftRISCVExtensionRegistry.inc"
#undef WEFT_RISCV_EXTENSION
} // namespace weft

namespace weft::riscv_emission {
#define WEFT_RISCV_EXTENSION(module)                                             \
  mlir::FailureOr<llvm::SmallVector<std::string>> emit##module##MMA(               \
      riscv::IMEFragmentMMAOp, llvm::ArrayRef<std::string>,                       \
      llvm::ArrayRef<std::string>, llvm::ArrayRef<std::string>,                  \
      const FragmentEmission &);
#include "WeftRISCVExtensionRegistry.inc"
#undef WEFT_RISCV_EXTENSION

mlir::FailureOr<llvm::SmallVector<std::string>> emitFragmentMMA(
    riscv::IMEFragmentMMAOp operation, llvm::ArrayRef<std::string> lhs,
    llvm::ArrayRef<std::string> rhs, llvm::ArrayRef<std::string> accumulators,
    const FragmentEmission &emission) {
  const auto *contract = findRISCVFragmentCapability(operation.getLeaf().getInstruction());
  if (!contract) {
    operation.emitOpError("selected fragment instruction has no registered contract");
    return mlir::failure();
  }
  // Dispatch the already selected instruction to its build-time owner. This
  // table is generated from the same module list as the runtime contracts.
#define WEFT_RISCV_EXTENSION(module)                                             \
  if (contract->extension == get##module##Extension().name)                     \
    return emit##module##MMA(operation, lhs, rhs, accumulators, emission);
#include "WeftRISCVExtensionRegistry.inc"
#undef WEFT_RISCV_EXTENSION
  operation.emitOpError("selected fragment extension has no terminal implementation");
  return mlir::failure();
}
} // namespace weft::riscv_emission
