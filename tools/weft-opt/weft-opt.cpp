#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVPipeline.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  weft::registerRISCVPasses();
  weft::registerRISCVPipelines();
  mlir::DialectRegistry registry;
  registry.insert<weft::kernel::WEFTKernelDialect,
                  weft::riscv::WEFTRISCVDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv,
                        "Weft Canonical and RISC-V Physical IR driver\n",
                        registry));
}
