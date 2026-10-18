#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/RISCVCompilation.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <string>

using namespace weft;

namespace {

bool feedsPartialProgramInBlock(mlir::Value source, mlir::Block *block) {
  llvm::SmallVector<mlir::Value> worklist{source};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *consumer = use.getOwner();
      if (consumer->getBlock() != block)
        continue;
      if (mlir::isa<riscv::RVVPartialSetOp,
                    riscv::RVVPartialCollectOp, riscv::RVVFinalizeWidenDotOp,
                    riscv::IMEFragmentMMAOp>(consumer))
        return true;
      if (consumer->getNumRegions() != 0 || consumer->getNumResults() != 1 ||
          !mlir::isMemoryEffectFree(consumer))
        continue;
      worklist.push_back(consumer->getResult(0));
    }
  }
  return false;
}

riscv::LeafAttr scalarPrimeLeaf(mlir::Builder &builder,
                                riscv::LeafAttr leaf,
                                llvm::StringRef instruction) {
  return riscv_internal::leaf(
      builder, leaf.getEngine(), leaf.getFamily(), instruction, instruction,
      leaf.getOperandGroups(), leaf.getResultGroups(),
      leaf.getTemporaryGroups(), leaf.getFragmentGroups(), leaf.getMask(),
      leaf.getTail(), leaf.getParameters().asArrayRef(), leaf.getLocalBytes());
}

bool selectFirstPartialStorageRead(mlir::scf::ForOp loop,
                                   mlir::Builder &builder) {
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (auto load = mlir::dyn_cast<riscv::IMEFragmentStorageLoadOp>(operation)) {
      auto type = load.getResult().getType();
      const int64_t parts = riscv::fragmentPartCount(type);
      if (parts < 2 || parts > 4 || type.getLayout().getValidity() != "full" ||
          type.getPacking().getStorageBits() != 8 ||
          !feedsPartialProgramInBlock(load.getResult(), loop.getBody()))
        continue;
      auto instruction = load.getLeaf().getInstruction();
      if (instruction == "ime.field-raw-load.scalar-prime" ||
          instruction == "ime.field-raw-load.grouped.scalar-prime") return true;
      if (instruction != "ime.field-raw-load" && instruction != "ime.field-raw-load.grouped") return false;
      load.setLeafAttr(scalarPrimeLeaf(builder, load.getLeaf(),
          instruction == "ime.field-raw-load.grouped" ? "ime.field-raw-load.grouped.scalar-prime"
                                                     : "ime.field-raw-load.scalar-prime"));
      return true;
    }
    if (auto load = mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(operation)) {
      if (load.getPlan().getKind() != "layered" ||
          load.getWindowOffsets().size() != 1 ||
          !feedsPartialProgramInBlock(load.getResult(), loop.getBody()))
        continue;
      llvm::StringRef instruction = load.getLeaf().getInstruction();
      if (instruction == "rvv.replica-storage-load.layered.scalar-prime")
        return true;
      if (instruction != "rvv.replica-storage-load.layered")
        return false;
      load.setLeafAttr(scalarPrimeLeaf(
          builder, load.getLeaf(),
          "rvv.replica-storage-load.layered.scalar-prime"));
      return true;
    }
    if (auto load = mlir::dyn_cast<riscv::RVVLayeredStorageLoadOp>(operation)) {
      if (!feedsPartialProgramInBlock(load.getResult(), loop.getBody()))
        continue;
      llvm::StringRef instruction = load.getLeaf().getInstruction();
      if (instruction == "rvv.layered-storage-load.scalar-prime")
        return true;
      if (instruction != "rvv.layered-storage-load")
        return false;
      load.setLeafAttr(scalarPrimeLeaf(
          builder, load.getLeaf(), "rvv.layered-storage-load.scalar-prime"));
      return true;
    }
  }
  return false;
}

class SelectRISCVScalarLoadPrimesPass
    : public mlir::PassWrapper<SelectRISCVScalarLoadPrimesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVScalarLoadPrimesPass)
  explicit SelectRISCVScalarLoadPrimesPass(bool value = false) { enabled = value; }
  SelectRISCVScalarLoadPrimesPass(const SelectRISCVScalarLoadPrimesPass &other)
      : PassWrapper(other) { copyOptionValuesFrom(&other); }

  llvm::StringRef getArgument() const override {
    return "weft-riscv-select-scalar-load-primes";
  }
  llvm::StringRef getDescription() const override {
    return "Select one target-local scalar-prime leaf for a typed contraction supply in each physical loop";
  }

  void runOnOperation() override {
    if (!enabled)
      return;
    mlir::Builder builder(&getContext());
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::scf::ForOp loop) { loops.push_back(loop); });
    bool selected = false;
    for (mlir::scf::ForOp loop : loops)
      selected |= selectFirstPartialStorageRead(loop, builder);
    if (!selected) {
      rejectRISCVCandidate(getOperation(), RISCVFailureCategory::Unsupported,
          "scalar-load-prime binding requires a grouped/layered partial supply "
          "or a complete multi-atom fragment storage supply");
      signalPassFailure();
    }
  }

private:
  Option<bool> enabled{*this, "enabled", llvm::cl::init(false),
      llvm::cl::desc("Require and select an eligible scalar-prime load")};
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createSelectRISCVScalarLoadPrimesPass(bool enabled) {
  return std::make_unique<SelectRISCVScalarLoadPrimesPass>(enabled);
}
