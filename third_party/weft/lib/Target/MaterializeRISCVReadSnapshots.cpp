#include "Weft/Target/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/MemoryEffects.h"

#include "RISCVPhysicalSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>
#include <memory>

using namespace weft;

namespace {

int64_t ownerDomain(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto level =
            parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      return level.getDomainId();
  return 0;
}

class MaterializeRISCVReadSnapshotsPass
    : public mlir::PassWrapper<MaterializeRISCVReadSnapshotsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVReadSnapshotsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-read-snapshots";
  }
  llvm::StringRef getDescription() const override {
    return "Preserve encoded read values across interfering memory effects";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    int64_t identity = 0;
    getOperation().walk([&](riscv::KernelOp kernel) {
      for (int64_t aliasSet : kernel.getArgAliasSets())
        identity = std::max(identity, aliasSet + 1);
    });
    getOperation().walk([&](riscv::LocalAllocOp allocation) {
      auto type = allocation.getResult().getType();
      identity = std::max({identity, type.getAliasSet() + 1,
                           type.getBirthId() + 1});
    });
    bool failed = false;
    getOperation().walk([&](riscv::LoadOp load) {
      if (load.getSnapshotStorage() ||
          !riscv_internal::needsReadSnapshot(load))
        return;
      auto kernel = load->getParentOfType<riscv::KernelOp>();
      auto memory = load.getRegion().getType();
      auto value = mlir::dyn_cast<riscv::ValueType>(load.getResult().getType());
      auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
      const int64_t extent = value && value.getShape().size() == 1
                                 ? value.getShape()[0]
                                 : 0;
      const int64_t recordBytes = memory.getStorageBits() / 8;
      if (!kernel || kernel.getResourcesMaterialized() || !value ||
          value.getLayout().getCarrier() != "local" ||
          value.getLayout().getValidity() != "full" ||
          encoding.getKind() != "base" || extent <= 0 ||
          extent % memory.getElements() || memory.getStorageBits() % 8 ||
          recordBytes <= 0 || memory.getInterleaveRows() != 0 ||
          memory.getStrides().size() != 1 || memory.getStrides()[0] != 1 ||
          extent / memory.getElements() >
              std::numeric_limits<int64_t>::max() / recordBytes) {
        load.emitError(
            "interfering encoded read requires a static contiguous full-record snapshot");
        failed = true;
        return;
      }
      const int64_t bytes = extent / memory.getElements() * recordBytes;
      if (bytes > kernel.getTarget().getMaxPrivateStackBytes()) {
        load.emitError("encoded read snapshot exceeds target local storage");
        failed = true;
        return;
      }
      const int64_t owner = ownerDomain(load);
      rewriter.setInsertionPoint(load);
      auto byteCount = mlir::arith::ConstantIndexOp::create(rewriter,
          load.getLoc(), bytes);
      auto storageType = riscv::LocalType::get(
          &getContext(), rewriter.getIntegerType(8),
          riscv_internal::integers(rewriter, {}),
          riscv_internal::integers(rewriter, {}), bytes, memory.getAlignment(),
          identity, "pack", owner, identity, owner, "read-snapshot");
      ++identity;
      auto storage = riscv::LocalAllocOp::create(rewriter,
          load.getLoc(), storageType, byteCount);
      riscv_internal::copyOrigin(load, storage);
      load.getSnapshotStorageMutable().assign(storage.getResult());
      load.setLeafAttr(
          riscv_internal::readSnapshotLeaf(rewriter, kernel.getTarget(), bytes));
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

bool weft::riscv_internal::needsReadSnapshot(riscv::LoadOp load) {
  auto encoding =
      mlir::cast<kernel::EncodingType>(load.getRegion().getType().getEncoding());
  if (encoding.getKind() == "dense")
    return false;
  llvm::SmallVector<mlir::Value> pending{load.getResult()};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!pending.empty()) {
    mlir::Value value = pending.pop_back_val();
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (riscv::storageWrittenBetween(load.getRegion(), load, user))
        return true;
      if (mlir::isMemoryEffectFree(user) &&
          !mlir::isa<riscv::RegisterMaterializeOp,
                     riscv::EncodedLocalBindOp>(user))
        pending.append(user->result_begin(), user->result_end());
    }
  }
  return false;
}

std::unique_ptr<mlir::Pass> weft::createMaterializeRISCVReadSnapshotsPass() {
  return std::make_unique<MaterializeRISCVReadSnapshotsPass>();
}
