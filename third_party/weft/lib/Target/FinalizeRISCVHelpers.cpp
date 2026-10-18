#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace weft;

namespace {
class FinalizeRISCVHelpersPass
    : public PassWrapper<FinalizeRISCVHelpersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeRISCVHelpersPass)
  StringRef getArgument() const override { return "weft-riscv-finalize-helpers"; }
  StringRef getDescription() const override { return "Close native helper captures after physical resource selection"; }
  void runOnOperation() override {
    getOperation().walk<WalkOrder::PostOrder>([&](riscv::HelperOp helper) {
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(helper.getBody(), captures);
      // Descriptor bindings retain their explicit origin and index values.
      // Capture those dependencies as well; numerical producers stay outside.
      for (unsigned i = 0; i < captures.size(); ++i) {
        Value value = captures[i];
        if (auto definition = value.getDefiningOp(); definition &&
            !isa<riscv::MemoryViewOp, riscv::PointerCastOp, riscv::LocalViewOp>(definition) &&
            (isa<riscv::MemDescType, riscv::DomainType, riscv::PointType>(value.getType()) ||
             isa<riscv::FieldOp>(definition)))
          captures.insert(definition->operand_begin(), definition->operand_end());
      }
      helper.getCapturesMutable().assign(captures.getArrayRef());
      helper->setAttr("native_abi", StringAttr::get(&getContext(), "selected-parts"));
    });
  }
};
}

std::unique_ptr<Pass> weft::createFinalizeRISCVHelpersPass() {
  return std::make_unique<FinalizeRISCVHelpersPass>();
}
