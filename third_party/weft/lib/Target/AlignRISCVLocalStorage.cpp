#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
std::optional<int64_t> vectorAccessBytes(Operation *operation) {
  int64_t width = 0;
  auto collect = [&](Type type) {
    auto numeric = dyn_cast<riscv::ValueType>(type);
    if (!numeric || numeric.getLayout().getCarrier() != "rvv") return;
    // The memory op's selected carrier gives a finite byte span. No scalar
    // source loop is converted into a vector access to justify alignment.
    const auto layout = numeric.getLayout();
    if (layout.getSew() < 8 || layout.getVl() <= 0) return;
    width = std::max(width, layout.getVl() * (layout.getSew() / 8));
  };
  for (Type type : operation->getOperandTypes()) collect(type);
  for (Type type : operation->getResultTypes()) collect(type);
  return width > 0 ? std::optional<int64_t>(width) : std::nullopt;
}

llvm::DenseMap<Value, int64_t> requests(riscv::KernelOp kernel, int64_t lineBytes) {
  llvm::DenseMap<Value, int64_t> result;
  kernel.walk([&](Operation *operation) {
    auto memory = dyn_cast<MemoryEffectOpInterface>(operation);
    auto access = operation->getAttrOfType<riscv::AccessAttr>("access");
    if (!memory || !access || (access.getForm() != "unit" && access.getForm() != "segment")) return;
    auto width = vectorAccessBytes(operation);
    if (!width) return;
    int64_t alignment = std::min(*width, lineBytes);
    // Only whole power-of-two physical spans have an alignment contract here.
    // This does not reinterpret a non-power-of-two active tail as a full VL.
    if (alignment < 16 || (alignment & (alignment - 1))) return;
    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);
    for (const auto &effect : effects) {
      if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) continue;
      Value descriptor = effect.getValue();
      if (!descriptor || !isa<riscv::MemDescType>(descriptor.getType())) continue;
      auto origins = riscv::storageOrigins(descriptor);
      if (!origins.isOwnedLocal()) continue;
      for (Value root : origins.allocations) {
        auto allocation = root.getDefiningOp<riscv::LocalAllocOp>();
        if (!allocation) continue;
        auto type = allocation.getResult().getType();
        // Improve compiler-created numeric representations. Author births,
        // external workspaces, artifact bytes, spills and dynamic capacities
        // retain their independently selected placement contracts.
        if (type.getPurpose() != "pack" || type.getSchema() != "physical-share" ||
            type.getSizeBytes() < lineBytes || alignment <= type.getAlignment()) continue;
        auto found = result.find(root);
        if (found == result.end()) result.try_emplace(root, alignment);
        else found->second = std::max(found->second, alignment);
      }
    }
  });
  return result;
}

class AlignRISCVLocalStoragePass
    : public PassWrapper<AlignRISCVLocalStoragePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlignRISCVLocalStoragePass)
  StringRef getArgument() const override { return "weft-riscv-align-local-storage"; }
  StringRef getDescription() const override { return "Align owned dense vector storage within the shared lexical byte budget"; }
  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](riscv::KernelOp kernel) {
      auto cache = dyn_cast<riscv::DataCacheAttr>(kernel.getTarget().getDataCache());
      if (!cache) return;
      if (kernel.getResourcesMaterialized()) {
        kernel.emitError("local alignment selection requires open physical resources");
        invalid = true;
        return;
      }
      auto footprint = staticLocalFootprint(kernel);
      if (!footprint) {
        kernel.emitError("local alignment cannot represent the current lexical storage footprint");
        invalid = true;
        return;
      }
      // Strengthening alignment must not reduce an unknown runtime shape's
      // available capacity. Static representations have an exact byte budget.
      if (footprint->hasDynamicObjects) return;
      // Preserve capacity for a program that still needs mandatory spills.
      // Only spend padding after the current complete vector/fragment live
      // set already fits; alignment is not allowed to force a new rejection.
      auto pressure = physicalRegisterPressure(kernel);
      if (!pressure || pressure->total() > kernel.getTarget().getVectorRegisters()) return;
      auto proposed = requests(kernel, cache.getLineBytes());
      if (proposed.empty()) return;
      llvm::DenseMap<Value, int64_t> selected;
      SmallVector<riscv::LocalAllocOp> allocations;
      kernel.walk([&](riscv::LocalAllocOp allocation) { allocations.push_back(allocation); });
      // Stable allocation order gives a deterministic selection when several
      // independently useful alignments cannot fit together. Estimate before
      // mutation; final resource closure still rechecks the realized program.
      for (auto allocation : allocations) {
        auto candidate = proposed.find(allocation.getResult());
        if (candidate == proposed.end()) continue;
        selected.try_emplace(candidate->first, candidate->second);
        auto trial = staticLocalFootprint(kernel, selected);
        if (!trial || trial->bytes > kernel.getTarget().getMaxPrivateStackBytes())
          selected.erase(candidate->first);
      }
      for (auto allocation : allocations) {
        auto placement = selected.find(allocation.getResult());
        if (placement == selected.end()) continue;
        auto type = allocation.getResult().getType();
        allocation.getResult().setType(riscv::LocalType::get(&getContext(), type.getElementType(),
            type.getShape(), type.getAxisIds(), type.getSizeBytes(), placement->second, type.getAliasSet(),
            type.getPurpose(), type.getOwnerDomainId(), type.getBirthId(), type.getLifetimeEndDomainId(), type.getSchema()));
      }
    });
    if (invalid) signalPassFailure();
  }
};
}

std::unique_ptr<Pass> weft::createAlignRISCVLocalStoragePass() {
  return std::make_unique<AlignRISCVLocalStoragePass>();
}
