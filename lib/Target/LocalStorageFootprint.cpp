#include "LocalMemory.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
class Footprint {
public:
  explicit Footprint(const llvm::DenseMap<Value, int64_t> &alignments)
      : alignments(alignments) {}

  std::optional<StaticLocalFootprint> run(riscv::KernelOp kernel) {
    visit(kernel.getBody().front(), 0);
    if (!known || overrides != alignments.size()) return std::nullopt;
    return StaticLocalFootprint{peak, dynamic};
  }

private:
  const llvm::DenseMap<Value, int64_t> &alignments;
  int64_t peak = 0;
  bool dynamic = false;
  bool known = true;
  size_t overrides = 0;

  void add(int64_t &live, int64_t bytes) {
    if (bytes < 0 || llvm::AddOverflow(live, bytes, live)) known = false;
  }

  int64_t visit(Block &block, int64_t live) {
    for (Operation &operation : block) {
      if (!known) return live;
      if (auto leaf = operation.getAttrOfType<riscv::LeafAttr>("leaf")) add(live, leaf.getLocalBytes());
      if (auto allocation = dyn_cast<riscv::LocalAllocOp>(operation)) {
        auto type = allocation.getResult().getType();
        int64_t alignment = type.getAlignment();
        auto replacement = alignments.find(allocation.getResult());
        if (replacement != alignments.end()) {
          ++overrides;
          if (replacement->second < alignment) { known = false; return live; }
          alignment = replacement->second;
        }
        if (alignment <= 0 || (alignment & (alignment - 1))) { known = false; return live; }
        if (type.getSizeBytes() < 0) {
          // Dynamic padding belongs to the dynamic guard's live-byte SSA, not
          // to the static capacity subtracted from that guard's target limit.
          dynamic = true;
        } else {
          add(live, type.getSizeBytes());
          add(live, alignment - 1);
        }
      }
      if (!known) return live;
      peak = std::max(peak, live);
      if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        // The while before-region may carry private storage to the after
        // region. Preserve the same lexical lifetime used by capacity guards.
        int64_t before = visit(*loop.getBeforeBody(), live);
        visit(*loop.getAfterBody(), before);
      } else {
        // Exclusive branches and repeated loop iterations do not coexist.
        // Their enclosing allocations remain live in each nested region.
        for (Region &region : operation.getRegions())
          for (Block &nested : region) visit(nested, live);
      }
    }
    return live;
  }
};
}

std::optional<StaticLocalFootprint> weft::riscv_internal::staticLocalFootprint(
    riscv::KernelOp kernel, const llvm::DenseMap<Value, int64_t> &alignments) {
  return Footprint(alignments).run(kernel);
}
