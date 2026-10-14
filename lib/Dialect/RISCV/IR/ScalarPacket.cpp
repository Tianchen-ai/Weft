#include "Weft/Dialect/RISCV/IR/ScalarPacket.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft::riscv;

LogicalResult ScalarPacketAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t axis,
    int64_t steps, int64_t parts, int64_t registers, int64_t liveRegisters) {
  if (axis <= 0 || (steps != 2 && steps != 4))
    return error() << "scalar packet requires an existing axis and two or four coordinates";
  if (parts < 4 || parts > 16 || parts % steps || registers != parts)
    return error() << "scalar packet must name its exact bounded FPR residency";
  // The supported full-V targets use the standard F/D register file. This
  // counts scalar floating registers, not RVV groups or architectural x regs.
  if (liveRegisters < registers || liveRegisters > 32)
    return error() << "scalar packet must fit the complete live FPR frontier";
  return success();
}

LogicalResult weft::riscv::verifyScalarPacket(ScalarTileLoadOp operation) {
  auto packet = operation.getPacket();
  if (!packet) return success();
  auto type = operation.getResult().getType();
  if (operation.getGuarded() || (!type.getElementType().isF32() && !type.getElementType().isF64()))
    return operation.emitOpError("closed scalar packet requires complete f32 or f64 loads");
  auto coordinates = RegisterCoordinates::get(type.getLayout());
  auto axes = type.getAxisIds().asArrayRef();
  auto position = llvm::find(axes, packet->getAxis());
  if (!coordinates || coordinates->lanes() != 1 || position == axes.end() ||
      type.getShape()[position - axes.begin()] != packet->getSteps() ||
      coordinates->parts() != packet->getParts())
    return operation.emitOpError("closed scalar packet disagrees with its exact numeric coordinates");
  auto instruction = operation.getLeaf().getInstruction();
  if (instruction != "scalar.tile-load.unit" && instruction != "scalar.tile-load.strided")
    return operation.emitOpError("closed scalar packet requires an already selected address form");
  auto parameters = operation.getLeaf().getParameters();
  if (parameters.size() != 1 || parameters[0] != packet->getAxis())
    return operation.emitOpError("closed scalar packet axis differs from its selected address recurrence");
  auto live = scalarPacketResidency(operation);
  if (!live || *live > packet->getLiveRegisters())
    return operation.emitOpError("closed scalar packet understates its complete live FPR frontier");
  return success();
}

namespace {
struct ScalarSlot {
  Value root;
  int64_t part;
};

std::optional<int64_t> floatingParts(Type type) {
  if (isa<FloatType>(type)) return 1;
  auto numeric = dyn_cast<ValueType>(type);
  if (!numeric || !isa<FloatType>(numeric.getElementType()) ||
      numeric.getLayout().getCarrier() != "scalar") return 0;
  auto coordinates = RegisterCoordinates::get(numeric.getLayout());
  if (!coordinates || coordinates->lanes() != 1 || coordinates->parts() > 4096) return std::nullopt;
  return coordinates->parts();
}

bool scalarProjection(Operation *operation) {
  auto extract = dyn_cast<ExtractOp>(operation);
  return extract && extract.getLeaf().getInstruction() == "scalar.replica-project";
}

std::optional<ScalarSlot> scalarSlot(Value value, int64_t part, unsigned depth = 0) {
  if (depth > 32) return std::nullopt;
  if (auto extract = value.getDefiningOp<ExtractOp>(); extract && scalarProjection(extract)) {
    auto parameters = extract.getLeaf().getParameters();
    if (part < 0 || part >= int64_t(parameters.size())) return std::nullopt;
    return scalarSlot(extract.getInput(), parameters[part], depth + 1);
  }
  auto count = floatingParts(value.getType());
  if (!count || part < 0 || part >= *count) return std::nullopt;
  return ScalarSlot{value, part};
}
}

std::optional<int64_t> weft::riscv::scalarPacketResidency(ScalarTileLoadOp operation) {
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel) return std::nullopt;
  DominanceInfo dominance(kernel);
  llvm::DenseMap<Value, llvm::SmallDenseSet<int64_t, 16>> slots;
  bool known = true;
  auto add = [&](Value value, Operation *outside) {
    auto count = floatingParts(value.getType());
    if (!count) { known = false; return; }
    for (int64_t part = 0; part < *count; ++part) {
      auto slot = scalarSlot(value, part);
      if (!slot) { known = false; return; }
      if (outside) {
        Operation *owner = slot->root.getParentBlock()->getParentOp();
        if (owner == outside || outside->isProperAncestor(owner)) continue;
      }
      if (slot->root.getDefiningOp() == operation || dominance.dominates(slot->root, operation))
        slots[slot->root].insert(slot->part);
    }
  };
  add(operation.getResult(), nullptr);
  // Follow future uses, not every historical definition. Numeric values born
  // after this frontier do not occupy it; projected packet rows share slots.
  // Include captures in nested blocks but exclude their newly born iter_args.
  unsigned visited = 0;
  auto uses = [&](Operation *subtree, Operation *outside) {
    subtree->walk([&](Operation *user) {
      if (++visited > 20000 || !known) { known = false; return; }
      if (scalarProjection(user)) return;
      for (Value operand : user->getOperands()) add(operand, outside);
    });
  };
  // Values used after an enclosing if/loop remain live while this packet is
  // consumed, even when that value has no use in the packet's own block.
  for (Operation *scope = operation; scope != kernel; scope = scope->getParentOp()) {
    for (Operation *next = scope->getNextNode(); next; next = next->getNextNode()) uses(next, nullptr);
    // An invariant used before the packet in a repeating loop is needed on
    // the next iteration. A value born in that loop body is a fresh value on
    // the next iteration and must not be counted merely due to a past use.
    auto parent = scope->getParentOp();
    if (isa<scf::ForOp, scf::WhileOp>(parent)) uses(parent, parent);
    if (!known) return std::nullopt;
  }
  int64_t registers = 0;
  for (const auto &entry : slots) registers += entry.second.size();
  return registers;
}
