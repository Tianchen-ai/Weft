#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/Atomic.h"
#include "Verification.h"

using namespace mlir;
using namespace weft::riscv;
namespace {
Type element(Type type) {
  if (auto value = dyn_cast<ValueType>(type)) return value.getElementType();
  return type;
}
bool domain(Type a, Type b) {
  auto x = dyn_cast<ValueType>(a), y = dyn_cast<ValueType>(b);
  if (!x || !y) return !x && !y;
  return x.getShape() == y.getShape() && x.getAxisIds() == y.getAxisIds();
}
LogicalResult verify(Operation *op, Value base, Value offsets, Value mask, Value value, LeafAttr leaf) {
  auto storage = cast<MemDescType>(base.getType());
  auto encoding = cast<weft::kernel::EncodingType>(storage.getEncoding());
  auto index = dyn_cast<IntegerType>(element(offsets.getType()));
  auto stride = op->getAttrOfType<IntegerAttr>("element_stride");
  if (op->hasAttr("element_stride") && (!stride || !stride.getType().isInteger(64)))
    return op->emitOpError("element_stride must be an i64 integer attribute");
  auto shaped = dyn_cast<ValueType>(value.getType());
  bool affine = stride && shaped && shaped.getShape().size() == 1 &&
                !isa<ValueType>(offsets.getType()) && stride.getInt() != 0;
  if (stride && !affine) return op->emitOpError("affine pointer memory requires a scalar base and a nonzero one-axis element stride");
  Type payload = element(value.getType());
  if (stride) {
    int64_t bytes = std::max<unsigned>(8, payload.getIntOrFloatBitWidth()) / 8;
    if (stride.getInt() > INT64_MAX / bytes || stride.getInt() < INT64_MIN / bytes)
      return op->emitOpError("affine pointer byte stride must fit signed XLEN");
  }
  std::string family;
  if (auto integer = dyn_cast<IntegerType>(payload))
    family = (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
  else { llvm::raw_string_ostream stream(family); payload.print(stream); }
  if (storage.getAddressClass() != "pointer" || !storage.getShape().empty() ||
      encoding.getKind() != "dense" || encoding.getFamily() != family || !index || !index.isSigned() || index.getWidth() != 64 ||
      !element(mask.getType()).isInteger(1) || (!affine && !domain(value.getType(), offsets.getType())) ||
      (isa<ValueType>(mask.getType()) && !domain(value.getType(), mask.getType())))
    return op->emitOpError("pointer memory requires an origin, i64 element offsets and an i1 mask on the value domain");
  if (leaf.getEngine() == "unselected") return success();
  bool vector = shaped && shaped.getLayout().getCarrier() == "rvv";
  std::string form = vector ? (affine ? (stride.getInt() == 1 ? "rvv.pointer-unit." : "rvv.pointer-strided.") : "rvv.pointer-indexed.") : "scalar.pointer-masked.";
  form += op->getNumResults() ? "load" : "store";
  auto offsetType = dyn_cast<ValueType>(offsets.getType());
  int64_t temporary = vector ? 1 + (affine ? 0 : (offsetType.getLayout().getLmulEighths() + 7) / 8) : 0;
  if (vector && op->getNumResults() && !isa<ValueType>(op->getOperand(3).getType()))
    temporary += (shaped.getLayout().getLmulEighths() + 7) / 8;
  if (leaf.getFamily() != "pointer-memory" || leaf.getInstruction() != form || leaf.getSpelling() != form ||
      leaf.getEngine() != (vector ? "rvv" : "scalar") || leaf.getMask() != "explicit" ||
      leaf.getTemporaryGroups() != temporary || leaf.getFragmentGroups() || leaf.getLocalBytes())
    return op->emitOpError("pointer memory requires its exact masked leaf");
  if (!shaped) return success();
  auto reference = shaped.getLayout();
  if ((reference.getCarrier() != "rvv" && reference.getCarrier() != "scalar") || reference.getReplicaGroup() != 1)
    return op->emitOpError("pointer memory requires a selected numeric carrier");
  for (Value input : {offsets, mask}) {
    if (!isa<ValueType>(input.getType())) continue;
    auto layout = cast<ValueType>(input.getType()).getLayout();
    if (layout.getCarrier() != reference.getCarrier() || layout.getTimeFactors() != reference.getTimeFactors() ||
        layout.getLaneFactors() != reference.getLaneFactors() || layout.getReplicaFactors() != reference.getReplicaFactors() ||
        layout.getFragmentFactors() != reference.getFragmentFactors() || layout.getLocalFactors() != reference.getLocalFactors() ||
        layout.getReplicaGroup() != 1)
      return op->emitOpError("pointer memory operands must preserve the same part/lane coordinates");
  }
  return success();
}
}
LogicalResult PointerCastOp::verify() {
  auto input = getInput().getType(), result = getResult().getType();
  if (input.getAddressClass() != "pointer" || result.getAddressClass() != "pointer" ||
      cast<weft::kernel::EncodingType>(input.getEncoding()).getKind() != "dense" ||
      cast<weft::kernel::EncodingType>(result.getEncoding()).getKind() != "dense" ||
      input.getElements() != 1 || result.getElements() != 1 || input.getInterleaveRows() || result.getInterleaveRows() ||
      input.getStorageBits() != result.getStorageBits() || input.getAlignment() != result.getAlignment() ||
      input.getAliasSet() != result.getAliasSet() || input.getAccess() != result.getAccess() ||
      !input.getShape().empty() || !result.getShape().empty())
    return emitOpError("pointer reinterpretation preserves storage width, alignment, access and alias provenance");
  return success();
}
LogicalResult PointerLoadOp::verify() {
  bool shapedFill = isa<ValueType>(getOther().getType());
  if ((shapedFill && !domain(getOther().getType(), getResult().getType())) || element(getOther().getType()) != element(getResult().getType()) ||
      (shapedFill && getLeaf().getEngine() != "unselected" && getOther().getType() != getResult().getType()))
    return emitOpError("fill must preserve the load domain and its selected result representation");
  return ::verify(*this, getBase(), getOffsets(), getMask(), getResult(), getLeaf());
}
LogicalResult PointerStoreOp::verify() {
  return ::verify(*this, getBase(), getOffsets(), getMask(), getValue(), getLeaf());
}

LogicalResult PointerAtomicOp::verify() {
  auto memory = getBase().getType();
  auto encoding = cast<weft::kernel::EncodingType>(memory.getEncoding());
  Type payload = element(getResult().getType());
  auto index = dyn_cast<IntegerType>(element(getOffsets().getType()));
  if (memory.getAddressClass() != "pointer" || memory.getAccess() != "readwrite" ||
      memory.getAlignment() < payload.getIntOrFloatBitWidth() / 8 ||
      encoding.getKind() != "dense" || !index || !index.isSigned() || index.getWidth() != 64 ||
      !element(getMask().getType()).isInteger(1) || !domain(getOffsets().getType(), getResult().getType()) ||
      element(getValue().getType()) != payload || element(getCompare().getType()) != payload)
    return emitOpError("atomic requires an aligned readwrite pointer and matching typed operands");
  for (Value value : {getMask(), getValue(), getCompare()})
    if (isa<ValueType>(value.getType()) && !domain(value.getType(), getResult().getType()))
      return emitOpError("atomic operands must share the pointer domain");
  if (failed(weft::verifyAtomicContract(*this, payload, getKind(), getOrdering(), getScope()))) return failure();
  if (getLeaf().getEngine() == "unselected") return success();
  if (isa<ValueType>(getResult().getType()) || getLeaf().getEngine() != "scalar" ||
      getLeaf().getFamily() != "pointer-atomic" || getLeaf().getInstruction() != "scalar.pointer-atomic." + getKind().str() ||
      getLeaf().getSpelling() != getLeaf().getInstruction() || getLeaf().getTemporaryGroups() || getLeaf().getLocalBytes())
    return emitOpError("atomic requires its explicit scalar instruction after lane materialization");
  return success();
}
