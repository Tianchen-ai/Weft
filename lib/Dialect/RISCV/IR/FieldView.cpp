#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/BuiltinOps.h"
#include <algorithm>

using namespace mlir;
using namespace weft;

LogicalResult riscv::FieldViewOp::verify() {
  auto field = getField().getDefiningOp<FieldOp>();
  auto source = getField().getType(), result = getResult().getType();
  auto owner = field ? dyn_cast<MemDescType>(field.getOwner().getType()) : MemDescType();
  auto access = field ? field.getAccess() : AccessAttr();
  if (!owner || owner.getShape().size() != 1 || source.getShape().size() != 1 ||
      owner.getShape()[0] != owner.getElements() || source.getShape()[0] <= 0 ||
      owner.getInterleaveRows() != 0 ||
      owner.getStrides().asArrayRef() != ArrayRef<int64_t>{1} ||
      !access || access.getMapping() != "natural" || access.getBitOffset() % 8 ||
      source.getStorageBits() % 8 || source.getStorageBits() < 8 || source.getStorageBits() > 64 ||
      owner.getElements() % source.getShape()[0] ||
      access.getStorageBits() / source.getStorageBits() != source.getShape()[0] ||
      access.getAlignment() < source.getStorageBits() / 8)
    return emitOpError("field view requires one complete aligned natural non-interleaved record field");
  auto encoding = cast<kernel::EncodingType>(owner.getEncoding());
  EncodingDeclOp declaration;
  for (auto candidate : getOperation()->getParentOfType<ModuleOp>().getOps<EncodingDeclOp>())
    if (candidate.getSymName() == encoding.getFamily()) declaration = candidate;
  if (encoding.getKind() != "base" || !declaration || declaration.getByteOrder() != "little" ||
      declaration.getBitOrder() != "lsb_first")
    return emitOpError("field view requires native little-endian natural element storage");
  if (result.getEncoding() != source.getEncoding() || result.getShape() != source.getShape() ||
      result.getAxisIds() != source.getAxisIds() || result.getAddressClass() != "slice" ||
      result.getAccess() != source.getAccess() || result.getAliasSet() != source.getAliasSet() ||
      result.getLayoutIdentity() != source.getLayoutIdentity() || result.getElements() != 1 ||
      result.getStorageBits() != source.getStorageBits() || result.getInterleaveRows() != 0 ||
      result.getAlignment() != std::min(access.getAlignment(), source.getStorageBits() / 8) || result.getStrides().asArrayRef() != ArrayRef<int64_t>{1} ||
      result.getOrigins().asArrayRef() != ArrayRef<int64_t>{-1})
    return emitOpError("field view must preserve storage identity and expose unit stride with an explicit origin");
  const int64_t divisor = owner.getElements() / source.getShape()[0];
  for (Value base = field.getOwner();;) {
    if (base.getDefiningOp<SubviewOp>())
      return emitOpError("field view does not support a subview of the encoded record owner");
    auto slice = base.getDefiningOp<SliceOp>();
    if (!slice) break;
    base = slice.getBase();
  }
  Value expected;
  Value cursor = field.getOwner();
  while (auto slice = cursor.getDefiningOp<SliceOp>()) {
    size_t index = 0;
    for (auto [axis, selector] : llvm::zip(slice.getBase().getType().getAxisIds().asArrayRef(), slice.getSelectors())) {
      auto kind = cast<StringAttr>(selector).getValue();
      if (kind == "all") continue;
      Value selected = slice.getIndices()[index++];
      if (axis == source.getAxisIds()[0] && kind == "domain") {
        auto point = selected.getDefiningOp<PhysicalPointOp>();
        if (!point) return emitOpError("record origin requires a materialized physical point");
        expected = point.getBase();
        break;
      }
    }
    if (expected) break;
    cursor = slice.getBase();
  }
  if (!expected) {
    if (!matchPattern(getOrigin(), m_Zero()))
      return emitOpError("record-local field view must have zero origin");
  } else if (divisor == 1) {
    if (getOrigin() != expected) return emitOpError("field origin must follow its selected record");
  } else {
    auto division = getOrigin().getDefiningOp<arith::DivUIOp>();
    llvm::APInt constant;
    if (!division || division.getLhs() != expected ||
        !matchPattern(division.getRhs(), m_ConstantInt(&constant)) || constant != divisor)
      return emitOpError("field origin must scale the record coordinate by its declared grouping");
  }
  return success();
}
