#include "Weft/Dialect/RISCV/IR/LocalTransfer.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include "Verification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
using namespace weft;
using namespace weft::riscv;

std::optional<int64_t> weft::riscv::localTransferBytes(Value source, Value destination) {
  auto from = dyn_cast<MemDescType>(source.getType()), to = dyn_cast<MemDescType>(destination.getType());
  if (!from || !to || from.getEncoding() != to.getEncoding() ||
      from.getShape() != to.getShape() || from.getAxisIds() != to.getAxisIds() ||
      from.getStrides() != to.getStrides() || from.getStorageBits() != to.getStorageBits() ||
      from.getAddressClass() != "local" || to.getAddressClass() != "local" ||
      (from.getAccess() != "read" && from.getAccess() != "readwrite") ||
      (to.getAccess() != "write" && to.getAccess() != "readwrite") ||
      from.getElements() != 1 || to.getElements() != 1 ||
      from.getInterleaveRows() || to.getInterleaveRows() ||
      from.getStorageBits() <= 0 || from.getStorageBits() % 8 ||
      cast<kernel::EncodingType>(from.getEncoding()).getKind() != "dense" ||
      !haveDisjointLocalStorage(source, destination)) return std::nullopt;
  int64_t elements = 1;
  for (size_t axis = from.getShape().size(); axis-- > 0;) {
    int64_t extent = from.getShape()[axis];
    if (extent <= 0 || from.getStrides()[axis] != elements ||
        from.getOrigins()[axis] || to.getOrigins()[axis] ||
        elements > std::numeric_limits<int64_t>::max() / extent) return std::nullopt;
    elements *= extent;
  }
  int64_t bytes = from.getStorageBits() / 8;
  if (elements > std::numeric_limits<int64_t>::max() / bytes) return std::nullopt;
  return elements * bytes;
}

namespace {
bool boundedIssue(LocalCopyOp op, int64_t bytes, int64_t capacity) {
  if (auto offset = getConstantIntValue(op.getByteOffset())) {
    auto active = getConstantIntValue(op.getActiveBytes());
    return active && *offset >= 0 && *offset <= bytes && *active > 0 &&
        *active <= capacity && *active <= bytes - *offset;
  }
  auto index = dyn_cast<BlockArgument>(op.getByteOffset());
  auto loop = index ? dyn_cast<scf::ForOp>(index.getOwner()->getParentOp()) : scf::ForOp();
  if (!loop || index != loop.getInductionVar()) return false;
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step != capacity ||
      *lower < 0 || *upper < *lower || *upper > bytes) return false;
  if (auto active = getConstantIntValue(op.getActiveBytes()))
    return *active == capacity && (*upper - *lower) % capacity == 0;
  auto tail = op.getActiveBytes().getDefiningOp<arith::SubIOp>();
  auto limit = tail ? getConstantIntValue(tail.getLhs()) : std::nullopt;
  return limit && *limit == *upper && tail.getRhs() == index && *upper - *lower <= capacity;
}
}

LogicalResult LocalCopyOp::verify() {
  auto bytes = localTransferBytes(getSource(), getDestination());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto parameters = getLeaf().getParameters().asArrayRef();
  if (!bytes || !kernel || !kernel.getTarget().getHasRVV() || parameters.size() != 2)
    return emitOpError("local copy requires two distinct complete contiguous private domains");
  auto target = kernel.getTarget();
  int64_t lmul = parameters[0], capacity = parameters[1];
  if ((lmul != 8 && lmul != 16 && lmul != 32) || capacity != target.getVlenBits() * lmul / 64 ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul) ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(), 8) ||
      !detail::exactLeaf(getLeaf(), "transfer", "copy", "rvv.local-copy", "none", "exact") ||
      getLeaf().getOperandGroups() || getLeaf().getResultGroups() ||
      getLeaf().getTemporaryGroups() != lmul / 8 || getLeaf().getFragmentGroups() ||
      getLeaf().getLocalBytes() || !boundedIssue(*this, *bytes, capacity))
    return emitOpError("local copy requires one exact bounded byte issue and its selected RVV resources");
  return detail::verifyLeafOperation(*this);
}

namespace {
bool sameExtent(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto a = getConstantIntValue(lhs), b = getConstantIntValue(rhs);
  if (a && b) return a == b;
  auto sa = lhs.getDefiningOp<SymbolOp>(), sb = rhs.getDefiningOp<SymbolOp>();
  return sa && sb && sa.getName() == sb.getName() &&
      sa->getParentOfType<KernelOp>() == sb->getParentOfType<KernelOp>();
}

bool checkedBytes(Value value, ValueRange extents, int64_t elementBytes) {
  auto size = value.getDefiningOp<LocalSizeOp>();
  if (!size || size.getElementBytes() != elementBytes || size.getExtents().size() != extents.size()) return false;
  for (auto [a, b] : llvm::zip(size.getExtents(), extents)) if (!sameExtent(a, b)) return false;
  return true;
}

bool coversBytes(Value capacity, ValueRange extents, int64_t elementBytes,
                 std::optional<int64_t> constantBytes) {
  SmallVector<Value> pending{capacity};
  llvm::DenseSet<Value> visited;
  for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
    if (cursor >= 64) return false;
    Value value = pending[cursor];
    if (!visited.insert(value).second) continue;
    if (checkedBytes(value, extents, elementBytes)) return true;
    if (auto size = getConstantIntValue(value); size && constantBytes && *size >= *constantBytes) return true;
    if (auto maximum = value.getDefiningOp<arith::MaxUIOp>()) {
      pending.push_back(maximum.getLhs());
      pending.push_back(maximum.getRhs());
    }
  }
  return false;
}
}

LogicalResult weft::riscv::verifyLocalBindingStorage(LocalBindOp binding) {
  auto type = binding.getResult().getType();
  auto storage = binding.getStorage().getType();
  auto kernel = binding->getParentOfType<KernelOp>();
  std::optional<int64_t> count = 1;
  for (auto [shape, extent] : llvm::zip(type.getShape().asArrayRef(), binding.getExtents())) {
    auto fixed = getConstantIntValue(extent);
    if (fixed && *fixed < 0) return binding.emitOpError("local extent cannot be negative");
    if (shape >= 0) {
      if (!fixed || *fixed != shape) return binding.emitOpError("local extent SSA must preserve its static logical dimension");
    } else if (storage.getSizeBytes() == -1) {
      if (!kernel || shape == std::numeric_limits<int64_t>::min())
        return binding.emitOpError("dynamic local extent has no enclosing shape identity");
      if (-shape > static_cast<int64_t>(kernel.getShapeSymbols().size())) {
        if (weft::localShapeExtent<LocalShapeOp>(binding, -shape) != extent)
          return binding.emitOpError("local extent lost its dominating shape definition");
      } else {
        auto symbol = extent.getDefiningOp<SymbolOp>();
        if (!symbol || symbol.getName() != cast<StringAttr>(kernel.getShapeSymbols()[-shape - 1]).getValue())
          return binding.emitOpError("local extent must name its declared shape symbol");
      }
    }
    if (count && fixed) {
      if (*fixed && *count > std::numeric_limits<int64_t>::max() / *fixed)
        return binding.emitOpError("local element product exceeds the index domain");
      *count *= *fixed;
    } else count.reset();
  }
  if (storage.getSizeBytes() != -1) return success();
  unsigned bits = detail::elementBitWidth(type.getElementType());
  if (!bits || (bits > 8 && bits % 8))
    return binding.emitOpError("local binding has no byte-addressable element representation");
  const int64_t elementBytes = std::max(8u, bits) / 8;
  std::optional<int64_t> bytes;
  if (count) {
    if (*count > std::numeric_limits<int64_t>::max() / elementBytes)
      return binding.emitOpError("local byte product exceeds the index domain");
    bytes = *count * elementBytes;
    if (getConstantIntValue(binding.getElements()) != count)
      return binding.emitOpError("local element count disagrees with its complete extents");
  } else {
    Value size = binding.getElements();
    bool checked = elementBytes == 1 && checkedBytes(size, binding.getExtents(), 1);
    if (auto quotient = size.getDefiningOp<arith::DivUIOp>())
      checked |= getConstantIntValue(quotient.getRhs()) == elementBytes &&
          checkedBytes(quotient.getLhs(), binding.getExtents(), elementBytes);
    if (!checked) return binding.emitOpError("dynamic element count must come from the checked extent byte product");
  }
  Value capacity;
  if (auto allocation = binding.getStorage().getDefiningOp<LocalAllocOp>()) capacity = allocation.getSizeBytes();
  else if (auto reused = binding.getStorage().getDefiningOp<LocalReuseOp>()) capacity = reused.getSizeBytes();
  if (!capacity || !coversBytes(capacity, binding.getExtents(), elementBytes, bytes))
    return binding.emitOpError("local allocation must cover the checked byte product of its complete extents");
  return success();
}
