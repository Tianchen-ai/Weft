#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/MemoryEffects.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Verification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace weft;
using namespace weft::riscv;
using namespace weft::riscv::detail;

namespace {
std::optional<int64_t> nonnegativeIndexUpperBound(mlir::Value value, unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value() >= 0 ? std::optional<int64_t>(constant.value()) : std::nullopt;
  if (auto remainder = value.getDefiningOp<mlir::arith::RemUIOp>()) {
    auto divisor = remainder.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (divisor && divisor.value() > 0)
      return divisor.value() - 1;
  }
  if (auto division = value.getDefiningOp<mlir::arith::DivUIOp>()) {
    auto divisor = division.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto bound = nonnegativeIndexUpperBound(division.getLhs(), depth + 1);
    if (divisor && divisor.value() > 0 && bound)
      return *bound / divisor.value();
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    auto lhs = nonnegativeIndexUpperBound(multiply.getLhs(), depth + 1);
    auto rhs = nonnegativeIndexUpperBound(multiply.getRhs(), depth + 1);
    if (lhs && rhs && (*rhs == 0 || *lhs <= std::numeric_limits<int64_t>::max() / *rhs))
      return *lhs * *rhs;
  }
  if (auto subtract = value.getDefiningOp<mlir::arith::SubIOp>()) {
    auto lhs = subtract.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto bound = nonnegativeIndexUpperBound(subtract.getRhs(), depth + 1);
    if (lhs && bound && lhs.value() >= *bound)
      return lhs.value();
  }
  return std::nullopt;
}

llvm::StringRef signedness(RISCVFragmentSignedness value) {
  return value == RISCVFragmentSignedness::Signed ? "signed" : "unsigned";
}

FragmentPackingAttr packingAttr(mlir::MLIRContext *context,
                                const RISCVFragmentPacking &packing) {
  return FragmentPackingAttr::get(
      context, packing.schema,
      mlir::DenseI64ArrayAttr::get(context, packing.axisOrder),
      packing.rowsPerTile, packing.columnsPerTile, packing.tileOrder,
      packing.elementOrder, packing.storageBits, packing.alignment);
}

FragmentCapabilityAttr fragmentCapability(mlir::Operation *operation,
                                          llvm::StringRef instruction) {
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel)
    return {};
  for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
    auto capability = mlir::cast<FragmentCapabilityAttr>(attribute);
    if (capability.getInstruction() == instruction)
      return capability;
  }
  return {};
}

} // namespace

mlir::FailureOr<llvm::SmallVector<llvm::StringRef>>
weft::riscv::usedInstructionExtensions(KernelOp kernel) {
  llvm::SmallVector<llvm::StringRef> extensions;
  auto walk = kernel.walk([&](IMEFragmentMMAOp mma) -> mlir::WalkResult {
    if (mlir::failed(mma.verify()))
      return mlir::WalkResult::interrupt();
    const auto *contract = findRISCVFragmentCapability(mma.getLeaf().getInstruction());
    extensions.push_back(contract->extension);
    return mlir::WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return mlir::failure();
  llvm::sort(extensions);
  extensions.erase(std::unique(extensions.begin(), extensions.end()), extensions.end());
  return extensions;
}

std::optional<AccessAttr> weft::riscv::fragmentFieldReadAccess(
    mlir::Value input, FragmentType result) {
  llvm::SmallVector<mlir::Value> indices;
  auto field = fieldReadProjection(input, &indices);
  auto source = mlir::dyn_cast<ValueType>(input.getType());
  auto owner = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  auto slice = owner ? owner.getRegion().getDefiningOp<SliceOp>() : SliceOp();
  auto view = slice ? slice.getBase().getDefiningOp<MemoryViewOp>() : MemoryViewOp();
  auto capability = field ? fragmentCapability(field, result.getFamily()) : FragmentCapabilityAttr();
  auto fieldType = field ? mlir::dyn_cast<ValueType>(field.getResult().getType()) : ValueType();
  if (!source || !owner || owner.getSnapshotStorage() || !slice || !view || indices.size() > 1 ||
      !capability ||
      !mlir::isa<mlir::BlockArgument>(view.getBase()) || !fieldType ||
      fieldType.getShape().size() != 2 || source.getShape().size() != 2 ||
      (result.getRole() != "lhs" && result.getRole() != "rhs") ||
      result.getPacking() != (result.getRole() == "lhs" ? capability.getLhsPacking()
                                                       : capability.getRhsPacking()) ||
      fragmentPartResourceGroups(result) != 1)
    return std::nullopt;
  auto access = field.getAccess();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
  const auto packing = result.getPacking();
  const int64_t rows = result.getShape()[packing.getAxisOrder()[0]];
  const int64_t columns = packing.getColumnsPerTile();
  const int64_t cohort = owner.getRegion().getType().getInterleaveRows();
  const int64_t rowAxis = result.getAxisIds()[packing.getAxisOrder()[0]];
  const int64_t innerAxis = result.getAxisIds()[packing.getAxisOrder()[1]];
  if (!integer || integer.isSignless() || integer.getWidth() > 8 ||
      source.getAxisIds()[0] != rowAxis || source.getAxisIds()[1] != innerAxis ||
      fieldType.getAxisIds() != source.getAxisIds() ||
      fieldType.getShape()[1] != owner.getRegion().getType().getElements() ||
      source.getShape()[1] != columns ||
      cohort < rows || cohort % rows || access.getBitOffset() % 8 ||
      access.getFieldChunkBytes() != columns || packing.getElementOrder() != "row_major" ||
      (access.getMapping() != "natural" && access.getMapping() != "grouped_layered"))
    return std::nullopt;
  if (access.getMapping() == "natural" ? integer.getWidth() != 8
      : (8 % integer.getWidth() || access.getLayerSize() != columns ||
         access.getGroupSize() != columns * (8 / integer.getWidth())))
    return std::nullopt;
  PhysicalPointOp rowPoint, innerPoint;
  for (mlir::Value index : slice.getIndices())
    if (auto point = index.getDefiningOp<PhysicalPointOp>()) {
      const int64_t axis = point.getResult().getType().getDomain().getAxisId();
      if (axis == rowAxis)
        rowPoint = point;
      if (axis == innerAxis)
        innerPoint = point;
    }
  for (mlir::Value index : indices) {
    auto point = index.getDefiningOp<PhysicalPointOp>();
    if (!point || point.getResult().getType().getDomain().getAxisId() != innerAxis)
      return std::nullopt;
    innerPoint = point;
  }
  auto hasPartition = [](PhysicalPointOp point, int64_t extent) {
    auto constant = point ? point.getPartition().getDefiningOp<mlir::arith::ConstantOp>()
                          : mlir::arith::ConstantOp();
    auto value = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()) : mlir::IntegerAttr();
    return value && value.getInt() == extent && knownIndexMultipleOf(point.getBase(), extent);
  };
  if (!hasPartition(rowPoint, rows) || !hasPartition(innerPoint, columns))
    return std::nullopt;
  return AccessAttr::get(input.getContext(), "unit", access.getMapping(), access.getAlignment(),
      0, 0, access.getGroupSize(), access.getLayerSize(), access.getJoinFields(),
      access.getJoinLowBits(), access.getJoinRole(), access.getBitOffset(), access.getStorageBits(),
      access.getOrder(), access.getFieldChunkBytes());
}

std::optional<AccessAttr> weft::riscv::fragmentWindowReadAccess(
    mlir::Value input, mlir::Value logicalBase, FragmentType result) {
  auto field = input.getDefiningOp<FieldOp>();
  auto owner = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  auto type = mlir::dyn_cast<ValueType>(input.getType());
  auto capability = field ? fragmentCapability(field, result.getFamily()) : FragmentCapabilityAttr();
  if (!field || !owner || owner.getSnapshotStorage() || !type || !capability ||
      type.getShape().size() != 2 || owner.getRegion().getType().getShape().size() != 2 ||
      owner.getRegion().getType().getInterleaveRows() != 0 ||
      owner.getRegion().getType().getStrides()[1] != 1 ||
      !logicalBase.getType().isIndex() || result.getLayout().getValidity() != "full" ||
      type.getLayout().getValidity() != "full" || type.getLayout().getReplicaGroup() != 1 ||
      (result.getRole() != "lhs" && result.getRole() != "rhs") ||
      result.getPacking() != (result.getRole() == "lhs" ? capability.getLhsPacking()
                                                       : capability.getRhsPacking()) ||
      fragmentPartResourceGroups(result) != 1)
    return std::nullopt;
  auto access = field.getAccess();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  auto packing = result.getPacking();
  auto target = field->getParentOfType<KernelOp>().getTarget();
  const int64_t rowPosition = packing.getAxisOrder()[0];
  const int64_t innerPosition = packing.getAxisOrder()[1];
  const int64_t rows = result.getShape()[rowPosition];
  const int64_t columns = packing.getColumnsPerTile();
  const int64_t extent = type.getShape()[1];
  const int64_t group = access.getGroupSize(), layer = access.getLayerSize();
  auto upper = nonnegativeIndexUpperBound(logicalBase);
  if (!integer || !integer.isUnsigned() ||
      (integer.getWidth() != 1 && integer.getWidth() != 2 && integer.getWidth() != 4) || rows <= 1 ||
      rows != fragmentPartCount(result) * packing.getRowsPerTile() ||
      result.getShape()[innerPosition] != columns ||
      type.getAxisIds()[0] != result.getAxisIds()[rowPosition] ||
      type.getAxisIds()[1] != result.getAxisIds()[innerPosition] ||
      type.getShape()[0] != rows || owner.getRegion().getType().getShape() != type.getShape() ||
      extent <= 0 || extent != owner.getRegion().getType().getElements() ||
      access.getMapping() != "grouped_layered" || access.getOrder() != "lo_first" ||
      access.getFieldChunkBytes() != 1 || access.getBitOffset() % 8 ||
      group <= 0 || layer <= 0 || columns <= 0 || group % layer || extent % group ||
      layer % columns || integer.getWidth() * (group / layer) > 8 ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), columns * 64 / target.getVlenBits()) ||
      packing.getElementOrder() != "row_major" || packing.getTileOrder() != "row_major" ||
      !upper || *upper >= extent || !knownIndexMultipleOf(logicalBase, columns))
    return std::nullopt;
  const int64_t bytes = extent / group * layer;
  const int64_t offset = access.getBitOffset() / 8;
  const int64_t recordBytes = owner.getRegion().getType().getStorageBits() / 8;
  if (bytes > access.getStorageBits() / 8 || offset > recordBytes || bytes > recordBytes - offset)
    return std::nullopt;
  return access;
}

std::optional<int64_t> weft::riscv::fragmentPointwiseTemporaryGroups(
    FragmentType lhs, mlir::Type rhs) {
  if (lhs.getRole() == "accumulator" || !integerSignednessMatches(lhs.getElementType(), "signed") ||
      elementBitWidth(lhs.getElementType()) != 8 || fragmentPartResourceGroups(lhs) != 1)
    return std::nullopt;
  if (rhs == lhs.getElementType() || rhs == lhs)
    return 0;
  auto value = mlir::dyn_cast<ValueType>(rhs);
  const int64_t rowAxis = lhs.getAxisIds()[lhs.getPacking().getAxisOrder()[0]];
  const int64_t rows = lhs.getShape()[lhs.getPacking().getAxisOrder()[0]];
  if (!value || value.getElementType() != lhs.getElementType() || value.getAxisIds().size() != 1 ||
      value.getAxisIds()[0] != rowAxis || value.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  auto layout = value.getLayout();
  const int64_t lmul = layout.getLmulEighths();
  if (layout.getLaneFactors()[0] != rows || layout.getSew() != 8 ||
      layout.getVl() != rows || layout.getValidity() == "mask" ||
      (layout.getValidity() != "full" && layout.getValidity() != lhs.getLayout().getValidity()) ||
      lmul <= 0 || lmul > 8 || (lmul & (lmul - 1)) ||
      layout.getRegisterGroups() != 1 ||
      layout.getTimeFactors()[0] != 1 || layout.getReplicaFactors()[0] != 1 ||
      layout.getFragmentFactors()[0] != 1 || layout.getLocalFactors()[0] != 1)
    return std::nullopt;
  return 2;
}

mlir::LogicalResult IMEFragmentRowBroadcastOp::verify() {
  auto fragment = getResult().getType();
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto temporary = fragmentPointwiseTemporaryGroups(fragment, getInput().getType());
  const int64_t parts = fragmentPartCount(fragment);
  const bool byteRepeat = getLeaf().getInstruction() == "ime.rows.byte-repeat";
  if (!kernel || !fragmentCapability(*this, fragment.getFamily()) ||
      !supportsRVVLayout(kernel.getTarget(), getInput().getType().getLayout()) ||
      !temporary || *temporary != 2 || fragment.getPacking().getColumnsPerTile() != 8 ||
      !exactLeaf(getLeaf(), "ime", "fragment-broadcast",
                 byteRepeat ? "ime.rows.byte-repeat" : "ime.rows.local", "none", "exact") ||
      getLeaf().getTemporaryGroups() != (byteRepeat ? 2 * parts : parts + 1) ||
      getLeaf().getFragmentGroups() || getLeaf().getLocalBytes())
    return emitOpError("fragment row broadcast requires a complete signed-byte row projection and local index resources");
  if (byteRepeat) {
    auto target = kernel.getTarget();
    auto layout = getInput().getType().getLayout();
    auto packing = fragment.getPacking();
    // Output byte (row, column) is input[row]: a tail prefix p maps only
    // to its own 8*p bytes, with no cross-row reduction or permutation.
    if (packing.getElementOrder() != "row_major" || packing.getTileOrder() != "row_major" ||
        fragment.getLayout().getReplicaFactors()[packing.getAxisOrder()[1]] != 1 ||
        layout.getLaneFactors()[0] != packing.getRowsPerTile() * parts ||
        packing.getRowsPerTile() * 64 != target.getVlenBits() ||
        layout.getLmulEighths() != parts || layout.getVl() * 64 != target.getVlenBits() * parts ||
        !llvm::is_contained(target.getSupportedSEW().asArrayRef(), 64) ||
        !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 8 * parts))
      return emitOpError("byte-repeat broadcast requires eight byte copies in each full u64 lane");
  }
  return mlir::success();
}

bool weft::riscv::supportsFragmentWordWindows(mlir::Value field, FragmentType result) {
  auto kernel = field.getParentRegion()->getParentOfType<KernelOp>();
  if (!kernel) return false;
  auto packing = result.getPacking();
  auto target = kernel.getTarget();
  return packing.getColumnsPerTile() == 8 && packing.getRowsPerTile() * 64 == target.getVlenBits() &&
      llvm::is_contained(target.getSupportedSEW().asArrayRef(), 64) &&
      llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 8);
}

namespace {
bool denseRowWindowSource(mlir::Value source, int64_t rowAxis, int64_t innerAxis,
                          int64_t rows, int64_t columns, int64_t alignment) {
  auto memory = mlir::dyn_cast<MemDescType>(source.getType());
  auto encoding = memory ? mlir::dyn_cast<kernel::EncodingType>(memory.getEncoding())
                         : kernel::EncodingType();
  if (!memory || !encoding || encoding.getKind() != "dense" ||
      encoding.getFamily() != "i8" || memory.getStorageBits() != 8 ||
      memory.getElements() != 1 || memory.getInterleaveRows() ||
      (memory.getAccess() != "read" && memory.getAccess() != "readwrite") ||
      memory.getShape().size() != 2 || memory.getAxisIds().size() != 2 ||
      memory.getStrides().size() != 2 || rowAxis < 0 || rowAxis >= 2 ||
      innerAxis < 0 || innerAxis >= 2 || rowAxis == innerAxis ||
      rows < 4 || rows > 16 || rows % 4 || columns != 8 || alignment != 8 ||
      memory.getStrides()[innerAxis] != 1 || !isAddressableDenseDescriptor(source))
    return false;
  if (!denseDescriptorExtent(source, rowAxis) ||
      !denseDescriptorExtent(source, innerAxis))
    return false;
  return true;
}
} // namespace

std::optional<AccessAttr> weft::riscv::fragmentDenseReadAccess(
    mlir::Value source, FragmentType result) {
  auto memory = mlir::dyn_cast<MemDescType>(source.getType());
  auto *owner = source.getParentRegion()->getParentOp();
  auto kernel = mlir::dyn_cast<KernelOp>(owner);
  if (!kernel) kernel = owner->getParentOfType<KernelOp>();
  auto packing = result.getPacking();
  FragmentCapabilityAttr capability;
  if (kernel)
    for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
      auto candidate = mlir::cast<FragmentCapabilityAttr>(attribute);
      if (candidate.getInstruction() == result.getFamily()) capability = candidate;
    }
  if (!memory || !kernel || !capability || result.getAxisIds().size() != 2 ||
      result.getShape().size() != 2 || packing.getAxisOrder().size() != 2 ||
      (result.getRole() != "lhs" && result.getRole() != "rhs") ||
      packing != (result.getRole() == "lhs" ? capability.getLhsPacking()
                                           : capability.getRhsPacking()) ||
      packing.getRowsPerTile() != 4 || packing.getColumnsPerTile() != 8 ||
      packing.getStorageBits() != 8 || packing.getElementOrder() != "row_major" ||
      packing.getTileOrder() != "row_major" || fragmentPartResourceGroups(result) != 1 ||
      !integerSignednessMatches(result.getElementType(), "signed") ||
      elementBitWidth(result.getElementType()) != 8)
    return std::nullopt;
  auto target = kernel.getTarget();
  const int64_t row = packing.getAxisOrder()[0], inner = packing.getAxisOrder()[1];
  const int64_t parts = fragmentPartCount(result), rows = result.getShape()[row];
  if (parts < 1 || parts > 4 || rows != parts * packing.getRowsPerTile() ||
      result.getShape()[inner] != packing.getColumnsPerTile() ||
      result.getLayout().getReplicaFactors()[inner] != 1 ||
      !llvm::all_of(result.getLayout().getTimeFactors().asArrayRef(), [](int64_t factor) { return factor == 1; }) ||
      target.getVlenBits() != packing.getRowsPerTile() * 64 ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(), 64) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 8))
    return std::nullopt;
  auto axes = memory.getAxisIds().asArrayRef();
  auto rowPosition = llvm::find(axes, result.getAxisIds()[row]);
  auto innerPosition = llvm::find(axes, result.getAxisIds()[inner]);
  if (rowPosition == axes.end() || innerPosition == axes.end() ||
      !denseRowWindowSource(source, rowPosition - axes.begin(), innerPosition - axes.begin(),
                            rows, packing.getColumnsPerTile(), 8))
    return std::nullopt;
  return AccessAttr::get(source.getContext(), "strided", "dense", 8,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1);
}

mlir::LogicalResult DenseRowWindowGuardOp::verify() {
  if (!denseRowWindowSource(getSource(), getRowAxis(), getInnerAxis(),
                            getRows(), getColumns(), getAlignment()))
    return emitOpError("dense row guard requires bounded signed-byte row windows, explicit extents and unit inner stride");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentDenseLoadOp::verify() {
  auto fragment = getResult().getType();
  auto expected = fragmentDenseReadAccess(getSource(), fragment);
  if (!expected || getAccess() != *expected ||
      !exactLeaf(getLeaf(), "ime", "fragment-dense-load", "ime.dense-row-words", "none", "agnostic") ||
      getLeaf().getTemporaryGroups() || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != fragment.getResourceGroups() ||
      !getLeaf().getParameters().empty())
    return emitOpError("dense fragment load requires its exact capability packing, word transfer and resource contract");
  auto packing = fragment.getPacking();
  auto axes = getSource().getType().getAxisIds().asArrayRef();
  const int64_t row = packing.getAxisOrder()[0], inner = packing.getAxisOrder()[1];
  const int64_t rowPosition = llvm::find(axes, fragment.getAxisIds()[row]) - axes.begin();
  const int64_t innerPosition = llvm::find(axes, fragment.getAxisIds()[inner]) - axes.begin();
  for (mlir::Operation *child = getOperation(), *parent = child->getParentOp();
       parent; child = parent, parent = parent->getParentOp()) {
    auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent);
    if (!branch || child->getParentRegion() != &branch.getThenRegion()) continue;
    auto guard = branch.getCondition().getDefiningOp<DenseRowWindowGuardOp>();
    if (guard && guard.getSource() == getSource() && guard.getRowAxis() == rowPosition &&
        guard.getInnerAxis() == innerPosition && guard.getRows() == fragment.getShape()[row] &&
        guard.getColumns() == packing.getColumnsPerTile() && guard.getAlignment() == 8)
      return mlir::success();
  }
  return emitOpError("dense fragment load requires the matching row-window and alignment guard true branch");
}

mlir::LogicalResult IMEFragmentWindowLoadOp::verify() {
  auto expected = fragmentWindowReadAccess(getField(), getLogicalBase(), getResult().getType());
  bool guardedWords = supportsFragmentWordWindows(getField(), getResult().getType()) &&
      exactLeaf(getLeaf(), "ime", "fragment-window-load", "ime.field-byte-windows.guarded-vlse64", "none", "agnostic");
  if (!expected || getAccess() != *expected ||
      (!guardedWords && !exactLeaf(getLeaf(), "ime", "fragment-window-load", "ime.field-byte-windows", "none", "agnostic")) ||
      getLeaf().getTemporaryGroups() != getResult().getType().getPacking().getRowsPerTile() || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != getResult().getType().getResourceGroups())
    return emitOpError("fragment byte windows require complete record rows, aligned in-layer byte spans, and exact fragment packing");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentStorageLoadOp::verify() {
  auto expected = fragmentFieldReadAccess(getInput(), getResult().getType());
  if (!expected || getAccess() != *expected ||
      !exactLeaf(getLeaf(), "ime", "fragment-storage-load", "ime.field-raw-load", "none", "agnostic") ||
      getLeaf().getTemporaryGroups() != 1 || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != getResult().getType().getResourceGroups())
    return emitOpError("fragment field load requires one proven contiguous row/chunk tile");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentDecodeOp::verify() {
  auto fragment = getInput().getType();
  const int64_t bits = getElementBits();
  auto maximum = nonnegativeIndexUpperBound(getShift());
  if (getResult().getType() != fragment || fragment.getRole() == "accumulator" ||
      fragmentPartResourceGroups(fragment) != 1 ||
      (bits != 1 && bits != 2 && bits != 4) || !maximum || *maximum > 8 - bits ||
      !knownIndexMultipleOf(getShift(), bits) ||
      !exactLeaf(getLeaf(), "ime", "fragment-decode", "ime.decode.bits", "none", "exact") ||
      getLeaf().getTemporaryGroups() != 1 || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != fragment.getResourceGroups())
    return emitOpError("fragment bit projection requires a bounded byte-local shift and closed replicas");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentPointwiseOp::verify() {
  auto temporaries = fragmentPointwiseTemporaryGroups(getLhs().getType(), getRhs().getType());
  auto rhs = mlir::dyn_cast<ValueType>(getRhs().getType());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  const bool reuse = getLhsPolicy() == "reuse";
  if (!temporaries || !fragmentCapability(*this, getResult().getType().getFamily()) ||
      (rhs && (!kernel || !supportsRVVLayout(kernel.getTarget(), rhs.getLayout()))) ||
      getResult().getType() != getLhs().getType() ||
      (getKind() != "add" && getKind() != "sub") ||
      (getLhsPolicy() != "copy" && !reuse) || (reuse && !canReuseFragmentPointwise(*this)) ||
      !exactLeaf(getLeaf(), "ime", "fragment-pointwise", ("ime.fragment." + getKind()).str(), "none", "exact") ||
      getLeaf().getTemporaryGroups() != *temporaries || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != (reuse ? 0 : getResult().getType().getResourceGroups()))
    return emitOpError("fragment pointwise requires its exact integer domain and scalar/row broadcast");
  return mlir::success();
}

FragmentCapabilityAttr weft::riscv::fragmentCapabilityAttr(
    mlir::MLIRContext *context, const RISCVFragmentCapability &capability) {
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Attribute> clobbers;
  for (const std::string &clobber : capability.clobbers)
    clobbers.push_back(builder.getStringAttr(clobber));
  return FragmentCapabilityAttr::get(
      context, capability.name, signedness(capability.lhsSignedness),
      signedness(capability.rhsSignedness), capability.lhsElementBits,
      capability.rhsElementBits, capability.accumulatorElementBits,
      capability.mFactor, capability.nFactor, capability.kFactor, capability.activeMFactor,
      capability.lhsResourceGroups, capability.rhsResourceGroups,
      capability.accumulatorResourceGroups,
      packingAttr(context, capability.lhsPacking),
      packingAttr(context, capability.rhsPacking),
      packingAttr(context, capability.accumulatorPacking),
      capability.mmaGroups, capability.mmaChunks, builder.getArrayAttr(clobbers),
      capability.volatileAsm, capability.memoryClobber);
}

bool weft::riscv::fragmentAccumulatorTypeMatches(
    FragmentCapabilityAttr capability, mlir::Type element) {
  const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
  return contract && elementBitWidth(element) == capability.getAccumulatorBits() &&
         integerSignednessMatches(element, signedness(contract->accumulatorSignedness));
}

int64_t weft::riscv::fragmentPartCount(FragmentType fragment) {
  auto replicas = fragment.getLayout().getReplicaFactors();
  return replicas[0] * replicas[1];
}

int64_t weft::riscv::fragmentPartResourceGroups(FragmentType fragment) {
  return fragment.getResourceGroups() / fragmentPartCount(fragment);
}

bool weft::riscv::canReuseFragmentAccumulator(IMEFragmentMMAOp operation) {
  mlir::Value accumulator = operation.getAccumulator();
  if (!accumulator.hasOneUse())
    return false;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(accumulator))
    return argument.getOwner() == operation->getBlock();
  auto *definition = accumulator.getDefiningOp();
  return definition && definition->getBlock() == operation->getBlock();
}

bool weft::riscv::canReuseFragmentPointwise(IMEFragmentPointwiseOp operation) {
  auto lhs = operation.getLhs();
  if (!lhs.hasOneUse()) return false;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(lhs))
    return argument.getOwner() == operation->getBlock();
  auto *definition = lhs.getDefiningOp();
  return definition && definition->getBlock() == operation->getBlock();
}

int64_t weft::riscv::fragmentMMAAdditionalGroups(FragmentCapabilityAttr capability, int64_t atoms,
                                               bool reuseAccumulator) {
  llvm::SmallVector<unsigned> registers;
  for (mlir::Attribute attribute : capability.getClobbers()) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(attribute).getValue();
    unsigned number = 0;
    if (name.consume_front("v") && !name.getAsInteger(10, number) &&
        number < 32 && !llvm::is_contained(registers, number))
      registers.push_back(number);
  }
  // A tied accumulator may need a copy when its input SSA value remains live.
  // Reserve that result group as well as any fixed registers beyond operands.
  return (reuseAccumulator ? 0 : capability.getAccumulatorResourceGroups() * atoms) +
         std::max<int64_t>(0, static_cast<int64_t>(registers.size()) -
                                capability.getLhsResourceGroups() -
                                capability.getRhsResourceGroups() -
                                capability.getAccumulatorResourceGroups());
}

mlir::LogicalResult FragmentPackingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef schema, mlir::DenseI64ArrayAttr axisOrder,
    int64_t rowsPerTile, int64_t columnsPerTile,
    llvm::StringRef tileOrder, llvm::StringRef elementOrder,
    int64_t storageBits, int64_t alignment) {
  if (schema.empty() || rowsPerTile <= 0 || columnsPerTile <= 0 ||
      !axisOrder || axisOrder.size() != 2 || axisOrder[0] == axisOrder[1] ||
      llvm::any_of(axisOrder.asArrayRef(),
                   [](int64_t axis) { return axis < 0 || axis > 1; }) ||
      storageBits <= 0 || storageBits % 8 || alignment <= 0)
    return emitError()
           << "fragment packing requires one rank-two byte-addressable tiled storage mapping";
  auto validOrder = [](llvm::StringRef order) {
    return order == "row_major" || order == "column_major";
  };
  if (!validOrder(tileOrder) || !validOrder(elementOrder))
    return emitError()
           << "fragment packing requires explicit tile and in-tile orders";
  return mlir::success();
}

mlir::LogicalResult FragmentCapabilityAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef instruction, llvm::StringRef lhsSignedness,
    llvm::StringRef rhsSignedness, int64_t lhsBits, int64_t rhsBits,
    int64_t accumulatorBits, int64_t mFactor, int64_t nFactor,
    int64_t kFactor, int64_t activeMFactor, int64_t lhsResourceGroups, int64_t rhsResourceGroups,
    int64_t accumulatorResourceGroups, FragmentPackingAttr lhsPacking,
    FragmentPackingAttr rhsPacking, FragmentPackingAttr accumulatorPacking,
    int64_t mmaGroups, int64_t mmaChunks, mlir::ArrayAttr clobbers,
    bool volatileAsm, bool memoryClobber) {
  const auto *contract = findRISCVFragmentCapability(instruction);
  if (!contract)
    return emitError() << "unsupported fragment instruction " << instruction;
  if (!lhsPacking || !rhsPacking || !accumulatorPacking || !clobbers)
    return emitError() << "fragment capability requires packing and clobbers";
  auto *context = lhsPacking.getContext();
  if (lhsSignedness != signedness(contract->lhsSignedness) ||
      rhsSignedness != signedness(contract->rhsSignedness) ||
      lhsBits != contract->lhsElementBits || rhsBits != contract->rhsElementBits ||
      accumulatorBits != contract->accumulatorElementBits ||
      mFactor != contract->mFactor || nFactor != contract->nFactor ||
      kFactor != contract->kFactor || activeMFactor != contract->activeMFactor ||
      lhsResourceGroups != contract->lhsResourceGroups ||
      rhsResourceGroups != contract->rhsResourceGroups ||
      accumulatorResourceGroups != contract->accumulatorResourceGroups ||
      lhsPacking != packingAttr(context, contract->lhsPacking) ||
      rhsPacking != packingAttr(context, contract->rhsPacking) ||
      accumulatorPacking != packingAttr(context, contract->accumulatorPacking) ||
      mmaGroups != contract->mmaGroups || mmaChunks != contract->mmaChunks ||
      volatileAsm != contract->volatileAsm || memoryClobber != contract->memoryClobber ||
      clobbers.size() != contract->clobbers.size())
    return emitError() << "fragment capability disagrees with its registered ISA contract";
  for (auto [attribute, name] : llvm::zip(clobbers, contract->clobbers)) {
    auto clobber = mlir::dyn_cast<mlir::StringAttr>(attribute);
    if (!clobber || clobber.getValue() != name)
      return emitError() << "fragment clobbers disagree with the registered ISA contract";
  }
  if (instruction.empty() || lhsBits <= 0 || rhsBits <= 0 ||
      accumulatorBits <= 0 || mFactor <= 0 || nFactor <= 0 || kFactor <= 0 ||
      activeMFactor <= 0 || activeMFactor > mFactor ||
      (lhsSignedness != "signed" && lhsSignedness != "unsigned") ||
      (rhsSignedness != "signed" && rhsSignedness != "unsigned") ||
      lhsResourceGroups <= 0 || rhsResourceGroups <= 0 ||
      accumulatorResourceGroups <= 0 || !lhsPacking || !rhsPacking ||
      !accumulatorPacking ||
      mmaGroups <= 0 || mmaChunks <= 0 || !volatileAsm ||
      clobbers.empty() ||
      llvm::any_of(clobbers, [](mlir::Attribute attribute) {
        auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
        return !value || value.getValue().empty();
      }))
    return emitError() << "fragment capability fields must be positive and complete";
  if (lhsPacking.getRowsPerTile() * lhsPacking.getColumnsPerTile() *
              lhsPacking.getStorageBits() / 8 <=
          0 ||
      rhsPacking.getRowsPerTile() * rhsPacking.getColumnsPerTile() *
              rhsPacking.getStorageBits() / 8 <=
          0 ||
      accumulatorPacking.getRowsPerTile() *
              accumulatorPacking.getColumnsPerTile() *
              accumulatorPacking.getStorageBits() / 8 <=
          0)
    return emitError() << "fragment packing tile geometry is incomplete";
  if (mFactor % lhsPacking.getRowsPerTile() ||
      kFactor % lhsPacking.getColumnsPerTile() ||
      nFactor % rhsPacking.getRowsPerTile() ||
      kFactor % rhsPacking.getColumnsPerTile() ||
      mFactor % accumulatorPacking.getRowsPerTile() ||
      nFactor % accumulatorPacking.getColumnsPerTile() ||
      lhsPacking.getStorageBits() < lhsBits ||
      rhsPacking.getStorageBits() < rhsBits ||
      accumulatorPacking.getStorageBits() < accumulatorBits)
    return emitError()
           << "fragment packing geometry does not cover the capability operands";
  return mlir::success();
}

mlir::LogicalResult IMEZeroOp::verify() {
  auto fragment = getResult().getType();
  auto capability = fragmentCapability(*this, fragment.getFamily());
  if (!capability || fragment.getRole() != "accumulator" ||
      fragment.getPacking() != capability.getAccumulatorPacking() ||
      !fragmentAccumulatorTypeMatches(capability, fragment.getElementType()) ||
      (!exactLeaf(getLeaf(), "ime", "fragment-init", "ime.zero", "none", "exact") &&
       !exactLeaf(getLeaf(), "ime", "fragment-init", "ime.zero.local", "none", "exact")) ||
      getLeaf().getLocalBytes() || getLeaf().getTemporaryGroups() ||
      getLeaf().getFragmentGroups())
    return emitOpError("fragment zero requires a closed accumulator representation");
  return mlir::success();
}

void IMEFragmentMMAOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  auto *storage = PrimitiveStorageResource::get();
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(0), storage);
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(1), storage);
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(2), storage);
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getOperation()->getResult(0), storage);
}

mlir::LogicalResult IMEFragmentMMAOp::verify() {
  FragmentType lhs = getLhs().getType();
  FragmentType rhs = getRhs().getType();
  FragmentType result = getResult().getType();
  auto capability = fragmentCapability(*this, getLeaf().getInstruction());
  const auto *contract = findRISCVFragmentCapability(getLeaf().getInstruction());
  const bool reuse = getAccumulatorPolicy() == "reuse";
  if ((getAccumulatorPolicy() != "copy" && !reuse) ||
      (reuse && !canReuseFragmentAccumulator(*this)))
    return emitOpError("fragment accumulator reuse requires its unique same-block final use");
  if (!capability || !contract ||
      !exactLeaf(getLeaf(), "ime", "fragment-mma",
                 capability.getInstruction(), "none", "exact") ||
      lhs.getFamily() != rhs.getFamily() || result.getFamily() != lhs.getFamily() ||
      lhs.getFamily() != capability.getInstruction() ||
      getAccumulator().getType() != result ||
      lhs.getRole() != "lhs" || rhs.getRole() != "rhs" ||
      result.getRole() != "accumulator" ||
      lhs.getPacking() != capability.getLhsPacking() ||
      rhs.getPacking() != capability.getRhsPacking() ||
      result.getPacking() != capability.getAccumulatorPacking() ||
      elementBitWidth(lhs.getElementType()) != capability.getLhsBits() ||
      elementBitWidth(rhs.getElementType()) != capability.getRhsBits() ||
      !integerSignednessMatches(lhs.getElementType(),
                                capability.getLhsSignedness()) ||
      !integerSignednessMatches(rhs.getElementType(),
                                capability.getRhsSignedness()) ||
      !fragmentAccumulatorTypeMatches(capability, result.getElementType()) ||
      getLeaf().getFragmentGroups() != fragmentMMAAdditionalGroups(capability, fragmentPartCount(result), reuse) ||
      getGroups() != capability.getMmaGroups() ||
      getChunks() != capability.getMmaChunks() ||
      getAsmClobbers() != capability.getClobbers() ||
      getVolatileAsm() != capability.getVolatileAsm() ||
      getMemoryClobber() != capability.getMemoryClobber() ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}) ||
      lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      getLeaf().getLocalBytes() != contract->mmaLocalBytes)
    return emitOpError("IME fragment MMA requires an IME leaf contract");
  if (lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      lhs.getShape()[0] != result.getShape()[0] ||
      lhs.getShape()[1] != capability.getKFactor() ||
      rhs.getShape()[0] != capability.getKFactor() ||
      rhs.getShape()[1] != result.getShape()[1] ||
      lhs.getAxisIds()[1] != rhs.getAxisIds()[0] ||
      lhs.getAxisIds()[0] != result.getAxisIds()[0] ||
      rhs.getAxisIds()[1] != result.getAxisIds()[1])
    return emitOpError(
        "IME fragment MMA axes and shapes do not match the selected capability");
  if (fragmentPartResourceGroups(lhs) != capability.getLhsResourceGroups() ||
      fragmentPartResourceGroups(rhs) != capability.getRhsResourceGroups() ||
      fragmentPartResourceGroups(result) != capability.getAccumulatorResourceGroups())
    return emitOpError(
        "IME fragment resources do not match the target capability");
  return mlir::success();
}

mlir::LogicalResult IMEPackOp::verify() {
  FragmentType result = getResult().getType();
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto capability = fragmentCapability(*this, result.getFamily());
  const auto *contract = findRISCVFragmentCapability(result.getFamily());
  if (!input || !capability || !contract ||
      (getRole() != "lhs" && getRole() != "rhs") ||
      !exactLeaf(getLeaf(), "ime", "fragment-pack",
                 getRole() == "lhs" ? "ime.pack.lhs" : "ime.pack.rhs",
                 "none", "exact") ||
      getPacking() != (getRole() == "lhs" ? capability.getLhsPacking()
                                           : capability.getRhsPacking()) ||
      result.getRole() != getRole() || result.getPacking() != getPacking() ||
      fragmentPartResourceGroups(result) !=
          (getRole() == "lhs" ? capability.getLhsResourceGroups()
                              : capability.getRhsResourceGroups()) ||
      getPacking().getStorageBits() != 8 ||
      result.getLayout().getCarrier() != "ime" ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}))
    return emitOpError("IME pack requires one typed fragment operand contract");
  const bool lhs = getRole() == "lhs";
  auto packing = getPacking();
  const int64_t expectedBits = lhs ? capability.getLhsBits()
                                   : capability.getRhsBits();
  const llvm::StringRef expectedSignedness =
      lhs ? capability.getLhsSignedness() : capability.getRhsSignedness();
  if (elementBitWidth(result.getElementType()) != expectedBits ||
      elementBitWidth(input.getElementType()) != expectedBits ||
      !integerSignednessMatches(input.getElementType(), expectedSignedness) ||
      input.getAxisIds().size() != result.getAxisIds().size() ||
      !llvm::all_of(input.getAxisIds().asArrayRef(), [&](int64_t axis) {
        return llvm::is_contained(result.getAxisIds().asArrayRef(), axis);
      }))
    return emitOpError("IME pack input and fragment logical domains must agree");
  auto layout = input.getLayout();
  int64_t laneAxes = 0;
  if (layout.getCarrier() != "rvv" || layout.getSew() != 8 ||
      input.getShape().size() != layout.getTimeFactors().size() ||
      input.getShape().size() != layout.getLaneFactors().size() ||
      input.getShape().size() != layout.getReplicaFactors().size() ||
      input.getShape().size() != layout.getFragmentFactors().size() ||
      input.getShape().size() != layout.getLocalFactors().size())
    return emitOpError("IME pack input requires one complete eight-bit RVV mapping");
  for (auto [axis, time, lane, replica, fragment, local] :
       llvm::zip(input.getAxisIds().asArrayRef(), layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef())) {
    const auto position = llvm::find(result.getAxisIds().asArrayRef(), axis) -
                          result.getAxisIds().asArrayRef().begin();
    const int64_t extent = result.getShape()[position];
    if (extent <= 0 || time <= 0 || lane <= 0 || replica <= 0 ||
        fragment <= 0 || local <= 0 ||
        time * lane * replica * fragment * local != extent)
      return emitOpError("IME pack input time/lane/register factors are incomplete");
    laneAxes += lane > 1;
  }
  if (laneAxes != 1)
    return emitOpError("IME pack input requires exactly one SIMD lane axis");
  if ((lhs && (result.getShape().size() != 2 ||
               result.getShape()[1] != capability.getKFactor())) ||
      (!lhs && (result.getShape().size() != 2 ||
                result.getShape()[0] != capability.getKFactor())))
    return emitOpError("IME pack logical shape disagrees with the target fragment");
  auto axisOrder = packing.getAxisOrder().asArrayRef();
  const int64_t rows = result.getShape()[axisOrder[0]];
  const int64_t columns = result.getShape()[axisOrder[1]];
  const int64_t innerAxis = result.getAxisIds()[axisOrder[1]];
  for (auto [axis, time, lane, replica] : llvm::zip(input.getAxisIds().asArrayRef(),
           layout.getTimeFactors().asArrayRef(), layout.getLaneFactors().asArrayRef(),
           layout.getReplicaFactors().asArrayRef()))
    if (time != 1 || lane != (axis == innerAxis ? columns : 1) ||
        replica != (axis == innerAxis ? 1 : rows))
      return emitOpError("register fragment pack requires its selected contiguous inner-axis slices");
  if (layout.getLmulEighths() > 8 || rows % packing.getRowsPerTile() ||
      columns != packing.getColumnsPerTile() || packing.getElementOrder() != "row_major")
    return emitOpError("register fragment pack requires selected row-major atom tiles");
  if (rows % packing.getRowsPerTile() ||
      columns % packing.getColumnsPerTile() ||
      packing.getStorageBits() < expectedBits)
    return emitOpError(
        "IME operand shape and element width do not fit the selected tiled packing");
  if (getLeaf().getLocalBytes() !=
          (lhs ? contract->lhsPackLocalBytes : contract->rhsPackLocalBytes) ||
      getLeaf().getTemporaryGroups() != contract->packTemporaryGroups)
    return emitOpError(
        "IME pack leaf must account for every primitive-private byte");
  if (getLeaf().getFragmentGroups() != result.getResourceGroups())
    return emitOpError("fragment pack must account for result replicas while input slices remain live");
  if (getAccess().getMapping() == "opaque" || getAccess().getForm() == "opaque")
    return emitOpError("IME operand has no selected physical access form");
  return mlir::success();
}

mlir::LogicalResult IMEUnpackOp::verify() {
  auto capability = fragmentCapability(*this, getInput().getType().getFamily());
  const auto *contract = findRISCVFragmentCapability(getInput().getType().getFamily());
  if (!capability || !contract ||
      !exactLeaf(getLeaf(), "ime", "fragment-unpack", "ime.unpack.rvv",
                 "none", "exact") ||
      getConversion().getKind() != "fragment_to_rvv" ||
      getConversion().getEffect() != "handoff" ||
      getInput().getType().getRole() != "accumulator" ||
      getInput().getType().getPacking() !=
          capability.getAccumulatorPacking() ||
      getResult().getType().getLayout().getCarrier() != "rvv" ||
      getInput().getType().getAxisIds() != getResult().getType().getAxisIds() ||
      getInput().getType().getElementType() !=
          getResult().getType().getElementType() ||
      !fragmentAccumulatorTypeMatches(capability, getResult().getType().getElementType()) ||
      getLeaf().getLocalBytes() != 0 ||
      getLeaf().getTemporaryGroups() != fragmentUnpackTemporaryGroups(
          getInput().getType(), getResult().getType()) ||
      getConversion().getTemporaryGroups() != getLeaf().getTemporaryGroups())
    return emitOpError("IME unpack must preserve a fragment logical domain into RVV");
  auto fragment = getInput().getType();
  auto layout = getResult().getType().getLayout();
  const bool grouped = isGroupedFragmentRVVHandoff(fragment, getResult().getType());
  if (layout.getAxisIds().size() != 2 ||
      (layout.getLaneFactors()[0] != 1 && !grouped && !isDirectFragmentRVVHandoff(fragment, getResult().getType())) ||
      (!grouped && (layout.getReplicaGroup() != 1 ||
                    layout.getLmulEighths() > fragmentPartResourceGroups(fragment) * 8)) ||
      llvm::any_of(layout.getFragmentFactors().asArrayRef(), [](int64_t factor) { return factor != 1; }) ||
      llvm::any_of(layout.getLocalFactors().asArrayRef(), [](int64_t factor) { return factor != 1; }))
    return emitOpError("fragment unpack requires selected contiguous output-row slices");
  for (auto [extent, time, lane, replica, fragmentFactor, local] :
       llvm::zip(fragment.getShape().asArrayRef(),
                 layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef()))
    if (time * lane * replica * fragmentFactor * local != extent)
      return emitOpError(
          "IME unpack result layout does not cover the complete fragment domain");
  return mlir::success();
}
