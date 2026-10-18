#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/Contraction.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Verification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;
using namespace weft::riscv;

std::optional<VectorTileGeometry> VectorTileGeometry::get(ValueType type, int64_t axis) {
  auto layout = type.getLayout();
  auto coordinates = RegisterCoordinates::get(layout);
  if (!coordinates || layout.getCarrier() != "rvv" || layout.getReplicaGroup() != 1 ||
      coordinates->parts() > 32 ||
      coordinates->lanes() != layout.getVl() || layout.getVl() <= 1 ||
      !llvm::is_contained(type.getAxisIds().asArrayRef(), axis)) return std::nullopt;
  VectorTileGeometry result;
  result.axis = axis;
  result.lanes = layout.getVl();
  for (auto [position, id] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
    if (layout.getTimeFactors()[position] < 1 || layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1) return std::nullopt;
    int64_t lanes = id == axis ? result.lanes : 1;
    if (layout.getLaneFactors()[position] != lanes || layout.getReplicaFactors()[position] <= 0 ||
        type.getShape()[position] != lanes * layout.getReplicaFactors()[position] * layout.getTimeFactors()[position])
      return std::nullopt;
    if (id == axis) result.lanePosition = position;
  }
  for (int64_t part = 0; part < coordinates->parts(); ++part) {
    auto point = coordinates->logical(part);
    if (!point) return std::nullopt;
    // The selected carrier must advance only its lane axis. This also rejects
    // an accidental interleaving of two axes that happens to have equal size.
    for (int64_t lane = 0; lane < result.lanes; ++lane) {
      auto expected = *point;
      expected[result.lanePosition] += lane;
      auto actual = coordinates->logical(part, lane);
      if (!actual || *actual != expected) return std::nullopt;
    }
    result.starts.push_back(std::move(*point));
  }
  return result;
}

bool weft::riscv::tileMemoryElementMatches(MemDescType memory, Type element) {
  auto encoding = dyn_cast<weft::kernel::EncodingType>(memory.getEncoding());
  if (!encoding || encoding.getKind() != "dense" || memory.getElements() != 1 || memory.getInterleaveRows()) return false;
  StringRef family = encoding.getFamily();
  if (element.isSignlessInteger(1)) return family == "i1" && memory.getStorageBits() == 8;
  if (isa<Float8E4M3FNType>(element)) return family == "f8E4M3FN";
  if (isa<Float8E5M2Type>(element)) return family == "f8E5M2";
  if (element.isF16()) return family == "f16";
  if (element.isBF16()) return family == "bf16";
  if (element.isF32()) return family == "f32";
  if (element.isF64()) return family == "f64";
  auto integer = dyn_cast<IntegerType>(element);
  return integer && !integer.isSignless() &&
      llvm::is_contained(ArrayRef<unsigned>({1, 2, 4, 6, 8, 16, 32, 64}), integer.getWidth()) &&
      memory.getStorageBits() == std::max(8u, integer.getWidth()) &&
      family == (std::string(integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth()));
}

bool weft::riscv::tileExtentsMatch(Value memory, ValueType value, ValueRange extents) {
  auto type = dyn_cast<MemDescType>(memory.getType());
  if (!type || type.getAxisIds().size() != value.getAxisIds().size() || extents.size() != value.getAxisIds().size()) return false;
  for (auto [position, axis] : llvm::enumerate(value.getAxisIds().asArrayRef())) {
    auto found = llvm::find(type.getAxisIds().asArrayRef(), axis);
    if (found == type.getAxisIds().asArrayRef().end()) return false;
    if (!sameDenseExtent(extents[position], memory, found - type.getAxisIds().asArrayRef().begin())) return false;
  }
  return true;
}

bool weft::riscv::tileIsCompleteMemory(TargetAttr target, Value descriptor, ValueType value) {
  auto memory = cast<MemDescType>(descriptor.getType());
  auto layout = value.getLayout();
  if (memory.getAxisIds().size() != value.getAxisIds().size() ||
      layout.getVl() != target.getVlenBits() * layout.getLmulEighths() / (8 * layout.getSew())) return false;
  for (auto [axis, size] : llvm::zip(value.getAxisIds().asArrayRef(), value.getShape().asArrayRef())) {
    auto found = llvm::find(memory.getAxisIds().asArrayRef(), axis);
    if (size <= 0 || found == memory.getAxisIds().asArrayRef().end() ||
        memory.getShape()[found - memory.getAxisIds().asArrayRef().begin()] != size) return false;
    auto actual = denseDescriptorExtent(descriptor, found - memory.getAxisIds().asArrayRef().begin());
    if (!actual) return false;
    auto count = getConstantIntValue(actual);
    if (!count)
      if (auto extent = dyn_cast<Value>(actual)) count = denseConstantIndex(extent);
    if (count != size) return false;
  }
  return true;
}

StringRef weft::riscv::tileContractInstruction(StringRef instruction) {
  if (instruction == "rvv.tile-vfmacc.vf" || instruction == "rvv.tile-vfmacc-full.vf") return "rvv.vfmacc.vf";
  if (instruction == "rvv.tile-vfwmacc.vf" || instruction == "rvv.tile-vfwmacc-full.vf") return "rvv.vfwmacc.vf";
  return {};
}

bool weft::riscv::tileHasCompleteFields(Operation *at, Value memory,
                                       ValueType value, int64_t fieldAxis) {
  auto type = dyn_cast<MemDescType>(memory.getType());
  auto field = llvm::find(value.getAxisIds().asArrayRef(), fieldAxis);
  if (!type || field == value.getAxisIds().asArrayRef().end()) return false;
  int64_t count = value.getShape()[field - value.getAxisIds().asArrayRef().begin()];
  auto position = llvm::find(type.getAxisIds().asArrayRef(), fieldAxis);
  if (count <= 0 || position == type.getAxisIds().asArrayRef().end()) return false;
  auto extent = denseDescriptorExtent(memory, position - type.getAxisIds().asArrayRef().begin());
  if (!extent) return false;
  if (auto constant = getConstantIntValue(extent)) return *constant >= count;
  auto index = dyn_cast<Value>(extent);
  return index && denseIndexBoundsAt(index, at).lower >= uint64_t(count);
}

std::optional<VectorTileSegments> VectorTileSegments::get(
    ValueType type, int64_t laneAxis, int64_t fieldAxis, int64_t fields) {
  auto tile = VectorTileGeometry::get(type, laneAxis);
  auto axis = llvm::find(type.getAxisIds().asArrayRef(), fieldAxis);
  if (!tile || fieldAxis == laneAxis || axis == type.getAxisIds().asArrayRef().end() ||
      !llvm::is_contained(ArrayRef<int64_t>({2, 4, 8}), fields) ||
      type.getLayout().getLmulEighths() * fields > 64) return std::nullopt;
  size_t position = axis - type.getAxisIds().asArrayRef().begin();
  if (type.getShape()[position] <= 0 || type.getShape()[position] % fields)
    return std::nullopt;
  VectorTileSegments result{std::move(*tile), fieldAxis, fields, position, {}};
  SmallVector<bool> covered(result.tile.parts(), false);
  for (int64_t part = 0; part < result.tile.parts(); ++part) {
    auto point = result.tile.starts[part];
    if (point[position] % fields) continue;
    const int64_t begin = point[position];
    SmallVector<int64_t> group;
    for (int64_t field = 0; field < fields; ++field) {
      point[position] = begin + field;
      auto found = llvm::find(result.tile.starts, point);
      if (found == result.tile.starts.end()) return std::nullopt;
      size_t index = found - result.tile.starts.begin();
      if (covered[index]) return std::nullopt;
      covered[index] = true;
      group.push_back(index);
    }
    result.groups.push_back(std::move(group));
  }
  if (llvm::any_of(covered, [](bool value) { return !value; })) return std::nullopt;
  return result;
}

std::optional<VectorTileProjection> VectorTileProjection::get(
    ValueType source, ValueType result, int64_t axis, ArrayRef<int64_t> offsets) {
  auto input = VectorTileGeometry::get(source, axis);
  auto output = VectorTileGeometry::get(result, axis);
  if (!input || !output || source.getElementType() != result.getElementType() ||
      source.getAxisIds() != result.getAxisIds() || offsets.size() != source.getShape().size() ||
      input->lanes < output->lanes || input->lanes % output->lanes) return std::nullopt;
  for (auto [position, offset] : llvm::enumerate(offsets))
    if (offset < 0 || result.getShape()[position] > source.getShape()[position] ||
        offset > source.getShape()[position] - result.getShape()[position]) return std::nullopt;
  VectorTileProjection projection;
  for (auto point : output->starts) {
    for (auto [position, offset] : llvm::enumerate(offsets)) point[position] += offset;
    auto start = point;
    int64_t offset = point[input->lanePosition] % input->lanes;
    start[input->lanePosition] -= offset;
    auto part = llvm::find(input->starts, start);
    if (part == input->starts.end() || offset % output->lanes || offset + output->lanes > input->lanes)
      return std::nullopt;
    projection.sourceParts.push_back(part - input->starts.begin());
    projection.laneOffsets.push_back(offset);
  }
  return projection;
}

namespace {
bool denseAccessContract(AccessAttr access, MemDescType memory, int64_t fields) {
  return access.getMapping() == "dense" && access.getAlignment() <= memory.getAlignment() &&
      access.getIndexSEW() == 0 && access.getSegmentFields() == fields &&
      access.getGroupSize() == 0 && access.getLayerSize() == 0 && access.getJoinFields() == 0 &&
      access.getJoinLowBits() == 0 && access.getJoinRole() == 0 && access.getBitOffset() == 0 &&
      access.getStorageBits() == 0 && access.getOrder() == "none" && access.getFieldChunkBytes() == 1;
}

LogicalResult memoryContract(Operation *op, Value memory, ValueType type,
                             ValueRange extents, int64_t axis, AccessAttr access, LeafAttr leaf, bool write) {
  auto kernel = op->getParentOfType<KernelOp>();
  auto descriptor = cast<MemDescType>(memory.getType());
  StringRef family = write ? "tile-store" : "tile-load";
  std::string instruction = write ? "rvv.tile-store" : "rvv.tile-load";
  bool segment = leaf.getInstruction().starts_with(instruction + ".segment");
  if (segment) instruction += ".segment";
  bool full = leaf.getInstruction() == instruction + ".full" ||
              leaf.getInstruction() == instruction + ".full.dynamic-stride";
  if (full) instruction += ".full";
  bool dynamic = leaf.getInstruction() == instruction + ".dynamic-stride";
  if (dynamic) instruction += ".dynamic-stride";
  if (!kernel) return op->emitOpError("vector tile transfer requires a target kernel");
  if (!VectorTileGeometry::get(type, axis))
    return op->emitOpError("vector tile transfer has no complete part geometry for axis ") << axis << ": " << type;
  if (!supportsRVVLayout(kernel.getTarget(), type.getLayout()))
    return op->emitOpError("vector tile transfer carrier is unsupported by its target: ") << type;
  if (!tileMemoryElementMatches(descriptor, type.getElementType()))
    return op->emitOpError("vector tile transfer element ") << type.getElementType()
        << " does not match dense storage " << descriptor.getEncoding();
  if (!tileExtentsMatch(memory, type, extents))
    return op->emitOpError("vector tile transfer bounds must belong to its exact descriptor and axes");
  if (descriptor.getAccess() != "readwrite" && descriptor.getAccess() != (write ? "write" : "read"))
    return op->emitOpError("vector tile transfer disagrees with the descriptor access mode");
  auto parameters = leaf.getParameters().asArrayRef();
  if (segment) {
    if (dynamic || parameters.size() != 2 || !kernel.getTarget().getHasSegmentMemory() ||
        !isAddressableDenseDescriptor(memory) ||
        descriptor.getStorageBits() != type.getElementType().getIntOrFloatBitWidth() ||
        !VectorTileSegments::get(type, axis, parameters[0], parameters[1]) ||
        access.getForm() != "segment" || !denseAccessContract(access, descriptor, parameters[1]) ||
        !hasUnitStrideProof(op, memory, parameters[0]) ||
        !tileHasCompleteFields(op, memory, type, parameters[0]))
      return op->emitOpError("segmented tile transfer requires complete contiguous fields, legal tuple registers and exact descriptor bounds");
    if (write && !hasUniqueRegisterUse(cast<RVVTileStoreOp>(op).getInput()))
      return op->emitOpError("zero-scratch segmented tile store requires the last use of its tuple fields");
  } else if (!parameters.empty() ||
             (access.getForm() != "unit" && access.getForm() != "strided") ||
             !denseAccessContract(access, descriptor, 0)) {
    return op->emitOpError("vector tile transfer requires an exact unit or strided dense access");
  }
  if (!weft::riscv::detail::exactLeaf(leaf, "transfer", family, instruction, "none", "exact") ||
      leaf.getTemporaryGroups() || leaf.getFragmentGroups() || leaf.getLocalBytes())
    return op->emitOpError("vector tile transfer requires its exact zero-scratch transfer leaf");
  if (full && !tileIsCompleteMemory(kernel.getTarget(), memory, type))
    return op->emitOpError("full vector tile transfer requires exact static bounds and complete target vector groups");
  if (access.getForm() == "unit" && !hasUnitStrideProof(op, memory, axis))
    return op->emitOpError("unit vector tile transfer requires a dominating or static stride proof");
  if (dynamic && (access.getForm() != "strided" || !hasDynamicDenseStride(descriptor, type)))
    return op->emitOpError("dynamic vector tile transfer requires the corresponding ABI lane stride");
  int64_t groups = type.getLayout().getRegisterGroups();
  if (leaf.getOperandGroups() != (write ? groups : 0) || leaf.getResultGroups() != (write ? 0 : groups))
    return op->emitOpError("vector tile transfer resources must cover every resident part");
  return success();
}
}

LogicalResult RVVTileLoadOp::verify() {
  return memoryContract(*this, getSource(), getResult().getType(), getExtents(), getAxis(), getAccess(), getLeaf(), false);
}

LogicalResult RVVTileSliceOp::verify() {
  auto input = getInput().getType(), result = getResult().getType();
  auto projection = VectorTileProjection::get(input, result, getAxis(), getOffsets());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  bool split = getLeaf().getInstruction() == "rvv.tile-slice.split";
  StringRef instruction = split ? "rvv.tile-slice.split" : "rvv.tile-slice";
  if (!kernel || !projection || !supportsRVVLayout(kernel.getTarget(), result.getLayout()) ||
      !weft::riscv::detail::exactLeaf(getLeaf(), "transfer", "tile-slice", instruction, "none", "exact") ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != result.getLayout().getRegisterGroups() ||
      getLeaf().getTemporaryGroups() || getLeaf().getFragmentGroups() ||
      getLeaf().getLocalBytes() || !getLeaf().getParameters().empty())
    return emitOpError("vector tile slice requires an exact logical subdomain and bounded register projection");
  if (split) {
    auto source = input.getLayout(), destination = result.getLayout();
    int64_t capacity = kernel.getTarget().getVlenBits() / destination.getSew() *
        (destination.getLmulEighths() / 8);
    if (!supportsRVVLayout(kernel.getTarget(), source) || source.getSew() != destination.getSew() ||
        destination.getLmulEighths() < 8 || source.getLmulEighths() <= destination.getLmulEighths() ||
        source.getLmulEighths() % destination.getLmulEighths() || capacity <= 0 ||
        destination.getVl() != capacity ||
        source.getVl() != kernel.getTarget().getVlenBits() / source.getSew() * (source.getLmulEighths() / 8) ||
        llvm::any_of(projection->laneOffsets, [&](int64_t offset) { return offset % capacity; }))
      return emitOpError("tile register split must select aligned complete hardware register groups");
  } else if (input.getLayout().getLmulEighths() != result.getLayout().getLmulEighths() ||
             llvm::any_of(projection->laneOffsets, [](int64_t offset) { return offset != 0; })) {
    return emitOpError("tile tuple selection cannot change the vector carrier");
  }
  return success();
}

LogicalResult RVVTileStoreOp::verify() {
  return memoryContract(*this, getDestination(), getInput().getType(), getExtents(), getAxis(), getAccess(), getLeaf(), true);
}

LogicalResult RVVTileContractOp::verify() {
  auto type = getResult().getType();
  auto instruction = tileContractInstruction(getLeaf().getInstruction());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto tile = VectorTileGeometry::get(type, getAxis());
  auto mapping = ContractStepGeometry::get(getLhs().getType(), getRhs().getType(), type,
                                          getReductionAxis(), getLaneOperand(), instruction);
  bool full = getLeaf().getInstruction() == "rvv.tile-vfmacc-full.vf" ||
              getLeaf().getInstruction() == "rvv.tile-vfwmacc-full.vf";
  if (!kernel || !tile || !mapping || getAccumulator().getType() != type ||
      getExtents().size() != type.getShape().size() || getAxis() == getReductionAxis() ||
      !weft::riscv::detail::exactLeaf(getLeaf(), "rvv", "tile-contract", getLeaf().getInstruction(), "none", full ? "exact" : "undisturbed") ||
      getLeaf().getParameters().asArrayRef() != ArrayRef<int64_t>({int64_t(getReductionAxis()), int64_t(getAxis())}) ||
      getLeaf().getTemporaryGroups() || getLeaf().getFragmentGroups() || getLeaf().getLocalBytes())
    return emitOpError("vector tile contraction requires a closed scalar/vector mapping and per-axis active extents");
  for (Value operand : getOperands())
    if (auto value = dyn_cast<ValueType>(operand.getType()); value && value.getLayout().getCarrier() == "rvv" &&
        !supportsRVVLayout(kernel.getTarget(), value.getLayout()))
      return emitOpError("vector tile contraction has an illegal SEW/LMUL carrier");
  for (auto [extent, capacity] : llvm::zip(getExtents(), type.getShape().asArrayRef())) {
    auto value = getConstantIntValue(extent);
    if (full && (!value || *value != capacity))
      return emitOpError("complete vector tile contraction requires every axis bound to equal its selected capacity");
    if (value && (*value < 0 || *value > capacity))
      return emitOpError("vector tile active extent exceeds its selected logical capacity");
  }
  if (full && tile->lanes != kernel.getTarget().getVlenBits() * type.getLayout().getLmulEighths() /
      (8 * type.getLayout().getSew()))
    return emitOpError("complete vector tile contraction must cover the complete target vector group");
  return success();
}
