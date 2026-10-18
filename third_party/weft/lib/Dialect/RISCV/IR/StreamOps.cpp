#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/Stream.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "Weft/Dialect/RISCV/IR/ScalarPacket.h"
#include "Verification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace weft;
using namespace weft::riscv;
using namespace weft::riscv::detail;

namespace {
bool canRead(llvm::StringRef access) { return access == "read" || access == "readwrite"; }
bool canWrite(llvm::StringRef access) { return access == "write" || access == "readwrite"; }

bool legalCarrier(mlir::Operation *operation, ValueType type) {
  auto kernel = operation->getParentOfType<KernelOp>();
  return kernel && supportsRVVLayout(kernel.getTarget(), type.getLayout());
}

bool provesComplete(mlir::Value predicate, mlir::Value extent, int64_t required) {
  if (auto both = predicate.getDefiningOp<mlir::arith::AndIOp>())
    return provesComplete(both.getLhs(), extent, required) ||
           provesComplete(both.getRhs(), extent, required);
  auto compare = predicate.getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare || compare.getPredicate() != mlir::arith::CmpIPredicate::uge) return false;
  auto bound = denseConstantIndex(compare.getRhs());
  return bound && *bound >= required && sameDenseIndex(compare.getLhs(), extent);
}

bool completeFreeAxis(mlir::Operation *load, mlir::Value extent, int64_t required) {
  if (denseIndexBounds(extent).lower >= uint64_t(required)) return true;
  for (mlir::Operation *child = load, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp())
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent);
        branch && child->getParentRegion() == &branch.getThenRegion() &&
        provesComplete(branch.getCondition(), extent, required)) return true;
  return false;
}

}

std::optional<StreamGeometry> StreamGeometry::get(ValueType type, int64_t axis) {
  auto layout = type.getLayout();
  auto coordinates = RegisterCoordinates::get(layout);
  if (!coordinates || layout.getCarrier() != "rvv" ||
      layout.getReplicaGroup() != 1 || coordinates->streams() != 1 ||
      coordinates->lanes() != layout.getVl() || layout.getVl() <= 1 ||
      !llvm::is_contained(type.getAxisIds().asArrayRef(), axis)) return std::nullopt;
  StreamGeometry result{axis, layout.getVl(), coordinates->parts(), {}, {}};
  for (auto [position, id] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1) return std::nullopt;
    if (id == axis) {
      if (layout.getLaneFactors()[position] != result.lanes ||
          layout.getReplicaFactors()[position] != 1 ||
          type.getShape()[position] != result.lanes) return std::nullopt;
    } else {
      int64_t extent = layout.getReplicaFactors()[position];
      if (extent <= 0 || layout.getLaneFactors()[position] != 1 ||
          type.getShape()[position] != extent) return std::nullopt;
      result.freeAxes.push_back(id);
      result.freeShape.push_back(extent);
    }
  }
  return result;
}

std::optional<StreamContractGeometry> StreamContractGeometry::get(
    ValueType lhs, ValueType rhs, ValueType accumulator, int64_t axis) {
  auto left = StreamGeometry::get(lhs, axis);
  auto right = StreamGeometry::get(rhs, axis);
  auto output = StreamGeometry::get(accumulator, axis);
  if (!left || !right || !output || left->lanes != output->lanes ||
      right->lanes != output->lanes || accumulator.getLayout().getValidity() != "full")
    return std::nullopt;
  bool widening = lhs.getElementType().isF16() && rhs.getElementType().isF16();
  bool ordinary = lhs.getElementType() == accumulator.getElementType() &&
                  rhs.getElementType() == accumulator.getElementType();
  if ((!widening && !ordinary) ||
      (!accumulator.getElementType().isF32() && !accumulator.getElementType().isF64()) ||
      (widening && !accumulator.getElementType().isF32()) ||
      lhs.getLayout().getLmulEighths() != rhs.getLayout().getLmulEighths() ||
      accumulator.getLayout().getLmulEighths() != lhs.getLayout().getLmulEighths() * (widening ? 2 : 1))
    return std::nullopt;
  llvm::DenseSet<int64_t> unionAxes;
  for (const auto *input : {&*left, &*right}) {
    for (auto [id, extent] : llvm::zip(input->freeAxes, input->freeShape)) {
      auto found = llvm::find(output->freeAxes, id);
      if (found == output->freeAxes.end() ||
          output->freeShape[found - output->freeAxes.begin()] != extent) return std::nullopt;
      unionAxes.insert(id);
    }
  }
  if (unionAxes.size() != output->freeAxes.size()) return std::nullopt;
  StreamContractGeometry result;
  for (int64_t part = 0; part < output->parts; ++part) {
    auto leftPart = projectCarrierPart(lhs, accumulator, part);
    auto rightPart = projectCarrierPart(rhs, accumulator, part);
    if (!leftPart || !rightPart || *leftPart >= left->parts || *rightPart >= right->parts)
      return std::nullopt;
    result.lhsParts.push_back(*leftPart);
    result.rhsParts.push_back(*rightPart);
  }
  return result;
}

std::optional<llvm::SmallVector<int64_t>> weft::riscv::streamFinalizeParts(
    ValueType input, mlir::Type output, int64_t axis) {
  auto geometry = StreamGeometry::get(input, axis);
  if (!geometry || input.getLayout().getValidity() != "full") return std::nullopt;
  if (geometry->freeAxes.empty() && output == input.getElementType())
    return llvm::SmallVector<int64_t>{0};
  auto type = mlir::dyn_cast<ValueType>(output);
  if (!type || type.getElementType() != input.getElementType() ||
      type.getLayout().getCarrier() != "scalar" ||
      type.getAxisIds().asArrayRef() != llvm::ArrayRef<int64_t>(geometry->freeAxes)) return std::nullopt;
  auto coordinates = RegisterCoordinates::get(type.getLayout());
  auto source = RegisterCoordinates::get(input.getLayout());
  if (!coordinates || !source || coordinates->parts() != geometry->parts ||
      coordinates->lanes() != 1 || coordinates->extents() != llvm::ArrayRef<int64_t>(geometry->freeShape))
    return std::nullopt;
  for (auto [position, extent] : llvm::enumerate(geometry->freeShape))
    if (type.getShape()[position] > 0 && type.getShape()[position] != extent) return std::nullopt;
  llvm::SmallVector<int64_t> parts;
  for (int64_t part = 0; part < coordinates->parts(); ++part) {
    auto point = coordinates->logical(part);
    if (!point) return std::nullopt;
    llvm::SmallVector<int64_t> sourcePoint;
    size_t freePosition = 0;
    for (int64_t id : input.getAxisIds().asArrayRef())
      sourcePoint.push_back(id == axis ? 0 : (*point)[freePosition++]);
    auto location = source->locate(sourcePoint);
    if (!location || location->lane != 0) return std::nullopt;
    parts.push_back(location->part);
  }
  return parts;
}


mlir::LogicalResult ScalarTileLoadOp::verify() {
  if (mlir::failed(verifyScalarPacket(*this))) return mlir::failure();
  auto type = getResult().getType();
  auto layout = type.getLayout();
  auto memory = getSource().getType();
  auto coordinates = RegisterCoordinates::get(layout);
  if (!canRead(memory.getAccess()) || layout.getCarrier() != "scalar" || !coordinates || coordinates->lanes() != 1 ||
      coordinates->parts() < 1 || coordinates->parts() > 16 ||
      !tileMemoryElementMatches(memory, type.getElementType()) ||
      memory.getAxisIds() != type.getAxisIds() || getExtents().size() != type.getShape().size())
    return emitOpError("scalar tile load requires a bounded dense scalar tuple with exact source axes");
  for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
    if (type.getShape()[position] <= 0 || coordinates->extents()[position] != type.getShape()[position])
      return emitOpError("scalar tile load must materialize every declared numeric coordinate");
    auto extent = getExtents()[position];
    if (!sameDenseExtent(extent, getSource(), position))
      return emitOpError("scalar tile load extent must belong to its exact projected descriptor");
    if (!getGuarded() && !completeFreeAxis(*this, extent, type.getShape()[position]))
      return emitOpError("unguarded scalar tile load requires a dominating complete-domain bound");
  }
  auto instruction = getLeaf().getInstruction();
  bool ordinary = instruction == "scalar.tile-load";
  if (!ordinary && instruction != "scalar.tile-load.unit" && instruction != "scalar.tile-load.strided" &&
      instruction != "scalar.tile-load.dynamic-stride")
    return emitOpError("scalar tile load has no selected address form");
  if (ordinary ? !getLeaf().getParameters().empty() : getLeaf().getParameters().size() != 1)
    return emitOpError("scalar tile address form requires its exact selected axis");
  if (!ordinary) {
    int64_t axis = getLeaf().getParameters()[0];
    auto position = llvm::find(type.getAxisIds().asArrayRef(), axis);
    if (position == type.getAxisIds().asArrayRef().end() ||
        type.getShape()[position - type.getAxisIds().asArrayRef().begin()] < 2 ||
        !isInnermostDenseAxis(getSource(), axis) || !denseArgumentStride(getSource(), axis))
      return emitOpError("scalar tile stride selection requires a repeated innermost ABI coordinate");
    if (instruction == "scalar.tile-load.unit" && !hasUnitStrideProof(*this, getSource(), axis))
      return emitOpError("unit scalar tile load requires a dominating exact ABI stride proof");
  }
  if (!exactLeaf(getLeaf(), "transfer", "scalar-tile-load", instruction, "none", "exact") ||
      getLeaf().getOperandGroups() || getLeaf().getResultGroups() || getLeaf().getTemporaryGroups() ||
      getLeaf().getFragmentGroups() || getLeaf().getLocalBytes())
    return emitOpError("scalar tile load has no exact zero-vector-resource leaf");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult ScalarTileStoreOp::verify() {
  auto type = getInput().getType();
  auto memory = getDestination().getType();
  auto coordinates = RegisterCoordinates::get(type.getLayout());
  if (!canWrite(memory.getAccess()) || type.getLayout().getCarrier() != "scalar" ||
      !coordinates || coordinates->lanes() != 1 || coordinates->parts() < 1 || coordinates->parts() > 16 ||
      !tileMemoryElementMatches(memory, type.getElementType()) || !tileExtentsMatch(getDestination(), type, getExtents()))
    return emitOpError("scalar tile store requires an exact bounded scalar tuple and writable dense descriptor");
  for (auto [position, size] : llvm::enumerate(type.getShape().asArrayRef())) {
    if (size <= 0 || coordinates->extents()[position] != size)
      return emitOpError("scalar tile store requires complete numeric coordinates");
    if (!getGuarded() && !completeFreeAxis(*this, getExtents()[position], size))
      return emitOpError("unguarded scalar tile store requires a complete destination bound");
  }
  auto leaf = getLeaf();
  if (!exactLeaf(leaf, "transfer", "scalar-tile-store", "scalar.tile-store", "none", "exact") ||
      leaf.getOperandGroups() || leaf.getResultGroups() || leaf.getTemporaryGroups() ||
      leaf.getFragmentGroups() || leaf.getLocalBytes() || !leaf.getParameters().empty())
    return emitOpError("scalar tile store has no exact zero-vector-resource leaf");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVStreamReduceStepOp::verify() {
  ValueType input = getInput().getType();
  ValueType accumulator = getAccumulator().getType();
  ValueType result = getResult().getType();
  bool full = getLeaf().getInstruction() == "rvv.stream-reduce-full-step";
  auto length = mlir::getConstantIntValue(getActive());
  if (full && (input.getLayout().getValidity() != "full" || !length || *length != input.getLayout().getVl()))
    return emitOpError("full stream reduction step requires the complete input carrier and constant full VL");
  if (!legalCarrier(*this, input) || !legalCarrier(*this, accumulator) ||
      streamAccumulatorType(input) != accumulator || result != accumulator ||
      (!result.getElementType().isF32() && !result.getElementType().isF64() &&
       !(result.getElementType().isInteger(32) && getKind() == "add")) ||
      (getKind() != "add" && getKind() != "max" && getKind() != "min") ||
      !exactLeaf(getLeaf(), "rvv", "stream-reduce-step",
                 full ? "rvv.stream-reduce-full-step" : "rvv.stream-reduce-step", "none", "exact") ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() + accumulator.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != result.getLayout().getRegisterGroups() ||
      getLeaf().getTemporaryGroups() != 0)
    return emitOpError("RVV stream reduce step requires a fully defined floating or i32-add accumulator and matching lane geometry");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVStreamContractStepOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  ValueType accumulator = getAccumulator().getType();
  ValueType result = getResult().getType();
  const bool ordinary = lhs.getElementType() == rhs.getElementType() &&
                        (lhs.getElementType().isF32() || lhs.getElementType().isF64());
  const bool widening = lhs.getElementType().isF16() && rhs.getElementType().isF16();
  const bool full = getLeaf().getInstruction() == "rvv.stream-contract-full-step" ||
                    getLeaf().getInstruction() == "rvv.stream-widen-contract-full-step";
  const llvm::StringRef instruction = widening
      ? (full ? "rvv.stream-widen-contract-full-step" : "rvv.stream-widen-contract-step")
      : (full ? "rvv.stream-contract-full-step" : "rvv.stream-contract-step");
  if (full) {
    auto active = mlir::getConstantIntValue(getActive());
    if (!active || *active != lhs.getLayout().getVl() ||
        lhs.getLayout().getValidity() != "full" || rhs.getLayout().getValidity() != "full")
      return emitOpError("full stream contract requires a constant complete lane window for both operands");
  }
  if (!legalCarrier(*this, lhs) || !legalCarrier(*this, rhs) || !legalCarrier(*this, accumulator) ||
      (!ordinary && !widening) ||
      !StreamContractGeometry::get(lhs, rhs, accumulator, getReductionAxis()) ||
      result != accumulator ||
      accumulator.getLayout().getValidity() != "full" ||
      !exactLeaf(getLeaf(), "rvv", "stream-contract-step", instruction, "none", "exact") ||
      getLeaf().getOperandGroups() != lhs.getLayout().getRegisterGroups() + rhs.getLayout().getRegisterGroups() +
                                        accumulator.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != result.getLayout().getRegisterGroups() ||
      getLeaf().getTemporaryGroups() != 0)
    return emitOpError("RVV stream contract step requires two typed floating lane windows and one fully defined f32/f64 accumulator topology");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVStreamFinalizeOp::verify() {
  ValueType input = getInput().getType();
  if (!legalCarrier(*this, input) || !streamFinalizeParts(input, getResult().getType(), getAxis()) ||
      (!input.getElementType().isF32() && !input.getElementType().isF64() &&
       !(input.getElementType().isInteger(32) && getKind() == "add")) ||
      (getKind() != "add" && getKind() != "max" && getKind() != "min") ||
      !exactLeaf(getLeaf(), "rvv", "stream-finalize", "rvv.stream-finalize", "none", "exact") ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != 0 || getLeaf().getTemporaryGroups() != 1)
    return emitOpError("RVV stream finalize requires fully defined floating or i32-add reduction lanes and an exact scalar free-axis projection");
  return verifyLeafOperation(*this);
}
