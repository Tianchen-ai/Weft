#include "Weft/Dialect/RISCV/IR/Contraction.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Verification.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace weft::riscv;
using namespace weft::riscv::detail;

namespace {
mlir::Type element(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type)) return value.getElementType();
  return type;
}

bool appendAxes(mlir::Type type, int64_t reductionAxis,
                llvm::SmallVectorImpl<int64_t> &axes,
                llvm::SmallVectorImpl<int64_t> &shape) {
  auto value = mlir::dyn_cast<ValueType>(type);
  if (!value) return mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
  if (value.getLayout().getCarrier() != "scalar" &&
      value.getLayout().getCarrier() != "rvv") return false;
  for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                      value.getShape().asArrayRef())) {
    if (axis == reductionAxis) return false;
    auto found = llvm::find(axes, axis);
    if (found == axes.end()) {
      axes.push_back(axis);
      shape.push_back(extent);
    } else if (shape[found - axes.begin()] != extent) return false;
  }
  return true;
}
}

std::optional<ContractStepGeometry> ContractStepGeometry::get(
    mlir::Type lhs, mlir::Type rhs, ValueType result,
    int64_t reductionAxis, llvm::StringRef laneOperand,
    llvm::StringRef instruction) {
  if (reductionAxis <= 0 || (laneOperand != "lhs" && laneOperand != "rhs") ||
      result.getLayout().getCarrier() != "rvv") return std::nullopt;
  llvm::SmallVector<int64_t> axes, shape;
  if (!appendAxes(lhs, reductionAxis, axes, shape) ||
      !appendAxes(rhs, reductionAxis, axes, shape) ||
      result.getAxisIds().size() != axes.size()) return std::nullopt;
  // Local operand projection may put a shared singleton batch axis on only
  // one operand. Preserve the result's canonical axis order and prove the
  // numeric mapping by identity, rather than deriving it from operand order.
  for (auto [axis, extent] : llvm::zip(result.getAxisIds().asArrayRef(), result.getShape().asArrayRef())) {
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || shape[found - axes.begin()] != extent) return std::nullopt;
  }
  axes.assign(result.getAxisIds().asArrayRef().begin(), result.getAxisIds().asArrayRef().end());
  auto lane = mlir::dyn_cast<ValueType>(laneOperand == "lhs" ? lhs : rhs);
  auto scalarType = laneOperand == "lhs" ? rhs : lhs;
  auto scalar = mlir::dyn_cast<ValueType>(scalarType);
  if (!lane || lane.getLayout().getCarrier() != "rvv" ||
      (scalar && scalar.getLayout().getCarrier() != "scalar")) return std::nullopt;
  ContractStepGeometry geometry;
  const bool integerWidening = instruction == "rvv.vwmacc.vx";
  geometry.widening = instruction == "rvv.vfwmacc.vf" || integerWidening;
  geometry.floating = instruction == "rvv.vfmacc.vf" || instruction == "rvv.vfwmacc.vf";
  auto outputElement = result.getElementType();
  if (integerWidening) {
    auto narrow = mlir::dyn_cast<mlir::IntegerType>(element(lhs));
    auto wide = mlir::dyn_cast<mlir::IntegerType>(outputElement);
    if (!narrow || !wide || !narrow.isSigned() || !wide.isSigned() ||
        narrow.getWidth() != 16 || wide.getWidth() != 32 || element(rhs) != narrow)
      return std::nullopt;
  } else if (geometry.widening) {
    if (!element(lhs).isF16() || !element(rhs).isF16() ||
        !outputElement.isF32()) return std::nullopt;
  } else if (geometry.floating) {
    if ((!outputElement.isF32() && !outputElement.isF64()) ||
        element(lhs) != outputElement || element(rhs) != outputElement) return std::nullopt;
  } else if (instruction != "rvv.vmacc.vx" ||
             !mlir::isa<mlir::IntegerType>(outputElement) ||
             element(lhs) != outputElement || element(rhs) != outputElement) return std::nullopt;
  // The intrinsic's source vector type is fixed by its result carrier.
  // Matching logical coordinates alone does not prove that C vector types
  // have the same LMUL, or that widening doubles the register group exactly.
  if (lane.getLayout().getLmulEighths() * (geometry.widening ? 2 : 1) !=
      result.getLayout().getLmulEighths()) return std::nullopt;
  auto outputCoordinates = RegisterCoordinates::get(result.getLayout());
  auto laneCoordinates = RegisterCoordinates::get(lane.getLayout());
  auto scalarCoordinates = scalar ? RegisterCoordinates::get(scalar.getLayout())
                                  : std::optional<RegisterCoordinates>();
  if (!outputCoordinates || !laneCoordinates ||
      (scalar && !scalarCoordinates) ||
      laneCoordinates->lanes() != outputCoordinates->lanes() ||
      (scalarCoordinates && scalarCoordinates->lanes() != 1)) return std::nullopt;
  // Verify every lane as well as every part. Equal carrier sizes alone do not
  // establish that a transpose or an interleaved replica has the same mapping.
  for (int64_t part = 0; part < outputCoordinates->parts(); ++part) {
    std::optional<int64_t> selectedLane, selectedScalar;
    for (int64_t index = 0; index < outputCoordinates->lanes(); ++index) {
      auto outputPoint = outputCoordinates->logical(part, index);
      if (!outputPoint) return std::nullopt;
      auto project = [&](ValueType input, const RegisterCoordinates &coordinates)
          -> std::optional<RegisterLocation> {
        llvm::SmallVector<int64_t> point;
        for (int64_t axis : input.getAxisIds().asArrayRef()) {
          auto position = llvm::find(axes, axis);
          if (position == axes.end()) return std::nullopt;
          point.push_back((*outputPoint)[position - axes.begin()]);
        }
        return coordinates.locate(point);
      };
      auto location = project(lane, *laneCoordinates);
      if (!location || location->lane != index ||
          (selectedLane && *selectedLane != location->part)) return std::nullopt;
      selectedLane = location->part;
      int64_t scalarPart = 0;
      if (scalar) {
        auto scalarLocation = project(scalar, *scalarCoordinates);
        if (!scalarLocation || scalarLocation->lane != 0) return std::nullopt;
        scalarPart = scalarLocation->part;
      }
      if (selectedScalar && *selectedScalar != scalarPart) return std::nullopt;
      selectedScalar = scalarPart;
    }
    if (!selectedLane || !selectedScalar) return std::nullopt;
    geometry.laneParts.push_back(*selectedLane);
    geometry.scalarParts.push_back(*selectedScalar);
  }
  return geometry;
}

mlir::LogicalResult RVVContractStepOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel) return emitOpError("contract step must belong to a target kernel");
  if (getLeaf().getInstruction() == "rvv.vwmacc.vx" && !kernel.getTarget().getHasWideningInteger())
    return emitOpError("integer widening contract step requires the target capability");
  for (auto operand : getOperands())
    if (auto type = mlir::dyn_cast<ValueType>(operand.getType());
        type && type.getLayout().getCarrier() == "rvv" && !supportsRVVLayout(kernel.getTarget(), type.getLayout()))
      return emitOpError("contract step operand has an unsupported SEW/LMUL carrier");
  auto geometry = ContractStepGeometry::get(getLhs().getType(), getRhs().getType(),
      getResult().getType(), getReductionAxis(), getLaneOperand(), getLeaf().getInstruction());
  if (getActive()) {
    auto layout = getResult().getType().getLayout();
    if (llvm::any_of(layout.getTimeFactors().asArrayRef(), [](int64_t n) { return n != 1; }) ||
        llvm::count_if(layout.getLaneFactors().asArrayRef(), [](int64_t n) { return n > 1; }) != 1)
      return emitOpError("explicit contract VL requires one issue along one output lane axis");
    if (auto active = mlir::getConstantIntValue(getActive()); active && (*active < 0 || *active > layout.getVl()))
      return emitOpError("contract active length exceeds its selected output carrier");
  }
  if (!geometry || getAccumulator().getType() != getResult().getType() ||
      !exactLeaf(getLeaf(), "rvv", "contract-step", getLeaf().getInstruction(),
                 "none", getActive() ? "undisturbed" : "agnostic") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({static_cast<int64_t>(getReductionAxis())}))
    return emitOpError("contract step requires fully projected numeric operands, a scalar/vector coordinate mapping, and an exact FMA leaf");
  return verifyLeafOperation(*this);
}
