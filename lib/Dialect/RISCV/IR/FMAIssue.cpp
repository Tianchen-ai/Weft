#include "Weft/Dialect/RISCV/IR/FMAIssue.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Verification.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft::riscv;

namespace {
Type element(Type type) {
  if (auto value = dyn_cast<ValueType>(type)) return value.getElementType();
  return type;
}

bool embeds(ValueType input, ValueType result) {
  for (auto [axis, size] : llvm::zip(input.getAxisIds().asArrayRef(),
                                   input.getShape().asArrayRef())) {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (found == result.getAxisIds().asArrayRef().end() || size <= 0 ||
        size != result.getShape()[found - result.getAxisIds().asArrayRef().begin()])
      return false;
  }
  return true;
}

std::optional<int64_t> project(ValueType input, ValueType result,
                               int64_t part, bool scalar) {
  auto source = RegisterCoordinates::get(input.getLayout());
  auto destination = RegisterCoordinates::get(result.getLayout());
  if (!source || !destination || (scalar && source->lanes() != 1) ||
      (!scalar && source->lanes() != destination->lanes())) return std::nullopt;
  std::optional<int64_t> selected;
  for (int64_t lane = 0; lane < destination->lanes(); ++lane) {
    auto point = destination->logical(part, lane);
    if (!point) return std::nullopt;
    SmallVector<int64_t> coordinates;
    for (int64_t axis : input.getAxisIds().asArrayRef()) {
      auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
      if (found == result.getAxisIds().asArrayRef().end()) return std::nullopt;
      coordinates.push_back((*point)[found - result.getAxisIds().asArrayRef().begin()]);
    }
    auto location = source->locate(coordinates);
    if (!location || location->lane != (scalar ? 0 : lane) ||
        (selected && *selected != location->part)) return std::nullopt;
    selected = location->part;
  }
  return selected;
}

bool groupFits(const FMAIssueGeometry &geometry, int64_t begin, int64_t end,
               int64_t budget) {
  if (begin < 0 || end <= begin || end > geometry.parts()) return false;
  llvm::DenseSet<int64_t> vectors, multipliers;
  for (int64_t part = begin; part < end; ++part) {
    vectors.insert(geometry.vectorParts[part]);
    multipliers.insert(geometry.multiplierParts[part]);
  }
  // The issue boundary has tied read/write outputs. GCC counts each twice.
  // Input FMA values remain ordinary intrinsic operands, not hidden asm loads.
  int64_t operands = 2 * (end - begin);
  int64_t groups = (end - begin) * geometry.outputGroups +
      (vectors.size() + (geometry.scalar ? 0 : multipliers.size())) * geometry.inputGroups;
  return operands <= 30 && groups <= budget;
}
} // namespace

std::optional<FMAIssueGeometry> FMAIssueGeometry::get(
    ValueType vector, Type multiplier, ValueType result) {
  auto other = dyn_cast<ValueType>(multiplier);
  auto output = RegisterCoordinates::get(result.getLayout());
  if (!output || output->parts() < 1 || output->parts() > 16 ||
      result.getLayout().getCarrier() != "rvv" ||
      vector.getLayout().getCarrier() != "rvv" || !embeds(vector, result) ||
      (other && (!embeds(other, result) ||
                 (other.getLayout().getCarrier() != "rvv" &&
                  other.getLayout().getCarrier() != "scalar")))) return std::nullopt;
  FMAIssueGeometry geometry;
  geometry.scalar = !other || other.getLayout().getCarrier() == "scalar";
  Type input = vector.getElementType(), product = result.getElementType();
  geometry.widening = input.isF16() && product.isF32();
  if (element(multiplier) != input ||
      (!geometry.widening && (input != product || (!product.isF32() && !product.isF64()))))
    return std::nullopt;
  int64_t width = vector.getLayout().getLmulEighths();
  if (width * (geometry.widening ? 2 : 1) != result.getLayout().getLmulEighths() ||
      (!geometry.scalar && other.getLayout().getLmulEighths() != width)) return std::nullopt;
  // Every output coordinate is supplied by an input. No implicit replication
  // over an axis absent from both operands is permitted by this local op.
  for (int64_t axis : result.getAxisIds().asArrayRef())
    if (!llvm::is_contained(vector.getAxisIds().asArrayRef(), axis) &&
        (!other || !llvm::is_contained(other.getAxisIds().asArrayRef(), axis))) return std::nullopt;
  geometry.lanes = output->lanes();
  if (geometry.lanes != result.getLayout().getVl() || geometry.lanes <= 1)
    return std::nullopt;
  geometry.inputGroups = (width + 7) / 8;
  geometry.outputGroups = (result.getLayout().getLmulEighths() + 7) / 8;
  for (int64_t part = 0; part < output->parts(); ++part) {
    auto first = project(vector, result, part, false);
    auto second = other ? project(other, result, part, geometry.scalar)
                        : std::optional<int64_t>(0);
    if (!first || !second) return std::nullopt;
    geometry.vectorParts.push_back(*first);
    geometry.multiplierParts.push_back(*second);
  }
  return geometry;
}

StringRef FMAIssueGeometry::instruction() const {
  if (widening) return scalar ? "rvv.issue-vfwmacc.vf" : "rvv.issue-vfwmacc.vv";
  return scalar ? "rvv.issue-vfmacc.vf" : "rvv.issue-vfmacc.vv";
}

bool weft::riscv::canReuseFMAAccumulator(Operation *operation, Value accumulator) {
  if (!hasUniqueRegisterUse(accumulator)) return false;
  // A same-block representation bridge can still alias a captured value.
  // Mutating that outer binding would change the next loop iteration's seed.
  llvm::DenseSet<Value> visited;
  while (Value source = registerAliasSource(accumulator)) {
    if (!visited.insert(accumulator).second || visited.size() > 128) return false;
    accumulator = source;
  }
  if (auto argument = dyn_cast<BlockArgument>(accumulator))
    return argument.getOwner() == operation->getBlock();
  auto definition = accumulator.getDefiningOp();
  return definition && definition->getBlock() == operation->getBlock();
}

int64_t weft::riscv::fmaIssueCopyGroups(RVVFMAIssueOp operation) {
  return operation.getAccumulatorPolicy() == "reuse" ? 0 :
      operation.getResult().getType().getLayout().getRegisterGroups();
}

std::optional<SmallVector<int64_t>> weft::riscv::selectFMAIssueGroups(
    const FMAIssueGeometry &geometry, int64_t budget) {
  SmallVector<int64_t> ends;
  for (int64_t begin = 0; begin < geometry.parts();) {
    int64_t end = begin;
    while (end < geometry.parts() && groupFits(geometry, begin, end + 1, budget)) ++end;
    if (end == begin) return std::nullopt;
    ends.push_back(end);
    begin = end;
  }
  return ends;
}

bool weft::riscv::legalFMAIssueGroups(const FMAIssueGeometry &geometry,
                                     ArrayRef<int64_t> ends, int64_t budget) {
  int64_t begin = 0;
  for (int64_t end : ends) {
    if (!groupFits(geometry, begin, end, budget)) return false;
    begin = end;
  }
  return !ends.empty() && begin == geometry.parts();
}

LogicalResult RVVFMAIssueOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto type = getResult().getType();
  auto geometry = FMAIssueGeometry::get(getVector().getType(), getMultiplier().getType(), type);
  if (!kernel || !geometry || getAccumulator().getType() != type ||
      getLengths().size() != size_t(geometry->parts()))
    return emitOpError("FMA issue requires one closed numeric coordinate mapping and one VL per output part");
  for (Value operand : getOperands())
    if (auto value = dyn_cast<ValueType>(operand.getType());
        value && value.getLayout().getCarrier() == "rvv" &&
        !supportsRVVLayout(kernel.getTarget(), value.getLayout()))
      return emitOpError("FMA issue has an unsupported numeric SEW/LMUL carrier");
  if (geometry->widening && !kernel.getTarget().getHasWideningFloat())
    return emitOpError("FMA issue requires target floating widening support");
  bool reuse = getAccumulatorPolicy() == "reuse";
  if ((!reuse && getAccumulatorPolicy() != "copy") ||
      (reuse && !canReuseFMAAccumulator(*this, getAccumulator())))
    return emitOpError("FMA accumulator reuse requires a unique same-block final use");
  if (!legalFMAIssueGroups(*geometry, getGroupEnds(), kernel.getTarget().getVectorRegisters()))
    return emitOpError("FMA issue groups exceed the register or assembly operand budget");
  if (getTail() != "agnostic" && getTail() != "undisturbed")
    return emitOpError("FMA issue requires an explicit lane-tail policy");
  for (Value length : getLengths()) {
    auto bounds = denseIndexBoundsAt(length, *this);
    if (bounds.upper > uint64_t(geometry->lanes))
      return emitOpError("FMA issue VL must be proven within its numeric carrier");
    if (getTail() == "agnostic") {
      auto constant = getConstantIntValue(length);
      auto layout = getVector().getType().getLayout();
      if (!constant || *constant != geometry->lanes || geometry->lanes !=
          kernel.getTarget().getVlenBits() * layout.getLmulEighths() / (8 * layout.getSew()))
        return emitOpError("agnostic FMA issue requires a constant complete target vector group");
    }
  }
  auto leaf = getLeaf();
  if (!detail::exactLeaf(leaf, "rvv", "fma-issue", geometry->instruction(), "none", getTail()) ||
      leaf.getTemporaryGroups() != fmaIssueCopyGroups(*this) ||
      leaf.getFragmentGroups() || leaf.getLocalBytes() || !leaf.getParameters().empty())
    return emitOpError("FMA issue lost its exact numeric instruction or accumulator-copy resource contract");
  return success();
}
