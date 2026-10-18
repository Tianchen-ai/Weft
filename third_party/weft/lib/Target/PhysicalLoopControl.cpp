#include "PhysicalLoopControl.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

FailureOr<bool> riscv_internal::descendingPhysicalLoop(scf::ForOp loop) {
  Attribute levelValue = loop->getAttr("weft.riscv.level");
  Attribute directionValue = loop->getAttr("weft.riscv.direction");
  auto level = dyn_cast_or_null<riscv::LevelAttr>(levelValue);
  auto direction = dyn_cast_or_null<StringAttr>(directionValue);
  if ((levelValue && !level) || (directionValue && !direction))
    return loop.emitError("physical loop direction requires typed Level and string direction attributes"), failure();
  StringRef selected = level ? level.getDirection() : direction ? direction.getValue() : StringRef();
  if (selected != "ascending" && selected != "descending")
    return loop.emitError("ordered physical loop has no selected ascending or descending direction"), failure();
  if (level && direction && level.getDirection() != direction.getValue())
    return loop.emitError("physical loop direction disagrees with its author Level"), failure();
  return selected == "descending";
}

FailureOr<PhysicalLoopControl> PhysicalLoopControl::get(scf::ForOp loop) {
  auto direction = descendingPhysicalLoop(loop);
  if (failed(direction)) return failure();
  riscv::TargetAttr target;
  if (auto kernel = loop->getParentOfType<riscv::KernelOp>())
    target = kernel.getTarget();
  else if (auto artifact = loop->getParentOfType<riscv::ArtifactPackOp>())
    target = artifact.getTarget();
  if (!target)
    return loop.emitError("physical loop arithmetic has no owning target kernel or artifact builder"), failure();
  auto type = loop.getInductionVar().getType();
  if (!type.isIndex() || loop.getLowerBound().getType() != type ||
      loop.getUpperBound().getType() != type || loop.getStep().getType() != type)
    return loop.emitError("ordered physical loop requires target index coordinates, bounds and step"), failure();
  StringRef march = target.getMarch();
  if (!march.starts_with("rv32") && !march.starts_with("rv64"))
    return loop.emitError("physical loop requires a target with an explicit index width"), failure();
  PhysicalLoopControl result{*direction, march.starts_with("rv32") ? 32u : 64u, {}, {}, {}};
  auto constant = [&](Value value) -> FailureOr<std::optional<llvm::APInt>> {
    llvm::APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer))) return std::optional<llvm::APInt>{};
    // Index constants have a host-side 64-bit spelling even for RV32. Accept
    // either a signed spelling or the complete unsigned target bit pattern;
    // truncating any additional significant bits would change the program.
    if (!integer.isSignedIntN(result.width) && !integer.isIntN(result.width))
      return loop.emitError("physical loop constant does not fit the target index width"), failure();
    return std::optional<llvm::APInt>(integer.sextOrTrunc(result.width));
  };
  auto lower = constant(loop.getLowerBound());
  auto upper = constant(loop.getUpperBound());
  auto step = constant(loop.getStep());
  if (failed(lower) || failed(upper) || failed(step)) return failure();
  result.lower = *lower;
  result.upper = *upper;
  result.step = *step;
  if (result.step) {
    if (result.step->isZero())
      return loop.emitError("physical loop has a zero induction step"), failure();
    if (result.step->isNegative() != result.descending)
      return loop.emitError("physical loop step disagrees with its selected direction"), failure();
  }
  return result;
}

llvm::APInt PhysicalLoopControl::coordinate(const llvm::APInt &value) const {
  // Two extra bits hold any difference, signed stride, and one exit increment.
  return descending ? value.sext(width + 2) : value.zext(width + 2);
}

std::optional<llvm::APInt> PhysicalLoopControl::count() const {
  if (!lower || !upper || !step) return std::nullopt;
  auto start = coordinate(*lower), stop = coordinate(*upper);
  if (descending ? start.sle(stop) : start.uge(stop)) return llvm::APInt(width + 2, 0);
  auto distance = descending ? start - stop : stop - start;
  auto stride = step->sext(width + 2);
  if (descending) stride = -stride;
  // Quotient/remainder form cannot overflow at SIZE_MAX. In particular, the
  // generic (distance + stride - 1) / stride is invalid in this finite domain.
  return distance.udiv(stride) + llvm::APInt(width + 2, !distance.urem(stride).isZero());
}

bool PhysicalLoopControl::fitsCoordinate(const llvm::APInt &value) const {
  return descending ? value.isSignedIntN(width) : value.isIntN(width);
}

LogicalResult PhysicalLoopControl::verifyFiniteEndpoint(scf::ForOp loop) const {
  auto trips = count();
  if (!trips || trips->isZero()) return success();
  auto endpoint = coordinate(*lower) + *trips * step->sext(width + 2);
  if (!fitsCoordinate(endpoint))
    return loop.emitError("physical loop exit induction wraps its target coordinate range; no finite ordered execution is representable");
  return success();
}
