#include "DenseIssue.h"
#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

Value riscv_internal::completeDenseExtent(IRRewriter &b, Location loc, Value extent, int64_t width) {
  if (auto size = riscv::denseConstantIndex(extent))
    return arith::ConstantIndexOp::create(b, loc, (*size / width) * width);
  if (riscv::denseIndexBounds(extent).upper < uint64_t(width))
    return arith::ConstantIndexOp::create(b, loc, 0);
  Value step = arith::ConstantIndexOp::create(b, loc, width);
  Value quotient = arith::DivUIOp::create(b, loc, extent, step);
  return arith::MulIOp::create(b, loc, quotient, step);
}

FailureOr<SmallVector<Value>> riscv_internal::emitDenseIssues(
    IRRewriter &b, Operation *anchor, Value extent, int64_t width,
    ValueRange initial, DenseIssueBody body, int64_t pipelineDepth, bool includePartial, bool registerPieces,
    bool separatePartial) {
  if (width <= 0 || !extent.getType().isIndex() ||
      (pipelineDepth != 1 && pipelineDepth != 2)) return failure();
  Location loc = anchor->getLoc();
  auto constant = [&](int64_t n) -> Value { return arith::ConstantIndexOp::create(b, loc, n); };
  Value zero = constant(0), step = constant(width);
  SmallVector<Value> carried(initial);
  if (registerPieces) {
    auto count = riscv::denseConstantIndex(extent);
    if (!count || *count <= 0 || *count % width || *count / width > 4 || pipelineDepth != 1)
      return anchor->emitError("register issue pieces require a closed sequence of at most four full carriers"), failure();
    for (int64_t offset = 0; offset < *count; offset += width) {
      auto next = body(constant(offset), step, true, carried);
      if (failed(next) || next->size() != carried.size()) return failure();
      for (auto [before, after] : llvm::zip(carried, *next))
        if (before.getType() != after.getType()) return failure();
      carried = std::move(*next);
    }
    return carried;
  }
  auto issue = [&](Value begin, Value end, bool full, bool capActive = false) -> LogicalResult {
    // A statically empty interval has no reads and no speculative computation.
    auto first = riscv::denseConstantIndex(begin), last = riscv::denseConstantIndex(end);
    if (begin == end || (first && last && *first == *last)) return success();
    auto loop = scf::ForOp::create(b, loc, begin, end, step, carried);
    loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    // Only the main traversal has a next issue. The remainder is at most one
    // iteration and retains the same carry without creating a pipeline buffer.
    if ((full || capActive) && pipelineDepth == 2)
      loop->setAttr("weft.riscv.schedule", schedule(b, 1, pipelineDepth));
    copyOrigin(anchor, loop);
    b.setInsertionPointToStart(loop.getBody());
    Value active = full ? step : Value(arith::SubIOp::create(b, loc, end, loop.getInductionVar()));
    if (!full && capActive) active = arith::MinUIOp::create(b, loc, active, step);
    auto next = body(loop.getInductionVar(), active, full, loop.getRegionIterArgs());
    if (failed(next) || next->size() != carried.size()) return failure();
    for (auto [before, after] : llvm::zip(carried, *next))
      if (before.getType() != after.getType()) return failure();
    if (!carried.empty()) scf::YieldOp::create(b, loc, *next);
    b.setInsertionPointAfter(loop);
    carried.assign(loop.getResults().begin(), loop.getResults().end());
    return success();
  };
  // A runtime extent would add a division, a second loop boundary and duplicate
  // issue code just to remove its per-issue minimum. In short Level blocks that
  // costs more than it saves. Keep one bounded traversal unless the complete
  // interval is known statically (or the caller already owns its scalar tail).
  if (includePartial && !separatePartial && !riscv::denseConstantIndex(extent)) {
    if (failed(issue(zero, extent, width == 1, true))) return failure();
    return carried;
  }
  // With no vector tail, the caller supplies the already-rounded full bound
  // and owns the scalar remainder. This path needs no synthetic domain point.
  Value full = includePartial ? completeDenseExtent(b, loc, extent, width) : extent;
  if (failed(issue(zero, full, true)) ||
      (includePartial && failed(issue(full, extent, width == 1))))
    return failure();
  return carried;
}

std::optional<DenseIssueCost> riscv_internal::denseIssueCost(
    Builder &b, ArrayRef<Operation *> order, ArrayRef<Value> results,
    ArrayRef<Value> exports, riscv::ValueType domain, ArrayRef<int64_t> tile, int64_t axis,
    riscv::TargetAttr target, int64_t accumulatorGroups, int64_t maximumLaneSpan) {
  if (tile.size() != domain.getShape().size() || accumulatorGroups < 0) return std::nullopt;
  auto pieceOf = [&](riscv::ValueType type) {
    return valueTileType(b, type, domain.getAxisIds().asArrayRef(), tile, axis, target, true, maximumLaneSpan);
  };
  DenseMap<Value, unsigned> last;
  DenseMap<Value, int64_t> groups;
  llvm::DenseSet<Value> scalarInputs;
  for (auto [position, operation] : llvm::enumerate(order)) {
    if (operation->getNumResults() != 1) return std::nullopt;
    auto type = dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
    if (!type) return std::nullopt;
    auto piece = pieceOf(type);
    if (!piece) return std::nullopt;
    auto issues = staticProduct(piece.getLayout().getTimeFactors().asArrayRef());
    if (!issues) return std::nullopt;
    // An interleaved issue keeps all of its numeric pieces live. The type's
    // one-issue registerGroups is not the residency of that complete Value.
    groups[operation->getResult(0)] = piece.getLayout().getRegisterGroups() * *issues;
    last.try_emplace(operation->getResult(0), position);
    for (Value input : operation->getOperands()) {
      if (isa<riscv::ValueType>(input.getType())) last[input] = position;
      else if (isa<IntegerType, FloatType, IndexType>(input.getType())) scalarInputs.insert(input);
    }
  }
  for (Value result : results) last[result] = order.size();
  // Exports are written immediately after their producer, not held until the
  // end of the traversal. Their stores overlap the producer's one issue.
  for (Value value : exports)
    if (!groups.count(value)) return std::nullopt;
  DenseIssueCost cost;
  cost.scalarValues = scalarInputs.size();
  int64_t live = accumulatorGroups;
  for (auto [position, operation] : llvm::enumerate(order)) {
    int64_t output = groups.lookup(operation->getResult(0));
    auto representation = pieceOf(cast<riscv::ValueType>(operation->getResult(0).getType()));
    auto coordinates = riscv::RegisterCoordinates::get(representation.getLayout());
    if (!coordinates) return std::nullopt;
    cost.operations += coordinates->parts();
    if (live > std::numeric_limits<int64_t>::max() - output) return std::nullopt;
    // Input/result overlap is a conservative upper bound. A destructive RVV
    // instruction may reuse its last-use input, but the planner need not assume it.
    live += output;
    int64_t temporary = 0;
    if (auto unary = dyn_cast<riscv::UnaryOp>(operation)) {
      auto type = pieceOf(cast<riscv::ValueType>(unary.getResult().getType()));
      auto scratch = type ? riscv::unaryTemporaryGroups(unary, type) : std::nullopt;
      if (!scratch) return std::nullopt;
      temporary = *scratch;
      if (unary.getKind() == "exp" || unary.getKind() == "exp2" || unary.getKind() == "log")
        cost.operations += 24 * coordinates->parts();
    }
    if (auto select = dyn_cast<riscv::SelectOp>(operation)) temporary = std::max<int64_t>(1, output);
    if (auto compare = dyn_cast<riscv::CompareOp>(operation)) temporary = std::max<int64_t>(1, output);
    if (live > std::numeric_limits<int64_t>::max() - temporary) return std::nullopt;
    cost.vectorGroups = std::max(cost.vectorGroups, live + temporary);
    llvm::DenseSet<Value> expired;
    for (Value input : operation->getOperands())
      if (groups.count(input) && last.lookup(input) == position && expired.insert(input).second)
        live -= groups.lookup(input);
    if (last.lookup(operation->getResult(0)) == position) live -= output;
  }
  // Finalization has a scalar seed carrier in addition to the live partials.
  cost.vectorGroups = std::max(cost.vectorGroups, live + (accumulatorGroups ? 1 : 0));
  return cost;
}

Value riscv_internal::denseReductionSeed(IRRewriter &b, Location loc,
    riscv::ValueType accumulator, StringRef kind) {
  double neutral = kind == "max" ? -std::numeric_limits<double>::infinity()
      : kind == "min" ? std::numeric_limits<double>::infinity() : 0.0;
  Type element = accumulator.getElementType();
  TypedAttr identity = element.isInteger(32) ? TypedAttr(b.getIntegerAttr(element, 0))
                                            : TypedAttr(b.getFloatAttr(element, neutral));
  Value scalar = riscv::ConstantOp::create(b, loc, element, identity);
  return riscv::RVVSplatOp::create(b, loc, accumulator, scalar,
      leaf(b, "rvv", "splat", "rvv.splat", "rvv.splat", 0, accumulator.getLayout().getRegisterGroups()));
}

Value riscv_internal::denseReductionStep(IRRewriter &b, Operation *origin,
    Value input, Value accumulator, Value active, StringRef kind) {
  auto type = cast<riscv::ValueType>(accumulator.getType());
  int64_t groups = type.getLayout().getRegisterGroups();
  auto inputType = cast<riscv::ValueType>(input.getType());
  auto length = getConstantIntValue(active);
  const bool full = inputType.getLayout().getValidity() == "full" && length && *length == type.getLayout().getVl();
  StringRef instruction = full ? "rvv.stream-reduce-full-step" : "rvv.stream-reduce-step";
  auto step = riscv::RVVStreamReduceStepOp::create(b, origin->getLoc(), type,
      input, accumulator, active, kind,
      leaf(b, "rvv", "stream-reduce-step", instruction, instruction, 2 * groups, groups));
  copyOrigin(origin, step);
  return step;
}

Value riscv_internal::denseReductionResult(IRRewriter &b, Operation *origin,
    Value accumulator, int64_t axis, StringRef kind, Type resultType) {
  auto type = cast<riscv::ValueType>(accumulator.getType());
  auto result = riscv::RVVStreamFinalizeOp::create(b, origin->getLoc(), resultType ? resultType : type.getElementType(),
      accumulator, axis, kind,
      leaf(b, "rvv", "stream-finalize", "rvv.stream-finalize", "rvv.stream-finalize",
          type.getLayout().getRegisterGroups(), 0, 1));
  copyOrigin(origin, result);
  return result;
}
