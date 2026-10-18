#include "Weft/Target/RISCVPasses.h"
#include "PipelineSchedule.h"
#include "PipelineShapeBindings.h"
#include "PhysicalLoopControl.h"
#include "ValueLiveness.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <memory>
#include <limits>
#include <tuple>

using namespace weft;

namespace {

constexpr llvm::StringLiteral kIssueOrderAttr =
    "weft.riscv.issue_operation_order";
constexpr llvm::StringLiteral kIssueIterationAttr =
    "weft.riscv.issue_iteration";

bool isReadOrPure(mlir::Operation *operation) {
  return riscv::hasOnlyReadEffects(operation);
}

mlir::LogicalResult scheduleOperationMajor(
    mlir::Operation *diagnosticOwner, mlir::Block *body,
    llvm::ArrayRef<mlir::Operation *> operations,
    mlir::Operation *insertionAnchor) {
  for (mlir::Operation *operation : operations) {
    if (operation->getNumRegions() != 0 ||
        (!riscv_internal::isPipelineShapeBinding(operation) && !isReadOrPure(operation)))
      return diagnosticOwner->emitError(
          "operation-major issue unroll requires one flat read-only or pure body");
  }
  if (operations.empty())
    return diagnosticOwner->emitError(
        "operation-major issue unroll produced no schedulable operations");

  llvm::SmallVector<std::pair<int64_t, int64_t>> priorities;
  for (mlir::Operation *operation : operations) {
    auto order = operation->getAttrOfType<mlir::IntegerAttr>(kIssueOrderAttr);
    auto iteration =
        operation->getAttrOfType<mlir::IntegerAttr>(kIssueIterationAttr);
    // The SCF unroller may insert induction-offset arithmetic that is not a
    // clone of a source body operation.  It has no side effects and must be
    // available before the first cloned operation that consumes it.
    const int64_t operationOrder = order ? order.getInt() : -1;
    const int64_t issueIteration = iteration ? iteration.getInt() : -1;
    priorities.emplace_back(operationOrder, issueIteration);
  }
  auto kernel = operations.front()->getParentOfType<riscv::KernelOp>();
  auto outside = riscv_internal::surroundingRegisterGroups(operations.front());
  if (!kernel || !outside)
    return diagnosticOwner->emitError("unrolled issues have no bounded target/surrounding register contract");
  // The common frontier counts values already live in this block. Only retain
  // captures from enclosing regions in the external reserve.
  auto local = riscv_internal::BlockRegisterLiveness::get(*body);
  auto first = local ? local->position(operations.front()) : std::nullopt;
  if (!first) return diagnosticOwner->emitError("unrolled issue window has no physical liveness position");
  int64_t localBefore = 0;
  for (const auto &interval : local->intervals())
    if ((mlir::isa<mlir::BlockArgument>(interval.value) || interval.begin < *first) && interval.end >= *first)
      localBefore += interval.groups;
  int64_t reserve = std::max<int64_t>(0, *outside - localBefore);
  auto scheduled = riscv_internal::orderReadOnlyIssues(operations.front(), *body, operations, priorities,
      std::max<int64_t>(1, kernel.getTarget().getVectorRegisters() - reserve));
  if (mlir::failed(scheduled)) return mlir::failure();
  for (mlir::Operation *operation : *scheduled) {
    operation->removeAttr(kIssueOrderAttr);
    operation->removeAttr(kIssueIterationAttr);
    operation->moveBefore(insertionAnchor);
  }
  return mlir::success();
}

class SelectedUnroller {
public:
  SelectedUnroller(mlir::scf::ForOp loop, const riscv_internal::PhysicalLoopControl &space,
                   unsigned factor, bool operationMajor, int64_t &shapeIdentity)
      : loop(loop), space(space), factor(factor), operationMajor(operationMajor),
        shapeIdentity(shapeIdentity), b(loop.getContext()), loc(loop.getLoc()) {}

  mlir::LogicalResult run() {
    if (operationMajor)
      for (mlir::Operation &operation : loop.getBody()->without_terminator())
        if (operation.getNumRegions() ||
            (!riscv_internal::isPipelineShapeBinding(&operation) && !isReadOrPure(&operation)))
          return operation.emitError("operation-major unroll requires a flat read-only issue body");
    auto count = space.count();
    if (count && count->isZero()) {
      b.replaceOp(loop, loop.getInitArgs());
      return mlir::success();
    }
    if (count && count->ule(factor)) return fullyUnroll(count->getZExtValue());
    if (count) {
      auto groupedStep = space.step->sext(space.width + 2) * llvm::APInt(space.width + 2, factor);
      auto groups = count->udiv(llvm::APInt(space.width + 2, factor));
      auto endpoint = space.coordinate(*space.lower) + groups * groupedStep;
      // Retain ordinary coordinate induction whenever its grouped step and end
      // fit. This keeps downstream affine memory proofs and pointer recurrences.
      if (groupedStep.isSignedIntN(space.width) && space.fitsCoordinate(endpoint))
        return staticGroups(groupedStep, endpoint, count->urem(llvm::APInt(space.width + 2, factor)).getZExtValue());
    }
    return countedGroups();
  }

private:
  mlir::Value index(const llvm::APInt &value) {
    return mlir::arith::ConstantOp::create(b, loc, b.getIndexType(),
        b.getIndexAttr(value.trunc(space.width).getSExtValue()));
  }
  mlir::Value index(int64_t value) { return mlir::arith::ConstantIndexOp::create(b, loc, value); }

  void attributes(mlir::Operation *target, bool keepLevel, bool counted = false) {
    for (mlir::NamedAttribute attribute : loop->getAttrs()) {
      auto name = attribute.getName();
      if (name == "weft.riscv.unroll_factor" || name == "weft.riscv.unroll_order" ||
          name == "weft.riscv.direction" || (!keepLevel && name == "weft.riscv.level")) continue;
      target->setAttr(name, attribute.getValue());
    }
    if (mlir::isa<mlir::scf::ForOp>(target))
      target->setAttr("weft.riscv.direction", b.getStringAttr(counted ? "ascending" : space.descending ? "descending" : "ascending"));
  }

  llvm::SmallVector<mlir::Value> iteration(mlir::Value coordinate,
      mlir::ValueRange incoming, unsigned instance, bool reorder) {
    mlir::IRMapping mapping;
    mapping.map(loop.getInductionVar(), coordinate);
    for (auto [argument, initial] : llvm::zip_equal(loop.getRegionIterArgs(), incoming)) mapping.map(argument, initial);
    llvm::DenseMap<mlir::Value, int64_t> shapes;
    unsigned order = 0;
    for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
      auto clone = riscv_internal::clonePipelineOperation(b, &operation, mapping, shapes, shapeIdentity);
      if (reorder) {
        clone->setAttr(kIssueOrderAttr, b.getI64IntegerAttr(order));
        clone->setAttr(kIssueIterationAttr, b.getI64IntegerAttr(instance));
      }
      ++order;
    }
    llvm::SmallVector<mlir::Value> outgoing;
    for (mlir::Value value : loop.getBody()->getTerminator()->getOperands()) outgoing.push_back(mapping.lookupOrDefault(value));
    return outgoing;
  }

  // Include offset arithmetic in the same contiguous window as its cloned
  // issue. Scheduling sees its SSA dependencies; no marker scan can accidentally
  // capture an adjacent unrolled loop or omit an induction definition.
  mlir::LogicalResult reorder(mlir::Operation *previous, mlir::Operation *anchor) {
    if (!operationMajor) return mlir::success();
    auto block = anchor->getBlock();
    auto first = previous ? previous->getNextNode() : &block->front();
    llvm::SmallVector<mlir::Operation *> operations;
    for (auto operation = first; operation != anchor; operation = operation->getNextNode()) operations.push_back(operation);
    if (operations.empty()) return mlir::success();
    return scheduleOperationMajor(anchor, block, operations, anchor);
  }

  mlir::LogicalResult fullyUnroll(unsigned count) {
    b.setInsertionPoint(loop);
    auto previous = loop->getPrevNode();
    llvm::SmallVector<mlir::Value> carried(loop.getInitArgs());
    mlir::Value coordinate = loop.getLowerBound();
    for (unsigned instance = 0; instance < count; ++instance) {
      carried = iteration(coordinate, carried, instance, operationMajor);
      if (instance + 1 < count) coordinate = mlir::arith::AddIOp::create(b, loc, coordinate, loop.getStep());
    }
    if (mlir::failed(reorder(previous, loop))) return mlir::failure();
    b.replaceOp(loop, carried);
    return mlir::success();
  }

  mlir::LogicalResult fillGroup(mlir::scf::ForOp main, bool coordinateCarry) {
    if (!main.getBody()->empty()) b.eraseOp(main.getBody()->getTerminator());
    b.setInsertionPointToStart(main.getBody());
    auto arguments = main.getRegionIterArgs();
    mlir::Value coordinate = coordinateCarry ? arguments.back() : main.getInductionVar();
    llvm::SmallVector<mlir::Value> carried(coordinateCarry ? arguments.drop_back() : arguments);
    for (unsigned instance = 0; instance < factor; ++instance) {
      carried = iteration(coordinate, carried, instance, operationMajor);
      if (coordinateCarry || instance + 1 < factor)
        coordinate = mlir::arith::AddIOp::create(b, loc, coordinate, loop.getStep());
    }
    if (coordinateCarry) carried.push_back(coordinate);
    auto yield = mlir::scf::YieldOp::create(b, loc, carried);
    return reorder(nullptr, yield);
  }

  llvm::SmallVector<mlir::Value> tail(mlir::Value start, mlir::ValueRange initial, bool keepLevel) {
    auto cleanup = mlir::scf::ForOp::create(b, loc, start, loop.getUpperBound(), loop.getStep(), initial);
    attributes(cleanup, keepLevel);
    if (!cleanup.getBody()->empty()) b.eraseOp(cleanup.getBody()->getTerminator());
    b.setInsertionPointToStart(cleanup.getBody());
    auto outgoing = iteration(cleanup.getInductionVar(), cleanup.getRegionIterArgs(), 0, false);
    mlir::scf::YieldOp::create(b, loc, outgoing);
    b.setInsertionPointAfter(cleanup);
    return llvm::SmallVector<mlir::Value>(cleanup.getResults());
  }

  mlir::LogicalResult staticGroups(const llvm::APInt &step,
                                  const llvm::APInt &endpoint, unsigned remainder) {
    b.setInsertionPoint(loop);
    auto end = index(endpoint), stride = index(step);
    auto main = mlir::scf::ForOp::create(b, loc, loop.getLowerBound(), end, stride, loop.getInitArgs());
    attributes(main, true);
    if (mlir::failed(fillGroup(main, false))) return mlir::failure();
    b.setInsertionPointAfter(main);
    llvm::SmallVector<mlir::Value> results(main.getResults());
    if (remainder) results = tail(end, results, true);
    b.replaceOp(loop, results);
    return mlir::success();
  }

  mlir::LogicalResult countedGroups() {
    b.setInsertionPoint(loop);
    auto predicate = space.descending ? mlir::arith::CmpIPredicate::sgt : mlir::arith::CmpIPredicate::ult;
    auto nonempty = mlir::arith::CmpIOp::create(b, loc, predicate, loop.getLowerBound(), loop.getUpperBound());
    auto guard = mlir::scf::IfOp::create(b, loc, loop.getResultTypes(), nonempty, true);
    attributes(guard, true);
    // The zero-result builder supplies empty yields; nonzero-result regions are
    // initially empty. Both forms receive the explicit handoff below.
    for (mlir::Region &region : guard->getRegions())
      if (!region.front().empty()) b.eraseOp(region.front().getTerminator());
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    auto zero = index(0), one = index(1), width = index(factor);
    auto distance = mlir::arith::SubIOp::create(b, loc,
        space.descending ? loop.getLowerBound() : loop.getUpperBound(),
        space.descending ? loop.getUpperBound() : loop.getLowerBound());
    mlir::Value magnitude = loop.getStep();
    if (space.descending) magnitude = mlir::arith::SubIOp::create(b, loc, zero, magnitude);
    auto count = mlir::arith::CeilDivUIOp::create(b, loc, distance, magnitude);
    auto groups = mlir::arith::DivUIOp::create(b, loc, count, width);
    llvm::SmallVector<mlir::Value> initial(loop.getInitArgs());
    initial.push_back(loop.getLowerBound());
    auto main = mlir::scf::ForOp::create(b, loc, zero, groups, one, initial);
    attributes(main, false, true);
    if (mlir::failed(fillGroup(main, true))) return mlir::failure();
    b.setInsertionPointAfter(main);
    auto outgoing = tail(main.getResults().back(), main.getResults().drop_back(), false);
    mlir::scf::YieldOp::create(b, loc, outgoing);
    b.setInsertionPointToStart(&guard.getElseRegion().front());
    mlir::scf::YieldOp::create(b, loc, loop.getInitArgs());
    b.replaceOp(loop, guard.getResults());
    return mlir::success();
  }

  mlir::scf::ForOp loop;
  riscv_internal::PhysicalLoopControl space;
  unsigned factor;
  bool operationMajor;
  int64_t &shapeIdentity;
  mlir::IRRewriter b;
  mlir::Location loc;
};

class UnrollRISCVLevelsPass
    : public mlir::PassWrapper<UnrollRISCVLevelsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrollRISCVLevelsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-unroll-levels";
  }
  llvm::StringRef getDescription() const override {
    return "Instantiate selected innermost Level unroll factors";
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>([&](mlir::scf::ForOp loop) {
      if (auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
              "weft.riscv.unroll_factor");
          factor && factor.getInt() > 1)
        loops.push_back(loop);
    });

    bool failed = false;
    bool changed = false;
    for (mlir::scf::ForOp loop : loops) {
      auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
          "weft.riscv.unroll_factor");
      auto order = loop->getAttrOfType<mlir::StringAttr>(
          "weft.riscv.unroll_order");
      const bool operationMajor = order && order.getValue() == "operation-major";
      if (order && !operationMajor) {
        loop.emitError("selected issue unroll has an unknown operation order");
        failed = true;
        continue;
      }
      if (factor.getInt() > 64) {
        loop.emitError("selected unroll exceeds the bounded physical issue factor");
        failed = true;
        continue;
      }
      auto space = riscv_internal::PhysicalLoopControl::get(loop);
      if (mlir::failed(space)) { failed = true; continue; }
      if (mlir::failed(space->verifyFiniteEndpoint(loop))) { failed = true; continue; }
      int64_t shapeIdentity = riscv_internal::nextPipelineShapeIdentity(
          loop->getParentOfType<riscv::KernelOp>());
      if (mlir::failed(SelectedUnroller(loop, *space, factor.getInt(), operationMajor, shapeIdentity).run())) {
        signalPassFailure();
        return;
      }
      changed = true;
    }
    // Cloning an iteration preserves operation attributes verbatim. A
    // partial birth is a target-local SSA lifetime identity, so cloned partial
    // definitions in one owner domain must receive fresh identities just as
    // cloned SSA definitions receive fresh results.
    if (changed && !failed) {
      llvm::DenseMap<int64_t, int64_t> maximumBirth;
      getOperation().walk([&](mlir::Operation *operation) {
        if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
          maximumBirth[partial.getOwnerDomainId()] =
              std::max(maximumBirth.lookup(partial.getOwnerDomainId()),
                       static_cast<int64_t>(partial.getBirthId()));
        if (auto capture =
                mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
          maximumBirth[capture.getOwnerDomainId()] =
              std::max(maximumBirth.lookup(capture.getOwnerDomainId()),
                       static_cast<int64_t>(capture.getBirthId()));
        if (auto collect =
                mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
          maximumBirth[collect.getOwnerDomainId()] =
              std::max(maximumBirth.lookup(collect.getOwnerDomainId()),
                       static_cast<int64_t>(collect.getBirthId()));
      });
      llvm::DenseSet<std::pair<int64_t, int64_t>> seen;
      mlir::Builder builder(&getContext());
      auto freshen = [&](mlir::Operation *operation, int64_t owner,
                         int64_t birth) {
        if (seen.insert({owner, birth}).second)
          return;
        if (maximumBirth.lookup(owner) == std::numeric_limits<int64_t>::max()) {
          operation->emitError("unrolled partial birth exceeds its owner identity domain");
          failed = true;
          return;
        }
        const int64_t replacement = ++maximumBirth[owner];
        operation->setAttr("birth_id",
                           builder.getI64IntegerAttr(replacement));
        seen.insert({owner, replacement});
      };
      getOperation().walk([&](mlir::Operation *operation) {
        if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
          freshen(operation, partial.getOwnerDomainId(), partial.getBirthId());
        if (auto capture =
                mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
          freshen(operation, capture.getOwnerDomainId(), capture.getBirthId());
        if (auto collect =
                mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
          freshen(operation, collect.getOwnerDomainId(), collect.getBirthId());
      });
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createUnrollRISCVLevelsPass() {
  return std::make_unique<UnrollRISCVLevelsPass>();
}
