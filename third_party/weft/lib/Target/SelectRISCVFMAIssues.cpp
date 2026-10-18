#include "RISCVPhysicalSupport.h"
#include "Weft/Dialect/RISCV/IR/FMAIssue.h"
#include "Weft/Dialect/RISCV/IR/Stream.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
struct Candidate {
  Operation *operation;
  Value vector, multiplier, accumulator;
  riscv::ValueType result;
  riscv::FMAIssueGeometry geometry;
  SmallVector<int64_t> groupEnds;
  bool full = false;
  Value active;
  SmallVector<Value> extents;
  std::optional<riscv::VectorTileGeometry> tile;

  static std::optional<Candidate> get(Operation *operation) {
    Candidate candidate;
    candidate.operation = operation;
    if (auto tile = dyn_cast<riscv::RVVTileContractOp>(operation)) {
      bool rhs = tile.getLaneOperand() == "rhs";
      candidate.vector = rhs ? tile.getRhs() : tile.getLhs();
      candidate.multiplier = rhs ? tile.getLhs() : tile.getRhs();
      candidate.accumulator = tile.getAccumulator();
      candidate.result = tile.getResult().getType();
      candidate.full = tile.getLeaf().getInstruction() == "rvv.tile-vfmacc-full.vf" ||
                       tile.getLeaf().getInstruction() == "rvv.tile-vfwmacc-full.vf";
      candidate.extents.assign(tile.getExtents().begin(), tile.getExtents().end());
      candidate.tile = riscv::VectorTileGeometry::get(candidate.result, tile.getAxis());
      if (!candidate.tile) return std::nullopt;
    } else if (auto stream = dyn_cast<riscv::RVVStreamContractStepOp>(operation)) {
      candidate.vector = stream.getLhs();
      candidate.multiplier = stream.getRhs();
      candidate.accumulator = stream.getAccumulator();
      candidate.result = stream.getResult().getType();
      candidate.active = stream.getActive();
      candidate.full = stream.getLeaf().getInstruction() == "rvv.stream-contract-full-step" ||
                       stream.getLeaf().getInstruction() == "rvv.stream-widen-contract-full-step";
    } else if (auto step = dyn_cast<riscv::RVVContractStepOp>(operation)) {
      bool rhs = step.getLaneOperand() == "rhs";
      candidate.vector = rhs ? step.getRhs() : step.getLhs();
      candidate.multiplier = rhs ? step.getLhs() : step.getRhs();
      candidate.accumulator = step.getAccumulator();
      candidate.result = step.getResult().getType();
      candidate.active = step.getActive();
      // A non-stream step without explicit VL may depend on several physical
      // points. Leave it to its existing exact leaf until those bounds are IR.
      if (!candidate.active) return std::nullopt;
    } else return std::nullopt;
    auto vector = dyn_cast<riscv::ValueType>(candidate.vector.getType());
    if (!vector) return std::nullopt;
    auto geometry = riscv::FMAIssueGeometry::get(vector, candidate.multiplier.getType(), candidate.result);
    if (!geometry) return std::nullopt;
    candidate.geometry = std::move(*geometry);
    if (candidate.tile && candidate.tile->parts() != candidate.geometry.parts()) return std::nullopt;
    auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
    auto groups = riscv::selectFMAIssueGroups(candidate.geometry, target.getVectorRegisters());
    if (!groups) return std::nullopt;
    candidate.groupEnds = std::move(*groups);
    return candidate;
  }

  bool requiresClosedIssue() const {
    int64_t supplies = geometry.scalar || vector == multiplier ? 1 : 2;
    int64_t resident = issueRegisterGroups(accumulator) + supplies * geometry.inputGroups;
    // A replica's input can die after its own consumers: do not charge every
    // replica load as simultaneously resident. Retain all independent carries,
    // the current instruction's supply and one look-ahead supply. Closing a
    // wider tuple unnecessarily would inhibit otherwise legal overlap.
    auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
    if (resident + geometry.inputGroups <= target.getVectorRegisters()) return false;
    // A single update in a loop with downstream unrolling disabled does not
    // expose a second load to the scheduler. Look for the actual next supply
    // in this block, not a hypothetical future iteration or a source name.
    auto follows = [](const Candidate &first, const Candidate &second) {
      if (first.operation->getBlock() != second.operation->getBlock() ||
          second.accumulator != first.operation->getResult(0) ||
          first.vector == second.vector) return false;
      Operation *supply = second.vector.getDefiningOp();
      return supply && supply->getBlock() == first.operation->getBlock() &&
             first.operation->isBeforeInBlock(supply) && supply->isBeforeInBlock(second.operation);
    };
    for (Operation *user : operation->getResult(0).getUsers())
      if (auto next = Candidate::get(user); next && follows(*this, *next)) return true;
    if (auto *producer = accumulator.getDefiningOp())
      if (auto previous = Candidate::get(producer); previous && follows(*previous, *this)) return true;
    return false;
  }

  Value length(IRRewriter &b, int64_t part) const {
    auto loc = operation->getLoc();
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    if (full) return constant(geometry.lanes);
    if (active) return b.createOrFold<arith::MinUIOp>(loc, active, constant(geometry.lanes));
    Value valid;
    Value length;
    for (auto [axis, coordinate] : llvm::enumerate(tile->starts[part])) {
      Value index = constant(coordinate);
      Value present = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, extents[axis], index);
      if (axis == tile->lanePosition) {
        Value remaining = b.createOrFold<arith::SubIOp>(loc, extents[axis], index);
        Value limited = b.createOrFold<arith::MinUIOp>(loc, remaining, constant(geometry.lanes));
        length = b.createOrFold<arith::SelectOp>(loc, present, limited, constant(0));
      } else valid = valid ? b.createOrFold<arith::AndIOp>(loc, valid, present) : present;
    }
    return valid ? b.createOrFold<arith::SelectOp>(loc, valid, length, constant(0)) : length;
  }

  void materialize(IRRewriter &b) const {
    b.setInsertionPoint(operation);
    if (auto loop = operation->getParentOfType<scf::ForOp>())
      loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    SmallVector<Value> lengths;
    for (int64_t part = 0; part < geometry.parts(); ++part) lengths.push_back(length(b, part));
    StringRef policy = riscv::canReuseFMAAccumulator(operation, accumulator) ? "reuse" : "copy";
    int64_t copies = policy == "reuse" ? 0 : result.getLayout().getRegisterGroups();
    auto instruction = geometry.instruction();
    auto selected = riscv::RVVFMAIssueOp::create(b, operation->getLoc(), result,
        vector, multiplier, accumulator, lengths, b.getDenseI64ArrayAttr(groupEnds),
        full ? "agnostic" : "undisturbed", policy,
        leaf(b, "rvv", "fma-issue", instruction, instruction,
             issueRegisterGroups(vector) + issueRegisterGroups(multiplier) + issueRegisterGroups(accumulator),
             result.getLayout().getRegisterGroups(), copies, 0, "none", full ? "agnostic" : "undisturbed"));
    copyOrigin(operation, selected);
    b.replaceOp(operation, selected.getResult());
  }
};

class SelectRISCVFMAIssuesPass
    : public PassWrapper<SelectRISCVFMAIssuesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVFMAIssuesPass)
  SelectRISCVFMAIssuesPass() = default;
  SelectRISCVFMAIssuesPass(const SelectRISCVFMAIssuesPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "weft-riscv-select-fma-issues"; }
  StringRef getDescription() const override {
    return "Close saturated numeric FMA issue frontiers with explicit tied-register contracts";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<riscv::WEFTRISCVDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    SmallVector<Operation *> operations;
    getOperation().walk([&](Operation *operation) {
      if (isa<riscv::RVVTileContractOp, riscv::RVVStreamContractStepOp, riscv::RVVContractStepOp>(operation))
        operations.push_back(operation);
    });
    SmallVector<Operation *> selectedOperations;
    for (Operation *operation : operations) {
      auto candidate = Candidate::get(operation);
      if (!candidate || !candidate->requiresClosedIssue()) continue;
      selectedOperations.push_back(operation);
    }
    for (Operation *operation : selectedOperations) {
      // Earlier replacements update the later accumulator uses. Query those
      // current SSA operands instead of retaining a pre-rewrite Value handle.
      auto candidate = Candidate::get(operation);
      if (!candidate) {
        operation->emitError("selected FMA issue lost its numeric contract during materialization");
        signalPassFailure();
        return;
      }
      if (operation->getParentOfType<riscv::KernelOp>().getResourcesMaterialized()) {
        operation->emitError("FMA issue selection requires an open physical resource boundary");
        signalPassFailure();
        return;
      }
      parts += candidate->geometry.parts();
      groups += candidate->groupEnds.size();
      candidate->materialize(b);
      ++selected;
    }
  }
private:
  Statistic selected{this, "weft-fma-issues", "Closed numeric FMA issues"};
  Statistic parts{this, "weft-fma-parts", "Independent lane-wise FMA updates in closed issues"};
  Statistic groups{this, "weft-fma-groups", "Bounded typed assembly groups"};
};
} // namespace

std::unique_ptr<Pass> weft::createSelectRISCVFMAIssuesPass() {
  return std::make_unique<SelectRISCVFMAIssuesPass>();
}
