#ifndef WEFT_TARGET_PIPELINE_SCHEDULE_H
#define WEFT_TARGET_PIPELINE_SCHEDULE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace weft::riscv_internal {

inline constexpr llvm::StringLiteral pipelineStage = "weft.riscv.pipeline_stage";
inline constexpr llvm::StringLiteral pipelineOrder = "weft.riscv.pipeline_order";
inline constexpr llvm::StringLiteral pipelineInvariant = "weft.riscv.pipeline_invariant";

// A query over one physical loop, never a persistent planning IR. Stage and
// total steady-state order are written on the operations themselves. Expansion
// reads those decisions back and validates them before making any copies.
struct PipelineSchedule {
  mlir::scf::ForOp loop;
  llvm::SmallVector<mlir::Operation *> producers;
  llvm::SmallVector<mlir::Operation *> consumers;
  llvm::SmallVector<mlir::Operation *> ordered;
  llvm::SmallVector<mlir::Operation *> invariants;
  llvm::SmallVector<mlir::Value> transported;
  bool transportInduction = false;
  bool descending = false;

  bool isProducer(mlir::Operation *operation) const;
  bool isInvariant(mlir::Operation *operation) const;
  mlir::LogicalResult verify() const;
  void write() const;
  static mlir::FailureOr<PipelineSchedule> read(mlir::scf::ForOp loop);
};

// No split is a normal absence of an eligible producer/consumer cluster.
// Invalid effects, bindings or dependency contracts are diagnostics, not an
// invitation to retry another implementation of the source program.
mlir::FailureOr<std::optional<PipelineSchedule>> schedulePipeline(mlir::scf::ForOp loop);

struct PipelineExpansion {
  mlir::scf::IfOp guard;
  mlir::scf::ForOp steady;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> steadyOperations;
};

mlir::FailureOr<PipelineExpansion> expandPipeline(
    const PipelineSchedule &schedule, mlir::IRRewriter &rewriter,
    int64_t &shapeIdentity);

// Reorders the two already selected stage chains on a disposable expanded
// physical loop. The returned order refers to the original scheduled ops.
// Both stages retain their own dependency/effect order; final residency uses
// the same region-aware analysis as ordinary resource closure.
mlir::FailureOr<llvm::SmallVector<mlir::Operation *>> orderPipelineFrontiers(
    const PipelineSchedule &schedule, const PipelineExpansion &expansion,
    int64_t registerBudget);

// The same lifetime frontier also orders an explicitly unrolled read-only
// issue window. Priorities are the selected operation/iteration order; actual
// SSA and local-shape dependencies remain hard constraints.
mlir::FailureOr<llvm::SmallVector<mlir::Operation *>> orderReadOnlyIssues(
    mlir::Operation *owner, mlir::Block &body,
    llvm::ArrayRef<mlir::Operation *> operations,
    llvm::ArrayRef<std::pair<int64_t, int64_t>> priorities,
    int64_t registerBudget);

} // namespace weft::riscv_internal
#endif
