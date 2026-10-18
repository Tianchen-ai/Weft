#ifndef WEFT_TARGET_DENSE_PRODUCER_GRAPH_H
#define WEFT_TARGET_DENSE_PRODUCER_GRAPH_H

#include "DenseIssue.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace weft::riscv_internal {

struct DenseAliasRequirement {
  mlir::Value lhs, rhs;
  bool allowEqualMapping;
};

// A bounded query of the current use-def graph. It is consumed immediately by
// the physical rewrite; none of these records survive as an alternate program.
class DenseProducerGraph {
public:
  explicit DenseProducerGraph(llvm::ArrayRef<mlir::Operation *> sinks);
  bool analyze(bool required = false, bool allowAliasGuards = false);
  bool needsRealization() const;
  bool ownsConsumer(mlir::Operation *operation) const;
  bool canStreamRead(riscv::LoadOp read) const;

  mlir::Operation *root;
  mlir::Value result;
  riscv::ValueType domain;
  llvm::SmallVector<mlir::Operation *> order;
  llvm::SmallPtrSet<mlir::Operation *, 32> members;
  llvm::SmallVector<riscv::LoadOp> reads;
  llvm::SmallVector<riscv::ConvertLayoutOp> localReads;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> exports;
  llvm::SmallVector<riscv::ReduceOp> reductions;
  llvm::SmallVector<riscv::StoreOp> stores;
  llvm::SmallVector<mlir::Operation *> sinks;
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<riscv::PhysicalPointOp> points;
  riscv::PhysicalPointOp point;
  int64_t lanePosition = 0;
  int64_t width = 0;
  int64_t laneWidth = 0;
  int64_t issueCopies = 1;
  llvm::SmallVector<int64_t> tileShape;
  DenseIssueCost issueCost;
  llvm::StringRef failureReason;
  llvm::SmallVector<DenseAliasRequirement> aliasRequirements;

private:
  bool projects(riscv::ValueType type) const;
  bool collect(mlir::Value value);
  bool collectConsumers();
  bool canDelayStores();
  bool preservesOrder() const;
  bool chooseIssue();
  bool boundary(mlir::Value value);
  bool allowAliasGuards = false;
  bool requireDisjoint(mlir::Value lhs, mlir::Value rhs);
};

bool isDensePointwise(mlir::Operation *operation);

} // namespace weft::riscv_internal
#endif
