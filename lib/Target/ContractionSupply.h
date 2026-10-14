#ifndef WEFT_LIB_TARGET_CONTRACTIONSUPPLY_H
#define WEFT_LIB_TARGET_CONTRACTIONSUPPLY_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::scf { class ForOp; }

namespace weft::riscv_internal {

// A query of the current SSA snapshot, not permission to move its read. Origin
// remains provenance; explicit numeric materialization cuts the reload path.
struct ContractionSupply {
  mlir::Value value;
  mlir::Value frontier;
  riscv::LoadOp origin;
  riscv::LoadOp read;
  riscv::AccessAttr access;
  // Exact floating promotions, spelled by either CastOp or WidenOp.
  llvm::SmallVector<mlir::Operation *> wideningCasts;
  llvm::SmallVector<int64_t> valueToReadAxes;
  llvm::SmallVector<int64_t> contiguousAxes;
  // Dense unit-stride axes at this frozen value's actual birth. These are
  // layout facts only; they never grant permission to reload its origin.
  llvm::SmallVector<int64_t> birthUnitAxes;
  bool frozen = false;
  bool memorySource = false;

  bool hasMemorySupply() const { return !frozen && memorySource; }
  std::optional<int64_t> readAxis(int64_t valueAxis) const;
  // Existing stream consumers accept the original, unchanged descriptor.
  riscv::LoadOp directRead(mlir::Operation *consumer) const;
};

ContractionSupply queryContractionSupply(mlir::Value value);
// A stable complete dense read, transporting only a one-to-one axis rename.
// Empty success means that this numeric value cannot be supplied by re-reading.
mlir::FailureOr<mlir::Value> materializeContractionReadView(
    mlir::IRRewriter &rewriter, mlir::Value value, mlir::Operation *consumer);
// The original typed storage for an exact widening supply. Consumers must
// explicitly reconstruct its numeric conversion before using the loaded value.
mlir::FailureOr<mlir::Value> materializeContractionStorageView(
    mlir::IRRewriter &rewriter, mlir::Value value, mlir::Operation *consumer);
riscv::ValueType projectedReductionType(mlir::Builder &builder,
                                      riscv::ValueType input,
                                      int64_t reductionAxis);
// Project explicit logical coordinates from a stable dense read. Remaining
// axes, storage dtype and exact promotions are checked against projectedType.
// Empty success retains the existing numeric SSA and requires a numeric
// projection; failure means a malformed/unsupported requested representation.
mlir::FailureOr<mlir::Value> projectContractionRead(
    mlir::IRRewriter &rewriter, mlir::Value value,
    mlir::Type projectedType,
    llvm::ArrayRef<std::pair<int64_t, mlir::Value>> coordinates,
    mlir::Operation *consumer);

// Instantiate the bounded parts of an already frozen numeric K window. Other
// reduction loops retain their existing issue organization.
mlir::LogicalResult specializeContractionParts(mlir::scf::ForOp loop);

} // namespace weft::riscv_internal
#endif
