#ifndef WEFT_LIB_TARGET_LOCALMEMORY_H
#define WEFT_LIB_TARGET_LOCALMEMORY_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace weft::riscv_internal {
struct StaticLocalFootprint {
  int64_t bytes;
  bool hasDynamicObjects;
};

// One lexical footprint model shared by physical placement and final resource
// closure. Overrides price a proposed alignment without mutating the program.
// Dynamic payloads remain unknown and are checked by explicit capacity guards.
std::optional<StaticLocalFootprint> staticLocalFootprint(riscv::KernelOp kernel,
    const llvm::DenseMap<mlir::Value, int64_t> &alignments = llvm::DenseMap<mlir::Value, int64_t>());

// An active extent proved from the actual point and its source loop bounds.
std::optional<int64_t> completePointExtent(riscv::PhysicalPointOp point);
// A physical tile of an existing logical domain. Non-lane dimensions become
// register replicas; omitted and broadcast dimensions retain a single value.
riscv::ValueType valueTileType(mlir::Builder &builder, riscv::ValueType type,
    llvm::ArrayRef<int64_t> axes, llvm::ArrayRef<int64_t> shape, int64_t laneAxis,
    riscv::TargetAttr target, bool tail, int64_t maximumLaneSpan = 0);

// The identity cursor must be refreshed after another materializer changes the
// module. Allocation and SSA ownership use this common source of identities.
int64_t lastLocalIdentity(mlir::ModuleOp module);
riscv::ValueType localValueType(mlir::Builder &builder, riscv::ValueType type);
mlir::FailureOr<mlir::Value> allocateLocalValue(
    mlir::IRRewriter &builder, riscv::ValueType type, mlir::Operation *anchor,
    int64_t &identity, bool preserveBirth = false,
    llvm::StringRef schema = "local-value-copy");
// Initialize at the original birth, using bounded vector fills or a copy of
// the existing numeric initializer. This does not create a new source state.
mlir::LogicalResult initializeLocalValue(mlir::IRRewriter &builder,
    mlir::Value destination, mlir::Value initial, mlir::Operation *anchor);

// Project one existing logical domain into a bounded physical issue. Missing
// axes remain scalar broadcasts; all retained axes keep their identity/order.
riscv::ValueType valuePieceType(mlir::Builder &builder, riscv::ValueType type,
                               int64_t laneAxis, int64_t width,
                               riscv::TargetAttr target, bool tail = false);
mlir::FailureOr<mlir::Value> valueMemoryWindow(
    mlir::IRRewriter &builder, mlir::Value memory, riscv::ValueType domain,
    llvm::ArrayRef<mlir::Value> coordinates, riscv::ValueType piece,
    mlir::Operation *anchor);

// Traverse exactly the numerical Value shape. This does not infer an active
// domain from a nearby point: a completed state can outlive that point.
mlir::LogicalResult forEachValuePiece(
    mlir::IRRewriter &builder, riscv::ValueType type, mlir::Operation *anchor,
    llvm::function_ref<mlir::LogicalResult(llvm::ArrayRef<mlir::Value>, int64_t, int64_t)> body,
    std::optional<std::pair<int64_t, int64_t>> issue = std::nullopt,
    llvm::ArrayRef<mlir::Value> extents = {});

// Expose an existing numeric local value. The descriptor borrows that value's
// actual extent operands; it neither allocates nor extends its lifetime.
mlir::FailureOr<mlir::Value> localView(mlir::IRRewriter &builder,
                                    mlir::Value source,
                                    mlir::Operation *anchor);
mlir::Value completeLocalSlice(mlir::IRRewriter &builder, mlir::Value view);

// Full selected register coordinates can read a completed static local value.
// Time may occur on any axis, independently of the lane axis.
bool isCompleteLocalRead(riscv::ConvertLayoutOp conversion);

// Transfer an identical, fully initialized private object to a new numerical
// snapshot when the previous identity is dead. Author stage metadata remains
// on the destination binding; the removed copy does not erase that boundary.
mlir::LogicalResult forwardLocalCopies(mlir::ModuleOp module, mlir::IRRewriter &builder);

} // namespace weft::riscv_internal

#endif
