#ifndef WEFT_DIALECT_RISCV_IR_MEMORY_H
#define WEFT_DIALECT_RISCV_IR_MEMORY_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <utility>

namespace mlir {
class Operation;
class Block;
}

namespace weft::riscv {
class MemDescType;

// One RVV lane axis whose element stride is an explicit ABI/runtime operand.
// A selected dynamic-stride leaf can test that same operand for unit stride;
// both instruction arms preserve the address, active lanes and access order.
bool hasDynamicDenseStride(MemDescType memory, mlir::Type value);
bool isInnermostDenseAxis(mlir::Value memory, int64_t axis);
std::optional<std::pair<mlir::Value, int64_t>> denseArgumentStride(mlir::Value memory, int64_t axis);
bool hasUnitStrideProof(mlir::Operation *operation, mlir::Value memory, int64_t axis);
bool hasDenseStrideProof(mlir::Operation *operation, mlir::Value memory,
                         int64_t axis, int64_t stride);
// A concrete dense address, rather than a logical field projection whose
// Encoding still owns bit/record addressing. No numerical producer is replayed.
bool isAddressableDenseDescriptor(mlir::Value memory);
// A natural field occupying every byte of each record in a complete,
// contiguous memory view has the same address map as its dense element view.
bool isContiguousRecordField(mlir::Value field);
std::optional<int64_t> contiguousLocalElements(mlir::Value memory);
bool isStorageMetadataUse(mlir::OpOperand &use);

// Storage provenance, not permission to replay a numeric producer. A register
// value has no storage origins; an unresolved addressable value is unknown.
// Argument alias groups apply within one kernel. Private allocations alias by
// SSA identity, including a reused allocation and all control-flow carries.
struct StorageOrigins {
  llvm::SmallVector<mlir::Value, 2> arguments;
  llvm::SmallVector<mlir::Value, 2> allocations;
  bool unknown = false;

  bool isOwnedLocal() const {
    return !unknown && arguments.empty() && !allocations.empty();
  }
  bool isNumeric() const {
    return !unknown && arguments.empty() && allocations.empty();
  }
};

StorageOrigins storageOrigins(mlir::Value value);
bool mayAliasStorage(mlir::Value lhs, mlir::Value rhs);
// A dynamic disjointness proof is valid only inside its dominating true arm.
// It supplements, never replaces, the complete storage provenance query.
bool mayAliasStorageAt(mlir::Value lhs, mlir::Value rhs, mlir::Operation *at);
// Full pinned dense view, with explicit ABI extent/stride metadata. Projections
// remain attached to this storage but cannot themselves assert a full range.
mlir::Value fullDenseArgumentView(mlir::Value storage);
// A pointwise issue can overwrite an input only when its range is disjoint or
// its retained coordinates share an injective, dynamically equal full mapping.
// Different fixed fields of a common record may still belong to one issue.
bool densePointwiseIndependentAt(mlir::Value lhs, mlir::Value rhs, mlir::Operation *at);
bool haveDisjointLocalStorage(mlir::Value lhs, mlir::Value rhs);

// Append only results/region arguments which can alias this operand. These
// edges also define lexical escape checking; numerical results do not borrow
// their input storage simply because their type uses a local carrier.
void appendStorageAliases(mlir::OpOperand &use,
                          llvm::SmallVectorImpl<mlir::Value> &aliases);
bool storageAliasesRemainInScope(mlir::Value value, mlir::Block *scope);

// Read-only is an effect fact, not speculation or motion permission. Unknown
// operations and trapping guards are barriers. Recursive queries understand
// only operations which declare RecursiveMemoryEffects.
bool hasOnlyReadEffects(mlir::Operation *operation, bool recursive = false);
bool mayWriteStorage(mlir::Operation *operation, mlir::Value storage);

// The original read must precede the consumer in its enclosing block. Nested
// loops include writes from earlier iterations; branches inspect only the
// branch leading to the consumer. No query authorizes hoisting a read into a
// path where the original operation would not execute.
bool storageWrittenBetween(mlir::Value storage, mlir::Operation *read,
                           mlir::Operation *consumer);
bool readCrossesWrite(mlir::Operation *read, mlir::Operation *consumer);

// The final program must identify the storage touched by every ordinary
// memory effect. Primitive-private and shape-identity resources have separate
// contracts and deliberately do not expose an addressable operand.
mlir::LogicalResult verifyMemoryEffects(mlir::Operation *operation);

} // namespace weft::riscv

#endif
