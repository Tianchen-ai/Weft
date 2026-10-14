#ifndef WEFT_TARGET_LOCAL_LIFETIME_H
#define WEFT_TARGET_LOCAL_LIFETIME_H

#include "mlir/IR/Value.h"
namespace mlir { class Operation; class DominanceInfo; }

namespace weft::riscv_internal {
bool isPrivateLocalState(mlir::Value value);

// True only when the old state version has no observer after this boundary.
// The proof follows view aliases and ownership transfers through ordered
// control; sibling if branches are mutually exclusive. The blocking use is
// returned for a diagnostic. This does not grant permission to replay reads.
bool hasExclusiveLocalVersion(mlir::Value value, mlir::Operation *boundary,
                              mlir::DominanceInfo &dominance,
                              mlir::Operation *&unproven);
}
#endif
