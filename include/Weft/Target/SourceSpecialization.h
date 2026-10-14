#ifndef WEFT_TARGET_SOURCESPECIALIZATION_H
#define WEFT_TARGET_SOURCESPECIALIZATION_H

#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace weft {

/// Bind the declared source-auto domain on a candidate copy of Canonical IR.
/// This pass has no target, layout, resource or physical parameter inputs.
std::unique_ptr<mlir::Pass> createBindSourceParametersPass(
    const llvm::StringMap<int64_t> &bindings = {});

} // namespace weft

#endif // WEFT_TARGET_SOURCESPECIALIZATION_H
