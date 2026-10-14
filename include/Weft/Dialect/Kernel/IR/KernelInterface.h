#ifndef WEFT_DIALECT_KERNEL_IR_KERNEL_INTERFACE_H
#define WEFT_DIALECT_KERNEL_IR_KERNEL_INTERFACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace weft::kernel {

// Read one verified canonical kernel's public arguments and finite source-auto
// domains. No target profile, physical pass or executable code is constructed.
// This is the interface shared by external canonical producers and the runtime;
// it is not a substitute for the target compiler's final typed native ABI.
mlir::FailureOr<std::string> describeKernelInterface(mlir::ModuleOp module);

} // namespace weft::kernel
#endif
