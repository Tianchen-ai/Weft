#ifndef WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
#define WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.h.inc"

namespace weft::kernel {
class FunctionControlResource
    : public mlir::SideEffects::Resource::Base<FunctionControlResource> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunctionControlResource)
  llvm::StringRef getName() const final { return "weft.function-control"; }
};

class LocalShapeResource
    : public mlir::SideEffects::Resource::Base<LocalShapeResource> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LocalShapeResource)
  llvm::StringRef getName() const final { return "weft.local-shape-binding"; }
};

bool isUnitAxisProjection(llvm::ArrayRef<int64_t> inputShape,
                          llvm::ArrayRef<int64_t> inputAxes,
                          llvm::ArrayRef<int64_t> resultShape,
                          llvm::ArrayRef<int64_t> resultAxes,
                          llvm::ArrayRef<int64_t> inputOrder);

struct FieldInterleaveSpan {
  int64_t byteOffset;
  int64_t byteCount;
  int64_t chunkBytes;
};

mlir::FailureOr<llvm::SmallVector<FieldInterleaveSpan>> fieldInterleaveSpans(
    int64_t storageBits, llvm::ArrayRef<int64_t> fieldBitOffsets,
    llvm::ArrayRef<int64_t> fieldStorageBits, llvm::ArrayRef<int64_t> padding,
    llvm::ArrayRef<int64_t> fieldChunks);
} // namespace weft::kernel

#endif // WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
