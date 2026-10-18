#ifndef WEFT_RISCV_IR_VERIFICATION_H
#define WEFT_RISCV_IR_VERIFICATION_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

namespace weft::riscv::detail {
mlir::LogicalResult verifyLeafOperation(mlir::Operation *operation);
bool exactLeaf(LeafAttr leaf, llvm::StringRef engine, llvm::StringRef family,
               llvm::StringRef instruction, llvm::StringRef mask,
               llvm::StringRef tail);
unsigned elementBitWidth(mlir::Type type);
bool integerSignednessMatches(mlir::Type type, llvm::StringRef signedness);
} // namespace weft::riscv::detail

#endif
