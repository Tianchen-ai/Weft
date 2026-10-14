#ifndef WEFT_DIALECT_RISCV_IR_LOCAL_TRANSFER_H
#define WEFT_DIALECT_RISCV_IR_LOCAL_TRANSFER_H
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
namespace weft::riscv {
class LocalBindOp;
// A complete contiguous private domain, with separate source/destination
// objects. The byte extent follows the actual descriptor storage width.
std::optional<int64_t> localTransferBytes(mlir::Value source, mlir::Value destination);
mlir::LogicalResult verifyLocalBindingStorage(LocalBindOp binding);
}
#endif
