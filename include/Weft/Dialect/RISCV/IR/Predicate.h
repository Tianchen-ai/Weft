#ifndef WEFT_DIALECT_RISCV_IR_PREDICATE_H
#define WEFT_DIALECT_RISCV_IR_PREDICATE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include <optional>

namespace weft::riscv {
// These selected local producers carry packed RVV masks. Ordinary i1 values
// still have byte-register/storage representation; the two are never confused
// by guessing from element width or by changing the public Boolean ABI.
bool isPredicateRegister(mlir::Value value);
bool uniformPredicate(mlir::Type type);
bool predicateGeometry(ValueType predicate, ValueType data);
bool predicateDataType(ValueType type);
std::optional<int64_t> predicatePartCount(ValueType type);
std::optional<int64_t> predicateIssueGroups(ValueType type);
llvm::StringRef predicateTail(ValueType type);
} // namespace weft::riscv
#endif
