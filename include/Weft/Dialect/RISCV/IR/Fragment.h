#ifndef WEFT_DIALECT_RISCV_IR_FRAGMENT_H
#define WEFT_DIALECT_RISCV_IR_FRAGMENT_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVFragment.h"

namespace weft::riscv {
FragmentCapabilityAttr fragmentCapabilityAttr(
    mlir::MLIRContext *context, const RISCVFragmentCapability &capability);
bool fragmentAccumulatorTypeMatches(FragmentCapabilityAttr capability,
                                     mlir::Type element);
int64_t fragmentMMAAdditionalGroups(FragmentCapabilityAttr capability, int64_t atoms = 1,
                                    bool reuseAccumulator = false);
bool canReuseFragmentAccumulator(IMEFragmentMMAOp operation);
bool canReuseFragmentPointwise(IMEFragmentPointwiseOp operation);
int64_t fragmentPartCount(FragmentType fragment);
int64_t fragmentPartResourceGroups(FragmentType fragment);
std::optional<AccessAttr> fragmentFieldReadAccess(mlir::Value input, FragmentType result);
std::optional<AccessAttr> fragmentWindowReadAccess(mlir::Value field,
    mlir::Value logicalBase, FragmentType result);
// Exact row-word supply geometry. The alignment in this access is usable only
// under the matching DenseRowWindowGuard true branch, checked by the load op.
std::optional<AccessAttr> fragmentDenseReadAccess(mlir::Value source,
                                                FragmentType result);
bool supportsFragmentWordWindows(mlir::Value field, FragmentType result);
std::optional<int64_t> fragmentPointwiseTemporaryGroups(FragmentType lhs, mlir::Type rhs);
} // namespace weft::riscv

#endif
