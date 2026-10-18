#ifndef WEFT_DIALECT_RISCV_IR_RISCVOPSTRAITS_H
#define WEFT_DIALECT_RISCV_IR_RISCVOPSTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait {

template <typename ConcreteType>
class WeftRISCVTerminal
    : public TraitBase<ConcreteType, WeftRISCVTerminal> {};

template <typename ConcreteType>
class WeftRISCVLeaf : public TraitBase<ConcreteType, WeftRISCVLeaf> {};

} // namespace mlir::OpTrait

#endif
