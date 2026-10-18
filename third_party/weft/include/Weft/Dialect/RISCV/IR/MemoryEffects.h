#ifndef WEFT_DIALECT_RISCV_IR_MEMORYEFFECTS_H
#define WEFT_DIALECT_RISCV_IR_MEMORYEFFECTS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace weft::riscv {

// Primitive-private storage never escapes as an addressable MemDesc. Its
// contents are exposed only by the operation's typed SSA results.
class PrimitiveStorageResource
    : public mlir::SideEffects::Resource::Base<PrimitiveStorageResource> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrimitiveStorageResource)
  llvm::StringRef getName() const final { return "weft.primitive-storage"; }
};

} // namespace weft::riscv

#endif
