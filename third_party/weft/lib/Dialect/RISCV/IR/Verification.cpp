#include "Verification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

bool weft::riscv::knownIndexMultipleOf(mlir::Value value, int64_t divisor, unsigned depth) {
  if (!value || !value.getType().isIndex() || divisor <= 0 ||
      (divisor & (divisor - 1)) || depth > 24)
    return false;
  if (divisor == 1)
    return true;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt() % divisor == 0;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
        loop && argument == loop.getInductionVar())
      return knownIndexMultipleOf(loop.getLowerBound(), divisor, depth + 1) &&
             knownIndexMultipleOf(loop.getStep(), divisor, depth + 1);
  auto *operation = value.getDefiningOp();
  if (operation && mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp>(operation))
    return knownIndexMultipleOf(operation->getOperand(0), divisor, depth + 1) &&
           knownIndexMultipleOf(operation->getOperand(1), divisor, depth + 1);
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>())
    return knownIndexMultipleOf(multiply.getLhs(), divisor, depth + 1) ||
           knownIndexMultipleOf(multiply.getRhs(), divisor, depth + 1);
  if (auto remainder = value.getDefiningOp<mlir::arith::RemUIOp>())
    if (auto constant = remainder.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
        constant && constant.value() > 0 && constant.value() % divisor == 0)
      return knownIndexMultipleOf(remainder.getLhs(), divisor, depth + 1);
  return false;
}

namespace weft::riscv::detail {

mlir::LogicalResult verifyLeafOperation(mlir::Operation *operation) {
  if (!operation->getAttrOfType<LeafAttr>("leaf"))
    return operation->emitOpError("target-local operation requires a leaf contract");
  return mlir::success();
}

bool exactLeaf(LeafAttr leaf, llvm::StringRef engine,
               llvm::StringRef family, llvm::StringRef instruction,
               llvm::StringRef mask, llvm::StringRef tail) {
  return leaf && leaf.getEngine() == engine && leaf.getFamily() == family &&
         leaf.getInstruction() == instruction &&
         leaf.getSpelling() == instruction && leaf.getMask() == mask &&
         leaf.getTail() == tail;
}

unsigned elementBitWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (type.isIndex())
    return 64;
  return 0;
}

bool integerSignednessMatches(mlir::Type type, llvm::StringRef signedness) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && !integer.isSignless() &&
         ((signedness == "signed" && integer.isSigned()) ||
          (signedness == "unsigned" && integer.isUnsigned()));
}

} // namespace weft::riscv::detail
