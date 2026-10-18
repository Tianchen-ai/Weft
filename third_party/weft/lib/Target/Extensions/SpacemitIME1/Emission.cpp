#include "../../IME/FragmentEmission.h"

namespace weft::riscv_emission {
mlir::FailureOr<llvm::SmallVector<std::string>> emitSpacemitIME1MMA(
    riscv::IMEFragmentMMAOp operation, llvm::ArrayRef<std::string> lhs,
    llvm::ArrayRef<std::string> rhs, llvm::ArrayRef<std::string> accumulators,
    const FragmentEmission &emission) {
  llvm::SmallVector<std::string> outputs;
  for (const std::string &accumulator : accumulators) {
    outputs.push_back(operation.getAccumulatorPolicy() == "reuse"
                          ? accumulator : emission.fresh("ime_acc"));
    if (operation.getAccumulatorPolicy() == "copy")
      emission.line("vint32m2_t " + outputs.back() + " = " + accumulator + ";");
  }
  emission.line("__asm__ volatile(");
  emission.line("  \".option push\\n\\t.option arch, +xsmtvdot\\n\\t\"");
  emission.line("  \"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
  for (size_t row = 0; row < lhs.size(); ++row)
    for (size_t column = 0; column < rhs.size(); ++column)
      emission.line("  \"smt.vmadot %[acc" + std::to_string(row * rhs.size() + column) +
                    "], %[lhs" + std::to_string(row) + "], %[rhs" +
                    std::to_string(column) + "]\\n\\t\"");
  emission.line("  \".option pop\\n\\t\"");
  std::string outputOperands;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    if (index)
      outputOperands += ", ";
    outputOperands += "[acc" + std::to_string(index) + "] \"+&vr\"(" + output + ")";
  }
  emission.line("  : " + outputOperands);
  std::string inputOperands;
  auto appendInputs = [&](llvm::StringRef prefix, llvm::ArrayRef<std::string> parts) {
    for (auto [index, part] : llvm::enumerate(parts)) {
      if (!inputOperands.empty())
        inputOperands += ", ";
      inputOperands += "[" + prefix.str() + std::to_string(index) + "] \"vr\"(" + part + ")";
    }
  };
  appendInputs("lhs", lhs);
  appendInputs("rhs", rhs);
  emission.line("  : " + inputOperands);
  std::string clobbers;
  for (mlir::Attribute attribute : operation.getAsmClobbers()) {
    if (!clobbers.empty())
      clobbers += ", ";
    clobbers += "\"" + mlir::cast<mlir::StringAttr>(attribute).getValue().str() + "\"";
  }
  emission.line("  : " + clobbers + ");");
  return outputs;
}
} // namespace weft::riscv_emission
