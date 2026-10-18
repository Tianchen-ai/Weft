#include "Emitter.h"

namespace weft::riscv_emission {
namespace {
// arith integer types denote finite bit patterns. C integer promotions and a
// binding's spelling must not decide signedness, division or wrap semantics.
struct ScalarInteger {
  std::string storage, unsignedType, signedType, computation;
  unsigned width;

  static std::optional<ScalarInteger> get(mlir::Type type) {
    if (type.isIndex()) return ScalarInteger{"size_t", "size_t", "ptrdiff_t", "size_t", 0};
    auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
    if (!integer || !llvm::is_contained(llvm::ArrayRef<unsigned>{1, 8, 16, 32, 64}, integer.getWidth()))
      return std::nullopt;
    unsigned width = integer.getWidth();
    std::string bits = std::to_string(std::max(8u, width));
    std::string storage = (integer.isUnsigned() ? "uint" : "int") + bits + "_t";
    return ScalarInteger{width == 1 ? "bool" : storage, "uint" + bits + "_t",
                         "int" + bits + "_t", width <= 32 ? "uint32_t" : "uint64_t", width};
  }

  static std::string cast(llvm::StringRef type, llvm::StringRef expression) {
    return "((" + type.str() + ")(" + expression.str() + "))";
  }
  std::string bits(llvm::StringRef expression) const {
    if (width == 1) return "(" + cast("uint32_t", expression) + " & UINT32_C(1))";
    return cast(unsignedType, expression);
  }
  std::string unsignedValue(llvm::StringRef expression) const {
    return cast(computation, bits(expression));
  }
  std::string signedValue(llvm::StringRef expression) const {
    // A signed i1 has the values 0 and -1, not C's unsigned boolean values.
    if (width == 1) return "(-" + cast("int32_t", bits(expression)) + ")";
    return cast(signedType, bits(expression));
  }
  std::string result(llvm::StringRef expression) const {
    return width == 1 ? cast("bool", bits(expression)) : cast(storage, expression);
  }
};
} // namespace

mlir::LogicalResult Emitter::compileScalarControl(mlir::Operation &operation) {
  auto operand = [&](mlir::Value value) -> std::optional<std::string> {
    auto found = bindings.lookup(value);
    if (found.kind != Binding::Kind::Scalar) return std::nullopt;
    return found.scalar;
  };
  auto bind = [&](mlir::Value value, std::string expression) -> mlir::LogicalResult {
    auto form = ScalarInteger::get(value.getType());
    if (!form) return fail(&operation, "scalar control result requires index or a supported finite integer width");
    expression = form->result(expression);
    if (value.hasOneUse()) bindings[value] = scalar(expression);
    else {
      std::string name = fresh("index_value");
      line("const " + form->storage + " " + name + " = " + expression + ";");
      line("(void)" + name + ";");
      bindings[value] = scalar(name);
    }
    return mlir::success();
  };
  auto binary = [&](mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                    llvm::StringRef symbol, bool signedOperands = false) -> mlir::LogicalResult {
    auto form = ScalarInteger::get(lhs.getType());
    auto a = operand(lhs), b = operand(rhs);
    if (!form || lhs.getType() != rhs.getType() || !a || !b)
      return fail(&operation, "scalar control arithmetic requires matching typed integer bindings");
    auto left = signedOperands ? form->signedValue(*a) : form->unsignedValue(*a);
    auto right = signedOperands ? form->signedValue(*b) : form->unsignedValue(*b);
    return bind(result, "(" + left + " " + symbol.str() + " " + right + ")");
  };
  if (auto op = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "+");
  if (auto op = mlir::dyn_cast<mlir::arith::SubIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "-");
  if (auto op = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "*");
  if (auto op = mlir::dyn_cast<mlir::arith::DivUIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "/");
  if (auto op = mlir::dyn_cast<mlir::arith::DivSIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "/", true);
  if (auto op = mlir::dyn_cast<mlir::arith::RemUIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "%");
  if (auto op = mlir::dyn_cast<mlir::arith::RemSIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "%", true);
  if (auto op = mlir::dyn_cast<mlir::arith::AndIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "&");
  if (auto op = mlir::dyn_cast<mlir::arith::OrIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "|");
  if (auto op = mlir::dyn_cast<mlir::arith::XOrIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "^");
  if (auto op = mlir::dyn_cast<mlir::arith::ShLIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), "<<");
  if (auto op = mlir::dyn_cast<mlir::arith::ShRUIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), ">>");
  if (auto op = mlir::dyn_cast<mlir::arith::ShRSIOp>(operation))
    return binary(op.getLhs(), op.getRhs(), op.getResult(), ">>", true);

  if (auto op = mlir::dyn_cast<mlir::arith::CmpIOp>(operation)) {
    auto form = ScalarInteger::get(op.getLhs().getType());
    auto a = operand(op.getLhs()), b = operand(op.getRhs());
    if (!form || !a || !b) return fail(op, "scalar integer comparison requires typed scalar operands");
    llvm::StringRef symbol;
    bool signedComparison = false;
    using P = mlir::arith::CmpIPredicate;
    switch (op.getPredicate()) {
    case P::eq: symbol = "=="; break;
    case P::ne: symbol = "!="; break;
    case P::slt: signedComparison = true; [[fallthrough]];
    case P::ult: symbol = "<"; break;
    case P::sle: signedComparison = true; [[fallthrough]];
    case P::ule: symbol = "<="; break;
    case P::sgt: signedComparison = true; [[fallthrough]];
    case P::ugt: symbol = ">"; break;
    case P::sge: signedComparison = true; [[fallthrough]];
    case P::uge: symbol = ">="; break;
    }
    std::string left = signedComparison ? form->signedValue(*a) : form->unsignedValue(*a);
    std::string right = signedComparison ? form->signedValue(*b) : form->unsignedValue(*b);
    return bind(op.getResult(), "(" + left + " " + symbol.str() + " " + right + ")");
  }
  auto extremum = [&](mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                      bool maximum, bool signedOperands) -> mlir::LogicalResult {
    auto form = ScalarInteger::get(lhs.getType());
    auto a = operand(lhs), b = operand(rhs);
    if (!form || !a || !b) return fail(&operation, "scalar integer extremum requires typed scalar operands");
    std::string left = signedOperands ? form->signedValue(*a) : form->unsignedValue(*a);
    std::string right = signedOperands ? form->signedValue(*b) : form->unsignedValue(*b);
    return bind(result, "((" + left + (maximum ? " > " : " < ") + right + ") ? " + left + " : " + right + ")");
  };
  if (auto op = mlir::dyn_cast<mlir::arith::MinUIOp>(operation))
    return extremum(op.getLhs(), op.getRhs(), op.getResult(), false, false);
  if (auto op = mlir::dyn_cast<mlir::arith::MaxUIOp>(operation))
    return extremum(op.getLhs(), op.getRhs(), op.getResult(), true, false);
  if (auto op = mlir::dyn_cast<mlir::arith::MinSIOp>(operation))
    return extremum(op.getLhs(), op.getRhs(), op.getResult(), false, true);
  if (auto op = mlir::dyn_cast<mlir::arith::MaxSIOp>(operation))
    return extremum(op.getLhs(), op.getRhs(), op.getResult(), true, true);

  auto divideRounded = [&](mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                           bool signedOperands, bool ceiling) -> mlir::LogicalResult {
    auto form = ScalarInteger::get(lhs.getType());
    auto a = operand(lhs), b = operand(rhs);
    if (!form || !a || !b) return fail(&operation, "rounded integer division requires typed scalar operands");
    std::string type = signedOperands ? form->signedType : form->computation;
    std::string left = signedOperands ? form->signedValue(*a) : form->unsignedValue(*a);
    std::string right = signedOperands ? form->signedValue(*b) : form->unsignedValue(*b);
    if (!signedOperands) {
      // One division, with no overflowing a+b-1 intermediate. The subtraction
      // is only observed when a is positive; the final sum is at most a.
      return bind(result, "(" + left + " == 0 ? 0 : ((" + left + " - 1) / " + right + " + 1))");
    }
    auto quotient = fresh("index_quotient"), remainder = fresh("index_remainder");
    line("const " + type + " " + quotient + " = " + left + " / " + right + ";");
    line("const " + type + " " + remainder + " = " + left + " % " + right + ";");
    // C division truncates toward zero. Only a nonzero remainder whose sign
    // requires another integer step changes the quotient for floor or ceil.
    std::string correction = "(" + remainder + " != 0)";
    if (signedOperands)
      correction = "(" + correction + " && ((" + remainder + " < 0) " +
          (ceiling ? "==" : "!=") + " (" + right + " < 0)))";
    return bind(result, "(" + quotient + (ceiling ? " + " : " - ") + correction + ")");
  };
  if (auto op = mlir::dyn_cast<mlir::arith::CeilDivUIOp>(operation))
    return divideRounded(op.getLhs(), op.getRhs(), op.getResult(), false, true);
  if (auto op = mlir::dyn_cast<mlir::arith::CeilDivSIOp>(operation))
    return divideRounded(op.getLhs(), op.getRhs(), op.getResult(), true, true);
  if (auto op = mlir::dyn_cast<mlir::arith::FloorDivSIOp>(operation))
    return divideRounded(op.getLhs(), op.getRhs(), op.getResult(), true, false);

  auto cast = [&](mlir::Value input, mlir::Value result, bool signExtend) -> mlir::LogicalResult {
    auto source = ScalarInteger::get(input.getType());
    auto value = operand(input);
    if (!source || !value || !ScalarInteger::get(result.getType()))
      return fail(&operation, "integer/index cast requires supported source and destination scalar widths");
    return bind(result, signExtend ? source->signedValue(*value) : source->unsignedValue(*value));
  };
  if (auto op = mlir::dyn_cast<mlir::arith::IndexCastOp>(operation))
    return cast(op.getIn(), op.getOut(), true);
  if (auto op = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(operation))
    return cast(op.getIn(), op.getOut(), false);
  if (auto op = mlir::dyn_cast<mlir::arith::ExtSIOp>(operation))
    return cast(op.getIn(), op.getOut(), true);
  if (auto op = mlir::dyn_cast<mlir::arith::ExtUIOp>(operation))
    return cast(op.getIn(), op.getOut(), false);
  if (auto op = mlir::dyn_cast<mlir::arith::TruncIOp>(operation))
    return cast(op.getIn(), op.getOut(), false);

  if (auto op = mlir::dyn_cast<mlir::arith::SelectOp>(operation)) {
    auto condition = operand(op.getCondition());
    auto a = operand(op.getTrueValue()), b = operand(op.getFalseValue());
    if (!op.getCondition().getType().isInteger(1) || !condition || !a || !b ||
        !scalarCType(op.getType()) || mlir::isa<riscv::ValueType, riscv::FragmentType, riscv::LocalType>(op.getType()))
      return fail(op, "terminal arith.select requires a scalar predicate and scalar values");
    std::string expression = "((" + *condition + ") ? (" + *a + ") : (" + *b + "))";
    if (ScalarInteger::get(op.getType())) return bind(op.getResult(), expression);
    bindings[op.getResult()] = scalar(expression);
    return mlir::success();
  }
  return fail(&operation, "unsupported scalar arith operation at the terminal control boundary");
}
} // namespace weft::riscv_emission
