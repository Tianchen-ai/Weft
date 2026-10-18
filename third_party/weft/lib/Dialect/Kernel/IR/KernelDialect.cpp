#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/Atomic.h"
#include "Weft/Dialect/Kernel/IR/ReductionRegion.h"
#include "Weft/Dialect/Kernel/IR/StrideABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "Weft/Dialect/Kernel/IR/ShapeBindings.h"
#include <functional>
#include <limits>
#include <optional>
#include <tuple>

using namespace weft::kernel;

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"

bool weft::kernel::isUnitAxisProjection(
    llvm::ArrayRef<int64_t> inputShape, llvm::ArrayRef<int64_t> inputAxes,
    llvm::ArrayRef<int64_t> resultShape, llvm::ArrayRef<int64_t> resultAxes,
    llvm::ArrayRef<int64_t> inputOrder) {
  if (inputShape.size() != inputAxes.size() ||
      resultShape.size() != resultAxes.size() || inputOrder != inputAxes)
    return false;
  size_t source = 0;
  for (auto [extent, axis] : llvm::zip(resultShape, resultAxes)) {
    if (extent == 1) continue;
    while (source < inputShape.size() && inputShape[source] == 1) ++source;
    if (source == inputShape.size() || inputShape[source] != extent ||
        inputAxes[source] != axis)
      return false;
    ++source;
  }
  while (source < inputShape.size() && inputShape[source] == 1) ++source;
  return source == inputShape.size();
}

namespace {

bool isScalar(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

bool isEncoding(mlir::Type type) { return mlir::isa<EncodingType>(type); }

bool isViewLike(mlir::Type type) {
  return mlir::isa<ViewType, SliceType>(type);
}

bool isLocalValue(mlir::Type type) {
  return isScalar(type) || isEncoding(type) || mlir::isa<ValueType>(type);
}

bool isCanonicalNumericElement(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth() == 1 || !integer.isSignless();
  return type.isIndex() || mlir::isa<mlir::FloatType>(type);
}

bool isCanonicalProgramType(mlir::Type type) {
  if (isCanonicalNumericElement(type))
    return true;
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return isEncoding(value.getElementType()) ||
           isCanonicalNumericElement(value.getElementType());
  return mlir::isa<EncodingType, ViewType, SliceType, PointerType, DomainType, PointType>(type);
}

mlir::Type logicalElement(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getElementType();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getEncoding();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getEncoding();
  return type;
}

llvm::ArrayRef<int64_t> logicalShape(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getShape().asArrayRef();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getAxisIds().asArrayRef();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getAxisIds().asArrayRef();
  return {};
}

bool sameDomain(mlir::Type lhs, mlir::Type rhs) {
  return logicalShape(lhs) == logicalShape(rhs) &&
         logicalAxes(lhs) == logicalAxes(rhs);
}

std::optional<int64_t> staticElementCount(mlir::Type type) {
  int64_t count = 1;
  for (int64_t extent : logicalShape(type)) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    count *= extent;
  }
  return count;
}

bool isNumeric(mlir::Type type) {
  type = logicalElement(type);
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type) || type.isIndex();
}

std::optional<unsigned> bitWidth(mlir::Type type) {
  type = logicalElement(type);
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return std::nullopt;
}

std::string denseFamilyFor(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    llvm::StringRef prefix = integer.isUnsigned() ? "u" : "i";
    return (prefix + llvm::Twine(integer.getWidth())).str();
  }
  if (type.isBF16())
    return "bf16";
  if (mlir::isa<mlir::Float8E4M3FNType>(type)) return "f8E4M3FN";
  if (mlir::isa<mlir::Float8E5M2Type>(type)) return "f8E5M2";
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return ("f" + llvm::Twine(floating.getWidth())).str();
  return {};
}

EncodingDeclOp findEncodingDeclaration(mlir::Operation *operation,
                                       llvm::StringRef family) {
  EncodingDeclOp result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
  return result;
}

mlir::LogicalResult verifyEncodingReference(mlir::Operation *operation,
                                            EncodingType encoding) {
  if (encoding.getKind() == "dense") {
    if (encoding.getLayoutIdentity() !=
        ("dense." + encoding.getFamily()).str()) {
      operation->emitOpError()
          << "dense Encoding layout identity must be dense.<family>";
      return mlir::failure();
    }
    return mlir::success();
  }

  auto module = operation->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    operation->emitOpError("Encoding reference must belong to one module");
    return mlir::failure();
  }
  if (encoding.getKind() == "base") {
    unsigned matches = 0;
    module.walk([&](EncodingDeclOp declaration) {
      if (declaration.getSymName() == encoding.getFamily() &&
          declaration.getLayoutIdentity() == encoding.getLayoutIdentity())
        ++matches;
    });
    if (matches != 1) {
      operation->emitOpError()
          << "base Encoding must resolve to exactly one declaration with the "
             "same family and pinned layout identity";
      return mlir::failure();
    }
    return mlir::success();
  }
  if (encoding.getKind() == "derived_instance") {
    unsigned matches = 0;
    module.walk([&](DeriveOp derive) {
      if (derive.getResultFamily() == encoding.getFamily() &&
          derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derive.getParameterValues() == encoding.getParameters().asArrayRef())
        ++matches;
    });
    if (matches != 1) {
      operation->emitOpError()
          << "derived Encoding instance must resolve to exactly one derive "
             "builder with the same family, parameters, and pinned identity";
      return mlir::failure();
    }
    return mlir::success();
  }
  operation->emitOpError(
      "abstract derived Encoding family cannot appear in a runtime program type");
  return mlir::failure();
}

mlir::LogicalResult verifyEncodingInType(mlir::Operation *operation,
                                         mlir::Type type) {
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return verifyEncodingReference(
        operation, mlir::cast<EncodingType>(view.getEncoding()));
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return verifyEncodingReference(
        operation, mlir::cast<EncodingType>(slice.getEncoding()));
  if (auto value = mlir::dyn_cast<ValueType>(type)) {
    if (auto encoding = mlir::dyn_cast<EncodingType>(value.getElementType()))
      return verifyEncodingReference(operation, encoding);
    return mlir::success();
  }
  if (auto encoding = mlir::dyn_cast<EncodingType>(type))
    return verifyEncodingReference(operation, encoding);
  return mlir::success();
}

llvm::StringRef baseFamilyFor(mlir::Operation *operation,
                              EncodingType encoding) {
  if (encoding.getKind() == "base")
    return encoding.getFamily();
  if (encoding.getKind() != "derived_instance")
    return {};
  llvm::StringRef source;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DeriveOp derive) {
      if (source.empty() && derive.getResultFamily() == encoding.getFamily() &&
          derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derive.getParameterValues() ==
              encoding.getParameters().asArrayRef())
        source = derive.getSourceFamily();
    });
  return source;
}

mlir::LogicalResult verifyShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes,
    bool allowEmpty) {
  if (!allowEmpty && shape.empty())
    return emitError() << "logical Value must have positive rank";
  if (shape.size() != axes.size())
    return emitError() << "logical shape and axis identity rank must match";
  llvm::DenseSet<int64_t> seen;
  for (auto [extent, axis] : llvm::zip(shape, axes)) {
    if (extent == 0)
      return emitError() << "logical extents must be positive or symbolic";
    if (axis <= 0)
      return emitError() << "logical axis identities must be positive";
    if (!seen.insert(axis).second)
      return emitError() << "one logical axis identity cannot occur twice";
  }
  return mlir::success();
}

mlir::LogicalResult verifyStringArray(mlir::Operation *operation,
                                      mlir::ArrayAttr values,
                                      llvm::StringRef name) {
  if (llvm::any_of(values, [](mlir::Attribute value) {
        return !mlir::isa<mlir::StringAttr>(value);
      }))
    return operation->emitOpError() << name << " must contain only strings";
  return mlir::success();
}

mlir::LogicalResult verifyPointwise(mlir::Operation *operation,
                                    mlir::Type lhs, mlir::Type rhs,
                                    mlir::Type result) {
  if (!isLocalValue(lhs) || !isLocalValue(rhs) || !isLocalValue(result))
    return operation->emitOpError("pointwise operands must be local Values");
  if (!isNumeric(lhs) || !isNumeric(rhs) || !isNumeric(result))
    return operation->emitOpError("pointwise operands and result must be numeric");
  if (logicalElement(lhs) != logicalElement(rhs))
    return operation->emitOpError("pointwise element types must match");
  if (logicalElement(result) != logicalElement(lhs))
    return operation->emitOpError("pointwise result element type must match");
  llvm::SmallVector<int64_t> expectedAxes(logicalAxes(lhs));
  for (int64_t axis : logicalAxes(rhs))
    if (!llvm::is_contained(expectedAxes, axis))
      expectedAxes.push_back(axis);
  if (!llvm::equal(expectedAxes, logicalAxes(result)))
    return operation->emitOpError(
        "pointwise result axes must follow lhs order then rhs-only axes");
  for (auto [axis, extent] : llvm::zip(logicalAxes(result), logicalShape(result))) {
    std::optional<int64_t> expectedExtent;
    for (mlir::Type operand : {lhs, rhs}) {
      auto operandAxes = logicalAxes(operand);
      auto found = llvm::find(operandAxes, axis);
      if (found == operandAxes.end())
        continue;
      size_t position = static_cast<size_t>(found - operandAxes.begin());
      int64_t operandExtent = logicalShape(operand)[position];
      if (!expectedExtent || *expectedExtent == 1)
        expectedExtent = operandExtent;
      else if (operandExtent != 1 && operandExtent != *expectedExtent)
        return operation->emitOpError(
            "pointwise operands disagree on a shared logical axis extent");
    }
    if (!expectedExtent || extent != *expectedExtent)
      return operation->emitOpError(
          "pointwise result extent must equal its broadcast operand extent");
  }
  return mlir::success();
}

mlir::LogicalResult verifyProjection(mlir::Operation *operation,
                                     mlir::Type base,
                                     mlir::ValueRange indices,
                                     mlir::ArrayAttr selectors,
                                     mlir::Type projected,
                                     bool allowGather) {
  if (selectors.size() > logicalAxes(base).size())
    return operation->emitOpError("projection has more selectors than input axes");
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  size_t indexCursor = 0;
  constexpr int64_t dynamicProjection = std::numeric_limits<int64_t>::min();
  for (size_t position = 0; position < logicalAxes(base).size(); ++position) {
    llvm::StringRef selector =
        position < selectors.size()
            ? mlir::dyn_cast<mlir::StringAttr>(selectors[position]).getValue()
            : llvm::StringRef("all");
    if (selector == "all") {
      expectedAxes.push_back(logicalAxes(base)[position]);
      expectedShape.push_back(logicalShape(base)[position]);
      continue;
    }
    if (indexCursor >= indices.size())
      return operation->emitOpError(
          "projection selector has no corresponding index operand");
    mlir::Value index = indices[indexCursor++];
    if (selector == "domain" || selector == "group_index") {
      auto point = mlir::dyn_cast<PointType>(index.getType());
      if (!point || point.getDomain().getAxisId() != logicalAxes(base)[position])
        return operation->emitOpError(
            "level projection point must select the corresponding logical axis");
      if (selector == "domain") {
        expectedAxes.push_back(logicalAxes(base)[position]);
        expectedShape.push_back(dynamicProjection);
      }
      continue;
    }
    if (selector == "index") {
      if (!index.getType().isIndex())
        return operation->emitOpError("scalar projection index must have index type");
      continue;
    }
    if (selector != "gather" || !allowGather)
      return operation->emitOpError("unknown or disallowed projection selector");
    auto gather = mlir::dyn_cast<ValueType>(index.getType());
    auto integer = gather
                       ? mlir::dyn_cast<mlir::IntegerType>(gather.getElementType())
                       : mlir::IntegerType();
    if (!integer || integer.isSigned())
      return operation->emitOpError(
          "gather projection requires shaped unsigned indices");
    for (auto [axis, extent] : llvm::zip(gather.getAxisIds().asArrayRef(),
                                        gather.getShape().asArrayRef())) {
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return operation->emitOpError(
            "gather projection disagrees on a shared logical axis extent");
      }
    }
  }
  if (indexCursor != indices.size() ||
      !llvm::equal(expectedAxes, logicalAxes(projected)) ||
      expectedShape.size() != logicalShape(projected).size())
    return operation->emitOpError(
        "projection operands and result logical axes do not match");
  for (auto [expected, actual] : llvm::zip(expectedShape, logicalShape(projected)))
    if (expected != dynamicProjection && expected != actual)
      return operation->emitOpError(
          "projection result has an incorrect logical extent");
  return mlir::success();
}

mlir::Type promotedElement(mlir::Type lhs, mlir::Type rhs) {
  lhs = logicalElement(lhs);
  rhs = logicalElement(rhs);
  auto lhsFloat = mlir::dyn_cast<mlir::FloatType>(lhs);
  auto rhsFloat = mlir::dyn_cast<mlir::FloatType>(rhs);
  if (lhsFloat || rhsFloat) {
    unsigned width = std::max(lhsFloat ? lhsFloat.getWidth() : 0u,
                              rhsFloat ? rhsFloat.getWidth() : 0u);
    if (width > 32)
      return mlir::Float64Type::get(lhs.getContext());
    return mlir::Float32Type::get(lhs.getContext());
  }
  auto lhsInteger = mlir::dyn_cast<mlir::IntegerType>(lhs);
  auto rhsInteger = mlir::dyn_cast<mlir::IntegerType>(rhs);
  if (!lhsInteger || !rhsInteger || lhsInteger.isSignless() ||
      rhsInteger.isSignless())
    return {};
  unsigned width = std::max(lhsInteger.getWidth(), rhsInteger.getWidth());
  for (unsigned candidate : {8u, 16u, 32u, 64u})
    if (width <= candidate) {
      auto signedness = lhsInteger.isSigned() || rhsInteger.isSigned()
                            ? mlir::IntegerType::Signed
                            : mlir::IntegerType::Unsigned;
      return mlir::IntegerType::get(lhs.getContext(), candidate, signedness);
    }
  return {};
}

mlir::LogicalResult verifyContractLike(mlir::Operation *operation,
                                       mlir::Value lhs, mlir::Value rhs,
                                       llvm::ArrayRef<int64_t> over,
                                       std::optional<mlir::Type> accType,
                                       mlir::Type result) {
  if (!isLocalValue(lhs.getType()) || !isLocalValue(rhs.getType()) ||
      !isLocalValue(result))
    return operation->emitOpError("contraction operands must be local Values");
  if (!isNumeric(lhs.getType()) || !isNumeric(rhs.getType()) || !isNumeric(result))
    return operation->emitOpError("contraction operands and result must be numeric");
  if (over.empty())
    return operation->emitOpError("contraction requires at least one reduction axis");
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : over) {
    if (axis <= 0 || !llvm::is_contained(logicalAxes(lhs.getType()), axis) ||
        !llvm::is_contained(logicalAxes(rhs.getType()), axis) ||
        !reduction.insert(axis).second)
      return operation->emitOpError(
          "each contraction axis must occur once in both operands");
  }
  for (int64_t axis : logicalAxes(lhs.getType())) {
    auto rhsAxes = logicalAxes(rhs.getType());
    auto found = llvm::find(rhsAxes, axis);
    if (found == rhsAxes.end())
      continue;
    size_t lhsPosition = static_cast<size_t>(
        llvm::find(logicalAxes(lhs.getType()), axis) - logicalAxes(lhs.getType()).begin());
    size_t rhsPosition = static_cast<size_t>(found - rhsAxes.begin());
    if (logicalShape(lhs.getType())[lhsPosition] !=
        logicalShape(rhs.getType())[rhsPosition])
      return operation->emitOpError(
          "contraction operands disagree on a shared logical axis extent");
  }
  llvm::SmallVector<int64_t> expected;
  for (int64_t axis : logicalAxes(lhs.getType()))
    if (!reduction.contains(axis))
      expected.push_back(axis);
  for (int64_t axis : logicalAxes(rhs.getType()))
    if (!reduction.contains(axis) && !llvm::is_contained(expected, axis))
      expected.push_back(axis);
  if (!llvm::equal(expected, logicalAxes(result)))
    return operation->emitOpError(
        "contraction result axes must follow lhs free axes then rhs-only free axes");
  for (auto [axis, extent] : llvm::zip(logicalAxes(result), logicalShape(result))) {
    auto lhsAxes = logicalAxes(lhs.getType());
    auto lhsPosition = llvm::find(lhsAxes, axis);
    auto rhsAxes = logicalAxes(rhs.getType());
    auto rhsPosition = llvm::find(rhsAxes, axis);
    int64_t expectedExtent =
        lhsPosition != lhsAxes.end()
            ? logicalShape(lhs.getType())[static_cast<size_t>(lhsPosition -
                                                             lhsAxes.begin())]
            : logicalShape(rhs.getType())[static_cast<size_t>(rhsPosition -
                                                             rhsAxes.begin())];
    if (extent != expectedExtent)
      return operation->emitOpError(
          "contraction result extent must follow its free operand axis");
  }
  mlir::Type expectedElement =
      accType ? *accType : promotedElement(lhs.getType(), rhs.getType());
  if (!expectedElement || logicalElement(result) != expectedElement)
    return operation->emitOpError(
        "contraction result element must match acc_type or canonical promotion");
  return mlir::success();
}

} // namespace

mlir::LogicalResult EncodingType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, llvm::StringRef kind,
    llvm::StringRef layoutIdentity, mlir::DenseI64ArrayAttr parameters) {
  if (family.empty())
    return emitError() << "encoding family must not be empty";
  if (kind != "base" && kind != "dense" && kind != "derived_family" &&
      kind != "derived_instance")
    return emitError() << "unknown encoding kind";
  if ((kind == "base" || kind == "dense" || kind == "derived_instance") &&
      layoutIdentity.empty())
    return emitError() << "concrete encoding requires a layout identity";
  if (kind == "derived_family" && !layoutIdentity.empty())
    return emitError() << "abstract derived encoding cannot carry a layout identity";
  if ((kind == "base" || kind == "dense" || kind == "derived_family") &&
      !parameters.empty())
    return emitError() << "only a concrete derived encoding carries static parameters";
  if (kind == "derived_instance") {
    if (parameters.empty())
      return emitError() << "derived instance requires static parameters";
    if (llvm::any_of(parameters.asArrayRef(),
                     [](int64_t value) { return value <= 0; }))
      return emitError() << "derived encoding parameters must be positive";
  }
  return mlir::success();
}

mlir::LogicalResult PointerType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType) {
  if (!isCanonicalNumericElement(elementType) || elementType.isIndex())
    return emitError() << "pointer element must be a storage scalar";
  unsigned bits = elementType.getIntOrFloatBitWidth();
  if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return emitError() << "pointer element must have a native storage width";
  return mlir::success();
}

static mlir::LogicalResult verifyPointerMemory(mlir::Operation *op, mlir::Value base,
    mlir::Value offsets, mlir::Value mask, mlir::Value value) {
  auto element = mlir::cast<PointerType>(base.getType()).getElementType();
  if (logicalElement(value.getType()) != element ||
      !logicalElement(mask.getType()).isInteger(1) ||
      !mlir::isa<mlir::IntegerType>(logicalElement(offsets.getType())) ||
      !sameDomain(value.getType(), offsets.getType()) ||
      !sameDomain(value.getType(), mask.getType()))
    return op->emitOpError("pointer memory requires typed element offsets, i1 mask and matching value domains");
  return mlir::success();
}

mlir::LogicalResult PointerCastOp::verify() {
  auto bytes = [](PointerType type) { return std::max(8u, type.getElementType().getIntOrFloatBitWidth()) / 8; };
  if (bytes(getInput().getType()) != bytes(getResult().getType()))
    return emitOpError("pointer reinterpretation preserves storage element width");
  return mlir::success();
}

mlir::LogicalResult PointerLoadOp::verify() {
  if (getOther().getType() != getResult().getType())
    return emitOpError("masked load fill must have the result type");
  return verifyPointerMemory(*this, getBase(), getOffsets(), getMask(), getOther());
}

mlir::LogicalResult PointerStoreOp::verify() {
  return verifyPointerMemory(*this, getBase(), getOffsets(), getMask(), getValue());
}

mlir::LogicalResult PointerAtomicOp::verify() {
  auto element = logicalElement(getResult().getType());
  auto pointer = mlir::cast<PointerType>(getBase().getType());
  auto index = mlir::dyn_cast<mlir::IntegerType>(logicalElement(getOffsets().getType()));
  if (pointer.getElementType() != element || !index || !index.isSigned() || index.getWidth() != 64 ||
      !logicalElement(getMask().getType()).isInteger(1) || !sameDomain(getOffsets().getType(), getResult().getType()))
    return emitOpError("atomic pointer offset/result domain and element types must match");
  for (mlir::Value value : {getMask(), getValue(), getCompare()})
    if (!logicalShape(value.getType()).empty() && !sameDomain(value.getType(), getResult().getType()))
      return emitOpError("atomic operands must be scalar or share the pointer domain");
  if (logicalElement(getValue().getType()) != element || logicalElement(getCompare().getType()) != element)
    return emitOpError("atomic value and comparison preserve the storage dtype");
  return weft::verifyAtomicContract(*this, element, getKind(), getOrdering(), getScope());
}

mlir::LogicalResult ViewType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isEncoding(encoding))
    return emitError() << "View encoding must be an Encoding type";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true);
}

mlir::LogicalResult ValueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isCanonicalNumericElement(elementType) && !isEncoding(elementType))
    return emitError() << "Value element must be numeric or encoded";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(),
                     isCanonicalNumericElement(elementType));
}

mlir::LogicalResult SliceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isEncoding(encoding))
    return emitError() << "slice encoding must be an Encoding type";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true);
}

mlir::LogicalResult DomainType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef axisName, int64_t domainId, int64_t parentDomainId,
    int64_t axisId, llvm::StringRef relation, llvm::StringRef tail) {
  if (axisName.empty())
    return emitError() << "domain axis name must not be empty";
  if (relation != "root" && relation != "rows" && relation != "cols" &&
      relation != "tiles" && relation != "blocks" && relation != "subs")
    return emitError() << "unknown level relation";
  if (relation == "root") {
    if (domainId != 0 || parentDomainId != -1 || axisId != 0)
      return emitError()
             << "root domain has id zero, no parent, and axis identity zero";
  } else if (domainId <= 0 || parentDomainId < 0 || axisId <= 0) {
    return emitError()
           << "child domain has positive domain/axis identities and a parent";
  }
  if (tail != "exact" && tail != "tail")
    return emitError() << "domain tail must be exact or tail";
  return mlir::success();
}

mlir::LogicalResult PointType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DomainType domain) {
  if (!domain)
    return emitError() << "point must reference a domain";
  return mlir::success();
}

mlir::LogicalResult EncodingDeclOp::verify() {
  if (getKind() != "base")
    return emitOpError("encoding declaration must define a base layout");
  if (getLayoutIdentity() != getSymName())
    return emitOpError(
        "base Encoding layout identity must equal its declared family symbol");
  if (getBitOrder() != "lsb_first" && getBitOrder() != "msb_first")
    return emitOpError("bit_order must be lsb_first or msb_first");
  if (getByteOrder() != "little" && getByteOrder() != "big")
    return emitOpError("byte_order must be little or big");
  if (getAlignment() <= 0 || getElements() <= 0 || getStorageBits() <= 0)
    return emitOpError("alignment, elements, and storage_bits must be positive");
  if (failed(verifyStringArray(*this, getFieldNames(), "field_names")))
    return mlir::failure();
  const size_t count = getFieldNames().size();
  if (count == 0)
    return emitOpError("encoding declaration requires at least one field");
  if (getFieldTypes().size() != count || getFieldShapes().size() != count ||
      getFieldLayouts().size() != count ||
      getFieldBitOffsets().size() != count ||
      getFieldStorageBits().size() != count)
    return emitOpError("all encoding field arrays must have the same length");
  int64_t previousEnd = 0;
  int64_t joinedOffset = -1;
  int64_t joinedWidth = 0;
  int64_t joinedFields = 0;
  int64_t joinedNextRole = 0;
  int64_t joinedSize = 0;
  int64_t joinedLowBits = 0;
  std::string joinedOrder;
  llvm::SmallVector<std::pair<int64_t, int64_t>> spans;
  llvm::DenseSet<llvm::StringRef> fieldNames;
  for (size_t index = 0; index < count; ++index) {
    llvm::StringRef fieldName =
        mlir::cast<mlir::StringAttr>(getFieldNames()[index]).getValue();
    if (fieldName.empty() || !fieldNames.insert(fieldName).second)
      return emitOpError("encoding field names must be non-empty and unique");
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(getFieldTypes()[index]);
    auto shapeAttr =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(getFieldShapes()[index]);
    auto layouts = mlir::dyn_cast<mlir::ArrayAttr>(getFieldLayouts()[index]);
    if (!typeAttr || !shapeAttr || !layouts || layouts.empty() ||
        llvm::any_of(shapeAttr.asArrayRef(),
                     [](int64_t extent) { return extent <= 0; }))
      return emitOpError("field_types and field_shapes have invalid elements");
    if (!isCanonicalNumericElement(typeAttr.getValue()) ||
        typeAttr.getValue().isIndex())
      return emitOpError(
          "encoding fields require fixed-width signed, unsigned, or floating elements");
    auto kind = [](mlir::Attribute attribute) -> llvm::StringRef {
      auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
      auto value = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                              : mlir::StringAttr();
      return value ? value.getValue() : llvm::StringRef();
    };
    bool sharedJoined = false;
    int64_t joinedRole = -1;
    if (kind(layouts[0]) == "natural") {
      if (layouts.size() != 1)
        return emitOpError("natural field layout cannot be combined");
      auto width = bitWidth(typeAttr.getValue());
      int64_t elements = 1;
      for (int64_t extent : shapeAttr.asArrayRef())
        elements *= extent;
      if (!width || getFieldStorageBits()[index] !=
                        static_cast<int64_t>(*width) * elements)
        return emitOpError(
            "natural field storage width must match its typed logical shape");
    } else if (kind(layouts[0]) == "joined") {
      if (layouts.size() != 1)
        return emitOpError("joined field layout cannot be combined");
      auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      auto group = joined.getAs<mlir::IntegerAttr>("size");
      auto fields = joined.getAs<mlir::IntegerAttr>("fields");
      auto lowBits = joined.getAs<mlir::IntegerAttr>("low_bits");
      auto role = joined.getAs<mlir::IntegerAttr>("role");
      auto order = joined.getAs<mlir::StringAttr>("order");
      auto integer = mlir::dyn_cast<mlir::IntegerType>(typeAttr.getValue());
      if (!group || !fields || !lowBits || !role || !order || !integer ||
          shapeAttr.size() != 1 || group.getInt() <= 0 || fields.getInt() < 2 ||
          role.getInt() < 0 || role.getInt() >= fields.getInt() ||
          shapeAttr[0] != 2 * group.getInt() || lowBits.getInt() <= 0 ||
          lowBits.getInt() >= integer.getWidth() ||
          lowBits.getInt() * fields.getInt() != 8 ||
          integer.getWidth() + integer.getWidth() - lowBits.getInt() != 8 ||
          (order.getValue() != "lo_first" && order.getValue() != "hi_first"))
        return emitOpError("invalid joined field layout");
      joinedRole = role.getInt();
      sharedJoined = true;
      const int64_t expectedStorage =
          group.getInt() * (fields.getInt() + 1) * 8;
      if (getFieldStorageBits()[index] != expectedStorage)
        return emitOpError(
            "joined field storage span must be uniquely implied by its layout");
    } else {
      if (layouts.size() != 2 || kind(layouts[0]) != "grouped" ||
          kind(layouts[1]) != "layered")
        return emitOpError(
            "field layout must be natural or grouped followed by layered");
      auto grouped = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      auto layered = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
      auto group = grouped.getAs<mlir::IntegerAttr>("size");
      auto layer = layered.getAs<mlir::IntegerAttr>("size");
      auto order = layered.getAs<mlir::StringAttr>("order");
      if (!group || !layer || !order || group.getInt() <= 0 ||
          layer.getInt() <= 0 || group.getInt() % layer.getInt() ||
          (order.getValue() != "lo_first" &&
           order.getValue() != "hi_first"))
        return emitOpError("invalid grouped/layered field layout");
      auto integer = mlir::dyn_cast<mlir::IntegerType>(typeAttr.getValue());
      if (!integer || shapeAttr.size() != 1 ||
          shapeAttr[0] % group.getInt() ||
          integer.getWidth() * (group.getInt() / layer.getInt()) != 8)
        return emitOpError(
            "grouped/layered field must form one byte per layer position");
      if (getFieldStorageBits()[index] !=
          integer.getWidth() * shapeAttr[0])
        return emitOpError(
            "grouped/layered field storage span must match its complete logical field");
    }
    int64_t offset = getFieldBitOffsets()[index];
    int64_t width = getFieldStorageBits()[index];
    if (offset < 0 || width <= 0 || offset + width > getStorageBits())
      return emitOpError("encoding fields overlap or exceed storage_bits");
    if (sharedJoined) {
      auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      int64_t fields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
      if (joinedRole == 0) {
        if (offset < previousEnd)
          return emitOpError("joined storage overlaps a previous field");
        joinedOffset = offset;
        joinedWidth = width;
        joinedFields = fields;
        joinedNextRole = 1;
        joinedSize = joined.getAs<mlir::IntegerAttr>("size").getInt();
        joinedLowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
        joinedOrder = joined.getAs<mlir::StringAttr>("order").getValue().str();
        previousEnd = offset + width;
        spans.emplace_back(offset, offset + width);
      } else if (offset != joinedOffset || width != joinedWidth ||
                 fields != joinedFields || joinedRole != joinedNextRole++ ||
                 joined.getAs<mlir::IntegerAttr>("size").getInt() != joinedSize ||
                 joined.getAs<mlir::IntegerAttr>("low_bits").getInt() !=
                     joinedLowBits ||
                 joined.getAs<mlir::StringAttr>("order").getValue() !=
                     joinedOrder) {
        return emitOpError(
            "joined logical fields must be consecutive roles over one storage span");
      }
    } else {
      if (joinedOffset >= 0 && joinedNextRole != joinedFields)
        return emitOpError("joined storage is missing one or more logical fields");
      joinedOffset = -1;
      joinedNextRole = 0;
      if (offset < previousEnd)
        return emitOpError("encoding fields overlap");
      previousEnd = offset + width;
      spans.emplace_back(offset, offset + width);
    }
  }
  if (joinedOffset >= 0 && joinedNextRole != joinedFields)
    return emitOpError("joined storage is missing one or more logical fields");
  if (getPadding().size() % 3)
    return emitOpError("padding is a sequence of offset, width, fill triples");
  for (size_t index = 0; index < getPadding().size(); index += 3) {
    if (getPadding()[index] < 0 || getPadding()[index + 1] <= 0 ||
        getPadding()[index] + getPadding()[index + 1] > getStorageBits() ||
        getPadding()[index + 2] < 0 || getPadding()[index + 2] > 255)
      return emitOpError("padding triple is outside the declared layout");
    if (getPadding()[index] % 8 || getPadding()[index + 1] % 8)
      return emitOpError("padding byte ranges must be byte aligned");
    spans.emplace_back(getPadding()[index],
                       getPadding()[index] + getPadding()[index + 1]);
  }
  llvm::sort(spans);
  int64_t cursor = 0;
  for (auto [begin, end] : spans) {
    if (begin != cursor)
      return emitOpError("layout gaps must be represented by explicit padding");
    cursor = end;
  }
  if (cursor != getStorageBits())
    return emitOpError("fields and padding must cover storage_bits exactly");
  return mlir::success();
}

mlir::LogicalResult DeriveOp::verify() {
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily())
    return emitOpError("derive requires distinct source and result families");
  if (getLayoutIdentity().empty() ||
      failed(verifyStringArray(*this, getParameterNames(), "parameter_names")))
    return mlir::failure();
  if (getParameterNames().size() != getParameterValues().size() ||
      getParameterNames().empty())
    return emitOpError(
        "derive requires one typed value for every static parameter name");
  llvm::DenseSet<llvm::StringRef> parameterNames;
  for (auto [nameAttr, value] :
       llvm::zip(getParameterNames(), getParameterValues())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    if (name.empty() || !parameterNames.insert(name).second || value <= 0)
      return emitOpError(
          "derive parameter names are unique and values are positive");
  }
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1 || !mlir::isa<ViewType>(body.getArgument(0).getType()))
    return emitOpError("derive body takes one source View");
  auto source = mlir::cast<ViewType>(body.getArgument(0).getType()).getEncoding();
  auto sourceEncoding = mlir::cast<EncodingType>(source);
  if (sourceEncoding.getKind() != "base" ||
      sourceEncoding.getFamily() != getSourceFamily())
    return emitOpError("derive source View family does not match source_family");
  if (failed(verifyEncodingReference(*this, sourceEncoding)))
    return mlir::failure();
  auto yield = mlir::dyn_cast<DeriveYieldOp>(body.getTerminator());
  if (!yield)
    return emitOpError("derive body must end in derive_yield");
  auto resultEncoding = yield.getValue().getType().getEncoding();
  auto result = mlir::cast<EncodingType>(resultEncoding);
  if (result.getKind() != "derived_instance" ||
      result.getFamily() != getResultFamily() ||
      result.getLayoutIdentity() != getLayoutIdentity() ||
      result.getParameters().asArrayRef() != getParameterValues())
    return emitOpError(
        "derive result must be one concrete instance of its derived encoding family");
  if (failed(verifyEncodingReference(*this, result)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult DeriveYieldOp::verify() {
  if (!mlir::isa_and_nonnull<DeriveOp>(getOperation()->getParentOp()))
    return emitOpError("derive_yield terminates only a derive body");
  return mlir::success();
}

mlir::LogicalResult KernelOp::verify() {
  mlir::Block &body = getBody().front();
  if (getArgNames().size() != body.getNumArguments() ||
      getArgAccess().size() != body.getNumArguments() ||
      getArgAliasSets().size() != body.getNumArguments())
    return emitOpError(
        "argument names, access, and alias sets must match kernel entry arguments");
  if (failed(verifyStringArray(*this, getArgNames(), "arg_names")) ||
      failed(verifyStringArray(*this, getArgAccess(), "arg_access")) ||
      failed(verifyStringArray(*this, getShapeSymbols(), "shape_symbols")))
    return mlir::failure();
  llvm::DenseSet<llvm::StringRef> names;
  for (auto [index, nameAttr] : llvm::enumerate(getArgNames())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    if (name.empty() || !names.insert(name).second)
      return emitOpError("kernel argument names must be non-empty and unique");
    llvm::StringRef access =
        mlir::cast<mlir::StringAttr>(getArgAccess()[index]).getValue();
    mlir::Type type = body.getArgument(index).getType();
    if (access != "none" && access != "read" && access != "write" &&
        access != "readwrite")
      return emitOpError("unknown kernel argument access mode");
    if (mlir::isa<ViewType, PointerType>(type)) {
      if (getArgAliasSets()[index] < 0)
        return emitOpError("View argument requires a non-negative alias set");
    } else if (access != "none" || getArgAliasSets()[index] != -1) {
      return emitOpError(
          "scalar argument has no View access effect or alias set");
    }
  }
  if (!mlir::isa<ReturnOp>(body.getTerminator()))
    return emitOpError("kernel body must end in weft_kernel.return");
  if (failed(weft::verifyStrideABI<ViewType>(*this, body, getArgNames(),
                                            getShapeSymbols(), getStridedArgsAttr())))
    return mlir::failure();

  bool typeFailure = false;
  auto verifyTypes = [&](mlir::Operation *owner, mlir::Block &block) {
    for (mlir::BlockArgument argument : block.getArguments()) {
      if (!isCanonicalProgramType(argument.getType())) {
        owner->emitOpError(
            "region argument type is outside the canonical Weft type set");
        typeFailure = true;
      } else if (failed(verifyEncodingInType(owner, argument.getType()))) {
        typeFailure = true;
      }
    }
  };
  verifyTypes(*this, body);
  getBody().walk([&](mlir::Operation *operation) {
    llvm::StringRef name = operation->getName().getStringRef();
    const bool standardCanonical =
        name == "arith.constant" || name == "arith.ceildivui" ||
        name == "scf.for" || name == "scf.if" || name == "scf.while" ||
        name == "scf.yield" || name == "scf.condition";
    if (operation->getDialect() != getOperation()->getDialect() &&
        !standardCanonical) {
      operation->emitOpError(
          "operation is outside the canonical Weft/arith/SCF contract");
      typeFailure = true;
    }
    for (mlir::Value operand : operation->getOperands())
      if (!isCanonicalProgramType(operand.getType())) {
        operation->emitOpError("operand type is outside the canonical Weft type set");
        typeFailure = true;
      }
    for (mlir::Value result : operation->getResults())
      if (!isCanonicalProgramType(result.getType())) {
        operation->emitOpError("result type is outside the canonical Weft type set");
        typeFailure = true;
      } else if (failed(verifyEncodingInType(operation, result.getType())))
        typeFailure = true;
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &nested : region)
        verifyTypes(operation, nested);
  });
  if (typeFailure)
    return mlir::failure();

  llvm::DenseSet<int64_t> localShapes;
  getBody().walk([&](LocalShapeOp shape) {
    if (!localShapes.insert(shape.getId()).second) {
      shape.emitOpError("local shape identities must be unique within a kernel");
      typeFailure = true;
    }
  });
  auto checkLocalShapeUses = [&](mlir::Operation *owner, mlir::Type type) {
    for (int64_t extent : logicalShape(type)) {
      if (extent >= -static_cast<int64_t>(getShapeSymbols().size())) continue;
      if (extent == std::numeric_limits<int64_t>::min() ||
          !weft::localShapeExtent<LocalShapeOp>(owner, -extent)) {
        owner->emitOpError("local shape reference escapes its SSA definition scope");
        typeFailure = true;
      }
    }
  };
  getBody().walk([&](mlir::Operation *op) {
    for (mlir::Type type : op->getOperandTypes()) checkLocalShapeUses(op, type);
    for (mlir::Type type : op->getResultTypes()) checkLocalShapeUses(op, type);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (auto argument : block.getArguments()) checkLocalShapeUses(op, argument.getType());
  });
  if (typeFailure) return mlir::failure();

  llvm::DenseSet<int64_t> domainIds;
  unsigned roots = 0;
  getBody().walk([&](mlir::Operation *operation) {
    if (auto root = mlir::dyn_cast<RootDomainOp>(operation)) {
      ++roots;
      domainIds.insert(root.getResult().getType().getDomainId());
      return;
    }
    if (auto domain = mlir::dyn_cast<DomainOp>(operation)) {
      if (!domainIds.insert(domain.getResult().getType().getDomainId()).second) {
        domain.emitOpError("domain identity must be unique within one kernel");
        typeFailure = true;
      }
      return;
    }
    if (auto level = mlir::dyn_cast<LevelOp>(operation))
      if (!level.getDomain().getDefiningOp<DomainOp>()) {
        level.emitOpError("Level domain must be produced by a canonical domain op");
        typeFailure = true;
      }
  });
  if (roots != 1) {
    emitOpError("kernel must contain exactly one canonical root domain");
    return mlir::failure();
  }
  if (typeFailure)
    return mlir::failure();

  auto rootArgument = [&](mlir::Value value) -> std::optional<unsigned> {
    while (mlir::Operation *producer = value.getDefiningOp()) {
      if (auto slice = mlir::dyn_cast<SliceOp>(producer)) {
        value = slice.getBase();
        continue;
      }
      if (auto subview = mlir::dyn_cast<SubviewOp>(producer)) {
        value = subview.getBase();
        continue;
      }
      if (auto field = mlir::dyn_cast<FieldOp>(producer)) {
        value = field.getOwner();
        continue;
      }
      return std::nullopt;
    }
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || argument.getOwner() != &body)
      return std::nullopt;
    return argument.getArgNumber();
  };
  getBody().walk([&](mlir::Operation *operation) {
    mlir::Value region;
    llvm::StringRef required;
    if (auto admit = mlir::dyn_cast<AdmitOp>(operation)) {
      region = admit.getRegion();
      required = "read";
    } else if (auto commit = mlir::dyn_cast<CommitOp>(operation)) {
      region = commit.getRegion();
      required = "write";
    } else if (auto atomic = mlir::dyn_cast<AtomicRMWOp>(operation)) {
      region = atomic.getRegion();
      required = "readwrite";
    } else {
      return;
    }
    auto index = rootArgument(region);
    if (!index)
      return;
    llvm::StringRef access =
        mlir::cast<mlir::StringAttr>(getArgAccess()[*index]).getValue();
    const bool allowed = required == "readwrite" ? access == "readwrite"
                         : required == "read"
                             ? access == "read" || access == "readwrite"
                             : access == "write" || access == "readwrite";
    if (!allowed) {
      operation->emitOpError()
          << required << " effect contradicts the kernel argument access mode";
      typeFailure = true;
    }
  });
  if (typeFailure)
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult HelperOp::verify() {
  if (getName().empty() || getBody().front().getNumArguments())
    return emitOpError("helper needs a source name and a closure body");
  bool found = false;
  getBody().walk([&](HelperReturnOp ret) {
    if (ret->getParentOfType<HelperOp>() == *this) found = true;
  });
  if (!found) return emitOpError("helper has no return path");
  return mlir::success();
}

mlir::LogicalResult HelperReturnOp::verify() {
  auto helper = (*this)->getParentOfType<HelperOp>();
  if (!helper || getValues().getTypes() != helper.getResultTypes())
    return emitOpError("helper return values must match their enclosing function signature");
  return mlir::success();
}

void HelperReturnOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), FunctionControlResource::get());
}

mlir::LogicalResult ReturnOp::verify() {
  if (!mlir::isa_and_nonnull<KernelOp>(getOperation()->getParentOp()))
    return emitOpError("return terminates only a kernel body");
  if (!getValues().empty())
    return emitOpError("current kernels return through commit, not SSA results");
  return mlir::success();
}

mlir::LogicalResult RootDomainOp::verify() {
  DomainType type = getResult().getType();
  if (type.getRelation() != "root" || type.getDomainId() != 0 ||
      type.getParentDomainId() != -1 || type.getAxisId() != 0)
    return emitOpError("root_domain must produce the canonical root domain");
  return mlir::success();
}

mlir::LogicalResult SymbolOp::verify() {
  if (getName().empty() || (getKind() != "shape" && getKind() != "source_auto"))
    return emitOpError("symbol is a named shape or source auto value");
  if (getKind() == "shape" && !getChoices().empty())
    return emitOpError("shape symbol cannot carry source-auto choices");
  if (llvm::any_of(getChoices(),
                   [](int64_t value) { return value <= 0; }))
    return emitOpError("source-auto choices must be positive");
  return mlir::success();
}

mlir::LogicalResult DomainOp::verify() {
  DomainType parent = getParent().getType();
  DomainType child = getResult().getType();
  if (child.getRelation() == "root")
    return emitOpError("child domain cannot be another root");
  if (child.getParentDomainId() != parent.getDomainId())
    return emitOpError("child domain must name its parent domain identity");
  if (child.getAxisId() == parent.getAxisId() &&
      child.getAxisName().split('.').first !=
          parent.getAxisName().split('.').first)
    return emitOpError("a refined logical axis must preserve its axis name");
  auto [axisBase, axisSerial] = child.getAxisName().rsplit('.');
  if (axisBase.empty() || axisSerial != std::to_string(child.getDomainId()))
    return emitOpError(
        "child domain axis name must carry its unique domain identity suffix");
  auto parentProducer = getParent().getDefiningOp();
  if (!mlir::isa_and_nonnull<RootDomainOp, DomainOp>(parentProducer))
    return emitOpError("domain parent must be a canonical root or child domain");
  auto multiplicity = getMultiplicity().getDefiningOp<mlir::arith::CeilDivUIOp>();
  if (!multiplicity || multiplicity.getLhs() != getExtent() ||
      multiplicity.getRhs() != getPartition())
    return emitOpError(
        "domain multiplicity must be ceildiv(extent, partition)");
  for (auto [value, name] :
       {std::pair(getExtent(), llvm::StringRef("extent")),
        std::pair(getPartition(), llvm::StringRef("partition"))}) {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
          integer && integer.getInt() <= 0)
        return emitOpError() << "domain " << name << " must be positive";
  }
  return mlir::success();
}

mlir::LogicalResult LevelOp::verify() {
  if (getCarried().size() != getResults().size())
    return emitOpError("level carried operands and results must match");
  for (auto [operand, result] : llvm::zip(getCarried(), getResults()))
    if (operand.getType() != result.getType())
      return emitOpError("level carried operand/result types must match");
  DomainType domain = getDomain().getType();
  PointType point = PointType::get(getContext(), domain);
  mlir::Block &state = getStateBirths().front();
  mlir::Block &staged = getStagedBirths().front();
  mlir::Block &body = getBody().front();
  if (state.getNumArguments() != 1 || state.getArgument(0).getType() != point ||
      staged.getNumArguments() != 1 || staged.getArgument(0).getType() != point)
    return emitOpError("both births regions take the current level point");
  auto stateYield = mlir::dyn_cast<BirthsYieldOp>(state.getTerminator());
  auto stagedYield = mlir::dyn_cast<BirthsYieldOp>(staged.getTerminator());
  if (!stateYield || !stagedYield)
    return emitOpError("births regions must end in births_yield");
  for (mlir::Value value : stateYield.getValues())
    if (!value.getDefiningOp<NewOp>() ||
        value.getParentRegion() != &getStateBirths())
      return emitOpError("state_births may yield only values defined by new");
  for (mlir::Value value : stagedYield.getValues())
    if (!value.getDefiningOp<MaterializeOp>() ||
        value.getParentRegion() != &getStagedBirths())
      return emitOpError(
          "staged_births may yield only values defined by materialize");

  const size_t expectedArguments = 1 + getCarried().size() +
                                   stateYield.getValues().size() +
                                   stagedYield.getValues().size();
  if (body.getNumArguments() != expectedArguments ||
      body.getArgument(0).getType() != point)
    return emitOpError("level body signature does not match domain and births");
  size_t cursor = 1;
  for (mlir::Value carried : getCarried())
    if (body.getArgument(cursor++).getType() != carried.getType())
      return emitOpError("level body carried argument type mismatch");
  for (mlir::Value birth : stateYield.getValues())
    if (body.getArgument(cursor++).getType() != birth.getType())
      return emitOpError("state birth type mismatch");
  for (mlir::Value birth : stagedYield.getValues())
    if (body.getArgument(cursor++).getType() != birth.getType())
      return emitOpError("staged birth type mismatch");
  auto handoff = mlir::dyn_cast<HandoffOp>(body.getTerminator());
  if (!handoff || handoff.getValues().getTypes() != getResults().getTypes())
    return emitOpError("level body handoff must match level results");
  return mlir::success();
}

mlir::LogicalResult BirthsYieldOp::verify() {
  auto level = mlir::dyn_cast_or_null<LevelOp>(getOperation()->getParentOp());
  if (!level || (getOperation()->getParentRegion() != &level.getStateBirths() &&
                 getOperation()->getParentRegion() != &level.getStagedBirths()))
    return emitOpError("births_yield terminates a level births region");
  return mlir::success();
}

mlir::LogicalResult HandoffOp::verify() {
  auto level = mlir::dyn_cast_or_null<LevelOp>(getOperation()->getParentOp());
  if (!level || getOperation()->getParentRegion() != &level.getBody())
    return emitOpError("handoff terminates a level body");
  return mlir::success();
}

mlir::LogicalResult ConstantOp::verify() {
  if (!isScalar(getResult().getType()) || getResult().getType().isIndex())
    return emitOpError(
        "constant represents a signed or unsigned scalar numeric literal");
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(getValue());
  if (!typed || typed.getType() != getResult().getType())
    return emitOpError("constant attribute and result type must match");
  return mlir::success();
}

mlir::LogicalResult IotaOp::verify() {
  auto result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  if ((!element && !result.getElementType().isIndex()) ||
      (element && element.isSigned()) || result.getShape().size() != 1 ||
      result.getAxisIds().size() != 1)
    return emitOpError("iota produces one unsigned integer or index logical axis");
  if (getEnd() <= getStart() || result.getShape()[0] != getEnd() - getStart())
    return emitOpError("iota shape must equal its positive [start, end) extent");
  return mlir::success();
}

mlir::LogicalResult ValueFactOp::verify() {
  if (getKind() != "divisibility" && getKind() != "contiguity" && getKind() != "constancy")
    return emitOpError("unknown logical-axis fact");
  auto inputs = getInputs();
  if (inputs.size() != 1 && inputs.size() != 2)
    return emitOpError("value facts require a numeric value or a pointer origin and offsets");
  mlir::Type numeric = inputs.back().getType();
  if (!isNumeric(numeric) || (inputs.size() == 2 &&
      (!mlir::isa<PointerType>(inputs.front().getType()) || !logicalElement(numeric).isInteger(64))))
    return emitOpError("value facts require a numeric domain and explicit pointer provenance");
  size_t rank = logicalShape(numeric).size();
  if (getKind() == "divisibility") rank = std::max(size_t{1}, rank);
  if (getValues().size() != rank || llvm::any_of(getValues(), [](int64_t n) { return n <= 0; }))
    return emitOpError("value facts require one positive integer per logical axis");
  return mlir::success();
}

mlir::LogicalResult NewOp::verify() {
  if (!isLocalValue(getResult().getType()))
    return emitOpError("new produces a local Value");
  if (getInitialized() != static_cast<bool>(getInitial()))
    return emitOpError("initialized must agree with the optional initial value");
  if (!getInitial())
    return emitOpError(
        "new requires an explicit initial value; uninitialized state has no canonical semantics");
  mlir::Type initial = getInitial().getType();
  mlir::Type result = getResult().getType();
  if (initial == result)
    return mlir::success();
  if (!logicalShape(initial).empty() || logicalElement(initial) != logicalElement(result))
    return emitOpError("new initial value must match or scalar-broadcast to the result");
  return mlir::success();
}

mlir::LogicalResult LocalShapeOp::verify() {
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!owner || getId() <= static_cast<int64_t>(owner.getShapeSymbols().size()))
    return emitOpError("local shape identity must be distinct from every public ABI shape");
  if (auto constant = getExtent().getDefiningOp<mlir::arith::ConstantIndexOp>();
      constant && constant.value() < 0)
    return emitOpError("local storage extent cannot be negative");
  return mlir::success();
}

void LocalShapeOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), LocalShapeResource::get());
}

mlir::LogicalResult EmptyOp::verify() {
  auto type = getResult().getType();
  if (!isNumeric(type) || type.getShape().size() != getExtents().size())
    return emitOpError("empty storage requires one explicit extent per numeric value axis");
  auto owner = (*this)->getParentOfType<KernelOp>();
  if (!owner) return emitOpError("empty storage must be task-local");
  for (auto [shape, extent] : llvm::zip(type.getShape().asArrayRef(), getExtents())) {
    if (auto constant = extent.getDefiningOp<mlir::arith::ConstantIndexOp>();
        constant && (constant.value() < 0 || (shape > 0 && constant.value() != shape)))
      return emitOpError("empty storage extent disagrees with its declared shape");
    if (shape < -static_cast<int64_t>(owner.getShapeSymbols().size()) &&
        weft::localShapeExtent<LocalShapeOp>(*this, -shape) != extent)
      return emitOpError("local shape must be bound by the dominating actual extent SSA");
  }
  return mlir::success();
}

mlir::LogicalResult MaterializeOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("materialize preserves its logical value type");
  return mlir::success();
}

mlir::LogicalResult AdmitOp::verify() {
  if (!isViewLike(getRegion().getType()) || !isLocalValue(getResult().getType()))
    return emitOpError("admit maps a View region to a local Value");
  mlir::Type resultElement = logicalElement(getResult().getType());
  if (auto encoded = mlir::dyn_cast<EncodingType>(resultElement)) {
    auto regionEncoding =
        mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
    if (encoded != regionEncoding ||
        !sameDomain(getRegion().getType(), getResult().getType()))
      return emitOpError(
          "encoded admit preserves the selected logical record domain");
  } else if (!sameDomain(getRegion().getType(), getResult().getType())) {
    return emitOpError("dense admit must preserve the logical domain");
  }
  return mlir::success();
}

mlir::LogicalResult CommitOp::verify() {
  if (!isLocalValue(getValue().getType()) || !isViewLike(getRegion().getType()))
    return emitOpError("commit maps a local Value to a View region");
  if (!sameDomain(getValue().getType(), getRegion().getType()))
    return emitOpError("commit value and destination must have the same domain");
  auto destination =
      mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
  if (destination.getKind() == "dense") {
    if (!isNumeric(getValue().getType()) ||
        destination.getFamily() != denseFamilyFor(logicalElement(getValue().getType())))
      return emitOpError(
          "dense commit requires an explicitly converted matching element type");
  } else if (logicalElement(getValue().getType()) != destination) {
    return emitOpError("encoded commit requires identical encoding");
  }
  return mlir::success();
}

mlir::LogicalResult AtomicRMWOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getValue().getType());
  if (!integer || integer.isSignless() ||
      (integer.getWidth() != 32 && integer.getWidth() != 64) ||
      getResult().getType() != integer ||
      !logicalShape(getRegion().getType()).empty())
    return emitOpError("atomic update requires one scalar i32/u32/i64/u64 address and matching old-value result");
  auto encoding = mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
  if (encoding.getKind() != "dense" ||
      encoding.getFamily() != denseFamilyFor(integer))
    return emitOpError("atomic update preserves the dense storage element type");
  if (getKind() != "add")
    return emitOpError("unsupported atomic update kind");
  if (getOrdering() != "relaxed" && getOrdering() != "acquire" &&
      getOrdering() != "release" && getOrdering() != "acq_rel" &&
      getOrdering() != "seq_cst")
    return emitOpError("unsupported atomic memory ordering");
  return mlir::success();
}

mlir::LogicalResult SliceOp::verify() {
  if (!isViewLike(getBase().getType()))
    return emitOpError("slice base must be a View or slice");
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return mlir::failure();
  if (logicalElement(getBase().getType()) != logicalElement(getResult().getType()))
    return emitOpError("slice preserves its View encoding");
  return verifyProjection(*this, getBase().getType(), getIndices(),
                          getSelectors(), getResult().getType(), false);
}

mlir::LogicalResult SubviewOp::verify() {
  if (!isViewLike(getBase().getType()))
    return emitOpError("subview base must be a View or slice");
  if (logicalElement(getBase().getType()) !=
          logicalElement(getResult().getType()) ||
      !llvm::equal(logicalAxes(getBase().getType()),
                   logicalAxes(getResult().getType())))
    return emitOpError("subview preserves Encoding and logical axes");
  return verifySubviewBounds(*this, logicalShape(getBase().getType()),
      logicalShape(getResult().getType()), getOffsetsAttr(), getExtentsAttr(),
      getDynamicBounds());
}

mlir::LogicalResult ExtentOp::verify() {
  if (!isViewLike(getBase().getType()) || getAxis() < 0 ||
      static_cast<size_t>(getAxis()) >= logicalShape(getBase().getType()).size())
    return emitOpError("extent requires a View or slice and an in-range axis position");
  return mlir::success();
}

mlir::LogicalResult ReshapeOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  if (input.getElementType() != result.getElementType())
    return emitOpError("reshape preserves the logical element type");
  auto inputCount = staticElementCount(input);
  auto resultCount = staticElementCount(result);
  bool axisRename = input.getShape() == result.getShape() &&
                    llvm::equal(getOrder(), input.getAxisIds().asArrayRef());
  bool unitProjection = isUnitAxisProjection(input.getShape(), input.getAxisIds(),
      result.getShape(), result.getAxisIds(), getOrder());
  if (!axisRename && !unitProjection && (!inputCount || !resultCount || *inputCount != *resultCount))
    return emitOpError(
        "reshape requires equal static element counts, an ordered shape rename, or a unit-axis projection");
  auto order = getOrder();
  auto inputAxes = input.getAxisIds().asArrayRef();
  if (order.size() != inputAxes.size())
    return emitOpError("reshape order must name every input logical axis");
  llvm::DenseSet<int64_t> seen;
  for (int64_t axis : order)
    if (!llvm::is_contained(inputAxes, axis) || !seen.insert(axis).second)
      return emitOpError(
          "reshape order must be a permutation of the input logical axes");
  return mlir::success();
}

mlir::LogicalResult FieldOp::verify() {
  mlir::Type owner = getOwner().getType();
  mlir::Type result = getResult().getType();
  if ((!isLocalValue(owner) && !isViewLike(owner)) ||
      !isEncoding(logicalElement(owner)) || getName().empty())
    return emitOpError(
        "field access requires a named field of an encoded Value or View region");
  auto ownerEncoding = mlir::cast<EncodingType>(logicalElement(owner));
  llvm::StringRef baseFamily = baseFamilyFor(*this, ownerEncoding);
  EncodingDeclOp declaration = findEncodingDeclaration(*this, baseFamily);
  if (!declaration)
    return emitOpError("encoded field owner has no matching base declaration");
  size_t fieldIndex = declaration.getFieldNames().size();
  for (auto [index, attribute] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(attribute).getValue() == getName()) {
      fieldIndex = index;
      break;
    }
  if (fieldIndex == declaration.getFieldNames().size())
    return emitOpError("field name is absent from the owner Encoding declaration");
  mlir::Type fieldType =
      mlir::cast<mlir::TypeAttr>(declaration.getFieldTypes()[fieldIndex]).getValue();
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[fieldIndex])
          .asArrayRef();
  auto ownerShape = logicalShape(owner);
  auto ownerAxes = logicalAxes(owner);
  auto resultShape = logicalShape(result);
  auto resultAxes = logicalAxes(result);
  if (ownerShape.empty() || ownerShape.back() != declaration.getElements())
    return emitOpError(
        "encoded field owner must expose exactly one complete record axis");
  llvm::SmallVector<int64_t> expectedShape(ownerShape.drop_back());
  expectedShape.append(fieldShape.begin(), fieldShape.end());
  if (resultAxes.size() < ownerAxes.size() - 1 ||
      (!fieldShape.empty() && resultAxes.empty()) ||
      !llvm::equal(resultShape, expectedShape) ||
      !llvm::equal(resultAxes.take_front(ownerAxes.size() - 1),
                   ownerAxes.drop_back()) ||
      (!fieldShape.empty() && resultAxes.back() != ownerAxes.back()))
    return emitOpError(
        "field result must replace the record axis with its declared field shape");
  if (isLocalValue(owner)) {
    if (!isLocalValue(result) || !isNumeric(result))
      return emitOpError("encoded Value field access produces a numeric local Value");
    if (logicalElement(result) != fieldType)
      return emitOpError("field result element must match the declared field type");
    return mlir::success();
  }
  auto resultSlice = mlir::dyn_cast<SliceType>(result);
  auto resultEncoding = resultSlice
                            ? mlir::dyn_cast<EncodingType>(resultSlice.getEncoding())
                            : EncodingType();
  if (!resultEncoding || resultEncoding.getKind() != "dense")
    return emitOpError("encoded View field access produces a dense numeric slice");
  if (resultEncoding.getFamily() != denseFamilyFor(fieldType))
    return emitOpError("field slice encoding must match the declared field type");
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return emitOpError("extract requires a shaped local Value and selectors");
  if (logicalElement(getInput().getType()) != logicalElement(getResult().getType()))
    return emitOpError("extract preserves its local element type");
  return verifyProjection(*this, getInput().getType(), getIndices(),
                          getSelectors(), getResult().getType(), true);
}

mlir::LogicalResult UpdateOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      getResult().getType() != getInput().getType())
    return emitOpError("update preserves one shaped state Value");
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return mlir::failure();
  if (logicalElement(getValue().getType()) !=
      mlir::cast<ValueType>(getInput().getType()).getElementType())
    return emitOpError("updated element type must match the state Value");
  return verifyProjection(*this, getInput().getType(), getIndices(),
                          getSelectors(), getValue().getType(), true);
}

mlir::LogicalResult BitcastOp::verify() {
  auto sourceWidth = bitWidth(getInput().getType());
  auto targetWidth = bitWidth(getResult().getType());
  if (!isLocalValue(getInput().getType()) || !isLocalValue(getResult().getType()) ||
      !isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sourceWidth || !targetWidth || *sourceWidth != *targetWidth ||
      *sourceWidth < 8 || !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("bitcast preserves each numeric element's bit width and logical domain");
  return mlir::success();
}

mlir::LogicalResult UnaryOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      !isNumeric(getInput().getType()))
    return emitOpError("unary operation preserves its logical type");
  if (getKind() != "neg" && getKind() != "abs" && getKind() != "exp" && getKind() != "exp2" &&
      getKind() != "sqrt" && getKind() != "rsqrt" && getKind() != "log")
    return emitOpError("unknown unary numeric operation");
  if ((getKind() == "exp" || getKind() == "exp2" || getKind() == "sqrt" ||
       getKind() == "rsqrt" || getKind() == "log") &&
      !mlir::isa<mlir::FloatType>(logicalElement(getInput().getType())))
    return emitOpError() << getKind() << " requires floating-point elements";
  return mlir::success();
}

mlir::LogicalResult BinaryOp::verify() {
  constexpr llvm::StringLiteral allowed[] = {
      "add", "sub", "mul", "div", "mod", "and", "or",
      "xor", "shl", "shr", "max", "min", "maximum", "minimum"};
  if (!llvm::is_contained(allowed, getKind()))
    return emitOpError("unknown binary numeric operation");
  if (failed(verifyPointwise(*this, getLhs().getType(), getRhs().getType(),
                             getResult().getType())))
    return mlir::failure();
  if ((getKind() == "maximum" || getKind() == "minimum") &&
      !mlir::isa<mlir::FloatType>(logicalElement(getResult().getType())))
    return emitOpError("propagating maximum/minimum requires floating elements");
  if (getKind() == "mod" || getKind() == "and" || getKind() == "or" ||
      getKind() == "xor" || getKind() == "shl" || getKind() == "shr") {
    mlir::Type element = logicalElement(getResult().getType());
    if (!element.isIndex() && !mlir::isa<mlir::IntegerType>(element))
      return emitOpError("integer-only binary operation has non-integer elements");
  }
  return mlir::success();
}

mlir::LogicalResult CompareOp::verify() {
  constexpr llvm::StringLiteral predicates[] = {"eq", "ne", "lt",
                                                 "le", "gt", "ge"};
  if (!llvm::is_contained(predicates, getPredicate()))
    return emitOpError("unknown comparison predicate");
  if (!isLocalValue(getLhs().getType()) || !isLocalValue(getRhs().getType()) ||
      !isNumeric(getLhs().getType()) || !isNumeric(getRhs().getType()) ||
      logicalElement(getLhs().getType()) != logicalElement(getRhs().getType()))
    return emitOpError("compare operands must have matching element types");
  auto predicate = mlir::dyn_cast<mlir::IntegerType>(logicalElement(getResult().getType()));
  if (!predicate || predicate.getWidth() != 1)
    return emitOpError("compare result must have i1 elements");
  llvm::SmallVector<int64_t> expected(logicalAxes(getLhs().getType()));
  for (int64_t axis : logicalAxes(getRhs().getType()))
    if (!llvm::is_contained(expected, axis))
      expected.push_back(axis);
  if (!llvm::equal(expected, logicalAxes(getResult().getType())))
    return emitOpError(
        "compare result axes must follow lhs order then rhs-only axes");
  for (auto [axis, extent] :
       llvm::zip(logicalAxes(getResult().getType()),
                 logicalShape(getResult().getType()))) {
    std::optional<int64_t> expectedExtent;
    for (mlir::Type operand : {getLhs().getType(), getRhs().getType()}) {
      auto operandAxes = logicalAxes(operand);
      auto found = llvm::find(operandAxes, axis);
      if (found == operandAxes.end())
        continue;
      int64_t operandExtent =
          logicalShape(operand)[static_cast<size_t>(found - operandAxes.begin())];
      if (!expectedExtent || *expectedExtent == 1)
        expectedExtent = operandExtent;
      else if (operandExtent != 1 && operandExtent != *expectedExtent)
        return emitOpError(
            "compare operands disagree on a shared logical axis extent");
    }
    if (!expectedExtent || extent != *expectedExtent)
      return emitOpError(
          "compare result extent must equal its broadcast operand extent");
  }
  return mlir::success();
}

mlir::LogicalResult SelectOp::verify() {
  mlir::Type result = getResult().getType();
  if (!isLocalValue(getCondition().getType()) ||
      !logicalElement(getCondition().getType()).isInteger(1) ||
      !isNumeric(getTrueValue().getType()) || !isNumeric(getFalseValue().getType()) ||
      !isLocalValue(result) ||
      logicalElement(getTrueValue().getType()) != logicalElement(result) ||
      logicalElement(getFalseValue().getType()) != logicalElement(result))
    return emitOpError("select requires an i1 predicate and matching numeric elements");
  llvm::SmallVector<int64_t> axes, shape;
  for (mlir::Value value : {getTrueValue(), getFalseValue(), getCondition()}) {
    if (!isLocalValue(value.getType())) return emitOpError("select operands must be local Values");
    for (auto [axis, extent] : llvm::zip(logicalAxes(value.getType()), logicalShape(value.getType()))) {
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) { axes.push_back(axis); shape.push_back(extent); }
      else {
        int64_t &previous = shape[found - axes.begin()];
        if (previous == 1) previous = extent;
        else if (extent != 1 && extent != previous)
          return emitOpError("select operands disagree on a shared axis extent");
      }
    }
  }
  if (!llvm::equal(axes, logicalAxes(result)) || !llvm::equal(shape, logicalShape(result)))
    return emitOpError("select result must preserve true/false/predicate broadcast axes in order");
  return mlir::success();
}

mlir::LogicalResult CastOp::verify() {
  if (!isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("cast converts numeric elements and preserves the logical domain");
  if (logicalElement(getInput().getType()) ==
      logicalElement(getResult().getType()))
    return emitOpError("identity conversion is not a canonical cast");
  return mlir::success();
}

mlir::LogicalResult NarrowOp::verify() {
  if (!isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError(
        "narrow converts numeric elements and preserves the logical domain");
  if (getRounding() != "rne" && getRounding() != "rtz" &&
      getRounding() != "rdn" && getRounding() != "rup" &&
      getRounding() != "dynamic")
    return emitOpError("unknown numeric narrowing rounding mode");
  auto inputWidth = bitWidth(getInput().getType());
  auto resultWidth = bitWidth(getResult().getType());
  if (!inputWidth || !resultWidth || *resultWidth >= *inputWidth)
    return emitOpError("narrow must reduce the element bit width");
  if (getSaturate() &&
      !mlir::isa<mlir::IntegerType>(logicalElement(getResult().getType())))
    return emitOpError("saturation is defined only for an integer result");
  return mlir::success();
}

mlir::LogicalResult MacGroupsOp::verify() {
  if (getGroup() <= 0 || getOverflow() != "wrap")
    return emitOpError("mac_groups requires positive group and wrap overflow");
  if (!mlir::isa<ValueType>(getLhs().getType()) ||
      !mlir::isa<ValueType>(getRhs().getType()) ||
      !mlir::isa<ValueType>(getResult().getType()) ||
      !isNumeric(getLhs().getType()) || !isNumeric(getRhs().getType()) ||
      !isNumeric(getResult().getType()))
    return emitOpError("mac_groups consumes and produces shaped numeric Values");
  auto lhs = mlir::cast<ValueType>(getLhs().getType());
  auto rhs = mlir::cast<ValueType>(getRhs().getType());
  auto result = mlir::cast<ValueType>(getResult().getType());
  auto lhsAxes = lhs.getAxisIds().asArrayRef();
  auto rhsAxes = rhs.getAxisIds().asArrayRef();
  if (lhsAxes.empty() || rhsAxes.empty() || lhsAxes.back() != rhsAxes.back())
    return emitOpError("mac_groups requires one final common grouped axis");
  int64_t groupedAxis = lhsAxes.back();
  llvm::SmallVector<int64_t> inputAxes(lhsAxes.drop_back());
  llvm::SmallVector<int64_t> inputShape(
      lhs.getShape().asArrayRef().drop_back());
  for (auto [axis, extent] : llvm::zip(rhsAxes.drop_back(),
                                      rhs.getShape().asArrayRef().drop_back())) {
    auto found = llvm::find(inputAxes, axis);
    if (found == inputAxes.end()) {
      inputAxes.push_back(axis);
      inputShape.push_back(extent);
      continue;
    }
    size_t position = static_cast<size_t>(found - inputAxes.begin());
    if (inputShape[position] != extent)
      return emitOpError(
          "mac_groups operands disagree on a shared logical axis extent");
  }
  inputAxes.push_back(groupedAxis);
  inputShape.push_back(lhs.getShape()[lhs.getShape().size() - 1]);
  if (inputShape.back() != rhs.getShape()[rhs.getShape().size() - 1])
    return emitOpError(
        "mac_groups operands disagree on the grouped axis extent");
  if (inputAxes.empty() ||
      !llvm::equal(result.getAxisIds().asArrayRef(), inputAxes) ||
      result.getShape().size() != inputShape.size())
    return emitOpError(
        "mac_groups result axes must follow lhs order then rhs-only axes");
  for (size_t index = 0; index + 1 < inputShape.size(); ++index)
    if (result.getShape()[index] != inputShape[index])
      return emitOpError("mac_groups preserves non-grouped extents");
  if (inputShape.back() <= 0 || inputShape.back() % getGroup() ||
      result.getShape()[result.getShape().size() - 1] !=
          inputShape.back() / getGroup())
    return emitOpError("mac_groups divides its final logical extent by group");
  return mlir::success();
}

mlir::LogicalResult WidenOp::verify() {
  if (!sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("widen preserves the logical domain");
  auto inputWidth = bitWidth(getInput().getType());
  auto resultWidth = bitWidth(getResult().getType());
  mlir::Type inputElement = logicalElement(getInput().getType());
  mlir::Type resultElement = logicalElement(getResult().getType());
  const bool integerToFloat = mlir::isa<mlir::IntegerType>(inputElement) &&
                              mlir::isa<mlir::FloatType>(resultElement);
  if (!inputWidth || !resultWidth ||
      (integerToFloat ? *resultWidth < *inputWidth
                      : *resultWidth <= *inputWidth))
    return emitOpError(
        "widen must increase width or convert an integer to no-narrower float");
  return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
  if (getKind() != "add" && getKind() != "max" && getKind() != "min")
    return emitOpError("reduce kind must be add, max, or min");
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      !isLocalValue(getResult().getType()) || !isNumeric(getInput().getType()) ||
      !isNumeric(getResult().getType()))
    return emitOpError("reduce consumes and produces numeric Values");
  auto input = mlir::cast<ValueType>(getInput().getType());
  auto inputShape = input.getShape().asArrayRef();
  auto inputAxes = input.getAxisIds().asArrayRef();
  auto resultShape = logicalShape(getResult().getType());
  auto resultAxes = logicalAxes(getResult().getType());
  if (getAxis() < 0 || static_cast<size_t>(getAxis()) >= inputShape.size() ||
      resultShape.size() + 1 != inputShape.size() ||
      logicalElement(getResult().getType()) != input.getElementType())
    return emitOpError("reduce result must remove exactly its selected axis");
  size_t resultIndex = 0;
  for (size_t inputIndex = 0; inputIndex < inputShape.size(); ++inputIndex) {
    if (inputIndex == static_cast<size_t>(getAxis()))
      continue;
    if (resultShape[resultIndex] != inputShape[inputIndex] ||
        resultAxes[resultIndex] != inputAxes[inputIndex])
      return emitOpError("reduce must preserve every non-reduced logical axis");
    ++resultIndex;
  }
  return mlir::success();
}

mlir::LogicalResult Fold2Op::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      !mlir::isa<ValueType>(getResult().getType()))
    return emitOpError("fold2 consumes and produces shaped Values");
  auto input = mlir::cast<ValueType>(getInput().getType());
  auto result = mlir::cast<ValueType>(getResult().getType());
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  llvm::ArrayRef<int64_t> inputShape = input.getShape().asArrayRef();
  llvm::ArrayRef<int64_t> resultShape = result.getShape().asArrayRef();
  if (inputShape.empty() || inputShape.size() != resultShape.size() ||
      inputShape.back() <= 0 || inputShape.back() % 2 ||
      resultShape.back() != inputShape.back() / 2 ||
      !inputElement || !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 ||
      input.getAxisIds() != result.getAxisIds() ||
      !llvm::equal(inputShape.drop_back(), resultShape.drop_back()))
    return emitOpError("fold2 halves an even final logical extent");
  return mlir::success();
}

mlir::LogicalResult ReduceRegionOp::verify() {
  return weft::verifyReductionRegion<ValueType, ReduceYieldOp>(*this, getAxis(), getCombine(),
      [](mlir::Operation *op) {
        return mlir::isa<ConstantOp, mlir::arith::ConstantOp, UnaryOp, BinaryOp,
            CompareOp, SelectOp, CastOp, BitcastOp, WidenOp, NarrowOp>(op);
      });
}

mlir::LogicalResult ScanOp::verify() {
  return weft::verifyReductionRegion<ValueType, ScanYieldOp>(*this, getAxis(), getCombine(),
      [](mlir::Operation *op) {
        return mlir::isa<ConstantOp, mlir::arith::ConstantOp, UnaryOp, BinaryOp,
            CompareOp, SelectOp, CastOp, BitcastOp, WidenOp, NarrowOp>(op);
      }, true);
}

mlir::LogicalResult DotOp::verify() {
  if (getOver().size() != 1 || logicalAxes(getLhs().getType()).empty() ||
      logicalAxes(getRhs().getType()).empty() ||
      getOver()[0] != logicalAxes(getLhs().getType()).back() ||
      getOver()[0] != logicalAxes(getRhs().getType()).back())
    return emitOpError("dot reduces the final common logical axis");
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult ContractOp::verify() {
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult OuterContractOp::verify() {
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : getOver())
    reduction.insert(axis);
  for (int64_t axis : logicalAxes(getLhs().getType()))
    if (!reduction.contains(axis) &&
        llvm::is_contained(logicalAxes(getRhs().getType()), axis))
      return emitOpError(
          "outer_contract requires disjoint lhs and rhs free axes");
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult LookupOp::verify() {
  if (getBounds() != "in_bounds")
    return emitOpError("current canonical lookup requires bounds = in_bounds");
  mlir::Type indexElement = logicalElement(getIndices().getType());
  auto index = mlir::dyn_cast<mlir::IntegerType>(indexElement);
  if (!mlir::isa<ValueType>(getTable().getType()) ||
      !isNumeric(getTable().getType()) ||
      (!indexElement.isIndex() && (!index || index.isSigned())))
    return emitOpError(
        "lookup requires a numeric local table and unsigned integer indices");
  if (!sameDomain(getIndices().getType(), getResult().getType()))
    return emitOpError("lookup result must follow the index domain");
  if (logicalElement(getTable().getType()) != logicalElement(getResult().getType()))
    return emitOpError("lookup result element type must match the table element type");
  return mlir::success();
}

mlir::FailureOr<llvm::SmallVector<FieldInterleaveSpan>>
weft::kernel::fieldInterleaveSpans(
    int64_t storageBits, llvm::ArrayRef<int64_t> fieldBitOffsets,
    llvm::ArrayRef<int64_t> fieldStorageBits, llvm::ArrayRef<int64_t> padding,
    llvm::ArrayRef<int64_t> fieldChunks) {
  if (storageBits <= 0 || storageBits % 8 ||
      fieldChunks.size() != fieldStorageBits.size() ||
      fieldBitOffsets.size() != fieldStorageBits.size() ||
      llvm::any_of(fieldChunks, [](int64_t n) { return n <= 0; }))
    return mlir::failure();
  if (llvm::all_of(fieldChunks, [](int64_t n) { return n == 1; }))
    return llvm::SmallVector<FieldInterleaveSpan>{{0, storageBits / 8, 1}};
  llvm::SmallVector<FieldInterleaveSpan> spans;
  for (auto [offset, bits, chunk] :
       llvm::zip(fieldBitOffsets, fieldStorageBits, fieldChunks)) {
    if (offset < 0 || offset % 8 || bits <= 0 || bits % 8 ||
        bits / 8 % chunk || offset > storageBits - bits)
      return mlir::failure();
    spans.push_back({offset / 8, bits / 8, chunk});
  }
  if (padding.size() % 3)
    return mlir::failure();
  for (size_t i = 0; i < padding.size(); i += 3) {
    if (padding[i] < 0 || padding[i] % 8 || padding[i + 1] <= 0 ||
        padding[i + 1] % 8 || padding[i] > storageBits - padding[i + 1])
      return mlir::failure();
    spans.push_back({padding[i] / 8, padding[i + 1] / 8, 1});
  }
  llvm::sort(spans, [](const auto &a, const auto &b) {
    return std::tie(a.byteOffset, a.byteCount, a.chunkBytes) <
           std::tie(b.byteOffset, b.byteCount, b.chunkBytes);
  });
  llvm::SmallVector<FieldInterleaveSpan> result;
  int64_t end = 0;
  for (const auto &span : spans) {
    if (!result.empty() && span.byteOffset == result.back().byteOffset &&
        span.byteCount == result.back().byteCount &&
        span.chunkBytes == result.back().chunkBytes)
      continue;
    if (span.byteOffset != end)
      return mlir::failure();
    result.push_back(span);
    end += span.byteCount;
  }
  if (end != storageBits / 8)
    return mlir::failure();
  return result;
}

mlir::LogicalResult InterleaveOp::verify() {
  if (getRows() <= 0 || !sameDomain(getView().getType(), getResult().getType()))
    return emitOpError("interleave requires positive rows and preserves the View domain");
  if (mlir::cast<EncodingType>(getResult().getType().getEncoding()).getKind() !=
      "derived_instance")
    return emitOpError("interleave in derive produces a concrete derived instance");
  auto derive = mlir::dyn_cast_or_null<DeriveOp>(getOperation()->getParentOp());
  if (!derive)
    return emitOpError("interleave is valid only inside a derive body");
  auto resultEncoding =
      mlir::cast<EncodingType>(getResult().getType().getEncoding());
  if (derive.getParameterValues().size() != 1 ||
      derive.getParameterValues()[0] != getRows() ||
      resultEncoding.getParameters().asArrayRef() !=
          derive.getParameterValues() ||
      resultEncoding.getLayoutIdentity() != derive.getLayoutIdentity())
    return emitOpError(
        "interleave rows and result identity must match its derive instance");
  auto source = mlir::SymbolTable::lookupNearestSymbolFrom<EncodingDeclOp>(
      *this, mlir::StringAttr::get(getContext(), derive.getSourceFamily()));
  if (!source || mlir::failed(fieldInterleaveSpans(
                     source.getStorageBits(), source.getFieldBitOffsets(),
                     source.getFieldStorageBits(), source.getPadding(),
                     getFieldChunks())))
    return emitOpError(
        "field interleave chunks must partition the complete byte-addressable source storage without conflicting aliases");
  return mlir::success();
}

void WEFTKernelDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
}
