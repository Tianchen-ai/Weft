#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

namespace weft::riscv_emission {

std::string Emitter::vectorSuffix(mlir::Value value) const {
  riscv::LayoutAttr layout = layoutOf(value);
  int64_t sew = layout.getSew();
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  char category = 'i';
  if (usesRawFloatStorage(element))
    category = 'u';
  else if (mlir::isa<mlir::FloatType>(element))
    category = 'f';
  else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
    category = integer.isUnsigned() ? 'u' : 'i';
  else if (element.isIndex())
    category = 'u';
  return std::string(1, category) + std::to_string(sew) +
         lmulSpelling(layout.getLmulEighths());
}

std::string Emitter::vectorType(mlir::Value value) const {
  std::string suffix = vectorSuffix(value);
  char category = suffix.front();
  std::string prefix = category == 'f' ? "vfloat" : category == 'u' ? "vuint"
                                                                       : "vint";
  return prefix + suffix.substr(1) + "_t";
}

int64_t Emitter::streamPartCount(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    return std::max<int64_t>(1, product(layout.getTimeFactors()));
  return 1;
}

std::optional<Binding::Kind>
Emitter::selectedBindingKind(mlir::Value value) const {
  auto layout = layoutOf(value);
  if (!layout)
    return Binding::Kind::Scalar;
  llvm::StringRef carrier = layout.getCarrier();
  if (carrier == "rvv")
    return Binding::Kind::Vector;
  if (carrier == "scalar" && scalarPartCount(value) > 1)
    return Binding::Kind::ScalarTuple;
  if (carrier == "scalar")
    return Binding::Kind::Scalar;
  if (carrier == "local")
    return Binding::Kind::LocalArray;
  return std::nullopt;
}

int64_t Emitter::registerPartCount(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    return std::max<int64_t>(1, product(layout.getReplicaFactors()) / layout.getReplicaGroup());
  return 1;
}

int64_t Emitter::scalarPartCount(mlir::Value value) const {
  return registerPartCount(value) * streamPartCount(value);
}

std::optional<llvm::SmallVector<int64_t, 4>>
Emitter::scalarCoordinates(mlir::Value value, size_t part) const {
  auto layout = layoutOf(value);
  if (!layout || layout.getCarrier() != "scalar") return std::nullopt;
  auto geometry = riscv::RegisterCoordinates::get(layout);
  if (!geometry || part >= static_cast<size_t>(geometry->parts())) return std::nullopt;
  auto coordinates = geometry->logical(part);
  return llvm::SmallVector<int64_t, 4>(coordinates->begin(), coordinates->end());
}

llvm::SmallVector<int64_t, 4>
Emitter::registerAxesFor(mlir::Value value) const {
  llvm::SmallVector<int64_t, 4> result;
  if (auto layout = layoutOf(value))
    for (auto [axis, factor] : llvm::zip(layout.getAxisIds().asArrayRef(),
                                         layout.getReplicaFactors().asArrayRef()))
      if (factor > 1)
        result.push_back(axis);
  return result;
}

llvm::SmallVector<int64_t, 4>
Emitter::registerExtentsFor(mlir::Value value) const {
  llvm::SmallVector<int64_t, 4> result;
  if (auto layout = layoutOf(value))
    for (int64_t factor : layout.getReplicaFactors().asArrayRef())
      if (factor > 1)
        result.push_back(factor);
  if (!result.empty())
    return result;
  result.assign(registerAxesFor(value).size(), 1);
  if (result.size() == 1)
    result.front() = registerPartCount(value);
  return result;
}

std::optional<llvm::SmallVector<int64_t, 4>>
Emitter::registerCoordinates(mlir::Value value, size_t registerPart) const {
  llvm::SmallVector<int64_t, 4> axes = registerAxesFor(value);
  llvm::SmallVector<int64_t, 4> extents = registerExtentsFor(value);
  if (axes.size() != extents.size())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> coordinates(axes.size(), 0);
  int64_t remaining = static_cast<int64_t>(registerPart);
  for (int64_t index = static_cast<int64_t>(axes.size()) - 1; index >= 0;
       --index) {
    if (extents[index] <= 0)
      return std::nullopt;
    coordinates[index] = remaining % extents[index];
    remaining /= extents[index];
  }
  if (remaining != 0)
    return std::nullopt;
  return coordinates;
}

std::optional<size_t> Emitter::projectPart(mlir::Value source,
                                           mlir::Value result,
                                           size_t resultPart) const {
  auto from = mlir::dyn_cast<riscv::ValueType>(source.getType());
  auto to = mlir::dyn_cast<riscv::ValueType>(result.getType());
  auto projected = riscv::projectCarrierPart(from, to, resultPart);
  return projected ? std::optional<size_t>(*projected) : std::nullopt;
}

std::optional<size_t> Emitter::mappedPart(mlir::Operation *operation,
                                          size_t operand,
                                          size_t resultPart) const {
  if (operand >= operation->getNumOperands() || operation->getNumResults() != 1)
    return std::nullopt;
  return projectPart(operation->getOperand(operand), operation->getResult(0),
                     resultPart);
}

std::optional<size_t>
Emitter::projectRegisterPart(mlir::Value source, mlir::Value result,
                             size_t resultRegisterPart) const {
  auto resultCoordinates = registerCoordinates(result, resultRegisterPart);
  llvm::SmallVector<int64_t, 4> resultAxes = registerAxesFor(result);
  llvm::SmallVector<int64_t, 4> sourceAxes = registerAxesFor(source);
  llvm::SmallVector<int64_t, 4> sourceExtents = registerExtentsFor(source);
  if (!resultCoordinates || resultCoordinates->size() != resultAxes.size() ||
      sourceAxes.size() != sourceExtents.size())
    return std::nullopt;
  int64_t sourceRegister = 0;
  for (auto [axis, extent] : llvm::zip(sourceAxes, sourceExtents)) {
    auto found = llvm::find(resultAxes, axis);
    if (found == resultAxes.end() || extent <= 0)
      return std::nullopt;
    size_t position = static_cast<size_t>(found - resultAxes.begin());
    if ((*resultCoordinates)[position] >= extent)
      return std::nullopt;
    sourceRegister = sourceRegister * extent + (*resultCoordinates)[position];
  }
  if (sourceRegister < 0 || sourceRegister >= registerPartCount(source))
    return std::nullopt;
  return static_cast<size_t>(sourceRegister);
}

std::optional<std::string>
Emitter::extractVectorLane(mlir::Value source, const Binding &binding,
                           size_t sourcePart, int64_t laneOffset,
                           LaneExtractionCache &localExtractions,
                           std::string *failureReason) {
  auto reject = [&](llvm::StringRef reason) -> std::optional<std::string> {
    if (failureReason)
      *failureReason = reason.str();
    return std::nullopt;
  };
  if (binding.kind != Binding::Kind::Vector ||
      sourcePart >= binding.parts.size())
    return reject("source binding has no selected RVV vector part");
  if (laneOffset < 0 || laneOffset >= physicalLanes(source))
    return reject("selected RVV lane is outside the typed lane extent");
  const auto cacheKey = std::make_pair(sourcePart, laneOffset);
  if (auto found = localExtractions.find(cacheKey); found != localExtractions.end())
    return found->second;

  const std::string suffix = vectorSuffix(source);
  const std::string shifted = laneOffset == 0 ? binding.parts[sourcePart] :
      "__riscv_vslidedown_vx_" + suffix + "(" + binding.parts[sourcePart] +
      ", " + std::to_string(laneOffset) + ", " + partVL(source, sourcePart) +
      ")";
  mlir::Type element = riscv_internal::logicalElement(source.getType());
  auto sourceLayout = layoutOf(source);
  const unsigned physicalSew =
      sourceLayout ? static_cast<unsigned>(sourceLayout.getSew())
                   : riscv_internal::logicalBitWidth(source.getType());
  std::string expression;
  if (usesFloatingRegisters(element))
    expression = "__riscv_vfmv_f_s_" + suffix + "_f" +
                 std::to_string(physicalSew) + "(" + shifted + ")";
  else {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    if (!integer && !usesRawFloatStorage(element))
      return reject("source lane element is neither integer nor floating point");
    expression = "__riscv_vmv_x_s_" + suffix + "_" +
                 std::string(usesRawFloatStorage(element) || integer.isUnsigned() ? "u" : "i") +
                 std::to_string(physicalSew) + "(" + shifted + ")";
  }
  auto type = scalarCType(element);
  if (!type)
    return reject("source lane element has no scalar C spelling");
  std::string materialized = fresh("layout_convert");
  line(*type + " " + materialized + " = " + expression + ";");
  localExtractions[cacheKey] = materialized;
  return materialized;
}

std::optional<std::string> Emitter::extractLaneForRegisterBroadcast(
    mlir::Value source, mlir::Value result, size_t resultPart,
    const Binding &binding, LaneExtractionCache &localExtractions,
    std::string *failureReason) {
  auto reject = [&](llvm::StringRef reason) -> std::optional<std::string> {
    if (failureReason)
      *failureReason = reason.str();
    return std::nullopt;
  };
  if (binding.kind != Binding::Kind::Vector)
    return reject("source binding is not an RVV vector");
  auto input = mlir::dyn_cast<riscv::ValueType>(source.getType());
  auto output = mlir::dyn_cast<riscv::ValueType>(result.getType());
  if (!input || !output || output.getLayout().getCarrier() != "scalar")
    return reject("lane extraction requires an explicitly selected scalar representation");
  auto from = riscv::RegisterCoordinates::get(input.getLayout());
  auto to = riscv::RegisterCoordinates::get(output.getLayout());
  auto logical = to ? to->logical(resultPart) : std::nullopt;
  if (!from || !to || to->lanes() != 1 || !logical)
    return reject("lane extraction has no complete typed scalar coordinate");
  // Both time and replica factors contribute to a scalar tuple coordinate.
  // Projecting just replica IDs loses axes that became scalar issue time, such
  // as a [row, column] tile whose rows were vector-register replicas.
  llvm::SmallVector<int64_t> point;
  for (auto [position, axis] : llvm::enumerate(input.getAxisIds().asArrayRef())) {
    auto found = llvm::find(output.getAxisIds().asArrayRef(), axis);
    if (input.getShape()[position] == 1) point.push_back(0);
    else if (found != output.getAxisIds().asArrayRef().end())
      point.push_back((*logical)[found - output.getAxisIds().asArrayRef().begin()]);
    else return reject("lane extraction drops a non-singleton logical coordinate");
  }
  auto location = from->locate(point);
  if (!location || location->part < 0 || location->part >= static_cast<int64_t>(binding.parts.size()))
    return reject("projected source vector part is absent");
  return extractVectorLane(source, binding, location->part, location->lane,
                           localExtractions, failureReason);
}

mlir::FailureOr<Binding>
Emitter::projectBinding(mlir::Value source, mlir::Value result,
                        const Binding &binding) {
  if (binding.kind == Binding::Kind::Fragment) {
    if (source.getType() != result.getType())
      return mlir::failure();
    return binding;
  }
  if (binding.kind == Binding::Kind::LocalArray)
    return binding;
  if (binding.kind == Binding::Kind::Scalar)
    return binding;
  if (binding.kind != Binding::Kind::ScalarTuple &&
      binding.kind != Binding::Kind::Vector)
    return binding;
  const int64_t parts = binding.kind == Binding::Kind::Vector
                            ? vectorPartCount(result)
                            : scalarPartCount(result);
  if (parts <= 0)
    return mlir::failure();
  Binding projected;
  projected.kind = binding.kind;
  for (int64_t part = 0; part < parts; ++part) {
    auto sourcePart = projectPart(source, result, part);
    if (!sourcePart || *sourcePart >= binding.parts.size())
      return mlir::failure();
    projected.parts.push_back(binding.parts[*sourcePart]);
  }
  if (projected.kind == Binding::Kind::ScalarTuple && parts == 1) {
    projected.kind = Binding::Kind::Scalar;
    projected.scalar = projected.parts.front();
    projected.parts.clear();
  }
  return projected;
}

mlir::LogicalResult
Emitter::assignBinding(mlir::Operation *operation, mlir::Value targetValue,
                       Binding &target, mlir::Value sourceValue,
                       const Binding &source, llvm::StringRef mismatch) {
  if (targetValue.getType() != sourceValue.getType())
    return fail(operation,
                mismatch.str() +
                    " (physical IR omitted a typed convert_layout operation)");
  if (target.kind == Binding::Kind::Memory && source.kind == Binding::Kind::Memory) {
    auto type = mlir::dyn_cast<riscv::MemDescType>(targetValue.getType());
    if (!type || type.getAddressClass() != "pointer")
      return fail(operation, "memory carry requires an explicit pointer origin");
    line(target.memory.name + " = " + source.memory.name + ";");
    return mlir::success();
  }
  if (target.kind == Binding::Kind::Fragment && source.kind == Binding::Kind::Fragment) {
    auto fragment = mlir::cast<riscv::FragmentType>(targetValue.getType());
    if (target.fragmentFamily != source.fragmentFamily ||
        target.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)) ||
        source.parts.size() != target.parts.size())
      return fail(operation, mismatch.str() + " (fragment register carriers disagree)");
    for (auto [destination, value] : llvm::zip(target.parts, source.parts))
      line(destination + " = " + value + ";");
    return mlir::success();
  }
  if (source.kind == Binding::Kind::LocalArray &&
      (target.kind == Binding::Kind::None ||
       target.kind == Binding::Kind::LocalArray)) {
    if (target.kind == Binding::Kind::None || target.scalar.empty()) {
      target = source;
      return mlir::success();
    }
    if (target.scalar == source.scalar)
      return mlir::success();
    return fail(operation,
                mismatch.str() + " (distinct local arrays require an explicit copy op)");
  }
  if (target.kind == Binding::Kind::Scalar &&
      source.kind == Binding::Kind::Scalar) {
    line(target.scalar + " = " + source.scalar + ";");
    return mlir::success();
  }
  if (target.kind == Binding::Kind::Window &&
      source.kind == Binding::Kind::Window) {
    if (target.windowFamily != source.windowFamily ||
        target.windowLhs.size() != source.windowLhs.size() ||
        target.windowRhs.size() != source.windowRhs.size() ||
        target.windowValidity.size() != source.windowValidity.size())
      return fail(operation,
                  mismatch.str() + " (physical window schemas disagree)");
    for (auto [destination, expression] :
         llvm::zip(target.windowLhs, source.windowLhs))
      line(destination + " = " + expression + ";");
    for (auto [destination, expression] :
         llvm::zip(target.windowRhs, source.windowRhs))
      line(destination + " = " + expression + ";");
    for (auto [destination, expression] :
         llvm::zip(target.windowValidity, source.windowValidity))
      line(destination + " = " + expression + ";");
    return mlir::success();
  }
  if ((target.kind != Binding::Kind::Vector &&
       target.kind != Binding::Kind::ScalarTuple) ||
      target.kind != source.kind)
    return fail(operation,
                mismatch.str() + " (target/source binding kinds disagree: " +
                    std::to_string(static_cast<int>(target.kind)) + "/" +
                    std::to_string(static_cast<int>(source.kind)) + ")");
  const int64_t parts = target.kind == Binding::Kind::Vector
                            ? vectorPartCount(targetValue)
                            : scalarPartCount(targetValue);
  if (parts != static_cast<int64_t>(target.parts.size()))
    return fail(operation,
                mismatch.str() + " (target part count disagrees)");
  for (int64_t part = 0; part < parts; ++part) {
    auto sourcePart = projectPart(sourceValue, targetValue, part);
    if (!sourcePart || *sourcePart >= source.parts.size())
      return fail(operation,
                  mismatch.str() + " (source part projection disagrees)");
    line(target.parts[part] + " = " + source.parts[*sourcePart] + ";");
  }
  return mlir::success();
}

int64_t Emitter::physicalLanes(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    return std::max<int64_t>(1, product(layout.getLaneFactors()) * layout.getReplicaGroup());
  return 1;
}

int64_t Emitter::laneAxisFor(mlir::Value value) const {
  if (auto layout = layoutOf(value))
    for (auto [axis, factor] : llvm::zip(layout.getAxisIds().asArrayRef(),
                                         layout.getLaneFactors().asArrayRef()))
      if (factor > 1)
        return axis;
  return 0;
}

std::string Emitter::partOffset(mlir::Value value, size_t part) const {
  const int64_t streams = streamPartCount(value);
  int64_t stream = static_cast<int64_t>(part % streams);
  auto layout = layoutOf(value);
  int64_t laneCoordinate = 0;
  if (layout) {
    auto axes = layout.getAxisIds().asArrayRef();
    auto extents = layout.getTimeFactors().asArrayRef();
    for (int64_t index = static_cast<int64_t>(axes.size()) - 1; index >= 0;
         --index) {
      int64_t coordinate = stream % extents[index];
      stream /= extents[index];
      if (axes[index] == laneAxisFor(value))
        laneCoordinate = coordinate;
    }
  }
  return std::to_string(laneCoordinate * physicalLanes(value));
}

std::optional<int64_t> Emitter::timeCoordinate(mlir::Value value, size_t part,
                                               int64_t axis) const {
  auto layout = layoutOf(value);
  const int64_t streams = streamPartCount(value);
  if (!layout || streams <= 0)
    return std::nullopt;
  int64_t stream = static_cast<int64_t>(part % streams);
  int64_t result = 0;
  bool found = false;
  for (int64_t index = static_cast<int64_t>(layout.getAxisIds().size()) - 1;
       index >= 0; --index) {
    const int64_t factor = layout.getTimeFactors()[index];
    if (factor <= 0)
      return std::nullopt;
    const int64_t coordinate = stream % factor;
    stream /= factor;
    if (layout.getAxisIds()[index] == axis) {
      result = coordinate;
      found = true;
    }
  }
  if (stream != 0 || !found)
    return std::nullopt;
  return result;
}

std::string Emitter::partVL(mlir::Value value, size_t part) const {
  const int64_t laneAxis = laneAxisFor(value);
  if (auto layout = layoutOf(value); layout && layout.getValidity() == "full")
    return std::to_string(physicalLanes(value));
  if (auto layout = layoutOf(value);
      layout && (layout.getReplicaGroup() > 1 ||
                 llvm::count_if(layout.getLaneFactors().asArrayRef(),
                                [](int64_t n) { return n > 1; }) > 1)) {
    // A cartesian tile's active rectangle need not be a lane prefix. Local
    // pointwise consumers execute the complete carrier; the explicit boundary
    // repack restores row slices before memory observes the active domain.
    return std::to_string(physicalLanes(value));
  }
  auto scope = axisScopes.find(laneAxis);
  if (scope == axisScopes.end() || scope->second.empty())
    return std::to_string(physicalLanes(value));
  const std::string active = scope->second.back().active;
  auto layout = layoutOf(value);
  auto geometry = riscv::RegisterCoordinates::get(layout);
  auto coordinates = geometry ? geometry->logical(part) : std::nullopt;
  auto position = layout ? llvm::find(layout.getAxisIds().asArrayRef(), laneAxis)
                         : llvm::ArrayRef<int64_t>::iterator();
  if (!coordinates || position == layout.getAxisIds().asArrayRef().end()) return {};
  const std::string offset = std::to_string((*coordinates)[position - layout.getAxisIds().asArrayRef().begin()]);
  const std::string lanes = std::to_string(physicalLanes(value));
  return "((" + active + " > " + offset + ") ? ((" + active + " - " +
         offset + " < " + lanes + ") ? " + active + " - " + offset + " : " +
         lanes + ") : 0)";
}

int64_t Emitter::vectorPartCount(mlir::Value value) const {
  return registerPartCount(value) * streamPartCount(value);
}

mlir::FailureOr<Binding>
Emitter::makeVector(mlir::Value value, llvm::StringRef prefix,
                    std::optional<Binding> initial,
                    mlir::Value initialValue) {
  Binding result;
  result.kind = Binding::Kind::Vector;
  const int64_t count = vectorPartCount(value);
  const std::string type = vectorType(value);
  const std::string suffix = vectorSuffix(value);
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  auto elementType = scalarCType(element);
  if (!elementType) {
    kernel.emitError("intrinsic-C emission does not support the selected vector element type");
    return mlir::failure();
  }
  for (int64_t index = 0; index < count; ++index) {
    std::string name = fresh(prefix);
    std::string expression;
    if (initial) {
      if (initial->kind == Binding::Kind::Scalar) {
        if (usesFloatingRegisters(element))
          expression = "__riscv_vfmv_v_f_" + suffix + "((" +
                       *elementType + ")(" + initial->scalar + "), " +
                       partVL(value, index) + ")";
        else
          expression = "__riscv_vmv_v_x_" + suffix + "((" +
                       *elementType + ")(" + initial->scalar + "), " +
                       partVL(value, index) + ")";
      } else if (initial->kind == Binding::Kind::Vector ||
                 initial->kind == Binding::Kind::ScalarTuple) {
        if (!initialValue) {
          kernel.emitError(
              "vector initialization from a structured value requires its physical source");
          return mlir::failure();
        }
        auto sourcePart = projectPart(initialValue, value, index);
        if (!sourcePart || *sourcePart >= initial->parts.size()) {
          kernel.emitError(
              "vector initializer cannot be projected to the selected result mapping");
          return mlir::failure();
        }
        if (initial->kind == Binding::Kind::Vector)
          expression = initial->parts[*sourcePart];
        else if (usesFloatingRegisters(element))
          expression = "__riscv_vfmv_v_f_" + suffix + "((" +
                       *elementType + ")(" + initial->parts[*sourcePart] +
                       "), " + partVL(value, index) + ")";
        else
          expression = "__riscv_vmv_v_x_" + suffix + "((" +
                       *elementType + ")(" + initial->parts[*sourcePart] +
                       "), " + partVL(value, index) + ")";
      }
    }
    if (expression.empty())
      expression = "(" + type + "){0}";
    line(type + " " + name + " = " + expression + ";");
    result.parts.push_back(std::move(name));
  }
  return result;
}

mlir::FailureOr<Binding>
Emitter::declareMutableBinding(mlir::Value value, llvm::StringRef prefix) {
  if (auto pointer = mlir::dyn_cast<riscv::MemDescType>(value.getType());
      pointer && pointer.getAddressClass() == "pointer") {
    auto element = denseElementType(pointer.getEncoding());
    auto type = element ? storageCType(*element) : std::nullopt;
    if (!type) return mlir::failure();
    Binding result;
    result.kind = Binding::Kind::Memory;
    result.memory.encoding = pointer.getEncoding();
    result.memory.elementType = *element;
    result.memory.isConst = pointer.getAccess() != "write" && pointer.getAccess() != "readwrite";
    result.memory.name = fresh(prefix);
    line(std::string(result.memory.isConst ? "const " : "") + *type + " *" + result.memory.name + ";");
    return result;
  }
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(value.getType())) {
    auto type = fragmentVectorType(fragment);
    if (!type)
      return mlir::failure();
    Binding result;
    result.kind = Binding::Kind::Fragment;
    result.fragmentFamily = fragment.getFamily().str();
    for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
      result.parts.push_back(fresh(prefix));
      line(*type + " " + result.parts.back() + ";");
    }
    return result;
  }
  if (Binding existing = bindings.lookup(value);
      existing.kind == Binding::Kind::LocalArray)
    return existing;
  std::optional<Binding::Kind> selected = selectedBindingKind(value);
  if (!selected)
    return mlir::failure();

  Binding result;
  result.kind = *selected;
  mlir::Type element = riscv_internal::logicalElement(value.getType());
  if (result.kind == Binding::Kind::Scalar) {
    auto type = scalarCType(element);
    if (!type)
      return mlir::failure();
    result.scalar = fresh(prefix);
    line(*type + " " + result.scalar + ";");
    return result;
  }

  int64_t count = 0;
  std::string type;
  if (result.kind == Binding::Kind::Vector) {
    count = vectorPartCount(value);
    type = vectorType(value);
  } else if (result.kind == Binding::Kind::ScalarTuple) {
    count = scalarPartCount(value);
    auto scalarType = scalarCType(element);
    if (!scalarType)
      return mlir::failure();
    type = *scalarType;
  } else {
    return mlir::failure();
  }
  if (count <= 0)
    return mlir::failure();
  for (int64_t part = 0; part < count; ++part) {
    std::string name = fresh(prefix);
    line(type + " " + name + ";");
    result.parts.push_back(std::move(name));
  }
  return result;
}

mlir::LogicalResult
Emitter::compileConvertLayout(riscv::ConvertLayoutOp conversion) {
  Binding input = bindings.lookup(conversion.getInput());
  if (input.kind == Binding::Kind::None)
    return fail(conversion, "layout conversion input has no emitted value");
  if (input.kind == Binding::Kind::Field)
    return fail(conversion, "encoded conversion requires an explicit field_read");
  llvm::StringRef kind = conversion.getConversion().getKind();
  if (kind == "lane_repack") {
    auto slices = riscv::rvvLayoutRepackSlices(conversion.getInput().getType(),
                                               conversion.getResult().getType());
    if (!slices || input.kind != Binding::Kind::Vector ||
        instructionOf(conversion) != "rvv.lane-repack")
      return fail(conversion, "lane repack has no selected ordered numeric slice contract");
    Binding result;
    result.kind = Binding::Kind::Vector;
    const std::string from = vectorSuffix(conversion.getInput());
    const std::string to = vectorSuffix(conversion.getResult());
    const auto sourceLayout = conversion.getInput().getType().getLayout();
    const auto resultLayout = conversion.getResult().getType().getLayout();
    auto target = conversion->getParentOfType<riscv::KernelOp>().getTarget();
    const bool splitRegisters = riscv::isRVVSubregisterRepack(target,
        conversion.getInput().getType(), conversion.getResult().getType());
    const bool joinRegisters = riscv::isRVVSubregisterRepack(target,
        conversion.getResult().getType(), conversion.getInput().getType());
    for (const auto &slice : *slices) {
      if (slice.sourcePart >= static_cast<int64_t>(input.parts.size()))
        return fail(conversion, "lane repack source part is absent");
      std::string expression = input.parts[slice.sourcePart];
      if (splitRegisters) {
        const int64_t lanes = target.getVlenBits() * resultLayout.getLmulEighths() /
                              (8 * resultLayout.getSew());
        result.parts.push_back(fresh("register_extract"));
        line(vectorType(conversion.getResult()) + " " + result.parts.back() +
             " = __riscv_vget_v_" + from + "_" + to + "(" + expression + ", " +
             std::to_string(slice.sourceOffset / lanes) + ");");
        continue;
      }
      if (joinRegisters) {
        const int64_t lanes = target.getVlenBits() * sourceLayout.getLmulEighths() /
                              (8 * sourceLayout.getSew());
        if (!slice.resultOffset) {
          result.parts.push_back(fresh("register_join"));
          line(vectorType(conversion.getResult()) + " " + result.parts.back() +
               " = __riscv_vundefined_" + to + "();");
        }
        line(result.parts[slice.resultPart] + " = __riscv_vset_v_" + from + "_" + to +
             "(" + result.parts[slice.resultPart] + ", " + std::to_string(slice.resultOffset / lanes) +
             ", " + expression + ");");
        continue;
      }
      if (slice.sourceOffset)
        expression = "__riscv_vslidedown_vx_" + from + "(" + expression + ", " +
                     std::to_string(slice.sourceOffset) + ", " + std::to_string(slice.count) + ")";
      if (sourceLayout.getLmulEighths() != resultLayout.getLmulEighths())
        expression = "__riscv_vlmul_" +
            std::string(sourceLayout.getLmulEighths() > resultLayout.getLmulEighths() ? "trunc" : "ext") +
            "_v_" + from + "_" + to + "(" + expression + ")";
      if (!slice.resultOffset) {
        result.parts.push_back(fresh("layout_repack"));
        line(vectorType(conversion.getResult()) + " " + result.parts.back() + " = " + expression + ";");
      } else {
        line(result.parts[slice.resultPart] + " = __riscv_vslideup_vx_" + to + "_tu(" +
             result.parts[slice.resultPart] + ", " + expression + ", " +
             std::to_string(slice.resultOffset) + ", " +
             std::to_string(slice.resultOffset + slice.count) + ");");
      }
    }
    bindings[conversion.getResult()] = std::move(result);
    return mlir::success();
  }
  if (kind == "local_load") {
    if (input.kind == Binding::Kind::LocalArray)
      return fail(conversion,
                  "local-load conversion requires its explicit typed memory read");
    mlir::FailureOr<Binding> loaded =
        materializeNumeric(conversion.getResult(), std::move(input));
    if (mlir::failed(loaded))
      return mlir::failure();
    bindings[conversion.getResult()] = std::move(*loaded);
    return mlir::success();
  }
  if (kind == "splat") {
    if (input.kind != Binding::Kind::Scalar)
      return fail(conversion, "RVV splat requires one scalar input");
    mlir::FailureOr<Binding> result = makeVector(
        conversion.getResult(), "layout_splat", input, conversion.getInput());
    if (mlir::failed(result))
      return mlir::failure();
    bindings[conversion.getResult()] = std::move(*result);
    return mlir::success();
  }
  if (kind == "register_to_lane" && input.kind == Binding::Kind::Slice) {
    // The conversion's source layout is an explicit register tuple.  Load that
    // source representation first, then perform the typed tuple-to-lane pack.
    // Loading the field directly as the result layout would bypass a genuine
    // physical conversion and loses tuple-valued gather coordinates.
    mlir::FailureOr<Binding> materialized =
        materializeNumeric(conversion.getInput(), std::move(input));
    if (mlir::failed(materialized))
      return mlir::failure();
    input = std::move(*materialized);
  }
  if ((kind == "register_to_lane" || kind == "time_to_lane") &&
      input.kind == Binding::Kind::Vector) {
    auto sourceType = conversion.getInput().getType();
    auto resultType = conversion.getResult().getType();
    auto sourceLayout = sourceType.getLayout();
    auto resultLayout = resultType.getLayout();
    const auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    const auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
    const auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    const auto targetTime = resultLayout.getTimeFactors().asArrayRef();
    const auto targetLane = resultLayout.getLaneFactors().asArrayRef();
    const auto targetReplica = resultLayout.getReplicaFactors().asArrayRef();
    if (sourceType.getElementType() != resultType.getElementType() ||
        sourceType.getShape() != resultType.getShape() ||
        sourceType.getAxisIds() != resultType.getAxisIds() ||
        sourceLayout.getSew() != resultLayout.getSew() ||
        sourceTime.size() != targetTime.size())
      return fail(conversion,
                  "vector register-to-lane conversion changes its logical domain");

    auto selectedPieces = riscv::rvvPartToLanePieces(sourceType, resultType);
    if (!selectedPieces)
      return fail(conversion,
                  "vector part-to-lane conversion has no closed relative layout mapping");
    llvm::SmallVector<size_t> movedAxes;
    llvm::SmallVector<int64_t> pieceFactors;
    const int64_t pieces = *selectedPieces;
    for (size_t position = 0; position < sourceTime.size(); ++position) {
      const bool moved =
          targetLane[position] > sourceLane[position] &&
          (sourceReplica[position] > targetReplica[position] ||
           sourceTime[position] > targetTime[position]);
      if (moved) {
        const int64_t sourceExtent =
            sourceTime[position] * sourceLane[position] *
            sourceReplica[position];
        const int64_t targetExtent =
            targetTime[position] * targetLane[position] *
            targetReplica[position];
        if (sourceLane[position] <= 0 ||
            targetLane[position] % sourceLane[position] ||
            sourceExtent != targetExtent)
          return fail(conversion,
                      "vector register-to-lane conversion has no closed moved-axis geometry");
        movedAxes.push_back(position);
        pieceFactors.push_back(targetLane[position] / sourceLane[position]);
      }
    }
    const int64_t sourceLanes = physicalLanes(conversion.getInput());
    const int64_t resultLanes = physicalLanes(conversion.getResult());
    const int64_t sourceStreams = streamPartCount(conversion.getInput());
    const int64_t resultStreams = streamPartCount(conversion.getResult());
    const int64_t resultRegisters = registerPartCount(conversion.getResult());
    if (movedAxes.empty() || pieces <= 1 || sourceLanes <= 0 ||
        resultLanes <= 0 ||
        resultLanes != sourceLanes * pieces || sourceStreams <= 0 ||
        resultStreams <= 0 || resultRegisters <= 0 ||
        resultLayout.getLmulEighths() !=
            sourceLayout.getLmulEighths() * pieces)
      return fail(conversion,
                  "vector register-to-lane conversion has incomplete pack geometry");

    auto decode = [](int64_t linear, llvm::ArrayRef<int64_t> factors) {
      llvm::SmallVector<int64_t> coordinates(factors.size(), 0);
      for (int64_t position = static_cast<int64_t>(factors.size()) - 1;
           position >= 0; --position) {
        coordinates[static_cast<size_t>(position)] =
            linear % factors[static_cast<size_t>(position)];
        linear /= factors[static_cast<size_t>(position)];
      }
      return std::pair{std::move(coordinates), linear};
    };
    auto encode = [](llvm::ArrayRef<int64_t> coordinates,
                     llvm::ArrayRef<int64_t> factors) {
      int64_t linear = 0;
      for (auto [coordinate, factor] : llvm::zip(coordinates, factors))
        linear = linear * factor + coordinate;
      return linear;
    };
    const std::string sourceSuffix = vectorSuffix(conversion.getInput());
    const std::string resultSuffix = vectorSuffix(conversion.getResult());
    const char category = resultSuffix.front();
    auto suffixFor = [&](int64_t lmulEighths) {
      return std::string(1, category) +
             std::to_string(resultLayout.getSew()) +
             lmulSpelling(lmulEighths);
    };
    auto typeFor = [&](int64_t lmulEighths) {
      const std::string suffix = suffixFor(lmulEighths);
      const std::string prefix =
          category == 'f' ? "vfloat" : category == 'u' ? "vuint" : "vint";
      return prefix + suffix.substr(1) + "_t";
    };
    if (sourceSuffix != suffixFor(sourceLayout.getLmulEighths()) ||
        resultSuffix != suffixFor(resultLayout.getLmulEighths()))
      return fail(conversion,
                  "vector register-to-lane conversion has inconsistent RVV spelling");
    Binding packed;
    packed.kind = Binding::Kind::Vector;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto [targetRegisters, remainingRegister] =
          decode(resultRegister, targetReplica);
      if (remainingRegister)
        return fail(conversion,
                    "vector register-to-lane target register is out of range");
      for (int64_t resultStream = 0; resultStream < resultStreams;
           ++resultStream) {
        auto [targetTimes, remainingStream] = decode(resultStream, targetTime);
        if (remainingStream)
          return fail(conversion,
                      "vector register-to-lane target stream is out of range");
        llvm::SmallVector<std::string> level;
        for (int64_t piece = 0; piece < pieces; ++piece) {
          auto [pieceCoordinates, remainingPiece] = decode(piece, pieceFactors);
          if (remainingPiece)
            return fail(conversion,
                        "vector register-to-lane moved-axis coordinate is out of range");
          llvm::SmallVector<int64_t> sourceTimes(sourceTime.size(), 0);
          llvm::SmallVector<int64_t> sourceRegisters(sourceReplica.size(), 0);
          for (size_t position = 0; position < sourceTime.size(); ++position) {
            int64_t logicalBase =
                (targetTimes[position] * targetReplica[position] +
                 targetRegisters[position]) *
                targetLane[position];
            auto moved = llvm::find(movedAxes, position);
            if (moved != movedAxes.end()) {
              const size_t movedPosition = static_cast<size_t>(
                  moved - movedAxes.begin());
              logicalBase += pieceCoordinates[movedPosition] *
                             sourceLane[position];
            }
            const int64_t sourceTile =
                sourceLane[position] * sourceReplica[position];
            if (sourceTile <= 0 || logicalBase < 0 ||
                logicalBase % sourceLane[position] ||
                logicalBase / sourceTile >= sourceTime[position])
              return fail(conversion,
                          "vector register-to-lane source coordinate is not aligned");
            sourceTimes[position] = logicalBase / sourceTile;
            sourceRegisters[position] =
                (logicalBase % sourceTile) / sourceLane[position];
          }
          const int64_t sourcePart =
              encode(sourceRegisters, sourceReplica) * sourceStreams +
              encode(sourceTimes, sourceTime);
          if (sourcePart < 0 ||
              sourcePart >= static_cast<int64_t>(input.parts.size()))
            return fail(conversion,
                        "vector register-to-lane source part is out of range");
          level.push_back(input.parts[static_cast<size_t>(sourcePart)]);
        }
        int64_t currentLMUL = sourceLayout.getLmulEighths();
        int64_t currentLanes = sourceLayout.getVl();
        while (level.size() > 1) {
          llvm::SmallVector<std::string> next;
          const int64_t nextLMUL = currentLMUL * 2;
          const int64_t nextLanes = currentLanes * 2;
          for (size_t index = 0; index < level.size(); index += 2) {
            std::string name = fresh("layout_pack");
            std::string expression;
            const int64_t vectorBits =
                conversion->getParentOfType<riscv::KernelOp>()
                    .getTarget().getVlenBits();
            const bool fullCarrier =
                currentLanes * resultLayout.getSew() * 8 ==
                vectorBits * currentLMUL;
            if (currentLMUL >= 8 && fullCarrier) {
              const std::string set = "__riscv_vset_v_" + suffixFor(currentLMUL) +
                                      "_" + suffixFor(nextLMUL);
              expression = set + "(" + set + "(__riscv_vundefined_" +
                           suffixFor(nextLMUL) + "(), 0, " + level[index] +
                           "), 1, " + level[index + 1] + ")";
            } else {
              const std::string extend =
                  "__riscv_vlmul_ext_v_" + suffixFor(currentLMUL) + "_" +
                  suffixFor(nextLMUL);
              expression = "__riscv_vslideup_vx_" + suffixFor(nextLMUL) +
                           "(" + extend + "(" + level[index] + "), " +
                           extend + "(" + level[index + 1] + "), " +
                           std::to_string(currentLanes) + ", " +
                           std::to_string(nextLanes) + ")";
            }
            line(typeFor(nextLMUL) + " " + name + " = " + expression + ";");
            next.push_back(std::move(name));
          }
          level = std::move(next);
          currentLMUL = nextLMUL;
          currentLanes = nextLanes;
        }
        if (level.size() != 1 || currentLMUL != resultLayout.getLmulEighths())
          return fail(conversion,
                      "vector register-to-lane pack did not reach its target LMUL");
        std::string packedValue = std::move(level.front());
        packed.parts.push_back(std::move(packedValue));
      }
    }
    bindings[conversion.getResult()] = std::move(packed);
    return mlir::success();
  }
  if (kind == "register_to_lane" &&
      input.kind == Binding::Kind::ScalarTuple) {
    auto sourceLayout = layoutOf(conversion.getInput());
    auto resultLayout = layoutOf(conversion.getResult());
    const int64_t lanes = physicalLanes(conversion.getResult());
    const int64_t streams = streamPartCount(conversion.getResult());
    const int64_t resultRegisters =
        registerPartCount(conversion.getResult());
    if (!sourceLayout || !resultLayout || lanes <= 1 ||
        streams <= 0 || resultRegisters <= 0 ||
        streamPartCount(conversion.getInput()) != 1 ||
        sourceLayout.getAxisIds() != resultLayout.getAxisIds() ||
        sourceLayout.getReplicaFactors().size() !=
            resultLayout.getReplicaFactors().size())
      return fail(conversion,
                  "register-to-lane conversion has incomplete typed factors");

    auto decode = [](int64_t linear, llvm::ArrayRef<int64_t> factors)
        -> std::optional<llvm::SmallVector<int64_t>> {
      llvm::SmallVector<int64_t> coordinates(factors.size(), 0);
      for (int64_t position = static_cast<int64_t>(factors.size()) - 1;
           position >= 0; --position) {
        const int64_t factor = factors[static_cast<size_t>(position)];
        if (factor <= 0)
          return std::nullopt;
        coordinates[static_cast<size_t>(position)] = linear % factor;
        linear /= factor;
      }
      if (linear)
        return std::nullopt;
      return coordinates;
    };
    const auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    const auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    const auto resultTime = resultLayout.getTimeFactors().asArrayRef();
    const auto resultLane = resultLayout.getLaneFactors().asArrayRef();
    const auto resultReplica = resultLayout.getReplicaFactors().asArrayRef();
    if (llvm::any_of(sourceTime, [](int64_t factor) { return factor != 1; }) ||
        static_cast<int64_t>(input.parts.size()) !=
            registerPartCount(conversion.getInput()))
      return fail(conversion,
                  "register-to-lane scalar tuple must reside entirely in register replicas");

    mlir::Type element =
        riscv_internal::logicalElement(conversion.getResult().getType());
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    const bool floating = usesFloatingRegisters(element);
    if (!floating && !integer && !usesRawFloatStorage(element))
      return fail(conversion,
                  "register-to-lane conversion has no scalar element type");
    const std::string suffix = vectorSuffix(conversion.getResult());
    const std::string vectorCType = vectorType(conversion.getResult());
    const std::string broadcast =
        floating ? "__riscv_vfmv_v_f_" : "__riscv_vmv_v_x_";
    const std::string slide =
        floating ? "__riscv_vfslide1up_vf_" : "__riscv_vslide1up_vx_";

    Binding packed;
    packed.kind = Binding::Kind::Vector;
    for (int64_t resultRegister = 0; resultRegister < resultRegisters;
         ++resultRegister) {
      auto resultRegisterCoordinates = decode(resultRegister, resultReplica);
      if (!resultRegisterCoordinates)
        return fail(conversion,
                    "register-to-lane result has no register coordinates");
      for (int64_t stream = 0; stream < streams; ++stream) {
        auto resultTimeCoordinates = decode(stream, resultTime);
        if (!resultTimeCoordinates)
          return fail(conversion,
                      "register-to-lane result has no issue-time coordinates");
        llvm::SmallVector<size_t, 16> sourceParts;
        for (int64_t lane = 0; lane < lanes; ++lane) {
          auto resultLaneCoordinates = decode(lane, resultLane);
          if (!resultLaneCoordinates)
            return fail(conversion,
                        "register-to-lane result has no lane coordinates");
          int64_t sourcePart = 0;
          for (size_t position = 0; position < sourceReplica.size(); ++position) {
            const int64_t coordinate =
                (((*resultTimeCoordinates)[position] * resultReplica[position] +
                  (*resultRegisterCoordinates)[position]) *
                     resultLane[position] +
                 (*resultLaneCoordinates)[position]);
            if (sourceReplica[position] <= 0 || coordinate < 0 ||
                coordinate >= sourceReplica[position])
              return fail(conversion,
                          "register-to-lane coordinate exceeds source tuple");
            sourcePart = sourcePart * sourceReplica[position] + coordinate;
          }
          if (sourcePart < 0 ||
              sourcePart >= static_cast<int64_t>(input.parts.size()))
            return fail(conversion,
                        "register-to-lane source part is outside its tuple");
          sourceParts.push_back(static_cast<size_t>(sourcePart));
        }
        const int64_t resultPart = resultRegister * streams + stream;
        const std::string vl = partVL(conversion.getResult(), resultPart);
        std::string expression = broadcast + suffix + "(" +
                                 input.parts[sourceParts.back()] + ", " + vl +
                                 ")";
        for (int64_t lane = lanes - 2; lane >= 0; --lane)
          expression = slide + suffix + "(" + expression + ", " +
                       input.parts[sourceParts[static_cast<size_t>(lane)]] +
                       ", " + vl + ")";
        std::string name = fresh("layout_pack");
        line(vectorCType + " " + name + " = " + expression + ";");
        packed.parts.push_back(std::move(name));
      }
    }
    bindings[conversion.getResult()] = std::move(packed);
    return mlir::success();
  }
  if (kind == "tuple" || kind == "reshape" ||
      kind == "register_to_lane") {
    if (input.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> materialized =
          materializeNumeric(conversion.getInput(), std::move(input));
      if (mlir::failed(materialized))
        return mlir::failure();
      input = std::move(*materialized);
    }
    auto sourceType = conversion.getInput().getType();
    auto resultType = conversion.getResult().getType();
    auto sourceLayout = sourceType.getLayout();
    auto resultLayout = resultType.getLayout();
    bool sameRepartitionDomain =
        sourceLayout.getAxisIds() == resultLayout.getAxisIds() &&
        sourceLayout.getReplicaFactors() == resultLayout.getReplicaFactors() &&
        sourceLayout.getFragmentFactors() == resultLayout.getFragmentFactors() &&
        sourceLayout.getLocalFactors() == resultLayout.getLocalFactors();
    if (sameRepartitionDomain)
      for (size_t position = 0;
           position < sourceLayout.getAxisIds().size(); ++position)
        sameRepartitionDomain &=
            sourceLayout.getTimeFactors()[position] *
                    sourceLayout.getLaneFactors()[position] ==
            resultLayout.getTimeFactors()[position] *
                resultLayout.getLaneFactors()[position];
    if (input.kind == Binding::Kind::Vector &&
        sourceLayout.getCarrier() == "rvv" &&
        resultLayout.getCarrier() == "rvv" &&
        sourceType.getElementType() == resultType.getElementType() &&
        sameRepartitionDomain) {
      const int64_t sourceLanes = physicalLanes(conversion.getInput());
      const int64_t resultLanes = physicalLanes(conversion.getResult());
      const int64_t sourceStreams = streamPartCount(conversion.getInput());
      const int64_t resultStreams = streamPartCount(conversion.getResult());
      const int64_t resultRegisters = registerPartCount(conversion.getResult());
      const std::string sourceSuffix = vectorSuffix(conversion.getInput());
      const std::string resultSuffix = vectorSuffix(conversion.getResult());
      const std::string resultCType = vectorType(conversion.getResult());
      if (sourceLanes <= 0 || resultLanes <= 0 || sourceStreams <= 0 ||
          resultStreams <= 0 || resultRegisters <= 0)
        return fail(conversion, "RVV repartition has incomplete typed factors");

      Binding repartitioned;
      repartitioned.kind = Binding::Kind::Vector;
      if (sourceLanes == resultLanes && sourceSuffix == resultSuffix) {
        for (int64_t resultPart = 0;
             resultPart < resultRegisters * resultStreams; ++resultPart) {
          auto sourcePart = projectPart(conversion.getInput(),
                                        conversion.getResult(), resultPart);
          if (!sourcePart || *sourcePart >= input.parts.size())
            return fail(conversion,
                        "RVV repartition cannot project its typed coordinates");
          repartitioned.parts.push_back(input.parts[*sourcePart]);
        }
        bindings[conversion.getResult()] = std::move(repartitioned);
        return mlir::success();
      }
      for (int64_t resultRegister = 0; resultRegister < resultRegisters;
           ++resultRegister) {
        auto sourceRegister = projectRegisterPart(
            conversion.getInput(), conversion.getResult(), resultRegister);
        if (!sourceRegister)
          return fail(conversion,
                      "RVV repartition cannot preserve register-axis coordinates");
        for (int64_t resultStream = 0; resultStream < resultStreams;
             ++resultStream) {
          const int64_t logicalOffset = resultStream * resultLanes;
          std::string expression;
          if (resultLanes <= sourceLanes) {
            const int64_t sourceStream = logicalOffset / sourceLanes;
            const int64_t laneOffset = logicalOffset % sourceLanes;
            const size_t sourcePart =
                *sourceRegister * sourceStreams + sourceStream;
            if (sourceStream >= sourceStreams || sourcePart >= input.parts.size())
              return fail(conversion,
                          "RVV split repartition exceeds its source streams");
            expression = input.parts[sourcePart];
            if (laneOffset)
              expression = "__riscv_vslidedown_vx_" + sourceSuffix + "(" +
                           expression + ", " + std::to_string(laneOffset) +
                           ", " + std::to_string(sourceLanes) + ")";
            if (sourceSuffix != resultSuffix) {
              const int64_t sourceLMUL = sourceLayout.getLmulEighths();
              const int64_t resultLMUL = resultLayout.getLmulEighths();
              if (sourceLMUL < resultLMUL && resultLMUL % sourceLMUL == 0)
                expression = "__riscv_vlmul_ext_v_" + sourceSuffix + "_" +
                             resultSuffix + "(" + expression + ")";
              else if (sourceLMUL > resultLMUL && sourceLMUL % resultLMUL == 0)
                expression = "__riscv_vlmul_trunc_v_" + sourceSuffix + "_" +
                             resultSuffix + "(" + expression + ")";
              else
                return fail(conversion,
                            "RVV repartition requires one integral LMUL relation");
            }
          } else {
            if (resultLanes % sourceLanes != 0)
              return fail(conversion,
                          "RVV merge repartition requires integral lane groups");
            const int64_t pieces = resultLanes / sourceLanes;
            auto sourceLayout = layoutOf(conversion.getInput());
            auto resultLayout = layoutOf(conversion.getResult());
            if (pieces == 2 && sourceLayout && resultLayout &&
                sourceLayout.getSew() == resultLayout.getSew() &&
                sourceLayout.getLmulEighths() * 2 ==
                    resultLayout.getLmulEighths()) {
              const int64_t firstStream = resultStream * pieces;
              const size_t firstPart =
                  *sourceRegister * sourceStreams + firstStream;
              const size_t secondPart = firstPart + 1;
              if (firstStream + 1 >= sourceStreams ||
                  secondPart >= input.parts.size())
                return fail(conversion,
                            "RVV create repartition exceeds its source streams");
              if (sourceLayout.getLmulEighths() >= 8) {
                const std::string set = "__riscv_vset_v_" + sourceSuffix + "_" + resultSuffix;
                expression = set + "(" + set + "(__riscv_vundefined_" +
                             resultSuffix + "(), 0, " + input.parts[firstPart] +
                             "), 1, " + input.parts[secondPart] + ")";
              } else {
                const std::string extend =
                    "__riscv_vlmul_ext_v_" + sourceSuffix + "_" + resultSuffix;
                expression = "__riscv_vslideup_vx_" + resultSuffix + "(" +
                             extend + "(" + input.parts[firstPart] + "), " +
                             extend + "(" + input.parts[secondPart] + "), " +
                             std::to_string(sourceLanes) + ", " +
                             partVL(conversion.getResult(),
                                    resultRegister * resultStreams +
                                        resultStream) +
                             ")";
              }
              std::string name = fresh("layout_repartition");
              line(resultCType + " " + name + " = " + expression + ";");
              repartitioned.parts.push_back(std::move(name));
              continue;
            }
            mlir::Type element = resultType.getElementType();
            std::string merged =
                std::string(usesFloatingRegisters(element)
                                ? "__riscv_vfmv_v_f_"
                                : "__riscv_vmv_v_x_") +
                resultSuffix + "(" +
                (usesFloatingRegisters(element) ? "0.0" : "0") + ", " +
                partVL(conversion.getResult(),
                       resultRegister * resultStreams + resultStream) +
                ")";
            for (int64_t piece = 0; piece < pieces; ++piece) {
              const int64_t sourceStream =
                  resultStream * pieces + piece;
              const size_t sourcePart =
                  *sourceRegister * sourceStreams + sourceStream;
              if (sourceStream >= sourceStreams || sourcePart >= input.parts.size())
                return fail(conversion,
                            "RVV merge repartition exceeds its source streams");
              std::string extended = input.parts[sourcePart];
              if (sourceSuffix != resultSuffix)
                extended = "__riscv_vlmul_ext_v_" + sourceSuffix + "_" +
                           resultSuffix + "(" + extended + ")";
              merged = "__riscv_vslideup_vx_" + resultSuffix + "(" + merged +
                       ", " + extended + ", " +
                       std::to_string(piece * sourceLanes) + ", " +
                       partVL(conversion.getResult(),
                              resultRegister * resultStreams + resultStream) +
                       ")";
            }
            expression = std::move(merged);
          }
          std::string name = fresh("layout_repartition");
          line(resultCType + " " + name + " = " + expression + ";");
          repartitioned.parts.push_back(std::move(name));
        }
      }
      bindings[conversion.getResult()] = std::move(repartitioned);
      return mlir::success();
    }
    mlir::FailureOr<Binding> projected =
        projectBinding(conversion.getInput(), conversion.getResult(), input);
    if (mlir::failed(projected))
      return fail(conversion,
                  "layout conversion is not a projection of its typed layouts");
    bindings[conversion.getResult()] = std::move(*projected);
    return mlir::success();
  }
  if (kind == "extract" || kind == "lane_to_register") {
    LaneExtractionCache localExtractions;
    if (input.kind == Binding::Kind::Slice) {
      mlir::FailureOr<Binding> materialized =
          materializeNumeric(conversion.getInput(), std::move(input));
      if (mlir::failed(materialized))
        return mlir::failure();
      input = std::move(*materialized);
    }
    if (input.kind != Binding::Kind::Vector || input.parts.empty())
      return fail(conversion, "RVV extract requires one vector input");
    mlir::Type element =
        riscv_internal::logicalElement(conversion.getResult().getType());
    auto scalarType = scalarCType(element);
    if (!scalarType)
      return fail(conversion, "RVV extract result has no scalar C type");
    int64_t parts = scalarPartCount(conversion.getResult());
    if (parts == 1) {
      std::string failureReason;
      auto extracted = extractLaneForRegisterBroadcast(
          conversion.getInput(), conversion.getResult(), 0, input,
          localExtractions, &failureReason);
      if (!extracted) {
        std::string operationText;
        llvm::raw_string_ostream operationStream(operationText);
        conversion.print(operationStream);
        return fail(conversion,
                    llvm::Twine("RVV extract cannot preserve its typed register coordinates; part=0") +
                        ", source_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getInput())) +
                        ", result_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getResult())) +
                        ", source_register_parts=" +
                        llvm::Twine(registerPartCount(conversion.getInput())) +
                        ", source_stream_parts=" +
                        llvm::Twine(streamPartCount(conversion.getInput())) +
                        ", input_binding_parts=" +
                        llvm::Twine(input.parts.size()) + ", reason=" +
                        failureReason + ", operation=" + operationText);
      }
      bindings[conversion.getResult()] = scalar(*extracted);
      return mlir::success();
    }
    Binding tuple;
    tuple.kind = Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < parts; ++part) {
      std::string failureReason;
      auto extracted = extractLaneForRegisterBroadcast(
          conversion.getInput(), conversion.getResult(), part, input,
          localExtractions, &failureReason);
      if (!extracted) {
        std::string operationText;
        llvm::raw_string_ostream operationStream(operationText);
        conversion.print(operationStream);
        return fail(conversion,
                    llvm::Twine("RVV extract cannot preserve its typed register coordinates; part=") +
                        llvm::Twine(part) + ", source_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getInput())) +
                        ", result_lane_axis=" +
                        llvm::Twine(laneAxisFor(conversion.getResult())) +
                        ", source_register_parts=" +
                        llvm::Twine(registerPartCount(conversion.getInput())) +
                        ", source_stream_parts=" +
                        llvm::Twine(streamPartCount(conversion.getInput())) +
                        ", input_binding_parts=" +
                        llvm::Twine(input.parts.size()) + ", reason=" +
                        failureReason + ", operation=" + operationText);
      }
      tuple.parts.push_back(*extracted);
    }
    bindings[conversion.getResult()] = std::move(tuple);
    return mlir::success();
  }
  return fail(conversion,
              "layout conversion kind is not a terminal RVV representation operation");
}

mlir::LogicalResult Emitter::compileProjectReductionOperand(
    riscv::ProjectReductionOperandOp operation) {
  mlir::FailureOr<Binding> materialized = materializeNumeric(
      operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(materialized))
    return mlir::failure();
  Binding input = std::move(*materialized);
  Binding index = bindings.lookup(operation.getReductionIndex());
  if ((input.kind != Binding::Kind::Vector && input.kind != Binding::Kind::ScalarTuple) ||
      index.kind != Binding::Kind::Scalar) {
    std::string detail;
    llvm::raw_string_ostream stream(detail);
    stream << "reduction operand projection requires a numeric tuple and scalar "
              "index; input type="
           << operation.getInput().getType() << ", input binding="
           << static_cast<unsigned>(input.kind) << ", index binding="
           << static_cast<unsigned>(index.kind);
    return fail(operation, stream.str());
  }
  auto inputType = operation.getInput().getType();
  auto resultType = operation.getResult().getType();
  riscv::LayoutAttr inputLayout = inputType.getLayout();
  auto found = llvm::find(inputType.getAxisIds().asArrayRef(),
                          operation.getReductionAxis());
  if (found == inputType.getAxisIds().asArrayRef().end())
    return fail(operation, "reduction projection axis is absent from its input");
  const size_t axisIndex = static_cast<size_t>(
      found - inputType.getAxisIds().asArrayRef().begin());
  const int64_t lanes = inputLayout.getLaneFactors()[axisIndex];
  const int64_t streams = inputLayout.getTimeFactors()[axisIndex];
  if (inputLayout.getCarrier() == "scalar") {
    auto source = riscv::RegisterCoordinates::get(inputLayout);
    auto destination = riscv::RegisterCoordinates::get(resultType.getLayout());
    auto ctype = scalarCType(resultType.getElementType());
    if (!source || !destination || !ctype || input.kind != Binding::Kind::ScalarTuple)
      return fail(operation, "scalar reduction projection has no complete register coordinates");
    Binding result;
    result.kind = destination->parts() == 1 ? Binding::Kind::Scalar : Binding::Kind::ScalarTuple;
    for (int64_t part = 0; part < destination->parts(); ++part) {
      auto point = destination->logical(part);
      if (!point) return fail(operation, "scalar reduction result coordinate is absent");
      point->insert(point->begin() + axisIndex, 0);
      std::string expression;
      for (int64_t reduced = source->extents()[axisIndex] - 1; reduced >= 0; --reduced) {
        (*point)[axisIndex] = reduced;
        auto location = source->locate(*point);
        if (!location || location->lane != 0 || location->part >= static_cast<int64_t>(input.parts.size()))
          return fail(operation, "scalar reduction source coordinate is absent");
        const auto &value = input.parts[location->part];
        expression = expression.empty() ? value :
            "(" + index.scalar + " == " + std::to_string(reduced) + " ? " + value + " : " + expression + ")";
      }
      std::string name = fresh("scalar_projection");
      line(*ctype + " " + name + " = " + expression + ";");
      if (result.kind == Binding::Kind::Scalar) result.scalar = std::move(name);
      else result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (lanes == 1 && streams > 1 &&
      resultType.getLayout().getCarrier() == "rvv") {
    if (laneAxisFor(operation.getInput()) != laneAxisFor(operation.getResult()) ||
        laneAxisFor(operation.getResult()) == operation.getReductionAxis())
      return fail(operation,
                  "time-axis projection does not preserve its independent SIMD axis");
    const int64_t inputStreams = streamPartCount(operation.getInput());
    const int64_t resultStreams = streamPartCount(operation.getResult());
    if (inputStreams <= 0 || resultStreams <= 0)
      return fail(operation,
                  "time-axis projection has incomplete stream factors");
    auto resultLayout = resultType.getLayout();
    Binding result;
    result.kind = Binding::Kind::Vector;
    for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
      const int64_t resultRegister = part / resultStreams;
      int64_t resultStream = part % resultStreams;
      auto sourceRegister = projectRegisterPart(
          operation.getInput(), operation.getResult(), resultRegister);
      if (!sourceRegister)
        return fail(operation,
                    "time-axis projection cannot preserve register coordinates");
      llvm::DenseMap<int64_t, int64_t> resultTimeCoordinates;
      for (int64_t dimension =
               static_cast<int64_t>(resultType.getAxisIds().size()) - 1;
           dimension >= 0; --dimension) {
        int64_t factor = resultLayout.getTimeFactors()[dimension];
        if (factor <= 0)
          return fail(operation,
                      "time-axis projection has an invalid result factor");
        resultTimeCoordinates[resultType.getAxisIds()[dimension]] =
            resultStream % factor;
        resultStream /= factor;
      }
      if (resultStream != 0)
        return fail(operation,
                    "time-axis projection result stream is out of range");
      std::string expression;
      for (int64_t reduced = streams - 1; reduced >= 0; --reduced) {
        int64_t sourceStream = 0;
        for (auto [axis, factor] :
             llvm::zip(inputLayout.getAxisIds().asArrayRef(),
                       inputLayout.getTimeFactors().asArrayRef())) {
          if (factor <= 0)
            return fail(operation,
                        "time-axis projection has an invalid input factor");
          int64_t coordinate = reduced;
          if (axis != operation.getReductionAxis()) {
            auto coordinateIt = resultTimeCoordinates.find(axis);
            coordinate = coordinateIt == resultTimeCoordinates.end()
                             ? 0
                             : coordinateIt->second;
          }
          if (coordinate < 0 || coordinate >= factor)
            return fail(operation,
                        "time-axis projection coordinate is out of range");
          sourceStream = sourceStream * factor + coordinate;
        }
        size_t sourcePart = *sourceRegister * inputStreams + sourceStream;
        if (sourcePart >= input.parts.size())
          return fail(operation,
                      "time-axis projection source part exceeds its typed layout");
        expression = expression.empty()
                         ? input.parts[sourcePart]
                         : "(" + index.scalar + " == " +
                               std::to_string(reduced) + " ? " +
                               input.parts[sourcePart] + " : " + expression +
                               ")";
      }
      std::string name = fresh("time_projection");
      line(vectorType(operation.getResult()) + " " + name + " = " +
           expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (lanes <= 1 || streams <= 0 || laneAxisFor(operation.getInput()) !=
                                      operation.getReductionAxis())
    return fail(operation,
                "current reduction projection requires the eliminated axis in RVV lanes");
  if (resultType.getLayout().getCarrier() != "scalar")
    return fail(operation,
                "current reduction projection must produce a register scalar tuple");

  mlir::Type element = resultType.getElementType();
  auto scalarType = scalarCType(element);
  if (!scalarType)
    return fail(operation, "reduction projection result has no scalar C type");
  const std::string suffix = vectorSuffix(operation.getInput());
  const std::string streamIndex = "((" + index.scalar + ") / " +
                                  std::to_string(lanes) + ")";
  const std::string laneIndex = "((" + index.scalar + ") % " +
                                std::to_string(lanes) + ")";
  auto extract = [&](size_t sourcePart) {
    const std::string shifted =
        "__riscv_vslidedown_vx_" + suffix + "(" + input.parts[sourcePart] +
        ", " + laneIndex + ", " +
        partVL(operation.getInput(), sourcePart) + ")";
    if (usesFloatingRegisters(element))
      return "__riscv_vfmv_f_s_" + suffix + "_f" +
             std::to_string(riscv_internal::logicalBitWidth(element)) + "(" +
             shifted + ")";
    auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
    if (!integer && !usesRawFloatStorage(element))
      return std::string();
    return "__riscv_vmv_x_s_" + suffix + "_" +
           std::string(usesRawFloatStorage(element) || integer.isUnsigned() ? "u" : "i") +
           std::to_string(riscv_internal::logicalBitWidth(element)) + "(" + shifted + ")";
  };

  Binding result;
  const int64_t resultParts = registerPartCount(operation.getResult());
  result.kind = resultParts == 1 ? Binding::Kind::Scalar
                                 : Binding::Kind::ScalarTuple;
  for (int64_t resultPart = 0; resultPart < resultParts; ++resultPart) {
    auto sourceRegister = projectRegisterPart(
        operation.getInput(), operation.getResult(), resultPart);
    if (!sourceRegister)
      return fail(operation,
                  "reduction projection cannot preserve the free-axis register mapping");
    std::string expression;
    for (int64_t stream = streams - 1; stream >= 0; --stream) {
      size_t sourcePart = *sourceRegister * streamPartCount(operation.getInput()) +
                          static_cast<size_t>(stream);
      if (sourcePart >= input.parts.size())
        return fail(operation,
                    "reduction projection source part exceeds its typed layout");
      std::string value = extract(sourcePart);
      if (value.empty())
        return fail(operation,
                    "reduction projection has no intrinsic scalar extraction");
      expression = expression.empty()
                       ? value
                       : "(" + streamIndex + " == " + std::to_string(stream) +
                             " ? " + value + " : " + expression + ")";
    }
    std::string name = fresh("reduction_operand");
    line(*scalarType + " " + name + " = " + expression + ";");
    if (result.kind == Binding::Kind::Scalar)
      result.scalar = std::move(name);
    else
      result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
