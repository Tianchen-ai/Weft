#include "Internal.h"

#include <algorithm>
#include <limits>

namespace weft::runtime {

CallContract::CallContract(const std::vector<Argument> &metadata,
                           const std::vector<std::string> &dimensionNames,
                           const std::map<std::string, size_t> &bindings)
    : dimensions(dimensionNames) {
  for (size_t index = 0; index < dimensions.size(); ++index) {
    const auto &name = dimensions[index];
    require(!name.empty() && dimensionIndices.emplace(name, index).second &&
                !bindings.count(name),
            WEFT_ABI_ERROR, "duplicate, empty or static runtime dimension: " + name);
  }
  arguments.reserve(metadata.size());
  for (size_t index = 0; index < metadata.size(); ++index) {
    const auto &argument = metadata[index];
    require(!argument.name.empty() && argumentIndices.emplace(argument.name, index).second &&
                !dimensionIndices.count(argument.name),
            WEFT_ABI_ERROR, "duplicate or colliding ABI argument: " + argument.name);
    ArgumentBinding binding{index, {}, strideCount};
    binding.shape.reserve(argument.shape.size());
    for (const auto &spelling : argument.shape) {
      size_t extent;
      if (!llvm::StringRef(spelling).getAsInteger(10, extent)) {
        binding.shape.push_back({false, extent});
        continue;
      }
      auto fixed = bindings.find(spelling);
      if (fixed != bindings.end()) {
        binding.shape.push_back({false, fixed->second});
        continue;
      }
      auto dynamic = dimensionIndices.find(spelling);
      require(dynamic != dimensionIndices.end(), WEFT_ABI_ERROR,
              "unknown View dimension: " + spelling);
      binding.shape.push_back({true, dynamic->second});
    }
    if (argument.kind == WEFT_VIEW) {
      for (size_t previous : viewIndices)
        if (metadata[previous].alias != argument.alias)
          aliasPairs.emplace_back(previous, index);
      viewIndices.push_back(index);
      if (argument.strided)
        strideCount = add(strideCount, argument.shape.size(), "ABI stride count");
    } else {
      scalarIndices.push_back(index);
    }
    arguments.push_back(std::move(binding));
  }
  parameterCount = add(add(metadata.size(), dimensions.size(), "ABI parameter count"),
                       strideCount, "ABI parameter count");
}

DimensionValues CallContract::bind(const weft_dimension *input, size_t count) const {
  if (!input && count)
    throw Failure(WEFT_INVALID_ARGUMENT, "dimension array is null");
  DimensionValues result{std::vector<size_t>(dimensions.size()),
                         std::vector<bool>(dimensions.size(), false)};
  for (size_t index = 0; index < count; ++index) {
    if (!input[index].name)
      throw Failure(WEFT_INVALID_ARGUMENT, "dimension name is null");
    auto found = dimensionIndices.find(input[index].name);
    if (found == dimensionIndices.end())
      throw Failure(WEFT_INVALID_ARGUMENT,
                    "unknown runtime dimension: " + std::string(input[index].name));
    size_t ordinal = found->second;
    if (result.known[ordinal])
      throw Failure(WEFT_INVALID_ARGUMENT, "duplicate dimension: " + dimensions[ordinal]);
    result.values[ordinal] = input[index].value;
    result.known[ordinal] = true;
  }
  return result;
}

void CallContract::constrain(size_t argument, const Argument &metadata,
                             const weft_view &view, DimensionValues &values) const {
  if (metadata.pointer) return;
  const auto &terms = arguments[argument].shape;
  if (view.rank != terms.size() || (!view.shape && view.rank))
    throw Failure(WEFT_INVALID_ARGUMENT, metadata.name + " has the wrong View rank");
  for (size_t axis = 0; axis < terms.size(); ++axis) {
    const auto &term = terms[axis];
    size_t actual = view.shape[axis];
    if (term.dynamic && !values.known[term.value]) {
      values.values[term.value] = actual;
      values.known[term.value] = true;
    } else if (actual != (term.dynamic ? values.values[term.value] : term.value)) {
      throw Failure(WEFT_INVALID_ARGUMENT,
                    metadata.name + " disagrees with dimension " + metadata.shape[axis]);
    }
  }
}

std::vector<size_t> CallContract::shape(size_t argument, const DimensionValues &values) const {
  const auto &terms = arguments[argument].shape;
  std::vector<size_t> result;
  result.reserve(terms.size());
  for (const auto &term : terms) {
    if (term.dynamic && !values.known[term.value])
      throw Failure(WEFT_INVALID_ARGUMENT, "missing dimension: " + dimensions[term.value]);
    result.push_back(term.dynamic ? values.values[term.value] : term.value);
  }
  return result;
}

void CallContract::complete(const DimensionValues &values) const {
  for (size_t index = 0; index < dimensions.size(); ++index)
    if (!values.known[index])
      throw Failure(WEFT_INVALID_ARGUMENT, "missing runtime dimension: " + dimensions[index]);
}

size_t CallContract::argumentIndex(const char *name) const {
  if (!name || !*name)
    throw Failure(WEFT_INVALID_ARGUMENT, "argument name is empty");
  auto found = argumentIndices.find(name);
  if (found == argumentIndices.end())
    throw Failure(WEFT_INVALID_ARGUMENT, "kernel has no argument: " + std::string(name));
  return found->second;
}

DimensionValues Kernel::bindDimensions(const weft_dimension *values, size_t count) const {
  return contract->bind(values, count);
}

size_t Kernel::storageSize(size_t index, const DimensionValues &values) const {
  if (index >= arguments.size())
    throw Failure(WEFT_INVALID_ARGUMENT, "argument index is outside the ABI");
  const auto &argument = arguments[index];
  if (argument.kind != WEFT_VIEW || argument.strided || argument.pointer)
    throw Failure(WEFT_INVALID_ARGUMENT,
                  "storage_size requires a contiguous View; strided storage needs an explicit span");
  auto extents = contract->shape(index, values);
  // Allocation queries and callable Views use exactly the same Encoding,
  // cohort, arithmetic and signed-pointer bounds.
  return StorageSpan(argument, extents.data(), extents.size(), nullptr, 0).bytes;
}

} // namespace weft::runtime
