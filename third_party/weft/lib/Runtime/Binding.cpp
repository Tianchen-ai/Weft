#include "Internal.h"

#include <algorithm>

namespace weft::runtime {

static void *scalarAddress(weft_value &value, const Argument &argument) {
  if (value.kind != WEFT_SCALAR || value.dtype != argument.type)
    throw Failure(WEFT_INVALID_ARGUMENT, argument.name + " scalar dtype differs from its ABI");
  switch (argument.type) {
  case WEFT_I1: return &value.scalar.i1;
  case WEFT_I8: return &value.scalar.i8;
  case WEFT_U8: return &value.scalar.u8;
  case WEFT_I16: return &value.scalar.i16;
  case WEFT_U16: return &value.scalar.u16;
  case WEFT_I32: return &value.scalar.i32;
  case WEFT_U32: return &value.scalar.u32;
  case WEFT_I64: return &value.scalar.i64;
  case WEFT_U64: return &value.scalar.u64;
  case WEFT_F32: return &value.scalar.f32;
  case WEFT_F64: return &value.scalar.f64;
  case WEFT_INDEX: return &value.scalar.index;
  default: throw Failure(WEFT_UNSUPPORTED, argument.name + " by-value dtype requires a direct typed C call");
  }
}

Call::Call(std::shared_ptr<Kernel> kernel, const weft_value *input, size_t count,
           const weft_dimension *explicitDimensions, size_t dimensionCount)
    : kernel(std::move(kernel)) {
  require(count == this->kernel->arguments.size() && (input || !count), WEFT_INVALID_ARGUMENT,
          "argument count differs from compiled ABI");
  const auto &contract = *this->kernel->contract;
  auto extents = this->kernel->bindDimensions(explicitDimensions, dimensionCount);
  this->explicitDimensions = extents;
  if (count) values.assign(input, input + count);
  pointerShapes.resize(count);
  pointerStrides.resize(count);
  parameters.resize(contract.parameterCount);
  strides.resize(contract.strideCount);
  struct Span { uintptr_t begin = 0, end = 0; };
  std::vector<Span> spans(count);
  for (size_t i = 0; i < count; ++i) {
    const auto &arg = this->kernel->arguments[i];
    auto &value = values[i];
    if (value.kind != arg.kind)
      throw Failure(WEFT_INVALID_ARGUMENT, arg.name + " has the wrong argument kind");
  }
  for (size_t i : contract.scalarIndices)
    parameters[i] = scalarAddress(values[i], this->kernel->arguments[i]);
  for (size_t i : contract.viewIndices) {
    const auto &arg = this->kernel->arguments[i];
    const auto &value = values[i];
    const auto &view = value.view;
    if (!view.encoding || view.encoding != arg.encoding)
      throw Failure(WEFT_INVALID_ARGUMENT, arg.name + " has the wrong Encoding");
    contract.constrain(i, arg, view, extents);
    StorageSpan layout(arg, view.shape, view.rank, view.strides, view.strides ? view.rank : 0);
    if (arg.pointer) {
      if (view.rank) pointerShapes[i].assign(view.shape, view.shape + view.rank);
      pointerStrides[i] = layout.strides;
    }
    if (arg.strided)
      std::copy(layout.strides.begin(), layout.strides.end(), strides.begin() + contract.arguments[i].strideBegin);
    size_t bytes = layout.bytes;
    if (bytes > view.byte_length)
      throw Failure(WEFT_INVALID_ARGUMENT, arg.name + " storage is shorter than its View");
    if (arg.writable() && view.readonly)
      throw Failure(WEFT_INVALID_ARGUMENT, arg.name + " requires writable storage");
    uintptr_t address = reinterpret_cast<uintptr_t>(view.data);
    if (bytes) {
      if (!address || address % arg.alignment)
        throw Failure(WEFT_INVALID_ARGUMENT, arg.name + " does not meet Encoding alignment");
      uintptr_t end = add(address, bytes, arg.name + " address");
      spans[i] = {address, end};
    }
    parameters[i] = view.data;
  }
  for (const auto &[left, right] : contract.aliasPairs) {
    const auto &a = spans[left], &b = spans[right];
    if (a.begin != a.end && b.begin != b.end && a.begin < b.end && b.begin < a.end)
      throw Failure(WEFT_INVALID_ARGUMENT, this->kernel->arguments[right].name + " overlaps " +
                    this->kernel->arguments[left].name + " across declared alias groups");
  }
  contract.complete(extents);
  dimensions = std::move(extents.values);
  rebuildParameters();
}

void Call::rebuildParameters() {
  // Every backing vector has reached its final capacity before taking pointers.
  // A copied scalar packet must never retain pointers into another Call's union
  // values or dimension arrays, even if its View storage is intentionally shared.
  const auto &contract = *kernel->contract;
  parameters.resize(contract.parameterCount);
  for (size_t index : contract.scalarIndices)
    parameters[index] = scalarAddress(values[index], kernel->arguments[index]);
  for (size_t index : contract.viewIndices)
    parameters[index] = values[index].view.data;
  size_t slot = values.size();
  for (auto &extent : dimensions) parameters[slot++] = &extent;
  for (auto &stride : strides) parameters[slot++] = &stride;
}

Call::Call(const Call &source, const std::vector<Update> &updates)
    : pointerShapes(source.pointerShapes), pointerStrides(source.pointerStrides),
      kernel(source.kernel), values(source.values), dimensions(source.dimensions),
      strides(source.strides), storage(source.storage), explicitDimensions(source.explicitDimensions) {
  // Only the scalar bytes change. Shape, alias, capacity and stride proofs are
  // immutable facts of the source binding and remain valid in this new call.
  // Tensor metadata is a Python responsibility above this raw borrowed C ABI.
  for (const auto &update : updates) values[update.index] = update.value;
  rebuildParameters();
}

std::shared_ptr<Call> Call::rebind(const weft_named_value *updates, size_t count,
                                 const weft_dimension *dimensionUpdates, size_t dimensionCount) const {
  if (!updates && count)
    throw Failure(WEFT_INVALID_ARGUMENT, "call updates are null");
  const auto &contract = *kernel->contract;
  std::vector<bool> changed(values.size(), false);
  std::vector<Update> selectedUpdates;
  selectedUpdates.reserve(count);
  bool scalarOnly = dimensionCount == 0;
  for (size_t index = 0; index < count; ++index) {
    size_t ordinal = contract.argumentIndex(updates[index].name);
    if (changed[ordinal])
      throw Failure(WEFT_INVALID_ARGUMENT, "duplicate call update: " + kernel->arguments[ordinal].name);
    const auto &argument = kernel->arguments[ordinal];
    if (updates[index].value.kind != argument.kind)
      throw Failure(WEFT_INVALID_ARGUMENT, argument.name + " has the wrong argument kind");
    scalarOnly &= argument.kind == WEFT_SCALAR;
    selectedUpdates.push_back({ordinal, updates[index].value});
    changed[ordinal] = true;
  }
  if (scalarOnly)
    return std::shared_ptr<Call>(new Call(*this, selectedUpdates));
  auto selected = values;
  // Public descriptor arrays may have died immediately after prepare. Recover
  // their exact captured shapes from the typed contract and owned ABI extents,
  // never from the stale descriptor pointers in the original input structures.
  DimensionValues captured{dimensions, std::vector<bool>(dimensions.size(), true)};
  std::vector<std::vector<size_t>> shapes(values.size());
  for (size_t index : contract.viewIndices) {
    const auto &argument = kernel->arguments[index];
    shapes[index] = argument.pointer ? pointerShapes[index] : contract.shape(index, captured);
    auto &view = selected[index].view;
    view.encoding = argument.encoding.c_str();
    view.shape = shapes[index].data();
    view.strides = argument.pointer ? pointerStrides[index].data() : argument.strided && view.rank ?
        strides.data() + contract.arguments[index].strideBegin : nullptr;
  }
  for (const auto &update : selectedUpdates) selected[update.index] = update.value;
  auto extents = explicitDimensions;
  auto overrides = contract.bind(dimensionUpdates, dimensionCount);
  std::vector<weft_dimension> explicitValues;
  explicitValues.reserve(contract.dimensions.size());
  for (size_t index = 0; index < contract.dimensions.size(); ++index) {
    if (overrides.known[index]) {
      extents.values[index] = overrides.values[index];
      extents.known[index] = true;
    }
    if (extents.known[index])
      explicitValues.push_back({contract.dimensions[index].c_str(), extents.values[index]});
  }
  auto result = std::make_shared<Call>(kernel, selected.data(), selected.size(),
                                     explicitValues.data(), explicitValues.size());
  // Retaining an unchanged SDK allocation is necessary even when its outer
  // buffer/bindings handle has already been released. New descriptors remain
  // borrowed exactly as at ordinary prepare; no hidden allocation is inferred.
  result->storage = storage;
  return result;
}

} // namespace weft::runtime
