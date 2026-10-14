#include "APIBoundary.h"

#include <algorithm>
#include <cstdlib>
#include <set>

namespace weft::runtime {

Storage::Storage(size_t bytes, size_t alignment) : bytes(bytes) {
  require(alignment && (alignment & (alignment - 1)) == 0, WEFT_INVALID_ARGUMENT,
          "buffer alignment must be a nonzero power of two");
  require(posix_memalign(&data, std::max(alignment, sizeof(void *)), std::max(bytes, size_t(1))) == 0,
          WEFT_RESOURCE_ERROR, "cannot allocate aligned buffer");
}

Storage::~Storage() { std::free(data); }

BoundValue::BoundValue(const weft_value &input) : value(input) {
  require(input.kind == WEFT_VIEW || input.kind == WEFT_SCALAR,
          WEFT_INVALID_ARGUMENT, "invalid argument kind");
  if (input.kind != WEFT_VIEW) return;
  require(input.view.encoding && (input.view.shape || !input.view.rank),
          WEFT_INVALID_ARGUMENT, "View requires an Encoding and shape");
  encoding = input.view.encoding;
  if (input.view.rank) {
    shape.assign(input.view.shape, input.view.shape + input.view.rank);
    if (input.view.strides)
      strides.assign(input.view.strides, input.view.strides + input.view.rank);
  }
  // Descriptor pointers are rebuilt from owned metadata after every move.
  value.view.encoding = nullptr;
  value.view.shape = nullptr;
  value.view.strides = nullptr;
}

weft_value BoundValue::descriptor() const {
  auto result = value;
  if (result.kind == WEFT_VIEW) {
    result.view.encoding = encoding.c_str();
    result.view.shape = shape.data();
    result.view.strides = strides.empty() ? nullptr : strides.data();
  }
  return result;
}

size_t Bindings::index(const char *name) const {
  return kernel->contract->argumentIndex(name);
}

void Bindings::set(size_t index, BoundValue value) {
  const auto &argument = kernel->arguments[index];
  require(value.value.kind == argument.kind, WEFT_INVALID_ARGUMENT,
          argument.name + " has the wrong argument kind");
  if (argument.kind == WEFT_VIEW)
    require(value.encoding == argument.encoding && value.shape.size() == argument.shape.size(),
            WEFT_INVALID_ARGUMENT, argument.name + " has the wrong Encoding or rank");
  else
    require(value.value.dtype == argument.type, WEFT_INVALID_ARGUMENT,
            argument.name + " has the wrong scalar dtype");
  values[index] = std::move(value);
}

std::shared_ptr<Call> Bindings::prepare() const {
  std::vector<weft_value> ordered;
  std::vector<std::shared_ptr<Storage>> retained;
  ordered.reserve(values.size());
  for (size_t i = 0; i < kernel->arguments.size(); ++i) {
    const auto &value = values[i];
    if (!value)
      throw Failure(WEFT_INVALID_ARGUMENT, "unbound argument: " + kernel->arguments[i].name);
    ordered.push_back(value->descriptor());
    if (value->storage) retained.push_back(value->storage);
  }
  std::vector<weft_dimension> extents;
  for (const auto &[name, value] : dimensions) extents.push_back({name.c_str(), value});
  auto result = std::make_shared<Call>(kernel, ordered.data(), ordered.size(), extents.data(), extents.size());
  result->storage = std::move(retained);
  return result;
}

Sequence::Sequence(std::vector<std::shared_ptr<Call>> values) : calls(std::move(values)) {
  for (const auto &call : calls) {
    const Context *context = call->kernel->context.get();
    auto found = std::find_if(guards.begin(), guards.end(),
                              [context](const Guard &guard) { return guard.context == context; });
    if (found == guards.end()) {
      guards.push_back({context, call->kernel->usedExtensions});
      continue;
    }
    for (const auto &extension : call->kernel->usedExtensions)
      if (std::find(found->extensions.begin(), found->extensions.end(), extension) == found->extensions.end())
        found->extensions.push_back(extension);
  }
}

void Sequence::invoke() const {
  // Validate every target before the first visible write. The caller must not
  // alter affinity/vector state during this synchronous execution scope.
  for (const auto &guard : guards) guard.context->check(guard.extensions);
  for (const auto &call : calls)
    call->kernel->dispatch(call->kernel->entry, call->parameters.data());
}

} // namespace weft::runtime

using namespace weft::runtime;

extern "C" {
weft_status weft_bindings_create(weft_kernel *kernel, weft_bindings **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(kernel, WEFT_INVALID_ARGUMENT, "kernel is null");
    *result = new weft_bindings{std::make_unique<Bindings>(kernel->value)};
  });
}

void weft_bindings_release(weft_bindings *bindings) { delete bindings; }

weft_status weft_bindings_set(weft_bindings *bindings, const char *name,
                              const weft_value *value, weft_error *error) {
  return api(error, [&] {
    require(bindings && value, WEFT_INVALID_ARGUMENT, "bindings and value are required");
    bindings->value->set(bindings->value->index(name), BoundValue(*value));
  });
}

weft_status weft_bindings_dimension(weft_bindings *bindings, const char *name,
                                    size_t value, weft_error *error) {
  return api(error, [&] {
    require(bindings, WEFT_INVALID_ARGUMENT, "bindings are null");
    weft_dimension dimension{name, value};
    bindings->value->kernel->bindDimensions(&dimension, 1);
    bindings->value->dimensions.insert_or_assign(name, value);
  });
}

weft_status weft_bindings_buffer(weft_bindings *bindings, const char *name,
                                 weft_buffer *buffer, size_t offset, size_t rank,
                                 const size_t *shape, const size_t *strides,
                                 bool readonly, weft_error *error) {
  return api(error, [&] {
    require(bindings && buffer, WEFT_INVALID_ARGUMENT, "bindings and buffer are required");
    auto index = bindings->value->index(name);
    const auto &argument = bindings->value->kernel->arguments[index];
    require(argument.kind == WEFT_VIEW, WEFT_INVALID_ARGUMENT, "buffer can only bind a View argument");
    require(offset <= buffer->value->bytes, WEFT_INVALID_ARGUMENT, "buffer offset is outside its allocation");
    weft_value descriptor{};
    descriptor.kind = WEFT_VIEW;
    descriptor.view = {static_cast<char *>(buffer->value->data) + offset,
                       buffer->value->bytes - offset, argument.encoding.c_str(), rank,
                       shape, strides, readonly};
    BoundValue value(descriptor);
    value.storage = buffer->value;
    bindings->value->set(index, std::move(value));
  });
}

weft_status weft_bindings_prepare(const weft_bindings *bindings, weft_call **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(bindings, WEFT_INVALID_ARGUMENT, "bindings are null");
    *result = new weft_call{bindings->value->prepare()};
  });
}

weft_status weft_sequence_create(weft_call *const *calls, size_t count,
                                 weft_sequence **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(calls && count, WEFT_INVALID_ARGUMENT, "sequence requires at least one prepared call");
    std::vector<std::shared_ptr<Call>> retained;
    retained.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      require(calls[i], WEFT_INVALID_ARGUMENT, "sequence contains a null call");
      retained.push_back(calls[i]->value);
    }
    *result = new weft_sequence{std::make_unique<Sequence>(std::move(retained))};
  });
}

weft_status weft_sequence_invoke(const weft_sequence *sequence, weft_error *error) {
  return api(error, [&] {
    require(sequence, WEFT_INVALID_ARGUMENT, "sequence is null");
    sequence->value->invoke();
  });
}

void weft_sequence_release(weft_sequence *sequence) { delete sequence; }

weft_status weft_sequence_replace(const weft_sequence *sequence, size_t index,
                                   const weft_call *call, weft_sequence **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(sequence && call, WEFT_INVALID_ARGUMENT, "sequence and replacement call are required");
    require(index < sequence->value->calls.size(), WEFT_INVALID_ARGUMENT,
            "sequence call index is out of range");
    auto calls = sequence->value->calls;
    calls[index] = call->value;
    *result = new weft_sequence{std::make_unique<Sequence>(std::move(calls))};
  });
}

weft_status weft_sequence_rebind(const weft_sequence *sequence,
                                  const weft_sequence_update *updates, size_t count,
                                  weft_sequence **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(sequence && (updates || !count), WEFT_INVALID_ARGUMENT,
            "sequence and update descriptors are required");
    auto calls = sequence->value->calls;
    std::vector<bool> changed(calls.size(), false);
    for (size_t index = 0; index < count; ++index) {
      const auto &update = updates[index];
      if (update.call_index >= calls.size())
        throw Failure(WEFT_INVALID_ARGUMENT, "sequence call index is out of range");
      if (changed[update.call_index])
        throw Failure(WEFT_INVALID_ARGUMENT, "duplicate sequence call update");
      calls[update.call_index] = calls[update.call_index]->rebind(
          update.values, update.value_count, update.dimensions, update.dimension_count);
      changed[update.call_index] = true;
    }
    *result = new weft_sequence{std::make_unique<Sequence>(std::move(calls))};
  });
}
}
