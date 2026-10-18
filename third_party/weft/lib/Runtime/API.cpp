#include "APIBoundary.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

using namespace weft::runtime;

extern "C" {

const char *weft_status_name(weft_status status) {
  switch (status) {
  case WEFT_OK: return "ok";
  case WEFT_INVALID_ARGUMENT: return "invalid-argument";
  case WEFT_IO_ERROR: return "io-error";
  case WEFT_ABI_ERROR: return "abi-error";
  case WEFT_TARGET_ERROR: return "target-error";
  case WEFT_UNSUPPORTED: return "unsupported";
  case WEFT_RESOURCE_ERROR: return "resource-error";
  }
  return "unknown-status";
}

weft_status weft_context_create(weft_context **result, weft_error *error) {
  return api(error, [&] { output(result); *result = new weft_context{std::make_shared<Context>()}; });
}
void weft_context_release(weft_context *context) { delete context; }

weft_status weft_context_target_json(const weft_context *context, const char **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(context, WEFT_INVALID_ARGUMENT, "context is null");
    *result = context->value->json.c_str();
  });
}

weft_status weft_kernel_open(weft_context *context, const char *directory,
                             weft_kernel **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(context && directory && *directory, WEFT_INVALID_ARGUMENT, "kernel needs a context and artifact directory");
    *result = new weft_kernel{std::make_shared<Kernel>(context->value, directory)};
  });
}
void weft_kernel_release(weft_kernel *kernel) { delete kernel; }

weft_status weft_kernel_open_native(weft_context *context, const char *libraryPath,
                                    const char *manifestJSON, weft_kernel **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(context && libraryPath && manifestJSON, WEFT_INVALID_ARGUMENT,
            "native kernel needs a context, library path and manifest");
    auto parsed = llvm::json::parse(manifestJSON);
    if (!parsed)
      throw Failure(WEFT_ABI_ERROR, llvm::toString(parsed.takeError()));
    auto *manifest = parsed->getAsObject();
    require(manifest, WEFT_ABI_ERROR, "native kernel manifest must be an object");
    *result = new weft_kernel{std::make_shared<Kernel>(context->value, libraryPath, *manifest)};
  });
}

int weft_kernel_check_worker(const void *kernel, char *message, size_t capacity) {
  weft_error error{};
  auto status = weft_kernel_check_thread(static_cast<const weft_kernel *>(kernel), &error);
  if (message && capacity) std::snprintf(message, capacity, "%s", error.message);
  return static_cast<int>(status);
}

weft_status weft_kernel_describe(const weft_kernel *kernel, weft_kernel_info *result, weft_error *error) {
  return api(error, [&] {
    require(kernel && result, WEFT_INVALID_ARGUMENT, "kernel and result are required");
    const auto &value = *kernel->value;
    *result = {value.symbol.c_str(), value.march.c_str(), value.abi.c_str(), value.vlen,
               value.arguments.size(), value.dimensionNames.size(), value.dimensionNames.data()};
  });
}

weft_status weft_kernel_argument(const weft_kernel *kernel, size_t index,
                                 weft_argument_info *result, weft_error *error) {
  return api(error, [&] {
    require(kernel && result, WEFT_INVALID_ARGUMENT, "kernel and result are required");
    require(index < kernel->value->arguments.size(), WEFT_INVALID_ARGUMENT, "argument index is outside the ABI");
    const auto &arg = kernel->value->arguments[index];
    *result = {arg.name.c_str(), arg.kind, arg.type, arg.encoding.c_str(), arg.access, arg.alias,
               arg.shapeNames.size(), arg.shapeNames.data(), arg.storageBytes, arg.recordElements,
               arg.interleaveRows, arg.alignment, arg.strided};
  });
}

weft_status weft_kernel_check_thread(const weft_kernel *kernel, weft_error *error) {
  return api(error, [&] {
    require(kernel, WEFT_INVALID_ARGUMENT, "kernel is null");
    kernel->value->context->check(kernel->value->usedExtensions);
  });
}

weft_status weft_kernel_bind_thread(const weft_kernel *kernel, weft_error *error) {
  return api(error, [&] {
    require(kernel, WEFT_INVALID_ARGUMENT, "kernel is null");
    kernel->value->context->bindThread(kernel->value->usedExtensions);
  });
}

weft_status weft_kernel_get_entry(const weft_kernel *kernel, weft_native_entry *result, weft_error *error) {
  return api(error, [&] {
    require(kernel && result, WEFT_INVALID_ARGUMENT, "kernel and result are required");
    *result = kernel->value->entry;
  });
}

weft_status weft_call_prepare(weft_kernel *kernel, const weft_value *arguments, size_t count,
                              const weft_dimension *dimensions, size_t dimension_count,
                              weft_call **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(kernel, WEFT_INVALID_ARGUMENT, "kernel is null");
    *result = new weft_call{std::make_shared<Call>(kernel->value, arguments, count, dimensions, dimension_count)};
  });
}

weft_status weft_call_invoke(const weft_call *call, weft_error *error) {
  return api(error, [&] {
    require(call, WEFT_INVALID_ARGUMENT, "call is null");
    const auto &kernel = call->value->kernel;
    kernel->context->check(kernel->usedExtensions);
    kernel->dispatch(kernel->entry, call->value->parameters.data());
  });
}

weft_status weft_call_arguments(const weft_call *call, void *const **arguments,
                                 size_t *count, weft_error *error) {
  return api(error, [&] {
    output(arguments);
    require(call && count, WEFT_INVALID_ARGUMENT, "call, arguments and count are required");
    *count = call->value->parameters.size();
    *arguments = call->value->parameters.data();
  });
}
void weft_call_release(weft_call *call) { delete call; }

weft_status weft_call_rebind(const weft_call *call, const weft_named_value *updates, size_t count,
                             const weft_dimension *dimensions, size_t dimensionCount,
                             weft_call **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(call, WEFT_INVALID_ARGUMENT, "call is null");
    *result = new weft_call{call->value->rebind(updates, count, dimensions, dimensionCount)};
  });
}

weft_status weft_kernel_storage_size(const weft_kernel *kernel, size_t argument,
                                     const weft_dimension *dimensions, size_t dimension_count,
                                     size_t *bytes, weft_error *error) {
  return api(error, [&] {
    require(kernel && bytes, WEFT_INVALID_ARGUMENT, "kernel and size result are required");
    *bytes = kernel->value->storageSize(argument, kernel->value->bindDimensions(dimensions, dimension_count));
  });
}

weft_status weft_buffer_allocate(size_t bytes, size_t alignment, weft_buffer **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    *result = new weft_buffer{std::make_shared<Storage>(bytes, alignment)};
  });
}

weft_status weft_buffer_data(const weft_buffer *buffer, void **data, size_t *bytes, weft_error *error) {
  return api(error, [&] {
    require(buffer && data && bytes, WEFT_INVALID_ARGUMENT, "buffer, data and size outputs are required");
    *data = buffer->value->data; *bytes = buffer->value->bytes;
  });
}
void weft_buffer_release(weft_buffer *buffer) { delete buffer; }

weft_status weft_package_open(weft_context *context, const char *directory, weft_package **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(context && directory && *directory, WEFT_INVALID_ARGUMENT, "package needs a context and directory");
    *result = new weft_package{std::make_unique<Package>(context->value, directory)};
  });
}
weft_status weft_package_count(const weft_package *package, size_t *count, weft_error *error) {
  return api(error, [&] {
    require(package && count, WEFT_INVALID_ARGUMENT, "package and result are required");
    *count = package->value->names.size();
  });
}
weft_status weft_package_name(const weft_package *package, size_t index, const char **name, weft_error *error) {
  return api(error, [&] {
    output(name);
    require(package && index < package->value->names.size(), WEFT_INVALID_ARGUMENT, "package index is outside its entries");
    *name = package->value->names[index].c_str();
  });
}
weft_status weft_package_kernel(weft_package *package, const char *name, weft_kernel **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(package && name, WEFT_INVALID_ARGUMENT, "package and kernel name are required");
    auto found = package->value->kernels.find(name);
    require(found != package->value->kernels.end(), WEFT_INVALID_ARGUMENT, "package has no entry: " + std::string(name));
    *result = new weft_kernel{found->second};
  });
}
void weft_package_release(weft_package *package) { delete package; }

weft_status weft_package_has_binding_group(const weft_package *package, const char *group,
                                          bool *present, weft_error *error) {
  return api(error, [&] {
    require(package && group && *group && present, WEFT_INVALID_ARGUMENT, "package, named group and result are required");
    *present = package->value->bindings.find(group) != package->value->bindings.end();
  });
}

weft_status weft_package_binding_count(const weft_package *package, const char *group,
                                       size_t *count, weft_error *error) {
  return api(error, [&] {
    require(package && group && count, WEFT_INVALID_ARGUMENT, "package, group and count are required");
    auto found = package->value->bindings.find(group);
    require(found != package->value->bindings.end(), WEFT_INVALID_ARGUMENT, "package has no binding group: " + std::string(group));
    *count = found->second.size();
  });
}
weft_status weft_package_binding(const weft_package *package, const char *group, size_t index,
                                 const char **role, const char **kernel, weft_error *error) {
  return api(error, [&] {
    output(role); output(kernel);
    require(package && group, WEFT_INVALID_ARGUMENT, "package and binding group are required");
    auto found = package->value->bindings.find(group);
    require(found != package->value->bindings.end() && index < found->second.size(),
            WEFT_INVALID_ARGUMENT, "package binding index is outside its group");
    *role = found->second[index].first.c_str();
    *kernel = found->second[index].second.c_str();
  });
}

} // extern "C"
