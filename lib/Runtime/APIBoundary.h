#ifndef WEFT_RUNTIME_API_BOUNDARY_H
#define WEFT_RUNTIME_API_BOUNDARY_H

#include "Internal.h"
#include <cstdio>

namespace weft::runtime {
template <typename F> weft_status api(weft_error *error, F &&function) noexcept {
  try {
    function();
    if (error) { error->status = WEFT_OK; error->message[0] = '\0'; }
    return WEFT_OK;
  } catch (const Failure &failure) {
    if (error) {
      error->status = failure.status;
      std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
    }
    return failure.status;
  } catch (const std::exception &exception) {
    if (error) {
      error->status = WEFT_RESOURCE_ERROR;
      std::snprintf(error->message, sizeof(error->message), "%s", exception.what());
    }
    return WEFT_RESOURCE_ERROR;
  } catch (...) {
    if (error) {
      error->status = WEFT_RESOURCE_ERROR;
      std::snprintf(error->message, sizeof(error->message), "nonstandard exception at the native SDK boundary");
    }
    return WEFT_RESOURCE_ERROR;
  }
}

template <typename T> void output(T **pointer) {
  require(pointer != nullptr, WEFT_INVALID_ARGUMENT, "result pointer is null");
  *pointer = nullptr;
}
} // namespace weft::runtime
#endif
