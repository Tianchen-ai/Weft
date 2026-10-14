#ifndef WEFT_GGML_H
#define WEFT_GGML_H

#include "weft/runtime.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_weft_counters {
    uint64_t nodes;
    uint64_t extra_buffer_nodes;
    uint64_t vec_dot_calls;
    uint64_t dot_products;
    uint64_t quantize_calls;
    uint64_t quantized_elements;
} ggml_weft_counters;

/* GGML_WEFT is an explicit build choice. An enabled backend requires a native
   package in WEFT_GGML_PACKAGE; loading/ABI/execution failures are fatal GGML
   errors. A type absent from that package retains its upstream implementation. */
WEFT_API void ggml_weft_initialize(void);
WEFT_API bool ggml_weft_registered(enum ggml_type type);
WEFT_API bool ggml_weft_counters_get(enum ggml_type type, ggml_weft_counters *result);
WEFT_API const char *ggml_weft_kernel_name(enum ggml_type type);

/* CPU backend call-site hooks; callers preserve the original graph, worker
   assignment, quantization workspace, weight layout and synchronization. */
WEFT_API void ggml_weft_begin_node(const struct ggml_tensor *tensor, int ith, bool use_ref);
WEFT_API void ggml_weft_extra_node(const struct ggml_tensor *tensor, int ith);
WEFT_API ggml_vec_dot_t ggml_weft_select_vec_dot(enum ggml_type type, ggml_vec_dot_t original);
WEFT_API void ggml_weft_from_float(ggml_from_float_t original, const float *x, void *y, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
