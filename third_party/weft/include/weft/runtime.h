#ifndef WEFT_RUNTIME_H
#define WEFT_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define WEFT_API __attribute__((visibility("default")))
#else
#define WEFT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum weft_status {
    WEFT_OK = 0,
    WEFT_INVALID_ARGUMENT,
    WEFT_IO_ERROR,
    WEFT_ABI_ERROR,
    WEFT_TARGET_ERROR,
    WEFT_UNSUPPORTED,
    WEFT_RESOURCE_ERROR
} weft_status;

typedef struct weft_error {
    weft_status status;
    char message[1024];
} weft_error;

typedef enum weft_dtype {
    WEFT_DTYPE_NONE = 0,
    WEFT_I1, WEFT_I8, WEFT_U8, WEFT_I16, WEFT_U16,
    WEFT_I32, WEFT_U32, WEFT_I64, WEFT_U64,
    WEFT_F16, WEFT_BF16, WEFT_F32, WEFT_F64, WEFT_INDEX
} weft_dtype;

typedef enum weft_access {
    WEFT_ACCESS_NONE = 0,
    WEFT_READ, WEFT_WRITE, WEFT_READWRITE
} weft_access;

typedef enum weft_value_kind { WEFT_VIEW = 1, WEFT_SCALAR } weft_value_kind;

typedef struct weft_view {
    void *data;
    size_t byte_length;
    const char *encoding;
    size_t rank;
    const size_t *shape;
    /* NULL declares contiguous storage; otherwise one element stride per axis. */
    const size_t *strides;
    bool readonly;
} weft_view;

typedef union weft_scalar {
    bool i1;
    int8_t i8;
    uint8_t u8;
    int16_t i16;
    uint16_t u16;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    float f32;
    double f64;
    size_t index;
} weft_scalar;

typedef struct weft_value {
    weft_value_kind kind;
    weft_dtype dtype;
    weft_view view;
    weft_scalar scalar;
} weft_value;

typedef struct weft_dimension {
    const char *name;
    size_t value;
} weft_dimension;

typedef struct weft_named_value {
    const char *name;
    weft_value value;
} weft_named_value;

typedef struct weft_argument_info {
    const char *name;
    weft_value_kind kind;
    weft_dtype dtype;
    const char *encoding;
    weft_access access;
    int64_t alias_set;
    size_t rank;
    const char *const *shape;
    size_t storage_bytes;
    size_t record_elements;
    size_t interleave_rows;
    size_t alignment;
    bool strided;
} weft_argument_info;

typedef struct weft_kernel_info {
    const char *symbol;
    const char *march;
    const char *abi;
    size_t vlen_bits;
    size_t argument_count;
    size_t dimension_count;
    const char *const *dimensions;
} weft_kernel_info;

typedef struct weft_context weft_context;
typedef struct weft_kernel weft_kernel;
typedef struct weft_call weft_call;
typedef struct weft_package weft_package;
typedef struct weft_buffer weft_buffer;
typedef struct weft_bindings weft_bindings;
typedef struct weft_sequence weft_sequence;
typedef struct weft_layout weft_layout;
typedef void (*weft_native_entry)(void);

WEFT_API const char *weft_status_name(weft_status status);
/* Discovery uses the calling thread's current CPU affinity. It never widens it. */
WEFT_API weft_status weft_context_create(weft_context **result, weft_error *error);
WEFT_API void weft_context_release(weft_context *context);
WEFT_API weft_status weft_context_target_json(const weft_context *context,
                                              const char **result, weft_error *error);

WEFT_API weft_status weft_kernel_open(weft_context *context, const char *directory,
                                     weft_kernel **result, weft_error *error);
/* Load a Triton cache/export artifact without a second on-disk cache. */
WEFT_API weft_status weft_kernel_open_native(weft_context *context, const char *library_path,
                                            const char *manifest_json,
                                            weft_kernel **result, weft_error *error);
/* CPU launcher callback; checks the actual OpenMP worker before visible writes. */
WEFT_API int weft_kernel_check_worker(const void *kernel, char *message, size_t capacity);
/* Calls retain their library and target context independently of this handle. */
WEFT_API void weft_kernel_release(weft_kernel *kernel);
WEFT_API weft_status weft_kernel_describe(const weft_kernel *kernel,
                                        weft_kernel_info *result, weft_error *error);
WEFT_API weft_status weft_kernel_argument(const weft_kernel *kernel, size_t index,
                                         weft_argument_info *result, weft_error *error);
WEFT_API weft_status weft_kernel_check_thread(const weft_kernel *kernel, weft_error *error);
/* Explicitly narrow the caller's current affinity to the artifact's capable
   CPUs. Never expands affinity; fails if the intersection is empty. */
WEFT_API weft_status weft_kernel_bind_thread(const weft_kernel *kernel, weft_error *error);
/* For a consumer that checked the exact typed ABI and owns a bounded worker
   execution scope. The handle must outlive every direct call. Check the worker
   before each scope, and do not change affinity/vector state within that scope. */
WEFT_API weft_status weft_kernel_get_entry(const weft_kernel *kernel,
                                          weft_native_entry *result, weft_error *error);

WEFT_API weft_status weft_call_prepare(weft_kernel *kernel,
                                      const weft_value *arguments, size_t count,
                                      const weft_dimension *dimensions, size_t dimension_count,
                                      weft_call **result, weft_error *error);
/* Binding copies metadata, not tensor bytes. Caller storage must remain live
   and unchanged in size/layout until release or until a new call is prepared. */
WEFT_API weft_status weft_call_invoke(const weft_call *call, weft_error *error);
WEFT_API void weft_call_release(weft_call *call);

/* Borrow the validated typed ABI packet for a native launcher. The packet and
   scalar storage live until call_release; View bytes remain caller-owned.
   The launcher must use this call's kernel entry and check every executing
   worker, just as a direct entry consumer does. This accessor does not execute
   the kernel or replace worker capability checks. */
WEFT_API weft_status weft_call_arguments(const weft_call *call,
                                        void *const **arguments, size_t *count,
                                        weft_error *error);

/* Prepare a new call from named updates. The source call is immutable; omitted
   values retain their captured storage and scalar values. Inferred dimensions
   are inferred again; explicitly supplied dimensions stay fixed unless updated.
   Updates borrow their storage under the same rules as weft_call_prepare. */
WEFT_API weft_status weft_call_rebind(const weft_call *call,
                                     const weft_named_value *updates, size_t count,
                                     const weft_dimension *dimensions, size_t dimension_count,
                                     weft_call **result, weft_error *error);
/* Named binding builders copy descriptors and scalars at bind time. Their
   mutation does not change any already prepared call. Borrowed storage still
   belongs to the caller; bind_buffer retains an SDK allocation for the call. */
WEFT_API weft_status weft_bindings_create(weft_kernel *kernel, weft_bindings **result,
                                         weft_error *error);
WEFT_API void weft_bindings_release(weft_bindings *bindings);
WEFT_API weft_status weft_bindings_set(weft_bindings *bindings, const char *name,
                                      const weft_value *value, weft_error *error);
WEFT_API weft_status weft_bindings_dimension(weft_bindings *bindings, const char *name,
                                            size_t value, weft_error *error);
WEFT_API weft_status weft_bindings_buffer(weft_bindings *bindings, const char *name,
                                         weft_buffer *buffer, size_t offset,
                                         size_t rank, const size_t *shape,
                                         const size_t *strides, bool readonly,
                                         weft_error *error);
WEFT_API weft_status weft_bindings_prepare(const weft_bindings *bindings,
                                          weft_call **result, weft_error *error);
/* An ordinary, synchronous host call sequence. No graph rewriting, scheduling,
   allocation, packing, implicit workers or state reset occurs on invocation.
   The sequence retains each prepared call. Aliasing between successive calls
   is allowed: the host explicitly owns their order and storage dependencies. */
WEFT_API weft_status weft_sequence_create(weft_call *const *calls, size_t count,
                                         weft_sequence **result, weft_error *error);
WEFT_API weft_status weft_sequence_invoke(const weft_sequence *sequence, weft_error *error);
WEFT_API void weft_sequence_release(weft_sequence *sequence);

typedef struct weft_sequence_update {
    size_t call_index;
    const weft_named_value *values;
    size_t value_count;
    const weft_dimension *dimensions;
    size_t dimension_count;
} weft_sequence_update;

/* Derive immutable sequences. All updates are prepared before publishing the
   result; no entry is executed. Unchanged calls retain their captured bindings,
   owners and execution domains. Cross-call shape/workspace changes are explicit. */
WEFT_API weft_status weft_sequence_replace(const weft_sequence *sequence, size_t index,
                                          const weft_call *call,
                                          weft_sequence **result, weft_error *error);
WEFT_API weft_status weft_sequence_rebind(const weft_sequence *sequence,
                                         const weft_sequence_update *updates, size_t count,
                                         weft_sequence **result, weft_error *error);
WEFT_API weft_status weft_kernel_storage_size(const weft_kernel *kernel, size_t argument,
                                             const weft_dimension *dimensions, size_t dimension_count,
                                             size_t *bytes, weft_error *error);
/* Resolve a View's concrete host storage contract before allocating it. Strides
   are in Encoding elements; NULL requests ordinary contiguous storage. The
   returned shape/stride pointers are borrowed until layout_release. */
typedef struct weft_layout_info {
    const char *encoding;
    size_t rank;
    const size_t *shape;
    const size_t *strides;
    size_t byte_length;
    size_t alignment;
    weft_access access;
} weft_layout_info;
WEFT_API weft_status weft_kernel_layout(const weft_kernel *kernel, size_t argument,
                                       const weft_dimension *dimensions, size_t dimension_count,
                                       const size_t *strides, size_t stride_count,
                                       weft_layout **result, weft_error *error);
WEFT_API weft_status weft_layout_describe(const weft_layout *layout, weft_layout_info *result,
                                         weft_error *error);
WEFT_API weft_status weft_layout_allocate(const weft_layout *layout, weft_buffer **result,
                                         weft_error *error);
WEFT_API void weft_layout_release(weft_layout *layout);

/* The author/host owns workspace and persistent storage. These functions only
   allocate bytes at an explicit size/alignment; they never invent a workspace. */
WEFT_API weft_status weft_buffer_allocate(size_t bytes, size_t alignment,
                                         weft_buffer **result, weft_error *error);
WEFT_API weft_status weft_buffer_data(const weft_buffer *buffer, void **data,
                                     size_t *bytes, weft_error *error);
WEFT_API void weft_buffer_release(weft_buffer *buffer);

WEFT_API weft_status weft_package_open(weft_context *context, const char *directory,
                                      weft_package **result, weft_error *error);
WEFT_API weft_status weft_package_count(const weft_package *package, size_t *count, weft_error *error);
WEFT_API weft_status weft_package_name(const weft_package *package, size_t index,
                                      const char **name, weft_error *error);
WEFT_API weft_status weft_package_kernel(weft_package *package, const char *name,
                                        weft_kernel **result, weft_error *error);
/* Framework bindings map role names to explicit package entries. They do not
   select or alter a compiled program. Returned strings have package lifetime. */
WEFT_API weft_status weft_package_has_binding_group(const weft_package *package, const char *group,
                                                   bool *present, weft_error *error);
WEFT_API weft_status weft_package_binding_count(const weft_package *package, const char *group,
                                               size_t *count, weft_error *error);
WEFT_API weft_status weft_package_binding(const weft_package *package, const char *group, size_t index,
                                         const char **role, const char **kernel, weft_error *error);
WEFT_API void weft_package_release(weft_package *package);

#ifdef __cplusplus
}
#endif
#endif
