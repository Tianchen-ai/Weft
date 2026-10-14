#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

struct forward_case {
  const char *name;
  const char *implementation;
  const char *model_shape;
};

#if defined(GGML_BASELINE_K1)
const forward_case cases[] = {
    {"add", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"mul", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"scale", "rvv_intrinsic", "hidden[128,4096]"},
    {"cpy", "scalar_memcpy", "hidden[128,4096]"},
    {"gelu", "scalar", "ffn[128,14336]"},
    {"silu", "rvv_intrinsic", "ffn[128,14336]"},
    {"rms_norm", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"softmax", "rvv_intrinsic", "attention[heads=32,Q=128,K=128]"},
    {"rope", "scalar", "q[head_dim=128,heads=32,tokens=128]"},
    {"sub", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"div", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"norm", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"cont", "handwritten_rvv_inline_asm", "transpose(hidden[128,4096])"},
    {"get_rows", "scalar", "embedding[vocab=128256,hidden=4096],tokens=128"},
    {"repeat", "spacemit_rvv_intrinsic", "norm_weight[4096]->hidden[128,4096]"},
    {"sum_rows", "spacemit_rvv_intrinsic", "hidden[128,4096]"},
    {"concat", "scalar", "hidden[64+64,4096]"},
    {"flash_attn", "scalar_mixed", "Q=128,KV=128,H=32,Hkv=8,D=128"},
    {"get_rows_f32", "handwritten_rvv_inline_asm", "TinyLlama embedding[vocab=32000,hidden=2048],tokens=128"},
    {"concat_dim0", "spacemit_scalar", "hidden[128,2048+2048]"},
};
#else
const forward_case cases[] = {
    {"add", "scalar", "hidden[128,4096]"},
    {"mul", "scalar", "hidden[128,4096]"},
    {"scale", "rvv_intrinsic", "hidden[128,4096]"},
    {"cpy", "scalar_memcpy", "hidden[128,4096]"},
    {"gelu", "scalar", "ffn[128,14336]"},
    {"silu", "rvv_intrinsic", "ffn[128,14336]"},
    {"rms_norm", "scalar", "hidden[128,4096]"},
    {"softmax", "rvv_intrinsic", "attention[heads=32,Q=128,K=128]"},
    {"rope", "scalar", "q[head_dim=128,heads=32,tokens=128]"},
    {"sub", "scalar", "hidden[128,4096]"},
    {"div", "scalar", "hidden[128,4096]"},
    {"norm", "scalar", "hidden[128,4096]"},
    {"cont", "scalar_memcpy", "transpose(hidden[128,4096])"},
    {"get_rows", "scalar", "embedding[vocab=128256,hidden=4096],tokens=128"},
    {"repeat", "scalar", "norm_weight[4096]->hidden[128,4096]"},
    {"sum_rows", "scalar", "hidden[128,4096]"},
    {"concat", "scalar", "hidden[64+64,4096]"},
    {"flash_attn", "scalar_mixed", "Q=128,KV=128,H=32,Hkv=8,D=128"},
};
#endif

struct built_case {
  ggml_tensor *a = nullptr;
  ggml_tensor *b = nullptr;
  ggml_tensor *c = nullptr;
  ggml_tensor *mask = nullptr;
  ggml_tensor *indices = nullptr;
  ggml_tensor *positions = nullptr;
  ggml_tensor *output = nullptr;
};

volatile std::uint64_t flush_sink = 0;

const forward_case *find_case(const char *name) {
  for (const forward_case &candidate : cases) {
    if (std::strcmp(name, candidate.name) == 0) {
      return &candidate;
    }
  }
  return nullptr;
}

std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0) {
    return 0;
  }
  return static_cast<std::size_t>(value);
}

void evict_cache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t i = 0; i < buffer.size(); i += 64) {
    buffer[i] = static_cast<std::uint8_t>(buffer[i] + 1U);
    flush_sink += buffer[i];
  }
}

built_case build(ggml_context *ctx, const char *name) {
  built_case result;
  if (std::strcmp(name, "add") == 0 || std::strcmp(name, "sub") == 0 ||
      std::strcmp(name, "mul") == 0 || std::strcmp(name, "div") == 0 ||
      std::strcmp(name, "cpy") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    if (std::strcmp(name, "add") == 0) {
      result.output = ggml_add(ctx, result.a, result.b);
    } else if (std::strcmp(name, "sub") == 0) {
      result.output = ggml_sub(ctx, result.a, result.b);
    } else if (std::strcmp(name, "mul") == 0) {
      result.output = ggml_mul(ctx, result.a, result.b);
    } else if (std::strcmp(name, "div") == 0) {
      result.output = ggml_div(ctx, result.a, result.b);
    } else {
      result.output = ggml_cpy(ctx, result.a, result.b);
    }
    return result;
  }
  if (std::strcmp(name, "scale") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.output = ggml_scale(ctx, result.a, 0.5F);
    return result;
  }
  if (std::strcmp(name, "gelu") == 0 || std::strcmp(name, "silu") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 14336, 128);
    result.output = std::strcmp(name, "gelu") == 0 ? ggml_gelu(ctx, result.a)
                                                   : ggml_silu(ctx, result.a);
    return result;
  }
  if (std::strcmp(name, "rms_norm") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.output = ggml_rms_norm(ctx, result.a, 1.0e-5F);
    return result;
  }
  if (std::strcmp(name, "norm") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.output = ggml_norm(ctx, result.a, 1.0e-5F);
    return result;
  }
  if (std::strcmp(name, "softmax") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 32, 1);
    result.output = ggml_soft_max(ctx, result.a);
    return result;
  }
  if (std::strcmp(name, "rope") == 0) {
    result.a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 32, 128);
    result.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 128);
    result.output =
        ggml_rope(ctx, result.a, result.positions, 128, GGML_ROPE_TYPE_NEOX);
    return result;
  }
  if (std::strcmp(name, "cont") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.output = ggml_cont(ctx, ggml_transpose(ctx, result.a));
    return result;
  }
  if (std::strcmp(name, "get_rows") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 128256);
    result.indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 128);
    result.output = ggml_get_rows(ctx, result.a, result.indices);
    return result;
  }
  if (std::strcmp(name, "get_rows_f32") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 32000);
    result.indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 128);
    result.output = ggml_get_rows(ctx, result.a, result.indices);
    return result;
  }
  if (std::strcmp(name, "repeat") == 0) {
    result.a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.output = ggml_repeat(ctx, result.a, result.b);
    return result;
  }
  if (std::strcmp(name, "sum_rows") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 128);
    result.output = ggml_sum_rows(ctx, result.a);
    return result;
  }
  if (std::strcmp(name, "concat") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 64);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 64);
    result.output = ggml_concat(ctx, result.a, result.b, 1);
    return result;
  }
  if (std::strcmp(name, "concat_dim0") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
    result.output = ggml_concat(ctx, result.a, result.b, 0);
    return result;
  }
  if (std::strcmp(name, "flash_attn") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 32, 1);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 128, 8, 1);
    result.c = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 128, 8, 1);
    result.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 128, 1, 1);
    result.output = ggml_flash_attn_ext(
        ctx, result.a, result.b, result.c, result.mask,
        1.0F / std::sqrt(128.0F), 0.0F, 0.0F);
    return result;
  }
  return result;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <forward-kernel> <repetitions>\n", argv[0]);
    return 2;
  }
  const forward_case *selected = find_case(argv[1]);
  if (selected == nullptr) {
    std::fprintf(stderr, "unsupported GGML forward kernel: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  const std::size_t graph_nodes = 64;
  ggml_init_params params = {
      ggml_tensor_overhead() * 32 +
          ggml_graph_overhead_custom(graph_nodes, false),
      nullptr,
      true,
  };
  ggml_context *ctx = ggml_init(params);
  if (ctx == nullptr) {
    std::fprintf(stderr, "failed to create GGML context\n");
    return 1;
  }
  built_case operation = build(ctx, selected->name);
  if (operation.output == nullptr) {
    std::fprintf(stderr, "failed to build GGML forward kernel: %s\n", selected->name);
    ggml_free(ctx);
    return 1;
  }

  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, 1);
  if (!ggml_backend_supports_op(backend, operation.output)) {
    std::fprintf(stderr, "GGML CPU backend does not support: %s\n", selected->name);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buffer == nullptr) {
    std::fprintf(stderr, "failed to allocate GGML forward buffers\n");
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  if (operation.a->type == GGML_TYPE_F32) {
    std::vector<float> a_values(
        static_cast<std::size_t>(ggml_nelements(operation.a)));
    for (std::size_t i = 0; i < a_values.size(); ++i) {
      a_values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0F;
    }
    ggml_backend_tensor_set(operation.a, a_values.data(), 0,
                            ggml_nbytes(operation.a));
  } else if (operation.a->type == GGML_TYPE_Q4_K) {
    constexpr std::size_t embedding_width = 4096;
    const std::size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, embedding_width);
    std::vector<float> source_row(embedding_width);
    for (std::size_t i = 0; i < source_row.size(); ++i) {
      source_row[i] =
          static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0F;
    }
    std::vector<std::uint8_t> quantized_row(row_bytes);
    quantize_row_q4_K(source_row.data(), quantized_row.data(), embedding_width);
    std::vector<std::uint8_t> embedding(ggml_nbytes(operation.a));
    for (std::size_t offset = 0; offset < embedding.size(); offset += row_bytes) {
      std::memcpy(embedding.data() + offset, quantized_row.data(), row_bytes);
    }
    ggml_backend_tensor_set(operation.a, embedding.data(), 0,
                            ggml_nbytes(operation.a));
  } else {
    std::fprintf(stderr, "unsupported input type for %s\n", selected->name);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  if (operation.b != nullptr) {
    if (operation.b->type == GGML_TYPE_F32) {
      std::vector<float> b_values(
          static_cast<std::size_t>(ggml_nelements(operation.b)));
      for (std::size_t i = 0; i < b_values.size(); ++i) {
        b_values[i] = std::strcmp(selected->name, "div") == 0
                          ? static_cast<float>((i % 17) + 1) / 9.0F
                          : static_cast<float>(static_cast<int>(i % 17) - 8) /
                                9.0F;
      }
      ggml_backend_tensor_set(operation.b, b_values.data(), 0,
                              ggml_nbytes(operation.b));
    } else if (operation.b->type == GGML_TYPE_F16) {
      std::vector<ggml_fp16_t> b_values(
          static_cast<std::size_t>(ggml_nelements(operation.b)));
      for (std::size_t i = 0; i < b_values.size(); ++i) {
        const float value =
            static_cast<float>(static_cast<int>(i % 17) - 8) / 9.0F;
        b_values[i] = ggml_fp32_to_fp16(value);
      }
      ggml_backend_tensor_set(operation.b, b_values.data(), 0,
                              ggml_nbytes(operation.b));
    }
  }
  if (operation.c != nullptr) {
    std::vector<ggml_fp16_t> c_values(
        static_cast<std::size_t>(ggml_nelements(operation.c)));
    for (std::size_t i = 0; i < c_values.size(); ++i) {
      const float value =
          static_cast<float>(static_cast<int>(i % 13) - 6) / 7.0F;
      c_values[i] = ggml_fp32_to_fp16(value);
    }
    ggml_backend_tensor_set(operation.c, c_values.data(), 0,
                            ggml_nbytes(operation.c));
  }
  if (operation.mask != nullptr) {
    std::vector<ggml_fp16_t> mask_values(
        static_cast<std::size_t>(ggml_nelements(operation.mask)));
    for (std::size_t query = 0; query < 128; ++query) {
      for (std::size_t key = 0; key < 128; ++key) {
        const float value =
            key <= query ? 0.0F : -std::numeric_limits<float>::infinity();
        mask_values[query * 128 + key] = ggml_fp32_to_fp16(value);
      }
    }
    ggml_backend_tensor_set(operation.mask, mask_values.data(), 0,
                            ggml_nbytes(operation.mask));
  }
  if (operation.indices != nullptr) {
    std::vector<std::int32_t> index_values(
        static_cast<std::size_t>(ggml_nelements(operation.indices)));
    const std::size_t vocabulary =
        std::strcmp(selected->name, "get_rows_f32") == 0 ? 32000U : 128256U;
    for (std::size_t i = 0; i < index_values.size(); ++i) {
      index_values[i] =
          static_cast<std::int32_t>((i * 997U) % vocabulary);
    }
    ggml_backend_tensor_set(operation.indices, index_values.data(), 0,
                            ggml_nbytes(operation.indices));
  }
  std::vector<std::int32_t> position_values;
  if (operation.positions != nullptr) {
    position_values.resize(
        static_cast<std::size_t>(ggml_nelements(operation.positions)));
    for (std::size_t i = 0; i < position_values.size(); ++i) {
      position_values[i] = static_cast<std::int32_t>(i);
    }
    ggml_backend_tensor_set(operation.positions, position_values.data(), 0,
                            ggml_nbytes(operation.positions));
  }

  ggml_cgraph *graph = ggml_new_graph_custom(ctx, graph_nodes, false);
  ggml_build_forward_expand(graph, operation.output);
  ggml_status status = ggml_backend_graph_compute(backend, graph);
  if (status != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "warmup failed for %s: %s\n", selected->name,
                 ggml_status_to_string(status));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  std::vector<std::uint8_t> flush_buffer(kFlushBytes, 1);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush_buffer);
    const auto begin = std::chrono::steady_clock::now();
    status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    if (status != GGML_STATUS_SUCCESS) {
      std::fprintf(stderr, "compute failed for %s: %s\n", selected->name,
                   ggml_status_to_string(status));
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }

  std::vector<float> output_values(
      static_cast<std::size_t>(ggml_nelements(operation.output)));
  ggml_backend_tensor_get(operation.output, output_values.data(), 0,
                          ggml_nbytes(operation.output));
  for (float value : output_values) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "non-finite output from %s\n", selected->name);
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }
  }
  const float output_sample = output_values[0];

  std::printf("family=forward\n");
  std::printf("kernel=%s\n", selected->name);
  std::printf("implementation=%s\n", selected->implementation);
  std::printf("model=%s\n",
              std::strcmp(selected->name, "get_rows_f32") == 0
                  ? "TinyLlama-1.1B"
                  : "DeepSeek-R1-Distill-Llama-8B");
  std::printf("model_shape=%s\n", selected->model_shape);
  std::printf("threads=1\n");
  std::printf("vlen_bits=%d\n", ggml_cpu_get_rvv_vlen() * 8);
  std::printf("cold_protocol=64MiB-evict-then-single-op-graph\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", median(samples));
  std::printf("output_sample=%.9g\n", output_sample);

  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  ggml_free(ctx);
  return 0;
}
