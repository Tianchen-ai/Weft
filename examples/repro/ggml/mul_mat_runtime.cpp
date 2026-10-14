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
#include <string>
#include <vector>

namespace {

constexpr std::size_t kN = 4096;
constexpr std::size_t kK = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

struct format_case {
  const char *name;
  ggml_type type;
  const char *implementation;
};

const format_case formats[] = {
    {"f32", GGML_TYPE_F32, "rvv_f32_vec_dot"},
    {"f16", GGML_TYPE_F16, "rvv_f16_vec_dot"},
    {"q1_0", GGML_TYPE_Q1_0, "rvv_quantized_vec_dot"},
    {"q4_0", GGML_TYPE_Q4_0, "rvv_quantized_vec_dot"},
    {"q4_1", GGML_TYPE_Q4_1, "rvv_quantized_vec_dot"},
    {"q5_0", GGML_TYPE_Q5_0, "rvv_quantized_vec_dot"},
    {"q5_1", GGML_TYPE_Q5_1, "rvv_quantized_vec_dot"},
    {"q8_0", GGML_TYPE_Q8_0, "rvv_quantized_vec_dot"},
    {"q2_K", GGML_TYPE_Q2_K, "rvv_quantized_vec_dot"},
    {"q3_K", GGML_TYPE_Q3_K, "rvv_quantized_vec_dot"},
    {"q4_K", GGML_TYPE_Q4_K, "rvv_quantized_vec_dot"},
    {"q5_K", GGML_TYPE_Q5_K, "rvv_quantized_vec_dot"},
    {"q6_K", GGML_TYPE_Q6_K, "rvv_quantized_vec_dot"},
    {"iq1_s", GGML_TYPE_IQ1_S, "rvv_quantized_vec_dot"},
    {"iq1_m", GGML_TYPE_IQ1_M, "rvv_quantized_vec_dot"},
    {"iq2_s", GGML_TYPE_IQ2_S, "rvv_quantized_vec_dot"},
    {"iq2_xs", GGML_TYPE_IQ2_XS, "rvv_quantized_vec_dot"},
    {"iq2_xxs", GGML_TYPE_IQ2_XXS, "rvv_quantized_vec_dot"},
    {"iq3_s", GGML_TYPE_IQ3_S, "rvv_quantized_vec_dot"},
    {"iq3_xxs", GGML_TYPE_IQ3_XXS, "rvv_quantized_vec_dot"},
    {"iq4_nl", GGML_TYPE_IQ4_NL, "rvv_quantized_vec_dot"},
    {"iq4_xs", GGML_TYPE_IQ4_XS, "rvv_quantized_vec_dot"},
    {"tq1_0", GGML_TYPE_TQ1_0, "rvv_quantized_vec_dot"},
    {"tq2_0", GGML_TYPE_TQ2_0, "rvv_quantized_vec_dot"},
    {"mxfp4", GGML_TYPE_MXFP4, "rvv_quantized_vec_dot"},
    {"nvfp4", GGML_TYPE_NVFP4, "scalar_quantized_vec_dot"},
};

struct invocation {
  const format_case *format = nullptr;
  const char *phase = nullptr;
  std::size_t m = 0;
};

volatile std::uint64_t flush_sink = 0;

bool ends_with(const std::string &text, const char *suffix) {
  const std::size_t suffix_size = std::strlen(suffix);
  return text.size() >= suffix_size &&
         text.compare(text.size() - suffix_size, suffix_size, suffix) == 0;
}

invocation find_invocation(const char *name) {
  const std::string requested(name);
  const char *suffix = nullptr;
  invocation result;
  if (ends_with(requested, "_decode")) {
    suffix = "_decode";
    result.phase = "decode";
    result.m = 1;
  } else if (ends_with(requested, "_prefill")) {
    suffix = "_prefill";
    result.phase = "prefill";
    result.m = 128;
  } else {
    return result;
  }

  const std::string format_name =
      requested.substr(0, requested.size() - std::strlen(suffix));
  for (const format_case &candidate : formats) {
    if (format_name == candidate.name) {
      result.format = &candidate;
      return result;
    }
  }
  result.phase = nullptr;
  result.m = 0;
  return result;
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
    std::fprintf(stderr, "usage: %s <weight-format-phase> <repetitions>\n", argv[0]);
    return 2;
  }
  const invocation selected = find_invocation(argv[1]);
  if (selected.format == nullptr) {
    std::fprintf(stderr, "unsupported GGML MUL_MAT case: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  const std::size_t graph_nodes = 16;
  ggml_init_params params = {
      ggml_tensor_overhead() * 8 +
          ggml_graph_overhead_custom(graph_nodes, false),
      nullptr,
      true,
  };
  ggml_context *ctx = ggml_init(params);
  if (ctx == nullptr) {
    std::fprintf(stderr, "failed to create GGML MUL_MAT context\n");
    return 1;
  }

  ggml_tensor *weight =
      ggml_new_tensor_2d(ctx, selected.format->type, kK, kN);
  ggml_tensor *activation =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, selected.m);
  ggml_tensor *output = ggml_mul_mat(ctx, weight, activation);

  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, 1);
  if (!ggml_backend_supports_op(backend, output)) {
    std::fprintf(stderr, "GGML CPU backend does not support MUL_MAT %s\n", argv[1]);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buffer == nullptr) {
    std::fprintf(stderr, "failed to allocate GGML MUL_MAT buffers\n");
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  std::vector<float> activation_values(selected.m * kK);
  if (selected.format->type == GGML_TYPE_F32) {
    std::fill(activation_values.begin(), activation_values.end(),
              1.0F / 4096.0F);
  } else {
    for (std::size_t i = 0; i < activation_values.size(); ++i)
      activation_values[i] =
          static_cast<float>(static_cast<int>((i * 37U) % 251U) - 125) /
          17.0F;
  }

  const std::size_t weight_row_bytes =
      ggml_row_size(selected.format->type, kK);
  std::vector<std::uint8_t> weight_row(weight_row_bytes, 0);
  const char *input_policy = nullptr;
  if (selected.format->type == GGML_TYPE_F32) {
    std::vector<float> dense_row(kK, 1.0F);
    std::memcpy(weight_row.data(), dense_row.data(), weight_row_bytes);
    input_policy = "dense-fixed-values";
  } else if (selected.format->type == GGML_TYPE_F16) {
    for (std::size_t i = 0; i < kK; ++i) {
      const float source =
          static_cast<float>(static_cast<int>((i * 19U) % 127U) - 63) /
          13.0F;
      const ggml_fp16_t value = ggml_fp32_to_fp16(source);
      std::memcpy(weight_row.data() + i * sizeof(value), &value, sizeof(value));
    }
    input_policy = "dense-fixed-values";
  } else {
    const std::int64_t weight_block_elements =
        ggml_blck_size(selected.format->type);
    const std::size_t weight_record_bytes =
        ggml_type_size(selected.format->type);
    if (weight_block_elements <= 0 || weight_record_bytes == 0 ||
        kK % static_cast<std::size_t>(weight_block_elements) != 0 ||
        weight_row_bytes % weight_record_bytes != 0) {
      std::fprintf(stderr, "incomplete quantized input contract for %s\n",
                   argv[1]);
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }

    const std::size_t input_case =
        static_cast<std::size_t>(selected.format - formats - 1);
    std::vector<float> weight_source(
        static_cast<std::size_t>(weight_block_elements));
    std::vector<float> importance(
        static_cast<std::size_t>(weight_block_elements), 1.0F);
    for (std::size_t index = 0; index < weight_source.size(); ++index)
      weight_source[index] =
          static_cast<float>(static_cast<int>(
                                 (index * 19U + input_case * 7U) % 127U) -
                             63) /
          13.0F;
    std::vector<std::uint8_t> weight_record(weight_record_bytes);
    const std::size_t encoded_bytes = ggml_quantize_chunk(
        selected.format->type, weight_source.data(), weight_record.data(), 0,
        1, weight_block_elements, importance.data());
    if (encoded_bytes != weight_record_bytes) {
      std::fprintf(stderr, "failed to construct valid encoded weights\n");
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }
    for (std::size_t offset = 0; offset < weight_row.size();
         offset += weight_record.size())
      std::memcpy(weight_row.data() + offset, weight_record.data(),
                  weight_record.size());
    input_policy = "valid-quantized-record-replicated";
  }
  for (std::size_t row = 0; row < kN; ++row) {
    ggml_backend_tensor_set(weight, weight_row.data(), row * weight_row_bytes,
                            weight_row_bytes);
  }

  ggml_backend_tensor_set(activation, activation_values.data(), 0,
                          ggml_nbytes(activation));

  ggml_cgraph *graph = ggml_new_graph_custom(ctx, graph_nodes, false);
  ggml_build_forward_expand(graph, output);
  ggml_status status = ggml_backend_graph_compute(backend, graph);
  if (status != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "MUL_MAT warmup failed for %s: %s\n", argv[1],
                 ggml_status_to_string(status));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  std::vector<float> output_values(
      static_cast<std::size_t>(ggml_nelements(output)));
  ggml_backend_tensor_get(output, output_values.data(), 0, ggml_nbytes(output));
  for (float value : output_values) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "non-finite MUL_MAT output from %s\n", argv[1]);
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }
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
      std::fprintf(stderr, "MUL_MAT compute failed for %s: %s\n", argv[1],
                   ggml_status_to_string(status));
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double cold_median_us = median(samples);
  const double operations = 2.0 * static_cast<double>(selected.m) *
                            static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=mul_mat\n");
  std::printf("kernel=mul_mat_%s\n", selected.format->name);
  std::printf("phase=%s\n", selected.phase);
  std::printf("implementation=%s\n", selected.format->implementation);
  std::printf("model_shape=Llama-8B.hidden_projection\n");
  std::printf("M=%zu\nN=%zu\nK=%zu\n", selected.m, kN, kK);
  std::printf("threads=1\n");
  std::printf("vlen_bits=%d\n", ggml_cpu_get_rvv_vlen() * 8);
  std::printf("cold_protocol=64MiB-evict-then-single-op-graph\n");
  std::printf("input_policy=%s\n", input_policy);
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", cold_median_us);
  std::printf("cold_gop_s=%.6f\n", operations / cold_median_us / 1.0e3);
  std::printf("output_sample=%.9g\n", output_values[0]);

  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  ggml_free(ctx);
  return 0;
}
