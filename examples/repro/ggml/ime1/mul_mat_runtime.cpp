#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ime.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sched.h>

#include "contract.h"

using namespace ime1_repro;

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <ime1-kernel-phase> <repetitions>\n", argv[0]);
    return 2;
  }
  const ime_case *selected = find_case(argv[1]);
  if (selected == nullptr) {
    std::fprintf(stderr, "unsupported GGML IME1 case: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }
  if (std::fegetround() != FE_TONEAREST) {
    std::fprintf(stderr, "IME1 comparison requires round-to-nearest-even\n");
    return 2;
  }
  cpu_set_t execution_cpus;
  if (sched_getaffinity(0, sizeof(execution_cpus), &execution_cpus) != 0 ||
      CPU_COUNT(&execution_cpus) != 1) {
    std::fprintf(stderr, "IME1 baseline requires one explicitly selected CPU\n");
    return 2;
  }

  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, 1);

  ggml_init_params weight_params = {
      ggml_tensor_overhead() * 4,
      nullptr,
      true,
  };
  ggml_context *weight_ctx = ggml_init(weight_params);
  ggml_tensor *weight =
      ggml_new_tensor_2d(weight_ctx, selected->weight_type, kK, kN);
  ggml_backend_buffer_type_t spacemit_buft =
      ggml_backend_cpu_riscv64_spacemit_buffer_type();
  ggml_backend_buffer_t weight_buffer =
      ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, spacemit_buft);
  if (weight_buffer == nullptr || weight->extra == nullptr) {
    std::fprintf(stderr, "IME1 repack trait unavailable for %s\n", selected->name);
    if (weight_buffer != nullptr) {
      ggml_backend_buffer_free(weight_buffer);
    }
    ggml_free(weight_ctx);
    ggml_backend_free(backend);
    return 1;
  }

  std::vector<float> source_row(kK);
  for (std::size_t i = 0; i < source_row.size(); ++i) {
    source_row[i] =
        static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0F;
  }
  const std::size_t plain_row_bytes =
      ggml_row_size(selected->weight_type, kK);
  std::vector<std::uint8_t> plain_row(plain_row_bytes);
  selected->quantize_weight(source_row.data(), plain_row.data(), kK);
  std::vector<std::uint8_t> plain_weight(plain_row_bytes * kN);
  for (std::size_t row = 0; row < kN; ++row) {
    std::memcpy(plain_weight.data() + row * plain_row_bytes, plain_row.data(),
                plain_row_bytes);
  }
  ggml_backend_tensor_set(weight, plain_weight.data(), 0, ggml_nbytes(weight));
  const auto expected_weights = reference_weights(*selected, plain_row);
  if (!check_packed_weights(*selected, weight->data, expected_weights)) {
    std::fprintf(stderr, "IME1 persistent repack disagrees with the source projection\n");
    return 1;
  }

  const std::size_t graph_nodes = 16;
  ggml_init_params graph_params = {
      ggml_tensor_overhead() * 8 +
          ggml_graph_overhead_custom(graph_nodes, false),
      nullptr,
      true,
  };
  ggml_context *graph_ctx = ggml_init(graph_params);
  ggml_tensor *activation =
      ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, kK, selected->m);
  ggml_tensor *output = ggml_mul_mat(graph_ctx, weight, activation);
  ggml_backend_buffer_t graph_buffer =
      ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
  if (graph_buffer == nullptr || !ggml_backend_supports_op(backend, output)) {
    std::fprintf(stderr, "IME1 MUL_MAT graph unavailable for %s\n", selected->name);
    if (graph_buffer != nullptr) {
      ggml_backend_buffer_free(graph_buffer);
    }
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_free(backend);
    return 1;
  }

  std::vector<float> activation_values(selected->m * kK);
  for (std::size_t i = 0; i < activation_values.size(); ++i) {
    activation_values[i] =
        static_cast<float>(static_cast<int>(i % 17) - 8) / 9.0F;
  }
  ggml_backend_tensor_set(activation, activation_values.data(), 0,
                          ggml_nbytes(activation));
  const auto expected_output = reference_output(*selected, expected_weights,
                                                activation_values);

  ggml_cgraph *graph = ggml_new_graph_custom(graph_ctx, graph_nodes, false);
  ggml_build_forward_expand(graph, output);
  ggml_status status = ggml_backend_graph_compute(backend, graph);
  if (status != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "IME1 warmup failed for %s: %s\n", selected->name,
                 ggml_status_to_string(status));
    ggml_backend_buffer_free(graph_buffer);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_free(backend);
    return 1;
  }

  std::vector<float> output_values(
      static_cast<std::size_t>(ggml_nelements(output)));
  ggml_backend_tensor_get(output, output_values.data(), 0, ggml_nbytes(output));
  double max_absolute = 0.0, max_relative = 0.0;
  for (std::size_t index = 0; index < output_values.size(); ++index) {
    const float expected = expected_output[index / kN];
    const double absolute = std::fabs(static_cast<double>(output_values[index]) - expected);
    const double relative = absolute / std::max(std::fabs(static_cast<double>(expected)), 1.0e-30);
    if (!std::isfinite(output_values[index]) || !std::isfinite(expected) ||
        absolute > kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected)) {
      std::fprintf(stderr, "IME1 output mismatch at %zu: actual=%a expected=%a\n",
                   index, output_values[index], expected);
      return 1;
    }
    max_absolute = std::max(max_absolute, absolute);
    max_relative = std::max(max_relative, relative);
  }
  cpu_set_t actual_cpus;
  if (sched_getaffinity(0, sizeof(actual_cpus), &actual_cpus) != 0 ||
      !CPU_EQUAL(&execution_cpus, &actual_cpus)) {
    std::fprintf(stderr, "GGML changed the explicitly selected execution CPU\n");
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
      std::fprintf(stderr, "IME1 compute failed for %s: %s\n", selected->name,
                   ggml_status_to_string(status));
      ggml_backend_buffer_free(graph_buffer);
      ggml_free(graph_ctx);
      ggml_backend_buffer_free(weight_buffer);
      ggml_free(weight_ctx);
      ggml_backend_free(backend);
      return 1;
    }
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }

  ggml_backend_tensor_get(output, output_values.data(), 0, ggml_nbytes(output));
  for (float value : output_values) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "non-finite IME1 output from %s\n", selected->name);
      ggml_backend_buffer_free(graph_buffer);
      ggml_free(graph_ctx);
      ggml_backend_buffer_free(weight_buffer);
      ggml_free(weight_ctx);
      ggml_backend_free(backend);
      return 1;
    }
  }
  const float output_sample = output_values[0];

  const double cold_median_us = median(samples);
  const double operations = 2.0 * static_cast<double>(selected->m) *
                            static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=ime1\n");
  std::printf("kernel=ime1_%s\n", selected->format);
  std::printf("phase=%s\n", selected->phase);
  std::printf("implementation=handwritten_ime1_asm\n");
  std::printf("target=SpacemiT-X60\n");
  std::printf("model_shape=Llama-8B.hidden_projection\n");
  std::printf("M=%zu\nN=%zu\nK=%zu\n", selected->m, kN, kK);
  std::printf("vlen_bits=%d\n", ggml_cpu_get_rvv_vlen() * 8);
  std::printf("weight_buffer=%s\n", ggml_backend_buffer_name(weight_buffer));
  std::printf("numeric=within-tolerance\nweight_storage=exact\nrounding=nearest-even\n");
  std::printf("absolute_tolerance=%.9g\nrelative_tolerance=%.9g\n", kAbsoluteTolerance, kRelativeTolerance);
  std::printf("max_absolute_error=%.9g\nmax_relative_error=%.9g\n", max_absolute, max_relative);
  std::printf("input_policy=quantized-period31-weight-row-replicated;period17-activation\n");
  std::printf("scope=production-activation-quantize-plus-ime1-gemm\n");
  std::printf("cold_protocol=64MiB-evict-then-single-op-graph\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", cold_median_us);
  std::printf("cold_gop_s=%.6f\n", operations / cold_median_us / 1.0e3);
  std::printf("output_sample=%.9g\n", output_sample);

  ggml_backend_buffer_free(graph_buffer);
  ggml_free(graph_ctx);
  ggml_backend_buffer_free(weight_buffer);
  ggml_free(weight_ctx);
  ggml_backend_free(backend);
  return 0;
}
