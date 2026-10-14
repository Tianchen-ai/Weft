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
#include <vector>

namespace {

constexpr std::size_t kK = 4096;
constexpr std::size_t kN = 14336;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

using dot_fn = void (*)(int, float *, std::size_t, const void *, std::size_t,
                        const void *, std::size_t, int);
struct dot_case {
  const char *name;
  ggml_type weight_type;
  ggml_type activation_type;
  dot_fn dot;
  const char *implementation;
};

const dot_case cases[] = {
    {"q1_0_q8_0", GGML_TYPE_Q1_0, GGML_TYPE_Q8_0,
     ggml_vec_dot_q1_0_q8_0, "rvv_intrinsic"},
    {"q4_0_q8_0", GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
     ggml_vec_dot_q4_0_q8_0, "rvv_intrinsic"},
    {"q4_1_q8_1", GGML_TYPE_Q4_1, GGML_TYPE_Q8_1,
     ggml_vec_dot_q4_1_q8_1, "rvv_intrinsic"},
    {"q5_0_q8_0", GGML_TYPE_Q5_0, GGML_TYPE_Q8_0,
     ggml_vec_dot_q5_0_q8_0, "rvv_intrinsic"},
    {"q5_1_q8_1", GGML_TYPE_Q5_1, GGML_TYPE_Q8_1,
     ggml_vec_dot_q5_1_q8_1, "rvv_intrinsic"},
    {"q8_0_q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
     ggml_vec_dot_q8_0_q8_0, "rvv_intrinsic"},
    {"q2_K_q8_K", GGML_TYPE_Q2_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q2_K_q8_K, "rvv_intrinsic"},
    {"q3_K_q8_K", GGML_TYPE_Q3_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q3_K_q8_K, "rvv_intrinsic"},
    {"q4_K_q8_K", GGML_TYPE_Q4_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q4_K_q8_K, "rvv_intrinsic"},
    {"q5_K_q8_K", GGML_TYPE_Q5_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q5_K_q8_K, "rvv_intrinsic"},
    {"q6_K_q8_K", GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q6_K_q8_K, "rvv_intrinsic"},
    {"iq1_s_q8_K", GGML_TYPE_IQ1_S, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq1_s_q8_K, "rvv_intrinsic"},
    {"iq1_m_q8_K", GGML_TYPE_IQ1_M, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq1_m_q8_K, "rvv_intrinsic"},
    {"iq2_s_q8_K", GGML_TYPE_IQ2_S, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq2_s_q8_K, "rvv_intrinsic"},
    {"iq2_xs_q8_K", GGML_TYPE_IQ2_XS, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq2_xs_q8_K, "rvv_intrinsic"},
    {"iq2_xxs_q8_K", GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq2_xxs_q8_K, "rvv_intrinsic"},
    {"iq3_s_q8_K", GGML_TYPE_IQ3_S, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq3_s_q8_K, "rvv_intrinsic"},
    {"iq3_xxs_q8_K", GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq3_xxs_q8_K, "rvv_intrinsic"},
    {"iq4_nl_q8_0", GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0,
     ggml_vec_dot_iq4_nl_q8_0, "rvv_intrinsic"},
    {"iq4_xs_q8_K", GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq4_xs_q8_K, "rvv_intrinsic"},
    {"tq1_0_q8_K", GGML_TYPE_TQ1_0, GGML_TYPE_Q8_K,
     ggml_vec_dot_tq1_0_q8_K, "rvv_intrinsic"},
    {"tq2_0_q8_K", GGML_TYPE_TQ2_0, GGML_TYPE_Q8_K,
     ggml_vec_dot_tq2_0_q8_K, "rvv_intrinsic"},
    {"mxfp4_q8_0", GGML_TYPE_MXFP4, GGML_TYPE_Q8_0,
     ggml_vec_dot_mxfp4_q8_0, "rvv_intrinsic"},
    {"nvfp4_q8_0", GGML_TYPE_NVFP4, GGML_TYPE_Q8_0,
     ggml_vec_dot_nvfp4_q8_0, "scalar"},
};

volatile std::uint64_t flush_sink = 0;

const dot_case *find_case(const char *name) {
  for (const dot_case &candidate : cases) {
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

void run_projection(const dot_case &selected, const std::uint8_t *weights,
                    const std::uint8_t *activation, std::size_t weight_row_bytes,
                    float *output) {
  for (std::size_t row = 0; row < kN; ++row) {
    selected.dot(static_cast<int>(kK), &output[row], 0,
                 weights + row * weight_row_bytes, 0, activation, 0, 1);
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
    std::fprintf(stderr, "usage: %s <vec-dot-kernel> <repetitions>\n", argv[0]);
    return 2;
  }

  const dot_case *selected = find_case(argv[1]);
  if (selected == nullptr) {
    std::fprintf(stderr, "unsupported GGML vec-dot kernel: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  const std::size_t weight_row_bytes = ggml_row_size(selected->weight_type, kK);
  const std::size_t activation_row_bytes =
      ggml_row_size(selected->activation_type, kK);
  const std::int64_t weight_block_elements =
      ggml_blck_size(selected->weight_type);
  const std::int64_t activation_block_elements =
      ggml_blck_size(selected->activation_type);
  const std::size_t weight_record_bytes =
      ggml_type_size(selected->weight_type);
  if (weight_block_elements <= 0 || activation_block_elements <= 0 ||
      weight_block_elements % activation_block_elements != 0 ||
      weight_record_bytes == 0 ||
      kK % static_cast<std::size_t>(weight_block_elements) != 0) {
    std::fprintf(stderr, "incompatible vec-dot block geometry for %s\n",
                 argv[1]);
    return 1;
  }
  const std::size_t activation_record_bytes =
      ggml_type_size(selected->activation_type) *
      static_cast<std::size_t>(weight_block_elements /
                               activation_block_elements);
  if (activation_record_bytes == 0 ||
      weight_row_bytes % weight_record_bytes != 0 ||
      activation_row_bytes % activation_record_bytes != 0) {
    std::fprintf(stderr, "incompatible vec-dot block geometry for %s\n",
                 argv[1]);
    return 1;
  }

  const std::size_t input_case = static_cast<std::size_t>(selected - cases);
  std::vector<float> weight_source(
      static_cast<std::size_t>(weight_block_elements));
  std::vector<float> activation_source(
      static_cast<std::size_t>(weight_block_elements));
  std::vector<float> importance(
      static_cast<std::size_t>(weight_block_elements), 1.0F);
  for (std::size_t index = 0; index < weight_source.size(); ++index) {
    weight_source[index] =
        static_cast<float>(static_cast<int>(
                               (index * 19U + input_case * 7U) % 127U) -
                           63) /
        13.0F;
    activation_source[index] =
        static_cast<float>(static_cast<int>((index * 37U) % 251U) - 125) /
        17.0F;
  }
  std::vector<std::uint8_t> weight_record(weight_record_bytes);
  std::vector<std::uint8_t> activation_record(activation_record_bytes);
  const std::size_t encoded_bytes = ggml_quantize_chunk(
      selected->weight_type, weight_source.data(), weight_record.data(), 0, 1,
      weight_block_elements, importance.data());
  const auto *activation_traits =
      ggml_get_type_traits_cpu(selected->activation_type);
  if (encoded_bytes != weight_record_bytes || activation_traits == nullptr ||
      activation_traits->from_float == nullptr) {
    std::fprintf(stderr, "failed to construct valid encoded input for %s\n",
                 argv[1]);
    return 1;
  }
  activation_traits->from_float(activation_source.data(),
                                activation_record.data(),
                                weight_block_elements);

  float record_output = 0.0F;
  ggml_cpu_init();
  selected->dot(static_cast<int>(weight_block_elements), &record_output, 0,
                weight_record.data(), 0, activation_record.data(), 0, 1);
  if (!std::isfinite(record_output)) {
    std::fprintf(stderr, "valid encoded input produced a non-finite result\n");
    return 1;
  }

  std::vector<std::uint8_t> weight_row(weight_row_bytes);
  for (std::size_t offset = 0; offset < weight_row.size();
       offset += weight_record.size())
    std::memcpy(weight_row.data() + offset, weight_record.data(),
                weight_record.size());
  std::vector<std::uint8_t> weights(weight_row_bytes * kN);
  for (std::size_t row = 0; row < kN; ++row) {
    std::memcpy(weights.data() + row * weight_row_bytes, weight_row.data(),
                weight_row_bytes);
  }

  std::vector<std::uint8_t> activation(activation_row_bytes);
  for (std::size_t offset = 0; offset < activation.size();
       offset += activation_record.size())
    std::memcpy(activation.data() + offset, activation_record.data(),
                activation_record.size());
  std::vector<float> output(kN);
  std::vector<std::uint8_t> flush_buffer(kFlushBytes, 1);

  ggml_backend_t backend = ggml_backend_cpu_init();
  const int vlen_bytes = ggml_cpu_get_rvv_vlen();

  run_projection(*selected, weights.data(), activation.data(), weight_row_bytes,
                 output.data());
  for (float value : output) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "non-finite output from %s\n", argv[1]);
      ggml_backend_free(backend);
      return 1;
    }
  }

  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush_buffer);
    const auto begin = std::chrono::steady_clock::now();
    run_projection(*selected, weights.data(), activation.data(), weight_row_bytes,
                   output.data());
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double cold_median_us = median(samples);
  const double operations = 2.0 * static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=vec_dot\n");
  std::printf("kernel=%s\n", selected->name);
  std::printf("implementation=%s\n", selected->implementation);
  std::printf("model_shape=DeepSeek-R1-Distill-Llama-8B.ffn_up\n");
  std::printf("M=1\nN=%zu\nK=%zu\n", kN, kK);
  std::printf("vlen_bits=%d\n", vlen_bytes * 8);
  std::printf("cold_protocol=64MiB-evict-then-full-projection\n");
  std::printf("input_policy=valid-quantized-record-replicated\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", cold_median_us);
  std::printf("cold_gop_s=%.6f\n", operations / cold_median_us / 1.0e3);
  std::printf("output_sample=%.9g\n", output[0]);

  ggml_backend_free(backend);
  return 0;
}
