#include "ggml-quants.h"
#include "ggml.h"
#include "quants.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef WEFT_VEC_DOT_FORMAT
#error "WEFT_VEC_DOT_FORMAT must select one quantized vec-dot kernel"
#endif

namespace {
constexpr double kAbsoluteTolerance = 1.0e-4;
constexpr double kRelativeTolerance = 2.0e-3;

bool within_tolerance(float actual, float expected, double &max_absolute,
                      double &max_relative) {
  if (!std::isfinite(actual) || !std::isfinite(expected))
    return false;
  const double absolute =
      std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
  const double relative =
      absolute / std::max(std::fabs(static_cast<double>(expected)), 1.0e-30);
  max_absolute = std::max(max_absolute, absolute);
  max_relative = std::max(max_relative, relative);
  return absolute <=
         kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
}

#if defined(WEFT_TARGET_K1)
constexpr const char *kTarget = "K1/X60";
#else
constexpr const char *kTarget = "SG2044";
#endif
constexpr int kElements = 4096;
constexpr std::size_t kRows = 14336;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
volatile std::uint64_t flushSink = 0;

constexpr ggml_type kWeightTypes[] = {
    GGML_TYPE_Q1_0,    GGML_TYPE_Q4_0,   GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,    GGML_TYPE_Q5_1,   GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,    GGML_TYPE_Q3_K,   GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,    GGML_TYPE_Q6_K,   GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_S,  GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,  GGML_TYPE_IQ4_XS, GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,   GGML_TYPE_MXFP4,  GGML_TYPE_NVFP4,
};

[[maybe_unused]] void quantize_activation(const float *source,
                                          block_q8_0 *destination,
                                          std::int64_t count) {
  quantize_row_q8_0_ref(source, destination, count);
}

[[maybe_unused]] void quantize_activation(const float *source,
                                          block_q8_1 *destination,
                                          std::int64_t count) {
  quantize_row_q8_1_ref(source, destination, count);
}

[[maybe_unused]] void quantize_activation(const float *source,
                                          block_q8_K *destination,
                                          std::int64_t count) {
  quantize_row_q8_K_ref(source, destination, count);
}

std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  return *text != '\0' && *end == '\0' && value != 0
             ? static_cast<std::size_t>(value)
             : 0;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() & 1U ? values[middle]
                            : 0.5 * (values[middle - 1] + values[middle]);
}

void evict_cache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64) {
    buffer[offset] = static_cast<std::uint8_t>(buffer[offset] + 1U);
    flushSink += buffer[offset];
  }
}

std::vector<std::int8_t> grid64(const std::uint64_t *table,
                                std::size_t entries) {
  std::vector<std::int8_t> result(entries * 8);
  for (std::size_t entry = 0; entry < entries; ++entry) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      result[entry * 8 + lane] = static_cast<std::int8_t>(
          (table[entry] >> (8 * lane)) & 0xffU);
    }
  }
  return result;
}

std::vector<std::int8_t> grid32(const std::uint32_t *table,
                                std::size_t entries) {
  std::vector<std::int8_t> result(entries * 4);
  for (std::size_t entry = 0; entry < entries; ++entry) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      result[entry * 4 + lane] = static_cast<std::int8_t>(
          (table[entry] >> (8 * lane)) & 0xffU);
    }
  }
  return result;
}

std::vector<std::int8_t> sign_table() {
  std::vector<std::int8_t> result(128 * 8);
  for (std::size_t sign = 0; sign < 128; ++sign) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      result[sign * 8 + lane] =
          (ksigns_iq2xs[sign] & kmask_iq2xs[lane]) ? -1 : 1;
    }
  }
  return result;
}

std::vector<float> f16_table() {
  std::vector<float> result(65536);
  for (std::uint32_t bits = 0; bits < 65536; ++bits) {
    const std::uint16_t packed = static_cast<std::uint16_t>(bits);
    _Float16 value;
    std::memcpy(&value, &packed, sizeof(value));
    result[bits] = static_cast<float>(value);
  }
  return result;
}

std::vector<float> e8m0_table() {
  std::vector<float> result(256);
  for (std::uint32_t value = 0; value < 256; ++value) {
    const std::uint32_t bits =
        value < 2 ? 0x00200000U << value : (value - 1U) << 23U;
    std::memcpy(&result[value], &bits, sizeof(bits));
  }
  return result;
}

std::vector<float> ue4m3_table() {
  std::vector<float> result(256);
  for (std::uint32_t value = 0; value < 256; ++value) {
    if (value == 0 || value == 0x7fU) {
      result[value] = 0.0f;
      continue;
    }
    const int exponent = static_cast<int>((value >> 3U) & 0xfU);
    const int mantissa = static_cast<int>(value & 7U);
    const float raw = exponent == 0
                          ? __builtin_ldexpf(static_cast<float>(mantissa), -9)
                          : __builtin_ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                                             exponent - 7);
    result[value] = raw * 0.5f;
  }
  return result;
}

} // namespace

#if WEFT_VEC_DOT_FORMAT == 0
using selected_weight = block_q1_0;
using selected_activation = block_q8_0;
#define selected_wqk 128
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q1_0_q8_0_generic
extern "C" void quantized_vec_dot_q1_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q1_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 1
using selected_weight = block_q4_0;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_0_q8_0_generic
extern "C" void quantized_vec_dot_q4_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q4_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 2
using selected_weight = block_q4_1;
using selected_activation = block_q8_1;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_1_q8_1_generic
extern "C" void quantized_vec_dot_q4_1_q8_1(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q4_1_q8_1(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 3
using selected_weight = block_q5_0;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_0_q8_0_generic
extern "C" void quantized_vec_dot_q5_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q5_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 4
using selected_weight = block_q5_1;
using selected_activation = block_q8_1;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_1_q8_1_generic
extern "C" void quantized_vec_dot_q5_1_q8_1(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q5_1_q8_1(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 5
using selected_weight = block_q8_0;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q8_0_q8_0_generic
extern "C" void quantized_vec_dot_q8_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q8_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 6
using selected_weight = block_q2_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q2_K_q8_K_generic
extern "C" void quantized_vec_dot_q2_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q2_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 7
using selected_weight = block_q3_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q3_K_q8_K_generic
extern "C" void quantized_vec_dot_q3_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q3_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 8
using selected_weight = block_q4_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q4_K_q8_K_generic
extern "C" void quantized_vec_dot_q4_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q4_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 9
using selected_weight = block_q5_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q5_K_q8_K_generic
extern "C" void quantized_vec_dot_q5_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q5_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 10
using selected_weight = block_q6_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q6_K_q8_K_generic
extern "C" void quantized_vec_dot_q6_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q6_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 11
using selected_weight = block_iq1_s;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_s_q8_K_generic
extern "C" void quantized_vec_dot_iq1_s_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq1_s_q8_k(w, x, iq1.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 12
using selected_weight = block_iq1_m;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_m_q8_K_generic
extern "C" void quantized_vec_dot_iq1_m_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq1_m_q8_k(w, x, iq1.data(), fp16.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 13
using selected_weight = block_iq2_s;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_s_q8_K_generic
extern "C" void quantized_vec_dot_iq2_s_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq2_s_q8_k(w, x, iq2s.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 14
using selected_weight = block_iq2_xs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xs_q8_K_generic
extern "C" void quantized_vec_dot_iq2_xs_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq2_xs_q8_k(w, x, iq2xs.data(), signs.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 15
using selected_weight = block_iq2_xxs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xxs_q8_K_generic
extern "C" void quantized_vec_dot_iq2_xxs_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq2_xxs_q8_k(w, x, iq2xxs.data(), signs.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 16
using selected_weight = block_iq3_s;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_s_q8_K_generic
extern "C" void quantized_vec_dot_iq3_s_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq3_s_q8_k(w, x, iq3s.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 17
using selected_weight = block_iq3_xxs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_xxs_q8_K_generic
extern "C" void quantized_vec_dot_iq3_xxs_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq3_xxs_q8_k(w, x, iq3xxs.data(), signs.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 18
using selected_weight = block_iq4_nl;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_iq4_nl_q8_0_generic
extern "C" void quantized_vec_dot_iq4_nl_q8_0(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq4_nl_q8_0(w, x, iq4.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 19
using selected_weight = block_iq4_xs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq4_xs_q8_K_generic
extern "C" void quantized_vec_dot_iq4_xs_q8_k(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq4_xs_q8_k(w, x, iq4.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 20
using selected_weight = block_tq1_0;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq1_0_q8_K_generic
extern "C" void quantized_vec_dot_tq1_0_q8_k(const std::uint8_t *, const std::uint8_t *, const std::uint32_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_tq1_0_q8_k(w, x, powers, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 21
using selected_weight = block_tq2_0;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq2_0_q8_K_generic
extern "C" void quantized_vec_dot_tq2_0_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_tq2_0_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 22
using selected_weight = block_mxfp4;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_mxfp4_q8_0_generic
extern "C" void quantized_vec_dot_mxfp4_q8_0(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_mxfp4_q8_0(w, x, fp4.data(), e8m0.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 23
using selected_weight = block_nvfp4;
using selected_activation = block_q8_0;
#define selected_wqk 64
#define selected_xqk 32
#define selected_reference ggml_vec_dot_nvfp4_q8_0
extern "C" void quantized_vec_dot_nvfp4_q8_0(const std::uint8_t *, const std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_nvfp4_q8_0(w, x, fp4.data(), ue4m3.data(), y, kElements)
#else
#error "unknown WEFT_VEC_DOT_FORMAT"
#endif

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[1]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be positive\n");
    return 2;
  }

  static_assert(WEFT_VEC_DOT_FORMAT >= 0 && WEFT_VEC_DOT_FORMAT < 24);
  std::vector<float> weightSource(selected_wqk);
  std::vector<float> activationSource(selected_wqk);
  std::vector<float> importance(selected_wqk, 1.0F);
  for (std::size_t index = 0; index < weightSource.size(); ++index) {
    weightSource[index] =
        static_cast<float>(static_cast<int>(
                               (index * 19U + WEFT_VEC_DOT_FORMAT * 7U) %
                               127U) -
                           63) /
        13.0F;
    activationSource[index] =
        static_cast<float>(static_cast<int>((index * 37U) % 251U) - 125) /
        17.0F;
  }
  selected_weight weightRecord{};
  std::vector<selected_activation> activationRecord(selected_wqk /
                                                     selected_xqk);
  const std::size_t encodedBytes = ggml_quantize_chunk(
      kWeightTypes[WEFT_VEC_DOT_FORMAT], weightSource.data(), &weightRecord, 0,
      1, selected_wqk, importance.data());
  quantize_activation(activationSource.data(), activationRecord.data(),
                      selected_wqk);
  if (encodedBytes != sizeof(selected_weight)) {
    std::fprintf(stderr, "failed to construct valid encoded weight record\n");
    return 2;
  }
  float recordExpected = 0.0f;
  selected_reference(selected_wqk, &recordExpected, 0, &weightRecord, 0,
                     activationRecord.data(), 0, 1);
  if (!__builtin_isfinite(recordExpected)) {
    std::fprintf(stderr, "valid encoded input produced a non-finite result\n");
    return 2;
  }

  std::vector<selected_weight> weightRow(kElements / selected_wqk,
                                         weightRecord);
  std::vector<selected_activation> activation(kElements / selected_xqk);
  for (std::size_t index = 0; index < activation.size(); ++index)
    activation[index] = activationRecord[index % activationRecord.size()];
  float expected = 0.0f;
  selected_reference(kElements, &expected, 0, weightRow.data(), 0,
                     activation.data(), 0, 1);
  if (!__builtin_isfinite(expected)) {
    std::fprintf(stderr, "non-finite full-row reference\n");
    return 2;
  }
  std::vector<selected_weight> weights(kRows * weightRow.size());
  for (std::size_t row = 0; row < kRows; ++row)
    std::memcpy(weights.data() + row * weightRow.size(), weightRow.data(),
                weightRow.size() * sizeof(selected_weight));

  const std::vector<std::int8_t> iq1 = grid64(iq1s_grid, 2048);
  const std::vector<std::int8_t> iq2xxs = grid64(iq2xxs_grid, 256);
  const std::vector<std::int8_t> iq2xs = grid64(iq2xs_grid, 512);
  const std::vector<std::int8_t> iq2s = grid64(iq2s_grid, 1024);
  const std::vector<std::int8_t> iq3xxs = grid32(iq3xxs_grid, 256);
  const std::vector<std::int8_t> iq3s = grid32(iq3s_grid, 512);
  const std::vector<std::int8_t> signs = sign_table();
  const std::vector<float> fp16 = f16_table();
  const std::vector<float> e8m0 = e8m0_table();
  const std::vector<float> ue4m3 = ue4m3_table();
  std::vector<std::int8_t> iq4(16);
  std::vector<std::int8_t> fp4(16);
  for (std::size_t index = 0; index < 16; ++index) {
    iq4[index] = kvalues_iq4nl[index];
    fp4[index] = kvalues_mxfp4[index];
  }
  const std::uint32_t powers[5] = {1, 3, 9, 27, 81};
  (void)iq1;
  (void)iq2xxs;
  (void)iq2xs;
  (void)iq2s;
  (void)iq3xxs;
  (void)iq3s;
  (void)signs;
  (void)fp16;
  (void)e8m0;
  (void)ue4m3;
  (void)iq4;
  (void)fp4;
  (void)powers;

  std::vector<float> output(kRows, 0.0f);
  auto run = [&] {
    for (std::size_t row = 0; row < kRows; ++row)
      selected_call(
          reinterpret_cast<const std::uint8_t *>(weights.data() +
                                                 row * weightRow.size()),
          reinterpret_cast<const std::uint8_t *>(activation.data()),
          &output[row]);
  };
  run();
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    if (!within_tolerance(output[row], expected, maxAbsolute, maxRelative)) {
      std::fprintf(stderr,
                   "mismatch at row %zu: actual=%a expected=%a\n", row,
                   output[row], expected);
      return 1;
    }
  }
  std::vector<std::uint8_t> flush(kFlushBytes, 1);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush);
    const auto begin = std::chrono::steady_clock::now();
    run();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double medianUs = median(samples);
  const double operations = 2.0 * static_cast<double>(kRows) * kElements;
  std::printf("target=%s\nM=1\nN=%zu\nK=%d\n", kTarget, kRows,
              kElements);
  std::printf("input_policy=valid-quantized-record-replicated\n");
  std::printf("numeric=within-tolerance\nmax_absolute_error=%.9g\n"
              "max_relative_error=%.9g\nrepetitions=%zu\n",
              maxAbsolute, maxRelative, repetitions);
  std::printf("cold_median_us=%.3f\ncold_gop_s=%.6f\n", medianUs,
              operations / medianUs / 1.0e3);
  std::printf("output_sample=%.9g\n", output[0]);
  return 0;
}
