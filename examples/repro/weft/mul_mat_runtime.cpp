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

#ifndef WEFT_MUL_MAT_FORMAT
#error "WEFT_MUL_MAT_FORMAT must select one production MUL_MAT kernel"
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
constexpr std::size_t kN = 4096;
constexpr std::size_t kK = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
std::size_t runtimeM = 0;
volatile std::uint64_t flushSink = 0;

constexpr ggml_type kQuantizedWeightTypes[] = {
    GGML_TYPE_Q1_0,    GGML_TYPE_Q4_0,   GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,    GGML_TYPE_Q5_1,   GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,    GGML_TYPE_Q3_K,   GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,    GGML_TYPE_Q6_K,   GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_S,  GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,  GGML_TYPE_IQ4_XS, GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,   GGML_TYPE_MXFP4,  GGML_TYPE_NVFP4,
};

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

[[maybe_unused]] std::vector<float> grid64(const std::uint64_t *table, std::size_t entries) {
  std::vector<float> result(entries * 8);
  for (std::size_t entry = 0; entry < entries; ++entry)
    for (std::size_t lane = 0; lane < 8; ++lane)
      result[entry * 8 + lane] = static_cast<float>(
          static_cast<std::int8_t>((table[entry] >> (8 * lane)) & 0xffU));
  return result;
}

[[maybe_unused]] std::vector<std::int8_t> grid64_i8(
    const std::uint64_t *table, std::size_t entries) {
  std::vector<std::int8_t> result(entries * 8);
  for (std::size_t entry = 0; entry < entries; ++entry)
    for (std::size_t lane = 0; lane < 8; ++lane)
      result[entry * 8 + lane] = static_cast<std::int8_t>(
          (table[entry] >> (8 * lane)) & 0xffU);
  return result;
}

[[maybe_unused]] std::vector<float> grid32(const std::uint32_t *table, std::size_t entries) {
  std::vector<float> result(entries * 4);
  for (std::size_t entry = 0; entry < entries; ++entry)
    for (std::size_t lane = 0; lane < 4; ++lane)
      result[entry * 4 + lane] = static_cast<float>(
          static_cast<std::int8_t>((table[entry] >> (8 * lane)) & 0xffU));
  return result;
}

[[maybe_unused]] std::vector<std::int8_t> grid32_i8(
    const std::uint32_t *table, std::size_t entries) {
  std::vector<std::int8_t> result(entries * 4);
  for (std::size_t entry = 0; entry < entries; ++entry)
    for (std::size_t lane = 0; lane < 4; ++lane)
      result[entry * 4 + lane] = static_cast<std::int8_t>(
          (table[entry] >> (8 * lane)) & 0xffU);
  return result;
}

[[maybe_unused]] std::vector<float> sign_table() {
  std::vector<float> result(128 * 8);
  for (std::size_t sign = 0; sign < 128; ++sign)
    for (std::size_t lane = 0; lane < 8; ++lane)
      result[sign * 8 + lane] =
          (ksigns_iq2xs[sign] & kmask_iq2xs[lane]) ? -1.0f : 1.0f;
  return result;
}

[[maybe_unused]] std::vector<std::int8_t> sign_table_i8() {
  std::vector<std::int8_t> result(128 * 8);
  for (std::size_t sign = 0; sign < 128; ++sign)
    for (std::size_t lane = 0; lane < 8; ++lane)
      result[sign * 8 + lane] =
          (ksigns_iq2xs[sign] & kmask_iq2xs[lane]) ? -1 : 1;
  return result;
}

[[maybe_unused]] std::vector<float> f16_table() {
  std::vector<float> result(65536);
  for (std::uint32_t bits = 0; bits < 65536; ++bits) {
    const std::uint16_t packed = static_cast<std::uint16_t>(bits);
    _Float16 value;
    std::memcpy(&value, &packed, sizeof(value));
    result[bits] = static_cast<float>(value);
  }
  return result;
}

[[maybe_unused]] std::vector<float> e8m0_table() {
  std::vector<float> result(256);
  for (std::uint32_t value = 0; value < 256; ++value) {
    const std::uint32_t bits =
        value < 2 ? 0x00200000U << value : (value - 1U) << 23U;
    std::memcpy(&result[value], &bits, sizeof(bits));
  }
  return result;
}

[[maybe_unused]] std::vector<float> ue4m3_table() {
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

std::vector<float> activation_values(std::size_t count) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index)
    result[index] = static_cast<float>(static_cast<int>((index * 37U) % 251U) - 125) /
                    17.0f;
  return result;
}

} // namespace

#if WEFT_MUL_MAT_FORMAT == 0
constexpr int kElements = 32;
extern "C" void production_mul_mat_f16(const _Float16 *, const float *, _Float16 *,
                                        float *, std::size_t, std::size_t, std::size_t);
#elif WEFT_MUL_MAT_FORMAT == 1
using selected_weight = block_q1_0;
using selected_activation = block_q8_0;
constexpr int kElements = 128;
#define selected_wqk 128
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q1_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_Q10_DECODE)
extern "C" void production_mul_mat_q1_0_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q1_0_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q1_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q1_0(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 2
using selected_weight = block_q4_0;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_Q40_DECODE)
extern "C" void production_mul_mat_q4_0_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_0_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q4_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_0(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 3
using selected_weight = block_q4_1;
using selected_activation = block_q8_1;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_1_q8_1_generic
#define selected_quantize quantize_row_q8_1_ref
#if defined(WEFT_Q41_DECODE)
extern "C" void production_mul_mat_q4_1_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_1_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q4_1(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_1(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 4
using selected_weight = block_q5_0;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_Q50_DECODE)
extern "C" void production_mul_mat_q5_0_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_0_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q5_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_0(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 5
using selected_weight = block_q5_1;
using selected_activation = block_q8_1;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_1_q8_1_generic
#define selected_quantize quantize_row_q8_1_ref
#if defined(WEFT_Q51_DECODE)
extern "C" void production_mul_mat_q5_1_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_1_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q5_1(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_1(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 6
using selected_weight = block_q8_0;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q8_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_Q80_DECODE)
extern "C" void production_mul_mat_q8_0_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q8_0_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q8_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q8_0(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 7
using selected_weight = block_q2_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q2_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_Q2K_DECODE)
extern "C" void production_mul_mat_q2_k_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q2_k_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q2_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q2_k(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 8
using selected_weight = block_q3_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q3_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_Q3K_DECODE)
#if defined(WEFT_Q3K_PREDECODED_SCALES)
extern "C" void production_mul_mat_q3_k_predecoded_scales_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q3_k_predecoded_scales_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q3_k_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q3_k_decode(w, x, xq, y, kN, kK, runtimeM)
#endif
#else
#if defined(WEFT_Q3K_PREDECODED_SCALES)
extern "C" void production_mul_mat_q3_k_predecoded_scales(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q3_k_predecoded_scales(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q3_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q3_k(w, x, xq, y, kN, kK, runtimeM)
#endif
#endif
#elif WEFT_MUL_MAT_FORMAT == 9
using selected_weight = block_q4_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q4_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_Q4K_DECODE)
extern "C" void production_mul_mat_q4_k_decode(
    const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t,
    std::size_t, std::size_t);
#define selected_call(w, x, xq, y)                                            \
  production_mul_mat_q4_k_decode(w, x, xq, y, kN, kK, runtimeM)
#elif defined(WEFT_Q4_DERIVED)
extern "C" std::size_t production_mul_mat_q4_k_persistent_W_packed_size(
    std::size_t, std::size_t);
extern "C" void production_mul_mat_q4_k_persistent_W_pack(
    const std::uint8_t *, std::uint8_t *, std::size_t, std::size_t);
extern "C" void production_mul_mat_q4_k_persistent(
    const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t,
    std::size_t, std::size_t);
#define selected_packed_size production_mul_mat_q4_k_persistent_W_packed_size
#define selected_pack production_mul_mat_q4_k_persistent_W_pack
#define selected_call(w, x, xq, y)                                            \
  production_mul_mat_q4_k_persistent(w, x, xq, y, runtimeM, kK, kN)
#elif defined(WEFT_Q4_STAGED)
extern "C" void production_mul_mat_q4_k_staged(
    const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t,
    std::size_t, std::size_t);
#define selected_call(w, x, xq, y)                                            \
  production_mul_mat_q4_k_staged(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q4_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_k(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 10
using selected_weight = block_q5_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q5_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_Q5K_DECODE)
extern "C" void production_mul_mat_q5_k_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_k_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q5_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_k(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 11
using selected_weight = block_q6_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q6_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_Q6K_DECODE)
extern "C" void production_mul_mat_q6_k_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q6_k_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_q6_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q6_k(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 12
using selected_weight = block_iq1_s;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_s_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_IQ1S_DECODE)
extern "C" void production_mul_mat_iq1_s_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq1_s_decode(w, x, xq, iq1.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq1_s(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq1_s(w, x, xq, iq1.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 13
using selected_weight = block_iq1_m;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_m_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_IQ1M_DECODE)
extern "C" void production_mul_mat_iq1_m_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq1_m_decode(w, x, xq, iq1.data(), fp16.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq1_m(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq1_m(w, x, xq, iq1.data(), fp16.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 14
using selected_weight = block_iq2_s;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_s_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_IQ2_S_STAGED)
extern "C" void production_mul_mat_iq2_s_staged(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_s_staged(w, x, xq, iq2s.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq2_s(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_s(w, x, xq, iq2s.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 15
using selected_weight = block_iq2_xs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_IQ2_XS_STAGED)
extern "C" void production_mul_mat_iq2_xs_staged(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_xs_staged(w, x, xq, iq2xs.data(), signs.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq2_xs(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_xs(w, x, xq, iq2xs.data(), signs.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 16
using selected_weight = block_iq2_xxs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xxs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_IQ2_XXS_DECODE)
extern "C" void production_mul_mat_iq2_xxs_decode(
    const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *,
    const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y)                                            \
  production_mul_mat_iq2_xxs_decode(w, x, xq, iq2xxs.data(), signs.data(), y, \
                                    kN, kK, runtimeM)
#elif defined(WEFT_IQ2_XXS_STAGED)
extern "C" void production_mul_mat_iq2_xxs_staged(
    const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *,
    const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y)                                            \
  production_mul_mat_iq2_xxs_staged(w, x, xq, iq2xxs_i8.data(),          \
                                         signs_i8.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq2_xxs(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_xxs(w, x, xq, iq2xxs.data(), signs.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 17
using selected_weight = block_iq3_s;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_s_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq3_s(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq3_s(w, x, xq, iq3s.data(), y, kN, kK, runtimeM)
#elif WEFT_MUL_MAT_FORMAT == 18
using selected_weight = block_iq3_xxs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_xxs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq3_xxs(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq3_xxs(w, x, xq, iq3xxs.data(), signs.data(), y, kN, kK, runtimeM)
#elif WEFT_MUL_MAT_FORMAT == 19
using selected_weight = block_iq4_nl;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_iq4_nl_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_IQ4_NL_DECODE)
extern "C" void production_mul_mat_iq4_nl_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq4_nl_decode(w, x, xq, iq4.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq4_nl(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq4_nl(w, x, xq, iq4.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 20
using selected_weight = block_iq4_xs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq4_xs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_IQ4_XS_DECODE)
extern "C" void production_mul_mat_iq4_xs_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq4_xs_decode(w, x, xq, iq4.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_iq4_xs(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq4_xs(w, x, xq, iq4.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 21
using selected_weight = block_tq1_0;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq1_0_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_TQ10_DECODE)
extern "C" void production_mul_mat_tq1_0_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::uint32_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_tq1_0_decode(w, x, xq, powers, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_tq1_0(const std::uint8_t *, const float *, std::uint8_t *, const std::uint32_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_tq1_0(w, x, xq, powers, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 22
using selected_weight = block_tq2_0;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq2_0_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
#if defined(WEFT_TQ20_DECODE)
extern "C" void production_mul_mat_tq2_0_decode(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_tq2_0_decode(w, x, xq, y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_tq2_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_tq2_0(w, x, xq, y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 23
using selected_weight = block_mxfp4;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_mxfp4_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_MXFP4_DECODE)
extern "C" void production_mul_mat_mxfp4_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_mxfp4_decode(w, x, xq, fp4.data(), e8m0.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_mxfp4(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_mxfp4(w, x, xq, fp4.data(), e8m0.data(), y, kN, kK, runtimeM)
#endif
#elif WEFT_MUL_MAT_FORMAT == 24
using selected_weight = block_nvfp4;
using selected_activation = block_q8_0;
constexpr int kElements = 64;
#define selected_wqk 64
#define selected_xqk 32
#define selected_reference ggml_vec_dot_nvfp4_q8_0
#define selected_quantize quantize_row_q8_0_ref
#if defined(WEFT_NVFP4_DECODE)
extern "C" void production_mul_mat_nvfp4_decode(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_nvfp4_decode(w, x, xq, fp4.data(), ue4m3.data(), y, kN, kK, runtimeM)
#else
extern "C" void production_mul_mat_nvfp4(const std::uint8_t *, const float *, std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_nvfp4(w, x, xq, fp4.data(), ue4m3.data(), y, kN, kK, runtimeM)
#endif
#else
#error "unknown WEFT_MUL_MAT_FORMAT"
#endif

#if WEFT_MUL_MAT_FORMAT == 9 &&                                                \
    (defined(WEFT_Q4_DERIVED) || defined(WEFT_Q4_STAGED))
float reference_f32_mul(float lhs, float rhs) {
  volatile float result = lhs * rhs;
  return result;
}

float reference_f32_sub(float lhs, float rhs) {
  volatile float result = lhs - rhs;
  return result;
}

float reference_f32_add(float lhs, float rhs) {
  volatile float result = lhs + rhs;
  return result;
}

std::uint8_t q4k_scale(const block_q4_K &record, std::size_t group) {
  if (group < 4)
    return static_cast<std::uint8_t>(record.scales[group] & 0x3fU);
  return static_cast<std::uint8_t>(
      (record.scales[group + 4] & 0x0fU) |
      ((record.scales[group - 4] >> 6U) << 4U));
}

std::uint8_t q4k_minimum(const block_q4_K &record, std::size_t group) {
  if (group < 4)
    return static_cast<std::uint8_t>(record.scales[group + 4] & 0x3fU);
  return static_cast<std::uint8_t>(
      (record.scales[group + 4] >> 4U) |
      ((record.scales[group] >> 6U) << 4U));
}

std::uint8_t q4k_value(const block_q4_K &record, std::size_t element) {
  const std::size_t group = element / 64;
  const std::size_t within = element % 64;
  const std::uint8_t packed = record.qs[group * 32 + within % 32];
  return static_cast<std::uint8_t>(
      (packed >> (4U * static_cast<unsigned>(within / 32))) & 0x0fU);
}

float q4k_blocked_reference(const block_q4_K *weights,
                            const block_q8_K *activation) {
  float result = 0.0f;
  for (std::size_t block = 0; block < kK / QK_K; ++block) {
    std::int32_t scaled = 0;
    std::int32_t minimum = 0;
    for (std::size_t group = 0; group < 8; ++group) {
      std::int32_t partial = 0;
      for (std::size_t element = 0; element < 32; ++element) {
        const std::size_t logical = group * 32 + element;
        partial += static_cast<std::int32_t>(
                       q4k_value(weights[block], logical)) *
                   static_cast<std::int32_t>(activation[block].qs[logical]);
      }
      scaled += partial *
                static_cast<std::int32_t>(q4k_scale(weights[block], group));
      minimum +=
          static_cast<std::int32_t>(q4k_minimum(weights[block], group)) *
          static_cast<std::int32_t>(activation[block].bsums[2 * group] +
                                    activation[block].bsums[2 * group + 1]);
    }
    const float scaleTerm = reference_f32_mul(
        ggml_fp16_to_fp32(weights[block].d), static_cast<float>(scaled));
    const float minimumTerm = reference_f32_mul(
        ggml_fp16_to_fp32(weights[block].dmin), static_cast<float>(minimum));
    const float blockTerm = reference_f32_mul(
        activation[block].d, reference_f32_sub(scaleTerm, minimumTerm));
    result = reference_f32_add(result, blockTerm);
  }
  return result;
}
#endif

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <decode|prefill> <repetitions>\n", argv[0]);
    return 2;
  }
  const char *phase = argv[1];
  if (std::strcmp(phase, "decode") == 0)
    runtimeM = 1;
  else if (std::strcmp(phase, "prefill") == 0)
    runtimeM = 128;
  else {
    std::fprintf(stderr, "unsupported phase: %s\n", phase);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be positive\n");
    return 2;
  }
  const std::vector<float> activation = activation_values(runtimeM * kK);
  std::vector<float> expected(runtimeM * kN, 0.0f);
  std::vector<float> actual(runtimeM * kN, 0.0f);

#if WEFT_MUL_MAT_FORMAT == 0
  std::vector<_Float16> weightRow(kK);
  for (std::size_t index = 0; index < weightRow.size(); ++index)
    weightRow[index] = static_cast<_Float16>(
        static_cast<float>(static_cast<int>((index * 19U) % 127U) - 63) / 13.0f);
  std::vector<_Float16> weights(kN * kK);
  for (std::size_t column = 0; column < kN; ++column)
    std::memcpy(weights.data() + column * kK, weightRow.data(),
                kK * sizeof(_Float16));
  std::vector<_Float16> expected_workspace(runtimeM * kK);
  std::vector<_Float16> actual_workspace(runtimeM * kK);
  for (std::size_t index = 0; index < expected_workspace.size(); ++index)
    expected_workspace[index] = static_cast<_Float16>(activation[index]);
  for (std::size_t row = 0; row < runtimeM; ++row) {
    float sum = 0.0f;
    for (std::size_t k = 0; k < kK; ++k)
      sum += static_cast<float>(weightRow[k]) *
             static_cast<float>(expected_workspace[row * kK + k]);
    std::fill_n(expected.data() + row * kN, kN, sum);
  }
  auto run = [&] {
    production_mul_mat_f16(weights.data(), activation.data(),
                            actual_workspace.data(), actual.data(), kN, kK,
                            runtimeM);
  };
  run();
  if (actual_workspace != expected_workspace) {
    std::fprintf(stderr, "f16 staging mismatch\n");
    return 1;
  }
#else
  std::vector<selected_activation> expected_workspace(runtimeM * kK /
                                                       selected_xqk);
  std::vector<selected_activation> actual_workspace(runtimeM * kK /
                                                     selected_xqk);
  for (std::size_t row = 0; row < runtimeM; ++row)
    selected_quantize(activation.data() + row * kK,
                      expected_workspace.data() + row * kK / selected_xqk,
                      kK);

  const std::vector<std::int8_t> iq1 = grid64_i8(iq1s_grid, 2048);
  const std::vector<std::int8_t> iq2xxs = grid64_i8(iq2xxs_grid, 256);
  const std::vector<std::int8_t> iq2xxs_i8 = grid64_i8(iq2xxs_grid, 256);
  const std::vector<std::int8_t> iq2xs = grid64_i8(iq2xs_grid, 512);
  const std::vector<std::int8_t> iq2s = grid64_i8(iq2s_grid, 1024);
  const std::vector<std::int8_t> iq3xxs = grid32_i8(iq3xxs_grid, 256);
  const std::vector<std::int8_t> iq3s = grid32_i8(iq3s_grid, 512);
  const std::vector<std::int8_t> signs = sign_table_i8();
  const std::vector<std::int8_t> signs_i8 = sign_table_i8();
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
  (void)iq2xxs_i8;
  (void)iq2xs;
  (void)iq2s;
  (void)iq3xxs;
  (void)iq3s;
  (void)signs;
  (void)signs_i8;
  (void)fp16;
  (void)e8m0;
  (void)ue4m3;
  (void)iq4;
  (void)fp4;
  (void)powers;

  static_assert(WEFT_MUL_MAT_FORMAT >= 1 && WEFT_MUL_MAT_FORMAT <= 24);
  std::vector<float> weightSource(selected_wqk);
  std::vector<float> importance(selected_wqk, 1.0F);
  for (std::size_t index = 0; index < weightSource.size(); ++index)
    weightSource[index] =
        static_cast<float>(static_cast<int>(
                               (index * 19U + WEFT_MUL_MAT_FORMAT * 7U) %
                               127U) -
                           63) /
        13.0F;
  selected_weight weightRecord{};
  std::vector<selected_weight> weightRow(kK / selected_wqk);
  std::vector<float> expectedRows(runtimeM, 0.0f);
  const std::size_t encodedBytes = ggml_quantize_chunk(
      kQuantizedWeightTypes[WEFT_MUL_MAT_FORMAT - 1], weightSource.data(),
      &weightRecord, 0, 1, selected_wqk, importance.data());
  if (encodedBytes != sizeof(selected_weight)) {
    std::fprintf(stderr, "failed to construct valid encoded weight record\n");
    return 2;
  }
  std::fill(weightRow.begin(), weightRow.end(), weightRecord);
  bool finite = true;
  for (std::size_t row = 0; row < runtimeM; ++row) {
#if WEFT_MUL_MAT_FORMAT == 9 &&                                                \
    (defined(WEFT_Q4_DERIVED) || defined(WEFT_Q4_STAGED))
    expectedRows[row] = q4k_blocked_reference(
        weightRow.data(), expected_workspace.data() + row * kK / selected_xqk);
#else
    selected_reference(kK, &expectedRows[row], 0, weightRow.data(), 0,
                       expected_workspace.data() + row * kK / selected_xqk, 0,
                       1);
#endif
    finite = finite && __builtin_isfinite(expectedRows[row]);
  }
  if (!finite) {
    std::fprintf(stderr, "valid encoded input produced a non-finite result\n");
    return 2;
  }
  std::vector<selected_weight> weights(kN * weightRow.size());
  for (std::size_t column = 0; column < kN; ++column)
    std::memcpy(weights.data() + column * weightRow.size(), weightRow.data(),
                weightRow.size() * sizeof(selected_weight));
  const std::uint8_t *kernelWeights =
      reinterpret_cast<const std::uint8_t *>(weights.data());
#if WEFT_MUL_MAT_FORMAT == 9 && defined(WEFT_Q4_DERIVED)
  std::vector<std::uint8_t> packedWeights(
      selected_packed_size(kN, kK));
  selected_pack(kernelWeights, packedWeights.data(), kN, kK);
  kernelWeights = packedWeights.data();
#endif
  for (std::size_t row = 0; row < runtimeM; ++row)
    std::fill_n(expected.data() + row * kN, kN, expectedRows[row]);
  auto run = [&] {
    selected_call(kernelWeights, activation.data(),
                  reinterpret_cast<std::uint8_t *>(actual_workspace.data()),
                  actual.data());
  };
  run();
  // Quantizers may choose sign-equivalent scales when equal-magnitude extrema
  // tie.  The standalone quantize repro owns byte-layout validation; this
  // fused repro validates the resulting numerical projection below.
#endif

  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!within_tolerance(actual[index], expected[index], maxAbsolute,
                          maxRelative)) {
      std::fprintf(stderr, "output mismatch at %zu: actual=%a expected=%a\n",
                   index, actual[index], expected[index]);
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
  const double operations = 2.0 * static_cast<double>(runtimeM) * kN * kK;
  std::printf("target=%s\nphase=%s\nM=%zu\nN=%zu\nK=%zu\n", kTarget,
              phase, runtimeM, kN, kK);
#if WEFT_MUL_MAT_FORMAT == 0
  std::printf("input_policy=dense-fixed-values\n");
#else
  std::printf("input_policy=valid-quantized-record-replicated\n");
#endif
  std::printf("numeric=within-tolerance\nmax_absolute_error=%.9g\n"
              "max_relative_error=%.9g\nrepetitions=%zu\n",
              maxAbsolute, maxRelative, repetitions);
  std::printf("cold_median_us=%.3f\ncold_gop_s=%.6f\n", medianUs,
              operations / medianUs / 1.0e3);
  std::printf("output_sample=%.9g\n", actual[0]);
  return 0;
}
