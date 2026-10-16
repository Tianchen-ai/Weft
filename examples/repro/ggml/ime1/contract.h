#ifndef WEFT_EXAMPLES_GGML_IME1_CONTRACT_H
#define WEFT_EXAMPLES_GGML_IME1_CONTRACT_H

#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ime1_repro {

inline std::size_t kN = 4096;
inline std::size_t kK = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
constexpr double kAbsoluteTolerance = 1.0e-4;
constexpr double kRelativeTolerance = 2.0e-3;

using quantize_fn = void (*)(const float *, void *, std::int64_t);

struct ime_case {
  const char *name;
  const char *format;
  const char *phase;
  ggml_type weight_type;
  quantize_fn quantize_weight;
  std::size_t m;
};

inline const ime_case cases[] = {
    {"q4_0_decode", "q4_0", "decode", GGML_TYPE_Q4_0,
     quantize_row_q4_0, 1},
    {"q4_0_prefill", "q4_0", "prefill", GGML_TYPE_Q4_0,
     quantize_row_q4_0, 128},
    {"q4_1_decode", "q4_1", "decode", GGML_TYPE_Q4_1,
     quantize_row_q4_1, 1},
    {"q4_1_prefill", "q4_1", "prefill", GGML_TYPE_Q4_1,
     quantize_row_q4_1, 128},
    {"q4_K_decode", "q4_K", "decode", GGML_TYPE_Q4_K,
     quantize_row_q4_K, 1},
    {"q4_K_prefill", "q4_K", "prefill", GGML_TYPE_Q4_K,
     quantize_row_q4_K, 128},
};

inline volatile std::uint64_t flush_sink = 0;

inline const ime_case *find_case(const char *name) {
  for (const ime_case &candidate : cases) {
    if (std::strcmp(name, candidate.name) == 0) {
      return &candidate;
    }
  }
  return nullptr;
}

struct reference_block {
  float scale;
  int zero;
  std::array<std::int8_t, 32> q;
};

inline std::vector<reference_block>
reference_weights(const ime_case &selected,
                  const std::vector<std::uint8_t> &plain_row) {
  std::vector<reference_block> blocks(kK / 32);
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    auto &out = blocks[b];
    const std::uint8_t *q;
    std::size_t half = 0;
    if (selected.weight_type == GGML_TYPE_Q4_0) {
      const auto &in = reinterpret_cast<const block_q4_0 *>(plain_row.data())[b];
      out.scale = ggml_fp16_to_fp32(in.d);
      out.zero = 8;
      q = in.qs;
    } else if (selected.weight_type == GGML_TYPE_Q4_1) {
      const auto &in = reinterpret_cast<const block_q4_1 *>(plain_row.data())[b];
      out.scale = ggml_fp16_to_fp32(in.d);
      out.zero = static_cast<int>(std::min(15.0F, std::max(
          0.0F, -std::nearbyintf(ggml_fp16_to_fp32(in.m) / out.scale))));
      q = in.qs;
    } else {
      const auto &in = reinterpret_cast<const block_q4_K *>(plain_row.data())[b / 8];
      const std::size_t j = b % 8;
      const unsigned sc = j < 4 ? in.scales[j] & 63U
          : (in.scales[j + 4] & 15U) | ((in.scales[j - 4] >> 6) << 4);
      const unsigned mn = j < 4 ? in.scales[j + 4] & 63U
          : (in.scales[j + 4] >> 4) | ((in.scales[j] >> 6) << 4);
      out.scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
          ggml_fp16_to_fp32(in.d) * static_cast<float>(sc)));
      const float minimum = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
          -(ggml_fp16_to_fp32(in.dmin) * static_cast<float>(mn))));
      out.zero = static_cast<int>(std::min(15.0F, std::max(
          0.0F, -std::nearbyintf(minimum / out.scale))));
      q = in.qs + (j / 2) * 32;
      half = j % 2;
    }
    for (std::size_t k = 0; k < 32; ++k) {
      const unsigned value = selected.weight_type == GGML_TYPE_Q4_K
          ? (q[k] >> (half * 4)) & 15U
          : (q[k % 16] >> ((k / 16) * 4)) & 15U;
      out.q[k] = static_cast<std::int8_t>(static_cast<int>(value) - out.zero);
    }
  }
  return blocks;
}

inline bool check_packed_weights(const ime_case &selected, const void *packed,
                          const std::vector<reference_block> &expected,
                          std::size_t begin_row = 0, std::size_t end_row = kN) {
  const auto *bytes = static_cast<const std::uint8_t *>(packed);
  const bool zero_points = selected.weight_type != GGML_TYPE_Q4_0;
  const std::size_t q_offset = 32 + (zero_points ? 16 : 0);
  const std::size_t record_bytes = q_offset + 256;
  for (std::size_t n = begin_row; n < end_row; ++n) {
    for (std::size_t b = 0; b < kK / 32; ++b) {
      const auto *record = bytes + ((n / 16) * (kK / 32) + b) * record_bytes;
      ggml_fp16_t stored_scale;
      std::memcpy(&stored_scale, record + (n % 16) * 2, sizeof(stored_scale));
      if (stored_scale != ggml_fp32_to_fp16(expected[b].scale) ||
          (zero_points && record[32 + n % 16] != expected[b].zero))
        return false;
      for (std::size_t k = 0; k < 32; ++k) {
        const std::size_t offset = q_offset + (k / 16) * 128 +
                                   (n % 16) * 8 + k % 8;
        const int value = (record[offset] >> (((k / 8) % 2) * 4)) & 15U;
        if (value - expected[b].zero != expected[b].q[k])
          return false;
      }
    }
  }
  return true;
}

inline std::vector<float> reference_output(
    const ime_case &selected, const std::vector<reference_block> &weights,
    const std::vector<float> &activation) {
  std::vector<float> output(selected.m, 0.0F);
  for (std::size_t m = 0; m < selected.m; ++m) {
    for (std::size_t b = 0; b < kK / 32; ++b) {
      const float *x = activation.data() + m * kK + b * 32;
      float maximum = 0.0F;
      for (std::size_t k = 0; k < 32; ++k)
        maximum = std::max(maximum, std::fabs(x[k]));
      const float scale = maximum * (1.0F / 127.0F);
      const float reciprocal = 1.0F / scale;
      std::int32_t partial = 0;
      for (std::size_t k = 0; k < 32; ++k) {
        const int quantized = static_cast<int>(std::nearbyintf(x[k] * reciprocal));
        partial += std::clamp(quantized, -128, 127) * weights[b].q[k];
      }
      output[m] = std::fma(static_cast<float>(partial),
                           scale * weights[b].scale, output[m]);
    }
  }
  return output;
}

inline std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0) {
    return 0;
  }
  return static_cast<std::size_t>(value);
}

inline void evict_cache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t i = 0; i < buffer.size(); i += 64) {
    buffer[i] = static_cast<std::uint8_t>(buffer[i] + 1U);
    flush_sink += buffer[i];
  }
}

inline double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}


} // namespace ime1_repro

#endif
