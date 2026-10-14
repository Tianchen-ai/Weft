#include "ggml-quants.h"

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
#include <random>
#include <vector>

#ifndef WEFT_ROW_FORMAT
#error "WEFT_ROW_FORMAT must select one row-dequantization kernel"
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
constexpr std::size_t kRows = 1024;
constexpr std::size_t kElements = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
volatile std::uint64_t flushSink = 0;

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
    std::uint32_t bits =
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

template <typename Block, typename Reference>
Block finite_random_record(Reference reference, std::size_t block_elements) {
  std::mt19937 generator(0x57454654U + WEFT_ROW_FORMAT);
  std::uniform_int_distribution<unsigned> bytes(0, 255);
  Block record{};
  std::vector<float> probe(block_elements);
  for (int attempt = 0; attempt < 4096; ++attempt) {
    auto *raw = reinterpret_cast<std::uint8_t *>(&record);
    for (std::size_t index = 0; index < sizeof(Block); ++index)
      raw[index] = static_cast<std::uint8_t>(bytes(generator));
    reference(&record, probe.data(), block_elements);
    if (std::all_of(probe.begin(), probe.end(), [](float value) {
          return __builtin_isfinite(value);
        })) {
      return record;
    }
  }
  std::fprintf(stderr, "failed to generate finite random encoded data\n");
  std::exit(2);
}

} // namespace

#if WEFT_ROW_FORMAT == 0
using selected_block = block_q1_0;
#define selected_reference dequantize_row_q1_0
#define selected_qk 128
extern "C" void row_dequantize_q1_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 1
using selected_block = block_q4_0;
#define selected_reference dequantize_row_q4_0
#define selected_qk 32
extern "C" void row_dequantize_q4_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 2
using selected_block = block_q4_1;
#define selected_reference dequantize_row_q4_1
#define selected_qk 32
extern "C" void row_dequantize_q4_1(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 3
using selected_block = block_q5_0;
#define selected_reference dequantize_row_q5_0
#define selected_qk 32
extern "C" void row_dequantize_q5_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 4
using selected_block = block_q5_1;
#define selected_reference dequantize_row_q5_1
#define selected_qk 32
extern "C" void row_dequantize_q5_1(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 5
using selected_block = block_q8_0;
#define selected_reference dequantize_row_q8_0
#define selected_qk 32
extern "C" void row_dequantize_q8_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 6
using selected_block = block_q2_K;
#define selected_reference dequantize_row_q2_K
#define selected_qk 256
extern "C" void row_dequantize_q2_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 7
using selected_block = block_q3_K;
#define selected_reference dequantize_row_q3_K
#define selected_qk 256
extern "C" void row_dequantize_q3_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 8
using selected_block = block_q4_K;
#define selected_reference dequantize_row_q4_K
#define selected_qk 256
extern "C" void row_dequantize_q4_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 9
using selected_block = block_q5_K;
#define selected_reference dequantize_row_q5_K
#define selected_qk 256
extern "C" void row_dequantize_q5_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 10
using selected_block = block_q6_K;
#define selected_reference dequantize_row_q6_K
#define selected_qk 256
extern "C" void row_dequantize_q6_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 11
using selected_block = block_iq1_s;
#define selected_reference dequantize_row_iq1_s
#define selected_qk 256
extern "C" void row_dequantize_iq1_s(const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 12
using selected_block = block_iq1_m;
#define selected_reference dequantize_row_iq1_m
#define selected_qk 256
extern "C" void row_dequantize_iq1_m(const std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 13
using selected_block = block_iq2_s;
#define selected_reference dequantize_row_iq2_s
#define selected_qk 256
extern "C" void row_dequantize_iq2_s(const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 14
using selected_block = block_iq2_xs;
#define selected_reference dequantize_row_iq2_xs
#define selected_qk 256
extern "C" void row_dequantize_iq2_xs(const std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 15
using selected_block = block_iq2_xxs;
#define selected_reference dequantize_row_iq2_xxs
#define selected_qk 256
extern "C" void row_dequantize_iq2_xxs(const std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 16
using selected_block = block_iq3_s;
#define selected_reference dequantize_row_iq3_s
#define selected_qk 256
extern "C" void row_dequantize_iq3_s(const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 17
using selected_block = block_iq3_xxs;
#define selected_reference dequantize_row_iq3_xxs
#define selected_qk 256
extern "C" void row_dequantize_iq3_xxs(const std::uint8_t *, const std::int8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 18
using selected_block = block_iq4_nl;
#define selected_reference dequantize_row_iq4_nl
#define selected_qk 32
extern "C" void row_dequantize_iq4_nl(const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 19
using selected_block = block_iq4_xs;
#define selected_reference dequantize_row_iq4_xs
#define selected_qk 256
extern "C" void row_dequantize_iq4_xs(const std::uint8_t *, const std::int8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 20
using selected_block = block_tq1_0;
#define selected_reference dequantize_row_tq1_0
#define selected_qk 256
extern "C" void row_dequantize_tq1_0(const std::uint8_t *, const std::uint32_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 21
using selected_block = block_tq2_0;
#define selected_reference dequantize_row_tq2_0
#define selected_qk 256
extern "C" void row_dequantize_tq2_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 22
using selected_block = block_mxfp4;
#define selected_reference dequantize_row_mxfp4
#define selected_qk 32
extern "C" void row_dequantize_mxfp4(const std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 23
using selected_block = block_nvfp4;
#define selected_reference dequantize_row_nvfp4
#define selected_qk 64
extern "C" void row_dequantize_nvfp4(const std::uint8_t *, const std::int8_t *, const float *, float *, std::size_t);
#else
#error "unknown WEFT_ROW_FORMAT"
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
  const selected_block record =
      finite_random_record<selected_block>(selected_reference, selected_qk);
  std::vector<selected_block> inputRow(kElements / selected_qk, record);
  std::vector<float> expectedRow(kElements);
  selected_reference(inputRow.data(), expectedRow.data(), kElements);
  std::vector<selected_block> input(kRows * inputRow.size());
  for (std::size_t row = 0; row < kRows; ++row)
    std::memcpy(input.data() + row * inputRow.size(), inputRow.data(),
                inputRow.size() * sizeof(selected_block));
  std::vector<float> actual(kRows * kElements);

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
  (void)powers;

#if WEFT_ROW_FORMAT == 0
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q1_0(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 1
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q4_0(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 2
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q4_1(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 3
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q5_0(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 4
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q5_1(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 5
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q8_0(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 6
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q2_k(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 7
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q3_k(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 8
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q4_k(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 9
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q5_k(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 10
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_q6_k(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 11
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq1_s(source, iq1.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 12
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq1_m(source, iq1.data(), fp16.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 13
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq2_s(source, iq2s.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 14
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq2_xs(source, iq2xs.data(), signs.data(), target,
                          kElements);
  };
#elif WEFT_ROW_FORMAT == 15
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq2_xxs(source, iq2xxs.data(), signs.data(), target,
                           kElements);
  };
#elif WEFT_ROW_FORMAT == 16
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq3_s(source, iq3s.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 17
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq3_xxs(source, iq3xxs.data(), signs.data(), target,
                           kElements);
  };
#elif WEFT_ROW_FORMAT == 18
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq4_nl(source, iq4.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 19
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_iq4_xs(source, iq4.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 20
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_tq1_0(source, powers, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 21
  auto runRow = [](const std::uint8_t *source, float *target) {
    row_dequantize_tq2_0(source, target, kElements);
  };
#elif WEFT_ROW_FORMAT == 22
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_mxfp4(source, fp4.data(), e8m0.data(), target, kElements);
  };
#elif WEFT_ROW_FORMAT == 23
  auto runRow = [&](const std::uint8_t *source, float *target) {
    row_dequantize_nvfp4(source, fp4.data(), ue4m3.data(), target, kElements);
  };
#endif

  auto run = [&] {
    for (std::size_t row = 0; row < kRows; ++row)
      runRow(reinterpret_cast<const std::uint8_t *>(
                 input.data() + row * inputRow.size()),
             actual.data() + row * kElements);
  };
  run();
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t index = 0; index < kElements; ++index) {
      const float value = actual[row * kElements + index];
      if (!within_tolerance(value, expectedRow[index], maxAbsolute,
                            maxRelative)) {
        std::fprintf(stderr,
                     "mismatch at row=%zu element=%zu: actual=%a expected=%a\n",
                     row, index, value, expectedRow[index]);
        return 1;
      }
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
  const double elements = static_cast<double>(kRows) * kElements;
  std::printf("target=%s\nN=%zu\nK=%zu\n", kTarget, kRows, kElements);
  std::printf("input_policy=finite-random-record-replicated\n");
  std::printf("input_seed=%u\n", 0x57454654U + WEFT_ROW_FORMAT);
  std::printf("numeric=within-tolerance\nmax_absolute_error=%.9g\n"
              "max_relative_error=%.9g\nrepetitions=%zu\n",
              maxAbsolute, maxRelative, repetitions);
  std::printf("cold_median_us=%.3f\ncold_melements_s=%.6f\n", medianUs,
              elements / medianUs);
  return 0;
}
