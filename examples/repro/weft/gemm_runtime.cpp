#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void gemm_f32(const float *A, const float *B, float *C,
                          std::size_t M, std::size_t K, std::size_t N);

#if defined(WEFT_TARGET_K1)
#define WEFT_TARGET_NAME "K1/X60"
#else
#define WEFT_TARGET_NAME "SG2044"
#endif

namespace {
constexpr std::size_t kK = 4096;
constexpr std::size_t kN = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
volatile std::uint64_t flush_sink = 0;
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
    flush_sink += buffer[offset];
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <decode|prefill> <repetitions>\n", argv[0]);
    return 2;
  }
  const char *phase = argv[1];
  const std::size_t m = std::strcmp(phase, "decode") == 0
                            ? 1
                            : std::strcmp(phase, "prefill") == 0 ? 128 : 0;
  if (m == 0) {
    std::fprintf(stderr, "unsupported phase: %s\n", phase);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0)
    return 2;
  std::vector<float> lhs(m * kK, 1.0F / 4096.0F);
  std::vector<float> rhs(kK * kN, 1.0F);
  std::vector<float> output(m * kN, 0.0F);
  gemm_f32(lhs.data(), rhs.data(), output.data(), m, kK, kN);
  const float expected = 1.0F;
  double max_absolute = 0.0;
  double max_relative = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (!within_tolerance(output[index], expected, max_absolute,
                          max_relative)) {
      std::fprintf(stderr,
                   "numeric mismatch index=%zu expected=%.9g actual=%.9g\n",
                   index, expected, output[index]);
      return 1;
    }
  }
  std::vector<std::uint8_t> flush(kFlushBytes, 1);
  std::vector<double> samples;
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    std::fill(output.begin(), output.end(), 0.0F);
    evict_cache(flush);
    const auto begin = std::chrono::steady_clock::now();
    gemm_f32(lhs.data(), rhs.data(), output.data(), m, kK, kN);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double median_us = median(samples);
  const double operations =
      2.0 * static_cast<double>(m) * kK * static_cast<double>(kN);
  std::printf("kernel=gemm_f32\ntarget=%s\nphase=%s\nM=%zu\nN=%zu\nK=%zu\n",
              WEFT_TARGET_NAME, phase, m, kN, kK);
  std::printf("input_policy=dense-fixed-values\n");
  std::printf("numeric=within-tolerance\nmax_absolute_error=%.9g\n"
              "max_relative_error=%.9g\nrepetitions=%zu\n",
              max_absolute, max_relative, repetitions);
  std::printf("cold_median_us=%.3f\ncold_gop_s=%.6f\n", median_us,
              operations / median_us / 1.0e3);
  std::printf("output_sample=%.9g\n", output[0]);
  return 0;
}
