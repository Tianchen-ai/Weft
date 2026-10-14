#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void gemv_f32(const float *W, const float *X, float *Y,
                          std::size_t M, std::size_t K);

#if defined(WEFT_TARGET_K1)
#define WEFT_TARGET_NAME "K1/X60"
#else
#define WEFT_TARGET_NAME "SG2044"
#endif

namespace {
constexpr std::size_t kM = 14336;
constexpr std::size_t kK = 4096;
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
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[1]);
  if (repetitions == 0)
    return 2;
  std::vector<float> weights(kM * kK, 0.5F);
  std::vector<float> activation(kK, 1.0F / 4096.0F);
  std::vector<float> output(kM, 0.0F);
  gemv_f32(weights.data(), activation.data(), output.data(), kM, kK);
  const float expected = 0.5F;
  double max_absolute = 0.0;
  double max_relative = 0.0;
  for (std::size_t row = 0; row < output.size(); ++row) {
    if (!within_tolerance(output[row], expected, max_absolute, max_relative)) {
      std::fprintf(stderr,
                   "numeric mismatch row=%zu expected=%.9g actual=%.9g\n", row,
                   expected, output[row]);
      return 1;
    }
  }
  std::vector<std::uint8_t> flush(kFlushBytes, 1);
  std::vector<double> samples;
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush);
    const auto begin = std::chrono::steady_clock::now();
    gemv_f32(weights.data(), activation.data(), output.data(), kM, kK);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double median_us = median(samples);
  const double operations = 2.0 * static_cast<double>(kM) * kK;
  std::printf("kernel=gemv_f32\ntarget=%s\nM=%zu\nK=%zu\n", WEFT_TARGET_NAME,
              kM, kK);
  std::printf("numeric=within-tolerance\nmax_absolute_error=%.9g\n"
              "max_relative_error=%.9g\nrepetitions=%zu\n",
              max_absolute, max_relative, repetitions);
  std::printf("cold_median_us=%.3f\ncold_gop_s=%.6f\n", median_us,
              operations / median_us / 1.0e3);
  return 0;
}
