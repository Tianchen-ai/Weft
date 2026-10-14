#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef WEFT_Q8_KIND
#error "WEFT_Q8_KIND is required"
#endif

#if WEFT_Q8_KIND == 0
#define WEFT_Q8_ENTRY q8_0_quantize
#elif WEFT_Q8_KIND == 1
#define WEFT_Q8_ENTRY q8_1_quantize
#elif WEFT_Q8_KIND == 2
#define WEFT_Q8_ENTRY q8_K_quantize
#else
#error "unknown WEFT_Q8_KIND"
#endif

extern "C" void WEFT_Q8_ENTRY(const float *input, std::uint8_t *output,
                              std::size_t columns);

#if defined(WEFT_TARGET_K1)
#define WEFT_TARGET_NAME "K1/X60"
#else
#define WEFT_TARGET_NAME "SG2044"
#endif

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 14336;
#if WEFT_Q8_KIND == 0
constexpr const char *kKernel = "q8_0_quantize";
constexpr std::size_t kBlock = 32;
constexpr std::size_t kRecordBytes = 34;
#elif WEFT_Q8_KIND == 1
constexpr const char *kKernel = "q8_1_quantize";
constexpr std::size_t kBlock = 32;
constexpr std::size_t kRecordBytes = 36;
#elif WEFT_Q8_KIND == 2
constexpr const char *kKernel = "q8_K_quantize";
constexpr std::size_t kBlock = 256;
constexpr std::size_t kRecordBytes = 292;
#else
#error "unknown WEFT_Q8_KIND"
#endif
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

volatile std::uint64_t flush_sink = 0;

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

std::int8_t rounded_i8(float value) {
  const float rounded = std::nearbyint(value);
  return static_cast<std::int8_t>(
      std::max(-128.0F, std::min(127.0F, rounded)));
}

void reference(const std::vector<float> &input,
               std::vector<std::uint8_t> &output) {
  const std::size_t blocks = kColumns / kBlock;
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < blocks; ++block) {
      const float *source = input.data() + row * kColumns + block * kBlock;
      std::uint8_t *target =
          output.data() + (row * blocks + block) * kRecordBytes;
#if WEFT_Q8_KIND == 2
      float maximum = -INFINITY;
      float minimum = INFINITY;
      for (std::size_t element = 0; element < kBlock; ++element) {
        maximum = std::max(maximum, source[element]);
        minimum = std::min(minimum, source[element]);
      }
      const float extreme =
          std::fabs(maximum) > std::fabs(minimum) ? maximum : minimum;
      const float inverse = extreme == 0.0F ? 0.0F : -127.0F / extreme;
      const float scale = inverse == 0.0F ? 0.0F : 1.0F / inverse;
      std::memcpy(target, &scale, sizeof(scale));
      auto *codes = reinterpret_cast<std::int8_t *>(target + 4);
      auto *sums = reinterpret_cast<std::int16_t *>(target + 260);
      for (std::size_t group = 0; group < 16; ++group) {
        std::int32_t sum = 0;
        for (std::size_t element = 0; element < 16; ++element) {
          const std::size_t index = group * 16 + element;
          codes[index] = rounded_i8(source[index] * inverse);
          sum += codes[index];
        }
        std::memcpy(sums + group, &sum, sizeof(std::int16_t));
      }
#else
      float maximum = 0.0F;
      for (std::size_t element = 0; element < kBlock; ++element)
        maximum = std::max(maximum, std::fabs(source[element]));
      const float scale = maximum / 127.0F;
      const float inverse = maximum == 0.0F ? 0.0F : 1.0F / scale;
      const _Float16 storedScale = static_cast<_Float16>(scale);
      std::memcpy(target, &storedScale, sizeof(storedScale));
      auto *codes = reinterpret_cast<std::int8_t *>(target + 2 + 2 * WEFT_Q8_KIND);
#if WEFT_Q8_KIND == 1
      std::int32_t sum = 0;
#endif
      for (std::size_t element = 0; element < kBlock; ++element) {
        codes[element] = rounded_i8(source[element] * inverse);
#if WEFT_Q8_KIND == 1
        sum += codes[element];
#endif
      }
#if WEFT_Q8_KIND == 1
      const _Float16 storedSum = static_cast<_Float16>(static_cast<float>(sum) * scale);
      std::memcpy(target + 2, &storedSum, sizeof(storedSum));
#endif
#endif
    }
  }
}

void run_weft(const std::vector<float> &input,
              std::vector<std::uint8_t> &output) {
  const std::size_t rowBytes = (kColumns / kBlock) * kRecordBytes;
  for (std::size_t row = 0; row < kRows; ++row)
    WEFT_Q8_ENTRY(input.data() + row * kColumns,
                  output.data() + row * rowBytes, kColumns);
}

} // namespace

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
  std::vector<float> input(kRows * kColumns);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) /
                   16.0F;
  const std::size_t outputBytes =
      kRows * (kColumns / kBlock) * kRecordBytes;
  std::vector<std::uint8_t> expected(outputBytes, 0);
  std::vector<std::uint8_t> actual(outputBytes, 0);
  reference(input, expected);
  run_weft(input, actual);
  if (actual != expected) {
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    std::fprintf(stderr,
                 "numeric mismatch byte=%zu expected=%u actual=%u\n",
                 static_cast<std::size_t>(mismatch.first - actual.begin()),
                 static_cast<unsigned>(*mismatch.second),
                 static_cast<unsigned>(*mismatch.first));
    return 1;
  }
  std::vector<std::uint8_t> flush(kFlushBytes, 1);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush);
    const auto begin = std::chrono::steady_clock::now();
    run_weft(input, actual);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double medianMs = median(samples);
  std::printf("kernel=%s\ntarget=%s\nM=%zu\nK=%zu\n", kKernel,
              WEFT_TARGET_NAME, kRows, kColumns);
  std::printf("input_policy=fixed-float-mod31\n");
  std::printf("numeric=bit-exact\nrepetitions=%zu\n", repetitions);
  std::printf("cold_median_ms=%.6f\n", medianMs);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kRows * kColumns) / medianMs / 1.0e3);
  std::printf("output_sample=%u\n", static_cast<unsigned>(actual[0]));
  return 0;
}
