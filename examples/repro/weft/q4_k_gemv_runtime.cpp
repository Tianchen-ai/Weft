#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#if defined(WEFT_Q4_IME)
std::size_t q4_k_q8_k_gemv_contract_W_packed_size(std::size_t M, std::size_t K);
void q4_k_q8_k_gemv_contract_W_pack(const std::uint8_t *source,
                                std::uint8_t *target, std::size_t M,
                                std::size_t K);
void q4_k_q8_k_gemv_contract(const std::uint8_t *W, const std::uint8_t *X, float *Y,
                        std::size_t M, std::size_t K);
#elif defined(WEFT_Q4_GROUPS4)
std::size_t q4_k_q8_k_gemv_groups4_W_packed_size(std::size_t M,
                                                 std::size_t K);
void q4_k_q8_k_gemv_groups4_W_pack(const std::uint8_t *source,
                                    std::uint8_t *target, std::size_t M,
                                    std::size_t K);
void q4_k_q8_k_gemv_groups4(const std::uint8_t *W, const std::uint8_t *X,
                            float *Y, std::size_t M, std::size_t K);
#else
std::size_t q4_k_q8_k_gemv_W_packed_size(std::size_t M, std::size_t K);
void q4_k_q8_k_gemv_W_pack(const std::uint8_t *source, std::uint8_t *target,
                            std::size_t M, std::size_t K);
void q4_k_q8_k_gemv(const std::uint8_t *W, const std::uint8_t *X, float *Y,
                    std::size_t M, std::size_t K);
#endif
}

#if defined(WEFT_Q4_IME)
#define WEFT_Q4_PACKED_SIZE q4_k_q8_k_gemv_contract_W_packed_size
#define WEFT_Q4_PACK q4_k_q8_k_gemv_contract_W_pack
#define WEFT_Q4_KERNEL q4_k_q8_k_gemv_contract
#define WEFT_Q4_KERNEL_NAME "q4_k_q8_k_gemv_contract"
#elif defined(WEFT_Q4_GROUPS4)
#define WEFT_Q4_PACKED_SIZE q4_k_q8_k_gemv_groups4_W_packed_size
#define WEFT_Q4_PACK q4_k_q8_k_gemv_groups4_W_pack
#define WEFT_Q4_KERNEL q4_k_q8_k_gemv_groups4
#define WEFT_Q4_KERNEL_NAME "q4_k_q8_k_gemv_groups4"
#else
#define WEFT_Q4_PACKED_SIZE q4_k_q8_k_gemv_W_packed_size
#define WEFT_Q4_PACK q4_k_q8_k_gemv_W_pack
#define WEFT_Q4_KERNEL q4_k_q8_k_gemv
#define WEFT_Q4_KERNEL_NAME "q4_k_q8_k_gemv"
#endif

#if defined(WEFT_TARGET_K1)
#define WEFT_TARGET_NAME "K1/X60"
#else
#define WEFT_TARGET_NAME "SG2044"
#endif

namespace {
constexpr double kAbsoluteTolerance = 1.25e-1;
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

constexpr std::size_t kM = 14336;
constexpr std::size_t kK = 4096;
constexpr std::size_t kQ4RecordBytes = 144;
constexpr std::size_t kQ8RecordBytes = 292;
constexpr std::size_t kBlock = 256;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

volatile std::uint64_t flush_sink = 0;

std::uint32_t random_word(std::uint32_t &state) {
  state ^= state << 13U;
  state ^= state >> 17U;
  state ^= state << 5U;
  return state;
}

void random_bytes(std::uint8_t *target, std::size_t bytes,
                  std::uint32_t &state) {
  for (std::size_t index = 0; index < bytes; ++index)
    target[index] = static_cast<std::uint8_t>(random_word(state));
}

std::uint8_t q4k_scale(const std::uint8_t *record, std::size_t group) {
  const std::uint8_t *scales = record + 4;
  if (group < 4)
    return static_cast<std::uint8_t>(scales[group] & 0x3FU);
  return static_cast<std::uint8_t>((scales[group + 4] & 0x0FU) |
                                   ((scales[group - 4] >> 6U) << 4U));
}

std::uint8_t q4k_minimum(const std::uint8_t *record, std::size_t group) {
  const std::uint8_t *scales = record + 4;
  if (group < 4)
    return static_cast<std::uint8_t>(scales[group + 4] & 0x3FU);
  return static_cast<std::uint8_t>((scales[group + 4] >> 4U) |
                                   ((scales[group] >> 6U) << 4U));
}

std::uint8_t q4k_value(const std::uint8_t *record, std::size_t element) {
  const std::size_t group64 = element / 64;
  const std::size_t within = element % 64;
  const std::uint8_t packed = record[16 + group64 * 32 + within % 32];
  return static_cast<std::uint8_t>((packed >> (4U * (within / 32))) & 0x0FU);
}

void store_f16(std::uint8_t *target, _Float16 value) {
  std::memcpy(target, &value, sizeof(value));
}

void store_f32(std::uint8_t *target, float value) {
  std::memcpy(target, &value, sizeof(value));
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

void initialize_q4(std::vector<std::uint8_t> &weights) {
  const std::size_t blocks = kK / kBlock;
  std::uint32_t random_state = 0x4b1d2a73U;
  for (std::size_t row = 0; row < kM; ++row) {
    for (std::size_t block = 0; block < blocks; ++block) {
      std::uint8_t *record =
          weights.data() + (row * blocks + block) * kQ4RecordBytes;
      random_bytes(record, kQ4RecordBytes, random_state);
      const float d = 0.125F +
                      static_cast<float>(random_word(random_state) & 31U) /
                          128.0F;
      const float dmin = 0.0625F +
                         static_cast<float>(random_word(random_state) & 15U) /
                             256.0F;
      store_f16(record, static_cast<_Float16>(d));
      store_f16(record + 2, static_cast<_Float16>(dmin));
    }
  }
}

void initialize_q8(std::vector<std::uint8_t> &activation) {
  const std::size_t blocks = kK / kBlock;
  std::uint32_t random_state = 0x8ac6f251U;
  for (std::size_t block = 0; block < blocks; ++block) {
    std::uint8_t *record = activation.data() + block * kQ8RecordBytes;
    random_bytes(record, kQ8RecordBytes, random_state);
    const float d = 0.125F +
                    static_cast<float>(random_word(random_state) & 31U) /
                        128.0F;
    store_f32(record, d);
  }
}

float reference_row(const std::uint8_t *weights, const std::uint8_t *activation,
                    std::size_t row) {
  const std::size_t blocks = kK / kBlock;
  float result = 0.0F;
  for (std::size_t block = 0; block < blocks; ++block) {
    const std::uint8_t *w =
        weights + (row * blocks + block) * kQ4RecordBytes;
    const std::uint8_t *x = activation + block * kQ8RecordBytes;
    _Float16 d16;
    _Float16 dmin16;
    float ds;
    std::memcpy(&d16, w, sizeof(d16));
    std::memcpy(&dmin16, w + 2, sizeof(dmin16));
    std::memcpy(&ds, x, sizeof(ds));
    std::int32_t scaled = 0;
    for (std::size_t group = 0; group < 8; ++group) {
      std::int32_t partial = 0;
      for (std::size_t element = 0; element < 32; ++element) {
        const std::int32_t q =
            static_cast<std::int32_t>(q4k_value(w, group * 32 + element));
        const std::int32_t v = static_cast<std::int8_t>(x[4 + group * 32 + element]);
        partial += q * v;
      }
      scaled += partial * static_cast<std::int32_t>(q4k_scale(w, group));
    }
    std::int32_t minimum = 0;
    for (std::size_t group = 0; group < 8; ++group) {
      std::int16_t first;
      std::int16_t second;
      std::memcpy(&first, x + 260 + 4 * group, sizeof(first));
      std::memcpy(&second, x + 262 + 4 * group, sizeof(second));
      minimum += static_cast<std::int32_t>(q4k_minimum(w, group)) *
                 static_cast<std::int32_t>(first + second);
    }
    const float scale_term = static_cast<float>(d16) * static_cast<float>(scaled);
    const float min_term = static_cast<float>(dmin16) * static_cast<float>(minimum);
    result += ds * (scale_term - min_term);
  }
  return result;
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
  const std::size_t blocks = kK / kBlock;
  std::vector<std::uint8_t> source_weights(kM * blocks * kQ4RecordBytes, 0);
  std::vector<std::uint8_t> activation(blocks * kQ8RecordBytes, 0);
  initialize_q4(source_weights);
  initialize_q8(activation);
  std::vector<std::uint8_t> packed(
      WEFT_Q4_PACKED_SIZE(kM, kK));
  WEFT_Q4_PACK(source_weights.data(), packed.data(), kM, kK);
  std::vector<float> output(kM, 0.0F);
  WEFT_Q4_KERNEL(packed.data(), activation.data(), output.data(), kM, kK);
  double max_absolute = 0.0;
  double max_relative = 0.0;
  for (std::size_t row = 0; row < kM; ++row) {
    const float expected = reference_row(source_weights.data(), activation.data(), row);
    if (!within_tolerance(output[row], expected, max_absolute, max_relative)) {
      std::fprintf(stderr,
                   "numeric mismatch row=%zu expected=%.9g actual=%.9g\n", row,
                   expected, output[row]);
      return 1;
    }
  }
  std::vector<std::uint8_t> flush(kFlushBytes, 1);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush);
    const auto begin = std::chrono::steady_clock::now();
    WEFT_Q4_KERNEL(packed.data(), activation.data(), output.data(), kM, kK);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double median_us = median(samples);
  const double operations = 2.0 * static_cast<double>(kM) * kK;
  std::printf("kernel=%s\n", WEFT_Q4_KERNEL_NAME);
  std::printf("target=%s\nM=1\nN=%zu\nK=%zu\n", WEFT_TARGET_NAME, kM,
              kK);
  std::printf("numeric=within-tolerance\nmax_absolute_error=%.9g\n"
              "max_relative_error=%.9g\nrepetitions=%zu\n",
              max_absolute, max_relative, repetitions);
  std::printf("cold_median_us=%.3f\n", median_us);
  std::printf("cold_gop_s=%.6f\n", operations / median_us / 1.0e3);
  std::printf("output_sample=%.9g\n", output.front());
  return 0;
}
