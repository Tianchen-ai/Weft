#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void ime_i8_contract(const std::int8_t *A, const std::int8_t *B,
                                std::int32_t *C, std::size_t M,
                                std::size_t K, std::size_t N);

namespace {
constexpr std::size_t kM = 4;
constexpr std::size_t kN = 4;
constexpr std::size_t kK = 8;

std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  return *text != '\0' && *end == '\0' && value != 0
             ? static_cast<std::size_t>(value)
             : 0;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  const std::size_t repetitions = parse_repetitions(argv[1]);
  if (repetitions == 0)
    return 2;

  std::vector<std::int8_t> lhs(kM * kK);
  std::vector<std::int8_t> rhs(kK * kN);
  for (std::size_t row = 0; row < kM; ++row)
    for (std::size_t k = 0; k < kK; ++k)
      lhs[row * kK + k] = static_cast<std::int8_t>((row * 5 + k * 3) % 17 - 8);
  for (std::size_t k = 0; k < kK; ++k)
    for (std::size_t column = 0; column < kN; ++column)
      rhs[k * kN + column] =
          static_cast<std::int8_t>((k * 7 + column * 2) % 19 - 9);

  std::vector<std::int32_t> expected(kM * kN, 0);
  for (std::size_t row = 0; row < kM; ++row)
    for (std::size_t column = 0; column < kN; ++column)
      for (std::size_t k = 0; k < kK; ++k)
        expected[row * kN + column] +=
            static_cast<std::int32_t>(lhs[row * kK + k]) *
            static_cast<std::int32_t>(rhs[k * kN + column]);

  std::vector<std::int32_t> output(kM * kN, 0);
  ime_i8_contract(lhs.data(), rhs.data(), output.data(), kM, kK, kN);
  if (output != expected) {
    auto mismatch = std::mismatch(output.begin(), output.end(), expected.begin());
    const std::size_t index =
        static_cast<std::size_t>(mismatch.first - output.begin());
    std::fprintf(stderr,
                 "numeric mismatch index=%zu expected=%d actual=%d\n", index,
                 expected[index], output[index]);
    return 1;
  }

  std::vector<double> samples;
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    std::fill(output.begin(), output.end(), 0);
    const auto begin = std::chrono::steady_clock::now();
    ime_i8_contract(lhs.data(), rhs.data(), output.data(), kM, kK, kN);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  const double median_us = samples[samples.size() / 2];
  const double operations = 2.0 * kM * kN * kK;
  std::printf("kernel=ime_i8_contract\ntarget=K1/X60\nM=4\nN=4\nK=8\n");
  std::printf("numeric=exact\nrepetitions=%zu\nmedian_us=%.3f\ngop_s=%.6f\n",
              repetitions, median_us, operations / median_us / 1.0e3);
  return 0;
}
