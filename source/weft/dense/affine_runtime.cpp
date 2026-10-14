#include "kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

int main() {
  constexpr std::size_t count = 129;
  alignas(64) std::array<float, count> input{};
  alignas(64) std::array<float, count> output{};
  for (std::size_t index = 0; index < count; ++index)
    input[index] = static_cast<float>(index) / 16.0f - 4.0f;
  affine(input.data(), output.data(), 1.25f, -0.5f, count);
  float maximum = 0.0f;
  for (std::size_t index = 0; index < count; ++index) {
    float expected = input[index] * 1.25f - 0.5f;
    float error = std::abs(output[index] - expected);
    if (!std::isfinite(output[index]) || error > 1e-4f + 2e-3f * std::abs(expected)) {
      std::fprintf(stderr, "affine mismatch at %zu: actual=%g expected=%g\n",
                   index, output[index], expected);
      return 1;
    }
    maximum = std::max(maximum, error);
  }
  std::printf("numeric=within-tolerance\nmax_absolute_error=%g\n", maximum);
  return 0;
}
