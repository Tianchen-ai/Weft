#include "../../ggml/ime1/contract.h"

#include <cfenv>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <memory>
#include <sched.h>
#include <type_traits>

using namespace ime1_repro;

using pack_fn = void (*)(const std::uint8_t *, std::uint8_t *, std::size_t, std::size_t);
using kernel_fn = void (*)(const std::uint8_t *, const float *, std::uint8_t *, float *,
                           std::size_t, std::size_t, std::size_t);

namespace {

bool check_activation(const ime_case &selected, const std::vector<float> &input,
                      const std::vector<std::uint8_t> &packed) {
  const std::size_t rows = selected.m == 1 ? 1 : 4;
  for (std::size_t m = 0; m < selected.m; ++m)
    for (std::size_t b = 0; b < kK / 32; ++b) {
      const float *x = input.data() + m * kK + b * 32;
      float maximum = 0;
      for (std::size_t k = 0; k < 32; ++k)
        maximum = std::max(maximum, std::fabs(x[k]));
      const float expected_scale = maximum * (1.0F / 127.0F);
      const auto *record = packed.data() + ((m / rows) * (kK / 32) + b) * (36 * rows);
      float scale;
      std::memcpy(&scale, record + (m % rows) * 4, sizeof(scale));
      if (std::memcmp(&scale, &expected_scale, sizeof(scale)) != 0)
        return false;
      const float inverse = 1.0F / expected_scale;
      for (std::size_t k = 0; k < 32; ++k) {
        const int expected = std::clamp(static_cast<int>(std::nearbyintf(x[k] * inverse)), -128, 127);
        const std::size_t offset = 4 * rows + (k / 8) * (8 * rows) + (m % rows) * 8 + k % 8;
        if (static_cast<std::int8_t>(record[offset]) != expected)
          return false;
      }
    }
  return true;
}

bool check_value(float actual, float expected, double &max_absolute, double &max_relative) {
  const double absolute = std::fabs(static_cast<double>(actual) - expected);
  const double relative = absolute / std::max(std::fabs(static_cast<double>(expected)), 1.0e-30);
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      absolute > kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected))
    return false;
  max_absolute = std::max(max_absolute, absolute);
  max_relative = std::max(max_relative, relative);
  return true;
}

} // namespace

template <class Function, class... Arguments>
bool invoke(Function function, Arguments... arguments) {
  if constexpr (std::is_void_v<std::invoke_result_t<Function, Arguments...>>) {
    function(arguments...);
    return true;
  } else {
    return function(arguments...) == 0;
  }
}

template <class Pack, class Kernel>
int run_case(const ime_case *selected, std::size_t repetitions, Pack pack, Kernel kernel) {
  if (!selected || !pack || std::fegetround() != FE_TONEAREST) {
    std::fprintf(stderr, "a declared IME1 case, pack callable and RNE are required\n");
    return 2;
  }
  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) != 0 || CPU_COUNT(&original) != 1) {
    std::fprintf(stderr, "select one confirmed IME1 CPU before running this repro\n");
    return 2;
  }
  ggml_init_params init = {1024 * 1024, nullptr, true};
  std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(init), ggml_free);
  if (!context)
    return 1;
  std::vector<float> source_row(kK);
  for (std::size_t k = 0; k < kK; ++k)
    source_row[k] = static_cast<float>(static_cast<int>(k % 31) - 15) / 16.0F;
  const std::size_t row_bytes = ggml_row_size(selected->weight_type, kK);
  std::vector<std::uint8_t> plain_row(row_bytes), plain(row_bytes * kN);
  selected->quantize_weight(source_row.data(), plain_row.data(), kK);
  for (std::size_t n = 0; n < kN; ++n)
    std::memcpy(plain.data() + n * row_bytes, plain_row.data(), row_bytes);
  const auto expected_weights = reference_weights(*selected, plain_row);
  const std::size_t record_bytes = selected->weight_type == GGML_TYPE_Q4_0 ? 18 : 19;
  std::vector<std::uint8_t> packed(record_bytes * kN * (kK / 32), 0xA5);
  if (!invoke(pack, plain.data(), packed.data(), kN, kK))
    return 1;
  if (!check_packed_weights(*selected, packed.data(), expected_weights)) {
    std::fprintf(stderr, "Weft persistent weight storage is not exact\n");
    return 1;
  }
  std::vector<float> activation(selected->m * kK), output(selected->m * kN);
  for (std::size_t i = 0; i < activation.size(); ++i)
    activation[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 9.0F;
  std::vector<std::uint8_t> quantized(36 * selected->m * (kK / 32), 0xA5);
  std::vector<double> samples;
  double max_absolute = 0.0, max_relative = 0.0;
  if (kernel) {
    const auto expected = reference_output(*selected, expected_weights, activation);
    if (!invoke(kernel, packed.data(), activation.data(), quantized.data(), output.data(), kN, kK, selected->m))
      return 1;
    if (!check_activation(*selected, activation, quantized)) {
      std::fprintf(stderr, "Weft activation scale/byte storage is not exact\n");
      return 1;
    }
    for (std::size_t i = 0; i < output.size(); ++i)
      if (!check_value(output[i], expected[i / kN], max_absolute, max_relative)) {
        std::fprintf(stderr, "Weft output mismatch at %zu: actual=%a expected=%a\n", i, output[i], expected[i / kN]);
        return 1;
      }
    std::vector<std::uint8_t> flush(repetitions ? kFlushBytes : 0, 1);
    for (std::size_t i = 0; i < repetitions; ++i) {
      evict_cache(flush);
      const auto begin = std::chrono::steady_clock::now();
      if (!invoke(kernel, packed.data(), activation.data(), quantized.data(), output.data(), kN, kK, selected->m))
        return 1;
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    for (float value : output)
      if (!std::isfinite(value)) {
        std::fprintf(stderr, "Weft timed output is not finite\n");
        return 1;
      }
  }

  // A second real input distinguishes columns; it is run after the timed case.
  for (std::size_t n = 0; n < kN; ++n) {
    for (std::size_t k = 0; k < kK; ++k)
      source_row[k] = static_cast<float>(static_cast<int>((n * 7 + k * 13) % 61) - 30) /
                      static_cast<float>(8 + n % 3);
    selected->quantize_weight(source_row.data(), plain.data() + n * row_bytes, kK);
  }
  if (!invoke(pack, plain.data(), packed.data(), kN, kK))
    return 1;
  ime_case independent = *selected;
  independent.m = std::min<std::size_t>(selected->m, 4);
  if (kernel && !invoke(kernel, packed.data(), activation.data(), quantized.data(), output.data(), kN, kK, independent.m))
    return 1;
  for (std::size_t n = 0; n < kN; ++n) {
    std::memcpy(plain_row.data(), plain.data() + n * row_bytes, row_bytes);
    const auto weights = reference_weights(*selected, plain_row);
    if (!check_packed_weights(*selected, packed.data(), weights, n, n + 1)) {
      std::fprintf(stderr, "Weft distinct-column persistent weight mismatch at column %zu\n", n);
      return 1;
    }
    if (kernel) {
      const auto expected = reference_output(independent, weights, activation);
      for (std::size_t m = 0; m < independent.m; ++m)
        if (!check_value(output[m * kN + n], expected[m], max_absolute, max_relative)) {
          std::fprintf(stderr, "Weft distinct-column output mismatch at %zu,%zu: actual=%a expected=%a\n",
                       m, n, output[m * kN + n], expected[m]);
          return 1;
        }
    }
  }
  cpu_set_t actual;
  if (sched_getaffinity(0, sizeof(actual), &actual) != 0 || !CPU_EQUAL(&original, &actual)) {
    std::fprintf(stderr, "repro changed the explicitly selected CPU\n");
    return 1;
  }
  std::printf("family=ime1\nkernel=ime1_%s\nphase=%s\n", selected->format, selected->phase);
  std::printf("execution=%s\nweight_storage=exact\nsecond_input=distinct-columns-exact\n",
              repetitions ? "artifact-call" : "public-native-jit");
  if (kernel) {
    std::printf("M=%zu\nN=%zu\nK=%zu\nnumeric=within-tolerance\nactivation_storage=exact\n",
                selected->m, kN, kK);
    std::printf("absolute_tolerance=%.9g\nrelative_tolerance=%.9g\n", kAbsoluteTolerance, kRelativeTolerance);
    std::printf("max_absolute_error=%.9g\nmax_relative_error=%.9g\n", max_absolute, max_relative);
    if (repetitions) {
      const double cold_us = median(samples);
      const double operations = 2.0 * selected->m * kN * kK;
      std::printf("repetitions=%zu\ncold_median_ms=%.6f\ncold_gops=%.6f\n", repetitions, cold_us / 1000.0,
                  operations / (cold_us * 1000.0));
    }
  } else {
    std::printf("artifact_boundary=persistent-pack\n");
  }
  return 0;
}

using pack_callback = int (*)(const std::uint8_t *, std::uint8_t *, std::size_t, std::size_t);
using kernel_callback = int (*)(const std::uint8_t *, const float *, std::uint8_t *, float *,
                               std::size_t, std::size_t, std::size_t);

extern "C" int ime1_run_native_jit(const char *name, pack_callback pack, kernel_callback kernel) {
  return run_case(find_case(name), 0, pack, kernel);
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 7) {
    std::fprintf(stderr, "usage: %s <case> <repetitions> <pack.so> <pack-symbol> [<kernel.so> <kernel-symbol>]\n", argv[0]);
    return 2;
  }
  const auto *selected = find_case(argv[1]);
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (!selected || !repetitions)
    return 2;
  void *pack_library = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
  if (!pack_library) {
    std::fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
  const auto pack = reinterpret_cast<pack_fn>(dlsym(pack_library, argv[4]));
  if (!pack) {
    std::fprintf(stderr, "%s\n", dlerror());
    dlclose(pack_library);
    return 1;
  }
  void *kernel_library = nullptr;
  kernel_fn kernel = nullptr;
  if (argc == 7) {
    kernel_library = dlopen(argv[5], RTLD_NOW | RTLD_LOCAL);
    if (!kernel_library) {
      std::fprintf(stderr, "%s\n", dlerror());
      dlclose(pack_library);
      return 1;
    }
    kernel = reinterpret_cast<kernel_fn>(dlsym(kernel_library, argv[6]));
    if (!kernel) {
      std::fprintf(stderr, "%s\n", dlerror());
      dlclose(kernel_library);
      dlclose(pack_library);
      return 1;
    }
  }
  const int result = run_case(selected, repetitions, pack, kernel);
  if (kernel_library)
    dlclose(kernel_library);
  dlclose(pack_library);
  return result;
}
