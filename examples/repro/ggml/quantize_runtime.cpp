#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kM = 128;
constexpr std::size_t kK = 14336;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

using quantize_fn = void (*)(const float *, void *, std::int64_t);

struct quantize_case {
  const char *name;
  ggml_type type;
  quantize_fn quantize;
};

const quantize_case cases[] = {
    {"q8_0", GGML_TYPE_Q8_0, quantize_row_q8_0},
    {"q8_1", GGML_TYPE_Q8_1, quantize_row_q8_1},
    {"q8_K", GGML_TYPE_Q8_K, quantize_row_q8_K},
};

volatile std::uint64_t flush_sink = 0;

const quantize_case *find_case(const char *name) {
  for (const quantize_case &candidate : cases) {
    if (std::strcmp(name, candidate.name) == 0) {
      return &candidate;
    }
  }
  return nullptr;
}

std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0) {
    return 0;
  }
  return static_cast<std::size_t>(value);
}

void evict_cache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t i = 0; i < buffer.size(); i += 64) {
    buffer[i] = static_cast<std::uint8_t>(buffer[i] + 1U);
    flush_sink += buffer[i];
  }
}

void run_quantize(const quantize_case &selected, const float *input,
                  std::uint8_t *output, std::size_t output_row_bytes) {
  for (std::size_t row = 0; row < kM; ++row) {
    selected.quantize(input + row * kK, output + row * output_row_bytes, kK);
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <q8_0|q8_1|q8_K> <repetitions>\n", argv[0]);
    return 2;
  }
  const quantize_case *selected = find_case(argv[1]);
  if (selected == nullptr) {
    std::fprintf(stderr, "unsupported GGML activation quantizer: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  std::vector<float> input(kM * kK);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0F;
  }
  const std::size_t output_row_bytes = ggml_row_size(selected->type, kK);
  std::vector<std::uint8_t> output(kM * output_row_bytes);
  std::vector<std::uint8_t> flush_buffer(kFlushBytes, 1);

  ggml_backend_t backend = ggml_backend_cpu_init();
  const int vlen_bytes = ggml_cpu_get_rvv_vlen();

  run_quantize(*selected, input.data(), output.data(), output_row_bytes);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush_buffer);
    const auto begin = std::chrono::steady_clock::now();
    run_quantize(*selected, input.data(), output.data(), output_row_bytes);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double cold_median_us = median(samples);
  const double elements = static_cast<double>(kM) * static_cast<double>(kK);
  std::printf("family=quantize\n");
  std::printf("kernel=quantize_row_%s\n", selected->name);
  std::printf("implementation=rvv_intrinsic\n");
  std::printf("model_shape=DeepSeek-R1-Distill-Llama-8B.ffn_down_activation\n");
  std::printf("M=%zu\nK=%zu\n", kM, kK);
  std::printf("vlen_bits=%d\n", vlen_bytes * 8);
  std::printf("cold_protocol=64MiB-evict-then-full-activation\n");
  std::printf("input_policy=fixed-float-mod31\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", cold_median_us);
  std::printf("cold_melements_s=%.6f\n", elements / cold_median_us);
  std::printf("output_sample=%u\n", static_cast<unsigned>(output[0]));

  ggml_backend_free(backend);
  return 0;
}
