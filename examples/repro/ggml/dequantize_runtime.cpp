#include "ggml-quants.h"
#include "ggml.h"
#include "quants.h"

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

namespace {

constexpr std::size_t kN = 1024;
constexpr std::size_t kK = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;

using dequantize_fn = void (*)(const void *, float *, std::int64_t);

template <typename Block,
          void (*Function)(const Block *, float *, std::int64_t)>
void dequantize_adapter(const void *input, float *output, std::int64_t count) {
  Function(static_cast<const Block *>(input), output, count);
}

struct dequantize_case {
  const char *name;
  ggml_type type;
  dequantize_fn dequantize;
  std::size_t block_elements;
  unsigned input_seed_offset;
};

const dequantize_case cases[] = {
    {"q1_0", GGML_TYPE_Q1_0,
     dequantize_adapter<block_q1_0, dequantize_row_q1_0>, 128, 0},
    {"q4_0", GGML_TYPE_Q4_0,
     dequantize_adapter<block_q4_0, dequantize_row_q4_0>, 32, 1},
    {"q4_1", GGML_TYPE_Q4_1,
     dequantize_adapter<block_q4_1, dequantize_row_q4_1>, 32, 2},
    {"q5_0", GGML_TYPE_Q5_0,
     dequantize_adapter<block_q5_0, dequantize_row_q5_0>, 32, 3},
    {"q5_1", GGML_TYPE_Q5_1,
     dequantize_adapter<block_q5_1, dequantize_row_q5_1>, 32, 4},
    {"q8_0", GGML_TYPE_Q8_0,
     dequantize_adapter<block_q8_0, dequantize_row_q8_0>, 32, 5},
    {"mxfp4", GGML_TYPE_MXFP4,
     dequantize_adapter<block_mxfp4, dequantize_row_mxfp4>, 32, 22},
    {"nvfp4", GGML_TYPE_NVFP4,
     dequantize_adapter<block_nvfp4, dequantize_row_nvfp4>, 64, 23},
    {"q2_K", GGML_TYPE_Q2_K,
     dequantize_adapter<block_q2_K, dequantize_row_q2_K>, 256, 6},
    {"q3_K", GGML_TYPE_Q3_K,
     dequantize_adapter<block_q3_K, dequantize_row_q3_K>, 256, 7},
    {"q4_K", GGML_TYPE_Q4_K,
     dequantize_adapter<block_q4_K, dequantize_row_q4_K>, 256, 8},
    {"q5_K", GGML_TYPE_Q5_K,
     dequantize_adapter<block_q5_K, dequantize_row_q5_K>, 256, 9},
    {"q6_K", GGML_TYPE_Q6_K,
     dequantize_adapter<block_q6_K, dequantize_row_q6_K>, 256, 10},
    {"tq1_0", GGML_TYPE_TQ1_0,
     dequantize_adapter<block_tq1_0, dequantize_row_tq1_0>, 256, 20},
    {"tq2_0", GGML_TYPE_TQ2_0,
     dequantize_adapter<block_tq2_0, dequantize_row_tq2_0>, 256, 21},
    {"iq2_xxs", GGML_TYPE_IQ2_XXS,
     dequantize_adapter<block_iq2_xxs, dequantize_row_iq2_xxs>, 256, 15},
    {"iq2_xs", GGML_TYPE_IQ2_XS,
     dequantize_adapter<block_iq2_xs, dequantize_row_iq2_xs>, 256, 14},
    {"iq2_s", GGML_TYPE_IQ2_S,
     dequantize_adapter<block_iq2_s, dequantize_row_iq2_s>, 256, 13},
    {"iq3_xxs", GGML_TYPE_IQ3_XXS,
     dequantize_adapter<block_iq3_xxs, dequantize_row_iq3_xxs>, 256, 17},
    {"iq3_s", GGML_TYPE_IQ3_S,
     dequantize_adapter<block_iq3_s, dequantize_row_iq3_s>, 256, 16},
    {"iq1_s", GGML_TYPE_IQ1_S,
     dequantize_adapter<block_iq1_s, dequantize_row_iq1_s>, 256, 11},
    {"iq1_m", GGML_TYPE_IQ1_M,
     dequantize_adapter<block_iq1_m, dequantize_row_iq1_m>, 256, 12},
    {"iq4_nl", GGML_TYPE_IQ4_NL,
     dequantize_adapter<block_iq4_nl, dequantize_row_iq4_nl>, 32, 18},
    {"iq4_xs", GGML_TYPE_IQ4_XS,
     dequantize_adapter<block_iq4_xs, dequantize_row_iq4_xs>, 256, 19},
};

volatile std::uint64_t flush_sink = 0;

const dequantize_case *find_case(const char *name) {
  for (const dequantize_case &candidate : cases) {
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

void run_dequantize(const dequantize_case &selected,
                    const std::uint8_t *input, float *output,
                    std::size_t input_row_bytes) {
  for (std::size_t row = 0; row < kN; ++row) {
    selected.dequantize(input + row * input_row_bytes, output + row * kK, kK);
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
    std::fprintf(stderr, "usage: %s <dequantize-format> <repetitions>\n", argv[0]);
    return 2;
  }
  const dequantize_case *selected = find_case(argv[1]);
  if (selected == nullptr) {
    std::fprintf(stderr, "unsupported GGML dequantizer: %s\n", argv[1]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  const std::size_t input_row_bytes = ggml_row_size(selected->type, kK);
  const std::size_t record_bytes =
      ggml_row_size(selected->type, selected->block_elements);
  std::vector<std::uint8_t> input_record(record_bytes);
  std::vector<float> probe(selected->block_elements);
  std::mt19937 generator(0x57454654U + selected->input_seed_offset);
  std::uniform_int_distribution<unsigned> bytes(0, 255);
  bool found_finite_record = false;
  for (int attempt = 0; attempt < 4096; ++attempt) {
    for (std::uint8_t &byte : input_record) {
      byte = static_cast<std::uint8_t>(bytes(generator));
    }
    selected->dequantize(input_record.data(), probe.data(),
                         selected->block_elements);
    found_finite_record =
        std::all_of(probe.begin(), probe.end(),
                    [](float value) { return std::isfinite(value); });
    if (found_finite_record) {
      break;
    }
  }
  if (!found_finite_record) {
    std::fprintf(stderr, "failed to generate finite random encoded data\n");
    return 2;
  }
  if (input_row_bytes % record_bytes != 0) {
    std::fprintf(stderr, "encoded row is not a whole number of records\n");
    return 2;
  }
  std::vector<std::uint8_t> input_row(input_row_bytes);
  for (std::size_t offset = 0; offset < input_row_bytes;
       offset += record_bytes) {
    std::memcpy(input_row.data() + offset, input_record.data(), record_bytes);
  }
  std::vector<std::uint8_t> input(kN * input_row_bytes);
  for (std::size_t row = 0; row < kN; ++row) {
    std::memcpy(input.data() + row * input_row_bytes, input_row.data(),
                input_row_bytes);
  }
  std::vector<float> output(kN * kK);
  std::vector<std::uint8_t> flush_buffer(kFlushBytes, 1);

  run_dequantize(*selected, input.data(), output.data(), input_row_bytes);
  for (float value : output) {
    if (!std::isfinite(value)) {
      std::fprintf(stderr, "non-finite output from dequantize_row_%s\n", argv[1]);
      return 1;
    }
  }

  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evict_cache(flush_buffer);
    const auto begin = std::chrono::steady_clock::now();
    run_dequantize(*selected, input.data(), output.data(), input_row_bytes);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double cold_median_us = median(samples);
  const double elements = static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=dequantize\n");
  std::printf("kernel=dequantize_row_%s\n", selected->name);
  std::printf("implementation=scalar\n");
  std::printf("model_shape=DeepSeek-R1-Distill-Llama-8B.attn_k\n");
  std::printf("N=%zu\nK=%zu\n", kN, kK);
  std::printf("input_policy=finite-random-record-replicated\n");
  std::printf("input_seed=%u\n",
              0x57454654U + selected->input_seed_offset);
  std::printf("cold_protocol=64MiB-evict-then-full-tensor\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", cold_median_us);
  std::printf("cold_melements_s=%.6f\n", elements / cold_median_us);
  std::printf("output_sample=%.9g\n", output[0]);
  return 0;
}
