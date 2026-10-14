#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef WEFT_GGML_OBSERVE
#include <weft/ggml.h>
#endif

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

static void compare(const std::vector<float> &actual, const std::vector<float> &expected) {
  require(actual.size() == expected.size(), "output sizes differ");
  double maximum = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double error = std::fabs(double(actual[i]) - expected[i]);
    maximum = std::max(maximum, error);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error > 1e-4 + 2e-3 * std::fabs(expected[i])) {
      std::fprintf(stderr, "element %zu: actual %.9g reference %.9g\n", i, actual[i], expected[i]);
      throw std::runtime_error("GGML graph numerical mismatch");
    }
  }
  std::printf(" max_abs=%.9g", maximum);
}

struct Reference {
  std::ofstream record;
  std::map<std::string, std::vector<float>> values;
  bool reading = false;
  Reference(int argc, char **argv) {
    if (argc == 1) return;
    require(argc == 3, "usage: ggml-nodes [--record FILE | --reference FILE]");
    if (std::string(argv[1]) == "--record") {
#ifdef WEFT_GGML_OBSERVE
      throw std::runtime_error("record the normal GGML build with integration OFF");
#else
      record.open(argv[2]);
      require(bool(record), "cannot create normal-path output record");
      record << std::setprecision(std::numeric_limits<float>::max_digits10);
#endif
    } else {
      require(std::string(argv[1]) == "--reference", "unknown reference argument");
      reading = true;
      std::ifstream input(argv[2]);
      require(bool(input), "cannot open normal-path outputs");
      std::string key;
      size_t count;
      while (input >> key >> count) {
        require(count <= 1000000 && !values.count(key), "invalid or duplicate normal-path output record");
        auto &row = values[key];
        row.resize(count);
        for (auto &value : row) require(bool(input >> value), "truncated normal-path output record");
      }
      require(input.eof() && !values.empty(), "invalid normal-path output file");
    }
  }
  void save(const std::string &key, const std::vector<float> &output) {
    if (!record.is_open()) return;
    record << key << ' ' << output.size();
    for (float value : output) record << ' ' << value;
    record << '\n';
    require(bool(record), "cannot write normal-path outputs");
  }
  std::vector<float> take(const std::string &key) {
    auto found = values.find(key);
    require(found != values.end(), "normal build did not provide this type/shape/threads/buffer case");
    auto output = std::move(found->second);
    values.erase(found);
    return output;
  }
  void finish() {
    require(!reading || values.empty(), "normal-path outputs contain unmatched cases");
    if (record.is_open()) {
      record.close();
      require(bool(record), "cannot finish normal-path output record");
    }
  }
};

struct Backend {
  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_threadpool_t pool = nullptr;
  explicit Backend(int threads) {
    require(backend, "CPU backend creation failed");
    auto parameters = ggml_threadpool_params_default(threads);
    pool = ggml_threadpool_new(&parameters);
    require(pool, "CPU threadpool creation failed");
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_backend_cpu_set_threadpool(backend, pool);
  }
  ~Backend() {
    ggml_backend_free(backend);
    ggml_threadpool_free(pool);
  }
};

struct Graph {
  static constexpr int64_t k = 1024, rows = 64, experts = 4, selected = 2;
  ggml_context *weights = nullptr, *context = nullptr;
  ggml_backend_buffer_t weightBuffer = nullptr, buffer = nullptr;
  ggml_tensor *w, *x, *ids = nullptr, *result;
  ggml_cgraph *graph;
  bool supported = false;
  Graph(Backend &backend, ggml_type type, int64_t tokens, bool indirect, bool prequant,
        ggml_backend_buffer_type_t extra = nullptr) {
    weights = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
    context = ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true});
    require(weights && context, "graph metadata allocation failed");
    w = ggml_new_tensor_3d(weights, type, k, rows, indirect ? experts : 1);
    ggml_set_name(w, "projection.weight");
    auto inputType = prequant ? ggml_get_type_traits_cpu(type)->vec_dot_type : GGML_TYPE_F32;
    x = indirect ? ggml_new_tensor_3d(context, inputType, k, 1, tokens)
                 : ggml_new_tensor_2d(context, inputType, k, tokens);
    ggml_set_name(x, "activation");
    if (indirect) {
      ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, selected, tokens);
      result = ggml_mul_mat_id(context, w, x, ids);
    } else {
      result = ggml_mul_mat(context, w, x);
    }
    // Include a normal downstream activation node in the actual graph.
    result = ggml_silu(context, result);
    graph = ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(graph, result);
    weightBuffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, extra ? extra : ggml_backend_cpu_buffer_type());
    buffer = ggml_backend_alloc_ctx_tensors(context, backend.backend);
    require(weightBuffer && buffer, "graph buffer allocation failed");
    ggml_backend_buffer_set_usage(weightBuffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    supported = ggml_backend_dev_supports_op(ggml_backend_get_device(backend.backend), result->src[0]);
    if (!supported) return;
    std::vector<float> wf(ggml_nelements(w)), xf(ggml_nelements(x));
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = float(int((i * 19 + 7) % 127) - 63) / 127;
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = float(int((i * 37 + 11) % 251) - 125) / 251;
    std::vector<uint8_t> packed(ggml_nbytes(w));
    auto quantized = ggml_quantize_chunk(type, wf.data(), packed.data(), 0, ggml_nrows(w), k, nullptr);
    require(quantized == packed.size(), "weight quantization size mismatch");
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    if (prequant) {
      packed.resize(ggml_nbytes(x));
      auto quantize = ggml_get_type_traits_cpu(inputType)->from_float;
      require(quantize, "activation type has no quantizer");
      quantize(xf.data(), packed.data(), xf.size());
      ggml_backend_tensor_set(x, packed.data(), 0, packed.size());
    } else {
      ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    }
    if (ids) {
      std::vector<int32_t> routing(selected * tokens);
      for (int64_t token = 0; token < tokens; ++token)
        for (int64_t expert = 0; expert < selected; ++expert)
          routing[token * selected + expert] = (token + expert) % experts;
      ggml_backend_tensor_set(ids, routing.data(), 0, routing.size() * sizeof(int32_t));
    }
  }
  ~Graph() {
    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(weightBuffer);
    ggml_free(context);
    ggml_free(weights);
  }
  std::vector<float> run(Backend &backend, bool reference) {
    ggml_backend_cpu_set_use_ref(backend.backend, reference);
    require(ggml_backend_graph_compute(backend.backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<float> output(ggml_nelements(result));
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    return output;
  }
};

static bool run_case(Reference &reference, Backend &backend, ggml_type type, int threads, int64_t tokens,
                     bool indirect, bool prequant, ggml_backend_buffer_type_t extra = nullptr) {
  Graph graph(backend, type, tokens, indirect, prequant, extra);
  if (!graph.supported) {
    require(extra, "ordinary CPU graph is unsupported");
    std::printf("type=%s buffer=%s upstream reports unsupported on this target\n",
                ggml_type_name(type), ggml_backend_buft_name(extra));
    return false;
  }
#ifdef WEFT_GGML_OBSERVE
  ggml_weft_counters before{}, after{};
  require(ggml_weft_counters_get(type, &before), "counter query failed");
#endif
  auto actual = graph.run(backend, false);
  auto repeated = graph.run(backend, false);
#ifdef WEFT_GGML_OBSERVE
  require(ggml_weft_counters_get(type, &after), "counter query failed");
  const auto dots = after.dot_products - before.dot_products;
  const auto quantized = after.quantized_elements - before.quantized_elements;
  const auto packed = after.extra_buffer_nodes - before.extra_buffer_nodes;
  require(after.nodes - before.nodes == 2, "matrix node count does not match graph executions");
  if (extra) {
    require(packed == 2 && dots == 0 && quantized == 0, "normal packed path was intercepted or misreported");
  } else {
    const bool registered = ggml_weft_registered(type);
    require(dots == (registered ? uint64_t(2 * Graph::rows * tokens * (indirect ? Graph::selected : 1)) : 0),
            "vec-dot count differs from mathematical output rows");
    require(quantized == (prequant ? 0 : uint64_t(2 * Graph::k * tokens)),
            "activation quantization was repeated or omitted");
  }
#endif
  std::string bufferName = extra ? ggml_backend_buft_name(extra) : "CPU";
  std::string key = std::string(ggml_type_name(type)) + ":" + std::to_string(threads) + ":" +
                    std::to_string(tokens) + ":" + std::to_string(indirect) + ":" +
                    std::to_string(prequant) + ":" + bufferName;
  std::vector<float> expected;
  if (reference.reading) {
    expected = reference.take(key);
  } else if (!extra) {
    Graph original(backend, type, tokens, indirect, prequant);
    expected = original.run(backend, true);
  } else {
    require(reference.record.is_open(), "packed nodes need --reference from the matching ordinary build");
    // The framework's packed path can use a different activation quantizer.
    // Record that actual path; do not substitute a different mathematical input.
    expected = actual;
  }
  reference.save(key, actual);
  std::printf("type=%s threads=%d tokens=%lld indirect=%d prequant=%d buffer=%s",
              ggml_type_name(type), threads, static_cast<long long>(tokens), indirect, prequant,
              extra ? ggml_backend_buft_name(extra) : "CPU");
  compare(actual, expected);
  compare(repeated, expected);
#ifdef WEFT_GGML_OBSERVE
  std::printf(" dots=%llu quantized=%llu packed_nodes=%llu", (unsigned long long)dots,
              (unsigned long long)quantized, (unsigned long long)packed);
#endif
  std::puts(reference.record.is_open() ? " normal-path recorded" : " passed");
  return true;
}

int main(int argc, char **argv) {
  try {
    Reference reference(argc, argv);
    for (int threads : {1, 4}) {
      Backend backend(threads);
      for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
        run_case(reference, backend, type, threads, 1, false, false);
        run_case(reference, backend, type, threads, 9, false, true);
        run_case(reference, backend, type, threads, 7, true, false);
      }
      auto registration = ggml_backend_cpu_reg();
      auto extras = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
          ggml_backend_reg_get_proc_address(registration, "ggml_backend_dev_get_extra_bufts"));
      require(extras, "normal CPU backend does not expose extra buffer types");
      auto types = extras(ggml_backend_reg_dev_get(registration, 0));
      bool any = false;
      for (size_t i = 0; types && types[i]; ++i) {
        std::string name = ggml_backend_buft_name(types[i]);
        if (name != "CPU_REPACK" && name != "CPU_RISCV64_SPACEMIT") continue;
        any |= run_case(reference, backend, GGML_TYPE_Q4_0, threads, 1, false, false, types[i]);
        any |= run_case(reference, backend, GGML_TYPE_Q4_0, threads, 9, false, false, types[i]);
      }
      std::printf("threads=%d available_packed_path_checked=%d\n", threads, any);
    }
    reference.finish();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
