#include "weft/ggml.h"
#include "weft/runtime.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {
using Dot = void (*)(const uint8_t *, const uint8_t *, float *, size_t);

struct StorageContract {
  ggml_type weight;
  const char *weightEncoding;
  const char *activationEncoding;
};

constexpr StorageContract contracts[] = {
    {GGML_TYPE_Q4_0, "Q4_0", "Q8_0"}, {GGML_TYPE_Q4_1, "Q4_1", "Q8_1"},
    {GGML_TYPE_Q5_0, "Q5_0", "Q8_0"}, {GGML_TYPE_Q5_1, "Q5_1", "Q8_1"},
    {GGML_TYPE_Q8_0, "Q8_0", "Q8_0"}, {GGML_TYPE_Q2_K, "Q2_K", "Q8_K"},
    {GGML_TYPE_Q3_K, "Q3_K", "Q8_K"}, {GGML_TYPE_Q4_K, "Q4_K", "Q8_K"},
    {GGML_TYPE_Q5_K, "Q5_K", "Q8_K"}, {GGML_TYPE_Q6_K, "Q6_K", "Q8_K"}};

[[noreturn]] void fail(const std::string &message) {
  GGML_ABORT("Weft CPU integration: %s", message.c_str());
}

void require(bool condition, const std::string &message) {
  if (!condition) fail(message);
}

struct AtomicCounters {
  std::atomic<uint64_t> nodes{0}, extra{0}, calls{0}, dots{0}, quantizes{0}, elements{0};
  ggml_weft_counters snapshot() const {
    return {nodes.load(std::memory_order_relaxed), extra.load(std::memory_order_relaxed),
             calls.load(std::memory_order_relaxed), dots.load(std::memory_order_relaxed),
             quantizes.load(std::memory_order_relaxed), elements.load(std::memory_order_relaxed)};
  }
};

struct alignas(64) ThreadCounters {
  std::array<AtomicCounters, GGML_TYPE_COUNT> types;
};

void accumulate(ggml_weft_counters &into, const ggml_weft_counters &value) {
  into.nodes += value.nodes;
  into.extra_buffer_nodes += value.extra_buffer_nodes;
  into.vec_dot_calls += value.vec_dot_calls;
  into.dot_products += value.dot_products;
  into.quantize_calls += value.quantize_calls;
  into.quantized_elements += value.quantized_elements;
}

struct CounterRegistry {
  std::mutex mutex;
  std::array<ggml_weft_counters, GGML_TYPE_COUNT> retired{};
  std::vector<ThreadCounters *> active;
  void attach(ThreadCounters *counters) {
    std::lock_guard<std::mutex> lock(mutex);
    active.push_back(counters);
  }
  void detach(ThreadCounters *counters) {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t type = 0; type < retired.size(); ++type) accumulate(retired[type], counters->types[type].snapshot());
    for (auto it = active.begin(); it != active.end(); ++it) {
      if (*it == counters) { active.erase(it); return; }
    }
    fail("worker counters were not registered");
  }
  ggml_weft_counters snapshot(ggml_type type) {
    std::lock_guard<std::mutex> lock(mutex);
    auto result = retired[type];
    for (const auto *counters : active) accumulate(result, counters->types[type].snapshot());
    return result;
  }
};

struct ThreadState {
  std::shared_ptr<CounterRegistry> registry;
  ThreadCounters counters;
  ggml_type type = GGML_TYPE_COUNT;
  bool reference = false;
  ~ThreadState() { if (registry) registry->detach(&counters); }
};
thread_local ThreadState worker;

struct Registration {
  weft::kernel kernel;
  Dot function;
  size_t block, weightBytes, activationBytes;
  std::string name;
  Registration(weft::kernel value, const StorageContract &contract) : kernel(std::move(value)) {
    auto info = kernel.info();
    require(info.argument_count == 3 && info.dimension_count == 1 && std::strcmp(info.dimensions[0], "K") == 0,
            "vec-dot requires three Views and exactly one runtime K");
    const auto *traits = ggml_get_type_traits_cpu(contract.weight);
    require(traits && traits->vec_dot && traits->nrows == 1, "upstream vec-dot row contract differs from the registered ABI");
    auto activation = traits->vec_dot_type;
    block = ggml_blck_size(contract.weight);
    weightBytes = ggml_type_size(contract.weight);
    activationBytes = ggml_type_size(activation);
    require(block == static_cast<size_t>(ggml_blck_size(activation)), "activation block geometry differs from the compiled dot");
    for (size_t index = 0; index < 3; ++index) {
      auto arg = kernel.argument(index);
      const char *encoding = index == 0 ? contract.weightEncoding : index == 1 ? contract.activationEncoding : "dense.f32";
      require(arg.kind == WEFT_VIEW && !arg.strided && arg.interleave_rows == 0 && arg.rank == 1 &&
              std::strcmp(arg.shape[0], index == 2 ? "1" : "K") == 0 && std::strcmp(arg.encoding, encoding) == 0 &&
              arg.access == (index == 2 ? WEFT_WRITE : WEFT_READ) &&
              arg.record_elements == (index == 2 ? 1 : block) &&
              arg.storage_bytes == (index == 0 ? weightBytes : index == 1 ? activationBytes : sizeof(float)) &&
              arg.alignment <= (index == 2 || (index == 1 && activation == GGML_TYPE_Q8_K) ? alignof(float) : 2),
              "artifact View/storage ABI differs from the actual GGML type " + std::string(ggml_type_name(contract.weight)));
    }
    function = reinterpret_cast<Dot>(kernel.entry());
    name = info.symbol;
  }
};

// The explicit framework bindings name artifacts; the compiler sees only the
// ordinary typed author programs and never selects by these role names.
struct Adapter {
  weft::context context;
  weft::package package;
  std::array<std::unique_ptr<Registration>, GGML_TYPE_COUNT> entries;
  std::shared_ptr<CounterRegistry> counters = std::make_shared<CounterRegistry>();
  explicit Adapter(const char *directory) : package(context, directory) {
    for (const auto &[typeName, name] : package.bindings("ggml.vec_dot")) {
      bool matched = false;
      for (const auto &contract : contracts) {
        if (typeName != ggml_type_name(contract.weight)) continue;
        require(!entries[contract.weight], "duplicate registered weight type");
        entries[contract.weight] = std::make_unique<Registration>(package.get_kernel(name), contract);
        matched = true;
        break;
      }
      require(matched, "package requests an unsupported GGML vec-dot storage contract: " + typeName);
    }
    bool any = false;
    for (const auto &entry : entries) any |= bool(entry);
    require(any, "package does not contain any ggml.vec_dot bindings");
  }
};

Adapter &adapter() {
  static std::unique_ptr<Adapter> result = [] {
    const char *directory = std::getenv("WEFT_GGML_PACKAGE");
    require(directory && *directory, "GGML_WEFT requires an explicit WEFT_GGML_PACKAGE directory");
    try { return std::make_unique<Adapter>(directory); }
    catch (const std::exception &error) { fail(error.what()); }
  }();
  return *result;
}

ThreadState &threadState() {
  if (!worker.registry) {
    worker.registry = adapter().counters;
    worker.registry->attach(&worker.counters);
  }
  return worker;
}

template <ggml_type Type>
void dot(int n, float *s, size_t bs, const void *x, size_t bx,
         const void *y, size_t by, int nrc) {
  const auto &entry = *adapter().entries[Type];
  require(n >= 0 && static_cast<size_t>(n) % entry.block == 0 && nrc == 1 && s && x && y,
          "selected vec-dot call violates its row/block contract");
  (void)bs; (void)bx; (void)by; // The upstream RISC-V traits explicitly have nrows=1.
  entry.function(static_cast<const uint8_t *>(x), static_cast<const uint8_t *>(y), s, n);
  auto &count = threadState().counters.types[Type];
  count.calls.fetch_add(1, std::memory_order_relaxed);
  count.dots.fetch_add(1, std::memory_order_relaxed);
}

ggml_vec_dot_t entryFunction(ggml_type type) {
  switch (type) {
  case GGML_TYPE_Q4_0: return dot<GGML_TYPE_Q4_0>;
  case GGML_TYPE_Q4_1: return dot<GGML_TYPE_Q4_1>;
  case GGML_TYPE_Q5_0: return dot<GGML_TYPE_Q5_0>;
  case GGML_TYPE_Q5_1: return dot<GGML_TYPE_Q5_1>;
  case GGML_TYPE_Q8_0: return dot<GGML_TYPE_Q8_0>;
  case GGML_TYPE_Q2_K: return dot<GGML_TYPE_Q2_K>;
  case GGML_TYPE_Q3_K: return dot<GGML_TYPE_Q3_K>;
  case GGML_TYPE_Q4_K: return dot<GGML_TYPE_Q4_K>;
  case GGML_TYPE_Q5_K: return dot<GGML_TYPE_Q5_K>;
  case GGML_TYPE_Q6_K: return dot<GGML_TYPE_Q6_K>;
  default: fail("no adapter entry for the selected weight type");
  }
}

bool matrixNode(const ggml_tensor *tensor) {
  return tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID;
}
}

extern "C" {
void ggml_weft_initialize(void) { (void)adapter(); }

bool ggml_weft_registered(enum ggml_type type) {
  return type >= 0 && type < GGML_TYPE_COUNT && bool(adapter().entries[type]);
}

const char *ggml_weft_kernel_name(enum ggml_type type) {
  return ggml_weft_registered(type) ? adapter().entries[type]->name.c_str() : nullptr;
}

bool ggml_weft_counters_get(enum ggml_type type, ggml_weft_counters *result) {
  if (!result || type < 0 || type >= GGML_TYPE_COUNT) return false;
  *result = adapter().counters->snapshot(type);
  return true;
}

void ggml_weft_begin_node(const struct ggml_tensor *tensor, int ith, bool use_ref) {
  auto &state = threadState();
  state.reference = use_ref;
  state.type = matrixNode(tensor) ? tensor->src[0]->type : GGML_TYPE_COUNT;
  if (state.type != GGML_TYPE_COUNT && ith == 0)
    state.counters.types[state.type].nodes.fetch_add(1, std::memory_order_relaxed);
}

void ggml_weft_extra_node(const struct ggml_tensor *tensor, int ith) {
  if (matrixNode(tensor) && ith == 0)
    threadState().counters.types[tensor->src[0]->type].extra.fetch_add(1, std::memory_order_relaxed);
}

ggml_vec_dot_t ggml_weft_select_vec_dot(enum ggml_type type, ggml_vec_dot_t original) {
  auto &state = threadState();
  if (state.reference || !ggml_weft_registered(type)) return original;
  // One guard per GGML worker chunk. The ordinary loop does not change affinity
  // or vector control between this check and its direct microkernel calls.
  try { adapter().entries[type]->kernel.check_thread(); }
  catch (const std::exception &error) { fail(error.what()); }
  return entryFunction(type);
}

void ggml_weft_from_float(ggml_from_float_t original, const float *x, void *y, int64_t n) {
  require(original && n >= 0, "invalid upstream activation quantization call");
  original(x, y, n);
  auto &state = threadState();
  require(state.type != GGML_TYPE_COUNT, "quantization hook is outside a matrix node");
  auto &count = state.counters.types[state.type];
  count.quantizes.fetch_add(1, std::memory_order_relaxed);
  count.elements.fetch_add(n, std::memory_order_relaxed);
}
}
