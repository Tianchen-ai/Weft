#include <weft/runtime.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sched.h>
#include <cerrno>
#include <vector>

static void close(float actual, float expected) {
  if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-4 + 2e-3 * std::fabs(expected))
    throw std::runtime_error("native SDK numerical mismatch");
}

static void rejected(const std::function<void()> &action, unsigned line = __builtin_LINE()) {
  try { action(); }
  catch (const weft::error &error) {
    if (error.status != WEFT_INVALID_ARGUMENT) throw;
    return;
  }
  throw std::runtime_error("invalid SDK binding was accepted at line " + std::to_string(line));
}

static weft::sequence prepare_state(const char *directory, float *output) {
  weft::context context;
  weft::package package(context, directory);
  auto update = package.get_kernel("state_update");
  auto scale = package.get_kernel("state_scale");
  auto affine = package.get_kernel("strided");
  auto layout = update.storage_layout(0, {{"M", 1}, {"N", 17}});
  auto state = layout.allocate(), delta = layout.allocate();
  for (int i = 0; i < 17; ++i) {
    static_cast<float *>(state.data())[i] = i;
    static_cast<float *>(delta.data())[i] = 1;
  }
  auto u = update.bind();
  u.retain("State", state, layout).retain("Delta", delta, layout).set("decay", weft::scalar(0.5f));
  auto first = u.prepare();
  // Changing the builder cannot mutate the already prepared scalar argument.
  u.set("decay", weft::scalar(99.0f));
  auto s = scale.bind();
  s.retain("State", state, {1, 17}).set("scale", weft::scalar(2.0f));
  auto second = s.prepare();
  auto a = affine.bind();
  a.retain("X", state, {1, 17}, {}, 0, true)
   .set("Y", weft::view(output, 17 * sizeof(float), "dense.f32", {1, 17}))
   .set("scale", weft::scalar(1.0f)).set("bias", weft::scalar(0.0f));
  auto third = a.prepare();
  return weft::sequence{&first, &second, &third};
}

static void binding_contracts(weft::package &package) {
  auto kernel = package.get_kernel("strided");
  std::vector<float> source(129), destination(3 * 129, -77);
  for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(i) / 129;
  weft::view input(source.data(), source.size() * sizeof(float), "dense.f32", {3, 129}, {0, 1}, true);
  weft::view output(destination.data(), destination.size() * sizeof(float), "dense.f32", {3, 129});
  auto run = kernel.prepare({input.argument(), output.argument(), weft::scalar(2.f), weft::scalar(1.f)});
  run();
  for (size_t row = 0; row < 3; ++row)
    for (size_t column = 0; column < 129; ++column)
      close(destination[row * 129 + column], source[column] * 2 + 1);

  rejected([&] {
    auto invalid = kernel.prepare({input.argument(), output.argument(), weft::scalar(1.f), weft::scalar(0.f)},
                                  {{"M", 3}, {"M", 3}});
  });
  rejected([&] {
    auto invalid = kernel.prepare({input.argument(), output.argument(), weft::scalar(1.f), weft::scalar(0.f)},
                                  {{"B", 64}});
  });
  rejected([&] {
    weft::view writable_broadcast(destination.data(), destination.size() * sizeof(float),
                                   "dense.f32", {3, 129}, {0, 1});
    auto invalid = kernel.prepare({input.argument(), writable_broadcast.argument(),
                                   weft::scalar(1.f), weft::scalar(0.f)});
  });
  rejected([&] {
    weft::view readonly(destination.data(), destination.size() * sizeof(float),
                         "dense.f32", {3, 129}, {}, true);
    auto invalid = kernel.prepare({input.argument(), readonly.argument(), weft::scalar(1.f), weft::scalar(0.f)});
  });
  weft::view aliased(source.data(), source.size() * sizeof(float), "dense.f32", {1, 129});
  auto inplace = kernel.prepare({aliased.argument(), aliased.argument(), weft::scalar(1.f), weft::scalar(0.f)});
  inplace();
  for (size_t i = 0; i < source.size(); ++i) close(source[i], static_cast<float>(i) / 129);
  rejected([&] {
    auto invalid = kernel.storage_layout(1, {{"M", 3}, {"N", 129}},
                                           {static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / 4 + 1, 1});
  });
  rejected([&] { auto invalid = kernel.storage_layout(1, {{"M", 3}}); });
  auto empty = kernel.storage_layout(1, {{"M", 3}, {"N", 0}});
  if (empty.info().byte_length != 0) throw std::runtime_error("empty View acquired a nonempty span");

  auto packed = package.get_kernel("dot_q5_0");
  if (packed.storage_size(0, {{"K", 1024}}) != packed.storage_layout(0, {{"K", 1024}}).info().byte_length)
    throw std::runtime_error("allocation query and prepared View disagree about packed bytes");
  const size_t too_many = std::numeric_limits<size_t>::max() / 32 * 32;
  rejected([&] { (void)packed.storage_size(0, {{"K", too_many}}); });
  rejected([&] { auto invalid = packed.storage_layout(0, {{"K", too_many}}); });
  rejected([&] { (void)packed.storage_size(0, {{"K", 33}}); });

  auto bindings = kernel.bind();
  bindings.set("X", input).set("Y", output)
          .set("scale", weft::scalar(1.f)).set("bias", weft::scalar(0.f));
  auto first = bindings.prepare();
  rejected([&] { bindings.set("scale", weft::scalar(7.0)); });
  auto still_valid = bindings.prepare();
  bindings.set("scale", weft::scalar(2.f));
  auto second = bindings.prepare();
  first();
  close(destination[0], source[0]);
  close(destination[128], source[128]);
  second();
  close(destination[128], 2 * source[128]);
  still_valid();
  close(destination[128], source[128]);

  auto scalar_change = first.rebind({{"scale", weft::scalar(3.f)}});
  scalar_change();
  close(destination[128], 3 * source[128]);
  first();
  close(destination[128], source[128]);
  rejected([&] { auto invalid = first.rebind({{"scale", weft::scalar(1.0)}}); });
  rejected([&] {
    auto invalid = first.rebind({{"scale", weft::scalar(1.f)}, {"scale", weft::scalar(2.f)}});
  });
  rejected([&] { auto invalid = first.rebind({{"missing", weft::scalar(1.f)}}); });
  rejected([&] { auto invalid = first.rebind({}, {{"N", 17}}); });
  std::vector<float> smaller(2 * 17, -5);
  weft::view small_input(source.data(), source.size() * sizeof(float), "dense.f32", {2, 17}, {0, 1}, true);
  weft::view small_output(smaller.data(), smaller.size() * sizeof(float), "dense.f32", {2, 17});
  auto resized = first.rebind({{"X", small_input.argument()}, {"Y", small_output.argument()}});
  resized();
  for (size_t i = 0; i < smaller.size(); ++i) close(smaller[i], source[i % 17]);
  auto fixed = kernel.prepare({input.argument(), output.argument(), weft::scalar(1.f), weft::scalar(0.f)},
                              {{"M", 3}, {"N", 129}});
  rejected([&] { auto invalid = fixed.rebind({{"X", small_input.argument()}, {"Y", small_output.argument()}}); });
  auto explicitly_resized = fixed.rebind({{"X", small_input.argument()}, {"Y", small_output.argument()}},
                                         {{"M", 2}, {"N", 17}});
  explicitly_resized();
  for (size_t i = 0; i < smaller.size(); ++i) close(smaller[i], source[i % 17]);
}

static weft::call rebind_owned_storage(const char *directory, float *output) {
  weft::context context;
  weft::package package(context, directory);
  auto kernel = package.get_kernel("strided");
  auto layout = kernel.storage_layout(0, {{"M", 1}, {"N", 17}});
  auto storage = layout.allocate();
  for (int i = 0; i < 17; ++i) static_cast<float *>(storage.data())[i] = static_cast<float>(i);
  auto builder = kernel.bind();
  builder.retain("X", storage, layout, 0, true)
         .set("Y", weft::view(output, 17 * sizeof(float), "dense.f32", {1, 17}))
         .set("scale", weft::scalar(1.f)).set("bias", weft::scalar(0.f));
  auto original = builder.prepare();
  return original.rebind({{"scale", weft::scalar(2.f)}});
}

class SavedAffinity {
  cpu_set_t original;
public:
  SavedAffinity() {
    if (sched_getaffinity(0, sizeof(original), &original) != 0)
      throw std::runtime_error(std::string("getaffinity: ") + std::strerror(errno));
  }
  const cpu_set_t &get() const { return original; }
  static void set(const cpu_set_t &value) {
    if (sched_setaffinity(0, sizeof(value), &value) != 0)
      throw std::runtime_error(std::string("setaffinity: ") + std::strerror(errno));
  }
  ~SavedAffinity() {
    if (sched_setaffinity(0, sizeof(original), &original) != 0) std::terminate();
  }
};

static void sequence_prewrite_guards(const char *directory) {
  SavedAffinity affinity;
  const auto original = affinity.get();
  if (CPU_COUNT(&original) != 1)
    throw std::runtime_error("native SDK guard repro requires a single pinned CPU");
  int cpu = 0;
  while (!CPU_ISSET(cpu, &original)) ++cpu;
  // The two official multi-core targets have an adjacent CPU. Discovery must
  // still prove a common ISA for the widened context before loading any kernel.
  int other = cpu ^ 1;
  cpu_set_t wide = original;
  CPU_SET(other, &wide);
  SavedAffinity::set(wide);
  weft::context wide_context;
  weft::package wide_package(wide_context, directory);
  auto early = wide_package.get_kernel("strided");
  SavedAffinity::set(original);
  weft::context narrow_context;
  weft::package narrow_package(narrow_context, directory);
  auto late = narrow_package.get_kernel("strided");
  float x[17], y[17], z[17];
  for (int i = 0; i < 17; ++i) {
    x[i] = i;
    y[i] = -77;
    z[i] = -91;
  }
  weft::view input(x, sizeof(x), "dense.f32", {1, 17}, {}, true);
  weft::view middle(y, sizeof(y), "dense.f32", {1, 17});
  weft::view output(z, sizeof(z), "dense.f32", {1, 17});
  auto first = early.prepare({input.argument(), middle.argument(), weft::scalar(2.f), weft::scalar(1.f)});
  auto second = late.prepare({middle.argument(), output.argument(), weft::scalar(3.f), weft::scalar(2.f)});
  weft::sequence sequence{&first, &second};
  cpu_set_t changed;
  CPU_ZERO(&changed);
  CPU_SET(other, &changed);
  SavedAffinity::set(changed);
  bool refused = false;
  try { sequence(); }
  catch (const weft::error &error) {
    if (error.status != WEFT_TARGET_ERROR) throw;
    refused = true;
  }
  if (!refused) throw std::runtime_error("sequence ignored a later call's narrower CPU domain");
  for (int i = 0; i < 17; ++i) {
    close(y[i], -77);
    close(z[i], -91);
  }
  SavedAffinity::set(original);
  sequence();
  for (int i = 0; i < 17; ++i) close(z[i], (i * 2 + 1) * 3 + 2);
}

static void measure_binding(weft::package &package) {
  auto kernel = package.get_kernel("strided");
  std::vector<float> x(3 * 129), y(3 * 129);
  for (size_t index = 0; index < x.size(); ++index) x[index] = static_cast<float>(index) / 100;
  weft::view input(x.data(), x.size() * sizeof(float), "dense.f32", {3, 129}, {}, true);
  weft::view output(y.data(), y.size() * sizeof(float), "dense.f32", {3, 129});
  auto bind = [&] {
    return kernel.prepare({input.argument(), output.argument(), weft::scalar(1.25f), weft::scalar(-0.5f)});
  };
  auto prepared = bind();
  prepared();
  std::vector<double> prepare_samples, invoke_samples;
  using Clock = std::chrono::steady_clock;
  for (int sample = 0; sample < 10; ++sample) {
    auto start = Clock::now();
    for (int repeat = 0; repeat < 100; ++repeat) {
      auto call = bind();
      call();
    }
    prepare_samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / 100);
    start = Clock::now();
    for (int repeat = 0; repeat < 100; ++repeat) prepared();
    invoke_samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / 100);
    for (size_t index = 0; index < y.size(); ++index) close(y[index], x[index] * 1.25f - 0.5f);
  }
  auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[4] + values[5]) / 2;
  };
  std::printf("{\"entry\":\"native-sdk\",\"prepare_and_call_us\":%.9f,\"prepared_call_us\":%.9f,"
              "\"shape\":[3,129],\"samples\":10,\"repetitions\":100,\"prepare_samples_us\":[",
              median(prepare_samples), median(invoke_samples));
  for (size_t index = 0; index < prepare_samples.size(); ++index)
    std::printf("%s%.9f", index ? "," : "", prepare_samples[index]);
  std::printf("],\"invoke_samples_us\":[");
  for (size_t index = 0; index < invoke_samples.size(); ++index)
    std::printf("%s%.9f", index ? "," : "", invoke_samples[index]);
  std::printf("]}\n");
}

int main(int argc, char **argv) {
  try {
    if (argc != 2 && !(argc == 3 && std::strcmp(argv[2], "--measure") == 0))
      throw std::runtime_error("usage: native-sdk PACKAGE [--measure]");
    weft::context context;
    weft::package package(context, argv[1]);
    if (argc == 3) {
      measure_binding(package);
      return 0;
    }
    binding_contracts(package);
    sequence_prewrite_guards(argv[1]);
    auto affine = package.get_kernel("strided");
    std::vector<float> x(3 * 263, -77), y(3 * 265, -91);
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 129; ++col) x[1 + row * 263 + 2 * col] = (row + col) / 17.0f;
    weft::view input(x.data() + 1, (x.size() - 1) * 4, "dense.f32", {3, 129}, {263, 2}, true);
    weft::view output(y.data() + 1, (y.size() - 1) * 4, "dense.f32", {3, 129}, {265, 2});
    auto prepared = affine.prepare({input.argument(), output.argument(), weft::scalar(1.25f), weft::scalar(-0.5f)});
    prepared(); prepared();
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 129; ++col) {
        close(y[1 + row * 265 + 2 * col], x[1 + row * 263 + 2 * col] * 1.25f - 0.5f);
        close(y[row * 265 + 2 * col], -91);
      }
    rejected([&] {
      auto bad = affine.prepare({input.argument(), output.argument(), weft::scalar(1.0), weft::scalar(0.0f)});
    });
    rejected([&] {
      auto bad = affine.prepare({input.argument(), output.argument(), weft::scalar(1.0f), weft::scalar(0.0f)}, {{"N", 130}});
    });
    rejected([&] {
      weft::view short_view(y.data(), 4, "dense.f32", {3, 129});
      auto bad = affine.prepare({input.argument(), short_view.argument(), weft::scalar(1.0f), weft::scalar(0.0f)});
    });
    auto b = affine.bind();
    auto shape = affine.storage_layout(1, {{"M", 3}, {"N", 129}}, {265, 2});
    if (shape.info().byte_length != (2 * 265 + 2 * 128 + 1) * sizeof(float))
      throw std::runtime_error("strided allocation span differs from its ABI");
    auto packed_layout = package.get_kernel("dot_q5_0").storage_layout(0, {{"K", 1024}});
    if (packed_layout.info().byte_length != 1024 / 32 * 22)
      throw std::runtime_error("packed allocation span differs from its Encoding");
    rejected([&] { auto bad = affine.storage_layout(1, {{"M", 3}, {"N", 129}}, {1, 1}); });
    rejected([&] { b.set("unknown", weft::scalar(1.0f)); });
    rejected([&] { auto bad = b.prepare(); });
    weft::view empty(nullptr, 0, "dense.f32", {3, 0});
    auto empty_call = affine.prepare({empty.argument(), empty.argument(), weft::scalar(1.0f), weft::scalar(0.0f)});
    empty_call();
    float state_output[17];
    auto state = prepare_state(argv[1], state_output);
    state();
    for (int i = 0; i < 17; ++i) close(state_output[i], i + 2);
    state();
    for (int i = 0; i < 17; ++i) close(state_output[i], i + 4);
    auto rebound_owner = rebind_owned_storage(argv[1], state_output);
    rebound_owner();
    for (int i = 0; i < 17; ++i) close(state_output[i], i * 2);
    auto derived_sequence = [&] {
      auto initial = prepare_state(argv[1], state_output);
      return initial.rebind(1, {{"scale", weft::scalar(3.f)}});
    }();
    derived_sequence();
    for (int i = 0; i < 17; ++i) close(state_output[i], i * 1.5f + 3);
    rejected([&] { auto invalid = derived_sequence.rebind(3, {{"scale", weft::scalar(1.f)}}); });
    rejected([&] {
      weft_named_value update{"scale", weft::scalar(1.f)};
      auto invalid = derived_sequence.rebind({{1, &update, 1, nullptr, 0}, {1, &update, 1, nullptr, 0}});
    });
    auto replaced = derived_sequence.replace(2, rebound_owner);
    replaced();
    for (int i = 0; i < 17; ++i) close(state_output[i], i * 2);
    auto unchanged = derived_sequence.rebind({});
    unchanged();
    for (int i = 0; i < 17; ++i) close(state_output[i], i * 3.375f + 14.25f);
    std::printf("native-sdk: strided/scalar/empty/rejection/retained-state/sequence/rebind passed\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
