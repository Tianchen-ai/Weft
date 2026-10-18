#ifndef WEFT_RUNTIME_INTERNAL_H
#define WEFT_RUNTIME_INTERNAL_H

#include "weft/runtime.h"
#include "Weft/Target/RISCVTargetProfile.h"
#include "llvm/Support/JSON.h"

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

namespace weft::runtime {

struct Failure : std::runtime_error {
  weft_status status;
  Failure(weft_status status, const std::string &message)
      : std::runtime_error(message), status(status) {}
};

void require(bool condition, weft_status status, const std::string &message);
llvm::json::Object readObject(const std::string &path);
const llvm::json::Object &object(const llvm::json::Object &parent, const char *name);
const llvm::json::Array &array(const llvm::json::Object &parent, const char *name);
std::string string(const llvm::json::Object &parent, const char *name);
int64_t integer(const llvm::json::Object &parent, const char *name);
bool boolean(const llvm::json::Object &parent, const char *name);
std::vector<std::string> strings(const llvm::json::Object &parent, const char *name);
size_t add(size_t a, size_t b, const std::string &what);
size_t multiply(size_t a, size_t b, const std::string &what);
weft_dtype dtype(const std::string &name);

struct Context {
  RISCVNativeTarget native;
  std::string json;
#if defined(__linux__)
  cpu_set_t allowedCPUs;
  std::map<std::string, cpu_set_t> extensionCPUs;
#endif
  Context();
  void requireAvailable(const std::vector<std::string> &extensions) const;
  void check(const std::vector<std::string> &extensions) const;
  void bindThread(const std::vector<std::string> &extensions) const;
private:
  void verifyThread(const std::vector<std::string> &extensions, bool bind) const;
};

struct Argument {
  std::string name, encoding;
  weft_value_kind kind;
  weft_dtype type;
  weft_access access;
  int64_t alias;
  size_t storageBytes, recordElements, interleaveRows, alignment;
  bool strided;
  bool pointer;
  std::vector<std::string> shape;
  std::vector<const char *> shapeNames;
  explicit Argument(const llvm::json::Object &metadata);
  bool writable() const { return access == WEFT_WRITE || access == WEFT_READWRITE; }
};

struct DimensionValues {
  std::vector<size_t> values;
  std::vector<bool> known;
};

struct ShapeTerm {
  bool dynamic;
  size_t value; // Constant extent or ordinal in the runtime dimension vector.
};

struct ArgumentBinding {
  size_t index;
  std::vector<ShapeTerm> shape;
  size_t strideBegin;
};

// Immutable host ABI facts. This is not a compiler program representation:
// all layout, instruction and schedule choices already belong to the artifact.
struct CallContract {
  std::vector<std::string> dimensions;
  std::map<std::string, size_t, std::less<>> dimensionIndices, argumentIndices;
  std::vector<ArgumentBinding> arguments;
  std::vector<size_t> viewIndices, scalarIndices;
  std::vector<std::pair<size_t, size_t>> aliasPairs;
  size_t strideCount = 0, parameterCount = 0;
  CallContract(const std::vector<Argument> &arguments,
               const std::vector<std::string> &dimensions,
               const std::map<std::string, size_t> &bindings);
  DimensionValues bind(const weft_dimension *values, size_t count) const;
  void constrain(size_t argument, const Argument &metadata, const weft_view &view,
                 DimensionValues &values) const;
  std::vector<size_t> shape(size_t argument, const DimensionValues &values) const;
  void complete(const DimensionValues &values) const;
  size_t argumentIndex(const char *name) const;
};

struct Kernel {
  using Dispatch = void (*)(weft_native_entry, void *const *);
  std::shared_ptr<Context> context;
  std::string directory, symbol, march, abi;
  size_t xlen, vlen;
  std::vector<Argument> arguments;
  std::vector<std::string> dimensions, usedExtensions;
  std::vector<const char *> dimensionNames;
  std::map<std::string, size_t> bindings;
  std::unique_ptr<CallContract> contract;
  void *library = nullptr;
  weft_native_entry entry = nullptr;
  Dispatch dispatch = nullptr;
  Kernel(std::shared_ptr<Context> context, const std::string &directory);
  Kernel(std::shared_ptr<Context> context, const std::string &libraryPath,
         const llvm::json::Object &manifest);
  ~Kernel();
  Kernel(const Kernel &) = delete;
  DimensionValues bindDimensions(const weft_dimension *values, size_t count) const;
  size_t storageSize(size_t argument, const DimensionValues &values) const;
};

struct Call {
  std::vector<std::vector<size_t>> pointerShapes, pointerStrides;
  struct Update { size_t index; weft_value value; };
  std::shared_ptr<Kernel> kernel;
  std::vector<weft_value> values;
  std::vector<size_t> dimensions, strides;
  std::vector<void *> parameters;
  std::vector<std::shared_ptr<struct Storage>> storage;
  DimensionValues explicitDimensions;
  Call(std::shared_ptr<Kernel> kernel, const weft_value *values, size_t count,
       const weft_dimension *dimensions, size_t dimensionCount);
  std::shared_ptr<Call> rebind(const weft_named_value *updates, size_t count,
                             const weft_dimension *dimensions, size_t dimensionCount) const;
  Call(const Call &) = delete;
  Call &operator=(const Call &) = delete;
private:
  Call(const Call &source, const std::vector<Update> &scalars);
  void rebuildParameters();
};

struct Storage {
  void *data = nullptr;
  size_t bytes;
  Storage(size_t bytes, size_t alignment);
  ~Storage();
  Storage(const Storage &) = delete;
};

struct BoundValue {
  weft_value value{};
  std::string encoding;
  std::vector<size_t> shape, strides;
  std::shared_ptr<Storage> storage;
  explicit BoundValue(const weft_value &value);
  weft_value descriptor() const;
};

struct StorageSpan {
  size_t bytes = 0;
  std::vector<size_t> strides;
  StorageSpan(const Argument &argument, const size_t *shape, size_t rank,
              const size_t *strides, size_t strideCount);
};

struct Layout {
  std::string encoding;
  std::vector<size_t> shape, strides;
  size_t bytes, alignment;
  weft_access access;
  Layout(const Argument &argument, const size_t *shape, size_t rank,
         const size_t *strides, size_t strideCount);
};

struct Bindings {
  std::shared_ptr<Kernel> kernel;
  std::vector<std::optional<BoundValue>> values;
  std::map<std::string, size_t> dimensions;
  explicit Bindings(std::shared_ptr<Kernel> kernel)
      : kernel(std::move(kernel)), values(this->kernel->arguments.size()) {}
  size_t index(const char *name) const;
  void set(size_t index, BoundValue value);
  std::shared_ptr<Call> prepare() const;
};

struct Sequence {
  struct Guard {
    const Context *context;
    std::vector<std::string> extensions;
  };
  std::vector<std::shared_ptr<Call>> calls;
  std::vector<Guard> guards;
  explicit Sequence(std::vector<std::shared_ptr<Call>> calls);
  void invoke() const;
};

struct Package {
  std::vector<std::string> names;
  std::map<std::string, std::shared_ptr<Kernel>> kernels;
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> bindings;
  Package(std::shared_ptr<Context> context, const std::string &directory);
};

} // namespace weft::runtime

struct weft_context { std::shared_ptr<weft::runtime::Context> value; };
struct weft_kernel { std::shared_ptr<weft::runtime::Kernel> value; };
struct weft_call { std::shared_ptr<weft::runtime::Call> value; };
struct weft_package { std::unique_ptr<weft::runtime::Package> value; };
struct weft_buffer { std::shared_ptr<weft::runtime::Storage> value; };
struct weft_bindings { std::unique_ptr<weft::runtime::Bindings> value; };
struct weft_sequence { std::unique_ptr<weft::runtime::Sequence> value; };
struct weft_layout { std::unique_ptr<weft::runtime::Layout> value; };
#endif
