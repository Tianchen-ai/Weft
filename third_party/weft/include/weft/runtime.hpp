#ifndef WEFT_RUNTIME_HPP
#define WEFT_RUNTIME_HPP

#include "runtime.h"
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace weft {

class error : public std::runtime_error {
public:
  weft_status status;
  explicit error(const weft_error &value) : std::runtime_error(value.message), status(value.status) {}
};
inline void check(weft_status status, const weft_error &detail) {
  if (status != WEFT_OK) throw error(detail);
}

template <typename T, void (*Release)(T *)> class owner {
protected:
  T *value_ = nullptr;
  explicit owner(T *value) : value_(value) {}
  owner() = default;
public:
  owner(const owner &) = delete;
  owner &operator=(const owner &) = delete;
  owner(owner &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
  owner &operator=(owner &&other) noexcept {
    if (this != &other) { Release(value_); value_ = std::exchange(other.value_, nullptr); }
    return *this;
  }
  ~owner() { Release(value_); }
  T *get() const { return value_; }
};

class context : public owner<weft_context, weft_context_release> {
public:
  context() {
    weft_error detail{};
    check(weft_context_create(&value_, &detail), detail);
  }
  std::string target_json() const {
    weft_error detail{};
    const char *text;
    check(weft_context_target_json(value_, &text, &detail), detail);
    return text;
  }
};

class view {
  void *data_;
  size_t bytes_;
  std::string encoding_;
  std::vector<size_t> shape_, strides_;
  bool readonly_;
public:
  view(void *data, size_t bytes, std::string encoding, std::vector<size_t> shape,
       std::vector<size_t> strides = {}, bool readonly = false)
      : data_(data), bytes_(bytes), encoding_(std::move(encoding)), shape_(std::move(shape)),
        strides_(std::move(strides)), readonly_(readonly) {
    if (!strides_.empty() && strides_.size() != shape_.size())
      throw std::invalid_argument("view stride count differs from its rank");
  }
  weft_value argument() const {
    weft_value value{};
    value.kind = WEFT_VIEW;
    value.view = {data_, bytes_, encoding_.c_str(), shape_.size(), shape_.data(),
                  strides_.empty() ? nullptr : strides_.data(), readonly_};
    return value;
  }
};

inline weft_value scalar(float value) {
  weft_value result{}; result.kind = WEFT_SCALAR; result.dtype = WEFT_F32; result.scalar.f32 = value; return result;
}
inline weft_value scalar(double value) {
  weft_value result{}; result.kind = WEFT_SCALAR; result.dtype = WEFT_F64; result.scalar.f64 = value; return result;
}
inline weft_value scalar(int32_t value) {
  weft_value result{}; result.kind = WEFT_SCALAR; result.dtype = WEFT_I32; result.scalar.i32 = value; return result;
}
inline weft_value scalar(int64_t value) {
  weft_value result{}; result.kind = WEFT_SCALAR; result.dtype = WEFT_I64; result.scalar.i64 = value; return result;
}
inline weft_value index(size_t value) {
  weft_value result{}; result.kind = WEFT_SCALAR; result.dtype = WEFT_INDEX; result.scalar.index = value; return result;
}

class buffer : public owner<weft_buffer, weft_buffer_release> {
  friend class layout;
  explicit buffer(weft_buffer *value) : owner(value) {}
public:
  buffer(size_t bytes, size_t alignment) {
    weft_error detail{};
    check(weft_buffer_allocate(bytes, alignment, &value_, &detail), detail);
  }
  void *data() const {
    weft_error detail{}; void *pointer; size_t bytes;
    check(weft_buffer_data(value_, &pointer, &bytes, &detail), detail);
    return pointer;
  }
  size_t size() const {
    weft_error detail{}; void *pointer; size_t bytes;
    check(weft_buffer_data(value_, &pointer, &bytes, &detail), detail);
    return bytes;
  }
  view as_view(std::string encoding, std::vector<size_t> shape,
               std::vector<size_t> strides = {}, bool readonly = false) const {
    return view(data(), size(), std::move(encoding), std::move(shape), std::move(strides), readonly);
  }
};

class call : public owner<weft_call, weft_call_release> {
  friend class kernel;
  friend class bindings;
  explicit call(weft_call *value) : owner(value) {}
public:
  void operator()() const {
    weft_error detail{};
    check(weft_call_invoke(value_, &detail), detail);
  }
  call rebind(std::initializer_list<weft_named_value> updates,
              std::initializer_list<weft_dimension> dimensions = {}) const {
    weft_error detail{};
    weft_call *result;
    check(weft_call_rebind(value_, updates.begin(), updates.size(), dimensions.begin(),
                          dimensions.size(), &result, &detail), detail);
    return call(result);
  }
};

class bindings;
class layout;

class kernel : public owner<weft_kernel, weft_kernel_release> {
  friend class package;
  explicit kernel(weft_kernel *value) : owner(value) {}
public:
  kernel(context &device, const std::string &directory) {
    weft_error detail{};
    check(weft_kernel_open(device.get(), directory.c_str(), &value_, &detail), detail);
  }
  weft_kernel_info info() const {
    weft_error detail{}; weft_kernel_info result{};
    check(weft_kernel_describe(value_, &result, &detail), detail);
    return result;
  }
  weft_argument_info argument(size_t index) const {
    weft_error detail{}; weft_argument_info result{};
    check(weft_kernel_argument(value_, index, &result, &detail), detail);
    return result;
  }
  call prepare(std::initializer_list<weft_value> values,
               std::initializer_list<weft_dimension> dimensions = {}) const {
    weft_error detail{}; weft_call *result;
    check(weft_call_prepare(value_, values.begin(), values.size(), dimensions.begin(), dimensions.size(), &result, &detail), detail);
    return call(result);
  }
  size_t storage_size(size_t argument, std::initializer_list<weft_dimension> dimensions) const {
    weft_error detail{}; size_t result;
    check(weft_kernel_storage_size(value_, argument, dimensions.begin(), dimensions.size(), &result, &detail), detail);
    return result;
  }
  void check_thread() const {
    weft_error detail{};
    check(weft_kernel_check_thread(value_, &detail), detail);
  }
  void bind_thread() const {
    weft_error detail{};
    check(weft_kernel_bind_thread(value_, &detail), detail);
  }
  weft_native_entry entry() const {
    weft_error detail{}; weft_native_entry result;
    check(weft_kernel_get_entry(value_, &result, &detail), detail);
    return result;
  }
  bindings bind() const;
  layout storage_layout(size_t argument, std::initializer_list<weft_dimension> dimensions,
                        std::initializer_list<size_t> strides = {}) const;
};

class layout : public owner<weft_layout, weft_layout_release> {
  friend class kernel;
  layout(const kernel &source, size_t argument, std::initializer_list<weft_dimension> dimensions,
         std::initializer_list<size_t> strides) {
    weft_error detail{};
    check(weft_kernel_layout(source.get(), argument, dimensions.begin(), dimensions.size(),
                             strides.size() ? strides.begin() : nullptr, strides.size(), &value_, &detail), detail);
  }
public:
  weft_layout_info info() const {
    weft_error detail{};
    weft_layout_info result{};
    check(weft_layout_describe(value_, &result, &detail), detail);
    return result;
  }
  buffer allocate() const {
    weft_error detail{};
    weft_buffer *result;
    check(weft_layout_allocate(value_, &result, &detail), detail);
    return buffer(result);
  }
  view borrow(void *data, size_t capacity, bool readonly = false) const {
    auto contract = info();
    if (capacity < contract.byte_length)
      throw std::invalid_argument("allocation is smaller than the resolved View span");
    std::vector<size_t> shape, strides;
    if (contract.rank) {
      shape.assign(contract.shape, contract.shape + contract.rank);
      strides.assign(contract.strides, contract.strides + contract.rank);
    }
    return view(data, capacity, contract.encoding, std::move(shape), std::move(strides), readonly);
  }
};

inline layout kernel::storage_layout(size_t argument, std::initializer_list<weft_dimension> dimensions,
                                     std::initializer_list<size_t> strides) const {
  return layout(*this, argument, dimensions, strides);
}

class bindings : public owner<weft_bindings, weft_bindings_release> {
public:
  explicit bindings(const kernel &source) {
    weft_error detail{};
    check(weft_bindings_create(source.get(), &value_, &detail), detail);
  }
  bindings &set(const std::string &name, const weft_value &value) {
    weft_error detail{};
    check(weft_bindings_set(value_, name.c_str(), &value, &detail), detail);
    return *this;
  }
  bindings &set(const std::string &name, const view &value) {
    return set(name, value.argument());
  }
  bindings &dimension(const std::string &name, size_t value) {
    weft_error detail{};
    check(weft_bindings_dimension(value_, name.c_str(), value, &detail), detail);
    return *this;
  }
  bindings &retain(const std::string &name, const buffer &storage,
                   std::initializer_list<size_t> shape,
                   std::initializer_list<size_t> strides = {},
                   size_t byte_offset = 0, bool readonly = false) {
    if (strides.size() && strides.size() != shape.size())
      throw std::invalid_argument("buffer stride count differs from its rank");
    weft_error detail{};
    check(weft_bindings_buffer(value_, name.c_str(), storage.get(), byte_offset,
                               shape.size(), shape.begin(), strides.size() ? strides.begin() : nullptr,
                               readonly, &detail), detail);
    return *this;
  }
  bindings &retain(const std::string &name, const buffer &storage, const layout &layout,
                   size_t byte_offset = 0, bool readonly = false) {
    auto contract = layout.info();
    weft_error detail{};
    check(weft_bindings_buffer(value_, name.c_str(), storage.get(), byte_offset,
                               contract.rank, contract.shape, contract.strides, readonly, &detail), detail);
    return *this;
  }
  call prepare() const {
    weft_error detail{};
    weft_call *result;
    check(weft_bindings_prepare(value_, &result, &detail), detail);
    return call(result);
  }
};

inline bindings kernel::bind() const { return bindings(*this); }

class sequence : public owner<weft_sequence, weft_sequence_release> {
  explicit sequence(weft_sequence *value) : owner(value) {}
public:
  explicit sequence(std::initializer_list<const call *> calls) {
    std::vector<weft_call *> handles;
    for (const auto *call : calls) {
      if (!call) throw std::invalid_argument("sequence contains a null call");
      handles.push_back(call->get());
    }
    weft_error detail{};
    check(weft_sequence_create(handles.data(), handles.size(), &value_, &detail), detail);
  }
  void operator()() const {
    weft_error detail{};
    check(weft_sequence_invoke(value_, &detail), detail);
  }
  sequence replace(size_t index, const call &replacement) const {
    weft_error detail{};
    weft_sequence *result;
    check(weft_sequence_replace(value_, index, replacement.get(), &result, &detail), detail);
    return sequence(result);
  }
  sequence rebind(std::initializer_list<weft_sequence_update> updates) const {
    weft_error detail{};
    weft_sequence *result;
    check(weft_sequence_rebind(value_, updates.begin(), updates.size(), &result, &detail), detail);
    return sequence(result);
  }
  sequence rebind(size_t index, std::initializer_list<weft_named_value> values,
                  std::initializer_list<weft_dimension> dimensions = {}) const {
    return rebind({{index, values.begin(), values.size(), dimensions.begin(), dimensions.size()}});
  }
};

class package : public owner<weft_package, weft_package_release> {
public:
  package(context &device, const std::string &directory) {
    weft_error detail{};
    check(weft_package_open(device.get(), directory.c_str(), &value_, &detail), detail);
  }
  std::vector<std::string> names() const {
    weft_error detail{}; size_t count;
    check(weft_package_count(value_, &count, &detail), detail);
    std::vector<std::string> result;
    for (size_t index = 0; index < count; ++index) {
      const char *name;
      check(weft_package_name(value_, index, &name, &detail), detail);
      result.emplace_back(name);
    }
    return result;
  }
  kernel get_kernel(const std::string &name) const {
    weft_error detail{}; weft_kernel *result;
    check(weft_package_kernel(value_, name.c_str(), &result, &detail), detail);
    return kernel(result);
  }
  bool has_binding_group(const std::string &group) const {
    weft_error detail{}; bool present;
    check(weft_package_has_binding_group(value_, group.c_str(), &present, &detail), detail);
    return present;
  }
  std::vector<std::pair<std::string, std::string>> bindings(const std::string &group) const {
    weft_error detail{}; size_t count;
    check(weft_package_binding_count(value_, group.c_str(), &count, &detail), detail);
    std::vector<std::pair<std::string, std::string>> result;
    for (size_t index = 0; index < count; ++index) {
      const char *role, *name;
      check(weft_package_binding(value_, group.c_str(), index, &role, &name, &detail), detail);
      result.emplace_back(role, name);
    }
    return result;
  }
};

} // namespace weft
#endif
