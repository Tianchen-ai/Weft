#include "weft/runtime.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {
using WorkerCheck = int (*)(const void *, char *, size_t);
using GridLaunch = int (*)(weft_native_entry, void *const *, size_t, size_t,
                          size_t, int, const void *, WorkerCheck, char *, size_t);

[[noreturn]] void overflow(const std::string &message) {
  PyErr_SetString(PyExc_OverflowError, message.c_str());
  throw py::error_already_set();
}

void check(weft_status status, const weft_error &error) {
  if (status == WEFT_OK) return;
  if (status == WEFT_INVALID_ARGUMENT) throw py::value_error(error.message);
  throw std::runtime_error(error.message);
}

size_t checkedAdd(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw py::value_error("View byte span overflows the native ABI");
  return left + right;
}

size_t checkedMultiply(size_t left, size_t right) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw py::value_error("View byte span overflows the native ABI");
  return left * right;
}

size_t index(py::handle value, const char *what) {
  if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
    throw py::type_error(std::string(what) + " must be an integer");
  size_t result = PyLong_AsSize_t(value.ptr());
  if (PyErr_Occurred()) throw py::error_already_set();
  return result;
}

std::vector<size_t> indices(py::handle value, const char *what) {
  std::vector<size_t> result;
  for (py::handle item : py::reinterpret_borrow<py::iterable>(value))
    result.push_back(index(item, what));
  return result;
}

const std::unordered_map<std::string, std::string> &torchEncodings() {
  static const std::unordered_map<std::string, std::string> encodings = {
      {"torch.int8", "dense.i8"}, {"torch.uint8", "dense.u8"},
      {"torch.int16", "dense.i16"}, {"torch.uint16", "dense.u16"},
      {"torch.int32", "dense.i32"}, {"torch.uint32", "dense.u32"},
      {"torch.int64", "dense.i64"}, {"torch.uint64", "dense.u64"},
      {"torch.float16", "dense.f16"}, {"torch.bfloat16", "dense.bf16"},
      {"torch.float32", "dense.f32"}, {"torch.float64", "dense.f64"},
      {"torch.bool", "dense.i1"}};
  return encodings;
}

std::string bufferEncoding(const Py_buffer &view) {
  const char *format = view.format ? view.format : "B";
  uint16_t one = 1;
  bool little = *reinterpret_cast<unsigned char *>(&one) != 0;
  if (view.itemsize > 1 &&
      (((format[0] == '>' || format[0] == '!') && little) ||
       (format[0] == '<' && !little)))
    throw py::type_error("non-native-endian buffers require explicit conversion");
  while (*format && std::strchr("@=<>!", *format)) ++format;
  if (std::strlen(format) != 1)
    throw py::type_error("buffer format needs an explicit Encoding");
  std::string dtype;
  switch (*format) {
  case '?': dtype = "i1"; break;
  case 'b': dtype = "i8"; break;
  case 'B': dtype = "u8"; break;
  case 'h': dtype = "i16"; break;
  case 'H': dtype = "u16"; break;
  case 'i': dtype = "i32"; break;
  case 'I': dtype = "u32"; break;
  case 'q': dtype = "i64"; break;
  case 'Q': dtype = "u64"; break;
  case 'e': dtype = "f16"; break;
  case 'f': dtype = "f32"; break;
  case 'd': dtype = "f64"; break;
  case 'l': case 'L':
    if (view.itemsize != 4 && view.itemsize != 8)
      throw py::type_error("unsupported native long buffer format");
    dtype = std::string(*format == 'l' ? "i" : "u") +
            std::to_string(view.itemsize * 8);
    break;
  default: throw py::type_error("buffer format needs an explicit Encoding");
  }
  return "dense." + dtype;
}

weft_dtype scalarType(const std::string &type) {
  static const std::unordered_map<std::string, weft_dtype> types = {
      {"i1", WEFT_I1}, {"i8", WEFT_I8}, {"u8", WEFT_U8},
      {"i16", WEFT_I16}, {"u16", WEFT_U16}, {"i32", WEFT_I32},
      {"u32", WEFT_U32}, {"i64", WEFT_I64}, {"u64", WEFT_U64},
      {"index", WEFT_INDEX}, {"f32", WEFT_F32}, {"f64", WEFT_F64}};
  auto found = types.find(type);
  if (found == types.end())
    throw py::type_error("by-value ABI does not support " + type);
  return found->second;
}

struct Parameter {
  std::string name;
  std::string encoding;
  weft_dtype dtype = WEFT_DTYPE_NONE;
  size_t sourceIndex = 0;
  bool view = false;
  bool writable = false;
  int gridAxis = -1;
  bool programID = false;
};

struct BorrowedView {
  Py_buffer exportView{};
  py::object storage;
  py::object tensor;
  py::object writeHook;
  std::string encoding;
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  void *data = nullptr;
  size_t bytes = 0;
  bool readonly = false;

  BorrowedView() = default;
  BorrowedView(const BorrowedView &) = delete;
  ~BorrowedView() {
    if (exportView.obj) PyBuffer_Release(&exportView);
  }

  weft_view descriptor() const {
    return {data, bytes, encoding.c_str(), shape.size(), shape.data(),
            strides.data(), readonly};
  }
};

struct NativeAPI {
  decltype(&weft_call_prepare) prepare;
  decltype(&weft_call_arguments) arguments;
  decltype(&weft_call_release) release;
  GridLaunch grid;
  WorkerCheck checkWorker;
};

struct CallOwner {
  weft_call *call = nullptr;
  decltype(&weft_call_release) release;
  explicit CallOwner(decltype(release) function) : release(function) {}
  ~CallOwner() { if (call) release(call); }
};

class NativeLauncher {
  // The module owns both dlopen handles, the SDK kernel and its target context.
  // No Tensor, pointer packet or mutable call state is cached on the launcher.
  py::object module;
  py::object bufferType;
  py::object integralType;
  py::object realType;
  NativeAPI api;
  weft_kernel *kernel;
  weft_native_entry entry;
  size_t sourceCount;
  int threads;
  std::vector<Parameter> parameters;

  weft_value scalar(const Parameter &parameter, py::handle input) const {
    weft_value result{};
    result.kind = WEFT_SCALAR;
    result.dtype = parameter.dtype;
    if (parameter.dtype == WEFT_F32 || parameter.dtype == WEFT_F64) {
      if (!py::isinstance(input, realType))
        throw py::type_error(parameter.name + " requires a real scalar");
      double value = PyFloat_AsDouble(input.ptr());
      if (PyErr_Occurred()) throw py::error_already_set();
      if (parameter.dtype == WEFT_F64) result.scalar.f64 = value;
      else {
        result.scalar.f32 = static_cast<float>(value);
        if (std::isfinite(value) && !std::isfinite(result.scalar.f32))
          overflow(parameter.name + " cannot be represented by f32");
      }
      return result;
    }
    if (!py::isinstance(input, integralType))
      throw py::type_error(parameter.name + " requires an integer scalar");
    py::object integer = py::reinterpret_steal<py::object>(PyNumber_Index(input.ptr()));
    if (!integer) throw py::error_already_set();
    bool isUnsigned = parameter.dtype == WEFT_I1 || parameter.dtype == WEFT_U8 ||
                      parameter.dtype == WEFT_U16 || parameter.dtype == WEFT_U32 ||
                      parameter.dtype == WEFT_U64 || parameter.dtype == WEFT_INDEX;
    if (isUnsigned) {
      auto value = PyLong_AsUnsignedLongLong(integer.ptr());
      if (PyErr_Occurred()) throw py::error_already_set();
      unsigned bits = parameter.dtype == WEFT_I1 ? 1 : parameter.dtype == WEFT_U8 ? 8 :
                      parameter.dtype == WEFT_U16 ? 16 : parameter.dtype == WEFT_U32 ? 32 : 64;
      if (bits < 64 && value >= (uint64_t(1) << bits))
        overflow(parameter.name + " exceeds its unsigned scalar ABI");
      switch (parameter.dtype) {
      case WEFT_I1: result.scalar.i1 = value; break;
      case WEFT_U8: result.scalar.u8 = value; break;
      case WEFT_U16: result.scalar.u16 = value; break;
      case WEFT_U32: result.scalar.u32 = value; break;
      case WEFT_U64: result.scalar.u64 = value; break;
      case WEFT_INDEX: result.scalar.index = value; break;
      default: throw std::logic_error("invalid unsigned scalar contract");
      }
    } else {
      auto value = PyLong_AsLongLong(integer.ptr());
      if (PyErr_Occurred()) throw py::error_already_set();
      unsigned bits = parameter.dtype == WEFT_I8 ? 8 : parameter.dtype == WEFT_I16 ? 16 :
                      parameter.dtype == WEFT_I32 ? 32 : 64;
      if (bits < 64 && (value < -(int64_t(1) << (bits - 1)) || value >= (int64_t(1) << (bits - 1))))
        overflow(parameter.name + " exceeds its signed scalar ABI");
      switch (parameter.dtype) {
      case WEFT_I8: result.scalar.i8 = value; break;
      case WEFT_I16: result.scalar.i16 = value; break;
      case WEFT_I32: result.scalar.i32 = value; break;
      case WEFT_I64: result.scalar.i64 = value; break;
      default: throw std::logic_error("invalid signed scalar contract");
      }
    }
    return result;
  }

  void borrowExplicit(BorrowedView &view, const Parameter &parameter, py::handle value) const {
    value.attr("validate")(py::arg("writable") = parameter.writable);
    view.data = reinterpret_cast<void *>(index(value.attr("address"), "buffer address"));
    view.bytes = index(value.attr("nbytes"), "buffer capacity");
    view.encoding = py::cast<std::string>(value.attr("encoding"));
    view.shape = indices(value.attr("shape"), "View extent");
    view.strides = indices(value.attr("strides"), "View stride");
    view.readonly = py::cast<bool>(value.attr("readonly"));
    if (parameter.writable) view.writeHook = value.attr("mark_written");
  }

  void borrowTensor(BorrowedView &view, const Parameter &parameter,
                    py::handle value, py::handle torch, bool grad, bool inference) const {
    if (py::cast<std::string>(value.attr("device").attr("type")) != "cpu" ||
        !value.attr("layout").is(torch.attr("strided")))
      throw py::type_error("Weft requires strided CPU Tensors");
    if (py::cast<bool>(value.attr("is_conj")()) || py::cast<bool>(value.attr("is_neg")()))
      throw py::value_error("Tensor conjugate/negative views require explicit materialization");
    std::string dtype = py::str(value.attr("dtype"));
    auto found = torchEncodings().find(dtype);
    if (found == torchEncodings().end()) throw py::type_error("unsupported Tensor storage dtype: " + dtype);
    view.encoding = found->second;
    view.shape = indices(value.attr("shape"), "Tensor extent");
    view.strides = indices(value.attr("stride")(), "Tensor stride");
    view.storage = value.attr("untyped_storage")();
    size_t capacity = index(view.storage.attr("nbytes")(), "Tensor storage capacity");
    size_t offset = checkedMultiply(index(value.attr("storage_offset")(), "Tensor offset"),
                                    index(value.attr("element_size")(), "Tensor element size"));
    if (offset > capacity) throw py::value_error("Tensor offset exceeds its backing storage");
    view.bytes = capacity - offset;
    view.data = reinterpret_cast<void *>(index(value.attr("data_ptr")(), "Tensor address"));
    if (grad && py::cast<bool>(value.attr("requires_grad"))) {
      PyErr_SetString(PyExc_NotImplementedError,
                     "raw Tensor calls do not define autograd; use an explicit backward or torch.no_grad()");
      throw py::error_already_set();
    }
    if (parameter.writable && !inference && py::cast<bool>(value.attr("is_inference")()))
      throw std::runtime_error("an inference Tensor can only be written inside torch.inference_mode()");
    if (parameter.writable) view.tensor = py::reinterpret_borrow<py::object>(value);
  }

  void borrowBuffer(BorrowedView &view, py::handle value) const {
    if (PyObject_GetBuffer(value.ptr(), &view.exportView, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
      throw py::error_already_set();
    const auto &buffer = view.exportView;
    if (buffer.ndim < 0 || buffer.itemsize <= 0 || buffer.len < 0 ||
        (buffer.ndim && (!buffer.shape || !buffer.strides)))
      throw py::value_error("invalid exported buffer metadata");
    view.encoding = bufferEncoding(buffer);
    bool empty = false;
    size_t last = static_cast<size_t>(buffer.itemsize);
    for (int axis = 0; axis < buffer.ndim; ++axis) {
      if (buffer.suboffsets && buffer.suboffsets[axis] >= 0)
        throw py::type_error("indirect buffer storage is unsupported");
      if (buffer.shape[axis] < 0 || buffer.strides[axis] < 0 || buffer.strides[axis] % buffer.itemsize)
        throw py::value_error("buffer extents and whole-element strides must be nonnegative");
      size_t extent = buffer.shape[axis];
      view.shape.push_back(extent);
      view.strides.push_back(buffer.strides[axis] / buffer.itemsize);
      empty |= extent == 0;
      if (extent) last = checkedAdd(last, checkedMultiply(extent - 1, buffer.strides[axis]));
    }
    view.bytes = empty ? 0 : last;
    view.data = buffer.buf;
    view.readonly = buffer.readonly;
  }

public:
  NativeLauncher(py::object module, py::object metadata, const py::dict &functions)
      : module(std::move(module)) {
    auto address = [&](const char *name) { return py::cast<uintptr_t>(functions[name]); };
    api = {reinterpret_cast<decltype(api.prepare)>(address("prepare")),
           reinterpret_cast<decltype(api.arguments)>(address("arguments")),
           reinterpret_cast<decltype(api.release)>(address("release")),
           reinterpret_cast<GridLaunch>(address("grid")),
           reinterpret_cast<WorkerCheck>(address("check_worker"))};
    kernel = reinterpret_cast<weft_kernel *>(address("kernel"));
    entry = reinterpret_cast<weft_native_entry>(address("entry"));
    if (!kernel || !entry || !api.prepare || !api.arguments || !api.release || !api.grid || !api.checkWorker)
      throw py::value_error("native launcher needs a complete SDK and kernel binding");
    threads = py::cast<int>(metadata.attr("num_cpu_threads"));
    if (threads < 0) throw py::value_error("num_cpu_threads must be nonnegative");
    bufferType = py::module_::import("triton.runtime.weft.buffer").attr("Buffer");
    auto numbers = py::module_::import("numbers");
    integralType = numbers.attr("Integral");
    realType = numbers.attr("Real");
    auto names = py::cast<std::vector<std::string>>(metadata.attr("launch_args"));
    sourceCount = names.size();
    std::unordered_map<std::string, size_t> sourceIndices;
    for (size_t i = 0; i < names.size(); ++i) sourceIndices.emplace(names[i], i);
    auto abi = py::cast<py::dict>(metadata.attr("kernel_abi"));
    for (auto item : py::cast<py::list>(abi["arguments"])) {
      auto description = py::cast<py::dict>(item);
      Parameter parameter;
      parameter.name = py::cast<std::string>(description["name"]);
      const auto kind = py::cast<std::string>(description["kind"]);
      parameter.view = kind == "view" || kind == "pointer";
      parameter.encoding = py::cast<std::string>(description["encoding"]);
      auto access = py::cast<std::string>(description["access"]);
      parameter.writable = access == "write" || access == "readwrite";
      if (!parameter.view) parameter.dtype = scalarType(py::cast<std::string>(description["dtype"]));
      for (int axis = 0; axis < 3; ++axis) {
        if (parameter.name == "__triton_pid_" + std::to_string(axis)) {
          parameter.gridAxis = axis;
          parameter.programID = true;
        }
        if (parameter.name == "__triton_grid_" + std::to_string(axis)) parameter.gridAxis = axis;
      }
      if (parameter.gridAxis < 0) {
        auto found = sourceIndices.find(parameter.name);
        if (found == sourceIndices.end()) throw py::value_error("native ABI argument is absent from the JIT signature");
        parameter.sourceIndex = found->second;
      }
      parameters.push_back(std::move(parameter));
    }
  }

  void launch(const py::args &args) const {
    constexpr size_t prefix = 9; // CompiledKernel's shared launcher protocol.
    if (args.size() != prefix + sourceCount)
      throw py::type_error("launch arguments do not match the compiled Triton signature");
    std::array<size_t, 3> grid;
    for (size_t axis = 0; axis < 3; ++axis) {
      if (!PyLong_CheckExact(args[axis].ptr()))
        throw py::value_error("grid dimensions must be nonnegative native index integers");
      grid[axis] = index(args[axis], "grid dimension");
      if (grid[axis] > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        throw py::value_error("grid dimension exceeds the supported launch domain");
    }
    py::object torch;
    auto *modules = PyImport_GetModuleDict();
    if (auto *found = PyDict_GetItemString(modules, "torch"))
      torch = py::reinterpret_borrow<py::object>(found);
    py::object tensorType;
    bool grad = false, inference = false;
    if (torch) {
      tensorType = torch.attr("Tensor");
      grad = py::cast<bool>(torch.attr("is_grad_enabled")());
      inference = py::cast<bool>(torch.attr("is_inference_mode_enabled")());
    }
    std::vector<weft_value> values(parameters.size());
    std::vector<std::unique_ptr<BorrowedView>> borrows;
    borrows.reserve(parameters.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      const auto &parameter = parameters[i];
      if (parameter.gridAxis >= 0) {
        values[i].kind = WEFT_SCALAR;
        values[i].dtype = WEFT_INDEX;
        values[i].scalar.index = parameter.programID ? 0 : grid[parameter.gridAxis];
        continue;
      }
      py::handle value = args[prefix + parameter.sourceIndex];
      if (!parameter.view) { values[i] = scalar(parameter, value); continue; }
      auto view = std::make_unique<BorrowedView>();
      if (py::isinstance(value, bufferType)) borrowExplicit(*view, parameter, value);
      else if (torch && py::isinstance(value, tensorType)) borrowTensor(*view, parameter, value, torch, grad, inference);
      else borrowBuffer(*view, value);
      if (view->encoding != parameter.encoding)
        throw py::type_error(parameter.name + " requires " + parameter.encoding + ", got " + view->encoding);
      if (parameter.writable && view->readonly)
        throw py::type_error(parameter.name + " requires writable storage");
      values[i].kind = WEFT_VIEW;
      values[i].view = view->descriptor();
      borrows.push_back(std::move(view));
    }
    CallOwner call(api.release);
    weft_error error{};
    check(api.prepare(kernel, values.data(), values.size(), nullptr, 0, &call.call, &error), error);
    void *const *packet = nullptr;
    size_t packetSize = 0;
    check(api.arguments(call.call, &packet, &packetSize, &error), error);
    auto enter = args[7], exit = args[8], metadata = args[6];
    if (!enter.is_none()) enter(metadata);
    try {
      int status;
      {
        py::gil_scoped_release release;
        status = api.grid(entry, packet, grid[0], grid[1], grid[2], threads,
                          kernel, api.checkWorker, error.message, sizeof(error.message));
      }
      if (status) throw std::runtime_error(error.message[0] ? error.message : "native grid exceeds the supported launch domain");
      if (grid[0] && grid[1] && grid[2]) {
        py::object increment;
        for (const auto &view : borrows) {
          if (view->writeHook) view->writeHook();
          if (view->tensor) {
            if (!increment) increment = torch.attr("autograd").attr("graph").attr("increment_version");
            increment(view->tensor);
          }
        }
      }
    } catch (...) {
      if (!exit.is_none()) exit(metadata);
      throw;
    }
    if (!exit.is_none()) exit(metadata);
  }
};
} // namespace

void init_weft_runtime(py::module_ &m) {
  py::class_<NativeLauncher>(m, "NativeLauncher")
      .def(py::init<py::object, py::object, const py::dict &>())
      .def("__call__", &NativeLauncher::launch);
}
