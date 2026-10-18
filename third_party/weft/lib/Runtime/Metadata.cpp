#include "Internal.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <limits>
#include <set>

namespace weft::runtime {

void require(bool condition, weft_status status, const std::string &message) {
  if (!condition) throw Failure(status, message);
}

llvm::json::Object readObject(const std::string &path) {
  auto input = llvm::MemoryBuffer::getFile(path);
  require(bool(input), WEFT_IO_ERROR, "cannot read " + path + (input ? "" : ": " + input.getError().message()));
  auto parsed = llvm::json::parse((*input)->getBuffer());
  if (!parsed) throw Failure(WEFT_ABI_ERROR, path + ": " + llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  require(root != nullptr, WEFT_ABI_ERROR, path + ": expected a JSON object");
  return std::move(*root);
}

const llvm::json::Object &object(const llvm::json::Object &parent, const char *name) {
  auto *value = parent.getObject(name);
  require(value != nullptr, WEFT_ABI_ERROR, std::string("missing object: ") + name);
  return *value;
}

const llvm::json::Array &array(const llvm::json::Object &parent, const char *name) {
  auto *value = parent.getArray(name);
  require(value != nullptr, WEFT_ABI_ERROR, std::string("missing array: ") + name);
  return *value;
}

std::string string(const llvm::json::Object &parent, const char *name) {
  auto value = parent.getString(name);
  require(bool(value), WEFT_ABI_ERROR, std::string("missing string: ") + name);
  return value->str();
}

int64_t integer(const llvm::json::Object &parent, const char *name) {
  auto value = parent.getInteger(name);
  require(bool(value), WEFT_ABI_ERROR, std::string("missing integer: ") + name);
  return *value;
}

bool boolean(const llvm::json::Object &parent, const char *name) {
  auto value = parent.getBoolean(name);
  require(bool(value), WEFT_ABI_ERROR, std::string("missing boolean: ") + name);
  return *value;
}

std::vector<std::string> strings(const llvm::json::Object &parent, const char *name) {
  std::vector<std::string> result;
  for (const auto &entry : array(parent, name)) {
    auto value = entry.getAsString();
    require(bool(value), WEFT_ABI_ERROR, std::string(name) + " contains a non-string");
    result.push_back(value->str());
  }
  return result;
}

size_t add(size_t a, size_t b, const std::string &what) {
  require(a <= std::numeric_limits<size_t>::max() - b, WEFT_INVALID_ARGUMENT, what + " overflows size_t");
  return a + b;
}

size_t multiply(size_t a, size_t b, const std::string &what) {
  require(!b || a <= std::numeric_limits<size_t>::max() / b, WEFT_INVALID_ARGUMENT, what + " overflows size_t");
  return a * b;
}

weft_dtype dtype(const std::string &name) {
  static const std::map<std::string, weft_dtype> types = {
      {"", WEFT_DTYPE_NONE}, {"i1", WEFT_I1}, {"i8", WEFT_I8}, {"u8", WEFT_U8},
      {"i16", WEFT_I16}, {"u16", WEFT_U16}, {"i32", WEFT_I32}, {"u32", WEFT_U32},
      {"i64", WEFT_I64}, {"u64", WEFT_U64}, {"f16", WEFT_F16}, {"bf16", WEFT_BF16},
      {"f32", WEFT_F32}, {"f64", WEFT_F64}, {"index", WEFT_INDEX}};
  auto found = types.find(name);
  require(found != types.end(), WEFT_UNSUPPORTED, "unsupported ABI dtype: " + name);
  return found->second;
}

Argument::Argument(const llvm::json::Object &metadata) {
  name = string(metadata, "name");
  encoding = string(metadata, "encoding");
  auto kindName = string(metadata, "kind");
  require(kindName == "view" || kindName == "pointer" || kindName == "scalar", WEFT_ABI_ERROR, "invalid argument kind: " + kindName);
  pointer = kindName == "pointer";
  kind = kindName == "scalar" ? WEFT_SCALAR : WEFT_VIEW;
  type = dtype(string(metadata, "dtype"));
  auto accessName = string(metadata, "access");
  static const std::map<std::string, weft_access> accesses = {
      {"none", WEFT_ACCESS_NONE}, {"read", WEFT_READ}, {"write", WEFT_WRITE}, {"readwrite", WEFT_READWRITE}};
  auto accessValue = accesses.find(accessName);
  require(accessValue != accesses.end(), WEFT_ABI_ERROR, "invalid access for " + name);
  access = accessValue->second;
  alias = integer(metadata, "alias_set");
  strided = boolean(metadata, "strided");
  shape = strings(metadata, "shape");
  require(!pointer || (shape.empty() && !strided && encoding.rfind("dense.", 0) == 0),
          WEFT_ABI_ERROR, "pointer requires dense storage with source-defined offsets");
  auto size = integer(metadata, "storage_bytes"), elements = integer(metadata, "record_elements");
  auto rows = integer(metadata, "interleave_rows"), align = integer(metadata, "alignment");
  require(size >= 0 && elements >= 0 && rows >= 0 && align >= 0, WEFT_ABI_ERROR, "negative storage contract for " + name);
  storageBytes = size; recordElements = elements; interleaveRows = rows; alignment = align;
  if (kind == WEFT_SCALAR) {
    require(type != WEFT_DTYPE_NONE && access == WEFT_ACCESS_NONE && alias == -1 &&
            shape.empty() && !strided, WEFT_ABI_ERROR, "scalar carries a View contract: " + name);
  } else {
    require(storageBytes && recordElements && alignment && (alignment & (alignment - 1)) == 0 &&
            alias >= 0 && !encoding.empty(), WEFT_ABI_ERROR, "incomplete storage contract for " + name);
    require(!strided || (encoding.rfind("dense.", 0) == 0 && recordElements == 1),
            WEFT_ABI_ERROR, "strided ABI requires dense storage: " + name);
  }
}

Package::Package(std::shared_ptr<Context> context, const std::string &directory) {
  auto manifest = readObject(directory + "/package.json");
  require(string(manifest, "kind") == "weft-kernel-package", WEFT_ABI_ERROR, "unsupported package kind");
  for (const auto &entry : array(manifest, "kernels")) {
    auto *value = entry.getAsObject();
    require(value != nullptr, WEFT_ABI_ERROR, "package entry is not an object");
    auto name = string(*value, "name"), path = string(*value, "directory");
    require(!name.empty() && !path.empty() && path.front() != '/' && path.find("..") == std::string::npos,
            WEFT_ABI_ERROR, "package entry must name a relative artifact directory");
    require(kernels.count(name) == 0, WEFT_ABI_ERROR, "duplicate package entry: " + name);
    kernels.emplace(name, std::make_shared<Kernel>(context, directory + "/" + path));
    names.push_back(name);
  }
  require(!names.empty(), WEFT_ABI_ERROR, "package contains no kernels");
  for (const auto &[group, value] : object(manifest, "bindings")) {
    auto *mapping = value.getAsObject();
    require(mapping && !group.str().empty(), WEFT_ABI_ERROR, "invalid framework binding group");
    auto &entries = bindings[group.str()];
    for (const auto &[role, target] : *mapping) {
      auto name = target.getAsString();
      require(name && !role.str().empty() && kernels.count(name->str()), WEFT_ABI_ERROR,
              "framework binding names an absent package entry");
      entries.emplace_back(role.str(), name->str());
    }
  }
}

} // namespace weft::runtime
