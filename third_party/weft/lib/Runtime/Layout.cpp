#include "APIBoundary.h"

#include <algorithm>
#include <limits>

namespace weft::runtime {

namespace {
// Successful binding performs arithmetic, not diagnostic string construction.
// All callers share these overflow checks and the same signed-pointer limit.
class StorageArithmetic {
  const std::string &name;
public:
  explicit StorageArithmetic(const std::string &name) : name(name) {}

  [[noreturn]] void invalid(const char *message) const {
    throw Failure(WEFT_INVALID_ARGUMENT, name + " " + message);
  }

  size_t product(size_t a, size_t b, const char *message) const {
    if (b && a > std::numeric_limits<size_t>::max() / b) invalid(message);
    return a * b;
  }

  size_t sum(size_t a, size_t b, const char *message) const {
    if (a > std::numeric_limits<size_t>::max() - b) invalid(message);
    return a + b;
  }
};
} // namespace

StorageSpan::StorageSpan(const Argument &arg, const size_t *extents, size_t rank,
                         const size_t *steps, size_t strideCount) {
  StorageArithmetic arithmetic(arg.name);
  if (arg.kind != WEFT_VIEW || (!arg.pointer && rank != arg.shape.size()) || (!extents && rank))
    arithmetic.invalid("needs a View shape matching its rank");
  if ((!steps && strideCount) || (steps && strideCount != rank))
    arithmetic.invalid("stride count differs from rank");
  size_t elements = 1;
  bool empty = false;
  for (size_t axis = 0; axis < rank; ++axis) {
    size_t extent = extents[axis];
    empty |= extent == 0;
    elements = arithmetic.product(elements, extent, "shape overflows size_t");
  }
  if (rank && extents[rank - 1] % arg.recordElements)
    arithmetic.invalid("requires complete Encoding records");
  if (arg.interleaveRows && (rank != 2 || extents[0] % arg.interleaveRows))
    arithmetic.invalid("requires complete row cohorts");
  strides.resize(rank);
  size_t contiguous = 1;
  for (size_t axis = rank; axis-- > 0;) {
    strides[axis] = steps ? steps[axis] : contiguous;
    if (!arg.strided && !arg.pointer && !empty && extents[axis] > 1 && strides[axis] != contiguous)
      arithmetic.invalid("requires contiguous storage");
    contiguous = arithmetic.product(contiguous, extents[axis], "contiguous stride overflows size_t");
  }
  bytes = arithmetic.product(elements / arg.recordElements, arg.storageBytes, "storage overflows size_t");
  const size_t limit = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
  if (arg.strided || arg.pointer) {
    size_t last = empty ? 0 : 1;
    std::vector<std::pair<size_t, size_t>> axes;
    if (arg.writable() && !arg.pointer && !empty) axes.reserve(rank);
    for (size_t axis = 0; axis < rank; ++axis) {
      if (strides[axis] > limit / arg.storageBytes)
        arithmetic.invalid("byte stride exceeds signed pointer range");
      if (!empty)
        last = arithmetic.sum(last, arithmetic.product(extents[axis] - 1, strides[axis],
                              "stride span overflows size_t"), "span overflows size_t");
      if (arg.writable() && !arg.pointer && !empty && extents[axis] > 1)
        axes.emplace_back(strides[axis], extents[axis]);
    }
    if (arg.writable() && !arg.pointer && !empty) {
      std::sort(axes.begin(), axes.end());
      size_t occupied = 1;
      for (const auto &[stride, extent] : axes) {
        if (stride < occupied) arithmetic.invalid("writable View overlaps itself");
        occupied = arithmetic.sum(occupied, arithmetic.product(extent - 1, stride,
                                   "stride span overflows size_t"), "span overflows size_t");
      }
    }
    bytes = arithmetic.product(last, arg.storageBytes, "span overflows size_t");
  }
  if (bytes > limit) arithmetic.invalid("storage exceeds signed pointer range");
}

Layout::Layout(const Argument &arg, const size_t *extents, size_t rank,
               const size_t *steps, size_t strideCount)
    : encoding(arg.encoding), alignment(arg.alignment), access(arg.access) {
  StorageSpan span(arg, extents, rank, steps, strideCount);
  if (rank) shape.assign(extents, extents + rank);
  bytes = span.bytes;
  strides = std::move(span.strides);
}

} // namespace weft::runtime

using namespace weft::runtime;

extern "C" {
weft_status weft_kernel_layout(const weft_kernel *kernel, size_t index,
                                const weft_dimension *dimensions, size_t dimensionCount,
                                const size_t *strides, size_t strideCount,
                                weft_layout **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(kernel && index < kernel->value->arguments.size(), WEFT_INVALID_ARGUMENT,
            "kernel and valid argument index are required");
    const auto &argument = kernel->value->arguments[index];
    auto values = kernel->value->bindDimensions(dimensions, dimensionCount);
    auto shape = kernel->value->contract->shape(index, values);
    *result = new weft_layout{std::make_unique<Layout>(argument, shape.data(), shape.size(), strides, strideCount)};
  });
}

weft_status weft_layout_describe(const weft_layout *layout, weft_layout_info *result, weft_error *error) {
  return api(error, [&] {
    require(layout && result, WEFT_INVALID_ARGUMENT, "layout and result are required");
    const auto &value = *layout->value;
    *result = {value.encoding.c_str(), value.shape.size(), value.shape.data(), value.strides.data(),
               value.bytes, value.alignment, value.access};
  });
}

weft_status weft_layout_allocate(const weft_layout *layout, weft_buffer **result, weft_error *error) {
  return api(error, [&] {
    output(result);
    require(layout, WEFT_INVALID_ARGUMENT, "layout is null");
    *result = new weft_buffer{std::make_shared<Storage>(layout->value->bytes, layout->value->alignment)};
  });
}

void weft_layout_release(weft_layout *layout) { delete layout; }
}
