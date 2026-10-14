#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
constexpr std::size_t kChunkElements = 1U << 20;
volatile std::uint64_t flushSink = 0;

struct built_case {
  ggml_tensor *a = nullptr;
  ggml_tensor *b = nullptr;
  ggml_tensor *c = nullptr;
  ggml_tensor *d = nullptr;
  ggml_tensor *e = nullptr;
  ggml_tensor *f = nullptr;
  ggml_tensor *g = nullptr;
  ggml_tensor *output = nullptr;
  const char *model = nullptr;
  const char *modelShape = nullptr;
  const char *implementation = nullptr;
  const char *comparison = nullptr;
  const char *metric = nullptr;
  double work = 0.0;
  double metricDivisor = 1.0e3;
};

std::size_t parseRepetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0)
    return 0;
  return static_cast<std::size_t>(value);
}

void evictCache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    flushSink += buffer[index];
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U)
    return values[middle];
  return 0.5 * (values[middle - 1U] + values[middle]);
}

template <typename Generator>
void setF32(ggml_tensor *tensor, Generator generator) {
  const std::size_t elements =
      static_cast<std::size_t>(ggml_nelements(tensor));
  std::vector<float> values(std::min(elements, kChunkElements));
  for (std::size_t begin = 0; begin < elements; begin += values.size()) {
    const std::size_t count = std::min(values.size(), elements - begin);
    for (std::size_t index = 0; index < count; ++index)
      values[index] = generator(begin + index);
    ggml_backend_tensor_set(tensor, values.data(), begin * sizeof(float),
                            count * sizeof(float));
  }
}

template <typename Generator>
void setI32(ggml_tensor *tensor, Generator generator) {
  const std::size_t elements =
      static_cast<std::size_t>(ggml_nelements(tensor));
  std::vector<std::int32_t> values(elements);
  for (std::size_t index = 0; index < elements; ++index)
    values[index] = generator(index);
  ggml_backend_tensor_set(tensor, values.data(), 0,
                          values.size() * sizeof(std::int32_t));
}

float centered(std::size_t index, std::size_t modulus, float scale) {
  return static_cast<float>(static_cast<int>(index % modulus) -
                            static_cast<int>(modulus / 2U)) /
         scale;
}

built_case build(ggml_context *ctx, const char *name) {
  built_case result;
  if (std::strcmp(name, "top_k") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
    result.output = ggml_top_k(ctx, result.a, 8);
    result.model = "MoE-router";
    result.modelShape = "tokens=512,experts=256,k=8";
    result.implementation = "ggml_cpu_partial_sort";
    result.comparison = "same_top_k_set_different_order_contract";
    return result;
  }
  if (std::strcmp(name, "ssm_conv") == 0) {
    result.a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 131, 2048, 1);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2048);
    result.output = ggml_ssm_conv(ctx, result.a, result.b);
    result.model = "Mamba-short-convolution";
    result.modelShape = "sequences=1,tokens=128,channels=2048,taps=4";
    result.implementation = "ggml_cpu_ssm_conv";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gop_s";
    result.work = 2.0 * 128.0 * 2048.0 * 4.0;
    return result;
  }
  if (std::strcmp(name, "ssm_scan") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 64, 32, 1);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 32, 128, 1);
    result.c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 128, 1);
    result.d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 32);
    result.e = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 8, 128, 1);
    result.f = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 8, 128, 1);
    result.g = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    result.output = ggml_ssm_scan(ctx, result.a, result.b, result.c, result.d,
                                  result.e, result.f, result.g);
    result.model = "Mamba2";
    result.modelShape =
        "sequences=1,tokens=128,heads=32,dim=64,state=16,groups=8";
    result.implementation = "ggml_cpu_ssm_scan";
    result.comparison = "same_math_and_layout_different_output_abi";
    return result;
  }
  if (std::strcmp(name, "rwkv_wkv6") == 0) {
    result.a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 128);
    result.b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 128);
    result.c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 128);
    result.d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
    result.e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 128);
    result.f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64 * 64 * 32, 1);
    result.output = ggml_rwkv_wkv6(ctx, result.a, result.b, result.c, result.d,
                                   result.e, result.f);
    result.model = "RWKV6";
    result.modelShape = "tokens=128,heads=32,head_size=64";
    result.implementation = "ggml_cpu_rwkv_wkv6";
    result.comparison = "same_math_and_layout_different_output_abi";
    return result;
  }
  if (std::strcmp(name, "add_id") == 0) {
    result.a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 8, 128);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 256);
    result.c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 128);
    result.output = ggml_add_id(ctx, result.a, result.b, result.c);
    result.model = "MoE-combine";
    result.modelShape = "tokens=128,slots=8,experts=256,hidden=4096";
    result.implementation = "ggml_cpu_add_id";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gbytes_s";
    result.work = 3.0 * 128.0 * 8.0 * 4096.0 * sizeof(float);
    return result;
  }
  if (std::strcmp(name, "get_rows_back") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
    result.b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 128);
    result.c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 32000);
    result.output = ggml_get_rows_back(ctx, result.a, result.b, result.c);
    result.model = "Embedding-backward";
    result.modelShape = "rows=32000,items=128,hidden=2048";
    result.implementation = "ggml_cpu_get_rows_back";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_zero_fill_gbytes_s";
    result.work = 32000.0 * 2048.0 * sizeof(float);
    return result;
  }
  if (std::strcmp(name, "mul_mat_id") == 0) {
    result.a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 4096, 8);
    result.b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 2, 64);
    result.c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 64);
    result.output = ggml_mul_mat_id(ctx, result.a, result.b, result.c);
    result.model = "MoE-indexed-contraction";
    result.modelShape = "tokens=64,slots=2,experts=8,N=4096,K=4096";
    result.implementation = "ggml_cpu_f32_mul_mat_id";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gop_s";
    result.work = 2.0 * 64.0 * 2.0 * 4096.0 * 4096.0;
    return result;
  }
  if (std::strcmp(name, "depthwise_conv2d") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 32);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 112, 112, 32, 1);
    result.output =
        ggml_conv_2d_dw_direct(ctx, result.a, result.b, 1, 1, 1, 1, 1, 1);
    result.model = "MobileNet";
    result.modelShape = "N=1,H=112,W=112,C=32,K=3,pad=1";
    result.implementation = "ggml_cpu_direct_depthwise_whcn";
    result.comparison = "same_math_native_layouts_weft_cwhn_ggml_whcn";
    result.metric = "cold_gop_s";
    result.work = 2.0 * 112.0 * 112.0 * 32.0 * 3.0 * 3.0;
    return result;
  }
  if (std::strcmp(name, "max_pool2d") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 112, 112, 64, 1);
    result.output =
        ggml_pool_2d(ctx, result.a, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    result.model = "Vision-pooling";
    result.modelShape = "N=1,C=64,H=112,W=112,K=2,S=2";
    result.implementation = "ggml_cpu_max_pool2d";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gbytes_s";
    result.work = (112.0 * 112.0 * 64.0 + 56.0 * 56.0 * 64.0) *
                  sizeof(float);
    return result;
  }
  if (std::strcmp(name, "bilinear_upscale") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 64, 1);
    result.output = ggml_interpolate(ctx, result.a, 256, 256, 64, 1,
                                     GGML_SCALE_MODE_BILINEAR);
    result.model = "Vision-upscale";
    result.modelShape = "N=1,H=128,W=128,C=64,to=256x256";
    result.implementation = "ggml_cpu_bilinear_whcn";
    result.comparison = "same_math_native_layouts_weft_cwhn_ggml_whcn";
    result.metric = "cold_gbytes_s";
    result.work = 5.0 * 256.0 * 256.0 * 64.0 * sizeof(float);
    return result;
  }
  if (std::strcmp(name, "argsort") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32000, 8);
    result.output = ggml_argsort(ctx, result.a, GGML_SORT_ORDER_ASC);
    result.model = "Batched-logits-sort";
    result.modelShape = "rows=8,vocab=32000";
    result.implementation = "ggml_cpu_std_sort";
    result.comparison = "same_semantics_shape_and_layout";
    return result;
  }
  if (std::strcmp(name, "set_rows") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 4096, 8, 1);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 8, 1);
    result.c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 128);
    result.output = ggml_set_rows(ctx, result.a, result.b, result.c);
    result.model = "KV-cache-row-update";
    result.modelShape =
        "heads=8,capacity=4096,updates=128,head_dim=128";
    result.implementation = "ggml_cpu_set_rows";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gbytes_s";
    result.work = 2.0 * 8.0 * 128.0 * 128.0 * sizeof(float);
    return result;
  }
  if (std::strcmp(name, "window_partition") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 768, 64, 64, 1);
    result.output = ggml_win_part(ctx, result.a, 14);
    result.model = "SAM-window-partition";
    result.modelShape = "C=768,H=64,W=64,window=14";
    result.implementation = "ggml_cpu_win_part";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gbytes_s";
    result.work = (64.0 * 64.0 * 768.0 + 25.0 * 14.0 * 14.0 * 768.0) *
                  sizeof(float);
    return result;
  }
  if (std::strcmp(name, "dense_conv2d") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 256, 256);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 64, 256, 1);
    result.output =
        ggml_conv_2d_direct(ctx, result.a, result.b, 1, 1, 1, 1, 1, 1);
    result.model = "SAM-dense-convolution";
    result.modelShape = "N=1,H=64,W=64,IC=256,OC=256,K=3x3,pad=1";
    result.implementation = "ggml_cpu_direct_conv2d";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gop_s";
    result.work = 2.0 * 64.0 * 64.0 * 256.0 * 256.0 * 3.0 * 3.0;
    return result;
  }
  if (std::strcmp(name, "conv_transpose2d") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 128, 256);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 256, 1);
    result.output = ggml_conv_transpose_2d_p0(ctx, result.a, result.b, 2);
    result.model = "Decoder-upsample";
    result.modelShape =
        "N=1,H=16,W=16,IC=256,OC=128,K=2x2,stride=2";
    result.implementation = "ggml_cpu_conv_transpose2d";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gop_s";
    result.work = 2.0 * 16.0 * 16.0 * 256.0 * 128.0 * 2.0 * 2.0;
    return result;
  }
  if (std::strcmp(name, "rms_norm_backward") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 512);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 512);
    result.output = ggml_rms_norm_back(ctx, result.a, result.b, 1.0e-5F);
    result.model = "Training-hidden-gradient";
    result.modelShape = "rows=512,hidden=4096";
    result.implementation = "ggml_cpu_rms_norm_back";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_melements_s";
    result.work = 512.0 * 4096.0;
    result.metricDivisor = 1.0;
    return result;
  }
  if (std::strcmp(name, "out_product") == 0) {
    result.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5632, 32);
    result.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 32);
    result.output = ggml_out_prod(ctx, result.a, result.b);
    result.model = "TinyLlama-linear-gradient";
    result.modelShape = "rows=2048,columns=5632,samples=32";
    result.implementation = "ggml_cpu_out_prod";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_gop_s";
    result.work = 2.0 * 2048.0 * 5632.0 * 32.0;
    return result;
  }
  if (std::strcmp(name, "im2col_backward") == 0) {
    result.a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2304, 64, 64, 1);
    result.b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 256, 1);
    std::int64_t inputShape[4] = {64, 64, 256, 1};
    result.output = ggml_im2col_back(ctx, result.a, result.b, inputShape, 1, 1,
                                     1, 1, 1, 1, true);
    result.model = "SAM-im2col-backward";
    result.modelShape =
        "N=1,H=64,W=64,IC=256,K=3x3,pad=1,stride=1";
    result.implementation = "ggml_cpu_im2col_back";
    result.comparison = "same_semantics_shape_and_layout";
    result.metric = "cold_moutput_s";
    result.work = 64.0 * 64.0 * 256.0;
    result.metricDivisor = 1.0;
    return result;
  }
  return result;
}

void initialize(const char *name, const built_case &operation) {
  if (std::strcmp(name, "top_k") == 0) {
    setF32(operation.a, [](std::size_t index) {
      const std::size_t row = index / 256U;
      const std::size_t column = index % 256U;
      return static_cast<float>((row * 977U + column * 811U) % 65521U) /
                 1024.0F +
             static_cast<float>(column) / 1048576.0F;
    });
    return;
  }
  if (std::strcmp(name, "ssm_conv") == 0) {
    setF32(operation.a,
           [](std::size_t index) { return centered(index, 257, 256.0F); });
    setF32(operation.b,
           [](std::size_t index) { return centered(index, 31, 32.0F); });
    return;
  }
  if (std::strcmp(name, "ssm_scan") == 0) {
    setF32(operation.a,
           [](std::size_t index) { return centered(index, 31, 128.0F); });
    setF32(operation.b,
           [](std::size_t index) { return centered(index, 67, 64.0F); });
    setF32(operation.c,
           [](std::size_t index) { return centered(index, 29, 32.0F); });
    setF32(operation.d, [](std::size_t index) {
      return -0.01F * static_cast<float>(index + 1U);
    });
    setF32(operation.e,
           [](std::size_t index) { return centered(index, 23, 128.0F); });
    setF32(operation.f, [](std::size_t index) {
      return centered(index * 3U + 1U, 37, 64.0F);
    });
    setI32(operation.g, [](std::size_t) { return 0; });
    return;
  }
  if (std::strcmp(name, "rwkv_wkv6") == 0) {
    setF32(operation.a,
           [](std::size_t index) { return centered(index, 29, 64.0F); });
    setF32(operation.b, [](std::size_t index) {
      return centered(index * 3U + 1U, 31, 64.0F);
    });
    setF32(operation.c, [](std::size_t index) {
      return centered(index * 5U + 2U, 37, 128.0F);
    });
    setF32(operation.d,
           [](std::size_t index) { return centered(index, 19, 128.0F); });
    setF32(operation.e, [](std::size_t index) {
      return 0.90F + static_cast<float>(index % 17U) / 200.0F;
    });
    setF32(operation.f,
           [](std::size_t index) { return centered(index, 23, 256.0F); });
    return;
  }
  if (std::strcmp(name, "add_id") == 0) {
    setF32(operation.a,
           [](std::size_t index) { return centered(index, 257, 128.0F); });
    setF32(operation.b,
           [](std::size_t index) { return centered(index, 127, 64.0F); });
    setI32(operation.c, [](std::size_t index) {
      const std::size_t token = index / 8U;
      const std::size_t slot = index % 8U;
      return static_cast<std::int32_t>((token * 37U + slot * 53U) % 256U);
    });
    return;
  }
  if (std::strcmp(name, "get_rows_back") == 0) {
    setF32(operation.a, [](std::size_t index) {
      const std::size_t item = index / 2048U;
      const std::size_t column = index % 2048U;
      return centered(item + column, 257, 256.0F);
    });
    setI32(operation.b, [](std::size_t item) {
      return static_cast<std::int32_t>((item * 7919U) % 97U);
    });
    return;
  }
  if (std::strcmp(name, "mul_mat_id") == 0) {
    setF32(operation.a, [](std::size_t index) {
      constexpr std::array<float, 4> localValues = {1.0F, 0.5F, -1.0F,
                                                    2.0F};
      const std::size_t inner = index % 4096U;
      const std::size_t row = (index / 4096U) % 4096U;
      const std::size_t expert = index / (4096U * 4096U);
      return localValues[(expert + row + 3U * inner + 1U) % 4U];
    });
    setF32(operation.b, [](std::size_t index) {
      constexpr std::array<float, 4> localValues = {1.0F, 0.5F, -1.0F,
                                                    2.0F};
      const std::size_t inner = index % 4096U;
      const std::size_t item = index / 4096U;
      const std::size_t slot = item % 2U;
      const std::size_t token = item / 2U;
      return localValues[(token + 5U * slot + 3U * inner + 2U) % 4U];
    });
    setI32(operation.c, [](std::size_t index) {
      const std::size_t token = index / 2U;
      const std::size_t slot = index % 2U;
      return static_cast<std::int32_t>((token * 5U + slot * 3U) % 8U);
    });
    return;
  }
  if (std::strcmp(name, "depthwise_conv2d") == 0) {
    setF32(operation.a, [](std::size_t index) {
      const std::size_t kernelIndex = index % 9U;
      const std::size_t channel = index / 9U;
      return centered(kernelIndex * 32U + channel, 31, 64.0F);
    });
    setF32(operation.b, [](std::size_t index) {
      const std::size_t plane = 112U * 112U;
      const std::size_t channel = index / plane;
      const std::size_t spatial = index % plane;
      return centered(spatial * 32U + channel, 257, 256.0F);
    });
    return;
  }
  if (std::strcmp(name, "max_pool2d") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return static_cast<float>(
                 static_cast<int>((index * 811U) % 65521U) - 32760) /
             1024.0F;
    });
    return;
  }
  if (std::strcmp(name, "bilinear_upscale") == 0) {
    setF32(operation.a, [](std::size_t index) {
      const std::size_t plane = 128U * 128U;
      const std::size_t channel = index / plane;
      const std::size_t spatial = index % plane;
      return centered(spatial * 64U + channel, 1021, 512.0F);
    });
    return;
  }
  if (std::strcmp(name, "argsort") == 0) {
    setF32(operation.a, [](std::size_t index) {
      const std::size_t row = index / 32000U;
      const std::size_t column = index % 32000U;
      return static_cast<float>((row * 104729U + column * 65537U) %
                                16777213U) +
             static_cast<float>(column) / 65536.0F;
    });
    return;
  }
  if (std::strcmp(name, "set_rows") == 0) {
    setF32(operation.a, [](std::size_t) { return -7.0F; });
    setF32(operation.b, [](std::size_t index) {
      return centered(index, 257, 128.0F);
    });
    setI32(operation.c, [](std::size_t update) {
      return static_cast<std::int32_t>((update * 31U + 17U) % 4096U);
    });
    return;
  }
  if (std::strcmp(name, "window_partition") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return centered(index, 1021, 512.0F);
    });
    return;
  }
  if (std::strcmp(name, "dense_conv2d") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return centered(index * 5U, 131, 512.0F);
    });
    setF32(operation.b, [](std::size_t index) {
      return centered(index, 127, 256.0F);
    });
    return;
  }
  if (std::strcmp(name, "conv_transpose2d") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return centered(index * 7U, 131, 512.0F);
    });
    setF32(operation.b, [](std::size_t index) {
      return centered(index, 127, 256.0F);
    });
    return;
  }
  if (std::strcmp(name, "rms_norm_backward") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return centered(index * 17U, 251, 256.0F);
    });
    setF32(operation.b, [](std::size_t index) {
      return centered(index, 257, 128.0F);
    });
    return;
  }
  if (std::strcmp(name, "out_product") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return centered(index * 3U, 131, 128.0F);
    });
    setF32(operation.b, [](std::size_t index) {
      return centered(index, 127, 64.0F);
    });
    return;
  }
  if (std::strcmp(name, "im2col_backward") == 0) {
    setF32(operation.a, [](std::size_t index) {
      return centered(index * 11U, 251, 256.0F);
    });
    setF32(operation.b, [](std::size_t) { return 0.0F; });
  }
}

bool validateTopK(ggml_tensor *output) {
  constexpr std::size_t rows = 512;
  constexpr std::size_t columns = 256;
  constexpr std::size_t selected = 8;
  std::vector<std::int32_t> actual(rows * selected);
  ggml_backend_tensor_get(output, actual.data(), 0,
                          actual.size() * sizeof(std::int32_t));
  std::vector<std::size_t> order(columns);
  std::iota(order.begin(), order.end(), 0U);
  for (std::size_t row = 0; row < rows; ++row) {
    auto score = [row](std::size_t column) {
      return static_cast<float>((row * 977U + column * 811U) % 65521U) /
                 1024.0F +
             static_cast<float>(column) / 1048576.0F;
    };
    std::partial_sort(order.begin(), order.begin() + selected, order.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                        return score(lhs) > score(rhs);
                      });
    std::array<std::int32_t, selected> expected{};
    std::array<std::int32_t, selected> observed{};
    for (std::size_t rank = 0; rank < selected; ++rank) {
      expected[rank] = static_cast<std::int32_t>(order[rank]);
      observed[rank] = actual[row * selected + rank];
    }
    std::sort(expected.begin(), expected.end());
    std::sort(observed.begin(), observed.end());
    if (expected != observed)
      return false;
  }
  return true;
}

bool validateArgsort(ggml_tensor *output) {
  constexpr std::size_t rows = 8;
  constexpr std::size_t columns = 32000;
  std::vector<std::int32_t> actual(rows * columns);
  ggml_backend_tensor_get(output, actual.data(), 0,
                          actual.size() * sizeof(std::int32_t));
  std::vector<std::size_t> order(columns);
  for (std::size_t row = 0; row < rows; ++row) {
    std::iota(order.begin(), order.end(), 0U);
    auto value = [row](std::size_t column) {
      return static_cast<float>((row * 104729U + column * 65537U) %
                                16777213U) +
             static_cast<float>(column) / 65536.0F;
    };
    std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                               std::size_t rhs) {
      return value(lhs) < value(rhs);
    });
    for (std::size_t column = 0; column < columns; ++column)
      if (actual[row * columns + column] !=
          static_cast<std::int32_t>(order[column]))
        return false;
  }
  return true;
}

bool validateI32(const char *name, ggml_tensor *output) {
  if (std::strcmp(name, "top_k") == 0)
    return validateTopK(output);
  if (std::strcmp(name, "argsort") == 0)
    return validateArgsort(output);
  return false;
}

bool validateF32(ggml_tensor *output, float &sample) {
  const std::size_t elements =
      static_cast<std::size_t>(ggml_nelements(output));
  const std::array<std::size_t, 3> offsets = {0U, elements / 2U,
                                              elements - 1U};
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    float value = 0.0F;
    ggml_backend_tensor_get(output, &value, offsets[index] * sizeof(float),
                            sizeof(float));
    if (!std::isfinite(value))
      return false;
    if (index == 0)
      sample = value;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <kernel> <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repetitions = parseRepetitions(argv[2]);
  if (repetitions == 0) {
    std::fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  constexpr std::size_t graphNodes = 64;
  ggml_init_params params = {
      ggml_tensor_overhead() * 32U +
          ggml_graph_overhead_custom(graphNodes, false),
      nullptr,
      true,
  };
  ggml_context *ctx = ggml_init(params);
  if (ctx == nullptr) {
    std::fprintf(stderr, "failed to create GGML context\n");
    return 1;
  }
  const built_case operation = build(ctx, argv[1]);
  if (operation.output == nullptr) {
    std::fprintf(stderr, "unsupported unfamiliar GGML kernel: %s\n", argv[1]);
    ggml_free(ctx);
    return 2;
  }

  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, 1);
  if (!ggml_backend_supports_op(backend, operation.output)) {
    std::fprintf(stderr, "GGML CPU backend does not support: %s\n", argv[1]);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buffer == nullptr) {
    std::fprintf(stderr, "failed to allocate GGML buffers for %s\n", argv[1]);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }
  initialize(argv[1], operation);

  ggml_cgraph *graph = ggml_new_graph_custom(ctx, graphNodes, false);
  ggml_build_forward_expand(graph, operation.output);
  ggml_status status = ggml_backend_graph_compute(backend, graph);
  ggml_backend_synchronize(backend);
  if (status != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "warmup failed for %s: %s\n", argv[1],
                 ggml_status_to_string(status));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  float outputSample = 0.0F;
  const bool valid =
      operation.output->type == GGML_TYPE_I32
          ? validateI32(argv[1], operation.output)
          : validateF32(operation.output, outputSample);
  if (!valid) {
    std::fprintf(stderr, "invalid output from %s\n", argv[1]);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 1;
  }

  std::vector<std::uint8_t> flushBuffer(kFlushBytes, 1);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    evictCache(flushBuffer);
    const auto begin = std::chrono::steady_clock::now();
    status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    if (status != GGML_STATUS_SUCCESS) {
      std::fprintf(stderr, "compute failed for %s: %s\n", argv[1],
                   ggml_status_to_string(status));
      ggml_backend_buffer_free(buffer);
      ggml_backend_free(backend);
      ggml_free(ctx);
      return 1;
    }
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double coldMedianUs = median(samples);
  std::printf("family=unfamiliar\n");
  std::printf("kernel=%s\n", argv[1]);
  std::printf("implementation=%s\n", operation.implementation);
  std::printf("model=%s\n", operation.model);
  std::printf("model_shape=%s\n", operation.modelShape);
  std::printf("comparison=%s\n", operation.comparison);
  std::printf("threads=1\n");
  std::printf("vlen_bits=%d\n", ggml_cpu_get_rvv_vlen() * 8);
  std::printf("cold_protocol=64MiB-evict-then-single-op-graph\n");
  std::printf("timed_scope=graph_compute_and_synchronize\n");
  std::printf("repetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\n", coldMedianUs);
  if (operation.metric != nullptr)
    std::printf("%s=%.6f\n", operation.metric,
                operation.work / coldMedianUs / operation.metricDivisor);
  std::printf("output_sample=%.9g\n", outputSample);

  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);
  ggml_free(ctx);
  return 0;
}
