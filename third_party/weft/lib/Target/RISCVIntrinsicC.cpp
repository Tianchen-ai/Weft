#include "Emission/Emitter.h"

using namespace weft;
using namespace weft::riscv_emission;


mlir::LogicalResult weft::emitSelectedRISCVIntrinsicC(mlir::ModuleOp module,
                                                      std::string &result,
                                                      std::vector<RISCVKernelABI> &kernels) {
  std::string body;
  llvm::raw_string_ostream output(body);
  output << "#include <stddef.h>\n"
         << "#include <stdint.h>\n"
         << "#include <stdbool.h>\n"
         << "#include <math.h>\n"
         << "#include <string.h>\n"
         << "#include <riscv_vector.h>\n\n"
         << "static inline __attribute__((unused)) float weft_f8e4m3fn_to_f32(uint8_t x) {\n"
         << "  uint32_t m = x & 127, bits; float y;\n"
         << "  if (m < 8) { y = (float)m * 0x1p-9f; memcpy(&bits, &y, sizeof(bits)); }\n"
         << "  else bits = m == 127 ? UINT32_C(0x7fc00000) : (m << 20) + (UINT32_C(120) << 23);\n"
         << "  bits |= (uint32_t)(x & 128) << 24; memcpy(&y, &bits, sizeof(y)); return y;\n"
         << "}\n"
         << "static inline __attribute__((unused)) float weft_f8e5m2_to_f32(uint8_t x) {\n"
         << "  uint32_t m = x & 127, bits; float y;\n"
         << "  if (m < 4) { y = (float)m * 0x1p-16f; memcpy(&bits, &y, sizeof(bits)); }\n"
         << "  else bits = m >= 124 ? (m == 124 ? UINT32_C(0x7f800000) : UINT32_C(0x7fc00000)) : (m << 21) + (UINT32_C(112) << 23);\n"
         << "  bits |= (uint32_t)(x & 128) << 24; memcpy(&y, &bits, sizeof(y)); return y;\n"
         << "}\n"
         << "static inline __attribute__((unused)) float weft_bf16_to_f32(uint16_t x) {\n"
         << "  uint32_t bits = (uint32_t)x << 16; float y; memcpy(&y, &bits, sizeof(y)); return y;\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint16_t weft_f32_to_bf16(float x) {\n"
         << "  uint32_t bits; memcpy(&bits, &x, sizeof(bits));\n"
         << "  if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) return (uint16_t)((bits >> 16) | 64);\n"
         << "  return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1)) >> 16);\n"
         << "}\n"
         << "static inline __attribute__((unused)) int16_t weft_load_i16_le(const uint8_t *p) {\n"
         << "  int16_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint16_t weft_load_u16_le(const uint8_t *p) {\n"
         << "  uint16_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) uint32_t weft_load_u32_le(const uint8_t *p) {\n"
         << "  uint32_t value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) _Float16 weft_load_f16_le(const uint8_t *p) {\n"
         << "  _Float16 value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) float weft_load_f32_le(const uint8_t *p) {\n"
         << "  float value; memcpy(&value, p, sizeof(value)); return value;\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_i16_le(uint8_t *p, int16_t value) {\n"
         << "  uint16_t bits = (uint16_t)value; p[0] = (uint8_t)bits; p[1] = (uint8_t)(bits >> 8);\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_f16_le(uint8_t *p, _Float16 value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
         << "}\n"
         << "static inline __attribute__((unused)) void weft_store_f32_le(uint8_t *p, float value) {\n"
         << "  memcpy(p, &value, sizeof(value));\n"
         << "}\n"
         << "\n";
  output << R"C(static inline __attribute__((always_inline, unused)) float weft_exp_f32(float x) {
  if (x != x) return x;
  if (x >= 89.0f) return INFINITY;
  if (x <= -104.0f) return 0.0f;
  float z = __builtin_fmaf(x, 0x1.715476p+0f, 0x1.8p23f);
  float n = z - 0x1.8p23f;
  float r = __builtin_fmaf(-n, 0x1.62e4p-1f, x);
  r = __builtin_fmaf(-n, 0x1.7f7d1cp-20f, r);
  float p = 1.0f / 40320.0f;
  p = __builtin_fmaf(p, r, 1.0f / 5040.0f);
  p = __builtin_fmaf(p, r, 1.0f / 720.0f);
  p = __builtin_fmaf(p, r, 1.0f / 120.0f);
  p = __builtin_fmaf(p, r, 1.0f / 24.0f);
  p = __builtin_fmaf(p, r, 1.0f / 6.0f);
  p = __builtin_fmaf(p, r, 0.5f);
  p = __builtin_fmaf(p, r, 1.0f);
  p = __builtin_fmaf(p, r, 1.0f);
  int32_t exponent = (int32_t)n;
  int32_t first = exponent < -126 ? -126 : exponent > 127 ? 127 : exponent;
  int32_t second = exponent - first;
  uint32_t a = (uint32_t)(first + 127) << 23;
  uint32_t b = (uint32_t)(second + 127) << 23;
  float scale_a, scale_b;
  __builtin_memcpy(&scale_a, &a, sizeof(a));
  __builtin_memcpy(&scale_b, &b, sizeof(b));
  return (p * scale_b) * scale_a;
}
static inline __attribute__((always_inline, unused)) float weft_log_f32(float x) {
  if (x != x || x < 0.0f) return NAN;
  if (x == 0.0f) return -INFINITY;
  if (x == INFINITY) return INFINITY;
  int subnormal = x < 0x1p-126f;
  float normalized = subnormal ? x * 0x1p23f : x;
  uint32_t bits;
  __builtin_memcpy(&bits, &normalized, sizeof(bits));
  int32_t exponent = (int32_t)(bits >> 23) - 127 - (subnormal ? 23 : 0);
  bits = (bits & UINT32_C(0x007fffff)) | UINT32_C(0x3f800000);
  float m;
  __builtin_memcpy(&m, &bits, sizeof(m));
  if (m > 0x1.6a09e6p+0f) { m *= 0.5f; ++exponent; }
  float z = (m - 1.0f) / (m + 1.0f), square = z * z;
  float p = 1.0f / 11.0f;
  p = __builtin_fmaf(p, square, 1.0f / 9.0f);
  p = __builtin_fmaf(p, square, 1.0f / 7.0f);
  p = __builtin_fmaf(p, square, 1.0f / 5.0f);
  p = __builtin_fmaf(p, square, 1.0f / 3.0f);
  p = __builtin_fmaf(p, square, 1.0f);
  float value = (2.0f * z) * p;
  value = __builtin_fmaf((float)exponent, 0x1.7f7d1cp-20f, value);
  return __builtin_fmaf((float)exponent, 0x1.62e4p-1f, value);
}

)C";
  bool found = false;
  kernels.clear();
  for (riscv::ArtifactPackOp artifact :
       module.getOps<riscv::ArtifactPackOp>()) {
    Emitter emitter(module, artifact, output);
    if (mlir::failed(emitter.emitArtifact()))
      return mlir::failure();
  }
  for (riscv::KernelOp kernel : module.getOps<riscv::KernelOp>()) {
    Emitter emitter(module, kernel, output);
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
    auto abi = emitter.kernelABI();
    if (mlir::failed(abi))
      return mlir::failure();
    kernels.push_back(std::move(*abi));
    found = true;
  }
  if (!found)
    return module.emitError("intrinsic-C emission requires a verified RISC-V physical kernel");
  output.flush();
  result = std::move(body);
  return mlir::success();
}
