from __future__ import annotations

from dataclasses import dataclass


FORMATS = ("q1_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
           "iq1_s", "iq1_m", "iq2_s", "iq2_xs", "iq2_xxs", "iq3_s", "iq3_xxs", "iq4_nl", "iq4_xs",
           "tq1_0", "tq2_0", "mxfp4", "nvfp4")
PARTNERS = ("q8_0", "q8_0", "q8_1", "q8_0", "q8_1", "q8_0", "q8_k", "q8_k", "q8_k", "q8_k", "q8_k",
            "q8_k", "q8_k", "q8_k", "q8_k", "q8_k", "q8_k", "q8_k", "q8_0", "q8_k", "q8_k", "q8_k", "q8_0", "q8_0")


@dataclass(frozen=True)
class Workload:
    name: str
    runtime: str
    expected_symbol: str
    defines: tuple[str, ...] = ()
    phase: str | None = None
    matrix_extension: str = "none"
    ggml: bool = False


def select(runner: str, name: str, phase: str | None) -> Workload:
    if runner == "kernel":
        simple = {"q4_k_gemv": "q4_k_gemv_runtime.cpp", "gemv_f32": "gemv_runtime.cpp",
                  "gemm_f32": "gemm_runtime.cpp", "ime_i8_contract": "ime_i8_contract_runtime.cpp"}
        if name in simple:
            return Workload(name, simple[name], name, phase="prefill" if name == "gemm_f32" else None,
                            matrix_extension="spacemit-ime1" if name == "ime_i8_contract" else "none")
        if name == "q4_k_gemv_groups4":
            return Workload(name, "q4_k_gemv_runtime.cpp", name, ("-DWEFT_Q4_GROUPS4=1",))
        if name in ("q8_0_quantize", "q8_1_quantize", "q8_K_quantize"):
            kind = ("q8_0_quantize", "q8_1_quantize", "q8_K_quantize").index(name)
            return Workload(name, "q8_quantize_runtime.cpp", name, (f"-DWEFT_Q8_KIND={kind}",))
        raise ValueError(f"unsupported kernel: {name}")
    if runner == "vec-dot":
        index = FORMATS.index(name)
        return Workload(name, "vec_dot_runtime.cpp", f"quantized_vec_dot_{name}_{PARTNERS[index]}",
                        (f"-DWEFT_VEC_DOT_FORMAT={index}",), ggml=True)
    if runner == "row-dequantize":
        index = FORMATS.index(name)
        return Workload(name, "row_dequantize_runtime.cpp", f"row_dequantize_{name}",
                        (f"-DWEFT_ROW_FORMAT={index}",), ggml=True)
    if runner != "mul-mat" or phase not in ("decode", "prefill"):
        raise ValueError("mul-mat requires an explicit decode or prefill phase")
    if name == "f32":
        return Workload(name, "gemm_runtime.cpp", "gemm_f32", phase=phase)
    formats = ("f16", *FORMATS)
    base = name.removesuffix("_persistent").removesuffix("_staged")
    index = formats.index(base)
    if phase == "prefill" and name in ("q4_k", "iq2_s", "iq2_xs"):
        name += "_staged"
    defines = [f"-DWEFT_MUL_MAT_FORMAT={index}"]
    variants = {"q4_k_persistent": "Q4_DERIVED", "q4_k_staged": "Q4_STAGED",
                "iq2_xxs_staged": "IQ2_XXS_STAGED", "iq2_xs_staged": "IQ2_XS_STAGED",
                "iq2_s_staged": "IQ2_S_STAGED"}
    if name in variants:
        defines.append(f"-DWEFT_{variants[name]}=1")
    symbol = f"production_mul_mat_{name}"
    decode = dict(zip(("q1_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k",
                       "q5_k", "q6_k", "iq1_s", "iq1_m", "iq2_xxs", "iq4_nl", "iq4_xs", "tq1_0", "tq2_0", "mxfp4", "nvfp4"),
                      ("Q10", "Q40", "Q41", "Q50", "Q51", "Q80", "Q2K", "Q3K", "Q4K", "Q5K", "Q6K",
                       "IQ1S", "IQ1M", "IQ2_XXS", "IQ4_NL", "IQ4_XS", "TQ10", "TQ20", "MXFP4", "NVFP4")))
    if phase == "decode" and name in decode:
        symbol += "_decode"
        defines.append(f"-DWEFT_{decode[name]}_DECODE=1")
    return Workload(name, "mul_mat_runtime.cpp", symbol, tuple(defines), phase=phase, ggml=True)
