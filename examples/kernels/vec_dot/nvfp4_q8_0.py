from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.NVFP4, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    ue4m3_scale: wl.View[wl.f32, (256,)],
):
    table = wl.stage(wl.load(codebook))
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=64) as wb:
        w = wl.load(W[wb])
        with wl.level.subtiles(wb, extent=32) as xb:
            x = wl.load(X[xb])
            with wl.level.subtiles(xb, extent=16) as sub:
                q = qf.small_nonlinear_lookup(table, w.q[sub])
                integer = wl.reduce_dot(q, x.q[sub], over="k", acc_dtype=wl.i32)
                scale = wl.f32(x.d) * qf.exponent_scale(ue4m3_scale, w.d[sub])
                result += scale * wl.f32(integer)
    return result


@weft.kernel
def quantized_vec_dot_nvfp4_q8_0(
    W: wl.View[ggml.NVFP4, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    scale: wl.View[wl.f32, (256,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, codebook, scale))
