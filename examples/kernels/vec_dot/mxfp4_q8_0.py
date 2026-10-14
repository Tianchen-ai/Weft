from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.MXFP4, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    e8m0_scale: wl.View[wl.f32, (256,)],
):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        integer = qf.dot_codebook32(w.q, x.q, codebook)
        scale = wl.f32(x.d) * qf.exponent_scale(e8m0_scale, w.e)
        result += scale * wl.f32(integer)
    return result


@weft.kernel
def quantized_vec_dot_mxfp4_q8_0(
    W: wl.View[ggml.MXFP4, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    scale: wl.View[wl.f32, (256,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, codebook, scale))
