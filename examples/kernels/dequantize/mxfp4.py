from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_mxfp4(
    W: wl.View[ggml.MXFP4, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    scale: wl.View[wl.f32, (256,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        decoded_scale = qf.exponent_scale(scale, w.e)
        q = qf.nonlinear_lookup(codebook, w.q)
        wl.store(Y[kb], qf.fp4_codebook(q, decoded_scale))
