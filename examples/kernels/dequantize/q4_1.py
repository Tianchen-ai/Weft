from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_q4_1(W: wl.View[ggml.Q4_1, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        wl.store(Y[kb], qf.min_affine(w.q, w.d, w.m))
