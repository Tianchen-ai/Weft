from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_tq2_0(W: wl.View[ggml.TQ2_0, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        q = wl.i32(wl.i8(w.q) - wl.i8(1))
        wl.store(Y[kb], qf.ternary_radix(q, w.d))
