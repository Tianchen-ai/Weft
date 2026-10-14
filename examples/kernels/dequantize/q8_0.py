from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_q8_0(W: wl.View[ggml.Q8_0, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        wl.store(Y[kb], qf.symmetric_integer(w.q, w.d))
