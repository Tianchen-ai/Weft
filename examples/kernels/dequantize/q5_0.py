from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_q5_0(W: wl.View[ggml.Q5_0, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        q = wl.i32(wl.u8(w.q) | wl.u8(w.qh) << wl.u8(4)) - wl.i32(16)
        wl.store(Y[kb], qf.symmetric_integer(q, w.d))
