from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_q5_k(W: wl.View[ggml.Q5_K, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        with wl.level.subtiles(kb, extent=32) as sub:
            q = wl.i32(w.q[sub]) | wl.i32(w.qh[sub]) << wl.u32(4)
            wl.store(Y[kb][sub], qf.k_superblock(q, w.sc[sub], w.d, w.m[sub], w.dmin))
