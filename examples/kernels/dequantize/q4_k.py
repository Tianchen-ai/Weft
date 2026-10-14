from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_q4_k(W: wl.View[ggml.Q4_K, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        group = wl.index(0)
        with wl.level.subtiles(kb, extent=64) as group_point:
            lane = wl.arange(0, 64, dtype=wl.u32, axis="k")
            scale_index = wl.u32(group) * wl.u32(2) + lane // wl.u32(32)
            wl.store(Y[kb][group_point], qf.k_superblock(
                    w.q[group_point], w.sc[scale_index], w.d, w.m[scale_index], w.dmin
                ))
            group += wl.index(1)
