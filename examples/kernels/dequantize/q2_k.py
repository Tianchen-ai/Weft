from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_q2_k(W: wl.View[ggml.Q2_K, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        with wl.level.subtiles(kb, extent=16) as sub:
            metadata = w.scales[sub]
            scale = qf.extract_bits(metadata, 0, 4)
            minimum = qf.extract_bits(metadata, 4, 4)
            wl.store(Y[kb][sub], qf.k_superblock(w.q[sub], scale, w.d, minimum, w.dmin))
