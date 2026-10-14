from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def _signed_scale(raw, zero: int = 0):
    return wl.i32(raw) - wl.i32(zero)


@weft.kernel
def row_dequantize_iq4_xs(
    W: wl.View[ggml.IQ4_XS, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (K,)],
):
    table = wl.stage(wl.load(codebook))
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        sub_index = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as sub:
            low = qf.extract_bits(
                w.scales_l[sub_index // wl.index(2)],
                sub_index % wl.index(2) * wl.index(4),
                4,
            )
            high = qf.extract_bits(w.scales_h, sub_index * wl.index(2), 2)
            scale = _signed_scale(low | high << wl.u32(4), zero=32)
            q = qf.small_nonlinear_lookup(table, w.q[sub])
            wl.store(Y[kb][sub], qf.iq_codebook(wl.i32(q), wl.f32(w.d) * wl.f32(scale)))
            sub_index += wl.index(1)
