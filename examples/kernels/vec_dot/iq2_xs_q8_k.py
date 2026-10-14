from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def quantized_vec_dot_iq2_xs_q8_k(
    W: wl.View[ggml.IQ2_XS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (512, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    sumf = wl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        scale_group = wl.arange(0, 16, dtype=wl.u32, axis="scale_group")
        entry = wl.arange(0, 2, dtype=wl.u32, axis="entry")
        payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
        linear_entry = scale_group * wl.u32(2) + entry
        entry_offset = scale_group * wl.u32(16) + entry * wl.u32(8)
        block_sum = qf.iq2_xs_entry_reduce(
            w.q[linear_entry],
            w.scales[scale_group // wl.u32(2)],
            x.q[entry_offset + wl.u32(payload)],
            grid_values,
            sign_values,
            scale_group,
            payload,
        )
        sumf += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    wl.store(Y[0], wl.f32(0.125) * sumf)
