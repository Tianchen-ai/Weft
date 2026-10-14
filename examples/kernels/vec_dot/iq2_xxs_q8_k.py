from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ2_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
):
    sumf = wl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        group = wl.arange(0, 8, dtype=wl.u32, axis="group")
        entry = wl.arange(0, 4, dtype=wl.u32, axis="entry")
        payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
        group_word = group * wl.u32(4)
        word0 = wl.widen(w.q[group_word + (entry >> wl.u32(1))], wl.u32)
        grid_index = word0 >> ((entry & wl.u32(1)) * wl.u32(8)) & wl.u32(255)
        word1 = wl.widen(w.q[group_word + wl.u32(2)], wl.u32) | wl.widen(
            w.q[group_word + wl.u32(3)], wl.u32
        ) << wl.u32(16)
        sign_index = wl.narrow(
            word1 >> entry * wl.u32(7), wl.u16, rounding="rtz", saturation=False
        ) & wl.u16(127)
        weight = wl.lookup(
            grid_values, grid_index * wl.u32(8) + payload, bounds="in_bounds"
        )
        sign = wl.lookup(
            sign_values, sign_index * wl.u16(8) + payload, bounds="in_bounds"
        )
        activation = x.q[group * wl.u32(32) + entry * wl.u32(8) + wl.u32(payload)]
        partial = wl.reduce_dot(activation, weight * sign, over=("entry", "payload"), acc_dtype=wl.i32)
        scale = wl.i32((word1 >> wl.u32(28)) * wl.u32(2) + wl.u32(1))
        block_sum = wl.reduce(partial * scale, axis="group")
        sumf += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    return wl.f32(0.125) * sumf


@weft.kernel
def quantized_vec_dot_iq2_xxs_q8_k(
    W: wl.View[ggml.IQ2_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid, signs))
