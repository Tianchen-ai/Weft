from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ3_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X4, (256, 4)],
    signs: wl.View[ggml.I8X8, (128, 8)],
):
    sumf = wl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        half = wl.arange(0, 2, dtype=wl.u32, axis="half")
        pair_entry = wl.arange(0, 4, dtype=wl.u32, axis="pair_entry")
        pair = wl.arange(0, 2, dtype=wl.u32, axis="pair")
        entry_code = pair_entry * wl.u32(2) + pair
        payload = wl.arange(0, 4, dtype=wl.u16, axis="payload")
        sign_payload = wl.arange(0, 8, dtype=wl.u16, axis="sign_payload")
        block_sum = wl.i32(0)
        group_index = wl.index(0)
        with wl.level.subtiles(kb, extent=64) as group:
            metadata = w.metadata[wl.u32(group_index) * wl.u32(2) + half]
            grid_index = wl.widen(
                w.q[
                    wl.u32(group_index) * wl.u32(16)
                    + half * wl.u32(8)
                    + entry_code
                ],
                wl.u32,
            )
            sign_index = metadata >> pair_entry * wl.u32(7) & wl.u32(127)
            weight = wl.lookup(
                grid_values, grid_index * wl.u32(4) + payload, bounds="in_bounds"
            )
            sign_bytes = wl.lookup(
                sign_values, sign_index * wl.u32(8) + sign_payload, bounds="in_bounds"
            )
            sign = wl.reshape(
                sign_bytes,
                shape=(2, 4, 2, 4),
                axes=("half", "pair_entry", "pair", "payload"),
                order=("half", "pair_entry", "sign_payload"),
            )
            activation = x.q[
                wl.u32(group_index) * wl.u32(64)
                + half * wl.u32(32)
                + entry_code * wl.u32(4)
                + wl.u32(payload)
            ]
            local = wl.reduce_dot(
                activation,
                weight * sign,
                over=("pair_entry", "pair", "payload"),
                acc_dtype=wl.i32,
            )
            scale = wl.i32((metadata >> wl.u32(28)) * wl.u32(2) + wl.u32(1))
            block_sum += wl.reduce(local * scale, axis="half")
            group_index += wl.index(1)
        sumf += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    return wl.f32(0.25) * sumf


@weft.kernel
def quantized_vec_dot_iq3_xxs_q8_k(
    W: wl.View[ggml.IQ3_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X4, (256, 4)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid, signs))
