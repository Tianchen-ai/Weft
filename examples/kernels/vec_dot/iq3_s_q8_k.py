from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ3_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X4, (512, 4)],
):
    result = wl.f32(0.0)
    grid_values = grid.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        half = wl.arange(0, 2, dtype=wl.u32, axis="half")
        entry = wl.arange(0, 4, dtype=wl.u32, axis="entry")
        code = wl.arange(0, 2, dtype=wl.u32, axis="code")
        payload = wl.arange(0, 4, dtype=wl.u16, axis="payload")
        block_sum = wl.i32(0)
        group_index = wl.index(0)
        with wl.level.subtiles(kb, extent=64) as group:
            storage_coordinate = (
                wl.u32(group_index) * wl.u32(16)
                + half * wl.u32(8)
                + entry * wl.u32(2)
                + code
            )
            grid_index = wl.widen(w.q[storage_coordinate], wl.u32) | wl.widen(
                w.qh[storage_coordinate], wl.u32
            ) << wl.u32(8)
            weight = wl.lookup(
                grid_values, grid_index * wl.u32(4) + payload, bounds="in_bounds"
            )
            sign_bit = w.signs[
                wl.u32(group_index) * wl.u32(64)
                + half * wl.u32(32)
                + entry * wl.u32(8)
                + code * wl.u32(4)
                + wl.u32(payload)
            ]
            signed_weight = weight * (wl.i8(1) - wl.i8(sign_bit) * wl.i8(2))
            activation = x.q[
                wl.u32(group_index) * wl.u32(64)
                + half * wl.u32(32)
                + entry * wl.u32(8)
                + code * wl.u32(4)
                + wl.u32(payload)
            ]
            local = wl.reduce_dot(
                activation,
                signed_weight,
                over=("entry", "code", "payload"),
                acc_dtype=wl.i32,
            )
            scale = wl.i32(w.scales[wl.u32(group_index) * wl.u32(2) + half]) * wl.i32(
                2
            ) + wl.i32(1)
            block_sum += wl.reduce(local * scale, axis="half")
            group_index += wl.index(1)
        result += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    return result


@weft.kernel
def quantized_vec_dot_iq3_s_q8_k(
    W: wl.View[ggml.IQ3_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X4, (512, 4)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid))
