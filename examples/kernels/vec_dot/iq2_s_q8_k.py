from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ2_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (1024, 8)],
):
    sumf = wl.f32(0.0)
    grid_values = grid.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        scale_group = wl.arange(0, 4, dtype=wl.u32, axis="scale_group")
        entry = wl.arange(0, 2, dtype=wl.u32, axis="entry")
        payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
        block_sum = wl.i32(0)
        group_index = wl.index(0)
        with wl.level.subtiles(kb, extent=64) as group:
            linear_entry = scale_group * wl.u32(2) + entry
            grid_index = wl.widen(
                w.q[wl.u32(group_index) * wl.u32(8) + linear_entry], wl.u32
            ) | (
                (
                    wl.widen(w.qh[group_index * wl.index(2)], wl.u32)
                    | wl.widen(w.qh[group_index * wl.index(2) + wl.index(1)], wl.u32)
                    << wl.u32(8)
                )
                >> linear_entry * wl.u32(2)
                & wl.u32(3)
            ) << wl.u32(
                8
            )
            sign_bit = w.signs[
                wl.u32(group_index) * wl.u32(64)
                + linear_entry * wl.u32(8)
                + wl.u32(payload)
            ]
            weight = wl.lookup(
                grid_values, grid_index * wl.u32(8) + payload, bounds="in_bounds"
            )
            signed_weight = weight * (wl.i8(1) - wl.i8(sign_bit) * wl.i8(2))
            activation = x.q[
                wl.u32(group_index) * wl.u32(64)
                + linear_entry * wl.u32(8)
                + wl.u32(payload)
            ]
            partial = wl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=wl.i32)
            metadata = wl.widen(
                w.scales[wl.u32(group_index) * wl.u32(2) + scale_group // wl.u32(2)],
                wl.u32,
            )
            scale = wl.i32(
                (metadata >> scale_group % wl.u32(2) * wl.u32(4) & wl.u32(15))
                * wl.u32(2)
                + wl.u32(1)
            )
            block_sum += wl.reduce(partial * scale, axis="scale_group")
            group_index += wl.index(1)
        sumf += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    return wl.f32(0.125) * sumf


@weft.kernel
def quantized_vec_dot_iq2_s_q8_k(
    W: wl.View[ggml.IQ2_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (1024, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid))
