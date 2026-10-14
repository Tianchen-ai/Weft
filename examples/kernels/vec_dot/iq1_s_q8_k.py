from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ1_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
):
    result = wl.f32(0.0)
    grid_values = grid.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        group = wl.arange(0, 8, dtype=wl.u16, axis="group")
        entry = wl.arange(0, 4, dtype=wl.u16, axis="entry")
        payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
        metadata = w.qh[group]
        scale = wl.i32(
            (metadata >> wl.u16(12) & wl.u16(7)) * wl.u16(2) + wl.u16(1)
        )
        grid_index = wl.widen(w.q[group * wl.u16(4) + entry], wl.u16) | (
            metadata >> entry * wl.u16(3) & wl.u16(7)
        ) << wl.u16(8)
        weight = wl.lookup(
            grid_values, grid_index * wl.u16(8) + payload, bounds="in_bounds"
        )
        activation = x.q[
            group * wl.u16(32) + entry * wl.u16(8) + payload
        ]
        group_sum = wl.reduce_dot(activation, weight, over=("entry", "payload"), acc_dtype=wl.i32)
        delta = wl.i32((wl.i16(metadata) >> wl.i16(15)) | wl.i16(1))
        main = wl.reduce(scale * group_sum, axis="group")
        correction = wl.reduce(
            scale
            * delta
            * (
                wl.i32(x.bsum[group * wl.u16(2)])
                + wl.i32(x.bsum[group * wl.u16(2) + wl.u16(1)])
            ),
            axis="group",
        )
        combined = wl.f32(main) + wl.f32(0.125) * wl.f32(correction)
        result += wl.f32(w.d) * wl.f32(x.ds) * combined
    return result


@weft.kernel
def quantized_vec_dot_iq1_s_q8_k(
    W: wl.View[ggml.IQ1_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid))
