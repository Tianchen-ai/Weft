from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q4_K, (K,)], X: wl.View[ggml.Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        group = wl.arange(0, 8, dtype=wl.u8, axis="group")
        lane = wl.arange(0, 32, dtype=wl.u8, axis="k")
        coordinate = group * wl.u8(32) + lane
        partial = wl.reduce_dot(
            w.q[coordinate], x.q[coordinate], over="k", acc_dtype=wl.i32
        )
        total = wl.reduce(partial * wl.i32(w.sc[group]), axis="group")
        min_group = wl.arange(0, 8, dtype=wl.u32, axis="min_group")
        minimum = wl.reduce(
            wl.widen(w.m[min_group], wl.i32)
            * (
                wl.widen(x.bsum[min_group * wl.u32(2)], wl.i32)
                + wl.widen(x.bsum[min_group * wl.u32(2) + wl.u32(1)], wl.i32)
            ),
            axis="min_group",
        )
        result += wl.f32(x.ds) * (
            wl.f32(w.d) * wl.f32(total) - wl.f32(w.dmin) * wl.f32(minimum)
        )
    return result


@weft.kernel
def quantized_vec_dot_q4_k_q8_k(
    W: wl.View[ggml.Q4_K, (K,)], X: wl.View[ggml.Q8_K, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
