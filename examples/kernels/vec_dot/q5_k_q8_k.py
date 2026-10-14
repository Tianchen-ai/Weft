from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q5_K, (K,)], X: wl.View[ggml.Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        integer = wl.i32(0)
        with wl.level.subtiles(extent=32) as sub:
            q = wl.u8(w.q[sub]) | wl.u8(w.qh[sub]) << wl.u8(4)
            integer += wl.reduce_dot(q, x.q[sub], over="k", acc_dtype=wl.i32) * wl.i32(
                w.sc[sub]
            )
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
            wl.f32(w.d) * wl.f32(integer) - wl.f32(w.dmin) * wl.f32(minimum)
        )
    return result


@weft.kernel
def quantized_vec_dot_q5_k_q8_k(
    W: wl.View[ggml.Q5_K, (K,)], X: wl.View[ggml.Q8_K, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
