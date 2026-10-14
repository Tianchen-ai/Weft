from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q1_0, (K,)], X: wl.View[ggml.Q8_0, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=128) as wb:
        w = wl.load(W[wb])
        subtotal = wl.f32(0.0)
        with wl.level.subtiles(wb, extent=32) as xb:
            x = wl.load(X[xb])
            centered = wl.i8(w.q[xb]) * wl.i8(2) - wl.i8(1)
            integer = wl.reduce_dot(x.q, centered, over="k", acc_dtype=wl.i32)
            subtotal += wl.f32(x.d) * wl.f32(integer)
        result += wl.f32(w.d) * subtotal
    return result


@weft.kernel
def quantized_vec_dot_q1_0_q8_0(
    W: wl.View[ggml.Q1_0, (K,)], X: wl.View[ggml.Q8_0, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
