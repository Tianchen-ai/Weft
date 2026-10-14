from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q4_1, (K,)], X: wl.View[ggml.Q8_1, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        integer = wl.reduce_dot(w.q, x.q, over="k", acc_dtype=wl.i32)
        result += wl.f32(w.d) * wl.f32(x.d) * wl.f32(integer) + wl.f32(w.m) * wl.f32(
            x.s
        )
    return result


@weft.kernel
def quantized_vec_dot_q4_1_q8_1(
    W: wl.View[ggml.Q4_1, (K,)], X: wl.View[ggml.Q8_1, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
