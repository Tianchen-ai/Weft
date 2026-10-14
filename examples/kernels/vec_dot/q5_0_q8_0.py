from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q5_0, (K,)], X: wl.View[ggml.Q8_0, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        q = wl.u8(w.q) | wl.u8(w.qh) << wl.u8(4)
        centered = wl.i8(q) - wl.i8(16)
        integer = wl.reduce_dot(centered, x.q, over="k", acc_dtype=wl.i32)
        result += wl.f32(w.d) * wl.f32(x.d) * wl.f32(integer)
    return result


@weft.kernel
def quantized_vec_dot_q5_0_q8_0(
    W: wl.View[ggml.Q5_0, (K,)], X: wl.View[ggml.Q8_0, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
