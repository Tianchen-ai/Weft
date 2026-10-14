from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.TQ2_0, (K,)], X: wl.View[ggml.Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        q = wl.i8(w.q) - wl.i8(1)
        integer = wl.reduce_dot(x.q, q, over="k", acc_dtype=wl.i32)
        scale = wl.f32(x.ds) * wl.f32(w.d)
        result += wl.widen(integer, wl.f32) * scale
    return result


@weft.kernel
def quantized_vec_dot_tq2_0_q8_k(
    W: wl.View[ggml.TQ2_0, (K,)], X: wl.View[ggml.Q8_K, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
