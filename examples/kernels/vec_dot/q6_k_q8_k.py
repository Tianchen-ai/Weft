from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q6_K, (K,)], X: wl.View[ggml.Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        integer_acc = wl.i32(0)
        half_index = wl.index(0)
        with wl.level.subtiles(kb, extent=128) as half:
            plane = wl.arange(0, 4, dtype=wl.u8)
            scale_part = wl.arange(0, 2, dtype=wl.u8)
            lane = wl.arange(0, 16, dtype=wl.u8, axis="k")
            coordinate = (
                wl.u8(half_index) * wl.u8(128)
                + plane * wl.u8(32)
                + scale_part * wl.u8(16)
                + lane
            )
            q = wl.i8(w.ql[coordinate]) | wl.i8(w.qh[coordinate]) << wl.u8(4)
            q -= wl.i8(32)
            products = wl.reduce_dot(x.q[coordinate], q, over="k", acc_dtype=wl.i32)
            scale_coordinate = (
                wl.u8(half_index) * wl.u8(8) + plane * wl.u8(2) + scale_part
            )
            integer_acc += wl.reduce(
                wl.reduce(products * wl.i32(w.scales[scale_coordinate]), axis=1), axis=0
            )
            half_index += wl.index(1)
        result += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(integer_acc)
    return result


@weft.kernel
def quantized_vec_dot_q6_k_q8_k(
    W: wl.View[ggml.Q6_K, (K,)], X: wl.View[ggml.Q8_K, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))
