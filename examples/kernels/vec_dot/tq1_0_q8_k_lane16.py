from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.TQ1_0, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    powers: wl.View[wl.u32, (5,)],
):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        lane = wl.arange(0, 16, dtype=wl.u32, axis="k")
        tail = qf.radix3_digit_i8(
            powers, w.qh[lane % wl.u32(4)], lane // wl.u32(4)
        )
        partial = wl.widen(x.q[wl.u32(240) + lane], wl.i16) * wl.widen(
            tail, wl.i16
        )
        packed0 = w.q[lane]
        packed1 = w.q[wl.u32(16) + lane]
        packed2 = w.q[wl.u32(32) + lane]
        # Each lane sums 1 + 5 * 3 products of magnitude at most 128: 2048.
        for digit in range(5):
            q0 = qf.radix3_digit_i8(powers, packed0, wl.u32(digit))
            q1 = qf.radix3_digit_i8(powers, packed1, wl.u32(digit))
            q2 = qf.radix3_digit_i8(powers, packed2, wl.u32(digit))
            partial += wl.widen(
                x.q[wl.u32(digit) * wl.u32(32) + lane], wl.i16
            ) * wl.widen(q0, wl.i16)
            partial += wl.widen(
                x.q[wl.u32(digit) * wl.u32(32) + wl.u32(16) + lane], wl.i16
            ) * wl.widen(q1, wl.i16)
            partial += wl.widen(
                x.q[wl.u32(160) + wl.u32(digit) * wl.u32(16) + lane], wl.i16
            ) * wl.widen(q2, wl.i16)
        integer = wl.reduce(wl.widen(partial, wl.i32), axis="k")
        result += wl.f32(integer) * (wl.f32(w.d) * wl.f32(x.ds))
    return result


@weft.kernel
def quantized_vec_dot_tq1_0_q8_k_lane16(
    W: wl.View[ggml.TQ1_0, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    powers: wl.View[wl.u32, (5,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, powers))
