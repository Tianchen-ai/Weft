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
    power_table = wl.stage(wl.load(powers))
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        lane0 = wl.arange(0, 32, axis="k")
        digit0 = wl.arange(0, 5, dtype=wl.u32, axis="digit0")
        q0 = qf.radix3_digit_i8(power_table, w.q[lane0], digit0)
        partial0 = wl.reduce_dot(x.q[digit0 * wl.u32(32) + lane0], q0, over="k", acc_dtype=wl.i32)
        integer = wl.reduce(partial0, axis="digit0")
        lane1 = wl.arange(0, 16, axis="k")
        digit1 = wl.arange(0, 5, dtype=wl.u32, axis="digit1")
        q1 = qf.radix3_digit_i8(power_table, w.q[wl.u32(32) + lane1], digit1)
        partial1 = wl.reduce_dot(
            x.q[wl.u32(160) + digit1 * wl.u32(16) + lane1],
            q1,
            over="k",
            acc_dtype=wl.i32,
        )
        integer += wl.reduce(partial1, axis="digit1")
        lane2 = wl.arange(0, 4, axis="k")
        digit2 = wl.arange(0, 4, dtype=wl.u32, axis="digit2")
        q2 = qf.radix3_digit_i8(power_table, w.qh[lane2], digit2)
        partial2 = wl.reduce_dot(
            x.q[wl.u32(240) + digit2 * wl.u32(4) + lane2],
            q2,
            over="k",
            acc_dtype=wl.i32,
        )
        integer += wl.reduce(partial2, axis="digit2")
        result += wl.f32(integer) * (wl.f32(w.d) * wl.f32(x.ds))
    return result


@weft.kernel
def quantized_vec_dot_tq1_0_q8_k(
    W: wl.View[ggml.TQ1_0, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    powers: wl.View[wl.u32, (5,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, powers))
