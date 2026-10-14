from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_tq1_0(
    W: wl.View[ggml.TQ1_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    powers: wl.View[wl.u32, (5,)],
    Y: wl.View[wl.f32, (M, N)],
):
    power_table = wl.stage(wl.load(powers))
    quant_q8_k.quantize_matrix(X, Xq)
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        wp = wl.stage(wl.load(W[nc]))
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            xp = wl.stage(wl.load(Xq[mc]))
            with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    acc = wl.state(wl.f32, [MR, NR], init=wl.f32(0.0))
                    with wl.level.blocks(K, extent=256) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        integer = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                        lane0 = wl.arange(0, 32, axis="k")
                        digit0 = wl.arange(0, 5, dtype=wl.u32, axis="digit0")
                        q0 = qf.radix3_digit_i8(power_table, w.q[:, lane0], digit0)
                        partial0 = wl.reduce_dot(
                            x.q[:, digit0 * wl.u32(32) + lane0],
                            q0,
                            over="k",
                            acc_dtype=wl.i32,
                        )
                        integer += wl.reduce(partial0, axis="digit0")
                        lane1 = wl.arange(0, 16, axis="k")
                        digit1 = wl.arange(0, 5, dtype=wl.u32, axis="digit1")
                        q1 = qf.radix3_digit_i8(
                            power_table, w.q[:, wl.u32(32) + lane1], digit1
                        )
                        partial1 = wl.reduce_dot(
                            x.q[:, wl.u32(160) + digit1 * wl.u32(16) + lane1],
                            q1,
                            over="k",
                            acc_dtype=wl.i32,
                        )
                        integer += wl.reduce(partial1, axis="digit1")
                        lane2 = wl.arange(0, 16, axis="k")
                        q2 = qf.radix3_digit_i8(
                            power_table, w.qh[:, lane2 // wl.u32(4)], lane2 % wl.u32(4)
                        )
                        integer += wl.dot(x.q[
                                :,
                                wl.u32(240)
                                + lane2 % wl.u32(4) * wl.u32(4)
                                + lane2 // wl.u32(4),
                            ], q2, over="k", acc_dtype=wl.i32)
                        acc += wl.f32(w.d) * wl.f32(x.ds) * wl.widen(integer, wl.f32)
                    wl.store(Y[mb, nb], acc)
