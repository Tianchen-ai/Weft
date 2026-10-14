from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_iq1_m(
    W: wl.View[ggml.IQ1_M, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    grid_values = grid.values
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

                        sc0 = wl.u32(w.scales[:, 0]) | wl.u32(w.scales[:, 1]) << wl.u32(8)
                        sc1 = wl.u32(w.scales[:, 2]) | wl.u32(w.scales[:, 3]) << wl.u32(8)
                        sc2 = wl.u32(w.scales[:, 4]) | wl.u32(w.scales[:, 5]) << wl.u32(8)
                        sc3 = wl.u32(w.scales[:, 6]) | wl.u32(w.scales[:, 7]) << wl.u32(8)
                        scale_bits = sc0 >> wl.u32(12) | sc1 >> wl.u32(8) & wl.u32(240)
                        scale_bits = scale_bits | sc2 >> wl.u32(4) & wl.u32(3840)
                        scale_bits = scale_bits | sc3 & wl.u32(61440)
                        block_scale = qf.nonlinear_lookup(f16_bits, scale_bits)

                        group = wl.arange(0, 8, dtype=wl.u32, axis="group")
                        scale_part = wl.arange(0, 2, dtype=wl.u32, axis="scale_part")
                        entry = wl.arange(0, 2, dtype=wl.u32, axis="entry")
                        payload = wl.arange(0, 8, dtype=wl.u32, axis="payload")
                        qh = wl.widen(w.qh[:, group * wl.u32(2) + scale_part], wl.u32)
                        grid_index = wl.widen(
                            w.q[
                                :,
                                group * wl.u32(4)
                                + scale_part * wl.u32(2)
                                + entry,
                            ],
                            wl.u32,
                        ) | (qh >> entry * wl.u32(4) & wl.u32(7)) << wl.u32(8)
                        weight = wl.lookup(
                            grid_values,
                            grid_index * wl.u32(8) + payload,
                            bounds="in_bounds",
                        )
                        activation = x.q[
                            :,
                            group * wl.u32(32)
                            + scale_part * wl.u32(16)
                            + entry * wl.u32(8)
                            + payload,
                        ]
                        dot = wl.reduce_dot(activation, weight, over=("entry", "payload"), acc_dtype=wl.i32)
                        scale_word = wl.widen(
                            w.scales[:, group // wl.u32(2) * wl.u32(2)], wl.u32
                        ) | wl.widen(
                            w.scales[
                                :,
                                group // wl.u32(2) * wl.u32(2) + wl.u32(1),
                            ],
                            wl.u32,
                        ) << wl.u32(8)
                        scale_shift = group % wl.u32(2) * wl.u32(6) + scale_part * wl.u32(3)
                        scale = wl.i32(
                            (scale_word >> scale_shift & wl.u32(7)) * wl.u32(2)
                            + wl.u32(1)
                        )
                        main = wl.reduce(
                            wl.reduce(dot * scale, axis="scale_part"), axis="group"
                        )

                        delta_shift = wl.u32(3) + entry * wl.u32(4)
                        delta_bit_u16 = wl.narrow(
                            qh >> delta_shift & wl.u32(1),
                            wl.u16,
                            rounding="rtz",
                            saturation=False,
                        )
                        delta_bit = wl.narrow(
                            delta_bit_u16,
                            wl.u8,
                            rounding="rtz",
                            saturation=False,
                        )
                        delta = wl.i8(1) - wl.i8(delta_bit) * wl.i8(2)
                        payload_u16 = wl.narrow(
                            payload, wl.u16, rounding="rtz", saturation=False
                        )
                        payload_u8 = wl.narrow(
                            payload_u16,
                            wl.u8,
                            rounding="rtz",
                            saturation=False,
                        )
                        delta = delta + wl.i8(payload_u8) * wl.i8(0)
                        correction_part = wl.reduce_dot(
                            activation,
                            delta,
                            over=("entry", "payload"),
                            acc_dtype=wl.i32,
                        )
                        correction = wl.reduce(
                            wl.reduce(correction_part * scale, axis="scale_part"),
                            axis="group",
                        )
                        combined = wl.f32(main) + wl.f32(0.125) * wl.f32(correction)
                        acc += wl.f32(block_scale) * wl.f32(x.ds) * combined
                    wl.store(Y[mb, nb], acc)
