from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q3_k(
    W: wl.View[ggml.Q3_K, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
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
                        decoded_scales = wl.i8(
                            (wl.u8(w.scale_low) | wl.u8(w.scale_high) << wl.u8(4))
                            - wl.u8(32)
                        )
                        integer_acc = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
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
                            q = wl.i8(
                                wl.u8(w.q[:, coordinate])
                                | wl.u8(w.hmask[:, coordinate]) << wl.u8(2)
                            ) - wl.i8(4)
                            products = wl.reduce_dot(x.q[:, coordinate], q, over="k", acc_dtype=wl.i32)
                            scale_coordinate = (
                                wl.u8(half_index) * wl.u8(8)
                                + plane * wl.u8(2)
                                + scale_part
                            )
                            scale = wl.i32(decoded_scales[:, scale_coordinate])
                            integer_acc += wl.reduce(
                                wl.reduce(products * scale, axis=1), axis=1
                            )
                            half_index += wl.index(1)
                        acc += (
                            wl.f32(x.ds) * wl.f32(w.d) * wl.widen(integer_acc, wl.f32)
                        )
                    wl.store(Y[mb, nb], acc)
