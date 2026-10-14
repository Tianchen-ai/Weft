from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q2_k(
    W: wl.View[ggml.Q2_K, (N, K)],
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
                        low_scales = wl.stage(
                            wl.u8(wp[nb, kb].scales) & wl.u8(15)
                        )
                        integer = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                        with wl.level.subtiles(kb, extent=16) as sub:
                            partial = wl.dot(x.q[:, sub], w.q[:, sub], over="k", acc_dtype=wl.i32)
                            scale = wl.i32(low_scales[:, sub])
                            integer += partial * scale
                        mins = wl.widen(wl.u8(w.scales) >> wl.u8(4), wl.i16)
                        correction = wl.dot(x.bsum, mins, over="k", acc_dtype=wl.i32)
                        acc += wl.f32(x.ds) * (
                            wl.f32(w.d) * wl.widen(integer, wl.f32)
                            - wl.f32(w.dmin) * wl.widen(correction, wl.f32)
                        )
                    wl.store(Y[mb, nb], acc)
