from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q8_0(
    W: wl.View[ggml.Q8_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_0, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        wp = wl.stage(wl.load(W[nc]))
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            xp = wl.stage(wl.load(Xq[mc]))
            with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    acc = wl.state(wl.f32, [MR, NR], init=wl.f32(0.0))
                    with wl.level.blocks(K, extent=32) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        integer = wl.dot(x.q, w.q, over="k", acc_dtype=wl.i32)
                        acc += wl.f32(w.d) * wl.f32(x.d) * wl.widen(integer, wl.f32)
                    wl.store(Y[mb, nb], acc)
