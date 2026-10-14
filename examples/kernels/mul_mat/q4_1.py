from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_1 as quant_q8_1
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_q4_1(
    W: wl.View[ggml.Q4_1, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_1, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_1.quantize_matrix(X, Xq)
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
                        integer = wl.dot(x.q, wl.i8(w.q), over="k", acc_dtype=wl.i32)
                        acc += wl.f32(w.d) * wl.f32(x.d) * wl.widen(
                            integer, wl.f32
                        ) + wl.f32(w.m) * wl.f32(x.s)
                    wl.store(Y[mb, nb], acc)
