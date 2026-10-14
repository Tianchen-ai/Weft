from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_q1_0(
    W: wl.View[ggml.Q1_0, (N, K)],
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
                    with wl.level.blocks(K, extent=128) as wb:
                        w = wp[nb, wb]
                        with wl.level.subtiles(wb, extent=32) as xb:
                            x = xp[mb, xb]
                            centered = wl.i8(w.q[:, xb]) * wl.i8(2) - wl.i8(1)
                            integer = wl.dot(x.q, centered, over="k", acc_dtype=wl.i32)
                            acc += wl.f32(w.d) * wl.f32(x.d) * wl.widen(integer, wl.f32)
                    wl.store(Y[mb, nb], acc)
