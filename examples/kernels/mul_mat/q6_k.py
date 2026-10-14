from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_q6_k(
    W: wl.View[ggml.Q6_K, (N, K)],
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
                        integer_acc = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                        with wl.level.subtiles(kb, extent=64) as pair:
                            q = wl.i8(w.ql[:, pair]) | wl.i8(w.qh[:, pair]) << wl.u8(4)
                            q -= wl.i8(32)
                            with wl.level.subtiles(pair, extent=16) as sub:
                                integer = wl.dot(x.q[:, sub], q[:, sub], over="k", acc_dtype=wl.i32)
                                integer_acc += integer * wl.i32(w.scales[:, sub])
                        acc += (
                            wl.f32(x.ds) * wl.f32(w.d) * wl.widen(integer_acc, wl.f32)
                        )
                    wl.store(Y[mb, nb], acc)
