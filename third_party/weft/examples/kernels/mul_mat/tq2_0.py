from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import triton
import triton.language as wl


@triton.jit(model="weft")
def production_mul_mat_tq2_0(
    W: wl.View[ggml.TQ2_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
 *, MC: wl.constexpr, MR: wl.constexpr, NC: wl.constexpr, NR: wl.constexpr):
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
                        q = wl.i8(w.q) - wl.i8(1)
                        integer = wl.dot(x.q, q, over="k", acc_dtype=wl.i32)
                        acc += wl.f32(w.d) * wl.f32(x.ds) * wl.widen(integer, wl.f32)
                    wl.store(Y[mb, nb], acc)
