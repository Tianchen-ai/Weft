from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


@weft.kernel
def q4_k_q8_k_gemv_contract(
    W: wl.View[ggml.Q4K_I[16], (M, K)],
    X: wl.View[ggml.Q8_K, (K,)],
    Y: wl.View[wl.f32, (M,)],
):
    with wl.level.rows(M, group=16) as mb:
        f32_acc = wl.state(wl.f32, [16], init=0)
        with wl.level.blocks(K, extent=256) as kb:
            w = wl.load(W[mb, kb])
            x = wl.load(X[kb])
            i32_acc = wl.state(wl.i32, [16], init=0)
            with wl.level.subtiles(extent=32) as s:
                p32 = wl.reduce_dot(w.q[s], x.q[s], over="k", acc_dtype=wl.i32)
                i32_acc += p32 * w.sc[s]
            mins = wl.sum_pairs(x.bsum)
            min_term = wl.reduce_dot(w.m, mins, over="k")
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        wl.store(Y[mb], f32_acc)
