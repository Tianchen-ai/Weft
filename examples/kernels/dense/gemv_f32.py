from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def gemv_f32(
    W: wl.View[wl.f32, (M, K)], X: wl.View[wl.f32, (K,)], Y: wl.View[wl.f32, (M,)]
):
    with wl.level.rows(M, group=wl.auto("MR")) as mb:
        acc = wl.state(wl.f32, [MR], init=0)
        with wl.level.blocks(K, extent=wl.auto("KB")) as kb:
            x = wl.load(X[kb])
            acc += wl.reduce_dot(wl.load(W[mb, kb]), x, over="k")
        wl.store(Y[mb], acc)
