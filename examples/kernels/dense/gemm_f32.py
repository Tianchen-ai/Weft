from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def gemm_f32(
    X: wl.View[wl.f32, (M, K)], W: wl.View[wl.f32, (N, K)], Y: wl.View[wl.f32, (M, N)]
):
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    product = wl.dot(wl.load(X[mb]), wl.load(W[nb]), over="k")
                    wl.store(Y[mb, nb], product)
