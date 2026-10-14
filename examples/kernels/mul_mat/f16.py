from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_f16(
    W: wl.View[wl.f16, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xh: wl.View[wl.f16, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    for row in range(M):
        with wl.level.blocks(K, extent=1) as kb:
            value = wl.load(X[row, kb])
            wl.store(Xh[row, kb], wl.f16(value))
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], wl.f32(0.0))
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.level.tiles(K, extent=wl.auto("KC")) as kc:
            wp = wl.stage(wl.load(W[nc, kc]))
            with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
                xp = wl.stage(wl.load(Xh[mc, kc]))
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                        acc = wl.state(wl.f32, [MR, NR], init=wl.load(Y[mb, nb]))
                        with wl.level.blocks(kc, extent=wl.auto("KB")) as kb:
                            weights = wp[nb, kb]
                            activations = xp[mb, kb]
                            acc += wl.dot(activations, weights, over="k", acc_dtype=wl.f32)
                        wl.store(Y[mb, nb], acc)
