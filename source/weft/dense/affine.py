from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def affine(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
           scale: wl.f32 = 1.0, *, bias: wl.f32 = 0.0):
    with wl.level.blocks(N, extent=wl.auto("B")) as block:
        value = wl.load(X[block])
        wl.store(Y[block], value * scale + bias)
