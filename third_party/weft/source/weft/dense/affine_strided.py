from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("X", "Y"))
def affine_strided(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)],
                   scale: wl.f32 = 1.0, *, bias: wl.f32 = 0.0, B: wl.constexpr):
    with wl.level.rows(M, group=1) as row:
        with wl.level.blocks(N, extent=wl.auto("B")) as block:
            wl.store(Y[row, block], wl.load(X[row, block]) * scale + bias)
