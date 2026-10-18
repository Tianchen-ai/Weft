from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("Grad", "DX"))
def affine_backward(Grad: wl.View[wl.f32, (M, N)], DX: wl.View[wl.f32, (M, N)],
                    scale: wl.f32 = 1.0):
    # Zero-stride broadcast gradients are ordinary read-only dense Views.
    with wl.level.rows(M, group=1) as row:
        with wl.level.blocks(N, extent=wl.auto(64)) as block:
            wl.store(DX[row, block], wl.load(Grad[row, block]) * scale)
