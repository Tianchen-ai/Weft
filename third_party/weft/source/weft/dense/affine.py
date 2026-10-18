from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft")
def affine(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
           scale: wl.f32 = 1.0, *, bias: wl.f32 = 0.0, B: wl.constexpr = 64):
    with wl.level.blocks(N, extent=wl.auto("B")) as block:
        value = wl.load(X[block])
        wl.store(Y[block], value * scale + bias)


@triton.jit(model="weft")
def affine_grid(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
                scale: wl.f32 = 1.0, *, bias: wl.f32 = 0.0, B: wl.constexpr = 64):
    start = wl.program_id(0) * B
    if start < N:
        count = wl.minimum(B, N - start)
        x = wl.subview(X, offsets=(start,), extents=(count,))
        y = wl.subview(Y, offsets=(start,), extents=(count,))
        wl.store(y, wl.load(x) * scale + bias)
