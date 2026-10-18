"""Ordinary bounded numerical programs for range-sensitive physical lowering."""
from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("X", "Y"))
def exp_nonpositive(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], wl.exp(-wl.abs(wl.load(X[row, part]))))


@triton.jit(model="weft", strided=("X", "Y"))
def exp_clamped(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.maximum(wl.minimum(wl.load(X[row, part]), 87.0), -87.0)
            wl.store(Y[row, part], wl.exp(value))


@triton.jit(model="weft", strided=("X", "Y"))
def log_clamped(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.maximum(wl.minimum(wl.load(X[row, part]), 1.2676506002282294e30), 7.888609052210118e-31)
            wl.store(Y[row, part], wl.log(value))


@triton.jit(model="weft", strided=("X", "Y"))
def log_exp_normalizer(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], wl.log(1.0 + wl.exp(-wl.abs(wl.load(X[row, part])))))


@triton.jit(model="weft", strided=("X", "Y"))
def selected_exp(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part])
            upper = wl.select(value > 5.0, 5.0, value)
            bounded = wl.select(upper < -5.0, -5.0, upper)
            wl.store(Y[row, part], wl.exp(bounded))


@triton.jit(model="weft", strided=("X", "Y"))
def masked_exp(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part])
            in_range = (value >= -5.0) & (value <= 5.0)
            wl.store(Y[row, part], wl.exp(wl.select(in_range, value, 0.0)))


@triton.jit(model="weft", strided=("X", "Y"))
def complemented_exp(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part])
            outside = ((value >= -5.0) & (value <= 5.0)) ^ wl.i1(True)
            wl.store(Y[row, part], wl.exp(wl.select(outside, 0.0, value)))
