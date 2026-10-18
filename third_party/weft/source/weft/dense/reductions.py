from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def statistics(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (3,)]):
    value = wl.load(X)
    total = wl.reduce(value, op="add", axis="n")
    maximum = wl.reduce(value, op="max", axis="n")
    minimum = wl.reduce(value, op="min", axis="n")
    wl.store(Y[0], total)
    wl.store(Y[1], maximum)
    wl.store(Y[2], minimum)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def product_sum(X: wl.View[wl.f32, (N,)], Weight: wl.View[wl.f32, (N,)],
                Y: wl.View[wl.f32, (1,)], *, BLOCK: wl.constexpr):
    total = wl.f32(0.0)
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        product = wl.load(X[part]) * wl.load(Weight[part])
        total += wl.reduce(product, op="add", axis="n")
    wl.store(Y[0], total)


@triton.jit(model="weft", alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dot_f16(X: wl.View[wl.f16, (N,)], Weight: wl.View[wl.f16, (N,)],
            Y: wl.View[wl.f32, (1,)], *, BLOCK: wl.constexpr):
    total = wl.f32(0.0)
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        total += wl.reduce_dot(wl.f32(wl.load(X[part])), wl.f32(wl.load(Weight[part])), over="n")
    wl.store(Y[0], total)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def mixed_dot_f16(X: wl.View[wl.f32, (N,)], Weight: wl.View[wl.f16, (N,)],
                  Y: wl.View[wl.f32, (1,)], *, BLOCK: wl.constexpr):
    total = wl.f32(0.0)
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        total += wl.reduce_dot(wl.load(X[part]), wl.f32(wl.load(Weight[part])), over="n")
    wl.store(Y[0], total)


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def mixed_dot_bf16(X: wl.View[wl.f32, (N,)], Weight: wl.View[wl.bf16, (N,)],
                   Y: wl.View[wl.f32, (1,)], *, BLOCK: wl.constexpr):
    total = wl.f32(0.0)
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        total += wl.reduce_dot(wl.load(X[part]), wl.f32(wl.load(Weight[part])), over="n")
    wl.store(Y[0], total)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def transformed_statistics(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, 3)],
                           shift: wl.f32, scale: wl.f32, *, N: wl.constexpr, ROWS: wl.constexpr):
    with wl.level.blocks(M, extent=wl.auto("ROWS")) as rows:
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            transformed = (wl.load(X[rows, columns]) - shift) * scale
            squared = transformed * transformed
            total = wl.reduce(squared, op="add", axis="n")
            maximum = wl.reduce(squared, op="max", axis="n")
            minimum = wl.reduce(squared, op="min", axis="n")
            wl.store(Y[rows, 0], total)
            wl.store(Y[rows, 1], maximum)
            wl.store(Y[rows, 2], minimum)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def dynamic_statistics(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, 3)],
                       shift: wl.f32, scale: wl.f32):
    value = (wl.load(X) - shift) * scale
    squared = value * value
    total = wl.reduce(squared, op="add", axis="n")
    maximum = wl.reduce(squared, op="max", axis="n")
    minimum = wl.reduce(squared, op="min", axis="n")
    wl.store(Y[:, 0], total)
    wl.store(Y[:, 1], maximum)
    wl.store(Y[:, 2], minimum)
