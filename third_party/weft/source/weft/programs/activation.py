from __future__ import annotations

import triton
import triton.language as wl


@triton.jit
def sigmoid_value(value):
    small = wl.exp(-wl.abs(value))
    numerator = wl.select(value >= 0.0, 1.0, small)
    return numerator / (1.0 + small)


@triton.jit
def silu_value(value):
    return value * sigmoid_value(value)


@triton.jit
def softplus_value(value):
    return wl.maximum(value, 0.0) + wl.log(1.0 + wl.exp(-wl.abs(value)))


@triton.jit(model="weft", strided=("X", "Y"))
def exponential(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], wl.exp(wl.load(X[row, part])))


@triton.jit(model="weft", strided=("X", "Y"))
def logarithm(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], wl.log(wl.load(X[row, part])))


@triton.jit(model="weft", strided=("X", "Y"))
def softplus(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], softplus_value(wl.load(X[row, part])))


@triton.jit(model="weft", strided=("X", "Y"))
def silu(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], silu_value(wl.load(X[row, part])))


@triton.jit(model="weft", strided=("Gate", "Up", "Y"))
def swiglu(Gate: wl.View[wl.f32, (M, N)], Up: wl.View[wl.f32, (M, N)],
            Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = silu_value(wl.load(Gate[row, part])) * wl.load(Up[row, part])
            wl.store(Y[row, part], value)


@triton.jit(model="weft", strided=("X", "Residual", "Y"))
def add(X: wl.View[wl.f32, (M, N)], Residual: wl.View[wl.f32, (M, N)],
         Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], wl.load(X[row, part]) + wl.load(Residual[row, part]))


@triton.jit(model="weft", strided=("X", "Y"))
def copy(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            wl.store(Y[row, part], wl.load(X[row, part]))


@triton.jit(model="weft", strided=("Y",))
def fill(Y: wl.View[wl.f32, (M, N)], value: wl.f32 = 0.0, *, BLOCK: wl.constexpr, ROWS: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("ROWS")) as row:
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            tile = wl.state(wl.f32, [ROWS, BLOCK], init=value)
            wl.store(Y[row, part], tile)


@triton.jit(model="weft", strided=("Condition", "X", "Other", "Y"),
             alias_groups={"Condition": "condition", "X": "activation", "Other": "activation", "Y": "activation"})
def where(Condition: wl.View[wl.i1, (M, N)], X: wl.View[wl.f32, (M, N)],
           Other: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            condition = wl.load(Condition[row, part])
            selected = wl.load(X[row, part])
            other = wl.load(Other[row, part])
            wl.store(Y[row, part], wl.select(condition, selected, other))
