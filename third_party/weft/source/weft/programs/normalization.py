from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Weight": "parameters"})
def rms_norm(X: wl.View[wl.f32, (M, N)], Weight: wl.View[wl.f32, (N,)],
             Y: wl.View[wl.f32, (M, N)], epsilon: wl.f32 = 1e-5, *, BLOCK: wl.constexpr):
    for row in range(M):
        row_value = wl.load(X[row])
        squares = wl.reduce(row_value * row_value, op="add", axis="n")
        inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part])
            weight = wl.load(Weight[part])
            wl.store(Y[row, part], value * inverse * weight)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "activation", "Y": "activation",
                           "Weight": "parameters", "Bias": "parameters"})
def layer_norm(X: wl.View[wl.f32, (M, N)], Weight: wl.View[wl.f32, (N,)],
               Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)],
               epsilon: wl.f32 = 1e-5, *, BLOCK: wl.constexpr):
    for row in range(M):
        total = wl.reduce(wl.load(X[row]), op="add", axis="n")
        mean = total / wl.f32(N)
        centered = wl.load(X[row]) - mean
        squares = wl.reduce(centered * centered, op="add", axis="n")
        inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part])
            weight = wl.load(Weight[part])
            bias = wl.load(Bias[part])
            wl.store(Y[row, part], (value - mean) * inverse * weight + bias)


@triton.jit(model="weft", strided=("X", "Residual", "Weight", "Y"),
             alias_groups={"X": "activation", "Residual": "activation",
                           "Y": "activation", "Weight": "parameters"})
def residual_rms_norm(X: wl.View[wl.f32, (M, N)], Residual: wl.View[wl.f32, (M, N)],
                      Weight: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)],
                      epsilon: wl.f32 = 1e-5, *, BLOCK: wl.constexpr):
    for row in range(M):
        row_value = wl.load(X[row]) + wl.load(Residual[row])
        squares = wl.reduce(row_value * row_value, op="add", axis="n")
        inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part]) + wl.load(Residual[row, part])
            wl.store(Y[row, part], value * inverse * wl.load(Weight[part]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Weight": "parameters"})
def rms_norm_row(X: wl.View[wl.f32, (M, N)], Weight: wl.View[wl.f32, (N,)],
                 Y: wl.View[wl.f32, (M, N)], epsilon: wl.f32 = 1e-5, *, N: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            value = wl.load(X[row, columns])
            squares = wl.reduce(value * value, op="add", axis="n")
            inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
            wl.store(Y[row, columns], value * inverse * wl.load(Weight[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "activation", "Y": "activation",
                           "Weight": "parameters", "Bias": "parameters"})
def layer_norm_row(X: wl.View[wl.f32, (M, N)], Weight: wl.View[wl.f32, (N,)],
                   Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)],
                   epsilon: wl.f32 = 1e-5, *, N: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            value = wl.load(X[row, columns])
            mean = wl.reduce(value, op="add", axis="n") / wl.f32(N)
            centered = value - mean
            squares = wl.reduce(centered * centered, op="add", axis="n")
            inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
            wl.store(Y[row, columns], centered * inverse * wl.load(Weight[columns]) + wl.load(Bias[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Weight": "parameters"})
def rms_norm_tile(X: wl.View[wl.f32, (M, N)], Weight: wl.View[wl.f32, (N,)],
                  Y: wl.View[wl.f32, (M, N)], epsilon: wl.f32 = 1e-5, *, N: wl.constexpr, ROWS: wl.constexpr):
    with wl.level.blocks(M, extent=wl.auto("ROWS")) as rows:
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            value = wl.load(X[rows, columns])
            squares = wl.reduce(value * value, op="add", axis="n")
            inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
            wl.store(Y[rows, columns], value * inverse * wl.load(Weight[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "activation", "Y": "activation",
                           "Weight": "parameters", "Bias": "parameters"})
def layer_norm_tile(X: wl.View[wl.f32, (M, N)], Weight: wl.View[wl.f32, (N,)],
                    Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)],
                    epsilon: wl.f32 = 1e-5, *, N: wl.constexpr, ROWS: wl.constexpr):
    with wl.level.blocks(M, extent=wl.auto("ROWS")) as rows:
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            value = wl.load(X[rows, columns])
            mean = wl.reduce(value, op="add", axis="n") / wl.f32(N)
            centered = value - mean
            squares = wl.reduce(centered * centered, op="add", axis="n")
            inverse = wl.rsqrt(squares / wl.f32(N) + epsilon)
            wl.store(Y[rows, columns], centered * inverse * wl.load(Weight[columns]) + wl.load(Bias[columns]))
