from __future__ import annotations

import triton
import triton.language as wl
from .activation import silu_value


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def batched_matmul(X: wl.View[wl.f32, (B, M, K)], Weight: wl.View[wl.f32, (B, K, N)],
                   Y: wl.View[wl.f32, (B, M, N)]):
    wl.store(Y, wl.reduce_dot(wl.load(X), wl.load(Weight), over="k"))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def matmul(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
           Y: wl.View[wl.f32, (M, N)], *, MC: wl.constexpr, NC: wl.constexpr):
    with wl.level.tiles(M, extent=wl.auto("MC")) as rows:
        with wl.level.tiles(N, extent=wl.auto("NC")) as columns:
            value = wl.dot(wl.load(X[rows]), wl.load(Weight[:, columns]), over="k")
            wl.store(Y[rows, columns], value)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def matmul_bias(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)], *, MC: wl.constexpr, NC: wl.constexpr):
    with wl.level.tiles(M, extent=wl.auto("MC")) as rows:
        with wl.level.tiles(N, extent=wl.auto("NC")) as columns:
            value = wl.dot(wl.load(X[rows]), wl.load(Weight[:, columns]), over="k")
            wl.store(Y[rows, columns], value + wl.load(Bias[columns]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def linear_silu(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                 Y: wl.View[wl.f32, (M, N)], *, MC: wl.constexpr, NC: wl.constexpr):
    with wl.level.tiles(M, extent=wl.auto("MC")) as rows:
        with wl.level.tiles(N, extent=wl.auto("NC")) as columns:
            value = wl.dot(wl.load(X[rows]), wl.load(Weight[columns]), over="k")
            wl.store(Y[rows, columns], silu_value(value))


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def linear_bias_silu(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                      Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)], *, MC: wl.constexpr, NC: wl.constexpr):
    with wl.level.tiles(M, extent=wl.auto("MC")) as rows:
        with wl.level.tiles(N, extent=wl.auto("NC")) as columns:
            value = wl.dot(wl.load(X[rows]), wl.load(Weight[columns]), over="k") + wl.load(Bias[columns])
            wl.store(Y[rows, columns], silu_value(value))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def linear(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
            Y: wl.View[wl.f32, (M, N)], *, MC: wl.constexpr, MR: wl.constexpr, NC: wl.constexpr, NR: wl.constexpr):
    if M == 1:
        with wl.level.cols(N, group=1) as out:
            wl.store(Y[0, out], wl.reduce_dot(wl.load(Weight[out]), wl.load(X[0]), over="k"))
    else:
        with wl.level.tiles(N, extent=wl.auto("NC")) as columns:
            with wl.level.tiles(M, extent=wl.auto("MC")) as rows:
                with wl.level.cols(columns, group=wl.auto("NR")) as out:
                    with wl.level.rows(rows, group=wl.auto("MR")) as row:
                        value = wl.dot(wl.load(X[row]), wl.load(Weight[out]), over="k")
                        wl.store(Y[row, out], value)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def linear_bias(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                 Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)], *, MC: wl.constexpr, MR: wl.constexpr, NC: wl.constexpr, NR: wl.constexpr):
    if M == 1:
        with wl.level.cols(N, group=1) as out:
            wl.store(Y[0, out], wl.reduce_dot(wl.load(Weight[out]), wl.load(X[0]), over="k") + wl.load(Bias[out]))
    else:
        with wl.level.tiles(N, extent=wl.auto("NC")) as columns:
            with wl.level.tiles(M, extent=wl.auto("MC")) as rows:
                with wl.level.cols(columns, group=wl.auto("NR")) as out:
                    with wl.level.rows(rows, group=wl.auto("MR")) as row:
                        value = wl.dot(wl.load(X[row]), wl.load(Weight[out]), over="k")
                        wl.store(Y[row, out], value + wl.load(Bias[out]))


@triton.jit(model="weft", strided=("X", "Weight", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Y": "output"})
def linear_tile(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr, MROWS: wl.constexpr, NROWS: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("MROWS")) as row:
        with wl.level.cols(N, group=wl.auto("NROWS")) as out:
            accumulator = wl.state(wl.f32, [MROWS, NROWS], init=0)
            with wl.level.blocks(K, extent=wl.auto("BLOCK")) as part:
                accumulator += wl.dot(wl.load(X[row, part]), wl.load(Weight[out, part]), over="k")
            wl.store(Y[row, out], accumulator)


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "inputs", "Weight": "inputs", "Bias": "inputs", "Y": "output"})
def linear_bias_tile(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                     Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr, MROWS: wl.constexpr, NROWS: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("MROWS")) as row:
        with wl.level.cols(N, group=wl.auto("NROWS")) as out:
            accumulator = wl.state(wl.f32, [MROWS, NROWS], init=0)
            with wl.level.blocks(K, extent=wl.auto("BLOCK")) as part:
                accumulator += wl.dot(wl.load(X[row, part]), wl.load(Weight[out, part]), over="k")
            wl.store(Y[row, out], accumulator + wl.load(Bias[out]))
