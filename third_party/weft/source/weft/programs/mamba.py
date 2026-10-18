from __future__ import annotations

import triton
import triton.language as wl

from .activation import silu_value, softplus_value


@triton.jit(model="weft", strided=("X", "Weight", "Bias", "History", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Bias": "input",
                           "History": "history", "Y": "output"})
def causal_conv1d(X: wl.View[wl.f32, (B, T, D)], Weight: wl.View[wl.f32, (D, W)],
                   Bias: wl.View[wl.f32, (D,)], History: wl.View[wl.f32, (B, W, D)],
                   Y: wl.View[wl.f32, (B, T, D)], *, BLOCK: wl.constexpr):
    for batch in range(B):
        for token in range(T):
            with wl.level.blocks(D, extent=wl.auto("BLOCK")) as part:
                for lag in range(W - 1):
                    wl.store(History[batch, lag, part], wl.load(History[batch, lag + 1, part]))
                wl.store(History[batch, W - 1, part], wl.load(X[batch, token, part]))
                accumulator = wl.load(Bias[part])
                for lag in range(W):
                    accumulator += wl.load(History[batch, lag, part]) * wl.load(Weight[part, lag])
                wl.store(Y[batch, token, part], silu_value(accumulator))


@triton.jit(model="weft", strided=("X", "Delta", "Bcoef", "Ccoef", "Alog", "Skip", "Gate", "Bias", "State", "Y"),
             alias_groups={"X": "input", "Delta": "input", "Bcoef": "input", "Ccoef": "input",
                           "Alog": "input", "Skip": "input", "Gate": "input", "Bias": "input",
                           "State": "state", "Y": "output"})
def selective_scan(X: wl.View[wl.f32, (B, T, D)], Delta: wl.View[wl.f32, (B, T, D)],
                     Bcoef: wl.View[wl.f32, (B, T, N)], Ccoef: wl.View[wl.f32, (B, T, N)],
                     Alog: wl.View[wl.f32, (D, N)], Skip: wl.View[wl.f32, (D,)],
                     Gate: wl.View[wl.f32, (B, T, D)], Bias: wl.View[wl.f32, (D,)],
                     State: wl.View[wl.f32, (B, D, N)], Y: wl.View[wl.f32, (B, T, D)], *, N: wl.constexpr):
    for batch in range(B):
        for token in range(T):
            for channel in range(D):
                dt = softplus_value(wl.load(Delta[batch, token, channel]) + wl.load(Bias[channel]))
                value = wl.load(X[batch, token, channel])
                with wl.level.blocks(N, extent=wl.auto("N")) as dimensions:
                    decay = wl.exp(-wl.exp(wl.load(Alog[channel, dimensions])) * dt)
                    state = wl.load(State[batch, channel, dimensions])
                    updated = state * decay + wl.load(Bcoef[batch, token, dimensions]) * (dt * value)
                    wl.store(State[batch, channel, dimensions], updated)
                    total = wl.reduce(updated * wl.load(Ccoef[batch, token, dimensions]), op="add", axis="n")
                    gated = (total + wl.load(Skip[channel]) * value) * silu_value(wl.load(Gate[batch, token, channel]))
                    wl.store(Y[batch, token, channel], gated)
