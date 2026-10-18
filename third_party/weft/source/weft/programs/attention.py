from __future__ import annotations

import triton
import triton.language as wl

inf = wl.constexpr(float("inf"))


@triton.jit(model="weft", strided=("X", "Y"))
def softmax(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, N: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            maximum = wl.reduce(wl.load(X[row, columns]), op="max", axis="n")
            value = wl.load(X[row, columns])
            exponential = wl.exp(value - maximum)
            denominator = wl.reduce(exponential, op="add", axis="n")
            inverse = wl.f32(1.0) / denominator
            wl.store(Y[row, columns], exponential * inverse)


@triton.jit(model="weft", strided=("X", "Y"))
def softmax_row(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)], *, N: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("N")) as columns:
            value = wl.load(X[row, columns])
            maximum = wl.reduce(value, op="max", axis="n")
            exponential = wl.exp(value - maximum)
            total = wl.reduce(exponential, op="add", axis="n")
            inverse = wl.f32(1.0) / total
            wl.store(Y[row, columns], exponential * inverse)


@triton.jit(model="weft", strided=("X", "Cos", "Sin", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Cos": "cache", "Sin": "cache"})
def rope_split(X: wl.View[wl.f32, (T, H, 2, P)], Cos: wl.View[wl.f32, (T, P)],
               Sin: wl.View[wl.f32, (T, P)], Y: wl.View[wl.f32, (T, H, 2, P)], *, BLOCK: wl.constexpr):
    for token in range(T):
        for head in range(H):
            with wl.level.blocks(P, extent=wl.auto("BLOCK")) as part:
                first = wl.load(X[token, head, 0, part])
                second = wl.load(X[token, head, 1, part])
                cosine = wl.load(Cos[token, part])
                sine = wl.load(Sin[token, part])
                wl.store(Y[token, head, 0, part], first * cosine - second * sine)
                wl.store(Y[token, head, 1, part], second * cosine + first * sine)


@triton.jit(model="weft", strided=("X", "Cos", "Sin", "Y"),
             alias_groups={"X": "activation", "Y": "activation", "Cos": "cache", "Sin": "cache"})
def rope_interleaved(X: wl.View[wl.f32, (T, H, P, 2)], Cos: wl.View[wl.f32, (T, P)],
                     Sin: wl.View[wl.f32, (T, P)], Y: wl.View[wl.f32, (T, H, P, 2)], *, BLOCK: wl.constexpr):
    for token in range(T):
        for head in range(H):
            with wl.level.blocks(P, extent=wl.auto("BLOCK")) as part:
                first = wl.load(X[token, head, part, 0])
                second = wl.load(X[token, head, part, 1])
                cosine = wl.load(Cos[token, part])
                sine = wl.load(Sin[token, part])
                wl.store(Y[token, head, part, 0], first * cosine - second * sine)
                wl.store(Y[token, head, part, 1], second * cosine + first * sine)


@triton.jit(model="weft", strided=("K", "V", "KeyCache", "ValueCache"),
             alias_groups={"K": "input", "V": "input", "KeyCache": "keys", "ValueCache": "values"})
def kv_update(K: wl.View[wl.f32, (B, T, G, D)], V: wl.View[wl.f32, (B, T, G, D)],
               KeyCache: wl.View[wl.f32, (B, S, G, D)], ValueCache: wl.View[wl.f32, (B, S, G, D)],
               start: wl.index, *, BLOCK: wl.constexpr):
    for batch in range(B):
        for token in range(T):
            for head in range(G):
                with wl.level.blocks(D, extent=wl.auto("BLOCK")) as part:
                    wl.store(KeyCache[batch, start + token, head, part], wl.load(K[batch, token, head, part]))
                    wl.store(ValueCache[batch, start + token, head, part], wl.load(V[batch, token, head, part]))


@triton.jit(model="weft", strided=("Q", "K", "V", "Y"),
             alias_groups={"Q": "input", "K": "input", "V": "input", "Y": "output"})
def causal_attention(Q: wl.View[wl.f32, (B, T, H, D)], K: wl.View[wl.f32, (B, S, G, D)],
                      V: wl.View[wl.f32, (B, S, G, D)], Y: wl.View[wl.f32, (B, T, H, D)],
                      start: wl.index, scale: wl.f32, *, D: wl.constexpr):
    for batch in range(B):
        for token in range(T):
            for head in range(H):
                group = head / (H / G)
                with wl.level.blocks(D, extent=wl.auto("D")) as dimensions:
                    query = wl.stage(wl.load(Q[batch, token, head, dimensions]))
                    accumulator = wl.state(wl.f32, [D], init=0)
                    maximum = wl.f32(-inf)
                    denominator = wl.f32(0.0)
                    for key in range(start + token + 1):
                        score = wl.reduce_dot(query, wl.load(K[batch, key, group, dimensions]), over="d") * scale
                        next_max = wl.maximum(maximum, score)
                        keep = wl.exp(maximum - next_max)
                        probability = wl.exp(score - next_max)
                        accumulator = accumulator * keep + wl.load(V[batch, key, group, dimensions]) * probability
                        denominator = denominator * keep + probability
                        maximum = next_max
                    wl.store(Y[batch, token, head, dimensions], accumulator / denominator)
