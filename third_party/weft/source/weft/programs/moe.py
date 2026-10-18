from __future__ import annotations

import triton
import triton.language as wl

inf = wl.constexpr(float("inf"))


@triton.jit(model="weft", strided=("Logits", "Indices", "Weights"),
             alias_groups={"Logits": "input", "Indices": "indices", "Weights": "weights"})
def router(Logits: wl.View[wl.f32, (T, E)], Indices: wl.View[wl.i32, (T, K)],
            Weights: wl.View[wl.f32, (T, K)]):
    for token in range(T):
        for slot in range(K):
            best = wl.f32(-inf)
            chosen = wl.index(0)
            for expert in range(E):
                taken = wl.i1(False)
                for previous in range(slot):
                    taken = taken | (wl.load(Indices[token, previous]) == wl.i32(expert))
                if taken == False:
                    score = wl.load(Logits[token, expert])
                    if score > best:
                        best = score
                        chosen = expert
            wl.store(Indices[token, slot], wl.i32(chosen))
            wl.store(Weights[token, slot], best)
        maximum = wl.load(Weights[token, 0])
        total = wl.f32(0.0)
        for slot in range(K):
            value = wl.exp(wl.load(Weights[token, slot]) - maximum)
            wl.store(Weights[token, slot], value)
            total += value
        for slot in range(K):
            wl.store(Weights[token, slot], wl.load(Weights[token, slot]) / total)


@triton.jit(model="weft", strided=("X", "Indices", "Offsets", "Cursor", "Positions", "Grouped"),
             alias_groups={"X": "input", "Indices": "indices", "Offsets": "offsets",
                           "Cursor": "cursor", "Positions": "positions", "Grouped": "grouped"})
def group_tokens(X: wl.View[wl.f32, (T, D)], Indices: wl.View[wl.i32, (T, K)],
                  Offsets: wl.View[wl.i32, (EP,)], Cursor: wl.View[wl.i32, (E,)],
                  Positions: wl.View[wl.i32, (T, K)], Grouped: wl.View[wl.f32, (P, D)], *, BLOCK: wl.constexpr):
    for expert in range(E + 1):
        wl.store(Offsets[expert], wl.i32(0))
    for token in range(T):
        for slot in range(K):
            expert = wl.index(wl.load(Indices[token, slot]))
            wl.store(Offsets[expert + 1], wl.load(Offsets[expert + 1]) + wl.i32(1))
    for expert in range(E):
        wl.store(Offsets[expert + 1], wl.load(Offsets[expert + 1]) + wl.load(Offsets[expert]))
        wl.store(Cursor[expert], wl.load(Offsets[expert]))
    for token in range(T):
        for slot in range(K):
            expert = wl.index(wl.load(Indices[token, slot]))
            position = wl.index(wl.load(Cursor[expert]))
            wl.store(Positions[token, slot], wl.i32(position))
            wl.store(Cursor[expert], wl.i32(position + 1))
            with wl.level.blocks(D, extent=wl.auto("BLOCK")) as part:
                wl.store(Grouped[position, part], wl.load(X[token, part]))


@triton.jit(model="weft", strided=("X", "Weight", "Offsets", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Offsets": "input", "Y": "output"})
def expert_linear(X: wl.View[wl.f32, (P, K)], Weight: wl.View[wl.f32, (E, N, K)],
                    Offsets: wl.View[wl.i32, (EP,)], Y: wl.View[wl.f32, (P, N)], *, ROWS: wl.constexpr):
    for expert in range(E):
        begin = wl.index(wl.load(Offsets[expert]))
        end = wl.index(wl.load(Offsets[expert + 1]))
        if K == 0:
            for row in range(begin, end):
                for out in range(N):
                    wl.store(Y[row, out], wl.f32(0.0))
        else:
            if begin < end:
                count = end - begin
                if count == 1:
                    with wl.level.rows(N, group=1) as out:
                        value = wl.reduce_dot(wl.load(Weight[expert, out]), wl.load(X[begin]), over="k")
                        wl.store(Y[begin, out], value)
                else:
                    with wl.level.rows(N, group=wl.auto("ROWS")) as out:
                        activations = wl.subview(X, offsets=(begin, 0), extents=(count, K))
                        outputs = wl.subview(Y, offsets=(begin, 0), extents=(count, N))
                        value = wl.dot(wl.load(activations), wl.load(Weight[expert, out]), over="k")
                        wl.store(outputs[:, out], value)


@triton.jit(model="weft", strided=("Grouped", "Positions", "Weights", "Y"),
             alias_groups={"Grouped": "input", "Positions": "input", "Weights": "input", "Y": "output"})
def combine(Grouped: wl.View[wl.f32, (P, D)], Positions: wl.View[wl.i32, (T, K)],
             Weights: wl.View[wl.f32, (T, K)], Y: wl.View[wl.f32, (T, D)], *, BLOCK: wl.constexpr):
    for token in range(T):
        with wl.level.blocks(D, extent=wl.auto("BLOCK")) as part:
            accumulator = wl.state(wl.f32, [BLOCK], init=0)
            for slot in range(K):
                position = wl.index(wl.load(Positions[token, slot]))
                accumulator += wl.load(Grouped[position, part]) * wl.load(Weights[token, slot])
            wl.store(Y[token, part], accumulator)
