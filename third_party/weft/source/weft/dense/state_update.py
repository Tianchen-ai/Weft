from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("State", "Delta"))
def state_update(State: wl.View[wl.f32, (M, N)], Delta: wl.View[wl.f32, (M, N)],
                 decay: wl.f32 = 1.0, *, B: wl.constexpr):
    with wl.level.rows(M, group=1) as row:
        with wl.level.blocks(N, extent=wl.auto("B")) as block:
            state = wl.load(State[row, block])
            delta = wl.load(Delta[row, block])
            wl.store(State[row, block], state * decay + delta)


@triton.jit(model="weft", strided=("State",))
def state_scale(State: wl.View[wl.f32, (M, N)], scale: wl.f32 = 1.0, *, B: wl.constexpr):
    with wl.level.rows(M, group=1) as row:
        with wl.level.blocks(N, extent=wl.auto("B")) as block:
            wl.store(State[row, block], wl.load(State[row, block]) * scale)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "activation", "Y": "activation"})
def recurrent_accumulate(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)],
                         steps: wl.index, decay: wl.f32 = 0.5, *, BLOCK: wl.constexpr):
    # Alternate an input update and decay with explicit ordered state.
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as block:
            state = wl.state(wl.f32, [BLOCK], init=0)
            iteration = wl.index(0)
            while iteration < steps:
                if iteration % 2 == 0:
                    state = state + wl.load(X[row, block])
                else:
                    state = state * decay
                iteration += 1
            wl.store(Y[row, block], state)


@triton.jit(model="weft", strided=("X", "Y", "Z"),
             alias_groups={"X": "input", "Y": "first", "Z": "second"})
def coupled_recurrence(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
                       Z: wl.View[wl.f32, (N,)], steps: wl.index):
    first = wl.state(wl.f32, [N], init=0)
    second = wl.state(wl.f32, [N], init=1)
    iteration = wl.index(0)
    while iteration < steps:
        previous = first
        first = second + wl.load(X)
        if iteration % 2 == 0:
            second = previous * 0.5 + second * 0.25
        else:
            second = previous - second * 0.125
        iteration += 1
    wl.store(Y, first)
    wl.store(Z, second)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def branch_recurrence(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
                      steps: wl.index):
    state = wl.state(wl.f32, [N], init=0)
    iteration = wl.index(0)
    while iteration < steps:
        if iteration % 2 == 0:
            state = state + wl.load(X)
        else:
            state = state * 0.5
        iteration += 1
    wl.store(Y, state)


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def initialized_recurrence(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
                           steps: wl.index):
    state = wl.state(wl.f32, [N], init=wl.load(X))
    for iteration in range(steps):
        state = state * 0.5 + 1.0
    wl.store(Y, state)


@triton.jit(model="weft", strided=("Y",))
def reset_recurrence(Y: wl.View[wl.f32, (N,)], steps: wl.index):
    state = wl.state(wl.f32, [N], init=-1)
    for iteration in range(steps):
        state = wl.state(wl.f32, [N], init=3)
    wl.store(Y, state)


@triton.jit(model="weft", strided=("X", "Y", "Z", "W", "Initial"),
             alias_groups={"X": "activation", "Y": "activation", "Z": "second",
                           "W": "third", "Initial": "snapshot"})
def parallel_carries(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
                     Z: wl.View[wl.f32, (N,)], W: wl.View[wl.f32, (N,)],
                     Initial: wl.View[wl.f32, (N,)], steps: wl.index):
    original = wl.load(X)
    first = wl.state(wl.f32, [N], init=original)
    second = wl.state(wl.f32, [N], init=original * 2.0)
    third = wl.state(wl.f32, [N], init=original + 3.0)
    for iteration in range(steps):
        if iteration % 3 == 0:
            previous = first
            first = second
            second = third
            third = previous
        elif iteration % 3 == 1:
            first = second
            third = third + original
        else:
            previous = first
            first = third + original
            third = second
            second = previous
    wl.store(Y, first)
    wl.store(Z, second)
    wl.store(W, third)
    wl.store(Initial, original)
