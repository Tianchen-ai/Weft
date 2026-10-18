from __future__ import annotations

import triton
import triton.language as wl


@triton.jit
def dot_tile(A, B, mb, nb, K, MR: wl.constexpr, NR: wl.constexpr, KB: wl.constexpr):
    acc = wl.state(wl.i32, [MR, NR], init=0)
    with wl.level.blocks(K, extent=wl.auto("KB")) as kb:
        acc += wl.dot(wl.load(A[mb, kb]), wl.load(B[kb, nb]),
                      over="k", acc_dtype=wl.i32)
    return acc


@triton.jit
def bias_clamp(value, bias):
    return wl.minimum(wl.maximum(value + bias, wl.i32(0)), wl.i32(127))


@triton.jit(model="weft")
def i8_bias_clamp(A: wl.View[wl.i8, (M, K)],
                  B: wl.View[wl.i8, (K, N)],
                  Bias: wl.View[wl.i32, (N,)],
                  C: wl.View[wl.i32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr, KB: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("MR")) as mb:
        with wl.level.cols(N, group=wl.auto("NR")) as nb:
            result = bias_clamp(dot_tile(A, B, mb, nb, K, MR, NR, KB), wl.load(Bias[nb]))
            wl.store(C[mb, nb], result)
