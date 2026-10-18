from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import kernels.vec_dot.q5_k_q8_k as vd_q5_k_q8_k
import triton
import triton.language as wl


@triton.jit(model="weft")
def production_mul_mat_q5_k_decode(
    W: wl.View[ggml.Q5_K, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], vd_q5_k_q8_k.compute(W[column], Xq[row]))
