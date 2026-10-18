from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import kernels.vec_dot.tq1_0_q8_k as vd_tq1_0_q8_k
import triton
import triton.language as wl


@triton.jit(model="weft")
def production_mul_mat_tq1_0_decode(
    W: wl.View[ggml.TQ1_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    powers: wl.View[wl.u32, (5,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], vd_tq1_0_q8_k.compute(W[column], Xq[row], powers))
