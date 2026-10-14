from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import kernels.vec_dot.q5_0_q8_0 as vd_q5_0_q8_0
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q5_0_decode(
    W: wl.View[ggml.Q5_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_0, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], vd_q5_0_q8_0.compute(W[column], Xq[row]))
