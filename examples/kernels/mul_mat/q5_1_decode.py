from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_1 as quant_q8_1
import kernels.vec_dot.q5_1_q8_1 as vd_q5_1_q8_1
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q5_1_decode(
    W: wl.View[ggml.Q5_1, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_1, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_1.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], vd_q5_1_q8_1.compute(W[column], Xq[row]))
