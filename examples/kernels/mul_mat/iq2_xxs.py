from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import kernels.vec_dot.iq2_xxs_q8_k as vd_iq2_xxs_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_iq2_xxs(
    W: wl.View[ggml.IQ2_XXS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_iq2_xxs_q8_k.compute(W[column], Xq[row], grid, signs)
            wl.store(Y[row, column], value)
