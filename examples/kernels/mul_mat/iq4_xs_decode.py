from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import kernels.vec_dot.iq4_xs_q8_k as vd_iq4_xs_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_iq4_xs_decode(
    W: wl.View[ggml.IQ4_XS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_iq4_xs_q8_k.compute(W[column], Xq[row], codebook)
            wl.store(Y[row, column], value)
