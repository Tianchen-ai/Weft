from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import kernels.vec_dot.mxfp4_q8_0 as vd_mxfp4_q8_0
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_mxfp4_decode(
    W: wl.View[ggml.MXFP4, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_0, (M, K)],
    codebook: wl.View[wl.i8, (16,)],
    scale: wl.View[wl.f32, (256,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_mxfp4_q8_0.compute(W[column], Xq[row], codebook, scale)
            wl.store(Y[row, column], value)
