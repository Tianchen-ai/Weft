from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import kernels.vec_dot.iq4_nl_q8_0 as vd_iq4_nl_q8_0
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq4_nl_decode(
    W: wl.View[ggml.IQ4_NL, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_0, (M, K)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_iq4_nl_q8_0.compute(W[column], Xq[row], codebook)
            wl.store(Y[row, column], value)
