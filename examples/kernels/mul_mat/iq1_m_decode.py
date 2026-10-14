from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import kernels.vec_dot.iq1_m_q8_k as vd_iq1_m_q8_k
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq1_m_decode(
    W: wl.View[ggml.IQ1_M, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vd_iq1_m_q8_k.compute(
                W[column], Xq[row], grid, f16_bits
            )
            wl.store(Y[row, column], value)
