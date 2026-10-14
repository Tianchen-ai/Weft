from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import kernels.vec_dot.iq3_s_q8_k as vd_iq3_s_q8_k
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq3_s(
    W: wl.View[ggml.IQ3_S, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X4, (512, 4)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row_base in range(0, M, 16):
        for column_base in range(0, N, 16):
            for row in range(row_base, wl.minimum(M, row_base + 16)):
                for column in range(column_base, wl.minimum(N, column_base + 16)):
                    wl.store(Y[row, column], vd_iq3_s_q8_k.compute(W[column], Xq[row], grid))
