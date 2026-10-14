from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl
from kernels.mul_mat.iq2_xxs_staged import _compute


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xxs_staged(
    W: wl.View[ggml.IQ2_XXS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (M, N)],
):
    _compute(W, X, Xq, grid, signs, Y, True)
