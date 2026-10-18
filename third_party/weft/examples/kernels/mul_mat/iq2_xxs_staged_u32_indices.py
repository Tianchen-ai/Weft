from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as wl
from kernels.mul_mat.iq2_xxs_staged import _compute


@triton.jit(model="weft", alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xxs_staged(
    W: wl.View[ggml.IQ2_XXS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (M, N)],
    *, NC: wl.constexpr, KC: wl.constexpr, MC: wl.constexpr, MR: wl.constexpr, NR: wl.constexpr,
):
    _compute(W, X, Xq, grid, signs, Y, True, NC, KC, MC, MR, NR)
