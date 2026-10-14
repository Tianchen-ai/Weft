from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


def _iq2_xs_entry_products(w, x, grid, signs, scale_group, entry, payload):
    linear_entry = scale_group * wl.u32(2) + entry
    code = w.q[:, linear_entry]
    entry_offset = scale_group * wl.u32(16) + entry * wl.u32(8)
    metadata = w.scales[:, scale_group // wl.u32(2)]
    return qf.iq2_xs_entry_reduce(
        code,
        metadata,
        x.q[:, entry_offset + wl.u32(payload)],
        grid,
        signs,
        scale_group,
        payload,
    )


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xs(
    W: wl.View[ggml.IQ2_XS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (512, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    grid_values = grid.values
    sign_values = signs.values
    with wl.level.rows(M, group=1) as mb:
        with wl.level.cols(N, group=wl.auto("NR")) as nb:
            f32_acc = wl.state(wl.f32, [1, NR], init=wl.f32(0.0))
            with wl.level.blocks(K, extent=256) as kb:
                w = wl.load(W[nb, kb])
                x = wl.load(Xq[mb, kb])
                scale_group = wl.arange(0, 16, dtype=wl.u32, axis="scale_group")
                entry = wl.arange(0, 2, dtype=wl.u32, axis="entry")
                payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
                block_sum = _iq2_xs_entry_products(
                    w, x, grid_values, sign_values, scale_group, entry, payload
                )
                f32_acc += (
                    wl.f32(0.125)
                    * wl.f32(w.d)
                    * wl.f32(x.ds)
                    * wl.widen(block_sum, wl.f32)
                )
            wl.store(Y[mb, nb], f32_acc)
