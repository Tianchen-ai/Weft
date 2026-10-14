from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_iq2_s(
    W: wl.View[ggml.IQ2_S, (K,)],
    grid: wl.View[wl.i8, (8192,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        group = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as group_point:
            high = w.qh[group]
            scale_bits = w.scales[group]
            entry = wl.index(0)
            with wl.level.subtiles(group_point, extent=8) as entry_point:
                lane = wl.arange(0, 8, dtype=wl.u32, axis="k")
                grid_index = wl.u32(w.q[group * wl.index(4) + entry]) | qf.extract_bits(
                    high, wl.u32(entry) * wl.u32(2), 2
                ) << wl.u32(8)
                value = qf.nonlinear_lookup(grid, grid_index * wl.u32(8) + lane)
                sign = wl.i32(1) - wl.i32(w.signs[entry_point]) * wl.i32(2)
                subscale = qf.extract_bits(
                    scale_bits, wl.u32(entry // wl.index(2)) * wl.u32(4), 4
                )
                scale = wl.f32(w.d) * (wl.f32(0.5) + wl.f32(subscale)) * wl.f32(0.25)
                wl.store(Y[kb][group_point][entry_point], qf.iq_codebook(wl.i32(value) * sign, scale))
                entry += wl.index(1)
            group += wl.index(1)
