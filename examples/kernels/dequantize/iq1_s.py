from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_iq1_s(
    W: wl.View[ggml.IQ1_S, (K,)],
    grid: wl.View[wl.i8, (16384,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        group = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as group_point:
            metadata = w.qh[group]
            delta = wl.f32(0.125)
            if qf.extract_bits(metadata, 15) != wl.u32(0):
                delta = wl.f32(-0.125)
            scale = wl.f32(w.d) * wl.f32(
                wl.i32(qf.extract_bits(metadata, 12, 3)) * wl.i32(2) + wl.i32(1)
            )
            entry = wl.index(0)
            with wl.level.subtiles(group_point, extent=8) as entry_point:
                lane = wl.arange(0, 8, dtype=wl.u16, axis="k")
                grid_index = wl.u32(w.q[group * wl.index(4) + entry]) | qf.extract_bits(
                    metadata, wl.u32(entry) * wl.u32(3), 3
                ) << wl.u32(8)
                grid_value = wl.lookup(
                    grid, grid_index * wl.u32(8) + wl.u32(lane), bounds="in_bounds"
                )
                wl.store(
                    Y[kb][group_point][entry_point],
                    qf.iq_codebook(wl.f32(wl.i32(grid_value)) + delta, scale),
                )
                entry += wl.index(1)
            group += wl.index(1)
