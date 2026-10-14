from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_iq2_xxs(
    W: wl.View[ggml.IQ2_XXS, (K,)],
    grid: wl.View[wl.i8, (2048,)],
    signs: wl.View[wl.i8, (1024,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        group = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as group_point:
            group_word = group * wl.index(4)
            metadata = wl.widen(w.q[group_word + wl.index(2)], wl.u32) | wl.widen(
                w.q[group_word + wl.index(3)], wl.u32
            ) << wl.u32(16)
            subscale = wl.f32(0.5) + wl.f32(qf.extract_bits(metadata, 28, 4))
            scale = wl.f32(w.d) * subscale * wl.f32(0.25)
            entry = wl.index(0)
            with wl.level.subtiles(group_point, extent=8) as entry_point:
                lane = wl.arange(0, 8, dtype=wl.u16, axis="k")
                indices = w.q[group_word + entry // wl.index(2)]
                grid_index = wl.narrow(
                    qf.extract_bits(indices, wl.u32(entry % wl.index(2)) * wl.u32(8), 8),
                    wl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                sign_index = wl.narrow(
                    qf.extract_bits(metadata, wl.u32(entry) * wl.u32(7), 7),
                    wl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                grid_value = wl.lookup(
                    grid, grid_index * wl.u16(8) + lane, bounds="in_bounds"
                )
                sign_value = wl.lookup(
                    signs, sign_index * wl.u16(8) + lane, bounds="in_bounds"
                )
                wl.store(
                    Y[kb][group_point][entry_point],
                    qf.iq_codebook(wl.i32(grid_value) * wl.i32(sign_value), scale),
                )
                entry += wl.index(1)
            group += wl.index(1)
