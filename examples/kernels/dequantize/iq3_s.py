from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_iq3_s(
    W: wl.View[ggml.IQ3_S, (K,)],
    grid: wl.View[wl.i8, (2048,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        group = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as group_point:
            subscale = w.scales[group]
            scale = wl.f32(w.d) * wl.f32(wl.i32(1) + wl.i32(2) * wl.i32(subscale))
            pair = wl.index(0)
            with wl.level.subtiles(group_point, extent=8) as pair_point:
                lane = wl.arange(0, 8, dtype=wl.u16, axis="k")
                half = wl.u32(lane // wl.u16(4))
                payload = wl.u32(lane % wl.u16(4))
                storage_coordinate = group * wl.index(8) + pair * wl.index(2)
                grid_index0 = wl.u32(w.q[storage_coordinate]) | wl.u32(
                    w.qh[storage_coordinate]
                ) << wl.u32(8)
                grid_index1 = wl.u32(w.q[storage_coordinate + wl.index(1)]) | wl.u32(
                    w.qh[storage_coordinate + wl.index(1)]
                ) << wl.u32(8)
                grid_index = grid_index0 * (wl.u32(1) - half) + grid_index1 * half
                value = qf.nonlinear_lookup(grid, grid_index * wl.u32(4) + payload)
                sign = wl.i32(1) - wl.i32(w.signs[pair_point]) * wl.i32(2)
                wl.store(Y[kb][group_point][pair_point], qf.iq_codebook(value * sign, scale))
                pair += wl.index(1)
            group += wl.index(1)
