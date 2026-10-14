from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def _grid_delta(grid, grid_index, lane, delta):
    value = wl.lookup(
        grid,
        wl.u32(grid_index) * wl.u32(8) + wl.u32(lane),
        bounds="in_bounds",
    )
    return wl.f32(wl.i32(value)) + wl.f32(delta)


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_iq1_m(
    W: wl.View[ggml.IQ1_M, (K,)],
    grid: wl.View[wl.i8, (16384,)],
    f16_bits: wl.View[wl.f32, (65536,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        sc0 = wl.u32(w.scales[0]) | wl.u32(w.scales[1]) << wl.u32(8)
        sc1 = wl.u32(w.scales[2]) | wl.u32(w.scales[3]) << wl.u32(8)
        sc2 = wl.u32(w.scales[4]) | wl.u32(w.scales[5]) << wl.u32(8)
        sc3 = wl.u32(w.scales[6]) | wl.u32(w.scales[7]) << wl.u32(8)
        scale_bits = sc0 >> wl.u32(12) | sc1 >> wl.u32(8) & wl.u32(240)
        scale_bits = scale_bits | sc2 >> wl.u32(4) & wl.u32(3840)
        scale_bits = scale_bits | sc3 & wl.u32(61440)
        d = qf.nonlinear_lookup(f16_bits, scale_bits)
        group = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as group_point:
            scale_word = wl.u32(
                w.scales[wl.index(2) * (group // wl.index(2))]
            ) | wl.u32(
                w.scales[wl.index(2) * (group // wl.index(2)) + wl.index(1)]
            ) << wl.u32(
                8
            )
            entry = wl.index(0)
            with wl.level.subtiles(group_point, extent=8) as entry_point:
                lane = wl.arange(0, 8, dtype=wl.u32, axis="k")
                qh = w.qh[group * wl.index(2) + entry // wl.index(2)]
                half = wl.u32(entry % wl.index(2))
                high_shift = half * wl.u32(4)
                delta_shift = wl.u32(3) + high_shift
                grid_index = wl.u32(w.q[group * wl.index(4) + entry]) | qf.extract_bits(
                    qh, high_shift, 3
                ) << wl.u32(8)
                delta = wl.f32(0.125)
                if qf.extract_bits(qh, delta_shift) != wl.u32(0):
                    delta = wl.f32(-0.125)
                subscale_shift = wl.u32(group % wl.index(2)) * wl.u32(6) + wl.u32(
                    entry // wl.index(2)
                ) * wl.u32(3)
                subscale = wl.i32(
                    qf.extract_bits(scale_word, subscale_shift, 3)
                ) * wl.i32(2) + wl.i32(1)
                wl.store(Y[kb][group_point][entry_point], qf.iq_codebook(
                        _grid_delta(grid, grid_index, lane, delta),
                        d * wl.f32(subscale),
                    ))
                entry += wl.index(1)
            group += wl.index(1)
