from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ1_M, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
):
    result = wl.f32(0.0)
    grid_values = grid.values
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        sc0 = wl.u32(w.scales[0]) | wl.u32(w.scales[1]) << wl.u32(8)
        sc1 = wl.u32(w.scales[2]) | wl.u32(w.scales[3]) << wl.u32(8)
        sc2 = wl.u32(w.scales[4]) | wl.u32(w.scales[5]) << wl.u32(8)
        sc3 = wl.u32(w.scales[6]) | wl.u32(w.scales[7]) << wl.u32(8)
        scale_bits = sc0 >> wl.u32(12) | sc1 >> wl.u32(8) & wl.u32(240)
        scale_bits = scale_bits | sc2 >> wl.u32(4) & wl.u32(3840)
        scale_bits = scale_bits | sc3 & wl.u32(61440)
        block_scale = qf.nonlinear_lookup(f16_bits, scale_bits)
        group = wl.arange(0, 8, dtype=wl.u16, axis="group")
        scale_part = wl.arange(0, 2, dtype=wl.u16, axis="scale_part")
        entry = wl.arange(0, 2, dtype=wl.u16, axis="entry")
        payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
        qh = wl.widen(w.qh[group * wl.u16(2) + scale_part], wl.u16)
        high_shift = entry * wl.u16(4)
        grid_index = wl.widen(
            w.q[group * wl.u16(4) + scale_part * wl.u16(2) + entry], wl.u16
        ) | (qh >> high_shift & wl.u16(7)) << wl.u16(8)
        weight = wl.lookup(
            grid_values, grid_index * wl.u16(8) + payload, bounds="in_bounds"
        )
        activation = x.q[
            group * wl.u16(32)
            + scale_part * wl.u16(16)
            + entry * wl.u16(8)
            + payload
        ]
        dot = wl.reduce_dot(activation, weight, over=("entry", "payload"), acc_dtype=wl.i32)
        scale_word = wl.widen(
            w.scales[group // wl.u16(2) * wl.u16(2)], wl.u16
        ) | wl.widen(
            w.scales[group // wl.u16(2) * wl.u16(2) + wl.u16(1)], wl.u16
        ) << wl.u16(8)
        scale_shift = group % wl.u16(2) * wl.u16(6) + scale_part * wl.u16(3)
        scale = wl.i32(
            (scale_word >> scale_shift & wl.u16(7)) * wl.u16(2) + wl.u16(1)
        )
        main = wl.reduce(wl.reduce(dot * scale, axis="scale_part"), axis="group")
        delta_shift = wl.u16(3) + entry * wl.u16(4)
        delta_bit = wl.narrow(
            qh >> delta_shift, wl.u8, rounding="rtz", saturation=False
        ) & wl.u8(1)
        delta = wl.i8(1) - wl.i8(delta_bit) * wl.i8(2)
        payload_u8 = wl.narrow(payload, wl.u8, rounding="rtz", saturation=False)
        delta = delta + wl.i8(payload_u8) * wl.i8(0)
        correction_part = wl.reduce_dot(activation, delta, over=("entry", "payload"), acc_dtype=wl.i32)
        correction = wl.reduce(
            wl.reduce(correction_part * scale, axis="scale_part"), axis="group"
        )
        combined = wl.f32(main) + wl.f32(0.125) * wl.f32(correction)
        result += wl.f32(block_scale) * wl.f32(x.ds) * combined
    return result


@weft.kernel
def quantized_vec_dot_iq1_m_q8_k(
    W: wl.View[ggml.IQ1_M, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid, f16_bits))
