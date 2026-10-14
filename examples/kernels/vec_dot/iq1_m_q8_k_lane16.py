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
        entry = wl.arange(0, 2, dtype=wl.u32, axis="entry")
        payload = wl.arange(0, 8, dtype=wl.u32, axis="payload")
        part = wl.arange(0, 2, dtype=wl.u32, axis="scale_part")
        main = wl.i32(entry) * wl.i32(0) + wl.i32(payload) * wl.i32(0)
        correction = main
        # Arbitrary signed-i8 grid values: each lane <= 16 * 16384 * 15.
        # The full i32 sum is bounded by 62914560; correction by 491520.
        for group in range(8):
            qh = wl.widen(w.qh[wl.u32(group) * wl.u32(2) + part], wl.u32)
            index = wl.widen(
                w.q[wl.u32(group) * wl.u32(4) + part * wl.u32(2) + entry],
                wl.u32,
            ) | (qh >> (entry * wl.u32(4)) & wl.u32(7)) << wl.u32(8)
            weight = wl.lookup(
                grid_values, index * wl.u32(8) + payload, bounds="in_bounds"
            )
            activation = x.q[
                wl.u32(group) * wl.u32(32) + part * wl.u32(16)
                + entry * wl.u32(8) + payload
            ]
            scale_word = wl.u32(w.scales[group // 2 * 2]) | wl.u32(
                w.scales[group // 2 * 2 + 1]
            ) << wl.u32(8)
            shift = wl.u32(group % 2) * wl.u32(6) + part * wl.u32(3)
            scale = wl.i16(wl.narrow(
                (scale_word >> shift & wl.u32(7)) * wl.u32(2) + wl.u32(1),
                wl.u16, rounding="rtz", saturation=False,
            ))
            delta16 = wl.narrow(
                qh >> (wl.u32(3) + entry * wl.u32(4)) & wl.u32(1),
                wl.u16, rounding="rtz", saturation=False,
            )
            delta8 = wl.narrow(delta16, wl.u8, rounding="rtz", saturation=False)
            delta = wl.i8(1) - wl.i8(delta8) * wl.i8(2)
            product = wl.widen(activation, wl.i16) * wl.widen(weight, wl.i16)
            correction_product = wl.widen(activation, wl.i16) * wl.widen(delta, wl.i16)
            main += wl.reduce(
                wl.widen(product, wl.i32) * wl.widen(scale, wl.i32),
                axis="scale_part",
            )
            correction += wl.reduce(
                wl.widen(correction_product, wl.i32) * wl.widen(scale, wl.i32),
                axis="scale_part",
            )
        main_lanes = wl.reshape(
            main, shape=(16,), axes=("lane",), order=("entry", "payload")
        )
        correction_lanes = wl.reshape(
            correction, shape=(16,), axes=("lane",), order=("entry", "payload")
        )
        main_sum = wl.reduce(main_lanes, axis="lane")
        correction_sum = wl.reduce(correction_lanes, axis="lane")
        combined = wl.f32(main_sum) + wl.f32(0.125) * wl.f32(correction_sum)
        result += wl.f32(block_scale) * wl.f32(x.ds) * combined
    return result


@weft.kernel
def quantized_vec_dot_iq1_m_q8_k_lane16(
    W: wl.View[ggml.IQ1_M, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, grid, f16_bits))
