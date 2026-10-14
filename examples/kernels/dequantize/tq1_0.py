from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel(alias_groups={"Y": "output"})
def row_dequantize_tq1_0(
    W: wl.View[ggml.TQ1_0, (K,)],
    powers: wl.View[wl.u32, (5,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        lane0 = wl.arange(0, 32, axis="k")
        digit0 = wl.arange(0, 5, dtype=wl.u32, axis="tq1_digit0")
        q0 = qf.radix3_digit_i8(powers, w.q[lane0], digit0)
        y0 = wl.reshape(
            qf.ternary_radix(q0, w.d),
            shape=(160,),
            axes=("k",),
            order=("tq1_digit0", "k"),
        )
        wl.store(wl.subview(Y[kb], offsets=(0,), extents=(160,)), y0)
        lane1 = wl.arange(0, 16, axis="k")
        digit1 = wl.arange(0, 5, dtype=wl.u32, axis="tq1_digit1")
        q1 = qf.radix3_digit_i8(powers, w.q[wl.u32(32) + lane1], digit1)
        y1 = wl.reshape(
            qf.ternary_radix(q1, w.d),
            shape=(80,),
            axes=("k",),
            order=("tq1_digit1", "k"),
        )
        wl.store(wl.subview(Y[kb], offsets=(160,), extents=(80,)), y1)
        lane2 = wl.arange(0, 4, axis="k")
        digit2 = wl.arange(0, 4, dtype=wl.u32, axis="tq1_digit2")
        q2 = qf.radix3_digit_i8(powers, w.qh[lane2], digit2)
        y2 = wl.reshape(
            qf.ternary_radix(q2, w.d),
            shape=(16,),
            axes=("k",),
            order=("tq1_digit2", "k"),
        )
        wl.store(wl.subview(Y[kb], offsets=(240,), extents=(16,)), y2)
