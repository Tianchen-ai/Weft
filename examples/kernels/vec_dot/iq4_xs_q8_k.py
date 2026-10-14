from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ4_XS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    codebook: wl.View[wl.i8, (16,)],
):
    table = wl.stage(wl.load(codebook))
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        sub_index = wl.index(0)
        with wl.level.subtiles(kb, extent=32) as sub:
            low = wl.u32(w.scales_l[sub_index // wl.index(2)])
            shift = sub_index % wl.index(2) * wl.index(4)
            scale_low = low >> wl.u32(shift) & wl.u32(15)
            scale_high = wl.u32(w.scales_h) >> wl.u32(sub_index * wl.index(2)) & wl.u32(
                3
            )
            scale = wl.i32(scale_low | scale_high << wl.u32(4)) - wl.i32(32)
            q = qf.small_nonlinear_lookup(table, w.q[sub])
            integer = wl.reduce_dot(q, x.q[sub], over="k", acc_dtype=wl.i32)
            block_scale = wl.f32(w.d) * wl.f32(x.ds) * wl.f32(scale)
            result += block_scale * wl.f32(integer)
            sub_index += wl.index(1)
    return result


@weft.kernel
def quantized_vec_dot_iq4_xs_q8_k(
    W: wl.View[ggml.IQ4_XS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.store(Y[0], compute(W, X, codebook))
