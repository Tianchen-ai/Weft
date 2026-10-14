from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


def _iq2_s_group_products(w, x, grid, scale_group, entry, payload, group):
    group_u32 = wl.u32(group)
    linear_entry = scale_group * wl.u32(2) + entry
    grid_index = wl.widen(w.q[:, group_u32 * wl.u32(4) + linear_entry], wl.u32) | (
        wl.widen(w.qh[:, group], wl.u32) >> linear_entry * wl.u32(2) & wl.u32(3)
    ) << wl.u32(8)
    sign_bit = w.signs[:, group_u32 * wl.u32(32) + linear_entry * wl.u32(8) + payload]
    weight = wl.lookup(grid, grid_index * wl.u32(8) + payload, bounds="in_bounds")
    sign_value = wl.i8(sign_bit)
    signed_weight = weight * (wl.i8(1) - sign_value * wl.i8(2))
    entry_offset = scale_group * wl.u32(16) + entry * wl.u32(8)
    activation = x.q[:, group_u32 * wl.u32(32) + entry_offset + payload]
    partial = wl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=wl.i32)
    metadata = wl.widen(w.scales[:, group], wl.u32)
    scale = wl.i32(
        (metadata >> scale_group * wl.u32(4) & wl.u32(15)) * wl.u32(2) + wl.u32(1)
    )
    return wl.reduce(partial * scale, axis="scale_group")


@weft.kernel
def production_mul_mat_iq2_s_staged(
    W: wl.View[ggml.IQ2_S, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[wl.i8, (8192,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], wl.f32(0.0))
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.level.tiles(K, extent=wl.auto("KC")) as kc:
            wp = wl.stage(wl.load(W[nc, kc]))
            with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                        f32_acc = wl.state(wl.f32, [MR, NR], init=wl.load(Y[mb, nb]))
                        with wl.level.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = wl.load(Xq[mb, kb])
                            scale_group = wl.arange(0, 2, dtype=wl.u32, axis="scale_group")
                            entry = wl.arange(0, 2, dtype=wl.u32, axis="entry")
                            payload = wl.arange(0, 8, dtype=wl.u32, axis="payload")
                            block_sum = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                            group_index = wl.index(0)
                            with wl.level.subtiles(kb, extent=32) as group:
                                block_sum += _iq2_s_group_products(
                                    w, x, grid, scale_group, entry, payload, group_index
                                )
                                group_index += wl.index(1)
                            f32_acc += (
                                wl.f32(0.125)
                                * wl.f32(w.d)
                                * wl.f32(x.ds)
                                * wl.widen(block_sum, wl.f32)
                            )
                        wl.store(Y[mb, nb], f32_acc)
