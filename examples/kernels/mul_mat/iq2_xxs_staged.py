from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


def _iq2_xxs_group_products(
    w, x, grid, signs, entry_lane, codebook_lane, group, word_indices, activation_indices
):
    group_word = group * 4
    indices = wl.widen(w.q[:, word_indices], wl.u32)
    grid_index = indices >> wl.u32((entry_lane % 2) * 8) & wl.u32(255)
    word1 = wl.widen(w.q[:, group_word + 2], wl.u32) | wl.widen(
        w.q[:, group_word + 3], wl.u32
    ) << wl.u32(16)
    sign_index = word1 >> wl.u32(entry_lane * 7) & wl.u32(127)
    weight = wl.lookup(grid, grid_index * wl.u32(8) + codebook_lane, bounds="in_bounds")
    sign = wl.lookup(signs, sign_index * wl.u32(8) + codebook_lane, bounds="in_bounds")
    signed_weight = weight * sign
    activation = x.q[:, activation_indices]
    integer = wl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=wl.i32)
    scale = wl.i32((word1 >> wl.u32(28)) * wl.u32(2) + wl.u32(1))
    return integer * scale


def _compute(
    W: wl.View[ggml.IQ2_XXS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (M, N)],
    bounded_indices,
):
    grid_values = grid.values
    sign_values = signs.values
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
                            entry_lane = wl.arange(0, 4, axis="entry")
                            codebook_lane = wl.arange(0, 8, axis="payload")
                            entry_u32 = wl.arange(0, 4, dtype=wl.u32, axis="entry")
                            payload_u32 = wl.arange(0, 8, dtype=wl.u32, axis="payload")
                            block_sum = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                            group_index = wl.index(0)
                            with wl.level.subtiles(kb, extent=32) as group:
                                if bounded_indices:
                                    group_partial = _iq2_xxs_group_products(
                                        w, x, grid_values, sign_values, entry_u32, payload_u32,
                                        group_index,
                                        wl.u32(group_index) * wl.u32(4) + entry_u32 // wl.u32(2),
                                        wl.u32(group_index) * wl.u32(32) + entry_u32 * wl.u32(8) + payload_u32,
                                    )
                                else:
                                    group_partial = _iq2_xxs_group_products(
                                        w, x, grid_values, sign_values, entry_lane, codebook_lane,
                                        group_index,
                                        group_index * 4 + entry_lane // 2,
                                        group_index * 32 + entry_lane * 8 + codebook_lane,
                                    )
                                block_sum += group_partial
                                group_index += wl.index(1)
                            f32_acc += (
                                wl.f32(w.d) * wl.f32(x.ds) * wl.widen(block_sum, wl.f32)
                            )
                        wl.store(Y[mb, nb], f32_acc)
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                    value = wl.load(Y[mb, nb])
                    wl.store(Y[mb, nb], wl.f32(0.125) * value)


@weft.kernel(alias_groups={"Xq": "workspace", "Y": "output"})
def production_mul_mat_iq2_xxs_staged(
    W: wl.View[ggml.IQ2_XXS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (M, N)],
):
    _compute(W, X, Xq, grid, signs, Y, False)
