from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import kernels.quantize.q8_0 as quant_q8_0
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_nvfp4(
    W: wl.View[ggml.NVFP4, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_0, (M, K)],
    codebook: wl.View[wl.i8, (16,)],
    scale: wl.View[wl.f32, (256,)],
    Y: wl.View[wl.f32, (M, N)],
):
    table = wl.stage(wl.load(codebook))
    quant_q8_0.quantize_matrix(X, Xq)
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        wp = wl.stage(wl.load(W[nc]))
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            xp = wl.stage(wl.load(Xq[mc]))
            with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    acc = wl.state(wl.f32, [MR, NR], init=wl.f32(0.0))
                    with wl.level.blocks(K, extent=64) as wb:
                        w = wp[nb, wb]
                        with wl.level.subtiles(wb, extent=32) as xb:
                            x = xp[mb, xb]
                            with wl.level.subtiles(xb, extent=16) as sub:
                                q = wl.lookup(
                                    table, wl.u8(w.q[:, sub]), bounds="in_bounds"
                                )
                                integer = wl.dot(x.q[:, sub], q, over="k", acc_dtype=wl.i32)
                                block_scale = qf.exponent_scale(scale, w.d[:, sub])
                                acc += (
                                    wl.f32(x.d)
                                    * block_scale
                                    * wl.widen(integer, wl.f32)
                                )
                    wl.store(Y[mb, nb], acc)
