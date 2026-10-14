from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def quantize_row(X: wl.View[wl.f32, (K,)], Y: wl.View[ggml.Q8_0, (K,)]):
    with wl.level.blocks(K, extent=32) as kb:
        x = wl.load(X[kb])
        amax = wl.reduce(wl.abs(x), op="max")
        d = amax / wl.f32(127.0)
        inverse = wl.f32(0.0)
        if amax != wl.f32(0.0):
            inverse = wl.f32(1.0) / d
        q = wl.narrow(x * inverse, wl.i8, rounding="dynamic", saturation=False)
        wl.store(Y[kb].d, wl.f16(d))
        wl.store(Y[kb].q, q)


def quantize_matrix(X: wl.View[wl.f32, (M, K)], Y: wl.View[ggml.Q8_0, (M, K)]):
    for row in range(M):
        quantize_row(X[row], Y[row])


@weft.kernel
def q8_0_quantize(X: wl.View[wl.f32, (K,)], Y: wl.View[ggml.Q8_0, (K,)]):
    quantize_row(X, Y)
