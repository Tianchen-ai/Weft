from __future__ import annotations

import formats.ggml as ggml
import triton
import triton.language as wl


@triton.encoding
class IMEQ4Centered:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    q: wl.u4[32] @ wl.grouped(elements=16) @ wl.bit_layers(elements=8, order=wl.lo_first)


@triton.encoding
class IMEQ4Affine:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    elements = 32
    d: wl.f16
    zp: wl.u8
    q: wl.u4[32] @ wl.grouped(elements=16) @ wl.bit_layers(elements=8, order=wl.lo_first)


@triton.encoding
class IMEQ8:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 4
    elements = 32
    d: wl.f32
    q: wl.i8[32]


@triton.derive
def IMEQ4Centered_I(W: wl.View[IMEQ4Centered, (N, K)], *, rows: wl.static[int]) -> wl.View[IMEQ4Centered_I[rows], (N, K)]:
    return wl.interleave(W, rows=rows, field_chunks={"d": 2, "q": 8})


@triton.derive
def IMEQ4Affine_I(W: wl.View[IMEQ4Affine, (N, K)], *, rows: wl.static[int]) -> wl.View[IMEQ4Affine_I[rows], (N, K)]:
    return wl.interleave(W, rows=rows, field_chunks={"d": 2, "zp": 1, "q": 8})


@triton.derive
def IMEQ8_I(X: wl.View[IMEQ8, (M, K)], *, rows: wl.static[int]) -> wl.View[IMEQ8_I[rows], (M, K)]:
    return wl.interleave(X, rows=rows, field_chunks={"d": 4, "q": 8})


@triton.jit(model="weft")
def affine_zero_point(d, m):
    ratio = -wl.f32(m) / wl.f32(d)
    clipped = wl.minimum(wl.f32(15.0), wl.maximum(wl.f32(0.0), ratio))
    return wl.narrow(clipped, wl.u8, rounding="dynamic", saturation=True)


@triton.jit(model="weft")
def pack_q4_0(W: wl.View[ggml.Q4_0, (N, K)], P: wl.View[IMEQ4Centered_I[16], (N, K)]):
    for row in range(N):
        with wl.level.blocks(K, extent=32) as kb:
            w = wl.load(W[row, kb])
            wl.store(P[row, kb].d, w.d)
            wl.store(P[row, kb].q, w.q)


@triton.jit(model="weft")
def pack_q4_1(W: wl.View[ggml.Q4_1, (N, K)], P: wl.View[IMEQ4Affine_I[16], (N, K)]):
    for row in range(N):
        with wl.level.blocks(K, extent=32) as kb:
            w = wl.load(W[row, kb])
            wl.store(P[row, kb].d, w.d)
            wl.store(P[row, kb].zp, affine_zero_point(w.d, w.m))
            wl.store(P[row, kb].q, w.q)


@triton.jit(model="weft")
def pack_q4_k(W: wl.View[ggml.Q4_K, (N, K)], P: wl.View[IMEQ4Affine_I[16], (N, K)]):
    for row in range(N):
        with wl.level.blocks(K, extent=256) as kb:
            w = wl.load(W[row, kb])
            with wl.level.subtiles(extent=32) as s:
                d = wl.f16(wl.f32(w.d) * wl.f32(w.sc[s]))
                m = wl.f16(-wl.f32(w.dmin) * wl.f32(w.m[s]))
                wl.store(P[row, s].d, d)
                wl.store(P[row, s].zp, affine_zero_point(d, m))
                wl.store(P[row, s].q, w.q[s])


@triton.jit(model="weft")
def quantize_activation(X, Xq):
    for row in range(M):
        with wl.level.blocks(K, extent=32) as kb:
            x = wl.load(X[row, kb])
            amax = wl.reduce(wl.abs(x), op="max")
            d = amax * wl.f32(0.007874015748031496)
            inverse = wl.f32(0.0)
            if amax != wl.f32(0.0):
                inverse = wl.f32(1.0) / d
            q = wl.narrow(x * inverse, wl.i8, rounding="dynamic", saturation=True)
            wl.store(Xq[row, kb].d, d)
            wl.store(Xq[row, kb].q, q)


@triton.jit(model="weft")
def quantize_ime1_prefill(X: wl.View[wl.f32, (M, K)], Xq: wl.View[IMEQ8_I[4], (M, K)]):
    quantize_activation(X, Xq)


@triton.jit(model="weft")
def mul_mat_centered(W, X, Xq, Y, *, MR: wl.constexpr, NR: wl.constexpr):
    quantize_activation(X, Xq)
    packed_mat_centered(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def packed_mat_centered(W, Xq, Y, *, MR: wl.constexpr, NR: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("MR")) as mb:
        with wl.level.cols(N, group=wl.auto("NR")) as nb:
            acc = wl.state(wl.f32, [MR, NR], init=0)
            with wl.level.blocks(K, extent=32) as kb:
                x = wl.load(Xq[mb, kb])
                w = wl.load(W[nb, kb])
                partial = wl.state(wl.i32, [MR, NR], init=0)
                with wl.level.subtiles(extent=8) as s:
                    q = wl.i8(w.q[s]) - wl.i8(8)
                    partial += wl.dot(x.q[s], q, over="k", acc_dtype=wl.i32)
                scale = x.d * wl.f32(w.d)
                acc += wl.f32(partial) * scale
            wl.store(Y[mb, nb], acc)


@triton.jit(model="weft")
def mul_mat_affine(W, X, Xq, Y, *, MR: wl.constexpr, NR: wl.constexpr):
    quantize_activation(X, Xq)
    packed_mat_affine(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def packed_mat_affine(W, Xq, Y, *, MR: wl.constexpr, NR: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("MR")) as mb:
        with wl.level.cols(N, group=wl.auto("NR")) as nb:
            acc = wl.state(wl.f32, [MR, NR], init=0)
            with wl.level.blocks(K, extent=32) as kb:
                x = wl.load(Xq[mb, kb])
                w = wl.load(W[nb, kb])
                partial = wl.state(wl.i32, [MR, NR], init=0)
                with wl.level.subtiles(extent=8) as s:
                    q = wl.i8(w.q[s]) - wl.i8(w.zp)
                    partial += wl.dot(x.q[s], q, over="k", acc_dtype=wl.i32)
                scale = x.d * wl.f32(w.d)
                acc += wl.f32(partial) * scale
            wl.store(Y[mb, nb], acc)


@triton.jit(model="weft")
def q4_centered_decode(W: wl.View[IMEQ4Centered_I[16], (N, K)], X: wl.View[wl.f32, (M, K)], Xq: wl.View[IMEQ8_I[1], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    mul_mat_centered(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def q4_centered_prefill(W: wl.View[IMEQ4Centered_I[16], (N, K)], X: wl.View[wl.f32, (M, K)], Xq: wl.View[IMEQ8_I[4], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    mul_mat_centered(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def q4_affine_decode(W: wl.View[IMEQ4Affine_I[16], (N, K)], X: wl.View[wl.f32, (M, K)], Xq: wl.View[IMEQ8_I[1], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    mul_mat_affine(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft")
def q4_affine_prefill(W: wl.View[IMEQ4Affine_I[16], (N, K)], X: wl.View[wl.f32, (M, K)], Xq: wl.View[IMEQ8_I[4], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    mul_mat_affine(W, X, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_centered_packed_decode(W: wl.View[IMEQ4Centered_I[16], (N, K)], Xq: wl.View[IMEQ8_I[1], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    packed_mat_centered(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_affine_packed_decode(W: wl.View[IMEQ4Affine_I[16], (N, K)], Xq: wl.View[IMEQ8_I[1], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    packed_mat_affine(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_centered_packed_prefill(W: wl.View[IMEQ4Centered_I[16], (N, K)], Xq: wl.View[IMEQ8_I[4], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    packed_mat_centered(W, Xq, Y, MR=MR, NR=NR)


@triton.jit(model="weft", strided=("Y",))
def q4_affine_packed_prefill(W: wl.View[IMEQ4Affine_I[16], (N, K)], Xq: wl.View[IMEQ8_I[4], (M, K)], Y: wl.View[wl.f32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    packed_mat_affine(W, Xq, Y, MR=MR, NR=NR)
