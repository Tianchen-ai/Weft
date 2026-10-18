from __future__ import annotations

import triton
import triton.language as wl

inf = wl.constexpr(float("inf"))


@triton.jit(model="weft")
def _rowmax(value):
    return wl.reduce(value, op="max", axis="tk")


@triton.jit(model="weft")
def _rowsum(value):
    return wl.reduce(value, op="add", axis="tk")


@triton.jit(model="weft")
def online_flash_attention(
    Q: wl.View[wl.f16, (Tq, D)],
    K: wl.View[wl.f16, (Tk, D)],
    V: wl.View[wl.f16, (Tk, D)],
    O: wl.View[wl.f16, (Tq, D)],
 *, BK: wl.constexpr, BQ: wl.constexpr):
    with wl.level.rows(Tq, group=wl.auto("BQ")) as qb:
        q = wl.stage(wl.load(Q[qb, :]))
        m = wl.state(wl.f32, [BQ], init=-inf)
        l = wl.state(wl.f32, [BQ], init=0)
        o = wl.state(wl.f32, [BQ, D], init=0)
        with wl.level.blocks(Tk, extent=wl.auto("BK")) as kb:
            k = wl.load(K[kb, :])
            v = wl.load(V[kb, :])
            s = wl.dot(q, k, over="d", acc_dtype=wl.f32)
            m_new = wl.maximum(m, _rowmax(s))
            p = wl.exp(s - m_new)
            alpha = wl.exp(m - m_new)
            l = l * alpha + _rowsum(p)
            o = o * alpha + wl.dot(p, v, over="tk")
            m = m_new
        wl.store(O[qb, :], wl.narrow(o / l, wl.f16, rounding="rne", saturation=False))
