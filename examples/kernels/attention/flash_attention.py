from __future__ import annotations

from math import inf

import weft
import weft.language as wl


def _rowmax(value):
    return wl.reduce(value, op="max", axis="tk")


def _rowsum(value):
    return wl.reduce(value, op="add", axis="tk")


@weft.kernel
def online_flash_attention(
    Q: wl.View[wl.f16, (Tq, D)],
    K: wl.View[wl.f16, (Tk, D)],
    V: wl.View[wl.f16, (Tk, D)],
    O: wl.View[wl.f16, (Tq, D)],
):
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
