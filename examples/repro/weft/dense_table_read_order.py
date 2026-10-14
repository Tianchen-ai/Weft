from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def dense_table_read_order(
    A: wl.View[wl.u32, (5,)],
    S: wl.View[wl.u32, (1,)],
    Y: wl.View[wl.u32, (5,)],
):
    saved = wl.stage(wl.load(A))
    wl.store(A[0], wl.load(S[0]))
    for index in range(5):
        value = wl.lookup(saved, wl.u32(index), bounds="in_bounds")
        wl.store(Y[index], value)
