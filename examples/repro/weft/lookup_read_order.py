from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def lookup_read_order(
    A: wl.View[wl.i32, (4,)],
    R: wl.View[wl.i32, (1,)],
    B: wl.View[wl.i32, (1,)],
    C: wl.View[wl.i32, (1,)],
):
    before = wl.lookup(A, wl.u32(0), bounds="in_bounds")
    wl.store(R[0], wl.i32(7))
    after = wl.lookup(A, wl.u32(0), bounds="in_bounds")
    wl.store(B[0], before)
    wl.store(C[0], after)
