from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def scalar_read_order(
    A: wl.View[wl.i32, (N,)],
    B: wl.View[wl.i32, (N,)],
    S: wl.View[wl.i32, (N,)],
):
    with wl.level.blocks(N, extent=1) as block:
        previous = wl.load(A[block])
        replacement = wl.load(S[block])
        wl.store(A[block], replacement)
        wl.store(B[block], previous)
