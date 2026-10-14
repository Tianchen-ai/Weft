from __future__ import annotations

import weft
import weft.language as wl


@weft.encoding
class Word:
    layout = wl.bitorder.lsb_first, wl.byteorder.little
    elements = 1
    alignment = 4
    value: wl.i32


@weft.kernel
def encoded_read_order(
    A: wl.View[Word, (N,)],
    R: wl.View[wl.i32, (1,)],
    B: wl.View[wl.i32, (1,)],
    S: wl.View[wl.i32, (1,)],
):
    with wl.level.blocks(N, extent=1) as item:
        previous = wl.load(A[item])
        replacement = wl.load(S[0])
        wl.store(R[0], replacement)
        wl.store(B[0], previous.value)
