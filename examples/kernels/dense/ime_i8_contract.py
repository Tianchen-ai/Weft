from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def ime_i8_contract(
    A: wl.View[wl.i8, (M, K)], B: wl.View[wl.i8, (K, N)], C: wl.View[wl.i32, (M, N)]
):
    with wl.level.rows(M, group=wl.auto("MR")) as mb:
        with wl.level.cols(N, group=wl.auto("NR")) as nb:
            acc = wl.state(wl.i32, [MR, NR], init=0)
            with wl.level.blocks(K, extent=wl.auto("KB")) as kb:
                acc += wl.dot(wl.load(A[mb, kb]), wl.load(B[kb, nb]), over="k", acc_dtype=wl.i32)
            wl.store(C[mb, nb], acc)
