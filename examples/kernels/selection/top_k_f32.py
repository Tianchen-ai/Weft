from __future__ import annotations

from math import inf

import weft
import weft.language as wl


@weft.kernel
def top_k_f32(X: wl.View[wl.f32, (N,)], out: wl.View[wl.i32, (K_,)]):
    heap = wl.state(wl.f32, [K_], init=-inf)
    idx = wl.state(wl.i32, [K_], init=-1)
    for i in range(N):
        score = X[i]
        if score > heap[K_ - 1]:
            pos = K_ - 1
            j = K_ - 1
            while j > 0:
                if score > heap[j - 1]:
                    pos = j - 1
                j -= 1
            for j in range(K_ - 1, pos, -1):
                heap[j] = heap[j - 1]
                idx[j] = idx[j - 1]
            heap[pos] = score
            idx[pos] = wl.i32(i)
    wl.store(out, idx)
