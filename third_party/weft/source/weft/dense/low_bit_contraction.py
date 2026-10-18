from __future__ import annotations

import triton
import triton.language as wl


@triton.encoding
class PackedU4:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 1
    elements = 32
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@triton.encoding
class SignedI8:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 1
    elements = 32
    q: wl.i8[32]


@triton.jit(model="weft", alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def low_bit_projection(X: wl.View[SignedI8, (M, K)],
                       Weight: wl.View[PackedU4, (N, K)], Y: wl.View[wl.i32, (M, N)], *, MR: wl.constexpr, NR: wl.constexpr):
    with wl.level.rows(M, group=wl.auto("MR")) as rows:
        with wl.level.cols(N, group=wl.auto("NR")) as columns:
            accumulator = wl.state(wl.i32, [MR, NR], init=0)
            with wl.level.blocks(K, extent=32) as part:
                x = wl.load(X[rows, part])
                weight = wl.load(Weight[columns, part])
                accumulator += wl.dot(x.q, weight.q, over="k", acc_dtype=wl.i32)
            wl.store(Y[rows, columns], accumulator)
