from __future__ import annotations

import weft
import weft.language as wl


def choices(left, right, output, part):
    wl.store(output[0, part], wl.select(left == right, left, right))
    wl.store(output[1, part], wl.select(left != right, left, right))
    wl.store(output[2, part], wl.select(left < right, left, right))
    wl.store(output[3, part], wl.select(left <= right, left, right))
    wl.store(output[4, part], wl.select(left > right, left, right))
    wl.store(output[5, part], wl.select(left >= right, left, right))


@weft.kernel(strided=("X", "R", "Y"))
def compare_f32(X: wl.View[wl.f32, (N,)], R: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (6, N)]):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(wl.load(X[part]), wl.load(R[part]), Y, part)


@weft.kernel(strided=("X", "R", "Y"))
def compare_f64(X: wl.View[wl.f64, (N,)], R: wl.View[wl.f64, (N,)], Y: wl.View[wl.f64, (6, N)]):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(wl.load(X[part]), wl.load(R[part]), Y, part)


@weft.kernel(strided=("X", "R", "Y"))
def compare_i32(X: wl.View[wl.i32, (N,)], R: wl.View[wl.i32, (N,)], Y: wl.View[wl.i32, (6, N)]):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(wl.load(X[part]), wl.load(R[part]), Y, part)


@weft.kernel(strided=("X", "R", "Y"))
def compare_u32(X: wl.View[wl.u32, (N,)], R: wl.View[wl.u32, (N,)], Y: wl.View[wl.u32, (6, N)]):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(wl.load(X[part]), wl.load(R[part]), Y, part)


@weft.kernel(strided=("X", "Y"))
def compare_scalar_f32(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (6, N)], pivot: wl.f32):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(pivot, wl.load(X[part]), Y, part)


@weft.kernel(strided=("X", "Y"))
def compare_scalar_f64(X: wl.View[wl.f64, (N,)], Y: wl.View[wl.f64, (6, N)], pivot: wl.f64):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(pivot, wl.load(X[part]), Y, part)


@weft.kernel(strided=("X", "Y"))
def compare_scalar_i32(X: wl.View[wl.i32, (N,)], Y: wl.View[wl.i32, (6, N)], pivot: wl.i32):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(pivot, wl.load(X[part]), Y, part)


@weft.kernel(strided=("X", "Y"))
def compare_scalar_u32(X: wl.View[wl.u32, (N,)], Y: wl.View[wl.u32, (6, N)], pivot: wl.u32):
    with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
        choices(pivot, wl.load(X[part]), Y, part)
