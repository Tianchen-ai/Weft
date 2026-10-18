"""Shared conditions over ordinary Weft numeric values and explicit outputs."""
from __future__ import annotations

import triton
import triton.language as wl


@triton.jit(model="weft", strided=("X", "Residual", "Y", "Z"),
             alias_groups={"X": "input", "Residual": "input", "Y": "first", "Z": "second"})
def masked_affine(X: wl.View[wl.f32, (M, N)], Residual: wl.View[wl.f32, (M, N)],
                  Y: wl.View[wl.f32, (M, N)], Z: wl.View[wl.f32, (M, N)],
                  lower: wl.f32 = -1.0, upper: wl.f32 = 1.0, scale: wl.f32 = 0.75, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            residual = wl.load(Residual[row, part])
            selected = (x >= lower) & (x <= upper)
            wl.store(Y[row, part], wl.select(selected, x * scale + residual, residual))
            wl.store(Z[row, part], wl.select(selected, x, 0.0))


@triton.jit(model="weft", strided=("X", "Ids", "Scores", "SelectedIds"),
             alias_groups={"X": "input", "Ids": "input", "Scores": "scores", "SelectedIds": "ids"})
def threshold_indices(X: wl.View[wl.f32, (M, N)], Ids: wl.View[wl.i32, (M, N)],
                      Scores: wl.View[wl.f32, (M, N)], SelectedIds: wl.View[wl.i32, (M, N)],
                      threshold: wl.f32 = 0.25, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            value = wl.load(X[row, part])
            identity = wl.load(Ids[row, part])
            selected = value >= threshold
            wl.store(Scores[row, part], wl.select(selected, value, 0.0))
            wl.store(SelectedIds[row, part], wl.select(selected, identity, wl.i32(-1)))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def logical_gate(X: wl.View[wl.f32, (M, N)], R: wl.View[wl.f32, (M, N)],
                 Y: wl.View[wl.f32, (M, N)], Z: wl.View[wl.f32, (M, N)], enabled: wl.i1, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = ((x > r) ^ (r > 0.0)) & enabled
            inverse = selected ^ wl.i1(True)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(inverse, x, r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def signed_choices(X: wl.View[wl.i32, (M, N)], R: wl.View[wl.i32, (M, N)],
                   Y: wl.View[wl.i32, (M, N)], Z: wl.View[wl.i32, (M, N)], threshold: wl.i32, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r < x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.i32(-7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def unsigned_choices(X: wl.View[wl.u32, (M, N)], R: wl.View[wl.u32, (M, N)],
                     Y: wl.View[wl.u32, (M, N)], Z: wl.View[wl.u32, (M, N)], threshold: wl.u32, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.u32(7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def double_choices(X: wl.View[wl.f64, (M, N)], R: wl.View[wl.f64, (M, N)],
                   Y: wl.View[wl.f64, (M, N)], Z: wl.View[wl.f64, (M, N)], threshold: wl.f64, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.f64(-7.0), r))


@triton.jit(model="weft", strided=("X", "Predicate", "Y"),
             alias_groups={"X": "input", "Predicate": "predicate", "Y": "output"})
def observed_predicate(X: wl.View[wl.f32, (M, N)], Predicate: wl.View[wl.i1, (M, N)],
                       Y: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            selected = x > 0.0
            wl.store(Predicate[row, part], selected)
            wl.store(Y[row, part], wl.select(selected, x, 0.0))


@triton.jit(model="weft", strided=("X", "R", "Mask", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Mask": "input", "Y": "first", "Z": "second"})
def loaded_gate(X: wl.View[wl.f32, (M, N)], R: wl.View[wl.f32, (M, N)],
                Mask: wl.View[wl.i1, (M, N)], Y: wl.View[wl.f32, (M, N)], Z: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = wl.load(Mask[row, part]) & (x >= 0.0)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, 1.0, x))


@triton.jit(model="weft", strided=("X", "R", "Mask", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Mask": "input", "Y": "first", "Z": "second"})
def loaded_choices(X: wl.View[wl.f32, (M, N)], R: wl.View[wl.f32, (M, N)],
                   Mask: wl.View[wl.i1, (M, N)], Y: wl.View[wl.f32, (M, N)], Z: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = wl.load(Mask[row, part])
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, -1.0, x))


@triton.jit(model="weft", strided=("X", "Mask", "Selected", "Y", "Z"),
             alias_groups={"X": "input", "Mask": "input", "Selected": "predicate", "Y": "first", "Z": "second"})
def observed_gate(X: wl.View[wl.f32, (M, N)], Mask: wl.View[wl.i1, (M, N)],
                  Selected: wl.View[wl.i1, (M, N)], Y: wl.View[wl.f32, (M, N)], Z: wl.View[wl.f32, (M, N)], *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            selected = wl.load(Mask[row, part]) | (x < 0.0)
            wl.store(Selected[row, part], selected)
            wl.store(Y[row, part], wl.select(selected, x, 0.0))
            wl.store(Z[row, part], wl.select(selected, 1.0, x))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def signed_byte_choices(X: wl.View[wl.i8, (M, N)], R: wl.View[wl.i8, (M, N)],
                        Y: wl.View[wl.i8, (M, N)], Z: wl.View[wl.i8, (M, N)], threshold: wl.i8, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r < x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.i8(-7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def unsigned_half_choices(X: wl.View[wl.u16, (M, N)], R: wl.View[wl.u16, (M, N)],
                          Y: wl.View[wl.u16, (M, N)], Z: wl.View[wl.u16, (M, N)], threshold: wl.u16, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.u16(7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def signed_wide_choices(X: wl.View[wl.i64, (M, N)], R: wl.View[wl.i64, (M, N)],
                        Y: wl.View[wl.i64, (M, N)], Z: wl.View[wl.i64, (M, N)], threshold: wl.i64, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r < x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.i64(-7), r))


@triton.jit(model="weft", strided=("X", "R", "Y", "Z"),
             alias_groups={"X": "input", "R": "input", "Y": "first", "Z": "second"})
def unsigned_wide_choices(X: wl.View[wl.u64, (M, N)], R: wl.View[wl.u64, (M, N)],
                          Y: wl.View[wl.u64, (M, N)], Z: wl.View[wl.u64, (M, N)], threshold: wl.u64, *, BLOCK: wl.constexpr):
    for row in range(M):
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as part:
            x = wl.load(X[row, part])
            r = wl.load(R[row, part])
            selected = (x >= threshold) | (r > x)
            wl.store(Y[row, part], wl.select(selected, x, r))
            wl.store(Z[row, part], wl.select(selected, wl.u64(7), r))
