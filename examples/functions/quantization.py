from __future__ import annotations

import weft.language as wl


def symmetric_integer(q, d, zero: int = 0):
    return wl.f32(wl.i32(q) - wl.i32(zero)) * wl.f32(d)


def min_affine(q, d, minimum):
    return wl.f32(wl.i32(q)) * wl.f32(d) + wl.f32(minimum)


def k_superblock(q, scale, d, minimum, dmin):
    scaled = wl.f32(d) * wl.f32(wl.i32(scale))
    minimum_scaled = wl.f32(dmin) * wl.f32(wl.i32(minimum))
    return scaled * wl.f32(wl.i32(q)) - minimum_scaled


def ternary_radix(q, d):
    return wl.f32(wl.i32(q)) * wl.f32(d)


def iq_codebook(q, scale):
    return wl.f32(q) * wl.f32(scale)


def fp4_codebook(q, scale):
    return wl.f32(wl.i32(q)) * wl.f32(scale)


def extract_bits(value, shift, width: int = 1):
    mask = (1 << width) - 1
    return wl.u32(value) >> wl.u32(shift) & wl.u32(mask)


def nonlinear_lookup(codebook, code):
    return wl.lookup(codebook, wl.u32(code), bounds="in_bounds")


def small_nonlinear_lookup(codebook, code):
    return wl.lookup(codebook, wl.u8(code), bounds="in_bounds")


def exponent_scale(table, raw):
    return wl.lookup(table, wl.u32(raw), bounds="in_bounds")


def radix3_digit_i8(powers, packed, digit):
    power16 = wl.narrow(
        wl.lookup(powers, wl.u32(digit), bounds="in_bounds"),
        wl.u16,
        rounding="rtz",
        saturation=False,
    )
    power = wl.narrow(power16, wl.u8, rounding="rtz", saturation=False)
    wrapped = power * wl.u8(packed)
    decoded = wl.narrow(
        wl.widen(wrapped, wl.u16) * wl.u16(3) >> wl.u16(8),
        wl.u8,
        rounding="rtz",
        saturation=False,
    )
    return wl.i8(decoded) - wl.i8(1)


def dot_codebook32(q, x, codebook):
    return wl.reduce_dot(small_nonlinear_lookup(codebook, q), x, over="k", acc_dtype=wl.i32)


def iq2_xs_entry_reduce(code, metadata, activation, grid, signs, scale_group, payload):
    code_bits = wl.u16(code)
    grid_index = code_bits & wl.u16(511)
    sign_index = code_bits >> wl.u16(9)
    weight = wl.lookup(grid, grid_index * wl.u16(8) + payload, bounds="in_bounds")
    sign = wl.lookup(signs, sign_index * wl.u16(8) + payload, bounds="in_bounds")
    partial = wl.reduce_dot(activation, weight * sign, over=("entry", "payload"), acc_dtype=wl.i32)
    scale = wl.i32(
        (wl.widen(metadata, wl.u32) >> scale_group % wl.u32(2) * wl.u32(4) & wl.u32(15))
        * wl.u32(2)
        + wl.u32(1)
    )
    return wl.reduce(partial * scale, axis="scale_group")
