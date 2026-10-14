from __future__ import annotations

import weft
import weft.language as wl


@weft.encoding
class Q1_0:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 128
    d: wl.f16
    q: wl.u1[128] @ wl.grouped(elements=8) @ wl.bit_layers(elements=1, order=wl.lo_first)


@weft.encoding
class Q4_0:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class Q4_1:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    m: wl.f16
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class Q5_0:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    qh: wl.u1[32] @ wl.grouped(elements=8) @ wl.bit_layers(elements=1, order=wl.lo_first)
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class Q5_1:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    m: wl.f16
    qh: wl.u1[32] @ wl.grouped(elements=8) @ wl.bit_layers(elements=1, order=wl.lo_first)
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class Q8_0:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    q: wl.i8[32]


@weft.encoding
class Q2_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    scales: wl.u8[16]
    q: wl.u2[256] @ wl.grouped(elements=128) @ wl.bit_layers(elements=32, order=wl.lo_first)
    d: wl.f16
    dmin: wl.f16


@weft.encoding
class Q3_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    hmask: wl.u1[256] @ wl.grouped(elements=256) @ wl.bit_layers(elements=32, order=wl.lo_first)
    q: wl.u2[256] @ wl.grouped(elements=128) @ wl.bit_layers(elements=32, order=wl.lo_first)
    scale_low: wl.u4[16] @ wl.grouped(elements=16) @ wl.bit_layers(elements=8, order=wl.lo_first)
    scale_high: wl.u2[16] @ wl.grouped(elements=16) @ wl.bit_layers(elements=4, order=wl.lo_first)
    d: wl.f16


@weft.encoding
class Q4_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    dmin: wl.f16
    sc: wl.u6[8] @ wl.pack_fields(group=4, fields=2, low_bits=4, order=wl.lo_first)
    m: wl.u6[8] @ wl.pack_fields(group=4, fields=2, low_bits=4, order=wl.lo_first)
    q: wl.u4[256] @ wl.grouped(elements=64) @ wl.bit_layers(elements=32, order=wl.lo_first)


@weft.encoding
class Q5_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    dmin: wl.f16
    sc: wl.u6[8] @ wl.pack_fields(group=4, fields=2, low_bits=4, order=wl.lo_first)
    m: wl.u6[8] @ wl.pack_fields(group=4, fields=2, low_bits=4, order=wl.lo_first)
    qh: wl.u1[256] @ wl.grouped(elements=256) @ wl.bit_layers(elements=32, order=wl.lo_first)
    q: wl.u4[256] @ wl.grouped(elements=64) @ wl.bit_layers(elements=32, order=wl.lo_first)


@weft.encoding
class Q6_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    elements = 256
    ql: wl.u4[256] @ wl.grouped(elements=128) @ wl.bit_layers(elements=64, order=wl.lo_first)
    qh: wl.u2[256] @ wl.grouped(elements=128) @ wl.bit_layers(elements=32, order=wl.lo_first)
    scales: wl.i8[16] @ wl.grouped(elements=16) @ wl.bit_layers(elements=16, order=wl.lo_first)
    d: wl.f16


@weft.encoding
class IQ1_S:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    q: wl.u8[32]
    qh: wl.u16[8]


@weft.encoding
class IQ1_M:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    elements = 256
    q: wl.u8[32]
    qh: wl.u8[16]
    scales: wl.u8[8]


@weft.encoding
class IQ2_S:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    q: wl.u8[32]
    signs: wl.u1[256] @ wl.grouped(elements=8) @ wl.bit_layers(elements=1, order=wl.lo_first)
    qh: wl.u8[8]
    scales: wl.u8[8]


@weft.encoding
class IQ2_XS:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    q: wl.u16[32]
    scales: wl.u8[8]


@weft.encoding
class IQ2_XXS:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    q: wl.u16[32]


@weft.encoding
class I8X8:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 8
    elements = 8
    values: wl.i8[8]


@weft.encoding
class I8X4:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 4
    elements = 4
    values: wl.i8[4]


@weft.encoding
class IQ3_S:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    q: wl.u8[64]
    qh: wl.u1[64] @ wl.grouped(elements=8) @ wl.bit_layers(elements=1, order=wl.lo_first)
    signs: wl.u1[256] @ wl.grouped(elements=8) @ wl.bit_layers(elements=1, order=wl.lo_first)
    scales: wl.u4[8] @ wl.grouped(elements=2) @ wl.bit_layers(elements=1, order=wl.lo_first)


@weft.encoding
class IQ3_XXS:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    q: wl.u8[64]
    metadata: wl.u32[8]


@weft.encoding
class IQ4_NL:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class IQ4_XS:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    scales_h: wl.u16
    scales_l: wl.u8[4]
    q: wl.u4[256] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class TQ1_0:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    q: wl.u8[48]
    qh: wl.u8[4]
    d: wl.f16


@weft.encoding
class TQ2_0:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    q: wl.u2[256] @ wl.grouped(elements=128) @ wl.bit_layers(elements=32, order=wl.lo_first)
    d: wl.f16


@weft.encoding
class MXFP4:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    elements = 32
    e: wl.u8
    q: wl.u4[32] @ wl.grouped(elements=32) @ wl.bit_layers(elements=16, order=wl.lo_first)


@weft.encoding
class NVFP4:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    elements = 64
    d: wl.u8[4]
    q: wl.u4[64] @ wl.grouped(elements=16) @ wl.bit_layers(elements=8, order=wl.lo_first)


@weft.encoding
class Q8_1:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 32
    d: wl.f16
    s: wl.f16
    q: wl.i8[32]


@weft.encoding
class Q8_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 4
    elements = 256
    ds: wl.f32
    q: wl.i8[256]
    bsum: wl.i16[16]


@weft.derive
def Q4K_I(
    W: wl.View[Q4_K, (M, K)], *, rows: wl.static[int]
) -> wl.View[Q4K_I[rows], (M, K)]:
    return wl.interleave(W, rows=rows)
