"""Ordinary Triton tensor programs for the Weft CPU target."""
import triton
import triton.language as tl


@triton.jit(model="weft")
def affine(X, Y, N, scale, bias, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    active = offsets < N
    values = tl.load(X + offsets, active, other=0.0)
    tl.store(Y + offsets, values * scale + bias, active)


@triton.jit(model="weft")
def softmax_rows(X, Y, N, stride_x, stride_y, B: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, B)
    values = tl.load(X + row * stride_x + columns, columns < N, other=-float("inf"))
    numerator = tl.exp(values - tl.max(values, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * stride_y + columns, result, columns < N)


@triton.jit(model="weft")
def matmul(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + reduction
        left = tl.load(A + rows[:, None] * K + k[None, :],
                       (rows[:, None] < M) & (k[None, :] < K), other=0.0)
        right = tl.load(B + k[:, None] * N + columns[None, :],
                        (k[:, None] < K) & (columns[None, :] < N), other=0.0)
        accumulator = tl.dot(left, right, accumulator)
    tl.store(C + rows[:, None] * N + columns[None, :], accumulator,
             (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit(model="weft")
def scan_rows(X, Y, N, B: tl.constexpr, REVERSE: tl.constexpr = False):
    row = tl.program_id(0)
    columns = tl.arange(0, B)
    values = tl.load(X + row * N + columns, columns < N, other=0)
    result = tl.cumsum(values, axis=0, reverse=REVERSE)
    tl.store(Y + row * N + columns, result, columns < N)


@triton.jit(model="weft")
def histogram(Indices, Counts, N, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    indices = tl.load(Indices + offsets, offsets < N, other=0)
    # The application supplies indices in [0, number_of_bins).
    tl.atomic_add(Counts + indices, 1, mask=offsets < N, sem="relaxed")
