from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dense_projection(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                     Y: wl.View[wl.f32, (M, N)]):
    wl.store(Y, wl.dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Bias", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Bias": "input", "Y": "output"})
def dense_projection_bias(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                          Bias: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (M, N)], scale: wl.f32):
    value = wl.dot(wl.load(X), wl.load(Weight), over="k")
    wl.store(Y, value * scale + wl.load(Bias))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def renamed_projection(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (N, K)],
                       Y: wl.View[wl.f32, (M, N)]):
    left = wl.reshape(wl.load(X), shape=(M, K), axes=("m", "feature"), order=("m", "k"))
    right = wl.reshape(wl.load(Weight), shape=(N, K), axes=("n", "feature"), order=("n", "k"))
    wl.store(Y, wl.dot(left, right, over="feature"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def row_major_matmul(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                     Y: wl.View[wl.f32, (M, N)]):
    with wl.level.rows(M, group=wl.auto("MR")) as row:
        with wl.level.cols(N, group=wl.auto("NR")) as column:
            wl.store(Y[row, column], wl.dot(wl.load(X[row]), wl.load(Weight[:, column]), over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def row_major_matmul_f16(X: wl.View[wl.f16, (M, K)], Weight: wl.View[wl.f16, (K, N)],
                         Y: wl.View[wl.f32, (M, N)]):
    with wl.level.rows(M, group=wl.auto("MR")) as row:
        with wl.level.cols(N, group=wl.auto("NR")) as column:
            wl.store(Y[row, column], wl.dot(wl.load(X[row]), wl.load(Weight[:, column]), over="k", acc_dtype=wl.f32))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def batched_inner_product(X: wl.View[wl.f32, (B, P, K)],
                          Weight: wl.View[wl.f32, (B, P, K)], Y: wl.View[wl.f32, (B,)]):
    with wl.level.blocks(B, extent=wl.auto("ROWS")) as batch:
        value = wl.reduce_dot(wl.load(X[batch]), wl.load(Weight[batch]), over=("p", "k"))
        wl.store(Y[batch], value)


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def pairwise_inner_product(X: wl.View[wl.f32, (M, P, K)],
                           Weight: wl.View[wl.f32, (N, P, K)], Y: wl.View[wl.f32, (M, N)]):
    with wl.level.rows(M, group=wl.auto("MR")) as row:
        with wl.level.cols(N, group=wl.auto("NR")) as column:
            value = wl.reduce_dot(wl.load(X[row]), wl.load(Weight[column]), over=("p", "k"))
            wl.store(Y[row, column], value)


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dot_f64(X: wl.View[wl.f64, (K,)], Weight: wl.View[wl.f64, (K,)],
             Y: wl.View[wl.f64, (1,)]):
    wl.store(Y[0], wl.reduce_dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "weight", "Y": "output"})
def snapshot_matmul(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                    Y: wl.View[wl.f32, (M, N)]):
    with wl.level.rows(M, group=wl.auto("M")) as row:
        with wl.level.cols(N, group=wl.auto("N")) as column:
            saved = wl.stage(wl.load(Weight[:, column]))
            wl.store(Weight[:, column], saved * 0.0)
            wl.store(Y[row, column], wl.dot(wl.load(X[row]), saved, over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def dynamic_matmul(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                   Y: wl.View[wl.f32, (M, N)]):
    wl.store(Y, wl.dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def lhs_lane_matmul(X: wl.View[wl.f32, (K, M)], Weight: wl.View[wl.f32, (N, K)],
                    Y: wl.View[wl.f32, (M, N)]):
    wl.store(Y, wl.dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def output_lane_f64(X: wl.View[wl.f64, (M, K)], Weight: wl.View[wl.f64, (K, N)],
                    Y: wl.View[wl.f64, (M, N)]):
    wl.store(Y, wl.dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def matmul_cast(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                Y: wl.View[wl.f64, (M, N)]):
    with wl.level.rows(M, group=wl.auto("MR")) as row:
        with wl.level.cols(N, group=wl.auto("NR")) as column:
            value = wl.dot(wl.load(X[row]), wl.load(Weight[:, column]), over="k")
            wl.store(Y[row, column], wl.f64(value))


@weft.kernel(strided=("X", "Weight", "Y", "Norm"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output", "Norm": "norm"})
def matmul_observers(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                     Y: wl.View[wl.f32, (M, N)], Norm: wl.View[wl.f32, (M,)]):
    with wl.level.rows(M, group=wl.auto("M")) as row:
        with wl.level.cols(N, group=wl.auto("N")) as column:
            value = wl.dot(wl.load(X[row]), wl.load(Weight[:, column]), over="k")
            wl.store(Y[row, column], value)
            wl.store(Norm[row], wl.reduce(value * value, op="add", axis="n"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def batched_matmul(X: wl.View[wl.f32, (B, M, K)], Weight: wl.View[wl.f32, (B, K, N)],
                   Y: wl.View[wl.f32, (B, M, N)]):
    wl.store(Y, wl.reduce_dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Y": "output"})
def batched_lhs_lane(X: wl.View[wl.f32, (B, K, M)], Weight: wl.View[wl.f32, (B, N, K)],
                     Y: wl.View[wl.f32, (B, M, N)]):
    wl.store(Y, wl.reduce_dot(wl.load(X), wl.load(Weight), over="k"))


@weft.kernel(strided=("X", "Weight", "Bias", "Gate", "Y"),
             alias_groups={"X": "input", "Weight": "input", "Bias": "input", "Gate": "input", "Y": "output"})
def matmul_epilogue(X: wl.View[wl.f32, (M, K)], Weight: wl.View[wl.f32, (K, N)],
                    Bias: wl.View[wl.f32, (N,)], Gate: wl.View[wl.f32, (M,)],
                    Y: wl.View[wl.f32, (M, N)], scale: wl.f32):
    with wl.level.rows(M, group=wl.auto("MR")) as row:
        with wl.level.cols(N, group=wl.auto("NR")) as column:
            value = wl.dot(wl.load(X[row]), wl.load(Weight[:, column]), over="k")
            left = value + wl.load(Bias[column])
            right = value * wl.load(Gate[row])
            wl.store(Y[row, column], left * right * scale)


@weft.kernel(strided=("X", "Weight0", "Weight1", "Bias0", "Bias1", "Y0", "Y1"),
             alias_groups={"X": "input", "Weight0": "input", "Weight1": "input",
                           "Bias0": "input", "Bias1": "input", "Y0": "first", "Y1": "second"})
def paired_projections(X: wl.View[wl.f32, (M, K)], Weight0: wl.View[wl.f32, (K, N)],
                       Weight1: wl.View[wl.f32, (K, N)], Bias0: wl.View[wl.f32, (N,)],
                       Bias1: wl.View[wl.f32, (N,)], Y0: wl.View[wl.f32, (M, N)],
                       Y1: wl.View[wl.f32, (M, N)]):
    with wl.level.rows(M, group=wl.auto("MR")) as row:
        with wl.level.cols(N, group=wl.auto("NR")) as column:
            first = wl.dot(wl.load(X[row]), wl.load(Weight0[:, column]), over="k")
            wl.store(Y0[row, column], first + wl.load(Bias0[column]))
            second = wl.dot(wl.load(X[row]), wl.load(Weight1[:, column]), over="k")
            wl.store(Y1[row, column], second + wl.load(Bias1[column]))
