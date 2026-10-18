"""Run ordinary Triton programs on a RISC-V CPU.

From the repository root:
    PYTHONPATH=third_party/weft python -m examples.triton_basics --operation softmax
"""
import argparse
import numpy as np
import triton

from source.weft.dense.triton_ops import affine, softmax_rows, matmul, scan_rows, histogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("affine", "softmax", "matmul", "scan", "histogram"), default="affine")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    launch = {"num_cpu_threads": args.threads}
    if args.operation == "affine":
        x = np.linspace(-1, 1, 131, dtype=np.float32)
        y = np.empty_like(x)
        affine[(triton.cdiv(x.size, 64),)](x, y, x.size, 1.25, -0.5, B=64, **launch)
    elif args.operation == "softmax":
        x = np.linspace(-2, 2, 4 * 117, dtype=np.float32).reshape(4, 117)
        y = np.empty_like(x)
        softmax_rows[(x.shape[0],)](x, y, x.shape[1], x.shape[1], y.shape[1], B=128, **launch)
    elif args.operation == "matmul":
        a = np.linspace(-1, 1, 16 * 32, dtype=np.float32).reshape(16, 32)
        b = np.linspace(-0.5, 0.5, 32 * 16, dtype=np.float32).reshape(32, 16)
        y = np.empty((16, 16), dtype=np.float32)
        matmul[(2, 2)](a, b, y, 16, 16, 32, BM=8, BN=8, BK=16, **launch)
    elif args.operation == "scan":
        x = np.arange(4 * 31, dtype=np.int32).reshape(4, 31) % 7 - 3
        y = np.empty_like(x)
        scan_rows[(4,)](x, y, 31, B=32, **launch)
    else:
        indices = np.arange(131, dtype=np.int32) % 16
        y = np.zeros(16, dtype=np.int32)
        histogram[(triton.cdiv(indices.size, 32),)](indices, y, indices.size, B=32, **launch)
    print(args.operation, y.shape, y.dtype)
    print(y.reshape(-1)[:16])


if __name__ == "__main__":
    main()
