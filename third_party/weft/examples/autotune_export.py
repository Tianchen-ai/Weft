"""Tune a CPU kernel, export its native artifact, then call the loaded artifact."""
import argparse
from pathlib import Path

import numpy as np
import triton

from source.weft.dense.triton_ops import affine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="a new directory for the selected native artifact")
    parser.add_argument("--size", type=int, default=4097)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    configurations = [triton.Config({"B": size}, backend_options={"lmul_eighths": 8},
                                   num_cpu_threads=args.threads) for size in (32, 64, 128)]
    tuned = triton.autotune(configs=configurations, key=["N"])(affine)
    x = np.linspace(-1, 1, args.size, dtype=np.float32)
    y = np.empty_like(x)
    compiled = tuned[lambda meta: (triton.cdiv(args.size, meta["B"]),)](x, y, args.size, 1.25, -0.5)
    compiled.export(args.output)
    loaded = triton.load(args.output)
    block = tuned.best_config.kwargs["B"]
    loaded[(triton.cdiv(args.size, block),)](x, y, args.size, 1.25, -0.5)
    print("Selected block:", block)
    print("Native artifact:", args.output)
    print(y[:8])


if __name__ == "__main__":
    main()
