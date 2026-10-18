"""Run and optionally export an ordinary Triton CPU kernel.

From third_party/weft: python -m examples.use --size 4097
Use TRITON_CACHE_DIR to select the application's compilation cache.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import triton
from source.weft.dense.affine import affine_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4097)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    x = np.arange(arguments.size, dtype=np.float32)
    y = np.empty_like(x)
    options = {"num_cpu_threads": arguments.threads}
    if arguments.toolchain is not None:
        options["toolchain"] = json.loads(arguments.toolchain.read_text())
    compiled = affine_grid[(triton.cdiv(x.size, 64),)](x, y, 1.25, bias=-0.5, B=64, **options)
    print(y[:8])
    if arguments.output is not None:
        print(compiled.export(arguments.output))


if __name__ == "__main__":
    main()
