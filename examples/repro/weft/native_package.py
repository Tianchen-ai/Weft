from __future__ import annotations

import argparse
import json
from pathlib import Path

from weft.api import load_definition
from weft.runtime import CompileOptions, PackageBuilder, Target, Toolchain


def main():
    parser = argparse.ArgumentParser(description="Build source artifacts for the native SDK and GGML node repros")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    target = Target(**json.loads(args.target.read_text()))
    toolchain = Toolchain(compiler=args.compiler)
    builder = PackageBuilder(target)
    specifications = {
        "affine": ("source/weft/dense/affine.py:affine", {"B": 64}),
        "strided": ("source/weft/dense/affine_strided.py:affine_strided", {"B": 64}),
        "state_update": ("source/weft/dense/state_update.py:state_update", {"B": 32}),
        "state_scale": ("source/weft/dense/state_update.py:state_scale", {"B": 32}),
        "dot_q4_0": ("examples/kernels/vec_dot/q4_0_q8_0.py:quantized_vec_dot_q4_0_q8_0", {}),
        "dot_q5_0": ("examples/kernels/vec_dot/q5_0_q8_0.py:quantized_vec_dot_q5_0_q8_0", {}),
        "dot_q5_1": ("examples/kernels/vec_dot/q5_1_q8_1.py:quantized_vec_dot_q5_1_q8_1", {}),
    }
    for name, (source, bindings) in specifications.items():
        builder.add(name, load_definition(str(root / source)), options=CompileOptions(meta=bindings))
    for role in ("q4_0", "q5_0", "q5_1"):
        builder.bind("ggml.vec_dot", role, "dot_" + role)
    package = builder.build(args.output, toolchain=toolchain, native_library=False)
    print(package.path)


if __name__ == "__main__":
    main()
