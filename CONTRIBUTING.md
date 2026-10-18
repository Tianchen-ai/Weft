# Contributing

Describe the concrete behavior changed, its reason and the relevant validation.
Keep changes focused on compiler, language, runtime or application contracts.

For the Weft model, start with the [component contribution guide](third_party/weft/CONTRIBUTING.md)
and [compiler handbook](third_party/weft/doc/index.md). Weft and Triton share
one LLVM/MLIR SDK, build and Python package. The Canonical IR preserves the
author's program; physical passes select target representations and operations.

Use the independent Weft Bench for affected numerical and performance cases.
Keep generated artifacts, fixtures and result files outside this repository.
Report the measurement boundary and the reference used; distinguish native
kernel, complete launch and model timings.

Build instructions are in the [installation guide](third_party/weft/doc/getting-started/installation.md).
Build the handbook with `make docs`. Preserve component licenses and attribution.

Changes intended for upstream Triton follow its
[contribution process](https://github.com/triton-lang/triton/blob/v3.8.0/CONTRIBUTING.md).
