# Weft compiler core inside Triton

This component contains Canonical Kernel IR, RISC-V Physical IR, target passes,
intrinsic C/local assembly emission and the native C/C++ runtime.

The Python language, JIT, autotuner and native backend are built and installed
through the enclosing Triton tree. Use `@triton.jit(model="weft")` and
`kernel[grid](...)`; this directory has no separate Python distribution or CLI.

See the [handbook](doc/index.md), [installation](doc/getting-started/installation.md)
and [native SDK](doc/compiler/native-sdk.md). Weft code retains its
[Apache-2.0 license](LICENSE); the enclosing Triton tree retains its own notices.
