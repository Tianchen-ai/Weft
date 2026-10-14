# Third-party notices

Weft's own code is distributed under Apache-2.0; this does not replace the
licenses of dependencies or retained source material.

## LLVM and MLIR

Weft builds against LLVM/MLIR. The compiler links MLIR components and the native
runtime uses LLVM support/target facilities. When built against shared LLVM,
the distribution includes the corresponding LLVM SONAME library. The upstream
license is Apache-2.0 with LLVM exceptions, with legacy notices retained in
[`licenses/LLVM.txt`](licenses/LLVM.txt), copied from LLVM's `llvmorg-20.1.8`
[`llvm/LICENSE.TXT`](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/LICENSE.TXT).

Notices are installed under the native SDK's `share/weft/licenses/` for CMake and
wheel consumers. Platform library dependencies are supplied by the operating
system, not bundled automatically. Distributors using a different LLVM build
must preserve any additional notices required by that build's components.

## Optional GGML / llama.cpp integration

The optional adapter and comparison examples build against the user's explicit
[llama.cpp checkout](https://github.com/ggml-org/llama.cpp), which retains its
[MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) and upstream
file notices. Upstream GGML source is not bundled in this public repository or
the Python distribution, and is not a compiler or runtime fallback.

## Optional dependencies and references

NumPy and PyTorch are optional, independently installed dependencies; their
distributions retain their own licenses. Triton, TileLang and hardware references
inform design comparisons but are not included in or linked by Weft. No local
reference tree is required to build or use the installed package.
