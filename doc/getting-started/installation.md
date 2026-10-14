# Installation

Weft runs on a development host to generate RISC-V code, or on a RISC-V host to
compile and execute it. Native JIT combines both on the execution host.

## Requirements

| Component | Requirement |
|---|---|
| Host | Linux; Python 3.10+ |
| Source build | pip 23+, CMake 3.20+, Ninja, C++17 compiler |
| Compiler SDK | Matching LLVM/MLIR CMake packages and TableGen tools; LLVM 20 is the build reference |
| Native execution | Little-endian RISC-V, OS-enabled RVV, compatible process ABI and VLEN |
| Native JIT/link | C compiler supporting the required RVV intrinsics and selected instructions |
| Arrays | Optional NumPy or CPU Torch |

Use LLVM and MLIR from one SDK. Its binaries must run on the build host.
Ubuntu development hosts can obtain packages from [apt.llvm.org](https://apt.llvm.org/);
the SDK needs `llvm-20-dev`, `libmlir-20-dev` and `mlir-20-tools`. RISC-V systems
need a native SDK. Installing only Clang does not provide MLIR development files.

## Install from a checkout

From the repository root:

```sh
python3 tools/install.py --environment /tmp/weft-env \
  --llvm-root /usr/lib/llvm-20 --build-dir /tmp/weft-build --jobs 4
. /tmp/weft-env/bin/activate
python -m pip install numpy
weft --help
weft-compile --version
weft sdk --prefix
```

Place environment and build directories outside the checkout. Reduce `--jobs`
if compilation exceeds memory. The installer supplies the Python build dependencies
and installs the compiler, SDK and program catalog together. For offline builds,
`--wheelhouse /path/to/wheels` supplies existing Python build dependency wheels.

A compiler-only development build uses:

```sh
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build -j4
```

This produces `build/tools/weft-compile/weft-compile` and
`build/tools/weft-opt/weft-opt`; it does not install the Python package.

## Select the kernel toolchain

On the RISC-V execution host:

```sh
export CC=clang
weft target
```

Target discovery checks ISA, ABI, VLEN and the allowed CPU set. A toolchain using
the external GNU assembler may require `CFLAGS=-fno-integrated-as`. SDK-specific
`--gcc-toolchain`, library directories and rpaths can also be passed through
`CFLAGS` and `LDFLAGS`. Use local paths rather than another machine's settings.

```python
from weft.runtime import Toolchain

tools = Toolchain(cc=("clang",), cflags=("-fno-integrated-as",))
# Pass toolchain=tools to weft.jit, weft.compile or weft.build.
```

`WEFT_COMPILER` selects an uninstalled Weft compiler during development.
`WEFT_CACHE_DIR` selects persistent artifact storage; the default uses the user's
XDG cache location. Closing a callable releases its loaded handle while complete
cache entries remain available across processes.

## Build a wheel from a source distribution

```sh
python -m pip install build
python -m build --sdist --outdir /tmp/weft-dist
python -m pip wheel /tmp/weft-dist/weft_riscv-*.tar.gz --no-deps \
  -Ccmake.define.LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -Ccmake.define.MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -Cbuild.tool-args=-j4 -w /tmp/weft-wheels
```

Use fresh output directories so the wildcard selects one archive. Wheels are
specific to their platform and compatible system libraries; they are not claimed
to be portable manylinux packages. Shared LLVM builds include the LLVM SONAME
dependency and notices. Inspect dynamic dependencies before moving a wheel across
distributions. Running the wheel needs no MLIR headers; JIT still needs a C compiler.

## Torch and troubleshooting

Install a CPU Torch build matching the target Python ABI and glibc. The
[runtime specification](../compiler/runtime.md#安装与命令行) links external RISC-V
distributions and explains their ABI boundaries. Weft's optional dependency
declaration cannot make an unavailable architecture wheel exist.

| Failure | Next step |
|---|---|
| `MLIRConfig.cmake` missing | Install the development package or select a complete SDK |
| TableGen cannot execute | Use matching SDK tools built for the build host |
| `weft target` fails on x86 | Use an explicit target and `--source-only` |
| C compiler rejects an intrinsic | Check RVV support, selected ISA and assembler flags |
| Shared library cannot load | Inspect system dependencies and host compatibility |
| Target or affinity mismatch | Discover and compile for the intended CPU set |
| Artifact directory already exists | Save to a new directory |

Continue with the [quick start](quickstart.md).
