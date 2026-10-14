# Weft

**A numerical kernel language and compiler for RISC-V.**

Licensed under [Apache-2.0](https://github.com/Tianchen-ai/Weft/blob/main/LICENSE).

[中文](https://github.com/Tianchen-ai/Weft/blob/main/README.zh-CN.md) · [Install](https://github.com/Tianchen-ai/Weft/blob/main/doc/getting-started/installation.md) ·
[Quick start](https://github.com/Tianchen-ai/Weft/blob/main/doc/getting-started/quickstart.md) · [Integrations](https://github.com/Tianchen-ai/Weft/blob/main/doc/integrations/overview.md) ·
[Handbook](https://tianchen-ai.github.io/Weft/) · [Contributing](https://github.com/Tianchen-ai/Weft/blob/main/CONTRIBUTING.md)

Weft lets you write numerical programs in a Python DSL, compile them for RVV and
supported local extensions, and call native kernels from Python, PyTorch, or
C/C++. Its goal is to make RISC-V kernel development as accessible as Triton
makes GPU kernel development, using an ordinary ordered CPU program and explicit
RISC-V capabilities.

Canonical Kernel IR preserves the author's numerical program. Physical passes
select layouts, memory transfers, schedules, resources and local instructions in
RISC-V Physical IR. A terminal emitter translates those decisions to intrinsic C
or typed local assembly for the system compiler.

![Weft compilation and application paths](https://raw.githubusercontent.com/Tianchen-ai/Weft/main/doc/assets/architecture.svg)

## What you can use

| Entry | What it provides |
|---|---|
| `weft.jit` / `weft.compile` | Native compilation, typed calls, persistent artifact cache and prepared calls |
| `weft.autotune` | Explicit finite configurations, numerical validation, private trial state and persistent selections |
| `weft.programs` | Norm, RoPE, attention, dense projections, MoE routing and experts, Mamba-1 state updates and model blocks |
| `weft.torch` | CPU Tensor borrowing, custom operators, fake/meta registration, authored backward and graph/export integration |
| `weft build` / `weft.load` | Inspectable artifacts and explicit source-only or native deployment |
| C/C++ SDK | Typed packages, retained storage, prepared calls and ordered native sequences |
| llama.cpp adapter | Explicitly registered vec-dot replacements within the normal GGML CPU path |

The public model catalog uses **f32 inference**. Packed quantized kernels have
explicit Encoding contracts and a separate source corpus. Python is 3.10+; native
execution requires little-endian RISC-V Linux with usable RVV and a compatible C
toolchain. SG2044 (VLEN 128) and K1 (VLEN 256) are the current hardware validation
targets. Development hosts can build source-only RISC-V artifacts.

## Install from source

Install CMake 3.20+, Ninja, a C++17 compiler and matching LLVM/MLIR development
packages. LLVM 20 is the current build reference. With a complete SDK at
`/usr/lib/llvm-20`:

```sh
git clone https://github.com/Tianchen-ai/Weft.git
cd Weft
python3 tools/install.py --environment /tmp/weft-env \
  --llvm-root /usr/lib/llvm-20 --build-dir /tmp/weft-build --jobs 4
. /tmp/weft-env/bin/activate
python -m pip install numpy
weft --help
```

The installer builds a platform wheel containing the Python API, compiler and
native SDK. JIT also needs an RVV-capable C compiler on the execution machine.
For SDK installation, flags, wheel portability and troubleshooting, see the
[installation guide](https://github.com/Tianchen-ai/Weft/blob/main/doc/getting-started/installation.md). Source installation does not depend
on an uploaded PyPI release.

## Run your first operator

On a RISC-V machine, select the native C compiler and inspect the discovered target:

```sh
export CC=clang
weft target
```

```python
import numpy as np
from weft.programs import Operators

x = np.linspace(-1, 1, 387, dtype=np.float32).reshape(3, 129)
weight = np.ones(129, dtype=np.float32)
with Operators() as operators:
    y = operators.rms_norm(x, weight)
expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)
print(y.shape)
```

First use compiles the program. Later calls reuse it; complete artifacts also
persist across processes. The [quick start](https://github.com/Tianchen-ai/Weft/blob/main/doc/getting-started/quickstart.md) continues
with your own DSL kernel, tuning, saved artifacts and generated-code inspection.

On a development host, generate an artifact without executing RISC-V code:

```sh
weft build --kernel weft.programs.normalization:rms_norm --meta BLOCK=64 \
  --march rv64imafdcv --abi lp64d --vlen-bits 128 \
  --source-only --output /tmp/weft-rms-artifact
weft inspect /tmp/weft-rms-artifact --emit operations --format text
```

Use a new output directory for each saved artifact. Compilation alone does not
establish numerical correctness on a target.

## Applications and hardware extensions

[Integrations](https://github.com/Tianchen-ai/Weft/blob/main/doc/integrations/overview.md) covers PyTorch, model blocks, offline
packages and llama.cpp. The adapter preserves upstream threading, activation
quantization and available repack/IME paths. Register kernels whose ABI and
measured workload match the intended replacement.

[Extending RISC-V support](https://github.com/Tianchen-ai/Weft/blob/main/doc/development/extensions.md) describes target facts,
local operation contracts, physical selection, emission and runtime discovery.
A new ISA name alone does not implement an extension.

## Performance and scope

Use the [measurement contract](https://github.com/Tianchen-ai/Weft/blob/main/doc/development/validation.md)
and the existing numerical examples to evaluate your workload. Compare results
only when target, input, dtype, toolchain, author program and timing scope match.
Runtime invocation, kernel execution and model throughput are different measurements.

Weft is ready for developer use and contributions, with these practical limits:

- Performance depends on operator, shape, source variant and target; there is no
  claim of uniform superiority over GGML or vendor kernels.
- Torch operators are opaque CPU calls. Graph capture/export is supported;
  arbitrary PyTorch graphs are not translated into Weft. Model modules do not
  supply automatic differentiation. Inductor performance is not established.
- Vendor instructions require implemented local contracts and supported
  discovery paths. A build target does not authorize native execution.
- The llama.cpp adapter covers registered single-row vec-dot ABIs, not every
  packed matrix path. Model throughput improvements are not established by
  kernel or integration checks.

## Participate

Read [CONTRIBUTING.md](https://github.com/Tianchen-ai/Weft/blob/main/CONTRIBUTING.md) for builds, existing repro entry points,
compiler responsibilities and evidence expected with a change. Report problems
through [GitHub issues](https://github.com/Tianchen-ai/Weft/issues), with a minimal kernel
and exact target/toolchain facts. See [SECURITY.md](https://github.com/Tianchen-ai/Weft/blob/main/SECURITY.md) for private
reporting guidance and [THIRD_PARTY_NOTICES.md](https://github.com/Tianchen-ai/Weft/blob/main/THIRD_PARTY_NOTICES.md) for notices.
