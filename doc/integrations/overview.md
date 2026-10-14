# Application integrations

All entry points use the same compiler and artifact ABI. Choose the integration
that matches your application's ownership of tensors, state and execution.

## PyTorch

On a RISC-V host with a matching CPU Torch installation:

```python
import torch
from weft.programs import get
from weft.torch import register

with get("rms_norm").jit(BLOCK=64) as kernel:
    operation = register("my_application::rms_norm", kernel)
    x = torch.randn(3, 129, dtype=torch.float32)
    weight = torch.ones(129, dtype=torch.float32)
    y = operation(x, weight)
    reference = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(y, reference, rtol=2e-3, atol=1e-4)
```

Register each name once per process. Keep the kernel alive while an operator or
captured graph can invoke it. CPU storage is borrowed directly; unsupported
devices, dtypes, strides or aliases are rejected rather than copied implicitly.

Functional registrations supply fake/meta behavior for graph capture. Explicit
output registrations expose mutation through `operation.out(..., out=y)`.
`torch.compile` retains an opaque CPU operator, and exported graphs require the
same operator registration in their consumer process. Authored backward functions
can be registered with `register_backward`; Weft does not differentiate arbitrary
DSL programs. See [Torch contracts](../compiler/runtime.md#torch-模块图与训练).

## Model blocks

`weft.programs` provides `TransformerBlock`, `SelfAttention`, `SwiGLU`, `MoE` and
`Mamba1`, with explicit weight dataclasses, scratch and state. These are f32
building blocks, not a checkpoint downloader or tokenizer. The caller supplies
weights, RoPE tables and state policy. MoE uses stable selected-top-k routing;
Mamba1 uses the Mamba-1 recurrence, not Mamba-2/SSD.

[Model contracts](../dsl/programs.md) describe the complete shapes, algorithms,
state invalidation and package interfaces. In a checkout, the existing usage
entry exercises full blocks and state behavior:

```sh
python examples/use.py --family models
python examples/use.py --family invocation --torch
```

## Offline packages and C/C++

`PackageBuilder` creates named kernels with fixed source/physical bindings;
`ProgramPackageBuilder` collects a model's required programs. Packages contain
code and ABI, with weights and state owned by the application. A source-only
package must be explicitly linked before execution.

Discover the installed SDK without assuming a site-packages layout:

```sh
weft sdk --prefix
weft sdk --cmake-dir
```

```cmake
find_package(Weft CONFIG REQUIRED)
add_executable(application application.cpp)
target_compile_features(application PRIVATE cxx_std_17)
target_link_libraries(application PRIVATE Weft::Runtime)
```

Pass the printed CMake directory as `-DWeft_DIR=...` when configuring the
application. The [SDK reference](../compiler/native-sdk.md) describes typed
bindings, storage layouts, call/sequence lifetimes and direct-entry contracts.
The checkout's `examples/repro/weft/native_sdk.c` and `.cpp` use this SDK.

## llama.cpp

Use a separate writable upstream checkout. The installer modifies it and rejects
unsupported source structures; it does not promise compatibility with every
llama.cpp revision. Prepare a native package with nonempty `ggml.vec_dot`
bindings for supported weight types, then explicitly install the adapter:

```sh
weft llama-install /path/to/llama.cpp
cmake -S /path/to/llama.cpp -B /path/to/llama-build \
  -DGGML_WEFT=ON -DWeft_DIR="$(weft sdk --cmake-dir)"
cmake --build /path/to/llama-build
export WEFT_GGML_PACKAGE=/path/to/selected-native-package
```

Keep the normal upstream target/backend options for the machine. The adapter
hooks worker-local vec-dot calls after GGML activation quantization and preserves
upstream threading, routing, output strides and available repack/IME selection.
Unregistered types use their existing GGML entry. Missing or incompatible Weft
packages fail initialization explicitly.

Use adapter counters to establish which paths actually ran. A registered kernel
may receive no calls when an earlier packed path handles the workload. Whole
matrix kernels owning different packing or quantization cannot be installed as
single-row vec-dot. The [integration contract](../compiler/native-sdk.md#llamacpp-的调用范围)
defines the boundary; integration correctness is separate from LLM throughput.
