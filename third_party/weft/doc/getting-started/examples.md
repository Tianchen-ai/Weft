# Runnable examples

Run these examples from the repository root, after installing this Triton
distribution on a supported RISC-V host. Set `PYTHONPATH=third_party/weft` so
the source corpus and the application examples are importable. NumPy examples
need NumPy; the PyTorch example additionally needs a supported CPU PyTorch build
and the [framework source seam](installation.md#可选应用依赖).

## Ordinary Triton

```sh
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation affine
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation softmax
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation matmul
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation scan
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation histogram --threads 2
```

The callable programs live in `source/weft/dense/triton_ops.py`. The application
allocates input/output storage and launches them through `kernel[grid]`.

| Example | Language and execution concepts |
|---|---|
| Affine | Runtime scalars, constexpr block size, masked tails |
| Row softmax | Strides, max/sum, stable exponential normalization |
| Matmul | Two-dimensional grid, broadcast addresses, accumulator `tl.dot`, K loop |
| Scan | Inclusive `tl.cumsum`, per-row grid, masked input/output |
| Histogram | Many programs update shared bins with `tl.atomic_add` |

Histogram input indices must lie in the allocated bin range. Atomic operations
serialize conflicting updates; they do not create a barrier across the grid.
The application initializes the count array before launch.

## Tune and deploy the selected kernel

```sh
PYTHONPATH=third_party/weft python -m examples.autotune_export /path/to/new/artifact
```

This selects a source block size using `triton.Config` and `triton.autotune`,
exports the selected `CompiledKernel`, then loads and calls that native artifact.
The export directory must not already exist. Loaded calls use the fixed source
configuration and need neither the original DSL definition nor a compiler.
For plain JIT without tuning, `python -m examples.use --output /path/to/new/artifact`
uses the existing View-based affine program.

## PyTorch

```sh
python -m triton.tools.pytorch
PYTHONPATH=third_party/weft python -m examples.pytorch_cpu
```

The example first passes CPU tensors to a hand-written Triton kernel, then asks
Inductor to generate a normalization kernel with `cpu_backend="triton"`.
Keep shape, stride and mutation semantics in the normal tensor workflow.
Gradient programs use explicit backward registration, as in
`source/weft/dense/torch_affine.py`; there is no implicit kernel autodifferentiation.

## Weft model features and larger programs

| Topic | Source to read | Contract |
|---|---|---|
| View, Level and subviews | `source/weft/dense/affine.py` | Runtime extents and source-owned data partitioning |
| Packed quantization | `examples/formats/` and `source/weft/ggml/vec_dot/` | Encoding fields, finite-width decode and accumulation |
| Matrix extensions | `examples/kernels/dense/ime_i8_contract.py` | Typed contraction; target capability selects local instructions |
| Attention | `examples/kernels/attention/flash_attention.py` | State update and reuse across nested levels |
| Model composition | `source/weft/programs/` | Host-owned inputs, caches, workspace and ordered kernel calls |
| llama.cpp | [Native SDK](../compiler/native-sdk.md) | Exported library and typed role binding; existing GGML workers |

These sources describe numerical programs, not a promise that every target
implements every possible configuration. A required extension must be present
and actually used by the resulting program. Inspect `asm["weftir"]`,
`asm["riscvir"]` and `asm["c"]` to understand the selected implementation.

Numerical verification and performance measurements belong to the independent
[Weft Bench](https://github.com/Tianchen-ai/Weft-Bench), which consumes these same
programs. The example scripts do not carry a separate benchmark runner.
