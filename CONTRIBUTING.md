# Contributing to Weft

Contributions and questions in English or Chinese are welcome. Start with the
[installation guide](doc/getting-started/installation.md) and [quick start](doc/getting-started/quickstart.md).
Use [issues](https://github.com/Tianchen-ai/Weft/issues) for reproducible bugs and design
discussions, and pull requests for concrete changes.

## Find the relevant layer

| Area | Location |
|---|---|
| Python DSL and frontend | `python/weft/language/`, `python/weft/frontend/` |
| Canonical Kernel IR | `include/Weft/Dialect/Kernel/`, `lib/Dialect/Kernel/` |
| Physical IR and verification | `include/Weft/Dialect/RISCV/`, `lib/Dialect/RISCV/` |
| Target passes and emission | `lib/Target/` |
| Python invocation, tuning and deployment | `python/weft/runtime/` |
| C/C++ runtime and SDK | `lib/Runtime/`, `include/weft/` |
| Author programs | `source/weft/`; installed model catalog in `source/weft/programs/` |
| Usage and numerical repros | `examples/use.py`, `examples/repro/weft/` |

The [design specification](doc/index.md) is authoritative. Preserve the two IR
layers and ordered CPU program model. Algorithm/value/effect changes belong to
the author; physical passes decide representations from typed dataflow, effects
and target facts. Emitters spell decisions already made in IR. New hardware
support should follow the [extension guide](doc/development/extensions.md).

## Build and validate a change

Use the compiler build or installed wheel route in the installation guide.
Run the existing repro closest to the affected behavior. For example, on native
RISC-V after installing the package:

```sh
python examples/use.py --family normalization
python examples/use.py --family invocation --torch
python examples/repro/weft/native_jit.py --arrays numpy
```

Run only the relevant families; Torch requires a compatible installation.
Source-only changes can be checked with `weft build --source-only`, but report
that boundary clearly. For compiler changes, examine the final Physical IR and
generated C as well as numerical execution. Use declared tolerances and exact
integer/storage contracts; do not relax them to make a change pass.

For performance work, explain the actual load/decode, address, conversion,
reduction, spill or waiting work removed and its resource cost. Compare matching
targets, inputs, compiler flags and timing boundaries. Put affected measurements
in a separate measurement artifact with their provenance. Avoid unrelated reruns
and separate microkernel, invocation and model-throughput claims.

CI builds a source distribution, builds and installs its wheel, and exercises
installed source-only compilation. It also builds documentation with strict link
checks. Hardware correctness and performance need the native repros; hosted CI
does not imply RISC-V execution.

Before submitting, run `git diff --check`. Keep caches, environments, logs,
generated artifacts and private machine configuration outside the repository.
Reuse the existing repro entry points rather than adding a parallel test tree.

## Pull requests

Describe the concrete problem, resulting behavior and affected layer. Include
the commands actually run and their results. For numerical or performance
claims, include target/toolchain and workload identity. If execution was not
available, say whether validation reached IR, C, object or native runtime.

Keep changes coherent and omit unrelated formatting, reports and generated
files. Discuss changes to language semantics or author-observable behavior
before implementing them. Documentation lives with the relevant contract or
user guide; avoid duplicating the design specification.

## Licensing and attribution

Weft is licensed under [Apache-2.0](LICENSE). Contributions are submitted under
that license. Preserve third-party copyright and license notices, and identify
the source and license when importing code. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Do not import local reference
repositories or turn them into runtime dependencies.

中文说明：欢迎直接用中文描述问题与提交修改。请区分生成代码、编译成功、运行正确和性能收益，
优先复用现有真实入口。修改语言语义前先讨论；运行记录应注明实际机器、工具链和输入。
