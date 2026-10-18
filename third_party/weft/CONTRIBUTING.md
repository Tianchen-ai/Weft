# Contributing to the Weft model

Start with the [handbook](doc/index.md) and [installation guide](doc/getting-started/installation.md).
Changes to this component build through the enclosing Triton source tree.

| Responsibility | Location relative to the repository root |
|---|---|
| Shared Python AST entry | python/triton/compiler/code_generator.py |
| Weft language and semantic lowering | python/triton/language/weft, python/triton/compiler/weft |
| Canonical and RISC-V physical IR | third_party/weft/include/Weft, third_party/weft/lib/Dialect |
| Physical passes and terminal emission | third_party/weft/lib/Target |
| JIT and autotuning | python/triton/runtime/jit.py, autotuner.py |
| CPU backend and C toolchain stage | python/triton/backends/weft |
| Native ABI/runtime and SDK | third_party/weft/lib/Runtime, include/weft |
| Framework model hooks | python/triton/experimental/torch |
| Author programs | third_party/weft/source |

Preserve author numerical, Encoding, Level, lifetime and effect contracts.
Target passes make physical decisions from typed dataflow and target facts;
terminal emission spells the decisions already written into IR.
Hardware extensions use capability facts and local operations, not kernel-name routing.

Build and install the unified Triton wheel. Inspect Canonical/Physical IR and C through
CompiledKernel.asm. Use the independent Weft Bench plugin for affected numerical/performance
cases and existing input/oracle/runner definitions. Keep its output and fixed baselines outside
this source tree. State separately whether a change generated code, compiled, ran correctly,
or improved performance. Do not relax tolerances or replace an actual integration check with
the presence of a decorator or generated artifact.

Pull requests should explain the concrete behavior change and relevant execution evidence.
Run git diff --check; do not include local credentials, build products, caches, historical
reports or an additional test runner. The build workflow checks wheel installation and
handbook links; target execution belongs to the external Bench environment.
