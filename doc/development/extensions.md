# Extending RISC-V support

Weft exposes target capabilities independently of the machine running Python:

```sh
weft target --catalog
weft target --march rv64imafdcv --abi lp64d --vlen-bits 128
```

The first lists registered local extensions; the second describes an explicit
build target. On RISC-V, `weft target` without explicit target flags discovers
native execution facts. These queries answer different questions.

## Where an extension belongs

Standard ISA parsing uses LLVM's RISC-V ISA machinery. Vendor local operations
use the built-in extension registry, with typed fragment contracts where needed.
The current registry is compiled into Weft; this is not a dynamic plugin API.

| Responsibility | Implementation entry |
|---|---|
| Extension identity and native matches | `include/Weft/Target/RISCVExtensions.h`, `lib/Target/RISCVExtensions.cpp` |
| Existing matrix atoms and packing/resource facts | `lib/Target/IME/FragmentContracts.cpp` |
| Target profile and native discovery | `lib/Target/RISCVTargetProfile.cpp`, `lib/Target/RISCVNativeTarget.cpp` |
| Fragment IR verification | `lib/Dialect/RISCV/IR/FragmentOps.cpp` |
| Selected local operation materialization | `lib/Target/IME/FragmentMaterialization.cpp` |
| Intrinsic/assembly spelling | `lib/Target/IME/FragmentEmission.cpp`, `lib/Target/Emission/` |
| Shared pipeline | `lib/Target/RISCVPipeline.cpp` |

For an atom using an existing contract, define its numerical types, logical
shape, packing, register groups, temporary bytes and clobbers together. Add its
legal selection and spelling. Native identity matches must establish the actual
feature, vendor/architecture, XLEN and VLEN; each execution CPU must qualify.
Unknown capability is not permission to emit or execute an instruction.

A mechanism introducing new effects or residency needs a corresponding general
IR contract. Adding a name to a table cannot implement those semantics. Read the
[local leaf specification](../compiler/leaves.md) and
[physical machine](../compiler/physical-machine.md) before changing these boundaries.

## Keep the author program intact

An extension leaf represents a closed local operation. It does not own outer
traversal, kernel ABI, workspace, persistent packing or the author program's
levels. Physical passes select and record representations; terminal emission
spells the selected operation and rejects missing decisions.

If exploiting hardware changes numerical grouping, persistent layout or the
value/effect graph, express that choice in a source/std program before lowering.
The [programming model](../programming-model/index.md) defines this distinction.

## Inspect and exercise the result

Inspect an artifact's `physical` IR, `operations` and `resources` through
`weft inspect`. Use the existing source/repro for the affected primitive to run
the full DSL → IR → C → native call path. Check the declared numerical contract,
then measure the affected workload on the actual target with matching input and
toolchain. State exactly where validation stopped if hardware is unavailable.

Compilation for a target and runtime discovery are separate evidence. The
[pass specification](../compiler/passes.md) and
[measurement protocol](validation.md) provide the detailed contracts.
