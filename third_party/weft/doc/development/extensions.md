# Extending RISC-V support

The extended Triton exposes three distinct capability queries:

```python
from triton.backends.compiler import CPUTarget
from triton.backends.weft.target import (
    extension_catalog, target_capabilities, native_capabilities,
)

catalog = extension_catalog()
target = CPUTarget("weft", "riscv64", "rv64gcv", "lp64d", 256)
rvv = target_capabilities(target)
matrix = target_capabilities(target, matrix_extension="spacemit-ime1")
# On a RISC-V execution host:
native = native_capabilities()
```

`extension_catalog()` lists the local extensions built into this compiler and
SDK, including numerical types, packing, finite atom repetitions, resources,
clobbers and native identity matches. `target_capabilities()` uses the same
target validator as compilation and returns the capabilities admitted for an
explicit ISA/ABI/VLEN and optional matrix extension. Both work on a cross-build
host without a C compiler, sysroot or device. An unknown extension, incompatible
VLEN or unsupported ISA is an error, not an empty successful capability set.

`native_capabilities()` reports the current host ISA, allowed CPU set and
`native_matrix_extensions`: the CPU IDs eligible for each detected extension.
It does not bind threads or enable an extension. Native discovery on an
unsupported host fails explicitly. The SDK still checks each executing worker;
the catalog alone never authorizes an instruction on a CPU.

Compilation continues through the ordinary Triton API. For an author program
whose typed contract can use a registered matrix atom:

```python
compiled = kernel[(1,)](
    W, X, Y,
    backend_options={
        "matrix_extension": "spacemit-ime1",
        "required_extensions": ("spacemit-ime1",),
    },
)
print(compiled.metadata.kernel_abi["used_extensions"])
print(compiled.asm["riscvir"])
```

`matrix_extension` admits a physical implementation. `required_extensions`
requires the final program to actually use it; it cannot turn an unrelated
program into a matrix operation. `triton.Config(..., backend_options=...)` uses
the same options during tuning. The C toolchain must assemble the selected
instruction, and deployment must provide qualifying execution CPUs.

## Where an extension belongs

Standard ISA parsing uses LLVM's RISC-V ISA machinery. Vendor local operations
use built-in extension modules, with typed fragment contracts where needed.
The registry is fixed at build time; there is no Python registration, runtime
library search or mutable global instruction registry.

| Responsibility | Implementation entry |
|---|---|
| Build-time module list | `lib/Target/Extensions/CMakeLists.txt` |
| Extension identity, native matches and local atom contracts | `lib/Target/Extensions/<Module>/Contracts.cpp` |
| Shared registry validation and tile admission | `lib/Target/RISCVExtensions.cpp`, `lib/Target/RISCVFragment.cpp` |
| Target profile and native discovery | `lib/Target/RISCVTargetProfile.cpp`, `lib/Target/RISCVNativeTarget.cpp` |
| Fragment IR verification | `lib/Dialect/RISCV/IR/FragmentOps.cpp` |
| Selected local operation materialization | `lib/Target/IME/FragmentMaterialization.cpp` |
| Extension's selected MMA spelling | `lib/Target/Extensions/<Module>/Emission.cpp` |
| Shared terminal dispatch and register/memory transfers | `lib/Target/IME/FragmentEmission.cpp`, `lib/Target/Emission/` |
| Shared pipeline | `lib/Target/RISCVPipeline.cpp` |

For another matrix extension using the existing shared-RVV fragment contract:

1. Add `Extensions/<Module>/Contracts.cpp`, defining
   `weft::get<Module>Extension()`. Its returned `RISCVExtension` has stable
   lifetime and owns distinct extension/instruction names, native matches and
   atom contracts. Define numerical types, logical/active shape, packing,
   register groups, temporary bytes, clobbers and finite row/column/product
   repetition limits together. A partial-row atom cannot repeat along M.
2. Add `Extensions/<Module>/Emission.cpp`, implementing
   `weft::riscv_emission::emit<Module>MMA` with the signature in
   `IME/FragmentEmission.cpp`. It consumes an already selected, verified
   `IMEFragmentMMAOp` and typed operand names. It only spells that local op
   through `FragmentEmission::fresh` and `line`; it cannot change layout,
   choose an algorithm or introduce a loop or ABI.
3. Add the module identifier to `extension_modules` in the module list.
   CMake derives both source lists and a shared registry include from it.
   `WeftRISCVContracts` contains only MLIR-free descriptions and validators;
   `WeftRISCVTarget` adds compiler-only emission. Runtime-only SDK builds
   consume the same descriptions without linking MLIR or emission code.

The existing `SpacemitIME1` module exercises this path for both M4 and M1 atoms.
Adding a module does not require editing a public instruction enum or a
vendor-specific switch in the shared emitter. Missing module sources or symbols
fail the build. Duplicate names, incomplete resource/packing contracts and
invalid repetition domains fail registry validation. Generic physical passes
still choose legal atom repetitions and account for their live resources.

Native identity matches must establish the actual feature, vendor/architecture,
XLEN and VLEN; each execution CPU must qualify. Unknown capability is not
permission to emit or execute an instruction.

A mechanism introducing new effects or residency needs a corresponding general
IR contract. Adding a name to a table cannot implement those semantics. Read the
[local leaf specification](../compiler/leaves.md) and
[physical machine](../compiler/physical-machine.md) before changing these boundaries.
The current module interface covers matrix MMA atoms with the existing carrier
and transfer semantics. A DSP instruction with different effects, an independent
tile register file or a new asynchronous engine needs its own physical contract
and lowering support. An x86 backend remains a separate target lowering from
Canonical IR; it is not registered as a RISC-V extension.

## Keep the author program intact

An extension leaf represents a closed local operation. It does not own outer
traversal, kernel ABI, workspace, persistent packing or the author program's
levels. Physical passes select and record representations; terminal emission
spells the selected operation and rejects missing decisions.

If exploiting hardware changes numerical grouping, persistent layout or the
value/effect graph, express that choice in a source/std program before lowering.
The [programming model](../programming-model/index.md) defines this distinction.

## Inspect and exercise the result

Inspect `CompiledKernel.asm["weftir"]`, `asm["riscvir"]`, `asm["c"]` and
`CompiledKernel.metadata.compilation` for selected operations and resources.
Use the existing Bench source/repro for the affected primitive to run
the full DSL → IR → C → native call path. Check the declared numerical contract,
then measure the affected workload on the actual target with matching input and
toolchain. State exactly where validation stopped if hardware is unavailable.

Compilation for a target and runtime discovery are separate evidence. The
[pass specification](../compiler/passes.md) and
[measurement protocol](validation.md) provide the detailed contracts.
