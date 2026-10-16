from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .driver import CompilationError, Toolchain, file_identity
from .compilation import CompilationPipeline
from .interface import KernelInterface, freeze
from .options import CompileOptions
from .target import NativeTarget, Target, isa_contract


def validate_compilation(result: dict, target: Target, options: CompileOptions) -> dict:
    if result["kind"] != "weft-riscv-artifact" or len(result["kernels"]) != 1:
        raise CompilationError("a callable artifact requires exactly one compiler-emitted kernel ABI")
    metadata = result["kernels"][0]
    missing = {"isa_extensions", "declaration", "arguments", "shape_parameters", "bindings",
               "matrix_extensions", "used_extensions", "required_extensions",
               "local_tile_strategy", "local_tile_realizations"} - metadata.keys()
    if missing:
        raise CompilationError(f"compiler artifact has an incomplete callable ABI: {sorted(missing)}")
    isa_contract(metadata["isa_extensions"])
    for name, value in options.meta.items():
        if metadata["bindings"].get(name) != value:
            raise CompilationError(f"compiled artifact disagrees with source binding {name}={value}")
    actual = Target(metadata["march"], metadata["abi"], metadata["vlen_bits"], metadata.get("data_cache"))
    if actual != target:
        raise CompilationError("compiled artifact disagrees with the requested build target")
    if metadata["local_tile_strategy"] != options.local_tile_strategy:
        raise CompilationError("compiled artifact disagrees with its local tile strategy")
    if options.local_tile_strategy != "target-default" and not metadata[
            "local_tile_realizations"].get(options.local_tile_strategy, 0):
        raise CompilationError("compiled artifact did not apply the requested local tile strategy")
    expected = [] if options.matrix_extension == "none" else [options.matrix_extension]
    if metadata["matrix_extensions"] != expected:
        raise CompilationError("compiled artifact disagrees with its matrix capability binding")
    if sorted(metadata["required_extensions"]) != list(options.require_extensions):
        raise CompilationError("compiled artifact disagrees with its required extensions")
    if not set(options.require_extensions).issubset(metadata["used_extensions"]) or not set(
            metadata["used_extensions"]).issubset(metadata["matrix_extensions"]):
        raise CompilationError("compiled artifact has inconsistent actual extension use")
    if "physical_resources" in metadata:
        validate_resources(metadata["physical_resources"])
    KernelInterface(metadata)
    return metadata


def validate_resources(resources):
    fields = {"vector_register_peak", "fragment_register_peak", "vector_register_budget",
              "local_storage_bytes", "local_storage_budget_bytes"}
    if not isinstance(resources, dict) or set(resources) != fields:
        raise CompilationError("Physical IR resource summary requires complete occupancy and capacity fields")
    if any(type(value) is not int or value < 0 for value in resources.values()):
        raise CompilationError("Physical IR resources must be nonnegative integer counts")
    registers = resources["vector_register_budget"]
    local = resources["local_storage_budget_bytes"]
    if registers <= 0 or local <= 0:
        raise CompilationError("Physical IR resource capacities must be positive")
    if resources["vector_register_peak"] + resources["fragment_register_peak"] > registers:
        raise CompilationError("Physical IR register residency exceeds its declared capacity")
    if resources["local_storage_bytes"] > local:
        raise CompilationError("Physical IR private storage exceeds its declared capacity")


def declaration_header(metadata: dict) -> str:
    # The signature is emitted by the compiler, never reconstructed from C text.
    declaration = metadata["declaration"].replace(" *restrict", " *__restrict")
    return ("#pragma once\n#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n"
            "#ifdef __cplusplus\nextern \"C\" {\n#endif\n" + declaration +
            "\n#ifdef __cplusplus\n}\n#endif\n")


class Artifact:
    """A durable compilation result. Inspecting it does not load native code."""

    def __init__(self, path: str | Path, *, cache_hit: bool | None = None) -> None:
        self.path = Path(path).expanduser().absolute()
        manifest_path = self.path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CompilationError(f"cannot read artifact manifest {manifest_path}: {error}") from error
        if manifest["kind"] != "weft-callable-artifact":
            raise CompilationError(f"unsupported artifact kind: {manifest['kind']}")
        if "dispatch" not in manifest or manifest["dispatch"] != "weft_kernel_dispatch":
            raise CompilationError("artifact has no typed native dispatch contract")
        self.options = CompileOptions(**manifest["options"])
        self.target = Target(**manifest["target"])
        self._raw_metadata = manifest["kernel"]
        self._compilation = (CompilationPipeline.from_dict(manifest["compilation"])
                             if "compilation" in manifest else None)
        validate_compilation({"kind": "weft-riscv-artifact", "kernels": [self._raw_metadata]},
                             self.target, self.options)
        self.interface = KernelInterface(self._raw_metadata, manifest["signature"])
        self.metadata = self.interface.metadata
        self.manifest = freeze(manifest)
        self.cache_hit = cache_hit
        required = ["kernel.c", "kernel.h", "dispatch.c", "canonical.mlir", "physical.mlir"]
        if manifest["native_library"]:
            required.append("kernel.so")
        missing = [name for name in required if not (self.path / name).is_file()]
        if missing:
            raise CompilationError(f"artifact {self.path} is missing files: {missing}")

    @property
    def cache_status(self) -> str:
        if self.cache_hit is None:
            return "loaded"
        return "hit" if self.cache_hit else "built"

    @property
    def source(self) -> str:
        return (self.path / "kernel.c").read_text(encoding="utf-8")

    @property
    def riscv_ir(self) -> str:
        return (self.path / "physical.mlir").read_text(encoding="utf-8")

    @property
    def compilation(self):
        """Actual compiler pass execution, retained across save/link/package."""
        if self._compilation is None:
            raise CompilationError("artifact does not contain a compiler pass execution report")
        return self._compilation

    @property
    def physical_resources(self):
        """Verified IR occupancy; downstream C compilation can still spill."""
        if "physical_resources" not in self.metadata:
            raise CompilationError("artifact does not contain a Physical IR resource summary")
        return self.metadata["physical_resources"]

    @property
    def physical_operations(self):
        """Inspect final selected operations and layouts without loading native code."""
        from .inspection import PhysicalSummary

        if "physical_operations" not in self._raw_metadata:
            raise CompilationError("artifact does not contain a final Physical IR operation summary")
        return PhysicalSummary.from_dict(self._raw_metadata["physical_operations"])

    @property
    def canonical_ir(self) -> str:
        return (self.path / "canonical.mlir").read_text(encoding="utf-8")

    @property
    def library_path(self) -> Path:
        if not self.manifest["native_library"]:
            raise CompilationError("this artifact contains C and IR only; link it before native loading")
        return self.path / "kernel.so"

    @property
    def header_path(self) -> Path:
        return self.path / "kernel.h"

    def save(self, destination: str | Path) -> Artifact:
        """Export a relocatable bundle, refusing to overwrite another artifact."""
        destination = Path(destination).expanduser().absolute()
        if destination == self.path:
            return self
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=".weft-export-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "artifact"
            staging.mkdir()
            for name in ("manifest.json", "kernel.c", "kernel.h", "dispatch.c", "canonical.mlir", "physical.mlir"):
                shutil.copy2(self.path / name, staging / name)
            if self.manifest["native_library"]:
                shutil.copy2(self.library_path, staging / "kernel.so")
            os.rename(staging, destination)
        return Artifact(destination)

    def link(self, *, toolchain: Toolchain | None = None, cache=None) -> Artifact:
        """Link the selected C program without running physical selection again."""
        from .cache import ArtifactCache

        if self.manifest["native_library"]:
            return self
        selected = (Toolchain.from_environment() if toolchain is None else toolchain).resolve_linker()
        selected_cache = cache if isinstance(cache, ArtifactCache) else ArtifactCache(cache)
        original = json.loads((self.path / "manifest.json").read_text())
        identity = {"operation": "link-artifact", "manifest": original,
                    "source": self.source, "dispatch": (self.path / "dispatch.c").read_text(), "canonical": self.canonical_ir,
                    "physical": self.riscv_ir, "toolchain": selected.link_identity(),
                    "runtime": [file_identity(path) for path in sorted(Path(__file__).parent.glob("*.py"))]}

        def build(staging: Path) -> None:
            for name in ("kernel.c", "kernel.h", "dispatch.c", "canonical.mlir", "physical.mlir"):
                shutil.copy2(self.path / name, staging / name)
            command = selected.link_library(staging / "kernel.c", staging / "kernel.so", self.target)
            manifest = {**original, "native_library": True, "link_command": command}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        path, hit = selected_cache.get_or_build(identity, build)
        return Artifact(path, cache_hit=hit)

    def load(self, *, target: NativeTarget | None = None, toolchain: Toolchain | None = None):
        from .invocation import CompiledKernel

        selected = Toolchain.from_environment() if toolchain is None else toolchain
        native = NativeTarget.discover(selected) if target is None else target
        return CompiledKernel(self, native, toolchain=selected)

    def check_isa_contract(self, *, toolchain: Toolchain | None = None):
        """Validate saved ISA requirements against the shared compiler authority."""
        facts = self.target.capabilities(options=self.options, toolchain=toolchain)
        declared = isa_contract(self.metadata["isa_extensions"])
        if declared != facts.isa_extensions:
            missing = sorted(facts.isa_extensions.keys() - declared.keys())
            extra = sorted(declared.keys() - facts.isa_extensions.keys())
            versions = sorted(name for name in declared.keys() & facts.isa_extensions.keys()
                              if declared[name] != facts.isa_extensions[name])
            raise CompilationError("artifact standard ISA contract disagrees with its target: "
                                   f"missing={missing}, extra={extra}, version_mismatch={versions}")
        return facts


def load(path: str | Path, *, target: NativeTarget | None = None, toolchain: Toolchain | None = None):
    return Artifact(path).load(target=target, toolchain=toolchain)
