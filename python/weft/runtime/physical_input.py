"""Selected Physical IR checkpoints on the ordinary compiler main chain."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .cache import ArtifactCache
from .compiler import KernelSource, _build
from .driver import CompilationError, Toolchain, run_compiler
from .options import CompileOptions
from .target import Target
from .contraction_inspection import ContractionSelections
from .panel_inspection import PanelTransferSelections


_OUTPUTS = {"contractions": "riscv-contraction-input", "layout": "riscv-layout-input",
            "panel-transfers": "riscv-panel-transfer-input"}


def _checkpoint(value):
    if not isinstance(value, str) or value not in _OUTPUTS:
        raise ValueError("checkpoint must be 'contractions', 'panel-transfers' or 'layout'")
    return value


@dataclass(frozen=True)
class PhysicalInput:
    """A bound numerical program at a selected physical compilation boundary.

    The IR contains its remaining bindings. Resuming does not accept replacement
    target or optimization options. It still runs resource and terminal checks;
    possession of a checkpoint is not proof that a native library can execute.
    """

    ir: str
    canonical_ir: str
    checkpoint: str
    target: Target
    options: CompileOptions
    signature: list[dict] | None = None
    interface: dict | None = None

    def __post_init__(self):
        _checkpoint(self.checkpoint)
        if not isinstance(self.ir, str) or not self.ir.strip():
            raise TypeError("PhysicalInput requires nonempty selected Physical IR")
        if not isinstance(self.canonical_ir, str) or not self.canonical_ir.strip():
            raise TypeError("PhysicalInput requires its original Canonical Kernel IR")
        if not isinstance(self.target, Target) or not isinstance(self.options, CompileOptions):
            raise TypeError("PhysicalInput requires explicit Target and CompileOptions")
        if self.signature is not None and (not isinstance(self.signature, list) or
                any(not isinstance(item, dict) for item in self.signature)):
            raise TypeError("PhysicalInput signature must be a list of argument records")
        if self.interface is not None and not isinstance(self.interface, dict):
            raise TypeError("PhysicalInput interface must be a canonical interface object")

    def complete(self, *, toolchain: Toolchain | None = None, native_library: bool = True, cache=None):
        """Materialize this selection through the same suffix used by build()."""
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        return _build(self.canonical_ir, self.signature, target=self.target, options=self.options,
                      toolchain=selected, native_library=native_library, cache=cache,
                      description=self.interface, physical_input=self.ir, checkpoint=self.checkpoint)

    def selections(self, *, toolchain: Toolchain | None = None):
        """Read compiler-verified tile, supply, panel and output decisions."""
        inspectors = {"contractions": ("contraction-selections", ContractionSelections),
                      "panel-transfers": ("panel-transfer-selections", PanelTransferSelections)}
        if self.checkpoint not in inspectors:
            raise ValueError("selected contraction and panel plans are consumed before the layout checkpoint")
        emit, schema = inspectors[self.checkpoint]
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        command = [selected.compiler_path(), f"--emit={emit}"]
        output = run_compiler(command, self.ir, stage=f"{self.checkpoint} selection inspection")
        try:
            value = json.loads(output)
        except ValueError as error:
            raise CompilationError("compiler returned invalid physical selection JSON", command=command,
                                   stdout=output) from error
        return schema.from_dict(value)

    def save(self, path: str | Path):
        """Save one self-contained, inspectable JSON file without generated code."""
        path = Path(path).expanduser().resolve()
        if path.exists():
            raise FileExistsError(f"physical input destination already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"kind": "weft-physical-input", "checkpoint": self.checkpoint,
                    "physical_ir": self.ir, "canonical_ir": self.canonical_ir,
                    "target": self.target.to_dict(), "options": self.options.to_dict(),
                    "signature": self.signature, "interface": self.interface}
        # Write completely before exposing the artifact. Link is exclusive, so
        # a concurrent writer cannot replace another saved numerical program.
        with TemporaryDirectory(prefix=".weft-input-", dir=path.parent) as directory:
            staging = Path(directory) / "input.json"
            staging.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            os.link(staging, path)
        return path

    @classmethod
    def load(cls, path: str | Path):
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        required = {"kind", "checkpoint", "physical_ir", "canonical_ir", "target", "options", "signature", "interface"}
        if not isinstance(value, dict) or set(value) != required or value["kind"] != "weft-physical-input":
            raise ValueError("file is not a complete Weft physical compilation input")
        return cls(value["physical_ir"], value["canonical_ir"], value["checkpoint"],
                   Target(**value["target"]), CompileOptions(**value["options"]),
                   value["signature"], value["interface"])


def _prepare(canonical, *, signature, interface, target, options, checkpoint, toolchain, cache):
    checkpoint = _checkpoint(checkpoint)
    if not isinstance(canonical, str) or not canonical.strip():
        raise TypeError("prepare_ir requires nonempty Canonical Kernel IR")
    if not isinstance(target, Target) or not isinstance(options, CompileOptions):
        raise TypeError("physical preparation requires Target and CompileOptions")
    selected = toolchain.resolve(native_library=False)
    arguments = [f"--emit={_OUTPUTS[checkpoint]}", *target.arguments(), *options.arguments()]
    identity = {"operation": "prepare-physical-input", "canonical": canonical,
                "target": target.to_dict(), "options": options.to_dict(),
                "arguments": arguments, "toolchain": selected.identity(native_library=False)}

    def build():
        command = [selected.compiler, *arguments]
        ir = run_compiler(command, canonical, stage=f"Weft {checkpoint} preparation")
        if not ir.strip():
            raise CompilationError("compiler returned an empty physical checkpoint", command=command)
        return {"checkpoint": checkpoint, "physical_ir": ir}

    storage = cache if isinstance(cache, ArtifactCache) else ArtifactCache(cache)
    record, _, _ = storage.record(identity, build)
    if set(record) != {"checkpoint", "physical_ir"} or record["checkpoint"] != checkpoint:
        raise CompilationError("cached physical preparation has an inconsistent checkpoint")
    return PhysicalInput(record["physical_ir"], canonical,
                         checkpoint, target, options, signature, interface)


def prepare(definition, *, target: Target, options: CompileOptions | None = None,
            checkpoint: str = "contractions", toolchain: Toolchain | None = None, cache=None):
    """Capture and bind a kernel, retaining inspectable selected Physical IR."""
    source = KernelSource.capture(definition)
    binding = source.options(CompileOptions() if options is None else options)
    return _prepare(source.canonical, signature=source.signature, interface=source.description,
                    target=target, options=binding, checkpoint=checkpoint,
                    toolchain=Toolchain.from_environment() if toolchain is None else toolchain, cache=cache)


def prepare_ir(canonical: str, *, target: Target, options: CompileOptions | None = None,
               checkpoint: str = "contractions", toolchain: Toolchain | None = None, cache=None):
    """Bind an external frontend's canonical program at a physical boundary."""
    return _prepare(canonical, signature=None, interface=None, target=target,
                    options=CompileOptions() if options is None else options, checkpoint=checkpoint,
                    toolchain=Toolchain.from_environment() if toolchain is None else toolchain, cache=cache)
