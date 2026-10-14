from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from weft.api import load_definition

from .artifact import Artifact
from .compiler import KernelSource
from .driver import CompilationError, Toolchain
from .options import CompileOptions
from .package import Package
from .target import Target
from .tuning import TuningResult


class PackageBuilder:
    """Capture explicit author programs and bindings for one deployment target.

    This is an offline build service, not an execution graph or a tuner. Kernel
    ordering, weight preparation, scratch reuse and synchronization remain the
    responsibility of the native consumer. Capturing at add time keeps a later
    Python module mutation from changing the selected program at build time.
    """

    def __init__(self, target: Target):
        if not isinstance(target, Target):
            raise TypeError("package builder requires an explicit Target")
        self.target = target
        self._entries = {}
        self._bindings = {}

    def _name(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
            raise ValueError("kernel names use letters, digits, underscore or hyphen")
        if name in self._entries:
            raise ValueError(f"duplicate package kernel: {name}")

    def add(self, name, definition, *, options=None, selection=None):
        self._name(name)
        if options is not None and selection is not None:
            raise ValueError("choose explicit options or a saved selection, not both")
        source = definition if isinstance(definition, KernelSource) else KernelSource.capture(definition)
        if selection is not None:
            if not isinstance(selection, TuningResult):
                raise TypeError("selection must be a TuningResult")
            selection.check_program(source.canonical, self.target)
            options = selection.winner.options
        elif options is None:
            options = CompileOptions()
        if not isinstance(options, CompileOptions):
            raise TypeError("package kernel options must be CompileOptions")
        # Resolve every finite source domain before any compilation begins.
        self._entries[name] = (source, source.options(options))
        return self

    def add_artifact(self, name, artifact):
        self._name(name)
        if not isinstance(artifact, Artifact):
            raise TypeError("expected a saved Artifact")
        if artifact.target != self.target:
            raise ValueError("artifact belongs to another explicit deployment target")
        self._entries[name] = artifact
        return self

    def bind(self, group, role, kernel):
        if not isinstance(group, str) or not group or not isinstance(role, str) or not role:
            raise ValueError("framework binding needs nonempty group and role names")
        if kernel not in self._entries:
            raise ValueError(f"binding names an absent package kernel: {kernel}")
        mapping = self._bindings.setdefault(group, {})
        if role in mapping:
            raise ValueError(f"duplicate framework role: {group}/{role}")
        mapping[role] = kernel
        return self

    def build(self, directory, *, toolchain=None, native_library=True, cache=None):
        if not self._entries:
            raise ValueError("package contains no kernels")
        destination = Path(directory).expanduser().absolute()
        if destination.exists():
            raise FileExistsError(destination)
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        artifacts = {}
        for name, entry in self._entries.items():
            try:
                if isinstance(entry, Artifact):
                    artifacts[name] = entry.link(toolchain=selected, cache=cache) if native_library else entry
                else:
                    source, options = entry
                    artifacts[name] = source.build(target=self.target, options=options,
                                                   toolchain=selected, native_library=native_library, cache=cache)
            except CompilationError as error:
                raise error.with_context(f"package entry {name}") from error
        # Package publication occurs only after every selected entry succeeds.
        return Package.create(destination, artifacts, bindings=self._bindings)

    @classmethod
    def from_file(cls, specification):
        path = Path(specification).expanduser().resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"target", "kernels", "bindings"}:
            raise ValueError("package build specification requires target, kernels and bindings")
        builder = cls(Target(**value["target"]))
        if not isinstance(value["kernels"], Mapping):
            raise TypeError("kernels must map names to explicit build entries")
        for name, entry in value["kernels"].items():
            if not isinstance(entry, dict):
                raise TypeError(f"package entry {name} must be an object")
            if "artifact" in entry:
                if set(entry) != {"artifact"}:
                    raise ValueError("an artifact entry cannot also specify source or build options")
                builder.add_artifact(name, Artifact(path.parent / entry["artifact"]))
                continue
            if not set(entry) <= {"source", "ir", "options", "selection"} or ("source" in entry) == ("ir" in entry):
                raise ValueError(f"{name} requires exactly one source entry or Canonical IR path")
            if "options" in entry and "selection" in entry:
                raise ValueError("a saved selection is a complete binding; options cannot override it")
            options = CompileOptions(**entry["options"]) if "options" in entry else None
            selection = TuningResult.load(path.parent / entry["selection"]) if "selection" in entry else None
            if "ir" in entry:
                # External frontends already supply canonical Weft IR. Its ABI
                # is emitted by the same compiler, without a Python signature.
                canonical = (path.parent / entry["ir"]).read_text(encoding="utf-8")
                builder.add_ir(name, canonical, options=options, selection=selection)
            else:
                location, separator, symbol = entry["source"].rpartition(":")
                if not separator:
                    raise ValueError("source must be file.py:kernel or module:kernel")
                if location.endswith(".py"):
                    location = str(path.parent / location)
                builder.add(name, load_definition(location + ":" + symbol), options=options, selection=selection)
        if not isinstance(value["bindings"], Mapping):
            raise TypeError("bindings must map groups to role mappings")
        for group, mapping in value["bindings"].items():
            if not isinstance(mapping, Mapping):
                raise TypeError("each binding group must contain a role mapping")
            for role, name in mapping.items():
                builder.bind(group, role, name)
        return builder

    def add_ir(self, name, canonical, *, options=None, selection=None):
        self._name(name)
        if not isinstance(canonical, str) or not canonical.strip():
            raise TypeError("expected nonempty Canonical Kernel IR")
        if options is not None and selection is not None:
            raise ValueError("choose explicit options or a saved selection, not both")
        if selection is not None:
            if not isinstance(selection, TuningResult):
                raise TypeError("selection must be a TuningResult")
            selection.check_program(canonical, self.target)
            options = selection.winner.options
        if options is None:
            options = CompileOptions()
        if not isinstance(options, CompileOptions):
            raise TypeError("package kernel options must be CompileOptions")
        self._entries[name] = (CanonicalSource(canonical), options)
        return self


class CanonicalSource:
    def __init__(self, canonical):
        self.canonical = canonical

    def build(self, **arguments):
        from .compiler import build_ir

        return build_ir(self.canonical, **arguments)
