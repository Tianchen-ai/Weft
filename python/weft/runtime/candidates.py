"""Build and deploy an explicit finite set of ordinary compiler artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifact import Artifact
from .cache import canonical_json
from .compilation import CompilationFailure
from .configuration import Config, resolve_configurations
from .driver import CandidateRejected, CompilationError, Toolchain, file_identity
from .target import Target


@dataclass(frozen=True)
class CandidateArtifact:
    """One successful compilation or a compiler-owned legal rejection."""

    config: Config
    artifact: Artifact | None = None
    failure: CompilationFailure | None = None
    message: str = ""

    def __post_init__(self):
        if not isinstance(self.config, Config):
            raise TypeError("candidate artifact requires a complete Config")
        if (self.artifact is None) == (self.failure is None):
            raise ValueError("candidate must contain exactly one artifact or compiler rejection")
        if self.artifact is not None:
            if not isinstance(self.artifact, Artifact) or self.artifact.options != self.config.options:
                raise ValueError("candidate artifact disagrees with its complete configuration")
            if self.message:
                raise ValueError("a successful candidate cannot contain a rejection message")
        elif not isinstance(self.failure, CompilationFailure) or not self.failure.rejects_candidate:
            raise ValueError("only an unsupported/resource compiler rejection may remove a candidate")
        if not isinstance(self.message, str):
            raise TypeError("candidate diagnostic message must be a string")

    @property
    def status(self):
        return "compiled" if self.artifact is not None else "compile-rejected"

    def require_artifact(self):
        if self.failure is not None:
            raise CandidateRejected(self.message, failure=self.failure)
        return self.artifact


class BuiltCandidates:
    """Relocatable alternatives for one author definition and explicit target.

    Every entry is independently compiled through the ordinary two IR layers.
    This object neither ranks artifacts nor changes their selected physical IR.
    Native tuning can consume it with no DSL source module or system C compiler.
    Compiler rejections remain in the declared order and are not retried or
    replaced by a fallback when the collection is loaded.
    """

    def __init__(self, source, target, entries):
        from .compiler import KernelSource

        if not isinstance(source, KernelSource) or not isinstance(target, Target):
            raise TypeError("built candidates require a captured author source and explicit target")
        entries = tuple(entries)
        if not entries or any(not isinstance(entry, CandidateArtifact) for entry in entries):
            raise ValueError("built candidates require a nonempty sequence of CandidateArtifact entries")
        configs, interface = resolve_configurations(source, tuple(entry.config for entry in entries),
                                                    max_candidates=len(entries))
        if configs != tuple(entry.config for entry in entries):
            raise ValueError("built candidates must retain complete resolved source bindings")
        expected_signature = canonical_json(source.signature)
        for entry in entries:
            if entry.artifact is None:
                continue
            artifact = entry.artifact
            if artifact.target != target:
                raise ValueError("candidate collection contains a different build target")
            if artifact.canonical_ir != source.canonical:
                raise ValueError("candidate collection contains a different canonical author definition")
            if artifact.interface.semantic_dict() != interface.semantic_dict():
                raise ValueError("candidate collection changes the public invocation contract")
            # Artifact.manifest is frozen; normalize its signature through JSON
            # rather than comparing list/tuple implementation representations.
            manifest = json.loads((artifact.path / "manifest.json").read_text(encoding="utf-8"))
            if canonical_json(manifest["signature"]) != expected_signature:
                raise ValueError("candidate collection changes Python defaults or parameter order")
        self.source = source
        self.target = target
        self.entries = entries
        self.configs = configs
        self.call_signature = interface

    @classmethod
    def build(cls, definition, configs, *, target, toolchain=None, native_library=True,
              cache=None, max_candidates=8):
        from .compiler import KernelSource

        source = definition if isinstance(definition, KernelSource) else KernelSource.capture(definition)
        configs, _ = resolve_configurations(source, configs, max_candidates=max_candidates)
        if not isinstance(target, Target):
            raise TypeError("candidate build requires an explicit Target")
        selected = Toolchain.from_environment() if toolchain is None else toolchain
        entries = []
        for config in configs:
            try:
                artifact = source.build(target=target, options=config.options, toolchain=selected,
                                        native_library=native_library, cache=cache)
                entries.append(CandidateArtifact(config, artifact=artifact))
            except CandidateRejected as error:
                entries.append(CandidateArtifact(config, failure=error.failure, message=str(error)))
        # Even an all-rejected collection is useful for precise compiler
        # diagnostics. It can be saved and inspected but cannot select a winner.
        return cls(source, target, entries)

    @property
    def artifacts(self):
        return tuple(entry.artifact for entry in self.entries if entry.artifact is not None)

    @property
    def rejections(self):
        return tuple(entry for entry in self.entries if entry.failure is not None)

    @property
    def native(self):
        return bool(self.artifacts) and all(artifact.manifest["native_library"] for artifact in self.artifacts)

    def entry(self, config):
        if not isinstance(config, Config):
            raise TypeError("candidate lookup requires a Config")
        for entry in self.entries:
            if entry.config == config:
                return entry
        raise ValueError("configuration is outside the declared candidate collection")

    def link(self, *, toolchain=None, cache=None):
        """Link existing C only. No candidate or physical operation is reselected."""
        linked = []
        for entry in self.entries:
            if entry.artifact is None:
                linked.append(entry)
            else:
                linked.append(CandidateArtifact(entry.config,
                    artifact=entry.artifact.link(toolchain=toolchain, cache=cache)))
        return BuiltCandidates(self.source, self.target, linked)

    def summary(self):
        """Inspect compilation state without native discovery or execution."""
        return {"target": self.target.to_dict(), "native": self.native,
                "candidates": [{"config": entry.config.to_dict(), "status": entry.status,
                                "native_library": (entry.artifact.manifest["native_library"]
                                                   if entry.artifact is not None else False),
                                "failure": None if entry.failure is None else entry.failure.to_dict(),
                                "message": entry.message} for entry in self.entries]}

    def identity(self):
        """Exact local executable identities; no content digest or performance claim."""
        entries = []
        for entry in self.entries:
            record = {"config": entry.config.to_dict()}
            if entry.artifact is None:
                record["failure"] = entry.failure.to_dict()
            else:
                artifact = entry.artifact
                record["manifest"] = json.loads((artifact.path / "manifest.json").read_text(encoding="utf-8"))
                record["library"] = file_identity(artifact.library_path)
            entries.append(record)
        return {"target": self.target.to_dict(), "entries": entries}

    def save(self, destination):
        """Publish all choices and rejections atomically into a new directory."""
        destination = Path(destination).expanduser().absolute()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".weft-candidates-", dir=destination.parent) as temporary:
            root = Path(temporary) / "candidates"
            root.mkdir()
            records = []
            for index, entry in enumerate(self.entries):
                name = f"candidate-{index}"
                if entry.artifact is not None:
                    entry.artifact.save(root / name)
                records.append({"config": entry.config.to_dict(),
                                "artifact": name if entry.artifact is not None else None,
                                "failure": None if entry.failure is None else entry.failure.to_dict(),
                                "message": entry.message})
            (root / "canonical.mlir").write_text(self.source.canonical, encoding="utf-8")
            value = {"kind": "weft-compiled-candidates", "target": self.target.to_dict(),
                     "signature": self.source.signature, "interface": self.source.description,
                     "candidates": records}
            (root / "candidates.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
            # Read before publishing, so malformed source metadata or copied
            # artifacts do not leave a directory that looks ready for deployment.
            BuiltCandidates.load(root)
            os.rename(root, destination)
        return BuiltCandidates.load(destination)

    @classmethod
    def load(cls, directory):
        from .compiler import KernelSource

        root = Path(directory).expanduser().absolute()
        value = json.loads((root / "candidates.json").read_text(encoding="utf-8"))
        required = {"kind", "target", "signature", "interface", "candidates"}
        if not isinstance(value, dict) or set(value) != required or value["kind"] != "weft-compiled-candidates":
            raise ValueError("compiled candidate manifest requires its complete source and target contract")
        if not isinstance(value["candidates"], list) or not value["candidates"]:
            raise ValueError("compiled candidate manifest must contain candidates")
        source = KernelSource((root / "canonical.mlir").read_text(encoding="utf-8"),
                              value["signature"], value["interface"])
        entries = []
        for index, record in enumerate(value["candidates"]):
            if not isinstance(record, dict) or set(record) != {"config", "artifact", "failure", "message"}:
                raise ValueError("candidate manifest entry requires config, artifact, failure and message")
            name = record["artifact"]
            if name is not None and name != f"candidate-{index}":
                raise ValueError("candidate artifact must name its own contained directory")
            artifact = None
            if name is not None:
                path = root / name
                if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
                    raise ValueError("candidate artifact directory escapes its collection")
                artifact = Artifact(path)
            failure = None if record["failure"] is None else CompilationFailure.from_dict(record["failure"])
            entries.append(CandidateArtifact(Config.from_dict(record["config"]), artifact, failure, record["message"]))
        return cls(source, Target(**value["target"]), entries)

    def autotune(self, *, key=(), policy=None, validator=None, validation_key=None,
                 toolchain=None, cache=None):
        """Select prebuilt native entries on actual arguments, with private trials."""
        from .autotune import AutotunedKernel

        if not self.native and self.artifacts:
            raise CompilationError("candidate collection contains source-only artifacts; link it explicitly before tuning")
        return AutotunedKernel.from_candidates(self, key=key, policy=policy, validator=validator,
            validation_key=validation_key, toolchain=toolchain, cache=cache)

    def fixed(self, selection):
        """Extract the measured winner as an ordinary deployable Artifact."""
        from .tuning import TuningResult

        if not isinstance(selection, TuningResult):
            raise TypeError("fixed candidate selection requires a TuningResult")
        selection.check_program(self.source.canonical, self.target)
        if tuple(item.config for item in selection.candidates) != self.configs:
            raise ValueError("selection belongs to a different declared candidate collection")
        return self.entry(selection.winner).require_artifact()


def build_candidates(definition, configs, *, target, toolchain=None, native_library=True,
                     cache=None, max_candidates=8):
    return BuiltCandidates.build(definition, configs, target=target, toolchain=toolchain,
        native_library=native_library, cache=cache, max_candidates=max_candidates)


def build_candidates_ir(canonical, configs, *, target, toolchain=None, native_library=True,
                        cache=None, max_candidates=8):
    from .compiler import KernelSource

    source = KernelSource.from_ir(canonical, toolchain=toolchain)
    return BuiltCandidates.build(source, configs, target=target, toolchain=toolchain,
        native_library=native_library, cache=cache, max_candidates=max_candidates)
