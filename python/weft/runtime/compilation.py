"""Compiler-owned pass execution and diagnostic records.

These records describe executed compiler passes. They do not add an IR layer,
infer decisions from diagnostic text, or estimate runtime performance.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .driver import CompilationError


def _record(value, names, description):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise CompilationError(f"{description} requires exactly {', '.join(names)}")


def _text(value, description, *, empty=False):
    if not isinstance(value, str) or (not empty and not value):
        raise CompilationError(f"{description} must be a {'possibly empty' if empty else 'nonempty'} string")
    return value


def _sequence(value, description):
    if not isinstance(value, (list, tuple)):
        raise CompilationError(f"{description} must be a sequence")
    return value


@dataclass(frozen=True)
class CompilationStep:
    stage: str
    pass_name: str
    status: str
    invocation: str

    @classmethod
    def from_dict(cls, value):
        _record(value, ("stage", "pass", "status", "invocation"), "compiler pipeline step")
        result = cls(_text(value["stage"], "compiler stage"),
                     _text(value["pass"], "compiler pass"), value["status"],
                     _text(value["invocation"], "compiler pass invocation"))
        if result.status not in ("completed", "failed"):
            raise CompilationError("compiler pass status must be completed or failed")
        return result

    def to_dict(self):
        return {"stage": self.stage, "pass": self.pass_name, "status": self.status,
                "invocation": self.invocation}


def _pipeline(value):
    result = tuple(CompilationStep.from_dict(step)
                   for step in _sequence(value, "compiler pipeline"))
    failures = [index for index, step in enumerate(result) if step.status == "failed"]
    if failures and failures != [len(result) - 1]:
        raise CompilationError("a failed compiler pass must terminate its execution record")
    return result


@dataclass(frozen=True)
class CompilationPipeline:
    """Actual ordered pass executions from the compiler's selected entry point."""

    entry: str
    pipeline: tuple[CompilationStep, ...]

    @classmethod
    def from_dict(cls, value):
        _record(value, ("entry", "pipeline"), "compilation report")
        if value["entry"] not in ("canonical", "layout-input", "contraction-input", "panel-transfer-input", "physical"):
            raise CompilationError("compilation report has an unknown entry point")
        steps = _pipeline(value["pipeline"])
        if not steps:
            raise CompilationError("a successful artifact requires its executed compiler passes")
        if any(step.status != "completed" for step in steps):
            raise CompilationError("a successful artifact contains a failed compiler pass")
        return cls(value["entry"], steps)

    @property
    def stages(self):
        """Stages in first-execution order; repeated passes remain in pipeline."""
        return tuple(dict.fromkeys(step.stage for step in self.pipeline))

    def matching(self, *, stage=None, pass_name=None):
        return tuple(step for step in self.pipeline
                     if (stage is None or step.stage == stage)
                     and (pass_name is None or step.pass_name == pass_name))

    def to_dict(self):
        return {"entry": self.entry, "pipeline": [step.to_dict() for step in self.pipeline]}

    def as_pass_pipeline(self):
        """The exact executed pass invocations for an MLIR module pass manager."""
        return "builtin.module(" + ",".join(step.invocation for step in self.pipeline) + ")"

    def render(self):
        lines = [f"Compiler entry: {self.entry}"]
        lines.extend(f"{index + 1:3d}  {step.stage}  {step.pass_name}  {step.status}"
                     for index, step in enumerate(self.pipeline))
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DiagnosticNote:
    message: str
    location: str

    @classmethod
    def from_dict(cls, value):
        _record(value, ("message", "location"), "compiler diagnostic note")
        return cls(_text(value["message"], "diagnostic note"),
                   _text(value["location"], "diagnostic note location", empty=True))

    def to_dict(self):
        return {"message": self.message, "location": self.location}


@dataclass(frozen=True)
class CompilationDiagnostic:
    severity: str
    message: str
    location: str
    notes: tuple[DiagnosticNote, ...]

    @classmethod
    def from_dict(cls, value):
        _record(value, ("severity", "message", "location", "notes"), "compiler diagnostic")
        if value["severity"] not in ("error", "warning", "remark", "note"):
            raise CompilationError("compiler diagnostic has an unknown severity")
        return cls(value["severity"], _text(value["message"], "diagnostic message"),
                   _text(value["location"], "diagnostic location", empty=True),
                   tuple(DiagnosticNote.from_dict(note)
                         for note in _sequence(value["notes"], "diagnostic notes")))

    def to_dict(self):
        return {"severity": self.severity, "message": self.message, "location": self.location,
                "notes": [note.to_dict() for note in self.notes]}


@dataclass(frozen=True)
class CompilationFailure:
    """Explicit compiler classification, never a classification of stderr text."""

    category: str
    stage: str
    pass_name: str
    diagnostics: tuple[CompilationDiagnostic, ...]
    pipeline: tuple[CompilationStep, ...]

    @classmethod
    def from_dict(cls, value):
        _record(value, ("kind", "category", "stage", "pass", "diagnostics", "pipeline"),
                "compiler failure")
        if value["kind"] != "weft-compilation-failure":
            raise CompilationError("unsupported compiler failure record")
        if value["category"] not in ("resource-limit", "unsupported", "invalid-input", "compiler-error"):
            raise CompilationError("compiler failure has an unknown category")
        result = cls(value["category"], _text(value["stage"], "failed compiler stage"),
                     _text(value["pass"], "failed compiler pass", empty=True),
                     tuple(CompilationDiagnostic.from_dict(item)
                           for item in _sequence(value["diagnostics"], "compiler diagnostics")),
                     _pipeline(value["pipeline"]))
        if not any(item.severity == "error" for item in result.diagnostics):
            raise CompilationError("a compiler failure must contain an error diagnostic")
        if result.pass_name:
            if not result.pipeline or result.pipeline[-1].status != "failed":
                raise CompilationError("a named failed pass requires its terminal failed execution")
            step = result.pipeline[-1]
            if (result.stage, result.pass_name) != (step.stage, step.pass_name):
                raise CompilationError("failed pass identity disagrees with its pipeline record")
        elif result.pipeline and result.pipeline[-1].status == "failed":
            raise CompilationError("a failed pass execution requires its stage and pass identity")
        return result

    @property
    def rejects_candidate(self):
        return self.category in ("resource-limit", "unsupported")

    def to_dict(self):
        return {"kind": "weft-compilation-failure", "category": self.category,
                "stage": self.stage, "pass": self.pass_name,
                "diagnostics": [item.to_dict() for item in self.diagnostics],
                "pipeline": [step.to_dict() for step in self.pipeline]}

    def as_pass_pipeline(self):
        """Replay the executed prefix through its failing pass on the same input."""
        if not self.pipeline:
            raise CompilationError("compiler failure has no executed pass prefix to replay")
        if not self.pass_name or self.pipeline[-1].status != "failed":
            raise CompilationError("failure occurred outside pass execution; its pass prefix does not reproduce it")
        return "builtin.module(" + ",".join(step.invocation for step in self.pipeline) + ")"

    def render(self):
        location = f"{self.stage}/{self.pass_name}" if self.pass_name else self.stage
        lines = [f"{self.category} at {location}"]
        for item in self.diagnostics:
            prefix = f"{item.location}: " if item.location else ""
            lines.append(f"{prefix}{item.severity}: {item.message}")
            for note in item.notes:
                prefix = f"{note.location}: " if note.location else ""
                lines.append(f"  {prefix}note: {note.message}")
        return "\n".join(lines)
