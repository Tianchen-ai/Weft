from __future__ import annotations

import json
import csv
import io
import math
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from .cache import ArtifactCache, canonical_json, publish_json
from .driver import CandidateRejected, file_identity
from .compilation import CompilationFailure
from .configuration import Config, expand_configurations


@dataclass(frozen=True)
class TuningPolicy:
    max_candidates: int = 8
    deadline_s: float = 120.0
    warmup: int = 2
    samples: int = 5
    repetitions: int = 5
    max_state_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        for name in ("max_candidates", "samples", "repetitions", "max_state_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        if not math.isfinite(self.deadline_s) or self.deadline_s <= 0:
            raise ValueError("deadline_s must be finite and positive")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ValidationError(ValueError):
    """A candidate violated the author's numerical contract."""


@dataclass(frozen=True)
class CandidateResult:
    config: Config
    status: str
    samples_ns: tuple[float, ...] = ()
    validated: bool = False
    message: str = ""
    failure: CompilationFailure | None = None

    def __post_init__(self):
        if not isinstance(self.config, Config):
            raise TypeError("candidate result requires a complete Config")
        object.__setattr__(self, "samples_ns", tuple(self.samples_ns))
        if self.status not in ("measured", "single-legal", "compile-rejected", "validation-rejected", "budget"):
            raise ValueError(f"unknown candidate status: {self.status}")
        if self.status == "measured" and not self.samples_ns:
            raise ValueError("a measured candidate must contain timing samples")
        if any(not math.isfinite(value) or value <= 0 for value in self.samples_ns):
            raise ValueError("candidate timing samples must be finite and positive")
        if type(self.validated) is not bool:
            raise TypeError("validated must be an explicit numerical verdict")
        if self.status == "compile-rejected":
            if not isinstance(self.failure, CompilationFailure) or not self.failure.rejects_candidate:
                raise ValueError("compile-rejected requires the compiler's resource/unsupported failure")
            if self.samples_ns or self.validated:
                raise ValueError("a compiler-rejected candidate cannot have execution evidence")
        elif self.failure is not None:
            raise ValueError("only compiler-rejected candidates carry a compiler failure")
        if self.status == "single-legal" and self.samples_ns:
            raise ValueError("single-legal skips timing and cannot contain samples")
        if self.status == "validation-rejected" and (self.validated or self.samples_ns):
            raise ValueError("a validation rejection cannot retain successful execution evidence")

    @property
    def median_ns(self):
        return statistics.median(self.samples_ns) if self.samples_ns else None

    def to_dict(self):
        return {"config": self.config.to_dict(), "status": self.status,
                "samples_ns": list(self.samples_ns), "validated": self.validated,
                "message": self.message,
                "failure": None if self.failure is None else self.failure.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
                "config", "status", "samples_ns", "validated", "message", "failure"}:
            raise ValueError("candidate result requires its complete timing and compiler failure contract")
        return cls(Config.from_dict(value["config"]), value["status"],
                   tuple(value["samples_ns"]), value["validated"], value["message"],
                   None if value["failure"] is None else CompilationFailure.from_dict(value["failure"]))


@dataclass(frozen=True)
class TuningResult:
    winner: Config
    candidates: tuple[CandidateResult, ...]
    timing_scope: str
    budget_exhausted: bool
    elapsed_s: float
    cache_hit: bool = False
    record_path: Path | None = None
    identity: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        configurations(tuple(item.config for item in self.candidates),
                       TuningPolicy(max_candidates=max(1, len(self.candidates))))
        selected = [item for item in self.candidates if item.config == self.winner]
        if len(selected) != 1 or selected[0].status not in ("measured", "single-legal"):
            raise ValueError("winner must identify one successfully selected candidate")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("selection elapsed time must be finite and nonnegative")
        if not self.timing_scope:
            raise ValueError("selection must retain its timing scope")
        if type(self.budget_exhausted) is not bool:
            raise TypeError("budget_exhausted must be an explicit Boolean")
        self._check_selection_contract()

    def _check_selection_contract(self):
        """A persisted winner must be justified by its declared complete study.

        Artifact publication and cache locking prevent partially written files.
        They do not prove that a decoded record contains every candidate, full
        samples or the winner implied by those samples. Check those obligations
        before a cache hit can skip execution or a deployment can bind a winner.
        """
        if any(item.status == "budget" for item in self.candidates) and not self.budget_exhausted:
            raise ValueError("incomplete candidates require an exhausted-budget record")
        single = [item for item in self.candidates if item.status == "single-legal"]
        measured = [item for item in self.candidates if item.status == "measured"]
        if single:
            if len(single) != 1 or measured or any(item.status == "budget" for item in self.candidates):
                raise ValueError("single-legal requires exactly one legal candidate and no omitted candidates")
        elif measured:
            expected = min(measured, key=lambda item: item.median_ns)
            if self.winner != expected.config:
                raise ValueError("winner is not the first minimum of the completed candidate timings")
        if not self.identity:
            return
        if not isinstance(self.identity, dict) or self.identity.get("operation") != "select-configuration":
            raise ValueError("selection identity must describe an explicit configuration study")
        required = {"operation", "workload", "selector", "configs", "policy", "timing_scope"}
        if set(self.identity) != required:
            raise ValueError("selection identity has an incomplete study contract")
        declared = tuple(Config.from_dict(value) for value in self.identity["configs"])
        if declared != tuple(item.config for item in self.candidates):
            raise ValueError("selection must preserve every declared candidate in declaration order")
        if self.timing_scope != self.identity["timing_scope"]:
            raise ValueError("selection timing scope disagrees with its declared study")
        policy = TuningPolicy(**self.identity["policy"])
        configurations(declared, policy)
        for item in self.candidates:
            if item.status == "measured" and len(item.samples_ns) != policy.samples:
                raise ValueError("measured candidate does not contain every requested timing sample")
            if item.status == "budget" and len(item.samples_ns) >= policy.samples:
                raise ValueError("budget candidate cannot contain a complete set of timing samples")
        workload = self.identity["workload"]
        if workload.get("kind") == "native-kernel" and workload.get("validation") is not None:
            if any(not item.validated for item in (*single, *measured)):
                raise ValueError("selected native candidates must satisfy the declared numerical validator")

    @property
    def selected(self):
        return next(item for item in self.candidates if item.config == self.winner)

    @property
    def source_groups(self):
        """Physical comparisons grouped by exactly equal author bindings.

        Each group winner is local to that author program. Comparing winners
        between groups changes source bindings and must not be presented as a
        backend-only optimization. Rejected and incomplete choices stay visible.
        """
        groups = {}
        for item in self.candidates:
            key = canonical_json(dict(item.config.source))
            groups.setdefault(key, []).append(item)
        result = []
        for key, candidates in groups.items():
            measured = [item for item in candidates if item.status == "measured"]
            best = min(measured, key=lambda item: item.median_ns) if measured else None
            if best is None:
                best = next((item for item in candidates if item.status == "single-legal"), None)
            result.append({"source": json.loads(key),
                           "winner": None if best is None else best.config.to_dict(),
                           "median_ns": None if best is None else best.median_ns,
                           "candidates": [item.to_dict() for item in candidates]})
        return tuple(result)

    def compare(self, reference, candidate=None):
        """Compare completed timings from this same study and workload.

        The classification concerns bindings, not causality: it does not claim
        that a physical option affected emitted instructions, or that an author
        binding improved model E2E. No ratio is made from incomplete timing.
        """
        if candidate is None:
            candidate = self.winner
        def find(config):
            matches = [item for item in self.candidates if item.config == config]
            if len(matches) != 1:
                raise ValueError("comparison requires two configurations from this selection")
            return matches[0]
        before, after = find(reference), find(candidate)
        changes = candidate.changes_from(reference)
        kind = ("source-and-physical-binding" if changes["physical"] else "source-binding") if changes["source"] else (
            "physical-binding" if changes["physical"] else "same-binding")
        measured = before.status == after.status == "measured"
        return {"reference": reference.to_dict(), "candidate": candidate.to_dict(),
                "kind": kind, "changes": changes, "timing_scope": self.timing_scope,
                "reference_median_ns": before.median_ns if measured else None,
                "candidate_median_ns": after.median_ns if measured else None,
                "speedup": before.median_ns / after.median_ns if measured else None,
                "validated": before.validated and after.validated}

    def summary(self):
        """A compact selection view, with explicit comparison provenance."""
        return {"winner": self.winner.to_dict(), "timing_scope": self.timing_scope,
                "budget_exhausted": self.budget_exhausted, "elapsed_s": self.elapsed_s,
                "source_groups": self.source_groups,
                "comparison_to_first": self.compare(self.candidates[0].config),
                "candidates": [{"config": item.config.to_dict(), "status": item.status,
                                "median_ns": item.median_ns, "validated": item.validated,
                                "message": item.message,
                                "failure": None if item.failure is None else item.failure.to_dict()}
                               for item in self.candidates]}

    def _indices(self, indices):
        if indices is None:
            return tuple(range(len(self.candidates)))
        indices = tuple(indices)
        if any(type(index) is not int or not 0 <= index < len(self.candidates) for index in indices):
            raise ValueError("candidate indices must identify saved candidates")
        if len(indices) != len(set(indices)):
            raise ValueError("candidate inspection cannot repeat an index")
        return indices

    def rows(self, *, reference=0, indices=None):
        """One flat row per candidate, suitable for application tables or CSV.

        Durations and ratios have different availability: incomplete samples
        remain visible as raw samples, but never supply a performance ratio.
        The comparison is within this exact recorded workload and policy.
        Source-group numbering follows first appearance in the declaration.
        """
        self._indices((reference,))
        original = self.candidates[reference].config
        groups = {}
        for item in self.candidates:
            key = canonical_json(dict(item.config.source))
            if key not in groups:
                groups[key] = len(groups)
        result = []
        for index in self._indices(indices):
            item = self.candidates[index]
            comparison = self.compare(original, item.config)
            failure = item.failure
            result.append({
                "candidate_index": index, "name": item.config.name, "status": item.status,
                "winner": item.config == self.winner, "validated": item.validated,
                "source_group": groups[canonical_json(dict(item.config.source))],
                "source_bindings": dict(item.config.source), "physical_bindings": dict(item.config.physical),
                "sample_count": len(item.samples_ns), "samples_ns": list(item.samples_ns),
                "median_ns": item.median_ns if item.status == "measured" else None,
                "reference_index": reference, "comparison_kind": comparison["kind"],
                "reference_over_candidate": comparison["speedup"],
                "changed_source_parameters": comparison["changes"]["source"],
                "changed_physical_parameters": comparison["changes"]["physical"],
                "failure_category": None if failure is None else failure.category,
                "failure_stage": None if failure is None else failure.stage,
                "failure_pass": None if failure is None else failure.pass_name,
                "message": item.message, "timing_scope": self.timing_scope,
                "budget_exhausted": self.budget_exhausted,
            })
        return tuple(result)

    def to_csv(self, *, reference=0, indices=None):
        """Return CSV text; no report path is created or silently overwritten."""
        rows = self.rows(reference=reference, indices=indices)
        fields = tuple(self.rows(reference=reference, indices=(0,))[0])
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
                             for name, value in row.items()})
        return stream.getvalue()

    def render(self, *, indices=None):
        """Explain selection state without conflating source and physical changes."""
        lines = [f"Timing scope: {self.timing_scope}",
                 f"Winner: {self.winner.name or '(unnamed configuration)'}",
                 f"Budget exhausted: {self.budget_exhausted}"]
        comparison = self.compare(self.candidates[0].config)
        lines.append(f"Winner versus first candidate: {comparison['kind']}")
        if comparison["speedup"] is not None:
            lines.append(f"Measured ratio: {comparison['speedup']:.6g}")
        if comparison["changes"]["source"]:
            lines.append("Source bindings changed; this ratio is not a backend-only improvement.")
        for index in self._indices(indices):
            item = self.candidates[index]
            selected = " (winner)" if item.config == self.winner else ""
            lines.extend(("", f"Candidate {index}: {item.config.name or '(unnamed configuration)'}{selected}",
                          f"  Status: {item.status}", f"  Numerically validated: {item.validated}",
                          "  Source: " + canonical_json(dict(item.config.source)),
                          "  Physical: " + canonical_json(dict(item.config.physical))))
            if item.samples_ns:
                label = "Incomplete timing" if item.status == "budget" else "Timing"
                lines.append(f"  {label}: median {item.median_ns:g} ns from {len(item.samples_ns)} samples")
            elif item.status == "single-legal":
                lines.append("  Timing: skipped for the only legal candidate")
            else:
                lines.append("  Timing: not measured")
            if item.failure is not None:
                lines.extend("  " + line for line in item.failure.render().splitlines())
            elif item.message:
                lines.extend("  " + line for line in item.message.splitlines())
        return "\n".join(lines)

    def to_dict(self):
        return {"winner": self.winner.to_dict(), "candidates": [item.to_dict() for item in self.candidates],
                "timing_scope": self.timing_scope, "budget_exhausted": self.budget_exhausted,
                "elapsed_s": self.elapsed_s, "identity": self.identity}

    @classmethod
    def from_dict(cls, value, *, cache_hit=False, record_path=None):
        if not isinstance(value, dict) or set(value) != {
                "winner", "candidates", "timing_scope", "budget_exhausted", "elapsed_s", "identity"}:
            raise ValueError("selection result requires the complete candidate and measurement record")
        return cls(Config.from_dict(value["winner"]),
                   tuple(CandidateResult.from_dict(item) for item in value["candidates"]),
                   value["timing_scope"], value["budget_exhausted"], value["elapsed_s"],
                   cache_hit, record_path, value["identity"])

    def save(self, path):
        return publish_json(Path(path), self.to_dict())

    def check_program(self, canonical, target):
        """Check a fixed deployment binding without claiming new measurements."""
        workload = self.identity["workload"]
        if workload["canonical"] != canonical:
            raise ValueError("selected binding belongs to another canonical author program")
        if workload["kind"] == "native-kernel":
            measured = workload["target"]
        elif workload["kind"] == "numerical-driver":
            measured = workload["machine"]["target"]
        else:
            raise ValueError("selection does not describe a native kernel or a numerical driver")
        if any(measured.get(name) != value for name, value in target.to_dict().items()):
            raise ValueError("selected binding was measured for another target ABI, ISA or VLEN")

    @classmethod
    def load(cls, path):
        """Read a selection for explicit fixed-configuration JIT/AOT use."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class TuningError(RuntimeError):
    def __init__(self, message, candidates):
        self.candidates = tuple(candidates)
        details = "\n".join(f"  {item.config.name or index}: {item.status}: {item.message}"
                            for index, item in enumerate(candidates))
        super().__init__(message + ("\n" + details if details else ""))


def configurations(values, policy: TuningPolicy) -> tuple[Config, ...]:
    if not isinstance(policy, TuningPolicy):
        raise TypeError("configuration selection requires a TuningPolicy")
    return expand_configurations(values, max_candidates=policy.max_candidates)


class Study:
    """Bounded selection shared by native calls and external numerical drivers.

    A trial owns its resources and implements validate(), measure_ns(), close().
    validate returns whether a numerical contract was checked. measure_ns must
    reset mutable state before each invocation and exclude the reset from its
    declared timing scope. Only typed compiler candidate rejections and numerical
    failures reject a candidate; tool, artifact and programming errors propagate.
    The deadline is cooperative at call
    boundaries, since interrupting native code cannot safely release its storage.
    """

    def __init__(self, configs, *, identity: dict, timing_scope: str,
                 policy: TuningPolicy | None = None, cache=None):
        self.policy = TuningPolicy() if policy is None else policy
        self.configs = configurations(configs, self.policy)
        if not identity or not isinstance(timing_scope, str) or not timing_scope:
            raise ValueError("selection needs an explicit workload identity and timing scope")
        self.identity = json.loads(canonical_json({"operation": "select-configuration", "workload": identity,
                                                   "selector": file_identity(__file__),
                                                   "configs": [config.to_dict() for config in self.configs],
                                                   "policy": self.policy.to_dict(), "timing_scope": timing_scope}))
        self.timing_scope = timing_scope
        self.cache = cache if isinstance(cache, ArtifactCache) else ArtifactCache(cache)

    def run(self, prepare, *, refresh=False) -> TuningResult:
        stored, hit, path = self.cache.record(self.identity, lambda: self._run(prepare).to_dict(), refresh=refresh)
        result = TuningResult.from_dict(stored, cache_hit=hit, record_path=path)
        if result.identity != self.identity or result.winner not in self.configs:
            raise TuningError("selection record disagrees with its complete configuration contract", result.candidates)
        return result

    def _run(self, prepare) -> TuningResult:
        started = time.monotonic()
        deadline = started + self.policy.deadline_s
        results = [None] * len(self.configs)
        live = {}
        samples = [[] for _ in self.configs]
        validated = [False] * len(self.configs)
        resources = ExitStack()

        def expired():
            return time.monotonic() >= deadline

        def finish(index):
            if any(item is None for item in results):
                raise RuntimeError("selection did not close every declared candidate")
            exhausted = expired() or any(item.status == "budget" for item in results)
            return TuningResult(self.configs[index], tuple(results), self.timing_scope,
                                exhausted, time.monotonic() - started, identity=self.identity)
        with resources:
            # Compile every admitted candidate before any measurement. Compilation
            # and native loading must not change the thermal/cache conditions in
            # the middle of a timing round.
            for index, config in enumerate(self.configs):
                if expired():
                    results[index] = CandidateResult(config, "budget", message="deadline before compilation")
                    continue
                try:
                    trial = prepare(config)
                except CandidateRejected as error:
                    results[index] = CandidateResult(config, "compile-rejected",
                        message=str(error), failure=error.failure)
                    continue
                cleanup = ExitStack()
                cleanup.callback(trial.close)
                resources.callback(cleanup.close)
                live[index] = (trial, cleanup)

            # Validation always precedes warmup and touches private trial state.
            # Keep declaration indices stable even when an early candidate fails.
            for index, (trial, cleanup) in tuple(live.items()):
                if expired():
                    results[index] = CandidateResult(self.configs[index], "budget",
                                                     message="deadline before validation")
                    cleanup.close()
                    del live[index]
                    continue
                try:
                    verdict = trial.validate()
                    if type(verdict) is not bool:
                        raise TypeError("trial.validate() must report an explicit Boolean verdict")
                    validated[index] = verdict
                except (ValidationError, AssertionError) as error:
                    results[index] = CandidateResult(self.configs[index], "validation-rejected",
                                                     message=str(error))
                    cleanup.close()
                    del live[index]

            # A sole survivor of complete legality and validation needs no timing.
            # Missing candidates due to budget are not evidence of sole legality.
            incomplete = any(item is not None and item.status == "budget" for item in results)
            if len(live) == 1 and not incomplete:
                index = next(iter(live))
                results[index] = CandidateResult(self.configs[index], "single-legal",
                                                 validated=validated[index])
                return finish(index)

            def ordered(round_index):
                indices = tuple(live)
                if not indices:
                    return ()
                shift = round_index % len(indices)
                indices = indices[shift:] + indices[:shift]
                return indices if round_index % 2 == 0 else tuple(reversed(indices))

            def measure(index):
                value = float(live[index][0].measure_ns())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("a trial must report finite positive native time")
                return value

            def reject(index, error):
                results[index] = CandidateResult(self.configs[index], "validation-rejected",
                                                 message=str(error))
                live[index][1].close()
                del live[index]
                samples[index].clear()

            for round_index in range(self.policy.warmup):
                for index in ordered(round_index):
                    if expired():
                        break
                    try:
                        measure(index)
                    except (ValidationError, AssertionError) as error:
                        reject(index, error)
                if expired():
                    break

            # A sample is a complete repetition block. Rotate candidate order
            # between rounds; no candidate gets its entire warm cache run before
            # another candidate has executed once.
            for round_index in range(self.policy.samples):
                if expired():
                    break
                for index in ordered(round_index):
                    values = []
                    try:
                        for _ in range(self.policy.repetitions):
                            if expired():
                                break
                            values.append(measure(index))
                    except (ValidationError, AssertionError) as error:
                        reject(index, error)
                        continue
                    if len(values) == self.policy.repetitions:
                        samples[index].append(statistics.median(values))
                if expired():
                    break

            for index in live:
                complete = len(samples[index]) == self.policy.samples
                results[index] = CandidateResult(self.configs[index], "measured" if complete else "budget",
                    tuple(samples[index]), validated[index],
                    "" if complete else "deadline during warmup or measurement")
            measured = [index for index, item in enumerate(results) if item.status == "measured"]
            if not measured:
                raise TuningError("no configuration completed the requested validation and measurement", results)
            winner = min(measured, key=lambda index: results[index].median_ns)
            return finish(winner)
