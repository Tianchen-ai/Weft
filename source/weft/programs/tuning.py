"""Explicit finite tuning recipes for public author programs.

The ordinary runtime owns trial storage, validation, timing and cached winners.
This module binds a recipe to a catalog entry and resolves fixed deployment
choices; it does not generate algorithms or infer a search space from a device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from weft.runtime import BuiltCandidates, Config, TuningPolicy, TuningResult
from weft.runtime.autotune import AutotunedKernel
from weft.runtime.compiler import KernelSource
from weft.runtime.tuning import configurations

from .catalog import get


@dataclass(frozen=True)
class Tuning:
    """A caller-owned candidate list and numerical validation contract.

    Source defaults fill omitted source parameters. Required ABI dimensions,
    such as a row program's N, are supplied by the public call and are identical
    in every candidate. A call cannot override a varying candidate parameter.
    Physical options are exactly those in each Config.
    """

    configs: tuple[Config, ...]
    policy: TuningPolicy = field(default_factory=TuningPolicy)
    key: tuple[str, ...] = ()
    validator: object = None
    validation_key: str | None = None
    candidates: BuiltCandidates | None = None

    def __post_init__(self):
        if not isinstance(self.policy, TuningPolicy):
            raise TypeError("tuning policy must be a TuningPolicy")
        object.__setattr__(self, "configs", configurations(self.configs, self.policy))
        if isinstance(self.key, str):
            raise TypeError("tuning key must be a sequence of scalar or shape names")
        key = tuple(self.key)
        if any(not isinstance(name, str) or not name for name in key) or len(set(key)) != len(key):
            raise ValueError("tuning key entries must be distinct nonempty names")
        object.__setattr__(self, "key", key)
        if self.validator is not None and not callable(self.validator):
            raise TypeError("tuning validator must be callable")
        if (self.validator is None) != (self.validation_key is None):
            raise ValueError("a validator and its validation_key must be supplied together")
        if self.validation_key is not None and (not isinstance(self.validation_key, str) or not self.validation_key):
            raise ValueError("validation_key must identify the numerical contract")
        if self.candidates is not None:
            if not isinstance(self.candidates, BuiltCandidates) or self.candidates.configs != self.configs:
                raise ValueError("prebuilt tuning recipe must preserve its complete candidate collection")

    @classmethod
    def from_candidates(cls, candidates, *, policy=None, key=(), validator=None, validation_key=None):
        """Use precompiled choices in Operators or Torch Library, with no JIT."""
        if not isinstance(candidates, BuiltCandidates):
            raise TypeError("expected a BuiltCandidates collection")
        return cls(candidates.configs, TuningPolicy() if policy is None else policy,
                   key, validator, validation_key, candidates)

    def bind(self, name, **bindings):
        entry = get(name)
        configs = entry.configurations(self.configs, max_candidates=self.policy.max_candidates, **bindings)
        if self.candidates is not None:
            if configs != self.candidates.configs:
                raise ValueError("prebuilt candidates cannot change source bindings for a public program call")
            expected = KernelSource.capture(entry.definition).interface(configs[0].options.meta).semantic_dict()
            if self.candidates.call_signature.semantic_dict() != expected:
                raise ValueError("prebuilt candidates have an incompatible public program ABI")
            if any(artifact.metadata["symbol"] != entry.definition.__name__ for artifact in self.candidates.artifacts):
                raise ValueError("prebuilt candidates name a different public author program")
        return configs

    def create(self, name, *, bindings, toolchain, cache):
        configs = self.bind(name, **bindings)
        if self.candidates is not None:
            return self.candidates.autotune(key=self.key, policy=self.policy,
                validator=self.validator, validation_key=self.validation_key, toolchain=toolchain, cache=cache)
        return get(name).autotune(configs, key=self.key,
                                 policy=self.policy, validator=self.validator,
                                 validation_key=self.validation_key,
                                 toolchain=toolchain, cache=cache)


def recipes(values, fixed):
    selected = dict(values or {})
    overlap = selected.keys() & fixed.keys()
    if overlap:
        raise ValueError(f"programs cannot have both fixed options and tuning recipes: {sorted(overlap)}")
    for name, recipe in selected.items():
        entry = get(name)
        if not isinstance(recipe, Tuning):
            raise TypeError(f"{name} requires a Tuning recipe")
        # Public ABI dimensions can be supplied later by a shape-aware call.
        # Check names now without inventing any value for those dimensions.
        allowed = set(dict(entry.defaults)) | set(entry.required)
        for config in recipe.configs:
            unknown = set(config.options.meta) - allowed
            if unknown:
                raise ValueError(f"{name} has no source parameters {sorted(unknown)}")
            if set(entry.required) <= config.options.meta.keys():
                entry.options(options=config.options)
        if recipe.candidates is not None:
            recipe.bind(name)
    return MappingProxyType(selected)


def candidate_artifacts(kernel):
    if isinstance(kernel, AutotunedKernel):
        return kernel.artifacts
    if kernel.compiled is None:
        return ()
    return (kernel.compiled.artifact,)


def selected_artifact(kernel, selection=None):
    """Select one already materialized configuration for a package role.

    Shape-specific winners can differ. Exporting such a callable requires an
    explicit measured Config or TuningResult instead of silently taking the
    last workload that happened to run in this process.
    """
    artifacts = candidate_artifacts(kernel)
    if not artifacts:
        raise ValueError("materialize or tune the program on a real workload before exporting")
    if not isinstance(kernel, AutotunedKernel):
        if selection is not None:
            raise ValueError("a fixed program does not accept a tuning selection")
        return artifacts[0]
    if selection is None:
        if len(artifacts) != 1:
            raise ValueError("multiple workload winners require an explicit selection for this package role")
        return artifacts[0]
    if not isinstance(selection, (Config, TuningResult)):
        raise TypeError("a deployment selection must be a measured Config or TuningResult")
    options = selection.winner.options if isinstance(selection, TuningResult) else selection.options
    if not any(result.winner.options == options for result in kernel.selections):
        raise ValueError("the requested configuration has not won a workload in this callable")
    matching = [artifact for artifact in artifacts if artifact.options == options]
    if len(matching) != 1:
        raise ValueError("selected configuration has no unique live native artifact")
    artifact = matching[0]
    if isinstance(selection, TuningResult):
        selection.check_program(kernel.source.canonical, artifact.target)
    return artifact


def selection_records(entries):
    records = {}
    for key, kernel in entries.items():
        if isinstance(kernel, AutotunedKernel):
            records[key] = kernel.selections
    return MappingProxyType(records)
