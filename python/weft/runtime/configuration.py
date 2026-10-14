"""Finite configuration domains, before either compiler IR is constructed."""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .cache import canonical_json, publish_json
from .options import CompileOptions, source_parameter_name


@dataclass(frozen=True, init=False)
class Config:
    """One complete author/source and target/physical binding."""

    options: CompileOptions
    name: str

    def __init__(self, *, meta=None, name="", **physical):
        if not isinstance(name, str):
            raise TypeError("a configuration name must be a string")
        object.__setattr__(self, "options", CompileOptions(meta={} if meta is None else meta, **physical))
        object.__setattr__(self, "name", name)

    @classmethod
    def from_options(cls, options, *, name=""):
        if not isinstance(options, CompileOptions):
            raise TypeError("a configuration requires CompileOptions")
        return cls(name=name, **options.to_dict())

    @property
    def source(self):
        return self.options.meta

    @property
    def physical(self):
        values = self.options.to_dict()
        del values["meta"]
        values["require_extensions"] = tuple(values["require_extensions"])
        return MappingProxyType(values)

    def changes_from(self, other):
        """Separate author bindings from physical bindings in a comparison."""
        if not isinstance(other, Config):
            raise TypeError("configuration comparison requires another Config")
        result = {}
        for kind, before, after in (("source", other.source, self.source),
                                    ("physical", other.physical, self.physical)):
            result[kind] = {name: {"before": before.get(name), "after": after.get(name)}
                            for name in sorted(before.keys() | after.keys())
                            if before.get(name) != after.get(name)}
        return result

    def to_dict(self):
        return {"name": self.name, "options": self.options.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"name", "options"}:
            raise ValueError("a configuration requires name and complete options")
        return cls(name=value["name"], **value["options"])


def _sequence(value, name):
    # The domain must already be finite. Do not exhaust an arbitrary generator
    # merely to discover that its Cartesian product exceeds a caller's budget.
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TypeError(f"{name} requires a nonempty finite sequence")
    return tuple(value)


def _domains(values, *, source):
    if not isinstance(values, Mapping):
        raise TypeError("configuration domains must be mappings")
    result = {}
    fields = CompileOptions.__dataclass_fields__.keys() - {"meta"}
    for name, choices in values.items():
        if source and not source_parameter_name(name):
            raise ValueError(f"invalid source parameter name {name!r}")
        if not source and name not in fields:
            raise ValueError(f"unknown physical parameter {name!r}")
        choices = _sequence(choices, f"domain {name}")
        normalized = []
        for value in choices:
            checked = CompileOptions(meta={name: value}) if source else CompileOptions(**{name: value})
            normalized.append(checked.meta[name] if source else getattr(checked, name))
        keys = [canonical_json(value) for value in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate values in domain {name!r}")
        result[name] = tuple(normalized)
    return MappingProxyType(result)


@dataclass(frozen=True, init=False)
class ConfigSpace:
    """An explicit finite product of source domains and physical domains.

    This is configuration input, not a program or an alternative lowering IR.
    Each expanded Config goes through the ordinary compiler independently.
    No source tree, target operation, layout or legal candidate is invented here.
    Use several spaces in expand_configurations for correlated choices instead
    of constructing a large product and silently pruning it.

    Example::

        ConfigSpace(base=CompileOptions(meta={"MC": 64, "NC": 64}),
                    source={"MR": (4, 32), "NR": (4, 64)},
                    physical={"lmul_eighths": (32, 64)}, name="projection")

    A domain overrides that field of base. Fields outside the declared domains
    remain exactly fixed. Expansion checks the full cardinality before building
    even the first Config, and respects the caller's explicit candidate budget.
    """

    base: CompileOptions
    name: str
    source: Mapping[str, tuple]
    physical: Mapping[str, tuple]

    def __init__(self, *, base=None, source=None, physical=None, name=""):
        if base is None:
            base = CompileOptions()
        if not isinstance(base, CompileOptions):
            raise TypeError("configuration space base must be CompileOptions")
        if not isinstance(name, str):
            raise TypeError("configuration space name must be a string")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", _domains({} if source is None else source, source=True))
        object.__setattr__(self, "physical", _domains({} if physical is None else physical, source=False))

    @property
    def size(self):
        size = 1
        for domain in (*self.source.values(), *self.physical.values()):
            size *= len(domain)
        return size

    def expand(self, *, max_candidates=8):
        _budget(max_candidates)
        if self.size > max_candidates:
            raise ValueError(f"configuration space has {self.size} candidates, above its {max_candidates}-candidate budget")
        source_names = tuple(self.source)
        physical_names = tuple(self.physical)
        domains = (*self.source.values(), *self.physical.values())
        result = []
        for index, values in enumerate(itertools.product(*domains)):
            selected = self.base.to_dict()
            selected["meta"].update(zip(source_names, values[:len(source_names)]))
            selected.update(zip(physical_names, values[len(source_names):]))
            name = f"{self.name}/{index}" if self.name else str(index)
            result.append(Config(name=name, **selected))
        return tuple(result)

    def specialize(self, **bindings):
        """Fix extra source parameters without replacing a declared domain."""
        conflicts = bindings.keys() & self.source.keys()
        if conflicts:
            raise ValueError(f"cannot override varying source domains: {sorted(conflicts)}")
        return ConfigSpace(base=self.base.specialize(**bindings), source=self.source,
                           physical=self.physical, name=self.name)

    def to_dict(self):
        return {"name": self.name, "base": self.base.to_dict(),
                "source": {name: list(values) for name, values in self.source.items()},
                "physical": {name: list(values) for name, values in self.physical.items()}}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"name", "base", "source", "physical"}:
            raise ValueError("configuration space requires name, base, source and physical")
        return cls(base=CompileOptions(**value["base"]), source=value["source"],
                   physical=value["physical"], name=value["name"])


def _budget(value):
    if type(value) is not int or value <= 0:
        raise ValueError("max_candidates must be a positive integer")


def expand_configurations(values, *, max_candidates=8):
    """Expand a finite list of Config/ConfigSpace, preserving declaration order.

    Reject overlapping spaces, even when their names differ. Candidate labels
    are presentation only; complete source/physical bindings identify candidates.
    The limit is checked before allocating their expanded Cartesian products.
    """
    _budget(max_candidates)
    if isinstance(values, (Config, ConfigSpace)):
        values = (values,)
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("candidates require Config or ConfigSpace objects")
    # Accept finite iterables of old-style complete configurations, but stop
    # consuming at the bound. A generator cannot bypass the allocation limit.
    entries = []
    size = 0
    for value in values:
        if not isinstance(value, (Config, ConfigSpace)):
            raise TypeError("candidates require Config or ConfigSpace objects")
        size += value.size if isinstance(value, ConfigSpace) else 1
        if size > max_candidates:
            raise ValueError(f"candidate domain exceeds its {max_candidates}-candidate budget")
        entries.append(value)
    if not entries:
        raise ValueError("candidate domain must not be empty")
    result = []
    seen = set()
    for value in entries:
        for config in value.expand(max_candidates=max_candidates) if isinstance(value, ConfigSpace) else (value,):
            key = canonical_json(config.options.to_dict())
            if key in seen:
                raise ValueError("duplicate complete configurations do not constitute a search space")
            seen.add(key)
            result.append(config)
    return tuple(result)


def save_configurations(path, values, *, max_candidates=8):
    """Save explicit complete choices; loading never executes Python code."""
    configs = expand_configurations(values, max_candidates=max_candidates)
    return publish_json(Path(path), [config.to_dict() for config in configs])


def load_configurations(path, *, max_candidates=8):
    """Read complete choices or finite spaces from a JSON array."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("configuration file must contain a JSON array")
    _budget(max_candidates)
    if len(values) > max_candidates:
        raise ValueError(f"configuration file exceeds its {max_candidates}-candidate budget")
    entries = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("configuration entry must be a JSON object")
        if "options" in value:
            entries.append(Config.from_dict(value))
        else:
            entries.append(ConfigSpace.from_dict(value))
    return expand_configurations(entries, max_candidates=max_candidates)


def resolve_configurations(source, values, *, max_candidates=8):
    """Close source domains and require a single public invocation contract."""
    configs = expand_configurations(values, max_candidates=max_candidates)
    resolved = tuple(Config.from_options(source.options(config.options), name=config.name) for config in configs)
    resolved = expand_configurations(resolved, max_candidates=max_candidates)
    interface = source.interface(resolved[0].options.meta)
    expected = interface.semantic_dict()
    for config in resolved[1:]:
        if source.interface(config.options.meta).semantic_dict() != expected:
            raise ValueError(f"configuration {config.name!r} changes the public invocation contract; use a separate kernel")
    return resolved, interface
