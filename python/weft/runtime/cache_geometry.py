"""Reported cache geometry used for physical selection, separately from ISA legality."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DataCache:
    size_bytes: int
    line_bytes: int
    ways: int
    sets: int

    def __post_init__(self):
        for name in ("size_bytes", "line_bytes", "ways", "sets"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"data cache {name} must be a positive integer")
        if self.line_bytes & (self.line_bytes - 1):
            raise ValueError("data cache line_bytes must be a power of two")
        if self.size_bytes != self.line_bytes * self.ways * self.sets:
            raise ValueError("data cache size must equal line bytes times ways times sets")
        if self.size_bytes > 2**31 - 1:
            raise ValueError("L1 data cache size exceeds the supported address-work domain")

    def to_dict(self):
        return {"size_bytes": self.size_bytes, "line_bytes": self.line_bytes,
                "ways": self.ways, "sets": self.sets}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {"size_bytes", "line_bytes", "ways", "sets"}:
            raise ValueError("data cache requires exactly size_bytes, line_bytes, ways and sets")
        return cls(**value)

    @classmethod
    def parse(cls, spelling):
        parts = spelling.split(":")
        if len(parts) != 4 or any(not part.isascii() or not part.isdecimal() for part in parts):
            raise ValueError("data cache requires SIZE_BYTES:LINE_BYTES:WAYS:SETS")
        return cls(*(int(part) for part in parts))

    def argument(self):
        return ":".join(str(value) for value in (self.size_bytes, self.line_bytes, self.ways, self.sets))


def cache_observations(values, cpus, common):
    """Check that the profile derives from one observation per affinity CPU."""
    if not isinstance(values, list):
        raise ValueError("native cache observations must be a list")
    result = []
    seen = set()
    for entry in values:
        if not isinstance(entry, dict) or set(entry) != {"cpu", "cache"}:
            raise ValueError("native cache observation requires cpu and cache")
        cpu = entry["cpu"]
        if type(cpu) is not int or cpu not in cpus or cpu in seen:
            raise ValueError("native cache observation must identify a distinct affinity CPU")
        seen.add(cpu)
        cache = None if entry["cache"] is None else DataCache.from_dict(entry["cache"])
        result.append((cpu, cache))
    if seen != set(cpus):
        raise ValueError("native cache observations do not cover the discovered CPU affinity")
    observed = {cache for _, cache in result}
    expected = next(iter(observed)) if len(observed) == 1 else None
    if common != expected:
        raise ValueError("native common cache disagrees with the per-CPU observations")
    return tuple(sorted(result))
