from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from .driver import CompilationError, Toolchain, run_compiler
from .cache_geometry import DataCache, cache_observations
from .signature import freeze


_libc = ctypes.CDLL(None, use_errno=True)


@dataclass(frozen=True)
class Target:
    """An explicit build target. It grants no permission to execute locally."""

    march: str
    abi: str
    vlen_bits: int
    data_cache: DataCache | None = None

    def __post_init__(self) -> None:
        if isinstance(self.data_cache, dict):
            object.__setattr__(self, "data_cache", DataCache.from_dict(self.data_cache))
        if self.data_cache is not None and not isinstance(self.data_cache, DataCache):
            raise TypeError("data_cache must be DataCache or None for unavailable geometry")
        if not self.march.startswith(("rv32", "rv64")):
            raise ValueError("a build target requires a RISC-V march")
        if self.abi not in ("ilp32", "ilp32f", "ilp32d", "lp64", "lp64f", "lp64d"):
            raise ValueError("unsupported RISC-V ABI")
        if self.abi.startswith("lp64") != (self.xlen == 64):
            raise ValueError("march and ABI have different XLEN")
        if type(self.vlen_bits) is not int or not 128 <= self.vlen_bits <= 65536 or (
                self.vlen_bits & (self.vlen_bits - 1)):
            raise ValueError("VLEN must be a power of two from 128 through 65536")

    @property
    def xlen(self) -> int:
        return 64 if self.march.startswith("rv64") else 32

    def to_dict(self) -> dict:
        result = {"march": self.march, "abi": self.abi, "vlen_bits": self.vlen_bits}
        if self.data_cache is not None:
            result["data_cache"] = self.data_cache.to_dict()
        return result

    def arguments(self) -> list[str]:
        arguments = [f"--march={self.march}", f"--abi={self.abi}", f"--vlen-bits={self.vlen_bits}"]
        if self.data_cache is not None:
            arguments.append("--data-cache=" + self.data_cache.argument())
        return arguments

    def capabilities(self, *, options=None, toolchain: Toolchain | None = None):
        """Query compiler-supported facts for this target without native execution."""
        from .options import CompileOptions

        binding = CompileOptions() if options is None else options
        if not isinstance(binding, CompileOptions):
            raise TypeError("target capability options must be CompileOptions")
        arguments = ["--query-target", f"--march={self.march}", f"--abi={self.abi}",
                     f"--vlen-bits={self.vlen_bits}",
                     f"--matrix-extension={binding.matrix_extension}",
                     f"--max-widening-combine-groups={binding.max_widening_combine_groups}",
                     f"--partial-combine-policy={binding.partial_combine_policy}",
                     f"--record-axis-policy={binding.record_axis_policy}",
                     f"--private-stack-bytes={binding.private_stack_bytes}"]
        result = TargetCapabilities.from_dict(_query(arguments, toolchain))
        if (result.target.march, result.target.abi, result.target.vlen_bits) != (
                self.march, self.abi, self.vlen_bits):
            raise CompilationError("target query returned another requested ISA, ABI or VLEN")
        expected = () if binding.matrix_extension == "none" else (binding.matrix_extension,)
        if result.matrix_extensions != expected:
            raise CompilationError("target query disagrees with its requested extension binding")
        return result


def _query(arguments, toolchain):
    selected = Toolchain.from_environment() if toolchain is None else toolchain
    command = [selected.compiler_path(), *arguments]
    output = run_compiler(command, stage="target capability query")
    try:
        return json.loads(output)
    except ValueError as error:
        raise CompilationError("target query returned invalid JSON", command=command,
                               stage="target capability query", stdout=output) from error


def _object(value, required, description):
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise CompilationError(f"{description} has incomplete or unknown fields")


def _names(value, description):
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        raise CompilationError(f"{description} must contain nonempty names")
    if len(set(value)) != len(value):
        raise CompilationError(f"{description} contains duplicate names")
    return tuple(value)


def _records(value, description, *, identity="name"):
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, Mapping) for item in value):
        raise CompilationError(f"{description} must contain capability objects")
    names = [item.get(identity) for item in value]
    _names(names, description)
    return tuple(freeze(dict(item)) for item in value)


def _plain(value):
    if isinstance(value, Mapping):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def isa_contract(value):
    """Check the shape of compiler-owned ISA facts without deriving any facts."""
    if not isinstance(value, Mapping) or not value:
        raise CompilationError("standard ISA contract must be a nonempty extension/version map")
    result = {}
    for name, version in value.items():
        if not isinstance(name, str) or not name or not isinstance(version, (list, tuple)) or len(version) != 2:
            raise CompilationError("standard ISA requirements need names and major/minor versions")
        if any(type(component) is not int or component < 0 for component in version):
            raise CompilationError("standard ISA versions must contain nonnegative integers")
        result[name] = tuple(version)
    return freeze(result)


@dataclass(frozen=True)
class TargetCapabilities:
    """Compiler facts; querying support does not grant permission to execute."""

    target: Target
    isa_extensions: Mapping
    capabilities: Mapping
    matrix_extensions: tuple[str, ...]
    fragments: tuple[Mapping, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, ("kind", "target", "isa_extensions", "capabilities", "matrix_extensions", "fragments"),
                "target capability query")
        if value["kind"] != "weft-target-capabilities":
            raise CompilationError("unsupported target capability query kind")
        selected = value["target"]
        _object(selected, ("march", "abi", "xlen", "vlen_bits"), "queried target")
        target = Target(selected["march"], selected["abi"], selected["vlen_bits"])
        if type(selected["xlen"]) is not int or selected["xlen"] != target.xlen:
            raise CompilationError("queried target XLEN disagrees with its ISA and ABI")
        if not isinstance(value["capabilities"], Mapping) or not value["capabilities"]:
            raise CompilationError("queried target has no physical capability facts")
        return cls(target, isa_contract(value["isa_extensions"]), freeze(dict(value["capabilities"])),
                   _names(value["matrix_extensions"], "enabled matrix extensions"),
                   _records(value["fragments"], "target fragments", identity="instruction"))

    def to_dict(self):
        return {"kind": "weft-target-capabilities", "target": {**self.target.to_dict(), "xlen": self.target.xlen},
                "isa_extensions": _plain(self.isa_extensions), "capabilities": _plain(self.capabilities),
                "matrix_extensions": list(self.matrix_extensions), "fragments": _plain(self.fragments)}


@dataclass(frozen=True)
class ExtensionCatalog:
    """Registered local extensions, independently of the machine running Python."""

    extensions: tuple[Mapping, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, ("kind", "extensions"), "extension catalog")
        if value["kind"] != "weft-extension-catalog":
            raise CompilationError("unsupported extension catalog kind")
        records = _records(value["extensions"], "registered extensions")
        for item in records:
            _object(item, ("name", "engine", "description", "native_matches", "fragments"), "registered extension")
            if not isinstance(item["engine"], str) or not item["engine"] or not isinstance(item["description"], str):
                raise CompilationError("registered extension requires its engine and description")
            if not isinstance(item["native_matches"], tuple):
                raise CompilationError("registered extension requires explicit native match records")
            for match in item["native_matches"]:
                _object(match, ("xlen", "vlen_bits", "vendor_id", "architecture_id", "isa_feature"), "native extension match")
            _records(item["fragments"], "extension fragments", identity="instruction")
        return cls(records)

    @property
    def names(self):
        return tuple(item["name"] for item in self.extensions)

    def get(self, name):
        for item in self.extensions:
            if item["name"] == name:
                return item
        raise KeyError(f"unregistered extension: {name}")

    def to_dict(self):
        return {"kind": "weft-extension-catalog", "extensions": _plain(self.extensions)}


def extension_catalog(*, toolchain: Toolchain | None = None):
    return ExtensionCatalog.from_dict(_query(["--list-extensions"], toolchain))


@dataclass(frozen=True)
class NativeTarget:
    march: str
    abi: str
    xlen: int
    vlen_bits: int
    cpus: frozenset[int]
    matrix_extensions: tuple[tuple[str, frozenset[int]], ...]
    isa_extensions: tuple[tuple[str, tuple[int, int]], ...]
    data_cache: DataCache | None
    cpu_data_caches: tuple[tuple[int, DataCache | None], ...]

    @property
    def build_target(self) -> Target:
        return Target(self.march, self.abi, self.vlen_bits, self.data_cache)

    @classmethod
    def discover(cls, toolchain: Toolchain | None = None) -> NativeTarget:
        selected = Toolchain() if toolchain is None else toolchain
        facts = json.loads(run_compiler([selected.compiler_path(), "--query-native-target"], stage="target discovery"))
        if "isa_extensions" not in facts:
            raise CompilationError("native probe lacks the standard ISA contract; use the compiler installed with this runtime")
        if "data_cache" not in facts or "cpu_data_caches" not in facts:
            raise CompilationError("native probe lacks cache observations; use the compiler installed with this runtime")
        common = None if facts["data_cache"] is None else DataCache.from_dict(facts["data_cache"])
        observations = cache_observations(facts["cpu_data_caches"], facts["cpus"], common)
        target = cls(facts["march"], facts["abi"], facts["xlen"], facts["vlen_bits"],
                     frozenset(facts["cpus"]),
                     tuple((name, frozenset(cpus)) for name, cpus in sorted(facts["matrix_extensions"].items())),
                     tuple((name, tuple(version)) for name, version in sorted(facts["isa_extensions"].items())),
                     common, observations)
        target.build_target
        if target.xlen != target.build_target.xlen or not target.cpus:
            raise CompilationError("native discovery returned incomplete hardware facts")
        if any(not cpus or not cpus.issubset(target.cpus) for _, cpus in target.matrix_extensions):
            raise CompilationError("native matrix capability exceeds its discovered CPU set")
        target.check_execution()
        return target

    def require(self, target: Target, extensions: tuple[str, ...], isa_extensions) -> None:
        if (target.abi, target.xlen, target.vlen_bits) != (self.abi, self.xlen, self.vlen_bits):
            raise CompilationError("artifact ABI or VLEN differs from the native execution target")
        available = dict(self.isa_extensions)
        missing = [name for name, version in isa_extensions.items()
                   if name not in available or available[name] < tuple(version)]
        if missing:
            raise CompilationError(f"artifact requires unavailable ISA extensions: {', '.join(missing)}")
        self.check_execution(extensions)

    def check_execution(self, matrix_extensions: tuple[str, ...] = ()) -> None:
        if sys.platform != "linux" or sys.byteorder != "little" or platform.machine() not in ("riscv32", "riscv64"):
            raise CompilationError("native execution requires little-endian RISC-V Linux")
        if ctypes.sizeof(ctypes.c_void_p) * 8 != self.xlen:
            raise CompilationError("process pointer ABI disagrees with the compiled target")
        control = _libc.prctl(ctypes.c_int(70), *(ctypes.c_ulong(0) for _ in range(4)))
        if control < 0:
            raise CompilationError(f"cannot query the calling thread's vector state: {os.strerror(ctypes.get_errno())}")
        if (control & 3) == 1:
            raise CompilationError("RISC-V vector state is disabled in the calling thread")
        cpus = os.sched_getaffinity(0)
        if not cpus.issubset(self.cpus):
            raise CompilationError("current CPU affinity exceeds the discovered execution target")
        capabilities = dict(self.matrix_extensions)
        for extension in matrix_extensions:
            if extension not in capabilities or not cpus.issubset(capabilities[extension]):
                raise CompilationError(f"matrix capability {extension} is unavailable on the current CPU affinity")

    def to_dict(self) -> dict:
        return {**self.build_target.to_dict(), "xlen": self.xlen, "cpus": sorted(self.cpus),
                "matrix_extensions": {name: sorted(cpus) for name, cpus in self.matrix_extensions},
                "data_cache": None if self.data_cache is None else self.data_cache.to_dict(),
                "cpu_data_caches": [{"cpu": cpu, "cache": None if cache is None else cache.to_dict()}
                                    for cpu, cache in self.cpu_data_caches],
                "isa_extensions": dict(self.isa_extensions)}
