from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .artifact import Artifact
from .cache import publish_json
from .driver import CompilationError, installed_tool


def sdk_prefix() -> Path:
    compiler = installed_tool("weft-compile")
    if compiler is None:
        raise CompilationError("install a Weft compiler wheel to obtain the C/C++ SDK")
    prefix = compiler.parent.parent
    for relative in ("include/weft/runtime.h", "include/weft/runtime.hpp", "lib/libweft_runtime.so",
                     "lib/cmake/Weft/WeftConfig.cmake"):
        if not (prefix / relative).is_file():
            raise CompilationError(f"installed SDK is incomplete: {prefix / relative}")
    return prefix


class Package:
    """A relocatable set of named, already selected artifacts."""

    def __init__(self, directory):
        self.path = Path(directory).expanduser().absolute()
        manifest = json.loads((self.path / "package.json").read_text(encoding="utf-8"))
        if manifest["kind"] != "weft-kernel-package":
            raise CompilationError("unsupported package kind")
        self.kernels = {}
        for entry in manifest["kernels"]:
            name = entry["name"]
            relative = Path(entry["directory"])
            if name in self.kernels or relative.is_absolute() or ".." in relative.parts:
                raise CompilationError("package contains duplicate names or a non-relative artifact")
            self.kernels[name] = Artifact(self.path / relative)
        if not self.kernels:
            raise CompilationError("package contains no kernels")
        self.bindings = manifest["bindings"]

    @classmethod
    def create(cls, directory, kernels: Mapping[str, Artifact], *, bindings=None):
        if not kernels or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) for name in kernels):
            raise ValueError("package requires distinct named artifacts using letters, digits, underscore or hyphen")
        if any(not isinstance(artifact, Artifact) for artifact in kernels.values()):
            raise TypeError("package entries must be Artifact objects")
        target_abis = {(artifact.target.abi, artifact.target.vlen_bits) for artifact in kernels.values()}
        if len(target_abis) != 1:
            raise ValueError("one package must use a common execution ABI and VLEN")
        registrations = {}
        for group, mapping in ({} if bindings is None else bindings).items():
            if not isinstance(group, str) or not group or not isinstance(mapping, Mapping):
                raise ValueError("binding groups require named role-to-kernel mappings")
            registrations[group] = dict(mapping)
            for role, name in mapping.items():
                if not isinstance(role, str) or not role or name not in kernels:
                    raise ValueError("binding role must name an existing package entry")
                if group == "ggml.vec_dot":
                    from weft.integrations.llama_cpp import validate_vec_dot

                    validate_vec_dot(kernels[name], role)
        destination = Path(directory).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=".weft-package-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            entries = []
            for name, artifact in kernels.items():
                relative = "kernels/" + name
                artifact.save(staging / relative)
                entries.append({"name": name, "directory": relative})
            publish_json(staging / "package.json", {"kind": "weft-kernel-package", "kernels": entries,
                         "bindings": registrations})
            os.rename(staging, destination)
            descriptor = os.open(destination.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return cls(destination)

    def link(self, directory, *, toolchain=None, cache=None):
        """System-compile source-only entries once, before deployment or model load."""
        return self.create(directory, {name: artifact.link(toolchain=toolchain, cache=cache)
                                       for name, artifact in self.kernels.items()}, bindings=self.bindings)


__all__ = ["Package", "sdk_prefix"]
