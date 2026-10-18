"""Named native artifacts for C/C++ consumers, including explicit GGML roles."""
import json
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping


def sdk_prefix():
    prefix = Path(__file__).parents[1] / "_C/weft"
    for name in ("include/weft/runtime.h", "include/weft/runtime.hpp",
                 "lib/libweft_runtime.so", "lib/cmake/Weft/WeftConfig.cmake"):
        if not (prefix / name).is_file():
            raise FileNotFoundError(f"the installed Triton native SDK is incomplete: {prefix / name}")
    return prefix


class Package:
    """A relocatable collection of already compiled, explicitly selected kernels."""

    def __init__(self, directory):
        from .weft.artifact import load
        self.path = Path(directory).expanduser().absolute()
        description = json.loads((self.path / "package.json").read_text())
        if description["kind"] != "weft-kernel-package" or description["producer"] != "triton":
            raise ValueError("expected a Triton native kernel package")
        self.kernels = {}
        for entry in description["kernels"]:
            name, relative = entry["name"], Path(entry["directory"])
            if name in self.kernels or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("duplicate package name or non-relative artifact path")
            self.kernels[name] = load(self.path / relative)
        if not self.kernels:
            raise ValueError("package contains no native kernels")
        self.bindings = description["bindings"]
        self._validate(self.kernels, self.bindings)

    @staticmethod
    def _validate(kernels, bindings):
        from triton.compiler import CompiledKernel
        from triton.tools.llama_cpp import validate_vec_dot, validate_quantizer, validate_ime1
        if not kernels or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) for name in kernels):
            raise ValueError("package requires named kernels using letters, digits, underscore or hyphen")
        for kernel in kernels.values():
            if not isinstance(kernel, CompiledKernel) or kernel.metadata.target.backend != "weft":
                raise TypeError("native package entries must be CPU CompiledKernel objects")
        if len({(k.metadata.target.abi, k.metadata.target.vlen_bits) for k in kernels.values()}) != 1:
            raise ValueError("one native package must use a common execution ABI and VLEN")
        validators = {"ggml.vec_dot": validate_vec_dot, "ggml.from_float": validate_quantizer,
                      "ggml.ime1": validate_ime1}
        for group, mapping in bindings.items():
            if not isinstance(group, str) or not group or not isinstance(mapping, Mapping):
                raise ValueError("bindings require named role-to-kernel mappings")
            for role, name in mapping.items():
                if not isinstance(role, str) or not role or name not in kernels:
                    raise ValueError("binding role must name an existing package entry")
                if group in validators:
                    validators[group](kernels[name], role)

    @classmethod
    def create(cls, directory, kernels, *, bindings=None):
        registrations = {} if bindings is None else {group: dict(mapping) for group, mapping in bindings.items()}
        cls._validate(kernels, registrations)
        destination = Path(directory).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=".triton-package-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            entries = []
            for name, kernel in kernels.items():
                relative = "kernels/" + name
                kernel.export(staging / relative)
                entries.append({"name": name, "directory": relative})
            (staging / "package.json").write_text(json.dumps({"kind": "weft-kernel-package",
                "producer": "triton", "kernels": entries, "bindings": registrations}, indent=2))
            os.rename(staging, destination)
        return cls(destination)
