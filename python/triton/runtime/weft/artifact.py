"""Export/load the same compiled kernel without invoking a frontend or compiler."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .native import manifest


def export(kernel, directory):
    destination = Path(directory).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix=".triton-artifact-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        _write_export(kernel, staging)
        os.rename(staging, destination)
    return destination


def _write_export(kernel, destination):
    for extension, contents in kernel.asm.items():
        if extension == "source":
            continue
        path = destination / ("kernel." + extension)
        path.write_bytes(contents) if isinstance(contents, bytes) else path.write_text(contents)
    metadata = dict(kernel.metadata._asdict())
    metadata["target"] = vars(metadata["target"])
    metadata["runtime_files"] = {}
    for name in kernel.metadata.runtime_files:
        source = kernel.metadata_group["__runtime__/" + name]
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        metadata["runtime_files"][name] = name
    (destination / "triton.json").write_text(json.dumps(metadata, indent=2))
    single = manifest(kernel.metadata)
    single["kernel"] = kernel.metadata.single_instance_abi
    single["dispatch"] = "triton_weft_instance_dispatch"
    single["grid"] = manifest(kernel.metadata)
    single["grid"]["launcher"] = "triton_weft_grid"
    (destination / "manifest.json").write_text(json.dumps(single, indent=2))
    header = "#pragma once\n#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n"
    header += '#ifdef __cplusplus\n#define TRITON_WEFT_RESTRICT __restrict__\nextern "C" {\n'
    header += '#else\n#define TRITON_WEFT_RESTRICT restrict\n#endif\n'
    declarations = kernel.metadata.kernel_abi["declaration"] + "\n" + single["kernel"]["declaration"] + "\n"
    header += re.sub(r"\brestrict\b", "TRITON_WEFT_RESTRICT", declarations)
    from triton.backends.weft.launcher import GRID_DECLARATION
    header += GRID_DECLARATION + "\n"
    header += '#ifdef __cplusplus\n}\n#endif\n#undef TRITON_WEFT_RESTRICT\n'
    (destination / "kernel.h").write_text(header)


def load(directory):
    from triton.compiler import CompiledKernel
    root = Path(directory).absolute()
    metadata_path = root / "triton.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["target"]["backend"] != "weft":
        raise ValueError("not a Triton CPU native artifact")
    group = {"triton.json": str(metadata_path)}
    for path in root.glob("kernel.*"):
        if path.suffix not in (".h", ".json"):
            group[path.name] = str(path)
    if "kernel.so" not in group:
        raise FileNotFoundError("native artifact has no kernel.so")
    for name in metadata["runtime_files"]:
        path = (root / name).resolve()
        if path.is_relative_to(root.resolve()) and path.is_file():
            group["__runtime__/" + name] = str(path)
        else:
            raise FileNotFoundError(f"artifact runtime dependency is absent: {name}")
    return CompiledKernel(None, group, metadata["hash"])
