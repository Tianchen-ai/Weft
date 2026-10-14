from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from .definitions import KernelDefinition


def load_definition(specification: str) -> KernelDefinition:
    """Resolve module:kernel or file.py:kernel without changing import paths."""
    location, separator, name = specification.rpartition(":")
    if not separator or not name.isidentifier():
        raise ValueError("a kernel entry must be module:kernel or file.py:kernel")
    if location.endswith(".py"):
        path = Path(location).absolute()
        module_name = "_weft_source_" + path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import kernel source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    else:
        module = importlib.import_module(location)
    definition = getattr(module, name)
    if not isinstance(definition, KernelDefinition):
        raise TypeError(f"{specification} is not an @weft.kernel definition")
    return definition
