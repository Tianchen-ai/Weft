from __future__ import annotations

import json
from pathlib import Path

from weft.runtime import CompileOptions, Config


PROJECT = Path(__file__).resolve().parents[2]


def find_configuration(runner: str, request: str) -> dict:
    key = runner + ":" + request
    matches = []
    for path in sorted((PROJECT / "examples/kernels").glob("*/tuning.json")):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        if key not in catalog["bindings"]:
            continue
        binding = catalog["bindings"][key]
        entry = path.parent / binding["entry"]
        if entry.parent != path.parent or entry.suffix != ".py" or not entry.is_file():
            raise ValueError(f"invalid declared kernel entry: {entry}")
        if not binding["symbol"].isascii() or not binding["symbol"].isidentifier():
            raise ValueError(f"invalid declared kernel symbol: {binding['symbol']}")
        physical = dict(catalog["physical_defaults"], **binding["physical"])
        # Historical fixed bindings remain useful defaults. Search is a public
        # sequence of Config objects; catalog domains no longer select programs.
        CompileOptions(meta=binding["source"], **physical)
        matches.append({"runner": runner, "request": request,
                        "entry": str(entry.relative_to(PROJECT)), "symbol": binding["symbol"],
                        "source": binding["source"], "physical": physical,
                        "binding_origin": binding.get("binding_origin", catalog["binding_origin"]),
                        "catalog": str(path.relative_to(PROJECT))})
    if len(matches) != 1:
        raise ValueError(f"expected one declared kernel configuration for {key}, got {len(matches)}")
    return matches[0]


def configured_binding(runner: str, request: str, target: dict, config: Config | None = None) -> dict:
    result = find_configuration(runner, request)
    result["target"] = target
    if config is None:
        options = CompileOptions(meta=result["source"], **result["physical"],
                                 matrix_extension=target["matrix_extension"])
    else:
        options = config.options
        if options.matrix_extension != target["matrix_extension"]:
            raise ValueError("candidate matrix capability differs from the workload's explicit target contract")
        result["binding_origin"] = "explicit-complete-config"
    values = options.to_dict()
    result["source"] = values.pop("meta")
    result["physical"] = values
    return result
