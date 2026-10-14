"""Typed inspection of compiler-verified, static contraction selections."""
from __future__ import annotations

from dataclasses import dataclass

from .driver import CompilationError


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise CompilationError(f"{label} has missing or unknown fields")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise CompilationError(f"{label} must be an integer >= {minimum}")
    return value


def _shape(value, label, *, symbolic=False):
    if not isinstance(value, list) or any(type(item) is not int or (not symbolic and item <= 0) for item in value):
        raise CompilationError(f"{label} must contain {'integer dimensions' if symbolic else 'positive dimensions'}")
    return tuple(value)


@dataclass(frozen=True)
class ContractionTile:
    shape: tuple[int, ...]
    lane_axis: int
    lane_input: int
    lanes: int
    lmul_eighths: int
    scalar_steps: int
    register_groups: int
    loads: int
    conversions: int

    @classmethod
    def from_dict(cls, value):
        fields = tuple(cls.__dataclass_fields__)
        _object(value, fields, "contraction tile")
        numbers = {name: _integer(value[name], name) for name in fields if name != "shape"}
        if numbers["lane_input"] not in (0, 1) or any(numbers[name] <= 0 for name in
                ("lane_axis", "lanes", "lmul_eighths", "scalar_steps", "register_groups")):
            raise CompilationError("contraction tile has an invalid selected representation")
        return cls(_shape(value["shape"], "tile shape"), **numbers)

    def to_dict(self):
        return {name: list(self.shape) if name == "shape" else getattr(self, name)
                for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ContractionEpilogue:
    mode: str
    shape: tuple[int, ...]
    lane_axis: int
    lanes: int
    lmul_eighths: int
    register_groups: int

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "contraction epilogue")
        if value["mode"] not in ("scalar", "direct", "vector"):
            raise CompilationError("contraction epilogue has an unknown representation")
        numbers = {name: _integer(value[name], name) for name in
                   ("lane_axis", "lanes", "lmul_eighths", "register_groups")}
        return cls(value["mode"], _shape(value["shape"], "epilogue shape"), **numbers)

    def to_dict(self):
        return {name: list(self.shape) if name == "shape" else getattr(self, name)
                for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ContractionCase:
    singleton_axis: int
    tile: ContractionTile
    panel_shape: tuple[int, ...]
    panel_k: int
    update_tile: ContractionTile
    output_tile: ContractionTile
    direct_epilogue: ContractionEpilogue
    epilogue: ContractionEpilogue

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "contraction case")
        return cls(_integer(value["singleton_axis"], "singleton axis"), ContractionTile.from_dict(value["tile"]),
                   _shape(value["panel_shape"], "panel shape"), _integer(value["panel_k"], "panel K"),
                   ContractionTile.from_dict(value["update_tile"]), ContractionTile.from_dict(value["output_tile"]),
                   ContractionEpilogue.from_dict(value["direct_epilogue"]),
                   ContractionEpilogue.from_dict(value["epilogue"]))

    def to_dict(self):
        return {"singleton_axis": self.singleton_axis, "tile": self.tile.to_dict(),
                "panel_shape": list(self.panel_shape), "panel_k": self.panel_k,
                "update_tile": self.update_tile.to_dict(), "output_tile": self.output_tile.to_dict(),
                "direct_epilogue": self.direct_epilogue.to_dict(),
                "epilogue": self.epilogue.to_dict()}


@dataclass(frozen=True)
class ContractionSelection:
    output_axes: tuple[int, ...]
    output_shape: tuple[int, ...]
    reduction_axes: tuple[int, ...]
    element: str
    streamed_store: bool
    unroll: int
    pipeline_depth: int
    cases: tuple[ContractionCase, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "contraction selection")
        if not isinstance(value["element"], str) or type(value["streamed_store"]) is not bool:
            raise CompilationError("contraction selection requires an element type and store frontier")
        if not isinstance(value["cases"], list) or not value["cases"]:
            raise CompilationError("contraction selection requires its ordered extent cases")
        return cls(_shape(value["output_axes"], "output axes"),
                   _shape(value["output_shape"], "output shape", symbolic=True),
                   _shape(value["reduction_axes"], "reduction axes"), value["element"], value["streamed_store"],
                   _integer(value["unroll"], "unroll", 1), _integer(value["pipeline_depth"], "pipeline depth", 1),
                   tuple(ContractionCase.from_dict(item) for item in value["cases"]))

    def to_dict(self):
        return {"output_axes": list(self.output_axes), "output_shape": list(self.output_shape),
                "reduction_axes": list(self.reduction_axes), "element": self.element,
                "streamed_store": self.streamed_store, "unroll": self.unroll, "pipeline_depth": self.pipeline_depth,
                "cases": [case.to_dict() for case in self.cases]}


@dataclass(frozen=True)
class KernelContractions:
    name: str
    vlen_bits: int
    contractions: tuple[ContractionSelection, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "kernel contraction selections")
        if not isinstance(value["name"], str) or not value["name"] or not isinstance(value["contractions"], list):
            raise CompilationError("kernel contraction selections require a name and operation list")
        return cls(value["name"], _integer(value["vlen_bits"], "VLEN", 128),
                   tuple(ContractionSelection.from_dict(item) for item in value["contractions"]))

    def to_dict(self):
        return {"name": self.name, "vlen_bits": self.vlen_bits,
                "contractions": [item.to_dict() for item in self.contractions]}


@dataclass(frozen=True)
class ContractionSelections:
    """Static selected issue counts and residency, not timing or machine spills."""

    kernels: tuple[KernelContractions, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, ("kind", "scope", "kernels"), "contraction inspection")
        if value["kind"] != "weft-contraction-selections" or value["scope"] != "static-physical-choices":
            raise CompilationError("unsupported contraction inspection scope")
        if not isinstance(value["kernels"], list) or not value["kernels"]:
            raise CompilationError("contraction inspection has no physical kernel")
        return cls(tuple(KernelContractions.from_dict(item) for item in value["kernels"]))

    def to_dict(self):
        return {"kind": "weft-contraction-selections", "scope": "static-physical-choices",
                "kernels": [item.to_dict() for item in self.kernels]}

    def render(self):
        lines = ["Static physical choices; load counts are per issue and register groups are compiler estimates."]
        for kernel in self.kernels:
            lines.append(f"{kernel.name}: VLEN={kernel.vlen_bits}, {len(kernel.contractions)} closed contractions")
            for index, contraction in enumerate(kernel.contractions):
                lines.append(f"  [{index}] {contraction.element} output={contraction.output_shape}, "
                             f"reduce axes={contraction.reduction_axes}, pipeline={contraction.pipeline_depth}")
                for case in contraction.cases:
                    condition = f"axis {case.singleton_axis} <= 1" if case.singleton_axis else "general"
                    tile = case.tile
                    lines.append(f"    {condition}: tile={tile.shape}, lanes={tile.lanes}, "
                                 f"LMUL={tile.lmul_eighths}/8, groups={tile.register_groups}, "
                                 f"loads={tile.loads}, conversions={tile.conversions}")
                    if case.panel_shape:
                        lines.append(f"      panel={case.panel_shape}, K={case.panel_k}, "
                                     f"update={case.update_tile.shape}, output={case.output_tile.shape}")
                        lines.append(f"      direct epilogue={case.direct_epilogue.mode}, "
                                     f"issue={case.direct_epilogue.shape}, axis={case.direct_epilogue.lane_axis}")
                    lines.append(f"      epilogue={case.epilogue.mode}, issue={case.epilogue.shape}")
        return "\n".join(lines) + "\n"
