"""Compiler-verified physical panel transfers before loop materialization."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .contraction_inspection import _integer, _object, _shape
from .driver import CompilationError


def _power(value, maximum, label):
    result = _integer(value, label, 1)
    if result > maximum or result & (result - 1):
        raise CompilationError(f"{label} must be a power of two <= {maximum}")
    return result


@dataclass(frozen=True)
class PanelTransferTile:
    axis: int
    fields: int
    lanes: int
    lmul_eighths: int
    register_groups: int
    loads: int
    stores: int
    work: int

    @property
    def elements(self):
        return self.fields * self.lanes

    @property
    def normalized_work(self):
        """Exact issued-operation and register-width ratio; never latency."""
        return Fraction(self.work, self.elements)

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "panel transfer tile")
        numbers = {key: _integer(item, key, 1) for key, item in value.items()}
        _power(numbers["fields"], 8, "fields")
        _power(numbers["lanes"], 65536, "lanes")
        _power(numbers["lmul_eighths"], 64, "LMUL eighths")
        groups = numbers["fields"] * ((numbers["lmul_eighths"] + 7) // 8)
        if numbers["lanes"] <= 1 or numbers["fields"] * numbers["lmul_eighths"] > 64:
            raise CompilationError("panel transfer has an illegal vector tuple")
        if numbers["register_groups"] != groups or groups > 8:
            raise CompilationError("panel transfer register witness disagrees with its tuple")
        if numbers["loads"] != numbers["fields"] or numbers["stores"] != 1:
            raise CompilationError("panel transfer must account for each source and destination issue")
        if numbers["work"] != numbers["loads"] + numbers["stores"] + 2 * groups:
            raise CompilationError("panel transfer work witness disagrees with its selected memory operations")
        return cls(**numbers)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PanelTransferCase:
    main: PanelTransferTile
    tail: PanelTransferTile

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "panel transfer case")
        main, tail = (PanelTransferTile.from_dict(value[name]) for name in ("main", "tail"))
        if main.axis != tail.axis or tail.fields != 1 or (main.fields == 1 and main != tail):
            raise CompilationError("panel transfer main and remainder must cover one exact orientation")
        return cls(main, tail)

    def to_dict(self):
        return {"main": self.main.to_dict(), "tail": self.tail.to_dict()}


@dataclass(frozen=True)
class PanelTransferSelection:
    axes: tuple[int, ...]
    element: str
    strategy: str
    source_shape: tuple[int, ...]
    destination_axes: tuple[int, ...]
    destination_shape: tuple[int, ...]
    maximum_lmul_eighths: int
    maximum_fields: int
    unit_axis: int
    nonunit_axis: int
    cases: tuple[PanelTransferCase, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "panel transfer selection")
        axes = _shape(value["axes"], "source axes")
        destination_axes = _shape(value["destination_axes"], "destination axes")
        source = _shape(value["source_shape"], "source shape", symbolic=True)
        destination = _shape(value["destination_shape"], "destination shape", symbolic=True)
        if len(axes) != 2 or len(set(axes)) != 2 or sorted(axes) != sorted(destination_axes):
            raise CompilationError("panel transfer must preserve two distinct logical axes")
        if len(source) != 2 or len(destination) != 2:
            raise CompilationError("panel transfer descriptors require rank-two shapes")
        if value["element"] not in ("f16", "bf16", "f32", "f64"):
            raise CompilationError("panel transfer requires supported floating storage")
        if value["strategy"] not in ("issue-work", "max-fields", "single-field"):
            raise CompilationError("panel transfer requires a known finite selection strategy")
        maximum_lmul = _power(value["maximum_lmul_eighths"], 64, "maximum LMUL")
        maximum_fields = _power(value["maximum_fields"], 8, "maximum fields")
        unit = _integer(value["unit_axis"], "unit axis")
        nonunit = _integer(value["nonunit_axis"], "nonunit axis")
        if (unit and unit not in axes) or (nonunit and (nonunit not in axes or nonunit == unit or not unit)):
            raise CompilationError("panel transfer predicate must use distinct descriptor axes")
        if not isinstance(value["cases"], list) or len(value["cases"]) != (2 if unit else 1):
            raise CompilationError("panel transfer does not cover its complete stride predicate")
        cases = tuple(PanelTransferCase.from_dict(item) for item in value["cases"])
        if unit and (cases[0].main.axis != unit or cases[1].main.axis == unit):
            raise CompilationError("panel transfer branch orientation disagrees with its predicate")
        for case in cases:
            for tile in (case.main, case.tail):
                if tile.axis not in axes or tile.fields > maximum_fields or tile.lmul_eighths > maximum_lmul:
                    raise CompilationError("panel transfer choice exceeds its saved finite domain")
        if value["strategy"] == "single-field" and any(case.main.fields != 1 for case in cases):
            raise CompilationError("panel transfer differs from its explicit single-field binding")
        return cls(axes, value["element"], value["strategy"], source, destination_axes, destination,
                   maximum_lmul, maximum_fields, unit, nonunit, cases)

    def to_dict(self):
        return {"axes": list(self.axes), "element": self.element, "strategy": self.strategy,
                "source_shape": list(self.source_shape),
                "destination_axes": list(self.destination_axes), "destination_shape": list(self.destination_shape),
                "maximum_lmul_eighths": self.maximum_lmul_eighths, "maximum_fields": self.maximum_fields,
                "unit_axis": self.unit_axis, "nonunit_axis": self.nonunit_axis,
                "cases": [case.to_dict() for case in self.cases]}


@dataclass(frozen=True)
class KernelPanelTransfers:
    name: str
    vlen_bits: int
    transfers: tuple[PanelTransferSelection, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, cls.__dataclass_fields__, "kernel panel transfers")
        if not isinstance(value["name"], str) or not value["name"] or not isinstance(value["transfers"], list):
            raise CompilationError("panel transfers require a named physical kernel and selection list")
        vlen = _power(value["vlen_bits"], 65536, "VLEN bits")
        if vlen < 128:
            raise CompilationError("panel transfer target requires full V")
        transfers = tuple(PanelTransferSelection.from_dict(item) for item in value["transfers"])
        for selection in transfers:
            bits = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}[selection.element]
            for case in selection.cases:
                for tile in (case.main, case.tail):
                    if tile.lanes * bits * 8 != vlen * tile.lmul_eighths:
                        raise CompilationError("panel transfer lane count disagrees with its target VLEN")
        return cls(value["name"], vlen, transfers)

    def to_dict(self):
        return {"name": self.name, "vlen_bits": self.vlen_bits,
                "transfers": [item.to_dict() for item in self.transfers]}


@dataclass(frozen=True)
class PanelTransferSelections:
    kernels: tuple[KernelPanelTransfers, ...]

    @classmethod
    def from_dict(cls, value):
        _object(value, ("kind", "scope", "kernels"), "panel transfer inspection")
        if value["kind"] != "weft-panel-transfer-selections" or value["scope"] != "static-physical-choices":
            raise CompilationError("unsupported panel transfer inspection scope")
        if not isinstance(value["kernels"], list) or not value["kernels"]:
            raise CompilationError("panel transfer inspection has no physical kernel")
        return cls(tuple(KernelPanelTransfers.from_dict(item) for item in value["kernels"]))

    def to_dict(self):
        return {"kind": "weft-panel-transfer-selections", "scope": "static-physical-choices",
                "kernels": [kernel.to_dict() for kernel in self.kernels]}

    def render(self):
        lines = ["Selected physical transfers; work counts issues and register widths, not measured latency."]
        for kernel in self.kernels:
            lines.append(f"{kernel.name}: VLEN={kernel.vlen_bits}, {len(kernel.transfers)} panel copies")
            for index, transfer in enumerate(kernel.transfers):
                lines.append(f"  [{index}] {transfer.element} axes={transfer.axes} -> {transfer.destination_axes}, "
                             f"strategy={transfer.strategy}")
                for branch, case in enumerate(transfer.cases):
                    condition = "always"
                    if transfer.unit_axis:
                        condition = f"stride(axis {transfer.unit_axis}) == 1"
                        if transfer.nonunit_axis:
                            condition += f" and stride(axis {transfer.nonunit_axis}) != 1"
                        if branch:
                            condition = f"otherwise ({condition} is false)"
                    tile = case.main
                    lines.append(f"    {condition}: copy axis={tile.axis}, fields={tile.fields}, lanes={tile.lanes}, "
                                 f"LMUL={tile.lmul_eighths}/8, groups={tile.register_groups}")
                    lines.append(f"      issues load/store={tile.loads}/{tile.stores}, work/element={tile.normalized_work}; "
                                 f"remainder fields=1, lanes={case.tail.lanes}")
        return "\n".join(lines) + "\n"
