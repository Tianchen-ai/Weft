"""Read-only views of final Physical IR choices exported by the compiler.

Occurrences count static selected operations across all branches and loop
bodies. They are useful for inspecting lowering, independently of native timing.
Layouts describe Value and Fragment carriers; storage descriptors remain in IR.
"""
from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields

from .driver import CompilationError


def _fields(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise CompilationError(f"{name} has incomplete or unknown diagnostic fields")


def _string(value, name):
    if not isinstance(value, str) or not value:
        raise CompilationError(f"{name} must be a nonempty diagnostic string")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise CompilationError(f"{name} must be an integer >= {minimum}")
    return value


def _integers(value, name):
    if not isinstance(value, (list, tuple)) or any(type(item) is not int for item in value):
        raise CompilationError(f"{name} must be an integer sequence")
    return tuple(value)


def _strings(value, name):
    if not isinstance(value, (list, tuple)):
        raise CompilationError(f"{name} must be a string sequence")
    return tuple(_string(item, name) for item in value)


@dataclass(frozen=True)
class PhysicalLayout:
    role: str
    index: int
    element_type: str
    carrier: str
    validity: str
    shape: tuple[int, ...]
    axis_ids: tuple[int, ...]
    layout_axis_ids: tuple[int, ...]
    time_factors: tuple[int, ...]
    lane_factors: tuple[int, ...]
    replica_factors: tuple[int, ...]
    fragment_factors: tuple[int, ...]
    local_factors: tuple[int, ...]
    sew: int
    lmul_eighths: int
    vl: int
    register_groups: int
    replica_group: int

    @classmethod
    def from_dict(cls, value):
        _fields(value, (item.name for item in fields(cls)), "physical layout")
        strings = ("role", "element_type", "carrier", "validity")
        numbers = ("index", "sew", "lmul_eighths", "vl", "register_groups", "replica_group")
        vectors = ("shape", "axis_ids", "layout_axis_ids", "time_factors", "lane_factors",
                   "replica_factors", "fragment_factors", "local_factors")
        data = {name: _string(value[name], name) for name in strings}
        data.update({name: _integer(value[name], name) for name in numbers})
        data.update({name: _integers(value[name], name) for name in vectors})
        result = cls(**data)
        if result.role not in ("operand", "result"):
            raise CompilationError("layout role must identify an operand or result")
        if len(result.shape) != len(result.axis_ids):
            raise CompilationError("layout logical shape and axes have different ranks")
        rank = len(result.layout_axis_ids)
        if any(len(getattr(result, name)) != rank for name in vectors[3:]):
            raise CompilationError("layout factors and physical axes have different ranks")
        if result.replica_group < 1:
            raise CompilationError("layout replica group must be positive")
        return result

    def to_dict(self):
        return {item.name: list(value) if isinstance(value := getattr(self, item.name), tuple) else value
                for item in fields(self)}

    @property
    def lmul(self):
        if self.lmul_eighths == 0:
            return "none"
        if self.lmul_eighths in (1, 2, 4):
            return f"mf{8 // self.lmul_eighths}"
        if self.lmul_eighths % 8 == 0:
            return f"m{self.lmul_eighths // 8}"
        return f"{self.lmul_eighths}/8"

    def describe(self):
        # Negative dimensions are compiler shape identities, not negative sizes.
        shape = "x".join(str(size) if size >= 0 else f"s{-size}" for size in self.shape) or "scalar"
        return (f"{self.role}[{self.index}] {self.element_type}[{shape}] {self.carrier} "
                f"SEW={self.sew} LMUL={self.lmul} VL={self.vl} "
                f"lanes={list(self.lane_factors)} replicas={list(self.replica_factors)} "
                f"groups={self.register_groups} {self.validity}")


@dataclass(frozen=True)
class PhysicalAccess:
    """The selected access attribute, including dense and encoded storage geometry."""
    form: str
    mapping: str
    alignment: int
    index_sew: int
    segment_fields: int
    group_size: int
    layer_size: int
    join_fields: int
    join_low_bits: int
    join_role: int
    bit_offset: int
    storage_bits: int
    order: str
    field_chunk_bytes: int

    @classmethod
    def from_dict(cls, value):
        _fields(value, (item.name for item in fields(cls)), "physical access")
        data = {}
        for item in fields(cls):
            name = item.name
            if name in ("form", "mapping", "order"):
                data[name] = _string(value[name], name)
            else:
                minimum = 1 if name in ("alignment", "field_chunk_bytes") else 0
                data[name] = _integer(value[name], name, minimum)
        return cls(**data)

    def to_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def describe(self):
        result = (f"{self.form}/{self.mapping} alignment={self.alignment} "
                  f"index_SEW={self.index_sew} fields={self.segment_fields}")
        if self.mapping in ("natural", "grouped_layered", "joined"):
            result += (f" group={self.group_size} layer={self.layer_size} "
                       f"offset_bits={self.bit_offset} storage_bits={self.storage_bits} "
                       f"order={self.order} chunk_bytes={self.field_chunk_bytes}")
        if self.mapping == "joined":
            result += (f" join_fields={self.join_fields} join_low_bits={self.join_low_bits} "
                       f"join_role={self.join_role}")
        return result


@dataclass(frozen=True)
class ScalarPacket:
    axis: int
    steps: int
    parts: int
    registers: int
    live_registers: int

    @classmethod
    def from_dict(cls, value):
        _fields(value, (item.name for item in fields(cls)), "scalar packet")
        numbers = {name: _integer(item, name, 1) for name, item in value.items()}
        if numbers["steps"] not in (2, 4) or not 4 <= numbers["parts"] <= 16:
            raise CompilationError("scalar packet has an unsupported coordinate domain")
        if numbers["parts"] % numbers["steps"] or numbers["registers"] != numbers["parts"]:
            raise CompilationError("scalar packet register count disagrees with its materialized parts")
        if not numbers["registers"] <= numbers["live_registers"] <= 32:
            raise CompilationError("scalar packet exceeds its complete floating register frontier")
        return cls(**numbers)

    def to_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class PhysicalOperation:
    operation: str
    engine: str
    family: str
    instruction: str
    mask: str
    tail: str
    access: PhysicalAccess | None
    scalar_packet: ScalarPacket | None
    occurrences: int
    operand_count: int
    result_count: int
    operand_groups: int
    result_groups: int
    temporary_groups: int
    fragment_groups: int
    local_bytes: int
    parameters: tuple[int, ...]
    locations: tuple[str, ...]
    layouts: tuple[PhysicalLayout, ...]

    @classmethod
    def from_dict(cls, value):
        _fields(value, (item.name for item in fields(cls)), "physical operation")
        text = ("operation", "engine", "family", "instruction", "mask", "tail")
        numbers = ("occurrences", "operand_count", "result_count", "operand_groups",
                   "result_groups", "temporary_groups", "fragment_groups", "local_bytes")
        data = {name: _string(value[name], name) for name in text}
        data.update({name: _integer(value[name], name, 1 if name == "occurrences" else 0) for name in numbers})
        data["parameters"] = _integers(value["parameters"], "leaf parameters")
        data["locations"] = _strings(value["locations"], "source locations")
        data["access"] = None if value["access"] is None else PhysicalAccess.from_dict(value["access"])
        data["scalar_packet"] = None if value["scalar_packet"] is None else ScalarPacket.from_dict(value["scalar_packet"])
        if not isinstance(value["layouts"], (list, tuple)):
            raise CompilationError("operation layouts must be a sequence")
        data["layouts"] = tuple(PhysicalLayout.from_dict(item) for item in value["layouts"])
        result = cls(**data)
        if result.scalar_packet is not None and result.operation != "weft_riscv.scalar_tile_load":
            raise CompilationError("scalar packet scheduling belongs to an explicit scalar tile load")
        identities = set()
        for layout in result.layouts:
            count = result.operand_count if layout.role == "operand" else result.result_count
            if layout.index >= count or (layout.role, layout.index) in identities:
                raise CompilationError("layout does not identify a distinct operation operand or result")
            identities.add((layout.role, layout.index))
        return result

    def to_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["parameters"] = list(self.parameters)
        result["locations"] = list(self.locations)
        result["layouts"] = [layout.to_dict() for layout in self.layouts]
        result["access"] = None if self.access is None else self.access.to_dict()
        result["scalar_packet"] = None if self.scalar_packet is None else self.scalar_packet.to_dict()
        return result

    @property
    def memory_form(self):
        return "none" if self.access is None else self.access.form

    @property
    def segment_fields(self):
        return 0 if self.access is None else self.access.segment_fields

    def value_layouts(self, role):
        if role not in ("operand", "result"):
            raise ValueError("layout role must be 'operand' or 'result'")
        return tuple(layout for layout in self.layouts if layout.role == role)


@dataclass(frozen=True)
class PhysicalSummary:
    scope: str
    selected_operation_count: int
    operations: tuple[PhysicalOperation, ...]

    @classmethod
    def from_dict(cls, value):
        _fields(value, (item.name for item in fields(cls)), "physical operation summary")
        if value["scope"] != "final-physical-ir/static-selected-operations":
            raise CompilationError("unsupported physical operation summary scope")
        count = _integer(value["selected_operation_count"], "selected operation count")
        if not isinstance(value["operations"], (list, tuple)):
            raise CompilationError("physical operation groups must be a sequence")
        operations = tuple(PhysicalOperation.from_dict(item) for item in value["operations"])
        if sum(item.occurrences for item in operations) != count:
            raise CompilationError("static operation count disagrees with its operation groups")
        return cls(value["scope"], count, operations)

    def to_dict(self):
        return {"scope": self.scope, "selected_operation_count": self.selected_operation_count,
                "operations": [operation.to_dict() for operation in self.operations]}

    def matching(self, *, family=None, engine=None, instruction=None, memory_form=None):
        """Select actual emitted groups by their exact leaf or access properties."""
        filters = {"family": family, "engine": engine, "instruction": instruction, "memory_form": memory_form}
        return tuple(operation for operation in self.operations
                     if all(value is None or getattr(operation, key) == value for key, value in filters.items()))

    def select(self, **filters):
        """Return a view containing only matching static operation groups."""
        selected = self.matching(**filters)
        return PhysicalSummary(self.scope, sum(item.occurrences for item in selected), selected)

    def render(self):
        lines = [f"Final Physical IR: {self.selected_operation_count} static selected operations "
                 f"in {len(self.operations)} groups across all control-flow paths."]
        for operation in self.operations:
            lines.append(f"{operation.occurrences:5d}  {operation.instruction}  [{operation.engine}/{operation.family}] "
                         f"memory={operation.memory_form} fields={operation.segment_fields} "
                         f"mask={operation.mask} tail={operation.tail}")
            lines.append(f"       groups in/out/temporary/fragment="
                         f"{operation.operand_groups}/{operation.result_groups}/"
                         f"{operation.temporary_groups}/{operation.fragment_groups}; local_bytes={operation.local_bytes}")
            if operation.access is not None:
                lines.append("       access: " + operation.access.describe())
            if operation.scalar_packet is not None:
                packet = operation.scalar_packet
                lines.append(f"       scalar packet: axis={packet.axis}, steps={packet.steps}, "
                             f"FPR packet/frontier={packet.registers}/{packet.live_registers}")
            for layout in operation.layouts:
                lines.append("       " + layout.describe())
            for location in operation.locations:
                lines.append("       " + location)
        return "\n".join(lines) + "\n"

    def to_csv(self):
        """Export one row per static operation group, preserving nested layout facts."""
        output = io.StringIO()
        names = [item.name for item in fields(PhysicalOperation)]
        writer = csv.DictWriter(output, fieldnames=["scope", "memory_form", "segment_fields", *names], lineterminator="\n")
        writer.writeheader()
        for operation in self.operations:
            row = operation.to_dict()
            row["scope"] = self.scope
            row["memory_form"] = operation.memory_form
            row["segment_fields"] = operation.segment_fields
            for name in ("parameters", "locations", "layouts", "access", "scalar_packet"):
                row[name] = json.dumps(row[name], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)
        return output.getvalue()
