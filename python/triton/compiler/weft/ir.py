from __future__ import annotations

import json
from dataclasses import dataclass, field

from triton.language.weft.diagnostics import SourceLocation

from .types import ValueType, ViewType, SliceType, PointerType, emit_type


@dataclass(frozen=True, slots=True)
class Value:
    id: int
    type: ValueType
    name_hint: str | None = None
    builder: object = field(default=None, compare=False, hash=False, repr=False)
    location: object = field(default=None, compare=False, hash=False, repr=False)
    producer: object = field(default=None, compare=False, hash=False, repr=False)

    @property
    def name(self) -> str:
        return f"%v{self.id}"

    def get_type(self):
        return self.type

    def get_loc(self):
        return self.location

    def set_loc(self, location):
        object.__setattr__(self, "location", location)

    def replace_all_uses_with(self, replacement):
        for region in self.builder.regions:
            for operation in region.operations:
                operation.operands = tuple(replacement if value is self else value
                                           for value in operation.operands)


@dataclass(slots=True)
class Region:
    arguments: tuple[Value, ...]
    operations: list[Operation] = field(default_factory=list)

    def arg(self, index):
        return self.arguments[index]

    def has_terminator(self):
        return self.terminated

    def erase(self):
        self.operations.clear()

    def merge_block_before(self, destination):
        if self.arguments:
            raise ValueError("cannot merge an argument-bearing canonical block")
        destination.operations[:0] = self.operations
        self.operations.clear()

    def get_parent(self):
        return self

    def size(self):
        return 1

    def used_values(self) -> set[Value]:
        values = set()
        for operation in self.operations:
            values.update(operation.operands)
            for region in operation.regions:
                values.update(region.used_values())
        return values

    @property
    def terminated(self) -> bool:
        return bool(self.operations) and self.operations[-1].name in {
            "weft_kernel.return",
            "scf.yield",
            "scf.condition",
            "weft_kernel.births_yield",
            "weft_kernel.handoff",
            "weft_kernel.derive_yield",
            "weft_kernel.reduce_yield",
            "weft_kernel.scan_yield",
        }


@dataclass(slots=True)
class Operation:
    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: dict[str, str]
    regions: tuple[Region, ...]
    location: SourceLocation

    def get_result(self, index):
        return self.results[index]

    def get_num_results(self):
        return len(self.results)

    def get_then_block(self):
        return self.regions[0]

    def get_else_block(self):
        return self.regions[1]

    def get_body(self, index=0):
        return self.regions[index]

    def get_region(self, index):
        return self.regions[index]

    def verify(self):
        # Native construction validates the complete Canonical operation after
        # the shared frontend has populated its combiner and SSA operands.
        return all(region.terminated for region in self.regions)

    def get_before(self):
        return self.regions[0]

    def get_after(self):
        return self.regions[1]

    def get_induction_var(self):
        return self.regions[0].arguments[0]

    def set_attr(self, name, value):
        self.attributes[name] = value


class IRBuilder:
    """One-way canonical MLIR assembly builder."""

    def __init__(self) -> None:
        self._next_value_id = 0
        self.regions = []

    def value(self, value_type: ValueType, name_hint: str | None = None) -> Value:
        value = Value(self._next_value_id, value_type, name_hint, self)
        self._next_value_id += 1
        return value

    def region(
        self,
        argument_types: tuple[ValueType, ...],
        argument_names: tuple[str | None, ...] = (),
    ) -> Region:
        names = argument_names or (None,) * len(argument_types)
        if len(names) != len(argument_types):
            raise ValueError("region argument names and types must match")
        region = Region(
            tuple(
                self.value(value_type, name)
                for value_type, name in zip(argument_types, names)
            )
        )
        self.regions.append(region)
        return region

    def operation(
        self,
        name: str,
        location: SourceLocation,
        *,
        operands: tuple[Value, ...] = (),
        result_types: tuple[ValueType, ...] = (),
        attributes: dict[str, str] | None = None,
        regions: tuple[Region, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> Operation:
        names = result_names or (None,) * len(result_types)
        if len(names) != len(result_types):
            raise ValueError("result names and types must match")
        results = tuple(
            self.value(value_type, hint)
            for value_type, hint in zip(result_types, names)
        )
        operation = Operation(
            name,
            tuple(operands),
            results,
            dict(attributes or {}),
            tuple(regions),
            location,
        )
        for result in results:
            object.__setattr__(result, "producer", operation)
        return operation

    def emit(
        self,
        block: Region,
        name: str,
        location: SourceLocation,
        **kwargs: object,
    ) -> tuple[Value, ...]:
        if block.terminated:
            raise ValueError("cannot emit after a block terminator")
        operation = self.operation(name, location, **kwargs)
        block.operations.append(operation)
        return operation.results


def argument_accesses(body: Region) -> list[set[str]]:
    """Infer ABI memory effects through the completed structured SSA graph.

    This also covers the common frontend's speculative carry discovery: only
    regions in the final program contribute effects. View joins may have more
    than one possible origin; every possible input retains the access effect.
    """
    roots = {value: {index} for index, value in enumerate(body.arguments)
             if isinstance(value.type, (ViewType, PointerType))}
    edges = []
    memory = []

    def connect(destinations, sources):
        for destination, source in zip(destinations, sources):
            if isinstance(destination.type, (ViewType, SliceType, PointerType)):
                edges.append((destination, source))

    def walk(block):
        for operation in block.operations:
            name, args, results, regions = operation.name, operation.operands, operation.results, operation.regions
            if name in ("weft_kernel.slice", "weft_kernel.subview", "weft_kernel.field", "weft_kernel.pointer_cast"):
                connect(results, args[:1])
            elif name == "scf.if":
                for region in regions:
                    connect(results, region.operations[-1].operands)
            elif name == "scf.for":
                carried = regions[0].arguments[1:]
                connect(carried, args[3:])
                connect(carried, regions[0].operations[-1].operands)
                connect(results, carried)
            elif name == "scf.while":
                before, after = regions
                connect(before.arguments, args)
                connect(before.arguments, after.operations[-1].operands)
                forwarded = before.operations[-1].operands[1:]
                connect(after.arguments, forwarded)
                connect(results, forwarded)
            elif name == "weft_kernel.helper":
                def return_edges(region):
                    for inner in region.operations:
                        if inner.name == "weft_kernel.helper_return":
                            connect(results, inner.operands)
                        elif inner.name != "weft_kernel.helper":
                            for nested in inner.regions:
                                return_edges(nested)
                return_edges(regions[0])
            elif name == "weft_kernel.level":
                state, staged, loop = regions
                carried = loop.arguments[1:1 + len(results)]
                connect(carried, args[1:])
                connect(carried, loop.operations[-1].operands)
                connect(results, carried)
                connect(loop.arguments[1 + len(results):],
                        state.operations[-1].operands + staged.operations[-1].operands)
            if name == "weft_kernel.admit":
                memory.append((args[0], ("read",)))
            elif name in ("weft_kernel.pointer_load", "weft_kernel.pointer_store"):
                memory.append((args[0], ("read" if name.endswith("load") else "write",)))
            elif name == "weft_kernel.pointer_atomic":
                memory.append((args[0], ("read", "write")))
            elif name == "weft_kernel.commit":
                memory.append((args[1], ("write",)))
            elif name == "weft_kernel.atomic_rmw":
                memory.append((args[1], ("read", "write")))
            for region in regions:
                walk(region)

    walk(body)
    changed = True
    while changed:
        changed = False
        for destination, source in edges:
            known = roots.setdefault(destination, set())
            incoming = roots.get(source, set())
            if not incoming.issubset(known):
                known.update(incoming)
                changed = True
    accesses = [set() for _ in body.arguments]
    for value, effects in memory:
        for index in roots.get(value, ()):
            accesses[index].update(effects)
    return accesses
