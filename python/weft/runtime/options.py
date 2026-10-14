from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType


def source_parameter_name(name) -> bool:
    if not isinstance(name, str):
        return False
    if name.isidentifier():
        return True
    parts = name.split("$")
    return len(parts) > 1 and parts[0] == "auto" and all(part.isdecimal() and int(part) > 0 for part in parts[1:])


@dataclass(frozen=True)
class CompileOptions:
    """One explicit source and physical binding, shared by JIT and AOT."""

    meta: Mapping[str, int] = field(default_factory=dict)
    lmul_eighths: int = 8
    unroll: int = 1
    pipeline_depth: int = 1
    scalar_load_prime: int = 0
    pointwise_issue_copies: int = 1
    contraction_column_copies: int = 4
    panel_transfer_fields: int = 4
    panel_transfer_strategy: str = "issue-work"
    contraction_panel_k: int = 128
    contraction_k_steps: int = 4
    partial_combine_policy: str = "independent-multilevel"
    record_axis_policy: str = "within-record"
    max_widening_combine_groups: int = 2
    matrix_extension: str = "none"
    require_extensions: tuple[str, ...] = ()
    local_tile_strategy: str = "target-default"
    private_stack_bytes: int = 65536

    def __post_init__(self) -> None:
        values = dict(self.meta)
        if any(not source_parameter_name(name) or
               type(value) is not int or value <= 0 for name, value in values.items()):
            raise ValueError("source bindings must be named positive integers")
        object.__setattr__(self, "meta", MappingProxyType(values))
        for name, choices in (("lmul_eighths", (1, 2, 4, 8, 16, 32, 64)),
                              ("pipeline_depth", (1, 2)), ("scalar_load_prime", (0, 1)),
                              ("pointwise_issue_copies", (1, 2, 4)),
                              ("contraction_column_copies", (1, 2, 4)),
                              ("panel_transfer_fields", (1, 2, 4, 8)),
                              ("contraction_panel_k", (32, 64, 128, 256)),
                              ("contraction_k_steps", (1, 2, 4))):
            value = getattr(self, name)
            if type(value) is not int or value not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        for name in ("unroll", "max_widening_combine_groups", "private_stack_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.partial_combine_policy not in ("independent-multilevel", "sequential"):
            raise ValueError("unknown partial-combine policy")
        if self.panel_transfer_strategy not in ("issue-work", "max-fields", "single-field"):
            raise ValueError("panel_transfer_strategy must be issue-work, max-fields or single-field")
        if self.record_axis_policy not in ("within-record", "across-records"):
            raise ValueError("unknown record-axis policy")
        if self.local_tile_strategy not in ("target-default", "atom-partitioned", "register-grouped"):
            raise ValueError("unknown local tile strategy")
        if not isinstance(self.matrix_extension, str) or not self.matrix_extension:
            raise ValueError("matrix_extension must name a capability or 'none'")
        if isinstance(self.require_extensions, str) or any(
                not isinstance(name, str) or not name for name in self.require_extensions):
            raise ValueError("required extensions must be a sequence of nonempty names")
        object.__setattr__(self, "require_extensions", tuple(sorted(set(self.require_extensions))))

    def specialize(self, **bindings: int) -> CompileOptions:
        return replace(self, meta={**self.meta, **bindings})

    def to_dict(self) -> dict:
        return {"meta": dict(self.meta), "lmul_eighths": self.lmul_eighths,
                "unroll": self.unroll, "pipeline_depth": self.pipeline_depth,
                "scalar_load_prime": self.scalar_load_prime,
                "pointwise_issue_copies": self.pointwise_issue_copies,
                "contraction_column_copies": self.contraction_column_copies,
                "panel_transfer_fields": self.panel_transfer_fields,
                "panel_transfer_strategy": self.panel_transfer_strategy,
                "contraction_panel_k": self.contraction_panel_k,
                "contraction_k_steps": self.contraction_k_steps,
                "partial_combine_policy": self.partial_combine_policy,
                "record_axis_policy": self.record_axis_policy,
                "max_widening_combine_groups": self.max_widening_combine_groups,
                "matrix_extension": self.matrix_extension,
                "require_extensions": list(self.require_extensions),
                "local_tile_strategy": self.local_tile_strategy,
                "private_stack_bytes": self.private_stack_bytes}

    def arguments(self) -> list[str]:
        arguments = [f"--meta={name}={value}" for name, value in sorted(self.meta.items())]
        for name, flag in (("lmul_eighths", "auto-lmul-eighths"), ("unroll", "auto-unroll"),
                           ("pipeline_depth", "auto-pipeline-depth"),
                           ("scalar_load_prime", "auto-scalar-load-prime"),
                           ("pointwise_issue_copies", "pointwise-issue-copies"),
                           ("contraction_column_copies", "contraction-column-copies"),
                           ("panel_transfer_fields", "panel-transfer-fields"),
                           ("panel_transfer_strategy", "panel-transfer-strategy"),
                           ("contraction_panel_k", "contraction-panel-k"),
                           ("contraction_k_steps", "contraction-k-steps"),
                           ("partial_combine_policy", "partial-combine-policy"),
                           ("record_axis_policy", "record-axis-policy"),
                           ("max_widening_combine_groups", "max-widening-combine-groups"),
                           ("matrix_extension", "matrix-extension"),
                           ("local_tile_strategy", "local-tile-strategy"),
                           ("private_stack_bytes", "private-stack-bytes")):
            arguments.append(f"--{flag}={getattr(self, name)}")
        arguments.extend(f"--require-extension={name}" for name in self.require_extensions)
        return arguments
