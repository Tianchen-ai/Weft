from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from types import FunctionType

from triton.language.weft.declarations import DerivedEncodingDefinition
from triton.language.weft.diagnostics import FrontendError, SourceLocation


@dataclass(frozen=True, slots=True)
class FunctionSource:
    function: ast.FunctionDef
    filename: str
    first_line: int
    bindings: dict[str, object]
    python_function: FunctionType

    @classmethod
    def from_definition(
        cls,
        definition,
    ) -> FunctionSource:
        from triton.runtime.jit import JITFunction
        if isinstance(definition, DerivedEncodingDefinition):
            definition = definition.jit
        if not isinstance(definition, JITFunction) or (definition.model != "weft" and definition.explicit_model):
            raise TypeError("expected @triton.jit(model='weft'); cross-model helpers require explicit semantics")
        return cls(definition.parse().body[0], definition.file_name,
                   definition.def_file_line_number, {**vars(builtins), **definition.get_capture_scope()}, definition.fn)

    def location(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            self.filename,
            self.first_line + getattr(node, "lineno", 1) - 1,
            getattr(node, "col_offset", 0),
        )

    def resolve(self, node: ast.AST) -> object:
        if isinstance(node, ast.Name):
            if node.id not in self.bindings:
                raise FrontendError(
                    f"unknown Python binding {node.id!r}", self.location(node)
                )
            return normalize_static(self.bindings[node.id])
        if isinstance(node, ast.Attribute):
            owner = self.resolve(node.value)
            try:
                return normalize_static(getattr(owner, node.attr))
            except AttributeError as error:
                raise FrontendError(
                    f"Python binding has no attribute {node.attr!r}", self.location(node)
                ) from error
        raise FrontendError("expected a static Python binding", self.location(node))


def normalize_static(value):
    """Resolve dtype annotations without replacing any Triton operation."""
    import triton.language as tl
    from triton.language.weft import dtypes
    if isinstance(value, tl.dtype):
        types = {tl.float16: dtypes.f16, tl.bfloat16: dtypes.bf16,
                 tl.float32: dtypes.f32, tl.float64: dtypes.f64,
                 tl.int1: dtypes.i1, tl.int8: dtypes.i8, tl.int16: dtypes.i16,
                 tl.int32: dtypes.i32, tl.int64: dtypes.i64,
                 tl.uint8: dtypes.u8, tl.uint16: dtypes.u16,
                 tl.uint32: dtypes.u32, tl.uint64: dtypes.u64}
        if value not in types:
            raise TypeError(f"dtype {value} is not supported by the Weft model")
        return types[value]
    return value
