from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from types import FunctionType

from weft.api import DerivedEncodingDefinition, InlineDefinition, KernelDefinition
from weft.diagnostics import FrontendError, SourceLocation


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
        definition: KernelDefinition[object, object]
        | DerivedEncodingDefinition
        | InlineDefinition
        | FunctionType,
    ) -> FunctionSource:
        if isinstance(definition, InlineDefinition):
            function = definition.python_function
            source = definition.source
        elif isinstance(definition, (KernelDefinition, DerivedEncodingDefinition)):
            function = definition.python_function
            source = definition.source
        elif isinstance(definition, FunctionType):
            function = definition
            from weft.api.definitions import _function_source

            source = _function_source(function, "inline function")
        else:
            raise TypeError("expected a captured Weft or ordinary Python function")
        tree = ast.parse(source.text, filename=source.filename)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        location = SourceLocation(source.filename, source.first_line, 0)
        if len(functions) != 1:
            raise FrontendError("a Weft definition must contain one function", location)
        closure = inspect.getclosurevars(function)
        bindings = dict(function.__globals__)
        bindings.update(closure.globals)
        bindings.update(closure.nonlocals)
        bindings.update(closure.builtins)
        return cls(functions[0], source.filename, source.first_line, bindings, function)

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
            return self.bindings[node.id]
        if isinstance(node, ast.Attribute):
            owner = self.resolve(node.value)
            try:
                return getattr(owner, node.attr)
            except AttributeError as error:
                raise FrontendError(
                    f"Python binding has no attribute {node.attr!r}", self.location(node)
                ) from error
        raise FrontendError("expected a static Python binding", self.location(node))
