from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass

from .diagnostics import DefinitionError, LanguageUseError


@dataclass(frozen=True)
class DeclarationSource:
    filename: str
    first_line: int
    text: str


class EncodingDefinition:
    """A storage declaration captured by the Triton frontend."""

    def __init__(self, declaration, bindings):
        if not isinstance(declaration, type):
            raise DefinitionError("@triton.encoding decorates a class")
        lines, first_line = inspect.getsourcelines(declaration)
        self.source = DeclarationSource(inspect.getfile(declaration), first_line,
                                        textwrap.dedent("".join(lines)))
        self.tree = ast.parse(self.source.text, filename=self.source.filename)
        self.bindings = bindings
        self.__name__ = declaration.__name__
        self.__qualname__ = declaration.__qualname__
        self.__module__ = declaration.__module__

    def dependencies(self):
        names = {node.id for node in ast.walk(self.tree)
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        return tuple(self.bindings[name] for name in sorted(names) if name in self.bindings)


class DerivedEncodingDefinition:
    """A compile-time layout builder with the same AST capture as a JIT helper."""

    def __init__(self, function):
        from triton.runtime.jit import JITFunction
        self.jit = JITFunction(function, model="weft")
        self.__name__ = self.jit.__name__
        self.__qualname__ = self.jit.__qualname__
        self.__module__ = self.jit.__module__

    def __getitem__(self, parameters):
        values = parameters if isinstance(parameters, tuple) else (parameters,)
        if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("derived Encoding parameters must be static integers")
        return DerivedEncodingInstance(self, values)

    def __call__(self, *args, **kwargs):
        raise LanguageUseError("derived Encodings are constructed during compilation")


@dataclass(frozen=True)
class DerivedEncodingInstance:
    family: DerivedEncodingDefinition
    parameters: tuple[int, ...]


@dataclass(frozen=True)
class OverloadSet:
    name: str
    definitions: tuple


def encoding(declaration):
    frame = inspect.currentframe()
    try:
        scope = dict(frame.f_back.f_globals)
        scope.update(frame.f_back.f_locals)
        return EncodingDefinition(declaration, scope)
    finally:
        del frame


def derive(function):
    return DerivedEncodingDefinition(function)


def overloads(name, *functions):
    from triton.runtime.jit import JITFunction
    if not name or not functions or any(not isinstance(fn, JITFunction) or fn.model != "weft" for fn in functions):
        raise DefinitionError("overloads requires a name and Weft model JIT helpers")
    return OverloadSet(name, functions)
