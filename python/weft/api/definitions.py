from __future__ import annotations

import functools
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable, Generic, Mapping, ParamSpec, TypeVar

from weft.diagnostics import DefinitionError, LanguageUseError


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class DefinitionSource:
    filename: str
    first_line: int
    text: str


def _function_source(function: FunctionType, noun: str) -> DefinitionSource:
    try:
        lines, first_line = inspect.getsourcelines(function)
    except (OSError, TypeError) as error:
        raise DefinitionError(f"Weft must be able to inspect the {noun}") from error
    return DefinitionSource(
        inspect.getsourcefile(function) or function.__code__.co_filename,
        first_line,
        textwrap.dedent("".join(lines)),
    )


class KernelDefinition(Generic[P, R]):
    """Captured source for one Weft kernel."""

    def __init__(
        self,
        function: Callable[P, R],
        *,
        alias_groups: Mapping[str, str] | None = None,
        strided: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(function, FunctionType):
            raise DefinitionError("Weft kernels must decorate a Python function")
        if alias_groups is not None and not isinstance(alias_groups, Mapping):
            raise DefinitionError(
                "kernel alias_groups must be a parameter-to-group mapping"
            )
        groups = dict(alias_groups or {})
        if any(
            not isinstance(parameter, str)
            or not parameter
            or not isinstance(group, str)
            or not group
            for parameter, group in groups.items()
        ):
            raise DefinitionError(
                "kernel alias_groups maps parameter names to non-empty group names"
            )
        self.python_function = function
        self.alias_groups = groups
        if isinstance(strided, str) or any(not isinstance(name, str) or not name.isidentifier() for name in strided):
            raise DefinitionError("strided must list dense View parameter names")
        if len(set(strided)) != len(strided):
            raise DefinitionError("strided parameter names must be unique")
        self.strided = tuple(strided)
        self.source = _function_source(function, "decorated DSL kernel")
        functools.update_wrapper(self, function)

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.python_function, eval_str=False)

    def source_text(self) -> str:
        return self.source.text

    def lower(self) -> str:
        from weft.frontend import lower_to_mlir

        return lower_to_mlir(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise LanguageUseError(
            f"kernel {self.__name__} is a definition; use weft.jit or weft.compile to execute it"
        )

    def __repr__(self) -> str:
        return f"<weft kernel {self.__module__}.{self.__qualname__}>"


class EncodingDefinition:
    """Captured pure-layout encoding declaration."""

    def __init__(self, declaration: type[object], bindings: dict[str, object]) -> None:
        if not isinstance(declaration, type):
            raise DefinitionError("@weft.encoding decorates one class declaration")
        try:
            lines, first_line = inspect.getsourcelines(declaration)
        except (OSError, TypeError) as error:
            raise DefinitionError("Weft must be able to inspect the encoding class") from error
        self.python_class = declaration
        self.bindings = bindings
        self.source = DefinitionSource(
            inspect.getsourcefile(declaration) or inspect.getfile(declaration),
            first_line,
            textwrap.dedent("".join(lines)),
        )
        self.__name__ = declaration.__name__
        self.__qualname__ = declaration.__qualname__
        self.__module__ = declaration.__module__

    def __repr__(self) -> str:
        return f"<weft encoding {self.__module__}.{self.__qualname__}>"


class DerivedEncodingDefinition:
    """Captured build-phase encoding-family generator."""

    def __init__(self, function: Callable[..., object]) -> None:
        if not isinstance(function, FunctionType):
            raise DefinitionError("@weft.derive decorates one Python function")
        self.python_function = function
        self.source = _function_source(function, "derived encoding definition")
        functools.update_wrapper(self, function)

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.python_function, eval_str=False)

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"derived encoding {self.__name__} is generated at build time"
        )

    def __getitem__(self, parameters: object) -> DerivedEncodingInstance:
        values = parameters if isinstance(parameters, tuple) else (parameters,)
        if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("derived Encoding parameters must be static integers")
        return DerivedEncodingInstance(self, values)


@dataclass(frozen=True, slots=True)
class DerivedEncodingInstance:
    family: DerivedEncodingDefinition
    parameters: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InlineDefinition:
    python_function: FunctionType
    source: DefinitionSource

    @classmethod
    def capture(cls, function: Callable[..., object]) -> InlineDefinition:
        if not isinstance(function, FunctionType):
            raise DefinitionError("Weft inline functions must be ordinary Python functions")
        return cls(function, _function_source(function, "inline function"))

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.python_function, eval_str=False)

    @property
    def name(self) -> str:
        return self.python_function.__name__


@dataclass(frozen=True, slots=True)
class OverloadSet:
    name: str
    definitions: tuple[InlineDefinition, ...]

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"overload set {self.name} is resolved only while lowering a Weft kernel"
        )


def kernel(
    function: Callable[P, R] | None = None,
    *,
    alias_groups: Mapping[str, str] | None = None,
    strided: tuple[str, ...] = (),
) -> Any:
    if function is None:
        return lambda decorated: KernelDefinition(
            decorated, alias_groups=alias_groups, strided=strided
        )
    return KernelDefinition(function, alias_groups=alias_groups, strided=strided)


def encoding(declaration: type[object]) -> EncodingDefinition:
    frame = inspect.currentframe()
    try:
        if frame is None or frame.f_back is None:
            raise DefinitionError("@weft.encoding cannot capture its definition scope")
        return EncodingDefinition(declaration, dict(frame.f_back.f_globals))
    finally:
        del frame


def derive(function: Callable[..., object]) -> DerivedEncodingDefinition:
    return DerivedEncodingDefinition(function)


def overloads(name: str, *functions: Callable[..., object]) -> OverloadSet:
    if not name or not functions:
        raise DefinitionError("weft.overloads expects a name and at least one function")
    return OverloadSet(name, tuple(InlineDefinition.capture(function) for function in functions))
