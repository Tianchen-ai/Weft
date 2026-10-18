from __future__ import annotations

from dataclasses import dataclass


class WeftError(Exception):
    """Base class for public Weft errors."""


class DefinitionError(WeftError):
    """Raised when a Python definition is not a valid Weft DSL kernel."""


class LanguageUseError(WeftError):
    """Raised when a compile-time-only Weft object is executed as Python."""


@dataclass(frozen=True, slots=True)
class SourceLocation:
    filename: str
    line: int
    column: int

    def format(self) -> str:
        return f"{self.filename}:{self.line}:{self.column + 1}"


class FrontendError(WeftError):
    def __init__(self, message: str, location: SourceLocation | None = None) -> None:
        self.message = message
        self.location = location
        prefix = f"{location.format()}: " if location is not None else ""
        super().__init__(prefix + message)
