from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from weft.diagnostics import LanguageUseError


class DTypeCategory(Enum):
    BOOL = "bool"
    INTEGER = "integer"
    INDEX = "index"
    FLOAT = "float"
    BFLOAT = "bfloat"


@dataclass(frozen=True, slots=True)
class DType:
    name: str
    category: DTypeCategory
    bits: int | None
    signedness: str | None = None

    def __call__(self, value: object) -> object:
        raise LanguageUseError(
            f"W.{self.name}(...) is valid only while lowering a Weft definition"
        )

    def __getitem__(self, shape: object) -> object:
        from .annotations import ArraySpec

        return ArraySpec(self, shape if isinstance(shape, tuple) else (shape,))

    def __matmul__(self, packing: object) -> object:
        from .annotations import ArraySpec

        return ArraySpec(self, ()) @ packing

    def __repr__(self) -> str:
        return f"W.{self.name}"


i1 = DType("i1", DTypeCategory.BOOL, 1)
index = DType("index", DTypeCategory.INDEX, None)
i4 = DType("i4", DTypeCategory.INTEGER, 4, "signed")
i8 = DType("i8", DTypeCategory.INTEGER, 8, "signed")
i16 = DType("i16", DTypeCategory.INTEGER, 16, "signed")
i32 = DType("i32", DTypeCategory.INTEGER, 32, "signed")
i64 = DType("i64", DTypeCategory.INTEGER, 64, "signed")
u1 = DType("u1", DTypeCategory.INTEGER, 1, "unsigned")
u2 = DType("u2", DTypeCategory.INTEGER, 2, "unsigned")
u4 = DType("u4", DTypeCategory.INTEGER, 4, "unsigned")
u6 = DType("u6", DTypeCategory.INTEGER, 6, "unsigned")
u8 = DType("u8", DTypeCategory.INTEGER, 8, "unsigned")
u16 = DType("u16", DTypeCategory.INTEGER, 16, "unsigned")
u32 = DType("u32", DTypeCategory.INTEGER, 32, "unsigned")
u64 = DType("u64", DTypeCategory.INTEGER, 64, "unsigned")
f16 = DType("f16", DTypeCategory.FLOAT, 16)
bf16 = DType("bf16", DTypeCategory.BFLOAT, 16)
f32 = DType("f32", DTypeCategory.FLOAT, 32)
f64 = DType("f64", DTypeCategory.FLOAT, 64)
