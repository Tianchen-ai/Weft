from __future__ import annotations

from dataclasses import dataclass

from triton.language.weft.diagnostics import LanguageUseError


@dataclass(frozen=True, slots=True)
class Intrinsic:
    name: str
    __triton_builtin__ = True

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"wl.{self.name} is valid only while lowering a Weft definition"
        )


@dataclass(frozen=True, slots=True)
class LevelConstructor:
    relation: str
    __triton_builtin__ = True

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError("Weft levels exist only while lowering a kernel")


class LevelNamespace:
    rows = LevelConstructor("rows")
    cols = LevelConstructor("cols")
    tiles = LevelConstructor("tiles")
    blocks = LevelConstructor("blocks")
    subtiles = LevelConstructor("subs")


level = LevelNamespace()


def _intrinsic(name: str) -> Intrinsic:
    return Intrinsic(name)


state = _intrinsic("state")
arange = _intrinsic("arange")
stage = _intrinsic("stage")
load = _intrinsic("load")
store = _intrinsic("store")
subview = _intrinsic("subview")
extent = _intrinsic("extent")
reshape = _intrinsic("reshape")

mac_pairs = _intrinsic("mac_pairs")
mac_groups = _intrinsic("mac_groups")
widen = _intrinsic("widen")
narrow = _intrinsic("narrow")
reduce = _intrinsic("reduce")
sum_pairs = _intrinsic("sum_pairs")
dot = _intrinsic("dot")
reduce_dot = _intrinsic("reduce_dot")
lookup = _intrinsic("lookup")
interleave = _intrinsic("interleave")

maximum = _intrinsic("maximum")
minimum = _intrinsic("minimum")
maximum_propagating = _intrinsic("maximum_propagating")
minimum_propagating = _intrinsic("minimum_propagating")
select = _intrinsic("select")
abs = _intrinsic("abs")
exp = _intrinsic("exp")
rsqrt = _intrinsic("rsqrt")
log = _intrinsic("log")
sqrt = _intrinsic("sqrt")
