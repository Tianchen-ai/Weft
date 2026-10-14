from .api import (
    DerivedEncodingDefinition,
    EncodingDefinition,
    KernelDefinition,
    derive,
    encoding,
    kernel,
    overloads,
)
from .diagnostics import DefinitionError, FrontendError, LanguageUseError, WeftError


def compile(definition, *, options=None, toolchain=None, cache=None):
    from .runtime import compile as native_compile

    return native_compile(definition, options=options, toolchain=toolchain, cache=cache)


def jit(definition, *, options=None, toolchain=None, cache=None):
    from .runtime import jit as native_jit

    return native_jit(definition, options=options, toolchain=toolchain, cache=cache)


def jit_ir(canonical, *, options=None, toolchain=None, cache=None):
    from .runtime import jit_ir as native_jit

    return native_jit(canonical, options=options, toolchain=toolchain, cache=cache)


def autotune(definition, configs, *, key=(), policy=None, validator=None,
             validation_key=None, toolchain=None, cache=None):
    from .runtime import autotune as native_autotune

    return native_autotune(definition, configs, key=key, policy=policy, validator=validator,
                           validation_key=validation_key, toolchain=toolchain, cache=cache)


def autotune_ir(canonical, configs, *, key=(), policy=None, validator=None,
                validation_key=None, toolchain=None, cache=None):
    from .runtime import autotune_ir as native_autotune

    return native_autotune(canonical, configs, key=key, policy=policy, validator=validator,
                           validation_key=validation_key, toolchain=toolchain, cache=cache)


def build(definition, *, target, options=None, toolchain=None, native_library=True, cache=None):
    from .runtime import build as build_artifact

    return build_artifact(definition, target=target, options=options, toolchain=toolchain,
                          native_library=native_library, cache=cache)


def build_ir(canonical, *, target, options=None, toolchain=None, native_library=True, cache=None):
    from .runtime import build_ir as build_artifact

    return build_artifact(canonical, target=target, options=options, toolchain=toolchain,
                          native_library=native_library, cache=cache)


def build_candidates(definition, configs, *, target, toolchain=None, native_library=True,
                     cache=None, max_candidates=8):
    from .runtime import build_candidates as build_set

    return build_set(definition, configs, target=target, toolchain=toolchain,
                     native_library=native_library, cache=cache, max_candidates=max_candidates)


def build_candidates_ir(canonical, configs, *, target, toolchain=None, native_library=True,
                        cache=None, max_candidates=8):
    from .runtime import build_candidates_ir as build_set

    return build_set(canonical, configs, target=target, toolchain=toolchain,
                     native_library=native_library, cache=cache, max_candidates=max_candidates)


def load(path, *, target=None, toolchain=None):
    from .runtime import load as load_artifact

    return load_artifact(path, target=target, toolchain=toolchain)


def prepare(definition, *, target, options=None, checkpoint="contractions", toolchain=None, cache=None):
    from .runtime import prepare as prepare_input

    return prepare_input(definition, target=target, options=options, checkpoint=checkpoint,
                         toolchain=toolchain, cache=cache)


def prepare_ir(canonical, *, target, options=None, checkpoint="contractions", toolchain=None, cache=None):
    from .runtime import prepare_ir as prepare_input

    return prepare_input(canonical, target=target, options=options, checkpoint=checkpoint,
                         toolchain=toolchain, cache=cache)


def lower_to_mlir(definition: KernelDefinition[object, object]) -> str:
    from .frontend import lower_to_mlir as lower

    return lower(definition)


__all__ = [
    "DefinitionError",
    "DerivedEncodingDefinition",
    "EncodingDefinition",
    "FrontendError",
    "KernelDefinition",
    "LanguageUseError",
    "WeftError",
    "derive",
    "build",
    "build_ir",
    "build_candidates",
    "build_candidates_ir",
    "jit_ir",
    "autotune_ir",
    "prepare",
    "prepare_ir",
    "compile",
    "encoding",
    "kernel",
    "jit",
    "autotune",
    "load",
    "lower_to_mlir",
    "overloads",
]
