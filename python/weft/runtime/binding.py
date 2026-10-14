from __future__ import annotations

import inspect
import keyword


MISSING = object()


class SignatureBinder:
    """Compile Python's argument binding rules once for an immutable signature.

    Only validated parameter identifiers enter the function text. Defaults are
    namespace objects, never repr/eval expressions. CPython consequently checks
    duplicate values, positional-only and keyword-only parameters before the
    View/ABI contract runs. This code marshals host arguments, not kernel IR.
    """

    def __init__(self, signature: inspect.Signature, *, optional=()):
        parameters = tuple(signature.parameters.values())
        names = tuple(parameter.name for parameter in parameters)
        optional = frozenset(optional)
        if not optional.issubset(names):
            raise ValueError("optional bindings must name signature parameters")
        for parameter in parameters:
            if (not parameter.name.isidentifier() or keyword.iskeyword(parameter.name)
                    or not parameter.name.isascii()):
                raise ValueError("kernel parameters must be ASCII Python identifiers")
            if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise NotImplementedError("kernel signatures cannot contain variadic parameters")

        namespace = {"__builtins__": {}}
        # A namespace name cannot shadow an argument, including deliberately
        # named arguments such as _weft_default_0 or _weft_bind.
        prefix = "_weft_"
        while any(name.startswith(prefix) for name in names):
            prefix += "_"
        function_name = prefix + "bind"
        missing_name = prefix + "missing"
        namespace[missing_name] = MISSING
        declaration = []
        positional_only = sum(parameter.kind == inspect.Parameter.POSITIONAL_ONLY for parameter in parameters)
        keyword_boundary = False
        for index, parameter in enumerate(parameters):
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_boundary:
                declaration.append("*")
                keyword_boundary = True
            value = parameter.name
            if parameter.default is not inspect.Parameter.empty:
                default_name = prefix + f"default_{index}"
                namespace[default_name] = parameter.default
                value += "=" + default_name
            elif optional:
                # An output may precede a required input. Give every otherwise
                # required parameter a sentinel default, then require every
                # non-output in the small wrapper below.
                value += "=" + missing_name
            declaration.append(value)
            if positional_only and index + 1 == positional_only:
                declaration.append("/")

        values = ", ".join(f"{name!r}: {name}" for name in names)
        source = f"def {function_name}({', '.join(declaration)}):\n    return {{{values}}}\n"
        exec(compile(source, "<weft argument binding>", "exec"), namespace)
        self._bind = namespace[function_name]
        self._optional = optional
        self._required = tuple(parameter.name for parameter in parameters
                               if parameter.name not in optional and parameter.default is inspect.Parameter.empty)

    def __call__(self, *args, **kwargs):
        values = self._bind(*args, **kwargs)
        if self._optional:
            for name in self._required:
                if values[name] is MISSING:
                    raise TypeError(f"missing kernel argument: {name}")
            for name in self._optional:
                if values[name] is MISSING:
                    del values[name]
        return values
