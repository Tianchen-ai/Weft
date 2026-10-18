"""Target-independent metadata from a verified Canonical program.

Framework analysis uses the same frontend as execution. It never selects a
device, borrows Tensor storage, or compiles a native library.
"""
import inspect

from triton._C.libtriton import ir, weft


def describe(function, arguments, *, signature=None):
    from triton.compiler import ASTSource
    from triton.backends.weft.compiler import WeftOptions
    if function.model != "weft":
        raise TypeError("Canonical interface analysis requires the Weft model")
    constants = {}
    for parameter in function.params:
        if not parameter.is_constexpr:
            continue
        if parameter.name in arguments:
            constants[parameter.name] = arguments[parameter.name]
        else:
            default = function.signature.parameters[parameter.name].default
            if default is inspect.Parameter.empty:
                raise TypeError(f"missing compile-time parameter {parameter.name!r}")
            constants[parameter.name] = default
    if signature is None:
        from triton.backends.weft.compiler import WeftBackend
        signature = {name: WeftBackend.specialize_argument("weft", value, False, False, False)[0]
                     for name, value in arguments.items() if name in function.arg_names and name not in constants}
    source = ASTSource(function, signature, constexprs=constants)
    context = ir.context()
    weft.load_dialects(context)
    from triton.backends.weft.compiler import WeftBackend
    module = source.make_ir(None, WeftOptions(), {}, WeftBackend.get_module_map(None), context)
    result = weft.interface(module)["interface"]
    result["arguments"] = [arg for arg in result["arguments"] if not arg["name"].startswith("__triton_")]
    return result
