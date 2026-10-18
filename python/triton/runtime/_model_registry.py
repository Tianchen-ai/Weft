"""Resolve live model callables used by cached framework wrappers.

The application still defines its Triton kernels. A generated graph refers to
the source/dependency identity instead of a process-local side-table integer.
This registry neither compiles nor owns native artifacts.
"""
import importlib
import inspect
import json
import weakref

_callables = []


def register(function):
    # JITFunction.__hash__ reads its dependency key; declarations can still be
    # under construction here, so registering must not request that hash.
    _callables.append(weakref.ref(function))


def _hook_identity(hook):
    if hook is None:
        return None
    if not inspect.isfunction(hook):
        raise NotImplementedError("cached model graph hooks must be Python functions")
    closure = inspect.getclosurevars(hook)
    values = {**closure.globals, **closure.nonlocals}
    bindings = {}
    for name, value in values.items():
        if inspect.ismodule(value):
            bindings[name] = {"module": value.__name__}
        elif hasattr(value, "cache_key"):
            bindings[name] = {"jit": value.cache_key}
        else:
            # Reject mutable opaque state rather than capturing its address.
            json.dumps(value)
            bindings[name] = value
    return (hook.__module__, hook.__qualname__, inspect.getsource(hook), bindings)


def reference(callable):
    function = getattr(callable, "jit_fn", callable)
    description = {"source": function.cache_key}
    if callable is not function:
        description.update(
            configs=[(config.all_kwargs(model=function.model), _hook_identity(config.pre_hook))
                     for config in callable.configs],
            key=callable.keys, reset=callable.reset_to_zero, restore=callable.restore_value,
            pre=_hook_identity(callable.pre_hook) if callable.user_defined_pre_hook else None,
            post=_hook_identity(callable.post_hook) if callable.user_defined_post_hook else None,
            prune=_hook_identity(callable.early_config_prune), perf=_hook_identity(callable.perf_model),
            top_k=callable.configs_top_k, benchmark=_hook_identity(callable._do_bench))
    return (function.module, function.fn.__qualname__, json.dumps(description, sort_keys=True))


def resolve(module, name, identity):
    importlib.import_module(module)
    _callables[:] = [item for item in _callables if item() is not None]
    for item in tuple(_callables):
        callable = item()
        if callable is None:
            continue
        function = getattr(callable, "jit_fn", callable)
        if function.module == module and function.fn.__qualname__ == name:
            if reference(callable)[2] == identity:
                return callable
    raise RuntimeError(f"cached graph requires the original Triton model definition {module}:{name}")
