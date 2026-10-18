"""Native model calls emitted by PyTorch's existing Triton HOP node."""
import platform


def supports_cpu():
    return platform.machine() == "riscv64"


def is_model_kernel(kernel):
    function = getattr(kernel, "jit_fn", kernel)
    return getattr(function, "model", "triton") == "weft"


def launch(kernel, grids, arguments):
    if len(grids) == 1:
        grid = tuple(grids[0])
    else:
        if len(grids) != len(kernel.configs):
            raise ValueError("framework grid candidates disagree with Triton configs")

        def grid(metadata):
            for candidate, dimensions in zip(kernel.configs, grids):
                if all(metadata.get(key) == value for key, value in candidate.kwargs.items()):
                    return tuple(dimensions)
            raise ValueError("selected Triton config has no framework grid")
    return kernel[grid](**arguments)


def codegen(node, wrapper):
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
    from torch._inductor.virtualized import V
    from triton.runtime._model_registry import reference
    if V.graph.cpp_wrapper:
        raise NotImplementedError("use the native Triton AOT artifact for C++ deployment")
    kernel = kernel_side_table.get_kernel(node.kernel_idx)
    handle = reference(kernel)
    key = ("weft-model", handle)
    if key not in wrapper.user_defined_kernel_cache:
        name = f"triton_model_{len(wrapper.user_defined_kernel_cache)}"
        wrapper.add_import_once("from triton.runtime._model_registry import resolve as resolve_triton_model")
        wrapper.add_import_once("from triton.experimental.torch.launch import launch as launch_triton_model")
        wrapper.header.writeline(f"{name} = resolve_triton_model(*{handle!r})")
        wrapper.user_defined_kernel_cache[key] = name
    name = wrapper.user_defined_kernel_cache[key]
    arguments = ", ".join(f"{key!r}: {wrapper.val_to_arg_str(node.get_kwargs_value(key))}"
                          for key in node.kernel_args)
    grids = wrapper.val_to_arg_str(node.grid)
    wrapper.writeline(f"launch_triton_model({name}, {grids}, {{{arguments}}})")
