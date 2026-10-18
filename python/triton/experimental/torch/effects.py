"""Canonical effects for PyTorch's existing user-defined Triton kernel HOP."""
from triton.compiler.weft.interface import describe
from triton.runtime.autotuner import Autotuner
from triton.runtime.weft.signature import CallSignature


def kernel_interface(kernel, arguments):
    import torch
    from torch._inductor.ir import TensorBox
    from triton.runtime.weft.buffer import _TORCH_DTYPES
    from triton.language.weft import dtypes
    from triton.compiler.weft.values import logical_dtype
    from triton.backends.weft.compiler import WeftBackend
    candidates = kernel.configs if isinstance(kernel, Autotuner) else (None,)
    function = kernel.jit_fn if isinstance(kernel, Autotuner) else kernel
    if function.model != "weft":
        raise TypeError("this framework hook requires the Weft model")
    result = None
    for config in candidates:
        values = dict(arguments)
        if config is not None:
            values.update(config.kwargs)
        signature = {}
        for parameter in function.params:
            if parameter.is_constexpr or parameter.name not in values:
                continue
            value = values[parameter.name]
            if isinstance(value, TensorBox):
                dtype = _TORCH_DTYPES[str(value.get_dtype()).removeprefix("torch.")]
                spelling = "*" + logical_dtype(getattr(dtypes, dtype)).name
            elif isinstance(value, torch.SymInt):
                spelling = "i64"
            elif isinstance(value, torch.SymFloat):
                spelling = "fp32"
            elif isinstance(value, torch.SymBool):
                spelling = "u1"
            else:
                spelling = WeftBackend.specialize_argument("weft", value, False, False, False)[0]
            signature[parameter.name] = spelling
        current = describe(function, values, signature=signature)
        if result is not None and current != result:
            raise ValueError("autotune candidates change the public View/effect contract")
        result = current
    if result is None:
        raise ValueError("autotuning has no candidate programs")
    return result


def kernel_accesses(kernel, arguments, tma_descriptor_metadata):
    """Return exact parameter read/write sets without querying a CPU or a pointer."""
    if tma_descriptor_metadata:
        raise NotImplementedError("Weft View parameters do not use GPU tensor descriptors")
    from torch import Tensor
    from torch._higher_order_ops.triton_kernel_wrap import TensorAccesses
    from torch._inductor.dependencies import ReadWrites, StarDep
    from torch._inductor.ir import TensorBox
    from torch.utils._ordered_set import OrderedSet

    function = kernel.jit_fn if isinstance(kernel, Autotuner) else kernel
    supplied = function.signature.bind_partial(**{key: value for key, value in arguments.items()
                                                  if key in function.arg_names})
    supplied.apply_defaults()
    arguments = {**arguments, **supplied.arguments}
    description = kernel_interface(kernel, arguments)
    names = {arg["name"] for arg in description["arguments"]}
    # FakeTensor metadata is checked with no storage access. Inductor TensorBox
    # uses the same checked graph inputs and is only consulted for dependencies.
    if all(not isinstance(arguments[name], TensorBox) for name in names):
        CallSignature(description).normalize((), {name: arguments[name] for name in names}, metadata_only=True)
    reads, writes = OrderedSet(), OrderedSet()
    for arg in description["arguments"]:
        if arg["kind"] not in ("view", "pointer"):
            continue
        name = arg["name"]
        if not isinstance(arguments[name], (Tensor, TensorBox)):
            raise TypeError(f"PyTorch kernel argument {name!r} must be a Tensor")
        if arg["access"] in ("read", "readwrite"):
            reads.add(StarDep(name))
        if arg["access"] in ("write", "readwrite"):
            writes.add(StarDep(name))
    return TensorAccesses(ReadWrites(reads=reads, writes=writes, index_exprs=OrderedSet()), can_fuse_epilogue=False)
