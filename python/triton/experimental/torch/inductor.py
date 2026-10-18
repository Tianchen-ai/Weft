"""CPU model hooks for Inductor's own Triton codegen and CachingAutotuner.

The source integration in triton.tools.pytorch calls these at the existing
compilation and launcher boundaries. Graph scheduling, fusion, configuration
enumeration and wrapper generation remain in Inductor.
"""
import math


def reduction_configs(configs, metadata):
    """Add finite CPU streaming tiles to Inductor's existing tuning domain.

    Only looped reductions have a variable reduction block. Persistent kernels
    and explicit user configurations are excluded by the call site. All
    candidates still pass the compiler's numerical/effect/resource verifiers.
    """
    import copy
    from triton.runtime.driver import driver

    if any(metadata.get(key, False) for key in
           ("deterministic", "batch_invariant", "are_deterministic_algorithms_enabled")):
        return configs
    target = driver.for_model("weft").get_current_target()
    vector_elements = target.vlen_bits // 32
    expanded = list(configs)
    for config in configs:
        blocks = [name for name in config.kwargs if name == "RBLOCK" or
                  (name.startswith("R") and name[1:-6].isdigit() and name.endswith("_BLOCK"))]
        if not blocks:
            continue
        for copies in (8, 16):
            tile = vector_elements * copies
            candidate = copy.deepcopy(config)
            for name in blocks:
                candidate.kwargs[name] = min(config.kwargs[name], tile)
            if candidate.kwargs != config.kwargs:
                expanded.append(candidate)
    return expanded


def require_cpu():
    from .launch import supports_cpu
    if not supports_cpu():
        raise RuntimeError("this Triton CPU model requires a supported RISC-V host")


def precompile(autotuner, config, cc_override=None):
    import triton
    from triton.compiler import ASTSource
    from triton.runtime.driver import driver
    from torch._inductor.runtime.triton_heuristics import TritonCompileResult

    if cc_override is not None:
        raise NotImplementedError("GPU compute-capability overrides do not describe a RISC-V CPU target")
    metadata = autotuner._create_compile_meta(config)
    target = driver.for_model("weft").get_current_target()
    options = dict(metadata.get("backend_options", {}))
    if config.backend_options is not None:
        options.update(config.backend_options)
    if config.num_cpu_threads is not None:
        options["num_cpu_threads"] = config.num_cpu_threads
    options["debug"] = metadata["debug"]
    if "enable_fp_fusion" in metadata and not metadata["enable_fp_fusion"]:
        raise NotImplementedError("disabling the model's declared local FMA contraction requires a numeric option")
    binary = triton.compile(ASTSource(autotuner.fn, metadata["signature"], metadata["constants"],
                                     metadata["configs"][0]), target=target, options=options)
    return TritonCompileResult(binary, config, metadata, autotuner.inductor_meta)


def make_launcher(result):
    import triton
    import torch
    from triton import knobs
    from torch._inductor.runtime.triton_heuristics import get_constexprs, triton_hash_to_path_key

    binary = result.kernel
    function = binary.src.fn
    binary._init_handles()
    arguments, parameters, _ = result._get_arg_lists(function.arg_names, get_constexprs(function))
    scope = {"grid_meta": result.config.kwargs, "math": math, "torch": torch, "triton": triton,
             "runner": binary.run, "function": binary.function, "metadata": binary.packed_metadata,
             "enter": knobs.runtime.launch_enter_hook, "exit": knobs.runtime.launch_exit_hook}
    launcher = result._gen_launcher_code(scope, parameters,
        ["grid_0", "grid_1", "grid_2", "stream", "function", "metadata", "None", "enter", "exit", *arguments])
    launcher.config = result.config
    launcher.n_regs = None
    launcher.n_spills = None
    launcher.shared = 0
    launcher.cache_hash = triton_hash_to_path_key(binary.hash)
    launcher.store_cubin = False
    launcher._is_static = False
    launcher.bin = binary
    launcher.fn = function
    return launcher


def clone_tuning_arguments(autotuner, excluded, args, kwargs):
    from .functional import clone_arguments

    if excluded:
        raise ValueError("CPU tuning keeps private storage; device-copy exclusions are not valid")
    names = [p.name for p in autotuner.fn.params if not p.is_constexpr]
    names += list(autotuner.inductor_meta.get("extra_launcher_args", ()))
    if len(args) > len(names):
        raise TypeError("tuning arguments exceed the generated kernel signature")
    positional = names[:len(args)]
    arguments = dict(zip(positional, args))
    if arguments.keys() & kwargs.keys():
        raise TypeError("tuning argument supplied twice")
    arguments.update(kwargs)
    private = clone_arguments(arguments, set(autotuner.mutated_arg_names) | set(autotuner.reset_to_zero_arg_names))
    return [private[name] for name in positional], {name: private[name] for name in kwargs}
