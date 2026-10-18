"""Install the model hooks in a selected PyTorch source/install tree.

This is a source integration step, not a runtime monkey patch. The supported
Triton HOP/effect and Inductor interfaces are checked before any file is written.
"""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import textwrap


def _function(source, path):
    body = ast.parse(source).body
    for name in path:
        nodes = [node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
        if len(nodes) != 1:
            raise NotImplementedError(f"PyTorch has no supported interface {'.'.join(path)}")
        node = nodes[0]
        body = node.body
    return node


def _insert(source, path, marker, body, arguments):
    node = _function(source, path)
    names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
    if not set(arguments) <= names:
        raise NotImplementedError(f"PyTorch interface {'.'.join(path)} has changed")
    lines = source.splitlines(keepends=True)
    if marker in ''.join(lines[node.lineno - 1:node.end_lineno]):
        return source
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        position = first.end_lineno
    else:
        position = first.lineno - 1
    addition = textwrap.indent(f"# {marker}\n" + textwrap.dedent(body).strip() + "\n", " " * (node.col_offset + 4))
    lines.insert(position, addition)
    return ''.join(lines)


def _replace(source, before, after):
    if after in source:
        return source
    if source.count(before) != 1:
        raise NotImplementedError("PyTorch model integration boundary has changed")
    return source.replace(before, after, 1)


def install(torch_directory):
    root = Path(torch_directory).resolve(strict=True)
    files = {name: (root / name).read_text() for name in
             ("_dynamo/device_interface.py", "_higher_order_ops/triton_kernel_wrap.py", "_inductor/ir.py",
              "utils/_triton.py", "_inductor/codegen/triton.py", "_inductor/runtime/triton_heuristics.py",
              "_inductor/select_algorithm.py")}
    updated = dict(files)
    name = "_dynamo/device_interface.py"
    updated[name] = _insert(updated[name], ("CpuInterface", "raise_if_triton_unavailable"),
        "TRITON_MODEL_CPU_CAPABILITY", '''
        from triton.experimental.torch.launch import supports_cpu
        if supports_cpu():
            return
        ''', ())

    name = "_higher_order_ops/triton_kernel_wrap.py"
    updated[name] = _insert(updated[name], ("identify_accessed_tensors",), "TRITON_MODEL_CANONICAL_EFFECTS", '''
        from triton.experimental.torch.launch import is_model_kernel
        if is_model_kernel(kernel):
            from triton.experimental.torch import kernel_accesses
            return kernel_accesses(kernel, kwargs, tma_descriptor_metadata)
        ''', ("kernel", "kwargs", "tma_descriptor_metadata"))
    updated[name] = _insert(updated[name], ("triton_kernel_wrapper_mutation_fake_tensor_mode",),
        "TRITON_MODEL_FAKE_ARGUMENTS", '''
        from triton.experimental.torch.launch import is_model_kernel
        kernel = kernel_side_table.get_kernel(kernel_idx)
        if is_model_kernel(kernel):
            from triton.experimental.torch import kernel_accesses
            constants = kernel_side_table.get_constant_args(constant_args_idx)
            kernel_accesses(kernel, {**kwargs, **constants}, tma_descriptor_metadata)
        ''', ("kernel_idx", "constant_args_idx", "kwargs", "tma_descriptor_metadata"))
    before = '''    kwargs = {
        key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
        for key, val in kwargs.items()
    }
'''
    after = '''    # TRITON_MODEL_ALIAS_PRESERVING_COPIES
    from triton.experimental.torch.launch import is_model_kernel
    if is_model_kernel(kernel_side_table.get_kernel(kernel_idx)):
        from triton.experimental.torch.functional import clone_arguments
        kwargs = clone_arguments(kwargs, tensors_to_clone)
    else:
        kwargs = {
            key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
            for key, val in kwargs.items()
        }
'''
    updated[name] = _replace(updated[name], before, after)
    updated[name] = _insert(updated[name], ("triton_kernel_wrapper_functional_fake_tensor_mode",),
        "TRITON_MODEL_FAKE_ALIASES", '''
        from triton.experimental.torch.launch import is_model_kernel
        if is_model_kernel(kernel_side_table.get_kernel(kernel_idx)):
            from triton.experimental.torch.functional import clone_arguments
            values = clone_arguments(kwargs, tensors_to_clone)
            return {name: values[name] for name in tensors_to_clone}
        ''', ("kernel_idx", "kwargs", "tensors_to_clone"))
    updated[name] = _replace(updated[name], '                special_kwargs[name] = self.get_value(val)\n', '''                # TRITON_MODEL_STATIC_LAUNCH_CONFIGURATION
                from triton.experimental.torch.launch import is_model_kernel
                if is_model_kernel(variable.kernel):
                    val = variable.specialize_symbolic(val)
                special_kwargs[name] = self.get_value(val)
''')

    name = "_inductor/ir.py"
    updated[name] = _insert(updated[name], ("UserDefinedTritonKernel", "_codegen"),
        "TRITON_MODEL_NATIVE_LAUNCH", '''
        from triton.experimental.torch.launch import is_model_kernel, codegen
        kernel, _, _, _ = self.get_kernel_and_metadata()
        if is_model_kernel(kernel):
            if epilogue_fusion is not None:
                raise NotImplementedError("model kernels own their numerical epilogue")
            return codegen(self, wrapper)
        ''', ("self", "wrapper", "epilogue_fusion"))

    name = "utils/_triton.py"
    updated[name] = _insert(updated[name], ("has_triton_cpu_backend",), "TRITON_MODEL_CPU_BACKEND", '''
        if has_triton_package():
            from triton.experimental.torch.launch import supports_cpu
            if supports_cpu():
                return True
        ''', ())
    updated[name] = _replace(updated[name], 'def triton_backend() -> Any:', 'def triton_backend(model=None) -> Any:')
    updated[name] = _replace(updated[name], 'target = driver.active.get_current_target()',
                            'target = driver.for_model(model).get_current_target() if model else driver.active.get_current_target()')
    updated[name] = _replace(updated[name], 'def triton_hash_with_backend() -> str:',
                            'def triton_hash_with_backend(model=None) -> str:')
    updated[name] = _replace(updated[name], '    backend = triton_backend()\n', '    backend = triton_backend(model)\n')

    name = "_inductor/codegen/triton.py"
    updated[name] = _replace(updated[name], '"backend_hash": torch.utils._triton.triton_hash_with_backend(),',
        '"backend_hash": torch.utils._triton.triton_hash_with_backend(model="weft" if V.graph.get_current_device_or_throw().type == "cpu" else None),')
    updated[name] = _replace(updated[name],
        'code.splice("triton_helpers.set_driver_to_cpu()")',
        'code.splice("from triton.experimental.torch.inductor import require_cpu\\nrequire_cpu()")')
    updated[name] = _replace(updated[name], '        code.splice(heuristics_line)\n', '''        # TRITON_MODEL_CPU_CODEGEN
        if V.graph.get_current_device_or_throw().type == "cpu":
            heuristics_line = heuristics_line.replace("@triton.jit", '@triton.jit(model="weft")')
        code.splice(heuristics_line)
''')
    updated[name] = _replace(updated[name],
        '            device_interface.device(V.graph.get_current_device_or_throw()),  # type: ignore[attr-defined]\n',
        '''            # CPU has no indexed device context; retain the normal fusion benchmark.
            contextlib.nullcontext() if V.graph.device_type == "cpu" else
            device_interface.device(V.graph.get_current_device_or_throw()),  # type: ignore[attr-defined]
''')
    updated[name] = _replace(updated[name], '            if launchers[0].n_spills > n_spills_threshold:\n',
        '            if launchers[0].n_spills is not None and launchers[0].n_spills > n_spills_threshold:\n')

    name = "_inductor/select_algorithm.py"
    updated[name] = _insert(updated[name], ("TritonTemplateKernel", "jit_lines"),
        "TRITON_MODEL_CPU_TEMPLATE", '''
        jit_decorator = '@triton.jit(model="weft")' if self.output_node.get_device().type == "cpu" else "@triton.jit"
        ''', ("self",))
    updated[name] = _replace(updated[name], '            return "@triton.jit"\n', '            return jit_decorator\n')
    updated[name] = _replace(updated[name], '            @triton.jit\n        """\n', '            {jit_decorator}\n        """\n')

    name = "_inductor/runtime/triton_heuristics.py"
    updated[name] = _insert(updated[name], ("CachingAutotuner", "_precompile_worker"),
        "TRITON_MODEL_CPU_CANDIDATES", '''
        from triton.compiler.errors import BackendCompilationError
        ''', ("self",))
    updated[name] = _replace(updated[name], '''            except (OutOfResources, PTXASError, IntelGPUError) as e:
                exc = e
''', '''            except (OutOfResources, PTXASError, IntelGPUError) as e:
                exc = e
            except BackendCompilationError as e:
                if getattr(self.fn, "model", None) != "weft" or not e.candidate_rejected:
                    raise
                exc = e
''')
    updated[name] = _insert(updated[name], ("cached_autotune",),
        "TRITON_MODEL_CPU_REDUCTION_CONFIGS", '''
        device = triton_meta.get("device")
        if (getattr(device, "type", None) == "cpu" and not custom_kernel and
                heuristic_type == HeuristicType.REDUCTION):
            from triton.experimental.torch.inductor import reduction_configs
            configs = reduction_configs(configs, inductor_meta or {})
        ''', ("triton_meta", "configs", "custom_kernel", "heuristic_type", "inductor_meta"))
    updated[name] = _insert(updated[name], ("CachingAutotuner", "_precompile_config"),
        "TRITON_MODEL_CPU_COMPILE", '''
        if self.device_props.type == "cpu" and getattr(self.fn, "model", None) == "weft":
            from triton.experimental.torch.inductor import precompile
            return precompile(self, cfg, cc_override)
        ''', ("self", "cfg", "cc_override"))
    updated[name] = _insert(updated[name], ("TritonCompileResult", "make_launcher"),
        "TRITON_MODEL_CPU_GRID", '''
        if self.kernel.metadata.target.backend == "weft":
            from triton.experimental.torch.inductor import make_launcher
            return make_launcher(self)
        ''', ("self",))
    updated[name] = _insert(updated[name], ("CachingAutotuner", "maybe_clone_args"),
        "TRITON_MODEL_CPU_TUNING_STORAGE", '''
        if self.device_props.type == "cpu" and getattr(self.fn, "model", None) == "weft":
            from triton.experimental.torch.inductor import clone_tuning_arguments
            return clone_tuning_arguments(self, exclude, args, kwargs)
        ''', ("self", "exclude"))
    updated[name] = _replace(updated[name],
        '            # reset to zero before evaluating any config\n            self.reset_to_zero_args(*args, **kwargs)\n', '''            # reset to zero before evaluating any config
            # TRITON_MODEL_CPU_TUNING_RESET
            if self.device_props.type == "cpu" and getattr(self.fn, "model", None) == "weft":
                self.reset_to_zero_args(*cloned_args, **cloned_kwargs)
            else:
                self.reset_to_zero_args(*args, **kwargs)
''')
    for name, source in updated.items():
        ast.parse(source, filename=str(root / name))
    changed = []
    for name, source in updated.items():
        if source == files[name]:
            continue
        destination = root / name
        descriptor, temporary = tempfile.mkstemp(prefix=".triton-model-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(source)
            os.chmod(temporary, destination.stat().st_mode)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        changed.append(str(destination))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-root", type=Path, help="the directory containing torch/__init__.py")
    options = parser.parse_args()
    root = options.torch_root
    if root is None:
        specification = importlib.util.find_spec("torch")
        if specification is None:
            raise ModuleNotFoundError("install the supported PyTorch distribution first")
        root = Path(specification.origin).parent
    for path in install(root):
        print(path)


if __name__ == "__main__":
    main()
