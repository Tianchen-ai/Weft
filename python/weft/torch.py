"""Optional torch.library integration over the ordinary Weft call contract."""
from __future__ import annotations

import inspect
import threading

import torch

from .runtime.artifact import Artifact
from .runtime.autotune import AutotunedKernel
from .runtime.buffer import empty_buffer
from .runtime.compiler import JITKernel
from .runtime.invocation import CompiledKernel
from .runtime.signature import require
from .runtime.binding import SignatureBinder


def _scalar_type(dtype: str) -> str:
    if dtype == "i1":
        return "bool"
    if dtype in ("f32", "f64"):
        return "float"
    if dtype in ("index", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
        return "int"
    raise NotImplementedError(f"torch.library scalar ABI does not support {dtype}")


def _schema(signature, *, out: bool):
    parameters = {parameter.name: parameter for parameter in signature.inputs}
    call = signature.functional_signature()
    arguments, mutations = [], set()
    keyword_only = False
    for parameter in call.parameters.values():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise NotImplementedError("Torch schemas require named public parameters")
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_only:
            arguments.append("*")
            keyword_only = True
        contract = parameters.get(parameter.name)
        if contract is None:
            kind = "SymInt"
        elif contract.kind == "scalar":
            kind = _scalar_type(contract.dtype)
        elif contract.writable:
            kind = f"Tensor(a{len(mutations)}!)"
            mutations.add(contract.name)
        else:
            kind = "Tensor"
        spelling = f"{kind} {parameter.name}"
        if parameter.default is not inspect.Parameter.empty:
            if parameter.default is None or type(parameter.default) not in (int, float, bool):
                raise NotImplementedError("Torch kernel defaults must be literal numeric scalars")
            spelling += "=" + repr(parameter.default)
        arguments.append(spelling)
    outputs = signature.outputs
    if out:
        output_names = [f"output_{index}" for index in range(len(outputs))]
        if {"out", *output_names} & call.parameters.keys():
            raise ValueError("out and indexed output names are reserved for the explicit output operation")
        if not keyword_only:
            arguments.append("*")
        for name in output_names:
            arguments.append(f"Tensor(a{len(mutations)}!) " + name)
            mutations.add(name)
        result = "()"
    elif len(outputs) == 1:
        result = "Tensor"
    else:
        result = "(" + ", ".join("Tensor" for _ in outputs) + ")"
    return "(" + ", ".join(arguments) + ") -> " + result, mutations


def _result(values):
    if len(values) == 1:
        return values[0]
    return tuple(values) if values else None


def _mutation_contract(signature, call):
    tensors = [(parameter, call.values[parameter.name]) for parameter in signature.arguments
               if parameter.kind == "view" and parameter.name in call.values]
    for index, (parameter, tensor) in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{parameter.name} requires a Tensor in torch.library calls")
        for other, other_tensor in tensors[:index]:
            if not (parameter.writable or other.writable):
                continue
            # Functionalization may clone arguments separately. Storage identity
            # is available on FakeTensor/meta without reading a pointer or data.
            if tensor.untyped_storage()._cdata == other_tensor.untyped_storage()._cdata:
                raise NotImplementedError(
                    f"Torch mutation cannot preserve shared storage between {parameter.name} and {other.name}; "
                    "express it as one readwrite View or use the raw Weft call")
        if parameter.writable and parameter.strided:
            # Keep the same conservative non-overlap condition as native binding.
            # Symbolic strides are ordered by dimension instead of concretized.
            axes = [(stride, extent) for stride, extent in zip(tensor.stride(), tensor.shape)
                    if not (type(extent) is int and extent <= 1)]
            axes.sort(key=lambda axis: axis[0])
            occupied = 1
            empty = tensor.numel() == 0
            for stride, extent in axes:
                require(empty | (extent <= 1) | (stride >= occupied),
                        f"writable output {parameter.name} overlaps itself")
                occupied += (extent - 1) * stride


class TorchKernel:
    """A named CPU custom operator with a separate explicit-output operation.

    Register this object in each process before loading a torch.export program.
    Registration/fake evaluation never builds or loads native code. Backward is
    an author-supplied torch.library formula, not an inferred Weft derivative.
    """

    def __init__(self, name: str, kernel):
        if not isinstance(name, str) or len(name.split("::")) != 2 or any(
                not part.isidentifier() for part in name.split("::")):
            raise ValueError("Torch operator name must be namespace::identifier")
        if isinstance(kernel, (JITKernel, AutotunedKernel)):
            signature = kernel.call_signature
        elif isinstance(kernel, (CompiledKernel, Artifact)):
            signature = kernel.interface
        else:
            raise TypeError("register a Weft JIT, autotuned kernel, compiled kernel or saved Artifact")
        if not any(parameter.kind == "view" for parameter in signature.inputs):
            raise NotImplementedError("CPU dispatch requires at least one Tensor input")
        if any(parameter.kind == "view" and not parameter.encoding.startswith("dense.") for parameter in signature.arguments):
            raise NotImplementedError("packed Encodings require the explicit Weft Buffer ABI")
        self.name = name
        self.kernel = kernel
        self.signature = signature
        self.schema, mutations = _schema(signature, out=False)
        out_schema = _schema(signature, out=True)[0] if signature.outputs else None
        namespace, symbol = name.split("::")
        names = (symbol, symbol + "_out") if out_schema is not None else (symbol,)
        if any(hasattr(getattr(torch.ops, namespace), candidate) for candidate in names):
            raise ValueError(f"Torch operator namespace already contains {name} or its output operation")
        for parameter in signature.functional_signature().parameters.values():
            contract = next((item for item in signature.inputs if item.name == parameter.name), None)
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and contract is not None and contract.kind == "view":
                raise NotImplementedError("Torch custom operators require Tensor inputs before the keyword-only boundary")
        self.mutates_args = frozenset(mutations)
        self._functional_signature = signature.functional_signature()
        self._functional_binder = SignatureBinder(self._functional_signature)
        self._loaded = None
        self._lock = threading.RLock()
        self.op = torch.library.custom_op(name, self._cpu, schema=self.schema,
                                          mutates_args=mutations, device_types="cpu")
        self.op.register_fake(self._fake)
        self.out_op = None
        self.out_schema = out_schema
        self._out_library = None
        if signature.outputs:
            self._out_library = torch.library.Library(namespace, "FRAGMENT")
            self._out_library.define(symbol + "_out" + self.out_schema)
            self._out_library.impl(symbol + "_out", self._cpu_out, "CPU")
            torch.library.register_fake(name + "_out", self._fake_out, lib=self._out_library)
            self.out_op = getattr(getattr(torch.ops, namespace), symbol + "_out").default

    def _arguments(self, args, kwargs, *, out=None, metadata_only=False, allocate=False):
        values = self._functional_binder(*args, **kwargs)
        if out is not None:
            values["out"] = out
        call = self.signature.normalize((), values, allow_outputs=True, metadata_only=metadata_only,
                                        allocate_outputs=allocate)
        _mutation_contract(self.signature, call)
        return call

    def _native(self):
        if not isinstance(self.kernel, Artifact):
            return self.kernel
        with self._lock:
            if self._loaded is None or self._loaded.closed:
                self._loaded = self.kernel.load()
            return self._loaded

    def _cpu(self, *args, **kwargs):
        call = self._arguments(args, kwargs, allocate=True)
        with torch.no_grad():
            return self._native()._prepare_call(call)()

    def _take_outputs(self, kwargs):
        values = tuple(kwargs.pop(f"output_{index}") for index in range(len(self.signature.outputs)))
        return values[0] if len(values) == 1 else values

    def _cpu_out(self, *args, **kwargs):
        outputs = self._take_outputs(kwargs)
        call = self._arguments(args, kwargs, out=outputs)
        if torch.is_grad_enabled() and any(isinstance(value, torch.Tensor) and value.requires_grad
                                          for value in call.values.values()):
            raise NotImplementedError("explicit-output operators do not define autograd; use the functional operator")
        missing = [parameter.name for parameter in self.signature.outputs if parameter.name not in call.values]
        if missing:
            raise TypeError(f"out must supply every output: {missing}")
        with torch.no_grad():
            self._native()._prepare_call(call)()
        return None

    def _fake(self, *args, **kwargs):
        call = self._arguments(args, kwargs, metadata_only=True)
        prototype = next(call.values[parameter.name] for parameter in self.signature.inputs if parameter.kind == "view")
        metadata = self.signature.output_metadata(call)
        return _result([empty_buffer(view.shape, parameter.dtype, like=prototype)
                        for parameter, view in zip(self.signature.outputs, metadata)])

    def _fake_out(self, *args, **kwargs):
        outputs = self._take_outputs(kwargs)
        self._arguments(args, kwargs, out=outputs, metadata_only=True)
        return None

    def out(self, *args, out, **kwargs):
        if self.out_op is None:
            raise TypeError("this kernel has no write-only outputs")
        # Distinct mutable Tensor arguments are functionalizable on all of the
        # supported Torch releases; mutable Tensor lists are not. Keep the
        # convenient public out= binding outside the opaque dispatcher schema.
        names = tuple(parameter.name for parameter in self.signature.outputs)
        if isinstance(out, dict):
            if set(out) != set(names):
                raise ValueError(f"out must provide exactly {names}")
            values = tuple(out[name] for name in names)
        elif len(names) == 1:
            values = (out,)
        elif isinstance(out, (tuple, list)) and len(out) == len(names):
            values = tuple(out)
        else:
            raise TypeError(f"out must provide {len(names)} output Tensors")
        return self.out_op(*args, **{f"output_{index}": value for index, value in enumerate(values)}, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)

    def register_backward(self, backward, *, setup_context=None):
        if self.mutates_args:
            raise NotImplementedError("author backward registration requires a functional operator")
        torch.library.register_autograd(self.op, backward, setup_context=setup_context)
        return self

    def close(self):
        with self._lock:
            if self._loaded is not None:
                self._loaded.close()


def register(name: str, kernel) -> TorchKernel:
    """Register an explicit CPU operator; the namespace owns its process lifetime."""
    return TorchKernel(name, kernel)


__all__ = ["TorchKernel", "register"]
