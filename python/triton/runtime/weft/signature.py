from __future__ import annotations

import inspect
import math
import sys
from dataclasses import dataclass
from types import MappingProxyType

from .buffer import Buffer, ViewMetadata, empty_buffer, view_metadata
from .binding import SignatureBinder
from .errors import CompilationError


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def require(condition, message: str) -> None:
    """Check an ordinary or Torch symbolic metadata condition without reading data."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(condition, torch.SymBool):
        torch._check(condition, lambda: message)
    elif not condition:
        raise ValueError(message)


def encode_signature(signature: inspect.Signature) -> list[dict]:
    result = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError("kernel signatures cannot contain variadic arguments")
        item = {"name": parameter.name, "kind": parameter.kind.name}
        if parameter.default is not parameter.empty:
            default = parameter.default
            if type(default) not in (type(None), bool, int, float, str):
                raise TypeError(f"{parameter.name}: persistent signatures require literal scalar defaults")
            if isinstance(default, float) and not math.isfinite(default):
                raise ValueError(f"{parameter.name}: nonfinite defaults cannot be serialized")
            item["default"] = default
        result.append(item)
    return result


def decode_signature(parameters: list[dict]) -> inspect.Signature:
    return inspect.Signature(inspect.Parameter(item["name"], getattr(inspect.Parameter, item["kind"]),
                             default=item.get("default", inspect.Parameter.empty)) for item in parameters)


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    dtype: str
    access: str
    encoding: str
    shape: tuple[str, ...]
    alias_set: int
    strided: bool

    @classmethod
    def from_dict(cls, value: dict):
        missing = cls.__dataclass_fields__.keys() - value.keys()
        if missing:
            raise CompilationError(f"incomplete parameter contract: {sorted(missing)}")
        result = cls(**{key: tuple(value[key]) if key == "shape" else value[key]
                        for key in cls.__dataclass_fields__})
        if not result.name.isidentifier() or result.kind not in ("scalar", "view", "pointer"):
            raise CompilationError("invalid compiler argument name or kind")
        if result.access not in ("read", "write", "readwrite", "none"):
            raise CompilationError(f"invalid access mode for {result.name}")
        if result.kind == "scalar" and (result.access != "none" or result.shape or result.alias_set != -1 or result.strided):
            raise CompilationError(f"scalar argument {result.name} carries a memory contract")
        if result.strided and not result.encoding.startswith("dense."):
            raise CompilationError("strided parameters require dense Encoding")
        if result.kind == "pointer" and (result.shape or result.strided or not result.encoding.startswith("dense.")):
            raise CompilationError("pointer parameters have a dense element and explicit source offsets")
        return result

    @property
    def writable(self) -> bool:
        return self.access in ("write", "readwrite")

    @property
    def output(self) -> bool:
        return self.kind == "view" and self.access == "write"

    def semantic_dict(self) -> dict:
        return {key: list(value) if key == "shape" else value
                for key in Parameter.__dataclass_fields__ if (value := getattr(self, key)) is not None}


@dataclass(frozen=True)
class CallArguments:
    values: dict[str, object]
    dimensions: dict[str, object]
    views: dict[str, ViewMetadata]
    buffers: dict[str, Buffer]
    explicit_shapes: dict[str, object]
    contract: tuple
    metadata_only: bool

    def __post_init__(self):
        for name in ("values", "dimensions", "views", "buffers", "explicit_shapes"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def keywords(self, signature: CallSignature) -> dict:
        missing = set(signature.shape_parameters) - self.dimensions.keys()
        if missing:
            raise ValueError(f"call needs explicit shape arguments: {sorted(missing)}")
        return {**self.values, **{name: self.dimensions[name] for name in signature.shape_parameters}}


class CallSignature:
    """Target-independent public arguments and output inference from the author program."""

    parameter_type = Parameter

    def __init__(self, description: dict, signature: list[dict] | None = None, *, bindings=None):
        self.arguments = tuple(self.parameter_type.from_dict(value) for value in description["arguments"])
        self.shape_parameters = tuple(description["shape_parameters"])
        self.bindings = MappingProxyType(dict({} if bindings is None else bindings))
        names = [argument.name for argument in self.arguments]
        if len(set(names)) != len(names) or set(names) & set(self.shape_parameters):
            raise CompilationError("kernel ABI contains colliding parameter names")
        self.signature = decode_signature(signature) if signature is not None else inspect.Signature(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names)
        if list(self.signature.parameters) != names:
            raise CompilationError("Python call signature disagrees with compiler argument order")
        self._outputs = tuple(parameter for parameter in self.arguments if parameter.output)
        self._inputs = tuple(parameter for parameter in self.arguments if not parameter.output)
        self._views = tuple(parameter for parameter in self.arguments if parameter.kind in ("view", "pointer"))
        self._binder = SignatureBinder(self.signature)
        self._output_binder = SignatureBinder(self.signature, optional=(parameter.name for parameter in self._outputs))
        self._shape_contracts = tuple((parameter, tuple(
            int(axis) if axis.isdecimal() else self.bindings.get(axis, axis) for axis in parameter.shape))
            for parameter in self._views)
        self._semantic_contract = tuple(
            (parameter.name, parameter.kind, parameter.dtype, parameter.access,
             parameter.encoding, tuple(str(self.bindings.get(axis, axis)) for axis in parameter.shape),
             parameter.alias_set, parameter.strided) for parameter in self.arguments), tuple(
                 name for name in self.shape_parameters if name not in self.bindings)

    @property
    def outputs(self) -> tuple[Parameter, ...]:
        return self._outputs

    @property
    def inputs(self) -> tuple[Parameter, ...]:
        return self._inputs

    def functional_signature(self) -> inspect.Signature:
        outputs = {parameter.name for parameter in self.outputs}
        parameters = [parameter for parameter in self.signature.parameters.values() if parameter.name not in outputs]
        inferred = {extent for parameter in self.inputs for extent in parameter.shape}
        parameters.extend(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in self.shape_parameters
                          if name not in inferred and name not in self.bindings)
        return inspect.Signature(parameters)

    def _split_shapes(self, kwargs: dict) -> dict:
        dimensions = dict(self.bindings)
        for name in self.shape_parameters:
            if name in kwargs:
                value = kwargs.pop(name)
                if type(value) is not int:
                    torch = sys.modules.get("torch")
                    if torch is None or not isinstance(value, torch.SymInt):
                        raise TypeError(f"shape {name} must be an integer")
                require(value >= 0, f"shape {name} must be non-negative")
                if name in self.bindings:
                    require(value == self.bindings[name], f"shape {name} disagrees with its source binding")
                dimensions[name] = value
        return dimensions

    def _bind_out(self, keywords: dict) -> None:
        if "out" not in keywords or "out" in self.signature.parameters:
            return
        out = keywords.pop("out")
        outputs = self.outputs
        if not outputs:
            raise TypeError("this kernel has no write-only output")
        if isinstance(out, dict):
            if set(out) != {parameter.name for parameter in outputs}:
                raise TypeError("out mapping must name every declared write-only output")
            selected = out
        elif len(outputs) == 1:
            selected = {outputs[0].name: out}
        else:
            if not isinstance(out, (tuple, list)) or len(out) != len(outputs):
                raise TypeError("out must contain one buffer per declared output")
            selected = dict(zip((parameter.name for parameter in outputs), out))
        for name, value in selected.items():
            if name in keywords:
                raise TypeError(f"output {name} was supplied twice")
            keywords[name] = value

    def normalize(self, args: tuple, kwargs: dict, *, allow_outputs: bool = False,
                  metadata_only: bool = False, allocate_outputs: bool = False) -> CallArguments:
        allow_outputs = allow_outputs or allocate_outputs
        keywords = dict(kwargs)
        if allow_outputs:
            self._bind_out(keywords)
        dimensions = self._split_shapes(keywords)
        binder = self._output_binder if allow_outputs else self._binder
        values = binder(*args, **keywords)
        explicit = {name: kwargs[name] for name in self.shape_parameters if name in kwargs}
        return self._capture(values, dimensions, explicit, metadata_only=metadata_only,
                             allocate_outputs=allocate_outputs)

    def _capture(self, values, dimensions, explicit, *, metadata_only, borrowed=None,
                 allocate_outputs=False):
        views, buffers = {}, {}
        for parameter, axes in self._shape_contracts:
            if parameter.name not in values:
                if not parameter.output:
                    raise TypeError(f"missing kernel argument: {parameter.name}")
                continue
            value = values[parameter.name]
            if metadata_only:
                metadata = view_metadata(value, allow_meta=True)
            else:
                if borrowed is not None and parameter.name in borrowed:
                    buffer = borrowed[parameter.name]
                else:
                    buffer = value if isinstance(value, Buffer) else Buffer(value)
                buffers[parameter.name] = buffer
                metadata = buffer.metadata
            views[parameter.name] = metadata
            if metadata.encoding != parameter.encoding:
                raise TypeError(f"{parameter.name} requires {parameter.encoding}, got {metadata.encoding}")
            if parameter.kind == "pointer":
                if parameter.writable and metadata.readonly:
                    raise TypeError(f"{parameter.name} requires writable storage")
                continue
            if len(metadata.shape) != len(parameter.shape):
                raise ValueError(f"{parameter.name} has the wrong logical rank")
            for spelling, extent in zip(axes, metadata.shape):
                if type(extent) is int:
                    if extent < 0:
                        raise ValueError(f"{parameter.name} has a negative extent")
                    expected = spelling if type(spelling) is int else dimensions.setdefault(spelling, extent)
                    if type(expected) is int:
                        if extent != expected:
                            raise ValueError(f"{parameter.name} disagrees with dimension {spelling}")
                        continue
                else:
                    require(extent >= 0, f"{parameter.name} has a negative extent")
                    expected = spelling if type(spelling) is int else dimensions.setdefault(spelling, extent)
                require(extent == expected, f"{parameter.name} disagrees with dimension {spelling}")
            if not parameter.strided:
                metadata.require_contiguous(parameter.name)
            elif any(type(stride) is int and stride < 0 for stride in metadata.strides):
                raise ValueError(f"{parameter.name} has a negative stride")
            if parameter.writable and metadata.readonly:
                raise TypeError(f"{parameter.name} requires writable storage")
        if allocate_outputs:
            self._allocate_values(values, dimensions, views, buffers, metadata_only=metadata_only)
        return CallArguments(values, dimensions, views, buffers, explicit, self._semantic_contract, metadata_only)

    def _require_call(self, call: CallArguments, *, native=False):
        if call.contract != self._semantic_contract:
            raise ValueError("captured invocation belongs to a different kernel interface")
        if native and call.metadata_only:
            raise TypeError("metadata-only invocation cannot enter a native kernel")

    def rebind(self, call: CallArguments, updates: dict) -> CallArguments:
        """Re-infer dimensions while retaining unchanged, current View borrows."""
        self._require_call(call, native=True)
        names = self.signature.parameters.keys() | set(self.shape_parameters)
        unknown = updates.keys() - names
        if unknown:
            raise TypeError(f"unknown rebind arguments: {sorted(unknown)}")
        values = {**call.values, **call.explicit_shapes, **updates}
        dimensions = self._split_shapes(values)
        explicit = {**call.explicit_shapes,
                    **{name: value for name, value in updates.items() if name in self.shape_parameters}}
        borrowed = {}
        for name, buffer in call.buffers.items():
            if name not in updates and (call.values[name] is buffer or buffer.matches_owner()):
                borrowed[name] = buffer
        return self._capture(values, dimensions, explicit, metadata_only=False, borrowed=borrowed)

    def _output_metadata(self, parameter: Parameter, call: CallArguments) -> ViewMetadata:
        return self._output_view(parameter, call.dimensions)

    def _output_view(self, parameter: Parameter, dimensions) -> ViewMetadata:
        if not parameter.encoding.startswith("dense."):
            raise NotImplementedError(f"allocate {parameter.name}'s encoded storage explicitly")
        missing = [extent for extent in parameter.shape if not extent.isdecimal() and extent not in dimensions]
        if missing:
            raise ValueError(f"output {parameter.name} needs explicit dimensions: {missing}")
        shape = tuple(int(extent) if extent.isdecimal() else dimensions[extent] for extent in parameter.shape)
        return ViewMetadata.contiguous(parameter.encoding, shape)

    def output_metadata(self, call: CallArguments) -> tuple[ViewMetadata, ...]:
        return tuple(self._output_metadata(parameter, call) for parameter in self.outputs)

    def allocate(self, call: CallArguments, *, like=None) -> CallArguments:
        self._require_call(call)
        values, views, buffers = dict(call.values), dict(call.views), dict(call.buffers)
        self._allocate_values(values, call.dimensions, views, buffers, metadata_only=call.metadata_only, like=like)
        return CallArguments(values, call.dimensions, views, buffers, call.explicit_shapes,
                             call.contract, call.metadata_only)

    def _allocate_values(self, values, dimensions, views, buffers, *, metadata_only, like=None):
        if like is None:
            like = next((values[parameter.name] for parameter in self.inputs if parameter.kind == "view"), None)
        for parameter in self.outputs:
            if parameter.name in values:
                continue
            metadata = self._output_view(parameter, dimensions)
            value = empty_buffer(metadata.shape, parameter.dtype, like=like)
            values[parameter.name] = value
            views[parameter.name] = metadata
            if not metadata_only:
                buffers[parameter.name] = Buffer(value)

    def semantic_dict(self) -> dict:
        arguments = []
        for parameter in self.arguments:
            value = parameter.semantic_dict()
            value["shape"] = [str(self.bindings[axis]) if axis in self.bindings else axis
                              for axis in parameter.shape]
            arguments.append(value)
        return {"arguments": arguments,
                "shape_parameters": [name for name in self.shape_parameters if name not in self.bindings]}
