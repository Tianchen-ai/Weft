"""Shape and ownership rules shared by native and framework program calls."""
from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Lock

from triton.runtime.weft.buffer import Buffer, empty_buffer, torch_module, view_metadata


def integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def tensor(value, name, *, rank=None, shape=None, dtype="f32", writable=False):
    metadata = view_metadata(value)
    if metadata.encoding != "dense." + dtype:
        raise TypeError(f"{name} requires {dtype}, got {metadata.encoding}")
    if rank is not None and len(metadata.shape) != rank:
        raise ValueError(f"{name} requires rank {rank}, got {metadata.shape}")
    if shape is not None and tuple(shape) != metadata.shape:
        raise ValueError(f"{name} requires shape {tuple(shape)}, got {metadata.shape}")
    if any(stride < 0 for stride in metadata.strides):
        raise ValueError(f"{name} requires nonnegative strides")
    if writable and metadata.readonly:
        raise ValueError(f"{name} must be writable")
    return metadata.shape


def same_backend(*values):
    kinds = {torch_module(value) is not None for value in values}
    if len(kinds) != 1:
        raise TypeError("one program invocation must use one buffer family (NumPy or Torch)")


def reshape(value, shape):
    """Borrow a different shape, refusing any reshape that would copy storage."""
    shape = tuple(shape)
    if math.prod(shape) != math.prod(value.shape):
        raise ValueError(f"cannot reshape {tuple(value.shape)} to {shape}")
    module = torch_module(value)
    if module is not None:
        return value.view(shape)
    import numpy as np

    if not isinstance(value, np.ndarray):
        raise TypeError("model composition requires a NumPy array or CPU Torch tensor")
    result = value.view()
    result.shape = shape
    return result


def rows(value):
    shape = tensor(value, "activation")
    if not shape or shape[-1] <= 0:
        raise ValueError("activation needs a positive last dimension")
    return reshape(value, (math.prod(shape[:-1]), shape[-1]))


def allocate(shape, *, like, dtype="f32", zero=False):
    result = empty_buffer(tuple(shape), dtype, like=like)
    if zero:
        clear(result)
    return result


def clear(value):
    # Storage initialization is a host responsibility; model arithmetic is Weft.
    module = torch_module(value)
    if module is None:
        value.fill(0)
    else:
        with module.no_grad():
            value.zero_()


def clone(value):
    module = torch_module(value)
    return value.copy() if module is None else value.detach().clone()


def copy_storage(destination, source):
    same_backend(destination, source)
    tensor(destination, "destination", shape=tuple(source.shape), writable=True)
    module = torch_module(destination)
    if module is None:
        destination[...] = source
    else:
        with module.no_grad():
            destination.copy_(source)


def output(value, shape, *, like):
    if value is None:
        return allocate(shape, like=like)
    tensor(value, "out", shape=shape, writable=True)
    same_backend(value, like)
    return value


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("workspace buffer needs a name")
        if self.dtype not in ("f32", "i32"):
            raise NotImplementedError("model workspace storage supports f32 and i32")
        if any(type(extent) is not int or extent < 0 for extent in self.shape):
            raise ValueError("workspace dimensions must be nonnegative integers")

    @property
    def nbytes(self):
        return math.prod(self.shape) * 4


class Workspace:
    """Explicit, reusable scratch belonging to one model invocation at a time."""

    def __init__(self, specs, *, like):
        self.specs = tuple(specs)
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("workspace buffer names must be unique")
        self.buffers = {spec.name: allocate(spec.shape, dtype=spec.dtype, like=like)
                        for spec in self.specs}
        self._torch = torch_module(like) is not None
        self._lock = Lock()

    @property
    def nbytes(self):
        return sum(spec.nbytes for spec in self.specs)

    def check(self, specs, *, like):
        if tuple(specs) != self.specs:
            raise ValueError("workspace shape does not match this invocation; allocate the requested workspace")
        if self._torch != (torch_module(like) is not None):
            raise TypeError("workspace and activation must use the same buffer family")
        for spec in self.specs:
            tensor(self.buffers[spec.name], spec.name, shape=spec.shape, dtype=spec.dtype, writable=True)
        return self

    def __getitem__(self, name):
        return self.buffers[name]

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("workspace already belongs to an active invocation")
        return self

    def __exit__(self, *exception):
        self._lock.release()


def use_workspace(workspace, specs, *, like):
    if workspace is None:
        return Workspace(specs, like=like)
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace must be a weft.programs.Workspace")
    return workspace.check(specs, like=like)


def parameter_buffers(weights, prefix="weights"):
    """Walk the declared parameter record without reading tensor contents."""
    from dataclasses import fields, is_dataclass

    if not is_dataclass(weights):
        raise TypeError("model parameters must use their declared weight record")
    values = {}
    for field in fields(weights):
        value = getattr(weights, field.name)
        name = prefix + "." + field.name
        if is_dataclass(value):
            values.update(parameter_buffers(value, name))
        elif value is not None:
            values[name] = value
    return values


def check_invocation(x, y, weights, workspace, *, state=None, extra_inputs=None):
    """Check whole-program ownership before the first kernel can write.

    Kernel alias groups cover one call. A later model call could otherwise
    overwrite a parameter read only by an earlier call. Model output, scratch
    and streaming state therefore have disjoint storage spans. Ordinary raw
    operators retain their more permissive declared alias contracts.
    """
    inputs = {"x": x, **parameter_buffers(weights), **(extra_inputs or {})}
    writes = {"out": y, **{"workspace." + name: value for name, value in workspace.buffers.items()},
              **(state or {})}
    borrowed = []
    for writable, values in ((False, inputs), (True, writes)):
        for name, value in values.items():
            buffer = Buffer(value)
            buffer.validate(writable=writable)
            if writable:
                if buffer.readonly:
                    raise ValueError(f"{name} is readonly")
                buffer.require_nonoverlapping()
            span = buffer.byte_span()
            if span:
                for previous_name, previous, previous_writable, previous_span in borrowed:
                    if not (writable or previous_writable):
                        continue
                    if buffer.address < previous.address + previous_span and previous.address < buffer.address + span:
                        raise ValueError(f"model storage spans overlap: {name} and {previous_name}")
            borrowed.append((name, buffer, writable, span))
