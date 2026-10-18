from __future__ import annotations

import ctypes
import itertools
import math
import sys
import weakref
from collections.abc import Sequence
from dataclasses import dataclass


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p), ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)), ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None

_DENSE_FORMATS = {
    "b": "i8", "B": "u8", "h": "i16", "H": "u16",
    "i": "i32", "I": "u32", "q": "i64", "Q": "u64",
    "e": "f16", "f": "f32", "d": "f64", "?": "i1",
}
_TORCH_DTYPES = {"int8": "i8", "uint8": "u8", "int16": "i16", "uint16": "u16",
                 "int32": "i32", "uint32": "u32", "int64": "i64", "uint64": "u64",
                 "float16": "f16", "bfloat16": "bf16", "float32": "f32", "float64": "f64", "bool": "i1"}


def torch_module(value: object):
    module = sys.modules.get("torch")
    return module if module is not None and isinstance(value, module.Tensor) else None


def _torch_metadata(tensor, storage=None) -> tuple:
    storage = tensor.untyped_storage() if storage is None else storage
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
            tensor.dtype, tensor.device, tensor.layout, tensor.is_conj(), tensor.is_neg(),
            storage._cdata, storage.nbytes())


def element_bytes(encoding: str) -> int:
    if not encoding.startswith("dense."):
        raise TypeError("element strides require dense Encoding")
    dtype = encoding.removeprefix("dense.")
    if dtype == "bf16":
        return 2
    if dtype[:1] not in ("i", "u", "f") or not dtype[1:].isdecimal():
        raise TypeError(f"unsupported dense storage dtype: {dtype}")
    return max(1, (int(dtype[1:]) + 7) // 8)


def contiguous_strides(shape: tuple) -> tuple:
    strides, running = [], 1
    for extent in reversed(shape):
        strides.append(running)
        running *= extent
    return tuple(reversed(strides))


@dataclass(frozen=True)
class ViewMetadata:
    encoding: str
    shape: tuple
    strides: tuple
    readonly: bool = False

    @classmethod
    def contiguous(cls, encoding: str, shape: tuple):
        return cls(encoding, shape, contiguous_strides(shape))

    def require_contiguous(self, name: str) -> None:
        from .signature import require

        if any(type(extent) is int and extent == 0 for extent in self.shape):
            return
        for extent, stride, expected in zip(self.shape, self.strides, contiguous_strides(self.shape)):
            if type(extent) is int and type(stride) is int and type(expected) is int:
                if extent > 1 and stride != expected:
                    raise ValueError(f"{name} requires contiguous Encoding storage")
            else:
                require((extent <= 1) | (stride == expected), f"{name} requires contiguous Encoding storage")


def _native_endian(view: memoryview) -> None:
    prefix = view.format[:1]
    if view.itemsize > 1 and ((prefix in (">", "!") and sys.byteorder == "little") or
                             (prefix == "<" and sys.byteorder == "big")):
        raise TypeError("non-native-endian buffers require an explicit conversion by the caller")


def _buffer_encoding(view: memoryview) -> str:
    _native_endian(view)
    spelling = view.format.lstrip("@=<>!")
    if spelling in ("l", "L") and view.itemsize in (4, 8):
        return "dense." + ("i" if spelling == "l" else "u") + str(view.itemsize * 8)
    if spelling in _DENSE_FORMATS:
        return "dense." + _DENSE_FORMATS[spelling]
    raise TypeError(f"buffer format {view.format!r} needs an explicit Encoding")


def view_metadata(value: object, *, allow_meta=False) -> ViewMetadata:
    if isinstance(value, Buffer):
        return value.metadata
    torch = torch_module(value)
    if torch is not None:
        if value.device.type not in (("cpu", "meta") if allow_meta else ("cpu",)) or value.layout != torch.strided:
            raise TypeError("Weft requires strided CPU Tensors")
        if value.is_conj() or value.is_neg():
            raise ValueError("Tensor conjugate/negative views require explicit materialization")
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in _TORCH_DTYPES:
            raise TypeError(f"unsupported Tensor storage dtype: {value.dtype}")
        return ViewMetadata("dense." + _TORCH_DTYPES[dtype], tuple(value.shape), tuple(value.stride()))
    view = memoryview(value)
    if view.suboffsets and any(offset >= 0 for offset in view.suboffsets):
        raise TypeError("indirect buffer storage is unsupported")
    if any(stride % view.itemsize for stride in view.strides):
        raise TypeError("buffer strides must be whole storage elements")
    return ViewMetadata(_buffer_encoding(view), tuple(view.shape),
                        tuple(stride // view.itemsize for stride in view.strides), view.readonly)


def storage_encoding(value: object) -> str:
    """JIT specialization needs storage type; geometry stays a runtime input."""
    if isinstance(value, Buffer):
        return value.encoding
    if torch_module(value) is not None:
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in _TORCH_DTYPES:
            raise TypeError(f"unsupported Tensor storage dtype: {value.dtype}")
        return "dense." + _TORCH_DTYPES[dtype]
    with memoryview(value) as view:
        return _buffer_encoding(view)


class Buffer:
    """Borrowed storage, retaining its owner without copying or repacking.

    Explicit logical shape/Encoding describes packed bytes. Dense arrays and CPU
    tensors infer their Encoding and strides. A kernel must explicitly declare
    strided View parameters before it can consume a non-contiguous dense buffer.
    """

    def __init__(self, data: object, *, shape: Sequence[int] | None = None,
                 encoding: str | None = None, strides: Sequence[int] | None = None) -> None:
        self._owner = data
        self._torch = torch_module(data)
        self._tensor_metadata = None
        self._view = None
        if self._torch is not None:
            inferred_shape, inferred_encoding = self._borrow_tensor(data)
        else:
            inferred_shape, inferred_encoding = self._borrow_buffer(data, encoding is None)
        if encoding is None:
            encoding = inferred_encoding
        if not isinstance(encoding, str) or not encoding:
            raise TypeError("buffer Encoding must be a nonempty layout identity")
        logical_shape = tuple(inferred_shape if shape is None else shape)
        if any(type(extent) is not int or extent < 0 for extent in logical_shape):
            raise ValueError("logical buffer extents must be non-negative integers")
        if not self._contiguous and (logical_shape != inferred_shape or encoding != inferred_encoding):
            raise ValueError("non-contiguous storage cannot be reinterpreted as another shape or Encoding")
        if strides is not None:
            if not self._contiguous:
                raise ValueError("explicit strides require a contiguous backing storage object")
            logical_strides = tuple(strides)
        elif self._contiguous:
            logical_strides = contiguous_strides(logical_shape)
        else:
            logical_strides = self._storage_strides
        if len(logical_strides) != len(logical_shape) or any(type(stride) is not int or stride < 0 for stride in logical_strides):
            raise ValueError("strides must be non-negative element offsets, one per logical dimension")
        self._metadata = ViewMetadata(encoding, logical_shape, logical_strides, self._readonly)
        if not encoding.startswith("dense."):
            if strides is not None or not self._contiguous:
                raise NotImplementedError("packed Encoding storage must be contiguous")
        elif self.byte_span() > self.nbytes:
            raise ValueError("logical View exceeds its backing storage span")

    @property
    def metadata(self) -> ViewMetadata:
        return self._metadata

    @property
    def owner(self):
        return self._owner

    @property
    def shape(self):
        return self._metadata.shape

    @property
    def strides(self):
        return self._metadata.strides

    @property
    def encoding(self):
        return self._metadata.encoding

    @property
    def readonly(self):
        return self._metadata.readonly

    @property
    def address(self):
        return self._address

    @property
    def nbytes(self):
        return self._nbytes

    def _borrow_tensor(self, tensor) -> tuple:
        if tensor.device.type != "cpu" or tensor.layout != self._torch.strided:
            raise TypeError("Weft's native CPU ABI requires a strided CPU Tensor")
        if tensor.is_conj() or tensor.is_neg():
            raise ValueError("Tensor conjugate/negative views require an explicit materialization by the caller")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in _TORCH_DTYPES:
            raise TypeError(f"unsupported Tensor storage dtype: {tensor.dtype}")
        self._storage = tensor.untyped_storage()
        snapshot = self._tensor_metadata = _torch_metadata(tensor, self._storage)
        self._address = snapshot[0]
        self._nbytes = snapshot[-1] - snapshot[3] * tensor.element_size()
        self._storage_strides = snapshot[2]
        self._contiguous = tensor.is_contiguous()
        self._readonly = False
        return snapshot[1], "dense." + _TORCH_DTYPES[dtype]

    def _borrow_buffer(self, data, infer_encoding: bool) -> tuple:
        view = memoryview(data)
        _native_endian(view)
        if view.suboffsets and any(offset >= 0 for offset in view.suboffsets):
            raise TypeError("indirect buffer storage is unsupported")
        if any(stride < 0 or stride % view.itemsize for stride in view.strides):
            raise ValueError("buffer strides must be non-negative whole storage elements")
        inferred = _buffer_encoding(view) if infer_encoding else ""
        storage = _PyBuffer()
        _get_buffer(view, ctypes.byref(storage), 0x1c)  # PyBUF_FORMAT | PyBUF_STRIDES
        self._release = weakref.finalize(self, _release_buffer, ctypes.byref(storage))
        self._storage = storage
        self._view = view
        self._address = storage.buf or 0
        self._nbytes = storage.len
        self._contiguous = view.c_contiguous
        self._storage_strides = tuple(stride // view.itemsize for stride in view.strides)
        if not view.c_contiguous:
            self._nbytes = 0 if not math.prod(view.shape) else view.itemsize + sum(
                (extent - 1) * stride for extent, stride in zip(view.shape, view.strides))
            if not infer_encoding:
                inferred = _buffer_encoding(view)
        self._readonly = bool(storage.readonly)
        return tuple(view.shape), inferred

    def byte_span(self) -> int:
        metadata = self._metadata
        if not math.prod(metadata.shape):
            return 0
        return element_bytes(metadata.encoding) * (1 + sum((extent - 1) * stride
                                                   for extent, stride in zip(metadata.shape, metadata.strides)))

    def require_nonoverlapping(self) -> None:
        metadata = self._metadata
        if not math.prod(metadata.shape):
            return
        occupied = 1
        for stride, extent in sorted((stride, extent) for extent, stride in zip(metadata.shape, metadata.strides) if extent > 1):
            if stride < occupied:
                raise ValueError("writable strided Views must not overlap themselves")
            occupied += (extent - 1) * stride

    def byte_regions(self, required: int):
        """Accessible byte runs only; holes in strided storage are never read."""
        if not required:
            return
        if not self.encoding.startswith("dense."):
            yield 0, required
            return
        itemsize = element_bytes(self.encoding)
        run, outer = itemsize, []
        for stride, extent in sorted((stride * itemsize, extent) for extent, stride in zip(self.shape, self.strides) if extent > 1):
            if stride <= run:
                run += (extent - 1) * stride
            else:
                outer.append((stride, extent))
        for coordinates in itertools.product(*(range(extent) for _, extent in outer)):
            yield sum(index * stride for index, (stride, _) in zip(coordinates, outer)), run

    def validate(self, *, writable: bool) -> None:
        if self._torch is None:
            return
        if _torch_metadata(self.owner) != self._tensor_metadata:
            raise ValueError("a prepared Tensor changed storage, shape, strides or dtype; prepare the call again")
        if self._torch.is_grad_enabled() and self.owner.requires_grad:
            raise NotImplementedError("raw Weft Tensor calls do not define autograd; use torch.no_grad() for inference")
        if writable and self.owner.is_inference() and not self._torch.is_inference_mode_enabled():
            raise RuntimeError("an inference Tensor can only be written inside torch.inference_mode()")

    def matches_owner(self) -> bool:
        """Whether an implicit borrow can be retained by an explicit rebind.

        Tensor storage is movable; a changed owner must be borrowed again.
        Buffer-protocol exports pin storage, but their owner's public shape or
        writability can change independently of the retained memoryview.
        """
        if self._torch is not None:
            return _torch_metadata(self.owner) == self._tensor_metadata
        with memoryview(self.owner) as current:
            original = self._view
            return (current.shape == original.shape and current.strides == original.strides
                    and current.format == original.format and current.itemsize == original.itemsize
                    and current.nbytes == original.nbytes and current.readonly == original.readonly)

    def mark_written(self) -> None:
        if self._torch is not None:
            self._torch.autograd.graph.increment_version(self.owner)


def empty_buffer(shape: tuple[int, ...], dtype: str, *, like: object):
    module = torch_module(like.owner if isinstance(like, Buffer) else like)
    if module is not None:
        reverse = {value: name for name, value in _TORCH_DTYPES.items()}
        if dtype not in reverse or not hasattr(module, reverse[dtype]):
            raise NotImplementedError(f"Torch allocation does not support {dtype}")
        prototype = like.owner if isinstance(like, Buffer) else like
        return module.empty(shape, dtype=getattr(module, reverse[dtype]), device=prototype.device)
    import numpy as np

    formats = {"i1": "bool", "i8": "int8", "u8": "uint8", "i16": "int16", "u16": "uint16",
               "i32": "int32", "u32": "uint32", "i64": "int64", "u64": "uint64",
               "f16": "float16", "f32": "float32", "f64": "float64"}
    if dtype not in formats:
        raise NotImplementedError(f"NumPy allocation does not support {dtype}")
    return np.empty(shape, dtype=formats[dtype])
