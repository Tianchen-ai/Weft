"""Owned ctypes bindings to the installed SDK's prepared-call execution API."""
from __future__ import annotations

import ctypes
import os
import threading
import weakref

from .driver import CompilationError
from .package import sdk_prefix


class _Error(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("message", ctypes.c_char * 1024)]


class _View(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("byte_length", ctypes.c_size_t),
                ("encoding", ctypes.c_char_p), ("rank", ctypes.c_size_t),
                ("shape", ctypes.POINTER(ctypes.c_size_t)),
                ("strides", ctypes.POINTER(ctypes.c_size_t)), ("readonly", ctypes.c_bool)]


class _Scalar(ctypes.Union):
    _fields_ = [("i1", ctypes.c_bool), ("i8", ctypes.c_int8), ("u8", ctypes.c_uint8),
                ("i16", ctypes.c_int16), ("u16", ctypes.c_uint16),
                ("i32", ctypes.c_int32), ("u32", ctypes.c_uint32),
                ("i64", ctypes.c_int64), ("u64", ctypes.c_uint64),
                ("f32", ctypes.c_float), ("f64", ctypes.c_double), ("index", ctypes.c_size_t)]


class _Value(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("dtype", ctypes.c_int), ("view", _View), ("scalar", _Scalar)]


class _Dimension(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("value", ctypes.c_size_t)]


_DTYPES = {name: index for index, name in enumerate(
    ("none", "i1", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64",
     "f16", "bf16", "f32", "f64", "index"))}


class NativeCallError(CompilationError):
    """An SDK error retaining its native status for callers and diagnostics."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _check(status, error):
    if status:
        message = bytes(error.message).decode("utf-8")
        if status == 1:
            raise ValueError(message)
        if status == 5:
            raise NotImplementedError(message)
        raise NativeCallError(status, message)


class _Owner:
    def __init__(self, handle, release):
        self.handle = handle
        self._release = weakref.finalize(self, release, handle)

    def close(self):
        self._release()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class _SDK:
    def __init__(self):
        self.path = sdk_prefix() / "lib/libweft_runtime.so"
        self.library = ctypes.CDLL(str(self.path), mode=os.RTLD_LOCAL)
        handle = ctypes.c_void_p
        result = ctypes.POINTER(handle)
        error = ctypes.POINTER(_Error)
        signatures = {
            "context_create": [result, error],
            "context_release": [handle],
            "kernel_open": [handle, ctypes.c_char_p, result, error],
            "kernel_release": [handle],
            "call_prepare": [handle, ctypes.POINTER(_Value), ctypes.c_size_t,
                             ctypes.POINTER(_Dimension), ctypes.c_size_t, result, error],
            "call_release": [handle],
            "sequence_create": [ctypes.POINTER(handle), ctypes.c_size_t, result, error],
            "sequence_invoke": [handle, error],
            "sequence_release": [handle],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, "weft_" + name)
            function.argtypes = arguments
            function.restype = None if name.endswith("_release") else ctypes.c_int
            setattr(self, name, function)

    def _create(self, kind, function, *arguments):
        result, error = ctypes.c_void_p(), _Error()
        _check(function(*arguments, ctypes.byref(result), ctypes.byref(error)), error)
        return _Owner(result, getattr(self, kind + "_release"))

    def context(self):
        return self._create("context", self.context_create)

    def kernel(self, context, artifact):
        return self._create("kernel", self.kernel_open, context.handle, os.fsencode(artifact.path))

    def call(self, kernel, prepared):
        """Copy typed descriptors into the SDK; data owners stay in PreparedCall."""
        interface = prepared.kernel.interface
        bound = prepared._bound
        values = (_Value * len(interface.arguments))()
        retained = []
        views = {view.argument.name: view for view in bound.binding.views}
        for index, parameter in enumerate(interface.arguments):
            value = values[index]
            if parameter.kind == "scalar":
                value.kind = 2
                value.dtype = _DTYPES[parameter.dtype]
                # The Python ABI already checked range and representability.
                # Copy its exact converted value, not the original Python object.
                setattr(value.scalar, parameter.dtype, bound.arguments[index].value)
                continue
            value.kind = 1
            buffer = views[parameter.name].buffer
            shape = (ctypes.c_size_t * len(buffer.shape))(*buffer.shape)
            strides = ((ctypes.c_size_t * len(buffer.strides))(*buffer.strides)
                       if parameter.strided else None)
            encoding = buffer.encoding.encode("utf-8")
            retained.extend((shape, strides, encoding))
            value.view = _View(buffer.address, buffer.nbytes, encoding, len(buffer.shape),
                               shape, strides, buffer.readonly)
        dimensions = (_Dimension * len(interface.shape_parameters))()
        for index, name in enumerate(interface.shape_parameters):
            spelling = name.encode("utf-8")
            retained.append(spelling)
            dimensions[index] = _Dimension(spelling, bound.dimensions[name])
        return self._create("call", self.call_prepare, kernel.handle, values, len(values),
                            dimensions, len(dimensions))

    def sequence(self, calls):
        handles = (ctypes.c_void_p * len(calls))(*(call.handle for call in calls))
        return self._create("sequence", self.sequence_create, handles, len(calls))


_sdk = None
_sdk_lock = threading.Lock()


def sdk():
    global _sdk
    with _sdk_lock:
        if _sdk is None:
            _sdk = _SDK()
        return _sdk


class NativeSequence:
    """One SDK sequence; all C descriptors are copied at construction."""

    def __init__(self, calls, *, previous=None):
        self._sdk = sdk()
        self._prepared = tuple(calls)
        # These Python owners are shared only by immutable sequences. Dropping
        # one sequence must not release a handle needed to derive another one.
        used = {call.kernel for call in calls}
        self._kernels = ({} if previous is None else
                         {kernel: owner for kernel, owner in previous._kernels.items() if kernel in used})
        self._calls = []
        self._owner = None
        context = None
        try:
            for index, call in enumerate(calls):
                if previous is not None and previous._prepared[index] is call:
                    self._calls.append(previous._calls[index])
                    continue
                kernel = self._kernels.get(call.kernel)
                if kernel is None:
                    if context is None:
                        context = self._sdk.context()
                    kernel = self._sdk.kernel(context, call.kernel.artifact)
                    self._kernels[call.kernel] = kernel
                self._calls.append(self._sdk.call(kernel, call))
            self._owner = self._sdk.sequence(self._calls)
        except BaseException:
            self._calls.clear()
            self._kernels.clear()
            raise
        finally:
            if context is not None:
                context.close()

    def invoke(self):
        error = _Error()
        _check(self._sdk.sequence_invoke(self._owner.handle, ctypes.byref(error)), error)

    def close(self):
        if self._owner is not None:
            self._owner.close()
            self._owner = None
        self._calls.clear()
        self._kernels.clear()
        self._prepared = ()
