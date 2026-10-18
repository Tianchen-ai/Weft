"""Synchronous native calls with the compiler-emitted ABI and SDK guards."""
import ctypes
import json
import os
from pathlib import Path
import weakref
from functools import cache
from _ctypes import dlclose

from .abi import Argument
from .signature import CallSignature


class Error(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("message", ctypes.c_char * 1024)]


@cache
def _runtime_library(path):
    return ctypes.CDLL(path, mode=os.RTLD_LOCAL)


def _release_kernel(release, handle, library):
    release(handle)
    dlclose(library)


class NativeSignature(CallSignature):
    parameter_type = Argument


def manifest(metadata):
    options = {name: getattr(metadata, name) for name in
               ("matrix_extension", "max_widening_combine_groups", "partial_combine_policy",
                "record_axis_policy", "local_tile_strategy")}
    options["meta"] = metadata.source_bindings
    options["require_extensions"] = metadata.required_extensions
    return {"kind": "weft-callable-artifact", "producer": "triton", "native_library": True,
            "dispatch": "weft_kernel_dispatch", "kernel": metadata.kernel_abi,
            "options": options, "target": vars(metadata.target)}


class NativeModule:
    def __init__(self, library_path, metadata):
        runtime_path = os.environ.get("TRITON_WEFT_RUNTIME")
        if runtime_path is None:
            runtime_path = str(Path(__file__).parents[2] / "_C/weft/lib/libweft_runtime.so")
        self.sdk = _runtime_library(runtime_path)
        pointer = ctypes.c_void_p
        error_ptr = ctypes.POINTER(Error)
        for name, args, result in (
            ("context_create", [ctypes.POINTER(pointer), error_ptr], ctypes.c_int),
            ("context_release", [pointer], None),
            ("kernel_open_native", [pointer, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(pointer), error_ptr], ctypes.c_int),
            ("kernel_release", [pointer], None),
            ("kernel_get_entry", [pointer, ctypes.POINTER(pointer), error_ptr], ctypes.c_int),
        ):
            fn = getattr(self.sdk, "weft_" + name)
            fn.argtypes, fn.restype = args, result
        context, handle, entry = pointer(), pointer(), pointer()
        error = Error()
        self.check(self.sdk.weft_context_create(ctypes.byref(context), ctypes.byref(error)), error)
        try:
            self.check(self.sdk.weft_kernel_open_native(context, os.fsencode(library_path),
                       json.dumps(manifest(metadata)).encode(), ctypes.byref(handle), ctypes.byref(error)), error)
        finally:
            self.sdk.weft_context_release(context)
        self._release = weakref.finalize(self, self.sdk.weft_kernel_release, handle)
        self.handle = handle
        self.check(self.sdk.weft_kernel_get_entry(handle, ctypes.byref(entry), ctypes.byref(error)), error)
        self.entry = entry
        self.library = ctypes.CDLL(str(library_path), mode=os.RTLD_LOCAL)
        self._release.detach()
        self._release = weakref.finalize(self, _release_kernel, self.sdk.weft_kernel_release,
                                        handle, self.library._handle)
        self.grid = self.library.triton_weft_grid
        self.grid.argtypes = [pointer, ctypes.POINTER(pointer), ctypes.c_size_t, ctypes.c_size_t,
                              ctypes.c_size_t, ctypes.c_int, pointer, pointer, pointer, ctypes.c_size_t]
        self.grid.restype = ctypes.c_int
        self.check_worker = ctypes.cast(self.sdk.weft_kernel_check_worker, pointer)

    @staticmethod
    def check(status, error):
        if status:
            raise RuntimeError(bytes(error.message).decode())

    def close(self):
        self._release()


def NativeLauncher(module, metadata):
    from triton._C.libtriton import weft
    functions = {name: ctypes.cast(getattr(module.sdk, "weft_call_" + name), ctypes.c_void_p).value
                 for name in ("prepare", "arguments", "release")}
    functions.update(grid=ctypes.cast(module.grid, ctypes.c_void_p).value,
                     check_worker=module.check_worker.value,
                     kernel=module.handle.value, entry=module.entry.value)
    return weft.NativeLauncher(module, metadata, functions)
