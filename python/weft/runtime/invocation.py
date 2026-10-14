from __future__ import annotations

import ctypes
import os
import threading
import time
import weakref
from _ctypes import dlclose
from contextlib import contextmanager

from .interface import BoundCall
from .target import NativeTarget


class PreparedCall:
    """A validated ABI binding retaining the kernel and all borrowed storage."""

    def __init__(self, kernel: CompiledKernel, bound: BoundCall) -> None:
        self.kernel = kernel
        self._bound = bound

    @property
    def outputs(self) -> tuple:
        return self._bound.outputs

    def __call__(self):
        self.kernel._invoke(self._bound)
        return self._bound.result()

    def rebind(self, **updates) -> PreparedCall:
        """Recheck changed buffers/scalars instead of retaining stale native addresses."""
        if self.kernel.closed:
            raise RuntimeError("the compiled kernel has been closed")
        return PreparedCall(self.kernel, self.kernel.interface.rebind_call(self._bound, updates))

    def time_ns(self) -> int:
        """Time the native entry including ctypes dispatch, excluding validation."""
        with self.kernel._invocation(self._bound) as function:
            begin = time.perf_counter_ns()
            function(*self._bound.arguments)
            elapsed = time.perf_counter_ns() - begin
            self._bound.mark_written()
        return elapsed


class CompiledKernel:
    def __init__(self, artifact, target: NativeTarget, *, toolchain=None) -> None:
        artifact.check_isa_contract(toolchain=toolchain)
        self.artifact = artifact
        self.target = target
        self.options = artifact.options
        self.metadata = artifact.metadata
        self.interface = artifact.interface
        self.library_path = artifact.library_path
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._closing = False
        target.require(artifact.target, self.metadata["used_extensions"], self.metadata["isa_extensions"])
        ctypes_arguments = self.interface.ctypes
        library = ctypes.CDLL(str(self.library_path), mode=os.RTLD_LOCAL)
        try:
            function = getattr(library, self.metadata["symbol"])
            function.argtypes = ctypes_arguments
            function.restype = None
        except BaseException:
            dlclose(library._handle)
            raise
        self._library = library
        self._function = function
        self._release = weakref.finalize(self, dlclose, library._handle)

    @property
    def source(self) -> str:
        return self.artifact.source

    @property
    def riscv_ir(self) -> str:
        return self.artifact.riscv_ir

    @property
    def compilation(self):
        return self.artifact.compilation

    @property
    def physical_resources(self):
        return self.artifact.physical_resources

    @property
    def physical_operations(self):
        return self.artifact.physical_operations

    @property
    def cache_hit(self) -> bool | None:
        return self.artifact.cache_hit

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed or self._closing

    def prepare(self, *args, **kwargs) -> PreparedCall:
        if self.closed:
            raise RuntimeError("the compiled kernel has been closed")
        return PreparedCall(self, self.interface.bind(args, kwargs))

    def _prepare_call(self, call) -> PreparedCall:
        if self.closed:
            raise RuntimeError("the compiled kernel has been closed")
        return PreparedCall(self, self.interface.bind_call(call))

    def __call__(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)()

    def run(self, *args, **kwargs):
        """Allocate omitted write-only dense outputs; workspace/state stays explicit."""
        return self.prepare_run(*args, **kwargs)()

    def prepare_run(self, *args, **kwargs) -> PreparedCall:
        if self.closed:
            raise RuntimeError("the compiled kernel has been closed")
        bound = self.interface.bind(args, kwargs, allocate_outputs=True)
        return PreparedCall(self, bound)

    def _invoke(self, bound: BoundCall) -> None:
        with self._invocation(bound) as function:
            function(*bound.arguments)
            bound.mark_written()

    @contextmanager
    def _invocation(self, bound: BoundCall):
        function = self._acquire()
        try:
            self.target.check_execution(self.metadata["used_extensions"])
            bound.validate()
            yield function
        finally:
            self._retire()

    def _acquire(self):
        with self._condition:
            if self._closed or self._closing:
                raise RuntimeError("the compiled kernel has been closed")
            self._active += 1
            return self._function

    def _retire(self):
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def save(self, destination):
        return self.artifact.save(destination)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.wait_for(lambda: self._active == 0)
            self._function = None
            self._library = None
            try:
                self._release()
            finally:
                self._closed = True
                self._condition.notify_all()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("the compiled kernel has been closed")
        return self

    def __exit__(self, *exception):
        self.close()
