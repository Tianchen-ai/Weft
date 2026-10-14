from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Mapping

from .invocation import PreparedCall
from .native_calls import NativeSequence
from .storage_guards import StorageGuards


class PreparedSequence:
    """An explicitly ordered group of already prepared, synchronous CPU calls.

    The installed C SDK executes the sequence without returning to Python
    between entries. Construction retains the borrowed storage and selected
    artifacts. Every invocation checks all Tensor bindings before any write,
    and the SDK checks the current thread's capabilities before dispatch.

    The caller owns state reset, sharing and concurrency. No kernel is fused,
    reordered, tuned or recompiled. The execution affinity must stay within
    the CPU set observed while constructing this sequence.
    """

    def __init__(self, *calls: PreparedCall):
        self._initialize(calls)

    def _initialize(self, calls, previous=None):
        if not calls or any(not isinstance(call, PreparedCall) for call in calls):
            raise TypeError("a sequence requires one or more PreparedCall objects")
        self._calls = tuple(calls)
        self._kernels = tuple(dict.fromkeys(call.kernel for call in calls))
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False
        self._closing = False
        self._native = None
        self._guards = StorageGuards((view.buffer, view.argument.writable)
                                    for call in calls for view in call._bound.binding.views)
        # Acquire every library's invocation lifetime before preparing its SDK
        # counterpart; a concurrent close cannot unload a partially built set.
        with self._invocation():
            # Establish that the context discovered below is a subset of every
            # originating Python target. Future SDK checks restrict execution
            # to that captured context, so this proof need not repeat per call.
            for kernel in self._kernels:
                kernel.target.check_execution(kernel.metadata["used_extensions"])
            self._native = NativeSequence(calls, previous=previous)

    @property
    def calls(self):
        return self._calls

    @property
    def outputs(self):
        """The declared write-only outputs of each call, in execution order."""
        return tuple(call.outputs for call in self._calls)

    @property
    def closed(self):
        with self._condition:
            return self._closed or self._closing

    @contextmanager
    def _invocation(self):
        with self._lifetime():
            self._guards.validate()
            yield

    @contextmanager
    def _lifetime(self):
        with self._condition:
            if self._closed or self._closing:
                raise RuntimeError("the prepared sequence has been closed")
            self._active += 1
        acquired = []
        try:
            for kernel in self._kernels:
                kernel._acquire()
                acquired.append(kernel)
            yield
        finally:
            for kernel in reversed(acquired):
                kernel._retire()
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def __call__(self):
        with self._invocation():
            self._native.invoke()
            self._guards.mark_written()
        return tuple(call._bound.result() for call in self._calls)

    def replace(self, index: int, call: PreparedCall):
        """Construct another sequence with one explicitly replaced binding."""
        if type(index) is not int or not 0 <= index < len(self._calls):
            raise IndexError("sequence call index is out of range")
        if not isinstance(call, PreparedCall):
            raise TypeError("sequence replacement must be a PreparedCall")
        calls = list(self._calls)
        calls[index] = call
        with self._lifetime():
            return self._derive(calls)

    def rebind(self, updates: Mapping[int, Mapping]):
        """Return a new sequence after explicit per-call argument updates.

        For example ``sequence.rebind({0: {"X": next_input}, 2: {"start": 7}})``
        replaces those arguments using PreparedCall.rebind. It does not guess
        dependencies, rename arguments across calls, or allocate new workspace.
        Unchanged calls share their SDK binding and selected library handles.
        """
        if not isinstance(updates, Mapping):
            raise TypeError("sequence updates must map call indices to argument mappings")
        for index, arguments in updates.items():
            if type(index) is not int or not 0 <= index < len(self._calls):
                raise IndexError("sequence call index is out of range")
            if not isinstance(arguments, Mapping) or any(not isinstance(name, str) for name in arguments):
                raise TypeError("each sequence update must map argument names to values")
        with self._lifetime():
            calls = list(self._calls)
            for index, arguments in updates.items():
                calls[index] = calls[index].rebind(**arguments)
            return self._derive(calls)

    def _derive(self, calls):
        result = type(self).__new__(type(self))
        result._initialize(calls, previous=self._native)
        return result

    def close(self):
        with self._condition:
            if self._closed:
                return
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.wait_for(lambda: self._active == 0)
            try:
                if self._native is not None:
                    self._native.close()
                self._native = None
            finally:
                self._closed = True
                self._condition.notify_all()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("the prepared sequence has been closed")
        return self

    def __exit__(self, *exception):
        self.close()
