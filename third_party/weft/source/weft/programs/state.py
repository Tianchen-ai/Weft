"""Caller-owned streaming state with explicit reset, fork and failure semantics."""
from __future__ import annotations

from threading import Lock

from .contracts import allocate, clear, clone, copy_storage, integer, same_backend, tensor


class _Use:
    def __init__(self, state, tokens):
        self.state = state
        self.tokens = tokens

    def __enter__(self):
        state = self.state
        if not state._lock.acquire(blocking=False):
            raise RuntimeError("streaming state is already in use")
        if not state.valid:
            state._lock.release()
            raise RuntimeError("a previous invocation failed; explicitly reset or restore this state")
        if isinstance(state, KVCache) and state.position + self.tokens > state.capacity:
            state._lock.release()
            raise ValueError("cache has insufficient capacity for this append")
        return state.position

    def __exit__(self, kind, value, traceback):
        state = self.state
        if kind is None:
            state.position += self.tokens
        else:
            # Native writes are ordered, not transactional. Never present a
            # possibly partially updated state as an unconsumed valid prefix.
            state.valid = False
        state._lock.release()


class StreamingState:
    def __init__(self):
        self.position = 0
        self.valid = True
        self._lock = Lock()

    def use(self, tokens):
        integer(tokens, "tokens", minimum=0)
        return _Use(self, tokens)

    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot change or copy state during an invocation")

    def _reset(self, buffers):
        self._exclusive()
        try:
            self.valid = False
            for value in buffers:
                clear(value)
            self.position = 0
            self.valid = True
        finally:
            self._lock.release()

    def _restore(self, source, destinations, originals):
        if source is self:
            raise ValueError("restore requires a separate state snapshot")
        self._exclusive()
        try:
            source._exclusive()
            try:
                if not source.valid:
                    raise RuntimeError("cannot restore from invalid state")
                for destination, original in zip(destinations, originals):
                    same_backend(destination, original)
                    tensor(destination, "destination state", shape=tuple(original.shape), writable=True)
                self.valid = False
                for destination, original in zip(destinations, originals):
                    copy_storage(destination, original)
                self.position = source.position
                self.valid = True
            finally:
                source._lock.release()
        finally:
            self._lock.release()


class KVCache(StreamingState):
    """Keys and values [batch,capacity,kv_heads,head_dim], append-only prefix."""

    def __init__(self, keys, values, *, position=0):
        super().__init__()
        self.shape = tensor(keys, "keys", rank=4, writable=True)
        tensor(values, "values", shape=self.shape, writable=True)
        same_backend(keys, values)
        for name, extent in zip(("batch", "capacity", "kv_heads", "head_dim"), self.shape):
            integer(extent, name)
        integer(position, "position", minimum=0)
        if position > self.capacity:
            raise ValueError("cache position exceeds capacity")
        self.keys = keys
        self.values = values
        self.position = position

    @classmethod
    def allocate(cls, batch, capacity, kv_heads, head_dim, *, like):
        shape = tuple(integer(value, name) for value, name in zip(
            (batch, capacity, kv_heads, head_dim), ("batch", "capacity", "kv_heads", "head_dim")))
        return cls(allocate(shape, like=like, zero=True), allocate(shape, like=like, zero=True))

    @property
    def capacity(self):
        return self.shape[1]

    @property
    def nbytes(self):
        b, s, g, d = self.shape
        return 2 * b * s * g * d * 4

    def check(self, *, batch, tokens, kv_heads, head_dim, like):
        tensor(self.keys, "keys", shape=self.shape, writable=True)
        tensor(self.values, "values", shape=self.shape, writable=True)
        if (self.shape[0], self.shape[2], self.shape[3]) != (batch, kv_heads, head_dim):
            raise ValueError("cache dimensions do not match the model activation")
        integer(tokens, "tokens", minimum=0)
        if self.position + tokens > self.capacity:
            raise ValueError("cache has insufficient capacity for this append")
        same_backend(self.keys, self.values, like)
        if not self.valid:
            raise RuntimeError("cache is invalid after a failed invocation; reset it first")
        return self

    def reset(self):
        self._reset((self.keys, self.values))

    def restore(self, snapshot):
        if not isinstance(snapshot, KVCache) or snapshot.shape != self.shape:
            raise ValueError("KV snapshot needs the same batch, capacity, heads and head dimension")
        self._restore(snapshot, (self.keys, self.values), (snapshot.keys, snapshot.values))

    def truncate(self, position):
        """Discard a suffix; future appends overwrite it before it can be read."""
        integer(position, "position", minimum=0)
        self._exclusive()
        try:
            if not self.valid or position > self.position:
                raise ValueError("truncate requires a position within the valid populated prefix")
            self.position = position
        finally:
            self._lock.release()

    def fork(self):
        self._exclusive()
        try:
            if not self.valid:
                raise RuntimeError("cannot fork invalid cache state")
            return KVCache(clone(self.keys), clone(self.values), position=self.position)
        finally:
            self._lock.release()


class MambaState(StreamingState):
    """Convolution history [B,W,I] and selective SSM state [B,I,N]."""

    def __init__(self, history, ssm, *, position=0):
        super().__init__()
        b, w, i = tensor(history, "history", rank=3, writable=True)
        sb, si, n = tensor(ssm, "ssm", rank=3, writable=True)
        if (b, i) != (sb, si):
            raise ValueError("convolution and SSM state batch/channel dimensions differ")
        for name, extent in zip(("batch", "width", "inner", "state_size"), (b, w, i, n)):
            integer(extent, name)
        same_backend(history, ssm)
        self.history = history
        self.ssm = ssm
        self.shape = (b, w, i, n)
        self.position = integer(position, "position", minimum=0)

    @classmethod
    def allocate(cls, batch, width, inner, state_size, *, like):
        for name, extent in zip(("batch", "width", "inner", "state_size"), (batch, width, inner, state_size)):
            integer(extent, name)
        return cls(allocate((batch, width, inner), like=like, zero=True),
                   allocate((batch, inner, state_size), like=like, zero=True))

    @property
    def nbytes(self):
        b, w, i, n = self.shape
        return b * i * (w + n) * 4

    def check(self, *, batch, width, inner, state_size, like):
        if self.shape != (batch, width, inner, state_size):
            raise ValueError("Mamba state dimensions differ from this model invocation")
        tensor(self.history, "history", shape=(batch, width, inner), writable=True)
        tensor(self.ssm, "ssm", shape=(batch, inner, state_size), writable=True)
        same_backend(self.history, self.ssm, like)
        if not self.valid:
            raise RuntimeError("Mamba state is invalid after a failed invocation; reset it first")
        return self

    def reset(self):
        self._reset((self.history, self.ssm))

    def restore(self, snapshot):
        if not isinstance(snapshot, MambaState) or snapshot.shape != self.shape:
            raise ValueError("Mamba snapshot needs the same batch, convolution and state dimensions")
        self._restore(snapshot, (self.history, self.ssm), (snapshot.history, snapshot.ssm))

    def fork(self):
        self._exclusive()
        try:
            if not self.valid:
                raise RuntimeError("cannot fork invalid Mamba state")
            return MambaState(clone(self.history), clone(self.ssm), position=self.position)
        finally:
            self._lock.release()
