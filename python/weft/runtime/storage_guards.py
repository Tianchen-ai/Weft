from __future__ import annotations

from .buffer import _torch_metadata


class StorageGuards:
    """Precomputed Tensor checks for one synchronous invocation scope.

    A Py_buffer export pins non-Tensor storage and already owns immutable View
    metadata. Tensors can replace their storage and require live checks. Two
    uses share a check only when both their owner identity and the captured
    metadata agree; conflicting snapshots therefore cannot hide a stale use.
    """

    def __init__(self, buffers):
        checks = {}
        writes = []
        self._torch = None
        for buffer, writable in buffers:
            if buffer._torch is None:
                continue
            self._torch = buffer._torch
            owner, snapshot = buffer.owner, buffer._tensor_metadata
            key = (id(owner), snapshot)
            inference_write = writable and owner.is_inference()
            if key in checks:
                previous = checks[key]
                checks[key] = (owner, snapshot, inference_write or previous[2])
            else:
                checks[key] = (owner, snapshot, inference_write)
            if writable:
                # Preserve one mutation notification per declared writable use,
                # including sequential writes to the same Tensor.
                writes.append(owner)
        self._checks = tuple(checks.values())
        self._writes = tuple(writes)

    def validate(self):
        if not self._checks:
            return
        grad_enabled = self._torch.is_grad_enabled()
        inference_enabled = self._torch.is_inference_mode_enabled()
        for owner, snapshot, inference_write in self._checks:
            if _torch_metadata(owner) != snapshot:
                raise ValueError("a prepared Tensor changed storage, shape, strides or dtype; prepare the call again")
            if grad_enabled and owner.requires_grad:
                raise NotImplementedError("raw Weft Tensor calls do not define autograd; use torch.no_grad() for inference")
            if inference_write and not inference_enabled:
                raise RuntimeError("an inference Tensor can only be written inside torch.inference_mode()")

    def mark_written(self):
        if self._writes:
            increment = self._torch.autograd.graph.increment_version
            for tensor in self._writes:
                increment(tensor)
