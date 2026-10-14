"""Optional CPU nn.Module adapters using the same catalog and torch.library ABI.

    Library registration happens at construction, outside graph capture.
    Forward methods contain tensor metadata operations and Weft calls only.
    These modules are inference programs; no derivative is implicitly invented.
"""
from __future__ import annotations

import torch
from torch import nn
from types import MappingProxyType
from math import prod

from weft.torch import register

from .catalog import get
from .contracts import integer, positive
from .tuning import recipes, selection_records
from weft.runtime.autotune import AutotunedKernel
from .mamba_block import MambaWeights
from .moe_block import MoEWeights
from .transformer import AttentionWeights, SwiGLUWeights, TransformerWeights


class Library:
    """One explicit dispatcher namespace shared by all modules in an application."""

    def __init__(self, namespace, *, options=None, tuning=None, toolchain=None, cache=None):
        if not isinstance(namespace, str) or not namespace.isidentifier():
            raise ValueError("namespace must be a Python identifier")
        self.namespace = namespace
        self.options = MappingProxyType(dict(options or {}))
        for name in self.options:
            get(name)
        self.tuning = recipes(tuning, self.options)
        self.toolchain = toolchain
        self.cache = cache
        self._entries = {}
        self._closed = False
        self._loaded = None

    @classmethod
    def from_package(cls, namespace, package, *, target=None):
        """Register packaged native programs without a JIT or autotuning path."""
        from .deployment import LoadedOperators

        result = cls(namespace)
        result._loaded = LoadedOperators(package, target=target)
        return result

    def kernel(self, name, **bindings):
        if self._closed:
            raise RuntimeError("Torch program library is closed")
        entry = get(name)
        recipe = self.tuning.get(name)
        loaded = self._loaded.kernel(name, **bindings) if self._loaded is not None else None
        if loaded is not None:
            options = loaded.artifact.options
        elif recipe is not None:
            options = recipe.bind(name, **bindings)[0].options
        else:
            options = entry.options(options=self.options.get(name), **bindings)
        key = (name, tuple(sorted(options.meta.items())))
        if key not in self._entries:
            suffix = "".join(f"_{parameter.lower()}{value}" for parameter, value in key[1])
            if loaded is not None:
                kernel = loaded
            elif recipe is not None:
                kernel = recipe.create(name, bindings=bindings, toolchain=self.toolchain, cache=self.cache)
            else:
                kernel = entry.jit(options=options, toolchain=self.toolchain, cache=self.cache)
            self._entries[key] = register(f"{self.namespace}::{name}{suffix}", kernel)
        return self._entries[key]

    def materialize(self):
        if self._closed:
            raise RuntimeError("Torch program library is closed")
        if self._loaded is not None:
            return self
        if any(isinstance(entry.kernel, AutotunedKernel) for entry in self._entries.values()):
            raise ValueError("tuned Torch programs require workload arguments; use tune() or an ordinary module call")
        for entry in self._entries.values():
            entry.kernel.materialize()
        return self

    def tune(self, name, *, bindings=None, refresh=False, **arguments):
        operation = self.kernel(name, **({} if bindings is None else bindings))
        if not isinstance(operation.kernel, AutotunedKernel):
            raise ValueError(f"{name} has no tuning recipe")
        return operation.kernel.tune(refresh=refresh, **arguments)

    @property
    def tuning_results(self):
        return selection_records({key: entry.kernel for key, entry in self._entries.items()})

    def save(self, directory, *, selections=None):
        """Freeze measured choices as the ordinary Weft program package."""
        from .deployment import save_program_kernels, save_operators

        if self._closed:
            raise RuntimeError("Torch program library is closed")
        if self._loaded is not None:
            return save_operators(self._loaded, directory, selections=selections)
        return save_program_kernels({key: entry.kernel for key, entry in self._entries.items()},
                                    directory, selections=selections)

    def close(self):
        self._closed = True
        if self._loaded is not None:
            self._loaded.close()
        else:
            for entry in self._entries.values():
                entry.kernel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def parameter(module, name, value):
    if value is None:
        module.register_buffer(name, None)
        return
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
        raise TypeError(f"{name} requires a CPU float32 Tensor")
    if value.requires_grad:
        raise NotImplementedError("model program modules require detached inference parameters")
    module.register_buffer(name, value)


class _RMSNorm(nn.Module):
    def __init__(self, weight, *, library, epsilon=1e-5):
        super().__init__()
        if weight.ndim != 1 or weight.numel() == 0:
            raise ValueError("RMSNorm needs a nonempty scale vector")
        parameter(self, "weight", weight)
        self.epsilon = positive(epsilon, "epsilon")
        bindings = {"N": weight.numel()} if get(self.program).required == ("N",) else {}
        self.operation = library.kernel(self.program, **bindings).op

    def forward(self, x):
        return self.operation(x.view(-1, x.shape[-1]), self.weight, self.epsilon).view(x.shape)


class RMSNorm(_RMSNorm):
    program = "rms_norm"


class RMSNormRow(_RMSNorm):
    program = "rms_norm_row"


class RMSNormTile(_RMSNorm):
    program = "rms_norm_tile"


class _LayerNorm(nn.Module):
    def __init__(self, weight, bias, *, library, epsilon=1e-5):
        super().__init__()
        if weight.ndim != 1 or weight.numel() == 0 or bias.shape != weight.shape:
            raise ValueError("LayerNorm needs equal nonempty scale and bias vectors")
        parameter(self, "weight", weight)
        parameter(self, "bias", bias)
        self.epsilon = positive(epsilon, "epsilon")
        bindings = {"N": weight.numel()} if get(self.program).required == ("N",) else {}
        self.operation = library.kernel(self.program, **bindings).op

    def forward(self, x):
        return self.operation(x.view(-1, x.shape[-1]), self.weight, self.bias, self.epsilon).view(x.shape)


class LayerNorm(_LayerNorm):
    program = "layer_norm"


class LayerNormRow(_LayerNorm):
    program = "layer_norm_row"


class LayerNormTile(_LayerNorm):
    program = "layer_norm_tile"


class SoftmaxRow(nn.Module):
    def __init__(self, width, *, library):
        super().__init__()
        self.width = integer(width, "row width")
        self.operation = library.kernel("softmax_row", N=self.width).op

    def forward(self, x):
        return self.operation(x.view(-1, x.shape[-1])).view(x.shape)


class Linear(nn.Module):
    plain = "linear"
    biased = "linear_bias"

    def __init__(self, weight, bias=None, *, library):
        super().__init__()
        if weight.ndim != 2 or min(weight.shape) <= 0:
            raise ValueError("Linear needs a nonempty [out,in] weight")
        if bias is not None and tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("Linear bias must match output features")
        parameter(self, "weight", weight)
        parameter(self, "bias", bias)
        self.operation = library.kernel(self.plain if bias is None else self.biased).op

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        if self.bias is None:
            y = self.operation(flat, self.weight)
        else:
            y = self.operation(flat, self.weight, self.bias)
        return y.view(*x.shape[:-1], self.weight.shape[0])


class LinearSiLU(Linear):
    """One author projection/activation program with the standard inference ABI."""

    plain = "linear_silu"
    biased = "linear_bias_silu"


class Matmul(nn.Module):
    """Inference projection with a caller-owned [in,out] weight layout."""

    def __init__(self, weight, bias=None, *, library):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("Matmul requires a [in,out] weight")
        if bias is not None and tuple(bias.shape) != (weight.shape[1],):
            raise ValueError("Matmul bias must match output features")
        parameter(self, "weight", weight)
        parameter(self, "bias", bias)
        self.operation = library.kernel("matmul" if bias is None else "matmul_bias").op

    def forward(self, x):
        if x.ndim == 0:
            raise ValueError("Matmul activation requires at least one dimension")
        flat = x.view(prod(x.shape[:-1]), x.shape[-1])
        if self.bias is None:
            y = self.operation(flat, self.weight)
        else:
            y = self.operation(flat, self.weight, self.bias)
        return y.view(*x.shape[:-1], self.weight.shape[1])


class BatchedMatmul(nn.Module):
    """Pairwise products of [batch,m,k] and [batch,k,n] tensors."""

    def __init__(self, *, library):
        super().__init__()
        self.operation = library.kernel("batched_matmul").op

    def forward(self, x, weight):
        return self.operation(x, weight)


class Where(nn.Module):
    """Equal-shaped Boolean/f32 selection through the ordinary native program."""

    def __init__(self, *, library):
        super().__init__()
        self.operation = library.kernel("where").op

    def forward(self, condition, x, other):
        torch._check(x.dim() >= 1)
        torch._check(x.shape[-1] > 0)
        torch._check(condition.dim() == x.dim())
        torch._check(other.dim() == x.dim())
        for axis in range(x.dim()):
            torch._check(condition.shape[axis] == x.shape[axis])
            torch._check(other.shape[axis] == x.shape[axis])
        width = x.shape[-1]
        return self.operation(condition.view(-1, width), x.view(-1, width), other.view(-1, width)).view(x.shape)


class SwiGLU(nn.Module):
    def __init__(self, weights: SwiGLUWeights, *, library):
        super().__init__()
        weights.dimensions()
        self.gate = Linear(weights.gate, library=library)
        self.up = Linear(weights.up, library=library)
        self.down = Linear(weights.down, library=library)
        self.operation = library.kernel("swiglu").op

    def forward(self, x):
        gate, up = self.gate(x), self.up(x)
        activated = self.operation(gate.view(-1, gate.shape[-1]), up.view(-1, up.shape[-1])).view(gate.shape)
        return self.down(activated)


class MoE(nn.Module):
    """Graph-visible complete MoE with stable routing and no dropped tokens."""

    def __init__(self, weights: MoEWeights, top_k, *, library):
        super().__init__()
        experts, _, self.model = weights.dimensions()
        self.experts = experts
        self.top_k = integer(top_k, "top_k")
        if top_k > experts:
            raise ValueError("top_k exceeds the number of experts")
        self.router_projection = Linear(weights.router, library=library)
        parameter(self, "gate", weights.gate)
        parameter(self, "up", weights.up)
        parameter(self, "down", weights.down)
        self.route = library.kernel("router").op
        self.group = library.kernel("group_tokens").op
        self.project = library.kernel("expert_linear").op
        self.activate = library.kernel("swiglu").op
        self.combine = library.kernel("combine").op

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        count = flat.shape[0]
        torch._check(count * self.top_k <= 2147483647)
        logits = self.router_projection(flat)
        indices = torch.empty((count, self.top_k), dtype=torch.int32, device=x.device)
        weights = torch.empty((count, self.top_k), dtype=torch.float32, device=x.device)
        offsets = torch.empty((self.experts + 1,), dtype=torch.int32, device=x.device)
        cursor = torch.empty((self.experts,), dtype=torch.int32, device=x.device)
        self.route(logits, indices, weights)
        positions, grouped = self.group(flat, indices, offsets, cursor, P=count * self.top_k)
        gate = self.project(grouped, self.gate, offsets)
        up = self.project(grouped, self.up, offsets)
        activated = self.activate(gate, up)
        projected = self.project(activated, self.down, offsets)
        return self.combine(projected, positions, weights).view(x.shape)


class Mamba1(nn.Module):
    """Mamba-1 with explicit mutable history/SSM Tensor inputs for graph capture."""

    def __init__(self, weights: MambaWeights, *, library):
        super().__init__()
        self.model, self.inner, self.state_size, self.rank, self.width = weights.dimensions()
        self.input_projection = Linear(weights.in_proj, weights.in_bias, library=library)
        self.parameter_projection = Linear(weights.x_proj, library=library)
        self.delta_projection = Linear(weights.dt_proj, library=library)
        self.output_projection = Linear(weights.out_proj, weights.out_bias, library=library)
        for name in ("conv", "conv_bias", "dt_bias", "a_log", "skip"):
            parameter(self, name, getattr(weights, name))
        self.convolution = library.kernel("causal_conv1d").op
        self.scan = library.kernel("selective_scan", N=self.state_size).op

    def new_state(self, batch):
        integer(batch, "batch")
        return (torch.zeros((batch, self.width, self.inner), dtype=torch.float32),
                torch.zeros((batch, self.inner, self.state_size), dtype=torch.float32))

    def forward(self, x, history, state):
        projected = self.input_projection(x)
        conv = self.convolution(projected[:, :, :self.inner], self.conv, self.conv_bias, history)
        parameters = self.parameter_projection(conv)
        delta = self.delta_projection(parameters[:, :, :self.rank])
        bcoef = parameters[:, :, self.rank:self.rank + self.state_size]
        ccoef = parameters[:, :, self.rank + self.state_size:]
        scanned = self.scan(conv, delta, bcoef, ccoef, self.a_log, self.skip,
                            projected[:, :, self.inner:], self.dt_bias, state)
        return self.output_projection(scanned)


class SelfAttention(nn.Module):
    """Explicit cache inputs and start position; forward mutates the KV prefix."""

    def __init__(self, weights: AttentionWeights, heads, kv_heads, *, library, rope_layout="split"):
        super().__init__()
        self.model, self.head_dim = weights.dimensions(heads, kv_heads)
        self.heads, self.kv_heads = heads, kv_heads
        if rope_layout not in ("split", "interleaved", None):
            raise ValueError("rope_layout must be split, interleaved or None")
        if rope_layout is not None and self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        self.rope_layout = rope_layout
        self.query = Linear(weights.query, library=library)
        self.key = Linear(weights.key, library=library)
        self.value = Linear(weights.value, library=library)
        self.output_projection = Linear(weights.output, library=library)
        self.rotary = None if rope_layout is None else library.kernel("rope_" + rope_layout).out_op
        self.update = library.kernel("kv_update").out_op
        self.attention = library.kernel("causal_attention", D=self.head_dim).op

    def new_cache(self, batch, capacity):
        integer(batch, "batch")
        integer(capacity, "capacity")
        shape = (batch, capacity, self.kv_heads, self.head_dim)
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)

    def _rope(self, x, cosine, sine):
        b, t, h, d = x.shape
        # The rotary source accepts [tokens,heads,...]; batch heads can be
        # traversed independently without expanding or copying the tables.
        result = torch.empty_like(x)
        for batch in range(b):
            if self.rope_layout == "split":
                internal = x[batch].view(t, h, 2, d // 2)
            else:
                internal = x[batch].view(t, h, d // 2, 2)
            self.rotary(internal, cosine, sine, output_0=result[batch].view(internal.shape))
        return result

    def forward(self, x, keys, values, start: int, cosine=None, sine=None):
        b, t, _ = x.shape
        torch._check(start >= 0)
        torch._check(start + t <= keys.shape[1])
        q = self.query(x).view(b, t, self.heads, self.head_dim)
        k = self.key(x).view(b, t, self.kv_heads, self.head_dim)
        v = self.value(x).view(b, t, self.kv_heads, self.head_dim)
        if self.rotary is not None:
            q = self._rope(q, cosine, sine)
            k = self._rope(k, cosine, sine)
        self.update(k, v, start, S=keys.shape[1], output_0=keys, output_1=values)
        y = self.attention(q, keys, values, start, self.head_dim ** -0.5)
        return self.output_projection(y.view(b, t, self.heads * self.head_dim))


class TransformerBlock(nn.Module):
    def __init__(self, weights: TransformerWeights, heads, kv_heads, *, library,
                 epsilon=1e-5, rope_layout="split"):
        super().__init__()
        model, _ = weights.attention.dimensions(heads, kv_heads)
        weights.check(model)
        self.attention_norm = RMSNorm(weights.attention_norm, library=library, epsilon=epsilon)
        self.attention = SelfAttention(weights.attention, heads, kv_heads, library=library, rope_layout=rope_layout)
        self.ffn_norm = RMSNorm(weights.ffn_norm, library=library, epsilon=epsilon)
        self.ffn = SwiGLU(weights.ffn, library=library)
        self.add = library.kernel("add").op

    def new_cache(self, batch, capacity):
        return self.attention.new_cache(batch, capacity)

    def forward(self, x, keys, values, start: int, cosine=None, sine=None):
        attended = self.attention(self.attention_norm(x), keys, values, start, cosine, sine)
        residual = self.add(x.view(-1, x.shape[-1]), attended.view(-1, x.shape[-1])).view(x.shape)
        feedforward = self.ffn(self.ffn_norm(residual))
        return self.add(residual.view(-1, x.shape[-1]), feedforward.view(-1, x.shape[-1])).view(x.shape)


__all__ = ["Library", "RMSNorm", "LayerNorm", "Linear", "Where", "SwiGLU", "SelfAttention", "TransformerBlock", "MoE", "Mamba1"]
