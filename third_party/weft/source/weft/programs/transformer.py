"""Pre-norm dense Transformer blocks, grouped-query attention and SwiGLU."""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import BufferSpec, check_invocation, integer, output, positive, reshape, same_backend, tensor, use_workspace, Workspace
from .operators import Operators
from .state import KVCache


@dataclass(frozen=True)
class AttentionWeights:
    query: object
    key: object
    value: object
    output: object

    def dimensions(self, heads, kv_heads):
        integer(heads, "heads")
        integer(kv_heads, "kv_heads")
        q, model = tensor(self.query, "query weight", rank=2)
        if heads % kv_heads or q % heads or q == 0 or model == 0:
            raise ValueError("attention needs H divisible by G and query features divisible by H")
        d = q // heads
        tensor(self.key, "key weight", shape=(kv_heads * d, model))
        tensor(self.value, "value weight", shape=(kv_heads * d, model))
        tensor(self.output, "attention output weight", shape=(model, heads * d))
        same_backend(self.query, self.key, self.value, self.output)
        return model, d


@dataclass(frozen=True)
class SwiGLUWeights:
    gate: object
    up: object
    down: object

    def dimensions(self):
        hidden, model = tensor(self.gate, "gate weight", rank=2)
        integer(hidden, "hidden")
        integer(model, "model")
        tensor(self.up, "up weight", shape=(hidden, model))
        tensor(self.down, "down weight", shape=(model, hidden))
        same_backend(self.gate, self.up, self.down)
        return model, hidden


class SwiGLU:
    def __init__(self, weights: SwiGLUWeights, *, operators=None):
        if not isinstance(weights, SwiGLUWeights):
            raise TypeError("weights must be SwiGLUWeights")
        self.weights = weights
        self.dimensions = weights.dimensions()
        self.operators = Operators() if operators is None else operators
        self._owns_operators = operators is None

    def workspace_specs(self, batch, tokens):
        integer(batch, "batch")
        integer(tokens, "tokens", minimum=0)
        h = self.dimensions[1]
        return tuple(BufferSpec(name, (batch, tokens, h)) for name in ("gate", "up", "activated"))

    def workspace(self, batch, tokens, *, like=None):
        return Workspace(self.workspace_specs(batch, tokens), like=self.weights.gate if like is None else like)

    @property
    def required_programs(self):
        return {"linear": {}, "swiglu": {}}

    def materialize(self):
        self.operators.materialize(self.required_programs)
        return self

    def _run(self, x, y, work):
        op, w = self.operators, self.weights
        for b in range(x.shape[0]):
            op.linear(x[b], w.gate, out=work["gate"][b])
            op.linear(x[b], w.up, out=work["up"][b])
        op.swiglu(work["gate"], work["up"], out=work["activated"])
        for b in range(x.shape[0]):
            op.linear(work["activated"][b], w.down, out=y[b])

    def __call__(self, x, *, out=None, workspace=None):
        b, t, d = tensor(x, "x", rank=3)
        if d != self.dimensions[0] or self.weights.dimensions() != self.dimensions:
            raise ValueError("SwiGLU input or parameters changed dimensions")
        same_backend(x, self.weights.gate)
        y = output(out, (b, t, d), like=x)
        work = use_workspace(workspace, self.workspace_specs(b, t), like=x)
        check_invocation(x, y, self.weights, work)
        with work:
            self._run(x, y, work)
        return y

    def close(self):
        if self._owns_operators:
            self.operators.close()


class SelfAttention:
    """Projection, optional RoPE, KV append, online attention and output projection."""

    def __init__(self, weights: AttentionWeights, heads, kv_heads, *, rope_layout="split", operators=None):
        if not isinstance(weights, AttentionWeights):
            raise TypeError("weights must be AttentionWeights")
        self.weights = weights
        self.heads = integer(heads, "heads")
        self.kv_heads = integer(kv_heads, "kv_heads")
        self.dimensions = weights.dimensions(heads, kv_heads)
        if rope_layout not in ("split", "interleaved", None):
            raise ValueError("rope_layout must be split, interleaved or None")
        if rope_layout is not None and self.dimensions[1] % 2:
            raise ValueError("RoPE needs an even head dimension")
        self.rope_layout = rope_layout
        self.operators = Operators() if operators is None else operators
        self._owns_operators = operators is None

    def new_cache(self, batch, capacity, *, like=None):
        return KVCache.allocate(batch, capacity, self.kv_heads, self.dimensions[1],
                                like=self.weights.query if like is None else like)

    def workspace_specs(self, batch, tokens):
        integer(batch, "batch")
        integer(tokens, "tokens", minimum=0)
        d = self.dimensions[1]
        return (BufferSpec("query", (batch, tokens, self.heads, d)),
                BufferSpec("key", (batch, tokens, self.kv_heads, d)),
                BufferSpec("value", (batch, tokens, self.kv_heads, d)),
                BufferSpec("attention", (batch, tokens, self.heads, d)))

    def workspace(self, batch, tokens, *, like=None):
        return Workspace(self.workspace_specs(batch, tokens), like=self.weights.query if like is None else like)

    @property
    def required_programs(self):
        result = {"linear": {}, "kv_update": {}, "causal_attention": {"D": self.dimensions[1]}}
        if self.rope_layout is not None:
            result["rope_" + self.rope_layout] = {}
        return result

    def materialize(self):
        self.operators.materialize(self.required_programs, **self.required_programs)
        return self

    def _check(self, x, cache, cosine, sine):
        b, t, d = tensor(x, "x", rank=3)
        if d != self.dimensions[0] or self.weights.dimensions(self.heads, self.kv_heads) != self.dimensions:
            raise ValueError("attention input or parameters changed dimensions")
        same_backend(x, self.weights.query)
        if not isinstance(cache, KVCache):
            raise TypeError("cache must be KVCache")
        cache.check(batch=b, tokens=t, kv_heads=self.kv_heads, head_dim=self.dimensions[1], like=x)
        if self.rope_layout is None:
            if cosine is not None or sine is not None:
                raise ValueError("rotary tables were supplied to attention without RoPE")
        else:
            tensor(cosine, "cosine", shape=(t, self.dimensions[1] // 2))
            tensor(sine, "sine", shape=(t, self.dimensions[1] // 2))
            same_backend(x, cosine, sine)
        return b, t, d

    def _run(self, x, cache, start, y, work, cosine, sine):
        op, w = self.operators, self.weights
        b, t, _ = x.shape
        for batch in range(b):
            for name, weight in (("query", w.query), ("key", w.key), ("value", w.value)):
                op.linear(x[batch], weight, out=reshape(work[name][batch], (t, weight.shape[0])))
            if self.rope_layout is not None:
                op.rope(work["query"][batch], cosine, sine, layout=self.rope_layout, out=work["query"][batch])
                op.rope(work["key"][batch], cosine, sine, layout=self.rope_layout, out=work["key"][batch])
        op.kv_update(work["key"], work["value"], cache.keys, cache.values, start=start)
        op.attention(work["query"], cache.keys, cache.values, start=start, out=work["attention"])
        for batch in range(b):
            op.linear(reshape(work["attention"][batch], (t, self.heads * self.dimensions[1])), w.output, out=y[batch])

    def __call__(self, x, cache, *, cosine=None, sine=None, out=None, workspace=None):
        b, t, d = self._check(x, cache, cosine, sine)
        y = output(out, (b, t, d), like=x)
        work = use_workspace(workspace, self.workspace_specs(b, t), like=x)
        check_invocation(x, y, self.weights, work, state={"keys": cache.keys, "values": cache.values},
                         extra_inputs={} if cosine is None else {"cosine": cosine, "sine": sine})
        with work, cache.use(t) as start:
            self._run(x, cache, start, y, work, cosine, sine)
        return y

    def close(self):
        if self._owns_operators:
            self.operators.close()


@dataclass(frozen=True)
class TransformerWeights:
    attention_norm: object
    attention: AttentionWeights
    ffn_norm: object
    ffn: SwiGLUWeights

    def check(self, model):
        tensor(self.attention_norm, "attention_norm", shape=(model,))
        tensor(self.ffn_norm, "ffn_norm", shape=(model,))
        if self.ffn.dimensions()[0] != model:
            raise ValueError("attention and feedforward model dimensions differ")
        same_backend(self.attention_norm, self.ffn_norm, self.attention.query, self.ffn.gate)


class TransformerBlock:
    """RMS pre-norm, causal self-attention, residual and RMS/SwiGLU residual."""

    def __init__(self, weights: TransformerWeights, heads, kv_heads, *, epsilon=1e-5,
                 rope_layout="split", operators=None):
        if not isinstance(weights, TransformerWeights):
            raise TypeError("weights must be TransformerWeights")
        self.weights = weights
        self.operators = Operators() if operators is None else operators
        self._owns_operators = operators is None
        self.attention = SelfAttention(weights.attention, heads, kv_heads,
                                       rope_layout=rope_layout, operators=self.operators)
        self.ffn = SwiGLU(weights.ffn, operators=self.operators)
        self.model = self.attention.dimensions[0]
        weights.check(self.model)
        self.epsilon = positive(epsilon, "epsilon")

    def new_cache(self, batch, capacity, *, like=None):
        return self.attention.new_cache(batch, capacity, like=like)

    def workspace_specs(self, batch, tokens):
        intermediate = tuple(BufferSpec(name, (batch, tokens, self.model))
                             for name in ("normalized", "attention_output", "residual", "ffn_output"))
        return intermediate + self.attention.workspace_specs(batch, tokens) + self.ffn.workspace_specs(batch, tokens)

    def workspace(self, batch, tokens, *, like=None):
        return Workspace(self.workspace_specs(batch, tokens),
                         like=self.weights.attention_norm if like is None else like)

    def materialize(self):
        self.operators.materialize(self.required_programs, **self.required_programs)
        return self

    @property
    def required_programs(self):
        return {"rms_norm": {}, "add": {}, **self.attention.required_programs, **self.ffn.required_programs}

    def __call__(self, x, cache, *, cosine=None, sine=None, out=None, workspace=None):
        b, t, d = self.attention._check(x, cache, cosine, sine)
        self.weights.check(self.model)
        y = output(out, (b, t, d), like=x)
        work = use_workspace(workspace, self.workspace_specs(b, t), like=x)
        check_invocation(x, y, self.weights, work, state={"keys": cache.keys, "values": cache.values},
                         extra_inputs={} if cosine is None else {"cosine": cosine, "sine": sine})
        op = self.operators
        with work, cache.use(t) as start:
            for batch in range(b):
                op.rms_norm(x[batch], self.weights.attention_norm, epsilon=self.epsilon,
                            out=work["normalized"][batch])
            self.attention._run(work["normalized"], cache, start, work["attention_output"], work, cosine, sine)
            for batch in range(b):
                op.add(x[batch], work["attention_output"][batch], out=work["residual"][batch])
            op.rms_norm(work["residual"], self.weights.ffn_norm, epsilon=self.epsilon, out=work["normalized"])
            self.ffn._run(work["normalized"], work["ffn_output"], work)
            for batch in range(b):
                op.add(work["residual"][batch], work["ffn_output"][batch], out=y[batch])
        return y

    def close(self):
        if self._owns_operators:
            self.operators.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
