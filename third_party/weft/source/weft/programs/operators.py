"""Explicit model composition over ordinary Triton kernel launches."""
from __future__ import annotations

from contextlib import AbstractContextManager
from functools import partial
from threading import RLock
from types import MappingProxyType
from math import prod
import triton

from .catalog import get
from .contracts import allocate, integer, output, positive, reshape, rows, same_backend, tensor


class Operators(AbstractContextManager):
    """Own application choices; compilation and tuning belong to Triton."""

    def __init__(self, *, configs=None, tuning=None, toolchain=None):
        self.configs = MappingProxyType(dict(configs or {}))
        self.tuning = MappingProxyType({name: tuple(values) for name, values in (tuning or {}).items()})
        if self.configs.keys() & self.tuning.keys():
            raise ValueError("a program cannot have both a fixed config and tuning candidates")
        for name in self.configs.keys() | self.tuning.keys():
            get(name)
        self.toolchain = toolchain
        self._calls, self._native = {}, {}
        self._lock = RLock()
        self._closed = False

    def kernel(self, name, **bindings):
        entry = get(name)
        config = entry.config(self.configs.get(name), **bindings)
        key = (name, tuple(sorted(config.kwargs.items())))
        with self._lock:
            if self._closed:
                raise RuntimeError("Operators is closed")
            if key not in self._calls:
                if name in self.tuning:
                    candidates = [entry.config(candidate, **bindings) for candidate in self.tuning[name]]
                    function = triton.autotune(configs=candidates, key=[])(entry.definition)
                    options = {}
                else:
                    function = entry.definition
                    options = config.all_kwargs(model="weft")
                if self.toolchain is not None:
                    options["toolchain"] = self.toolchain
                self._calls[key] = partial(function.run, grid=(1,), warmup=False, **options)
            if key in self._native:
                native = self._native[key]
                def invoke(*args, **kwargs):
                    arguments = entry.definition.signature.bind(*args, **{**config.kwargs, **kwargs})
                    arguments.apply_defaults()
                    return native[(1,)](**{parameter["name"]: arguments.arguments[parameter["name"]]
                                          for parameter in native.metadata.public_signature})
                return invoke
            return self._calls[key]

    def materialize(self, names, **bindings):
        """Explicit AOT compilation of fixed model requirements before writes."""
        result = []
        for name in names:
            if name in self.tuning:
                raise ValueError("autotuned programs require real arguments; call the Triton kernel to tune")
            entry = get(name)
            selected = entry.config(self.configs.get(name), **bindings.get(name, {}))
            key = (name, tuple(sorted(selected.kwargs.items())))
            with self._lock:
                if self._closed:
                    raise RuntimeError("Operators is closed")
                if key not in self._native:
                    self._native[key] = entry.compile(config=selected, toolchain=self.toolchain)
                result.append(self._native[key])
        return tuple(result)

    def close(self):
        with self._lock:
            self._closed = True
            self._calls.clear()
            self._native.clear()

    def __exit__(self, *exception):
        self.close()

    def _unary(self, name, x, *, out=None):
        shape = tensor(x, "x")
        y = output(out, shape, like=x)
        bindings = {"N": shape[-1]} if get(name).required == ("N",) else {}
        self.kernel(name, **bindings)(X=rows(x), Y=rows(y))
        return y

    def silu(self, x, *, out=None):
        return self._unary("silu", x, out=out)

    def logarithm(self, x, *, out=None):
        return self._unary("logarithm", x, out=out)

    def exponential(self, x, *, out=None):
        return self._unary("exponential", x, out=out)

    def softplus(self, x, *, out=None):
        return self._unary("softplus", x, out=out)

    def copy(self, x, *, out=None):
        return self._unary("copy", x, out=out)

    def where(self, condition, x, other, *, out=None):
        """Select equal-shaped f32 arrays using Boolean storage, without broadcasting."""
        shape = tensor(x, "x")
        tensor(condition, "condition", shape=shape, dtype="i1")
        tensor(other, "other", shape=shape)
        same_backend(condition, x, other)
        flat = rows(x)
        y = output(out, shape, like=x)
        self.kernel("where")(Condition=reshape(condition, flat.shape), X=flat,
                             Other=rows(other), Y=rows(y))
        return y

    def fill(self, out, value=0.0):
        tensor(out, "out", writable=True)
        self.kernel("fill")(Y=rows(out), value=value)
        return out

    def softmax(self, x, *, out=None):
        return self._unary("softmax", x, out=out)

    def add(self, x, residual, *, out=None):
        shape = tensor(x, "x")
        tensor(residual, "residual", shape=shape)
        same_backend(x, residual)
        y = output(out, shape, like=x)
        self.kernel("add")(X=rows(x), Residual=rows(residual), Y=rows(y))
        return y

    def swiglu(self, gate, up, *, out=None):
        shape = tensor(gate, "gate")
        tensor(up, "up", shape=shape)
        same_backend(gate, up)
        y = output(out, shape, like=gate)
        self.kernel("swiglu")(Gate=rows(gate), Up=rows(up), Y=rows(y))
        return y

    def _normalization(self, name, x, weight, *, bias=None, residual=None, epsilon=1e-5, out=None):
        shape = tensor(x, "x")
        flat = rows(x)
        tensor(weight, "weight", shape=(shape[-1],))
        inputs = {"X": flat, "Weight": weight}
        parameters = [x, weight]
        if bias is not None:
            tensor(bias, "bias", shape=(shape[-1],))
            inputs["Bias"] = bias
            parameters.append(bias)
        if residual is not None:
            tensor(residual, "residual", shape=shape)
            inputs["Residual"] = rows(residual)
            parameters.append(residual)
        same_backend(*parameters)
        y = output(out, shape, like=x)
        bindings = {"N": shape[-1]} if get(name).required == ("N",) else {}
        self.kernel(name, **bindings)(**inputs, Y=rows(y), epsilon=positive(epsilon, "epsilon"))
        return y

    def rms_norm(self, x, weight, *, epsilon=1e-5, out=None):
        return self._normalization("rms_norm", x, weight, epsilon=epsilon, out=out)

    def rms_norm_row(self, x, weight, *, epsilon=1e-5, out=None):
        return self._normalization("rms_norm_row", x, weight, epsilon=epsilon, out=out)

    def rms_norm_tile(self, x, weight, *, epsilon=1e-5, out=None):
        return self._normalization("rms_norm_tile", x, weight, epsilon=epsilon, out=out)

    def layer_norm(self, x, weight, bias, *, epsilon=1e-5, out=None):
        return self._normalization("layer_norm", x, weight, bias=bias, epsilon=epsilon, out=out)

    def layer_norm_row(self, x, weight, bias, *, epsilon=1e-5, out=None):
        return self._normalization("layer_norm_row", x, weight, bias=bias, epsilon=epsilon, out=out)

    def layer_norm_tile(self, x, weight, bias, *, epsilon=1e-5, out=None):
        return self._normalization("layer_norm_tile", x, weight, bias=bias, epsilon=epsilon, out=out)

    def residual_rms_norm(self, x, residual, weight, *, epsilon=1e-5, out=None):
        return self._normalization("residual_rms_norm", x, weight, residual=residual, epsilon=epsilon, out=out)

    def softmax_row(self, x, *, out=None):
        return self._unary("softmax_row", x, out=out)

    def linear(self, x, weight, bias=None, *, out=None):
        return self._linear("linear", "linear_bias", x, weight, bias, out=out)

    def linear_silu(self, x, weight, bias=None, *, out=None):
        return self._linear("linear_silu", "linear_bias_silu", x, weight, bias, out=out)

    def bmm(self, x, weight, *, out=None):
        batch, m, k = tensor(x, "x", rank=3)
        wb, wk, n = tensor(weight, "weight", rank=3)
        if batch != wb or k != wk:
            raise ValueError("bmm requires equal batch and contraction dimensions")
        same_backend(x, weight)
        y = output(out, (batch, m, n), like=x)
        self.kernel("batched_matmul")(X=x, Weight=weight, Y=y)
        return y

    def matmul(self, x, weight, bias=None, *, out=None):
        shape = tensor(x, "x")
        if not shape:
            raise ValueError("matmul activation requires at least one dimension")
        k, n = tensor(weight, "weight", rank=2)
        if k != shape[-1]:
            raise ValueError("matmul right operand input dimension differs from the activation")
        same_backend(x, weight)
        y = output(out, (*shape[:-1], n), like=x)
        count = prod(shape[:-1])
        arguments = {"X": reshape(x, (count, k)), "Weight": weight, "Y": reshape(y, (count, n))}
        if bias is not None:
            tensor(bias, "bias", shape=(n,))
            same_backend(x, bias)
            arguments["Bias"] = bias
        self.kernel("matmul" if bias is None else "matmul_bias")(**arguments)
        return y

    def linear_tile(self, x, weight, bias=None, *, out=None):
        return self._linear("linear_tile", "linear_bias_tile", x, weight, bias, out=out)

    def _linear(self, plain, biased, x, weight, bias, *, out):
        shape = tensor(x, "x")
        flat = rows(x)
        n, k = tensor(weight, "weight", rank=2)
        integer(n, "output features")
        if k != shape[-1]:
            raise ValueError("linear weight input dimension differs from the activation")
        same_backend(x, weight)
        y = output(out, (*shape[:-1], n), like=x)
        if bias is None:
            self.kernel(plain)(X=flat, Weight=weight, Y=rows(y))
        else:
            tensor(bias, "bias", shape=(n,))
            same_backend(x, bias)
            self.kernel(biased)(X=flat, Weight=weight, Bias=bias, Y=rows(y))
        return y

    def rope(self, x, cosine, sine, *, layout="split", out=None):
        shape = tensor(x, "x", rank=3)
        t, h, d = shape
        if d == 0 or d % 2:
            raise ValueError("RoPE head dimension must be positive and even")
        tensor(cosine, "cosine", shape=(t, d // 2))
        tensor(sine, "sine", shape=(t, d // 2))
        same_backend(x, cosine, sine)
        if layout not in ("split", "interleaved"):
            raise ValueError("RoPE layout must be split or interleaved")
        y = output(out, shape, like=x)
        internal = (t, h, 2, d // 2) if layout == "split" else (t, h, d // 2, 2)
        self.kernel("rope_" + layout)(X=reshape(x, internal), Cos=cosine, Sin=sine, Y=reshape(y, internal))
        return y

    def kv_update(self, k, v, key_cache, value_cache, *, start):
        b, t, g, d = tensor(k, "k", rank=4)
        tensor(v, "v", shape=(b, t, g, d))
        cb, capacity, cg, cd = tensor(key_cache, "key_cache", rank=4, writable=True)
        tensor(value_cache, "value_cache", shape=(cb, capacity, cg, cd), writable=True)
        if (cb, cg, cd) != (b, g, d):
            raise ValueError("KV cache and incoming key/value dimensions disagree")
        integer(start, "start", minimum=0)
        if start + t > capacity:
            raise ValueError("KV append exceeds cache capacity")
        same_backend(k, v, key_cache, value_cache)
        self.kernel("kv_update")(K=k, V=v, KeyCache=key_cache, ValueCache=value_cache, start=start)

    def attention(self, q, k, v, *, start=0, scale=None, out=None):
        b, t, h, d = tensor(q, "q", rank=4)
        kb, capacity, g, kd = tensor(k, "k", rank=4)
        tensor(v, "v", shape=(kb, capacity, g, kd))
        if (kb, kd) != (b, d) or d <= 0 or g <= 0 or h <= 0 or h % g:
            raise ValueError("attention requires matching batch/head dimensions and H divisible by G")
        integer(start, "start", minimum=0)
        if start + t > capacity:
            raise ValueError("attention prefix exceeds KV capacity")
        same_backend(q, k, v)
        y = output(out, (b, t, h, d), like=q)
        scale = d ** -0.5 if scale is None else positive(scale, "scale")
        self.kernel("causal_attention", D=d)(Q=q, K=k, V=v, Y=y, start=start, scale=scale)
        return y

    def route(self, logits, top_k, *, indices=None, weights=None):
        t, e = tensor(logits, "logits", rank=2)
        integer(top_k, "top_k")
        if top_k > e or e > 2147483647:
            raise ValueError("top_k must not exceed the i32-addressable expert count")
        if indices is None:
            indices = allocate((t, top_k), like=logits, dtype="i32")
        tensor(indices, "indices", shape=(t, top_k), dtype="i32", writable=True)
        weights = output(weights, (t, top_k), like=logits)
        same_backend(logits, indices, weights)
        self.kernel("router")(Logits=logits, Indices=indices, Weights=weights)
        return indices, weights
