"""Installed model programs, numerical reproduction and native operator timing."""
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

from weft.programs import activation, attention, mamba, normalization
from weft.runtime import CompileOptions
import weft


def check(name, actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
    error = float(np.max(np.abs(np.asarray(actual) - expected))) if np.size(expected) else 0.0
    print(json.dumps({"program": name, "status": "correct", "max_abs_error": error}), flush=True)


def random(shape, rng):
    return rng.normal(0, 0.3, shape).astype(np.float32)


def normalization_usage(block=32):
    from weft.programs import get

    rng = np.random.default_rng(410)
    for width in (1, 7, 33, 129, 16385):
        storage = random((4, width * 2), rng)
        x = storage[:, ::2]
        w = random((width,), rng)
        bias = random((width,), rng)
        residual = random(x.shape, rng)
        out_storage = np.full_like(storage, 1234)
        y = out_storage[:, ::2]
        for definition in (normalization.rms_norm, normalization.layer_norm, normalization.residual_rms_norm):
            with get(definition.__name__).jit(BLOCK=block) as kernel:
                if definition is normalization.rms_norm:
                    expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * w
                    kernel(X=x, Weight=w, Y=y)
                elif definition is normalization.layer_norm:
                    centered = x - x.mean(axis=-1, keepdims=True)
                    expected = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5) * w + bias
                    kernel(X=x, Weight=w, Bias=bias, Y=y)
                else:
                    value = x + residual
                    expected = value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-5) * w
                    kernel(X=x, Residual=residual, Weight=w, Y=y)
                check(f"{definition.__name__}/strided/{width}", y, expected)
                np.testing.assert_array_equal(out_storage[:, 1::2], 1234)
                inplace = x.copy()
                kwargs = {"X": inplace, "Weight": w, "Y": inplace}
                if definition is normalization.layer_norm:
                    kwargs["Bias"] = bias
                if definition is normalization.residual_rms_norm:
                    kwargs["Residual"] = residual
                kernel(**kwargs)
                check(f"{definition.__name__}/inplace/{width}", inplace, expected)
        with get("softmax").jit(N=width) as kernel:
            kernel(X=x, Y=y)
            exp = np.exp(x - x.max(axis=-1, keepdims=True))
            check(f"softmax/{width}", y, exp / exp.sum(axis=-1, keepdims=True))
            np.testing.assert_array_equal(out_storage[:, 1::2], 1234)
            inplace = x.copy()
            kernel(X=inplace, Y=inplace)
            check(f"softmax/inplace/{width}", inplace, exp / exp.sum(axis=-1, keepdims=True))
    from weft.programs import Operators

    values = np.array([[-100, -16, -1, 0, 1, 16, 100]], dtype=np.float32)
    with Operators() as ops:
        check("copy/public_api", ops.copy(values), values)
        small = np.exp(-np.abs(values))
        sigmoid = np.where(values >= 0, 1 / (1 + small), small / (1 + small))
        check("silu/extreme_finite", ops.silu(values), values * sigmoid)


def attention_usage(block=32):
    rng = np.random.default_rng(411)
    for layout in ("split", "interleaved"):
        x = random((3, 2, 14), rng)
        phase = random((3, 7), rng)
        cosine, sine = np.cos(phase), np.sin(phase)
        y = np.empty_like(x)
        shape = (3, 2, 2, 7) if layout == "split" else (3, 2, 7, 2)
        internal = x.reshape(shape)
        first = internal[:, :, 0, :] if layout == "split" else internal[:, :, :, 0]
        second = internal[:, :, 1, :] if layout == "split" else internal[:, :, :, 1]
        pair = [first * cosine[:, None] - second * sine[:, None],
                second * cosine[:, None] + first * sine[:, None]]
        expected = np.stack(pair, axis=2 if layout == "split" else 3).reshape(x.shape)
        with weft.jit(getattr(attention, "rope_" + layout), options=CompileOptions(meta={"BLOCK": block})) as kernel:
            kernel(X=internal, Cos=cosine, Sin=sine, Y=y.reshape(shape))
            check("rope_" + layout, y, expected)
            kernel(X=internal, Cos=cosine, Sin=sine, Y=internal)
            check("rope_" + layout + "/inplace", x, expected)
    for d in (8, 16, 33, 64, 128, 129, 257):
        q = random((1, 3, 4, d), rng)
        k = random((1, 6, 2, d), rng)
        v = random(k.shape, rng)
        y = np.empty_like(q)
        start = 2
        expected = np.empty_like(y)
        for token in range(3):
            for head in range(4):
                key = k[0, :start + token + 1, head // 2]
                val = v[0, :start + token + 1, head // 2]
                scores = key @ q[0, token, head] / np.sqrt(d)
                probs = np.exp(scores - scores.max())
                expected[0, token, head] = (probs / probs.sum()) @ val
        for lmul in ((4, 8) if d >= 128 else (8,)):
            with weft.jit(attention.causal_attention, options=CompileOptions(meta={"D": d}, lmul_eighths=lmul)) as kernel:
                kernel(Q=q, K=k, V=v, Y=y, start=start, scale=float(d ** -0.5))
                check(f"causal_attention/{d}/lmul={lmul}/8", y, expected)


def value_usage(with_torch=False):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from repro.weft.native_jit import value_program_calls
    from weft.runtime import Toolchain
    from weft.programs import get, Operators

    with TemporaryDirectory(prefix="weft-values-") as temporary:
        print(json.dumps(value_program_calls(Toolchain.from_environment(), Path(temporary))), flush=True)
    rng = np.random.default_rng(419)
    for width, row_tile in ((1, 4), (33, 4), (257, 4), (4096, 1)):
        x = random((5, width * 2), rng)[:, ::2]
        weight, bias = random((width,), rng), random((width,), rng)
        centered = x - x.mean(axis=-1, keepdims=True)
        references = {
            "rms_norm_tile": x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5) * weight,
            "layer_norm_tile": centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) * weight + bias,
        }
        for name, expected in references.items():
            with get(name).jit(N=width, ROWS=row_tile) as kernel:
                output = np.full((5, width * 2), 1234, dtype=np.float32)
                arguments = {"X": x, "Weight": weight, "Y": output[:, ::2]}
                if name == "layer_norm_tile": arguments["Bias"] = bias
                kernel(**arguments)
                check(f"{name}/rows={row_tile}/{width}", output[:, ::2], expected)
                np.testing.assert_array_equal(output[:, 1::2], 1234)
                inplace = x.copy()
                kernel(**{**arguments, "X": inplace, "Y": inplace})
                check(f"{name}/inplace/{width}", inplace, expected)
        if width == 33:
            with Operators() as operators:
                check("rms_norm_tile/public", operators.rms_norm_tile(x, weight), references["rms_norm_tile"])
                check("layer_norm_tile/public", operators.layer_norm_tile(x, weight, bias), references["layer_norm_tile"])
            if with_torch:
                import torch
                from weft.programs.torch import Library, LayerNormTile

                with Library("weft_value_usage", options={"layer_norm_tile": CompileOptions(meta={"ROWS": row_tile})}) as library:
                    tensors = tuple(torch.from_numpy(value) for value in (x, weight, bias))
                    module = LayerNormTile(tensors[1], tensors[2], library=library)
                    check("layer_norm_tile/torch", module(tensors[0]).numpy(), references["layer_norm_tile"])
                    check("layer_norm_tile/torch_compile", torch.compile(module, backend="eager", fullgraph=True)(tensors[0]).numpy(),
                          references["layer_norm_tile"])
                    check("layer_norm_tile/torch_export", torch.export.export(module, (tensors[0],)).module()(tensors[0]).numpy(),
                          references["layer_norm_tile"])


def row_usage(with_torch=False):
    from weft.programs import get, Operators

    rng = np.random.default_rng(416)
    # Initialized values are full; the destination's two tail dimensions still
    # bound every store. Exercise both register and private-memory placement.
    for row_block, column_block in ((1, 32), (16, 64)):
        with get("fill").jit(ROWS=row_block, BLOCK=column_block) as kernel:
            for shape in ((3, 17), (19, 65), (0, 17)):
                storage = np.full((shape[0] + 2, shape[1] * 2 + 2), 1234, dtype=np.float32)
                expected = storage.copy()
                expected[1:-1, 1:-1:2] = -2.5
                kernel(Y=storage[1:-1, 1:-1:2], value=-2.5)
                check(f"fill/{row_block}x{column_block}/{shape}", storage, expected)
    with Operators() as operators:
        destination = np.empty((3, 17), dtype=np.float32)
        result = operators.fill(destination, -1.25)
        assert result is destination
        check("fill/public", result, np.full_like(destination, -1.25))
    for width in (1, 33, 4096):
        storage = random((3, width * 2), rng)
        x = storage[:, ::2]
        weight = random((width,), rng)
        bias = random((width,), rng)
        centered = x - x.mean(axis=-1, keepdims=True)
        exponential = np.exp(x - x.max(axis=-1, keepdims=True))
        cases = (
            ("rms_norm_row", dict(X=x, Weight=weight),
             x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5) * weight),
            ("layer_norm_row", dict(X=x, Weight=weight, Bias=bias),
             centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) * weight + bias),
            ("softmax_row", dict(X=x), exponential / exponential.sum(axis=-1, keepdims=True)),
        )
        for name, inputs, expected in cases:
            with get(name).jit(N=width) as kernel:
                output = np.full_like(storage, 1234)
                kernel(**inputs, Y=output[:, ::2])
                check(f"{name}/strided/{width}", output[:, ::2], expected)
                np.testing.assert_array_equal(output[:, 1::2], 1234)
                inplace = x.copy()
                kernel(**{**inputs, "X": inplace}, Y=inplace)
                check(f"{name}/inplace/{width}", inplace, expected)
                shifted = np.empty(width + 1, dtype=np.float32)
                shifted[:-1] = x[0]
                before = shifted[:-1].copy()
                overlapping = shifted[:-1].reshape(1, width)
                kernel(**{**inputs, "X": overlapping}, Y=shifted[1:].reshape(1, width))
                check(f"{name}/overlap/{width}", shifted[1:], expected[0])
                np.testing.assert_array_equal(shifted[:1], before[:1])
        with Operators() as operators:
            for name, inputs, expected in cases:
                arguments = [inputs["X"]] + [inputs[key] for key in ("Weight", "Bias") if key in inputs]
                check(name + f"/public/{width}", getattr(operators, name)(*arguments), expected)
    if with_torch:
        import torch
        from weft.programs.torch import Library, RMSNormRow, LayerNormRow, SoftmaxRow

        with Library("weft_row_usage") as library:
            tensor = torch.from_numpy(x.copy())
            for name, inputs, expected in cases:
                parameters = [torch.from_numpy(inputs[key].copy()) for key in ("Weight", "Bias") if key in inputs]
                if name == "rms_norm_row": module = RMSNormRow(*parameters, library=library)
                elif name == "layer_norm_row": module = LayerNormRow(*parameters, library=library)
                else: module = SoftmaxRow(width, library=library)
                compiled = torch.compile(module, backend="eager", fullgraph=True)
                check(name + "/torch_compile", compiled(tensor).numpy(), expected)
                exported = torch.export.export(module, (tensor,)).module()
                check(name + "/torch_export", exported(tensor).numpy(), expected)


def silu_reference(x):
    values = np.asarray(x)
    wide = values.astype(np.float64)
    return (wide / (1 + np.exp(-wide))).astype(values.dtype)


def scan_reference(arguments):
    state = arguments["State"].copy()
    expected = np.empty_like(arguments["X"])
    rate = -np.exp(arguments["Alog"])
    for token in range(arguments["X"].shape[1]):
        delta = arguments["Delta"][:, token] + arguments["Bias"]
        dt = np.logaddexp(np.float32(0), delta)
        value = arguments["X"][:, token]
        state = state * np.exp(dt[:, :, None] * rate) + (
            (dt * value)[:, :, None] * arguments["Bcoef"][:, token, None, :])
        expected[:, token] = ((state * arguments["Ccoef"][:, token, None, :]).sum(axis=-1) +
            arguments["Skip"] * value) * silu_reference(arguments["Gate"][:, token])
    return expected, state


def scan_usage():
    from weft.programs import get

    rng = np.random.default_rng(418)
    for state_size in (1, 16, 33):
        batch, length, channels = 2, 5, 7
        arguments = {key: random((batch, length, 2 * channels), rng)[..., ::2]
                     for key in ("X", "Delta", "Gate")}
        arguments.update({key: random((batch, length, 2 * state_size), rng)[..., ::2]
                          for key in ("Bcoef", "Ccoef")})
        arguments.update({key: random((2 * channels,), rng)[::2] for key in ("Skip", "Bias")})
        arguments["Alog"] = random((channels, 2 * state_size), rng)[:, ::2]
        state_storage = np.full((batch, channels, 2 * state_size + 2), 19, dtype=np.float32)
        initial = random((batch, channels, state_size), rng)
        arguments["State"] = state_storage[..., 1:-1:2]
        np.copyto(arguments["State"], initial)
        expected, final_state = scan_reference(arguments)
        output_storage = np.full((batch, length, 2 * channels + 2), 23, dtype=np.float32)
        output = output_storage[..., 1:-1:2]
        with get("selective_scan").jit(N=state_size) as kernel:
            kernel(**arguments, Y=output)
            check(f"selective_scan/strided/{state_size}", output, expected)
            check(f"selective_scan/final_state/{state_size}", arguments["State"], final_state)
            np.testing.assert_array_equal(state_storage[..., ::2], 19)
            np.testing.assert_array_equal(state_storage[..., -1], 19)
            np.testing.assert_array_equal(output_storage[..., ::2], 23)
            np.testing.assert_array_equal(output_storage[..., -1], 23)
            np.copyto(arguments["State"], initial)
            for lo, hi in ((0, 0), (0, 2), (2, 5)):
                sliced = dict(arguments)
                for key in ("X", "Delta", "Gate", "Bcoef", "Ccoef"):
                    sliced[key] = arguments[key][:, lo:hi]
                kernel(**sliced, Y=output[:, lo:hi])
                if lo == hi:
                    np.testing.assert_array_equal(arguments["State"], initial)
            check(f"selective_scan/chunks/{state_size}", output, expected)
            check(f"selective_scan/chunks_state/{state_size}", arguments["State"], final_state)
            np.testing.assert_array_equal(state_storage[..., ::2], 19)
            np.testing.assert_array_equal(state_storage[..., -1], 19)
            np.testing.assert_array_equal(output_storage[..., ::2], 23)
            np.testing.assert_array_equal(output_storage[..., -1], 23)


def model_usage(with_torch=False):
    scan_usage()
    from weft.programs import (AttentionWeights, Mamba1, MambaWeights, MoE, MoEWeights,
                               SwiGLUWeights, TransformerBlock, TransformerWeights)

    rng = np.random.default_rng(412)
    b, t, d, h, g, hd, hidden = 2, 5, 13, 4, 2, 8, 17
    x = random((b, t, d), rng)
    aw = AttentionWeights(random((h * hd, d), rng), random((g * hd, d), rng),
                          random((g * hd, d), rng), random((d, h * hd), rng))
    fw = SwiGLUWeights(random((hidden, d), rng), random((hidden, d), rng), random((d, hidden), rng))
    tw = TransformerWeights(random((d,), rng), aw, random((d,), rng), fw)
    phase = random((t, hd // 2), rng)
    cosine, sine = np.cos(phase), np.sin(phase)

    def norm(z, weight):
        return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + 1e-5) * weight

    def rotary(z):
        first, second = np.split(z, 2, axis=-1)
        return np.concatenate((first * cosine[None, :, None] - second * sine[None, :, None],
                               second * cosine[None, :, None] + first * sine[None, :, None]), axis=-1)

    normalized = norm(x, tw.attention_norm)
    q = rotary((normalized @ aw.query.T).reshape(b, t, h, hd))
    k = rotary((normalized @ aw.key.T).reshape(b, t, g, hd))
    v = (normalized @ aw.value.T).reshape(b, t, g, hd)
    context = np.empty_like(q)
    for batch in range(b):
        for token in range(t):
            for head in range(h):
                score = k[batch, :token + 1, head // (h // g)] @ q[batch, token, head] / np.sqrt(hd)
                probability = np.exp(score - score.max())
                context[batch, token, head] = (probability / probability.sum()) @ v[batch, :token + 1, head // (h // g)]
    residual = x + context.reshape(b, t, h * hd) @ aw.output.T
    normalized = norm(residual, tw.ffn_norm)
    expected = residual + (silu_reference(normalized @ fw.gate.T) * (normalized @ fw.up.T)) @ fw.down.T
    with TransformerBlock(tw, h, g) as block:
        cache = block.new_cache(b, t + 2)
        work = block.workspace(b, t)
        try:
            block(x, cache, cosine=cosine, sine=sine, out=x, workspace=work)
        except ValueError as error:
            assert "overlap" in str(error)
        else:
            raise AssertionError("whole-model alias preflight accepted an overlapping output")
        assert cache.valid and cache.position == 0
        actual = block(x, cache, cosine=cosine, sine=sine, workspace=work)
        check("transformer/full", actual, expected)
        check("transformer/key_cache", cache.keys[:, :t], k)
        check("transformer/value_cache", cache.values[:, :t], v)
        cache.reset()
        pieces = []
        for lo, hi in ((0, 2), (2, 3), (3, 5)):
            pieces.append(block(x[:, lo:hi], cache, cosine=cosine[lo:hi], sine=sine[lo:hi]))
        check("transformer/chunks", np.concatenate(pieces, axis=1), expected)
        assert cache.position == t
        branch = cache.fork()
        cache.reset()
        check("transformer/fork", branch.keys[:, :t], k)
        assert branch.position == t and cache.position == 0
        cache.restore(branch)
        cache.truncate(2)
        check("transformer/restore_and_rewind", block(x[:, 2:], cache, cosine=cosine[2:], sine=sine[2:]), expected[:, 2:])
    if with_torch:
        torch_usage("transformer", tw, x, expected, heads=h, kv_heads=g, cosine=cosine, sine=sine)

    e, top_k = 4, 2
    mw = MoEWeights(random((e, d), rng), random((e, hidden, d), rng),
                    random((e, hidden, d), rng), random((e, d, hidden), rng))
    tokens = x.reshape(b * t, d)
    for tied in (False, True):
        if tied:
            mw.router.fill(0)
        logits = tokens @ mw.router.T
        indices = np.argsort(-logits, axis=-1, kind="stable")[:, :top_k]
        chosen = np.take_along_axis(logits, indices, axis=-1)
        weights = np.exp(chosen - chosen.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected_moe = np.zeros_like(tokens)
        for token in range(len(tokens)):
            for slot, expert in enumerate(indices[token]):
                inner = silu_reference(tokens[token] @ mw.gate[expert].T) * (tokens[token] @ mw.up[expert].T)
                expected_moe[token] += weights[token, slot] * (inner @ mw.down[expert].T)
        with MoE(mw, top_k) as block:
            work = block.workspace(len(tokens))
            actual = block(tokens, workspace=work)
            check(f"moe/full/ties={tied}", actual, expected_moe)
            np.testing.assert_array_equal(work["indices"], indices)
            counts = np.bincount(indices.ravel(), minlength=e)
            np.testing.assert_array_equal(work["offsets"], np.r_[0, np.cumsum(counts)])
            np.testing.assert_array_equal(np.sort(work["positions"].ravel()), np.arange(len(tokens) * top_k))
            check("moe/routing_weights", work["weights"], weights)
            empty = block(tokens[:0])
            assert empty.shape == (0, d)
        if with_torch and not tied:
            torch_usage("moe", mw, tokens, expected_moe, top_k=top_k)

    inner, state_size, rank, width = 11, 7, 3, 4
    weights = MambaWeights(
        random((2 * inner, d), rng), random((inner, width), rng), random((inner,), rng),
        random((rank + 2 * state_size, inner), rng), random((inner, rank), rng),
        random((inner,), rng), random((inner, state_size), rng), random((inner,), rng),
        random((d, inner), rng), random((2 * inner,), rng), random((d,), rng))
    projected = x @ weights.in_proj.T + weights.in_bias
    history = np.zeros((b, width, inner), dtype=np.float32)
    state = np.zeros((b, inner, state_size), dtype=np.float32)
    expected_mamba = np.empty_like(x)
    for token in range(t):
        history[:, :-1] = history[:, 1:]
        history[:, -1] = projected[:, token, :inner]
        conv = silu_reference((history * weights.conv.T[None]).sum(axis=1) + weights.conv_bias)
        parameters = conv @ weights.x_proj.T
        delta = parameters[:, :rank] @ weights.dt_proj.T + weights.dt_bias
        dt = np.logaddexp(0, delta)
        bcoef = parameters[:, rank:rank + state_size]
        ccoef = parameters[:, rank + state_size:]
        state = state * np.exp(-np.exp(weights.a_log)[None] * dt[:, :, None])
        state += bcoef[:, None] * (dt * conv)[:, :, None]
        y = ((state * ccoef[:, None]).sum(axis=-1) + weights.skip * conv) * silu_reference(projected[:, token, inner:])
        expected_mamba[:, token] = y @ weights.out_proj.T + weights.out_bias
    with Mamba1(weights) as block:
        stream = block.new_state(b)
        actual = block(x, stream, workspace=block.workspace(b, t))
        check("mamba1/full", actual, expected_mamba)
        check("mamba1/final_conv", stream.history, history)
        check("mamba1/final_ssm", stream.ssm, state)
        stream.reset()
        pieces = []
        for lo, hi in ((0, 2), (2, 3), (3, 5)):
            pieces.append(block(x[:, lo:hi], stream))
        check("mamba1/chunks", np.concatenate(pieces, axis=1), expected_mamba)
        check("mamba1/chunks_ssm", stream.ssm, state)
        stream.reset()
        pieces = [block(x[:, token:token + 1], stream) for token in range(t)]
        check("mamba1/token_stream", np.concatenate(pieces, axis=1), expected_mamba)
        check("mamba1/token_ssm", stream.ssm, state)
        assert stream.position == t
        branch = stream.fork()
        stream.reset()
        check("mamba1/fork", branch.ssm, state)
        stream.restore(branch)
        check("mamba1/restore", stream.ssm, state)
        assert stream.position == t
        import tempfile
        from pathlib import Path
        from weft.programs import LoadedOperators, save_operators

        with tempfile.TemporaryDirectory(prefix="weft-model-package-") as directory:
            package = save_operators(block.operators, Path(directory) / "programs")
            with LoadedOperators(package) as loaded:
                deployed = Mamba1(weights, operators=loaded)
                check("mamba1/saved_package", deployed(x, deployed.new_state(b)), expected_mamba)
    if with_torch:
        torch_usage("mamba1", weights, x, expected_mamba)


def torch_usage(family, weights, x, expected, **settings):
    import torch
    from dataclasses import fields, is_dataclass
    from weft.programs import torch as modules

    def convert(value):
        if is_dataclass(value):
            return type(value)(**{field.name: convert(getattr(value, field.name)) for field in fields(value)})
        return torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value

    inputs = torch.from_numpy(x.copy())
    with modules.Library("weft_use_" + family) as library:
        if family == "transformer":
            model = modules.TransformerBlock(convert(weights), settings["heads"], settings["kv_heads"], library=library)

            def arguments():
                return (inputs, *model.new_cache(x.shape[0], x.shape[1] + 2), 0,
                        convert(settings["cosine"]), convert(settings["sine"]))
        elif family == "moe":
            model = modules.MoE(convert(weights), settings["top_k"], library=library)

            def arguments():
                return (inputs,)
        else:
            model = modules.Mamba1(convert(weights), library=library)

            def arguments():
                return (inputs, *model.new_state(x.shape[0]))
        actual = model(*arguments())
        check(f"torch/{family}/module", actual.numpy(), expected)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        check(f"torch/{family}/captured_execution", compiled(*arguments()).numpy(), expected)
        exported = torch.export.export(model, arguments(), strict=True)
        check(f"torch/{family}/exported_execution", exported.module()(*arguments()).numpy(), expected)


def predicate_usage(block=32, compare_artifacts=None, *, lmul_eighths=8, pointwise_issue_copies=1):
    from repro.weft.native_jit import predicate_calls
    from weft.runtime import Toolchain
    from weft.runtime.cache import ArtifactCache

    options = CompileOptions(meta={"BLOCK": block}, lmul_eighths=lmul_eighths,
                             pointwise_issue_copies=pointwise_issue_copies)
    result = predicate_calls(Toolchain.from_environment(), ArtifactCache().directory, measure=True,
                             compare_artifacts=compare_artifacts, options=options)
    print(json.dumps(result), flush=True)


def selection_usage(with_torch=False):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from weft.programs import Operators, Tuning, LoadedOperators, ProgramPackageBuilder, save_operators
    from weft.runtime import Config, TuningPolicy, TuningResult

    def validate(original, trial):
        expected = np.where(original["Condition"], original["X"], original["Other"])
        np.testing.assert_array_equal(trial["Y"], expected)
        np.testing.assert_array_equal(np.signbit(trial["Y"]), np.signbit(expected))

    policy = TuningPolicy(deadline_s=600, warmup=1, samples=3, repetitions=2)
    recipe = Tuning(tuple(Config(meta={"BLOCK": 32}, pointwise_issue_copies=copies,
                                 name=f"pieces{copies}") for copies in (1, 2, 4)),
                    policy=policy, validator=validate, validation_key="where-f32-exact-signed-zero")
    rng = np.random.default_rng(428)
    condition = np.arange(6 * 33).reshape(6, 33) % 3 == 0
    x, other = random((6, 33), rng), random((6, 33), rng)
    x[0, :2], other[0, :2] = [0.0, -0.0], [-0.0, 0.0]
    expected = np.where(condition, x, other)
    with TemporaryDirectory(prefix="weft-selection-") as temporary:
        directory = Path(temporary)
        with Operators(tuning={"where": recipe}) as operators:
            y = np.full_like(x, 37)
            originals = tuple(value.copy() for value in (condition, x, other, y))
            result = operators.tune("where", Condition=condition, X=x, Other=other, Y=y)
            for value, original in zip((condition, x, other, y), originals):
                np.testing.assert_array_equal(value, original)
            assert len(result.candidates) == 3
            assert all(candidate.status == "measured" and candidate.validated for candidate in result.candidates)
            actual = operators.where(condition, x, other, out=y)
            assert actual is y
            np.testing.assert_array_equal(y, expected)
            np.testing.assert_array_equal(np.signbit(y), np.signbit(expected))
            result.save(directory / "selection.json")
            saved = TuningResult.load(directory / "selection.json")
            assert saved.winner.options == result.winner.options
            package = save_operators(operators, directory / "selected", selections={"where": saved})
            with LoadedOperators(package) as loaded:
                np.testing.assert_array_equal(loaded.where(condition, x, other), expected)
                assert loaded.selected_options("where") == saved.winner.options
            target = operators.kernel("where").artifacts[0].target
            fixed = ProgramPackageBuilder(target).add_selection("where", saved)
            deployed = fixed.build(directory / "rebuilt")
            with LoadedOperators(deployed) as loaded:
                np.testing.assert_array_equal(loaded.where(condition, x, other), expected)
            print(json.dumps({"program": "where/tuning_and_deployment", "status": "correct",
                              "selected": saved.winner.options.to_dict(),
                              "candidates": [candidate.to_dict() for candidate in result.candidates]}), flush=True)

        with Operators(options={"where": saved.winner.options}) as operators:
            for shape in ((1,), (0, 33), (2, 3, 33), (3, 129)):
                condition_storage = rng.integers(0, 2, (*shape[:-1], shape[-1] * 2), dtype=np.uint8).astype(np.bool_)
                condition_view = condition_storage[..., ::2]
                first = random((*shape[:-1], shape[-1] * 2), rng)
                second = random(first.shape, rng)
                x_view, other_view = first[..., ::2], second[..., ::2]
                reference = np.where(condition_view, x_view, other_view)
                output_storage = np.full_like(first, 17)
                operators.where(condition_view, x_view, other_view, out=output_storage[..., ::2])
                np.testing.assert_array_equal(output_storage[..., ::2], reference)
                np.testing.assert_array_equal(output_storage[..., 1::2], 17)
                first_before, second_before = first.copy(), second.copy()
                operators.where(condition_view, x_view, other_view, out=x_view)
                np.testing.assert_array_equal(x_view, reference)
                np.testing.assert_array_equal(first[..., 1::2], first_before[..., 1::2])
                first[...] = first_before
                operators.where(condition_view, x_view, other_view, out=other_view)
                np.testing.assert_array_equal(other_view, reference)
                np.testing.assert_array_equal(second[..., 1::2], second_before[..., 1::2])
            for bad_condition in (condition.astype(np.uint8), condition[:, :1]):
                try:
                    operators.where(bad_condition, x, other)
                except (ValueError, TypeError):
                    pass
                else:
                    raise AssertionError("where accepted a non-Boolean or broadcast condition")
            # Shifted aliases retain the author's block snapshot and ordered
            # traversal, including values a preceding block has overwritten.
            for shift in (-7, 1, 17):
                for output_branch in ("selected", "other"):
                    storage = np.linspace(-2, 3, 200, dtype=np.float32)
                    reference_storage = storage.copy()
                    start = 20 + max(0, -shift)
                    read = storage[start:start + 97].reshape(1, 97)
                    write = storage[start + shift:start + shift + 97].reshape(1, 97)
                    reference_read = reference_storage[start:start + 97].reshape(1, 97)
                    reference_write = reference_storage[start + shift:start + shift + 97].reshape(1, 97)
                    mask = np.arange(97).reshape(1, 97) % 3 != 0
                    other_input = np.linspace(1, -1, 97, dtype=np.float32).reshape(1, 97)
                    for begin in range(0, 97, 32):
                        end = min(begin + 32, 97)
                        snapshot = reference_read[:, begin:end].copy()
                        left, right = (snapshot, other_input[:, begin:end]) if output_branch == "selected" else (
                            other_input[:, begin:end], snapshot)
                        reference_write[:, begin:end] = np.where(mask[:, begin:end], left, right)
                    left, right = (read, other_input) if output_branch == "selected" else (other_input, read)
                    operators.where(mask, left, right, out=write)
                    np.testing.assert_array_equal(storage, reference_storage)
            print(json.dumps({"program": "where/strides_aliases_and_shapes", "status": "correct"}), flush=True)

        if with_torch:
            import torch
            from weft.programs.torch import Library, Where

            with Library("weft_selection_usage", options={"where": saved.winner.options}) as library:
                module = Where(library=library)
                arguments = tuple(torch.from_numpy(value.reshape(2, 3, 33)) for value in (condition, x, other))
                reference = torch.where(*arguments)
                torch.testing.assert_close(module(*arguments), reference, rtol=0, atol=0)
                compiled = torch.compile(module, backend="eager", fullgraph=True)
                torch.testing.assert_close(compiled(*arguments), reference, rtol=0, atol=0)
                exported = torch.export.export(module, arguments).module()
                torch.testing.assert_close(exported(*arguments), reference, rtol=0, atol=0)
                package = library.save(directory / "torch-selected")
            with Library.from_package("weft_selection_deployed", package) as library:
                module = Where(library=library)
                torch.testing.assert_close(module(*arguments), reference, rtol=0, atol=0)
                compiled = torch.compile(module, backend="eager", fullgraph=True)
                torch.testing.assert_close(compiled(*arguments), reference, rtol=0, atol=0)
            print(json.dumps({"program": "where/torch_graph_and_offline", "status": "correct"}), flush=True)


def projection_tuning_usage(with_torch=False):
    """Finite source/physical choices through public Linear and fixed deployment."""
    panel_transfer_tuning_usage()
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from weft.programs import Operators, Tuning, LoadedOperators, get, save_operators
    from weft.runtime import (BuiltCandidates, Config, NativeTarget, NumericalCheck,
                              Toolchain, TuningPolicy, TuningResult, TuningWorkload)

    program = get("linear")
    # Both sets are explicitly authored. The compiler cannot enlarge the 4x4
    # dot in the first set by moving its numerical boundary across source Levels.
    # Physical LMUL is varied independently within each author binding.
    domains = (
        program.space(physical={"lmul_eighths": (32, 64)}, name="small-author"),
        program.space(source={"MC": (32,), "NC": (64,), "MR": (32,), "NR": (64,)},
                      physical={"lmul_eighths": (32, 64)}, name="wide-author"),
    )
    configs = program.configurations(domains, max_candidates=4)
    assert len(configs) == 4
    assert configs[0].source == configs[1].source
    assert configs[2].source == configs[3].source
    assert configs[0].source != configs[2].source
    assert program.options().meta == {"MC": 64, "NC": 16, "MR": 4, "NR": 4}
    toolchain = Toolchain.from_environment()
    target = NativeTarget.discover(toolchain).build_target
    policy = TuningPolicy(max_candidates=4, deadline_s=600, warmup=1, samples=3, repetitions=1)
    rng = np.random.default_rng(617)
    x = rng.normal(0, .2, (35, 129)).astype(np.float32)
    weight = rng.normal(0, .2, (67, 129)).astype(np.float32)
    y = np.full((35, 67), -77, np.float32)
    expected = x @ weight.T

    def validate(original, trial):
        if hasattr(original["X"], "detach"):
            import torch

            torch.testing.assert_close(trial["Y"], original["X"] @ original["Weight"].T,
                                       rtol=2e-3, atol=1e-4)
        else:
            np.testing.assert_allclose(trial["Y"], original["X"] @ original["Weight"].T,
                                       rtol=2e-3, atol=1e-4)

    with TemporaryDirectory(prefix="weft-linear-tuning-") as temporary:
        directory = Path(temporary)
        built = program.build_candidates(configs, target=target, toolchain=toolchain,
                                          cache=directory / "build-cache", max_candidates=4)
        saved = built.save(directory / "choices")
        assert saved.native and len(saved.artifacts) == 4
        assert not saved.rejections
        prebuilt = BuiltCandidates.load(directory / "choices")
        # No system compiler is needed from this point onward. Weft's native
        # target query still checks ISA, VLEN, affinity and vector state.
        no_cc = Toolchain(toolchain.compiler, ("/unavailable-system-compiler",))
        workload = TuningWorkload({"X": x, "Weight": weight, "Y": y},
            checks={"Y": NumericalCheck(expected, rtol=2e-3, atol=1e-4)},
            validation_key="public-linear-f32-atol1e-4-rtol2e-3")
        result = workload.run(prebuilt, policy=policy, toolchain=no_cc,
            cache=directory / "selection-cache", selection=directory / "selection.json",
            artifact=directory / "fixed-linear")
        np.testing.assert_array_equal(y, -77)
        assert result.selected.validated
        assert len(result.source_groups) == 2
        assert result.compare(configs[0], configs[1])["kind"] == "physical-binding"
        assert result.compare(configs[0], configs[2])["kind"] == "source-binding"
        restored = TuningResult.load(directory / "selection.json")
        assert prebuilt.fixed(restored).options == result.winner.options
        with prebuilt.fixed(restored).load(toolchain=no_cc) as fixed:
            fixed(X=x, Weight=weight, Y=y)
            check("finite_linear/fixed", y, expected)
            # One selected executable continues to accept dynamic shapes,
            # including GEMV, empty token lists and noncontiguous tail views.
            # It does not claim to be the fastest choice for those new workloads.
            for m, n, k in ((1, 17, 33), (3, 19, 65), (0, 17, 33)):
                a = rng.normal(0, .2, (m, 2 * k)).astype(np.float32)[:, ::2]
                w = rng.normal(0, .2, (n, 2 * k)).astype(np.float32)[:, ::2]
                backing = np.full((m, 2 * n), 43, np.float32)
                output = backing[:, ::2]
                fixed(X=a, Weight=w, Y=output)
                check(f"finite_linear/fixed/{m}-{n}-{k}", output, a @ w.T)
                np.testing.assert_array_equal(backing[:, 1::2], 43)

        recipe = Tuning.from_candidates(prebuilt, policy=policy, validator=validate,
                                        validation_key="public-linear-f32-atol1e-4-rtol2e-3")
        with Operators(tuning={"linear": recipe}, toolchain=no_cc,
                       cache=directory / "operator-cache") as operators:
            y.fill(-77)
            chosen = operators.tune("linear", X=x, Weight=weight, Y=y)
            np.testing.assert_array_equal(y, -77)
            check("finite_linear/operators", operators.linear(x, weight, out=y), expected)
            package = save_operators(operators, directory / "programs", selections={"linear": chosen})
            with LoadedOperators(package) as loaded:
                check("finite_linear/program_package", loaded.linear(x, weight), expected)
                assert loaded.selected_options("linear") == chosen.winner.options
                assert not loaded.tuning_results

        if with_torch:
            import torch
            from torch._subclasses.fake_tensor import FakeTensorMode
            from weft.programs.torch import Library, Linear

            tx, tw = torch.from_numpy(x.copy()), torch.from_numpy(weight.copy())
            reference = tx @ tw.T
            library = Library("weft_finite_linear", tuning={"linear": recipe}, toolchain=no_cc,
                              cache=directory / "torch-cache")
            try:
                module = Linear(tw, library=library)
                kernel = library.kernel("linear").kernel
                assert not kernel.selections
                with FakeTensorMode(allow_non_fake_inputs=True) as mode:
                    fake = module(mode.from_tensor(tx))
                    assert tuple(fake.shape) == tuple(reference.shape)
                assert not kernel.selections
                with torch.no_grad():
                    torch.testing.assert_close(module(tx), reference, rtol=2e-3, atol=1e-4)
                    graph = torch.compile(module, backend="eager", fullgraph=True)
                    torch.testing.assert_close(graph(tx), reference, rtol=2e-3, atol=1e-4)
                assert kernel.last_result.selected.validated
            finally:
                library.close()
        print(json.dumps({"program": "finite_linear", "status": "correct",
                          "winner": result.winner.to_dict(), "source_groups": len(result.source_groups),
                          "prebuilt_tuning": True, "fixed_deployment": True,
                          "torch": with_torch}), flush=True)


def tuning_usage(with_torch=False):
    from dataclasses import replace
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from weft.programs import Operators, Tuning, LoadedOperators, ProgramPackageBuilder, save_operators, get
    from weft.runtime import Config, TuningPolicy, TuningResult, TuningError

    rng = np.random.default_rng(421)
    policy = TuningPolicy(deadline_s=600, warmup=1, samples=3, repetitions=2)

    def validate_product(original, trial):
        x, weight = np.asarray(original["X"]), np.asarray(original["Weight"])
        expected = x @ weight.T.copy().T
        np.testing.assert_allclose(trial["Y"], expected, atol=1e-4, rtol=2e-3)

    def validate_add(original, trial):
        expected = np.asarray(original["X"]) + np.asarray(original["Residual"])
        np.testing.assert_allclose(trial["Y"], expected, atol=1e-4, rtol=2e-3)

    def validate_softmax(original, trial):
        values = np.asarray(original["X"])
        expected = np.exp(values - values.max(axis=-1, keepdims=True))
        expected /= expected.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(trial["Y"], expected, atol=1e-4, rtol=2e-3)

    product = Tuning(tuple(Config.from_options(replace(get("matmul").options(),
                           contraction_column_copies=copies, panel_transfer_fields=fields,
                           contraction_panel_k=panel, contraction_k_steps=steps),
                           name=f"columns{copies}-fields{fields}-k{panel}-steps{steps}")
                           for copies, fields, panel, steps in ((1, 1, 64, 1), (2, 4, 128, 2), (4, 8, 256, 4))), policy=policy,
                     validator=validate_product, validation_key="public-matmul-f32-atol1e-4-rtol2e-3")
    add = Tuning(tuple(Config.from_options(get("add").options(BLOCK=block), name=f"block{block}")
                       for block in (16, 64)), policy=policy,
                 validator=validate_add, validation_key="public-add-f32-atol1e-4-rtol2e-3")
    row = Tuning(tuple(Config(meta={}, lmul_eighths=lmul, private_stack_bytes=262144, name=f"lmul{lmul}")
                       for lmul in (8, 16)), policy=policy,
                 validator=validate_softmax, validation_key="public-softmax-f32-atol1e-4-rtol2e-3")
    x, weight = random((35, 129), rng), random((67, 129), rng).T
    out = np.full((35, 67), 37, dtype=np.float32)
    arguments = {"X": x, "Weight": weight, "Y": out}
    with TemporaryDirectory(prefix="weft-program-tuning-") as directory:
        directory = Path(directory)
        with Operators(tuning={"matmul": product, "add": add}) as operators:
            cold = operators.kernel("matmul")
            assert not cold.selections and not cold.artifacts
            try:
                save_operators(operators, directory / "unselected")
            except ValueError:
                pass
            else:
                raise AssertionError("an unselected program must not be exported")
            assert not (directory / "unselected").exists()
            result = operators.tune("matmul", **arguments)
            np.testing.assert_array_equal(out, 37)
            assert len(result.candidates) == 3 and all(candidate.status == "measured" for candidate in result.candidates)
            assert all(candidate.validated for candidate in result.candidates if candidate.status == "measured")
            kernel = operators.kernel("matmul")
            assert len(kernel.selections) == 1 and len(kernel.artifacts) == 1
            physical = kernel.artifacts[0].physical_operations
            assert physical.matching(engine="rvv")
            assert any(item.layouts for item in physical.operations)
            before = kernel.selections
            operators.matmul(x, weight, out=out)
            check("tuning/matmul", out, x @ weight.T.copy().T)
            assert kernel.selections == before
            result.save(directory / "selection.json")
            saved = TuningResult.load(directory / "selection.json")
            assert operators.selected_options("matmul", selection=saved) == result.winner.options

            # Tune an exact input/output alias. The tuning trials must preserve
            # both the caller's state and the untouched gaps in its storage.
            storage = random((3, 66), rng)
            state = storage[:, ::2]
            residual = random(state.shape, rng)
            original = storage.copy()
            add_arguments = {"X": state, "Residual": residual, "Y": state}
            addition = operators.tune("add", **add_arguments)
            np.testing.assert_array_equal(storage, original)
            operators.add(state, residual, out=state)
            check("tuning/alias_once", state, original[:, ::2] + residual)
            np.testing.assert_array_equal(storage[:, 1::2], original[:, 1::2])

            smaller_x, smaller_w = random((1, 33), rng), random((33, 17), rng)
            check("tuning/second_workload", operators.matmul(smaller_x, smaller_w), smaller_x @ smaller_w)
            assert len(kernel.selections) >= 2
            if len(kernel.artifacts) > 1:
                try:
                    save_operators(operators, directory / "ambiguous")
                except ValueError:
                    pass
                else:
                    raise AssertionError("different workload winners need an explicit deployment choice")
                assert not (directory / "ambiguous").exists()

            package = save_operators(operators, directory / "selected-programs",
                                     selections={"matmul": saved, "add": addition})
            with LoadedOperators(package) as loaded:
                check("tuning/offline", loaded.matmul(x, weight), x @ weight.T.copy().T)
                check("tuning/offline_add", loaded.add(state, residual), state + residual)
                assert loaded.selected_options("matmul") == result.winner.options
                assert loaded.kernel("matmul").physical_operations == physical
                assert not loaded.tuning_results
            artifact = kernel.artifacts[0]
            wrong_program = ProgramPackageBuilder(artifact.target)
            try:
                wrong_program.add_selection("linear", saved)
            except ValueError:
                pass
            else:
                raise AssertionError("a matrix layout selection must not bind another author program")
            assert not wrong_program.roles
            rebuilt = (ProgramPackageBuilder(artifact.target).add_selection("matmul", saved)
                       .build(directory / "fixed-build"))
            with LoadedOperators(rebuilt) as loaded:
                check("tuning/fixed_build", loaded.matmul(x, weight), x @ weight.T.copy().T)
                assert loaded.kernel("matmul").physical_operations == physical
            print(json.dumps({"program": "tuning/selection", "winner": result.winner.name,
                              "static_selected_operations": physical.selected_operation_count,
                              "validated": True, "candidates": [item.to_dict() for item in result.candidates]}), flush=True)

        # A new owner sees the same persisted workload selection. The numeric
        # oracle and timing policy are part of the record identity, not globals.
        with Operators(tuning={"matmul": product}) as again:
            repeated = again.tune("matmul", **arguments)
            assert repeated.cache_hit
            assert repeated.winner == saved.winner
            check("tuning/reopened", again.matmul(x, weight), x @ weight.T.copy().T)

        # Required ABI dimensions are fixed separately from the candidate list.
        # A single recipe produces two distinct package roles and never tunes
        # the public width of a call as though it were a physical parameter.
        with Operators(tuning={"softmax": row}) as operators:
            widths = {}
            for width in (33, 65):
                values = random((2, width * 2), rng)[:, ::2]
                result = operators.softmax(values)
                expected = np.exp(values - values.max(axis=-1, keepdims=True))
                expected /= expected.sum(axis=-1, keepdims=True)
                check(f"tuning/required_width/{width}", result, expected)
                widths[width] = (values, expected)
            assert len(operators.tuning_results) == 2
            assert {dict(key[1])["N"] for key in operators.tuning_results} == {33, 65}
            package = save_operators(operators, directory / "row-specializations")
            assert set(package.bindings["weft.programs"]) == {"softmax/N=33", "softmax/N=65"}
            with LoadedOperators(package) as loaded:
                for width, (values, expected) in widths.items():
                    check(f"tuning/offline_width/{width}", loaded.softmax(values), expected)

        # A one-candidate selection without an oracle remains explicitly
        # unvalidated; successful invocation is checked separately below.
        unchecked = Tuning((Config.from_options(get("add").options(BLOCK=32), name="fixed"),), policy=policy)
        with Operators(tuning={"add": unchecked}) as operators:
            a, b = random((2, 33), rng), random((2, 33), rng)
            result = operators.tune("add", X=a, Residual=b)
            assert result.candidates[0].status == "single-legal"
            assert not result.candidates[0].validated and result.candidates[0].median_ns is None
            check("tuning/unvalidated_record", operators.add(a, b), a + b)

        # Failed validation is also an executed private trial. Public ownership
        # and export must not quietly acquire a winner on that exception path.
        def reject_trial(original, trial):
            validate_add(original, trial)
            raise AssertionError("application rejected this candidate")

        rejecting = Tuning(add.configs, policy=policy, validator=reject_trial,
                           validation_key="public-add-rejected-application-contract")
        with Operators(tuning={"add": rejecting}) as operators:
            state, residual = random((2, 33), rng), random((2, 33), rng)
            unchanged = state.copy()
            try:
                operators.tune("add", X=state, Residual=residual, Y=state)
            except TuningError as error:
                assert len(error.candidates) == len(add.configs)
                assert all(candidate.status == "validation-rejected" for candidate in error.candidates)
            else:
                raise AssertionError("a rejected search must not execute a default configuration")
            np.testing.assert_array_equal(state, unchanged)
            assert not operators.artifacts
            assert all(not records for records in operators.tuning_results.values())
            try:
                save_operators(operators, directory / "rejected")
            except ValueError:
                pass
            else:
                raise AssertionError("a failed search must not produce a deployable package")
            assert not (directory / "rejected").exists()
            check("tuning/rejection_preserves_state", state, unchanged)

        if with_torch:
            import torch
            from weft.programs.torch import Library, Matmul

            tx, tw = torch.from_numpy(x), torch.from_numpy(weight)
            output = torch.full((35, 67), 37, dtype=torch.float32)
            version = output._version
            with Library("weft_use_tuned", tuning={"matmul": product}) as library:
                module = Matmul(tw, library=library)
                chosen = library.tune("matmul", X=tx, Weight=tw, Y=output)
                torch.testing.assert_close(output, torch.full_like(output, 37), atol=0, rtol=0)
                assert output._version == version
                check("tuning/torch", module(tx), x @ weight.T.copy().T)
                captured = torch.compile(module, backend="eager", fullgraph=True)
                check("tuning/torch_capture", captured(tx), x @ weight.T.copy().T)
                exported = torch.export.export(module, (tx,)).module()
                check("tuning/torch_export", exported(tx), x @ weight.T.copy().T)
                package = library.save(directory / "torch-selected", selections={"matmul": chosen})
                assert library.tuning_results
            with Library.from_package("weft_use_fixed", package) as fixed:
                module = Matmul(tw, library=fixed)
                fixed.materialize()
                check("tuning/torch_offline", module(tx), x @ weight.T.copy().T)
                captured = torch.compile(module, backend="eager", fullgraph=True)
                check("tuning/torch_offline_capture", captured(tx), x @ weight.T.copy().T)
                assert not fixed.tuning_results


def matmul_usage(with_torch=False):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from weft.programs import Operators, LoadedOperators, save_operators

    rng = np.random.default_rng(419)
    with Operators() as operators:
        for batch, m, n, k in ((0, 5, 7, 13), (2, 0, 7, 13), (2, 3, 0, 5), (2, 3, 7, 0),
                               (2, 5, 17, 33), (2, 35, 67, 259)):
            x = random((batch, m, k * 2), rng)[..., ::2]
            weight = random((batch, k, n * 2), rng)[..., ::2]
            storage = np.full((batch, m, n * 2 + 2), 37, dtype=np.float32)
            y = storage[..., 1:-1:2]
            actual = operators.bmm(x, weight, out=y)
            assert actual is y
            check(f"bmm/{batch}/{m}/{n}/{k}", actual, x @ weight)
            np.testing.assert_array_equal(storage[..., ::2], 37)
            np.testing.assert_array_equal(storage[..., -1], 37)
        # A broadcast read borrows the original storage; each batch still has
        # independent writable output and its own closed contraction.
        x = random((3, 5, 17), rng)
        weight = np.broadcast_to(random((1, 17, 9), rng), (3, 17, 9))
        check("bmm/broadcast_read", operators.bmm(x, weight), x @ weight)
        for m, n, k in ((0, 7, 13), (3, 0, 5), (3, 7, 0), (1, 1, 1), (5, 7, 33), (35, 67, 65)):
            x = random((2, m, 2 * k), rng)[..., ::2]
            weight = random((k, 2 * n), rng)[:, ::2]
            bias = random((2 * n,), rng)[::2]
            storage = np.full((2, m, 2 * n + 2), 37, dtype=np.float32)
            y = storage[..., 1:-1:2]
            expected = x @ weight
            actual = operators.matmul(x, weight, out=y)
            assert actual is y
            check(f"matmul/strided/{m}/{n}/{k}", actual, expected)
            np.testing.assert_array_equal(storage[..., ::2], 37)
            np.testing.assert_array_equal(storage[..., -1], 37)
            check(f"matmul/bias/{m}/{n}/{k}", operators.matmul(x, weight, bias), expected + bias)
        x = random((35, 259), rng)
        weight = random((67, 259), rng).T
        bias = random((67,), rng)
        storage = np.full((35, 136), 37, dtype=np.float32)
        output = storage[:, 1:135:2]
        expected = x @ weight + bias
        operators.matmul(x, weight, bias, out=output)
        check("matmul/column_major_weight/panel_and_tail", output, expected)
        np.testing.assert_array_equal(storage[:, ::2], 37)
        np.testing.assert_array_equal(storage[:, -1], 37)
        x = random((3, 17), rng)
        weight = random((17, 9), rng)
        bias = random((9,), rng)
        with TemporaryDirectory(prefix="weft-matmul-package-") as directory:
            package = save_operators(operators, Path(directory) / "programs")
            with LoadedOperators(package) as loaded:
                check("matmul/package", loaded.matmul(x, weight), x @ weight)
                check("matmul/package_bias", loaded.matmul(x, weight, bias), x @ weight + bias)
                bx, bw = random((2, 3, 17), rng), random((2, 17, 9), rng)
                check("bmm/package", loaded.bmm(bx, bw), bx @ bw)
    if with_torch:
        import torch
        from weft.programs.torch import Library, Matmul, BatchedMatmul

        value = torch.from_numpy(random((2, 3, 17), rng))
        right = torch.from_numpy(weight)
        offset = torch.from_numpy(bias)
        with Library("weft_use_matmul") as library:
            left = torch.from_numpy(random((2, 5, 17), rng))
            batch_weight = torch.from_numpy(random((2, 17, 9), rng))
            batched = BatchedMatmul(library=library)
            expected = torch.bmm(left, batch_weight).numpy()
            check("bmm/torch", batched(left, batch_weight), expected)
            captured = torch.compile(batched, backend="eager", fullgraph=True)
            check("bmm/torch_capture", captured(left, batch_weight), expected)
            exported = torch.export.export(batched, (left, batch_weight)).module()
            check("bmm/torch_export", exported(left, batch_weight), expected)
            for name, addition in (("plain", None), ("bias", offset)):
                module = Matmul(right, addition, library=library)
                expected = value @ right
                if addition is not None:
                    expected = expected + addition
                check("matmul/torch/" + name, module(value), expected.numpy())
                captured = torch.compile(module, backend="eager", fullgraph=True)
                check("matmul/torch_capture/" + name, captured(value), expected.numpy())
                exported = torch.export.export(module, (value,)).module()
                check("matmul/torch_export/" + name, exported(value), expected.numpy())


def projection_activation_usage(with_torch=False):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from weft.programs import Operators, LoadedOperators, save_operators

    rng = np.random.default_rng(420)
    with Operators() as operators:
        for m, n, k in ((0, 7, 13), (1, 7, 1), (1, 67, 259), (35, 1, 259),
                        (5, 17, 33), (35, 67, 259)):
            x = random((2, m, 2 * k), rng)[..., ::2]
            weight = random((n, 2 * k), rng)[:, ::2]
            bias = random((2 * n,), rng)[::2]
            for label, offset in (("plain", None), ("bias", bias)):
                storage = np.full((2, m, 2 * n + 2), 37, dtype=np.float32)
                out = storage[..., 1:-1:2]
                expected = x @ weight.T
                if offset is not None:
                    expected += offset
                expected = silu_reference(expected)
                actual = operators.linear_silu(x, weight, offset, out=out)
                assert actual is out
                check(f"linear_silu/{label}/{m}/{n}/{k}", actual, expected)
                np.testing.assert_array_equal(storage[..., ::2], 37)
                np.testing.assert_array_equal(storage[..., -1], 37)
        # Raw kernels admit an empty contraction. The public model composition
        # API requires positive feature counts; exercise this IR boundary here
        # through the already materialized kernel without changing that API.
        for name in ("linear_silu", "linear_bias_silu"):
            x, weight = np.empty((33, 0), dtype=np.float32), np.empty((65, 0), dtype=np.float32)
            y = np.empty((33, 65), dtype=np.float32)
            arguments = dict(X=x, Weight=weight, Y=y)
            expected = np.zeros_like(y)
            if name == "linear_bias_silu":
                arguments["Bias"] = random((65,), rng)
                expected += arguments["Bias"]
            operators.kernel(name)(**arguments)
            check(name + "/empty_contraction", y, silu_reference(expected))
        x = random((35, 259), rng)
        weight = random((259, 67), rng).T
        bias = random((67,), rng)
        check("linear_silu/column_major_weight/panel_and_tail", operators.linear_silu(x, weight, bias),
              silu_reference(x @ weight.T + bias))
        # One compiled program selects both unit and nonunit scalar packets.
        # Odd output and K sizes keep the packed panel's final issue live.
        for x_step, weight_step in ((1, 1), (1, 2), (2, 1)):
            x = random((35, 259 * x_step), rng)[:, ::x_step]
            weight = random((67, 259 * weight_step), rng)[:, ::weight_step]
            bias = random((67,), rng)
            storage = np.full((35, 136), 37, dtype=np.float32)
            out = storage[:, 1:135:2]
            check(f"linear_silu/packet_strides/{x_step}/{weight_step}",
                  operators.linear_silu(x, weight, bias, out=out), silu_reference(x @ weight.T + bias))
            np.testing.assert_array_equal(storage[:, ::2], 37)
            np.testing.assert_array_equal(storage[:, -1], 37)
        # Stable activation remains well-defined on both exponent extremes.
        x = np.array([[-90, -20, -1, -0.0, 0.0, 1, 20, 90]], dtype=np.float32)
        weight = np.eye(x.shape[1], dtype=np.float32)
        check("linear_silu/exponent_range", operators.linear_silu(x, weight), silu_reference(x))
        with TemporaryDirectory(prefix="weft-projection-activation-") as directory:
            package = save_operators(operators, Path(directory) / "programs")
            with LoadedOperators(package) as loaded:
                check("linear_silu/package", loaded.linear_silu(x, weight), silu_reference(x))
                bias = np.linspace(-1, 1, 8, dtype=np.float32)
                check("linear_silu/package_bias", loaded.linear_silu(x, weight, bias), silu_reference(x + bias))
    if with_torch:
        import torch
        from weft.programs.torch import Library, LinearSiLU

        value = torch.from_numpy(random((2, 3, 17), rng))
        weight = torch.from_numpy(random((9, 17), rng))
        bias = torch.from_numpy(random((9,), rng))
        with Library("weft_use_linear_silu") as library:
            for label, offset in (("plain", None), ("bias", bias)):
                module = LinearSiLU(weight, offset, library=library)
                expected = torch.nn.functional.silu(torch.nn.functional.linear(value, weight, offset))
                check("linear_silu/torch/" + label, module(value), expected.numpy())
                captured = torch.compile(module, backend="eager", fullgraph=True)
                check("linear_silu/torch_capture/" + label, captured(value), expected.numpy())
                exported = torch.export.export(module, (value,)).module()
                check("linear_silu/torch_export/" + label, exported(value), expected.numpy())


def closed_contraction_usage():
    from weft.programs import get

    # The two-element cancellation must finish before the author state add.
    # Empty, single-iteration and uneven final windows also exercise pipeline
    # guards and the binding of each iteration's physical descriptor extents.
    values = np.array([1, 0, 1, 1, 1], dtype=np.float32)
    coefficients = np.array([1, 0, 1e8, -1e8, 2], dtype=np.float32)
    for depth, lmul in ((1, 8), (2, 4)):
        with get("linear_tile").jit(BLOCK=2, MROWS=1, NROWS=4,
                                   options=CompileOptions(pipeline_depth=depth, lmul_eighths=lmul)) as kernel:
            for terms, expected in ((0, 0), (1, 1), (4, 1), (5, 3)):
                x = values[:terms].reshape(1, terms)
                weight = np.tile(coefficients[:terms], (7, 1))
                y = np.empty((1, 7), dtype=np.float32)
                kernel(X=x, Weight=weight, Y=y)
                check(f"linear_tile/closed_dot_boundary/k={terms}/depth={depth}/lmul={lmul}",
                      y, np.full_like(y, expected))


def contraction_issue_usage():
    from weft.programs import get

    rng = np.random.default_rng(421)
    options = CompileOptions(lmul_eighths=64)
    # One fixed program exercises complete panels, replica tails, short K,
    # zero-trip loops and nonunit scalar supplies. Output guard storage exposes
    # an incorrect full-vector proof even when the numerical prefix is right.
    for name in ("linear", "linear_bias", "linear_silu", "linear_bias_silu"):
        bindings = {"MC": 32, "NC": 64}
        if not name.endswith("silu"):
            bindings.update(MR=32, NR=64)
        with get(name).jit(**bindings, options=options) as kernel:
            for m, n, k, stride in ((32, 64, 128, 1), (32, 64, 128, 0), (35, 67, 259, 2),
                                     (17, 31, 3, 1), (33, 65, 0, 1)):
                if stride == 0:
                    x = random((k, m), rng).T
                    weight = random((k, n), rng).T
                else:
                    x = random((m, k * stride), rng)[:, ::stride]
                    weight = random((n, k * stride), rng)[:, ::stride]
                guard = np.full((m + 2, n * 2 + 2), 37, dtype=np.float32)
                output = guard[1:-1, 1:-1:2]
                arguments = dict(X=x, Weight=weight, Y=output)
                expected = x @ weight.T
                if "bias" in name:
                    arguments["Bias"] = random((n,), rng)
                    expected += arguments["Bias"]
                if name.endswith("silu"):
                    expected = silu_reference(expected)
                kernel(**arguments)
                check(f"{name}/wide_issue/{m}/{n}/{k}/stride={stride}", output, expected)
                np.testing.assert_array_equal(guard[:, ::2], 37)
                np.testing.assert_array_equal(guard[:, -1], 37)
                np.testing.assert_array_equal(guard[[0, -1]], 37)


def panel_transfer_usage():
    """Prepare, inspect and deploy one physical transfer selection."""
    from weft.programs import get
    from weft.runtime import NativeTarget

    definition = get("linear").definition
    options = CompileOptions(meta={"MC": 32, "NC": 64, "MR": 32, "NR": 64}, lmul_eighths=32)
    selected = weft.prepare(definition, target=NativeTarget.discover().build_target, options=options, checkpoint="panel-transfers")
    choices = selected.selections()
    if not any(kernel.transfers for kernel in choices.kernels):
        raise AssertionError("wide linear did not retain its selected panel transfers")
    artifact = selected.complete()
    if artifact.compilation.entry != "panel-transfer-input" or '"weft_riscv.dense_panel_copy"' in artifact.riscv_ir:
        raise AssertionError("selected transfer was not consumed by the ordinary compilation suffix")
    rng = np.random.default_rng(424)
    with artifact.load() as kernel:
        x = random((35, 259), rng)
        weight = random((67, 259), rng)
        y = np.empty((35, 67), dtype=np.float32)
        kernel(X=x, Weight=weight, Y=y)
        check("linear/selected_panel_transfer", y, x @ weight.T)
    packets = [operation for operation in artifact.physical_operations.operations if operation.scalar_packet]
    print(json.dumps({"program": "linear/selected_supply", "status": "native-correct",
                      "panel_transfers": sum(len(kernel.transfers) for kernel in choices.kernels),
                      "scalar_packets": sum(operation.occurrences for operation in packets),
                      "maximum_packet_fpr_frontier": max((operation.scalar_packet.live_registers
                                                           for operation in packets), default=0)}), flush=True)

def panel_transfer_tuning_usage():
    """Tune a target-declared memory structure with one frozen author program."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from weft.programs import get
    from weft.runtime import (Artifact, BuiltCandidates, ConfigSpace, NativeTarget,
                              NumericalCheck, Toolchain, TuningPolicy, TuningResult, TuningWorkload)

    program = get("linear")
    options = CompileOptions(meta={"MC": 32, "NC": 64, "MR": 32, "NR": 64}, lmul_eighths=32)
    domain = ConfigSpace(base=options,
        physical={"panel_transfer_strategy": ("issue-work", "max-fields", "single-field")}, name="panel-memory")
    configs = program.configurations(domain, max_candidates=3)
    for config in configs[1:]:
        changes = config.changes_from(configs[0])
        if changes["source"] or set(changes["physical"]) != {"panel_transfer_strategy"}:
            raise AssertionError("memory strategy tuning changed the author program or another physical binding")
    toolchain = Toolchain.from_environment()
    target = NativeTarget.discover(toolchain).build_target
    rng = np.random.default_rng(426)
    x = random((35, 131), rng)
    weight = random((67, 131), rng)
    output = np.full((35, 67), 37, dtype=np.float32)
    expected = x @ weight.T
    workload = TuningWorkload({"X": x, "Weight": weight, "Y": output},
        checks={"Y": NumericalCheck(expected, rtol=2e-3, atol=1e-4)},
        validation_key="panel-transfer-f32-atol1e-4-rtol2e-3")
    policy = TuningPolicy(max_candidates=3, deadline_s=300, warmup=1, samples=3, repetitions=1)
    with TemporaryDirectory(prefix="weft-panel-memory-") as temporary:
        directory = Path(temporary)
        built = program.build_candidates(configs, target=target, toolchain=toolchain,
                                          cache=directory / "cache", max_candidates=3)
        if built.rejections or len(built.artifacts) != 3:
            raise AssertionError("a supported panel transfer strategy failed native compilation")
        if len({artifact.source for artifact in built.artifacts}) != 3:
            raise AssertionError("finite memory strategies did not form three real emitted programs")
        built.save(directory / "choices")
        deployed = BuiltCandidates.load(directory / "choices")
        unavailable_cc = Toolchain(toolchain.compiler, ("/unavailable-system-compiler",))
        result = workload.run(deployed, policy=policy, toolchain=unavailable_cc,
            cache=directory / "selection-cache", selection=directory / "selected.json",
            artifact=directory / "selected-linear")
        np.testing.assert_array_equal(output, 37)
        if not result.selected.validated or len(result.source_groups) != 1:
            raise AssertionError("memory strategy tuning lost its single-source numerical contract")
        selected = TuningResult.load(directory / "selected.json")
        if selected.winner != result.winner:
            raise AssertionError("saved memory tuning selection changed its complete binding")
        fixed = Artifact(directory / "selected-linear")
        if fixed.options.panel_transfer_strategy != result.winner.options.panel_transfer_strategy:
            raise AssertionError("fixed deployment changed the measured memory structure")
        with fixed.load(toolchain=unavailable_cc) as kernel:
            kernel(X=x, Weight=weight, Y=output)
        check("linear/panel_memory_tuning", output, expected)
        print(json.dumps({"program": "linear/panel_memory_tuning", "status": "native-correct",
                          "source_groups": len(result.source_groups), "native_candidates": len(deployed.artifacts),
                          "selected_strategy": result.winner.options.panel_transfer_strategy}), flush=True)


def projection_usage(with_torch=False):
    matmul_usage(with_torch)
    projection_activation_usage(with_torch)
    contraction_issue_usage()
    panel_transfer_usage()
    from weft.programs import get, Operators

    rng = np.random.default_rng(417)
    for rows, depth, unroll, lmul in ((4, 1, 1, 64), (32, 1, 1, 8), (32, 2, 1, 8), (4, 2, 3, 8)):
        options = CompileOptions(pipeline_depth=depth, unroll=unroll, lmul_eighths=lmul)
        for name in ("linear", "linear_bias", "expert_linear"):
            bindings = {"ROWS": rows} if name == "expert_linear" else {"NR": rows}
            with get(name).jit(**bindings, options=options) as kernel:
                for m, n, k in ((3, 7, 1), (1, 7, 35), (5, 33, 35), (3, 65, 259), (0, 17, 33), (2, 17, 0)):
                    x = random((m, 2 * k), rng)[:, ::2]
                    expert = name == "expert_linear"
                    weight = random((4, n, 2 * k) if expert else (n, 2 * k), rng)[..., ::2]
                    destination = np.full((m + 2, 2 * n + 2), 1234, dtype=np.float32)
                    y = destination[1:-1, 1:-1:2]
                    arguments = dict(X=x, Weight=weight, Y=y)
                    if expert:
                        offsets = np.array([0, 0, m // 2, m // 2, m], dtype=np.int32)
                        arguments["Offsets"] = offsets
                        expected = np.empty((m, n), dtype=np.float32)
                        for e, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
                            expected[lo:hi] = x[lo:hi] @ weight[e].T
                    else:
                        expected = x @ weight.T
                        if name == "linear_bias":
                            arguments["Bias"] = random((2 * n,), rng)[::2]
                            expected += arguments["Bias"]
                    kernel(**arguments)
                    check(f"{name}/rows={rows}/depth={depth}/unroll={unroll}/{m}x{n}x{k}", y, expected)
                    guard = np.full_like(destination, 1234)
                    guard[1:-1, 1:-1:2] = expected
                    np.testing.assert_allclose(destination, guard, atol=1e-4, rtol=2e-3)

    closed_contraction_usage()

    for name in ("linear_tile", "linear_bias_tile"):
        for depth in (1, 2):
            with get(name).jit(options=CompileOptions(pipeline_depth=depth)) as kernel:
                for m, n, k in ((1, 1, 1), (5, 7, 35), (7, 17, 259), (0, 7, 33)):
                    x = random((m, 2 * k), rng)[:, ::2]
                    weight = random((n, 2 * k), rng)[:, ::2]
                    y = np.empty((m, n), dtype=np.float32)
                    arguments = dict(X=x, Weight=weight, Y=y)
                    expected = x @ weight.T
                    if name == "linear_bias_tile":
                        arguments["Bias"] = random((n,), rng)
                        expected += arguments["Bias"]
                    kernel(**arguments)
                    check(f"{name}/depth={depth}/{m}x{n}x{k}", y, expected)
    with Operators() as operators:
        x, weight, bias = random((2, 3, 35), rng), random((7, 35), rng), random((7,), rng)
        check("linear_tile/public", operators.linear_tile(x, weight, bias), x @ weight.T + bias)

    if with_torch:
        import torch
        from weft.programs.torch import Library

        with Library("weft_projection_use") as library:
            operation = library.kernel("linear_bias_tile").op

            class Projection(torch.nn.Module):
                def forward(self, x, weight, bias):
                    return operation(x, weight, bias)

            args = tuple(torch.from_numpy(value) for value in (x.reshape(-1, 35), weight, bias))
            expected = (x @ weight.T + bias).reshape(-1, 7)
            module = Projection()
            check("linear_tile/torch", module(*args).numpy(), expected)
            check("linear_tile/torch_compile", torch.compile(module, backend="eager", fullgraph=True)(*args).numpy(), expected)
            check("linear_tile/torch_export", torch.export.export(module, args).module()(*args).numpy(), expected)


class GGMLCall:
    """The existing GGML CPU graph baseline with the operator's actual inputs."""

    @staticmethod
    def aligned_inputs(directory, arguments):
        import ctypes as c
        from pathlib import Path

        api = c.CDLL(str(Path(directory).resolve() / "libggml-cpu.so"))
        api.ggml_backend_cpu_buffer_type.restype = c.c_void_p
        api.ggml_backend_buft_get_alignment.argtypes = [c.c_void_p]
        api.ggml_backend_buft_get_alignment.restype = c.c_size_t
        alignment = api.ggml_backend_buft_get_alignment(api.ggml_backend_cpu_buffer_type())
        if alignment <= 0:
            raise RuntimeError("GGML CPU buffer returned an invalid alignment")
        result = dict(arguments)
        for name, array in arguments.items():
            if name == "Y":
                continue
            if not isinstance(array, np.ndarray) or array.dtype != np.float32 or not array.flags.c_contiguous:
                raise ValueError("GGML comparison requires contiguous f32 inputs before alignment")
            if array.ctypes.data % alignment == 0:
                continue
            storage = np.empty(array.nbytes + alignment - 1, dtype=np.uint8)
            start = -storage.ctypes.data % alignment
            aligned = storage[start:start + array.nbytes].view(array.dtype).reshape(array.shape)
            aligned[...] = array
            result[name] = aligned
        return result

    def __init__(self, directory, name, arguments):
        import ctypes as c
        from pathlib import Path

        supported = {"linear", "linear_bias", "linear_silu", "linear_bias_silu", "linear_tile", "linear_bias_tile", "matmul", "matmul_bias", "rms_norm", "layer_norm",
                     "residual_rms_norm", "rms_norm_row", "layer_norm_row", "softmax", "softmax_row", "silu", "swiglu",
                     "logarithm", "softplus"}
        if name not in supported:
            raise NotImplementedError(f"no matching GGML graph for {name}")
        self.library = str(Path(directory).resolve() / "libggml-cpu.so")
        self.api = c.CDLL(self.library)
        self.context = self.backend = self.buffer = None
        self.input_buffers, self.inputs = [], {}
        pointer, integer, size = c.c_void_p, c.c_int, c.c_size_t

        def bind(symbol, result, *args):
            function = getattr(self.api, symbol)
            function.restype, function.argtypes = result, args
            return function

        class Init(c.Structure):
            _fields_ = [("mem_size", size), ("mem_buffer", pointer), ("no_alloc", c.c_bool)]

        initialize = bind("ggml_init", pointer, Init)
        backend = bind("ggml_backend_cpu_init", pointer)
        threads = bind("ggml_backend_cpu_set_n_threads", None, pointer, integer)
        new_tensor = bind("ggml_new_tensor", pointer, pointer, integer, integer, c.POINTER(c.c_int64))
        new_graph = bind("ggml_new_graph", pointer, pointer)
        expand = bind("ggml_build_forward_expand", None, pointer, pointer)
        allocate = bind("ggml_backend_alloc_ctx_tensors", pointer, pointer, pointer)
        wrap = bind("ggml_backend_cpu_buffer_from_ptr", pointer, pointer, size)
        tensor_alloc = bind("ggml_backend_tensor_alloc", integer, pointer, pointer, pointer)
        self.tensor_get = bind("ggml_backend_tensor_get", None, pointer, pointer, size, size)
        self.compute = bind("ggml_backend_graph_compute", integer, pointer, pointer)
        self.synchronize = bind("ggml_backend_synchronize", None, pointer)
        self.free_buffer = bind("ggml_backend_buffer_free", None, pointer)
        self.free_backend = bind("ggml_backend_free", None, pointer)
        self.free_context = bind("ggml_free", None, pointer)
        mul = bind("ggml_mul", pointer, pointer, pointer, pointer)
        add = bind("ggml_add", pointer, pointer, pointer, pointer)
        dot = bind("ggml_mul_mat", pointer, pointer, pointer, pointer)
        transpose = bind("ggml_transpose", pointer, pointer, pointer) if name.startswith("matmul") else None
        contiguous = bind("ggml_cont", pointer, pointer, pointer) if name.startswith("matmul") else None
        rms = bind("ggml_rms_norm", pointer, pointer, pointer, c.c_float)
        norm = bind("ggml_norm", pointer, pointer, pointer, c.c_float)
        silu = bind("ggml_silu", pointer, pointer, pointer)
        softmax = bind("ggml_soft_max", pointer, pointer, pointer)
        unary = bind("ggml_log" if name == "logarithm" else "ggml_softplus", pointer, pointer, pointer) if name in ("logarithm", "softplus") else None
        try:
            self.context = initialize(Init(16 * 1024 * 1024, None, True))
            self.backend = backend()
            if not self.context or not self.backend:
                raise RuntimeError("GGML CPU context/backend allocation failed")
            threads(self.backend, 1)
            tensors = {}
            for key, array in arguments.items():
                if key == "Y":
                    continue
                if not isinstance(array, np.ndarray) or array.dtype != np.float32 or not array.flags.c_contiguous:
                    raise ValueError("GGML comparison requires the same contiguous f32 input arrays")
                dimensions = (c.c_int64 * array.ndim)(*reversed(array.shape))
                tensor = new_tensor(self.context, 0, array.ndim, dimensions)
                if not tensor:
                    raise RuntimeError(f"GGML tensor allocation failed for {key}")
                buffer = wrap(array.ctypes.data, array.nbytes)
                if not buffer:
                    raise RuntimeError(f"GGML could not borrow the {key} input storage")
                self.input_buffers.append(buffer)
                if tensor_alloc(buffer, tensor, array.ctypes.data) != 0:
                    raise RuntimeError(f"GGML tensor binding failed for {key}")
                tensors[key], self.inputs[key] = tensor, array
            if name.startswith(("linear", "matmul")):
                weight = contiguous(self.context, transpose(self.context, tensors["Weight"])) if transpose is not None else tensors["Weight"]
                output = dot(self.context, weight, tensors["X"])
                if "bias" in name:
                    output = add(self.context, output, tensors["Bias"])
                if name.endswith("_silu"):
                    output = silu(self.context, output)
            elif "norm" in name:
                value = tensors["X"]
                if name == "residual_rms_norm":
                    value = add(self.context, value, tensors["Residual"])
                output = (norm if name.startswith("layer") else rms)(self.context, value, 1e-5)
                output = mul(self.context, output, tensors["Weight"])
                if name.startswith("layer"):
                    output = add(self.context, output, tensors["Bias"])
            elif name.startswith("softmax"):
                output = softmax(self.context, tensors["X"])
            elif name == "swiglu":
                output = mul(self.context, silu(self.context, tensors["Gate"]), tensors["Up"])
            elif unary is not None:
                output = unary(self.context, tensors["X"])
            else:
                output = silu(self.context, tensors["X"])
            self.output, self.graph = output, new_graph(self.context)
            expand(self.graph, self.output)
            self.buffer = allocate(self.context, self.backend)
            if not self.buffer:
                raise RuntimeError("GGML graph buffer allocation failed")
            self.host_output = np.empty_like(arguments["Y"])
        except BaseException:
            self.close()
            raise

    def time_ns(self):
        from time import perf_counter_ns

        begin = perf_counter_ns()
        status = self.compute(self.backend, self.graph)
        self.synchronize(self.backend)
        elapsed = perf_counter_ns() - begin
        if status != 0:
            raise RuntimeError(f"GGML graph computation failed with status {status}")
        return elapsed

    def result(self):
        self.tensor_get(self.output, self.host_output.ctypes.data, 0, self.host_output.nbytes)
        return self.host_output

    def close(self):
        if self.buffer:
            self.free_buffer(self.buffer)
            self.buffer = None
        if self.backend:
            self.free_backend(self.backend)
            self.backend = None
        if self.context:
            self.free_context(self.context)
            self.context = None
        for buffer in self.input_buffers:
            self.free_buffer(buffer)
        self.input_buffers.clear()
        self.inputs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def artifact_comparison_contract(previous, current, *, compare_programs=False, compare_bindings=False):
    """Check what a paired run can attribute before either artifact is timed."""
    if compare_programs and compare_bindings:
        raise ValueError("choose either an author-program or a physical-binding comparison")
    target_fields = ("march", "abi", "vlen_bits")
    if any(getattr(previous.target, key) != getattr(current.target, key) for key in target_fields):
        raise ValueError("comparison requires identical ISA, ABI and VLEN")
    if previous.target.data_cache is not None and current.target.data_cache is not None and (
            previous.target.data_cache != current.target.data_cache):
        raise ValueError("comparison requires the same known cache geometry")

    same_program = (previous.path / "canonical.mlir").read_bytes() == (current.path / "canonical.mlir").read_bytes()
    same_source_bindings = previous.options.meta == current.options.meta
    same_options = previous.options == current.options
    old_options = json.loads((previous.path / "manifest.json").read_text())["options"]
    new_options = json.loads((current.path / "manifest.json").read_text())["options"]
    if not compare_programs:
        if not same_program:
            raise ValueError("compiler or physical-binding comparison requires the same canonical author program")
        if not same_source_bindings:
            raise ValueError("physical-binding comparison cannot change source auto bindings")
        if not compare_bindings and not same_options:
            raise ValueError("compiler comparison requires identical physical bindings; use --compare-bindings to measure a selected binding")

    def flags(artifact):
        return [item for item in artifact.manifest["link_command"] if not item.endswith((".c", ".so"))]

    if flags(previous) != flags(current):
        raise ValueError("comparison requires the same system compiler and flags")
    # Missing fields remain distinguishable from a recorded value. Historical
    # artifacts may predate a physical parameter; an inferred default is not an
    # observed identity and must not turn such a pair into a compiler-only claim.
    changed = {}
    for name in sorted(set(old_options) | set(new_options)):
        if name in old_options and name in new_options and old_options[name] == new_options[name]:
            continue
        change = {}
        if name in old_options:
            change["previous"] = old_options[name]
        if name in new_options:
            change["current"] = new_options[name]
        changed[name] = change
    kind = ("author-program" if not same_program else
            "source-binding-and-compiler" if not same_source_bindings else
            "compiler-and-physical-binding" if changed else "compiler")
    return {"kind": kind,
            "same_canonical_program": same_program,
            "same_source_bindings": same_source_bindings,
            "same_options": old_options == new_options,
            "same_resolved_options": same_options,
            "same_target_profile": previous.target == current.target,
            "target_profiles": {"previous": previous.target.to_dict(), "current": current.target.to_dict()},
            "new_physical_parameters": sorted(set(new_options) - set(old_options)),
            "changed_options": changed,
            "options": old_options}


def operator_usage(block, names, compare_artifacts=None, *, rows=None, tokens=128, compare_programs=False,
                   compare_bindings=False,
                   ggml_directory=None, lmul_eighths=None, pointwise_issue_copies=None,
                   contraction_column_copies=None, panel_transfer_fields=None, contraction_panel_k=None,
                   contraction_k_steps=None, pipeline_depth=None, unroll=None):
    """Measure the ordinary native entry after preparing its actual ABI call.

    This deliberately measures operators, not a model. Preparation, reference
    computation, warmup and eviction are outside the timed region. Inputs stay
    unchanged across the ten invocations. A mutable scan state is restored to
    its recorded initial value before every call, outside the timed region.
    """
    from weft.programs import get

    if compare_programs and compare_bindings:
        raise ValueError("author-program and physical-binding comparisons are mutually exclusive")
    if (compare_programs or compare_bindings) and compare_artifacts is None:
        raise ValueError("a comparison mode requires reference artifacts")

    eviction = np.zeros(64 * 1024 * 1024, dtype=np.uint8)
    previous = {}
    if compare_artifacts is not None:
        from pathlib import Path

        for line in Path(compare_artifacts).read_text().splitlines():
            if line.startswith("{"):
                record = json.loads(line)
                key = (record["program"], tuple(record["shape"])) if compare_programs else (
                    record["program"], tuple(record["shape"]), tuple(sorted(record["bindings"].items())))
                if key in previous:
                    raise ValueError(f"ambiguous comparison record for {key}")
                previous[key] = record
    selected = names or ("rms_norm", "layer_norm", "residual_rms_norm", "silu",
                         "swiglu", "softmax", "rope_split", "rope_interleaved")
    for name in selected:
        # An operator's inputs do not depend on which other operators ran first.
        rng = np.random.default_rng(415)
        initial_state = expected_state = None
        base_name = name.removesuffix("_row")
        bindings = {"BLOCK": block}
        if name in ("linear", "linear_bias", "linear_silu", "linear_bias_silu", "expert_linear", "linear_tile", "linear_bias_tile", "matmul", "matmul_bias"):
            shape = (tokens, 4096, 4096)
            x = random((shape[0], shape[2]), rng)
            expert = name == "expert_linear"
            natural = name.startswith("matmul")
            weight = random((4, shape[1], shape[2]) if expert else (shape[2], shape[1]) if natural else (shape[1], shape[2]), rng)
            arguments = dict(X=x, Weight=weight)
            if expert:
                offsets = np.array([0, 0, tokens // 2, tokens // 2, tokens], dtype=np.int32)
                arguments["Offsets"] = offsets
                expected = np.empty((tokens, shape[1]), dtype=np.float32)
                for e, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
                    expected[lo:hi] = x[lo:hi] @ weight[e].T
            else:
                # The numerical oracle is outside timing. Give NumPy a dense K
                # axis even on installations without an optimized BLAS; the
                # measured native operands keep their original [K,N] storage.
                expected = x @ (weight.T.copy().T if natural else weight.T)
                if "bias" in name:
                    arguments["Bias"] = random((shape[1],), rng)
                    expected += arguments["Bias"]
                if name.endswith("_silu"):
                    expected = silu_reference(expected)
            if natural or name.endswith("_silu"):
                bindings = dict(get(name).options().meta)
            elif name in ("linear", "linear_bias"):
                bindings = dict(get(name).options(**({"NR": rows} if rows is not None else {})).meta)
            elif expert:
                bindings = {"ROWS": 4 if rows is None else rows}
            else:
                bindings.update({"MROWS": 4, "NROWS": 4 if rows is None else rows})
            definition = get(name).definition
        elif name == "selective_scan":
            shape = (1, tokens, 1024, 16)
            batch, length, channels, state_size = shape
            arguments = {key: random((batch, length, channels), rng) for key in ("X", "Delta", "Gate")}
            arguments.update({key: random((batch, length, state_size), rng) for key in ("Bcoef", "Ccoef")})
            arguments.update({key: random((channels,), rng) for key in ("Skip", "Bias")})
            arguments["Alog"] = random((channels, state_size), rng)
            initial_state = random((batch, channels, state_size), rng)
            arguments["State"] = initial_state.copy()
            expected, expected_state = scan_reference(arguments)
            bindings = {"N": state_size}
            definition = mamba.selective_scan
        elif name.startswith("rope_"):
            shape = (128, 32, 2, 64) if name == "rope_split" else (128, 32, 64, 2)
            x = random(shape, rng)
            phase = random((128, 64), rng)
            cosine, sine = np.cos(phase), np.sin(phase)
            first = x[:, :, 0, :] if name == "rope_split" else x[:, :, :, 0]
            second = x[:, :, 1, :] if name == "rope_split" else x[:, :, :, 1]
            expected = np.stack((first * cosine[:, None] - second * sine[:, None],
                                 second * cosine[:, None] + first * sine[:, None]),
                                axis=2 if name == "rope_split" else 3)
            definition = getattr(attention, name)
            arguments = dict(X=x, Cos=cosine, Sin=sine)
        else:
            shape = (4096, 128) if base_name == "softmax" else (128, 4096)
            x = random(shape, rng)
            arguments = dict(X=x)
            if base_name in ("rms_norm", "layer_norm", "residual_rms_norm"):
                definition = getattr(normalization, name)
                weight = random((shape[-1],), rng)
                arguments["Weight"] = weight
                value = x
                if base_name == "residual_rms_norm":
                    arguments["Residual"] = random(shape, rng)
                    value = x + arguments["Residual"]
                if base_name == "layer_norm":
                    value = x - x.mean(axis=-1, keepdims=True)
                expected = value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-5) * weight
                if base_name == "layer_norm":
                    arguments["Bias"] = random(weight.shape, rng)
                    expected += arguments["Bias"]
            elif base_name == "softmax":
                definition = getattr(attention, name)
                if name == "softmax":
                    bindings = {"N": shape[-1]}
                expected = np.exp(x - x.max(axis=-1, keepdims=True))
                expected /= expected.sum(axis=-1, keepdims=True)
            elif name in ("logarithm", "exponential", "softplus"):
                definition = getattr(activation, name)
                if name == "logarithm":
                    arguments["X"] = np.exp(x)
                    expected = np.log(arguments["X"])
                elif name == "exponential":
                    expected = np.exp(x)
                else:
                    expected = np.logaddexp(np.float32(0), x)
            elif name in ("silu", "swiglu"):
                definition = getattr(activation, name)
                expected = silu_reference(x)
                if name == "swiglu":
                    up = random(shape, rng)
                    arguments = dict(Gate=x, Up=up)
                    expected *= up
            else:
                raise ValueError(f"unknown operator {name!r}")
        output = np.empty_like(expected)
        arguments["Y"] = output
        if name.endswith("_row"):
            bindings = {"N": shape[-1]}
        key = (name, tuple(shape)) if compare_programs else (name, tuple(shape), tuple(sorted(bindings.items())))
        record = previous[key] if compare_artifacts is not None else None
        alignment_directory = ggml_directory
        if alignment_directory is None and record is not None:
            alignment_directory = (record.get("input_alignment") or {}).get("ggml_directory")
        if alignment_directory is None and record is not None and record.get("reference") is not None:
            from pathlib import Path

            # Preserve the recorded input contract without rerunning a fixed
            # GGML baseline. Otherwise an artifact comparison quietly changes
            # from GGML-aligned storage to whatever alignment NumPy provided.
            alignment_directory = str(Path(record["reference"]["library"]).parent)
        if alignment_directory is not None:
            arguments = GGMLCall.aligned_inputs(alignment_directory, arguments)
        options = get(name).options(**bindings)
        physical = {key: value for key, value in (
            ("lmul_eighths", lmul_eighths), ("pointwise_issue_copies", pointwise_issue_copies),
            ("contraction_column_copies", contraction_column_copies),
            ("panel_transfer_fields", panel_transfer_fields), ("contraction_panel_k", contraction_panel_k),
            ("contraction_k_steps", contraction_k_steps), ("pipeline_depth", pipeline_depth), ("unroll", unroll))
            if value is not None}
        if physical:
            from dataclasses import replace

            options = replace(options, **physical)
        with weft.jit(definition, options=get(name).options(options=options, **bindings)) as kernel:
            prepared = kernel.prepare(**arguments)
            prepared()
            np.testing.assert_allclose(output, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
            if initial_state is not None:
                np.testing.assert_allclose(arguments["State"], expected_state, atol=1e-4, rtol=2e-3)
            error = float(np.max(np.abs(output - expected)))
            samples = []
            comparison = None
            from contextlib import ExitStack

            with ExitStack() as owners:
                calls = [(prepared, samples, arguments.get("State"))]
                reference = None
                if ggml_directory is not None:
                    ggml = owners.enter_context(GGMLCall(ggml_directory, name, arguments))
                    ggml.time_ns()
                    np.testing.assert_allclose(ggml.result(), expected, atol=1e-4, rtol=2e-3, equal_nan=False)
                    reference_samples = []
                    calls.append((ggml, reference_samples, None))
                if compare_artifacts is not None:
                    from weft.runtime import Artifact

                    if tuple(record["shape"]) != tuple(shape):
                        raise ValueError("comparison requires the same workload shape")
                    artifact = Artifact(record["artifact"])
                    current = prepared.kernel.artifact
                    contract = artifact_comparison_contract(artifact, current,
                        compare_programs=compare_programs, compare_bindings=compare_bindings)
                    old = owners.enter_context(artifact.load(target=prepared.kernel.target))
                    # The compiler comparison must use the same addresses,
                    # including the output. Separate NumPy allocations can
                    # change alignment/cache behaviour on the measured target.
                    # Every call is checked below before the next overwrites it.
                    old_output = output
                    old_arguments = {**arguments, "Y": old_output}
                    if initial_state is not None:
                        old_arguments["State"] = initial_state.copy()
                    old_call = old.prepare(**old_arguments)
                    old_call()
                    np.testing.assert_allclose(old_output, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
                    if initial_state is not None:
                        np.testing.assert_allclose(old_arguments["State"], expected_state, atol=1e-4, rtol=2e-3)
                    old_samples = []
                    calls.insert(0, (old_call, old_samples, old_arguments.get("State")))
                for repeat in range(10):
                    for call, destination, state in (calls if repeat % 2 == 0 else list(reversed(calls))):
                        if state is not None:
                            np.copyto(state, initial_state)
                        np.add(eviction, np.uint8(1), out=eviction)
                        destination.append(call.time_ns() / 1e6)
                        if compare_artifacts is not None and not isinstance(call, GGMLCall):
                            np.testing.assert_allclose(output, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
                            if state is not None:
                                np.testing.assert_allclose(state, expected_state, atol=1e-4, rtol=2e-3)
                if compare_artifacts is not None:
                    np.testing.assert_allclose(old_output, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
                    comparison = {"artifact": str(artifact.path), "median_ms": statistics.median(old_samples),
                                  "samples_ms": old_samples, "max_abs_error": float(np.max(np.abs(old_output - expected))),
                                  "speedup": statistics.median(old_samples) / statistics.median(samples),
                                  **contract,
                                  "execution_target": prepared.kernel.target.to_dict(),
                                  "bindings": record["bindings"],
                                  "scope": "same-input-alternating-native-artifacts",
                                  "same_output_storage": True}
                    if initial_state is not None:
                        np.testing.assert_allclose(old_arguments["State"], expected_state, atol=1e-4, rtol=2e-3)
                        comparison["state_max_abs_error"] = float(np.max(np.abs(old_arguments["State"] - expected_state)))
                if ggml_directory is not None:
                    actual = ggml.result()
                    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
                    reference = {"implementation": "GGML CPU graph", "library": ggml.library,
                                 "median_ms": statistics.median(reference_samples), "samples_ms": reference_samples,
                                 "max_abs_error": float(np.max(np.abs(actual - expected))),
                                 "speedup": statistics.median(reference_samples) / statistics.median(samples),
                                 "scope": "same-input-storage-complete-operator-including-affine;single-thread;64MiB-evict"}
                    if name.startswith("matmul"):
                        reference["layout_adapter"] = "ggml_cont(ggml_transpose(Weight));inside-timed-graph"
            np.testing.assert_allclose(output, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
            compiled = prepared.kernel
            state_result = {}
            if initial_state is not None:
                np.testing.assert_allclose(arguments["State"], expected_state, atol=1e-4, rtol=2e-3)
                state_result = {"state_reset": "same-initial-state-before-every-call;outside-timed-region",
                                "state_max_abs_error": float(np.max(np.abs(arguments["State"] - expected_state)))}
            print(json.dumps({"program": name, "shape": shape, "bindings": bindings,
                              "input_generation": {"algorithm": "PCG64", "seed": 415, "scope": "per-operator"},
                              "status": "correct", "max_abs_error": error,
                              "median_ms": statistics.median(samples), "samples_ms": samples,
                              "timing_scope": "native-entry-with-ctypes;64MiB-evict",
                              "artifact": str(compiled.artifact.path),
                              "target": compiled.artifact.target.to_dict(),
                              "options": compiled.artifact.options.to_dict(),
                              "compiler": compiled.artifact.manifest["link_command"],
                              "input_alignment": {"ggml_directory": alignment_directory,
                                  "address_mod_64": {key: int(value.ctypes.data % 64) for key, value in arguments.items()
                                                     if key != "Y" and isinstance(value, np.ndarray)}},
                              "comparison": comparison,
                              "reference": reference,
                              "physical_resources": dict(compiled.physical_resources), **state_result}), flush=True)


def invocation_usage(with_torch=False):
    """Prepare an explicit residual/norm/activation chain for repeated execution."""
    from weft.programs import Operators
    from weft.runtime import PreparedSequence

    rng = np.random.default_rng(710)
    for backend in (("numpy", "torch") if with_torch else ("numpy",)):
        x = random((4, 129), rng)
        residual = random(x.shape, rng)
        weight = random((129,), rng)
        intermediate, normalized, result = (np.empty_like(x) for _ in range(3))
        expected = x + residual
        expected = expected / np.sqrt(np.mean(expected * expected, axis=-1, keepdims=True) + 1e-5) * weight
        expected = expected / (1 + np.exp(-expected))
        if backend == "torch":
            import torch
            x, residual, weight, intermediate, normalized, result = (
                torch.from_numpy(value) for value in (x, residual, weight, intermediate, normalized, result))
        with Operators() as operators:
            add = operators.kernel("add").prepare(X=x, Residual=residual, Y=intermediate)
            norm = operators.kernel("rms_norm").prepare(X=intermediate, Weight=weight, Y=normalized)
            activation_call = operators.kernel("silu").prepare(X=normalized, Y=result)
            with PreparedSequence(add, norm, activation_call) as sequence:
                outputs = sequence()
                assert outputs[-1] is result
                check("prepared-residual-norm-silu-" + backend, result, expected)
                # Input bytes can change without rebuilding a descriptor. State
                # reset and the execution order remain explicit author actions.
                x *= 0.5
                sequence()
                updated = np.asarray(x) + np.asarray(residual)
                updated = updated / np.sqrt(np.mean(updated * updated, axis=-1, keepdims=True) + 1e-5) * np.asarray(weight)
                updated = updated / (1 + np.exp(-updated))
                check("prepared-updated-input-" + backend, result, updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("normalization", "attention", "rows", "values", "projections", "models", "selection", "predicates", "operators", "tuning", "projection-tuning", "invocation", "all"), default="all")
    parser.add_argument("--block", type=int, default=32, help="explicit source BLOCK binding for operators")
    parser.add_argument("--rows", type=int, help="override the catalog output-column cohort for projection timing")
    parser.add_argument("--tokens", type=int, default=128, help="activation rows for projection timing")
    parser.add_argument("--operator", action="append", help="operator to measure; repeat to select several")
    parser.add_argument("--compare-artifacts", help="earlier operator JSONL or predicate JSON; rerun its artifacts on identical inputs")
    comparison_mode = parser.add_mutually_exclusive_group()
    comparison_mode.add_argument("--compare-programs", action="store_true", help="compare different author programs under the same numerical workload")
    comparison_mode.add_argument("--compare-bindings", action="store_true",
                                 help="compare physical bindings while retaining the same canonical program and source auto bindings")
    parser.add_argument("--ggml-directory", help="GGML CPU shared-library directory built with the matching compiler/target/flags")
    parser.add_argument("--lmul-eighths", type=int, choices=(1, 2, 4, 8, 16, 32, 64),
                        help="explicit physical LMUL bound for operator timing; otherwise use the program default")
    parser.add_argument("--pointwise-issue-copies", type=int, choices=(1, 2, 4),
                        help="maximum independent pointwise pieces per physical issue")
    parser.add_argument("--contraction-column-copies", type=int, choices=(1, 2, 4),
                        help="maximum output-column vectors sharing one scalar contraction operand")
    parser.add_argument("--panel-transfer-fields", type=int, choices=(1, 2, 4, 8),
                        help="maximum contiguous fields in each private-panel vector transfer")
    parser.add_argument("--contraction-panel-k", type=int, choices=(32, 64, 128, 256),
                        help="maximum reduction extent in each private contraction panel")
    parser.add_argument("--contraction-k-steps", type=int, choices=(1, 2, 4),
                        help="maximum adjacent reduction coordinates in a scalar supply packet")
    parser.add_argument("--pipeline-depth", type=int, choices=(1, 2),
                        help="physical producer/consumer pipeline depth for the operator call")
    parser.add_argument("--unroll", type=int, choices=range(1, 17),
                        help="physical issue repetition within the closed numerical operation")
    parser.add_argument("--torch", action="store_true", help="also run complete Torch modules and captured/exported graphs")
    args = parser.parse_args()
    if (args.compare_programs or args.compare_bindings) and not args.compare_artifacts:
        parser.error("comparison mode requires --compare-artifacts")
    if (args.compare_programs or args.compare_bindings) and args.family != "operators":
        parser.error("author-program and physical-binding comparison modes belong to --family operators")
    if (args.pipeline_depth is not None or args.unroll is not None) and args.family != "operators":
        parser.error("explicit pipeline/unroll bindings require --family operators")
    if args.family in ("normalization", "all"):
        normalization_usage(args.block)
    if args.family in ("attention", "all"):
        attention_usage(args.block)
    if args.family in ("models", "all"):
        model_usage(args.torch)
    if args.family in ("rows", "all"):
        row_usage(args.torch)
    if args.family in ("values", "all"):
        value_usage(args.torch)
    if args.family in ("selection", "all"):
        selection_usage(args.torch)
    if args.family == "predicates":
        predicate_usage(args.block, args.compare_artifacts,
                        lmul_eighths=8 if args.lmul_eighths is None else args.lmul_eighths,
                        pointwise_issue_copies=1 if args.pointwise_issue_copies is None else args.pointwise_issue_copies)
    if args.family in ("projections", "all"):
        projection_usage(args.torch)
    if args.family == "operators":
        operator_usage(args.block, args.operator, args.compare_artifacts, rows=args.rows, tokens=args.tokens,
                       compare_programs=args.compare_programs, compare_bindings=args.compare_bindings,
                       ggml_directory=args.ggml_directory,
                       lmul_eighths=args.lmul_eighths, pointwise_issue_copies=args.pointwise_issue_copies,
                       contraction_column_copies=args.contraction_column_copies,
                       panel_transfer_fields=args.panel_transfer_fields, contraction_panel_k=args.contraction_panel_k,
                       contraction_k_steps=args.contraction_k_steps, pipeline_depth=args.pipeline_depth, unroll=args.unroll)
    if args.family == "projection-tuning":
        projection_tuning_usage(args.torch)
        return
    if args.family == "tuning":
        tuning_usage(args.torch)
    if args.family in ("invocation", "all"):
        invocation_usage(args.torch)


if __name__ == "__main__":
    main()
