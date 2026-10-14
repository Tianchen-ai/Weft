"""An ordinary nn.Module backed by registered Weft forward/backward programs."""
from __future__ import annotations

import torch
import weft
from weft.runtime import TuningPolicy
from weft.torch import register

from .affine_backward import affine_backward
from .affine_strided import affine_strided
from .configs import AFFINE_CONFIGS


def validate_affine(original, trial):
    torch.testing.assert_close(trial["Y"], original["X"] * original["scale"] + original["bias"],
                               rtol=2e-3, atol=1e-4)


def register_affine(namespace: str, *, toolchain=None, cache=None, tune=False):
    if tune:
        forward = weft.autotune(affine_strided, AFFINE_CONFIGS, key=("scale", "bias"),
                                policy=TuningPolicy(deadline_s=600), validator=validate_affine,
                                validation_key="affine-f32-rtol2e-3-atol1e-4", toolchain=toolchain, cache=cache)
    else:
        forward = weft.jit(affine_strided, options=AFFINE_CONFIGS[0].options, toolchain=toolchain, cache=cache)
    derivative = weft.jit(affine_backward, toolchain=toolchain, cache=cache)
    backward_op = register(namespace + "::affine_backward", derivative)
    forward_op = register(namespace + "::affine", forward)

    def setup_context(ctx, inputs, keyword_only_inputs, output):
        _, scale = inputs
        ctx.scale = scale

    def backward(ctx, gradient):
        return backward_op(gradient, ctx.scale), None

    forward_op.register_backward(backward, setup_context=setup_context)
    return forward_op


class Affine(torch.nn.Module):
    def __init__(self, operator, *, scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.operator = operator
        self.scale = scale
        self.bias = bias

    def forward(self, value):
        return self.operator(value, self.scale, bias=self.bias)
