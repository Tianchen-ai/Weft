"""A complete Mamba-1 mixer composed from public Weft author programs."""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import BufferSpec, check_invocation, integer, output, same_backend, tensor, use_workspace, Workspace
from .operators import Operators
from .state import MambaState


@dataclass(frozen=True)
class MambaWeights:
    in_proj: object
    conv: object
    conv_bias: object
    x_proj: object
    dt_proj: object
    dt_bias: object
    a_log: object
    skip: object
    out_proj: object
    in_bias: object = None
    out_bias: object = None

    def dimensions(self):
        inner, state_size = tensor(self.a_log, "a_log", rank=2)
        model, output_inner = tensor(self.out_proj, "out_proj", rank=2)
        dt_inner, rank = tensor(self.dt_proj, "dt_proj", rank=2)
        conv_inner, width = tensor(self.conv, "conv", rank=2)
        if (output_inner, dt_inner, conv_inner) != (inner, inner, inner):
            raise ValueError("Mamba projections, convolution and state need the same inner dimension")
        for name, extent in zip(("model", "inner", "state_size", "dt_rank", "conv_width"),
                                (model, inner, state_size, rank, width)):
            integer(extent, name)
        tensor(self.in_proj, "in_proj", shape=(2 * inner, model))
        tensor(self.conv_bias, "conv_bias", shape=(inner,))
        tensor(self.x_proj, "x_proj", shape=(rank + 2 * state_size, inner))
        tensor(self.dt_bias, "dt_bias", shape=(inner,))
        tensor(self.skip, "skip", shape=(inner,))
        values = [self.in_proj, self.conv, self.conv_bias, self.x_proj, self.dt_proj,
                  self.dt_bias, self.a_log, self.skip, self.out_proj]
        if self.in_bias is not None:
            tensor(self.in_bias, "in_bias", shape=(2 * inner,))
            values.append(self.in_bias)
        if self.out_bias is not None:
            tensor(self.out_bias, "out_bias", shape=(model,))
            values.append(self.out_bias)
        same_backend(*values)
        return model, inner, state_size, rank, width


class Mamba1:
    """f32 inference mixer with caller-visible convolution and SSM state.

    A_log stores log(-A). Delta is low-rank projected, then biased and passed
    through softplus inside the selective scan. B/C depend on every token.
    The mixer includes both input/output projections but no outer residual or
    normalization; those are explicit model-block choices made by the caller.
    """

    def __init__(self, weights: MambaWeights, *, operators=None):
        if not isinstance(weights, MambaWeights):
            raise TypeError("weights must be MambaWeights")
        self.weights = weights
        self.dimensions = weights.dimensions()
        self.operators = Operators() if operators is None else operators
        self._owns_operators = operators is None

    def new_state(self, batch, *, like=None):
        _, inner, n, _, width = self.dimensions
        return MambaState.allocate(batch, width, inner, n,
                                   like=self.weights.in_proj if like is None else like)

    def workspace_specs(self, batch, tokens):
        integer(batch, "batch")
        integer(tokens, "tokens", minimum=0)
        _, inner, n, rank, _ = self.dimensions
        return (BufferSpec("input", (batch, tokens, 2 * inner)),
                BufferSpec("conv", (batch, tokens, inner)),
                BufferSpec("parameters", (batch, tokens, rank + 2 * n)),
                BufferSpec("delta", (batch, tokens, inner)),
                BufferSpec("scan", (batch, tokens, inner)))

    def workspace(self, batch, tokens, *, like=None):
        return Workspace(self.workspace_specs(batch, tokens),
                         like=self.weights.in_proj if like is None else like)

    @property
    def required_programs(self):
        names = ["linear", "causal_conv1d", "selective_scan"]
        if self.weights.in_bias is not None or self.weights.out_bias is not None:
            names.append("linear_bias")
        result = {name: {} for name in names}
        result["selective_scan"] = {"N": self.dimensions[2]}
        return result

    def materialize(self):
        self.operators.materialize(self.required_programs, **self.required_programs)
        return self

    def __call__(self, x, state, *, out=None, workspace=None):
        b, t, d = tensor(x, "x", rank=3)
        model, inner, n, rank, width = self.dimensions
        if d != model or b <= 0:
            raise ValueError("Mamba input must have positive batch and the configured model dimension")
        if self.weights.dimensions() != self.dimensions:
            raise ValueError("Mamba parameter dimensions changed after construction")
        same_backend(x, self.weights.in_proj)
        if not isinstance(state, MambaState):
            raise TypeError("state must be MambaState")
        state.check(batch=b, width=width, inner=inner, state_size=n, like=x)
        y = output(out, (b, t, d), like=x)
        work = use_workspace(workspace, self.workspace_specs(b, t), like=x)
        check_invocation(x, y, self.weights, work, state={"history": state.history, "ssm": state.ssm})
        op, w = self.operators, self.weights
        with work, state.use(t):
            # Iterate batch only where flattening a sliced batch would require
            # a copy. Each matrix call accepts the original element strides.
            for batch in range(b):
                op.linear(x[batch], w.in_proj, w.in_bias, out=work["input"][batch])
            projected = work["input"]
            op.kernel("causal_conv1d")(X=projected[:, :, :inner], Weight=w.conv, Bias=w.conv_bias,
                                       History=state.history, Y=work["conv"])
            op.linear(work["conv"], w.x_proj, out=work["parameters"])
            parameters = work["parameters"]
            op.linear(parameters[:, :, :rank], w.dt_proj, out=work["delta"])
            op.kernel("selective_scan", N=n)(
                X=work["conv"], Delta=work["delta"], Bcoef=parameters[:, :, rank:rank + n],
                Ccoef=parameters[:, :, rank + n:], Alog=w.a_log, Skip=w.skip,
                Gate=projected[:, :, inner:], Bias=w.dt_bias, State=state.ssm, Y=work["scan"])
            for batch in range(b):
                op.linear(work["scan"][batch], w.out_proj, w.out_bias, out=y[batch])
        return y

    def close(self):
        if self._owns_operators:
            self.operators.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
