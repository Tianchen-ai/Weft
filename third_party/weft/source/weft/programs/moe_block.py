"""Stable top-k routed SwiGLU experts with explicit grouping workspace."""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import BufferSpec, check_invocation, integer, output, same_backend, tensor, use_workspace, Workspace
from .operators import Operators


@dataclass(frozen=True)
class MoEWeights:
    router: object
    gate: object
    up: object
    down: object

    def dimensions(self):
        experts, hidden, model = tensor(self.gate, "gate", rank=3)
        for name, extent in zip(("experts", "hidden", "model"), (experts, hidden, model)):
            integer(extent, name)
        tensor(self.router, "router", shape=(experts, model))
        tensor(self.up, "up", shape=(experts, hidden, model))
        tensor(self.down, "down", shape=(experts, model, hidden))
        same_backend(self.router, self.gate, self.up, self.down)
        if experts > 2147483647:
            raise ValueError("expert count exceeds i32 routing indices")
        return experts, hidden, model


class MoE:
    """Renormalized selected-top-k routing, stable grouping, SwiGLU and merge.

    This is an explicit model variant: no capacity dropping, auxiliary loss,
    shared experts or softmax over unselected experts. Parameters remain owned
    by the caller. Inspect workspace indices/offsets to diagnose real routing.
    """

    def __init__(self, weights: MoEWeights, top_k, *, operators=None):
        if not isinstance(weights, MoEWeights):
            raise TypeError("weights must be MoEWeights")
        self.weights = weights
        self.dimensions = weights.dimensions()
        self.top_k = integer(top_k, "top_k")
        if top_k > self.dimensions[0]:
            raise ValueError("top_k exceeds the number of experts")
        self.operators = Operators() if operators is None else operators
        self._owns_operators = operators is None

    def workspace_specs(self, tokens):
        integer(tokens, "tokens", minimum=0)
        e, h, d = self.dimensions
        k, p = self.top_k, tokens * self.top_k
        if p > 2147483647:
            raise ValueError("token assignments exceed i32 offset storage")
        return (BufferSpec("logits", (tokens, e)),
                BufferSpec("indices", (tokens, k), "i32"),
                BufferSpec("weights", (tokens, k)),
                BufferSpec("offsets", (e + 1,), "i32"),
                BufferSpec("cursor", (e,), "i32"),
                BufferSpec("positions", (tokens, k), "i32"),
                BufferSpec("grouped", (p, d)),
                BufferSpec("gate", (p, h)),
                BufferSpec("up", (p, h)),
                BufferSpec("activated", (p, h)),
                BufferSpec("projected", (p, d)))

    def workspace(self, tokens, *, like=None):
        return Workspace(self.workspace_specs(tokens), like=self.weights.router if like is None else like)

    @property
    def required_programs(self):
        return {name: {} for name in ("linear", "router", "group_tokens", "expert_linear", "swiglu", "combine")}

    def materialize(self):
        self.operators.materialize(self.required_programs)
        return self

    def __call__(self, x, *, out=None, workspace=None):
        t, d = tensor(x, "x", rank=2)
        if d != self.dimensions[2] or self.weights.dimensions() != self.dimensions:
            raise ValueError("MoE input or parameter dimensions differ from construction")
        same_backend(x, self.weights.router)
        y = output(out, (t, d), like=x)
        work = use_workspace(workspace, self.workspace_specs(t), like=x)
        check_invocation(x, y, self.weights, work)
        op, w = self.operators, self.weights
        with work:
            op.linear(x, w.router, out=work["logits"])
            op.route(work["logits"], self.top_k, indices=work["indices"], weights=work["weights"])
            op.kernel("group_tokens")(X=x, Indices=work["indices"], Offsets=work["offsets"],
                                       Cursor=work["cursor"], Positions=work["positions"], Grouped=work["grouped"])
            op.kernel("expert_linear")(X=work["grouped"], Weight=w.gate, Offsets=work["offsets"], Y=work["gate"])
            op.kernel("expert_linear")(X=work["grouped"], Weight=w.up, Offsets=work["offsets"], Y=work["up"])
            op.swiglu(work["gate"], work["up"], out=work["activated"])
            op.kernel("expert_linear")(X=work["activated"], Weight=w.down, Offsets=work["offsets"], Y=work["projected"])
            op.kernel("combine")(Grouped=work["projected"], Positions=work["positions"],
                                  Weights=work["weights"], Y=y)
        return y

    def close(self):
        if self._owns_operators:
            self.operators.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
