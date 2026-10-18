"""Author programs expressed by the embedded Triton model."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import triton
from triton.compiler import ASTSource
from triton.compiler.weft.interface import describe
from triton.runtime.weft.signature import CallSignature
from . import activation, attention, linear, mamba, moe, normalization


@dataclass(frozen=True)
class Program:
    name: str
    family: str
    definition: object
    description: str
    defaults: tuple[tuple[str, int], ...] = (("BLOCK", 32),)
    required: tuple[str, ...] = ()
    physical_defaults: tuple[tuple[str, int], ...] = ()

    def config(self, config=None, **bindings):
        selected = dict(self.defaults)
        physical = dict(self.physical_defaults)
        if config is not None:
            if not isinstance(config, triton.Config):
                raise TypeError("program configuration must be triton.Config")
            selected.update(config.kwargs)
            physical.update(config.backend_options or {})
        selected.update(bindings)
        unknown = selected.keys() - (dict(self.defaults).keys() | set(self.required))
        missing = set(self.required) - selected.keys()
        if unknown or missing:
            raise ValueError(f"{self.name}: unknown source parameters {sorted(unknown)}, missing {sorted(missing)}")
        return triton.Config(selected, backend_options=physical,
                             num_cpu_threads=None if config is None else config.num_cpu_threads)

    def signature(self, **bindings):
        configuration = self.config(**bindings)
        return CallSignature(describe(self.definition, configuration.kwargs), bindings=configuration.kwargs)

    def compile(self, *, target=None, config=None, toolchain=None, **bindings):
        configuration = self.config(config, **bindings)
        if target is None:
            from triton.runtime.driver import driver
            target = driver.for_model("weft").get_current_target()
        options = {"backend_options": configuration.backend_options}
        if configuration.num_cpu_threads is not None:
            options["num_cpu_threads"] = configuration.num_cpu_threads
        if toolchain is not None:
            options["toolchain"] = toolchain
        return triton.compile(ASTSource(self.definition, {}, constexprs=configuration.kwargs),
                              target=target, options=options)


_entries = (
    Program("rms_norm", "normalization", normalization.rms_norm,
            "Row RMS normalization with a learned scale.", physical_defaults=(("lmul_eighths", 32),)),
    Program("layer_norm", "normalization", normalization.layer_norm,
            "Centered-variance layer normalization with scale and bias.", physical_defaults=(("lmul_eighths", 32),)),
    Program("rms_norm_row", "normalization", normalization.rms_norm_row,
            "One row value shared by RMS reduction and learned scaling.", (), ("N",)),
    Program("layer_norm_row", "normalization", normalization.layer_norm_row,
            "One row value and one centered value shared across reductions and scaling.", (), ("N",)),
    Program("rms_norm_tile", "normalization", normalization.rms_norm_tile,
            "Explicit row tiles with a retained row axis through RMS reduction.", (("ROWS", 1),), ("N",)),
    Program("layer_norm_tile", "normalization", normalization.layer_norm_tile,
            "Explicit row tiles with centered variance and broadcast scale/bias.", (("ROWS", 1),), ("N",)),
    Program("residual_rms_norm", "normalization", normalization.residual_rms_norm,
            "Normalize the sum of activation and residual.", physical_defaults=(("lmul_eighths", 32),)),
    Program("silu", "activation", activation.silu, "Numerically stable SiLU.",
            physical_defaults=(("lmul_eighths", 16),)),
    Program("logarithm", "math", activation.logarithm, "Elementwise natural logarithm.",
            physical_defaults=(("lmul_eighths", 16),)),
    Program("exponential", "math", activation.exponential, "Elementwise exponential."),
    Program("softplus", "activation", activation.softplus, "Stable softplus used by selective state-space models.",
            physical_defaults=(("lmul_eighths", 16),)),
    Program("swiglu", "activation", activation.swiglu, "SiLU(gate) times up projection.",
            physical_defaults=(("lmul_eighths", 16),)),
    Program("add", "activation", activation.add, "Elementwise residual addition."),
    Program("copy", "memory", activation.copy, "Explicit strided activation copy."),
    Program("where", "selection", activation.where,
            "Select equal-shaped f32 values with an explicit Boolean condition; supports activation aliases."),
    Program("fill", "memory", activation.fill, "Fill the caller's view with one scalar value.",
            (("BLOCK", 32), ("ROWS", 1))),
    Program("linear", "projection", linear.linear, "Dense projection with weight [out,in].",
            (("MC", 64), ("NC", 16), ("MR", 4), ("NR", 4)), physical_defaults=(("lmul_eighths", 64),)),
    Program("matmul", "projection", linear.matmul, "Dense matrix product with right operand [in,out].",
            (("MC", 32), ("NC", 64)), physical_defaults=(("lmul_eighths", 32),)),
    Program("matmul_bias", "projection", linear.matmul_bias, "Dense matrix product followed by column bias.",
            (("MC", 32), ("NC", 64)), physical_defaults=(("lmul_eighths", 32),)),
    Program("batched_matmul", "projection", linear.batched_matmul, "Pairwise matrix products over a shared batch axis.",
            (), physical_defaults=(("lmul_eighths", 32),)),
    Program("linear_silu", "projection", linear.linear_silu, "Dense projection followed by stable SiLU.",
            (("MC", 32), ("NC", 64)), physical_defaults=(("lmul_eighths", 32),)),
    Program("linear_bias_silu", "projection", linear.linear_bias_silu, "Dense projection, column bias and stable SiLU.",
            (("MC", 32), ("NC", 64)), physical_defaults=(("lmul_eighths", 32),)),
    Program("linear_bias", "projection", linear.linear_bias, "Dense projection and bias.",
            (("MC", 64), ("NC", 16), ("MR", 4), ("NR", 4)), physical_defaults=(("lmul_eighths", 64),)),
    Program("linear_tile", "projection", linear.linear_tile,
            "Explicit token/output tiles with closed K-block dots.",
            (("BLOCK", 128), ("MROWS", 4), ("NROWS", 4))),
    Program("linear_bias_tile", "projection", linear.linear_bias_tile,
            "Explicit token/output tiles followed by broadcast bias.",
            (("BLOCK", 128), ("MROWS", 4), ("NROWS", 4))),
    Program("softmax", "attention", attention.softmax, "Stable row softmax.", (), ("N",),
            physical_defaults=(("lmul_eighths", 16), ("private_stack_bytes", 262144))),
    Program("softmax_row", "attention", attention.softmax_row,
            "One row read and one exponential value shared with normalization.", (), ("N",),
            (("lmul_eighths", 16), ("private_stack_bytes", 262144))),
    Program("rope_split", "attention", attention.rope_split,
            "Split-half rotary embedding with explicit cosine/sine tables."),
    Program("rope_interleaved", "attention", attention.rope_interleaved,
            "Even/odd rotary embedding with explicit cosine/sine tables."),
    Program("kv_update", "attention", attention.kv_update,
            "Append new keys and values to caller-owned contiguous-position caches."),
    Program("causal_attention", "attention", attention.causal_attention,
            "Online causal grouped-query attention over the populated KV prefix.", (), ("D",)),
    Program("router", "moe", moe.router,
            "Stable top-k selection and selected-score softmax; lower-index ties.", ()),
    Program("group_tokens", "moe", moe.group_tokens,
            "Stable grouping, exclusive expert offsets and inverse token positions."),
    Program("expert_linear", "moe", moe.expert_linear,
            "Ragged expert projection including empty expert ranges.",
            (("ROWS", 4),), physical_defaults=(("lmul_eighths", 64),)),
    Program("combine", "moe", moe.combine,
            "Weighted merge of grouped expert outputs into token order."),
    Program("causal_conv1d", "mamba", mamba.causal_conv1d,
            "Stateful depthwise causal convolution and SiLU."),
    Program("selective_scan", "mamba", mamba.selective_scan,
            "Token-major Mamba-1 recurrence over a complete state vector.",
            (), ("N",), (("lmul_eighths", 16),)),
)

catalog = MappingProxyType({entry.name: entry for entry in _entries})


def get(name):
    try:
        return catalog[name]
    except KeyError:
        raise KeyError(f"unknown Weft program {name!r}; available: {', '.join(catalog)}") from None


def programs(*, family=None):
    return tuple(entry for entry in catalog.values() if family is None or entry.family == family)
