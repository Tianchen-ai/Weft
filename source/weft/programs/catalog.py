"""Discoverable author programs; every entry uses the ordinary Weft compiler."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import weft
from weft.runtime import CompileOptions
from weft.runtime.compiler import KernelSource

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

    def options(self, *, options=None, **bindings):
        declared = dict(self.defaults)
        allowed = set(declared) | set(self.required)
        unknown = set(bindings) - allowed
        if unknown:
            raise ValueError(f"{self.name} has no source parameters {sorted(unknown)}")
        if options is None:
            options = CompileOptions(**dict(self.physical_defaults))
        unknown = set(options.meta) - allowed
        if unknown:
            raise ValueError(f"{self.name} has no source parameters {sorted(unknown)}")
        declared.update(options.meta)
        declared.update(bindings)
        missing = set(self.required) - declared.keys()
        if missing:
            raise ValueError(f"{self.name} requires explicit source bindings {sorted(missing)}")
        return replace(options, meta=declared)

    def jit(self, *, options=None, toolchain=None, cache=None, **bindings):
        return weft.jit(self.definition, options=self.options(options=options, **bindings),
                        toolchain=toolchain, cache=cache)

    def build(self, *, target, options=None, toolchain=None, cache=None,
              native_library=True, **bindings):
        return weft.build(self.definition, target=target,
                          options=self.options(options=options, **bindings),
                          toolchain=toolchain, cache=cache, native_library=native_library)

    def signature(self, **bindings):
        source = KernelSource.capture(self.definition)
        return source.interface(source.options(self.options(**bindings)).meta)

    def configurations(self, configs, *, max_candidates=8, **bindings):
        """Resolve finite choices against this author's declared catalog defaults."""
        from weft.runtime.configuration import Config, expand_configurations, resolve_configurations

        result = []
        for config in expand_configurations(configs, max_candidates=max_candidates):
            for name, value in bindings.items():
                if name in config.options.meta and config.options.meta[name] != value:
                    raise ValueError(f"call binding {name}={value} conflicts with candidate {config.name!r}")
            result.append(Config.from_options(self.options(options=config.options, **bindings), name=config.name))
        return resolve_configurations(KernelSource.capture(self.definition), result,
                                      max_candidates=max_candidates)[0]

    def space(self, *, source=None, physical=None, options=None, name=None, **bindings):
        """Describe an explicit finite domain without compiling or measuring.

        Source axes are author choices. Physical axes are target parameters;
        their actual layout and resource legality remain compiler decisions.
        Required ABI specializations must be fixed in bindings, never varied.
        """
        from weft.runtime import ConfigSpace

        if source is not None:
            unknown = source.keys() - dict(self.defaults).keys()
            if unknown:
                raise ValueError(f"{self.name} cannot tune undeclared or ABI source parameters {sorted(unknown)}")
        return ConfigSpace(base=self.options(options=options, **bindings), source=source,
                           physical=physical, name=self.name if name is None else name)

    def autotune(self, configs, **kwargs):
        from weft.runtime import TuningPolicy

        policy = kwargs.get("policy")
        limit = TuningPolicy().max_candidates if policy is None else policy.max_candidates
        return weft.autotune(self.definition, self.configurations(configs, max_candidates=limit), **kwargs)

    def build_candidates(self, configs, *, target, max_candidates=8, toolchain=None,
                         cache=None, native_library=True, **bindings):
        configs = self.configurations(configs, max_candidates=max_candidates, **bindings)
        return weft.build_candidates(self.definition, configs, target=target, toolchain=toolchain,
            cache=cache, native_library=native_library, max_candidates=max_candidates)


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
