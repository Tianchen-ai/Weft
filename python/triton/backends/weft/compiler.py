from dataclasses import asdict, dataclass, field, fields
import hashlib
import json

from triton._C.libtriton import weft
from triton.backends.compiler import BaseBackend, CPUTarget, Language


@dataclass(frozen=True)
class WeftOptions:
    # Shared builtin semantics, independent of CPU scheduling and physical tuning.
    allowed_dot_input_precisions = ("ieee",)
    default_dot_input_precision = "ieee"
    sanitize_overflow = False
    meta: dict = field(default_factory=dict)
    lmul_eighths: int = 8
    unroll: int = 1
    pipeline_depth: int = 1
    scalar_load_prime: int = 0
    pointwise_issue_copies: int = 1
    contraction_column_copies: int = 4
    panel_transfer_fields: int = 4
    panel_transfer_strategy: str = "issue-work"
    contraction_panel_k: int = 128
    contraction_k_steps: int = 4
    partial_combine_policy: str = "independent-multilevel"
    record_axis_policy: str = "within-record"
    max_widening_combine_groups: int = 2
    matrix_extension: str = "none"
    required_extensions: tuple = ()
    local_tile_strategy: str = "target-default"
    private_stack_bytes: int = 65536
    data_cache: dict | None = None
    num_cpu_threads: int = 0
    toolchain: dict = field(default_factory=dict)
    debug: bool = False
    instrumentation_mode: str = ""

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class WeftBackend(BaseBackend):
    binary_ext = "so"
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target):
        return isinstance(target, CPUTarget) and target.backend == "weft" and target.arch == "riscv64"

    def hash(self):
        return str(self.target)

    def parse_options(self, options):
        from .toolchain import Toolchain
        values = dict(options)
        physical = dict(values.pop("backend_options", {}))
        valid = {f.name for f in fields(WeftOptions)}
        if set(physical) - valid:
            raise ValueError(f"unknown Weft backend options: {sorted(set(physical) - valid)}")
        if values.keys() - valid:
            raise ValueError(f"unknown Weft compilation options: {sorted(values.keys() - valid)}")
        values.update(physical)
        values["toolchain"] = Toolchain.resolve(values.get("toolchain", {}), self.target).to_dict()
        result = WeftOptions(**values)
        if type(result.num_cpu_threads) is not int or result.num_cpu_threads < 0:
            raise ValueError("num_cpu_threads must be a nonnegative integer; 0 uses the OpenMP setting")
        if result.instrumentation_mode:
            raise NotImplementedError("GPU instrumentation does not implement Weft CPU effects")
        return result

    def load_dialects(self, context):
        weft.load_dialects(context)

    def get_module_map(self):
        from triton.language.weft import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def get_codegen_implementation(self, options):
        # Canonical contraction has no GPU tile-size floor. Physical legality
        # remains the responsibility of the selected RISC-V implementation.
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def pack_metadata(self, metadata):
        return metadata

    @staticmethod
    def specialize_argument(backend, value, is_const, specialize, align):
        from triton.runtime.weft.buffer import storage_encoding
        if isinstance(value, bool):
            return ("u1", "")
        if isinstance(value, int):
            return ("i64", "")
        if isinstance(value, float):
            return ("fp32", "")
        encoding = storage_encoding(value)
        if encoding.startswith("dense."):
            from triton.compiler.weft.values import logical_dtype
            from triton.language.weft import dtypes
            from triton.runtime.jit import _normalize_ty
            return ("*" + _normalize_ty(logical_dtype(getattr(dtypes, encoding.removeprefix("dense.")))), "")
        return ("view:" + encoding, "")

    @staticmethod
    def parse_attr(description):
        return []

    def add_stages(self, stages, options, language):
        if language != Language.WEFT:
            raise ValueError("the Weft backend requires @triton.jit(model='weft')")

        def canonical(module, metadata):
            if not module.verify():
                raise ValueError("invalid Weft canonical IR")
            metadata["interface"] = weft.interface(module)
            return module

        def physical(module, metadata):
            bindings = asdict(options)
            interface = metadata["interface"]["interface"]
            names = set(interface["source_parameters"]) | set(interface["shape_parameters"])
            bindings["meta"] = {name: value for name, value in options.meta.items() if name in names}
            bindings.update(march=self.target.march, abi=self.target.abi, vlen_bits=self.target.vlen_bits)
            lowered, report = weft.lower(module, bindings)
            lowered.context = module.context
            metadata["compilation"] = report
            metadata["source_bindings"] = bindings["meta"]
            return lowered

        def emit(module, metadata):
            result = weft.emit(module)
            if len(result["kernels"]) != 1:
                raise ValueError("a Triton JITFunction must produce exactly one native kernel")
            metadata["kernel_abi"] = result["kernels"][0]
            from .launcher import single_instance_abi
            metadata["single_instance_abi"] = single_instance_abi(metadata["kernel_abi"])
            metadata["name"] = metadata["kernel_abi"]["symbol"]
            metadata["compilation"] = result["compilation"]
            return result["intrinsic_c"]

        def native(source, metadata):
            from .toolchain import Toolchain
            return Toolchain(**options.toolchain).compile(source, metadata, self.target)

        stages["weftir"] = canonical
        stages["riscvir"] = physical
        stages["c"] = emit
        stages["so"] = native
