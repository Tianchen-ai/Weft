import platform
import time
import os
from functools import lru_cache

from triton.backends.compiler import CPUTarget
from triton.backends.driver import DriverBase


class WeftDriver(DriverBase):
    def __init__(self):
        self.utils = self

    @staticmethod
    def is_active():
        # This model is selected per JITFunction, not by global GPU discovery.
        return False

    @staticmethod
    def map_python_to_cpp_type(ty):
        types = {"i1": "bool", "u1": "bool", "i8": "int8_t", "u8": "uint8_t",
                 "i16": "int16_t", "u16": "uint16_t", "i32": "int32_t", "u32": "uint32_t",
                 "i64": "int64_t", "u64": "uint64_t", "fp32": "float", "fp64": "double"}
        return "void*" if ty.startswith(("*", "view:")) else types[ty]

    def get_current_target(self):
        return self._target_for_affinity(tuple(sorted(os.sched_getaffinity(0))))

    @staticmethod
    @lru_cache(maxsize=32)
    def _target_for_affinity(affinity):
        # Cache hardware facts for the discovered CPU set, not execution
        # permission. The SDK still checks each actual worker on every launch.
        from triton._C.libtriton import weft
        target = weft.native_target()["target"]
        return CPUTarget("weft", "riscv64", target["march"], target["abi"], target["vlen_bits"])

    def get_current_device(self):
        if platform.machine() != "riscv64":
            raise RuntimeError("Weft CPU JIT execution requires RISC-V; use triton.compile with an explicit cross target")
        return ("weft:cpu", tuple(sorted(os.sched_getaffinity(0))))

    def get_current_stream(self, device):
        return None

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):
        def benchmark(call, *, quantiles, **kwargs):
            for _ in range(5):
                call()
            timings = []
            deadline = time.perf_counter_ns() + 50_000_000
            while len(timings) < 20 or time.perf_counter_ns() < deadline:
                start = time.perf_counter_ns()
                call()
                timings.append((time.perf_counter_ns() - start) / 1e6)
            timings.sort()
            return [timings[round(q * (len(timings) - 1))] for q in quantiles]
        return benchmark

    def load_compiled(self, kernel):
        from triton.runtime.weft.native import NativeModule, NativeLauncher
        path = next(path for name, path in kernel.metadata_group.items() if name.endswith(".so"))
        module = NativeModule(path, kernel.metadata)
        kernel.module = module
        kernel.function = module.entry
        kernel._run = NativeLauncher(module, kernel.metadata)
