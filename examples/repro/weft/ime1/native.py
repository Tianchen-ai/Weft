from __future__ import annotations

import argparse
import array
import json
import importlib.util
import os
from pathlib import Path
import statistics
import struct
import sys

import weft
from kernels.dense.ime_i8_contract import ime_i8_contract
from weft.runtime import Artifact, Buffer, CompileOptions, Toolchain
from weft.runtime.driver import file_identity


def native_toolchain_identity(toolchain):
    selected = toolchain.resolve()
    identity = selected.identity(native_library=True)
    fields = ("cc", "cflags", "ldflags", "library_flags", "cc_file", "cc_version", "environment")
    return {**{name: identity[name] for name in fields},
            "declared_dependencies": [file_identity(path) for path in selected.dependencies]}


def native_link_binding(artifact):
    command = list(artifact.manifest["link_command"])
    if len(command) < 3 or command[-2] != "-o":
        raise ValueError("artifact lacks a complete native library link command")
    command[-1] = "<library-output>"
    for index, value in enumerate(command):
        if Path(value).name in ("kernel.c", "dispatch.c"):
            command[index] = "<" + Path(value).name + ">"
    return command


def compare_binding(previous, current, context, identity, shape):
    if context["kind"] != "weft-ime-contraction-comparison":
        raise ValueError("comparison artifact lacks this repro's native toolchain context")
    if context["native_toolchain"] != identity:
        raise ValueError("comparison changes the native C toolchain, flags, dependencies or environment")
    if context["shape"] != list(shape):
        raise ValueError("comparison artifact was saved for another input shape")
    if previous.target != current.target or previous.options != current.options:
        raise ValueError("comparison requires identical target and source/physical bindings")
    if previous.canonical_ir != current.canonical_ir:
        raise ValueError("comparison requires exactly the same canonical author program")
    if native_link_binding(previous) != native_link_binding(current):
        raise ValueError("comparison changes the recorded native compile/link invocation")
    if previous.interface.semantic_dict() != current.interface.semantic_dict():
        raise ValueError("comparison artifact has a different callable ABI")


def measure_calls(calls, output, expected, *, warmup, repetitions):
    import numpy as np

    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("native performance measurement requires one explicitly pinned CPU")
    eviction = np.zeros(64 * 1024 * 1024, dtype=np.uint8)
    for call in calls.values():
        for _ in range(warmup):
            call()
            if list(output) != expected:
                raise RuntimeError("contraction warmup disagrees with the exact scalar reference")
    samples = {name: [] for name in calls}
    order = tuple(calls)
    for repetition in range(repetitions):
        for name in order if repetition % 2 == 0 else tuple(reversed(order)):
            np.add(eviction, np.uint8(1), out=eviction)
            elapsed = calls[name].time_ns()
            if list(output) != expected:
                raise RuntimeError(f"{name} timed contraction disagrees with the exact scalar reference")
            samples[name].append(elapsed)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    cpu, = os.sched_getaffinity(0)
    frequency = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    return {"scope": "complete-contraction-native-entry-with-ctypes;64MiB-evict;shared-input-output-storage",
            "warmup": warmup, "repetitions": repetitions, "samples_ns": samples, "median_ns": medians,
            "frequency_khz": (frequency / "scaling_cur_freq").read_text().strip()
            if (frequency / "scaling_cur_freq").is_file() else None,
            "governor": (frequency / "scaling_governor").read_text().strip()
            if (frequency / "scaling_governor").is_file() else None,
            "previous_over_current": medians["previous"] / medians["current"] if "previous" in medians else None}


def derived_pack_calls(toolchain):
    source = Path(__file__).resolve().parents[4] / "source/weft/ggml/mul_mat/ime1/kernels.py"
    spec = importlib.util.spec_from_file_location("ime1_production", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("the unique IME production author source cannot be loaded")
    programs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = programs
    spec.loader.exec_module(programs)
    rows, inner = 32, 64
    plain = bytearray()
    packed = bytearray(rows * (inner // 32) * 18)
    expected_packed = bytearray(len(packed))
    for row in range(rows):
        for block in range(inner // 32):
            scale = struct.pack("<e", (row % 7 + 1) * 0.25)
            q = bytes((row * 7 + block * 11 + i * 13) & 255 for i in range(16))
            plain.extend(scale + q)
            record = ((row // 16) * (inner // 32) + block) * 288
            expected_packed[record + (row % 16) * 2:record + (row % 16) * 2 + 2] = scale
            for column in range(32):
                value = (q[column % 16] >> ((column // 16) * 4)) & 15
                offset = record + 32 + (column // 16) * 128 + (row % 16) * 8 + column % 8
                expected_packed[offset] |= value << (((column // 8) % 2) * 4)
    with weft.compile(programs.pack_q4_0, toolchain=toolchain,
                      options=CompileOptions(matrix_extension="spacemit-ime1")) as pack_kernel:
        pack_kernel(Buffer(plain, shape=(rows, inner), encoding="Q4_0"),
                    Buffer(packed, shape=(rows, inner), encoding="IMEQ4Centered_I[16]"))
        if packed != expected_packed:
            raise RuntimeError("native derived weight pack is not byte-exact")
    return {"derived_pack_numeric": "byte-exact", "derived_pack_shape": [rows, inner]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiled i8 contraction through native Weft JIT")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument("--ldflag", action="append", default=[])
    parser.add_argument("--family", choices=("dense", "derived-pack", "all"), default="all",
                        help="select dense contraction/composition, derived packing, or both")
    parser.add_argument("--mr", type=int, default=4)
    parser.add_argument("--nr", type=int, default=4)
    parser.add_argument("--matrix-extension", choices=("none", "spacemit-ime1"),
                        default="spacemit-ime1")
    parser.add_argument("--local-tile-strategy",
                        choices=("target-default", "atom-partitioned", "register-grouped"),
                        default="target-default")
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"),
                        help="explicit contraction shape, including non-multiple tails")
    parser.add_argument("--save-artifact", type=Path, help="save the selected contraction with its comparison context")
    parser.add_argument("--compare-artifact", type=Path, help="load an earlier saved contraction on the same buffers")
    parser.add_argument("--measure", action="store_true", help="measure full native contraction calls after exact checking")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.mr <= 0 or args.nr <= 0 or args.warmup <= 0 or args.repetitions <= 0:
        parser.error("tile extents, warmup and repetitions must be positive")
    toolchain = Toolchain(args.compiler, (args.cc,), tuple(args.cflag), tuple(args.ldflag))
    if args.family == "derived-pack":
        if args.shape is not None or args.save_artifact is not None or args.compare_artifact is not None or args.measure:
            parser.error("shape, artifact comparison and measurement options require the dense family")
        print(json.dumps({"execution": "native-jit", "family": args.family,
                          **derived_pack_calls(toolchain)}, sort_keys=True))
        return
    m, n, k = args.shape if args.shape is not None else (max(8, args.mr), max(8, args.nr), 16)
    if min(m, n, k) <= 0:
        parser.error("shape dimensions must be positive")
    lhs = array.array("b", ((index * 7 + 3) % 29 - 14 for index in range(m * k)))
    rhs = array.array("b", ((index * 11 + 5) % 31 - 15 for index in range(k * n)))
    output = array.array("i", [0]) * (m * n)
    expected = [((sum(lhs[row * k + inner] * rhs[inner * n + column]
                     for inner in range(k)) + 0x80000000) & 0xffffffff) - 0x80000000
                for row in range(m) for column in range(n)]
    views = (Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(k, n)),
             Buffer(output, shape=(m, n)))
    options = CompileOptions(meta={"MR": args.mr, "NR": args.nr, "KB": 8},
                             matrix_extension=args.matrix_extension,
                             local_tile_strategy=args.local_tile_strategy,
                             require_extensions=() if args.matrix_extension == "none"
                             else (args.matrix_extension,))
    kernel = weft.jit(ime_i8_contract, options=options, toolchain=toolchain)
    kernel(*views)
    if list(output) != expected:
        raise RuntimeError("native multi-tile/multi-K contraction is not exact")
    kernel(*views)
    if list(output) != expected:
        raise RuntimeError("reused native contraction is not exact")
    if weft.compile(ime_i8_contract, options=options, toolchain=toolchain) is not kernel.compiled:
        raise RuntimeError("same-binding native IME call did not reuse its artifact")
    identity = native_toolchain_identity(toolchain)
    current = kernel.compiled.artifact
    previous = None
    calls = {"current": kernel.prepare(*views)}
    if args.compare_artifact is not None:
        artifact = Artifact(args.compare_artifact)
        context = json.loads((artifact.path / "comparison.json").read_text())
        compare_binding(artifact, current, context, identity, (m, n, k))
        previous = artifact.load(target=kernel.compiled.target, toolchain=toolchain)
        calls["previous"] = previous.prepare(*views)
        calls["previous"]()
        if list(output) != expected:
            previous.close()
            raise RuntimeError("comparison contraction disagrees with the exact scalar reference")
    try:
        measurement = measure_calls(calls, output, expected, warmup=args.warmup,
                                    repetitions=args.repetitions) if args.measure else None
        if args.save_artifact is not None:
            saved = current.save(args.save_artifact)
            (saved.path / "comparison.json").write_text(json.dumps({
                "kind": "weft-ime-contraction-comparison", "shape": [m, n, k],
                "native_toolchain": identity, "measurement": measurement}, indent=2))
    finally:
        if previous is not None:
            previous.close()
    alternate_strategy = ("atom-partitioned" if kernel.compiled.metadata[
        "local_tile_realizations"].get("register-grouped", 0) else args.local_tile_strategy)
    alternative = weft.compile(ime_i8_contract, toolchain=toolchain,
                              options=CompileOptions(meta=options.meta,
                                                     unroll=1 if alternate_strategy != args.local_tile_strategy else 2,
                                                     local_tile_strategy=alternate_strategy,
                                                     matrix_extension=args.matrix_extension,
                                                     require_extensions=options.require_extensions))
    if alternative is kernel.compiled:
        raise RuntimeError("different native IME binding reused its artifact")
    alternative(*views)
    if list(output) != expected:
        raise RuntimeError("alternative native contraction is not exact")
    composition_source = Path(__file__).resolve().parents[4] / "source/weft/dense/i8_bias_clamp.py"
    composition_spec = importlib.util.spec_from_file_location("dense_composition", composition_source)
    if composition_spec is None or composition_spec.loader is None:
        raise RuntimeError("the unique dense composition source cannot be loaded")
    composition = importlib.util.module_from_spec(composition_spec)
    sys.modules[composition_spec.name] = composition
    composition_spec.loader.exec_module(composition)
    bias = array.array("i", (column * 13 - 47 for column in range(n)))
    composed_output = array.array("i", [0]) * (m * n)
    composed = weft.compile(composition.i8_bias_clamp, options=options, toolchain=toolchain)
    composed(views[0], views[1], Buffer(bias, shape=(n,)),
             Buffer(composed_output, shape=(m, n)))
    if list(composed_output) != [min(max(value + bias[index % n], 0), 127)
                                for index, value in enumerate(expected)]:
        raise RuntimeError("ordinary helper dot/bias/clamp composition is not exact")
    pack_result = derived_pack_calls(toolchain) if args.family == "all" else {}
    print(json.dumps({
        "execution": "native-jit", "family": args.family, "shape": [m, n, k], "numeric": "exact",
        "measurement": measurement, "native_toolchain": identity,
        "input_output_addresses": [view.address for view in views],
        "target": kernel.compiled.target.to_dict(),
        "host": {"kernel": os.uname().release, "machine": os.uname().machine,
                 "affinity": sorted(os.sched_getaffinity(0))},
        "saved_artifact": None if args.save_artifact is None else str(args.save_artifact.resolve()),
        "compared_artifact": None if args.compare_artifact is None else str(args.compare_artifact.resolve()),
        "source_bindings": dict(options.meta),
        "vlen_bits": kernel.compiled.target.vlen_bits,
        "matrix_extensions": kernel.compiled.metadata["matrix_extensions"],
        "required_extensions": kernel.compiled.metadata["required_extensions"],
        "used_extensions": kernel.compiled.metadata["used_extensions"],
        "local_tile_strategy": kernel.compiled.metadata["local_tile_strategy"],
        "local_tile_realizations": dict(kernel.compiled.metadata["local_tile_realizations"]),
        "alternate_local_tile_strategy": alternative.metadata["local_tile_strategy"],
        "matrix_cpu_sets": {name: sorted(cpus)
                            for name, cpus in kernel.compiled.target.matrix_extensions},
        "same_binding_reused": True, "different_binding_specialized": True,
        "different_strategy_specialized": alternate_strategy != args.local_tile_strategy,
        **pack_result,
        "helper_composition_numeric": "exact",
    }, sort_keys=True))
    kernel.compiled.close()
    alternative.close()
    composed.close()


if __name__ == "__main__":
    main()
