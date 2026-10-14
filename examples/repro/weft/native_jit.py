from __future__ import annotations

import argparse
import array
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import statistics
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import weft
from weft.cli import load_definition
from kernels.dense.gemm_f32 import gemm_f32
from weft.runtime import Artifact, Buffer, CompilationError, CompileOptions, Toolchain, PhysicalInput, extension_catalog


PROJECT = Path(__file__).resolve().parents[3]


def reduction_calls(toolchain: Toolchain, cache: Path) -> dict:
    import numpy as np

    source = PROJECT / "source/weft/dense/reductions.py"
    definition = load_definition(f"{source}:statistics")
    maximum = 0.0
    with weft.jit(definition, toolchain=toolchain, cache=cache) as kernel:
        for width in (0, 1, 3, 33, 257):
            storage = np.linspace(-3, 4, max(1, width * 2), dtype=np.float32)
            x = storage[:width * 2:2]
            y = np.empty(3, dtype=np.float32)
            kernel(X=x, Y=y)
            expected = np.array([x.sum(), x.max(initial=-np.inf), x.min(initial=np.inf)], dtype=np.float32)
            np.testing.assert_allclose(y, expected, atol=1e-4, rtol=2e-3)
            finite = np.isfinite(expected)
            maximum = max(maximum, float(np.abs(y[finite] - expected[finite]).max(initial=0)))

    definition = load_definition(f"{source}:product_sum")
    x, weight = np.empty(32, dtype=np.float32), np.empty(32, dtype=np.float32)
    x[:16], weight[:16] = -2 ** 20, 2 ** 20
    x[16:], weight[16:] = 2 ** 20 + 0.125, 2 ** 20 - 0.125
    expected = np.sum(np.multiply(x, weight, dtype=np.float32), dtype=np.float32)
    with weft.jit(definition, options=CompileOptions(meta={"BLOCK": 32}),
                  toolchain=toolchain, cache=cache) as kernel:
        y = np.empty(1, dtype=np.float32)
        kernel(X=x, Weight=weight, Y=y)
        np.testing.assert_allclose(y[0], expected, atol=1e-4, rtol=2e-3)
        rounded = float(y[0])
    from kernels.mul_mat.f16 import production_mul_mat_f16

    rng = np.random.default_rng(418)
    half_error = 0.0
    options = CompileOptions(meta={"NC": 32, "KC": 4096, "MC": 16, "MR": 4, "NR": 2, "KB": 4096})
    with weft.jit(production_mul_mat_f16, options=options, toolchain=toolchain, cache=cache) as kernel:
        for m, n, k in ((1, 3, 1), (5, 7, 33), (17, 35, 259)):
            x = rng.normal(0, 0.3, (m, k)).astype(np.float32)
            weight = rng.normal(0, 0.3, (n, k)).astype(np.float16)
            scratch = np.empty((m, k), dtype=np.float16)
            y = np.empty((m, n), dtype=np.float32)
            kernel(W=weight, X=x, Xh=scratch, Y=y)
            converted = x.astype(np.float16)
            np.testing.assert_allclose(scratch.astype(np.float32), converted.astype(np.float32),
                                       atol=1e-4, rtol=2e-3)
            expected = converted.astype(np.float32) @ weight.astype(np.float32).T
            np.testing.assert_allclose(y, expected, atol=1e-4, rtol=2e-3)
            half_error = max(half_error, float(np.abs(y - expected).max()))
    mixed_errors = {}
    for kind in ("f16", "bf16"):
        definition = load_definition(f"{source}:mixed_dot_{kind}")
        error = 0.0
        with weft.jit(definition, options=CompileOptions(meta={"BLOCK": 32}, pipeline_depth=2),
                      toolchain=toolchain, cache=cache) as kernel:
            for width in (1, 33, 257):
                x = rng.normal(0, 0.3, width).astype(np.float32)
                original = rng.normal(0, 0.3, width).astype(np.float32)
                if kind == "f16":
                    weight = original.astype(np.float16)
                    reference = weight.astype(np.float32)
                else:
                    words = (original.view(np.uint32) >> 16).astype(np.uint16)
                    weight = Buffer(words, shape=(width,), encoding="dense.bf16")
                    reference = (words.astype(np.uint32) << 16).view(np.float32)
                y = np.empty(1, dtype=np.float32)
                kernel(X=x, Weight=weight, Y=y)
                expected = x @ reference
                np.testing.assert_allclose(y[0], expected, atol=1e-4, rtol=2e-3)
                error = max(error, float(np.abs(y[0] - expected)))
        mixed_errors[kind] = error
    return {"shared_reduction_max_absolute_error": maximum,
            "separately_rounded_product_sum": rounded,
            "f16_contraction_max_absolute_error": half_error,
            "mixed_storage_max_absolute_error": mixed_errors,
            "shared_reduction_strided_and_empty": True}


def wide_fma_calls(toolchain: Toolchain, cache: Path) -> dict:
    """Complete, partial and zero-K issues through the ordinary native JIT."""
    import numpy as np

    source = PROJECT / "source/weft/dense/contraction.py"
    rng = np.random.default_rng(421)
    errors, issues = {}, {}
    for name, dtype, binding in (("output_lane_f64", np.float64, {}),
                                 ("row_major_matmul_f16", np.float16, {"MR": 4, "NR": 64})):
        definition = load_definition(f"{source}:{name}")
        options = CompileOptions(meta=binding, lmul_eighths=64, unroll=4)
        with weft.jit(definition, options=options, toolchain=toolchain, cache=cache) as kernel:
            maximum = 0.0
            for m, n, k in ((3, 64, 128), (5, 67, 131), (1, 9, 0)):
                x = rng.normal(0, 0.3, (m, k * 2)).astype(dtype)[:, ::2]
                weight = rng.normal(0, 0.3, (k, n * 2)).astype(dtype)[:, ::2]
                output_dtype = np.float64 if dtype == np.float64 else np.float32
                storage = np.full((m, n * 2 + 2), 37, dtype=output_dtype)
                output = storage[:, 1:-1:2]
                expected = x.astype(output_dtype) @ weight.astype(output_dtype)
                kernel(X=x, Weight=weight, Y=output)
                atol, rtol = (1e-12, 1e-12) if dtype == np.float64 else (1e-4, 2e-3)
                np.testing.assert_allclose(output, expected, atol=atol, rtol=rtol, equal_nan=False)
                np.testing.assert_array_equal(storage[:, ::2], 37)
                np.testing.assert_array_equal(storage[:, -1], 37)
                maximum = max(maximum, float(np.abs(output - expected).max(initial=0)))
            errors[name] = maximum
            issues[name] = kernel.compiled.artifact.riscv_ir.count('"weft_riscv.rvv_fma_issue"')
    return {"wide_fma_max_absolute_errors": errors, "closed_fma_issues": issues}


def panel_selection_calls(toolchain: Toolchain, cache: Path) -> dict:
    """Selected memory programs remain callable after checkpoint serialization."""
    import numpy as np
    from weft.programs import get
    from weft.runtime import NativeTarget, PanelTransferSelections, build

    definition = get("linear").definition
    options = CompileOptions(meta={"MC": 32, "NC": 64, "MR": 32, "NR": 64}, lmul_eighths=32)
    target = NativeTarget.discover(toolchain).build_target
    prepared = weft.prepare(definition, target=target, options=options, checkpoint="panel-transfers",
                            toolchain=toolchain, cache=cache)
    selections = prepared.selections(toolchain=toolchain)
    if PanelTransferSelections.from_dict(selections.to_dict()) != selections:
        raise AssertionError("panel inspection lost a typed physical choice")
    transfers = [transfer for kernel in selections.kernels for transfer in kernel.transfers]
    if not transfers or not any(transfer.unit_axis for transfer in transfers):
        raise AssertionError("projection did not expose the runtime transfer orientation")
    saved = prepared.save(cache / "selected-panel-transfer.json")
    restored = PhysicalInput.load(saved)
    if restored != prepared:
        raise AssertionError("serialized panel input changed its source, bindings or physical choices")
    artifact = restored.complete(toolchain=toolchain, cache=cache)
    direct = build(definition, target=target, options=options, toolchain=toolchain, native_library=False, cache=cache)
    if artifact.source != direct.source or artifact.riscv_ir != direct.riscv_ir:
        raise AssertionError("panel checkpoint did not resume the exact shared main suffix")
    passes = [step.pass_name for step in artifact.compilation.pipeline]
    if any("select-panel-transfers" in name for name in passes):
        raise AssertionError("resuming a selected transfer reran structural selection")
    if not any("materialize-panel-transfers" in name for name in passes):
        raise AssertionError("resuming a selected transfer skipped physical loop materialization")

    rng = np.random.default_rng(425)
    maximum = 0.0
    with artifact.load(toolchain=toolchain) as kernel:
        for shape, orientation in (((32, 64, 128), "rows"), ((35, 67, 259), "columns"),
                                   ((35, 67, 131), "strided"), ((3, 5, 0), "rows")):
            m, n, k = shape
            x = rng.normal(0, 0.2, (m, k)).astype(np.float32)
            weight = rng.normal(0, 0.2, (n, k)).astype(np.float32)
            if orientation == "columns":
                x, weight = np.asfortranarray(x), np.asfortranarray(weight)
            elif orientation == "strided":
                x_storage = np.full((m * 2, k * 2), np.nan, dtype=np.float32)
                w_storage = np.full((n * 2, k * 2), np.nan, dtype=np.float32)
                x_storage[::2, ::2], w_storage[::2, ::2] = x, weight
                x, weight = x_storage[::2, ::2], w_storage[::2, ::2]
            storage = np.full((m + 2, n * 2 + 2), 37, dtype=np.float32)
            y = storage[1:-1, 1:-1:2]
            expected = x @ weight.T
            kernel(X=x, Weight=weight, Y=y)
            np.testing.assert_allclose(y, expected, atol=1e-4, rtol=2e-3, equal_nan=False)
            np.testing.assert_array_equal(storage[:, ::2], 37)
            np.testing.assert_array_equal(storage[:, -1], 37)
            np.testing.assert_array_equal(storage[[0, -1]], 37)
            maximum = max(maximum, float(np.abs(y - expected).max(initial=0)))

    # These are mutations of the actual prepared program. The public consumer
    # must reject each broken physical contract before emitting a callable C
    # artifact; constructing a separate hand-written IR fixture is unnecessary.
    rejections = []

    def reject(ir, entry, message, label):
        result = subprocess.run([toolchain.compiler_path(), "--emit=intrinsic-c", *entry],
                                input=ir, text=True, capture_output=True)
        if result.returncode == 0 or message not in result.stderr:
            raise AssertionError(f"{label} was not rejected by its physical contract: {result.stderr}")
        rejections.append(label)

    marker = "weft.riscv.panel_transfer_input"
    if marker not in restored.ir:
        raise AssertionError("panel checkpoint omitted its remaining binding")
    reject(restored.ir, ["--resume-contraction-input"], "matching checkpoint", "wrong-checkpoint")
    reject(restored.ir, ["--resume-panel-transfer-input", "--auto-lmul-eighths=8"],
           "build bindings apply only", "replaced-binding")
    bound = re.search(r'maximum_fields = (\d+) : i64', restored.ir)
    if bound is None:
        raise AssertionError("panel copy omitted its finite field domain")
    damaged = restored.ir[:bound.start(1)] + "1" + restored.ir[bound.end(1):]
    reject(damaged, ["--resume-panel-transfer-input"], "field domain differs", "changed-field-domain")
    packet = re.search(r'scalar_packet<(\d+), (\d+), (\d+), (\d+), (\d+)>', artifact.riscv_ir)
    if packet is None:
        raise AssertionError("complete projection did not select a scalar packet")
    damaged = artifact.riscv_ir[:packet.start(1)] + "999999" + artifact.riscv_ir[packet.end(1):]
    reject(damaged, [], "numeric coordinates", "changed-packet-axis")
    damaged = artifact.riscv_ir[:packet.start(5)] + "33" + artifact.riscv_ir[packet.end(5):]
    reject(damaged, [], "live FPR frontier", "oversized-packet-frontier")

    packet_groups = [operation for operation in artifact.physical_operations.operations if operation.scalar_packet]
    if not packet_groups:
        raise AssertionError("final artifact omitted selected packet scheduling from inspection")
    if artifact.physical_operations.to_dict() != type(artifact.physical_operations).from_dict(
            artifact.physical_operations.to_dict()).to_dict():
        raise AssertionError("final physical operation inspection lost its scalar resource contract")

    # The two private panels fit in 24 KiB plus their original padding. A
    # stronger optional alignment cannot make this otherwise legal binding
    # fail merely because the extra padding no longer fits the exact budget.
    # Execute this separately selected artifact, rather than inferring its
    # behavior from the resource estimate of the larger-budget compilation.
    capacity = 24 * 1024 + 16
    bounded_options = CompileOptions(**{**options.to_dict(), "private_stack_bytes": capacity})
    bounded = build(definition, target=target, options=bounded_options, toolchain=toolchain, cache=cache)
    bounded_bytes = bounded.physical_resources["local_storage_bytes"]
    if bounded_bytes > capacity:
        raise AssertionError("optional local alignment exceeded the declared private storage capacity")
    with bounded.load(toolchain=toolchain) as kernel:
        bx = rng.normal(0, 0.2, (35, 131)).astype(np.float32)
        bw = rng.normal(0, 0.2, (67, 131)).astype(np.float32)
        by = np.empty((35, 67), dtype=np.float32)
        kernel(X=bx, Weight=bw, Y=by)
        np.testing.assert_allclose(by, bx @ bw.T, atol=1e-4, rtol=2e-3, equal_nan=False)
    if bounded.options.meta != options.meta:
        raise AssertionError("storage alignment selection changed an author binding")
    if bounded.options.private_stack_bytes != capacity:
        raise AssertionError("native completion replaced the requested storage budget")
    return {"panel_selection_max_absolute_error": maximum,
            "panel_selection_checkpoint_entry": artifact.compilation.entry,
            "panel_selection_rejections": rejections,
            "panel_selection_packet_operations": sum(operation.occurrences for operation in packet_groups),
            "panel_selection_fpr_frontier": max(operation.scalar_packet.live_registers for operation in packet_groups),
            "panel_selection_bounded_storage_bytes": bounded_bytes}


def contraction_calls(toolchain: Toolchain, cache: Path) -> dict:
    """Run the same numerical contracts with K lanes and free-axis lanes.

    Strided inputs, partial tiles, exact half promotion, shared batch axes and
    numeric snapshots exercise the operand-supply boundary of the compiler.
    """
    import numpy as np

    source = PROJECT / "source/weft/dense/contraction.py"
    rng = np.random.default_rng(924)
    errors = {}
    roundtripped = set()
    checkpoints = set()
    checkpoint_rejections = set()

    def resume_selection(definition, options, artifact, arguments, expected, *, scalar_tolerance):
        """Inspectable selection and direct compilation must use one main chain."""
        category = "panel" if artifact.physical_resources["local_storage_bytes"] > 0 else "direct"
        if category in checkpoints:
            return
        for boundary in ("contractions", "panel-transfers", "layout"):
            prepared = weft.prepare(definition, target=artifact.target, options=options,
                                    checkpoint=boundary, toolchain=toolchain, cache=cache)
            saved = prepared.save(cache / f"{category}-{boundary}.json")
            restored = PhysicalInput.load(saved)
            if restored.ir != prepared.ir or restored.canonical_ir != artifact.canonical_ir:
                raise AssertionError("saved physical input changed its numerical or physical program")
            resumed = restored.complete(toolchain=toolchain, cache=cache)
            if resumed.source != artifact.source or resumed.riscv_ir != artifact.riscv_ir:
                raise AssertionError("resumed main suffix disagrees with uninterrupted compilation")
            entry = {"contractions": "contraction-input", "panel-transfers": "panel-transfer-input",
                     "layout": "layout-input"}[boundary]
            if resumed.compilation.entry != entry:
                raise AssertionError("physical input used another compilation entry")
            with resumed.load(toolchain=toolchain) as loaded:
                loaded(**arguments)
            np.testing.assert_allclose(arguments["Y"], expected, atol=scalar_tolerance[0],
                                       rtol=scalar_tolerance[1], equal_nan=False)
            # Native completion and source-only replay share selection. Source
            # inspection must never reselect a new tile or emit another C body.
            emitted = restored.complete(toolchain=toolchain, native_library=False, cache=cache)
            if emitted.source != resumed.source or emitted.manifest["native_library"]:
                raise AssertionError("source-only checkpoint replay changed the selected program")
            cached = weft.prepare(definition, target=artifact.target, options=options,
                                 checkpoint=boundary, toolchain=toolchain, cache=cache)
            if cached.ir != restored.ir:
                raise AssertionError("cached preparation changed its selected Physical IR")
            wrong_entry = "--resume-layout-input" if boundary == "contractions" else "--resume-contraction-input"
            rejected = subprocess.run([toolchain.compiler_path(), "--emit=intrinsic-c", wrong_entry],
                                      input=restored.ir, text=True, capture_output=True)
            if rejected.returncode == 0 or "matching checkpoint" not in rejected.stderr:
                raise AssertionError("compiler accepted the wrong physical checkpoint entry")
            checkpoint_rejections.add("wrong-entry")
            rebound = subprocess.run([toolchain.compiler_path(), "--emit=intrinsic-c",
                                     f"--resume-{entry}",
                                     "--auto-lmul-eighths=8"],
                                    input=restored.ir, text=True, capture_output=True)
            if rebound.returncode == 0 or "build bindings apply only" not in rebound.stderr:
                raise AssertionError("compiler allowed a checkpoint's physical binding to be replaced")
            checkpoint_rejections.add("binding-replacement")
            if boundary == "contractions":
                if "#weft_riscv.contraction_plan<" not in restored.ir:
                    raise AssertionError("contraction checkpoint omitted its selected representation")
                # Destroy one selected tile's resource witness while leaving
                # its numerical operands intact. This must fail at consumption.
                match = re.search(r'contraction_case<\d+, (<\[[^]]*\], \d+, [01], \d+, \d+, [124], )(\d+)(, \d+, \d+>)',
                                  restored.ir)
                if match is None:
                    raise AssertionError("repro could not locate the typed contraction tile")
                before = "".join(match.groups())
                damaged = restored.ir.replace(before, match[1] + "1" + match[3])
                rejected = subprocess.run([toolchain.compiler_path(), "--emit=intrinsic-c", "--resume-contraction-input"],
                                          input=damaged, text=True, capture_output=True)
                if rejected.returncode == 0 or "residency cost" not in rejected.stderr:
                    raise AssertionError("compiler accepted an understated contraction resource witness")
                checkpoint_rejections.add("understated-residency")
            if boundary == "panel-transfers":
                choices = restored.selections(toolchain=toolchain)
                for selected_kernel in choices.kernels:
                    for transfer in selected_kernel.transfers:
                        for case in transfer.cases:
                            if case.main.register_groups > 8 or case.tail.fields != 1:
                                raise AssertionError("selected transfer omitted bounded residency or remainder coverage")
                if '"weft_riscv.dense_panel_copy"' in resumed.riscv_ir:
                    raise AssertionError("panel copy survived the shared materialization suffix")
        checkpoints.add(category)

    def check(name, x, weight, expected, bindings, *, unroll=1, pipeline_depth=1,
              k_steps=2, arithmetic=None):
        definition = load_definition(f"{source}:{name}")
        options = CompileOptions(meta=bindings, lmul_eighths=32,
                                 unroll=unroll, pipeline_depth=pipeline_depth,
                                 contraction_k_steps=k_steps)
        with weft.jit(definition, options=options, toolchain=toolchain, cache=cache) as kernel:
            shape = expected.shape[:-1] + (expected.shape[-1] * 2,)
            guarded = np.full(shape, 1234, dtype=expected.dtype)
            output = guarded[..., ::2]
            kernel(X=x, Weight=weight, Y=output)
            selected = kernel.compiled.artifact.riscv_ir
            reload_case = "full-segment" if '"rvv.tile-store.segment.full"' in selected else "ordinary"
            if reload_case not in roundtripped:
                artifact = kernel.compiled.artifact.path
                emitted = subprocess.check_output([toolchain.compiler_path(),
                    str(artifact / "physical.mlir"), "--emit=intrinsic-c"])
                if emitted != (artifact / "kernel.c").read_bytes():
                    raise AssertionError("physical IR reload changed the selected C program")
                roundtripped.add(reload_case)
            dtype = expected.dtype if arithmetic is None else np.dtype(arithmetic)
            atol, rtol = (1e-12, 1e-12) if dtype == np.float64 else (1e-4, 2e-3)
            np.testing.assert_allclose(output, expected, atol=atol, rtol=rtol, equal_nan=False)
            np.testing.assert_array_equal(guarded[..., 1::2], 1234)
            resume_selection(definition, options, kernel.compiled.artifact,
                             {"X": x, "Weight": weight, "Y": output}, expected,
                             scalar_tolerance=(atol, rtol))
            np.testing.assert_array_equal(guarded[..., 1::2], 1234)
            error = float(np.abs(output - expected).max(initial=0))
            errors[name] = max(errors.get(name, 0.0), error)

    for dtype, name in ((np.float32, "row_major_matmul"), (np.float16, "row_major_matmul_f16")):
        for m, n, k in ((1, 3, 1), (5, 7, 33), (4, 8, 65)):
            x = rng.normal(0, 0.2, (m, k * 2)).astype(dtype)[:, ::2]
            weight = rng.normal(0, 0.2, (k, n * 2)).astype(dtype)[:, ::2]
            expected = x.astype(np.float32) @ weight.astype(np.float32)
            check(name, x, weight, expected, {"MR": 2, "NR": 4}, unroll=3)
    for m, n, k in ((0, 7, 33), (3, 0, 5), (3, 7, 0), (1, 3, 1), (5, 17, 33), (35, 67, 259)):
        x = rng.normal(0, 0.2, (m, k * 2)).astype(np.float32)[:, ::2]
        weight = rng.normal(0, 0.2, (k, n * 2)).astype(np.float32)[:, ::2]
        check("dynamic_matmul", x, weight, x @ weight, {})
        check("lhs_lane_matmul", x.T, weight.T, x @ weight, {})
    # These author tiles contain multiple compiler microtiles and K panels.
    # The final K fragment and both output tails are live numerical boundaries.
    for dtype, name in ((np.float32, "row_major_matmul"), (np.float16, "row_major_matmul_f16")):
        for m, n, k in ((35, 67, 259), (33, 65, 0), (64, 128, 128)):
            x = rng.normal(0, 0.2, (m, k * 2)).astype(dtype)[:, ::2]
            weight = rng.normal(0, 0.2, (k, n * 2)).astype(dtype)[:, ::2]
            expected = x.astype(np.float32) @ weight.astype(np.float32)
            check(name, x, weight, expected, {"MR": 32, "NR": 64}, pipeline_depth=2)
            check(name, x, weight, expected, {"MR": 32, "NR": 64}, k_steps=4)
    for m, n, k in ((3, 7, 0), (5, 17, 33), (35, 67, 259)):
        x = rng.normal(0, 0.2, (m, k * 2)).astype(np.float64)[:, ::2]
        weight = rng.normal(0, 0.2, (k, n * 2)).astype(np.float64)[:, ::2]
        check("output_lane_f64", x, weight, x @ weight, {}, unroll=3)
        x32, w32 = x.astype(np.float32), weight.astype(np.float32)
        check("matmul_cast", x32, w32, (x32 @ w32).astype(np.float64), {"MR": 3, "NR": 8},
              arithmetic=np.float32)
    for batch, m, n, k in ((0, 3, 7, 13), (2, 5, 17, 33), (3, 1, 9, 0), (2, 35, 67, 259), (2, 67, 35, 129)):
        x = rng.normal(0, 0.2, (batch, m, k * 2)).astype(np.float32)[..., ::2]
        weight = rng.normal(0, 0.2, (batch, k, n * 2)).astype(np.float32)[..., ::2]
        check("batched_matmul", x, weight, x @ weight, {}, unroll=2)
        check("batched_lhs_lane", x.swapaxes(-1, -2), weight.swapaxes(-1, -2), x @ weight, {}, unroll=2)
    definition = load_definition(f"{source}:matmul_observers")
    for m, n, k in ((5, 17, 33), (7, 35, 129)):
        x = rng.normal(0, 0.2, (m, k)).astype(np.float32)
        weight = rng.normal(0, 0.2, (k, n)).astype(np.float32)
        expected = x @ weight
        with weft.jit(definition, options=CompileOptions(meta={"M": m, "N": n}, lmul_eighths=32),
                      toolchain=toolchain, cache=cache) as kernel:
            y, norm = np.empty_like(expected), np.empty(m, dtype=np.float32)
            kernel(X=x, Weight=weight, Y=y, Norm=norm)
            np.testing.assert_allclose(y, expected, atol=1e-4, rtol=2e-3)
            np.testing.assert_allclose(norm, (expected * expected).sum(axis=1), atol=1e-4, rtol=2e-3)
            errors["matmul_observers"] = max(errors.get("matmul_observers", 0.0), float(np.abs(y - expected).max()))
    definition = load_definition(f"{source}:matmul_epilogue")
    with weft.jit(definition, options=CompileOptions(meta={"MR": 32, "NR": 64}, lmul_eighths=32),
                  toolchain=toolchain, cache=cache) as kernel:
        for m, n, k in ((5, 17, 33), (35, 67, 259)):
            x = rng.normal(0, 0.2, (m, k)).astype(np.float32)
            weight = rng.normal(0, 0.2, (k, n)).astype(np.float32)
            bias = rng.normal(0, 0.2, n * 2).astype(np.float32)[::2]
            gate = rng.normal(0, 0.2, m * 2).astype(np.float32)[::2]
            guard = np.full((m, n * 2), 1234, dtype=np.float32)
            y = guard[:, ::2]
            kernel(X=x, Weight=weight, Bias=bias, Gate=gate, Y=y, scale=0.75)
            dot = x @ weight
            expected = (dot + bias) * (dot * gate[:, None]) * np.float32(0.75)
            np.testing.assert_allclose(y, expected, atol=1e-4, rtol=2e-3)
            np.testing.assert_array_equal(guard[:, 1::2], 1234)
            errors["matmul_epilogue"] = max(errors.get("matmul_epilogue", 0.0), float(np.abs(y - expected).max()))
    definition = load_definition(f"{source}:paired_projections")
    options = CompileOptions(meta={"MR": 32, "NR": 64}, lmul_eighths=32)
    with weft.jit(definition, options=options, toolchain=toolchain, cache=cache) as kernel:
        for m, n, k in ((1, 1, 0), (5, 17, 33), (35, 67, 259)):
            x = rng.normal(0, 0.2, (m, k * 2)).astype(np.float32)[:, ::2]
            weights = [rng.normal(0, 0.2, (k, n * 2)).astype(np.float32)[:, ::2] for _ in range(2)]
            biases = [rng.normal(0, 0.2, n * 2).astype(np.float32)[::2] for _ in range(2)]
            guards = [np.full((m, n * 2), 1234, dtype=np.float32) for _ in range(2)]
            outputs = [value[:, ::2] for value in guards]
            arguments = {"X": x, "Weight0": weights[0], "Weight1": weights[1],
                         "Bias0": biases[0], "Bias1": biases[1], "Y0": outputs[0], "Y1": outputs[1]}
            kernel(**arguments)
            for index, output in enumerate(outputs):
                expected = x @ weights[index] + biases[index]
                np.testing.assert_allclose(output, expected, atol=1e-4, rtol=2e-3)
                np.testing.assert_array_equal(guards[index][:, 1::2], 1234)
                errors["paired_projections"] = max(errors.get("paired_projections", 0.0),
                                                    float(np.abs(output - expected).max(initial=0)))
        prepared = weft.prepare(definition, target=kernel.compiled.artifact.target, options=options,
                                toolchain=toolchain, cache=cache)
        summary = prepared.selections(toolchain=toolchain)
        if len(summary.kernels[0].contractions) != 2:
            raise AssertionError("paired projection checkpoint lost an independent closed contraction")
        replay = prepared.complete(toolchain=toolchain, native_library=False, cache=cache)
        if replay.source != kernel.compiled.artifact.source:
            raise AssertionError("sequential panel lifetimes changed while replaying their selected plans")
    for m, n, k in ((0, 7, 33), (3, 0, 5), (3, 7, 0), (1, 3, 1), (5, 7, 33), (17, 35, 259)):
        x = rng.normal(0, 0.2, (m, k * 2)).astype(np.float32)[:, ::2]
        weight = rng.normal(0, 0.2, (n, k)).astype(np.float32)
        check("dense_projection", x, weight, x @ weight.T, {})
        check("renamed_projection", x, weight, x @ weight.T, {})
        definition = load_definition(f"{source}:dense_projection_bias")
        bias = rng.normal(0, 0.2, n * 2).astype(np.float32)[::2]
        expected = (x @ weight.T) * np.float32(0.75) + bias
        with weft.jit(definition, toolchain=toolchain, cache=cache,
                      options=CompileOptions(lmul_eighths=32)) as kernel:
            output = np.empty_like(expected)
            kernel(X=x, Weight=weight, Bias=bias, Y=output, scale=0.75)
            np.testing.assert_allclose(output, expected, atol=1e-4, rtol=2e-3)
            errors["dense_projection_bias"] = max(errors.get("dense_projection_bias", 0.0),
                                                  float(np.abs(output - expected).max(initial=0)))
    for b, p, k in ((1, 1, 1), (5, 3, 33), (7, 2, 65)):
        x = rng.normal(0, 0.2, (b, p, k * 2)).astype(np.float32)[:, :, ::2]
        weight = rng.normal(0, 0.2, x.shape).astype(np.float32)
        check("batched_inner_product", x, weight, (x * weight).sum(axis=(1, 2)), {"ROWS": 3}, unroll=2)
    for m, n, p, k in ((1, 1, 1, 1), (5, 7, 3, 33), (4, 8, 2, 65)):
        x = rng.normal(0, 0.2, (m, p, k)).astype(np.float32)
        weight = rng.normal(0, 0.2, (n, p, k)).astype(np.float32)
        expected = x.reshape(m, -1) @ weight.reshape(n, -1).T
        check("pairwise_inner_product", x, weight, expected, {"MR": 3, "NR": 4}, pipeline_depth=2)
    for k in (0, 1, 7, 33, 257):
        x = rng.normal(0, 0.2, k * 2).astype(np.float64)[::2]
        weight = rng.normal(0, 0.2, k).astype(np.float64)
        check("dot_f64", x, weight, np.array([x @ weight]), {}, unroll=2)
    x = rng.normal(0, 0.2, (2, 17)).astype(np.float32)
    weight = rng.normal(0, 0.2, (17, 4)).astype(np.float32)
    check("snapshot_matmul", x, weight, x @ weight, {"M": 2, "N": 4})
    np.testing.assert_array_equal(weight, np.zeros_like(weight))
    low_bit = load_definition(f"{PROJECT / 'source/weft/dense/low_bit_contraction.py'}:low_bit_projection")
    for mr, nr in ((2, 4), (4, 16)):
        m, n, k = mr + 1, nr + 3, 64
        x = rng.integers(-128, 128, (m, k), dtype=np.int8)
        weight = rng.integers(0, 16, (n, k), dtype=np.uint8)
        blocks = weight.reshape(n, k // 32, 32)
        packed = (blocks[:, :, :16] | (blocks[:, :, 16:] << 4)).copy()
        y = np.empty((m, n), dtype=np.int32)
        with weft.jit(low_bit, toolchain=toolchain, cache=cache,
                      options=CompileOptions(meta={"MR": mr, "NR": nr})) as kernel:
            kernel(X=Buffer(x, shape=x.shape, encoding="SignedI8"),
                   Weight=Buffer(packed, shape=weight.shape, encoding="PackedU4"), Y=y)
            np.testing.assert_array_equal(y, x.astype(np.int32) @ weight.astype(np.int32).T)
    return {**wide_fma_calls(toolchain, cache), **panel_selection_calls(toolchain, cache),
            "contraction_max_absolute_errors": errors, "physical_ir_roundtrip": "same-selected-C",
            "physical_ir_roundtrip_paths": sorted(roundtripped),
            "contraction_numeric_snapshot_preserved": True, "low_bit_integer_contractions": "exact",
            "selected_checkpoint_native_paths": sorted(checkpoints),
            "selected_checkpoint_rejections": sorted(checkpoint_rejections)}


def dense_field_calls(toolchain: Toolchain, cache: Path) -> dict:
    """Public multi-output programs preserve snapshots and ordered aliasing."""
    import numpy as np

    rng = np.random.default_rng(592)
    maximum = 0.0
    cases = 0
    rejected = 0
    paths = {}
    modes = ("contiguous", "field-strided", "lane-strided", "broadcast-read",
             "transposed-output", "in-place", "shifted-alias", "overlapping-fields", "zero-field-stride")
    for entry, block in (("rope_interleaved", 16), ("rope_interleaved", 32), ("rope_split", 32)):
        definition = load_definition(f"{PROJECT / 'source/weft/programs/attention.py'}:{entry}")
        with weft.jit(definition, options=CompileOptions(meta={"BLOCK": block}),
                      toolchain=toolchain, cache=cache) as kernel:
            for width in (0, 1, 7, 33, 65):
                cosine = rng.normal(0, 0.3, (2, max(1, width) * 2)).astype(np.float32)[:, :width * 2:2]
                sine = rng.normal(0, 0.3, (2, max(1, width) * 2)).astype(np.float32)[:, :width * 2:2]
                for mode in modes:
                    # Explicit backing buffers let the comparison cover holes,
                    # source/output overlap and every untouched guard element.
                    capacity = max(1, width) * 8 + 16
                    source = rng.normal(0, 0.3, (2, 3, capacity)).astype(np.float32)
                    destination = np.full_like(source, 1234)
                    input_steps = (2, 1)
                    output_steps = (2, 1)
                    offset = 2
                    if mode == "field-strided": input_steps = (4, 2)
                    if mode == "lane-strided": input_steps = (4, 1)
                    if mode == "broadcast-read": input_steps = (0, 1)
                    if mode == "transposed-output": output_steps = (1, max(1, width))
                    if mode == "overlapping-fields": output_steps = (1, 1)
                    if mode == "zero-field-stride": output_steps = (1, 0)
                    if mode in ("in-place", "shifted-alias"):
                        destination = source
                        if mode == "shifted-alias": offset += 2
                    expected_source = source.copy()
                    expected_destination = expected_source if destination is source else destination.copy()

                    def view(storage, steps, start):
                        return np.ndarray((2, 3, width, 2), dtype=np.float32,
                                          buffer=storage, offset=start * 4,
                                          strides=(*storage.strides[:2], steps[0] * 4, steps[1] * 4))

                    x = view(source, input_steps, 2)
                    y = view(destination, output_steps, offset)
                    ref_x = view(expected_source, input_steps, 2)
                    ref_y = view(expected_destination, output_steps, offset)
                    arguments = (x, y) if entry == "rope_interleaved" else (x.swapaxes(-1, -2), y.swapaxes(-1, -2))
                    invalid_output = (mode == "overlapping-fields" and width > 1) or (
                        mode == "zero-field-stride" and width > 0)
                    if invalid_output:
                        try:
                            kernel(X=arguments[0], Cos=cosine, Sin=sine, Y=arguments[1])
                        except ValueError as error:
                            if "must not overlap themselves" not in str(error): raise
                        else:
                            raise AssertionError("self-overlapping output bypassed the public View contract")
                        np.testing.assert_array_equal(destination, expected_destination)
                        rejected += 1
                        continue
                    # The author reads a complete BLOCK before either field
                    # store. A shifted alias may affect the next author BLOCK,
                    # so a whole-array NumPy expression would be a wrong oracle.
                    for token in range(2):
                        for head in range(3):
                            for begin in range(0, width, block):
                                end = min(width, begin + block)
                                first = ref_x[token, head, begin:end, 0].copy()
                                second = ref_x[token, head, begin:end, 1].copy()
                                c, s = cosine[token, begin:end], sine[token, begin:end]
                                ref_y[token, head, begin:end, 0] = first * c - second * s
                                ref_y[token, head, begin:end, 1] = second * c + first * s
                    kernel(X=arguments[0], Cos=cosine, Sin=sine, Y=arguments[1])
                    np.testing.assert_allclose(destination, expected_destination, atol=1e-4, rtol=2e-3)
                    np.testing.assert_allclose(source, expected_source, atol=1e-4, rtol=2e-3)
                    maximum = max(maximum, float(np.abs(destination - expected_destination).max(initial=0)))
                    cases += 1
            physical = kernel.compiled.artifact.riscv_ir
            if entry == "rope_interleaved" and '"weft_riscv.rvv_dense_segment_load"' not in physical:
                raise AssertionError("adjacent RoPE reads did not reach the shared dense segment operation")
            artifact = kernel.compiled.artifact.path
            emitted = subprocess.check_output([toolchain.compiler_path(), str(artifact / "physical.mlir"),
                                               "--emit=intrinsic-c"])
            if emitted != (artifact / "kernel.c").read_bytes():
                raise AssertionError("dense field replay changed the selected program")
            paths[f"{entry}:{block}"] = str(artifact)
    return {"dense_field_max_absolute_error": maximum, "dense_field_cases": cases,
            "dense_field_in_place_and_partial_views": True, "dense_field_physical_replay": True,
            "dense_field_ordered_aliases": True, "dense_field_artifacts": paths,
            "self_overlapping_outputs_rejected": rejected}


def value_program_calls(toolchain: Toolchain, cache: Path) -> dict:
    import numpy as np

    errors = {}
    for entry, options, shapes in (
        ("transformed_statistics", CompileOptions(meta={"ROWS": 4, "N": 33}), ((1, 33), (5, 33))),
        ("dynamic_statistics", CompileOptions(), ((0, 7), (2, 0), (1, 1), (5, 33), (2, 257))),
    ):
        definition = load_definition(f"{PROJECT / 'source/weft/dense/reductions.py'}:{entry}")
        maximum, paths = 0.0, set()
        with weft.jit(definition, options=options, toolchain=toolchain, cache=cache) as kernel:
            for m, n in shapes:
                x = np.linspace(-2, 3, max(1, m * n * 2), dtype=np.float32)[:m * n * 2].reshape(m, n * 2)[:, ::2]
                storage = np.full((m, 6), 1234, dtype=np.float32)
                y = storage[:, ::2]
                kernel(X=x, Y=y, shift=0.25, scale=0.75)
                values = ((x - np.float32(0.25)) * np.float32(0.75)) ** 2
                reference = np.stack((values.sum(axis=-1), values.max(axis=-1, initial=-np.inf),
                                      values.min(axis=-1, initial=np.inf)), axis=-1)
                np.testing.assert_allclose(y, reference, atol=1e-4, rtol=2e-3)
                np.testing.assert_array_equal(storage[:, 1::2], 1234)
                finite = np.isfinite(reference)
                maximum = max(maximum, float(np.abs(y[finite] - reference[finite]).max(initial=0)))
                paths.add(str(kernel.compiled.artifact.path))
        assert len(paths) == 1
        errors[entry] = maximum
    for entry in ("coupled_recurrence", "branch_recurrence", "initialized_recurrence", "reset_recurrence"):
        definition = load_definition(f"{PROJECT / 'source/weft/dense/state_update.py'}:{entry}")
        maximum, paths = 0.0, set()
        with weft.jit(definition, toolchain=toolchain, cache=cache) as kernel:
            for n in (0, 1, 33, 129):
                x = np.linspace(-1, 1, max(1, n * 2), dtype=np.float32)[:n * 2:2]
                for steps in (0, 1, 3, 4):
                    first, second = np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
                    if entry == "initialized_recurrence": first = x.copy()
                    if entry == "reset_recurrence": first.fill(-1)
                    for iteration in range(steps):
                        if entry == "coupled_recurrence":
                            previous = first
                            first = second + x
                            second = previous * 0.5 + second * 0.25 if iteration % 2 == 0 else previous - second * 0.125
                        elif entry == "branch_recurrence":
                            first = first + x if iteration % 2 == 0 else first * 0.5
                        elif entry == "initialized_recurrence":
                            first = first * 0.5 + 1.0
                        else:
                            first.fill(3)
                    y, z = np.full(n * 2, 1234, dtype=np.float32), np.full(n * 2, 1234, dtype=np.float32)
                    arguments = {"X": x, "Y": y[::2], "steps": steps}
                    if entry == "reset_recurrence": del arguments["X"]
                    if entry == "coupled_recurrence": arguments["Z"] = z[::2]
                    kernel(**arguments)
                    np.testing.assert_allclose(y[::2], first, atol=1e-4, rtol=2e-3)
                    np.testing.assert_array_equal(y[1::2], 1234)
                    maximum = max(maximum, float(np.abs(y[::2] - first).max(initial=0)))
                    if entry == "coupled_recurrence":
                        np.testing.assert_allclose(z[::2], second, atol=1e-4, rtol=2e-3)
                        np.testing.assert_array_equal(z[1::2], 1234)
                        maximum = max(maximum, float(np.abs(z[::2] - second).max(initial=0)))
                    paths.add(str(kernel.compiled.artifact.path))
        assert len(paths) == 1
        errors[entry] = maximum
    definition = load_definition(f"{PROJECT / 'source/weft/dense/state_update.py'}:parallel_carries")
    with weft.jit(definition, toolchain=toolchain, cache=cache) as kernel:
        for n in (0, 1, 33, 257):
            for steps in (0, 1, 2, 3, 7):
                storage = np.full(n * 2, 1234, dtype=np.float32)
                original = np.arange(n, dtype=np.float32) / 16 - 2
                x = storage[::2]
                x[:] = original
                first, second, third = original.copy(), original * 2, original + 3
                for iteration in range(steps):
                    if iteration % 3 == 0:
                        first, second, third = second, third, first
                    elif iteration % 3 == 1:
                        first, third = second, third + original
                    else:
                        first, second, third = third + original, first, second
                z, w, initial = (np.empty(n, dtype=np.float32) for _ in range(3))
                kernel(X=x, Y=x, Z=z, W=w, Initial=initial, steps=steps)
                for actual, expected in ((x, first), (z, second), (w, third), (initial, original)):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(storage[1::2], 1234)
    return {"value_program_max_absolute_error": errors, "dynamic_shapes_reuse_one_artifact": True,
            "parallel_copies_cycles_duplicate_sources_and_retained_snapshot": True}


def math_calls(toolchain: Toolchain, cache: Path) -> dict:
    import numpy as np
    from weft.programs import get

    rng = np.random.default_rng(416)
    bits = rng.integers(1, 0x7f800000, size=32768, dtype=np.uint32)
    positive = bits.view(np.float32)
    special = np.array([0.0, -0.0, -1.0, np.inf, -np.inf, np.nan,
                        np.nextafter(np.float32(0), np.float32(1)),
                        np.finfo(np.float32).tiny, np.finfo(np.float32).max,
                        1.0, np.nextafter(np.float32(1), np.float32(0)),
                        np.nextafter(np.float32(1), np.float32(2))], dtype=np.float32)
    values = np.concatenate((special, positive))
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.log(values.astype(np.float64)).astype(np.float32)
    maximum, maximum_ulp = 0.0, 0
    for block, lmul in ((1, 8), (32, 8), (33, 16), (129, 32)):
        with get("logarithm").jit(toolchain=toolchain, cache=cache,
                options=CompileOptions(lmul_eighths=lmul), BLOCK=block) as kernel:
            for width in (0, 1, 3, 33, values.size):
                storage = np.full((1, width * 2), 1234, dtype=np.float32)
                x = values[:width].reshape(1, width)
                y = storage[:, ::2]
                kernel(X=x, Y=y)
                reference = expected[:width].reshape(1, width)
                np.testing.assert_allclose(y, reference, atol=5e-7, rtol=5e-7, equal_nan=True)
                np.testing.assert_array_equal(storage[:, 1::2], 1234)
                finite = np.isfinite(reference)
                maximum = max(maximum, float(np.abs(y[finite] - reference[finite]).max(initial=0)))
                def ordered(value):
                    integer = np.ascontiguousarray(value).view(np.int32).astype(np.int64)
                    return np.where(integer < 0, -2147483648 - integer, integer)
                ulp = int(np.abs(ordered(y[finite]) - ordered(reference[finite])).max(initial=0))
                assert ulp <= 3, f"logarithm exceeds its 3-ULP check: {ulp}"
                maximum_ulp = max(maximum_ulp, ulp)
                inplace = x.copy()
                kernel(X=inplace, Y=inplace)
                np.testing.assert_allclose(inplace, reference, atol=5e-7, rtol=5e-7, equal_nan=True)
    exp_input = np.concatenate((np.array([-np.inf, -104.0, -100.0, -0.0, 0.0,
        88.0, 88.72283, 89.0, np.inf, np.nan], dtype=np.float32),
        rng.uniform(-104, 89, 32768).astype(np.float32))).reshape(1, -1)
    with np.errstate(over="ignore", invalid="ignore"):
        exp_reference = np.exp(exp_input.astype(np.float64)).astype(np.float32)
    exp_ulp = 0
    for block in (1, 32):
        with get("exponential").jit(toolchain=toolchain, cache=cache, BLOCK=block) as kernel:
            storage = np.full((1, 2 * exp_input.shape[1]), 1234, dtype=np.float32)
            actual = storage[:, ::2]
            kernel(X=exp_input, Y=actual)
            np.testing.assert_allclose(actual, exp_reference, atol=5e-7, rtol=5e-7, equal_nan=True)
            finite = np.isfinite(exp_reference)
            distance = int(np.abs(ordered(actual[finite]) - ordered(exp_reference[finite])).max(initial=0))
            assert distance <= 4, f"exponential exceeds its 4-ULP check: {distance}"
            exp_ulp = max(exp_ulp, distance)
            np.testing.assert_array_equal(storage[:, 1::2], 1234)
    errors = {}
    x = np.linspace(-100, 100, 513, dtype=np.float32).reshape(3, 171)
    for name in ("softplus", "silu"):
        if name == "softplus":
            reference = np.logaddexp(np.float64(0), x.astype(np.float64)).astype(np.float32)
        else:
            reference = (x.astype(np.float64) / (1 + np.exp(-x.astype(np.float64)))).astype(np.float32)
        with get(name).jit(toolchain=toolchain, cache=cache, BLOCK=33) as kernel:
            y = np.empty_like(x)
            kernel(X=x, Y=y)
            np.testing.assert_allclose(y, reference, atol=1e-4, rtol=2e-3)
            errors[name] = float(np.max(np.abs(y - reference)))
    return {"log_max_absolute_error": maximum, "log_max_ulp_error": maximum_ulp,
            "log_special_values_and_subnormals": True,
            "scalar_and_vector_exp_max_ulp_error": exp_ulp,
            "activation_max_absolute_error": errors,
            "pointwise_aliases": pointwise_alias_calls(toolchain, cache),
            "bounded_math": bounded_math_calls(toolchain, cache)}


def bounded_math_calls(toolchain: Toolchain, cache: Path) -> dict:
    import numpy as np

    rng = np.random.default_rng(426)
    random_bits = rng.integers(0, 2**32, 16384, dtype=np.uint32).view(np.float32)
    dense = np.linspace(-105, 90, 8193, dtype=np.float32)
    special = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0, -104, -103.97208,
                        -87, 87, 88.72283, np.finfo(np.float32).tiny,
                        np.nextafter(np.float32(0), np.float32(1))], dtype=np.float32)
    data = np.concatenate((special, dense, random_bits))
    errors = {}
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        negative = -np.abs(data)
        # The normalizer's add is an explicit f32 numerical boundary.
        exponential = np.exp(negative.astype(np.float64)).astype(np.float32)
        # Weft's minimumNumber chooses the numeric operand for either kind of
        # NaN. Some NumPy fmin implementations quiet a signaling NaN first,
        # returning NaN for that first call and changing a subsequent clamp.
        finite_for_exp = np.where(np.isnan(data), np.float32(87), data)
        finite_for_log = np.where(np.isnan(data), np.float32(2.0**100), data)
        references = {
            "exp_nonpositive": exponential,
            "exp_clamped": np.exp(np.clip(finite_for_exp, -87, 87).astype(np.float64)).astype(np.float32),
            "log_clamped": np.log(np.clip(finite_for_log, np.float32(2.0**-100), np.float32(2.0**100)).astype(np.float64)).astype(np.float32),
            "log_exp_normalizer": np.log((np.float32(1) + exponential).astype(np.float64)).astype(np.float32),
            "selected_exp": np.exp(np.minimum(np.maximum(data, -5), 5).astype(np.float64)).astype(np.float32),
        }
        masked = np.where((data >= -5) & (data <= 5), data, np.float32(0))
        references["masked_exp"] = np.exp(masked.astype(np.float64)).astype(np.float32)
        references["complemented_exp"] = references["masked_exp"]
    for entry, expected in references.items():
        definition = load_definition(f"{PROJECT / 'source/weft/dense/math.py'}:{entry}")
        maximum_ulp, maximum = 0, 0.0
        for block, lmul, copies in ((1, 8, 1), (32, 8, 1), (33, 16, 1), (129, 32, 1),
                                    (32, 8, 2), (33, 8, 4)):
            with weft.jit(definition, toolchain=toolchain, cache=cache,
                          options=CompileOptions(meta={"BLOCK": block}, lmul_eighths=lmul,
                                                 pointwise_issue_copies=copies)) as kernel:
                for width in (0, 1, 7, 33, data.size):
                    x = data[:width].reshape(1, width)
                    storage = np.full((1, 2 * width + 2), 71, dtype=np.float32)
                    y = storage[:, 1:-1:2]
                    kernel(X=x, Y=y)
                    reference = expected[:width].reshape(1, width)
                    np.testing.assert_allclose(y, reference, atol=5e-7, rtol=5e-7, equal_nan=True,
                                               err_msg=f"{entry}/{block}/{lmul}/{copies}/{width}")
                    np.testing.assert_array_equal(storage[:, ::2], 71)
                    np.testing.assert_array_equal(storage[:, -1], 71)
                    finite = np.isfinite(reference)
                    maximum = max(maximum, float(np.abs(y[finite] - reference[finite]).max(initial=0)))
                    if entry != "log_exp_normalizer":
                        def ordered(value):
                            integer = np.ascontiguousarray(value).view(np.int32).astype(np.int64)
                            return np.where(integer < 0, -2147483648 - integer, integer)
                        distance = int(np.abs(ordered(y[finite]) - ordered(reference[finite])).max(initial=0))
                        assert distance <= 4, f"{entry} exceeds 4 ULP: {distance}"
                        maximum_ulp = max(maximum_ulp, distance)
        errors[entry] = {"max_absolute_error": maximum, "max_ulp_error": maximum_ulp}
    return errors


def pointwise_alias_calls(toolchain: Toolchain, cache: Path) -> dict:
    """Exercise the public ordered program on actual overlapping storage."""
    import numpy as np
    from weft.programs import get

    def view(storage, shape, offset, strides):
        return np.ndarray(shape, dtype=np.float32, buffer=storage,
                          offset=offset * 4, strides=tuple(s * 4 for s in strides))

    def activation(values):
        wide = values.astype(np.float64)
        return (wide / (1 + np.exp(-wide))).astype(np.float32)

    cases = []
    for n in (0, 1, 7, 31, 32, 33, 129):
        for stride in (1, 2):
            row = stride * n + 19
            for shift in (0, 1, -1, 17, -33):
                start = 40 + max(0, -shift)
                cases.append((f"shift/{n}/{stride}/{shift}", (3, n),
                              start, (row, stride), start + shift, (row, stride)))
    # A broadcast input can overlap a writable output without having the same
    # element mapping. Equal pointers must not remove the block snapshot.
    cases.append(("broadcast-input", (2, 65), 40, (200, 0), 40, (200, 1)))
    # Both views are individually injective, but their coordinate orders differ.
    cases.append(("transposed-output", (3, 33), 40, (33, 1), 40, (1, 3)))
    cases.append(("column-major-inplace", (3, 33), 40, (1, 3), 40, (1, 3)))
    cases.append(("overlapping-envelopes-disjoint-cells", (3, 33), 40, (100, 2), 41, (100, 2)))
    cases.append(("disjoint-ranges", (3, 33), 40, (100, 2), 600, (100, 2)))
    maximum = {}
    for name in ("silu", "swiglu"):
        error = 0.0
        with get(name).jit(toolchain=toolchain, cache=cache, BLOCK=32) as kernel:
            for label, shape, start, strides, out_start, out_strides in cases:
                storage = np.linspace(-3, 3, 1600, dtype=np.float32)
                reference_storage = storage.copy()
                x = view(storage, shape, start, strides)
                y = view(storage, shape, out_start, out_strides)
                reference_x = view(reference_storage, shape, start, strides)
                reference_y = view(reference_storage, shape, out_start, out_strides)
                # The second input sometimes aliases a shifted output as well.
                # The fast arm requires every read/write relation to be safe.
                up = view(storage, shape, out_start + 3, out_strides)
                reference_up = view(reference_storage, shape, out_start + 3, out_strides)
                for row in range(shape[0]):
                    for begin in range(0, shape[1], 32):
                        end = min(shape[1], begin + 32)
                        before = reference_x[row, begin:end].copy()
                        multiplier = reference_up[row, begin:end].copy()
                        expected = activation(before)
                        if name == "swiglu": expected *= multiplier
                        reference_y[row, begin:end] = expected
                arguments = {"X": x} if name == "silu" else {"Gate": x, "Up": up}
                kernel(**arguments, Y=y)
                np.testing.assert_allclose(storage, reference_storage, atol=1e-4, rtol=2e-3,
                                           err_msg=f"{name}/{label}")
                error = max(error, float(np.abs(storage - reference_storage).max(initial=0)))
        maximum[name] = error
    return {"max_absolute_error": maximum, "ordered_block_overlaps": True,
            "inplace_padded_and_broadcast_views": True}


def predicate_calls(toolchain: Toolchain, cache: Path, *, measure=False, compare_artifacts=None, options=None) -> dict:
    import numpy as np
    from contextlib import ExitStack

    source = PROJECT / "source/weft/dense/predicates.py"
    selected = CompileOptions(meta={"BLOCK": 32}, lmul_eighths=8) if options is None else options
    previous = {} if compare_artifacts is None else {
        entry["program"]: entry for entry in json.loads(Path(compare_artifacts).read_text())["measurements"]}
    rng = np.random.default_rng(427)
    floating = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -1.0, 0.25, 1.0, 3.0], dtype=np.float32)
    maxima, measurements = {}, []

    def data(name, shape, special):
        m, n = shape
        dtype = {"unsigned_choices": np.uint32, "signed_choices": np.int32, "double_choices": np.float64,
                 "signed_byte_choices": np.int8, "unsigned_half_choices": np.uint16,
                 "signed_wide_choices": np.int64, "unsigned_wide_choices": np.uint64}.get(name, np.float32)
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            values = np.array([limits.min, limits.max, 0, 1, 7, 17], dtype=dtype)
        else:
            values = floating.astype(dtype)
        x = np.resize(values, (m, n)) if special else rng.uniform(-3, 3, (m, n)).astype(dtype)
        r = np.roll(x, 1, axis=-1).copy() if special else rng.uniform(-2, 2, (m, n)).astype(dtype)
        arguments = {"X": x}
        if name == "threshold_indices":
            indices = np.arange(m * n, dtype=np.int32).reshape(m, n) - 17
            predicate = x >= 0.25
            arguments.update(Ids=indices, threshold=0.25)
            expected = {"Scores": np.where(predicate, x, np.float32(0)),
                        "SelectedIds": np.where(predicate, indices, np.int32(-1))}
        elif name == "observed_predicate":
            predicate = x > 0
            expected = {"Predicate": predicate, "Y": np.where(predicate, x, np.float32(0))}
        elif name == "masked_affine":
            predicate = (x >= -1) & (x <= 1)
            arguments.update(Residual=r, lower=-1.0, upper=1.0, scale=0.75)
            with np.errstate(invalid="ignore"):
                expected = {"Y": np.where(predicate, x * np.float32(0.75) + r, r),
                            "Z": np.where(predicate, x, np.float32(0))}
        elif name == "logical_gate":
            arguments.update(R=r, enabled=True)
            predicate = (x > r) ^ (r > 0)
            expected = {"Y": np.where(predicate, x, r), "Z": np.where(~predicate, x, r)}
        elif name in ("loaded_gate", "loaded_choices", "observed_gate"):
            mask = (np.arange(m * n).reshape(m, n) % 3) == 0
            arguments["Mask"] = mask
            if name == "observed_gate":
                predicate = mask | (x < 0)
                expected = {"Selected": predicate, "Y": np.where(predicate, x, np.float32(0)),
                            "Z": np.where(predicate, np.float32(1), x)}
            else:
                arguments["R"] = r
                predicate = mask & (x >= 0) if name == "loaded_gate" else mask
                scalar = np.float32(1 if name == "loaded_gate" else -1)
                expected = {"Y": np.where(predicate, x, r), "Z": np.where(predicate, scalar, x)}
        else:
            threshold = np.iinfo(dtype).max // 2 if np.issubdtype(dtype, np.integer) else 0.25
            predicate = (x >= threshold) | ((r < x) if np.issubdtype(dtype, np.signedinteger) else (r > x))
            arguments.update(R=r, threshold=int(threshold) if np.issubdtype(dtype, np.integer) else threshold)
            value = 7 if np.issubdtype(dtype, np.unsignedinteger) else -7
            expected = {"Y": np.where(predicate, x, r), "Z": np.where(predicate, dtype(value), r)}
        return arguments, expected

    def verify(actual, expected, label):
        if np.issubdtype(expected.dtype, np.integer) or expected.dtype == np.bool_:
            np.testing.assert_array_equal(actual, expected, err_msg=label)
            return 0.0
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=2e-3, equal_nan=True, err_msg=label)
        if "masked_affine" not in label:
            zeros = expected == 0
            np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]), err_msg=label)
        finite = np.isfinite(expected)
        return float(np.abs(actual[finite] - expected[finite]).max(initial=0))

    names = ("masked_affine", "threshold_indices", "logical_gate", "signed_choices", "unsigned_choices", "double_choices",
             "observed_predicate", "loaded_gate", "loaded_choices", "observed_gate", "signed_byte_choices",
             "unsigned_half_choices", "signed_wide_choices", "unsigned_wide_choices")
    for name in names:
        definition = load_definition(f"{source}:{name}")
        maximum = 0.0
        with weft.jit(definition, toolchain=toolchain, cache=cache, options=selected) as kernel:
            for shape in ((0, 7), (3, 0), (1, 1), (3, 7), (3, 33), (5, 129)):
                contiguous, dense_reference = data(name, shape, True)
                contiguous.update({key: np.empty_like(value) for key, value in dense_reference.items()})
                kernel(**contiguous)
                for output, value in dense_reference.items():
                    maximum = max(maximum, verify(contiguous[output], value, f"{name}/contiguous/{output}/{shape}"))
                # Unit-stride and arbitrary-stride memory leaves are distinct
                # selected operations, even with identical predicate geometry.
                # Exercise both on the same special-value corpus before timing.
                arguments, expected = data(name, shape, True)
                owners = []
                for argument, value in tuple(arguments.items()):
                    if isinstance(value, np.ndarray):
                        storage = np.full((shape[0], shape[1] * 2 + 2), 13, dtype=value.dtype)
                        storage[:, 1:-1:2] = value
                        owners.append(storage)
                        arguments[argument] = storage[:, 1:-1:2]
                output_owners = {}
                for output, value in expected.items():
                    storage = np.full((shape[0], shape[1] * 2 + 2), 1, dtype=value.dtype)
                    output_owners[output] = storage
                    arguments[output] = storage[:, 1:-1:2]
                kernel(**arguments)
                for output, value in expected.items():
                    maximum = max(maximum, verify(arguments[output], value, f"{name}/{output}/{shape}"))
                    np.testing.assert_array_equal(output_owners[output][:, ::2], 1)
                    np.testing.assert_array_equal(output_owners[output][:, -1], 1)
                for owner in owners:
                    sentinel = np.array(13, dtype=owner.dtype)
                    np.testing.assert_array_equal(owner[:, ::2], sentinel)
                    np.testing.assert_array_equal(owner[:, -1], sentinel)
                if name == "logical_gate":
                    arguments["enabled"] = False
                    kernel(**arguments)
                    verify(arguments["Y"], arguments["R"], name + "/disabled/Y")
                    verify(arguments["Z"], arguments["X"], name + "/disabled/Z")
            if measure and name in ("masked_affine", "threshold_indices", "logical_gate"):
                arguments, expected = data(name, (128, 4096), False)
                arguments.update({key: np.empty_like(value) for key, value in expected.items()})
                call = kernel.prepare(**arguments)
                call()
                for output, value in expected.items(): verify(arguments[output], value, name + "/timed/" + output)
                samples, old_samples, comparison = [], [], None
                with ExitStack() as resources:
                    calls = [(call, samples)]
                    if compare_artifacts is not None:
                        artifact = Artifact(previous[name]["artifact"])
                        current = kernel.compiled.artifact
                        if artifact.target != current.target or artifact.canonical_ir != current.canonical_ir:
                            raise ValueError("predicate comparison requires the same target and canonical program")
                        def flags(a):
                            return [x for x in a.manifest["link_command"] if not x.endswith((".c", ".so"))]
                        if flags(artifact) != flags(current): raise ValueError("predicate comparison toolchains differ")
                        old = resources.enter_context(artifact.load(target=kernel.compiled.target, toolchain=toolchain))
                        old_arguments = {**arguments, **{key: np.empty_like(value) for key, value in expected.items()}}
                        old_call = old.prepare(**old_arguments)
                        old_call()
                        for output, value in expected.items(): verify(old_arguments[output], value, name + "/previous/" + output)
                        calls.insert(0, (old_call, old_samples))
                    eviction = np.zeros(64 * 1024 * 1024, dtype=np.uint8)
                    for repeat in range(10):
                        for prepared, destination in calls if repeat % 2 == 0 else reversed(calls):
                            eviction += np.uint8(1)
                            destination.append(prepared.time_ns() / 1e6)
                    if old_samples:
                        comparison = {"artifact": previous[name]["artifact"], "samples_ms": old_samples,
                                      "median_ms": float(np.median(old_samples)), "same_options": artifact.options == selected,
                                      "kind": "compiler" if artifact.options == selected else "compiler-and-physical-binding"}
                measurements.append({"program": name, "shape": [128, 4096], "artifact": str(kernel.compiled.artifact.path),
                                     "options": selected.to_dict(), "samples_ms": samples, "median_ms": float(np.median(samples)),
                                     "comparison": comparison, "max_abs_error": maximum,
                                     "compiler": list(kernel.compiled.artifact.manifest["link_command"]),
                                     "target": kernel.compiled.artifact.target.to_dict(),
                                     "physical_resources": dict(kernel.compiled.artifact.physical_resources),
                                     "timing_scope": "complete-operator;single-thread;64MiB-evict;alternating-artifacts",
                                     "input_generation": {"algorithm": "PCG64", "seed": 427, "scope": "predicate-corpus-order"}})
        maxima[name] = maximum
    return {"predicate_max_absolute_error": maxima, "mixed_float_integer_outputs": True,
            "integer_and_boolean_outputs": "exact", "measurements": measurements}


def comparison_calls(toolchain: Toolchain, cache: Path) -> dict:
    import numpy as np

    source = PROJECT / "source/weft/dense/comparison.py"
    for name, dtype in (("f32", np.float32), ("f64", np.float64), ("i32", np.int32), ("u32", np.uint32)):
        floating = np.issubdtype(dtype, np.floating)
        if floating:
            values = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1, -1, 3], dtype=dtype)
        else:
            limits = np.iinfo(dtype)
            values = np.array([0, 1, 3, limits.min, limits.max, limits.max - 1], dtype=dtype)
        definition = load_definition(f"{source}:compare_{name}")
        scalar_definition = load_definition(f"{source}:compare_scalar_{name}")
        options = CompileOptions(meta={"BLOCK": 32})
        with weft.jit(definition, options=options, toolchain=toolchain, cache=cache) as kernel, \
             weft.jit(scalar_definition, options=options, toolchain=toolchain, cache=cache) as scalar_kernel:
            for width in (0, 1, 33, 257):
                x_storage = np.full(2 * width, 17, dtype=dtype)
                r_storage = np.full(2 * width, 17, dtype=dtype)
                x_storage[::2] = np.resize(values, width)
                r_storage[::2] = np.resize(np.roll(values, 1), width)
                x, right = x_storage[::2], r_storage[::2]
                storage = np.full((6, 2 * width + 2), 17, dtype=dtype)
                y = storage[:, 1:-1:2]
                expected = np.array([np.where(predicate(x, right), x, right) for predicate in
                                     (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)])
                kernel(X=x, R=right, Y=y)
                np.testing.assert_array_equal(y, expected)
                if floating:
                    zero = expected == 0
                    np.testing.assert_array_equal(np.signbit(y[zero]), np.signbit(expected[zero]))
                np.testing.assert_array_equal(storage[:, ::2], 17)
                np.testing.assert_array_equal(storage[:, -1], 17)
                for pivot in ((0.0, float("nan")) if floating else (0, int(np.iinfo(dtype).max))):
                    expected = np.array([np.where(predicate(pivot, x), pivot, x) for predicate in
                                         (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)], dtype=dtype)
                    scalar_kernel(X=x, Y=y, pivot=pivot)
                    np.testing.assert_array_equal(y, expected)
                    if floating:
                        zero = expected == 0
                        np.testing.assert_array_equal(np.signbit(y[zero]), np.signbit(expected[zero]))
                    np.testing.assert_array_equal(storage[:, ::2], 17)
                    np.testing.assert_array_equal(storage[:, -1], 17)
    return {"direct_comparison_choices": "exact", "comparison_signed_zero_nan_and_integer_extremes": True}


def invocation_measurements(artifact: Path, previous_runtime: Path, backends: list[str]):
    """Paired warm API timings using identical generated code and input storage.

    previous_runtime names a saved Python package, not a compiler candidate or
    an execution fallback. Both runtimes load the same affine artifact. The
    native kernel is deliberately small so this reports host invocation work;
    these samples are not the cold-cache operator or model timing protocol.
    """
    import numpy as np
    from weft.runtime import PreparedSequence

    previous_runtime = previous_runtime.resolve()
    if not (previous_runtime / "__init__.py").is_file():
        raise ValueError("previous runtime must name a complete saved weft Python package")
    namespace = "weft_invocation_previous"
    if namespace in sys.modules:
        raise RuntimeError("a previous runtime is already loaded in this measurement process")
    specification = importlib.util.spec_from_file_location(
        namespace, previous_runtime / "__init__.py", submodule_search_locations=[str(previous_runtime)])
    module = importlib.util.module_from_spec(specification)
    sys.modules[namespace] = module
    specification.loader.exec_module(module)
    previous_api = importlib.import_module(namespace + ".runtime")
    # PyDLL caches mutable ctypes function prototypes. Isolate the two copies
    # of the same CPython buffer ABI without changing either binding algorithm.
    for name in ("weft", namespace):
        buffer_module = importlib.import_module(name + ".runtime.buffer")
        api = ctypes.PyDLL(None)
        get_buffer = api.PyObject_GetBuffer
        get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(buffer_module._PyBuffer), ctypes.c_int]
        get_buffer.restype = ctypes.c_int
        release_buffer = api.PyBuffer_Release
        release_buffer.argtypes = [ctypes.POINTER(buffer_module._PyBuffer)]
        release_buffer.restype = None
        buffer_module._get_buffer = get_buffer
        buffer_module._release_buffer = release_buffer

    current_artifact = Artifact(artifact)
    names = tuple(parameter.name for parameter in current_artifact.interface.arguments)
    if names != ("X", "Y", "scale", "bias"):
        raise ValueError("invocation measurement requires the existing dense affine artifact")
    records = []
    with current_artifact.load() as current:
        with previous_api.Artifact(artifact).load(target=current.target) as previous:
            for backend in backends:
                x = np.linspace(-3, 4, 129, dtype=np.float32)
                if backend == "torch":
                    import torch
                    from weft.torch import register
                    previous_register = importlib.import_module(namespace + ".torch").register
                    torch.set_num_threads(1)
                    x = torch.from_numpy(x).clone()
                    allocate = lambda: torch.empty_like(x)
                    old_operator = previous_register("weft_invocation_previous::affine", previous)
                    new_operator = register("weft_invocation_current::affine", current)
                elif backend == "numpy":
                    allocate = lambda: np.empty_like(x)
                else:
                    raise ValueError(f"unknown array backend: {backend}")
                y = allocate()
                old_call = previous.prepare(x, y, 1.25, bias=-0.5)
                new_call = current.prepare(x, y, 1.25, bias=-0.5)
                cases = [
                    ("prepared", old_call, new_call),
                    ("public", lambda: previous(x, y, 1.25, bias=-0.5),
                     lambda: current(x, y, 1.25, bias=-0.5)),
                    ("allocated", lambda: previous.run(x, scale=1.25, bias=-0.5),
                     lambda: current.run(x, scale=1.25, bias=-0.5)),
                ]
                if backend == "torch":
                    cases.extend([
                        ("torch-out", lambda: old_operator.out(x, 1.25, bias=-0.5, out=y),
                         lambda: new_operator.out(x, 1.25, bias=-0.5, out=y)),
                        ("torch-functional", lambda: old_operator(x, 1.25, bias=-0.5),
                         lambda: new_operator(x, 1.25, bias=-0.5)),
                    ])
                outputs = [allocate() for _ in range(8)]
                inputs = [x, *outputs[:-1]]
                old_calls = [previous.prepare(a, b, 1.01, bias=0.1) for a, b in zip(inputs, outputs)]
                new_calls = [current.prepare(a, b, 1.01, bias=0.1) for a, b in zip(inputs, outputs)]
                with PreparedSequence(*new_calls) as sequence:
                    cases.append(("sequence-8", lambda: tuple(call() for call in old_calls), sequence))
                    for entry, before, after in cases:
                        for _ in range(10):
                            before()
                            after()
                        samples = [[], []]
                        errors = [0.0, 0.0]
                        for sample in range(10):
                            # Alternate ordering to avoid assigning a monotonic
                            # clock/frequency drift entirely to one runtime.
                            for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                                function = (before, after)[index]
                                begin = time.perf_counter_ns()
                                for _ in range(100):
                                    result = function()
                                samples[index].append((time.perf_counter_ns() - begin) / 100)
                                if entry == "sequence-8":
                                    actual, expected = outputs[-1], x
                                    for _ in range(8):
                                        expected = expected * 1.01 + 0.1
                                else:
                                    actual = y if entry == "torch-out" else result
                                    expected = x * 1.25 - 0.5
                                for a, b in zip(actual.tolist(), expected.tolist()):
                                    errors[index] = max(errors[index], require_numeric(a, b))
                        old_ns, new_ns = (statistics.median(values) for values in samples)
                        record = {
                            "execution": "paired-public-invocation", "backend": backend, "entry": entry,
                            "artifact": str(current_artifact.path), "previous_runtime": str(previous_runtime),
                            "target": current.target.to_dict(), "shape": [129],
                            "source_bindings": dict(current.options.meta), "options": current.options.to_dict(),
                            "link_command": list(current_artifact.manifest["link_command"]),
                            "python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__ if backend == "torch" else None,
                            "system": list(os.uname()), "affinity": sorted(os.sched_getaffinity(0)),
                            "timing_scope": "warm-public-call-100-repetitions-per-sample",
                            "comparison": "same-native-artifact-and-input-addresses",
                            "median_us": new_ns / 1000, "previous_median_us": old_ns / 1000,
                            "speedup": old_ns / new_ns, "samples_ns": samples[1],
                            "previous_samples_ns": samples[0], "repetitions": 100, "samples": 10,
                            "max_absolute_error": errors[1], "previous_max_absolute_error": errors[0],
                            "atol": 1e-4, "rtol": 2e-3,
                        }
                        records.append(record)
                        print(json.dumps(record), flush=True)
    return records


def invocation_calls(toolchain: Toolchain, cache: Path, backends: list[str]) -> dict:
    """Exercise retained bindings and ordered SDK execution with real kernels."""
    import numpy as np
    from weft.runtime import PreparedSequence

    definition = load_definition(f"{PROJECT / 'source/weft/dense/affine.py'}:affine")
    errors = {}
    with weft.jit(definition, options=CompileOptions(meta={"B": 64}),
                  toolchain=toolchain, cache=cache) as lazy:
        for backend in backends:
            if backend == "torch":
                import torch
                x = torch.linspace(-2, 2, 129)
                y, z = torch.empty_like(x), torch.empty_like(x)
            elif backend == "numpy":
                x = np.linspace(-2, 2, 129, dtype=np.float32)
                y, z = np.empty_like(x), np.empty_like(x)
            else:
                raise ValueError(f"unknown array backend: {backend}")
            first = lazy.prepare(x, y, 1.25, bias=-0.5)
            second = lazy.prepare(y, z, 2.0, bias=3.0)
            with PreparedSequence(first, second) as sequence:
                results = sequence()
                assert results[0] is y and results[1] is z
                assert sequence.outputs == ((y,), (z,))
                expected = (x * 1.25 - 0.5) * 2 + 3
                errors[backend] = max(require_numeric(a, b) for a, b in zip(z.tolist(), expected.tolist()))
                replacement = second.rebind(scale=3.0)
                # A scalar rebind retains the immutable View ABI proof but owns
                # its converted scalar. The original prepared call is unchanged.
                assert replacement._bound.binding.views[0] is second._bound.binding.views[0]
                with sequence.replace(1, replacement) as changed:
                    changed()
                    for actual, reference in zip(z.tolist(), ((x * 1.25 - 0.5) * 3 + 3).tolist()):
                        require_numeric(actual, reference)
                sequence()
                for actual, reference in zip(z.tolist(), expected.tolist()):
                    require_numeric(actual, reference)

                changed = sequence.rebind({0: {"scale": 2.0}, 1: {"bias": -1.0}})
                assert changed._native._kernels[first.kernel] is sequence._native._kernels[first.kernel]
                with changed:
                    changed()
                    expected_change = (x * 2 - 0.5) * 2 - 1
                    for actual, reference in zip(z.tolist(), expected_change.tolist()):
                        require_numeric(actual, reference)
                    derived = changed.rebind({1: {"scale": 1.0}})
                # Closing the parent does not invalidate shared native bindings.
                with derived:
                    derived()
                    for actual, reference in zip(z.tolist(), (x * 2 - 1.5).tolist()):
                        require_numeric(actual, reference)
                sequence()
                for actual, reference in zip(z.tolist(), expected.tolist()):
                    require_numeric(actual, reference)

                if backend == "torch":
                    assert y._version >= 3 and z._version >= 3
                    # The second call fails validation. No earlier output may
                    # be modified by entering the native sequence prematurely.
                    y.fill_(-77)
                    z.resize_(130)
                    try:
                        sequence()
                    except ValueError:
                        pass
                    else:
                        raise RuntimeError("sequence executed with a stale later Tensor")
                    assert torch.all(y == -77)
                    z.resize_(129)
                    rebuilt = second.rebind(Y=z)
                    with sequence.replace(1, rebuilt) as current:
                        current()
                    x.requires_grad_(True)
                    try:
                        sequence()
                    except NotImplementedError:
                        pass
                    else:
                        raise RuntimeError("sequence bypassed raw Tensor autograd rejection")
                    with torch.no_grad():
                        with PreparedSequence(first, rebuilt) as inference:
                            inference()
                    x.requires_grad_(False)
            try:
                sequence()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("closed sequence executed")

        x = np.arange(17, dtype=np.float32)
        original = lazy.prepare_run(x, scale=2.0)
        x2 = np.arange(31, dtype=np.float32)
        rebound = original.rebind(X=x2, Y=np.empty_like(x2))
        assert rebound._bound.dimensions["N"] == 31
        np.testing.assert_allclose(rebound(), x2 * 2, rtol=2e-3, atol=1e-4)
        np.testing.assert_allclose(original(), x * 2, rtol=2e-3, atol=1e-4)
        fixed_shape = lazy.prepare(x, np.empty_like(x), N=17)
        try:
            fixed_shape.rebind(X=x2, Y=np.empty_like(x2))
        except ValueError:
            pass
        else:
            raise RuntimeError("rebind dropped an explicitly fixed shape")
        immutable = Buffer(x)
        for name, value in (("address", 1), ("shape", (300,)), ("readonly", False), ("owner", x2)):
            try:
                setattr(immutable, name, value)
            except AttributeError:
                pass
            else:
                raise RuntimeError("borrowed storage descriptor was mutable")

        compiled = lazy.materialize()
        saved = compiled.save(cache.parent / "invocation-artifact")
        loaded = saved.load(target=compiled.target, toolchain=toolchain)
        call = loaded.prepare_run(x, scale=3.0)
        with PreparedSequence(call) as sequence:
            np.testing.assert_allclose(sequence()[0], x * 3, rtol=2e-3, atol=1e-4)
            loaded.close()
            try:
                sequence()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("sequence ignored its Python kernel's close contract")
    return {"invocation_errors": errors, "ordered_native_sequence": True,
            "rebind_inferred_and_fixed_shapes": True, "immutable_storage_descriptor": True,
            "sequence_saved_artifact": str(saved.path), "sequence_close_contract": True}


def array_calls(toolchain: Toolchain, cache: Path, backends: list[str]) -> dict:
    import numpy as np

    definition = load_definition(f"{PROJECT / 'source/weft/dense/affine.py'}:affine")
    options = CompileOptions(meta={"B": 64})
    lazy = weft.jit(definition, options=options, toolchain=toolchain, cache=cache)
    if lazy.compiled is not None:
        raise RuntimeError("JIT compiled eagerly before its first call")
    errors = {}
    for backend in backends:
        if backend == "numpy":
            x = np.linspace(-3, 4, 133, dtype=np.float32)[3:132]
            y = np.empty_like(x)
        elif backend == "torch":
            import torch

            x = torch.linspace(-3, 4, 133, dtype=torch.float32)[3:132]
            y = torch.empty_like(x)
        else:
            raise ValueError(f"unknown array backend: {backend}")
        prepared = lazy.prepare(x, y, 1.25, bias=-0.5)
        prepared()
        prepared()
        expected = x * 1.25 - 0.5
        actual_values = y.tolist()
        errors[backend] = max(require_numeric(a, b) for a, b in zip(actual_values, expected.tolist()))
        allocated = lazy.run(x, scale=1.25, bias=-0.5)
        for actual, reference in zip(allocated.tolist(), expected.tolist()):
            require_numeric(actual, reference)
        if backend == "torch":
            if y._version != 2:
                raise RuntimeError("native Tensor writes did not advance its mutation version")
            x.requires_grad_(True)
            try:
                lazy(x, y)
            except NotImplementedError:
                pass
            else:
                raise RuntimeError("raw inference API accepted an undefined autograd operation")
            with torch.no_grad():
                lazy(x, y)
            x.requires_grad_(False)
            x.resize_(130)
            try:
                prepared()
            except ValueError:
                pass
            else:
                raise RuntimeError("prepared call accepted resized Tensor storage")
        wrong_dtype = np.zeros(129, dtype=np.float64)
        try:
            lazy(wrong_dtype, np.zeros(129, dtype=np.float32))
        except TypeError:
            pass
        else:
            raise RuntimeError("call accepted the wrong dense Encoding")
    # Separate output buffers permit concurrent invocation of the same library.
    def invoke(seed):
        x = np.arange(97, dtype=np.float32) + seed
        result = lazy.run(x, scale=2.0, bias=1.0)
        return max(require_numeric(a, b) for a, b in zip(result, x * 2 + 1))
    with ThreadPoolExecutor(max_workers=4) as executor:
        errors["concurrent"] = max(executor.map(invoke, range(8)))
    compiled = lazy.materialize()
    facts = compiled.artifact.target.capabilities(options=compiled.options, toolchain=toolchain)
    if facts.isa_extensions != compiled.metadata["isa_extensions"]:
        raise RuntimeError("compiled ISA contract disagrees with the public target query")
    catalog = extension_catalog(toolchain=toolchain)
    if not set(compiled.metadata["used_extensions"]).issubset(catalog.names):
        raise RuntimeError("compiled artifact uses an extension absent from the public catalog")
    saved = compiled.save(cache.parent / "saved-affine")
    manifest_path = saved.path / "manifest.json"
    original_manifest = manifest_path.read_text()
    try:
        for mutation in ("missing", "version"):
            manifest = json.loads(original_manifest)
            requirements = manifest["kernel"]["isa_extensions"]
            name = next(iter(requirements))
            if mutation == "missing":
                del requirements[name]
            else:
                requirements[name][0] += 1
            manifest_path.write_text(json.dumps(manifest))
            try:
                rejected = Artifact(saved.path).load(target=compiled.target, toolchain=toolchain)
            except CompilationError as error:
                if "standard ISA contract disagrees" not in str(error):
                    raise
            else:
                rejected.close()
                raise RuntimeError(f"native load accepted a {mutation} ISA requirement")
    finally:
        manifest_path.write_text(original_manifest)
    # Reuse native hardware facts; the compiler only validates target facts.
    # Loading does not rerun physicalization or invoke the system C compiler.
    loaded = saved.load(target=compiled.target, toolchain=toolchain)
    x = np.arange(17, dtype=np.float32)
    output = loaded.run(x)
    for actual, reference in zip(output, x):
        require_numeric(actual, reference)
    pinned = loaded.prepare(x, np.empty_like(x))
    loaded.close()
    try:
        pinned()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("prepared call executed after its library was closed")
    if not saved.library_path.is_file():
        raise RuntimeError("closing a callable deleted its persistent artifact")
    different = lazy.specialize(B=32).materialize()
    if different.artifact.path == compiled.artifact.path:
        raise RuntimeError("distinct source bindings reused an artifact")
    for actual, reference in zip(different.run(x), x):
        require_numeric(actual, reference)
    different.close()
    return {"array_errors": errors, "artifact": str(compiled.artifact.path),
            "saved": str(saved.path), "lazy": True, "prepare": True, "allocation": True,
            "compiler_capability_query": True, "registered_extensions": list(catalog.names),
            "isa_metadata_rejected_before_load": ["missing", "version"],
            "concurrent_calls": True, "lifecycle": True, "specialization": True}


def process_calls(toolchain: Toolchain, directory: Path, compiler_args: list[str]) -> dict:
    cache = directory / "process-cache"
    command = [sys.executable, str(Path(__file__).resolve()), *compiler_args,
               "--cache-worker", str(cache)]
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = []
    for worker in workers:
        stdout, stderr = worker.communicate()
        if worker.returncode:
            raise RuntimeError(f"cache worker failed: {stdout}\n{stderr}")
        results.append(json.loads(stdout))
    if len({result["artifact"] for result in results}) != 1 or sum(result["cache_hit"] for result in results) != 1:
        raise RuntimeError(f"simultaneous cold processes did not share one completed build: {results}")
    restarted = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(restarted.stdout)
    if not result["cache_hit"] or result["artifact"] != results[0]["artifact"]:
        raise RuntimeError("a new process did not reuse its persistent artifact")
    return {"parallel_build": True, "process_restart": True}


def require_numeric(actual: float, expected: float) -> float:
    error = abs(actual - expected)
    if not math.isfinite(actual) or not math.isfinite(expected) or error > 1e-4 + 2e-3 * abs(expected):
        raise RuntimeError(f"numeric mismatch: actual={actual}, expected={expected}")
    return float(error)


def memory_calls(toolchain: Toolchain, cache: Path) -> dict:
    import numpy as np

    def compile_case(name):
        source = PROJECT / f"examples/repro/weft/{name}.py"
        return weft.jit(load_definition(f"{source}:{name}"), toolchain=toolchain, cache=cache)

    with compile_case("scalar_read_order") as scalar:
        original = np.arange(17, dtype=np.int32) - 9
        state, output, replacement = original.copy(), np.empty_like(original), original + 101
        scalar(state, output, replacement)
        np.testing.assert_array_equal(output, original)
        np.testing.assert_array_equal(state, replacement)

    with compile_case("encoded_read_order") as encoded:
        state = np.array([-17], dtype=np.int32)
        output = np.empty(1, dtype=np.int32)
        replacement = np.array([91], dtype=np.int32)
        # The encoded argument and the dense write deliberately share bytes.
        encoded(Buffer(state.view(np.uint8), shape=(1,), encoding="Word"),
                state, output, replacement)
        np.testing.assert_array_equal(output, [-17])
        np.testing.assert_array_equal(state, replacement)

    with compile_case("lookup_read_order") as lookup:
        table = np.array([-3, 11, 19, 23], dtype=np.int32)
        before, after = np.empty(1, dtype=np.int32), np.empty(1, dtype=np.int32)
        lookup(table, table[:1], before, after)
        np.testing.assert_array_equal(before, [-3])
        np.testing.assert_array_equal(after, [7])

    with compile_case("dense_table_read_order") as snapshot:
        original = np.array([13, 29, 31, 43, 47], dtype=np.uint32)
        table, output = original.copy(), np.empty_like(original)
        snapshot(table, np.array([101], dtype=np.uint32), output)
        np.testing.assert_array_equal(output, original)
        np.testing.assert_array_equal(table, [101, 29, 31, 43, 47])

    source = PROJECT / "source/weft/dense/state_update.py"
    definition = load_definition(f"{source}:recurrent_accumulate")
    maximum = 0.0
    state_bindings = ((32, 8), (33, 8), (64, 4), (128, 4), (129, 4), (256, 4))
    for block, lmul in state_bindings:
        with weft.jit(definition, options=CompileOptions(meta={"BLOCK": block}, lmul_eighths=lmul),
                      toolchain=toolchain, cache=cache) as recurrence:
            for columns in (0, 1, 2 * block + 1):
                for steps in (0, 1, 5):
                    storage = np.arange(2 * columns * 2, dtype=np.float32).reshape(2, columns * 2)
                    x = storage[:, ::2]
                    original = x.copy()
                    expected = np.zeros_like(original)
                    for iteration in range(steps):
                        expected = expected + original if iteration % 2 == 0 else expected * np.float32(0.5)
                    output = np.empty_like(original)
                    recurrence(x, output, steps)
                    np.testing.assert_allclose(output, expected, atol=1e-4, rtol=2e-3)
                    recurrence(x, x, steps)
                    np.testing.assert_allclose(x, expected, atol=1e-4, rtol=2e-3)
                    if output.size:
                        maximum = max(maximum, float(np.abs(output - expected).max()),
                                      float(np.abs(x - expected).max()))
                    np.testing.assert_array_equal(storage[:, 1::2],
                        np.arange(2 * columns * 2, dtype=np.float32).reshape(2, columns * 2)[:, 1::2])
    return {"read_snapshots": "exact", "while_branch_state_max_absolute_error": maximum,
            "state_bindings": state_bindings, "strided_inplace": True, "empty_and_zero_step": True}


def encoded_calls(toolchain: Toolchain, library: Path, *, options: CompileOptions | None = None) -> dict:
    base = ctypes.CDLL(str(library / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
    cpu = ctypes.CDLL(str(library / "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)
    errors = {}
    # Public GGML record sizes from ggml-common.h; the same packed bytes feed
    # both callees. Multiple distinct records exercise all time/replica axes.
    for weight, activation, block, wbytes, xbytes in (
        ("q4_0", "q8_0", 32, 18, 34),
        ("q5_0", "q8_0", 32, 22, 34),
        ("q4_K", "q8_K", 256, 144, 292),
        ("q6_K", "q8_K", 256, 210, 292),
    ):
        name = f"{weight}_{activation}"
        source = PROJECT / f"examples/kernels/vec_dot/{name.lower()}.py"
        definition = load_definition(f"{source}:quantized_vec_dot_{name.lower()}")
        quantizers = [getattr(base, f"quantize_row_{kind}_ref") for kind in (weight, activation)]
        for quantize in quantizers:
            quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            quantize.restype = None
        reference = getattr(cpu, f"ggml_vec_dot_{name}_generic")
        reference.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                             ctypes.c_size_t, ctypes.c_int]
        reference.restype = None
        maximum = 0.0
        with weft.jit(definition, toolchain=toolchain, options=options) as encoded:
            for records in (1, 3):
                k = block * records
                weights = Buffer(array.array("f", [((i * 19 + 7) % 127 - 63) / 13.0 for i in range(k)]))
                activations = Buffer(array.array("f", [((i * 37) % 251 - 125) / 17.0 for i in range(k)]))
                w = Buffer(bytearray(wbytes * records), shape=(k,), encoding=weight.upper())
                x = Buffer(bytearray(xbytes * records), shape=(k,), encoding=activation.upper())
                quantizers[0](weights.address, w.address, k)
                quantizers[1](activations.address, x.address, k)
                expected = ctypes.c_float()
                reference(k, ctypes.byref(expected), 0, w.address, 0, x.address, 0, 1)
                output = array.array("f", [0.0])
                encoded(w, x, output)
                maximum = max(maximum, require_numeric(output[0], expected.value))
        errors[name] = maximum
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real dense and encoded kernels through native Weft JIT")
    parser.add_argument("--compiler")
    parser.add_argument("--cc", default="clang")
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument("--ldflag", action="append", default=[])
    parser.add_argument("--ggml-lib-dir", type=Path)
    parser.add_argument("--arrays", nargs="+", choices=("numpy", "torch"), default=["numpy"])
    parser.add_argument("--cache-worker", type=Path)
    parser.add_argument("--invocation-artifact", type=Path,
                        help="measure warm public calls to an existing dense affine artifact")
    parser.add_argument("--previous-runtime", type=Path,
                        help="saved weft package used only for paired invocation timing")
    args = parser.parse_args()
    if (args.invocation_artifact is None) != (args.previous_runtime is None):
        parser.error("invocation measurement requires both --invocation-artifact and --previous-runtime")
    if args.invocation_artifact is not None:
        invocation_measurements(args.invocation_artifact, args.previous_runtime, args.arrays)
        return
    toolchain = Toolchain(args.compiler, (args.cc,), tuple(args.cflag), tuple(args.ldflag))
    if args.cache_worker is not None:
        import numpy as np

        definition = load_definition(f"{PROJECT / 'source/weft/dense/affine.py'}:affine")
        compiled = weft.compile(definition, options=CompileOptions(meta={"B": 64}),
                                toolchain=toolchain, cache=args.cache_worker)
        x = np.arange(129, dtype=np.float32)
        for actual, expected in zip(compiled.run(x, scale=2.0), x * 2.0):
            require_numeric(actual, expected)
        print(json.dumps({"artifact": str(compiled.artifact.path), "cache_hit": compiled.cache_hit}))
        return
    compiler_args = ["--cc", args.cc]
    if args.compiler:
        compiler_args += ["--compiler", args.compiler]
    compiler_args.extend(f"--cflag={flag}" for flag in args.cflag)
    compiler_args.extend(f"--ldflag={flag}" for flag in args.ldflag)
    with tempfile.TemporaryDirectory(prefix="weft-native-product-") as temporary:
        directory = Path(temporary)
        product = array_calls(toolchain, directory / "cache", args.arrays)
        product.update(invocation_calls(toolchain, directory / "cache", args.arrays))
        product.update(memory_calls(toolchain, directory / "cache"))
        product.update(reduction_calls(toolchain, directory / "cache"))
        product.update(contraction_calls(toolchain, directory / "cache"))
        product.update(dense_field_calls(toolchain, directory / "cache"))
        product.update(value_program_calls(toolchain, directory / "cache"))
        product.update(math_calls(toolchain, directory / "cache"))
        product.update(comparison_calls(toolchain, directory / "cache"))
        product.update(predicate_calls(toolchain, directory / "cache"))
        product.update(process_calls(toolchain, directory, compiler_args))
        # A fresh installed process loads the saved ABI without the DSL module
        # or a system C compiler. Native capability discovery remains required.
        environment = {**os.environ, "CC": "/unavailable-C-compiler"}
        environment.pop("PYTHONPATH", None)
        if args.compiler:
            environment["WEFT_COMPILER"] = args.compiler
        code = ("import sys,numpy as np,weft; k=weft.load(sys.argv[1]); "
                "x=np.arange(17,dtype=np.float32); y=k.run(x); "
                "np.testing.assert_allclose(y,x,rtol=2e-3,atol=1e-4); k.close()")
        subprocess.run([sys.executable, "-c", code, product["saved"]],
                       cwd=directory, env=environment, check=True)
        product["load_without_source_or_cc"] = True
        print(json.dumps({"execution": "native-product", **product}, sort_keys=True))

    m, n, k = 16, 32, 256
    lhs = array.array("f", [1.0 / k]) * (m * k)
    rhs = array.array("f", [1.0]) * (n * k)
    output = array.array("f", [0.0]) * (m * n)
    views = (Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(n, k)),
             Buffer(output, shape=(m, n)))
    options = CompileOptions(meta={"MC": 64, "NC": 16, "MR": 2, "NR": 2}, lmul_eighths=16)
    dense = weft.jit(gemm_f32, options=options, toolchain=toolchain)
    dense(*views)
    dense(*views)
    dense_error = max(require_numeric(value, 1.0) for value in output)
    if weft.compile(gemm_f32, options=options, toolchain=toolchain) is not dense.compiled:
        raise RuntimeError("same-binding JIT invocation did not reuse its artifact")
    alternative = weft.compile(gemm_f32, options=CompileOptions(
        meta=options.meta, lmul_eighths=16, unroll=2), toolchain=toolchain)
    if alternative is dense.compiled:
        raise RuntimeError("different compile bindings reused the same artifact")
    alternative(*views)
    dense_error = max(dense_error, *(require_numeric(value, 1.0) for value in output))

    if args.ggml_lib_dir is None:
        print(json.dumps({"execution": "native-dense", "march": dense.compiled.target.march,
                          "vlen_bits": dense.compiled.target.vlen_bits, "dense_max_absolute_error": dense_error}))
        dense.close()
        alternative.close()
        return
    encoded_errors = encoded_calls(toolchain, args.ggml_lib_dir)
    print(json.dumps({
        "execution": "native-jit", "march": dense.compiled.target.march,
        "vlen_bits": dense.compiled.target.vlen_bits, "shape": [m, n, k],
        "dense_numeric": "within-tolerance", "dense_max_absolute_error": dense_error,
        "encoded_numeric": "within-tolerance", "encoded_max_absolute_error": encoded_errors,
        "same_binding_reused": True, "different_binding_specialized": True,
    }, sort_keys=True))
    dense.compiled.close()
    alternative.close()


if __name__ == "__main__":
    main()
