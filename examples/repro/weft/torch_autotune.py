from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import weft
from weft.runtime import (Artifact, Buffer, BuiltCandidates, CandidateRejected, CompilationError, Config,
                          ConfigSpace, CompileOptions, NativeTarget, NumericalCheck, Toolchain,
                          TuningError, TuningPolicy, TuningResult, TuningWorkload,
                          expand_configurations, load_configurations, save_configurations)
from weft.torch import register

from source.weft.dense.affine_strided import affine_strided
from source.weft.dense.configs import AFFINE_CONFIGS, STATE_CONFIGS
from source.weft.dense.state_update import state_scale, state_update
from source.weft.dense.torch_affine import Affine, register_affine


def check_affine(original, trial):
    np.testing.assert_allclose(trial["Y"], original["X"] * original["scale"] + original["bias"],
                               rtol=2e-3, atol=1e-4)


def check_state(original, trial):
    torch.testing.assert_close(trial["State"], original["State"] * original["decay"] + original["Delta"],
                               rtol=2e-3, atol=1e-4)


def aligned_array(shape):
    size = int(np.prod(shape)) * 4
    backing = bytearray(size + 256)
    offset = (-Buffer(backing).address) % 256
    return np.ndarray(shape, dtype=np.float32, buffer=backing, offset=offset)


def tuning_worker(toolchain, cache):
    x, y = aligned_array((3, 129)), aligned_array((3, 129))
    x[:] = np.arange(x.size, dtype=np.float32).reshape(x.shape) / 100
    y.fill(-77)
    with weft.autotune(affine_strided, AFFINE_CONFIGS, key=("scale", "bias"),
                       policy=TuningPolicy(deadline_s=600, samples=3, repetitions=3),
                       validator=check_affine, validation_key="affine-f32-rtol2e-3-atol1e-4",
                       toolchain=toolchain, cache=cache) as tuned:
        selected = tuned.tune(x, y, 1.25, bias=-0.5)
        np.testing.assert_array_equal(y, np.full_like(y, -77))
        tuned(x, y, 1.25, bias=-0.5)
        check_affine({"X": x, "scale": 1.25, "bias": -0.5}, {"Y": y})
        fixed = tuned.fixed()
        np.testing.assert_allclose(fixed.run(x, scale=1.25, bias=-0.5), y, rtol=2e-3, atol=1e-4)
        selected.save(cache.parent / "selected.json")
        assert TuningResult.load(cache.parent / "selected.json").winner == selected.winner
        return {"cache_hit": selected.cache_hit, "record": str(selected.record_path),
                "winner": selected.winner.name, "validated": all(item.validated for item in selected.candidates
                                                                  if item.status == "measured")}


def compiler_contract_calls(toolchain, cache, directory):
    """Real compiler resource rejection, artifact inspection and refresh failure."""
    x, y = aligned_array((3, 129)), aligned_array((3, 129))
    x[:] = np.arange(x.size, dtype=np.float32).reshape(x.shape) / 100
    y.fill(-77)
    configs = (Config(name="insufficient-private-storage", meta={"B": 1024}, private_stack_bytes=1),
               Config(name="bounded-affine", meta={"B": 64}))
    with weft.autotune(affine_strided, configs, key=("scale", "bias"),
                       policy=TuningPolicy(deadline_s=600, samples=3, repetitions=3),
                       validator=check_affine, validation_key="affine-f32-rtol2e-3-atol1e-4",
                       toolchain=toolchain, cache=cache) as tuned:
        selected = tuned.tune(x, y, 1.25, bias=-0.5)
        rejected = [item for item in selected.candidates if item.status == "compile-rejected"]
        assert len(rejected) == 1 and rejected[0].config == configs[0]
        failure = rejected[0].failure
        assert failure.category == "resource-limit"
        assert failure.pipeline[-1].status == "failed"
        assert failure.pass_name == failure.pipeline[-1].pass_name
        assert failure.diagnostics and any(item.severity == "error" for item in failure.diagnostics)
        assert selected.winner == configs[1]
        np.testing.assert_array_equal(y, np.full_like(y, -77))
        selected.save(directory / "resource-selection.json")
        restored = TuningResult.load(directory / "resource-selection.json")
        assert restored.candidates == selected.candidates
        rejected_index = selected.candidates.index(rejected[0])
        command = [sys.executable, "-m", "weft", "selection", str(directory / "resource-selection.json")]
        summary = subprocess.run(command, check=True, capture_output=True, text=True)
        assert json.loads(summary.stdout)["winner"] == selected.winner.to_dict()
        readable = subprocess.run([*command, "--format", "text", "--candidate", str(rejected_index)],
                                  check=True, capture_output=True, text=True)
        assert "compile-rejected" in readable.stdout and failure.pass_name in readable.stdout
        assert failure.diagnostics[0].message in readable.stdout
        prefix = subprocess.run([*command, "--format", "mlir", "--candidate", str(rejected_index)],
                                check=True, capture_output=True, text=True)
        assert prefix.stdout.strip() == failure.as_pass_pipeline()
        tuned(x, y, 1.25, bias=-0.5)
        check_affine({"X": x, "scale": 1.25, "bias": -0.5}, {"Y": y})

        artifact, = tuned.artifacts
        pipeline = artifact.compilation
        assert pipeline.entry == "canonical" and pipeline.pipeline
        assert all(step.status == "completed" for step in pipeline.pipeline)
        assert pipeline.matching(stage=pipeline.stages[0])
        assert pipeline.matching(pass_name="weft-convert-to-riscv")
        assert all(step.invocation for step in pipeline.pipeline)
        assert pipeline.as_pass_pipeline().startswith("builtin.module(")
        saved = artifact.save(directory / "compiler-artifact")
        assert Artifact(saved.path).compilation == pipeline
        with saved.load(toolchain=toolchain) as loaded:
            assert loaded.compilation == pipeline
            loaded(x, y, 1.25, bias=-0.5)
            check_affine({"X": x, "scale": 1.25, "bias": -0.5}, {"Y": y})

        # The repro owns this temporary cache. Simulate loss of a required
        # artifact file while the existing loaded winner remains alive.
        # Refresh must propagate the infrastructure failure and preserve the
        # prior durable selection; it cannot call this a rejected candidate.
        original_record = selected.record_path.read_bytes()
        held_library = directory / "held-winner.so"
        artifact.library_path.rename(held_library)
        try:
            try:
                tuned.tune(x, y, 1.25, bias=-0.5, refresh=True)
            except CandidateRejected as error:
                raise RuntimeError("missing artifact was misclassified as a compiler candidate") from error
            except CompilationError as error:
                assert error.failure is None
            else:
                raise RuntimeError("refresh accepted an incomplete cached artifact")
        finally:
            held_library.rename(artifact.library_path)
        assert selected.record_path.read_bytes() == original_record
        assert tuned.last_result is selected
        assert tuned.tune(x, y, 1.25, bias=-0.5) is selected
        tuned(x, y, 1.25, bias=-0.5)
        check_affine({"X": x, "scale": 1.25, "bias": -0.5}, {"Y": y})
        return {"typed_resource_rejection": True, "saved_compiler_failure": True,
                "selection_cli_formats": ["json", "text", "mlir"],
                "compiler_pass_count": len(pipeline.pipeline), "artifact_pipeline_reload": True,
                "fatal_refresh_preserved_selection": True}


def scalar_key_calls(toolchain, cache):
    """Finite scalar keys keep their persistent meaning during warm dispatch."""
    x = aligned_array((1, 17))
    x[:] = np.arange(17, dtype=np.float32)
    y = aligned_array(x.shape)
    observed = []
    with weft.autotune(affine_strided, [AFFINE_CONFIGS[0]], key=("scale",),
                       validator=check_affine, validation_key="affine-f32-rtol2e-3-atol1e-4",
                       toolchain=toolchain, cache=cache) as tuned:
        for scale in (1, 1.0, 0.0, -0.0):
            selected = tuned.tune(x, y, scale, bias=-0.5)
            observed.append(selected)
            tuned(x, y, scale, bias=-0.5)
            np.testing.assert_allclose(y, x * scale - 0.5, rtol=2e-3, atol=1e-4)
            assert tuned.last_result is selected
        if len(tuned.selections) != 4 or len({item.record_path for item in observed}) != 4:
            raise RuntimeError("warm workload keys merged distinct persistent scalar identities")
        for scale, selected in zip((1, 1.0, 0.0, -0.0), observed):
            assert tuned.tune(x, y, scale, bias=-0.5) is selected
        expected = tuple(item.record_path for item in observed)
    # Reopening uses the durable records under the same key semantics.
    with weft.autotune(affine_strided, [AFFINE_CONFIGS[0]], key=("scale",),
                       validator=check_affine, validation_key="affine-f32-rtol2e-3-atol1e-4",
                       toolchain=toolchain, cache=cache) as reopened:
        for scale, path in zip((1, 1.0, 0.0, -0.0), expected):
            selected = reopened.tune(x, y, scale, bias=-0.5)
            assert selected.cache_hit and selected.record_path == path
            reopened(x, y, scale, bias=-0.5)
            np.testing.assert_allclose(y, x * scale - 0.5, rtol=2e-3, atol=1e-4)
    return {"scalar_key_identity": ["integer", "real", "positive-zero", "negative-zero"],
            "warm_and_durable_selection_agree": True}


def state_calls(toolchain, cache):
    storage = torch.linspace(-2, 2, 6 * 260).reshape(6, 260)
    state = storage[1::2, 1:260:2]
    before = storage.clone()
    version = state._version
    with weft.autotune(state_update, STATE_CONFIGS, key=("decay",),
                       policy=TuningPolicy(deadline_s=600, samples=3, repetitions=3),
                       validator=check_state, validation_key="state-f32-rtol2e-3-atol1e-4",
                       toolchain=toolchain, cache=cache) as tuned:
        # Exact aliasing must survive every private warmup, validation and trial.
        result = tuned.tune(state, state, 0.5)
        torch.testing.assert_close(storage, before, rtol=0, atol=0)
        assert state._version == version
        prepared = tuned.prepare(state, state, 0.5)
        prepared()
        expected = before.clone()
        expected[1::2, 1:260:2] *= 1.5
        torch.testing.assert_close(storage, expected, rtol=2e-3, atol=1e-4)
        assert state._version > version
        prepared.rebind(decay=0.0)()
        torch.testing.assert_close(storage, expected, rtol=2e-3, atol=1e-4)
        assert result.winner in STATE_CONFIGS

    def reject(original, trial):
        raise AssertionError("deliberate rejection of the numerical candidate")

    rejecting = weft.autotune(state_update, STATE_CONFIGS, validator=reject,
                              validation_key="reject-state-candidate", toolchain=toolchain, cache=cache)
    unchanged, version = state.clone(), state._version
    try:
        rejecting.tune(state, state)
    except TuningError as error:
        assert any(item.status == "validation-rejected" for item in error.candidates)
        assert all(item.status in ("validation-rejected", "compile-rejected") for item in error.candidates)
    else:
        raise RuntimeError("rejected numerical candidates produced a winner")
    torch.testing.assert_close(state, unchanged, rtol=0, atol=0)
    assert state._version == version
    rejecting.close()
    return {"mutable_aliases": True, "trial_state_isolation": True, "rejected_state_unchanged": True}


def finite_candidates_calls(toolchain, cache, directory):
    """Compile/deploy finite choices, then tune with no available C compiler."""
    from weft.runtime.configuration import resolve_configurations
    from weft.runtime.compiler import KernelSource

    source = KernelSource.capture(affine_strided)
    domain = ConfigSpace(source={"B": (32, 64)}, physical={"lmul_eighths": (8, 16)}, name="affine")
    assert domain.size == 4
    configs, interface = resolve_configurations(source, domain, max_candidates=4)
    assert len(configs) == 4 and interface.outputs[0].name == "Y"
    physical = configs[1].changes_from(configs[0])
    author = configs[2].changes_from(configs[0])
    assert not physical["source"] and set(physical["physical"]) == {"lmul_eighths"}
    assert set(author["source"]) == {"B"} and not author["physical"]
    declarations = directory / "domains.json"
    declarations.write_text(json.dumps([domain.to_dict()]))
    assert load_configurations(declarations, max_candidates=4) == configs
    save_configurations(directory / "complete-configs.json", configs, max_candidates=4)
    assert load_configurations(directory / "complete-configs.json", max_candidates=4) == configs
    for operation in (lambda: expand_configurations(domain, max_candidates=3),
                      lambda: expand_configurations((domain, configs[0])),
                      lambda: ConfigSpace(source={"B": (32, 32)}),
                      lambda: ConfigSpace(physical={"lmul_eighths": (3,)}),
                      lambda: domain.specialize(B=128)):
        try:
            operation()
        except (TypeError, ValueError):
            pass
        else:
            raise RuntimeError("a malformed finite domain was accepted")

    native = NativeTarget.discover(toolchain)
    rejected = Config(name="private-capacity", meta={"B": 1024}, private_stack_bytes=1)
    complete = (*configs[:2], rejected, *configs[2:])
    built = weft.build_candidates(affine_strided, complete, target=native.build_target,
                                  toolchain=toolchain, native_library=False, cache=cache)
    assert len(built.artifacts) == 4 and len(built.rejections) == 1 and not built.native
    assert built.entries[2].failure.category == "resource-limit"
    saved_source = built.save(directory / "candidate-source")
    assert saved_source.configs == complete
    try:
        saved_source.autotune()
    except CompilationError:
        pass
    else:
        raise RuntimeError("source-only choices implicitly invoked a C compiler")
    saved = saved_source.link(toolchain=toolchain, cache=cache).save(directory / "candidate-native")
    assert saved.native
    assert saved.entries[2].failure == built.entries[2].failure
    assert all(a.source == b.source for a, b in zip(saved.artifacts, built.artifacts))
    assert all(a.riscv_ir == b.riscv_ir for a, b in zip(saved.artifacts, built.artifacts))
    unavailable_cc = Toolchain(toolchain.compiler, ("/no/system-C-compiler",))

    storage = aligned_array((4, 259))
    storage[:] = np.arange(storage.size, dtype=np.float32).reshape(storage.shape) / 300
    original = storage.copy()
    x = storage[:, 1:258:2]
    # Partial overlap and different offsets require one alias-preserving private
    # storage group shared by every candidate, not four separately copied arrays.
    y = storage[:, 2:259:2]
    arguments = {"X": x, "Y": y, "scale": 1.25, "bias": -0.5}
    pointers = []

    def validate(original_arguments, trial):
        check_affine(original_arguments, trial)
        pointers.append(int(trial["Y"].ctypes.data))
        assert trial["Y"].ctypes.data - trial["X"].ctypes.data == 4
        assert trial["Y"].strides == y.strides and trial["X"].strides == x.strides

    policy = TuningPolicy(deadline_s=600, warmup=1, samples=3, repetitions=2,
                          max_state_bytes=2 * (storage.nbytes + 512))
    with saved.autotune(key=("scale", "bias"), policy=policy, validator=validate,
                         validation_key="affine-strided-f32-atol1e-4-rtol2e-3",
                         toolchain=unavailable_cc, cache=cache) as tuned:
        result = tuned.tune(**arguments)
        np.testing.assert_array_equal(storage, original)
        assert len(pointers) == 4 and len(set(pointers)) == 1
        assert tuple(item.config for item in result.candidates) == complete
        assert result.candidates[2].status == "compile-rejected"
        assert result.compare(configs[0], configs[1])["kind"] == "physical-binding"
        assert result.compare(configs[0], configs[2])["kind"] == "source-binding"
        assert result.compare(rejected)["speedup"] is None
        assert len(result.source_groups) == 3
        result.save(directory / "finite-selection.json")
        restored = TuningResult.load(directory / "finite-selection.json")
        assert restored.candidates == result.candidates
        rows = list(csv.DictReader(io.StringIO(restored.to_csv())))
        assert len(rows) == 5 and rows[2]["status"] == "compile-rejected"
        assert rows[2]["reference_over_candidate"] == ""
        assert json.loads(rows[0]["source_bindings"]) == dict(configs[0].source)
        assert json.loads(rows[1]["changed_physical_parameters"])["lmul_eighths"]["after"] == 16
        broken = restored.to_dict()
        broken["candidates"] = broken["candidates"][::-1]
        try:
            TuningResult.from_dict(broken)
        except ValueError:
            pass
        else:
            raise RuntimeError("selection accepted reordered declared candidates")
        broken = restored.to_dict()
        index = next(i for i, item in enumerate(restored.candidates) if item.status == "measured")
        broken["candidates"][index]["samples_ns"] = broken["candidates"][index]["samples_ns"][:1]
        try:
            TuningResult.from_dict(broken)
        except ValueError:
            pass
        else:
            raise RuntimeError("selection accepted incomplete timings as a measured candidate")
        artifact = saved.fixed(restored)
        assert artifact.options == result.winner.options
        with tuned.fixed() as fixed:
            fixed(**arguments)
        np.testing.assert_allclose(y, original[:, 1:258:2] * 1.25 - 0.5, rtol=2e-3, atol=1e-4)
        np.testing.assert_array_equal(storage[:, 1:258:2], original[:, 1:258:2])

    # Reopening the collection preserves the exact record. A cache hit must not
    # re-run the numerical validator or require a C compilation toolchain.
    storage[:] = original
    pointers.clear()
    reloaded = BuiltCandidates.load(directory / "candidate-native")
    with reloaded.autotune(key=("scale", "bias"), policy=policy, validator=validate,
                           validation_key="affine-strided-f32-atol1e-4-rtol2e-3",
                           toolchain=unavailable_cc, cache=cache) as tuned:
        again = tuned.tune(**arguments)
        assert again.cache_hit and again.winner == result.winner
        assert not pointers
        np.testing.assert_array_equal(storage, original)

    command = [sys.executable, "-m", "weft"]
    summary = subprocess.run([*command, "candidates-inspect", str(directory / "candidate-native"),
                               "--format", "json"], check=True, capture_output=True, text=True)
    assert len(json.loads(summary.stdout)["candidates"]) == 5
    prefix = subprocess.run([*command, "candidates-inspect", str(directory / "candidate-native"),
                              "--format", "mlir", "--candidate", "2"],
                             check=True, capture_output=True, text=True)
    assert prefix.stdout.strip() == built.entries[2].failure.as_pass_pipeline()
    rendered = subprocess.run([*command, "configs", str(declarations), "--kernel",
                                "source.weft.dense.affine_strided:affine_strided"],
                               check=True, capture_output=True, text=True)
    assert "Source:" in rendered.stdout and "Physical:" in rendered.stdout
    return {"finite_source_physical_domains": True, "source_only_explicit_link": True,
            "shared_trial_snapshot": True, "prebuilt_no_c_compiler": True,
            "stable_rejection_order": True, "separated_binding_comparisons": True,
            "prebuilt_persistent_selection": True, "finite_candidate_cli": True}


def workload_calls(toolchain, cache, directory):
    """The installed tune command consumes arrays and independent references."""
    source = BuiltCandidates.load(directory / "candidate-native")
    x = np.arange(3 * 65, dtype=np.float32).reshape(3, 65) / 100
    y = np.full_like(x, -91)
    expected = x * 1.25 - 0.5
    check = NumericalCheck(expected, rtol=2e-3, atol=1e-4)
    workload = TuningWorkload({"X": x, "Y": y, "scale": 1.25, "bias": -0.5},
                              checks={"Y": check}, validation_key="archive-affine-f32")
    changed = TuningWorkload(workload.arguments, checks={"Y": NumericalCheck(expected, rtol=1e-3, atol=1e-4)},
                             validation_key="archive-affine-f32")
    assert changed.contract_key() != workload.contract_key()
    workload.check_interface(source.call_signature)
    inputs, references = directory / "inputs.npz", directory / "expected.npz"
    np.savez(inputs, X=x, Y=y)
    np.savez(references, Y=expected)
    scalars = directory / "scalars.json"
    scalars.write_text(json.dumps({"scale": 1.25, "bias": -0.5}))
    archived = TuningWorkload.from_archive(inputs, scalars={"scale": 1.25, "bias": -0.5},
        expected=references, rtol=2e-3, atol=1e-4, validation_key="archive-affine-f32")
    assert archived.contract_key() == workload.contract_key()
    command = [sys.executable, "-m", "weft", "tune", "--candidates", str(directory / "candidate-native"),
               "--inputs", str(inputs), "--expected", str(references), "--scalars", str(scalars),
               "--rtol", "0.002", "--atol", "0.0001", "--validation-key", "archive-affine-f32",
               "--cc", "/no/system-C-compiler", "--deadline", "600", "--warmup", "1",
               "--samples", "3", "--repetitions", "2", "--key", "scale", "--key", "bias",
               "--cache", str(cache), "--output", str(directory / "cli-selection.json")]
    if toolchain.compiler:
        command += ["--compiler", toolchain.compiler]
    response = subprocess.run([*command, "--artifact", str(directory / "cli-winner")],
                              check=True, capture_output=True, text=True)
    result = json.loads(response.stdout)
    assert result["validated"]
    selected = TuningResult.load(directory / "cli-selection.json")
    assert selected.selected.validated
    repeated = subprocess.run(command, check=True, capture_output=True, text=True)
    second = TuningResult.load(directory / "cli-selection.json")
    if second.identity == selected.identity:
        assert json.loads(repeated.stdout)["cache_hit"]
    else:
        # NPZ reload owns new arrays. Their natural allocator alignment can
        # change in a fresh process and must remain part of the workload key.
        before = json.loads(json.dumps(selected.identity))
        after = json.loads(json.dumps(second.identity))
        for identity in (before, after):
            for view in identity["workload"]["views"]:
                del view["alignment"]
        assert before == after
    unavailable_cc = Toolchain(toolchain.compiler, ("/no/system-C-compiler",))
    with Artifact(directory / "cli-winner").load(toolchain=unavailable_cc) as fixed:
        fixed(X=x, Y=y, scale=1.25, bias=-0.5)
        np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)

    # The same workload helper checks exact integer outputs and Torch state
    # without implicitly converting dtype, relaxing tolerances or editing state.
    integer = np.array([0, 1, -2147483648, 2147483647], dtype=np.int32)
    NumericalCheck(integer, rtol=1, atol=1).validate(integer.copy(), name="index")
    try:
        NumericalCheck(integer, rtol=1, atol=1).validate(integer + 1, name="index")
    except AssertionError:
        pass
    else:
        raise RuntimeError("integer reference accepted an inexact result")
    state = torch.linspace(-2, 2, 3 * 65).reshape(3, 65)
    delta = torch.ones_like(state)
    unchanged, version = state.clone(), state._version
    mutation = TuningWorkload({"State": state, "Delta": delta, "decay": 0.5},
        checks={"State": NumericalCheck(state * 0.5 + delta, rtol=2e-3, atol=1e-4)},
        validation_key="workload-state-f32")
    policy = TuningPolicy(deadline_s=600, warmup=1, samples=3, repetitions=1)
    selection = mutation.run(state_update, STATE_CONFIGS, policy=policy, toolchain=toolchain, cache=cache)
    assert selection.selected.validated
    torch.testing.assert_close(state, unchanged, rtol=0, atol=0)
    assert state._version == version
    return {"native_tune_cli": True, "independent_archive_references": True,
            "cli_fixed_artifact": True, "exact_integer_reference": True,
            "torch_workload_state_isolation": True}


def canonical_tuning_calls(toolchain, cache, directory):
    """External canonical input uses the same compiler, JIT, tuner and artifacts."""
    from weft.runtime.compiler import KernelSource
    from source.weft.programs import get

    captured = KernelSource.capture(affine_strided)
    canonical = captured.canonical
    described = KernelSource.from_ir(canonical, toolchain=toolchain)
    assert described.signature is None and described.description == captured.description
    # Public ABI specialization is author data. Varying it cannot share one
    # callable, even when every separate candidate would compile successfully.
    row = get("rms_norm_row")
    row_source = KernelSource.capture(row.definition)
    row_ir = KernelSource.from_ir(row_source.canonical, toolchain=toolchain)
    assert row_ir.description == row_source.description
    try:
        weft.autotune_ir(row_source.canonical,
                         (Config(meta={"N": 32}), Config(meta={"N": 64})), toolchain=toolchain)
    except ValueError:
        pass
    else:
        raise RuntimeError("canonical tuner accepted a changing public View dimension")

    arrays = np.arange(3 * 66, dtype=np.float32).reshape(3, 66)
    x = arrays[:, ::2]
    y = np.empty_like(x)
    expected = x * 1.25 - 0.5
    arguments = {"X": x, "Y": y, "scale": 1.25, "bias": -0.5}
    with weft.jit_ir(canonical, options=CompileOptions(meta={"B": 32}),
                      toolchain=toolchain, cache=cache) as compiled:
        assert compiled.compiled is None
        compiled(**arguments)
        np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)
        with compiled.specialize(B=64) as specialized:
            specialized(**arguments)
            np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)
    choices = ConfigSpace(source={"B": (32, 64)}, physical={"lmul_eighths": (8, 16)})
    policy = TuningPolicy(deadline_s=600, warmup=1, samples=3, repetitions=1)
    y.fill(-77)
    with weft.autotune_ir(canonical, choices, key=("scale", "bias"), policy=policy,
                          validator=check_affine, validation_key="canonical-affine-f32",
                          toolchain=toolchain, cache=cache) as tuned:
        result = tuned.tune(**arguments)
        assert result.selected.validated
        np.testing.assert_array_equal(y, -77)
        with tuned.fixed() as fixed:
            fixed(**arguments)
            np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)
        built = tuned.build_candidates(target=NativeTarget.discover(toolchain).build_target)
        saved = built.save(directory / "canonical-candidates")
        assert saved.source.signature is None
        assert saved.source.description == captured.description
    unavailable_cc = Toolchain(toolchain.compiler, ("/no/system-C-compiler",))
    with saved.autotune(key=("scale", "bias"), policy=policy,
                         validator=check_affine, validation_key="canonical-affine-f32",
                         toolchain=unavailable_cc, cache=cache) as tuned:
        tuned(**arguments)
        np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)

    # The frontend is absent in this command path: a plain MLIR file contains
    # the complete author program, and the compiler reports its source contract.
    ir_path = directory / "canonical.mlir"
    ir_path.write_text(canonical)
    configs_path = directory / "canonical-configs.json"
    configs_path.write_text(json.dumps([choices.to_dict()]))
    command = [sys.executable, "-m", "weft", "configs", str(configs_path), "--ir", str(ir_path), "--format", "json"]
    if toolchain.compiler:
        command += ["--compiler", toolchain.compiler]
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    assert len(json.loads(output.stdout)) == 4
    inputs_path, expected_path = directory / "canonical-inputs.npz", directory / "canonical-expected.npz"
    # Archive inputs have their own contiguous storage contract, explicitly
    # separate from the strided array used by the direct API above.
    np.savez(inputs_path, X=x, Y=np.full_like(expected, -77))
    np.savez(expected_path, Y=expected)
    scalars = directory / "canonical-scalars.json"
    scalars.write_text(json.dumps({"scale": 1.25, "bias": -0.5}))
    command = [sys.executable, "-m", "weft", "tune", "--ir", str(ir_path),
               "--configs", str(configs_path), "--inputs", str(inputs_path),
               "--expected", str(expected_path), "--scalars", str(scalars),
               "--validation-key", "canonical-cli-affine", "--rtol", "0.002", "--atol", "0.0001",
               "--deadline", "600", "--warmup", "1", "--samples", "2", "--repetitions", "1",
               "--cc", " ".join(toolchain.cc), "--cache", str(cache),
               "--output", str(directory / "canonical-cli-selection.json")]
    if toolchain.compiler:
        command += ["--compiler", toolchain.compiler]
    command += [f"--cflag={flag}" for flag in toolchain.cflags]
    command += [f"--ldflag={flag}" for flag in toolchain.ldflags]
    response = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(response.stdout)["validated"]
    successful = (directory / "canonical-cli-selection.json").read_bytes()
    np.savez(expected_path, Y=expected + 100)
    response = subprocess.run([*command, "--refresh"], capture_output=True, text=True)
    assert response.returncode != 0 and "validation-rejected" in response.stderr
    assert (directory / "canonical-cli-selection.json").read_bytes() == successful
    with np.load(inputs_path, allow_pickle=False) as inputs:
        np.testing.assert_array_equal(inputs["Y"], -77)
    # Cold tuning of an empty workload must also enforce private-storage budget;
    # an empty output's owner/alignment allocation is not free.
    empty = np.empty((0, 33), np.float32)
    with saved.autotune(policy=TuningPolicy(max_state_bytes=1, deadline_s=600),
                         validator=check_affine, validation_key="empty-affine",
                         toolchain=unavailable_cc, cache=cache) as tuned:
        try:
            tuned.tune(X=empty, Y=empty, scale=1.25, bias=-0.5)
        except ValueError as error:
            assert "tuning state needs" in str(error)
        else:
            raise RuntimeError("empty trial storage escaped its byte budget")
    return {"canonical_interface": True, "canonical_jit": True, "canonical_autotune": True,
            "canonical_fixed_deployment": True, "canonical_abi_domain_rejection": True,
            "canonical_native_cli": True, "failed_cli_preserved_selection": True,
            "empty_trial_storage_budget": True}


def view_calls(operator):
    storage = torch.arange(8 * 520, dtype=torch.float32).reshape(8, 520)
    x = storage[1::2, 3:516:2]
    expected = x * 1.25 - 0.5
    actual = operator(x, 1.25, bias=-0.5)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-4)
    backing = torch.full((4, 257 * 3), 47.0)
    out = backing[:, ::3]
    assert operator.out(x, 1.25, bias=-0.5, out=out) is None
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=1e-4)
    assert torch.all(backing[:, 1::3] == 47)
    assert torch.all(backing[:, 2::3] == 47)

    class OutModule(torch.nn.Module):
        def forward(self, value, destination):
            operator.out(value, 1.25, bias=-0.5, out=destination)
            return destination

    compiled_out = torch.compile(OutModule(), backend="aot_eager", fullgraph=True)
    compiled_out(x, out)
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=1e-4)
    transposed = torch.arange(9 * 17, dtype=torch.float32).reshape(17, 9).t()
    torch.testing.assert_close(operator(transposed), transposed)
    broadcast = torch.arange(17, dtype=torch.float32).expand(3, 17)
    torch.testing.assert_close(operator(broadcast), broadcast)
    for shape in ((0, 17), (3, 0), (0, 0)):
        empty = torch.empty(shape)
        assert operator(empty).shape == shape
        operator.out(empty, out=torch.empty_like(empty))
    try:
        operator.out(torch.ones(3, 17), out=torch.empty(17).expand(3, 17))
    except ValueError:
        pass
    else:
        raise RuntimeError("self-overlapping writable output was accepted")
    return {"strides_offsets": True, "transposed": True, "broadcast_reads": True,
            "explicit_out": True, "out_graph": True, "empty_views": True}


def graph_calls(toolchain, cache, directory, *, inductor):
    # Supplying nonexistent native tools makes accidental compilation observable.
    fake_jit = weft.jit(affine_strided, options=AFFINE_CONFIGS[0].options,
                        toolchain=Toolchain(compiler="/unavailable-weft-compiler"))
    fake_op = register("weft_repro_metadata::affine", fake_jit)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(3, 129)
        assert fake_op(x).shape == x.shape
        fake_op.out(x, out=torch.empty_like(x))
    assert fake_op(torch.empty(3, 129, device="meta")).device.type == "meta"
    dynamic = {"value": {0: torch.export.Dim("rows", min=2, max=16),
                          1: torch.export.Dim("columns", min=2, max=300)}}
    captured = torch.export.export(Affine(fake_op), (torch.empty(3, 129),), dynamic_shapes=dynamic)
    assert "weft_repro_metadata.affine" in str(captured.graph)
    assert fake_jit.compiled is None

    operator = register_affine("weft_repro", toolchain=toolchain, cache=cache)
    results = view_calls(operator)
    module = Affine(operator, scale=1.25, bias=-0.5)
    x = torch.linspace(-1, 1, 3 * 129).reshape(3, 129).requires_grad_()
    module(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 1.25), rtol=2e-3, atol=1e-4)
    checked = torch.library.opcheck(operator.op, (x, 1.25), {"bias": -0.5})
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    next_x = torch.randn(5, 73, requires_grad=True)
    compiled(next_x).sum().backward()
    torch.testing.assert_close(next_x.grad, torch.full_like(next_x, 1.25), rtol=2e-3, atol=1e-4)
    exported = torch.export.export(module, (x.detach(),), dynamic_shapes=dynamic)
    export_path = directory / "affine.pt2"
    torch.export.save(exported, export_path)
    loaded = torch.export.load(export_path).module()
    other = torch.randn(7, 61)
    torch.testing.assert_close(loaded(other), other * 1.25 - 0.5, rtol=2e-3, atol=1e-4)

    tuned_op = register_affine("weft_repro_tuned", toolchain=toolchain, cache=cache, tune=True)
    tuned_module = Affine(tuned_op, scale=1.25, bias=-0.5)
    torch.testing.assert_close(tuned_module(x.detach()), module(x.detach()), rtol=2e-3, atol=1e-4)
    tuned_graph = torch.compile(tuned_module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(tuned_graph(x.detach()), module(x.detach()), rtol=2e-3, atol=1e-4)
    assert tuned_op.kernel.last_result.winner in AFFINE_CONFIGS

    saved = operator.kernel.save(directory / "torch-artifact")
    consumer = ("import sys,torch; from weft.runtime import Artifact; from weft.torch import register; "
                "register('weft_repro::affine',Artifact(sys.argv[1])); "
                "m=torch.export.load(sys.argv[2]).module(); x=torch.arange(15,dtype=torch.float32).reshape(3,5); "
                "torch.testing.assert_close(m(x),x*1.25-0.5,rtol=2e-3,atol=1e-4)")
    environment = {**os.environ, "CC": "/unavailable-C-compiler"}
    environment.pop("PYTHONPATH", None)
    if toolchain.compiler:
        environment["WEFT_COMPILER"] = toolchain.compiler
    subprocess.run([sys.executable, "-c", consumer, str(saved.path), str(export_path)],
                   cwd=directory, env=environment, check=True)

    state_jit = weft.jit(state_scale, options=CompileOptions(meta={"B": 64}), toolchain=toolchain, cache=cache)
    state_op = register("weft_repro::state_scale", state_jit)

    class StateModule(torch.nn.Module):
        def forward(self, state):
            state_op(state, 0.5)
            return state

    mutable = torch.compile(StateModule(), backend="aot_eager", fullgraph=True)
    state = torch.arange(3 * 17, dtype=torch.float32).reshape(3, 17)
    initial = state.clone()
    mutable(state)
    torch.testing.assert_close(state, initial * 0.5)
    mutating = register("weft_repro::state_update", weft.jit(
        state_update, options=STATE_CONFIGS[0].options, toolchain=toolchain, cache=cache))
    try:
        mutating(state, state)
    except NotImplementedError:
        pass
    else:
        raise RuntimeError("Torch mutation accepted unsupported shared-storage inputs")
    if inductor:
        native_graph = torch.compile(module, fullgraph=True)
        torch.testing.assert_close(native_graph(other), other * 1.25 - 0.5, rtol=2e-3, atol=1e-4)
    results.update(fake_without_compiler=True, dynamic_export_reload=True, author_backward=True,
                   aot_autograd=True, mutable_graph=True, opcheck=checked, inductor=inductor,
                   autotuned_torch_graph=True, exported_source_free_consumer=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Weft autotune and Torch integration on the native RISC-V CPU")
    parser.add_argument("--cc", default="clang")
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument("--ldflag", action="append", default=[])
    parser.add_argument("--compiler")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--inductor", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    toolchain = Toolchain(args.compiler, (args.cc,), tuple(args.cflag), tuple(args.ldflag))
    if args.worker:
        print(json.dumps(tuning_worker(toolchain, args.worker)))
        return
    with tempfile.TemporaryDirectory(prefix="weft-framework-") as temporary:
        directory = Path(temporary)
        arguments = ["--cc", args.cc, *(f"--cflag={flag}" for flag in args.cflag),
                     *(f"--ldflag={flag}" for flag in args.ldflag)]
        if args.compiler:
            arguments += ["--compiler", args.compiler]
        command = [sys.executable, str(Path(__file__).resolve()), *arguments, "--worker", str(directory / "process-cache")]
        workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        records = []
        for worker in workers:
            stdout, stderr = worker.communicate()
            if worker.returncode:
                raise RuntimeError(f"selection worker failed: {stdout}\n{stderr}")
            records.append(json.loads(stdout))
        assert len({record["record"] for record in records}) == 1
        assert sum(record["cache_hit"] for record in records) == 1
        restart = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        assert restart["cache_hit"] and restart["record"] == records[0]["record"]
        result = {"concurrent_selection": True, "persistent_winner": True}
        result.update(compiler_contract_calls(toolchain, directory / "compiler-cache", directory))
        result.update(finite_candidates_calls(toolchain, directory / "finite-cache", directory))
        result.update(workload_calls(toolchain, directory / "finite-cache", directory))
        result.update(canonical_tuning_calls(toolchain, directory / "canonical-cache", directory))
        result.update(state_calls(toolchain, directory / "cache"))
        result.update(scalar_key_calls(toolchain, directory / "scalar-key-cache"))
        result.update(graph_calls(toolchain, directory / "cache", directory, inductor=args.inductor))
        print(json.dumps({"execution": "native-torch-autotune", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
