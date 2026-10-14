"""CLI for finite configuration construction, compilation and native selection."""
from __future__ import annotations

import json
from pathlib import Path

from .api import load_definition
from .runtime import TuningPolicy
from .runtime.candidates import BuiltCandidates, build_candidates, build_candidates_ir
from .runtime.compiler import KernelSource
from .runtime.configuration import load_configurations, resolve_configurations, save_configurations
from .runtime.workload import TuningWorkload, load_workload


COMMANDS = {"configs", "candidates-build", "candidates-link", "candidates-inspect", "tune"}


def add_commands(commands, *, add_tools):
    configs = commands.add_parser("configs", help="expand and inspect explicit source/physical configuration domains")
    configs.add_argument("path", type=Path)
    config_source = configs.add_mutually_exclusive_group()
    config_source.add_argument("--kernel", help="check source domains and common invocation ABI without compiling")
    config_source.add_argument("--ir", type=Path, help="check a canonical frontend's source interface")
    add_tools(configs)
    configs.add_argument("--max-candidates", type=int, default=8)
    configs.add_argument("--output", type=Path, help="save the expanded complete configurations")
    configs.add_argument("--format", choices=("json", "text"), default="text")

    build = commands.add_parser("candidates-build", help="compile every explicit candidate for a target, without measuring")
    build_source = build.add_mutually_exclusive_group(required=True)
    build_source.add_argument("--kernel", help="module:kernel or file.py:kernel")
    build_source.add_argument("--ir", type=Path, help="Canonical Kernel IR from any frontend")
    build.add_argument("--configs", type=Path, required=True)
    build.add_argument("--max-candidates", type=int, default=8)
    build.add_argument("--march")
    build.add_argument("--abi")
    build.add_argument("--vlen-bits", type=int)
    build.add_argument("--data-cache")
    build.add_argument("--source-only", action="store_true")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cache", type=Path)
    add_tools(build)

    link = commands.add_parser("candidates-link", help="link a saved set of selected C artifacts without recompiling IR")
    link.add_argument("path", type=Path)
    link.add_argument("--output", type=Path, required=True)
    link.add_argument("--cache", type=Path)
    add_tools(link)

    inspect = commands.add_parser("candidates-inspect", help="inspect candidates and compiler rejections without executing")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--candidate", type=int)
    inspect.add_argument("--format", choices=("json", "text", "mlir"), default="text")

    tune = commands.add_parser("tune", help="select finite native candidates using explicit inputs and a numerical contract")
    source = tune.add_mutually_exclusive_group(required=True)
    source.add_argument("--kernel", help="module:kernel or file.py:kernel; requires --configs")
    source.add_argument("--ir", type=Path, help="Canonical Kernel IR from any frontend; requires --configs")
    source.add_argument("--candidates", type=Path, help="precompiled native candidate directory")
    tune.add_argument("--configs", type=Path)
    inputs = tune.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--workload", help="module:factory or file.py:factory returning TuningWorkload")
    inputs.add_argument("--inputs", type=Path, help="NPZ with every named array argument, including outputs/state")
    tune.add_argument("--scalars", type=Path, help="JSON scalar/shape arguments for --inputs")
    tune.add_argument("--expected", type=Path, help="NPZ independently computed reference for every writable argument")
    tune.add_argument("--validation-key", help="explicit numerical contract identity for archive references")
    tune.add_argument("--rtol", type=float, default=0.0)
    tune.add_argument("--atol", type=float, default=0.0)
    tune.add_argument("--equal-nan", action="store_true")
    tune.add_argument("--key", action="append", default=[], help="additional scalar/shape workload key")
    tune.add_argument("--deadline", type=float, default=120.0)
    tune.add_argument("--warmup", type=int, default=2)
    tune.add_argument("--samples", type=int, default=5)
    tune.add_argument("--repetitions", type=int, default=5)
    tune.add_argument("--max-candidates", type=int, default=8)
    tune.add_argument("--max-state-bytes", type=int, default=256 * 1024 * 1024)
    tune.add_argument("--refresh", action="store_true")
    tune.add_argument("--cache", type=Path)
    tune.add_argument("--output", type=Path, required=True, help="save the complete TuningResult")
    tune.add_argument("--artifact", type=Path, help="export the fixed native winner for ordinary deployment")
    add_tools(tune)


def _configuration_text(configs):
    lines = [f"Candidates: {len(configs)}"]
    baseline = configs[0]
    for index, config in enumerate(configs):
        lines.append(f"\n{index}: {config.name or '(unnamed)'}")
        lines.append("  Source: " + json.dumps(dict(config.source), sort_keys=True))
        lines.append("  Physical: " + json.dumps(dict(config.physical), sort_keys=True))
        if index:
            changes = config.changes_from(baseline)
            lines.append("  Changed source fields: " + ", ".join(changes["source"]))
            lines.append("  Changed physical fields: " + ", ".join(changes["physical"]))
    return "\n".join(lines)


def _inspect(arguments):
    built = BuiltCandidates.load(arguments.path)
    indices = range(len(built.entries))
    if arguments.candidate is not None:
        if not 0 <= arguments.candidate < len(built.entries):
            raise ValueError("candidate index is outside the saved collection")
        indices = (arguments.candidate,)
    if arguments.format == "mlir":
        if arguments.candidate is None:
            raise ValueError("mlir inspection requires --candidate INDEX")
        failure = built.entries[arguments.candidate].failure
        if failure is None:
            raise ValueError("candidate has no compiler rejection to replay")
        print(failure.as_pass_pipeline())
    elif arguments.format == "json":
        summary = built.summary()
        if arguments.candidate is not None:
            summary["candidates"] = [summary["candidates"][arguments.candidate]]
        print(json.dumps(summary, indent=2))
    else:
        print("Target: " + json.dumps(built.target.to_dict(), sort_keys=True))
        for index in indices:
            entry = built.entries[index]
            print(f"\n{index}: {entry.config.name or '(unnamed)'}: {entry.status}")
            print("  Source: " + json.dumps(dict(entry.config.source), sort_keys=True))
            print("  Physical: " + json.dumps(dict(entry.config.physical), sort_keys=True))
            if entry.artifact is not None:
                print(f"  Native library: {entry.artifact.manifest['native_library']}")
            else:
                print(entry.failure.render())


def _workload(arguments):
    if arguments.workload is not None:
        if (arguments.scalars is not None or arguments.expected is not None or
                arguments.validation_key is not None or arguments.rtol or arguments.atol or arguments.equal_nan):
            raise ValueError("Python workload factories own their inputs and numerical contract; archive options cannot override them")
        return load_workload(arguments.workload)
    scalars = {} if arguments.scalars is None else json.loads(arguments.scalars.read_text(encoding="utf-8"))
    if not isinstance(scalars, dict):
        raise ValueError("scalar argument file must be a JSON object")
    if arguments.expected is None and (arguments.rtol or arguments.atol or arguments.equal_nan):
        raise ValueError("numerical tolerances require independent expected outputs")
    return TuningWorkload.from_archive(arguments.inputs, scalars=scalars, expected=arguments.expected,
        rtol=arguments.rtol, atol=arguments.atol, equal_nan=arguments.equal_nan,
        validation_key=arguments.validation_key)


def _tune(arguments, tools):
    authored = arguments.kernel is not None or arguments.ir is not None
    if authored != (arguments.configs is not None):
        raise ValueError("--kernel/--ir requires --configs; prebuilt --candidates already declare their configurations")
    policy = TuningPolicy(max_candidates=arguments.max_candidates, deadline_s=arguments.deadline,
        warmup=arguments.warmup, samples=arguments.samples, repetitions=arguments.repetitions,
        max_state_bytes=arguments.max_state_bytes)
    if authored:
        source = load_definition(arguments.kernel) if arguments.kernel is not None else arguments.ir.read_text(encoding="utf-8")
        configs = load_configurations(arguments.configs, max_candidates=policy.max_candidates)
    else:
        source = BuiltCandidates.load(arguments.candidates)
        configs = None
    workload = _workload(arguments)
    result = workload.run(source, configs, key=arguments.key, policy=policy, toolchain=tools,
        cache=arguments.cache, refresh=arguments.refresh, selection=arguments.output, artifact=arguments.artifact)
    first = result.candidates[0].config
    print(json.dumps({"selection": str(arguments.output), "winner": result.winner.to_dict(),
                      "cache_hit": result.cache_hit, "validated": result.selected.validated,
                      "timing_scope": result.timing_scope, "elapsed_s": result.elapsed_s,
                      "budget_exhausted": result.budget_exhausted,
                      "comparison_to_first": result.compare(first),
                      "source_groups": result.source_groups}, indent=2))


def run(arguments, *, toolchain, build_target):
    command = arguments.command
    if command == "configs":
        configs = load_configurations(arguments.path, max_candidates=arguments.max_candidates)
        if arguments.kernel is not None or arguments.ir is not None:
            source = (KernelSource.capture(load_definition(arguments.kernel)) if arguments.kernel is not None else
                      KernelSource.from_ir(arguments.ir.read_text(encoding="utf-8"), toolchain=toolchain(arguments)))
            configs, _ = resolve_configurations(source, configs, max_candidates=arguments.max_candidates)
        if arguments.output is not None:
            save_configurations(arguments.output, configs, max_candidates=arguments.max_candidates)
        if arguments.format == "json":
            print(json.dumps([config.to_dict() for config in configs], indent=2))
        else:
            print(_configuration_text(configs))
    elif command == "candidates-inspect":
        _inspect(arguments)
    else:
        tools = toolchain(arguments)
        if command == "candidates-build":
            configs = load_configurations(arguments.configs, max_candidates=arguments.max_candidates)
            builder = build_candidates if arguments.kernel is not None else build_candidates_ir
            source = load_definition(arguments.kernel) if arguments.kernel is not None else arguments.ir.read_text(encoding="utf-8")
            built = builder(source, configs,
                target=build_target(arguments, tools), toolchain=tools,
                native_library=not arguments.source_only, cache=arguments.cache,
                max_candidates=arguments.max_candidates)
            built.save(arguments.output)
            print(json.dumps({"path": str(arguments.output), **built.summary()}, indent=2))
        elif command == "candidates-link":
            linked = BuiltCandidates.load(arguments.path).link(toolchain=tools, cache=arguments.cache)
            linked.save(arguments.output)
            print(json.dumps({"path": str(arguments.output), **linked.summary()}, indent=2))
        elif command == "tune":
            _tune(arguments, tools)
        else:
            raise ValueError(f"unknown tuning command {command!r}")
