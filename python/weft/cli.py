from __future__ import annotations

import argparse
import json
import os
import runpy
import shlex
import sys
from pathlib import Path

from .diagnostics import WeftError
from .api import KernelDefinition, load_definition
from .frontend import lower_to_mlir
from .runtime import Artifact, ArtifactCache, CompileOptions, NativeTarget, Package, Target, Toolchain, TuningError, TuningResult, build_ir, extension_catalog, sdk_prefix
from .runtime.compiler import KernelSource
from .runtime.driver import installed_tool
from .runtime.cache_geometry import DataCache
from .runtime.physical_input import PhysicalInput, prepare, prepare_ir


def _toolchain(args) -> Toolchain:
    environment = Toolchain.from_environment()
    return Toolchain(args.compiler if args.compiler is not None else environment.compiler,
                     tuple(shlex.split(args.cc)) if args.cc else environment.cc,
                     (*environment.cflags, *args.cflag), (*environment.ldflags, *args.ldflag))


def _add_tools(parser) -> None:
    parser.add_argument("--compiler", help="Weft compiler executable")
    parser.add_argument("--cc", help="system C compiler command")
    parser.add_argument("--cflag", action="append", default=[], help="one C flag; use --cflag=-flag")
    parser.add_argument("--ldflag", action="append", default=[], help="one linker flag")


def _add_binding(parser) -> None:
    binding = parser.add_mutually_exclusive_group()
    binding.add_argument("--options", type=Path, help="CompileOptions JSON object")
    binding.add_argument("--selection", type=Path, help="use a saved TuningResult as a fixed AOT binding")
    parser.add_argument("--meta", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--cache", type=Path)


def _options(args) -> CompileOptions:
    if args.selection is not None:
        if args.meta:
            raise ValueError("a measured selection is a complete binding; --meta cannot override it")
        return TuningResult.load(args.selection).winner.options
    values = json.loads(args.options.read_text()) if args.options is not None else {}
    options = CompileOptions(**values)
    bindings = {}
    for value in args.meta:
        name, separator, extent = value.partition("=")
        if not separator or name in bindings:
            raise ValueError(f"expected a unique source binding NAME=VALUE, got {value!r}")
        bindings[name] = int(extent)
    return options.specialize(**bindings)


def _check_selection(path: Path, canonical: str, target: Target):
    TuningResult.load(path).check_program(canonical, target)


def _show_selection(arguments):
    selection = TuningResult.load(arguments.path)
    if arguments.identity and arguments.format != "json":
        raise ValueError("--identity requires selection --format json")
    index = arguments.candidate
    if index is not None and not 0 <= index < len(selection.candidates):
        raise ValueError(f"candidate index must be between 0 and {len(selection.candidates) - 1}")
    if arguments.format == "mlir":
        if index is None:
            raise ValueError("selection --format mlir requires --candidate INDEX")
        candidate = selection.candidates[index]
        if candidate.status != "compile-rejected" or candidate.failure is None:
            raise ValueError("candidate has no compiler rejection to replay")
        print(candidate.failure.as_pass_pipeline())
    elif arguments.format == "text":
        print(selection.render(indices=None if index is None else (index,)))
    elif arguments.format == "csv":
        print(selection.to_csv(indices=None if index is None else (index,)), end="")
    else:
        result = selection.summary()
        if index is not None:
            result = {"candidate_index": index, **result["candidates"][index]}
        if arguments.identity:
            result["identity"] = selection.identity
        print(json.dumps(result, indent=2))


def _named_entries(values):
    result = {}
    for spelling in values:
        name, separator, value = spelling.partition("=")
        if not separator or not name or not value or name in result:
            raise ValueError("expected a distinct NAME=VALUE entry")
        result[name] = value
    return result


def _show_target(arguments):
    tools = _toolchain(arguments)
    explicit = (arguments.march is not None, arguments.abi is not None, arguments.vlen_bits is not None)
    physical = {name: getattr(arguments, name) for name in
                ("matrix_extension", "max_widening_combine_groups", "partial_combine_policy",
                 "record_axis_policy", "private_stack_bytes") if getattr(arguments, name) is not None}
    if arguments.catalog:
        if any(explicit) or physical:
            raise ValueError("--catalog lists registered extensions independently of a target binding")
        print(json.dumps(extension_catalog(toolchain=tools).to_dict(), indent=2))
    elif any(explicit):
        if not all(explicit):
            raise ValueError("an explicit target query requires --march, --abi and --vlen-bits")
        target = Target(arguments.march, arguments.abi, arguments.vlen_bits)
        print(json.dumps(target.capabilities(options=CompileOptions(**physical), toolchain=tools).to_dict(), indent=2))
    else:
        if physical:
            raise ValueError("physical capability bindings require an explicit target query")
        print(json.dumps(NativeTarget.discover(tools).to_dict(), indent=2))


def _build_target(args, toolchain):
    explicit = (args.march is not None, args.abi is not None, args.vlen_bits is not None)
    if any(explicit) and not all(explicit):
        raise ValueError("explicit targets require --march, --abi and --vlen-bits together")
    if args.data_cache is not None and not all(explicit):
        raise ValueError("--data-cache requires an explicit build target")
    cache_geometry = DataCache.parse(args.data_cache) if args.data_cache is not None else None
    return Target(args.march, args.abi, args.vlen_bits, cache_geometry) if all(explicit) else NativeTarget.discover(toolchain).build_target


def _build(args) -> None:
    toolchain = _toolchain(args)
    target = _build_target(args, toolchain)
    parameters = dict(target=target, options=_options(args), toolchain=toolchain,
                      native_library=not args.source_only, cache=args.cache)
    if args.ir is not None:
        canonical = args.ir.read_text(encoding="utf-8")
        if args.selection is not None:
            _check_selection(args.selection, canonical, target)
        artifact = build_ir(canonical, **parameters)
    else:
        source = KernelSource.capture(load_definition(args.kernel))
        if args.selection is not None:
            _check_selection(args.selection, source.canonical, target)
        artifact = source.build(**parameters)
    if args.output is not None:
        saved = artifact.save(args.output)
        path = saved.path
    else:
        path = artifact.path
    print(json.dumps({"artifact": str(path), "cache_hit": artifact.cache_hit,
                      "symbol": artifact.metadata["symbol"], "target": target.to_dict()}))


def _prepare_input(args):
    toolchain = _toolchain(args)
    target = _build_target(args, toolchain)
    options = _options(args)
    parameters = dict(target=target, options=options, checkpoint=args.checkpoint,
                      toolchain=toolchain, cache=args.cache)
    if args.ir is not None:
        canonical = args.ir.read_text(encoding="utf-8")
        if args.selection is not None:
            _check_selection(args.selection, canonical, target)
        selected = prepare_ir(canonical, **parameters)
    else:
        definition = load_definition(args.kernel)
        if args.selection is not None:
            _check_selection(args.selection, KernelSource.capture(definition).canonical, target)
        selected = prepare(definition, **parameters)
    path = selected.save(args.output)
    print(json.dumps({"physical_input": str(path), "checkpoint": selected.checkpoint,
                      "target": selected.target.to_dict()}))


def _resume_input(args):
    selected = PhysicalInput.load(args.input)
    artifact = selected.complete(toolchain=_toolchain(args), native_library=not args.source_only, cache=args.cache)
    path = artifact.save(args.output).path if args.output is not None else artifact.path
    print(json.dumps({"artifact": str(path), "entry": artifact.compilation.entry,
                      "cache_hit": artifact.cache_hit, "symbol": artifact.metadata["symbol"]}))


def _inspect_input(args):
    selected = PhysicalInput.load(args.input)
    if args.format != "json" and args.emit != "selections":
        raise ValueError("--format text requires --emit selections")
    if args.emit == "physical":
        print(selected.ir, end="")
    elif args.emit == "canonical":
        print(selected.canonical_ir, end="")
    elif args.emit == "selections":
        selections = selected.selections(toolchain=_toolchain(args))
        if args.format == "text":
            print(selections.render(), end="")
        else:
            print(json.dumps(selections.to_dict(), indent=2))
    else:
        print(json.dumps({"checkpoint": selected.checkpoint, "target": selected.target.to_dict(),
                          "options": selected.options.to_dict(), "signature": selected.signature,
                          "interface": selected.interface}, indent=2))


def _add_source_target(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kernel", help="module:kernel or file.py:kernel")
    source.add_argument("--ir", type=Path, help="Canonical Kernel IR from any frontend")
    parser.add_argument("--march")
    parser.add_argument("--abi")
    parser.add_argument("--vlen-bits", type=int)
    parser.add_argument("--data-cache", help="known L1 cache SIZE_BYTES:LINE_BYTES:WAYS:SETS")
    _add_tools(parser)
    _add_binding(parser)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="weft", description="Build, inspect and deploy native Weft kernels")
    commands = result.add_subparsers(dest="command", required=True)
    from .tuning_cli import add_commands
    add_commands(commands, add_tools=_add_tools)
    lower_parser = commands.add_parser("lower", help="lower a Python kernel to Canonical Kernel IR")
    lower_parser.add_argument("source", type=Path)
    lower_parser.add_argument("--kernel", help="kernel name when a file defines several")
    compile_parser = commands.add_parser("build", help="build a native or source-only artifact")
    _add_source_target(compile_parser)
    compile_parser.add_argument("--source-only", action="store_true", help="emit C/IR/header without invoking a C compiler")
    compile_parser.add_argument("--output", type=Path, help="export a new relocatable artifact directory")
    prepare_parser = commands.add_parser("prepare", help="save selected Physical IR before contraction or layout materialization")
    _add_source_target(prepare_parser)
    prepare_parser.add_argument("--checkpoint", choices=("contractions", "panel-transfers", "layout"), default="contractions")
    prepare_parser.add_argument("--output", type=Path, required=True, help="new physical input JSON file")
    resume_parser = commands.add_parser("resume", help="complete a saved physical selection through the compiler main suffix")
    resume_parser.add_argument("input", type=Path)
    resume_parser.add_argument("--output", type=Path, help="new callable artifact directory")
    resume_parser.add_argument("--source-only", action="store_true")
    resume_parser.add_argument("--cache", type=Path)
    _add_tools(resume_parser)
    input_parser = commands.add_parser("inspect-input", help="inspect a saved physical selection without compiling or executing")
    input_parser.add_argument("input", type=Path)
    input_parser.add_argument("--emit", choices=("physical", "canonical", "binding", "selections"), default="physical")
    input_parser.add_argument("--format", choices=("json", "text"), default="json")
    _add_tools(input_parser)
    inspect_parser = commands.add_parser("inspect", help="inspect an artifact without executing it")
    inspect_parser.add_argument("artifact", type=Path)
    inspect_parser.add_argument("--emit", choices=("metadata", "c", "canonical", "physical", "header", "resources", "operations", "pipeline"), default="metadata")
    inspect_parser.add_argument("--format", choices=("json", "text", "csv", "mlir"),
                                help="format for --emit operations or pipeline (default: json)")
    for field in ("engine", "family", "instruction", "memory-form"):
        inspect_parser.add_argument(f"--{field}", help=f"filter --emit operations by exact {field}")
    signature_parser = commands.add_parser("signature", help="inspect source arguments, outputs and auto domains without compiling")
    signature_parser.add_argument("kernel", nargs="?", help="module:kernel or file.py:kernel")
    signature_parser.add_argument("--ir", type=Path, help="read a canonical frontend's invocation contract")
    _add_tools(signature_parser)
    selection_parser = commands.add_parser("selection", help="inspect a saved tuning decision and candidate rejections")
    selection_parser.add_argument("path", type=Path)
    selection_parser.add_argument("--format", choices=("json", "text", "csv", "mlir"), default="json",
                                  help="selection summary, readable diagnostics, or rejected pass prefix")
    selection_parser.add_argument("--candidate", type=int, metavar="INDEX",
                                  help="inspect one candidate by its zero-based record index")
    selection_parser.add_argument("--identity", action="store_true", help="include the exact workload and build identity")
    target_parser = commands.add_parser("target", help="query native CPU facts, an explicit target, or registered extensions")
    target_parser.add_argument("--catalog", action="store_true", help="list compiler-registered local extensions on any host")
    target_parser.add_argument("--march", help="explicit RISC-V ISA for a compiler capability query")
    target_parser.add_argument("--abi")
    target_parser.add_argument("--vlen-bits", type=int)
    target_parser.add_argument("--matrix-extension", help="registered matrix extension to enable")
    target_parser.add_argument("--max-widening-combine-groups", type=int)
    target_parser.add_argument("--partial-combine-policy")
    target_parser.add_argument("--record-axis-policy")
    target_parser.add_argument("--private-stack-bytes", type=int)
    _add_tools(target_parser)
    link_parser = commands.add_parser("link", help="link an existing selected C artifact")
    link_parser.add_argument("artifact", type=Path)
    link_parser.add_argument("--output", type=Path)
    link_parser.add_argument("--cache", type=Path)
    _add_tools(link_parser)
    cache_parser = commands.add_parser("cache", help="list completed cache entries")
    cache_parser.add_argument("--cache", type=Path)
    sdk_parser = commands.add_parser("sdk", help="locate the installed C/C++ runtime and CMake package")
    display = sdk_parser.add_mutually_exclusive_group()
    display.add_argument("--prefix", action="store_true")
    display.add_argument("--cmake-dir", action="store_true")
    package_parser = commands.add_parser("package", help="publish named artifacts for native consumers")
    package_parser.add_argument("--entry", action="append", required=True, metavar="NAME=ARTIFACT")
    package_parser.add_argument("--ggml-vec-dot", action="append", default=[], metavar="TYPE=NAME")
    package_parser.add_argument("--output", type=Path, required=True)
    package_link = commands.add_parser("package-link", help="link all source-only package entries before deployment")
    package_link.add_argument("package", type=Path)
    package_link.add_argument("--output", type=Path, required=True)
    package_link.add_argument("--cache", type=Path)
    _add_tools(package_link)
    package_build = commands.add_parser("package-build", help="build an explicit multi-kernel deployment specification")
    package_build.add_argument("specification", type=Path)
    package_build.add_argument("--output", type=Path, required=True)
    package_build.add_argument("--source-only", action="store_true")
    package_build.add_argument("--cache", type=Path)
    _add_tools(package_build)
    llama_parser = commands.add_parser("llama-install", help="install reviewed GGML build and call-site hooks in an upstream tree")
    llama_parser.add_argument("source", type=Path)
    return result


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        from .tuning_cli import COMMANDS, run as run_tuning
        if arguments.command in COMMANDS:
            run_tuning(arguments, toolchain=_toolchain, build_target=_build_target)
        elif arguments.command == "lower":
            scope = runpy.run_path(str(arguments.source))
            definitions = {name: value for name, value in scope.items() if isinstance(value, KernelDefinition)}
            if arguments.kernel is not None:
                if arguments.kernel not in definitions:
                    raise ValueError(f"source defines no kernel named {arguments.kernel!r}")
                definition = definitions[arguments.kernel]
            elif len(definitions) == 1:
                definition = next(iter(definitions.values()))
            else:
                raise ValueError("source must define exactly one kernel or select one with --kernel")
            print(lower_to_mlir(definition), end="")
        elif arguments.command == "build":
            _build(arguments)
        elif arguments.command == "prepare":
            _prepare_input(arguments)
        elif arguments.command == "resume":
            _resume_input(arguments)
        elif arguments.command == "inspect-input":
            _inspect_input(arguments)
        elif arguments.command == "target":
            _show_target(arguments)
        elif arguments.command == "signature":
            if (arguments.kernel is None) == (arguments.ir is None):
                raise ValueError("signature requires exactly one Python kernel or --ir canonical file")
            source = (KernelSource.capture(load_definition(arguments.kernel)) if arguments.kernel is not None else
                      KernelSource.from_ir(arguments.ir.read_text(encoding="utf-8"), toolchain=_toolchain(arguments)))
            print(json.dumps({**source.description, "parameters": source.signature}, indent=2))
        elif arguments.command == "selection":
            _show_selection(arguments)
        elif arguments.command == "cache":
            print(json.dumps([str(path) for path in ArtifactCache(arguments.cache).entries()], indent=2))
        elif arguments.command == "sdk":
            prefix = sdk_prefix()
            cmake = prefix / "lib/cmake/Weft"
            if arguments.prefix or arguments.cmake_dir:
                print(prefix if arguments.prefix else cmake)
            else:
                print(json.dumps({"prefix": str(prefix), "cmake_dir": str(cmake),
                                  "library": str(prefix / "lib/libweft_runtime.so")}, indent=2))
        elif arguments.command == "package":
            entries = {name: Artifact(path) for name, path in _named_entries(arguments.entry).items()}
            bindings = {"ggml.vec_dot": _named_entries(arguments.ggml_vec_dot)} if arguments.ggml_vec_dot else {}
            package = Package.create(arguments.output, entries, bindings=bindings)
            print(json.dumps({"package": str(package.path), "kernels": list(package.kernels)}))
        elif arguments.command == "package-link":
            package = Package(arguments.package).link(arguments.output, toolchain=_toolchain(arguments), cache=arguments.cache)
            print(json.dumps({"package": str(package.path), "kernels": list(package.kernels)}))
        elif arguments.command == "package-build":
            from .runtime.packaging import PackageBuilder

            builder = PackageBuilder.from_file(arguments.specification)
            package = builder.build(arguments.output, toolchain=_toolchain(arguments),
                                    native_library=not arguments.source_only, cache=arguments.cache)
            print(json.dumps({"package": str(package.path), "kernels": list(package.kernels),
                              "bindings": package.bindings}))
        elif arguments.command == "llama-install":
            from .integrations.llama_cpp import install_source

            print(json.dumps(install_source(arguments.source), indent=2))
        elif arguments.command == "link":
            artifact = Artifact(arguments.artifact).link(toolchain=_toolchain(arguments), cache=arguments.cache)
            path = artifact.save(arguments.output).path if arguments.output else artifact.path
            print(json.dumps({"artifact": str(path), "cache_hit": artifact.cache_hit}))
        elif arguments.command == "inspect":
            artifact = Artifact(arguments.artifact)
            filters = {name: getattr(arguments, name) for name in ("engine", "family", "instruction", "memory_form")}
            if arguments.emit != "operations" and any(value is not None for value in filters.values()):
                raise ValueError("operation filters require --emit operations")
            if arguments.emit not in ("operations", "pipeline") and arguments.format is not None:
                raise ValueError("--format requires --emit operations or pipeline")
            files = {"metadata": "manifest.json", "c": "kernel.c", "canonical": "canonical.mlir",
                     "physical": "physical.mlir", "header": "kernel.h"}
            if arguments.emit == "operations":
                if arguments.format == "mlir":
                    raise ValueError("mlir format requires --emit pipeline")
                summary = artifact.physical_operations.select(**filters)
                if arguments.format == "text":
                    print(summary.render(), end="")
                elif arguments.format == "csv":
                    print(summary.to_csv(), end="")
                else:
                    print(json.dumps(summary.to_dict(), indent=2))
            elif arguments.emit == "resources":
                print(json.dumps(dict(artifact.physical_resources), indent=2))
            elif arguments.emit == "pipeline":
                if arguments.format == "csv":
                    raise ValueError("pipeline inspection supports json, text or mlir")
                if arguments.format == "text":
                    print(artifact.compilation.render(), end="")
                elif arguments.format == "mlir":
                    print(artifact.compilation.as_pass_pipeline())
                else:
                    print(json.dumps(artifact.compilation.to_dict(), indent=2))
            else:
                print((artifact.path / files[arguments.emit]).read_text(), end="")
    except (WeftError, OSError, ValueError, TypeError, TuningError) as error:
        print(f"weft: {error}", file=sys.stderr)
        return 1
    return 0


def _native_tool(name: str) -> None:
    path = installed_tool(name)
    if path is None:
        raise FileNotFoundError(f"the Python environment has no installed native {name}")
    os.execv(str(path), [str(path), *sys.argv[1:]])


def compiler_main() -> None:
    _native_tool("weft-compile")


def optimizer_main() -> None:
    _native_tool("weft-opt")


if __name__ == "__main__":
    raise SystemExit(main())
