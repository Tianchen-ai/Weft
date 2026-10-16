from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import weft
from weft.runtime import Buffer, CompileOptions, Toolchain
from weft.runtime.target import NativeTarget


def run(command, *, source=None):
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command!r}\n{result.stderr}")
    return result.stdout


def run_native_jit(args, selected, source_dir, toolchain, common, ggml):
    spec = importlib.util.spec_from_file_location("ime1_production", source_dir / "kernels.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("the unique production author source cannot be loaded")
    programs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = programs
    spec.loader.exec_module(programs)
    pack = weft.compile(getattr(programs, selected["pack"]), toolchain=toolchain,
                        options=CompileOptions(matrix_extension=args.matrix_extension))
    kernel = None if args.pack_only else weft.compile(
        getattr(programs, selected["kernel"]), toolchain=toolchain,
        options=CompileOptions(meta=selected["source"], matrix_extension=args.matrix_extension,
                               local_tile_strategy=args.local_tile_strategy,
                               require_extensions=() if args.matrix_extension == "none" else (args.matrix_extension,)))
    reference = args.output_dir / "ime1-reference.so"
    run([args.cxx, *common, "-std=c++17", "-fPIC", "-shared",
         "-I" + str(ggml / "include"), "-I" + str(ggml / "src"),
         "-I" + str(ggml / "src/ggml-cpu"), str(Path(__file__).with_name("mul_mat_runtime.cpp")),
         "-L" + str(args.ggml_lib_dir), "-Wl,-rpath," + str(args.ggml_lib_dir),
         "-Wl,--no-as-needed", "-lggml", "-lggml-cpu", "-lggml-base",
         "-lm", "-ldl", "-pthread", "-o", str(reference)])
    library = ctypes.CDLL(str(reference.resolve()), mode=os.RTLD_LOCAL)
    pack_callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.c_size_t)
    kernel_callback = ctypes.CFUNCTYPE(ctypes.c_int, *([ctypes.c_void_p] * 4),
                                       *([ctypes.c_size_t] * 3))
    errors = []

    def invoke(compiled, addresses, dimensions):
        try:
            buffers = []
            for address, parameter in zip(addresses, compiled.metadata["arguments"], strict=True):
                shape = tuple(int(dimension) if dimension.isdecimal() else dimensions[dimension]
                              for dimension in parameter["shape"])
                size = math.prod(shape) // parameter["record_elements"] * parameter["storage_bytes"]
                storage = (ctypes.c_uint8 * size).from_address(address)
                buffers.append(Buffer(storage, shape=shape, encoding=parameter["encoding"]))
            compiled(*buffers)
            return 0
        except BaseException as error:
            # Exceptions must cross the C callback as an explicit failure, then
            # be re-raised by its Python caller; ctypes cannot propagate them.
            errors.append(error)
            return 1

    pack_call = pack_callback(lambda plain, packed, n, k:
                             invoke(pack, (plain, packed), {"N": n, "K": k}))
    kernel_call = kernel_callback() if kernel is None else kernel_callback(
        lambda weights, activation, workspace, output, n, k, m:
        invoke(kernel, (weights, activation, workspace, output), {"N": n, "K": k, "M": m}))
    entry = library.ime1_run_native_jit
    entry.argtypes = (ctypes.c_char_p, pack_callback, kernel_callback)
    entry.restype = ctypes.c_int
    try:
        status = entry(args.case.encode(), pack_call, kernel_call)
        if errors:
            raise errors[0]
        if status:
            raise RuntimeError(f"public native JIT production repro failed ({status})")
        for compiled in (pack,) if kernel is None else (pack, kernel):
            prefix = args.output_dir / compiled.metadata["symbol"]
            prefix.with_suffix(".riscv.mlir").write_text(compiled.riscv_ir)
            prefix.with_suffix(".c").write_text(compiled.source)
        print(json.dumps({"execution": "native-jit", "case": args.case,
                          "native_target": pack.target.march, "timing": "not-measured",
                          "used_extensions": [] if kernel is None else list(kernel.metadata["used_extensions"]),
                          "local_tile_strategy": None if kernel is None else kernel.metadata["local_tile_strategy"],
                          "local_tile_realizations": {} if kernel is None else dict(kernel.metadata["local_tile_realizations"]),
                          "source_bindings": selected["source"]}, sort_keys=True), flush=True)
    finally:
        if kernel is not None:
            kernel.close()
        pack.close()


def main():
    parser = argparse.ArgumentParser(description="Compile and run the production IME1 programs on RISC-V")
    parser.add_argument("case")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--ggml-source", type=Path, required=True)
    parser.add_argument("--ggml-lib-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--execution", choices=("native-aot", "native-jit"), default="native-aot")
    parser.add_argument("--nr", type=int)
    parser.add_argument("--matrix-extension", choices=("none", "spacemit-ime1"), default="spacemit-ime1",
                        help="explicit compiler capability; upstream M=1 and M=4 packed kernels both use IME1")
    parser.add_argument("--local-tile-strategy",
                        choices=("target-default", "atom-partitioned", "register-grouped"),
                        default="target-default")
    args = parser.parse_args()
    if args.pack_only and args.local_tile_strategy != "target-default":
        parser.error("an explicit local tile strategy requires a contraction, not pack-only execution")
    root = Path(__file__).resolve().parents[4]
    source_dir = root / "source/weft/ggml/mul_mat/ime1"
    bindings = json.loads((source_dir / "bindings.json").read_text())
    if args.case not in bindings["cases"] or args.repetitions <= 0:
        parser.error("select one declared case and positive repetitions")
    selected = bindings["cases"][args.case]
    if args.nr is not None:
        if args.nr not in bindings["source_parameter_domains"]["NR"]:
            parser.error("NR must belong to the co-located source parameter domain")
        selected = {**selected, "source": {**selected["source"], "NR": args.nr}}
    toolchain = Toolchain(args.compiler, (args.cc,), ("-fno-integrated-as",)).resolve()
    target = NativeTarget.discover(toolchain)
    required = () if args.matrix_extension == "none" else (args.matrix_extension,)
    target.check_execution(required)
    if len(os.sched_getaffinity(0)) != 1 or target.vlen_bits != 256 or target.abi != "lp64d":
        raise RuntimeError("this comparison requires one explicitly selected IME1 core, VLEN=256 and lp64d")

    # This explicit native AOT binding matches the fixed GGML baseline flags.
    # Native JIT's automatically discovered ISA is exercised by native.py.
    march = "rv64gcv_zfh_zvfh_zicbop_zihintpause_zba"
    common = ["-O3", "-Wall", "-Wextra", "-Werror", "-ffp-contract=fast",
              "-fno-integrated-as", f"-march={march}", "-mabi=lp64d"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ggml = args.ggml_source / "ggml"
    if args.execution == "native-jit":
        run_native_jit(args, selected, source_dir, toolchain, common, ggml)
        return

    def compile_entry(entry, meta, require_mma):
        canonical = run([sys.executable, "-m", "weft", "lower", str(source_dir / "kernels.py"), "--kernel", entry])
        arguments = [toolchain.compiler, "--emit=artifact", f"--march={march}",
                     "--abi=lp64d", "--vlen-bits=256", f"--matrix-extension={args.matrix_extension}"]
        arguments += [f"--meta={name}={value}" for name, value in sorted(meta.items())]
        if require_mma:
            if required:
                arguments.append(f"--require-extension={args.matrix_extension}")
            arguments.append(f"--local-tile-strategy={args.local_tile_strategy}")
        artifact = json.loads(run(arguments, source=canonical))
        metadata, = artifact["kernels"]
        expected_shapes = ["N", "K", "M"] if require_mma else ["N", "K"]
        if metadata["shape_parameters"] != expected_shapes or metadata["matrix_extensions"] != list(required):
            raise RuntimeError("the emitted ABI does not match the explicit repro calling convention")
        prefix = args.output_dir / entry
        prefix.with_suffix(".canonical.mlir").write_text(canonical)
        prefix.with_suffix(".riscv.mlir").write_text(artifact["riscv_ir"])
        prefix.with_suffix(".json").write_text(json.dumps(artifact, indent=2))
        prefix.with_suffix(".c").write_text(artifact["intrinsic_c"])
        library = prefix.with_suffix(".so")
        run([args.cc, *common, "-std=c11", "-fPIC", "-shared", str(prefix.with_suffix(".c")),
             "-lm", "-o", str(library)])
        return str(library.resolve()), metadata["symbol"]

    pack = compile_entry(selected["pack"], {}, False)
    kernel = () if args.pack_only else compile_entry(selected["kernel"], selected["source"], True)
    executable = args.output_dir / "ime1-repro"
    run([args.cxx, *common, "-std=c++17", "-I" + str(ggml / "include"),
         "-I" + str(ggml / "src"), "-I" + str(ggml / "src/ggml-cpu"),
         str(Path(__file__).with_name("mul_mat_runtime.cpp")),
         "-L" + str(args.ggml_lib_dir), "-Wl,-rpath," + str(args.ggml_lib_dir),
         "-Wl,--no-as-needed", "-lggml", "-lggml-cpu", "-lggml-base",
         "-lm", "-ldl", "-pthread", "-o", str(executable)])
    target.check_execution(required)
    result = subprocess.run([str(executable.resolve()), args.case, str(args.repetitions), *pack, *kernel])
    if result.returncode:
        raise SystemExit(result.returncode)
    print(json.dumps({"execution": "native-aot", "native_target": target.march, "comparison_march": march,
                      "local_tile_strategy": args.local_tile_strategy,
                      "source_bindings": selected["source"], "binding_origin": bindings["binding_origin"],
                      "case": args.case, "output_dir": str(args.output_dir.resolve())}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
