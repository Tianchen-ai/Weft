"""C-to-native stage of the Triton RISC-V backend."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import platform
import shutil
import subprocess
import tempfile


def native_gcc_runtime(abi):
    """Describe the installed C runtime inputs; Clang remains the compiler."""
    gcc = shutil.which("gcc")
    if gcc is None:
        return ()  # Clang can also discover runtime-only installations.
    gcc = str(Path(gcc).resolve(strict=True))
    triple = subprocess.run([gcc, "-dumpmachine"], check=True, capture_output=True, text=True).stdout.strip()
    if not triple.startswith("riscv64"):
        raise ValueError(f"native GCC runtime provider targets {triple}, expected riscv64; configure toolchain explicitly")
    def locate(name):
        spelling = subprocess.run([gcc, f"-mabi={abi}", f"-print-file-name={name}"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        if spelling == name or not Path(spelling).is_absolute():
            raise FileNotFoundError(f"native GCC cannot locate {name}; configure the target runtime in toolchain.cflags")
        return Path(spelling).resolve(strict=True)
    library = locate("libgcc.a")
    startup = locate("crtbeginS.o")
    if library.parent != startup.parent:
        raise ValueError("GCC runtime and shared startup files use different installations; configure toolchain explicitly")
    return ("--gcc-install-dir=" + str(library.parent),)


def gcc_runtime_identity(flags, abi):
    installation = None
    for position, flag in enumerate(flags):
        if flag.startswith("--gcc-install-dir="):
            installation = flag.partition("=")[2]
        elif flag == "--gcc-install-dir":
            if position + 1 == len(flags):
                raise ValueError("--gcc-install-dir needs an installation directory")
            installation = flags[position + 1]
    if installation is None:
        return None
    root = Path(installation).resolve(strict=True)
    inputs = {}
    for name in ("libgcc.a", "crtbeginS.o", "crtendS.o"):
        path = root / name
        if path.is_file():
            stat = path.stat()
            inputs[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            inputs[name] = None
    return {"installation": str(root), "abi": abi, "inputs": inputs}


@dataclass(frozen=True)
class Toolchain:
    cc: str
    target: str
    sysroot: str
    cflags: tuple = ()
    ldflags: tuple = ()
    sysroot_identity: str = ""
    runtime_files: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)

    @classmethod
    def resolve(cls, configuration, target):
        values = dict(configuration)
        bundled_compiler = False
        allowed = cls.__dataclass_fields__.keys()
        if values.keys() - allowed:
            raise ValueError(f"unknown toolchain fields: {sorted(values.keys() - allowed)}")
        for name in ("cflags", "ldflags"):
            flags = values.get(name, ())
            if isinstance(flags, str) or any(not isinstance(flag, str) for flag in flags):
                raise TypeError(f"toolchain.{name} must be a sequence of arguments")
            values[name] = tuple(flags)
        if "cc" not in values:
            configured = os.environ.get("TRITON_WEFT_CC")
            bundled = Path(__file__).parents[2] / "_C/weft/bin/clang"
            if configured:
                values["cc"] = configured
            elif bundled.is_file():
                bundled_compiler = True
                values["cc"] = str(bundled)
                prefix = bundled.parent.parent
                values["cflags"] += ("-I" + str(prefix / "include"),)
                values["ldflags"] += ("-L" + str(prefix / "lib"), "-Wl,-rpath,$ORIGIN/lib")
                libraries = sorted((prefix / "lib").glob("libomp*.so*"))
                bundled_libraries = {"lib/" + path.name: str(path) for path in libraries
                                     if path.name.startswith(("libomp.so", "libomp-"))}
                values["runtime_files"] = {**bundled_libraries, **values.get("runtime_files", {})}
            else:
                raise RuntimeError("RISC-V C toolchain is not configured; set toolchain.cc or TRITON_WEFT_CC")
        cc = shutil.which(values["cc"])
        if cc is None:
            raise FileNotFoundError(f"RISC-V compiler is not executable: {values['cc']}")
        values["cc"] = str(Path(cc).resolve())
        values.setdefault("target", "riscv64-linux-gnu")
        values.setdefault("sysroot", "")
        if not values["target"].startswith("riscv64"):
            raise ValueError("the Weft RISC-V backend requires a riscv64 toolchain target")
        if platform.machine() != "riscv64" and not values["sysroot"]:
            raise ValueError("cross compilation requires an explicit target sysroot")
        explicit_runtime = any(flag.startswith(("--gcc-", "--sysroot", "-rtlib=", "--rtlib="))
                               for flag in (*values["cflags"], *values["ldflags"]))
        if bundled_compiler and platform.machine() == "riscv64" and not values["sysroot"] and not explicit_runtime:
            values["cflags"] += native_gcc_runtime(target.abi)
        for flag in values["cflags"]:
            if flag.startswith("-march=") and flag != f"-march={target.march}":
                raise ValueError("toolchain ISA flags disagree with the compiler target")
            if flag.startswith("-mabi=") and flag != f"-mabi={target.abi}":
                raise ValueError("toolchain ABI flags disagree with the compiler target")
            if flag.startswith("-mrvv-vector-bits=") and flag != f"-mrvv-vector-bits={target.vlen_bits}":
                raise ValueError("toolchain vector length disagrees with the compiler target")
        vector_bits = f"-mrvv-vector-bits={target.vlen_bits}"
        if vector_bits not in values["cflags"]:
            # Physical layouts and runtime checks already require this VLEN.
            # Carry the same fact into native codegen instead of discarding it
            # at the intrinsic C boundary.
            values["cflags"] += (vector_bits,)
        if values["sysroot"]:
            root = Path(values["sysroot"]).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("toolchain.sysroot must be a directory")
            if not values.get("sysroot_identity"):
                raise ValueError("a cross sysroot needs its explicit SDK identity")
            values["sysroot"] = str(root)
        version = subprocess.run([values["cc"], "--version"], check=True, capture_output=True, text=True).stdout
        if "clang" not in version.lower():
            raise ValueError("this backend requires a Clang toolchain with the declared RISC-V intrinsics")
        targets = subprocess.run([values["cc"], "--print-targets"], check=True, capture_output=True, text=True).stdout
        if "riscv64" not in targets:
            raise ValueError(f"the selected Clang has no RISC-V code generator: {values['cc']}")
        stat = Path(cc).stat()
        values["identity"] = {"version": version, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "sysroot": values["sysroot"],
                              "sysroot_identity": values.get("sysroot_identity", ""),
                              "native_runtime": gcc_runtime_identity((*values["cflags"], *values["ldflags"]), target.abi),
                              "environment": {key: os.environ.get(key) for key in
                                              ("PATH", "CPATH", "C_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH",
                                               "GCC_EXEC_PREFIX")}}
        dependencies = {}
        for name, source in values.get("runtime_files", {}).items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parent == Path('.'):
                raise ValueError("runtime files must use relative library subdirectories")
            path = Path(source).resolve(strict=True)
            info = path.stat()
            dependencies[name] = {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
        values["identity"]["runtime_files"] = dependencies
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    def compile(self, source, metadata, target):
        from triton.runtime.cache import get_cache_manager
        from .launcher import launch_source
        cache = get_cache_manager(metadata["hash"])
        with tempfile.TemporaryDirectory(prefix="native-", dir=cache.cache_dir) as directory:
            root = Path(directory)
            kernel = root / "kernel.c"
            launcher = root / "launcher.c"
            output = root / "kernel.so"
            kernel.write_text(source)
            launcher.write_text(launch_source(metadata["kernel_abi"]))
            command = [self.cc, f"--target={self.target}", f"-march={target.march}",
                       f"-mabi={target.abi}", "-O3", "-ffp-contract=fast", "-std=gnu11", "-fPIC", "-shared", "-fopenmp",
                       "-Wl,-Bsymbolic-functions"]
            if self.sysroot:
                command += [f"--sysroot={self.sysroot}"]
            command += [*self.cflags, str(kernel), str(launcher), "-lm", *self.ldflags, "-o", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            metadata["native_build"] = {"toolchain": self.to_dict(), "language": "gnu11", "optimization": "O3",
                                        "command": command}
            if result.returncode:
                from triton.compiler.errors import BackendCompilationError
                raise BackendCompilationError({
                    "kind": "weft-compilation-failure", "stage": "native-compilation",
                    "pass": "clang", "category": "toolchain-error",
                    "pipeline": metadata["compilation"]["pipeline"],
                    "native_build": metadata["native_build"],
                    "exit_code": result.returncode,
                    "diagnostics": [{"severity": "error", "location": self.cc,
                                     "message": result.stderr.strip() or result.stdout.strip(),
                                     "notes": []}],
                })
            metadata["runtime_files"] = {name: cache.put(Path(path).read_bytes(), name)
                                         for name, path in self.runtime_files.items()}
            return output.read_bytes()
