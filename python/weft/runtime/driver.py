from __future__ import annotations

import os
import importlib.metadata
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from weft.diagnostics import WeftError


LIBRARY_FLAGS = ("-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-ffp-contract=fast", "-fPIC", "-shared")


class CompilationError(WeftError):
    """A failed build stage, including its command and original diagnostics."""

    def __init__(self, message, *, stage=None, command=(), returncode=None,
                 stdout="", stderr="", failure=None):
        self.stage = stage
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.failure = failure
        super().__init__(message)

    def with_context(self, context):
        """Add a consumer location while preserving the compiler's classification."""
        return type(self)(f"{context}: {self}", stage=self.stage, command=self.command,
                          returncode=self.returncode, stdout=self.stdout, stderr=self.stderr,
                          failure=self.failure)


class CandidateRejected(CompilationError):
    """An explicitly classified unsupported or over-budget compiler candidate."""

    def __init__(self, message, *, failure, **context):
        from .compilation import CompilationFailure

        if not isinstance(failure, CompilationFailure) or not failure.rejects_candidate:
            raise TypeError("candidate rejection requires an explicit compiler resource/unsupported failure")
        super().__init__(message, failure=failure, **context)


def run_compiler(command: list[str], source: str | None = None, *,
                 stage: str = "compilation", cwd: Path | None = None,
                 structured: bool = False) -> str:
    try:
        result = subprocess.run(command, input=source, text=True, capture_output=True, cwd=cwd)
    except OSError as error:
        raise CompilationError(f"{stage}: cannot execute {shlex.join(command)}: {error}",
                               stage=stage, command=command) from error
    if result.returncode:
        context = dict(stage=stage, command=command, returncode=result.returncode,
                       stdout=result.stdout, stderr=result.stderr)
        message = (f"{stage} exited with status {result.returncode}\n"
                   f"command: {shlex.join(command)}\n")
        # A signal, an ordinary tool failure or a malformed protocol is fatal.
        # Only an explicit successful decode of the compiler's failure envelope
        # permits a tuner to discard this physical candidate.
        if structured and result.returncode > 0:
            from .compilation import CompilationFailure

            try:
                failure = CompilationFailure.from_dict(json.loads(result.stdout))
            except (ValueError, TypeError, CompilationError) as error:
                raise CompilationError(message + result.stdout + result.stderr,
                                       **context) from error
            kind = CandidateRejected if failure.rejects_candidate else CompilationError
            raise kind(message + failure.render(), failure=failure, **context)
        raise CompilationError(message + result.stdout + result.stderr, **context)
    return result.stdout


def file_identity(path: str | Path) -> dict:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), "device": stat.st_dev, "inode": stat.st_ino,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def executable(name: str, purpose: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise CompilationError(f"{purpose} is unavailable: {name!r}")
    return str(Path(path).absolute())


def installed_tool(name: str) -> Path | None:
    local = Path(__file__).resolve().parents[1] / "_native" / "bin" / name
    if local.is_file():
        return local
    try:
        distribution = importlib.metadata.distribution("weft-riscv")
    except importlib.metadata.PackageNotFoundError:
        return None
    installed = Path(distribution.locate_file(f"weft/_native/bin/{name}"))
    return installed if installed.is_file() else None


@dataclass(frozen=True)
class Toolchain:
    """Tools for building artifacts; loading an existing library needs no C toolchain."""

    compiler: str | None = None
    cc: tuple[str, ...] = ("clang",)
    cflags: tuple[str, ...] = ()
    ldflags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("cc", "cflags", "ldflags", "dependencies"):
            value = getattr(self, name)
            if isinstance(value, str) or any(not isinstance(item, str) or not item for item in value):
                raise TypeError(f"{name} must be a sequence of nonempty strings")
            object.__setattr__(self, name, tuple(value))
        if not self.cc:
            raise ValueError("the C compiler command cannot be empty")

    @classmethod
    def from_environment(cls) -> Toolchain:
        return cls(compiler=os.environ.get("WEFT_COMPILER"),
                   cc=tuple(shlex.split(os.environ.get("CC", "clang"))),
                   cflags=tuple(shlex.split(os.environ.get("CFLAGS", ""))),
                   ldflags=tuple(shlex.split(os.environ.get("LDFLAGS", ""))))

    def compiler_path(self) -> str:
        if self.compiler is not None:
            return executable(self.compiler, "Weft compiler")
        installed = installed_tool("weft-compile")
        if installed is not None:
            return executable(str(installed), "installed Weft compiler")
        return executable("weft-compile", "Weft compiler; install a compiler wheel or set WEFT_COMPILER")

    def resolve(self, *, native_library: bool = True) -> Toolchain:
        cc = (executable(self.cc[0], "C compiler"), *self.cc[1:]) if native_library else self.cc
        return Toolchain(self.compiler_path(), cc, self.cflags, self.ldflags, self.dependencies)

    def identity(self, *, native_library: bool) -> dict:
        compiler = self.compiler_path()
        dependencies = [file_identity(path) for path in self.dependencies]
        libdir = Path(compiler).parent.parent / "lib"
        if libdir.is_dir():
            dependencies.extend(file_identity(path) for path in sorted(libdir.glob("*.so*"))
                                if path.is_file())
        result = {"compiler": file_identity(compiler),
                  "compiler_version": run_compiler([compiler, "--version"], stage="compiler identity"),
                  "dependencies": dependencies}
        if native_library:
            result.update(cc=list(self.cc), cflags=list(self.cflags), ldflags=list(self.ldflags),
                          library_flags=list(LIBRARY_FLAGS),
                          cc_file=file_identity(self.cc[0]),
                          cc_version=run_compiler([*self.cc, "--version"], stage="C toolchain identity"),
                          environment={name: os.environ.get(name) for name in
                                       ("CPATH", "C_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
                                        "GCC_EXEC_PREFIX", "COMPILER_PATH", "SDKROOT")})
        return result

    def link_library(self, source: Path, output: Path, target) -> list[str]:
        dispatch = source.with_name("dispatch.c")
        if not dispatch.is_file():
            raise CompilationError("artifact has no typed native dispatch source")
        command = [*self.cc, *self.cflags, *LIBRARY_FLAGS, f"-march={target.march}",
                   f"-mabi={target.abi}", str(source), str(dispatch), "-lm", *self.ldflags, "-o", str(output)]
        run_compiler(command, stage=f"native library build for {target.march}")
        return command
