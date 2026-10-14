from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .artifact import Artifact
from .driver import Toolchain, executable, file_identity, run_compiler
from .target import NativeTarget
from .tuning import ValidationError


@dataclass(frozen=True)
class Deployment:
    """An explicit SSH endpoint for running an already selected program."""

    host: str
    python: str
    workspace: str
    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.host or self.host.startswith("-") or not self.python or not self.workspace.startswith("/"):
            raise ValueError("deployment requires an SSH host, Python executable and absolute workspace")
        if not self.cpus or any(type(cpu) is not int or cpu < 0 for cpu in self.cpus):
            raise ValueError("deployment requires a nonempty CPU affinity")
        object.__setattr__(self, "cpus", tuple(sorted(set(self.cpus))))

    def command(self, arguments: list[str], *, input: str | None = None) -> str:
        return run_compiler(["ssh", self.host, shlex.join(arguments)], input, stage=f"remote deployment on {self.host}")

    def describe(self, *, toolchain: Toolchain, cxx: tuple[str, ...], dependencies=()) -> dict:
        """Read native target/tool identities under the declared execution affinity."""
        request = {"cpus": self.cpus, "compiler": toolchain.compiler, "cc": toolchain.cc,
                   "cflags": toolchain.cflags, "ldflags": toolchain.ldflags,
                   "dependencies": [*toolchain.dependencies, *dependencies], "cxx": cxx}
        code = ("import json,sys; from weft.runtime.deployment import describe_request; "
                "print(json.dumps(describe_request(json.load(sys.stdin))))")
        return {"host": self.host, "native": json.loads(self.command([self.python, "-c", code], input=json.dumps(request)))}

    def prepare(self, artifact: Artifact, runtime: Path, *, toolchain: Toolchain,
            cxx: tuple[str, ...], cxxflags: tuple[str, ...], link_flags: tuple[str, ...],
            arguments: tuple[str, ...], keep: bool = False) -> DeployedCall:
        """Compile once, then reuse the actual C++ driver's numerical/timing entry.

        The driver retains its own reference algorithm and timing boundaries.
        Hardware facts are checked after affinity is set on the receiving host.
        """
        create = ("import pathlib,tempfile,sys; p=pathlib.Path(sys.argv[1]); "
                  "p.mkdir(parents=True,exist_ok=True); print(tempfile.mkdtemp(prefix='run-',dir=p))")
        remote = self.command([self.python, "-c", create, self.workspace]).strip()
        remote_path = Path(remote)
        if remote_path.parent != Path(self.workspace) or not remote_path.name.startswith("run-"):
            raise RuntimeError(f"unexpected remote staging path: {remote!r}")
        try:
            with tempfile.TemporaryDirectory(prefix="weft-deploy-") as temporary:
                staging = Path(temporary)
                artifact.save(staging / "artifact")
                shutil.copy2(runtime, staging / "runtime.cpp")
                request = {"cpus": self.cpus, "cc": toolchain.cc, "cflags": toolchain.cflags,
                           "ldflags": toolchain.ldflags, "compiler": toolchain.compiler,
                           "cxx": cxx, "cxxflags": cxxflags, "link_flags": link_flags,
                           "arguments": arguments, "assembly": keep}
                (staging / "request.json").write_text(json.dumps(request), encoding="utf-8")
                # Tar transports only this freshly created bundle, avoiding SCP
                # remote-shell pathname interpretation and preserving filenames.
                archive = subprocess.Popen(["tar", "-C", str(staging), "-cf", "-", "."], stdout=subprocess.PIPE)
                receiver = subprocess.run(["ssh", self.host, shlex.join(["tar", "-C", remote, "-xf", "-"])],
                                          stdin=archive.stdout, capture_output=True, text=True)
                archive.stdout.close()
                sender_status = archive.wait()
                if sender_status or receiver.returncode:
                    raise RuntimeError(f"artifact transfer failed: sender={sender_status}, "
                                       f"receiver={receiver.returncode}: {receiver.stderr}")
            self.command([self.python, "-m", "weft.runtime.deployment", "build", str(remote_path / "request.json")])
        except BaseException:
            self._release(remote_path, keep)
            raise
        return DeployedCall(self, remote_path, keep)

    def _release(self, remote: Path, keep: bool):
        if keep:
            print(f"WEFT_REMOTE_ARTIFACTS={self.host}:{remote}", file=sys.stderr)
        else:
            self.command([self.python, "-c", "import shutil,sys; shutil.rmtree(sys.argv[1])", str(remote)])

    def run(self, artifact: Artifact, runtime: Path, **options) -> str:
        with self.prepare(artifact, runtime, **options) as call:
            return call()


class DeployedCall:
    def __init__(self, deployment: Deployment, directory: Path, keep: bool):
        self.deployment = deployment
        self.directory = directory
        self.keep = keep
        self.closed = False
        self._lock = threading.RLock()

    def __call__(self) -> str:
        with self._lock:
            if self.closed:
                raise RuntimeError("deployed numerical driver is closed")
            encoded = self.deployment.command([self.deployment.python, "-m", "weft.runtime.deployment",
                                               "run", str(self.directory / "request.json")])
            result = json.loads(encoded)
            if result["returncode"]:
                raise ValidationError(f"native numerical driver exited with {result['returncode']}\n"
                                      f"{result['stdout']}{result['stderr']}")
            if result["stderr"]:
                print(result["stderr"], end="", file=sys.stderr)
            return result["stdout"]

    def close(self):
        with self._lock:
            if not self.closed:
                self.deployment._release(self.directory, self.keep)
                self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("deployed numerical driver is closed")
        return self

    def __exit__(self, *exception):
        self.close()


def _affinity(request):
    cpus = set(request["cpus"])
    if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
        raise ValueError("requested deployment CPUs exceed the available affinity")
    os.sched_setaffinity(0, cpus)


def describe_request(request) -> dict:
    _affinity(request)
    toolchain = Toolchain(request["compiler"], tuple(request["cc"]), tuple(request["cflags"]),
                          tuple(request["ldflags"]), tuple(request["dependencies"])).resolve()
    cxx = (executable(request["cxx"][0], "C++ compiler"), *request["cxx"][1:])
    return {"target": NativeTarget.discover(toolchain).to_dict(), "host": os.uname().nodename,
            "toolchain": toolchain.identity(native_library=True),
            "cxx": file_identity(cxx[0]),
            "cxx_version": run_compiler([*cxx, "--version"], stage="C++ toolchain identity")}


def _request(path):
    directory = path.resolve().parent
    request = json.loads(path.read_text(encoding="utf-8"))
    _affinity(request)
    artifact = Artifact(directory / "artifact")
    toolchain = Toolchain(request["compiler"], tuple(request["cc"]),
                          tuple(request["cflags"]), tuple(request["ldflags"])).resolve()
    target = NativeTarget.discover(toolchain)
    target.require(artifact.target, artifact.metadata["used_extensions"], artifact.metadata["isa_extensions"])
    return directory, request, artifact, toolchain


def build_request(path: Path) -> None:
    directory, request, artifact, toolchain = _request(path)
    cflags = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-ffp-contract=fast",
              *toolchain.cflags, f"-march={artifact.target.march}", f"-mabi={artifact.target.abi}"]
    source = directory / "artifact" / "kernel.c"
    if request["assembly"]:
        run_compiler([*toolchain.cc, *cflags, "-S", str(source), "-o", str(directory / "kernel.s")],
                     stage="deployment assembly emission")
    run_compiler([*toolchain.cc, *cflags, "-c", str(source), "-o", str(directory / "kernel.o")],
                 stage="deployment object compilation")
    command = [*request["cxx"], "-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror",
               "-Wno-unused-const-variable", "-ffp-contract=fast", *toolchain.cflags,
               f"-march={artifact.target.march}", f"-mabi={artifact.target.abi}",
               *request["cxxflags"], str(directory / "runtime.cpp"), str(directory / "kernel.o"),
               *toolchain.ldflags, *request["link_flags"], "-o", str(directory / "runtime")]
    run_compiler(command, stage="deployment numerical driver link")


def run_request(path: Path) -> dict:
    directory, request, _, _ = _request(path)
    result = subprocess.run([str(directory / "runtime"), *request["arguments"]],
                            capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one transferred Weft artifact with its numerical driver")
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("request", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "build":
        build_request(arguments.request)
    else:
        print(json.dumps(run_request(arguments.request)))


if __name__ == "__main__":
    main()
