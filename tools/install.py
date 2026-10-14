from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, environment: dict | None = None) -> None:
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and install Weft into an isolated Python environment")
    parser.add_argument("--environment", type=Path, required=True, help="new or existing virtual environment")
    parser.add_argument("--llvm-root", type=Path, required=True, help="LLVM/MLIR SDK containing lib/cmake/{llvm,mlir}")
    parser.add_argument("--build-dir", type=Path, required=True, help="persistent compiler build directory")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--wheelhouse", type=Path, help="optional offline directory containing Python build dependencies")
    parser.add_argument("--cxx", help="C++ compiler used to build Weft itself")
    parser.add_argument("--cc", help="C compiler used to build Weft itself")
    arguments = parser.parse_args()
    if arguments.jobs <= 0:
        parser.error("--jobs must be positive")
    try:
        pip_version = importlib.metadata.version("pip")
    except importlib.metadata.PackageNotFoundError:
        parser.error("the invoking Python needs pip >= 23 to seed the isolated environment")
    if tuple(int(part) for part in pip_version.split(".")[:2]) < (23, 0):
        parser.error("the invoking Python needs pip >= 23; select a Python environment with an up-to-date pip")
    source = Path(__file__).resolve().parents[1]
    sdk = arguments.llvm_root.expanduser().resolve(strict=True)
    llvm = sdk / "lib/cmake/llvm"
    mlir = sdk / "lib/cmake/mlir"
    for path in (llvm / "LLVMConfig.cmake", mlir / "MLIRConfig.cmake"):
        if not path.is_file():
            parser.error(f"incomplete LLVM/MLIR SDK: missing {path}")
    environment = arguments.environment.expanduser().absolute()
    python = environment / "bin/python"
    if not (environment / "pyvenv.cfg").is_file():
        if environment.exists() and any(environment.iterdir()):
            parser.error("refusing to replace a nonempty directory that is not a virtual environment")
        # The invoking pip can seed an environment even on distributions that
        # separate ensurepip into an optional system package.
        run([sys.executable, "-m", "venv", "--without-pip", str(environment)])
    index = ["--no-index", "--find-links", str(arguments.wheelhouse.resolve())] if arguments.wheelhouse else []
    run([sys.executable, "-m", "pip", "--python", str(python), "install", *index,
         "pip>=23", "scikit-build-core>=0.11", "setuptools-scm>=8"])
    settings = [f"--config-settings=build-dir={arguments.build_dir.expanduser().absolute()}",
                f"--config-settings=cmake.define.LLVM_DIR={llvm}",
                f"--config-settings=cmake.define.MLIR_DIR={mlir}"]
    if arguments.cxx:
        settings.append(f"--config-settings=cmake.define.CMAKE_CXX_COMPILER={arguments.cxx}")
    if arguments.cc:
        settings.append(f"--config-settings=cmake.define.CMAKE_C_COMPILER={arguments.cc}")
    build_environment = {**os.environ, "CMAKE_BUILD_PARALLEL_LEVEL": str(arguments.jobs)}
    run([str(python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", *settings, str(source)],
        environment=build_environment)
    run([str(environment / "bin/weft-compile"), "--version"])
    run([str(environment / "bin/weft"), "--help"])
    print(f"Installed Weft: {python}; native kernel C toolchain is selected by CC/CFLAGS/LDFLAGS or Toolchain.")


if __name__ == "__main__":
    main()
