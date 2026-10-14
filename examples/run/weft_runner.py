from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from weft.cli import load_definition
from weft.runtime import Config, Target, Toolchain, TuningResult, ValidationError
from weft.runtime.compiler import KernelSource
from weft.runtime.deployment import Deployment
from weft.runtime.driver import file_identity

from kernel_configuration import PROJECT, configured_binding
from workloads import select


class DriverTrial:
    """The existing numerical driver's cold timing, excluding deployment/build."""

    def __init__(self, deployed, check_workload):
        self.deployed = deployed
        self.check_workload = check_workload
        self.last_measurement = None

    def _run(self):
        output = self.deployed()
        values = {}
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        if values.get("numeric") not in ("within-tolerance", "bit-exact", "exact"):
            raise ValidationError("numerical driver did not report a successful contract")
        units = {"cold_median_us": 1000, "cold_median_ms": 1000000, "median_us": 1000}
        times = [(name, float(values[name]) * scale) for name, scale in units.items() if name in values]
        if len(times) != 1 or not math.isfinite(times[0][1]) or times[0][1] <= 0:
            raise ValueError("driver must report one finite positive kernel median time")
        identity = {"shape": {name: values[name] for name in ("M", "N", "K") if name in values},
                    "input_policy": values.get("input_policy"), "time_metric": times[0][0],
                    "repetitions": values["repetitions"], "numeric": values["numeric"]}
        self.check_workload(identity)
        self.last_measurement = values
        return times[0][1]

    def validate(self):
        self._run()
        return True

    def measure_ns(self):
        # Each driver invocation initializes its own arrays and performs its
        # original numerical checks. It reports native timing, not SSH latency.
        return self._run()

    def close(self):
        self.deployed.close()


class NumericalWorkload:
    def __init__(self, runner: str, target_name: str, kernel: str, *, phase=None,
                 repetitions: int, targets: Path, keep=False):
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        self.machine = json.loads(Path(targets).read_text(encoding="utf-8"))[target_name]
        self.workload = select(runner, kernel, phase)
        self.target = Target(**self.machine["target"])
        self.deployment = Deployment(**self.machine["deployment"])
        self.tools = Toolchain(**self.machine["toolchain"])
        self.request = " ".join((target_name, self.workload.name,
                                 *([self.workload.phase] if runner == "mul-mat" else [])))
        self.runner = runner
        self.target_contract = {**self.target.to_dict(), "matrix_extension": self.workload.matrix_extension}
        self.binding = configured_binding(runner, self.request, self.target_contract)
        self.definition = load_definition(f"{PROJECT / self.binding['entry']}:{self.binding['symbol']}")
        self.source = KernelSource.capture(self.definition)
        self.compiler = Toolchain(compiler=os.environ.get("WEFT_COMPILER", str(PROJECT / "build/tools/weft-compile/weft-compile")))
        self.runtime = PROJECT / "examples/repro/weft" / self.workload.runtime
        self.arguments = tuple(([self.workload.phase] if self.workload.phase else []) + [str(repetitions)])
        self.keep = keep
        self._measurement_contract = None

    @property
    def default_config(self):
        return Config(name="catalog-binding", meta=self.binding["source"], **self.binding["physical"])

    def identity(self):
        dependencies = []
        if self.workload.ggml:
            directory = self.machine["ggml"]["libraries"]
            dependencies.extend(f"{directory}/lib{name}.so" for name in ("ggml", "ggml-cpu", "ggml-base"))
        return {"kind": "numerical-driver", "runner": self.runner, "request": self.request,
                "canonical": self.source.canonical, "driver": self.runtime.read_text(encoding="utf-8"),
                "machine": self.machine, "arguments": list(self.arguments),
                "compiler": self.compiler.resolve(native_library=False).identity(native_library=False),
                "adapter": file_identity(__file__),
                "execution": self.deployment.describe(toolchain=self.tools, cxx=tuple(self.machine["cxx"]),
                                                       dependencies=dependencies)}

    def _prepare(self, config):
        binding = configured_binding(self.runner, self.request, self.target_contract, config)
        artifact = self.source.build(target=self.target, options=config.options, native_library=False,
                                     toolchain=self.compiler)
        defines = [*self.machine["defines"], *self.workload.defines]
        if binding["symbol"] != self.workload.expected_symbol:
            defines.append(f"-D{self.workload.expected_symbol}={binding['symbol']}")
        link_flags = []
        if self.workload.ggml:
            source, libraries = self.machine["ggml"]["source"], self.machine["ggml"]["libraries"]
            defines.extend(f"-I{source}/{directory}" for directory in ("ggml/include", "ggml/src", "ggml/src/ggml-cpu"))
            link_flags = [f"-L{libraries}", f"-Wl,-rpath,{libraries}", "-Wl,--no-as-needed",
                          "-lggml", "-lggml-cpu", "-lggml-base", "-lgomp", "-lm", "-ldl", "-pthread"]
        return self.deployment.prepare(artifact, self.runtime, toolchain=self.tools,
                                       cxx=tuple(self.machine["cxx"]), cxxflags=tuple(defines),
                                       link_flags=tuple(link_flags), arguments=self.arguments, keep=self.keep)

    def run(self, config=None):
        config = self.default_config if config is None else config
        with self._prepare(config) as call:
            return call()

    def _check_measurement(self, identity):
        if self._measurement_contract is None:
            self._measurement_contract = identity
        elif identity != self._measurement_contract:
            raise ValueError("candidate changed the numerical driver's shape, input, timing or repetition contract")

    def trial(self, config):
        return DriverTrial(self._prepare(config), self._check_measurement)


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("runner", choices=("kernel", "mul-mat", "vec-dot", "row-dequantize"))
    parser.add_argument("target")
    parser.add_argument("kernel")
    parser.add_argument("arguments", nargs="+")
    parser.add_argument("--targets", type=Path, default=os.environ.get("WEFT_TARGETS"))
    parser.add_argument("--keep", action="store_true")
    return parser


def from_arguments(args, parser):
    expected = 2 if args.runner == "mul-mat" else 1
    if len(args.arguments) != expected:
        parser.error("mul-mat needs PHASE REPETITIONS; other runners need REPETITIONS")
    if args.targets is None:
        parser.error("set --targets or WEFT_TARGETS to the explicit machine configuration")
    return NumericalWorkload(args.runner, args.target, args.kernel,
                             phase=args.arguments[0] if args.runner == "mul-mat" else None,
                             repetitions=int(args.arguments[-1]), targets=args.targets, keep=args.keep)


def main():
    parser = parser_for("Build and run one existing Weft numerical workload")
    parser.add_argument("--config", type=Path, help="one serialized public Config")
    parser.add_argument("--selection", type=Path, help="use a public TuningResult winner as a fixed binding")
    args = parser.parse_args()
    if args.config is not None and args.selection is not None:
        parser.error("choose --config or --selection")
    workload = from_arguments(args, parser)
    config = workload.default_config
    if args.config:
        config = Config.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    if args.selection:
        result = TuningResult.load(args.selection)
        if result.identity["workload"] != workload.identity():
            raise ValueError("selection belongs to a different workload, compiler or native execution target")
        config = result.winner
    print("configuration=" + json.dumps(configured_binding(args.runner, workload.request,
                                                           workload.target_contract, config), sort_keys=True), flush=True)
    print(workload.run(config), end="")


if __name__ == "__main__":
    main()
