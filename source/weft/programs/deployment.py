"""Named model-program artifacts using the public native Package format."""
from __future__ import annotations

from threading import RLock

from weft.runtime import Package, PackageBuilder, TuningResult
from weft.runtime.compiler import KernelSource
from weft.runtime.signature import CallSignature

from .catalog import get
from .operators import Operators
from .tuning import selected_artifact


def role(name, bindings):
    entry = get(name)
    missing = set(entry.required) - bindings.keys()
    if missing:
        raise ValueError(f"{name} requires ABI specialization {sorted(missing)}")
    return name + "".join(f"/{key}={bindings[key]}" for key in sorted(entry.required))


class ProgramPackageBuilder:
    """Choose author programs and complete configurations for offline deployment.

    The produced package is also consumable by the C/C++ SDK. It owns code,
    never model weights, state or workspaces. Loop ordering remains the model's
    host composition, and build failures do not select another implementation.
    """

    def __init__(self, target):
        self.builder = PackageBuilder(target)
        self.roles = {}

    def add(self, name, *, options=None, **bindings):
        entry = get(name)
        selected = entry.options(options=options, **bindings)
        identity = role(name, dict(selected.meta))
        if identity in self.roles:
            raise ValueError(f"package already contains {identity}")
        symbol = name + "".join(f"_{key.lower()}{selected.meta[key]}" for key in sorted(entry.required))
        self.builder.add(symbol, entry.definition, options=selected)
        self.builder.bind("weft.programs", identity, symbol)
        self.roles[identity] = symbol
        return self

    def add_model(self, model, *, selections=None):
        """Capture the exact kernel requirements declared by a model composition."""
        if isinstance(model.operators, LoadedOperators):
            raise TypeError("a loaded model already owns a selected package; use save_operators to export it")
        selected = dict(selections or {})
        consumed, pending, identities = set(), [], set(self.roles)
        for name, bindings in model.required_programs.items():
            identity = role(name, bindings)
            if identity in identities:
                raise ValueError(f"package already contains {identity}")
            identities.add(identity)
            if name in model.operators.tuning:
                kernel = model.operators.kernel(name, **bindings)
                artifact = selected_artifact(kernel, selected.get(identity))
                selection = next(result for result in kernel.selections if result.winner.options == artifact.options)
                source = KernelSource.capture(get(name).definition)
                selection.check_program(source.canonical, self.builder.target)
                options = artifact.options
                if identity in selected:
                    consumed.add(identity)
            else:
                if identity in selected:
                    raise ValueError(f"fixed program {identity} does not accept a tuning selection")
                options = get(name).options(options=model.operators.options.get(name), **bindings)
            pending.append((name, options))
        unknown = selected.keys() - consumed
        if unknown:
            raise ValueError(f"selections name unrequested model roles: {sorted(unknown)}")
        # Resolve all choices first. A missing/ambiguous workload selection must
        # not leave half of a model captured in an otherwise reusable builder.
        for name, options in pending:
            self.add(name, options=options)
        return self

    def add_selection(self, name, selection):
        """Build one measured binding, checking its canonical program and target."""
        if not isinstance(selection, TuningResult):
            raise TypeError("add_selection requires a TuningResult")
        source = KernelSource.capture(get(name).definition)
        selection.check_program(source.canonical, self.builder.target)
        return self.add(name, options=selection.winner.options)

    def build(self, directory, *, toolchain=None, native_library=True, cache=None):
        return self.builder.build(directory, toolchain=toolchain,
                                   native_library=native_library, cache=cache)


class LoadedOperators(Operators):
    """Run model compositions from already compiled artifacts with no JIT path."""

    def __init__(self, package, *, target=None):
        # Deliberately do not construct a default Toolchain or call the JIT
        # initializer. The saved package is this object's entire authority.
        self.package = package if isinstance(package, Package) else Package(package)
        self.target = target
        self.options = {}
        self.tuning = {}
        self._kernels = {}
        self._lock = RLock()
        self._closed = False
        self._roles = dict(self.package.bindings.get("weft.programs", {}))
        if not self._roles:
            raise ValueError("package has no weft.programs bindings")
        self._verified = {}
        for identity, symbol in self._roles.items():
            if symbol not in self.package.kernels:
                raise ValueError(f"binding {identity} names missing artifact {symbol}")
            name = identity.split("/", 1)[0]
            entry = get(name)
            artifact = self.package.kernels[symbol]
            bindings = dict(artifact.manifest["options"]["meta"])
            if identity != role(name, bindings):
                raise ValueError(f"package role {identity} disagrees with its artifact specialization")
            selected = entry.options(options=artifact.options)
            if dict(selected.meta) != bindings:
                raise ValueError(f"package artifact {symbol} has incomplete source bindings")
            source = KernelSource.capture(entry.definition)
            expected = source.interface(bindings).semantic_dict()
            actual = CallSignature(artifact.manifest["kernel"], artifact.manifest["signature"],
                                   bindings=bindings).semantic_dict()
            if actual != expected:
                raise ValueError(f"package artifact {symbol} has an incompatible public program ABI")
            self._verified[identity] = artifact

    def _artifact(self, name, bindings):
        if self._closed:
            raise RuntimeError("LoadedOperators is closed")
        identity = role(name, bindings)
        if identity not in self._verified:
            raise NotImplementedError(f"package does not contain {identity}")
        artifact = self._verified[identity]
        for key, value in bindings.items():
            if key not in artifact.options.meta or artifact.options.meta[key] != value:
                raise ValueError("a loaded artifact cannot change its selected source bindings")
        return identity, artifact

    def kernel(self, name, **bindings):
        with self._lock:
            identity, artifact = self._artifact(name, bindings)
            if identity not in self._kernels:
                self._kernels[identity] = artifact.load(target=self.target)
            return self._kernels[identity]

    def materialize(self, names, **bindings):
        return tuple(self.kernel(name, **bindings.get(name, {})) for name in names)

    @property
    def artifacts(self):
        return tuple(self._verified.values())

    def selected_options(self, name, *, selection=None, **bindings):
        if selection is not None:
            raise ValueError("a loaded package already fixes its configurations")
        with self._lock:
            return self._artifact(name, bindings)[1].options


def save_operators(operators, directory, *, selections=None):
    if isinstance(operators, LoadedOperators):
        if selections:
            raise ValueError("a loaded package already fixes its configurations")
        artifacts = {symbol: operators.package.kernels[symbol] for symbol in operators._roles.values()}
        return Package.create(directory, artifacts, bindings={"weft.programs": operators._roles})
    if not isinstance(operators, Operators):
        raise TypeError("save_operators requires Operators or LoadedOperators")
    with operators._lock:
        if operators._closed:
            raise RuntimeError("Operators is closed")
        return save_program_kernels(operators._kernels, directory, selections=selections)


def save_program_kernels(kernels, directory, *, selections=None):
    selected = dict(selections or {})
    artifacts, roles, consumed = {}, {}, set()
    for (name, bindings), kernel in kernels.items():
        identity = role(name, dict(bindings))
        artifact = selected_artifact(kernel, selected.get(identity))
        if identity != role(name, dict(artifact.options.meta)):
            raise ValueError("tuning changed a public program ABI specialization")
        if identity in roles:
            raise ValueError(f"more than one selected configuration for {identity}")
        symbol = "program_" + str(len(artifacts))
        artifacts[symbol] = artifact
        roles[identity] = symbol
        if identity in selected:
            consumed.add(identity)
    unknown = selected.keys() - consumed
    if unknown:
        raise ValueError(f"selections name unrequested package roles: {sorted(unknown)}")
    if not artifacts:
        raise ValueError("no materialized programs to save")
    return Package.create(directory, artifacts, bindings={"weft.programs": roles})
