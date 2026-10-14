# Quick start

Complete [installation](installation.md) first. Native sections run on a RISC-V host;
source-only compilation also works on a development host.

## Use an installed operator

```python
import numpy as np
from weft.programs import Operators, programs

print([(entry.name, entry.family) for entry in programs()])
x = np.linspace(-1, 1, 387, dtype=np.float32).reshape(3, 129)
weight = np.ones(129, dtype=np.float32)
with Operators() as operators:
    y = operators.rms_norm(x, weight)
expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
np.testing.assert_allclose(y, expected, rtol=2e-3, atol=1e-4)
```

Weft borrows supported buffers without silently converting or repacking them.
See [program contracts](../dsl/programs.md) for shapes, aliasing and state.

## Write a kernel

Save this definition as `affine.py`, so the DSL can inspect its source:

```python
from __future__ import annotations

import weft
import weft.language as wl

@weft.kernel
def affine(X: wl.View[wl.f32, (N,)], Y: wl.View[wl.f32, (N,)],
           scale: wl.f32 = 1.0, *, bias: wl.f32 = 0.0):
    with wl.level.blocks(N, extent=wl.auto("B")) as block:
        value = wl.load(X[block])
        wl.store(Y[block], value * scale + bias)
```

`N` is inferred from views; `B` is an explicit source binding. The ordered block
traversal belongs to the author; its RVV representation belongs to the compiler.
A definition is not itself an executable Python function.

Run a separate script in the same directory:

```python
import numpy as np
import weft
from affine import affine
from weft.runtime import CompileOptions

x = np.linspace(-1, 1, 129, dtype=np.float32)
y = np.empty_like(x)
with weft.jit(affine, options=CompileOptions(meta={"B": 64})) as kernel:
    kernel(x, y, scale=1.25, bias=-0.5)
    np.testing.assert_allclose(y, x * 1.25 - 0.5, rtol=2e-3, atol=1e-4)
    call = kernel.prepare(x, y, scale=1.25, bias=-0.5)
    call()
    call.rebind(bias=0.5)()
    saved = kernel.save("/tmp/my-affine-artifact")

with weft.load(saved.path) as loaded:
    loaded(x, y, scale=1.25, bias=-0.5)
```

Prepared calls retain buffer owners. Rebinding creates a new call and preserves
the old one. `kernel.run(x, scale=..., bias=...)` can allocate declared write-only
output. Mutable state and workspace remain caller-owned.

## Inspect without a RISC-V machine

```sh
weft signature affine.py:affine
weft build --kernel affine.py:affine --meta B=64 \
  --march rv64imafdcv --abi lp64d --vlen-bits 128 \
  --source-only --output /tmp/my-affine-source
weft inspect /tmp/my-affine-source --emit header
weft inspect /tmp/my-affine-source --emit operations --format text
weft inspect /tmp/my-affine-source --emit resources
```

This saves the two IRs, C and typed ABI without invoking a RISC-V C compiler or
executing the kernel. On the target, `weft link` explicitly produces a native
artifact. See `weft link --help` for toolchain flags. Saved directories must be new.

## Tune a finite set of implementations

Continue the native script with `affine`, `x` and `y` defined above:

```python
from weft.runtime import Config, TuningPolicy

def validate(original, trial):
    expected = original["X"] * original["scale"] + original["bias"]
    np.testing.assert_allclose(trial["Y"], expected, rtol=2e-3, atol=1e-4)

configs = [Config(name="b64-m1", meta={"B": 64}, lmul_eighths=8),
           Config(name="b128-m2", meta={"B": 128}, lmul_eighths=16)]
with weft.autotune(affine, configs, validator=validate,
                  validation_key="affine-f32-rtol2e-3-atol1e-4",
                  policy=TuningPolicy(max_candidates=2)) as tuned:
    result = tuned.tune(x, y, scale=1.25, bias=-0.5)
    print(result.render())
    tuned(x, y, scale=1.25, bias=-0.5)
```

The tuner compiles legal candidates, validates private trial storage, measures
them and persists the selection. It does not invent another algorithm. Workload
keys include shape, layout, target and toolchain facts; use `key=(...)` for scalar
values that change work. Failed candidates retain diagnostics. See the
[runtime contract](../compiler/runtime.md#公共调优).

For broader numerical exercises in a checkout, use
`python examples/use.py --family normalization`. Other families cover attention,
models and prepared invocation. They require native hardware and the installed
package. Correctness exercises do not establish model throughput.
