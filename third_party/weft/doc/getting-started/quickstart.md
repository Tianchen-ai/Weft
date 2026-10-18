# 从 Triton 编写 CPU kernel

安装扩展后的 Triton 后，入口仍是 `import triton`。普通指针、`tl.arange`、广播、
masked load/store 和 `tl.dot(a, b, acc)` 保留 Triton 的调用方式。下面程序在 RISC-V
本机 JIT，实例的数据分区由源码表达：

```python
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def affine(X, Y, N, scale, bias, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    active = offsets < N
    values = tl.load(X + offsets, mask=active, other=0.0)
    tl.store(Y + offsets, values * scale + bias, mask=active)

x = np.arange(4097, dtype=np.float32)
y = np.empty_like(x)
compiled = affine[(triton.cdiv(x.size, 64),)](
    x, y, x.size, 1.25, -0.5, B=64, num_cpu_threads=2)
print(y[:8])
```

将定义保存为 Python 文件运行；JIT 需要函数源码。`B` 是编译期参数，`N`、scale、bias
是运行时标量。grid 表示 program instances 的数量，线程数独立控制 OpenMP 调度。
mask 为假的元素不发生访存。普通指针的 stride 由源码显式参与元素偏移，runtime 不复制
或自动变成连续 Tensor。

```python
print(compiled.asm["weftir"])
print(compiled.asm["riscvir"])
print(compiled.asm["c"])
compiled.export("/application/kernels/affine")
loaded = triton.load("/application/kernels/affine")
loaded[(triton.cdiv(x.size, 64),)](x, y, x.size, 1.25, -0.5)
```

导出已固定 B 和物理配置，加载只传 runtime 参数。CPU Tensor 也可直接作为 X、Y。
`View`、Encoding、Level 和 state/stage 是同一模型中的可选扩展。例如 `tl.View[tl.f32, (N,)]`
可将逻辑 shape 绑定为 N；它的动态 stride 使用装饰器 `strided=("X", "Y")` 声明。
这些扩展及其生命周期见[语言示例](../dsl/examples.md)。普通 helper 可用 `@triton.jit`
定义，并随调用者的模型编译。

从仓库根目录运行现有调用示例：

```sh
PYTHONPATH=third_party/weft python -m examples.use --size 4097 --threads 2
```

性能、数值覆盖和 llama.cpp 测量使用独立 Weft Bench，结果放在其输出目录。
完整调优、ABI 与工具链合同见[运行时参考](../compiler/runtime.md)。
