# Triton–Weft

[English](README.md) · [工具书](https://tianchen-ai.github.io/Weft/) ·
[示例指南](third_party/weft/doc/getting-started/examples.md) ·
[Weft Bench](https://github.com/Tianchen-ai/Weft-Bench)

Triton–Weft 在 Triton 中加入面向 RISC-V CPU 的编程模型与编译路线。
普通程序继续使用 `tl.load/store`、广播、归约、`tl.dot`、scan 和 atomic；
需要精确表达量化存储、有限位宽和跨层复用时，可以组合 Encoding、View、Level、state/stage。
同一前端形成 Canonical IR，再由 RISC-V 物理 passes 选择 RVV 与支持的 IME 局部指令。

用户使用同一份 `import triton`、JIT、autotune、编译缓存与原生产物。
CPU Tensor 可以直接传入；导出的 native 库可由现有 C/C++ 线程池调用，也可用于 llama.cpp。
上游 Triton 的 GPU 路线保留。x86/Intel AMX 与 TileLang 是后续独立后端和前端工作。

## 开始使用

本项目是一份定制 Triton，普通 `pip install triton` 不包含 Weft 扩展。
请按[安装指南](third_party/weft/doc/getting-started/installation.md)在独立环境构建，
使用匹配的 LLVM/MLIR SDK、Clang 与目标 ABI。一个环境只安装一份提供 `triton` 模块的发行包。

```python
import numpy as np
import triton
import triton.language as tl

@triton.jit(model="weft")
def affine(X, Y, N, scale, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    active = offsets < N
    x = tl.load(X + offsets, mask=active, other=0.0)
    tl.store(Y + offsets, x * scale, mask=active)

x = np.arange(131, dtype=np.float32)
y = np.empty_like(x)
affine[(triton.cdiv(x.size, 64),)](x, y, x.size, 1.25, B=64, num_cpu_threads=2)
print(y[:8])
```

将程序保存为 Python 文件，在支持的 RISC-V 主机上运行。
grid 表达独立实例的数据分区，`num_cpu_threads` 控制 OpenMP 线程数；两者各自独立。
mask 为假的元素不发生访存。`CompiledKernel.asm` 可检查语义 IR、物理 IR、C 和原生库。

从仓库根目录运行正式示例：

```sh
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation softmax
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation matmul
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation scan
PYTHONPATH=third_party/weft python -m examples.triton_basics --operation histogram
PYTHONPATH=third_party/weft python -m examples.autotune_export /path/to/new/artifact
```

PyTorch 图接入先按安装指南运行一次 `python -m triton.tools.pytorch`，
再使用正常 Triton kernel 调用及 `torch.compile(..., options={"cpu_backend": "triton"})`。
完整用法见[示例指南](third_party/weft/doc/getting-started/examples.md)和[框架接入](third_party/weft/doc/integrations/overview.md)。

## 文档与开发

- [工具书入口](third_party/weft/doc/index.md)：安装、语言、编译器、应用与扩展。
- [Encoding 与存储](third_party/weft/doc/dsl/encoding-and-artifacts.md)。
- [Level 与生命周期](third_party/weft/doc/dsl/values-and-levels.md)。
- [JIT、调优、缓存和 AOT](third_party/weft/doc/compiler/runtime.md)。
- [原生 SDK 与 llama.cpp](third_party/weft/doc/compiler/native-sdk.md)。
- [RISC-V 扩展合同](third_party/weft/doc/development/extensions.md)。
- [贡献指南](CONTRIBUTING.md)。

编译器、运行时和使用示例在本仓库；数值检查、benchmark、设备配置和实验输出由独立 Weft Bench 管理。

## 实测模型性能

![K1 与 SG2044 的 llama.cpp 两阶段加速比](third_party/weft/doc/assets/llama-performance.svg)

图中比较正常 llama.cpp 与加载原生替换包后的真实模型执行，prefill/decode 分开计时。
K1 DeepSeek 8B decode 为 **1.514×**；SG2044 Llama2 decode 为 **1.005×**，当前替换集合在该阶段基本持平。
这些数字没有叠加单算子倍率，也不代表所有模型都达到同样收益。

模型为 Llama2 7B Q4_0 与 DeepSeek-R1-Distill-Llama 8B Q4_K_M。
Prefill 为 128 tokens；decode 为 depth=512 时生成 32 tokens；K1 使用 8 线程，SG2044 使用 64 线程，
每项取 3 次计时中位数。编译和加载不计入窗口；实际调用计数确认替换命中。
数值检查使用独立的 16-token 固定 prompt，比较最后一个 prefill token 与两个 teacher-forced decode token 的全词表 logits，
使用 `atol=rtol=1e-3`。复现由 [Weft Bench](https://github.com/Tianchen-ai/Weft-Bench) 提供。
固定 llama.cpp 构建使用 Clang 18，Weft 原生替换库使用 Clang 23；这里的应用收益包含这项工具链选择。

## 来源与许可证

Triton 基于上游 3.8.0，保留 MIT 许可证；Weft 编译器、语言与运行时保留 Apache-2.0。
[NOTICE](NOTICE) 列出组件归属和第三方许可。本扩展由本仓库维护。
