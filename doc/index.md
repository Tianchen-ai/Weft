# Weft Handbook · 工具书

Weft 是面向 RISC-V 的数值 kernel 语言与编译器。作者通过 Python DSL 表达有序、有限位宽的
数值程序，编译器形成 RVV 或已支持扩展的物理实现，应用通过 Python、PyTorch 或 C/C++ 调用。

Weft compiles numerical programs into native RISC-V kernels. Start with the
English usage guides below; the detailed language and compiler specifications
are currently primarily in Chinese.

![Weft 编译与应用路径](assets/architecture.svg)

## 开始使用 · Getting started

- [安装](getting-started/installation.md)：源码、wheel、LLVM/MLIR SDK 和目标工具链。
- [快速上手](getting-started/quickstart.md)：运行 catalog、编写 kernel、JIT、调优和保存产物。
- [框架接入](integrations/overview.md)：PyTorch、模型 block、C/C++ SDK 和 llama.cpp。

## 语言与编程模型 · Language

先读[编程模型](programming-model/index.md)和[非目标](programming-model/non-goals.md)，
理解作者程序与目标编译器的职责；然后按需要查阅：

- [Encoding、View 与 Artifact](dsl/encoding-and-artifacts.md)：存储表示和调用边界。
- [Value、Level 与有序控制](dsl/values-and-levels.md)：逻辑值、生命周期和普通控制。
- [函数与 Operation](dsl/functions-and-operations.md)：标准库、有限 source 参数和 operation。
- [DSL 示例](dsl/examples.md)与[模型程序](dsl/programs.md)：数值程序和状态所有权。
- [Canonical Kernel IR](dsl/canonical-ir.md)：编译主链保存的作者语义。

## 编译器 · Compiler

[编译器总览](compiler/index.md)解释两层 IR：Canonical Kernel IR 保存作者程序，
[RISC-V Physical IR](compiler/riscv-ir.md)保存目标物理决定，然后直接生成 intrinsic C / local asm。

- [物理机器](compiler/physical-machine.md)：lane、time、replica、fragment 与局部存储。
- [Passes](compiler/passes.md)：分析、选择、物化和最终验证。
- [优化方法](compiler/optimization-principles.md)：动态工作量、数据供应、归约、流水和资源。
- [局部指令合同](compiler/leaves.md)与[代码生成](compiler/emission.md)。

## 运行时与接入 · Runtime

- [运行时参考](compiler/runtime.md)：调用 ABI、Tensor、JIT、调优、缓存和安装合同。
- [C/C++ 与 llama.cpp 参考](compiler/native-sdk.md)：具名包、生命周期和替换边界。

## 开发指南 · Development

- [扩展 RISC-V 支持](development/extensions.md)：能力、局部 op、选择、资源和执行权限。
- [数值与测量](development/validation.md)、[基线与归因](development/benchmarking.md)。
- [文档构建与发布](development/documentation.md)。
- [贡献指南](https://github.com/Tianchen-ai/Weft/blob/main/CONTRIBUTING.md)与
  [问题反馈](https://github.com/Tianchen-ai/Weft/issues)。

## 规格权威

本页链接的编程模型、语言、物理机器与编译器章节是当前设计规范。使用指南解释公开入口，
源码说明实现事实；运行结果和历史实现不能反向定义语言语义。程序 IR 始终只有 Canonical
Kernel IR 与 target-aware RISC-V Physical IR 两层。文档不保存开发流水账或历史性能成绩。
