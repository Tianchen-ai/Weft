# Triton 内部的 Weft 编译核心

本目录包含 Canonical Kernel IR、RISC-V Physical IR、目标 passes、intrinsic C/local asm
生成和 C/C++ native runtime。Python 语言、JIT、autotune、backend 与 Triton 一起构建安装。

公共入口是 `@triton.jit(model="weft")` 和 `kernel[grid](...)`，没有独立 Weft Python 发行包或 CLI。

从[工具书](doc/index.md)、[安装说明](doc/getting-started/installation.md)和
[native SDK](doc/compiler/native-sdk.md)开始。Weft 原代码保留 [Apache-2.0](LICENSE) 归属。
