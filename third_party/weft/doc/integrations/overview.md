# 应用接入

用户通过 Triton 编译、调优与调用 Weft 模型。Python、C/C++ 和 llama.cpp 共享同一份
编译 kernel，调用入口的区别只在参数绑定、实例启动和所属线程框架。

| 使用方式 | 入口 | 调用边界 |
|---|---|---|
| NumPy / CPU Tensor | `kernel[grid](...)` | 借用现有存储，按显式 grid 启动 |
| 调优 | triton.autotune / triton.Config | 有限 source/physical 配置，私有 trial state |
| Python AOT | CompiledKernel.export / triton.load | 不需要原 DSL 或现场编译器 |
| C/C++ | 导出的 typed header 和 native SDK | 单实例或显式 grid，资源由 host 管理 |
| llama.cpp | triton.Package + 显式 GGML role | 既有 worker 调用单实例，不重复分块/量化/线程团队 |
| PyTorch 用户 kernel | Triton HOP 的模型接缝 | 同一 JITFunction、准确 effects、明确 backward |
| PyTorch 自动图 | Inductor 的 CPU Triton 调度与代码生成 | 自动生成的普通 Triton kernel 使用同一编译和启动链 |

## CPU Tensor 与训练

Tensor 保留 dtype、shape、stride、storage offset 和底层 storage 关系。
View 参数的动态 stride 在 JIT 装饰器中声明；普通 pointer kernel 则通过显式 scalar 参数
计算元素偏移。没有隐式 contiguous、copy 或 repack。

source/weft/dense/torch_affine.py 给出普通 nn.Module 与 torch.autograd.Function，
forward/backward 都调用同模型 Triton kernel。梯度程序由作者显式写出。
模型源码中的 normalization、attention、MoE 和 Mamba 是普通作者程序；
host composition 保有参数、cache、workspace 和状态更新顺序。

图捕获中的 FakeTensor 只能读取元数据。Canonical interface 分析提供参数读写、
alias、逻辑 shape 和 stride 条件，不读取 Tensor bytes、不探测设备、不启动 Clang。
PyTorch 必须显式接入这个分析与模型启动接口；不能用 TTIR 分析失败的保守 fallback
代替模型支持。相关实现位于 Triton 内部框架模块，不要求用户注册 weft dispatcher 算子。

安装 PyTorch 源码接缝后，普通 Torch 程序可选择已有的 CPU Triton backend：

```python
compiled = torch.compile(lambda x: x * 1.25 - 0.5,
                         backend="inductor", options={"cpu_backend": "triton"})
result = compiled(cpu_tensor)
```

Inductor 负责图分解、调度、融合和 Triton 源码；CPU 编译接缝选择 Weft 模型及目标，
沿用原 CachingAutotuner 的配置枚举和 grid 表达式。生成代码、Canonical IR 和 native
产物可用于确认某项计算走了哪条路线。框架自身选择的外部算子不计为 Weft 编译覆盖。

自动生成的非 persistent 归约在同一配置列表中加入按 VLEN 计算的有限 CPU 流式
分块候选，仍由 Inductor 去重、应用最小分块约束、缓存和实测选择。显式用户配置、
persistent 归约和要求固定数值策略的配置不扩展。原候选保留，超出真实寄存器或
局部存储预算的候选由编译器拒绝；不能提高预算或把编译错误当作调优失败吞掉。

## Native 与模型线程

```python
package = triton.Package.create(
    "/application/kernels/model",
    {"projection": compiled_dot},
    bindings={"ggml.vec_dot": {"q4_0": "projection"}},
)
```

包只保存已经选定并编译的 native kernel，不重建数学程序。
GGML 角色检查真实 Encoding、shape、stride、参数顺序和已使用扩展。
vec-dot、activation quantizer 和 IME packed row tile 是不同合同，不能互相替代。
实际 llama.cpp 命中由模型测量窗口的调用计数确认，prefill/decode 分开记录。

部署细节见 [native SDK](../compiler/native-sdk.md)，
JIT、缓存、调优与 Python AOT 见[运行时参考](../compiler/runtime.md)。
