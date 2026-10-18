# 模型程序与 host composition

作者程序位于 third_party/weft/source/weft/programs。每个 kernel 使用
`@triton.jit(model="weft")`，可直接通过 Triton 编译、调优、grid 启动和导出。
它们是公开源码示例和程序库，不是另一份独立 Python 编译服务。

| 程序族 | 作者表达的工作 |
|---|---|
| normalization | RMSNorm、中心方差 LayerNorm、residual 融合、行与显式 tile 变体 |
| activation | SiLU、SwiGLU、softplus、add、copy、fill、where |
| linear | dense contraction、bias、SiLU epilogue、显式嵌套 Level 变体 |
| attention | split/interleaved RoPE、KV 写入、online causal grouped-query attention |
| MoE | router top-k、稳定 token 分组、ragged expert projection、加权合并 |
| Mamba | causal depthwise convolution、selective state recurrence |

Encoding、数值类型、归约顺序、Level 的 state/stage 初始化与 handoff 都是作者程序。
编译器负责同一程序的布局、conversion、访存、流水与局部指令选择。
模型名称不参与 backend 路由，也不会用模板实现取代源码。

## 直接调用

在仓库根目录将 third_party/weft 加入 Python path：

```python
from source.weft.programs.normalization import rms_norm
rms_norm[(1,)](X, Weight, Y, epsilon=1e-5, BLOCK=32,
              backend_options={"lmul_eighths": 32})
```

这里 X/Y 是二维 View，Weight 是一维 View；BLOCK 是显式 constexpr。
源码内部已经包含完整行循环，单实例即可覆盖这份程序。
想用多个 program instances 时，应写明 program_id 数据分区，不能机械地增加 grid。

## 配置与应用组合

catalog 描述源码、source 参数默认值、必须固定的维度及显式物理默认配置。
这些是应用选择，compiler 不读取 catalog 来决定数学算法。

```python
from source.weft.programs import get, Operators
configuration = get("rms_norm").config(BLOCK=64)
with Operators(configs={"rms_norm": configuration}) as operators:
    result = operators.rms_norm(x, weight)
```

Operators 是普通 host composition：分配显式输出、检查应用 shape、按源码次序调用 Triton。
它不维护另一套 JIT/cache/tuner。固定配置是 triton.Config；有限调优候选通过
Operators(tuning={"name": [config1, config2]}) 交给 triton.autotune。
也可直接对原 JITFunction 使用 Triton 的装饰器和 API。

Program.compile 使用 triton.compile 返回同一 CompiledKernel，可显式指定交叉 target/toolchain。
Operators.materialize 可在应用写入前 AOT 编译固定程序要求；
自动调优需要实际输入，普通调用由 Triton 在私有 trial state 上完成选择。

## 模型状态与生命周期

SelfAttention、SwiGLU、TransformerBlock、MoE 和 Mamba1 的 host composition 留在同一源码目录。
参数、cache、workspace 和 persistent state 都由调用方持有，kernel 不私自保存模型数据。
KVCache/MambaState 记录已消费的位置；状态调用串行使用。
若调用中途失败，状态会失效，避免把部分更新误当作一个完整模型步骤。

MoE 的 token grouping、offsets、expert 执行与合并顺序由模型程序给出；
Mamba 的 convolution 与 scan state 在调用之间显式传递；
online attention 的递推和因果范围由源码声明。
target pass 不能把这些算法选择包装成硬件 lowering。

## Tensor、训练与部署

相同 kernel 和 host composition 可接收 CPU Tensor，借用原有存储。
训练需作者显式给出 backward。dense/torch_affine.py 展示 torch.autograd.Function
及同模型 Triton backward，不使用独立 Weft dispatcher 注册层。

部署选择已编译的 CompiledKernel，通过 triton.Package.create 组成具名集合；
模型的 host 调用顺序保持显式。Python AOT 使用 triton.load，
C/C++ 使用 typed header 和共同 native SDK。LLM 替换角色及计时由独立 Bench 管理。
