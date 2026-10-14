# 模型作者程序与公开调用

`weft.programs` 是安装包内的正式作者程序集合。唯一源码在
`source/weft/programs/`，wheel 将该目录安装为 `weft.programs`；示例不另存算法副本。
这些程序经普通 JIT/AOT 入口生成 Canonical Kernel IR、RISC-V Physical IR 和 native library。
模型名称只组织公开 API，不参与 compiler 的局部操作选择。

## 数值与形状

模型程序使用 dense f32，路由索引、计数与前缀和使用 exact i32。权重采用 `[out,in]`，
专家权重采用 `[expert,out,in]`。动态维度从 View 推导；正的 feature/head/state 维度是
调用合同，零 token 输入允许不执行数据计算。输入数值应有限，softmax 不定义全 `-inf` 行
或含 NaN 的排序；这些内容条件由调用方保证，不逐次扫描张量。

浮点遵守[测量协议](../development/validation.md)的既定绝对/相对容差，不要求归约 bit-exact。
LayerNorm 使用先求均值再求中心方差的算法。RMSNorm 使用平方均值。SiLU 和 softplus 使用
避免正方向指数溢出的表达。`epsilon` 必须有限且大于零。

`wl.log`、`wl.sqrt` 与 `wl.exp`、`wl.rsqrt` 一样是逐元素数值 operation；是否有对应 RVV
局部实现是 target lowering 的责任。没有向量 leaf 的数学函数不能冒充硬件向量指令。

## Catalog 与单算子

```python
from weft.programs import Operators, get, programs

for program in programs():
    print(program.name, program.family, program.description)

with Operators() as ops:
    y = ops.layer_norm(x, scale, bias, epsilon=1e-5)
    ops.rms_norm(x, scale, out=x)

kernel = get("rms_norm").jit(BLOCK=64)
kernel(X=x, Weight=scale, Y=output)

# 明确选择整行作者程序，N 是固定的 ABI specialization。
row_kernel = get("rms_norm_row").jit(N=x.shape[-1])
row_kernel(X=x, Weight=scale, Y=output)
```

`Operators` 负责 lazy code handle 及复用，不拥有调用者权重。输出可由调用方提供，或按输入
buffer family 分配。CPU Torch 直接借用 storage，NumPy 使用 buffer protocol；没有 dtype
转换或 `.contiguous()` 修复。合并前导维度需要可借用的 reshape，不可借用时报错。
完整模型会逐 batch 调用投影，因此可消费带 batch 间空洞的序列切片。

| 程序 | 主要合同 |
|---|---|
| `rms_norm`, `layer_norm`, `residual_rms_norm` | 行归约，最后维度正；支持非负 stride；同一 activation alias group 允许原地输出 |
| `rms_norm_row`, `layer_norm_row`, `softmax_row` | 显式固定 `N`；整行读取形成一个 Value，归约与输出复用该 Value；同样支持非负 stride 与原地输出 |
| `silu`, `swiglu`, `softplus`, `logarithm`, `exponential`, `add`, `copy` | `[M,N]`；逐元素程序；只有显式声明的 alias group 才允许重叠 |
| `softmax` | 固定行宽 `N`，完整行最大值、一次指数、完整行求和与标量倒数；`Operators.softmax` 从输入绑定行宽 |
| `fill` | `[M,N]`；`ROWS`/`BLOCK` 绑定完整初始化值，写入范围限于目标 View；`ops.fill(out, value)` 返回原输出对象 |
| `linear`, `linear_bias` | `[M,K] × [N,K] -> [M,N]`；单行走单输出 GEMV，多行以 `MC/NC` 外层分块和 `MR/NR` 输出 cohort 遍历；均使用完整 K 域的闭合 dot |
| `linear_tile`, `linear_bias_tile` | 显式 `MROWS/NROWS/BLOCK`，每个 K block 的 dot 在外层状态加法前收敛；与完整 K dot 是不同作者程序 |
| `rope_split`, `rope_interleaved` | 公开 `ops.rope` 输入 `[T,H,D]`，`D` 正且偶数；内部 View 分别为 `[T,H,2,D/2]`、`[T,H,D/2,2]` |
| `kv_update` | `[B,T,G,D]` 写入 `[B,S,G,D]` 的 `[start,start+T)`，区间外保留 |
| `causal_attention` | `H % G == 0`；Q `[B,T,H,D]`，KV `[B,S,G,D]`，第 t 行读取 `[0,start+t]` |
| `router` | `[T,E]` 选 K 个不同 expert，分数相同时优先较小编号；只对选中分数归一化 |
| `group_tokens`, `expert_linear`, `combine` | 稳定分组、ragged expert 区间和恢复 token 顺序；空 expert 合法 |
| `causal_conv1d`, `selective_scan` | 顺序消费 token；原地更新调用者的卷积历史/SSM state；scan 固定 state 宽度 `N`，各通道独立递推 |

`get(name).options()` 展示默认完整配置。普通逐元素输出块为 `BLOCK=32`；完整 K 投影的
多行投影的默认 source 参数为 `MC=64, NC=16, MR=4, NR=4`。Norm 的默认物理绑定为
`lmul_eighths=32`，完整 K 投影为 `64`；编译器在闭合 operation 内按实际数据载体和资源预算
选择合法 issue。单行分支没有跨输出复用，允许选择更长的 K 向量；该分支属于作者程序，
编译器不按 Linear 名称接管 kernel。显式传入 `CompileOptions` 时使用调用方给出的完整物理配置。
`expert_linear` 按 expert offsets 建立实际 token 子区域，以 `ROWS=4` 划分输出列，并对完整
K 域求 dot。动态 token 数由编译器在闭合 operation 内有界分块；空 expert 不创建 subview，
单 token 区间使用单输出 GEMV，K 为零时显式写零。其 source 参数为 `ROWS`，物理 LMUL 默认上限为 `64`。
Attention 的 `D` 是必须给出的 ABI specialization，不能在同一 artifact 上改成另一
head dimension。`Program.build` 复用普通 `weft.build`。`Program.autotune` 要求调用方给出
完整 `Config` 列表和 validator/validation_key；构造模型不会暗中启动调优或改写 state。

默认 Norm 的每个归约覆盖完整行，`BLOCK` 只划分输出遍历，不在每个小块强制标量收敛。
默认 softmax 在最大值归约后再次读取输入，保存一份 exponential Value，再用完整行求和的
标量倒数归一化；输出参数只写。行宽由 source `N` specialization 固定，因而完整快照的
资源需求在构建时可检查。默认 LMUL 上限为 2、私有存储预算为 256 KiB；行数仍可动态变化。
`*_row` 是独立作者版本：RMS 共享读取，LayerNorm 还共享 centered Value，softmax 还共享
exponential Value；它们改变读取次数及完整中间值的生命周期。
比较时必须区分作者程序改变与固定程序的 compiler 改进。`N` 从构建绑定固定，`M` 可动态变化；
能否容纳必要快照由 target 的私有存储预算决定，超出预算明确拒绝，不自动换程序。
`Operators` 提供同名方法并从输入宽度绑定 `N`；Torch 提供 `RMSNormRow`、`LayerNormRow`、
`SoftmaxRow`，沿同一 `Library` 注册、编译和导出。原模型组合使用其明确声明的 catalog 条目。

直接使用低层路由 kernel 时，调用方保证 `EP=E+1`、`P=T*K`、`0<=Indices<E`、Offsets
是覆盖 `[0,P)` 的单调前缀和、Positions 是对应的合法位置，且赋值总数不超过 i32 上限。
完整 `MoE` 组合自己生成这些中间结果，不接受调用者伪造的 routing metadata。

## 完整 block 与存储所有权

`TransformerBlock`、`SelfAttention`、`SwiGLU`、`MoE`、`Mamba1` 使用显式权重 dataclass。
所有线性投影、激活、归约、路由及状态计算都由 catalog 的 Weft kernel 执行。主机只组织调用、
借用视图、初始化存储和管理状态。所有权检查在首个写入前检查完整模型参数、输出、scratch
和 state；模型输出/scratch/state 的 byte span 不得互相重叠或覆盖输入/参数。
单算子的原地 alias 合同不自动扩大为整个模型的原地合同。

```python
from weft.programs import TransformerBlock, TransformerWeights

block = TransformerBlock(weights, heads=32, kv_heads=8)
cache = block.new_cache(batch=1, capacity=4096)
workspace = block.workspace(batch=1, tokens=prompt.shape[1])
y = block(prompt, cache, cosine=cos_for_prompt, sine=sin_for_prompt,
          workspace=workspace)
y_next = block(next_token, cache, cosine=cos_for_next, sine=sin_for_next)
block.close()
```

Transformer 是 RMS pre-norm → Q/K/V projection → 可选 RoPE → causal GQA → output projection
→ residual → RMSNorm → SwiGLU MLP → residual。调用者按实际绝对位置提供 cosine/sine 表；
不隐式选择 RoPE base、频率缩放或 position policy。没有 rotary 的变体显式传 `rope_layout=None`。
`SelfAttention` 提供相同的 attention 子程序，`SwiGLU` 提供独立前馈子程序。

`Workspace.specs` 与 `nbytes` 描述具体 scratch，shape/dtype/存储变化会在使用前拒绝。
workspace 同时只能属于一个 invocation。独立 workspace/state 可以共享同一个 `Operators`。
模型借用外部 Operators 时不会在 close 时关闭它；自己创建的 Operators 随模型关闭。

`required_programs` 给出模型调用的 catalog 条目和 ABI specialization；`materialize()` 预先
编译它们，不读写 activation。编译失败发生在修改模型 state 之前。执行中的错误可能已经产生
部分 native 写入，因此 `KVCache`/`MambaState` 将自己标为 invalid，必须显式 reset/restore。

## KV 与 Mamba 状态

`KVCache.keys/values` 的形状是 `[B,capacity,G,D]`，`position` 是已填充前缀长度。append
必须有容量；`fork()` 复制独立分支，`restore(snapshot)` 恢复同形状快照，`truncate(position)`
丢弃已填充后缀。截断不会清零后缀；下一次 append 覆盖新范围，attention 不读取 position 以外。
`reset()` 清零并回到 position 0。状态对象不能被并发调用、复制或重置。

`Mamba1` 实现完整 Mamba-1 mixer：输入投影得到 x/z，causal depthwise convolution + SiLU，
token-dependent x projection 生成低秩 dt/B/C，dt projection，再进行：

```text
dt = softplus(delta + dt_bias)
A = -exp(A_log)
h = exp(dt*A) * h + (dt*x) * B
y = (sum(h*C) + skip*x) * SiLU(z)
output = out_projection(y)
```

权重 `MambaWeights` 包含 in/out projection、可选对应 bias、conv/conv_bias、x_proj、dt_proj、
dt_bias、a_log、skip。history 是 `[B,W,I]`，按从旧到新排列，SSM state 是 `[B,I,N]`。
`new_state(batch)` 创建零状态；一次调用消费 `[B,T,D]` 并保留最终状态。整段、分块与单 token
调用使用同一 recurrence、相同初态和相同参数，比较输出及最终 history/SSM。
`fork/reset/restore` 与 KV 相同；不能用 KV 式 truncate 推回递归状态。
外部 residual/norm 由模型作者组合；Mamba-2/SSD、训练反向和低精度变体需要独立作者程序。

`selective_scan` 的作者程序按 token 顺序遍历，再对每个通道的完整 `N` 维 state 更新和归约。
更新后的同一 Value 供给 State 写回及输出归约，不拆成每个小块的标量累加。
source 参数为必需的 `N`；`Mamba1`、Torch 模块和离线 package 都从权重的 state 维度绑定。
这一顺序使同一 token 的输入通道相邻访问；目标编译器决定闭合 operation 的分片、
私有存储与 RVV 载体，不自行交换时间与通道循环或延长跨 token 的状态生命周期。

`MoE` 采用 selected-top-k renormalization、稳定分组、每 expert 的 gate/up → SwiGLU → down，
最后按 token/slot 加权合并。没有 capacity dropping、shared expert 或辅助损失。workspace 的
indices/weights/offsets/positions 可直接检查。全路由集中在少量 expert 以及其余 expert 为空，
都遵守同一程序，没有主机分组或外部专家库 fallback。

## Torch 与离线部署

```python
from weft.programs.torch import Library, MoE

library = Library("my_model")
module = MoE(torch_weights, top_k=2, library=library)
compiled = torch.compile(module, backend="eager", fullgraph=True)
y = compiled(x)
```

`Library` 在模型构造阶段显式注册命名 CPU custom operators，forward 不进行注册。
Torch adapter 复用相同作者程序及 M2 的 fake/mutation ABI，不另写数值实现。
Norm、Linear、SwiGLU、Transformer、MoE、Mamba1 提供 `nn.Module`；模型参数是 CPU f32
inference buffers，不自动定义 backward。view 必须可借用；raw Torch 调用的 grad 限制仍适用。
Torch Transformer 接受显式 keys/values/start，Mamba 接受 history/state；这些 mutable Tensor
输入由 schema 声明。图入口不维护 Python 状态计数，调用者负责 start/状态生命周期。
`backend="eager"` 的图执行验证 opaque native 调用保留，不代表图优化或性能增益。

```python
from weft.programs import LoadedOperators, Mamba1, ProgramPackageBuilder, save_operators

block.materialize()
package = save_operators(block.operators, "/new/path/programs")
loaded = LoadedOperators(package)
deployed = Mamba1(mamba_weights, operators=loaded)  # package 包含该模型 required_programs
```

跨目标构建使用 `ProgramPackageBuilder(target).add_model(block).build(...)`，底层是同一
`PackageBuilder` 与 C/C++ SDK package 格式。package 只包含具名代码与绑定，不包含权重或 state。
`LoadedOperators` 检查 catalog ABI 与 source specialization，只加载包内 kernel；缺少程序明确
报错，不自动 JIT、改配置或调用参考库。`native_library=False` 只生成可链接源码产物，需要显式
`Package.link` 后才能加载。保存/加载不转移旧机器性能结论。

## 参考机制与职责

Triton 的 `python/tutorials/05-layer-norm.py:48` 用作者确定的行归约算法和 block load；
这里 `source/weft/programs/normalization.py:26` 同样明确中心方差，但用普通有序行循环及 Level
表达有限逻辑块，不引入 GPU program id。RVV layout、归约累加器有效性和临时资源属于
`lib/Target/MaterializeRISCVPrograms.cpp` 与 `lib/Dialect/RISCV/IR/StreamOps.cpp`。

TileLang 的 `examples/fusedmoe/example_fusedmoe_torch.py:8` 展示路由/分组/专家/合并的完整
结构；这里的 `source/weft/programs/moe_block.py:77` 通过正式 Weft 程序执行这些环节，并明确
selected-top-k 归一化这一算法选择。参考实现由各自上游维护，不进入 Weft 的构建或运行。
Mamba 的计算定义对应[官方 Mamba-1 实现](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)
和[selective scan reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)；
`source/weft/programs/mamba.py:31` 与 `mamba_block.py` 明确每步及持久状态，参考实现只用于理解合同。

`examples/use.py` 是安装后普通 API 的统一入口，组织输入、独立数值 reference、模型组合和
状态一致性检查。它不选择模型 benchmark baseline，也不写性能 CSV。
