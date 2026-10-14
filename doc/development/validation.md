# 正确性与性能测量协议

## 1. Correctness

### Encoding 与离散 Storage

bit-exact 只用于同一 packed bytes、bit/byte order、padding、`grouped/bit_layers/pack_fields` 的 storage mapping、离散 field，以及定义为 exact wrap/saturate 的整数结果。它用于发现 Encoding 错位、field 拼接和 ABI 错误。

### 浮点 Kernel

浮点输出检查finite/NaN/Inf政策、case预先声明的absolute/relative tolerance，以及符号、数量级、全零和明显量化错误。合法FMA contraction、reduction顺序或closed primitive partial可以产生末位差异。

不能为了float bit-exact交换作者scale tree、插入人为物化阻断contraction或修改合法leaf。tolerance必须在运行前定义，不能看到结果后放宽；数值失败不记录性能。

## 2. Toolchain

Weft generated C、runtime/harness和GGML baseline wrapper使用同一套Clang toolchain与共同flags。每次记录：

```text
Clang executable与完整版本
-O level
-ffp-contract policy
-march / target extensions
-mabi
LTO、fast-math及其它数值/代码生成flags
assembler / linker（IME需要时）
```

正式公共口径为`-O3 -ffp-contract=fast -mabi=lp64d`，`-march`与extension flags按target profile设置。不能让Weft开启contraction而baseline关闭，也不能用Clang编译Weft、用GCC编译baseline后给出比值。

Clang版本、target flags、assembler或数值flags改变时，相关baseline必须重编重测。generated C中的intrinsic、typed asm和保护已选physical形态的局部pragma属于编译器输出；公平性要求相同toolchain、target capability与数值政策，不要求两份source文本相同。

## 3. Target Machines

每次运行记录 OS/kernel、CPU identity、VLEN/extension probe、允许执行的 CPU 集合、目标
core、frequency/governor 和 toolchain。CPU 数量与可用扩展以当前系统探测为准，不从机器名称推定。

性能比只比较同一台机器上的 Weft 与 baseline。跨 target 检查同一 DSL tree 是否形成各自
合法的 layout、memory form、leaf 与 schedule，不能用不同机器的绝对速度计算编译器收益。
工具链不能生成或系统不允许执行某个扩展时，不报告该扩展的原生性能。

机器、工具链或运行协议变化时建立独立结果集，不能沿用不匹配的 baseline。

## 4. Kernel Timing

正式单线程kernel协议：

1. 初始化相同inputs与persistent artifacts；
2. 一次不计时warmup；
3. 每次timed invocation前遍历64 MiB eviction buffer；
4. 记录10次wall time；
5. 报告median；
6. 编译、动态链接、tensor初始化、persistent weight quantize/repack、warmup和eviction不计时。

activation quantize、kernel内local pack或其它operator-invocation工作，若baseline在timed graph内执行，Weft也必须计入。只有双方都属于跨调用persistent artifact phase时才可排除。每个case必须写明timing scope。

matrix/vec-dot按`2MNK`计算GOP/s；quantize/dequantize按logical elements计算MElements/s；没有一致work定义时只报告wall time。

模型级`llama-bench`沿用真实graph、线程和自身计时协议，不与kernel microbenchmark混在同一分布。

## 5. Result Files

将运行结果保存为独立 CSV/JSON 产物，明确区分固定 baseline、当前程序和对应比较。
baseline 只在机器、toolchain、flags、算法/shape 或测量协议变化时重测；Weft compiler
修改本身不要求 baseline 重跑。不匹配的旧协议与历史运行不得混成同一结果集。

只重跑受影响 case，并保留未测 case 的原始测量身份。失败或不可比较的结果不能覆盖有效
性能数字；如实记录失败和到达的 artifact 边界。局部定向运行不能表述为完整 workload 验收。

每行至少保存case identity、hardware/ISA、shape、timing scope、compiler/version、flags、physical configuration、repetitions、correctness、absolute/relative error、median与throughput。合同字段缺失的运行记录不得计算baseline ratio。

结果中的 `comparison_kind`、`comparison_median_ms`、`speedup` 字段明确标注本行
实际比较对象，`median_ms` 始终属于当前生成程序，`speedup` 为比较对象耗时除以 Weft 耗时。
固定作者程序的 compiler 对照必须使用同一输入、source/physical binding、目标、toolchain
和计时范围；不能借用历史记录耗时，也不能将该倍率标成 GGML 或模型收益。作者程序或默认
配置改变的对照必须单独标明。独立作者 variant 没有真实对照时保留空倍率。

公开程序与 GGML 比较时，以相同输入构造完整 CPU graph：RMSNorm 包含权重乘法，
LayerNorm 包含权重和偏置，residual RMSNorm 包含残差相加；不能用单独的 Norm 节点代替。
`examples/use.py --family operators --ggml-directory ...` 借用同一输入 storage，图与输出
buffer 在计时前建立，输出传回 Python 及数值检查不计时。保留所选 CPU 后端能力；固定一个
工作线程。reference 的库、输入、样本和完整计时范围随该行 configuration 保存；这类新工作
范围不覆盖既有固定 baseline 行。旧 Weft 与 GGML 同时参与时，分别保存各自对照，不能混合。
