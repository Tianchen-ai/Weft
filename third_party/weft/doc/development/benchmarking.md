# 基线选择与性能归因

## 性能开发入口

算子计时、baseline 比较和 llama.cpp E2E 使用独立的 [Weft Bench](https://github.com/Tianchen-ai/Weft-Bench) 仓库。
安装该仓库后，以外部 JSON 配置机器、编译器和模型：

```sh
export WEFT_BENCH_CONFIG=/path/to/bench.local.json
weft-bench show q4_k_dot_rows
weft-bench op q4_k_dot_rows --target k1 --weft /path/to/native-artifact --baseline /path/to/reference --output /path/to/latest
weft-bench e2e llama2-7b-q4_0 --target k1 --phase prefill --output /path/to/latest-model
weft-bench compare runs/before.json runs/after.json
```

默认输出原生调用耗时及 CSV/JSON；默认不采样热点、不重跑全部输入规模。
`--baseline` 复用已保存结果，`--profile` 按需采样模型阶段。
配置中的 source/package adapter 选择 Triton 路线；机器、模型和产物路径保留在外部配置中。
本仓库保留作者 kernel、编译器/runtime、llama.cpp 集成和使用示例；测量与回归由 Weft Bench 维护。
下文说明编译器与算法的职责，具体运行参数和输出接口以 bench 文档为准。

## 1. 实验服务于编译器

Weft 的目标是建立语言和编译器：

```text
作者写出确定的数值与实现树
    ↓
编译器生成 target-aware RISC-V program
    ↓
intrinsic C / local asm
```

实验用于检验：程序是否生成数值正确的真实target artifact；编译器是否形成高质量机器实现；相同语言事实换format、shape、consumer、VLEN或extension后是否仍成立。

实验可以否定语言或编译器假设，但不能成为隐藏语言。不能因为某行CSV慢，就增加kernel/format route、whole-kernel leaf、emitter matcher或fallback。

## 2. Baseline 的双重作用

### 2.1 决定作者比较哪一种实现

GGML/llama.cpp 上游实现既是性能参照，也是作者编写对应 DSL/std 函数时的算法参照。使用者应显式选择上游源码与构建配置；为了比较同一种实现，作者必须先确认并自然表达：

- algorithm variant与数值分解；
- outer traversal、loop order与blocking；
- staging、workspace与跨loop reuse；
- accumulator/state的诞生层和更新方式；
- persistent repack、derived Encoding与ABI；
- activation quantize、decode、prefill等phase；
- preprocessing与timed region边界。

若baseline的高性能依赖这些结构，而DSL没有写出，修改DSL/std tree；不能让target compiler从普通SSA猜回blocking、staging或persistent packing。若DSL无法表达，这是语言缺口，不能换更容易的baseline或缩小workload。

### 2.2 不决定 target physical realization

作者树冻结后，baseline不能指定LMUL、`vl`、lane/register/fragment mapping、memory form、invocation-local pack schema、RVV/IME leaf、pipeline、spill或intrinsic spelling。这些由RISC-V passes与[leaf contracts](../compiler/leaves.md)决定。

可以阅读 baseline 或硬件参考理解 microkernel、reuse 和指令组织，但知识必须分别进入 DSL tree、physical compiler 或 local leaf；baseline 函数本身不得调用、链接、包装或成为 fallback。

### 2.3 不参与编译选择

GGML 吞吐不参与 leaf selection 或 tuner。tuner 只在作者/std 或 target 明确声明的有限候选之间编译、验证和实测；source 参数与物理策略分开记录，数值失败不能进入 winner。baseline 在 winner 生成以后评价最终 artifact。

## 3. 可比较 Case

将性能差异归因于编译器优化时，比较双方需匹配下列合同：

```text
target machine
algorithm variant与数值语义
operator/helper boundary
shape、dtype、quant format与layout
decode/prefill phase
thread count
input bytes/data generation policy
persistent preprocessing与repack归属
timed region
system compiler与flags
correctness policy
measurement protocol
```

产品整体对比可以保留正常应用的既有工具链，但必须同时说明应用和替换库的编译器、
选项与运行依赖。此时倍率包含工具链选择的影响，不能解释成单个 pass 或编程模型的独立贡献。

production `MUL_MAT`、standalone vec-dot、activation quantize和row dequantize是不同case。不能用vec-dot数字代替完整MUL_MAT，也不能把persistent repack排除在一边、计入另一边。

当前production矩阵参照为`N=4096, K=4096`，decode为`M=1`、prefill为`M=128`。vec-dot helper是独立case，参照shape为`M=1, N=14336, K=4096`。改变shape形成新case，不能覆盖旧结果。

具体correctness、toolchain、machines、timing与CSV合同见[测量协议](validation.md)。

原生调用、source/AOT、JIT/cache 与完整 grid 的检查由独立 Weft Bench 复用 workload/oracle 执行。
它们服务于不同调用边界的验收，不能代替上面的 production 性能 case；
GGML 仅在 Bench 中生成参考输入/输出，不进入 Triton runtime。
JIT 编译与加载耗时不混入已有 kernel-only timing；需要测冷启动时必须另行声明计时边界。

## 4. 性能差距的所有权

按顺序判断：

1. **DSL/std tree：** baseline是否具有当前树未表达的variant、blocking、staging、state或persistent layout；
2. **Canonical IR：** axes、Encoding、Level、effects与numerical semantics是否保存；
3. **RISC-V passes：** layout、conversion、memory、reuse、pipeline与resources是否产生；
4. **Leaf selection：** 是否选择符合typed facts与target capability的RVV/IME op；
5. **Terminal spelling：** intrinsic/asm是否忠实，Clang是否保留已选physical形态；
6. **Measurement：** workload、timing、flags与cache状态是否匹配。

第1项不一致时，修改DSL/std并作为新的作者程序重新比较。若tree、Encoding、std与physical passes已经闭合，但达到baseline仍必须改变canonical logical Value集合或Level归属，应停止并报告[唯一职责判据](../programming-model/index.md#6-唯一职责判据)可能被证伪，不能让compiler暗中改tree。

性能比、分档和聚类只是定位共享能力缺口的工具，不是编译器架构。单条高性能不能证明泛化，平均数也不能代替逐case事实。

## 5. 外部反证，而不是内部自评

“有 pass”“emitter 只拼写”或一条深度优化路径达到 baseline，都不能证明抽象成立。至少要改变会迫使物理表示重新形成的外部条件：

- 同一 source tree 在 VLEN128、VLEN256 与合法 RVV/IME profile 上编译；
- 改变 cohort、source `auto` 绑定、shape、stride、predicate 或普通 consumer；
- 让相同 operation/Encoding 出现在不同 Level、use-def 与 memory context；
- 使用没有针对性后端改动的新格式或 std 函数。

这些变化后，作者不应手工补 LMUL、local-pack schema、lane/register mapping、fragment 或 pipeline。若必须补这些信息，说明 physical machine 或 passes 没有真正承载该决定。若必须改变 canonical Value 集合、Level 或 artifact 才能获得所需实现，则把它记录为作者程序差异或职责判据的设计信号，不能由后端静默完成。

跨机器只检验目标事实是否产生各自合法的 physical program；速度只与同一机器、同一 case 的 baseline 比较。K1 的 intrinsic C 能生成不能代替 K1 真机执行，SG2044 的结果也不能外推为 VLEN256/IME 结论。

## 6. 禁止实验驱动出隐藏语言

不允许：

- 为追一行数字增加kernel名、format名、exact closure或whole-kernel leaf；
- 通过GGML、仓库外reference调用、legacy path或silent scalar fallback通过测试；
- 数值失败仍记录性能；
- 为float bit-exact修改作者tree或禁用合法融合；
- 用更小shape、不同phase、timing scope或helper替换失败case；
- 将tuner结果、generated-C inspection或compile success冒充真实target性能。

实验回答“当前编译器在明确合同下做得怎样”，不回答“下一条后端特例写在哪里”。
