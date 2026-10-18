# 非 SIMT 物理抽象机器

## 1. 定位

Weft 源程序规定 logical values、axes、Level、数值 operation、Encoding、lifetime、handoff 和 effects。目标编译器必须把这些事实实现为一份单控制器机器程序。

本文件定义 lowering 面向的物理抽象机器。它不把物理对象加入 DSL；该机器由唯一的 target-aware [RISC-V IR](riscv-ir.md)承载，编译算法见[RISC-V Pass](passes.md)。

非 SIMT 的准确含义是：

> logical Value 不归属于虚拟 thread、warp、CTA 或 program instance。目标编译器直接把完整 logical Value 分解到顺序 issue、SIMD lane、寄存器副本、extension fragment 和局部存储。

“单控制器”只描述一条有序控制流能够直接调度这些资源，不表示机器只有一种执行单元，也不表示所有目标拥有相同的寄存器、fragment 或 memory hierarchy。

## 2. 源程序身份不可丢失

每个物理实体都必须能追溯到以下 source identity：

```text
canonical Value 或 closed primitive temporary
logical axes 与 coordinates
所属 Level 和 Level instance
ordinary control iteration / branch
producer、consumers 与 use edge
effect、alias、validity 与 order
```

物理化可以复制、分片、暂存或重算同一 Value，但不能改变：

- canonical logical value 集合；
- logical axes 与 extent；
- Value 的 Level 归属；
- births、handoff 和普通控制的可观察顺序；
- pinned View 的 Encoding 与 ABI；
- operation 的数值、overflow、结合与 effect 语义。

若一种高性能实现必须改变其中任何一项，它是另一棵作者程序，不是这台机器上的另一种表示。

## 3. 物理表示不是一个统一根类型

一个 logical coordinate 可以在执行期间与下列物理分量建立关系：

```text
logical coordinates
    → issue/time coordinates
    × SIMD lane coordinates
    × register-replica coordinates
    × extension-fragment coordinates
    × local-storage coordinates
```

这不是要求每个 Value 都映射到完整五元组的笛卡尔积。它是一条可组合、可部分定义、允许一对多的表示关系：

- scalar Value 可以只有 time 与一个 scalar carrier；
- RVV Value 可以使用 time、lane 和 register replica；
- IME operand/result 可以使用 time、fragment，并通过转换与 RVV/register Value 相接；
- staged Value 可以暂时只存在于 local storage；
- broadcast 可以让一个 logical coordinate 同时出现在多个 lane 或 replica；
- spill 可以让同一 Value 在不同时间先后存在于 register 和 local storage。

源语言因此仍没有统一的 tile、vector 或 fragment 根对象。统一的是 logical identity 与表示关系，不是某种物理 carrier。

## 4. 五类物理分量

### 4.1 Issue/time

time 表示同一 source instance 内的物理执行次序，包括：

- vector strip；
- primitive 内部顺序迭代；
- unrolled instance；
- pipeline stage 与 buffer version；
- prologue、steady state 和 epilogue；
- spill、reload 或 rematerialize 的发生点。

每个 physical time point 必须保留它对应的 source Level/domain point、ordinary loop iteration、tail validity 和 effect order。增加 physical time point 不能增加 `load`、`state`、`stage` 或 handoff 的逻辑次数。

### 4.2 SIMD lane

lane 表示一组 logical coordinates 在一次宽指令中的位置。lane mapping 必须说明：

- 承载哪些 logical axes；
- axis extent 与 active `vl`；
- tail/mask 怎样对应 logical validity；
- cast/widen/narrow 后 logical element identity 怎样保持；
- producer 与 consumer 的 lane mapping 是否一致。

普通有序 scalar loop 没有 shaped axis，不能仅因迭代同构而获得 lane mapping。

### 4.3 Register replica

register replica 表示同一时刻保留的多个 scalar/vector physical values。它可以实现：

- free axis 的物理 microtile；
- 多个输出 accumulator；
- broadcast 后的多个 consumer operand；
- closed reduction/contract 内部不可观察的 partial；
- pipeline 的 register buffer version。

replica 不是 canonical Value。replica 的拆分与合并必须在同一 source Value 或 closed primitive contract 内闭合；若多个 partial 要跨 source Level handoff、成为独立 state 或改变浮点合并位置，它们必须由作者显式写成 logical values。

### 4.4 Extension fragment

fragment 是 matrix/tensor/DSP extension 的目标 operand 或 result object。target operation contract 必须规定：

- 对应哪些 logical axes；
- fragment shape、dtype 与 accumulator type；
- logical coordinates 到 fragment coordinates 的合法关系；
- mask、tail 与 partial fragment 支持；
- fragment 与 lane/register/local-storage 表示之间的合法转换；
- fragment、temporary 和 accumulator 的资源占用。

fragment 可以是不可逐元素寻址的 opaque carrier，但不能暗中拥有 outer traversal、Level、workspace、persistent layout 或 kernel ABI。

### 4.5 Local storage

local storage 是 invocation 内、target 管理的可寻址临时载体。它可以承载：

- spill slot；
- target 选择的 invocation-local pack；
- pipeline buffer；
- engine handoff 的局部物化；
- primitive-private temporary。

它不是 DSL 中的 cache/shared/L1/stack scope。target profile 规定 capacity、alignment、addressing、bank/port 与 ordering 能力；physical program 为每个对象规定 size、lifetime、alias 和 producer/consumer。它不能替代 caller-visible workspace，也不能越过 pin boundary 形成未声明 ABI。

## 5. Axis preservation

每个 physical representation 必须显式保存 logical axis relation。Pointwise、broadcast、cast、memory、reduce 和 contract 通过 axis relation连接，不得只保存“当前 lane axis”而丢掉其它 free axes。

### 5.1 Pointwise 与 broadcast

- 同名 logical axes 保持一一对应；
- 缺失轴的 scalar/singleton operand可以复制到多个 physical carrier；
- 复制不创建新的 logical elements；
- consumer 要求不同 representation 时，在 use edge 插入 conversion。

### 5.2 Reduction

设输入 axes 为 `F ∪ R`，operation 消去 reduction axes `R`，结果 axes 必须恰好为 `F`。

编译器可以把 `R` 映射到 time、lane collective、register partial 或 fragment reduction；这些 partial 必须在 closed primitive 内合并。所有 `F` axes 及其 coordinates 必须继续存在于 result representation。

因此：

- 对 `[M,K]` 沿 K 求和，结果是 `[M]`，不能变成一个 scalar 再广播；
- codebook 的八元素轴只有在 source 中作为 shaped axis 存在时才能映射到 lane；八个独立 scalar lookup 不能由编译器重新发明为一条轴；
- output cohort 可以映射到 register replica，但 logical cohort width 与 handoff 不变。

### 5.3 Contract

contract operation 提供 free/reduction axis relation与数值语义。target compiler可以在 closed operation 内选择：

- 哪些 free axes 进入 lane、register replica 或 fragment；
- reduction axis 怎样进入 time、unroll 或 fragment K coordinate；
- operand broadcast、local pack、accumulator 与 result handoff。

它不能创建 source 未写的 outer loop、blocking、staging、persistent packing 或 accumulator state。

### 5.4 Lookup

lookup result 必须继承 shaped indices 的全部 logical axes。table entry axis 在每个 index coordinate 上被读取并消去；其它 free axes不能丢失。对 `[M,G]` indices，target可以把 G 映射到 lane、把 M 映射到 register replica，并选择 unit/indexed/gather/table instruction，但 result 仍是 `[M,G]`，沿 G 的后续 reduce 仍必须得到 `[M]`。

多个 output consumers 共享 packed/index/codebook window 只有在 source use-def、dominance 与 effect 允许时才能形成。physical program可以共享一个 producer representation或建立局部 window；不能按格式名、固定 lookup 数量或相邻 closure 识别 codebook axis。

## 6. Target operation contract

每个 target-local operation 必须声明：

```text
实现哪个 canonical op
使用哪个 physical engine class
operand/result representation constraints
dtype、shape、axis 与 mask/tail legality
memory、alias 与 ordering requirements
产生的 conversions 或 local temporaries
register、fragment、local-storage 与 buffer resources
instruction/intrinsic/asm realization
```

ISA 通常只给出合法区域，不给出唯一实现。target profile 因此还要为多个合法 structural realization 提供固定规则与优先级。同一个 canonical `contract` 可以拥有 RVV register-microkernel 和 IME fragment contracts；target 根据 operand axes、Encoding、representation、resources 与 build requirements 选择合法 engine 和 fragment family。

### 6.1 Local leaf

local leaf 是上述 target operation contract 的最终 ISA 落点，不是独立算法层。RVV/IME leaf 的typed合同、选择、opaque边界与失败规则统一定义在[RISC-V Local Leaf](leaves.md)；terminal spelling见[Emission](emission.md)。

## 7. Physical conversion

conversion 在不改变 canonical Value、logical axes 和 Level 归属的前提下，把一种表示变成另一种。它可以落成：

- lane shuffle、slide、gather 或 broadcast；
- register tuple 的 split/merge；
- widen/narrow 对应的 physical type change；
- register/fragment handoff；
- local store/load；
- target-selected local pack 的物化与读取。

conversion 必须是 physical program 中可观察、可验证、可消除的 operation，不能只是 emitter 旁边的字符串或缺失字段默认值。多个 consumer 可以共享一个支配它们的 representation；若 consumer 要求冲突，conversion 归属于具体 use edge。

pin boundary 不允许隐式 conversion。kernel 参数、caller workspace 与 persistent derived encoding 的 bytes 必须与声明的 Encoding identity 一致。

## 8. Local pack 的物理含义

local pack 是 target 为同一个 canonical Value 或 value-use edge选择的一种 invocation-local representation。源程序不包含 `pack(along=...)`；它只通过 Value/Level/use-def 给出：

- logical values、shape 与 axes；
- producer Encoding、address relation 与 validity；
- `stage` 建立的 logical birth、lifetime 与 reuse domain；
- all consumers 及其 typed operation requirements。

target compiler 据此决定是否需要 local pack、哪条 logical axis 对当前 consumer 连续、pack 的 carrier 与 schema，以及它在 register、fragment 或 local storage 中的表示。选择同时读取：

- producer storage mapping 与地址连续性；
- 全部 consumers 的 lane/register/fragment requirements；
- widening、decode 与 compute fusion；
- live interval、pipeline buffers 与 resources；
- physical engine handoff 与 local-storage 能力。

target 从显式声明的有限 pack 候选中按固定规则、分项成本或外部实测选择；schema 内的 lane factor、LMUL、microtile extent 和 buffer count 同样可以有限调优。候选来源、预算和估计遵守[物理优化方法](optimization-principles.md)。local pack 不得改变 canonical axes、Level birth、effects 或 staged value 的逻辑初始化次数，也不得越过 pin boundary。

若作者要改变 logical axes，使用 canonical reshape/transpose/index operation；若要改变跨调用 bytes、artifact 与 ABI，使用 derived Encoding builder。两者都不是 local pack。

## 9. 跨 Level、loop 与 consumer

### 9.1 Level

physical schedule 可以 strip-mine、分多次 issue 或建立局部 pipeline，但每个 physical instance 必须保持 source Level path、domain point、births 和 handoff identity。一个 Level 不能被物理调度变成多个 source Level。

### 9.2 Ordinary loop/control

ordinary `for/while/if` 保持有序语义。编译器可以移动不改变 effect/order 且已证明 loop invariant 的 physical operation；不能从普通迭代生成 shaped axis。loop-carried Value 的 representation 可以改变，但转换必须出现在明确的 loop edge，并保持 carry type/value identity。

### 9.3 Multiple consumers

同一 Value 的 producer representation 可由多个 consumer共享。编译器根据 dominance、live interval、effect 与资源决定：

- 保持并共享；
- 在某个 use edge conversion；
- 从同一 pinned/staged source reload；
- 对允许的 pure producer rematerialize；
- spill 后在各 consumer 前 reload。

它不能按源码闭包或固定 consumer 数识别新的 primitive，也不能把独立 source loads 按地址猜成同一个 logical Value。

## 10. Spill、reload 与 rematerialize

spill 将一个 physical representation 保存到不与 source View 混淆的 local-storage slot；reload 恢复同一 logical Value。slot 必须具有确定 size、alignment、lifetime、alias 与 use edges。

rematerialize 只允许重新执行 operation semantics 明确允许重算的 pure producer。带 effect、volatile/atomic memory、ordered state、不可重复外部读取或会跨越 source handoff 的 operation 不可 rematerialize。

staged Value 被 spill 不等于再次执行 source `stage`：它仍然是同一次 logical birth 的物理存储变化。任何会增加 source materialization 次数或越过 pin boundary 的方案都非法。

## 11. Local cluster 与 software pipeline

local cluster 是从作者已有 Level/ordinary loop 中抽取的一组物理 producer、conversion、memory operation 和 compute operation。它必须保留真实 use-def、alias、effect、state carry 和 tail validity。

对合法 cluster，target可以形成：

- sequential schedule；
- prologue / steady state / epilogue；
- stage assignment；
- single/double/multi buffer version；
- async or ordinary transfer、wait 与 local load；
- unroll 与 prefetch。

这些结构只改变同一 source instances 的 issue time和temporary lifetime；不能创建新的算法 pass、跨 Level workspace、persistent buffer或另一种state recurrence。pipeline depth、buffer count、unroll和prefetch distance是有限物理参数，可由构建期实测选择。

当前 RISC-V 实现已将 scheduler 与 expander 分开。depth=1 是顺序执行；depth=2
先从一个带 carry 的 physical Level 的 use-def/effect 推导 pure/read producer 和 carry-dependent
consumer，再为任意数量的跨 stage SSA value 与多个 source carries 生成真实 prologue/
steady-state/epilogue。当前只实现距离一迭代、depth=2/buffer=2；嵌套 region 与
write/unknown effect 还没有 predication/ordering 合同，会明确拒绝。async transfer、wait、
多于两个 buffer、显式 local-storage ping-pong 和非零 prefetch 尚未形成合法 physical
program，不能仅通过设置参数声称支持。

同一iteration内，闭合grouped/layered raw window可以选择一个target-local
`scalar-prime + vector-load` leaf。它只改变该memory edge的有限指令拼写，不读取下一iteration，
也没有distance、buffer version或跨stage SSA，因此不属于本节的software prefetch。

## 12. Resource model

每份完整物理参数绑定都必须计算同时 live 的：

- scalar/vector operands；
- register replicas与accumulators；
- mask与indices；
- cast/widen/narrow/decode temporaries；
- state representations；
- pipeline buffers；
- extension fragments；
- local-storage objects。

资源计算直接来自选定 representation、operation 与 schedule。资源不足时，只有三种合法结果：使用结构规则已明确允许的 spill、拒绝该物理参数绑定、或在固定优先级中判当前结构不合法。Emitter 不得缩小 LMUL、减少 accumulator、改变 fragment 或偷偷换 engine。

## 13. 三类编译决定

物理机器上的决定分为：

### 13.1 唯一合法推导

axis relation、Encoding mapping、typed conversion关系、effect/alias/order、engine/instruction/fragment legality。选定 base LMUL/lane parameter 后，cast/widen/narrow 链上的派生 element width 与 representation relation也属于唯一推导。结果错了会改变数值或生成非法程序，不能启发式。

### 13.2 结构性选择

同一 source tree 下的 engine、lane/register/fragment mapping、memory form、local pack、materialization/reload/rematerialize、local pipeline structure 与 fragment family，均可由 target 提供有界候选。固定优先级、分项成本或外部实测只比较合法候选，不能代替数值/effect/资源验证；选定结果必须保存在 Physical IR 中。

### 13.3 参数性选择

结构固定后的 LMUL、schema 内 physical microtile extent、unroll、pipeline depth、buffer count、
local memory-leaf variant和prefetch distance。target提供有限合法域，tuner通过实测选择；非法绑定
在emission前拒绝。当前RISC-V compiler API暴露LMUL、unroll、pipeline depth和boolean
scalar-load-prime；后者适用于已证明的单 window grouped/layered partial supply，或完整的
二至四 atom 连续 byte fragment supply，均只在当前局部读取范围内选择固定指令序列。
depth=2固定导出两个buffer，prefetch固定为0，后二者还不是独立可调域。

## 14. Target profile

一个完整 target implementation 必须为其实际支持的机器能力提供下列 facts；未提供的能力就是 unsupported，不能由 pass 或 emitter 假定默认存在：

```text
ISA 与 ABI
vector length / scalable-vector facts
physical engine classes 与可实现的 canonical ops
representation constructors 与合法 conversions
scalar/vector register resources
fragment families、operand/result constraints 与 resources
unit/strided/indexed/segment memory capabilities
alignment、mask、tail 与 gather/segment restrictions
local-storage capacity、addressing、alignment 与 ordering
operation latency class、transfer、async、wait、barrier 与 issue restrictions
spill/rematerialize legality
固定 structural rules 与 priorities
有限 physical parameter domains
intrinsic / local asm availability
```

当前 RISC-V `TargetAttr` 只物化已经有真实 consumer 的子集：ISA/ABI、显式 VLEN、32 个 vector registers、合法 SEW/LMUL、完整 `V` 的 indexed/segment/widening 能力、local-storage 上界、partial-combine 与 record-axis placement 的固定结构优先级，以及 typed IME fragment capability。partial-combine policy 只在多个 topology 均合法时规定 `independent-multilevel` 或 `sequential` 的优先关系；`max-widening-combine-groups` 进一步规定该 target 愿意为一次 widening combine 的结果保留多少 vector groups。record-axis policy 则在已经证明合法的 pointwise record loop 上规定优先保留 record 内 lane mapping，还是把多个完整 record instance 放进一个 lane cohort。这些都是 target 的固定结构规则，不是 source `auto`、tuner 参数或 target-name 分支。当前 backend 要求完整 `V` 和显式正 VLEN；`Zve`、纯标量 target、未知 VLEN、异步 transfer/wait/barrier、独立 prefetch 域及通用 latency model 均明确不在当前实现范围。它们没有占位字段，也不能通过默认值假装可用。

换目标时可以改变这些 profile facts、规则、参数域和最终指令；不能改变：

- source Value/axis/Level/operation语义；
- ordinary control的有序性；
- pin boundary 的 Encoding/layout identity；
- 本文件对 time/lane/replica/fragment/local-storage 表示关系的定义。

新 extension 通过增加 representation、target operation、conversion、resource rule与spelling接入；不能增加按 kernel/格式名接管 whole-kernel lowering的路径。

### 14.1 Build config

build config 是 canonical IR 之外、target lowering 的独立输入。它可以指定 target profile、source `auto` 绑定与对最终 physical program 的硬 requirement，例如：

```text
target = K1
require = uses_extension(IME)
```

requirement 是 physical legality 的额外约束：结构性选择前，不能满足它的 structures 被排除；选择后、emission 前还要对完整 physical program 验证。不满足就拒绝当前 build，不能静默改用 RVV，也不能修改 source tree。它约束的是 target artifact，不定义 canonical operation 或 Value identity。若高性能 IME 路径需要不同的 Level/stage/persistent Encoding，作者必须在进入 lowering 前选择另一份 std source tree。

## 15. 与 Triton/TileLang 可复用的机制边界

以下路径相对于本仓库根目录位于同级reference checkout `../ref/`。TritonGPU把 distributed layout 放在 tensor type encoding 上，`convert_layout` 是真实 typed op；coalescing、matmul acceleration、layout-conversion elimination、loop scheduling和pipeline都改写IR。对应源码位于：

- `../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738`；
- `../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32`；
- `../ref/triton/third_party/nvidia/backend/compiler.py:297`。

Weft 应复用“typed representation + explicit conversion + real program rewrite”的机制，不复用 logical coordinate→thread/warp/CTA ownership 作为 source 根布局。

TileLang 在 source 中显式提供 shared/local/fragment allocation、thread binding、copy/pipeline metadata和layout；对应源码位于 `../ref/tilelang/tilelang/language/allocate.py:1`、`kernel.py:149`、`loop.py:13`。Weft 不把这些 target storage/thread objects加入 DSL，但物理机器必须具有同等明确的 local storage、fragment、transfer 与 pipeline语义，不能只靠 emitter 私约定。

## 16. 与 physical IR 的边界

这台抽象机器不是 Kernel IR 与 RISC-V IR 之间的第三层。它是第二层 target-aware RISC-V IR 的语义：每个 representation、conversion、local storage、cluster、buffer version 和 target operation 都必须成为该程序中可验证、可改写的实体。

```text
Canonical Kernel IR
    ↓
RISC-V IR：本文件定义的物理机器
    ↓
intrinsic C / local asm
```

RISC-V IR可以混用`func/scf/arith`与`weft_riscv` operations；dialect namespace和pass数量不增加IR层次。设计不采用`weft_phys → weft_riscv`两级physical lowering，也不采用canonical program旁边的`problem/assignment`字典。

typed RISC-V IR只是必要载体，不是充分能力。有了dialect、type和op，不会自动产生：

- axis/layout propagation；
- coalescing 与 memory scheduling；
- conversion insertion/elimination；
- reuse 与 rematerialization analysis；
- cluster construction 与 software pipelining；
- resource allocation；
- target structural rules；
- physical parameter tuning。

这些是真实编译算法；它们作为pass改写同一份RISC-V IR，并用改变输入、target与consumer后的结果检验。

## 17. 尚未闭合的机器问题

下列问题不能由 backend convention 私下补齐：

1. **跨 engine 异步流水。** register/fragment/local-storage conversion、wait/barrier 与 source use-def/effect 的完整对应；若同步本身 source-visible，需要定义真实 effect operation，不能恢复通用 `stage_handoff`。
2. **fragment 的部分 spill 与 handoff。** opaque fragment 能否局部拆分、怎样保持 source Value identity、哪些 reduction 必须在fragment内闭合。
3. **local storage 的跨 engine ordering。** 不同 engine 共享 local object 时的 alias、coherence、wait/barrier合同。
4. **动态 loop 与 tail 的 physical time identity。** pipeline version、dynamic trip count、tail mask与source Level instance的完整对应规则。

在这些合同冻结前，target缺少合法实现时必须明确拒绝，不能由emitter补默认同步、默认fragment或默认storage行为。
