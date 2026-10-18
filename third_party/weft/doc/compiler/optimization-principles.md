# 非 SIMT 物理优化方法

## 1. 目的

Weft 的物理编译器面对的是一份已经确定 logical values、axes、Level、Encoding、
数值 operation 与 effects 的程序。优化不能改变这棵程序树；它要减少同一程序在
单控制器 RISC-V core 上执行的动态工作，并在寄存器预算内隐藏剩余延迟。

本文件给出一套可复用的诊断方法。它不是 pass 清单，也不以“新增了 typed op”作为
进展。每次优化前后都必须回答：哪类动态工作减少了，减少多少，代价转移到了哪里。

方法由一个前置门槛、五条物理原则和一个全局仲裁器组成：

```text
P0 先数工作
P1 载体归属        logical axis → lane / register replica / issue time / fragment
P2 供应唯一        distinct producer value 在其复用域内只产生一次
P3 形态跟随        memory instruction 服从 storage geometry 与 consumer layout
P4 延迟收敛        宽 partial 保持到该形态能完成的工作全部完成
P5 issue-time 重叠  在真实依赖允许时把 producer latency 与 compute 重叠

R  寄存器、fragment 与 local-storage live set 是共同合法性约束
```

P1–P5 不是互斥 pass。一个 kernel 可以同时违反多条；诊断必须把现象归到产生它的
物理实体，不能用一个总的“向量化不好”替代。

## 2. P0：命名不等于减少

新增 physical type、op、window、partial 或 cluster 只增加了可观察性。只有它消除了
动态 load、decode、index、conversion、MAC、reduction、spill 或等待，才是优化。

动手前建立一份 physical work ledger：

| 类别 | 静态实体 | 动态计数方法 |
|---|---|---|
| supply | load、decode、lookup、address/index producer | 静态 op 数 × 所在 loop trip count |
| compute | multiply、MAC、pointwise、cast | 每个 leaf 的指令数 × issue 次数 |
| convergence | partial combine、lane reduction、scalar extract | 每个 result/partial 的 reduce 次数 |
| movement | conversion、gather、slide、spill/reload | 指令数、bytes 与 live interval |
| schedule | stage、buffer、prefetch、wait | steady-state iteration 与 prologue/epilogue 开销 |

两个操作只有在完整 identity 相同并且一个结果支配所有 consumers 时才可合并。memory
identity 至少包含 source SSA、logical coordinate/offset、Encoding relation、access、result
layout、validity 与 effect。地址在运行时偶尔别名，不等于 SSA producer 相同。

对应参考机制：Triton `RemoveLayoutConversions.cpp:1044-1078` 比较 conversion 与
rematerialization 的估计成本；TileLang `layout_cost_model.cc:55-92` 计算 global-memory
segment bytes 与 issue-equivalent bytes。二者都不是“出现某种 IR 名称就认为更快”。

## 3. P1：载体归属

### 3.1 原则

每条 logical axis 必须在一个具体 value/use 上映射到：

- SIMD lane：同一条宽指令中的 coordinates；
- register replica：同时存活的 output/partial copies；
- issue time：顺序 strip、unroll instance 或 loop iteration；
- extension fragment：固定硬件坐标；
- local storage：可寻址临时载体。

选错载体会改变动态指令数和 live set。reduction axis 放进 register replica 会复制
partial；free/output axis 放进 lane 后过早 scalar extract 会失去多个 accumulator；本应顺序
issue 的小 reduction 全部同时存活会直接超出寄存器预算。

### 3.2 在代码里看什么

```text
final IR:  ValueType 的 time/lane/replica factors、register groups、vl
C/asm:     每个 logical output 的 vector op 数、vmv.x.s 数、tuple 数、spill/reload
```

先数每个 axis 的 physical factor，再数由该 factor 产生的 instruction multiplicity。仅看
LMUL 不够；相同 LMUL 下 replica 数可以相差一个数量级。reduction axis 的归属必须同时考虑
axis extent、SEW、VLEN、surviving free axes、consumer contract 与完整 live set。

### 3.3 责任与 pass

logical axis 是否存在、是否是普通 scalar loop，归作者；同一 shaped axis 的物理载体归编译器。
当前主要 owner 是 `SelectRISCVOperations`、`PropagateRISCVLayouts` 和
`PlanRISCVPartialTopologies`；`MaterializeRISCVResources` 只检查最终 live set。若达到性能
必须把 Python 展开的标量循环重新发明成 axis，
应停止并修改 std tree，而不是增加 matcher。

target 可以决定同一 canonical tree 的 LMUL、VL、lane/time 分解、physical 参数、memory
instruction、partial topology、pipeline 以及 RVV/IME realization。target 不能决定是否存在
logical axis、选择 scalar tree 还是 shaped tree、替换 std function、改变 reduction 结构，或
改变 Value/Level 集合。后几项一旦不同，就是不同作者程序；不能以 VLEN、target 名称或
实现能力作为 source-level 分支条件。多个确实不同的数值程序若都需要保留，由调用方以
entry 身份显式选择。tuner 可以枚举作者声明的 source 参数及 target 声明的有限物理结构与参数，但必须分别记录候选来源；物理比较固定已经实例化的 canonical tree。

Triton 的对应机制不是“SIMT 总工作固定”。`OptimizeThreadLocality.cpp:106-176` 会重分配
thread/warp ownership以减少 shuffle；`AccelerateMatmul.cpp:86-145,449-485` 会按 dot shape、
MMA instruction shape 与 warp 数重构 operand/result layout。固定的是 launch participants 的
某些覆盖关系，不是复制、shuffle、register 或动态 instruction 总量。Weft 的差异是 logical
value 没有预先分给 thread，P1 必须直接在完整值上决定 lane/replica/time。

## 4. P2：供应唯一

### 4.1 原则

一次 load、decode、lookup、index 或 address construction 应按它产生的 **distinct physical
value** 发生一次，而不是按 consumer 次数发生。共享需要同时满足：

1. source 中存在共同 SSA producer或可证明等价的 physical producer；
2. producer 支配所有 uses，且 effect/alias/order允许移动；
3. consumers 接受同一 representation，或共享点之前/之后有显式 conversion；
4. 扩大的 lifetime 不导致更昂贵的 spill。

P2 不允许把不同 logical offset、不同 table entry 或不同 decode layer 合并。那会改变程序，
不是 reuse。

### 4.2 在代码里看什么

```text
distinct source/offset identities
每个 identity 的 load/decode/index 指令次数
consumer 数与实际 producer 数
producer 定义点到 last use 的 live interval
因扩大 lifetime 新增的 spill/reload
```

### 4.3 责任与 pass

source Value 的 birth、Level 与 handoff 归作者；同一次 birth 的 physical placement、reload、
rematerialize和consumer sharing归编译器。`PlanRISCVMemory`拥有同一block内byte-aligned
natural encoded scalar的multi-consumer placement：它沿纯一元use-def链找到共同supply，
并以真实`physical-share` SSA value冻结一次load/decode。跨block、跨Level、vector/local
allocation以及混合effect的placement尚无统一owner；这些情况仍分别受
`CanonicalizeRISCVLayouts`的受限single-use rematerialization、
`ShareRISCVLayeredWindows`的typed layered geometry与显式local materialization约束。

Triton `OptimizeDotOperands.cpp:181-299` 把 local allocation/load 提到共同 base tensor并让
多个 views消费；`RemoveLayoutConversions.cpp:42-59` 以 anchor、propagation、conflict与
dominance rewrite组织 representation reuse。TileLang reducer只有同一个 epoch、structurally
equal plan的 update sites 才共享 partial storage；它同样不会按相似地址合并独立值。

## 5. P3：访问形态跟随 storage geometry

### 5.1 原则

memory form 必须由 Encoding/storage mapping、pointer relation、consumer layout和target
capability共同得到：

```text
contiguous coordinates       → unit load/store
fixed affine stride          → strided
entry/payload relation       → indexed/gather or entry window
interleaved fields           → segment
sub-byte layered geometry    → one raw window + mask/shift/merge
joined planes                → typed multi-source reconstruction
```

P3 决定“怎样取得同一个值”；P2 决定“取得几次”。二者不能合成一个含糊的 memory pass。

### 5.2 在代码里看什么

```text
vle/vlse/vluxei/vlseg 的数量与宽度
scalar address mul/add、index-vector construction、vrgather/vslide
每个 raw byte window服务的logical elements
mask/shift是否与Encoding geometry一致
```

### 5.3 责任与 pass

Encoding 若没有表达 bit/byte/group/layer/joined relation，作者必须修 Encoding；relation
已经存在时，unit/strided/indexed/segment/gather/unpack归编译器。当前主要由
`PlanRISCVMemory`、`FuseRISCVBitplanes`、`ShareRISCVLayeredWindows` 负责。joined high-plane、
radix storage projection和codebook entry payload仍有覆盖缺口。

Triton `CoalesceUtils.cpp:17-94` 读取 pointer contiguity/divisibility、shape、load/store语义与
thread count选择order和per-thread width；`Coalesce.cpp:82-119` 插入真实 layout conversion。
TileLang `loop_vectorize.cc:216-266,428-476` 从target最大load宽度和所有access约束的GCD决定
vector size。它们提供的是基于访问事实的规则，不是按kernel名称选择load模板。

## 6. P4：延迟收敛

### 6.1 原则

lane/register partial 应保持到该形态能做的乘加、scale combine和partial combine全部完成，
再发生 lane reduction 或 scalar extract。过早 reduction 会产生：

- 每个 contribution 一次 `vwredsum/vredsum`；
- `vmv.x.s` 后重新 broadcast；
- scalar carry串行依赖；
- 大量独立 partial spill后再reload。

但“延迟收敛”不能改变作者声明的整数宽度、溢出边界、浮点结合或 source partial集合。
`scale * reduce(product)` 与 `reduce(scale * product)` 若改变中间值，就是另一棵作者程序。

### 6.2 在代码里看什么

```text
每个block/half/output的reduction次数
vector partial slots、combine arity和tree depth
vmv.x.s / scalar carry update次数
partial spill/reload与最终reduce之前完成的MAC数量
```

### 6.3 责任与 pass

closed contract/reduce 内的 physical partial set、combine topology和final reduction归编译器；
改变 canonical partial、widening位置或Level carry归作者。当前
`PlanRISCVPartialTopologies`、`MaterializeRISCVPartialAccumulators` 与
`LowerRISCVComposites` 服务P4。planner 必须先冻结 product carrier、切片点、partial-set
类型、combine topology、scale supply 与资源合同；materializer 不得重新选择这些结构。
computed scale、codebook-derived operand和非统一SSA拼写仍会使同一物理关系落回顺序reduction。

Triton `ReduceOpToLLVM.cpp:228-351` 从register/lane bases和被消去axis组织local/lane tree
reduction；TileLang `reducer_plan_materialize.cc:559-946` 从update-site reduction axes投影
partial storage并选择narrow/wide/packed plan。可复用的是“先有typed distribution/plan，
再materialize combine”，不是GPU thread collective本身。

## 7. P5：issue-time 重叠

### 7.1 为什么单列

流水不一定减少指令；它可能增加guard、buffer version、prologue/epilogue和register
pressure，但通过把下一iteration的load/decode与当前compute重叠而缩短critical path。因此它
既不是P2供应唯一，也不是P4延迟reduction。

### 7.2 形成条件与计数

只有同时满足以下条件才尝试pipeline：

- source loop至少有两个动态iterations；
- physical IR中存在可移动的pure/read producer与carry-dependent consumer；
- effect、alias、tail和control允许跨iteration移动；
- producer latency足以覆盖buffer/guard开销；
- 增加的buffer live set不导致spill。

应检查：load-to-use距离、steady-state中producer/consumer交错、跨stage SSA数量、buffer
versions、prologue/epilogue占比、spill变化。只看到`pipeline_depth=2`不算实现。

### 7.3 调度边界

当前`ScheduleRISCVLevels`负责从use-def/effect写stage/order，`PipelineRISCVLevels`只做
SSA version与prologue/steady/epilogue展开。只实现distance=1、depth=2/buffer=2；没有typed
transfer和latency事实时，prefetch保持不存在，而不是设一个非零默认值。

同一 block 内，`ScheduleRISCVLocalSupplies` 可把多 atom 的 fragment raw read 及其纯地址依赖
移到紧邻的前一条 MMA 之前。它不跨写入、未知 effect、memory-clobber 或 birth，最多跨一次
MMA，并在改写后检查完整 kernel 的寄存器驻留；不能把此局部重排解释为跨 iteration 流水。

同一raw window上的scalar-prime leaf不构成上述prefetch：它没有future-iteration、distance、
buffer version或新增SSA，只是一个由完整memory edge和实测parameter共同选择的固定局部序列。

Triton `SoftwarePipeliner.cpp:21-27` 同样分schedule与expander；
`LowerLoops.cpp:49-117`、`AssignLatencies.cpp:172-231`读取async能力、shared encoding、use
distance、mask与MMA wait约束。TileLang `pipeline_planning.cc:1083-1337`读取source stage/order
或producer closure，`inject_pipeline.cc`再版本化buffer并生成同步。

## 8. R：资源仲裁

寄存器预算不是第六种优化，它是P1–P5共享的全局合法性约束。每项候选结构必须计算同时
live的operands、results、partials、indices、masks、conversions、pipeline buffers和fragments。
per-issue capacity 与完整值的实际驻留必须分开：type 上的 time factor 不是已经实施的
分时调度。numeric SSA 的全部物化 parts、显式 alias 的真实最后 use，以及局部 leaf 中
已形成输出与 scratch 的重叠都进入资源账；descriptor/storage projection 不当作数值寄存器。

冲突处理顺序：

1. 数值、axis、Encoding、effect与2.2边界不可让步；
2. 先拒绝没有减少工作的结构（P0）；
3. 先确定P1 carrier与P4 convergence topology，因为它们决定基本live set；
4. P3选择能直接供应该carrier的memory form；
5. P2只在不造成更昂贵spill时扩大共享lifetime；
6. P5最后增加buffer和stage；资源不足时该parameter binding非法，不回头让emitter换结构。

更宽的 load 可能减少静态 load 数，却扩大同时驻留的独立 SSA 集合并触发 spill。
需要联合核算供应与 partial consumer 的 input、temporary 和 output 资源，而不能只比较 load 条数。

Triton也不是资源无限：`OptimizePartitionWarps.cpp:143-253`用rough register estimate调整
partition warps并重新分配layouts。TileLang layout cost以register为次级成本，但最终register
allocation仍交给LLVM/NVCC。Weft必须在intrinsic C之前闭合RVV/IME resource groups，因为系统
compiler不能替Weft重新选择carrier、partial topology或fragment。

## 9. 可复用诊断动作

分析一个慢kernel时，固定执行以下动作，而不是从某个pass名称出发：

1. 从final RISC-V IR列出每个hot Level的axis→time/lane/replica映射与resource peak；
2. 按source origin、logical coordinate和loop trip count建立work ledger；
3. 对每个supply identity统计load/decode/index次数与consumer数；
4. 对每条memory edge统计unit/strided/indexed/segment及地址算术；
5. 对每个contraction/reduction统计partial slots、MAC、reduce、scalar extract与spill；
6. 仅在前五项合理后检查load-to-use距离和pipeline可行性；
7. 与donor逐段对照，先判断差异归作者tree还是physical compiler，再修改代码。

第五步必须把“先形成完整 product 再切片”和“先切 input、逐片 product、再拼回”分开计数。
二者可以有相同的数学乘法数，却有完全不同的 product 指令数、carrier LMUL、slide 数与
live set；只统计最终 reduction 数会漏掉这种 P1/P4 联合错误。

量化 contraction 还必须成对观察 standalone vec-dot 与同格式 MUL_MAT decode。两者的
logical contraction 相同；若优化只提升 blocked prefill，或只在某个 wrapper 中出现，说明
能力仍绑定在 GEMM tree/ABI 上，没有进入共享 vec-dot 底座。只有两项的 final IR 机制计数
与真实吞吐同步变化，才能把收益归给 contraction compiler。

新机制动手前必须回答：

```text
它删除或合并哪些动态工作？静态上限是多少？
它依赖哪些typed facts？有哪些独立输入证据，尚未覆盖哪些关系？
它是否改变logical values、Level、widening或artifact？
它增加多少live register/local/fragment资源？
收益应在IR、C和assembly的哪个计数上出现？
若计数不变，为什么还值得实现？
```

最后一问无法回答时，不实现。

## 10. 可证伪预测

每个新机制在实现前应明确作者前置条件、主要瓶颈类别、预期可见计数及独立输入的证据范围。
没有第二个现成格式不禁止实现一个合同闭合的关系，例如独立 half-byte projection；但一个
正例只能证明该输入闭合，不能宣称跨格式泛化。不得为了一个 donor 把多种关系捆成带隐藏
外围程序的大组合 op。预测与实测结果作为独立运行记录保存；不能事后改写原预测来迁就结果。
结果与预测不一致时，应重新检查工作量归因和适用条件。

预测中的格式名只用于实验定位，不授权按格式名实现pass。真正的compiler规则仍必须以axis、
storage geometry、use-def、effect、target facts和resources为输入。预测表、累计数字和待验证状态
是可变运行事实，不进入稳定设计规范。

## 11. 有限选择与成本估计

合法性与性能排序分开。axis/数值/effect/alias/目标指令合同必须先成立，最终完整资源检查
必须通过；任何估计分数或实测收益都不能覆盖它们。寄存器估计可以帮助排列候选，但不是
最终 live-set 合法性的替代物。成本模型是可解释、可验证的启发式，不是硬件定理。

target 可以声明少量完整的 memory、layout、partial、pack 或 schedule 候选，并以固定规则、
分项成本排序或外部实测进行有限选择。每个选择点与整次编译必须有明确的候选/搜索上限，
相同输入与配置具有确定的 tie-break。模型应公开采用的输入事实、分项估计和选择理由；
缺少所需事实时拒绝该估计，不能用无依据的零值或默认分数掩盖未知成本。

候选由作者/std 或 target 提供：前者的 source 参数可能实例化不同作者树；后者只改变固定
作者树的物理表示。tuner 不发明程序树或任意物理结构，也不限于数字参数，可以绑定 target
已声明的枚举策略。每个候选独立形成一份 Physical program，失败是该候选的可观察结果；
外部继续编译另一个显式候选不等于在 pass/emitter 内静默 fallback。

调优记录至少关联真实 source 文件/函数与 entry、输入表示和 shape、source bindings、target
facts、physical bindings、候选来源与完整枚举域、估计分项、拒绝原因、数值结果、计时与
winner。一个历史选定值、target 默认值或 production 手填值不自动等于完整候选域的 winner。
配置放在 Python、C++ 或 shell 不改变这项归属与证据要求。

仓库示例的 source bindings 与历史 physical bindings 放在对应 kernel 家族旁的
`examples/kernels/*/tuning.json`；runner 只解析该配置、绑定目标环境并编译/运行。
`binding_origin=historical-selection` 不宣称完整搜索的最优值。
公共 tuner 使用 Triton 的有限 `Config` 列表，source kwargs 与 backend_options 独立保存，
不自动从多个独立域生成笛卡尔积。候选枚举、pruning、hook 和计时使用同一 Triton autotuner。
候选仍须通过 C++ target 合法性检查；没有静态估计时只记录真实 native 时间，不伪造成本分数。

`triton.autotune` 负责配置选择，独立 Bench 负责数值验收和明确计时边界的性能比较。
source、target、toolchain 与配置身份随产物保存；试跑保持 readwrite/alias 合同。
历史固定绑定不因一次局部选择被自动改写。完整运行合同见
[运行时与安装](runtime.md#配置与-autotune)。

普通 `for/while/if` 仍按当前语言合同保持有序标量控制。本节不授权从普通迭代发明 shaped
axis，也不随成本政策调整 birth、alias、iteration identity 或 numerical boundary。

## 12. 机械验收的证明范围

parse/verify、指定的 canonicalization/CSE 和二次文本 diff 为零，只证明这个样本在实际
重放的流程上稳定；必须注明样本、输入 artifact 边界和重放 pass 列表。它不证明全部 pass
幂等、整个 pipeline 等价，也不证明生成 C 或 RISC-V 可执行程序数值正确。

final kernel 的 `resources_materialized=true` 会使 layout canonicalization 跳过主要
backward-rematerialization/layout-changing 分支，只保留不改变最终资源合同的 conversion
CSE。因此 final IR 的稳定重放不能代表 resource closure 之前的改写已被重新执行和验证。
layout 改写的重放从 `--emit=riscv-layout-input` 开始：解析后执行
`weft-riscv-propagate-consumer-layouts → weft-riscv-canonicalize-layouts → cse → weft-riscv-eliminate-dead-layouts`，
以固定点 dead-producer 清理消费 CSE 暴露的纯死链，再以 `--resume-layout-input` 运行同一收尾后缀，
重新选择 memory/leaf 并核算资源。应同时记录边界前后的实际改写、重放稳定性与完整编译
结果的比较；不得只观察一个被跳过的 pass。真机数值 repro 仍是独立且必须完成的证据。

也可对已通过 final verifier 的独立副本清除 `resources_materialized`，并将
`vector_register_peak`、`fragment_register_peak`、`local_storage_bytes` 清零，
通过 `triton-opt` 重放 partial-extraction coalescing、consumer layout propagation、layout canonicalization、CSE、dead-layout 清理及原有的
memory/replica-load/hoist/operation-selection/leaf-finalization/read-snapshot/
leaf-resource-closure/CSE/physical-issue-fusion/dead-layout/resource
收尾 passes，最后重新运行 final verifier。原有显式 spill、snapshot 和 selected target op
仍是输入 Physical graph 的一部分，不因清 marker 而删除；已有 scalar-load prime 保留，
不得据此推测新的参数绑定。每轮都重新清 marker 后重放，二次 diff 为零才表示此流程稳定。
若改写留下不匹配的 leaf 或无法重新闭合资源，应报告失败，不能补回旧 marker/统计使其通过。
rematerialization 必须使旧 `leaf`/`implementation` 失效，再由原有选择 pass 闭合新表示。
资源闭合可能使原本属性不同的 leaf 变得相同；若随后再次 CSE/dead-layout 清理，应先清除
final marker 与统计，并在清理后重新核算资源。二次 diff 比较必须包含这一收尾，不能保留旧统计。

pass 贡献只能来自固定作者树及其它 binding 下两份均合法的 Physical program，并实际运行
对应 artifacts。关闭必需 lowering 后编译失败不是性能消融，也不能把多个相互依赖 pass 的
联合收益拆成没有依据的贡献百分比。
