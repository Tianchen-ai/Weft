# 编程模型非目标

本附录列出不属于 Weft 编程模型的构造。每项都给出语义原因，不记录设计过程。

## 1. Worker、hart 或 program tile 作为根对象

Weft kernel 没有隐式 worker/hart identity，也不以“一个 program instance拥有一个tile”为语义。并行调用和任务分配属于调用方。

若把worker设为根，Level lifetime、Encoding和numerical tree会被迫依附于一个并不存在于所有target上的任务分解；同一kernel也无法自然表示一次调用内的完整有序控制。

## 2. Grid/SIMT ownership 作为根布局

语言不提供program id、thread id、warp/CTA owner或logical coordinate→thread owner映射。目标后端可以把shaped axis映射到它的物理执行资源，但该mapping不是source semantic。

否则CPU/vector/matrix目标必须先模拟一个SIMT ownership，再恢复单控制器的register/lifetime结构。

## 3. VLA region作为整门语言的根

宽执行只能来自显式shaped Value、Level cohort或closed primitive。语言不把所有程序包进一个VLA region，也不把普通scalar loop自动向量化。

VLA是RVV实现能力，不足以表达Encoding、state birth、staged lifetime、非归约handoff或普通不规则控制。

## 4. Block/tile作为统一源类型

Value有logical shape和axis，但不是目标tile：它没有owner、register layout、shared scope或静态2的幂限制。

把tile设为根会把某一种target物理对象提升为所有kernel必须采用的source结构；Weft只在target physical program中选择这些对象。

## 5. Stream + resident state作为根模型

stream、resident state、summary state和ordered carry可以描述部分kernel能力，但不能统一Encoding、persistent artifact、GEMM staging、ordinary control和多支路numeric tree。

Weft直接使用Level births/staged/handoff和普通SSA，不把其中一种常见组合命名为特权kernel类别。

## 6. 硬件storage scope进入DSL

不提供：

```text
alloc_l1
alloc_cache
alloc_vector_register
alloc_ime_fragment
```

CPU cache通常由硬件管理，物理register由下游compiler分配，fragment属于target realization。作者表达的是logical lifetime、materialization、workspace和persistent layout，不是它们最终落在哪级硬件。

## 7. Permission bit

不提供：

```text
allow_reassociation
allow_repack
allow_low_precision
allow_matrix
```

permission没有独立语义。若一种变化会改变logical tree，它必须成为另一个operation、type或stdtree；若不改变，它本来就在编译器表示权限内。精度和结合性必须写进具体operation语义，不使用全局许可。

这不等于成熟 DSL 不能有 assumption、schedule metadata 或 target hint。Weft 的选择是把 target requirement 放进 build config，把 physical organization 放进 target machine；core DSL 不用一个 `allow_*` 位混淆这些层次。

## 8. 通用 `ordered` 属性

ordinary control本身有序；state/attention等顺序由SSA use-def和effect明确；closed primitive需要的结合/顺序由primitive定义。

一个没有消费者、只用于阻止本来就禁止的tree rewrite的`ordered`属性不增加语义。

## 9. 从普通SSA识别高级primitive

编译器不从：

```text
mul + add + reduce
max + exp + sum
固定数量lookup
相邻同构operation
```

识别dot、matmul、online softmax、codebook axis或量化格式。需要这些语义时，作者调用显式basic op或std函数。

这避免kernel-name、exact-op-count、source-closure matcher成为隐藏语言。

## 10. Invocation-local `pack(along=...)`

core DSL 不提供只保持数值、shape、axes 与 Level 不变的 local-pack operation。`stage(expr)` 已经表达 staged Value 的诞生层、物化次数、lifetime 和复用域；local pack 的存在、连续方向、carrier 和 schema 属于 target physical representation。

若重排改变 logical axes，作者写真实的 reshape/transpose/index operation；若改变跨调用 bytes 与 ABI，作者写 derived Encoding。两者都不需要模糊的 pack 授权点。

## 11. Source engine role 与 `stage_handoff`

core DSL 不提供 `@wide`、`@matrix`、`@transfer` 或通用 `stage_handoff`。它们不改变 canonical operation/value/effect graph 或 Level 归属，只限制 target realization，因此属于 build config 和 physical machine。

`scalar/wide/matrix/transfer` 仍是 target engine 类别。若 build 必须使用 IME，写 `require=uses_extension(IME)`，不满足则失败。若 IME 需要另一棵 materialization/Level tree，作者选择另一份 std 函数；跨 engine 的 register/fragment/local-storage 交接是 physical conversion。

## 12. Built-in whole-kernel matmul或quant op

GEMM/GEMV/quantized MUL_MAT 不是 opaque language op。它们由同语言 std 函数组织 outer traversal、Level、stage、state 和局部 dot/reduce_dot。persistent packing 通过 derived Encoding 表达；invocation-local packing 由 target 物理化。

允许的closed primitive只拥有局部operand/result numerical relation，不能拥有kernel ABI、outer traversal、persistent layout或workspace。

## 13. 两套用户DSL

不区分“普通用户语言”和“专家kernel语言”。std作者、应用作者和新格式作者使用相同Encoding、Value、Level、control和basic ops。

库可以提供更高层普通函数和重载，但它们没有隐藏IR、特殊verifier或后端捷径。

## 14. 自动规范化作者tree

编译器不把作者的blocking、staging、accumulator lifetime、persistent packing、普通control或algorithm variant重写成标准模板。

等价source spelling可以通过相同typed facts得到相同物理类别；实现这一点依赖SSA/axis/use-def分析，不依赖把source tree改成统一形状。

## 15. 全局物理开关或全局约束求解器

SEW/LMUL属于value，instruction属于op，memory form属于memory edge，pipeline属于Level/local cluster。它们不是kernel-global C/D字典。

唯一事实由实体的 typed use 推导；多个合法物理实现可以按 target 固定规则、分项成本或有界实测选择。冲突用显式 physical conversion 表示；资源不足时执行已定义的 spill 或拒绝当前 candidate。有限局部比较不授权带回边的全局 solver 修改作者 tree。

这条属于语言与target compiler的职责边界；具体physical IR和pass结构不在本部分定义。

## 16. 用成本分数替代合法性

不允许成本估计覆盖数值、effect、alias、target instruction 或最终资源合同。可解释的局部成本模型可以比较多个合法结构，但必须公开输入事实、分项估计与选择理由；估计不是硬件定理，也不是实际延迟测量。

缺少估计所需事实时，不能以隐式默认值伪装成精确成本。模型与搜索边界由[物理优化方法](../compiler/optimization-principles.md)规定。

## 17. 无界或来源不明的结构搜索

禁止 tuner 发明任意作者树或没有 target 合同的物理结构。target 可以声明有限候选，由固定规则、成本排序或外部真机运行选择；每个选择点和整次编译必须有明确预算，一个 module 只保存一个已选程序。

可测量的配置包括作者/std 声明的 source `auto`，以及 target 声明的数值参数和枚举策略。二者候选来源分别记录，实际 entry、输入表示和选择结果必须可追溯。候选失败不能触发隐藏 fallback。

## 18. 第三层 Physical IR 与 side-record planning

Weft 的程序 IR 只有 Canonical Kernel IR 和 target-aware physical IR 两层。非SIMT物理机器是第二层的语义，不单独形成`weft_phys → weft_riscv`两级lowering；target profile、build config、tuner和pass stage也不是IR层。

物理决定不能只保存在`problem/assignment`字典或其它并行side record中，再让emitter同时读取canonical program与记录首次合成physical program。layout必须进入physical value type，conversion/memory/local object/pipeline/spill必须成为真实operation或region结构，pass结果必须能从改写后的RISC-V IR直接观察和验证。
