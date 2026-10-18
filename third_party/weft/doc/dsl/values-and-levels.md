# Value、Level 与有序控制

## 1. Value

Value 是带 element type、逻辑 shape 和 axis identity 的局部 SSA 值：

```text
Value[element, shape, axes]
```

Value 没有 physical owner、物理 lane、LMUL、register group、fragment 或 hardware storage scope。其可见性来自 SSA dominance、本层初始化与层结果。标量是 rank-0 Value；正 rank Value 的 axis identity 决定 pointwise、broadcast、reduce、dot/reduce_dot 和 memory relation。

Value 可以：

- 被多个 consumer 使用；
- 进入 pointwise、reduce、dot/reduce_dot、lookup 或 memory operation；
- 作为 Level state 跨迭代更新；
- 通过 handoff 交回外层；
- 在普通 `for/while/if` 中作为 SSA carry；
- 被 stage 为本层后代共享的 staged value。

structured operation 的结果仍是普通 Value，不形成只能 direct-store 的封闭路径。

## 2. 四个生命周期构造

### 2.1 `state`

```python
acc = state(f32, [MR, NR], init=0)
```

`state` 声明当前作用域中的可更新数值状态，可以由后续赋值、元素更新或内层控制继续更新。作用域可以是 kernel root 或 Level；把声明放到另一层，会改变初始化次数和跨迭代更新范围。它不分配某类物理寄存器或内存。Top-K 的 heap/indices 也是 state，不限于求和 accumulator。

`init` 是数值语义。`init=0`、`init=load(C[...])` 和未初始化不是可互换实现。

### 2.2 `stage`

```python
Bp = stage(load(B[kc, nc]))
```

`stage(expr)` 在本作用域求得一个供后代只读复用的值。与 `state` 不同，后代不能重新绑定这个值；与普通 load 不同，它显式声明一个有本层归属的复用边界。该边界进入 canonical IR，不是可忽略的优化 hint，也不承诺 cache、shared memory、物理复制或 pipeline stage。target 在保持 dtype、shape、axes、初始化频率与层归属的前提下选择其表示。

Level 顶层直接赋值的 `state(...)` 和 `stage(...)` 分别属于两个初始化区域，执行顺序是 state 初始化、stage 初始化、Level body。各初始化区域只能使用进入 Level 前已可见的值、当前 Level point，以及本区域中此前定义的值；不能引用 Level body 中新定义的值，也不能引用另一个初始化区域的新定义。源码中把 body 赋值写在前面不会改变这项限制。例如 `w = load(W[kb]); panel = stage(decode(w))` 在同一 Level 顶层不合法；应在初始化表达式中直接给出其输入，或从外层传入值。诊断必须点名 body 值或另一初始化区域，不能伪装成未知 Python 名称。

这些限制定义当前初始化区域的可见性，并不授权把 body 的读取、写入或计算复制到初始化区域。初始化中的同名变量按该区域可见环境解析，不能借书写位置假定它来自 body。普通 inline helper 也不建立跨区域捕获通道。

### 2.3 `load`

```python
x = load(X[kb])
```

`load(region)` 从 View region 读取一个 Value，后续使用的是这次读取的结果，而不是可以任意重新解引用的地址表达式。把读取放在更内层会增加逻辑读取次数，放在更外层会扩大生命周期；跨可能别名的写入移动读取必须有合法性证明。

### 2.4 `store`

```python
store(C[mb, nb], acc)
```

`store(region, value)` 将 Value 写入第一个参数指定的 View region。它不提交事务、同步其它执行者或触发隐式 barrier；目标区域、覆盖域和与其它读写的顺序属于源程序。

## 3. Level

Level 表达一段层级数值合成程序。它至少包含：

```text
domain
    当前 Level 覆盖的逻辑域及 axis identity

partition
    当前实例与父域的划分关系，包括 tail

multiplicity
    每个父实例中有多少个当前实例

births.state
    在本层由 state 诞生的状态

births.staged
    在本层由 stage 形成、供后代复用的值

body
    可以包含多棵数值子树、普通控制、多个 state update 和 effect

handoff
    body 的结果如何成为外层可见的 SSA value
```

Level 相比普通 `for` 的核心新增语义是 typed lifetime、层归属和 handoff；domain partition、multiplicity 与 tail 使这一层的逻辑覆盖范围可被精确保存。

Level 不是“一层一个 reduce”，也不是 target loop 的别名。同一 Level 可以包含多个块乘或乘积归约、pointwise、状态更新和 memory effect。

## 4. Domain 与 partition

常见 domain 构造：

```python
with level.tiles(N, extent=NC) as nc:
with level.rows(M, group=MR) as mb:
with level.cols(N, group=NR) as nb:
with level.blocks(K, extent=KB) as kb:
with level.subtiles(kb, extent=32) as s:
```

`level` 是构造逻辑层级的命名空间。`rows/cols` 指一组逻辑行或列，`tiles/blocks` 指指定宽度的域划分，`subtiles` 指父域内的进一步划分。这些名字不指定 lane/thread/fragment，也不要求嵌套某一种固定层级。第一个参数可以是 shape symbol 或父 point；省略时使用当前父域，示例优先写出父 point。

`extent` 或 `group` 定义当前实例覆盖的逻辑宽度；`multiplicity` 由父域范围和 partition 得出；不能整除时，tail 仍属于同一逻辑 domain。

编译器可以 strip-mine 一个 Level 的物理执行、分多次 issue 或使用 tail mask，但不能把一个 source Level 改成多个独立 source Level，因为 births、load 和 handoff 的次数会变化。

## 5. 逻辑 cohort

```python
with level.rows(M, group=16) as mb:
    acc = state(f32, [16], init=0)
```

一个 Level instance 逻辑上共同产生 16 个输出。这 16 个输出：

- 可以共同消费在本层发生的一次 `load` 或 birth；
- 共享 births/staged scope；
- 具有同一 handoff 边界。

`group=16` 不表示 16 个输出必须同时驻留在 16 个物理寄存器中。目标编译器可以把 cohort 分成多次 issue、spill/reload 或多个 fragment；但它不能把源程序改写成四个 `group=4` Level。

## 6. Handoff

### 6.1 Level handoff

表面语法使用普通 SSA 赋值：

```python
acc += block_term
m = m_new
o = o * alpha + dot(p, v, over="tk")
```

在 canonical Kernel IR 中，Level body 的终结必须保存每个 carried/state result 的名称、类型和来源。作者不需要学习独立 handoff 代数，但层归属不能从 IR 中消失。

handoff 不限于 `+=`。attention 中：

```python
o = o * alpha + contribution
```

先重标定旧 state，再加入当前 block，是一个可观察的非归约 handoff。

### 6.2 跨 physical engine 的 value use

```python
def local_decode(
    packed: Value[u4, (K,), ("k",)]
) -> Value[i8, (K,), ("k",)]:
    return i8(packed) - i8(8)

panel = stage(local_decode(packed))
acc += reduce_dot(panel, x, over="k")
```

`local_decode` 是保持 K axis 的普通 helper。`panel` 由 `stage` 建立的本层复用边界与普通 SSA use-def 已完整表达 source-visible value boundary；core DSL 不再增加 `stage_handoff` 或 source engine annotation。

若 target 让 producer 与 consumer 使用不同 physical engine，它在具体 use edge 上插入 register/fragment/local-storage conversion，并承担相应 ordering 与 resources。若同步、并发或 workspace 会改变 source-visible effects，则必须定义真实的 effect operation 或另一棵作者程序，不能由 engine convention 暗中补出。

## 7. 普通控制流

```python
for i in range(N):
    if predicate:
        ...

while condition:
    ...
```

普通 `for/while/if` 是有序标量控制：

- 迭代按程序顺序发生；
- 编译器不把迭代自动改成 shaped axis 或 SIMD domain；
- loop/branch carry 使用普通 SSA；
- effect、alias 和依赖顺序必须保留；
- 常量折叠、地址提升等不改变逻辑值集合的优化仍合法。

Top-K、heap update、依赖前一迭代的搜索和 CSR traversal 可以只用普通控制，不必使用 Level。

## 8. `arange` 与显式逻辑轴

```python
lane = arange(0, 8, dtype=u32)
within_k = arange(0, 32, dtype=u32, axis="k")
```

`arange(start, end)` 建立半开区间 `[start, end)`，两端必须是编译期整数，且 end 大于 start。当前 dtype 为 unsigned integer 或 index，默认 u32。未指定 `axis` 时建立匿名 logical axis；`axis="k"` 复用已有同名 axis，或建立新的具名 axis。它显式给出 field gather、bit 组合、lookup 与多轴乘积归约所需的 shaped index，不承诺任何 RVV lane 位置。

它与下面程序不同：

```python
for lane in range(8):
    value0 = lookup(table, index0)
    ...
```

八次 Python/DSL 标量展开只产生八组独立 SSA operation；不存在一个 8-element logical axis。编译器不得根据相同 op count、同构 source closure 或相邻语句重新发明轴。

若算法要求 codebook 的 8 个元素成为一个 shaped value，作者必须用 `arange`、shaped indexing 或返回 shaped Value 的函数显式写出。

## 9. 顺序性

顺序性来自 ordinary control、SSA use-def 和 operation effect，不使用通用 `ordered=True` 权限位。

例如 online softmax：

```text
m_new depends on m
alpha depends on old m and m_new
l_new depends on old l and alpha
o_new depends on old o, alpha and current contribution
```

这些依赖已经唯一规定合法顺序。若一个 reduce/scan primitive需要额外结合或顺序语义，它必须写入该 primitive 的定义，而不是成为全局 hint。
