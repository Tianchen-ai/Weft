# Encoding、View 与 Artifact

## 1. Encoding 是纯布局

Encoding 定义：

> 一个 logical field coordinate 怎样映射到 storage unit 和其中的 bit range。

它不定义 dequant 公式、scale 语义、zero point、不变量或某条 ISA 指令。

```python
@triton.encoding
class Q4_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    alignment = 1
    d:    f16
    dmin: f16
    sc:   u6[8]   @ pack_fields(group=4, fields=2, low_bits=4, order=lo_first)
    m:    u6[8]   @ pack_fields(group=4, fields=2, low_bits=4, order=lo_first)
    q:    u4[256] @ grouped(elements=64) @ bit_layers(elements=32, order=lo_first)

@triton.encoding
class Q8_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    alignment = 1
    ds:   f32
    q:    i8[256]
    bsum: i16[16]
```

一份完整 Encoding 必须给出：

- bit order；
- byte order；
- `elements`：一条 storage record 对应多少个逻辑元素；
- field 的逻辑 shape 和 element bit width；
- field storage span；
- grouping、layering 或 join 关系；
- alignment；省略时语义等于 `alignment = 1`；
- 非空 padding 的位置与内容要求；省略时语义等于没有 padding。

`elements` 不能从最大字段长度猜测。radix bytes、grid indices 或 metadata field 可以只覆盖 record 的一种存储分解，但整条 record 仍对应另一数量的逻辑元素。

字段声明顺序与自然排列、`grouped`、`bit_layers`、`pack_fields` 必须能唯一确定每个 field 占据的 storage span。构造参数全部以名称给出，避免把元素数、字段数、bit 数和顺序混在位置参数中。关系不完整时 Encoding 非法，不以隐式 padding 或猜测 offset 补齐。

`alignment` 是一条 serialized storage record 起始地址必须满足的最小对齐，不是某个 host C struct 的自然对齐，也不是外层 tensor allocator 的对齐。目标代码若要把地址转换成具有更强 C ABI 对齐要求的对象，必须在该使用点另行证明要求成立。

## 2. 基础布局关系

### 2.1 `natural`

逻辑元素按顺序占据自然 storage unit。Q8 字节数组和普通 dense scalar field 属于这一类。

### 2.2 `grouped(elements=N)`

`elements` 是一个重复存储组覆盖的逻辑元素数，不是 byte 数。每组独立应用后续 bit 布局；字段长度必须能被它整除。该名称只说明分组，不承诺 SIMD 分组或数值分组归约。

### 2.3 `bit_layers(elements=P, order=...)`

存储组内每 P 个逻辑元素形成一个 bit 层，不同层相同位置的元素占据同一 byte 的不同 bit range。`elements` 是每层元素数，`order` 指逻辑层从低位还是高位开始；它不是字节序。当前构造要求 `P` 整除组长，并且元素位宽乘层数恰为 8。

例如：

```text
Q4_K.q = grouped(elements=64) @ bit_layers(elements=32, order=lo_first)
```

唯一表示：

```text
group = i // 64
within = i % 64
storage byte = 32*group + (within % 32)
storage bits = low nibble  if within < 32
               high nibble otherwise
```

这个公式是布局关系的规范解释，不是要求作者在 kernel 中手写 byte/nibble 地址。

### 2.4 record-local field axis 与 Level point

Encoding field 的 shape 建立 record-local logical axes。一个覆盖该 record element axis 的 Level point 可以用来选择 field：

```python
with level.subtiles(extent=32) as sub:
    q32 = block.q[sub]   # 选择当前 32-element logical region
    sc1 = block.sc[sub]  # 选择与该 region 对应的一个 record-local field item
```

后一种投影只有在 `elements / field_extent == Level partition` 时唯一成立；其 field coordinate 是当前 point 在 record 内的 ordinal。选择会消去这个 record-local field axis，并保留其它 free axes：对单条 record，`block.sc[sub]` 是 scalar；对 16-row cohort，它是 `[16]`。若比例关系不唯一，作者必须显式给出 index/reshape relation，前端不能根据字段名或使用位置猜测。这个规则只定义 logical indexing，不声称 `sc` 是 scale，也不赋予 field 数值不变量。

### 2.5 保留但不公开的 `bit_planes`

全格式布局中确实存在“一个逻辑值的不同 bit 片段位于独立重复 storage plane”的关系，但 `bit_planes` 尚未形成唯一的 source type、verifier 与 lowering 合同，因此不是公开 DSL 构造。

在合同闭合前，Q5/Q6 等格式把 low/high bits 声明为真实、独立的 storage fields，由普通数值函数显式组合。前端不能接受半成品 `bit_planes` annotation，emitter 也不能根据格式名恢复 plane 关系。公开它的条件不是格式数量，而是同一声明能唯一决定 storage mapping，并有不依赖格式名的真实 lowering consumer。

### 2.6 `pack_fields(group=..., fields=..., low_bits=..., order=...)`

`pack_fields` 声明连续的 F 个逻辑字段共用一个紧凑 byte 容器，而不是将它们各自对齐存放。`fields=F` 指共同声明的字段数，`group=G` 指每个字段的半段元素数，因此各字段长度为 `2G`；`low_bits` 指尾段每个字段占据的低位片段宽度，`order` 决定字段片段排列方向。全部 F 个字段必须连续出现并使用相同参数。

当前规则要求 `low_bits * F == 8`，每个字段的位宽与其补充片段共同构成一个 byte，完整容器占 `G * (F + 1)` bytes。缺少字段、参数不一致或存在未覆盖 storage bits 都必须拒绝。它描述已有 bytes 的解释，不执行 pack kernel，也不表达 scale/min 的数值关系。

Q4_K 的 `sc/m` 使用：

```text
pack_fields(group=4, fields=2, low_bits=4, order=lo_first)
```

它说明两组 6-bit logical values 如何共用 12 bytes；不说明 scale/min 在数学公式中如何使用。

### 2.7 `padding`

padding 明确声明无逻辑 field 对应的 storage range。未声明的空洞不是隐式 padding；Encoding 必须完整覆盖 record storage。

## 3. Encoding 不包含的知识

下面关系都不属于布局：

```text
bsum[j] == Σ x[16j:16j+16]
sc 是 scale
m 是 minimum correction
q 应先减 zero point
某个 field 应由 gather 读取
某个格式应使用 IME
```

如果 kernel 要使用 `bsum`，作者显式读取它。如果派生 artifact 要生成 `bsum`，builder 显式计算并写入。编译器不能从 `q` 推断或替换这些字段。

同一 Encoding mapping 可以由 shift/mask、gather、segment 或专用 unpack 实现；这些是目标物理 realization，不改变 Encoding。

## 4. View

```python
View[Encoding, shape]
```

View 是带 Encoding 的内存对象。它规定：

- 逻辑 shape；
- 逻辑 axis identity；
- 每个逻辑元素所属的 storage record；
- 可观察的 encoding/layout identity；
- kernel 参数或外部 artifact 的内存边界；
- 参数级 access effect 与 alias relation。

`View[f32, (M, N)]` 是 dense Encoding 的简写。dense 仍然有明确的 element type、shape、axis 和 ABI，不代表“没有布局”。

kernel 默认接收连续 row-major dense storage。`@triton.jit(model="weft", strided=("X", ...))` 为指定 dense
参数声明运行时逐轴 element stride，保持逻辑 shape、axis、数值程序和 Encoding 不变。
stride 非负；只读 View 可以用 zero stride 表达 broadcast，可写 View 不允许自重叠。
storage offset 由传入的首元素地址表达。声明作为 `strided_args` 同时存在于 canonical/physical
kernel ABI，不能由目标编译器自行增加。packed/derived Encoding 不接受这项 dense stride 合同。

对 record Encoding，View 的 shape仍按 logical elements 计数；storage size 由 record span 与 `elements` 推出。若逻辑 extent 不能被 `elements` 整除，Encoding/derived builder 必须定义 tail record 与 padding，或前端拒绝该 View；不能假定越界 bytes 可读。

View 不等于一个已加载的 tile。对 View 做 slice 得到逻辑 region；`load` 才把 region 供应给当前 Level。

参数之间的 alias relation 是 kernel ABI 的事实，而不是允许某项优化的权限位。默认情况下，
所有未声明的 View 参数属于同一个 may-alias 组；编译器不能假设它们互不重叠。调用方能保证
关系时，可以在 kernel 定义上声明命名等价类：

```python
@triton.jit(model="weft", alias_groups={
    "weights": "weights",
    "activation": "activation",
    "output": "output",
})
def gemv(
    weights: View[Q4_K, (N, K)],
    activation: View[Q8_K, (K,)],
    output: View[f32, (N,)],
):
    ...
```

相同组名表示参数可能互相 alias；不同组名表示调用方承诺对应内存对象不重叠。省略的 View
仍留在共同的保守组中；任一显式命名组也承诺与这个省略组不重叠。因此只命名一个参数，
就表示它和所有未命名 View 都不重叠。alias group 不改变 View 的 Encoding、shape 或 logical axes；
它只进入 Canonical Kernel IR 的参数合同，并原样传播到 physical memory descriptor。违反该承诺
是调用方错误，target pass 不得自行拆分或合并组。

## 5. Base、derived family 与 derived instance

三种身份必须分开：

```text
base encoding
    Q4_K：调用方已有的 canonical packed record

derived family
    Q4K_I[rows]：由 build/load phase 生成的一类跨行交错表示

derived instance
    Q4K_I[16]：静态参数和具体 byte layout 已固定的 artifact layout identity
```

派生 family 由同一语言中的 builder 定义：

```python
@triton.derive
def Q4K_I(
    W: View[Q4_K, (N, K)], *, rows: static[int]
) -> View[Q4K_I[rows], (N, K)]:
    return interleave(W, rows=rows)
```

`Q4K_I` 是抽象 family；`Q4K_I[16]` 是 builder 静态实例化后具有确定 byte layout 的类型。方括号是派生 Encoding 的类型生成语法，不是 ordinary value indexing。`derive` 是类型与 artifact 生成，不是 kernel 最外层循环。

`@triton.derive` 将被装饰函数的名称同时引入 Encoding 类型命名空间，因此函数的 result annotation 可以引用正在定义的 family constructor。它不是运行时递归调用。

`interleave(W, rows=R)` 保留每一行的 base Encoding，并把每个 R-row cohort 写成 byte-major 顺序：对 record-local byte coordinate `b`，依次存放 cohort 中各行的第 `b` 个 byte，再进入 `b+1`。上面的 builder 要求行 extent 能被 `rows` 整除；需要 tail cohort 时，builder 必须另行显式声明 padding/layout，不能由 artifact runtime 默默补齐。

`interleave(W, rows=R, field_chunks={name: bytes, ...})` 可以显式声明每个字段的交错粒度。
该静态字典必须完整覆盖字段，字段须 byte-addressable，chunk 必须整除其 storage span；共享
同一 storage span 的字段必须选择相同 chunk。字段区间与显式 padding 形成完整、无重叠的
record 划分，padding 仍按单 byte 交错。对起始 byte 为 `F`、chunk 为 `U` 的字段，cohort
内第 `r` 行的 record byte `b` 对应 `F*R + ((b-F)/U)*U*R + r*U + (b-F)%U`。
省略 `field_chunks` 等价于全字段 `U=1`，仍是同一交错关系。chunk 属于作者声明的 persistent
layout identity，不是 target 可以自行修改的物理参数。

std 函数可以对 family 的静态参数进行重载或生成函数；导出 kernel 或 artifact 之前，参数必须实例化为固定的 layout identity。调用方 ABI 看到的是 derived instance，不是仍可变化的 family 参数。目标 profile 不必成为 identity 的一部分；多个目标可以接受同一实例，但不能各自私改它的 bytes。

## 6. Artifact phase 与 invocation phase

```text
artifact phase
    build、load 或模型初始化期间执行
    生成持久 packed object、size、alignment、endianness、layout identity

invocation phase
    kernel 每次调用时执行
    消费固定 View/layout identity
```

持久重排必须由作者程序显式声明，因为它改变：

- kernel ABI；
- 内存容量和 alignment；
- 初始化成本；
- 跨调用生命周期；
- 可与哪些 target artifact 兼容。

一旦调用方传入某个 derived instance，kernel lowering 不得改选另一 byte interleave。invocation 内部的 `stage` 只建立 staged Value 的 birth、Level 与复用范围；target compiler 可以为这个 Value 选择短生命周期 local pack，但 pack 的存在、方向和 physical schema 都不进入 canonical DSL，也不形成跨调用可见的 Encoding identity。

## 7. Storage ownership 与 lifetime

Weft 不用硬件 storage scope 区分对象，而用调用边界、Encoding 和逻辑生命周期区分：

| 对象 | 源程序表示 | 所有权与生命周期 |
|---|---|---|
| 普通输入/输出 | kernel 参数中的 `View` | caller 持有；在 invocation 边界可见 |
| caller-visible workspace | kernel 参数中的可写 `View` 与 effect/alias 约束 | caller 分配；kernel 在一次调用内按作者树使用 |
| persistent packed object | derived-instance `View` | artifact phase 生成；跨调用保留 |
| invocation-local staged Value | `stage(expr)` | 在声明它的逻辑作用域诞生，供后代复用，调用结束前消亡；可被 target 表示为 local pack |
| primitive-private temporary | 不进入 DSL | target realization 内部产生和销毁 |

workspace 不是靠参数名识别的特殊 pointer。它的 Encoding、shape、alignment、读写 effect，以及与其它 View 的 alias 约束都是 kernel 签名的一部分；函数不能私下要求一块签名中不存在的 scratch memory。

## 8. Pin boundary

下列表示跨越调用边界，因此是 pin：

- kernel 参数；
- caller-visible workspace；
- persistent packed object；
- returned builder metadata。

pin 的 Encoding/layout identity 必须由两侧共同知道。编译器不得在 pin 处静默插入 coercion；若需要跨调用转换，作者必须把 builder 或 workspace 写入程序。invocation 内的 physical conversion 只能作用于已经进入 kernel 的 Value，不改变 pin bytes。

primitive-private temporary、register pack 和 fragment 不跨调用边界，不进入 DSL 或 artifact metadata。

## 9. 布局正确性

Encoding 验证针对离散 storage 事实：

- 字段 storage span 不重叠；
- bit/byte order 完整；
- `grouped/bit_layers/pack_fields` 的映射在声明域内；
- padding 完整覆盖空洞；
- derived builder 的 result family 与声明一致；
- consumer 的 pinned layout identity 与 artifact 一致。

这类验证可以要求逐 byte 或逐 field 一致。它与浮点 kernel 的容差判据是两件事。
