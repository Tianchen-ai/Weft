# Canonical Kernel IR

Canonical Kernel IR 是 Weft 源程序的唯一算法与数值 authority。它保存作者写下的树，供任何目标后端读取；它不保存某次 target lowering 的机器表示。

本文只规定 canonical 边界。两层 lowering 见[编译器总览](../compiler/index.md)，target-aware 表示见[RISC-V Physical IR](../compiler/riscv-ir.md)。

## 1. 必须承载的内容

### 1.1 Kernel 与普通控制

- kernel 参数、View 类型、access effect 与 alias boundary；
- 普通 `for/while/if` 及其有序语义；
- loop/branch carried SSA values；
- source operation 的 use-def、region nesting 和 dominance。

普通 Triton void kernel 的提前 `return` 在共同前端中结构化：后续语句只放入尚未返回
的分支，仍经共同 scope/SSA 规则形成 `scf.if`。动态条件保留；不靠删除空输入检查来
适配模板。kernel 不能返回数值。

需要原生函数边界的 helper 保存为 `helper` region，`noinline` 明确保留作者的调用要求。
普通表达式、控制流和 helper 调用仍由共同 Triton CodeGenerator 构造。`helper_return`
退出最近的 helper，包括其中正在执行的普通循环和 Level，调用者继续执行。各返回路径
具有相同的 typed 结果签名；未返回路径不能用零值或隐式 None 冒充数值结果。
返回携带独立的 function-control effect，阻止把本来不会执行的访存提前到退出路径。
该边界不改变 Encoding、Value 的数值类型或 Level 的初始化合同，也不创建线程或 workspace。

### 1.2 Encoding 与内存语义

- base/dense/derived family/derived instance identity；
- bit order、byte order、elements、alignment和padding；
- field storage span；
- 源级 `grouped/bit_layers/pack_fields` 声明对应的 storage mapping；
- View/Slice 的 logical shape 和 axis identities；
- derive builder、persistent interleave 与 pinned artifact layout identity。

普通 Triton 指针使用 `pointer<element>` 保存入口存储来源，元素偏移是独立的有符号
64-bit scalar/Value。指针的广播、加法与 reshape 只改变这份偏移；普通控制的 SSA carry
显式携带来源和偏移两部分。它不隐式声明 View shape，不把 tensor 的逻辑 shape 当作允许
读取的范围。`pointer_load(base, offsets, mask, other)` 与
`pointer_store(base, offsets, mask, value)` 保留逐元素 mask、填充值、数值 domain 与实际
base 的读写 effect。只有 mask 为真的元素可解引用；非活动 load 元素返回 other。
来源沿普通控制 SSA 传播，参数 alias/effect 与 View 使用同一合同。
`pointer_atomic` 保存 storage origin、offset、mask、更新值、比较值、操作、ordering 和 scope。
它返回实际更新前的值；mask 为假时不访问内存，其返回值不供作者依赖。
普通 Triton `atomic_add/min/max/and/or/xor/xchg/cas` 使用共同 semantic 的类型与参数检查。
CPU 对 `cta/gpu/sys` 均提供系统范围原子性，不由这些 GPU 名称引入另一套线程所有权。
ordering 保留 relaxed/acquire/release/acq_rel；CAS 失败路径不携带 release 语义。
同 storage 字节宽度的 `pointer_cast` 只重解释元素类型，保留来源、偏移单位和 alias；
它不把指针变成没有存储来源的整数，也不增强地址对齐。

### 1.3 Logical Value

- element type；
- logical shape；
- axis identity；
- pointwise/broadcast/reduction/contraction axis relation；
- 显式cast/widen/narrow及其中间数值类型。

Triton 的 `assume` 保留 scalar i1 前提；`multiple_of/max_contiguous/max_constancy`
形成带 logical-axis 数组的 `value_fact`，指针事实同时携带 origin 和 offsets。
它们是作者对合法输入的承诺，不改变数值、不产生运行时检查，也不授权更换算法。
后端可以保守地不利用这些事实；当前 indexed pointer 访存仍使用完整 offsets 和 mask，
不凭事实标记扩大范围或增强 ABI 对齐。标量 assume 可传给下游 C compiler。

### 1.4 Level

- domain axis、relation和parent identity；
- partition、multiplicity、tail；
- 尚未实例化时的 source `auto` 参数 identity；进入 target lowering 的 candidate 则携带具体绑定；
- carried operands/results；
- `births.state` 与 `births.staged`；
- body region；
- handoff operand、result type和层归属；
- logical cohort width。

Level 不能在canonical化时退化成“一个loop加几个可丢弃attribute”。births和handoff必须是typed region/SSA结构。

### 1.5 Numerical operations

- pointwise operation；
- grouped MAC 的group width与accumulator type；
- reduce、sum_pairs、dot/reduce_dot 的输入、输出与轴；
- lookup/index relation；
- state update与effect；
- operation精度、顺序、overflow/wrap/saturate语义；
- reduce identity、空域政策、NaN/signed-zero与允许的结合自由度。

普通 Triton `tl.reduce` 的联合归约保存为 `reduce_region`，携带非空输入 tuple、
被消去的轴和封闭的纯标量 combiner region。N 个输入共享 shape/axis domain，
region 参数依次为完整左 tuple、完整右 tuple；`reduce_yield` 返回 N 个标量，
各端口保持自身 dtype，输出 domain 只去掉归约轴。combiner 可以组合纯标量数值操作，
不能访问外部 SSA、内存或嵌套控制。结合性由调用者保证；该 primitive 允许选择归约树，
不允许交换元素次序，也不隐式发明 identity 或填充空域。普通单值 add/min/max
继续保存为原 `reduce`，保留已有目标实现。

`scan` 保存相同的纯标量 tuple combiner、axis 与 reverse，结果保留完整输入 domain。
它是 inclusive scan；reverse 精确对应 `flip(scan(flip(input)))`，不能因常见加法可交换而
颠倒自定义 combiner 的左右参数。允许结合但不允许交换次序；没有额外 identity 元素。

同一局部作用域内，浮点乘法的结果若仅被一次同精度加法消费，允许目标将该直接
`mul + add` 对 contraction 成一次 FMA。它使用一次舍入，仍按声明的浮点容差验收；
不授权重结合 scale tree、改变 reduction 顺序，也不跨显式 cast/widen/narrow、Level
或其它可观察数值边界。Canonical IR 继续保存原始两条 operation，Physical pass
根据这项数值许可形成显式 FMA，而不是由 terminal 匹配表达式。

## 2. 按设计不承载的内容

Canonical Kernel IR 不包含：

- target ISA、机器 ABI（例如 `lp64d`）、VLEN或resource budget；
- SEW、LMUL、`vl`或tail loop实现；
- physical lane、register tuple或fragment layout；
- machine register编号；
- selected RVV/IME instruction；
- selected scalar/wide/matrix/transfer engine；
- unit/strided/indexed/segment memory form；
- invocation-local unpack/packing、schema、axis orientation 与 parameterized extents；
- physical layout conversion；
- unroll、prefetch、local pipeline和buffer version；
- live interval、spill、reload、rematerialization；
- intrinsic C或inline asm spelling；
- kernel/operator/quant-format专用backend route。

这些事实属于target candidate的物理程序，不得反向成为语言annotation。

## 3. Canonical verifier 的职责

canonical verifier必须检查程序是不是一份合法Weft程序：

- encoding mapping与storage span完整合法；
- View/Value shape、axis和element type一致；
- domain parent/partition/multiplicity/tail合法；
- Level births/body/handoff签名一致；
- 普通控制carry与branch result类型一致；
- load/stage/state/store 的 domain 与 value 关系一致；
- primitive operand/result shape和axis relation合法；
- `select` 的 i1 predicate 与同 dtype 数值分支按 axis identity 广播，按 true、false、predicate 的首次出现顺序组成结果 domain；
- `subview` 的 offsets/extents 与 base rank 一致；静态数组中的 `-1` 按 offsets、extents 的顺序一一绑定 index SSA operands，结果保留 Encoding 与 axes。静态已知边界须非负/正且不越界；动态范围由该次 operands 定义，不能被同轴的 ABI shape 替代；
- `extent` 的 base 是 View/slice，axis 是 rank 内的静态位置，结果是当前描述符范围的 index SSA；
- `reshape` 的输入 axis order 是完整排列，输入输出元素数和 dtype 相同，结果 shape/axes 与显式坐标映射一致；
- derive builder result与declared family一致；
- pinned layout identity在ABI边界明确。

verifier不负责：

- 证明程序等价于某个上游数学contract；
- 证明作者选择的i16/i32不会溢出；
- 证明`bsum`由正确公式生成；
- 证明某个candidate高性能；
- 选择另一棵stdtree；
- 证明浮点结果与某个reference bit-exact。

## 4. 作者树与closed primitive

canonical IR必须区分：

```text
开放SSA树中的logical values
closed primitive内部不可观察的physical temporaries
```

例如：

- 把一个source accumulator拆成四个canonical state再合并，改变logical value集合，必须由作者写；
- `reduce`后端使用四个register partial并在primitive内合并，不改变canonical operand/result contract，可以由编译器实现。

这个边界由IR可观察性决定，不由“代码里是否出现四个临时变量”决定。

## 5. 与target compiler的交接

目标编译器从canonical IR读取：

```text
typed SSA
logical axes
Level/domain/lifetime
memory/encoding relation
operation semantics
effects
```

并与独立的 target profile/build config 一起，为每个已实例化 std/auto candidate 形成一份 target-aware RISC-V IR module。target profile 提供 engines、representations、legality 与结构规则；build config 提供 target requirement，例如最终 physical program 必须使用 IME。requirement 在结构性选择时参与 legality 过滤，并在 selected program 上验证；二者都不进入 canonical Value identity。

```text
Canonical Kernel IR
    ↓ ConvertWeftToRISCV
RISC-V Physical IR
    ↓ terminal translation
intrinsic C / local asm
```

这是两层 MLIR。target profile、build config、tuner和各个physical pass都是转换输入或对第二层程序的改写，不构成额外IR层。canonical module保持不变；每个candidate从它独立建立一份瞬态RISC-V module。

唯一合法事实必须从这些输入唯一推导；多个合法物理结构可以按 target 声明的固定规则、分项成本或有限实测选择。每个候选独立形成 Physical program，合法性不受成本分数覆盖。若物理实现需要改变 canonical values、Level 归属、artifact ABI 或 numerical operation，它不能在 target lowering 中完成；必须返回到作者 tree 或另一个 std overload。

canonical IR本身不承担目标物理candidate的持久authority。一个candidate被拒绝或替换，不得修改源程序语义；选定layout、conversion、memory、schedule和target operations必须存在于RISC-V IR本身，不能只保存在side record中。

## 6. 源级拼写与 IR operation

源 API 不要求与 IR operation 同名。`load/store` 分别产生带读取/写入 effect 的 `admit/commit`；`state/stage` 产生 `new/materialize`，并保留初始化区域和层归属；`arange/sum_pairs` 对应 `iota/fold2`。Encoding 的 `bit_layers/pack_fields` 保存为 `layered/joined` storage mapping。这些是编译器内部词汇，不是第二套公开 DSL。

块乘 `dot` 保存为 `outer_contract`，包含双方不同的 free axes；一般 `reduce_dot` 保存为 `contract`，保留显式缩并轴、共享 free axes 和 accumulator type。前端不得因为目标性能偏好在两种源操作之间改选，目标后端也不根据源函数名接管数值树。
