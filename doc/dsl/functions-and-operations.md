# 函数、基本 Operation 与编译配置

## 1. 一门语言、一个用户层级

Weft 的 std 函数与应用 kernel 使用同一门语言：

```text
没有 privileged library IR
没有 opaque whole-kernel template
没有“普通用户 DSL / 专家 DSL”两套语义
```

std helper 是可读、可复制、可修改的普通函数。它们在 canonical Kernel IR 形成前 inline；函数内部的 Level、Value、Encoding access 和 operation 与调用者组成同一棵 SSA 程序。`@weft.derive` builder 在 artifact phase 单独求值并产生派生 Encoding，不进入 invocation kernel 的 SSA 树。

库可以维护多个 overload/特化：

```python
gemv(f32, f32)
gemv(Q4K_I[16], Q8_K)
gemv_panelized(Q4K_I[16], Q8_K)
```

前两者因 operand Encoding 不同而是不同程序；第三者若增加 `stage` 或改变 Level，则也是另一棵作者树。前端根据函数调用、参数类型和静态 source 参数唯一解析 overload。build config 不能替作者在多棵 std tree 中搜索算法结构。

## 2. 没有内置 GEMM

语言核心没有 whole-kernel `matmul` primitive。GEMM、GEMV、quantized vec-dot、attention 和 sort 都是 std 中的普通函数。

GEMM 使用：

```text
Level hierarchy
stage / load / store
state accumulator
dot / reduce_dot
```

GEMM 的局部二维块乘写作 `dot`；GEMV 去掉 column Level，以 `reduce_dot` 对一组行和向量求乘积和。两者都不拥有外围 traversal、blocking、只读复用值、persistent Encoding、state lifetime 或 kernel ABI。

## 3. `auto`

```python
with level.tiles(N, extent=auto("NC")) as nc:
with level.rows(mc, group=auto("MR")) as mb:
```

`auto` 声明由构建过程实例化的有限 source 参数域。`auto("MR")` 在所在函数中引入同名静态符号；同一作用域内同名 `auto` 必须绑定同一个值。std overload 或显式 build config 必须提供非空有限值域。

View annotation 中的 `M/N/K` 等未绑定名称引入同名 shape symbol；同一名称复用同一 logical axis identity。它们由调用 ABI 的 extent 提供，不要求在 kernel 中另写赋值。`auto("MR")` 引入的静态参数则由构建配置绑定，不能把它当作运行时 shape 输入。候选域可以在 std 或构建调用侧，kernel 不必重复列出；缺少必要绑定必须报错，不能从 target 或默认值猜测作者参数。

每组绑定形成一份固定 source candidate。其 Level tree、logical values、births 和 handoff 不再变化；target 只物理化这棵树。资源不足时 spill 或拒绝，不回头修改 source 参数。

target 还可以为已经确定的物理结构公开 LMUL、physical microtile extent、unroll、pipeline depth 和 buffer count 等有限 physical parameters。source `auto` 可以改变作者程序的 cohort 或 Level extent；physical parameter 只实例化同一 source tree。二者可以由同一构建工具实测，但不能混成一类参数。

## 4. 基本 operation 的要求

每个基本 operation 必须唯一规定：

- operand/result 的 element type；
- result logical shape 和 axis relation；
- 被消去或保留的 axes；
- 数值顺序、结合和精度语义；
- overflow/wrap/saturate 行为；
- memory/effect 行为。

operation 不声明 source engine role。target operation contract 另行判断该 canonical operation 能否由 scalar、RVV、IME 或 transfer engine 实现。编译器也不能从普通 SSA closure 识别出未写明的 dot、online softmax、codebook axis 或其它 primitive。

## 5. Pointwise 与类型 operation

```text
unary        neg / abs / exp / rsqrt
binary       add / sub / mul / div / maximum / minimum / bitwise
compare      typed comparison
cast         数值转换或明确 reinterpret
widen        保持逻辑元素与 axes，提升 element width
narrow       保持逻辑元素与 axes，缩窄 element width
```

Pointwise operands 按 axis identity 对齐。缺少某轴的 scalar/singleton operand 可以 broadcast；两个同名 axis extent 不一致是类型错误。`widen` 和 `narrow` 的位置是 canonical 数值语义，target 不能把转换移过会改变 overflow 或舍入的位置。

浮点 `maximum/minimum` 使用 IEEE 754 `maximumNumber/minimumNumber` 语义：单个 NaN 与数字比较时返回数字，两个 NaN 返回 NaN；`maximum(+0,-0)=+0`，`minimum(+0,-0)=-0`。整数版本使用相应的有符号/无符号全序。

`maximum_propagating/minimum_propagating` 仅接受浮点元素，任一输入为 NaN 时结果为 NaN，其余数值及 signed-zero 规则同上；不保证 NaN payload。canonical binary kind 分别为 `maximum/minimum`，原有接口仍为 `max/min`，不能互换。

`select(condition, true_value, false_value)` 按元素选择已经求值的两个 SSA 数值，不是短路控制流。predicate 为 i1，两路元素类型相同；三者按 axis identity 广播，结果轴顺序为 true 分支、false 分支、predicate 中首次出现的轴。

`rsqrt(value)` 接收浮点 scalar 或 shaped Value，逐元素计算平方根的倒数，保留完整
element type、shape 与 axis identity，没有 memory effect。其数值定义是同一 element
格式中的 `1 / sqrt(value)`：平方根和除法分别按 round-to-nearest-even 舍入，不承诺
对实数倒平方根只做一次舍入。`+0/-0` 分别产生 `+inf/-inf`，`+inf` 产生 `+0`，负的
非零输入及 NaN 产生 NaN；不隐式 flush subnormal，也不使用低精度 reciprocal-sqrt
estimate 替代。目标缺少相应 dtype 的合法实现时明确拒绝，不隐式转成其它 dtype。

## 6. Grouped multiply-accumulate

```python
mac_pairs(a, b, into=i16)          # [N] × [N] -> [N/2]
mac_groups(a, b, n=4, into=i16)   # [N] × [N] -> [N/n]
```

grouped result 继承被分组 axis 的 identity，只把 extent 除以 group width；后续 `reduce(..., axis="k")` 仍引用同一 logical axis。operand extent 必须能被 group width 整除。group width 与 `into` 改变 result shape、partial 数量和 overflow 行为，属于作者选择；RVV widening MAC、普通 multiply+reduce 或 extension instruction 是同一 operation 的物理 realization。

## 7. Reduction 与 contraction

### 7.1 `reduce`

```python
reduce(value, op="add", axis="k")
```

`reduce` 消去指定 logical axes，保留全部 free axes。logical reduction domain 必须非空；physical tail 只补 identity，不增加 logical elements。

integer `add` 使用声明的 accumulator overflow 规则；floating `add` 在 closed operation 内允许改变括号次序，但不允许移出 operation/Level boundary；`max/min` 使用与 pointwise 相同的 NaN 和 signed-zero 规则。每个新增 reduction operation 必须定义 identity、空域政策和结合自由度。

源程序只有一个 reduce result；target 可以在 closed operation 内使用多个 physical partial、lane collective 或树归约。这些 temporary 不成为 canonical Value。

### 7.2 `sum_pairs`

```python
sum_pairs(x)       # integer[N] -> i32[N/2]
```

`sum_pairs` 将最后一轴上的相邻两项提升并相加，结果固定为 `i32`。该轴长度必须为偶数，其 identity 与其它轴保持不变。它不是任意 dtype 的可调 reduction，也不改变配对位置。

### 7.3 块乘 `dot` 与乘积归约 `reduce_dot`

```python
dot(a, b, over="k", acc_dtype=f32)
reduce_dot(a, x, over="k", acc_dtype=i32)
reduce_dot(a, b, over=("entry", "payload"), acc_dtype=i32)
```

`dot` 接收两个二维 matrix blocks：`over` 指定双方共有的一条 reduction axis，双方各保留一条不同的 row/column axis。axis identity 决定对应关系，不要求 reduction axis 在两侧具有相同位置。结果轴按左侧 free axis、右侧 free axis 排列。

`reduce_dot` 接收一般 shaped Values，在显式 `over` 指定的一条或多条共同轴上求乘积和；共享的非缩并轴逐点对齐，非共享轴广播。结果先保留左侧 free axes，再添加右侧独有的 free axes。向量内积、matrix-vector、带共享 batch 轴的乘积，以及多轴 decode 支路都使用这一入口。

两者的 `over` 非空、无重复，且每条缩并轴都必须存在于双方；所有共享轴的 extent 必须一致。`acc_dtype` 是累加及结果 dtype，不是初始 accumulator Value；省略时采用 canonical 类型提升规则。它们都是闭合 numerical operation，不能为了改写源码而展开为 `reduce(a * b)`：先在窄输入 dtype 上求乘积可能引入不同的溢出边界。

core 不复制 Triton 的目标尺寸下限，也不保证任意 dtype/shape 都存在 target leaf。RVV/IME 的 operand、extent、alignment、tail 和资源要求由 target verifier 检查；没有合法实现时明确拒绝。名称不指定 engine，不能用 `dot` 请求 IME 或用 `reduce_dot` 强制标量执行。

## 8. Lookup 与显式 index domain

```python
indices = arange(0, 8, dtype=u32)
values = lookup(codebook, base + indices)
```

`lookup` 对每个 index coordinate 读取 table entry；result shape/axes 与 indices 完全相同。越界 policy、mask 与 fill 必须由调用显式给出。八个独立 scalar lookup 不等于一条 `[8]` lookup；target 只能为已经存在的 shaped index 选择 unit/indexed/gather 或专用 table realization。

## 9. Logical projection、subview、reshape 与 transpose

对 shaped Value 的 slice/projection 必须显式保存输入输出 axis relation：

```python
panel = stage(load(B[kc, nc]))
b = panel[kb, nb]
```

`subview` 表示同一个 View 或 slice 内的连续矩形子区域：

```python
tail = subview(Y[kb], offsets=(160,), extents=(80,))
store(tail, value)
```

`offsets` 和 `extents` 以 logical element coordinate 计数，每个 base axis 恰好对应一项；各项可以是编译期整数或显式 scalar integer expression，后者转换为 index SSA。每次执行必须满足 `offset >= 0`、`extent > 0`，且 `offset + extent` 不得越过当前 base extent；静态可判定的违反由 verifier 拒绝，动态条件是调用该 operation 的前置条件，不隐式 clamp、padding 或 copy。结果保留 base 的 Encoding、axis identity、access 与 alias relation，只把 logical shape 缩为当前 `extents`。动态 result extent 来自该次 SSA operand，不等于同名 axis 在 kernel ABI 中的完整长度。`subview` 不改变 Level domain、partition、multiplicity、birth 或 handoff。

`subview` 只定义逐轴 unit-step 的连续矩形，不改变 base stride，也不接受 shaped index、gather 或 scatter。结果可以用于 `load/store`、后续 slice/subview 或 extent 查询；访问权限仍由原 View 的 ABI 决定。多个 subview 是否重叠由作者负责，canonical verifier 不做跨 op 的重叠证明。

`extent(region, axis)` 查询该 View/slice 当前的一维逻辑长度，返回 index SSA；`axis` 是显式、在 rank 内的编译期位置。对 subview 返回该次选中的 extent，对 Level domain slice 返回当前有效范围，不回读原始 View 的完整 shape。它只查询描述符，没有内存读取 effect，也不产生隐式 task/hart identity。

`reshape` 表示 local shaped Value 的显式坐标重排：

```python
flat = reshape(
    tile,
    shape=(160,),
    axes=("k",),
    order=("digit", "k"),
)
```

`shape` 和 `axes` 完整声明输出 domain。`order` 必填，且必须把输入的每个 logical axis 恰好列出一次，按从最慢变化到最快变化的顺序定义输入坐标的线性序号；输出按 `axes` 的声明顺序从最慢到最快解码同一个线性序号。因而 `order=("digit", "k")` 与 `order=("k", "digit")` 是两个不同的 canonical program，编译器不得互换。

`reshape` 必须满足输入和输出 shape 的元素总数相同，元素 dtype 不变；结果 axis identity 由 `axes` 显式给出，输入 axis、`order` 和结果 axis/shape 一起保存在 canonical op 中，使坐标来源可追溯。它不执行 storage access，也不选择 lane、register、fragment、local pack 或 target instruction。

仅重命名 axis 时，输入输出的有序 shape 必须完全相同，`order` 保持输入轴顺序；这种位置一一对应允许保留动态 extent identity，不要求推导一般动态 reshape 的元素数等价性。

Python 入口在这种重命名中接受已有 shape symbol，例如
`reshape(load(X), shape=(M, K), axes=("m", "feature"), order=("m", "k"))`。
symbol 必须对应输入原位置上的同一 extent；动态转置、任意 shape 乘除及负整数字面量不由此获得支持。

`transpose` 和其它显式 index operation 若改变 logical axes、coordinates 或遍历关系，同样必须作为 canonical operation 写入程序，并完整定义坐标映射。它们不是 physical layout hint。

invocation-local pack 不是 core operation。target 可以根据 producer Encoding、all consumers、axis mapping、widening、reuse、pipeline 与 resources，为同一 canonical Value/use edge 选择 register window、local-storage panel、RVV tuple、IME operand 或其它 local pack。这个选择不得改变 Value 的 shape/axes、Level birth、effects 或 pin ABI。

```python
interleave(view, rows=16)
```

`interleave` 只允许出现在 derived Encoding builder 中，生成跨调用可见的 layout family。它改变 artifact bytes 与 ABI，因此属于 source semantics，而不是 invocation-local target 优化。

## 10. Build config 与 target requirement

core DSL 不提供 `@wide`、`@matrix`、`@transfer` 或通用 `stage_handoff`。`scalar/wide/matrix/transfer` 是 target physical machine 的 engine 类别。

如果 build 必须实际使用某项 target extension，可以声明硬 requirement：

```python
CompileOptions(matrix_extension="spacemit-ime1",
               require_extensions=("spacemit-ime1",))
```

requirement 是对 target physical program 的谓词。它先把不能满足要求的 physical structures 判为非法，再验证 selected program；不满足就拒绝 build，不静默 fallback。它不改变 canonical values，也不授权编译器改 source tree。普通 use-def 表达 value consumer，`stage` 表达 logical staged boundary；register↔fragment 或 wide↔matrix 的交接由 physical conversion 表达。

CLI 的对应绑定是可重复的 `--require-extension=spacemit-ime1`；要求列表按集合规范化，
随每个 physical kernel、构建结果和 native cache identity 保存。`matrix_extension` 仅声明
target capability，不证明实际使用。实际使用取自当前 kernel 已选 operation 的 ISA 合同；
仅用 RVV 的 pack/zero/pointwise、其它 kernel 或 artifact builder 均不能满足 IME 要求。
未注册要求、目标不具备能力或最终程序未使用时，在 terminal/system compile 前失败。
已解析的 Physical IR 自带要求，terminal translation 与 layout replay 不允许用 CLI 替换。

Weft core DSL 当前不定义 soft-hint 表面。target schedule preference、cache policy 或 instruction preference若存在，应作为 build/target metadata附着到明确实体，不得进入 numerical operation 或 Value identity。

## 11. 函数边界

普通 inline 函数可以接收/返回 shaped Value、包含 Level 和普通控制、调用其它 std 函数、形成多个 consumer，并由类型和显式静态 source 参数重载。

函数不能隐藏 caller-visible persistent artifact、workspace 或 effect。需要它们时必须出现在 kernel 签名、View 类型或 derive builder 中。
