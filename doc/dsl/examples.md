# DSL 示例

这些示例展示作者树的语义边界。它们不规定目标 LMUL、physical microtile、pack schema/extent、fragment、load form或pipeline。为突出关系，片段省略 import、decorator 与函数签名中可由上下文确定的静态符号声明；未定义的大写符号均为静态 shape/参数，不是运行时隐式变量。

## 1. Q4_K × Q8_K multi-output vec-dot

下面的程序在一个 16-output cohort 中计算 16 条共享同一 activation block 的 vec-dot，因此完整函数形状是 GEMV。局部数值树仍是 Q4_K × Q8_K vec-dot；scalar 版本去掉 output free axis，并直接接收 canonical `Q4_K`。二者不是由后端在同一 ABI 下互换的实现。

### 1.1 Encoding

```python
@weft.encoding
class Q4_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    alignment = 1
    d:    f16
    dmin: f16
    sc:   u6[8]   @ pack_fields(group=4, fields=2, low_bits=4, order=lo_first)
    m:    u6[8]   @ pack_fields(group=4, fields=2, low_bits=4, order=lo_first)
    q:    u4[256] @ grouped(elements=64) @ bit_layers(elements=32, order=lo_first)

@weft.encoding
class Q8_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    alignment = 1
    ds:   f32
    q:    i8[256]
    bsum: i16[16]

@weft.derive
def Q4K_I(
    W: View[Q4_K, (N, K)], *, rows: static[int]
) -> View[Q4K_I[rows], (N, K)]:
    return interleave(W, rows=rows)
```

`Q4K_I[rows]` 是可由 std 函数静态生成的派生 family；导出 artifact 时 `rows=16` 被实例化并形成固定 layout identity。这里的 kernel ABI 因而接收 `Q4K_I[16]`，不是 canonical `Q4_K`，也不是仍可变化的抽象 family。

### 1.2 Kernel

```python
def q4k_gemv(
    W: View[Q4K_I[16], (N, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (N,)],
):
    # static preconditions: N % 16 == 0, K % 256 == 0
    with level.rows(N, group=16) as nb:
        f32_acc = state(f32, [16], init=0)

        with level.blocks(K, extent=256) as kb:
            w = load(W[nb, kb])
            x = load(X[kb])
            i32_acc = state(i32, [16], init=0)

            with level.subtiles(kb, extent=32) as s:
                # [16,32] × [32] -> [16,16] i16 partials
                p16 = mac_pairs(w.q[s], x.q[s], into=i16)
                # 消去 element-pair 轴，保留 16-output free axis
                sub_sum = reduce(widen(p16, i32), op="add", axis="k")
                i32_acc += sub_sum * i32(w.sc[s])

            # bsum 粒度 16，m 粒度 32
            mins = sum_pairs(x.bsum)
            min_term = reduce_dot(widen(w.m, i32), mins, over="k")

            # ds 属于当前 Q8_K block，必须在 block handoff 处参与
            f32_acc += f32(x.ds) * (
                f32(w.d) * f32(i32_acc)
                - f32(w.dmin) * f32(min_term)
            )

        store(Y[nb], f32_acc)
```

### 1.3 作者写下的知识

| 知识 | 代码位置 |
|---|---|
| min 支路与 q-product 支路分开 | `i32_acc` 与 `min_term` |
| min 支路读取预存 bsum | `sum_pairs(x.bsum)` |
| 16/32 粒度错配 | `sum_pairs` |
| element/sub/block 三层频率 | `mac_pairs`、`level.subtiles`、`level.blocks` |
| i16→i32→f32 位宽层次 | `into`、`widen`、显式 cast |
| 同一 Level 共同产生 16 个输出 | `level.rows(..., group=16)` |
| 持久跨行布局 | `Q4K_I[16]` derive 与 kernel 参数类型 |

目标编译器可以改变 `p16` 的物理 lane/register representation、unpack、归约树和指令；不能删除 min 支路、移动 `ds`、把 16-output cohort改成四个独立Level，或把 persistent layout换成另一个ABI。

### 1.4 三种容易混淆的程序

下列程序不是同一路径的三个后端选项：

```text
single vec-dot
    Q4_K[K] × Q8_K[K] → scalar

persistent blocked MUL_MAT/GEMV
    View[Q4K_I[16], (N,K)]，跨调用已重排，多输出cohort在kernel内

canonical staged MUL_MAT
    Q4_K[N,K]，每次 invocation 内 stage canonical region；target 可选择 local pack
```

它们的ABI、物化次数和Level树不同，必须分别由作者函数表达。

## 2. Blocked MUL_MAT

```python
def mul_mat(
    A: View[f32, (M, K)],
    B: View[f32, (K, N)],
    C: View[f32, (M, N)],
):
    with level.tiles(N, extent=auto("NC")) as nc:
        with level.tiles(K, extent=auto("KC")) as kc:
            # 每个(NC,KC) staged region一次，被所有MC复用
            Bp = stage(load(B[kc, nc]))

            with level.tiles(M, extent=auto("MC")) as mc:
                # 每个(MC,KC) staged region一次
                Ap = stage(load(A[mc, kc]))

                with level.rows(mc, group=auto("MR")) as mb:
                    with level.cols(nc, group=auto("NR")) as nb:
                        # acc位于KC内部：每个KC从C读入并写回
                        acc = state(f32, [MR, NR], init=load(C[mb, nb]))

                        with level.blocks(kc, extent=auto("KB")) as kb:
                            a = Ap[mb, kb]
                            b = Bp[kb, nb]
                            acc += dot(a, b, over="k")

                        store(C[mb, nb], acc)
```

### 2.1 KC 不能省略

`Bp` 在 KC 层诞生，作用域是 `KC × NC`，被该 KC 下所有 MC 复用。没有 KC 层时，无法仅靠注释表达 staged region 的逻辑范围、物化频率和 C 的分段累加。

source 不规定 local pack 的连续方向。target compiler 根据 A/B 的 logical axes、Encoding/address relation、所有 consumers、widening、reuse、resources 与 chosen engine，决定是否建立 local pack 以及 axis orientation。若作者确实要改变 logical axes，必须写显式 transpose/reshape/index；若要改变跨调用 bytes，必须写 derived Encoding。

这里的 `dot` 只规定两个块的数值关系，可以由 RVV register microkernel 或 IME fragment 实现。若 build 必须使用 IME，调用侧声明 `require=uses_extension(IME)`；它不改变 source tree，不满足时 build 失败。若所需实现改变 staged value 或 Level，作者另写一份 std 函数。

### 2.2 Accumulator 作用域是数值决策

上面的 accumulator 位于 KC 内：

```text
每个KC：read C → accumulate current KC → write C
```

若把 accumulator 移到 KC 外：

```text
read/init once → accumulate entire K → write once
```

两者的 C traffic、state lifetime 和浮点累加顺序不同，是两棵作者程序。编译器不能根据cache或寄存器压力在两者之间切换。

量化 MUL_MAT 可以复用 NC/KC/MC/MR/NR 外层骨架，但内层量化树、activation quantize、workspace 和 persistent Encoding 必须由对应 std 函数显式写出；它们不是 `dot` 的隐式后端模式。invocation-local pack 属于同一树的 target physical representation。

## 3. GEMV 是 blocked MUL_MAT 的退化

```python
def gemv(
    W: View[f32, (M, K)],
    X: View[f32, (K,)],
    Y: View[f32, (M,)],
):
    with level.rows(M, group=auto("MR")) as mb:
        acc = state(f32, [MR], init=0)
        with level.blocks(K, extent=auto("KB")) as kb:
            # X block被当前MR行共同使用
            x = load(X[kb])
            acc += reduce_dot(load(W[mb, kb]), x, over="k")
        store(Y[mb], acc)
```

相对 GEMM：

- 没有 N/column Level；
- accumulator 从 `[MR,NR]` 降为 `[MR]`；
- X block 在 K Level 供应一次，被 MR 行共享；
- 没有因为名字叫GEMV而引入新primitive。

若作者改成 `group=1` 并把 `load(X)` 放进每行内部，就是朴素row-dot：X被读取M次。编译器不能把后者识别成前者。

## 4. Online attention 与非归约 handoff

```python
def rowmax(x):
    return reduce(x, op="max", axis="tk")

def rowsum(x):
    return reduce(x, op="add", axis="tk")

def flash_attention(
    Q: View[f16, (Tq, D)],
    K: View[f16, (Tk, D)],
    V: View[f16, (Tk, D)],
    O: View[f16, (Tq, D)],
):
    with level.rows(Tq, group=auto("BQ")) as qb:
        q = stage(load(Q[qb, :]))
        m = state(f32, [BQ], init=-inf)
        l = state(f32, [BQ], init=0)
        o = state(f32, [BQ, D], init=0)

        with level.blocks(Tk, extent=auto("BK")) as kb:
            k = load(K[kb, :])
            v = load(V[kb, :])

            s = dot(q, k, over="d", acc_dtype=f32)
            m_new = maximum(m, rowmax(s))
            p = exp(s - m_new)
            alpha = exp(m - m_new)

            l = l * alpha + rowsum(p)
            o = o * alpha + dot(p, v, over="tk", acc_dtype=f32)
            m = m_new

        out = narrow(o / l, f16)
        store(O[qb, :], out)
```

这份程序确立：

- `q` 是 qb 层 staged value；
- `m/l/o` 是 qb 层 state；
- Tk block顺序来自 old state→state state的SSA依赖；
- `o = o * alpha + contribution` 是先重标定再合并，不是普通 `+=`；
- 一个 Level 可包含两个块乘、多个 reduce/pointwise 和三个 state update。

编译器不能把普通 softmax换成online算法，也不能把该handoff简化成未重标定的累加。

## 5. Top-K 明确不用 Level

```python
def topk(X: View[f32, (N,)], out: View[i32, (K_,)]):
    # static precondition: 1 <= K_ <= N
    heap = state(f32, [K_], init=-inf)
    idx = state(i32, [K_], init=-1)

    for i in range(N):
        score = X[i]
        if score > heap[K_ - 1]:
            pos = K_ - 1
            j = K_ - 1
            while j > 0:
                if score > heap[j - 1]:
                    pos = j - 1
                j -= 1

            for j in range(K_ - 1, pos, -1):
                heap[j] = heap[j - 1]
                idx[j] = idx[j - 1]

            heap[pos] = score
            idx[pos] = i32(i)

    store(out, idx)
```

这个算法虽然有静态的 N/K 边界，但作者程序是一条数据依赖的有序 insertion control；它没有声明 shaped domain/cohort，也不需要跨Level birth或typed handoff，因此普通`for/while/if`就是完整表达。相等分数不触发插入，所以当前程序的 tie policy 是保留先出现的元素；NaN 比较为 false，因此 NaN 不进入结果，全 NaN 输入保留初始化的 `-1` indices。

编译器不会把循环自动向量化。宽Top-K必须由作者或std另写一棵包含显式shaped axis/Level的程序，不能从这份标量tree恢复。
