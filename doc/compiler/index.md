# 编译器总览

## 1. 两层 MLIR

Weft 的程序 IR 只有两层：

```text
Weft Kernel IR
    logical Value、axis、Level、Encoding、operation、effect
        ↓ ConvertWeftToRISCV
Weft RISC-V IR
    layout、conversion、memory、schedule、RVV/IME leaf
        ↓ TranslateToIntrinsicC
intrinsic C / local asm
        ↓
system C compiler
```

第一层对应 Triton TTIR，第二层对应 TTGIR。非 SIMT 物理机器是第二层的语义，不是第三层 IR；target profile、build config、tuner 和各个 pass 也不是 IR 层。RISC-V module 可以混用 `func`、`scf`、`arith` 与 `weft_riscv` operations，dialect namespace 不增加层次。

当前 RISC-V physical compiler 的正式目标范围是带完整 `V` 扩展且构建时显式给出正 `VLEN` 的 RV32/RV64 profile；`Zve` 子集、纯标量 target 和未知 VLEN 会在创建 RISC-V IR 前明确拒绝。scalar physical value/leaf 是同一 RVV kernel 内的表示与操作，不构成另一条 scalar backend。

## 2. 两层的 authority

完整 `V` 的 VLEN 必须为 128–65536 bits 范围内的二次幂；CLI profile 与 Physical IR target
verifier 使用相同的合法域，不能以任意正整数冒充真实 RVV target。

[Canonical Kernel IR](../dsl/canonical-ir.md) 是唯一持久作者程序，保存 logical tree、Level/lifetime、Encoding/ABI、ordinary control 与数值语义。target pass 不能增加 canonical Value、Level、effect、persistent artifact 或算法阶段。

[RISC-V IR](riscv-ir.md) 是一份瞬态、target-aware、可改写的 physical program。layout、conversion、memory form、local object、pipeline、resource handling 和 [RVV/IME leaf](leaves.md) 必须存在于这份程序中，不能只保存在 side record。

## 3. Candidate 与 tuner

一份 RISC-V module 对应：

```text
一个已实例化的 canonical/std candidate
× 一个 target profile
× 一组 physical parameter bindings
× 一个按有界 target 选择规则选定的 structural realization
```

source `auto` 在进入 lowering 前绑定。结构固定后的 LMUL、microtile extent、unroll、pipeline depth 等有限参数，每组绑定各自建立并编译一份 RISC-V module。depth=1表示顺序执行；当前两阶段 pipeline 只接受 depth=2/buffer=2，并且必须能从一个带 carry 的 physical Level 中推导出 pure/read producer 与 carry-dependent consumer cluster，buffer count不是独立 binding。当前 compiler API 每次只接受一组单值 binding；外部 tuner 可以枚举这些 module 并比较可执行 artifacts，但不共享 `assignment` 字典，也不生成新的物理结构。

结构性选择可以采用固定优先级、可观察的分项成本，或由外部 tuner 实测 target 声明的有限候选；预算、合法性与记录要求见[物理优化方法](optimization-principles.md)。一个 module 内不保存备用路线。选定结构在后续 legality/resource pass 中失败时该 module 直接失败，不在 pass 或 emitter 中回退。显式枚举下一个候选时，从同一 canonical candidate 重新建立 module。

## 4. 编译器文档

- [RISC-V IR](riscv-ir.md)：physical values、memory descriptors、conversion、local objects 与 schedule entities。
- [编译 Pass](passes.md)：同一份 RISC-V IR 上的分析、选择与真实 rewrite。
- [非 SIMT 物理优化方法](optimization-principles.md)：从生成 IR/C/汇编分类动态工作、定位 carrier、supply、memory、reduction 与 pipeline 缺口。
- [Local Leaf](leaves.md)：RVV intrinsic、IME/opaque asm leaf 的合同与选择。
- [Terminal Emission](emission.md)：已选 leaf 和已物化程序怎样确定地写成 C/asm。
- [非 SIMT 物理机器](physical-machine.md)：time/lane/register-replica/fragment/local-storage 的机器语义。

## 5. 明确排除

Weft 不采用：

- `weft_phys → weft_riscv` 两级 physical lowering；
- `weft_riscv.problem` / `weft_riscv.assignment` decision dictionaries；
- canonical program 加 side records，再由 emitter 首次构造 physical program；
- 同一语义的新旧 physical 路径、compatibility layer 或 fallback；
- final emitter 中的 layout/default/selector/source-closure reconstruction。

未来 target 实现同一非 SIMT 机器合同时，替换第二层 target-aware physical IR，不插入第三层，也不要求另一份 source DSL。

## 6. 原生 runtime 与编译产物

原生入口在 RISC-V Linux 进程中通过 `weft.compile(definition, options=..., toolchain=...)`
立即编译，或由 `weft.jit(...)` 在首次调用时编译；二者复用同一编译服务与进程内、持久缓存。
`CompileOptions` 分开保存 source `meta` 和有限 physical bindings，`Toolchain` 明确指定
Weft/system compiler 及构建 flags。JIT 不发明作者树、physical structure 或跨调用 workspace。

有限局部 tile 策略经 CLI `--local-tile-strategy`、C++ `RISCVCompilerOptions.localTileStrategy` 与
Python `CompileOptions.local_tile_strategy` 绑定，取值为 `target-default`、`atom-partitioned` 或
`register-grouped`。请求保存在 Physical KernelOp；artifact 同时保存 `local_tile_strategy` 和
按实际 unpack 实体统计的 `local_tile_realizations`。显式请求必须实际应用，不能作为可忽略
hint；它与 ISA capability、extension-use requirement 和 source meta 分开。该字段进入完整
compiler arguments/cache identity，不同策略不能复用错误 artifact。Physical IR replay 或
terminal translation 读取原 IR 的绑定，不允许 CLI 再覆盖。

`weft-compile --query-native-target` 在原生进程中取得 Linux 可用 ISA、进程 ABI 和实际
`vlenb`，并核对允许执行的 CPU 集合具有一致 VLEN。发现的硬件事实在进入 lowering 前成为
明确 target profile；它不意味着一份二进制自动跨不同 VLEN。不能取得必要事实、向量状态
不可用或执行 affinity 超出已发现集合时明确失败；每次调用还检查当前线程的向量状态，
不能用 exec 后编译子进程的状态替代它。标准 ISA 与 vendor matrix capability 分开发现；
`matrix_extensions` 为每个已确认的矩阵扩展记录支持它的 CPU 集合。厂商 capability 的
注册事实必须同时匹配每个 CPU 的 ISA feature、vendor/architecture identity、XLEN 和 VLEN，
不能仅根据机器名称、同型号 CPU 或固定 core 编号假定支持，也不把未知 vendor token 加进标准 `march`。
请求矩阵扩展时，当前 affinity 必须是对应能力集合的子集；编译前、产物加载前和每次调用均核对，
不静默改绑 CPU 或退回另一引擎。必要事实缺失、扩展未支持或执行集合不满足时明确拒绝。

`--emit=artifact` 返回同一次编译的 Physical IR、C 与 typed kernel ABI；ABI 描述入口、View
Encoding、逻辑 shape、storage record、`interleave_rows`、alignment、access/alias、动态 shape 参数顺序和
所声明 target 的 `matrix_extensions` 要求。loader 核对这些要求与编译绑定及实际执行能力一致。
这些是编译结果，不是第三层 planning IR，也不参与重选 lowering。原生服务再编译、加载
shared object，通过该 ABI 调用；同一 kernel 定义、canonical 程序、target、toolchain 与完整
绑定复用产物，不将不同 Python 定义的调用签名混入同一缓存对象。
`CompiledKernel.options` 保留绑定，`metadata` 只读；`close()` 等待活动调用结束并释放加载句柄，
持久 artifact 的生命周期独立于句柄。完整公开合同见[运行时与安装](runtime.md)。

Runtime `Buffer` 借用连续 Python buffer storage；dense buffer 可由其格式和 shape 得到
Encoding，packed buffer 必须显式提供逻辑 shape 与 Encoding identity。调用检查 storage
大小、record 与已声明 row cohort 的完整性、alignment、可写性和已声明 alias 关系，不偷偷复制、repack 或分配
签名之外的 workspace。动态 extents 按 ABI 输入，不强制每个 shape 产生一份编译产物。

开发机生成 C、目标机编译运行仍是可显式调用的部署方式，与原生入口共用 compiler API；
它不是原生调用失败后的自动 fallback。原生使用不依赖 SSH 或实验 runner。
