# Intrinsic C 与 Local ASM Emission

## 1. 输入边界

terminal translator只接收通过`VerifyFinalRISCV`的[RISC-V IR](riscv-ir.md)。此时control、
address relation、layout conversion、window、fragment packing、pipeline、spill与ABI都已经是
typed program entities；每个target operation也已有闭合的[local leaf](leaves.md)。

translator内部可以用临时binding map把SSA value对应到C expression、RVV variable、local array
或typed fragment register。该表只保存已发射值的文本，不是跨pass decision authority。

从已选 RVV carrier 提取静态第 0 个元素时，直接拼写 `vmv.x.s` / `vfmv.f.s`，不先发射
零偏移 `vslidedown`。其它 lane 仍按明确的坐标映射移动；这只消除恒等搬运，不改变读取、
有效域、整数位宽或浮点数值。多消费者的提取共享仍由局部 extraction binding 保存。

scalar unary 与 binary 的数值结果也以其声明 C dtype 建立一次 SSA 绑定。特别是 `exp/log`
不能作为文本宏在重复 use 或内层循环中重复执行；system compiler 可以对这些普通局部值做
等价优化，terminal 本身不把一条已求值的数值 operation 展开成多次调用。

纯表达式可以内联，但 memory-read SSA result 必须在对应的读取点求值并形成稳定绑定。
同一结果的重复 use 不能变成重复解引用，也不能将读取延迟到可能别名的 store 之后。
这项要求同时覆盖 scalar、scalar tuple、encoded field 与 local load；source record 到
field 的投影也必须保留原有读取/effect 边界，不能把描述符字符串当成已经加载的值。
带 `snapshot_storage` 的 encoded load 只按已选 byte-copy leaf 写入对应 local allocation，
随后绑定该私有存储；translator 不自行决定是否复制、复制范围或生命周期。

encoded projection 以新表示供应 numeric value 时，`field_read` 显式给出读取点、
source projection、access 和 result layout；terminal 不在 pure `time_to_lane` 中重新读取。
每个 part 的 active length 来自该 value 的 layout 与对应 `physical_point`，不扫描同 block
其它 result 推导 scope VL。跨 operation 的提取共享由 SSA/CSE 决定；单个 conversion 内
允许复用固定展开中相同的 source part/lane，不把这种临时量缓存扩到相邻 operation。
field/extract 的存储投影绑定独立保留；数值物化不能覆盖它，再让另一个显式 Read op
根据偶然的 binding kind 决定是否读取。projection map 只保存对应 IR 的地址/坐标描述，
不共享已发射的 load/decode，也不决定 layout 或资源。

已选 `vv.broadcast` binary 的单个 op 内允许保留上一个 source part 的局部 gather，坐标公式
与独立 axis-broadcast 相同；不得越过 op 边界缓存它。`lane_repack` 只按两端 type 已确定的
连续切片移动。多轴 pointwise 在完整 carrier 上计算后，显式转换与 memory leaf 共同保证
只有活动矩形可被观察；不能用第一个 lane axis 的 active 值当作整个 tile 的 VL。

数值的 `full` validity 仅声明数值成员已初始化，不扩大写入目标的有效 View。
普通 dense store 的每个 part 同时受目标 descriptor 的实际各轴 extent 约束，lane VL
取数值有效长度与目标剩余长度的交集；私有完整值写入尾块时也遵守该规则。
动态 stride 的连续访问判定由 physical pass 形成显式 `scf.if`，每条最终访存已固定为
unit 或 strided。terminal 不创建这项选择；原 operation 的全部静态 parts 在所选分支内
按原顺序访问。相同循环不变量的判定可以包围完整 physical issue loop。

## 2. 允许的确定 lowering

translator可以：

- 把`func/scf/arith`写成普通C control和scalar expression；
- 根据`ValueType/LayoutAttr`枚举静态time/register parts并选择已确定RVV C type；
- 根据`MemDescType/AccessAttr`打印已确定的pointer arithmetic、stride、gather和bit extraction；
- 根据typed `convert_layout` source/result映射打印固定slide/splat/split/merge序列；
- 根据WindowType与exact leaf展开有限group/term指令序列；
- 根据FragmentPackingAttr和IME leaf打印固定local packing与typed asm；
- 打印explicit local alloc/guard/spill/reload、ABI、headers和declarations；
- 在机器合同不变时，由toolchain adapter选择等价API spelling。

这些工作是TTGIR→LLVM conversion pattern意义上的terminal lowering：输入type/op已经唯一决定
结果。生成多条C语句不代表translator拥有结构选择权。

编译失败的结构化诊断保留实际 stage、pass、已执行的 pipeline 前缀，以及失败时的
`failed_ir`。它只供定位当前 pass 的部分改写状态，不能作为合法 checkpoint 重启或部署；
成功编译不生成这份副本。
Clang 编译或链接失败使用同一结构化异常，标记 `native-compilation/clang` 与
`toolchain-error`，保存实际命令、工具链身份和退出码。该错误不能作为非法调优候选
被跳过；跨框架编译进程传递异常时保留完整 report，而不是只传格式化字符串。

## 3. 禁止的决定

translator不能：

- 补layout、lane、part、stride、access、resource或capacity默认值；
- 从canonical op、kernel/格式名、VLEN、target型号或source closure选择RVV/IME；
- 改LMUL、lane/register mapping、memory form、operand swap、fragment或packing schema；
- 生成source Level、contract reduction loop、动态local-pack loop或pipeline结构；
- 决定spill/reload/rematerialize、workspace或persistent ABI；
- 同时读取Canonical Kernel IR与side assignment合成physical program；
- 调用GGML、仓库外reference code、legacy helper或fallback。

translator若发现binding/type/leaf之间不一致必须失败。这表示前序pass或verifier合同遗漏，不能在
这里加一个default或第二selector。

## 4. System compiler边界

Weft已经决定logical-to-physical mapping、target-local instruction sequence和loop-local schedule。
system C compiler继续负责最终register allocation、machine scheduling、peephole、constant
folding与machine code。生成C中的局部pragma可以阻止system compiler破坏已选physical形态，
但不能要求它重新发现SIMD、microkernel或pipeline。

Weft 的驻留分析覆盖其 numeric SSA、已选局部临时量与显式 storage；system compiler 仍可能
因额外 hoist、寄存器组约束或调度产生 spill。IR 的 local bytes 为零不是最终机器码无栈访问的声明。

## 5. 可调用 artifact

同一次 terminal translation 同时生成 C prototype 与 `RISCVKernelABI`。ABI 的参数顺序、
Encoding identity、record span、logical extents、alignment、access 和 alias groups 均来自
已验证的 kernel/MemDesc，而不由 runtime 解析 C 文本或按格式名补齐。
`RISCVCompilationResult` 保存代码和这些 ABI；Triton native bindings 将它们交给 backend 的 C→native stage，并由 CompiledKernel 导出。
动态 shape 参数顺序与实际 prototype 一致，source binding 与发现的 target 随产物保持明确。
加载器负责系统编译、句柄与调用生命周期，不参与 leaf 或资源选择。

每个入口参数明确区分 scalar 与 View，并记录 scalar dtype 或 View 的 dense dtype。
scalar 按其 C ABI 值传递，access 为 `none`、alias group 为 `-1`，不伪装成内存描述符。
公共 header 直接采用 compiler 生成的 declaration；Python 使用同一份 typed ABI 绑定 ctypes，
不从 C spelling 反推宽度。Python 不支持的 by-value 类型明确拒绝，不以整数位模式冒充浮点 ABI。
显式参数名不能与隐式 shape 参数冲突。

普通 Triton pointer 参数在 typed ABI 中标记为 `pointer`，保留 dtype、Encoding、access
和 alias set，shape 为空。它仍按原生 typed pointer 传参；Python/SDK 借用调用者的实际
storage offset、shape 与 stride 来保持存储生命周期、对齐和 alias 校验，不要求这些 metadata
与一个推定的 View rank 相同。地址偏移、访问范围与 mask 来自作者程序，runtime 不隐式
复制、repack 或补 stride 参数。SDK 不从 pointer 参数推断输出分配大小。

artifact service 同时生成机械的 `dispatch.c`，根据以上 typed metadata 读取通用调用槽并
调用显式提供的 native entry。这个 stub 不选择算法、layout、leaf 或参数，只服务 C/C++ SDK
的普通 CPU 调用；其函数指针来自当前 artifact，不能通过全局同名符号选择另一个 specialization。

ABI 分别保存 `matrix_extensions`（target capability）、`required_extensions`（用户硬要求）
和 `used_extensions`（当前 kernel 已选指令实际依赖的扩展）。actual-use 查询与 final
verifier 共享同一 ISA 合同来源，不解析 C 文本、不用 engine 标签代替。运行时核对硬要求与
实际使用的一致性，并对实际使用的扩展继续执行原生 CPU/affinity 合法性检查。

自然布局的 scalar field read 将已验证的访问对齐在读取点传给 system compiler；
注解只保留访问对齐、record byte stride 与读取宽度共同保证的二次幂因子。
交错字段还必须满足其 chunk 不拆散该 scalar。它不增强 kernel 参数的 ABI 对齐、不改变空 View 的调用条件，不把 byte storage
转成更强别名类型，也不根据字段或 kernel 名选择读取算法。

## 6. 实现模块边界

`lib/Target/RISCVIntrinsicC.cpp` 保留唯一公开 translation 入口；`Emission/Control.cpp`
与 `Emitter.h` 管理共同的 ABI、C control 和 SSA binding。`Memory.cpp`、`Encoding.cpp`、
`Layout.cpp` 和各 `RVV*.cpp` 分别实现存储、编码、表示与已选 RVV 局部操作，均为独立编译单元。

`Emission/IME.cpp` 拼写已选的局部 raw 字段读取、bit projection、pointwise、pack/unpack 与 typed fragment binding 交接；
`IME/FragmentEmission.cpp` 按已选 atom 的注册归属分派到 `Extensions/<Module>/Emission.cpp`，
后者经有限的 fresh-name/line 接口输出局部 C/asm。合同和终端实现由同一构建期模块表生成，
不接管公共遍历或 ABI。packing/MMA 私有存储与 clobber 的合同来自同一目标描述，不能在
发射时增加。IME1 的 operand 为一个 vector group，accumulator 为两个 group；M1 只暴露第一行，
仍保留完整硬件 accumulator 资源。MMA 通过 tied、early-clobber register operand 更新，不隐式
清零或写回数组；初始化和最终转换必须在 IR 中存在。普通 control 的同型赋值同时服务 RVV 与
fragment，但不因此把 fragment 改为 RVV ValueType。
