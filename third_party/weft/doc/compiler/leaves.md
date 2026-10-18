# RISC-V Local Leaf

## 1. 边界

local leaf 是 final RISC-V IR 中已经选定、合同闭合的 target-local operation。它可以是一条
RVV intrinsic、一个固定短指令序列、一个 IME fragment operation，或 ISA规定的固定 opaque
asm sequence。

leaf的完整合同不是一个 instruction字符串，而是：

```text
physical op kind
+ operand/result physical types
+ layout/access/conversion/window/fragment attributes
+ LeafAttr exact instruction、spelling、mask/tail与resources
```

translator可以根据这些 typed facts展开一个有限、唯一的指令序列。它不能在多个layout、
memory form、microtile、fragment、engine或pipeline之间再选一次。

leaf不能隐藏 canonical Level、outer traversal、blocking、state、source materialization、
caller workspace、persistent Encoding、dynamic pack loop、pipeline或完整operator ABI。固定序列的
slots、terms、trip count、effects和resources若不能由op合同静态闭合，就必须继续拆为真实IR。
primitive-private C array或fragment scratch也必须写入leaf的`local_bytes`，并由resource pass计入
kernel local-storage上界；不能只在terminal translator中出现。

## 2. RVV leaf

一对一RVV operation由leaf固定 intrinsic family，value type固定SEW/LMUL/`vl`/register parts，
access或conversion固定operand form。terminal lowering只把这些事实投影为Clang RVV spelling。

`rvv.convert.i16-f32` 按源 signedness 选择 `vfwcvt.f.x.v` 或 `vfwcvt.f.xu.v`。
转换保持逻辑坐标与 active lanes，结果 SEW/LMUL 为输入的两倍；i16/u16 的每个整数均可
由 f32 精确表示。该局部转换不改变前后整数运算或浮点累加的边界。

一般的 `rvv.convert.i<width>-f<width>` 同样固定源/目标位宽与 signedness：同宽使用
`vfcvt.f.x[u].v`，加宽一倍使用 `vfwcvt.f.x[u].v`，减半使用 `vfncvt.f.x[u].w`。
支持的浮点宽度为 16/32/64，源整数宽度和目标能力必须满足相应指令合同；LMUL 按位宽
同比例变化而逻辑 lane 不变。整数直接转换到目标浮点精度，不经中间浮点精度再次舍入。
半精度转换和 8-bit→f16 加宽依赖 [RVV/Zvfh 转换指令](https://docs.riscv.org/reference/isa/unpriv/v-st-ext)。

普通浮点缩窄的 leaf 固定源/目标宽度及显式 rounding；RVV 使用相应 `vfncvt`，显式
舍入使用带 rounding 参数的 intrinsic，scalar 使用带 rounding 字段的 `fcvt`。f64→f32
和 f32→f16 均保持原逻辑坐标、active lanes 和半宽 LMUL 关系。普通浮点转同宽整数
采用 `vfcvt.rtz.x[u].f.v`，对数值转换合同内的输入向零截断，不借用浮点默认舍入。

byte/boolean→f32 与标准 16/32/64 位整数向 8/16/32 位的无饱和截断由 composite pass
分解成显式转换操作；非标准有限位宽不套用这项逐次减半规则。中间整数
Value 的布局和活跃区间参与共同资源规划。它们不隐藏在 emitter 中，也不跨越作者的
有限位宽边界；有符号/无符号整数截断保留低位，最终 signedness 通过显式重解释表达。

f32 `log` 可以选择 `rvv.log-approx-f32` 的闭合数值序列。正次正规数先精确乘 `2^23`，
拆分 exponent 与 mantissa，将 mantissa 归约到 `[sqrt(1/2),sqrt(2)]`；随后用
`z=(m-1)/(m+1)` 的奇次级数到 11 次及分裂的 `ln(2)` 重建结果。级数来自
[DLMF 4.6.4](https://dlmf.nist.gov/4.6.E4)，不调用外部向量库。
零返回负无穷，正无穷保持，负数和 NaN 返回 NaN。输入/输出之外的 temporary 上界为
八个按 LMUL 缩放的数据载体及三个 mask，数值遵守声明的浮点容差；不宣称正确舍入。
该 leaf 只处理已经给定的 active parts，不拥有遍历、缓存或模型结构。

f32 scalar 的 `exp/log` 同样可选择内联数值 leaf：`scalar.exp-approx-f32` 与
`scalar.log-approx-f32`。exp 用分裂的 ln(2) 做范围归约，以 8 次 Taylor 多项式求局部值，
再以两个正常二次幂恢复比例，覆盖 subnormal 和溢出；级数见 [DLMF 4.2.19](https://dlmf.nist.gov/4.2.E19)。
log 使用与 RVV 相同的归约与奇次级数。两者保留 NaN/Inf/零的数值规则，不保证正确舍入。
固定 scalar FMA、bitcast 和指数缩放直接内联；不调用 libm，因而不会因为普通调用 ABI
破坏外层仍活跃的 RVV 值。[标准调用约定](https://riscv-non-isa.github.io/riscv-elf-psabi-doc/#_standard_calling_convention)
中的全部 vector registers 都是 caller-saved。f64 数学仍使用其独立的 scalar leaf。

`rvv_compare_select` 将仅供一个 select 使用的比较与 merge 闭合成局部序列。四个输入与
输出共享同一 RVV domain；比较 RHS 和 true 分支可以是严格同元素类型的 scalar，
对应 `vv/vf/vx` 比较和 `vvm/vfm/vxm` merge form 写入 leaf。比较 LHS 与 false 分支为
相同的 RVV type，必要 broadcast 在前方以 `rvv_splat` 显式形成。
比较 mask 直接进入 merge，不先扩成 byte predicate 再重新比较；额外资源恰为一个 mask。
需要保留或另作数值用途的 predicate 仍是独立 Value，这项选择不复制比较，也不扩大读范围。

`rvv_assemble_value` 将至多四个同 SEW/LMUL 的完整 issue carrier，按同一逻辑轴的
issue-time 顺序拼成一个数值表示。它只建立寄存器 part 绑定，没有算术、访存或临时寄存器；
其它坐标必须为单位维度。输入 pieces 完整覆盖输出，驻留资源仍按所有 time parts 计入。
与 `rvv_issue_slice` 配合时，可直接供应已在寄存器中的小型数值状态，不能借此重算 producer。

`rvv_multiply_high_scalar` 表示无符号 u8/u16 vector 与同宽 scalar 乘积的高半部分，
结果完整 type 必须等于 vector 输入，leaf 固定为 `rvv.vmulhu.vx`。它没有 rounding、
saturation、outer loop 或私有 scratch；active lanes 与全部 time/replica parts 仍由输入
layout 确定。数值语义为 `floor(unsigned_product / 2^SEW)`，遵守
[RVV 单宽整数乘法合同](https://docs.riscv.org/reference/isa/unpriv/v-st-ext)。
它是独立的 selected local op，不扩大普通 BinaryOp，也不授权 scalar fallback。

`rvv_byte_gather` 从连续 byte/word descriptor 或一个完整、byte-aligned natural word field
读取指定的原始字节，返回 unsigned u8 RVV value。byte offsets 是显式的 u16/u32 value，
所有 index/result axes 和 active lanes 必须一致，EEW/EMUL 与 indexed-memory 能力由
verifier 核验；leaf 固定为 `rvv.byte-gather`。它只读取所选字节，不在 emitter 中恢复
word、shift 或 mask。encoded value 的原 load/快照身份仍经 field operand 保留。

`rvv_byte_windows_load` 读取同一 storage 的一至四个连续字节窗口，按显式 `byte_bases`
operand 顺序拼成一个 full-valid signed/unsigned RVV byte value。每个 base 是相对 storage
的同宽 unsigned u16/u32 字节偏移；base 的计算与 wrap 已在 scalar SSA 中完成，窗口内读取
连续 `window_bytes` 字节，不再隐式做窄整数 wrap。窗口宽度与 result 的完整 lane 数必须闭合。
由 indexed read 选择此 leaf 时，pass 必须证明每个窗口内部不跨原 index 的 wrap 边界；不同窗口
可以来自不同动态 base，不要求 storage 地址有序或互不相交。leaf 固定为 `rvv.byte-windows-pack`：逐窗口 `vle8`，后续窗口以 tail-undisturbed
`vslideup` 拼接，不能增加读取字节、对齐要求或外围遍历。多窗口只保留一个额外载体，
其完整寄存器组数必须计入 leaf temporary；scalar bases 与原 storage/快照身份仍显式存在。

grouped MAC和encoded dot允许作为closed sequence：reduction loop已经在IR中，
window type固定slots/terms/result parts，access固定storage geometry，step leaf固定widen/MAC
sequence。改变group、unroll、lane operand或memory form会产生另一份physical op，而不是
emitter分支选择。

普通 `rvv_contract_step` 是纯数值 scalar/vector FMA；两端所有 reduction 坐标已由前置
projection 消去。完整 free-axis part/lane 映射由类型确定。两个单 use 的 signed-i16→i32
widen 可以与同宽整数 step 合成 `rvv.vwmacc.vx`：输入保持 i16，accumulator/result 为 i32，
lane 坐标一致且 LMUL 精确加倍，目标必须声明 widening integer 能力。其整数结果遵循
原乘加的有限位宽合同。存储解码仍是独立 op；step 不消费 descriptor 或隐式增加读取。

`project_reduction_operand` 从完整 numeric value 投影一个显式 reduction 坐标，保留所有
free-axis 的 time/lane/replica 映射。除 RVV lane/time 之外，它也支持 scalar tuple 的
time/replica 坐标；共享寄存器坐标分析确定每个结果 part 的来源，不增加 memory read。

`fma` 是普通逐元素浮点融合乘加，当前 leaf 为 `scalar.fma`、`rvv.vfmacc.vv/vf`。
它只消费同精度 scalar 或相同布局、相同有效域的 numeric carriers；不包含转换、读取、
归约或调度屏障。scalar 发射显式 FMA builtin，RVV 发射已选 intrinsic。

`rvv_integer_mac` 使用 `rvv.vmacc.vx` 实现同宽整数的模乘加。vector、accumulator 与 result
具有相同的完整 type，multiplier 是相同元素类型的 scalar；每个 part 保留原 active lanes。
它没有 widening、saturation、读取或额外临时载体，不能跨过显式整数精度边界。

两个同型整数 operand 分别扩到双倍位宽后相乘，可以选择已有的 `rvv_widen_multiply`。
选择要求相同 signedness、完整有效域、相同坐标分解与合法的双倍 SEW/LMUL；不跨越
其它 cast/narrow，也不复用仍被其它节点观察的 widen 结果。

`extract` 的 `rvv.extract.regular-scalar` leaf 消费一个完整的 rank-1 register table。
`index_pattern` 保存 base/stride/repeat，另有可选的 unsigned scalar base operand；坐标分析
必须证明每个结果 carrier 内的 index 相同。leaf 参数保存每个 carrier 的相对 index，
translator 直接发射 `vrgather.vx`，保留动态 base 的整数宽度与加法 wrap，不构造 index vector。
layout 改写后重新核验 carrier 映射并重选 leaf；最终 IR 不能留下未选择的 leaf。

`convert_layout`同样是terminal physical op。source/result type和`ConversionAttr`唯一决定
split/merge/slide/splat/extract；translator实现该映射不等于重新推导layout。
若完整 carrier 的切片恰好对齐目标的整数 LMUL 子寄存器，`lane_repack` 使用 `vget/vset`
拼写切分与合并；共享坐标分析同时证明 VLEN、完整覆盖与切片边界，资源合同不再计入
不需要的 slide scratch。其余切片保留显式 slide 与对应临时量。

普通 BinaryOp 的 `rvv.<instruction>.vv.broadcast` leaf 可消费轴严格嵌入结果的两个 RVV
operand。共享轴的 time/lane/replica 必须一致，缺失 lane 轴由 type 决定的 index/gather 在该
局部 operation 内展开；不同 source part 依次消费，相邻相同 part 可复用同一次局部 gather。
它不包含其它数值 operation 或外围循环。最多64个 result parts，逐 part 计算已形成输出、
仍需保留的输入、index/gather 与 carrier extension 的联合驻留；只有 SSA 最后 use 才能回收
输入。独立 axis-broadcast 与该 leaf 复用同一个坐标拼写，跨 operation 共享仍必须在 IR 中。
独立 gather 的输出不得与输入/index 重叠，资源计数同时保留这些载体；约束来自
[RVV vector gather 合同](https://docs.riscv.org/reference/isa/v20260120/unpriv/v-st-ext.html)。

`rvv.mul.row-scale` 只实现单个普通 f32 multiply：左输入为完整行 scalar replicas，右输入为
列 RVV value，结果为行/列二维成组 carrier。行数为 2 或 4，每 carrier 至多 32 lanes；共享
列轴和 group 映射必须一致，列 carrier 与每个 atom 的窄 carrier 均须完整且合法。每个 atom
先在较小 carrier 上逐行作 `vfmul.vf`，再用 tail-undisturbed slide 拼接行对，以 typed subgroup
insert 拼回成组布局。不会用多次宽 masked multiply 反复经过整个 tile，也不需要 scale gather。
该 leaf 不包含 cast、add、FMA、额外读取或其它数值节点。窄乘积、行对、已拼接 atom 和先前
结果的联合临时量由同一合同重算，只有输入最后 use 才能回收其载体。

`rvv.splat.local` 与 `ime.zero.local` 分别是已经选定的 scalar splat 和 fragment zero 的
局部生成方式。typed asm 使 system compiler 保留该生成点，不将它们提升成长期存活的
零模板。它们仍是独立 Physical operations，不把初始化藏进 MMA，也不引入用户 memory effect。
`ime.zero.local-prefix` 的 leaf 参数另行固定初始化载体的 LMUL。只有逻辑 cells 恰好填满
一个更窄的合法载体时才可选择；局部 asm 清零该前缀，结果与 MMA 仍保留完整硬件 fragment
类型及寄存器资源。非逻辑行不能成为可观察结果，不为它们复制或初始化整组零模板。

`rvv_masked_negate` 在 mask bit 为 1 时对 signed integer data 取负，否则保留 data；
result 的完整 type 不变，逐 lane 保留该整数宽度的运算边界。mask 来自同一读取点的
`rvv.bitmask-window-mask`，shape、axis、time/lane/replica mapping、active lanes 和
SEW/LMUL mask ratio 必须与 data 一致。leaf 固定为 `rvv.masked-negate`，每个 physical
part 只拼写同宽 `vneg` 与 `vmerge`；一个额外 data carrier 计入 temporary groups。
它不包含数据 widening、浮点 scale、reduction 或额外读取，也不在 translator 匹配表达式。

`rvv_regular_repeat_scalar_load` 的 natural、repeat=1 形式也可供应完整的多维 scalar
window。source field 是一维连续 i8/i16/i32 storage；`source_count` 等于结果 shape 的
完整乘积，`part_bases` 按结果 layout 的坐标映射到行优先线性偏移。每个源元素恰读一次，
结果必须 full-valid，不能以部分 physical extent 代替完整 shape。它复用普通 scalar 字段
读取，保留原 base 的求值宽度后再作地址偏移，不包含 vector extraction 或数值计算。

`field_read` 是 Read-effect 的 terminal transfer：input 保留 encoded field/projection 的
原 load/快照身份，access 固定读取与 bit mapping，result 是具有完整布局的 RVV/scalar numeric SSA。
它可以读取同一表示，也可以直接供应已选的另一表示；不能把一个已经计算的 numeric value
追溯回 storage 后重读。pure conversion 不承担这项存储供应职责。

`encoded.field-read.record-element` 读取非交错记录中同一 payload 坐标：payload 的最后一轴
被一个显式 scalar index 消去，保留轴按已选 record stride 进入 RVV lane。当前接受 byte 内
完整的 unsigned 1/2/4/8-bit natural 或低位优先 grouped/layered 元素；typed projection、完整
field bit 数、字节边界、布局和临时载体均须闭合。leaf 只做已选 `vlse8` 与 shift/mask，保留
实际 active lanes，widen/MAC 位于后续独立数值 op。跨 byte、joined 或交错记录遵守各自合同。

`encoded.field-read.record-gather` 将同一记录间 stride 用于已投影的 scalar 索引组；
每个 RVV part 使用一个明确的 payload 坐标，当前覆盖自然布局的 signed/unsigned byte。
对 gather 结果沿归约轴取值时，physical pass 先投影索引张量，不能把归约坐标加到地址上。

`encoded.field-read.indexed-bytes` 消费 byte-interleaved record 的显式 RVV 索引，
按固定交错 stride 读取 8/16/32-bit natural 元素的各字节，再按位组装同一数值。
索引与结果的 lane/part 映射及原读取位置保持不变；地址、byte carrier 和组装临时量计入资源。
packed bitmask window 在同样的 byte-interleaved 存储上选择 `strided` access，
按实际 mask 字节数执行 strided byte load，再机械重解释为 mask；普通存储仍选择 unit mask load。

interleaved scalar field 的 row chunk 等于元素字节数时，memory pass 可证明一次读取完整位于
同一 row cohort：物理 extent 整除 cohort，point partition 等于该 extent，base 是其整数倍。
只有该证明成立才选择 `encoded.field-read.unit`；不足元素对齐的 little-endian 字段选择
`encoded.field-read.unit-bytes`，以相同 LMUL 读取 `activeVL * elementBytes` 个 u8，再按位
重解释为目标元素。不得提高 alignment、扩大读取 footprint，或在 emitter 重新猜连续性。

spill/reload必须具有exact transfer leaf、typed local slot和完整value layout；translator不能仅凭
binding kind自行选择一条隐藏路径。

资源 pass 可将 spill 的 `reload → issue_slice` 合成 `reload_slice`。它显式保存原载体类型、
source parts 和 lane offsets；每个结果 part 只读取已验证的字节窗口，原 slot 的唯一 spill
必须支配读取，整个 slot 保持不可变。坐标验证同时限制保留轴的 lane 顺序和被投影轴的固定值。
中间的行主序 reshape 只有在逐 part/lane 线性坐标及载体容量完全相同时才能穿透；
slot 仍保留原 spill 类型，读取显式保存切片使用的逻辑视图，不能把轴置换当作免费别名。
资源只计入实际载入的结果载体，不能先要求整个旧 Value 驻留再做寄存器提取。

连续 lane slice 的载体容量按 `VLEN × LMUL / SEW` 计算，不按源 value 的 active lanes
比例推算 LMUL；未占满载体不是新的逻辑维度，也不能使合法窗口被误判成非整数 LMUL。
缩小载体还必须保证每个完整窗口的起点与终点位于同一寄存器组内；长度放得下不能替代
offset 合法性，跨组窗口保留能够完整容纳它的载体。

`rvv.partial-repack.split` 仅表示可直接寻址的完整寄存器组切片；fractional 结果或未占满
寄存器的窗口使用显式选定的 `rvv.partial-repack.slice`。后者的 leaf 固定 addressable carrier，
terminal translation 只据此展开 subgroup extract、intra-group slide 与 LMUL truncation。
不能输出不存在的 fractional `vget`，也不能在 emitter 临时换载体。
每个独立 partial slot 按至少一个寄存器组计数；窗口 materialization 的结果与临时载体
必须计入 leaf 资源，不能将 fractional slots 假定为共享同一寄存器的免费视图。
逐槽 reduction 的资源计数遵守消费时序，允许已消费输入与对应结果复用寄存器，不将
所有旧输入和所有新输出重复计为同时存活。

一个已经选定的grouped/layered byte-window load可以使用固定的
`scalar-prime + vector-load` leaf。其op必须只含一个raw window，logical base、byte offset、active
lane数和vector load form均已闭合；translator只把“读取最后一个active byte，再发原vector load”
拼成两条有序指令。该leaf没有跨iteration对象、distance、buffer或额外result，不能冒充通用
prefetch，也不能由translator临时加到普通load上。

## 3. IME / opaque asm leaf

IME capability和operations共同固定：

- fragment family、logical shape、dtype与axis relation；
- lhs/rhs/accumulator fragment role及其tiled packing schema、tile rows/columns、tile与
  element order、storage width与alignment；
- MMA group/chunk数和fragment resources；
- fragment→RVV handoff；
- exact asm spelling key、constraints、clobbers与memory effect。

当前SpaceMIT IME1硬件 atom是 signed-i8 `M4×K8` 与 signed-i8 `K8×N4`
的带 i32 accumulator contraction。注册的 logical M 为 4 或 1；M1 仅暴露第一行，仍使用
完整硬件 atom 与两个 accumulator register groups，不虚构 M1 指令。lhs按`[M,K]`排列，rhs按
`[N,K]`的物理次序装入同一个`[K,N]`逻辑值，result按`[M,N]`排列。packing tile geometry、
axis order和各role的resource groups都是typed attribute，不能只靠`role=lhs`在emitter中复原；
局部pack、`vmadot`和unpack序列可以opaque，因为shape、signedness、storage width和clobber集合
已由唯一capability/spelling合同封闭。窄字段可由显式 load/decode/cast 形成 i8 fragment，不代表
硬件支持 i4 MMA。非 VLEN256 的 IME1 target 明确 unsupported。

同一 atom 可在 M/N 上组成不超过四个 atom 的完整局部 product，K 仍为 8。FragmentType
以 fragment factors 保存单 atom 坐标、replica factors 保存有界重复数，shape 和资源组覆盖
完整 tile；M1 的部分有效硬件 atom 不沿 M 重复。selector 固定 atom 与重复数，pack、MMA 和
unpack 只展开该 typed geometry。不同 N atom 消费同一 lhs fragment，不重建其供应；多个
accumulator 交到 RVV 时按明确坐标拼接，不能改变作者 K 累加、Level 或 persistent ABI。

operand supply 有两种显式局部表示：`ime_pack` 消费已选的连续内轴 RVV slices，以寄存器拼接
完成 packing；`ime_fragment_storage_load` 只在 source field 的整 chunk、row cohort、point partition/
alignment 共同证明局部 tile 连续且不需要 snapshot 时执行 unit byte load，并按位重解释为 i8
fragment。sub-byte 数值由独立 pure `ime_fragment_decode` 恢复，shift 是可证明落在 byte 内的
显式 SSA；原始 byte supply 因而可以服务不同 layer，不能把每个 layer 的读取藏在 decode 中。
raw-load 的 packing 必须匹配已注册 capability 的 operand role；其 tail 为 `agnostic`，只定义
packing 的逻辑 byte cells。非逻辑行不承诺零填充，也不能经 MMA/handoff 成为可观察输出；
部分有效 atom 仍由 capability 的 active shape 限定，不能按较大硬件容量扩展作者 domain。
单个逻辑行的自然 i8 field 在非交错 record 中也可直接供应 fragment：row partition 必须为一，
内层 point 必须按 atom 宽度对齐并完整覆盖该次读取。它沿用原 record 地址，不要求额外交错或清零。
完整的二或四个 byte atom 各占一个 RVV register、且目标支持对应 LMUL 时，可选择
`ime.field-raw-load.grouped`。一次连续读取形成整个 register group，按固定 subregister
坐标取出原有 fragments；访问字节和 fragment 次序不变，结果本身承载整组，不另占 vector
temporary。相同几何下，`ime.decode.grouped-bits` 与 `ime.fragment.add/sub.grouped` 对整组
执行原有 byte-local 运算；后者要求同型 fragment operands，仍遵守显式 lhs copy/reuse。
`ime.field-raw-load.scalar-prime` 只适用于完整有效的二至四 atom 连续 byte 供给：先 scalar
读取整个供给的最后一个有效字节，再执行原有 vector loads。读取范围不扩展，原 packing、
fragment 数与临时资源不变；这是同一局部 load 的固定序列，不跨迭代预取或引入私有 buffer。
physical index 的有界同余推导可把该 shift 冻结成常量。unsigned byte-local decode 的
shift=0 省右移，shift=8-bits 省 mask；signed decode 仍作必要的 sign extension。未知 shift
保持完整已选 decode，不由 terminal 遍历 producer 图重新证明常量。
它不拥有外层遍历或 persistent layout。`ime_fragment_pointwise` 对同一 i8 fragment 作 add/sub，
rhs 只可为同 dtype scalar 或覆盖 row 轴的一份 RVV value，后者必须为 SEW=8、VL 等于 row tile、
LMUL 可合法扩到 m1；非 full validity 必须与 fragment 一致，不接受任意 mask。row broadcast
的 index/gather 临时资源属于该局部合同。多 atom 的 row 参数覆盖完整 row 域，逐 atom 的
静态 index offset 已由 type 决定。其余输入关系必须在此前显式转换，不能由 emitter 重选。

`ime_fragment_row_broadcast` 将上述完整 signed-byte row value 展开为具有相同 typed packing
的 operand fragment；首批只支持 atom 内列宽 8。`ime.rows.local` 用一个局部 index 依次生成
至多四个 atom 的 row gather，输出和 index 同时保留，temporary groups 为 atom 数加一。
二至四 atom 还可选择 `ime.rows.indices`，用显式 RVV intrinsic 形成各 atom 的常量 index
并执行 gather；这些纯常量允许 system compiler 移出重复执行区间。所有输出与 index 的
联合临时上界为两倍 atom 数，不能隐藏额外寄存器或将 index 生成解释成新的作者操作。
当前选择器仅在 partial-row atom 上采用该形式；完整行 atom 的浮点 epilogue 驻留更宽，
采用 local-index 拼写限制 index 的实际存活区间，避免系统编译器提升多个 index 后产生 spill。
完整有效硬件行的 realization 可选择 `ime.rows.byte-repeat`：source u8 位模式逐 lane 扩到
u64，乘 `0x0101010101010101`，再重解释并拆为原序 i8 fragments。source 与宽 carrier 必须
恰好填满对应 lanes，target 支持 SEW64 与所需 LMUL；最大乘积为 `UINT64_MAX`，负 i8 的
位模式也保持。每个源 lane 只复制到自身的八个列位置，inactive 行不混入其它有效行。
宽输入/结果的联合临时上界为两倍 atom 数；未覆盖完整 operand packing 行的供应保留
local-index realization。行覆盖按 operand role 的 packing 判断：M1 限制 lhs/结果的 M 行，
不会缩小 rhs 的 N 行；不能用全局 `activeMFactor/mFactor` 代替该检查。
byte-repeat 要求 row-major packing、只沿 row 轴重复且行数完全覆盖
source lanes；逻辑有效前缀 `p` 因而恰好映射到前 `8p` 个字节，支持原有 tail 合同。
同一个展开结果可供后续多个 pointwise 使用，原 numeric input 与读取身份不变。pointwise
也可直接消费同型 fragment rhs；`lhs_policy=copy/reuse` 是显式资源选择，reuse 必须证明 lhs
同 block 且仅有当前一次 use，verifier 与 terminal 原地赋值共同遵守，不推断跨控制流的别名。

ISA capability、packing、fragment groups、私有 scratch 与 clobbers 的唯一描述在
`lib/Target/Extensions/SpacemitIME1/Contracts.cpp`。有限 row/column/product atom 上限也来自该合同；
公共选择器与 verifier 通过 `supportsRISCVFragmentTile` 查询，不再内置 IME1 的四 atom 常量。
不依赖 MLIR 的 `WeftRISCVContracts` 同时由 target
profile 与 IR dialect 消费；`FragmentOps.cpp` 负责属性序列化及针对该合同的 verifier，
`IME/FragmentMaterialization.cpp` 物化 selected product，`IME/FragmentEmission.cpp` 分派到
`Extensions/<Module>/Emission.cpp` 拼写。合同和 emitter 由同一份构建期模块表登记，不保留厂商指令枚举。
属性不能用一个已知 instruction key 冒充不同的 dtype、shape、VLEN 或寄存器约束。
任何通过局部 asm 改写向量配置的拼写必须显式 clobber `vl` 与 `vtype`，包括 MMA、局部
初始化与 row-index 生成；普通 RVV local splat 同样遵守。两者是机器状态而非 vector register
groups，不能只声明 GPR scratch 并依赖 system compiler 把 asm 偶然保留在原位置。
MMA 使用 typed register operands 和 tied、early-clobber accumulator；vector register 编号交给
system compiler 的寄存器分配，但资源数与不重叠约束已固定。leaf 为仍被其它 SSA use 引用的
旧 accumulator 按 atom 数预留 result copy 的 fragment groups，并计入未被 operand 覆盖的固定
clobbers。资源 owner 可以显式选择 `accumulator_policy=reuse`，但必须证明输入只在当前
同 block 的这次 MMA 使用，且来自同 block 定义或 block argument；不能原地修改仍被后续 use
或外层迭代使用的值。terminal 只拼写已选 copy/reuse，不自选。这项规则之外，pack/load/decode/
pointwise/unpack 还计入固定展开中输入与已形成结果同时存活的资源。
MMA 不在内部清零、不访问用户内存、不把结果写回局部数组。`ime_zero` 与最终
`ime_unpack` 必须独立存在；跨 issue 保持的状态由真实 loop carry 表达。
同一个已选 multi-atom MMA op 内可用一个 asm block 与一次 VTYPE 设置拼写全部 atom；
每组 accumulator 独立使用 tied、early-clobber 输出，共享 lhs 仍是同一输入。该拼写不能
跨 MMA op 或 Level 合并，不能改变 atom 次序、copy/reuse、clobbers 或联合资源上界。
primitive-private register/effect resource 不与用户 MemDesc 别名，不能把这种局部写入当作
覆盖权重的任意 memory write，也不能因此忽略真正的用户内存写入。

使用已有 atom 的新算子在 primitive/数值/存储合同已覆盖时，只增加 source/std 表达与显式
绑定。同类 fragment 合同的新 atom 修改上述目标局部描述、合法性、packing/spelling 与静态
登记，不修改公共 pipeline、ABI 或 live-set 算法。新的 effect、residency 或数值机制仍需
显式扩充对应通用合同；这不承诺任意新硬件无需公共改动。

RVV 与 IME 共享 frontend/Canonical IR、外围 control、memory/effect、compiler API、ABI 和
region-aware live-set 分析；两者的寄存器需求进入同一预算。RVV 的 LMUL/partial/spill 策略
不因此自动适用于 fragment。标准 ISA 的 native JIT 发现不等于厂商 matrix 扩展发现。
IME1 的 native 匹配事实与 local capability 同处其模块的 `Contracts.cpp`：必须有每个执行
CPU 明确报告的 `ime` feature，且 vendor/architecture、XLEN、VLEN 均匹配。探测结果保存为
扩展对应的 CPU 集合，由共享 runtime 在编译和调用时校验；没有匹配能力仍明确 unsupported。

完整GEMM/GEMV、persistent repack、M/N/K dispatch和outer loop不属于IME leaf。若extension需要
不同Level/materialization或persistent Encoding，作者在lowering前选择另一份std tree。

## 4. 选择与失败

1. `SelectRISCVOperations`只选择局部structural family；
2. layout、memory与composite passes把representation和真实control闭合；
3. `FinalizeRISCVLeaves`为尚未终结的普通op写唯一exact instruction；
4. 已实例化的local memory-leaf参数可以在完整storage/use relation上改写exact leaf；
5. resource pass补全leaf resource counts，final verifier核对整个合同。

leaf/capability集合不按kernel、operator或量化格式注册。无合法leaf、packing/access不完整、
toolchain不支持spelling、resource/effect不闭合时当前module失败；不能回到emitter重选、静默
scalar化或调用仓库外reference/GGML helper。
