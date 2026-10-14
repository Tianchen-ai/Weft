# RISC-V Physical IR

## 1. Program identity

RISC-V IR 是 Weft 的第二层、也是最后一层程序 IR。它混用 `func`、`scf`、`arith` 与
`weft_riscv` dialect，直接承载同一 canonical program 在一个 RISC-V target 上的物理程序。
它不是 Kernel IR 旁边的 plan，也不另设通用 physical IR。

每个 target operation保留 `source_origin` 供诊断；真实 use-def、control、memory、Level、
conversion、schedule 与 resource entity必须存在于本 IR，emitter不能回查 source closure。

## 2. Physical value 与 layout

`!weft_riscv.value` 保留 element type、logical shape、axis identities，并把 target layout放在
type 中：

```text
logical axis
→ time factor × lane factor × register-replica factor
  × fragment factor × local-storage factor
```

layout还携带 carrier、SEW、LMUL、`vl`、register groups与validity。所有未被 numerical
operation消去的 logical axes必须保留；例如 `[M,K] -> [M]` reduction 的 M 轴不能退化为
一个 scalar 再广播。IME representation使用独立 `FragmentType`；type同时携带
lhs/rhs/accumulator role与完整`FragmentPackingAttr`，不能伪装成 generic
`ValueType(carrier=ime)`。

RVV layout 的 `replicaGroup` 默认为 1，可显式取 2，把最后一个非单位 replica 轴上相邻的
两个 replica 拼进一个载体。它不改变逐轴 factor 或逻辑 shape；每载体 lane 数为
`product(lane) * replicaGroup`，载体数为 `product(time) * product(replica) / replicaGroup`，
register groups 按实际 LMUL 和载体数计算。载体内先排列原有完整 lane tile，再排列下一
replica 的 tile，因此 M4/N4 atom 成组后的顺序是 `[column-atom, row, column-lane]`，不是
普通 M4/N8 行主序。坐标投影与 `lane_repack` 必须使用同一有序映射。
容量相同的逐 lane handoff 还必须保留每个非单位 lane 轴的线性 stride；不能只比较各轴
factor 的乘积，就把反序多轴输入视为相同排列。

成组仅接受单 time、无 fragment/local factors 的 RVV 表示。final program 只允许已闭合的
逐 lane 操作、初始化、control carry 与显式 `lane_repack` 消费该表示，不允许直接 memory、
reduction 或其它尚未实现的专用 consumer；任意 mask 明确拒绝。矩形 tail 在完整 carrier 上
计算，边界 repack 恢复原 row slices，memory 操作继续核对行有效性和实际 VL。不能把完整
carrier 的计算宽度当作允许扩大读取或写入范围的依据。

`registerGroups` 表示一轮 issue 的 replica 容量，不证明整个 numeric Value 只驻留这一轮。
numeric RVV SSA 的完整驻留量为 `registerGroups × product(time)`；真实 issue loop 已经把
窗口缩进 operand/result type，不能再按外层完整域计数。encoded record、field/extract 存储投影
与 local descriptor 不占 numeric vector 寄存器，显式 `field_read` 的结果才占用。零拷贝
share、reshape 和保序 fragment handoff 按同一 storage identity 延长到所有 alias 的最后 use，
不重复分配；loop init 与 carry 复用还要求整条 alias 链及底层值只有这次 use。

canonical `reshape` lower为真实的 `weft_riscv.reshape`。op保留作者给出的input-axis
order和不同的result shape/axis identities；layout pass必须为两端形成线性序号相同的carrier
partition。只有time/lane/register/fragment/local各自的总分解一致时，terminal translation才可
把它拼写成零拷贝binding；否则必须在此前插入显式physical conversion或把candidate判非法，
不能由emitter调整坐标顺序。

当 input-axis order 不变、前缀 axis/extent 完全相同，且被拆分或合并的尾部全部由完整
lane 承载时，layout propagation 保留前缀的逐轴分解，并按相同线性序号重分解尾部 lane。
这条关系不跨 time、replica、fragment 或 local 边界；consumer 需要不同表示时使用显式
conversion。索引表供应可以沿该 reshape 的不变前缀追踪后续 contraction，但不能凭尾部
复用的 axis 名称推断坐标相同。

完整表项的 packed indexed gather 同时约束 payload 与 storage-word carrier：分配给字节
payload 的 LMUL 还须在实际 32/64-bit storage SEW 下合法。较宽 VLEN 不取消该约束；必要时
保留较宽的 memory result，再通过显式 conversion 交给较窄 consumer，active VL 和读取范围不变。

## 3. Memory 与 local storage

`MemDescType` 保存 pinned Encoding、shape/axes、static stride/origin facts、alignment、alias、
effect、layout identity与record geometry。kernel入口随后用 `memory_view` 的 SSA operands明确
给出每个动态 extent、stride和origin；ABI descriptor不能绕过这个 view。

普通 memory effect 绑定实际访问的 operand 或分配的 result。存储来源沿 view、slice、subview、
reaxis、local bind、snapshot 与有序控制的 SSA 边传播；`addressClass="slice"` 不决定其底层是
pinned 参数还是私有 allocation。参数 alias set 只在所属 kernel 内比较；local reuse 仍指向
原 allocation，分支及循环 carry 合并所有可能来源。数值寄存器快照不继续借用原输入 bytes。
未知来源或未知 effect 不能作为不别名证明，最终验证拒绝没有存储对象的普通 memory effect。

对 `strided_args` 声明的 pinned dense 参数，入口 `weft_riscv.stride(view, axis)` 读取对应 ABI
element stride，再作为 `memory_view` 的 SSA operand。它只允许读取直接入口参数的动态 stride，
不形成内存访问，也不接管 layout 选择。C ABI 在普通参数和动态 shape 之后，按参数/axis 顺序
追加 `size_t` stride。terminal translator 只拼写该参数，实际 unit/strided/indexed instruction
仍由 physical memory/operation passes 决定。

单 RVV lane-axis 的 dense edge 若使用显式动态 element stride，可选
`rvv.load/store.dynamic-stride` 或 `rvv.stream-load.dynamic-stride` 局部 leaf。它固定为
stride 等于 1 时的 unit 指令与其余值的 strided 指令，两臂具有相同地址、active lanes、
数值类型与 effects；无额外向量 scratch。普通 strided leaf 仍是合法表示。选择在 memory
pass 完成，verifier 核对动态 stride 及单 lane-axis 的合同，emitter 不自行引入该分支。

闭合 dense contraction 可在其 K 遍历外生成 `scf.if`，判断 operand 的 ABI stride 是否为 1。
完整输出窗口且 stride 条件成立的分支使用无 free-axis guard 的 unit stream load；其余分支
保留明确的尾部和 strided 访问。verifier 沿保留轴的 Slice/Subview 及入口 MemoryView
核对支配的 stride 条件和 free-axis extent，不能只凭一个 unit 标记省略 stride 或边界检查。
该分支不移动作者 Level 或改变闭合 dot 的归约域。
完整 K issue 的 active 必须为静态完整 lane 数；对应的 full-step leaf 使用无需保留尾部的
FMA，避免 load 与 FMA 的尾部策略反复切换。剩余 K issue 使用保留非活动 lane 的 step，
与完整 issue 共享累加器后只做一次最终归约。verifier 核对 full-step 的 active 和 operand validity。

canonical `subview` lower为 `weft_riscv.subview`：base descriptor、每轴静态或 SSA offset/extent和
缩小后的result descriptor同时存在于IR，动态 operand 顺序与 canonical marker 一致。它保留
Encoding、axis、stride、access与alias，向当前 base address加入逐轴相对offset，保存所选动态
extent；读写或再次投影都消费这份描述符，不能把动态长度重新解释为完整 ABI axis。
`weft_riscv.extent` 只读取该描述符中显式的有效范围。Subview 不是 indexed memory edge，
terminal translator不为它选择gather/scatter、补齐尾部或创造遍历；不支持的 consumer 表示由
physical legality 拒绝。

Extent 可以折叠为描述符本来就携带的 SSA bound 或静态常量。访存验证核对解析后的
同一范围，不要求一个已折叠的 Extent 节点继续存在；Slice 的 `all`、位置不变的 Reaxis、
Subview 与 LocalShape 保持各自的范围来源。仅当有界整数分析证明数值固定时才能据此
省略空区间或选完整 issue；未知、可能 wrap 的算术和未解析来源不提供这种证明。

`reaxis` 只按位置重命名 dense descriptor 的逻辑轴，保持 shape、stride、origin、
Encoding、access 与 alias。它不转置或复制数据。布局 canonicalization 可将位置不变的
数值轴重命名下推到一次原位置的 load，使访存与 contraction 共用当前坐标关系。

动态 extent 的 shaped load/store 必须由 bounded value/issue realization 消费。闭合浮点
contraction 可沿源描述符的实际 extent 遍历动态 free axes；结果只有一个不与输入别名的
store 时，可在同一物理微块内写回。投影后的 scalar store 带完整各轴边界，不分配完整动态
结果矩阵。中间存在写入、其它结果观察者或不闭合的表示时，不采用这条关系。转换入口保留
动态 subview，final verifier 拒绝最终仍未被有界 realization 消除的动态 shaped access。

Encoding field access用 `AccessAttr` 保存 selected memory form以及 natural、grouped-layered、
joined 的完整 storage geometry。terminal lowering可以据此计算确定地址和bit extraction，
不能按量化格式名补布局。

派生交错布局的每个字段另有 `fieldChunkBytes`，必须与 canonical derived declaration 的
`field_chunks` 一致。地址由 cohort、record byte、字段起点和 chunk 共同确定；time/replica
分片分别投影到 row 与 record-local 坐标，不能把所有分片都当作 row-lane offset。
字段写入由 `field_store_plan` 固定 scalar/byte assembly、自然元素或完整 grouped-layered
byte packing，以及每次 store 的 byte offset、source part、bit shift 和临时资源。当前
向量形式要求完整的单轴 record-local 字段；不闭合的跨 byte 元素或多轴写入明确拒绝。

自然布局、整字节且满足元素对齐的非交错单轴字段可用 `field_view` 暴露为 dense memory
descriptor，再以普通 slice/subview/store 写入子区域。该节点保存原 field、显式 SSA origin、unit
stride 及同一 access/alias；origin 按完整 record 的逻辑坐标与字段分组比换算，避免对子 Level
的绝对 point 重复累加 record offset。它不改变字段 bytes、初始化或 lifetime，不负责 bit packing。
record owner 自身的 subview 偏移尚不支持，须在物化时明确拒绝。memory pass 形成该表示，
final verifier 拒绝仍绕过它的字段投影写回；terminal 仅拼写已确定的地址。

natural field 的有效对齐由 record/base alignment、field byte offset 与动态 entry stride共同约束。
当有效对齐不足以支持 element-width unit load 时，Physical IR 必须选择显式 byte-load leaf，
由该 leaf按 byte载入并机械重解释为已经选定的RVV value；terminal emitter不能自行把`vle32`
降级成 byte load。final verifier拒绝任何超过Encoding与offset事实的alignment声明，以及任何
在不足对齐上仍选择element-width vector load的physical edge。

natural field到RVV register的完整窗口用`rvv.replica_storage_load`表示。它的
`StorageWindowPlanAttr`给出logical lane axis、unit/strided projection、extent/alignment、record
坐标与每个time/replica part对应的source window；result layout给出唯一SEW、LMUL与实际`vl`，
leaf给出exact `vle`或`vlse`。因此strided与unit不是terminal emitter根据地址表达式临时选择的
两种拼写。若后续issue materialization缩小窗口，只能投影已选plan，不能重新决定memory form。

`rvv_segment_pair_load` 将同一个 natural field 的两路完整一维交错投影表示为一次双结果读取：
第一个结果对应 `field[base + 2*i]`，第二个对应 `field[base + 2*i + 1]`。静态 base、两端相同的
lane/time layout、element alignment、`segmentFields=2` 与 exact `vlseg2` leaf 必须闭合；每个
issue 的 NF×LMUL 不超过 8，所有静态 tuple parts 的临时资源显式计入 leaf。它不增加读取范围、
逻辑轴或数值合并；多轴、tail、动态 base 与跨干扰写入的配对不属于这个关系。

多轴entry/payload坐标若按行主序线性化为连续source区间，使用
`rvv_unit_entry_window_load`保存scalar base、entry axes/extents、payload axis/extent与exact unit
leaf。entry axes必须是source中不存在的新逻辑轴；payload axis可以是新轴，也可以复用唯一被投影
掉的source axis，后者明确表示`entry * payload_extent + payload`的连续后缀。其它source-axis
别名、非unit系数或不闭合的layout必须保留为typed indexed edge，terminal emitter不能自行把
indexed access改写成unit load。

对一个已经闭合、只有一个raw byte window的grouped/layered load，physical parameter可以把
exact leaf选为`scalar-prime + vector-load`：先对该window中最后一个active byte执行一次有序的
scalar byte read，再执行原来已经选定的vector load。这个选择直接写在load op的`LeafAttr`中；
未选择时仍是原来的单vector-load leaf。它不读取未来iteration、不产生SSA result、不建立buffer
version，也不改变window、carrier或control，因此不是prefetch/pipeline。多window edge若没有显式
给出prime对象则非法，terminal emitter不能自行挑一个window。

当 target 的固定 record-axis 优先级选择跨 Level instance 的 lane cohort 时，
`RecordCohortType` 将一个 physical point、静态 Level partition 与 cohort width 绑定为同一坐标
实体。`rvv_record_storage_load` 明确保存 source field、cohort、storage unit、record byte stride
及该 unit 覆盖的 logical indices；sub-byte decode 是独立 pure op；`rvv_record_store` 明确保存
dense destination、logical element offset 与 record byte stride。动态 Level trip count仍由真实
主 `scf.for` 和 scalar tail 表示，不能藏在上述 leaf 或 emitter 中。该表示只在 source/destination
的一维 unit-stride encoded-record geometry、完整 active partition 与 target VLMAX 都可证明时合法；
否则保留原来的 within-record mapping。

若regular-repeat relation使同一个encoded scalar覆盖一个或多个完整lane window，
`rvv_regular_repeat_scalar_load`明确保存source axis、dynamic source base、source count、repeat与
每个physical time/replica part对应的`part_bases`。其result仍保留原logical axes以及
time/lane/replica分解，但carrier是scalar；后续pointwise边界因此只能发射已选的scalar-vector
operand form。verifier必须从layout与repeat关系重算`part_bases`，不能让terminal emitter推测
哪个scalar服务哪个lane window。该形式只适用于已证明对齐的完整repeat window；其它relation
保留为typed gather或普通memory edge。

`LocalType` 是 invocation-local、target管理的可寻址对象，明确 size、alignment、alias、
purpose、owner domain、birth与lifetime。当前真实用途包括 state/handoff object、spill slot与
pipeline window；它不能替代 caller-visible workspace。动态对象必须由紧邻的
`local_capacity_guard` 消费请求大小与当前作用域 live-byte SSA，给出合计 target-bounded
上界，否则 resource pass 拒绝它。计数包含对齐余量，不能把全部对象的平均份额当容量。

`local_view` 把已有 numeric local state 暴露为 dense descriptor，显式消费每轴 extent 与
row-major stride SSA；它不分配、复制或延长原 state 的 lifetime。动态 extent 按 shape
symbol identity 绑定，不把 logical axis 编号当作 shape symbol 编号；i1 state 使用一字节
元素存储。连续、静态有界的 local extract/update 可形成普通 `subview/slice + load/store`，
未缩小的轴可保留由 SSA 绑定的完整动态 extent。窗口保留动态 origin、原索引次序与未写
区域，继续经过共同 memory/resource passes。原地更新
要求旧 state version 的其它读取先于该更新或处于互斥分支；旧版本仍需存活时不能直接复用
其 allocation。任意多轴不规则 gather/scatter 不由连续窗口关系替代。

`staged_view` 是 reload placement 的唯一 SSA birth authority；register materialization也有真实
SSA result。作者 staged birth携带 owner/birth/lifetime；编译器在同一block内冻结一次scalar
supply时使用`physical-share`，其作用域完全由SSA dominance给出，不伪造作者birth。
staged lifetime不靠无 result marker保存。

encoded `load` 可携带显式 `snapshot_storage` operand，指向 schema 为 `read-snapshot`
的私有 byte allocation。其 selected leaf 在原读取点复制完整连续 record range，field
与 extract 继续投影同一个 Value，但不再重新读取可能已被覆盖的 pinned bytes。allocation、
byte size、alignment、owner/lifetime 和复制的 read/write effects 都必须在 IR 中闭合；
没有配套存储的 snapshot leaf 或跨干扰写入的无快照 encoded load 均非法。
静态连续 dense table 的 `dense_snapshot` 是显式 Read/Write transfer：消费原 descriptor
和精确大小的私有 byte allocation，生成同一 logical table 的只读 local descriptor。
其 leaf 只复制已确定的字节数，不选择 layout 或遍历结构。encoded staging 的私有副本
由 `encoded_local_bind` 切断对原 load 的值依赖；之后的原地址写入不影响该副本。
RVV target 的 snapshot 使用精确 SEW=8 transfer，owner 在合法 m1/m2/m4 中以最少
load/store 对数选择，平局取较少 scratch；leaf 参数冻结 LMUL 与每次搬运字节数，
temporary resource 为相应的 1/2/4 组，仍须通过完整 live-set 检查。末块只访问剩余字节。
无 RVV 的 target 才选择 scalar copy leaf。emitter 不自行选择 memcpy 或 RVV 实现。

## 4. Explicit conversion

representation conflict是一个真实 `weft_riscv.convert_layout`：source/result type保存完整
layout，`ConversionAttr`保存 transfer kind/effect/temporaries，leaf保存 exact terminal form。
conversion保持 element、shape、axes和 logical Value identity，只改变 physical carrier或
coordinate decomposition。

storage projection 与 numeric SSA 不混同。`field_read` 是显式 Read-effect op：input 保留
field/extract 到原 load 或 snapshot 的来源，access 固定字节/bit mapping 与 memory form，
result type 固定数值元素、axes、每个 part 的布局与活跃范围。它不改变 logical domain，
也不允许把算术结果当作 descriptor 重新读取。pure `time_to_lane` 不再承载字段重供应。
projection 捕获的 index SSA 同时列为 `field_read.indices`，不能仅靠祖先 Extract 隐式延长
向量索引寿命。load/decode 的输出组与临时载体上界由共享几何函数计算，leaf verifier 核对，
资源 pass 与其它 live values 共同计数；转换的零临时数不能代替读取资源。

regular contraction 的 reduction operand projection 必须消费明确的 lane/time 坐标。
被消去轴的 physical factors 全为 1 不意味着 logical extent 为 1；compiler 不能忽略
仍变化的 reduction index。dense read 可以在无干扰写入、无 snapshot 替换且存储 dtype
不变的条件下，先形成显式 indexed slice/load，再对投影结果 widening；不能把 widened
numeric value 当成较宽存储重新读取。寄存器投影必须保留所有 free-axis 坐标，并符合
selected lane extraction 或 independent-lane time selection 的完整合同。

conversion canonicalization必须重写/删除该 op。final IR可以保留合同已经闭合的 terminal
conversion；intrinsic-C translator根据 source/result layout确定地展开 slide、splat、tuple
split/merge或local transfer。这与重新选择layout不同。
RVV part-to-lane 直接拼接必须保持源 lane 内部的逻辑轴顺序；已有较早 lane axis 时，
不能把较晚 axis 的 register/time pieces 当作连续向量尾部拼接。欠满的 carrier 按实际
active lane 数拼接，不能把整寄存器容量中的无效尾部带入结果。需要交错的关系必须在
资源闭合前由合法的 index/layout 改写处理，或明确拒绝。

`lane_repack` 是同域 RVV 表示间的显式有序坐标交换，允许一个轴从 lane 移出、另一轴移入。
两端 type 唯一确定 `part=replica×streams+time` 与行主序 lane 坐标；共享几何函数把最多4096个
元素合为连续 source-part/lane slices，verifier 与 translator 消费同一关系。它不依赖第三种
layout map，不改变 dtype、轴顺序、validity 或读取字节，完整输出及局部移动 scratch 进入预算。

RVV↔IME handoff不用 generic conversion冒充：`ime_pack` 或已证明连续的 `ime_fragment_storage_load`
形成 fragment；后者的 raw bytes 由独立 `ime_fragment_decode` 投影为 sub-byte 数值，显式 shift
必须有 byte-local 上界。`ime_fragment_pointwise` 保留同一 i8 domain 的标量/row broadcast；
`ime_zero`、带 accumulator 输入的 `ime_fragment_mma` 与 `ime_unpack` 分别表示初始化、更新和
最终 handoff。FragmentType 保留独立资源与 packing，C 使用 RVV vector 类型不改变其 IR 归属。
`new` 可以保存表示相同的 accumulator fragment 初值；普通 `scf.for` 通过同型参数/yield 携带
它，原有 state birth、owner、lifetime 和 Level identity 不变。

fragment factors 固定一个硬件 atom，replica factors 固定 M/N 上的有限重复，单 product 最多
四个 atom 且不重复 K。input/result packing 与完整 resources 由同一 type/leaf 合同约束；
共享 lhs、多个 accumulator 和最终 RVV 拼接都发生在这一层，不产生第三层 planning IR。

row-major accumulator 的 atom factors 也可成为普通 RVV Value 的 lane factors，有限 atom
replicas 保持原顺序。`ime_unpack` 在该合同下是 alias 或合法的低位 carrier truncation，
无需提前拆为行向量；cast、pointwise、同型 new/iter_arg/yield 可以保持这一表示。普通 f32
值始终是 RVV Value，不伪装成 i32 fragment。memory 或其它不支持的 consumer 通过显式
`lane_repack` 恢复所需布局。多 lane-axis 的局部 pointwise 使用完整 carrier，活动矩形之外的
单元不成为可观察结果；内存边界仍按各逻辑轴的有效范围访问，不能把矩形误作一个 lane prefix。

## 5. Composite 与 terminal target operations

进入 terminal emission 前，canonical composite会变成真实的局部程序：

- reduction/contract 的 `scf.for` 和 loop-carried accumulator；
- grouped/encoded `WindowType`、load-window与compute-step；
- RVV contraction step 与显式 operand projection、load/field_read、widen；
- 独立完整product的`rvv_partial_collect`、partial reduction与scale-combine use-def；
- IME fragment packing geometry、MMA groups/chunks与unpack；
- local alloc/bind/load/store、spill/reload；
- 对已支持window cluster展开后的pipeline prologue、steady state与epilogue。

Window与IME op可以对应一个固定 closed leaf sequence，但其 slots、terms、axis、access、
fragment packing、resources和trip count必须由 type/op attributes闭合。source Level loop、动态
outer traversal、workspace或pipeline不能隐藏在 leaf 中。

流式 reduce/dot/contract 的 input window 保留 tail 与实际 VL；累加器必须完整初始化所有
物理 lane，未参与输入的 lane 保存归约单位元。step 保留未更新的累加器 lane，finalize 只接受
`full` 累加器。不能把短输入的 tail-agnostic splat 当作可按完整 VL 归约的初始值。
该共同合同由 `StreamOps.cpp` 验证，`MaterializeRISCVPrograms` 写出真实 init/loop/finalize。

浮点 contraction 的多条 reduction axes 可由外围有序物理循环和最内 RVV K 窗口共同遍历，
所有循环携带同一完整初始化的 partial，只在原闭合 operation 的出口收敛。f32/f64 使用同型
FMA，双 f16 存储可使用显式 widening FMA。普通 free-axis FMA 只消费已投影的 scalar/tuple
与 RVV 数值；其共享坐标映射由 `ContractStepGeometry` 验证和拼写，不携带可隐式重读的地址。

普通 lane reduction 同样必须在零 active lane 时保留 scalar seed。RVV 的 VL=0 reduction 不更新
目的寄存器，不能把 agnostic destination 当作会自动复制 seed；exact leaf 将已初始化 seed 同时
作为 passthrough destination，terminal 拼写对应 `_tu` intrinsic。seed 及完整 time parts 合并所需
的临时 carrier 必须计入资源合同。

source auto 绑定同时实例化控制 SSA、Value/View shape 和公共 ABI 中对应的 shape symbol。
其中固定后的 extent 可以用于物理合法性证明；没有绑定的动态 shape 仍沿 ABI 输入。
标量 carrier 的物理 part 包含 replica 与 time 两种因子，状态初始化、carry、局部 reload 和
存储必须消费同一坐标映射，不能只遍历 replica。numeric staged 值转入有界 local storage 时，
`LocalType`/`LocalBindOp` 保存原 owner、birth 与 lifetime，真实副本不丢失作者 staged 身份。

`RegisterCoordinates` 从已选 Layout 建立 part/lane 与逐逻辑轴坐标的双向映射；同一轴上的 time、
replica 和 lane 因子共同参与坐标，不能任选其中一种。scalar tuple、dense memory、ordered control
与 RVV repack 消费同一映射。完成的 local value 可以按任意轴上的有限 time 分片读取；memory
pass 必须形成真实 `local_view → slice → load`，数值 local-load conversion 不能留给 emitter 补读。

## 6. Level、ordinary control 与 schedule

canonical Level转换时保留 domain/partition/multiplicity、birth counts、carried values、handoff
和direction。Level最终成为带 `LevelAttr` 的真实 `scf.for`。ordinary `for/while/if` 仍是有序
scalar control，不因进入RISC-V IR而获得一个编译器发明的 shaped axis。

birth count 统计 canonical 初始化区域和本层 body 中直接存在的 `new/materialize`，
不以初始化区域 yield 的参数数量代替。普通 helper 内联产生的本层 birth 保持原 SSA 位置与
owner，不为补齐计数而移到初始化区域，也不把子 Level 的 birth 计入父层。

pipeline参数只有在改写成真实 guard、prologue、steady-state、epilogue与buffer-version carry后
才算实现。未消费 `ScheduleAttr` 或只有一个 depth 字段的module不能进入emission。

## 7. Final IR 不变量

`VerifyFinalRISCV` 通过时必须同时成立：

- Canonical Kernel IR operations已经全部消失；
- every shaped value/block argument有完整、target-legal layout；
- representation冲突由typed conversion或specialized fragment/local op显式承担；
- memory edge具有完整 descriptor、Encoding mapping、access form与leaf；
- source Level births/handoff、ordinary control与ABI `memory_view` identity仍可验证；
- composite、source Level op、`implementation`与未展开 schedule均已消失；
- planner使用的partial layout/combine/nested/sequential/scaled/layered plan attributes已被真实
  use-def、loop与local operations取代，不再是terminal translator输入；
- every target-local terminal op具有唯一 exact [local leaf](leaves.md)；
- register/fragment summary与重新计算的 SSA 驻留峰值一致，且与local-storage一起不超过target profile。

kernel 的 `required_extensions` 是明确的 build 要求，不属于 target capability 或
canonical 数值身份。final verifier 对当前 kernel 的 selected instruction ISA 合同
求实际使用集合，逐项核对要求；能力列表与其它入口不能代替该集合，重放保留同一要求。

携带已选 RVV stream reduction 的保留循环必须显式带 downstream-unroll-disable
合同。它的 lane/time 分解和 loop-carried accumulators 已在 Physical IR 中冻结；terminal
translator 只把该合同拼写为本地 pragma，防止 system compiler 再次展开并改变已核算的
live range。缺少该合同由 final verifier 拒绝，不能依赖 Clang/GCC 的启发式阈值。

当前没有实现的 conversion、spill、pipeline或extension形态必须在某个 physical pass中明确
unsupported，不能由terminal emitter给默认值或走另一条路径。
