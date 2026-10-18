# C/C++ SDK 与 llama.cpp 接入

## 从同一个 CompiledKernel 导出

Triton 的 CPU backend 直接生成 native library。`CompiledKernel.export` 发布该库、typed header、
ABI/目标 metadata 和运行依赖；`triton.load` 读取同一产物，不需要原 DSL 或现场编译器。

```python
import triton
compiled_dot.export("/application/kernels/dot")
package = triton.Package.create(
    "/application/kernels/model",
    {"projection_dot": compiled_dot},
    bindings={"ggml.vec_dot": {"q4_0": "projection_dot"}},
)
```

Package 接收已经选定的 CompiledKernel，包括 triton.load 返回的对象。
它不编译、不调优、不生成算法或 workspace；绑定是应用明确选择的模型角色。
所有 entries 必须使用共同 ABI/VLEN，GGML 角色核对 Encoding、shape、stride 和已用扩展。
包内 kernel 路径相对 package.json，发布前完成全部 entries，已有目录拒绝覆盖。

一个 kernel.so 包含两个入口：manifest.json 顶层描述单实例函数和 typed dispatch；
其 grid 字段描述原始程序 ABI 和 OpenMP launcher。两者来自同一数值程序。
llama.cpp 使用单实例入口，不再创建线程团队；显式 grid 调用使用原始 entry，
不能把单实例 entry pointer 传入 grid launcher。

C/C++ SDK 随 Triton wheel 分发，路径可由以下入口取得：

```python
from triton.runtime.package import sdk_prefix
print(sdk_prefix())
```

SDK 包含 include/weft/runtime.h、runtime.hpp、lib/libweft_runtime.so、
lib/cmake/Weft/WeftConfig.cmake 和 share/llama.cpp。Weft 是内部 native ABI 名称，
不要求安装另一份 Python 包。

```cmake
find_package(Weft CONFIG REQUIRED)
add_executable(application application.cpp)
target_compile_features(application PRIVATE cxx_std_17)
target_link_libraries(application PRIVATE Weft::Runtime)
```

Weft_DIR 指向 SDK 的 lib/cmake/Weft。运行时依赖与目标 ABI 必须匹配。
SDK 共用编译器的 ISA/capability 解释，不链接 MLIR 编译管线，也不调用 Python。
数学运算全部来自 Canonical IR → RISC-V Physical IR → intrinsic C/local asm。

## 调用、存储与生命周期

C 入口返回 `weft_status` 与可选 `weft_error`，错误消息包含参数或执行能力不匹配的原因。
所有新建 handle 的输出在失败时为 null。C++ wrapper 将相同错误转为 `weft::error`，并提供
move-only RAII。by-value f16/bf16 没有通用 C scalar 表示时明确拒绝；dense/packed View 的
Encoding 与字节表示仍由 compiler ABI 决定。

| 对象 | 所有权与调用合同 |
|---|---|
| context | 在当前线程 affinity 上探测目标；kernel 持有其执行能力信息 |
| kernel | 保持 native library、入口和 immutable ABI；释放外层 handle 不使已有 call 失效 |
| View | 借用调用者存储；容量、shape、stride、Encoding、可写性和 alias 合同在 prepare 时检查 |
| buffer | SDK 显式分配的对齐字节存储；不自动量化、初始化、repack 或清零状态 |
| bindings | 按参数名保存 metadata 和 scalar 的副本；`retain` 使 builder/call 持有 SDK buffer |
| call | 保存完整 ABI 参数；prepare 后外层描述符可释放，借用的数据字节仍须保持存活 |
| sequence | 持有一列 prepared calls，按作者指定顺序同步执行；没有隐式 worker、融合或重排 |

`kernel.storage_layout(argument, dimensions, strides)` 解析 dense/packed record、row cohort、
alignment 与具体 View span；调用者可以 `allocate()`，也可 `borrow()` 现有容量。
strides 是非负 element strides；未声明 strided 的 View 要求自然连续布局。
可写自重叠、非完整 record/cohort、signed pointer 范围外的 span 均被拒绝。空 View 不访问存储。
allocation span 和 prepare 使用同一个 `Layout` 合同计算。

```cpp
weft::context device;
weft::package package(device, package_directory);
auto kernel = package.get_kernel("affine");
auto output_layout = kernel.storage_layout(1, {{"N", count}});
auto output = output_layout.allocate();
auto binding = kernel.bind();
binding.set("X", weft::view(input, input_bytes, "dense.f32", {count}, {}, true))
       .retain("Y", output, output_layout)
       .set("scale", weft::scalar(1.25f))
       .set("bias", weft::scalar(-0.5f));
auto call = binding.prepare();
call();
```

改变 builder 不修改已经准备的 call；需要重新 prepare 才能改变地址、shape、stride 或 scalar。
`weft::sequence{&prepare_weights, &compute}` 是普通 host 调用次序。对于跨次复用的权重，应在
初始化期单独执行 `prepare_weights`，重复阶段只调用 `compute`。workspace 和 persistent state
由 host 显式保有/共享，sequence 不替调用者决定清零、复用时机或并发同步。

context 的 CPU 集合是发现时的可用范围。每次通用 call 核对当前 affinity、RVV 状态和实际使用
的扩展；sequence 在第一项写入前核对所有入口。独立存储可以并发调用同一 kernel，共享可变
存储由 host 同步。应用不得在执行期间改变 affinity 或 RVV 控制状态。

prepare/rebind 只绑定和验证存储合同，执行权限由实际执行的线程检查。
`weft_call_arguments` 可将 prepared call 的 typed ABI 参数数组借给原生 launcher；
数组、scalar 和 dimension storage 由 call 持有，直到 call_release。它必须配合同一 kernel 的
entry 使用，launcher 仍须在每个实际 worker 写入前调用能力检查。

加载 artifact 或取得 entry pointer 不执行 kernel。加载时核对该 context 中至少存在一个同时
满足所有扩展的 CPU；调用时再要求实际线程 affinity 全部属于可执行集合。因而主线程可以
加载异构核心上的扩展程序，交给具备该能力的 worker 执行，不必改变整个应用的亲和性。

需要执行线程选择时，`kernel.bind_thread()` 显式将当前线程的亲和性收窄到原集合与 artifact
全部扩展可执行集合的交集；空交集报错，不扩大调用者的 CPU 权限。该绑定在返回后保留，
调用方拥有其生命周期。`check_thread()` 仍只检查，不改变亲和性。IME adapter 在实际 worker
的节点函数选择处使用此接口，与平台首次原生 IME 指令触发的核心绑定一致。

性能敏感的框架可在逐项核对 typed ABI 后使用 `weft_kernel_get_entry`。每个有界 worker 工作段
之前调用 `weft_kernel_check_thread`，段内不能改变执行能力，kernel handle 必须覆盖全部直接调用。
这个接口不替调用者检查每次裸指针的容量；相应类型、布局与容量证明由 adapter 承担。

## llama.cpp 的调用范围

`triton.tools.llama_cpp.install_source("/path/to/llama.cpp")` 给明确选定的上游源码树安装薄调用 hooks 和 `GGML_WEFT`
build option。工具逐项核对 GGML CPU 的调用边界；上游代码结构不匹配时报 unsupported，不能猜测
另一个替换位置。相同 hooks 可以重复安装。该操作不触碰 Weft 的只读 reference 目录。

构建时使用正常上游配置并增加：

```sh
cmake -S /path/to/llama.cpp -B /path/to/build \
  -DGGML_WEFT=ON -DWeft_DIR=/path/to/sdk/lib/cmake/Weft
cmake --build /path/to/build --target llama
```

运行前设置 `WEFT_GGML_PACKAGE=/path/to/native/package`。启用的集成必须能加载该包，且包内有
至少一个受支持的非空 binding group；缺失、ABI 不符或目标不匹配明确终止初始化。包可用
`has_binding_group(name)` 查询角色组是否存在，具名 `bindings(name)` 仍对不存在的组报错。`GGML_WEFT=OFF`
移除 adapter 依赖并保留原 GGML 调用表达式。

接入工具同时修正所支持上游的 vendor build 声明：其 IME 源码要求 C++20，且每个 IME
translation unit 必须服从上游自身的 assembler capability probes。这些纯构建修正在 ON/OFF
两种配置中相同，保留探测通过的 IME 实现。启用上游 Spacemit backend 时仍须按该目标配置
`GGML_CPU_RISCV64_SPACEMIT=ON` 与它要求的 `GGML_RV_ZBA=ON`；未提供目标指令的工具链不能假装支持。

角色名是 GGML 的 weight type 名，如 `q5_0`。adapter 核对实际链接 GGML 的 block size、record
bytes、activation Encoding、参数顺序、对齐和 `nrows`。受支持的合同是普通 RVV、单行 vec-dot，
三个 View 为 `W[K]`、`X[K]`、`Y[1]`。当前可声明 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0、Q2_K 至 Q6_K；
能声明 ABI 不表示所有配置已有性能收益，部署者只注册已选定的实现。

实际接入点位于 GGML `MUL_MAT` 和 `MUL_MAT_ID` 的 worker chunk 内部。GGML 先完成共享 activation
量化和 barrier，再调用所选 vec-dot；adapter 不重新量化，不改变输出 stride、广播、路由、chunk
划分或线程池。未注册类型继续使用 GGML 原入口，框架显式 reference 模式使用原入口。

`ggml.from_float` 可绑定 `q8_0` 或 `q8_K` activation quantizer，两个 View 为 `X[K]`、`Y[K]`，
参数顺序固定，输出 Encoding 与正常 GGML record 完全一致。调用方负责相同舍入环境；接入在
原量化调用点使用原 workspace，每个 row/chunk 只量化一次。未绑定的量化类型保留上游函数。

更早的 CPU extra-buffer/repack/厂商 IME 路径仍由上游选择；集成不关闭其 build options。
上游设备可能公开 extra buffer type，但不支持某个 type/shape/VLEN，使用者必须遵守其
`ggml_backend_dev_supports_op` 判定。whole-matrix Weft 程序如果拥有 activation quantization
或不同的 persistent weight layout，不能注册成 vec-dot。

`ggml.ime1` 的 `q4_centered_prefill` / `q4_affine_prefill` 与
`q4_centered_decode` / `q4_affine_decode` 是独立的 packed row-tile 合同。
三个 View 为 `W[N,K]`、`Xq[M,K]`、带显式行列 stride 的 `Y[M,N]`；维度顺序为 `N,K,M`。
权重分别使用 `IMEQ4Centered_I[16]` / `IMEQ4Affine_I[16]`；prefill 的激活使用
`IMEQ8_I[4]`，decode 使用 `IMEQ8_I[1]`。adapter 在上游选择 IME1 函数时绑定显式
package entry；实际剩余行数至少四行时消费四行，否则只消费一行。这与原 IME 的
M4/M1 分派及激活布局一致，单行角色也可处理 prefill 的尾行。每次要求 32-element block、
完整 16 列 cohort，并传入原始输出行距。权重准备、激活量化、分块和线程池仍归上游。
未绑定的角色及不满足该 row-tile 合同的调用保留原 IME 函数；reference 模式
在函数选择前跳过 Weft。此绑定要求生成 artifact 实际使用 IME1，不接受仅声明扩展的 RVV 程序。

`weft/ggml.h` 提供 registration、kernel name 和按 weight type 累积的 counters：矩阵节点数、
原 extra-buffer 节点数、Weft vec-dot 次数/结果数、通用路径的 activation quantize 次数/元素数。
并发线程使用独立 counters；vec-dot 在 worker chunk 入口获取函数和计数指针，原调用点
直接调用已选函数，node 结束时汇总；逐调用不查询 TLS、注册表或原子计数。
packed tile 在节点入口绑定函数与执行核心，上游 callback 捕获该 worker
binding 指针，逐 tile 不再查询 TLS；段末汇总计数。额外提供
Weft quantizer 次数及 `ggml_weft_ime1_calls()` packed tile 次数。
线程退出时归并，图运行前后差值给出该段工作的命中信息。计数只
表示实际覆盖范围，extra-buffer 内部的量化不冒充通用量化计数，也不能从命中数推导模型加速。

worker 在每次 graph 的 NUMA/核心设置后开启新的能力检查作用域。无外部 abort/custom
callback 的原生 graph 保持 RVV permission，原生 IME 的亲和性收窄也保持已有检查成立；
每个实际使用的 vec-dot/quantizer 入口在此作用域内只检查一次，packed IME 入口也只做
一次执行能力检查与必要的亲和性收窄。缓存按具体入口区分，每次 graph 开始时清空。图含外部 callback 时，
每个矩阵节点仍检查；graph 结束后的核心恢复不能把检查状态带入下一次 graph。

## 测量边界

原生单实例、完整 grid/OpenMP launch 和模型 E2E 分开计时。
通过独立 Weft Bench 验证导出产物、加载、数值与角色命中；prefill/decode 分开记录。
正常 llama.cpp 的 repack、IME、OpenMP 等能力继续保留，固定 baseline 仅在测量合同变化时补测。
集成相关源码位于 third_party/weft/integrations/llama.cpp，Python 安装/角色接口位于
python/triton/tools/llama_cpp.py，具名产物集合位于 python/triton/runtime/package.py。
