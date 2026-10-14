# C/C++ SDK 与 llama.cpp 接入

## 产物与构建边界

`weft.runtime.PackageBuilder` 将作者明确选择的程序、source auto 绑定和 target physical
绑定编译为具名 artifact 集合。`Package.create` 发布已有 artifact；`Package.link` 在部署前
系统编译 source-only 产物。三者都进入现有 artifact/build/link 服务，不引入另一套 lowering。
包不是程序 IR，也不描述计算图、调度、权重转换或隐式 workspace。

```python
from weft.runtime import CompileOptions, PackageBuilder, Target

builder = PackageBuilder(Target("rv64imafdcv", "lp64d", 128))
builder.add("projection_dot", my_dot, options=CompileOptions())
builder.bind("ggml.vec_dot", "q5_0", "projection_dot")
package = builder.build("/path/to/new/package", native_library=False)
```

`add(..., selection=result)` 复用 `TuningResult` 的完整 winner，核对 canonical program、ABI、ISA
与 VLEN；不能同时覆盖 options。`add_ir` 接受外部前端的 Canonical Kernel IR，`add_artifact`
接收相同目标的既有产物。离线选定配置不代表新的性能测量。

`weft package-build specification.json --output /path/to/package --source-only` 接受以下结构。
文件路径相对 specification 所在目录；模块导入使用调用者已有的 Python import path。
一个 entry 恰有 `source`、`ir`、`artifact` 之一；source/IR 可附 `options` 或保存结果的
`selection` 路径。artifact entry 不能重写其编译绑定。

```json
{
  "target": {"march": "rv64imafdcv", "abi": "lp64d", "vlen_bits": 128},
  "kernels": {
    "affine": {"source": "affine.py:affine", "options": {"meta": {"B": 64}}},
    "projection_dot": {"artifact": "selected-dot"}
  },
  "bindings": {"ggml.vec_dot": {"q5_0": "projection_dot"}}
}
```

包目录以 `package.json` 命名其 entries 与框架角色，内部 artifact 使用相对路径。构建完成所有
entries 后才发布目标目录；已有目标目录不会被覆盖。框架 bindings 是显式接入选择，不参与编译器
按 primitive、use-def/effects 和 target facts 做物理决定。

原生 SDK 消费已经链接的 `kernel.so`、compiler ABI metadata 和 typed header。`dispatch.c`
只把检查后的 View pointer、scalar、动态 shape 与 stride 按同一 prototype 排列；它接收该
library 的原始函数指针，避免多个同名 specialization 之间的动态符号抢占。所有数学运算仍来自
Canonical Kernel IR → Physical IR → intrinsic C。部署后加载及重复调用不启动 Python、Weft
compiler 或系统 C compiler；source-only 产物必须先显式 `weft package-link`。

安装后的 SDK 路径可由 `weft sdk --prefix` 或 `weft sdk --cmake-dir` 查询。native wheel 携带
`include/weft/runtime.h`、`runtime.hpp`、共享运行库和 CMake imported target：

```cmake
find_package(Weft CONFIG REQUIRED)
add_executable(application application.cpp)
target_compile_features(application PRIVATE cxx_std_17)
target_link_libraries(application PRIVATE Weft::Runtime)
```

应用使用 CMake 的 `Weft_DIR` 或 `CMAKE_PREFIX_PATH` 指向 SDK。部署需要同 ABI 的系统运行库；
共享 LLVM 配置由发行包携带对应 LLVM 库。SDK 与 compiler 共享 `WeftRISCVPlatform` 的原生探测和
LLVM ISA 解释，不链接 MLIR 编译主链，也不维护另一份 ISA capability 表。

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

性能敏感的框架可在逐项核对 typed ABI 后使用 `weft_kernel_get_entry`。每个有界 worker 工作段
之前调用 `weft_kernel_check_thread`，段内不能改变执行能力，kernel handle 必须覆盖全部直接调用。
这个接口不替调用者检查每次裸指针的容量；相应类型、布局与容量证明由 adapter 承担。

## llama.cpp 的调用范围

`weft llama-install /path/to/llama.cpp` 给明确选定的上游源码树安装薄调用 hooks 和 `GGML_WEFT`
build option。工具逐项核对 GGML CPU 的调用边界；上游代码结构不匹配时报 unsupported，不能猜测
另一个替换位置。相同 hooks 可以重复安装。该操作不触碰 Weft 的只读 reference 目录。

构建时使用正常上游配置并增加：

```sh
cmake -S /path/to/llama.cpp -B /path/to/build \
  -DGGML_WEFT=ON -DWeft_DIR=/path/to/sdk/lib/cmake/Weft
cmake --build /path/to/build --target llama
```

运行前设置 `WEFT_GGML_PACKAGE=/path/to/native/package`。启用的集成必须能加载该包，且包内有
非空 `ggml.vec_dot` bindings；缺失、ABI 不符或目标不匹配明确终止初始化。`GGML_WEFT=OFF`
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

更早的 CPU extra-buffer/repack/厂商 IME 路径仍由上游选择；集成不关闭其 build options。
上游设备可能公开 extra buffer type，但不支持某个 type/shape/VLEN，使用者必须遵守其
`ggml_backend_dev_supports_op` 判定。whole-matrix Weft 程序如果拥有 activation quantization
或不同的 persistent weight layout，不能注册成 vec-dot。它需要一个与布局和准备责任相匹配
的独立接入合同；此接口不声称已替换这种路径。

`weft/ggml.h` 提供 registration、kernel name 和按 weight type 累积的 counters：矩阵节点数、
原 extra-buffer 节点数、Weft vec-dot 次数/结果数、通用路径的 activation quantize 次数/元素数。
并发线程使用独立 counters；线程退出时归并。图运行前后差值给出该段工作的命中信息。计数只
表示实际覆盖范围，extra-buffer 内部的量化不冒充通用量化计数，也不能从命中数推导模型加速。

## 机制对照与复现入口

Triton `python/triton/tools/compile.py:149-185` 从 signature 构造 AOT C stub 参数。
Weft 的对应责任在 `python/weft/runtime/dispatch.py:8` 与 `lib/Runtime/Binding.cpp:26`：
从同一 typed metadata 排列普通 CPU 参数。Triton 的 GPU grid/launch 信息不进入 Weft kernel ABI。

上游 GGML `ggml/src/ggml-cpu/ggml-cpu.c` 的矩阵计算实现分别提供
worker vec-dot、共享量化与间接矩阵入口；`integrations/llama.cpp/adapter.cpp:234` 只选择显式
绑定并在 worker 工作段验证目标。这一边界让 GGML 保有 traversal、threadpool、workspace 和
persistent weight 布局，Weft compiler 保有自己程序内部的物理决定。

`examples/repro/weft/native_package.py` 用既有作者程序构建可消费包；同目录 CMake 入口构建
`native-c`、`native-sdk`，以及指定 `LLAMA_SOURCE`/`LLAMA_LIBRARIES` 后的 `ggml-nodes`。
前两者覆盖 typed native 调用及资源生命周期；后者运行真实矩阵节点和 downstream activation，
检查量化、命中、未注册路径与上游可用 packed 路径。它们按[数值合同](../development/validation.md)
检查结果，不报告模型端到端性能。

packed 路径可以具有自己的 activation quantizer，接入保持性必须和相同正常路径比较：先用
集成关闭的 `ggml-nodes --record /outside/normal-outputs.txt` 保存各 type/shape/thread/buffer
组合的真实输出，再用集成开启的 `ggml-nodes --reference /outside/normal-outputs.txt` 对照。
记录文件保存在仓库外，不是性能表；接入没有改变原 packed 路径，不等于证明其量化行为与另一条
generic vec-dot 路径相同。
