# 运行时与安装

## 调用与构建

公开的 Python 入口是 `weft.compile`、`weft.jit`、`weft.autotune`、`weft.build`、`weft.build_ir` 和 `weft.load`。
参数类型 `CompileOptions`、`Target`、`DataCache`、`NativeTarget`、`Toolchain`、`Buffer` 与 `Artifact` 位于
`weft.runtime`。所有构建进入同一个 compiler service；Canonical Kernel IR、Physical IR 和
terminal translation 的权责不因 JIT、AOT 或部署位置而改变。

C/C++ 消费同一 artifact；`Package`、`PackageBuilder` 提供部署前的具名产物集合与固定配置构建。
其安装、资源生命周期和框架调用合同见 [C/C++ SDK 与 llama.cpp](native-sdk.md)。

`compile(definition)` 发现原生目标并立即返回可调用对象；`jit(definition)` 延迟到首次调用、
`prepare`、`run` 或 `materialize` 才编译。`CompileOptions.meta` 是作者声明的 source auto
绑定，其余选项是目标的有限物理绑定。`specialize(**meta)` 生成另一个显式绑定，
`with_options(options)` 替换整组配置；不会隐式改变 dtype、Encoding 或算法树。
动态 shape 沿函数 ABI 输入。普通标量的数值改变不触发重新编译。

`build(definition, target=Target(...))` 生成指定目标的持久 artifact，不在构建主机执行它。
`build_ir(canonical, target=...)` 接受外部前端生成的 Canonical Kernel IR；Weft 不链接该前端。
`native_library=False` 停在 C、header、ABI 与两层 IR；`Artifact.link(toolchain=...)`
只编译已经选定的 C 程序。系统 C 编译器和对应 target ISA 支持是生成 native library 的前提。

```python
import weft
from weft.runtime import CompileOptions, Target, Toolchain

call = weft.jit(affine, options=CompileOptions(meta={"B": 64}))
call(X, Y, 1.25, bias=-0.5)
prepared = call.prepare(X, Y, 1.25, bias=-0.5)
prepared()
Y2 = call.run(X, scale=1.25, bias=-0.5)
artifact = call.save("/path/to/new/artifact")
loaded = weft.load(artifact.path)
loaded(X, Y)
loaded.close()
```

`prepare` 绑定并检查完整调用，持有 kernel 与 buffer owner，重复调用仍核对执行能力和 Tensor
存储身份。普通调用与 `run` 返回声明为 write-only 的输出：一个输出返回对象，多个返回 tuple，
无输出返回 `None`。`run` 可分配省略的 dense write-only 输出；shape 必须从已有 View 或显式
shape 参数确定。readwrite 状态、workspace、packed output 和 persistent artifact 仍由调用方提供。
`run(..., out=Y)` 显式提供单输出，多个输出可用 tuple 或按输出参数名的 dict。
`prepare_run` 同时完成输出分配与参数准备；`PreparedCall.rebind(**updates)` 重新检查变化的
参数并返回新调用，未显式固定的 shape 重新从 View 推导。`time_ns()` 只计 native entry 与 ctypes
dispatch，排除调用前的 metadata/target 验证和调优状态恢复，写入仍更新 Tensor version。

## 内存与 Tensor

Python buffer 和 NumPy 数组通过 buffer protocol 借用；CPU Torch Tensor 直接借用其
存储，不经 NumPy、不复制、不 repack。非零 storage offset 可以保留。
`@weft.kernel(strided=("X", "Y"))` 允许这些 dense View 参数传入动态非负 element strides，
包括跨行切片、转置和只读 zero-stride broadcast；未声明的参数仍要求连续 storage。
可写 View 必须不自重叠，且 span 和 byte stride 必须适合目标指针 ABI；空 View 不访问存储。
负 stride、非 CPU device、conjugate/negative view、间接 buffer 和非本机字节序明确拒绝。
显式 `Buffer(bytes, shape=..., encoding=...)`
用于 packed storage。shape、record/cohort、对齐、容量、可写性和声明的 alias group 在调用前检查。

Tensor dtype、形状、stride、storage offset 或地址在 `prepare` 后变化会使原绑定失效。
调用方负责避免其它线程在 native 调用期间替换或修改相同存储；独立 buffers 可以并发调用同一
kernel。`close()` 拒绝新调用，等待已进入的调用退出再卸载库。关闭后的 prepared call 明确失败。

直接 Tensor 调用提供 inference 语义。启用 grad 且参数 `requires_grad` 时明确报 unsupported；
调用方可以在 `torch.no_grad()` 中运行。写入更新 Tensor mutation version；inference Tensor
的写入必须处于 `torch.inference_mode()`。该入口不隐式注册 autograd、FakeTensor 或图捕获算子。

## 公共调优

`weft.autotune(definition, configs, key=..., policy=..., validator=..., validation_key=...)`
接收少量完整 `Config`。每项的 `meta` 是 source 参数，其余字段是 `CompileOptions` 的物理绑定。
具名 source auto 由这份有限列表绑定；`auto(32, 64)` 的内部参数名为 `auto$32$64`，绑定必须属于
声明域；singleton 如 `auto(64)` 唯一确定绑定。`weft signature file.py:kernel` 在不启动 native
compiler 的情况下展示参数、输出角色、source 域与 shape。物理合法性和完整资源合同仍由 compiler 验证。

稠密 contraction 的 `contraction_column_copies` 接受 1/2/4，限制输出列方向的向量副本数；
`panel_transfer_fields` 接受 1/2/4/8，限制私有 panel 搬运的 segment 字段数；
`contraction_panel_k` 接受 32/64/128/256，限制私有 K panel 的深度；
`contraction_k_steps` 接受 1/2/4，限制一次供应的相邻 K 标量数。这些是有限物理选择的
上限：目标能力、输入布局与资源预算可能要求更小的合法实现，最终选择可从 artifact 查看。
它们不改变作者的 source tile、Level、persistent storage 或闭合 contraction 的数值边界。
两个内层 K 输入转成私有 panel 时，计算期的标量供应还受已知 cache line 和同时存活的
scalar tuple 大小限制；自然输出 lane、外部内存直读、未知 cache geometry 或两级软件流水
使用逐步供应。K 尾部独立处理，累加顺序保持不变。

```python
from weft.runtime import Config, TuningPolicy, TuningResult

configs = [Config(name="b64-m1", meta={"B": 64}, lmul_eighths=8),
           Config(name="b128-m2", meta={"B": 128}, lmul_eighths=16)]
tuned = weft.autotune(affine, configs, key=("scale", "bias"),
                     policy=TuningPolicy(max_candidates=8, deadline_s=120),
                     validator=check_affine, validation_key="affine-f32-tolerance-contract")
selection = tuned.tune(X, Y, scale=1.25, bias=-0.5)
tuned(X, Y, scale=1.25, bias=-0.5)
selection.save("/path/to/selection.json")
fixed = tuned.fixed()
saved = fixed.save("/path/to/fixed-artifact")
options = TuningResult.load("/path/to/selection.json").winner.options
```

`tune()` 只选择；普通调用选择后在调用方存储上执行一次 winner。shape、dtype/Encoding、stride、
实际 byte alias/offset、对齐类别、target/affinity、完整配置及工具链身份始终属于 workload key。
显式 `key` 增加会影响工作负载的有限数值 scalar/shape，不读取 Tensor 内容作为 key。
source 参数若改变公共 View 维度，不能在同一个 callable 的配置列表中取不同值。
绑定对象存活期间工具链视为固定；更换工具链应新建对象或关闭后重新选择。

候选先形成真实 artifact，编译拒绝保留原始诊断。多个合法候选用 `TuningPolicy` 的 warmup、
samples 与 repetitions 计时，按完整样本的中位数选择，同值保留列表顺序。deadline 在编译/调用
边界检查，不中断正在执行的 native code；因此一次编译或调用可以越过预算。未完成的测量标为
`budget`，不能胜出。只有一项合法且没有遗漏候选时跳过没有选择价值的计时，记录 `single-legal`。

`validator(original, trial)` 在私有 trial 调用完成后检查数值，成功返回 `None`，失败抛出
`AssertionError` 或 `ValidationError`；其它程序错误直接传播。`original` 与 `trial` 都按原 kernel
参数名提供对象和 scalar。`validation_key` 显式标识作者的容差/检查合同，合同变化必须更换此身份。
没有 validator 的选择明确记为 `validated=false`，不能把运行完成写成数值正确。

所有与 writable View 共享 bytes 的输入组成同一私有 storage group，保持其相对地址、stride
和所需对齐。初次复制只读取各 View 可访问的 byte 区段，不读取空洞；每次 warmup、验证和计时
前恢复私有快照。调用方的 state、output 和 Tensor version 不受试跑影响，异常路径也释放私有状态。
`max_state_bytes` 限制快照与 trial 存储；超限明确报错，不退回在调用方 state 上试跑。

选择记录复用 artifact cache 的精确身份索引、进程锁和原子发布；并发进程只执行一次冷调优。
`refresh=True` 成功后才替换记录；失败保留旧记录并向当前调用者抛出错误。
`TuningResult.candidates` 保存编译/数值拒绝、预算状态、样本、数值验证标记与 winner；全部失败时
`TuningError.candidates` 仍可检查。`weft selection /path/to/selection.json` 展示选择，`--identity`
展示完整身份。`weft build --selection ...` 固定使用 winner，并核对 canonical 程序和目标 ABI/ISA/VLEN；
新工具链下的固定构建不自动继承旧测量的性能结论。

## Torch 模块、图与训练

显式 `from weft.torch import register` 后，用 `register("namespace::operation", kernel)` 注册
JIT、autotuned callable、CompiledKernel 或 Artifact。注册从共同调用签名生成 CPU schema、
write/readwrite mutation 与 fake/meta 实现；注册和 fake 只检查 metadata，不发现目标、编译或读 Tensor
数据。真实 CPU 路径仍执行同一个 Weft artifact。每个命名 operator 在进程中注册一次，重名明确拒绝。

```python
from weft.torch import register

operator = register("application::affine", tuned)
Y = operator(X, scale=1.25, bias=-0.5)
operator.out(X, scale=1.25, bias=-0.5, out=Y)
```

functional operator 隐去 write-only 输出参数并返回新 Tensor；只有 readwrite state 的程序返回
`None` 并按 schema 修改 state。显式输出版本名字为 `operation_out`，通过独立 mutable schema
接受 `out=`，返回 `None`，不返回带别名的 Tensor。多个输出用 Tensor 列表按声明顺序提供。
显式输出 schema 将每个输出单独声明为 `output_0`、`output_1` 等可变 Tensor；公开 `out=`
按声明顺序 tuple/list 或按参数名 dict 绑定这些输出，避免可变 Tensor 列表的 functionalization 限制。
输入 Tensor 必须位于 keyword-only 边界之前；普通 scalar 的 defaults/keyword-only 语义保留。
packed Encoding 的框架 schema 尚未定义，应使用显式 Buffer ABI。

Torch 图调用若两个参数共享 storage 且至少一个可写，明确报 unsupported，因为框架
functionalization 可能分别复制两个参数而改变 alias 语义。可改成一个 readwrite View；raw Weft
与 autotune 的私有状态合同允许真实别名。框架不会用 `.contiguous()` 或隐式复制修补 View 条件。

`operator.register_backward(backward, setup_context=...)` 使用 torch.library 标准 callback 合同。
它只适用于没有 mutable input 的 functional operator；显式 out 不定义 autograd。
backward 是作者程序，不是 Weft 自动求导。`source/weft/dense/torch_affine.py` 给出普通 `nn.Module`、
调优 forward 和另一份 Weft DSL backward 的完整用法，包括 zero-stride gradient。

`torch.compile` 将 Weft 算子作为 opaque CPU 调用，`torch.export` 保存名称、schema 和图；在新进程
加载图之前需要注册对应 operator。保存的 Artifact 可以用于注册，不依赖原 Python DSL 或系统 C
compiler。框架图的捕获、实际执行和图级性能是不同结论；Weft 不接管其它图节点的编译。
接口行为遵循 [PyTorch custom operator 合同](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)。

## 持久产物与缓存

artifact 保存 canonical/physical IR、C、精确 C declaration、typed ABI、显式 target、编译绑定、
Python 参数顺序/default/keyword-only 语义以及可选 shared object。`save()` 向新的目录原子发布；
已有目录不覆盖。`load()` 不需要 Python kernel 定义或系统 C 编译器；执行前仍需原生硬件发现。
已加载同一进程的 `NativeTarget` 可以显式传入 `Artifact.load(target=...)`。

编译产物的 `physical_resources` 记录 final verifier 已核对的 vector/fragment 驻留峰值、
共同寄存器预算、私有存储字节上界及其预算。`Artifact.physical_resources`、
`CompiledKernel.physical_resources` 和 `weft inspect ... --emit resources` 只读取这份结果。
动态对象的字节数是已验证 guard 保证的上界。它不是 Clang 最终栈帧大小、硬件寄存器实测
或性能预测；system compiler 仍可能产生 spill。不存在该诊断字段时明确报告缺失，不返回零。

`Artifact.physical_operations` 与 `CompiledKernel.physical_operations` 返回只读的
`PhysicalSummary`。它从最终 Physical IR 汇总已经选中的局部指令、访存形式、mask/tail 合同、
局部资源声明，以及 Value/Fragment 的逻辑轴和物理布局：SEW、LMUL、VL、lane/replica 因子。
`matching(engine=..., family=..., instruction=..., memory_form=...)` 按精确字段筛选，
`select(...)` 返回可直接打印和导出的子集。`PhysicalOperation.access` 保留已选访问的对齐、
索引宽度、segment 数量及编码的 group/layer/join 几何；没有访问属性时明确为 `None`。
操作、leaf、access 与各输入/输出布局相同的实例归为一组，`occurrences` 是包含全部分支与
循环体的静态实例数，不代表动态执行次数或最终机器指令数。存储 descriptor 和完整 use-def
仍在 `physical.mlir` 中；布局摘要不取代资源 verifier 或原生性能测量。

```sh
weft inspect /path/to/artifact --emit operations --format text
weft inspect /path/to/artifact --emit operations --memory-form segment --format text
weft inspect /path/to/artifact --emit operations --format csv > /outside/operations.csv
```

默认格式为 JSON；CSV 每行对应一个静态操作组，嵌套的布局和 source location 保留为 JSON
字段。查看不加载 native library，也不启动编译。保存、链接与 package 部署保留这份编译结果；
没有该字段的产物明确报告缺失，不从 C 文本猜测布局或重新编译。

运行 target 的 XLEN、ABI、VLEN 必须匹配，ISA 必须覆盖构建目标，vendor extension 及 affinity
继续按实际使用检查。可调用产物是 native code，加载者选择可信的产物来源。
`DataCache(size_bytes, line_bytes, ways, sets)` 保存已报告的 L1 data/unified cache geometry。
`Target.data_cache=None` 表示未知；交叉构建可显式提供，命令行使用
`--data-cache=SIZE_BYTES:LINE_BYTES:WAYS:SETS`。原生发现读取实际 affinity 中每个 CPU 的
cacheinfo，`NativeTarget.cpu_data_caches` 保留逐 CPU 观察；只有全部相同才形成共同
`data_cache`。缺失 cacheinfo 保持未知，已发布但不完整或矛盾的 geometry 报错。
这些事实进入编译与调优身份，并随 artifact 保存；它们用于物理工作估计，不是 ISA 运行要求，
也不把 L1 容量当成私有存储保证。执行同一 ISA/ABI/VLEN 的产物不要求 cache geometry 相等。
标准 ISA 的版本与蕴含扩展由 compiler 的 LLVM ISA parser 和原生能力发现导出，Python 不维护
第二份扩展依赖表。构建 flags 可表达的 ISA 不等于 Linux 允许用户态直接执行的 ISA；例如
`g` 包含 `zifencei`，不能仅凭基础整数扩展就假定用户态 `fence.i` 可用。

缓存位于 `WEFT_CACHE_DIR` 或 XDG cache 下。索引使用完整 canonical 程序、序列化签名、
target、选项、实际命令配置、runtime 模块、工具路径/stat/version、声明依赖与相关工具链环境的精确身份。
工具链的额外 SDK 文件通过 `Toolchain.dependencies` 显式加入；此身份不宣称覆盖任意系统文件
变化。更新 SDK 时应改变工具链身份或选用新的 cache directory。
`Artifact.cache_status` 区分 `built`、`hit` 与显式 `loaded`；后者没有进行编译缓存查询。
缓存条目的 `identity.json` 保存完整身份，可直接检查程序、签名、flags、目标或工具链的变化。

SQLite 事务分配随机 slot，同一 slot 的进程锁串行化首次构建，不同程序可独立构建。
仅完整生成并写入 manifest 的 staging 目录才原子发布。失败不产生成功条目；错误和损坏明确
报告，不能静默调用旧实现。关闭句柄不删除持久缓存。`weft cache` 只列出完整条目。

## 安装与命令行

运行环境需要 Linux、Python 3.10+。wheel 包含 Python API、Weft 编译器与 optimizer；SDK
使用共享 LLVM 时同时安装它的 SONAME 文件。wheel 是构建平台的本地产物，依赖该平台兼容的
系统运行库；不把本地 wheel 宣称为通用 manylinux 包。使用者不需要 LLVM headers 或 C++ 构建
工具来加载 wheel，但 JIT 仍需要支持目标 RVV intrinsic 的 C compiler。

源码安装需要 pip 23+、CMake、Ninja、C++17 编译器及 LLVM/MLIR SDK。`tools/install.py` 将 Python
依赖和 Weft 安装到指定虚拟环境，系统 Python 保持独立。构建依赖可从显式 wheelhouse 离线提供。

```sh
python3 tools/install.py --environment /path/to/venv \
  --llvm-root /path/to/llvm-sdk --build-dir /path/to/weft-build --jobs 4
/path/to/venv/bin/weft --help
/path/to/venv/bin/weft target
```

`CC`、`CFLAGS`、`LDFLAGS` 与 `WEFT_COMPILER` 配置默认工具链；也可以显式传 `Toolchain`。
命令按参数数组执行，不通过 shell 展开。构建错误包含阶段、命令及编译器原始诊断。

Torch 是可选依赖，需按 Python ABI 与 glibc 选择真正的 RISC-V wheel。
[RISE 的 wheel 索引](https://pypi.riseproject.dev/project/torch/)提供发行包；例如在符合其
wheel tag 的 Python 3.12 / glibc 2.39 环境中：

```sh
python -m pip install --extra-index-url https://pypi.riseproject.dev/simple/ --prefer-binary torch numpy
```

Python 3.11、较旧 glibc 可使用 [KumaTea 构建的对应 wheel](https://github.com/KumaTea/pytorch-riscv64/releases/tag/2.4)。
它的 Torch 2.4.1 使用 NumPy 1.x ABI，应在独立环境中一起指定依赖：

```sh
python -m pip install --only-binary=:all: --extra-index-url https://ext.kmtea.eu/simple 'numpy<2'
python -m pip install https://github.com/KumaTea/pytorch-riscv64/releases/download/2.4/torch-2.4.1-cp311-cp311-manylinux_2_35_riscv64.whl
```

这些是外部维护的框架发行包，Weft 不包装它们的计算实现。没有匹配包时需按该框架的构建
说明先完成原生安装；不能把 wheel 文件改名后强行加载到不匹配的 Python/系统 ABI。

```sh
weft lower source/weft/dense/affine.py
weft build --kernel source/weft/dense/affine.py:affine --meta B=64 \
  --march rv64imafdcv --abi lp64d --vlen-bits 128 --source-only --output /path/to/affine
weft inspect /path/to/affine --emit header
weft link /path/to/affine --cc clang --output /path/to/affine-native
```

四类数值 runner 使用相同 `weft.build` 和 `Deployment`。`WEFT_TARGETS` 指向仓库外的机器
配置 JSON，按 target 名提供 `target`、`deployment`、`toolchain`、`cxx`、`defines` 与需要时的
`ggml` 路径。`deployment` 显式声明 SSH host、目标 Python、workspace 与 CPU 列表。
目标 Python 环境必须已安装 Weft。部署只运行调用方指定的 C++ numerical driver，保留其原有
reference 与计时范围；不会启动模型 e2e，也不在原生 JIT 失败时自动转向 SSH。

`examples/run/weft-kernel-tune.sh RUNNER TARGET KERNEL [PHASE] REPETITIONS --configs configs.json
--output /outside/selection.json` 读取 `Config.to_dict()` 对象列表并调用同一个公共 `Study`。
driver 自己检查数值并报告 native median，SSH、编译和部署耗时不进入样本。每候选的 driver 只
构建一次，重复计时仍初始化独立输入；该计时范围在结果中明确标为 `numerical-driver-native-median-ns`。
普通 runner 的 `--config` 接受一项 Config，`--selection` 接受保存结果并检查完整 workload/目标身份。
