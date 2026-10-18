# Triton JIT、运行与部署

公开入口为 `import triton`、`import triton.language as tl` 和 `@triton.jit(model="weft")`。
Weft 使用同一个 JITFunction、ASTSource、CompiledKernel、缓存和 autotuner。
普通 Triton 程序保留上游 GPU 路线；模型选择不覆盖进程的全局 GPU driver。

## 编译边界

```text
Triton JITFunction / Python AST
  → 共同 CodeGenerator + Weft 类型、作用域和 SSA 语义
  → Canonical Kernel IR
  → RISC-V Physical passes
  → intrinsic C / 已选择的 local asm
  → Clang → native shared library
  → Triton CompiledKernel
```

Canonical 程序在同一进程通过 native bindings 建立和验证，不先转换为普通 TTIR，
不调用外部 Weft 编译器。编译库使用 Triton 选择的 LLVM/MLIR SDK。
Clang 子进程属于 C→native 编译阶段；已加载 kernel 的执行不启动编译器。
`CompiledKernel.asm` 提供 `weftir`、`riscvir`、`c` 和 `so`。

Encoding、derived builder、helper、模型、alias/stride 声明及 constexpr 进入编译依赖。
运行时 scalar、动态 shape 和 stride 由 typed ABI 传入；普通 scalar 的数值不自动改变程序。
`tl.constexpr` 明确固定作者程序参数。数值、归约、Level/handoff 合同不因复用 Triton 运算而放松。

## grid 与线程

```python
compiled = kernel[(instances,)](X, Y, B=64, num_cpu_threads=8)
```

grid 表示独立 program instances；num_cpu_threads 配置 OpenMP。
program_id 和 num_programs 在前端变为实际使用的显式入口 scalar；未使用的坐标不进入 ABI。Canonical 核心不增加
hart、warp、thread ownership 或隐式跨实例同步。每个实例执行一份有序程序，
源码负责数据分区和尾部。单实例直接调用；多实例单线程使用原生循环；
多线程使用原生 OpenMP launcher，各 worker 在写入前检查执行能力。
num_cpu_threads=0 使用 OpenMP 当前线程设置。

跨实例别名、数据分区与同步由作者保证。已有线程框架调用导出的单实例入口，
沿用其任务划分，不在每次算子调用内部建立另一组线程。

## View 与 CPU Tensor

NumPy 和 buffer protocol 对象借用原存储。CPU Torch Tensor 直接借用地址，
保留 dtype、shape、stride、storage offset 和别名，不经 NumPy、不复制、不 repack。
装饰器的 `strided=("X", "Y")` 允许非负 element strides；只读 zero-stride broadcast 合法，
可写 View 不允许自重叠。负 stride、非 CPU 设备、conjugate/negative view 和不匹配 Encoding 明确拒绝。

普通 pointer 参数只约束存储 dtype 和 access，不隐式约束 Tensor rank 或 contiguous。
其 stride/offset 由源程序显式计算，mask 决定实际访存。runtime 保留真实 storage offset，
不把一个跨 stride 的 Tensor 自动改成连续数组，也不根据空的 pointer shape 推断输出分配。

```python
packed = tl.View(storage_bytes, encoding="Q4_0", shape=(K,))
kernel[(1,)](packed, activation, output)
```

调用检查容量、record/cohort、对齐、可写性和 alias group。
热调用由原生 launcher 读取当前 Tensor 或 buffer metadata，并复用 C SDK 的参数合同与 ABI 打包；
不在 Python 中逐次构造 Buffer 和 ctypes 参数数组，也不缓存调用者的可变 Tensor metadata。
调用方不能在执行期间替换存储或改变线程能力。写入更新 Tensor mutation version。
原生 callable 持有模块和 SDK handle，最后一个 callable 释放后再卸载 kernel；
丢弃 CompiledKernel 不会提前释放仍在执行或被调用方持有的原生入口。
训练通过显式 backward 或框架注册表达，不自动推导梯度。

## 配置与 autotune

```python
configs = [
    triton.Config({"B": 64}, backend_options={"lmul_eighths": 8}),
    triton.Config({"B": 128}, backend_options={"lmul_eighths": 16}),
]
tuned = triton.autotune(configs=configs, key=[])(kernel)
tuned[(instances,)](X, Y)
```

Config.kwargs 绑定 source 参数，backend_options 绑定有限物理参数，num_cpu_threads 配置 launcher。
候选继续经过数值、effect 和资源合法性检查；资源或不支持候选可被拒绝，
compiler-error 传播原始诊断。计时不能覆盖非法程序。
候选枚举、pruning、hook、CPU benchmark 和缓存由 Triton autotuner 承担。

Weft 调优 key 包含 Encoding、shape、stride、对齐类别、真实别名偏移、目标、CPU affinity 和 launch 配置；
显式 key 可再包含影响工作量的 scalar。试跑在私有存储中进行，共享 writable bytes 的所有输入
保持相对地址和 alias。每次试跑恢复私有快照，调用者 state、output 与 Tensor version 不因测量改变。
选定后只在调用者存储上执行一次 winner。数值正确性由应用与 Bench 按声明容差检查，
计时成功本身不表示数值已经验收。

驱动按当前允许 CPU 集合复用硬件事实，避免每次已调优调用重新扫描 sysfs；
实际 worker 的 RVV 权限、affinity 和扩展检查仍在每次执行前进行。
改变调用线程的 affinity 后，JIT 使用对应集合的加载上下文；ISA 和工具链相同的原生代码
仍可命中编译缓存。调用期间的存储与线程条件由调用方保持。

## 工具链与缓存

toolchain 描述 cc、target、sysroot、cflags、ldflags、sysroot_identity 和 runtime_files。
发行包可携带 Clang、完整 resource headers、OpenMP headers 和运行库；
本机也可通过 TRITON_WEFT_CC 或显式 toolchain 选择实际支持 RISC-V 的 Clang。
交叉构建必须声明 sysroot 和其 SDK 身份，不使用 SSH 编译服务。

ISA/ABI flags 必须与编译目标一致。工具路径、版本、stat、相关环境、sysroot 身份、
编译/链接参数和运行依赖进入编译缓存身份。更改 sysroot 内容时必须更新其身份。
明确的 target VLEN 同时通过 `-mrvv-vector-bits=N` 交给 Clang，不能在 intrinsic C
边界丢失已经用于 physical layout 和执行检查的机器事实；显式冲突的 VLEN 参数拒绝。
CPU 调度参数可通过工具链的 `-mtune` 显式声明，它不替代 ISA、ABI 或实际执行能力检查。
只使用 Triton cache，通过 TRITON_CACHE_DIR 选择目录，不建立第二套 Weft 缓存。
开发测量的 cache、dump、native 产物均由 Bench 配置在其输出目录。

## 导出与加载

```python
compiled.export("/application/kernels/affine")
loaded = triton.load("/application/kernels/affine")
loaded[(instances,)](X, Y, scale=1.25, bias=-0.5)
```

加载 callable 接收固定 constexpr 之后的 runtime 参数，保留默认值和 keyword-only 合同；
grid 可为一至三维。已有导出目录不覆盖，完整产物在临时目录写完后发布。
加载不需要原 kernel、现场 Clang 或外部编译服务，执行仍检查目标 ABI/ISA/VLEN 和能力。

目录包含 kernel.so、kernel.h、triton.json、manifest.json、IR/C 及运行依赖。
顶层 manifest 是无线程团队的单实例 ABI，grid 字段是完整 launcher ABI。
二者引用同一份编译程序；不能把单实例 entry pointer 传给 grid launcher。
triton.Package.create 将已选择的 CompiledKernel 组成具名集合，见 [native SDK](native-sdk.md)。

## PyTorch 分析边界

eager CPU Tensor 使用普通 Triton 调用。Canonical interface 分析不访问存储、不发现设备，
也不进行 native 编译，提供参数读写、alias、shape 和 stride 合同。
torch.compile 必须由 PyTorch 的 Triton HOP 显式接入该模型接口并保持模型启动方式；
只识别 JITFunction，或在 TTIR 分析失败后假定所有参数可写，都不构成支持。
框架接入不提供另一套 Weft dispatcher 注册 API，不承担 Inductor 的整个 CPU 图编译。
自动图使用 Inductor 现有 `cpu_backend="triton"`：其代码生成器保留普通 Triton 表达式，
CPU kernel 明确绑定 Weft 模型；编译、缓存身份与 launcher 按模型选择，避免改动全局 GPU driver。
调优候选的 unsupported/resource-limit 由原候选枚举处理，compiler-error 保持原诊断；
readwrite 和 alias 输入的 trial copies 保持共享存储关系，reset 作用于这些私有副本。
