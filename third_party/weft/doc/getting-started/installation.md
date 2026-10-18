# 安装扩展 Triton

本仓库构建一份提供 import triton 的定制 Triton。不要在同一环境同时安装两份覆盖
triton 模块的发行包。Weft 编译核心位于 third_party/weft，没有需要单独安装的 Weft Python 包。

## 源码构建

使用独立 Python 3.10+ 环境、CMake、Ninja 和支持 C++20 的编译器。
LLVM/MLIR 必须匹配仓库 cmake/llvm-info.json 指定的 revision；Weft、Triton、
TableGen 与 native bindings 共用这一份 SDK，不能链接旧的独立 Weft LLVM 产物。

```sh
python -m venv /path/to/environment
/path/to/environment/bin/python -m pip install -r python/requirements.txt
LLVM_SYSPATH=/path/to/matched-llvm-sdk \
TRITON_WEFT_CLANG_ROOT=/path/to/native-clang-23 \
TRITON_WEFT_OPENMP_LIBRARY=/path/to/libomp.so \
TRITON_WEFT_OPENMP_INCLUDE_DIR=/path/to/openmp-headers \
TRITON_BUILD_DIR=/path/to/build \
TRITON_BUILD_PROTON=OFF \
/path/to/environment/bin/python -m pip install --no-build-isolation .
```

PROTON=OFF 只关闭可选 profiler，不移除标准 GPU 编译路线。
这份命令同时安装本机 JIT 所需的 Clang 与 OpenMP；路径应选择运行 wheel 的机器架构。
只构建编译库时可以省略这三个打包参数，但运行 kernel 前需要显式配置目标工具链。
开发安装可使用 pip install -e .；build、compiler cache 和测量输出使用仓库外目录。
RISC-V 机器需要适配其 Python ABI、libc 和 C++ runtime 的编译环境。
交叉构建 Python extension 时可通过 TRITON_TARGET_PYTHON_INCLUDE_DIR 指定目标 headers，
构建工具仍使用主机 Python；架构 pyconfig.h 必须来自目标 Python。
交叉 wheel 还必须给出目标实际报告的 TRITON_TARGET_PYTHON_SOABI，并选择
`bdist_wheel --plat-name linux_riscv64`。打包器核对 SOABI 与目标 headers 的 Python 版本，
不会把主机 CPython ABI 标签贴到目标扩展上。

## 发行包中的 C 工具链

发行工具链使用 Clang 23。构建 wheel 时指定完整 Clang 和同架构 OpenMP：

```sh
TRITON_WEFT_CLANG_ROOT=/path/to/native-clang \
TRITON_WEFT_OPENMP_LIBRARY=/path/to/libomp.so \
TRITON_WEFT_OPENMP_INCLUDE_DIR=/path/to/omp-headers \
LLVM_SYSPATH=/path/to/matched-llvm-sdk \
TRITON_BUILD_DIR=/path/to/build \
TRITON_BUILD_PROTON=OFF \
python -m pip wheel --no-build-isolation --no-deps . -w /path/to/wheels
```

Clang 目录必须包含 bin/clang、LICENSE.TXT、完整 lib/clang/<version>/include 和 RISC-V codegen。
OpenMP headers 必须含 omp.h。构建会将 compiler、resource headers、runtime、
C/C++ headers、llama.cpp adapter 和许可材料放入同一 Triton wheel。

wheel 是构建平台和 Python ABI 的本地产物，不默认承诺 manylinux 兼容。
部署到 RISC-V 的 Clang、libtriton 和 runtime 都必须是目标原生机器码；
主机 x86 Clang 可以用于交叉编译，但不能复制到 RISC-V 当作本机 JIT compiler。
RISC-V wheel 安装 Weft CPU backend，不携带无法在 RISC-V 执行的 CUDA/ROCm 工具。
标准 GPU backend 保留在源码以及其支持的主机发行包中。

## 本机 JIT 与交叉构建

本机 JIT 默认使用 wheel 中的 Clang，也可以显式设置 TRITON_WEFT_CC 或传 toolchain 字典。
JIT、AOT 与模型 package 使用同一工具链描述；不会为了某个算子静默切换旧 Clang。
系统 libc、headers、linker 与启动文件由声明的目标环境提供。未配置或缺少所需 ISA 支持时报错。
本机使用 bundled Clang 时，会查询系统 GCC 提供的 ABI、libgcc 和共享库启动文件，
将经过核对的安装目录传给 Clang，并把运行库身份纳入编译缓存；编译器始终是随包 Clang。
没有 GCC driver 时，Clang 使用自己的系统运行库发现机制。显式 sysroot/GCC/runtime 参数保持优先。
若需选择另一套已声明的 GCC 运行库，可传入
`toolchain={"cflags": ["--gcc-install-dir=/path/to/target/gcc/version"]}`。
使用 bundled Clang 时，额外 flags 保留随包提供的 OpenMP include、library 和运行依赖；
显式指定 cc 时则由完整 toolchain 描述这些依赖。不会自动改用系统 GCC 编译 kernel。

交叉构建使用 triton.compile(ASTSource(...), target=CPUTarget(...), options=...)，
给出 cc、target triple、sysroot、sysroot_identity、cflags、ldflags 和必要 runtime_files。
这个入口只产生目标代码，不尝试在构建主机运行目标指令，也不依赖 SSH。

```python
from triton.backends.compiler import CPUTarget
from triton.compiler import ASTSource
compiled = triton.compile(
    ASTSource(kernel, signature={}, constexprs={"B": 64}),
    target=CPUTarget("weft", "riscv64", target_march, target_abi, target_vlen),
    options={"toolchain": toolchain},
)
compiled.export("/application/kernels/selected")
```

ABI、ISA、VLEN 和工具链必须与目标实际能力匹配。AOT 加载不需要原 DSL 或现场 Clang。
SDK 路径通过 triton.runtime.package.sdk_prefix() 取得。

## 可选应用依赖

NumPy 和 PyTorch 由应用选择适配目标的发行包。CPU Tensor eager 走相同 native kernel。
PyTorch 图接入使用 2.14 的 Triton HOP/Inductor 接口，并提供最小源码接缝安装工具：

```sh
python -m triton.tools.pytorch
```

在专用环境安装时运行该命令；也可用 --torch-root 指向明确选择的 Torch 源码目录。
工具在写入前检查所需源码接口的结构和函数签名，接口不匹配时报 unsupported。
它保留原 Triton HOP，增加模型的 CPU capability、Canonical effects/FakeTensor、alias
functionalization 与 native launch 接口；用户仍使用 kernel[grid] 和 torch.compile，
不注册独立 Weft dispatcher。Inductor 的 `cpu_backend="triton"` 通过同一接缝选择 CPU
模型、编译目标和 native grid launcher，保留原有图调度、融合和候选枚举。
运行时不 monkey patch。旧 Torch 环境可使用 eager CPU Tensor；
图接入需要上述接口，不能把 import 成功当作支持。
数值与性能验证由独立 Weft Bench 承担，不随产品安装测试 runner。
