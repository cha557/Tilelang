# Sunmmio Device Backend Design

## 1. 概述

提交 `80d95ea081993bd861d2fd476ecc6dd827a85813` 的核心目标，是把 TileLang 已有的 Sunmmio 编译变换能力接入到 JIT 执行体系中，使用户能够通过 `target="llvm -mcpu=sunmmio-a4e ..."` 和 `execution_backend="sunmmio"` 走一条面向 Sunmmio 设备的编译、封装、缓存和调用链路。

在这个提交之前，TileLang 已经有面向 Sunmmio 的算子语义、layout 推导、SRAM scope 推导、同步注入和部分目标代码生成逻辑，但这些能力还没有被标准 JIT backend 统一承载。用户侧的 `tilelang.jit.compile`、`tilelang.cache.cached` 和 `JITKernel` 仍然主要围绕 CUDA、Metal、CuTeDSL、Cython 或 TVM FFI backend 组织，无法把 Sunmmio target 自动解析成专用执行后端，也无法保存 Sunmmio 相关的 Python wrapper、launcher library、MLIR 或 ELF 产物。

本方案将 Sunmmio 作为一个独立 execution backend 接入，而不是把它伪装成 CUDA 或 CPU backend。这样做的关键原因是 Sunmmio 的编译产物和运行时调用模型与现有 backend 不同：设备侧需要先生成 Sunmmio 方言或设备 MLIR，再经外部工具链得到 ELF；host 侧需要生成一个面向 SuTensor/Sunmmio runtime 的 launcher；Python 侧还需要继续保持 PyTorch 风格的调用接口，负责输出 tensor 分配、动态 shape 解析、stream/device 信息传递和缓存恢复。

提交中的实现仍带有实验性质。它已经把 backend 名称、target 判定、JIT adapter、wrapper 生成、launcher 编译和 cache 类型接入主流程，但 `compile_mlir()` 仍是空实现，lowering 中存在临时 CUDA target 绕行，C++ launcher 里也还保留占位逻辑。因此这份文档把它整理为一个可落地的方案：先说明已经建立的接口和模块边界，再明确需要补齐的编译器、runtime 和测试闭环。

## 2. 设计目标

Sunmmio backend 的用户入口应与 TileLang 现有 JIT 入口保持一致。用户不需要直接调用底层 codegen 或 runtime，只需要在 `tilelang.jit.compile`、`@tilelang.jit` 或 `tilelang.cache.cached` 中指定 Sunmmio target；当 target 满足 `llvm` 且 `mcpu` 以 `sunmmio-` 开头时，backend 解析逻辑应选择 `sunmmio`，并把后续 lowering、wrapper 生成、runtime 加载和缓存恢复都交给专用 adapter。

这条链路需要满足三个约束。第一，编译阶段要保留 TileLang 的通用 lowering pipeline，让 Sunmmio 已有的 layout、scope、sync、pipeline 和 operator lowering pass 继续工作。第二，设备侧产物不能依赖 TVM FFI runtime module 的 CUDA launch 模型，而应输出 Sunmmio 设备代码，并允许后续接入 MLIR 到 ELF 的外部编译器。第三，运行时调用要保持 PyTorch 友好，用户仍然传入 torch tensor，adapter 负责按 PrimFunc 参数顺序组装输入和输出，调用预编译的 Python wrapper/launcher。

方案边界也需要明确。该提交不定义 Sunmmio operator 语义本身，也不修改 `dma_copy`、`mma_sunmmio`、layout inference 或 SRAM scope 的 compiler contract；这些 contract 仍由已有 Sunmmio pass 和算子实现负责。该提交关注的是 JIT backend 集成，即如何从已经 lower 后的 host/device module 生成可加载、可缓存、可调用的 Sunmmio kernel package。

## 3. 总体链路

整体链路可以拆成 target/backend 解析、TileLang lowering、Sunmmio adapter 构造、wrapper/launcher 生成、产物加载和缓存复用六个阶段。

```text
User API
  |
  | tilelang.jit.compile / tilelang.cache.cached
  v
Target and backend resolution
  |
  | target = llvm -mcpu=sunmmio-a4e ...
  | execution_backend = sunmmio
  v
TileLang lowering
  |
  | PrimFunc -> host_mod + device_mod + params + device source
  v
SunmmioKernelAdapter
  |
  | compile device MLIR to ELF
  | generate Python wrapper
  | generate C++ launcher library
  v
Python callable
  |
  | allocate outputs
  | resolve dynamic shapes
  | call launcher with tensor handles, stream, device_id
  v
Sunmmio runtime
```

在 API 层，`execution_backend` 的类型集合新增 `sunmmio`，`tilelang.cache.__init__` 中的 dispatch map 也新增 `SunmmioKernelCache`。在 target 层，`is_sunmmio_target` 使用 `target.kind.name == "llvm"` 和 `target.attrs["mcpu"]` 中的 `sunmmio-` 前缀识别 Sunmmio 设备；`resolve_execution_backend` 在 target 命中时自动选择 `sunmmio`。

在 lowering 层，`tilelang.engine.lower.device_codegen_without_compile` 对 Sunmmio target 调用 `target.build.tilelang_sunmmio_without_compile`，让 device module 输出 Sunmmio 设备源码而不是 LLVM host code。CMake 同时把 `src/target/codegen_sunmmio.cc` 和 `src/target/rt_mod_sunmmio.cc` 编入 TileLang runtime，使对应 TVM global function 能够在 Python lowering 阶段被找到。

在 adapter 层，`JITKernel._compile_and_create_adapter` 增加 `SunmmioKernelAdapter` 分支。该 adapter 接收 lowering 产出的 `params`、`host_mod`、`device_mod` 和 `device_kernel_source`，再由 `SunmmioLibraryGenerator` 负责设备编译、launcher 编译和 Python module 加载。

## 4. 核心模块职责

`tilelang.jit.execution_backend` 负责把 Sunmmio target 纳入统一 backend 解析。它的设计语义是：只要 target 是 `llvm` 且 `mcpu` 表示 Sunmmio 设备，就不走默认 CPU `cython` fallback，而是只允许 `sunmmio` backend。这样可以避免 Sunmmio target 被误当作普通 LLVM CPU target 编译，也能让用户使用 `execution_backend="auto"` 时获得正确行为。

`tilelang.engine.lower` 负责把 Sunmmio 设备 module 导出为后续工具链可以消费的源码。提交中接入的是 `target.build.tilelang_sunmmio_without_compile`，表示 TileLang 先完成设备代码生成，但暂不在 TVM 内部完成最终设备编译。这个决策和 Sunmmio 工具链形态相匹配：TileLang 负责从 TIR 到 Sunmmio MLIR 或设备源的转换，MLIR 到 ELF 的动作由 `SunmmioLibraryGenerator.compile_mlir()` 承接。

`SunmmioKernelAdapter` 是 JIT 层的核心桥接对象。它继承 `BaseKernelAdapter`，对外暴露 PyTorch callable，对内维护 PrimFunc 参数信息、输出索引、动态符号映射、编译产物路径和已加载 Python module。初始化时它先检查 Sunmmio 环境，再缓存每个参数的 dtype 和 shape，把动态 shape/stride 中出现的 `tir.Var` 映射回某个输入 tensor 的 shape 或 stride。这个映射用于运行时自动分配输出 tensor，避免用户显式传入 output buffer。

`TLSunmmioSourceWrapper` 负责从 TileLang IRModule 中恢复 host 调用顺序，并生成两类 host 侧代码。第一类是 Python wrapper，它通过 `tvm_ffi.load_module` 加载 C++ launcher，并把 torch tensor 包装成可传给 TVM FFI 的 opaque pointer。第二类是 C++ launcher，它定义 `launch_kernel` 入口，后续应在这里完成 SuTensor descriptor 提取、ELF 加载、kernel 参数组装和 runtime launch。提交里的 launcher 仍是骨架，只生成函数签名和 buffer handle 入口，实际 Sunmmio runtime 调用需要继续补齐。

`SunmmioLibraryGenerator` 负责把 wrapper 源码、launcher 源码和设备源码落到临时目录并构造成可 import 的 Python module。当前实现会生成 `kernel_launcher.py`，用 `gcc -shared -fPIC -std=c++17` 编译 `launcher_lib.so`，并通过 `importlib.util.spec_from_file_location` 加载 Python wrapper。它还预留了 `kernel_elf_path`，用于保存 MLIR 到 ELF 编译后的设备产物。

`SunmmioKernelCache` 负责把 Sunmmio backend 的多产物缓存到磁盘。相比普通 CUDA/Cython cache 只关心 kernel source 和 `.so`，Sunmmio cache 需要额外保存 `kernel.mlir`、`kernel.elf`、`kernel_launcher.py` 和 `launcher_lib.so`。这样从数据库恢复时，`SunmmioKernelAdapter.from_database` 可以直接加载已生成的 Python wrapper 和 launcher library，而不必重复 lowering 和编译。

## 5. 运行时参数与动态 shape

Sunmmio adapter 运行时调用遵循 PrimFunc 参数顺序，而不是简单地把用户输入原样转发给 launcher。adapter 会先根据 `result_idx` 找出哪些参数是输出，再把用户输入填回非输出参数位置。对于输出参数，adapter 根据初始化阶段缓存的 dtype 和 shape 创建 `torch.empty`，并将其放回对应参数位置。

动态 shape 的处理依赖 `_process_dynamic_symbolic()`。它先扫描输入 tensor 的 buffer shape，再扫描输出 tensor 的 buffer shape，然后扫描输入和输出 tensor 的 stride。扫描顺序与 CUDA wrapper 的动态符号语义保持一致，但优先从输入 tensor 解析符号，因为输出 tensor 在分配前尚不存在。运行时如果输出 shape 中包含 `tir.Var`，adapter 会查表找到引用的输入 tensor 维度或 stride，并用实际 tensor 属性填充输出 shape。

完成参数物化后，adapter 会推断 stream 和 device。提交中的逻辑仍沿用了 CUDA 风格：如果 target 字符串以 `cuda` 开头且 CUDA 可用，则取当前 CUDA stream，否则 stream 传 `0`；`device_id` 从第一个 torch tensor 的 device index 获得。最终调用路径是 `self.pymodule.call(*args, stream=stream, device_id=device_id)`，由生成的 Python wrapper 继续转发到 C++ launcher。

这里有一个需要后续收敛的设计点：Sunmmio 设备上的 tensor 句柄、stream 语义和 device id 不应长期复用 CUDA 判断逻辑。最终实现中，`SuTensorHandle` 应通过 torch-sunmmio 或 SuDeck/SuBase 暴露的 API 获取真实 SuTensor 指针或 descriptor，而不是直接使用 `tensor.data_ptr()` 作为 opaque pointer。

## 6. 编译产物与缓存布局

Sunmmio backend 的缓存目录应包含足够恢复一次 JIT 编译结果的全部信息。提交中的命名约定如下：

```text
<TILELANG_CACHE_DIR>/<key>/
  kernel.mlir
  kernel.elf
  kernel_launcher.py
  launcher_lib.so
  params.pkl
```

其中 `kernel.mlir` 对应 TileLang 设备侧 source，`kernel.elf` 是 Sunmmio 外部工具链生成的最终设备二进制，`kernel_launcher.py` 是 Python wrapper，`launcher_lib.so` 是 host C++ launcher，`params.pkl` 保存 `KernelParam` 信息。cache key 已经包含 `execution_backend`、target、compile flags、pass configs、TileLang 版本和 runtime library stamp，因此同一个 TIR 在不同 Sunmmio target 或不同编译选项下不会错误复用。

从磁盘恢复时，通用 `KernelCache` 会重新构造 `JITKernel.from_database`，再由 `SunmmioKernelAdapter.from_database` 调用 `SunmmioLibraryGenerator.load_lib(kernel_lib_path)` 加载 `kernel_launcher.py`。这意味着缓存恢复路径的最小闭环是 Python wrapper 能够在其所在目录找到配套的 `launcher_lib.so` 和设备 ELF。后续实现需要保证这些相对路径在保存和恢复后仍然成立。

## 7. 当前提交中的临时实现

这个提交已经建立了 Sunmmio backend 的大部分 Python 侧接口，但仍有几个明确的临时点不能作为最终行为。

首先，`JITKernel._compile_and_create_adapter` 中把 target 临时改成了 `cuda` 再调用 `tilelang.lower`，随后又恢复为原始 target。这说明当前 lowering pipeline 可能还依赖 CUDA target 才能跑通部分通用 pass 或 codegen 分支。最终方案应移除这段绕行，让 Sunmmio target 从 PassContext 到 device codegen 全程保持一致。

其次，`tilelang.engine.lower.lower` 中将 `codegen_mod` 替换成了固定字符串 `cuda_c_src`，并注释掉了真实 `device_codegen` 调用。这会导致 `device_kernel_source` 不是实际 Sunmmio MLIR，也让后续 `compile_mlir()` 无法获得有效输入。最终实现应恢复 `device_codegen_without_compile(device_mod, target)`，并让 Sunmmio target 返回 `DeviceSourceModule` 中的真实 source。

再次，`SunmmioLibraryGenerator.compile_mlir()` 仍为空实现。完整方案中它需要调用 Sunmmio MLIR compiler，把 `device_kernel_source` 编译为 `kernel.elf`，同时把 ELF 路径记录到 `kernel_elf_path`，供 launcher 生成和 cache 保存使用。

最后，`TLSunmmioSourceWrapper` 生成的 C++ launcher 还没有完成 runtime launch。当前模板只生成 `launch_kernel` 函数、buffer 参数和占位打印，尚未把 Python 传入的 opaque pointer 转成 SuTensor descriptor，也没有加载 ELF、构造 kernel 参数或调用 SuBase/SuDeck runtime。后续需要以 SuDeck/SuBase 的 runtime contract 为准补齐这部分，而不能只依赖 downstream 的 tensor `data_ptr()` 推断。

## 8. 落地步骤

第一步是收敛 target 和 lowering。需要删除 JIT 中临时改写 CUDA target 的逻辑，让 `target_is_sunmmio` 控制 Sunmmio pass 和 codegen 分支，并恢复 `lower()` 中真实的 `device_codegen_without_compile` 调用。完成后，`artifact.kernel_source` 应稳定输出 Sunmmio MLIR 或设备源。

第二步是实现 `compile_mlir()`。该函数应把 `device_kernel_source` 写入工作目录，调用 Sunmmio compiler 生成 ELF，并把 stdout/stderr、命令行和失败诊断在 verbose 模式下完整暴露。编译命令需要支持 `compile_flags`，并把 target 中的 mesh、mcpu、mattr 信息传给外部工具链。

第三步是补齐 C++ launcher。launcher 应以 SuTensor/SuBase/SuDeck 的公开 API 为 contract，从 Python 传入的 tensor handle 获取设备地址、shape、stride、dtype 和 layout 信息，加载或引用 `kernel.elf`，按 host module 中 `T.call_packed` 的顺序组织 kernel 参数，并在指定 stream/device 上提交执行。多 kernel 的调用顺序应由 `TLSunmmioSourceWrapper.parse_source_information()` 从 host module 恢复，而不是依赖字典遍历顺序。

第四步是完善 cache 恢复路径。保存时应把 `kernel_launcher.py`、`launcher_lib.so` 和 `kernel.elf` 放在同一 cache 目录，并让 Python wrapper 通过相对路径加载配套产物。恢复时不应重新编译 launcher，除非 cache key 或运行时库 stamp 变化。

第五步是补测试。单元测试需要覆盖 backend 解析、Sunmmio target cache key、动态 shape 输出分配、wrapper 代码生成和 from-database 恢复。集成测试需要使用最小 Sunmmio kernel 验证从 `tilelang.jit.compile(..., execution_backend="sunmmio")` 到实际 runtime launch 的闭环，并对无 Sunmmio 环境给出清晰 skip 或错误信息。

## 9. 验收标准

方案完成后，一个最小 Sunmmio kernel 应能通过标准 TileLang JIT 入口完成编译和调用。用户显式指定 `execution_backend="sunmmio"` 时应进入 Sunmmio adapter；用户指定 Sunmmio target 且 backend 为 `auto` 时，也应自动选择 `sunmmio`。编译产物应包含真实 Sunmmio device source、ELF、Python wrapper 和 launcher library，并能被 cache 复用。

运行时层面，adapter 应能正确处理输入 tensor、自动分配输出 tensor、解析动态 shape，并把参数按 PrimFunc 顺序传给 launcher。launcher 应使用 Sunmmio runtime contract 执行 ELF，而不是保留占位打印或 CUDA-only stream 判断。错误路径也应清晰：缺少 Sunmmio 工具链、MLIR 编译失败、launcher 编译失败、runtime launch 失败和 cache 产物缺失都应抛出带上下文的信息。

从工程角度看，最终代码不应包含 `TEMP` target 改写、固定 `cuda_c_src`、空的 `compile_mlir()` 或硬编码 ELF 路径。Sunmmio backend 应像其他 TileLang backend 一样，通过统一 JIT、cache 和 profiler 接口被使用，同时把设备特有逻辑隔离在 `tilelang/jit/adapter/sunmmio` 和 `src/target/*sunmmio*` 模块内。
