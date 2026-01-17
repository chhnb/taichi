# DSL 到 Taichi IR 转换方案（Numba 优先）

## 目标
- 统一将多个 Python 数值 DSL 转换为 Taichi IR。
- 先实现 Numba（CPU/CUDA），再扩展 Warp 与 Devito。
- 产出的 Taichi IR 尽量后端无关；设备特性操作单独标记。

## 非目标（阶段 1）
- 完整的 Python 语义或 object mode。
- 无法静态推断的动态 shape 或 Python 侧控制流。
- 覆盖全部 Numba CUDA 高级特性或全部 intrinsic。

## 总体架构

### 统一中间层（Taichi 前端 IR）
- 本方案以 Taichi 前端 IR 作为统一中间层（也是 Taichi IR 体系的一层）。
- 好处：与 Taichi 原生前端对齐、后端无关、可复用现有编译流水线。
- 后续如需更贴近后端的优化，再考虑下沉到更低层 Taichi IR。

### 组件
1. **DSL 前端适配器（每个 DSL 一个）**
   - 提取 typed IR（或等价的有类型图）。
   - 规范化为小而明确的 kernel IR。

2. **DSL→Taichi 翻译核心（共享）**
   - 以抽象基类定义翻译契约。
   - 通过最小 IR builder 接口构建 Taichi IR。

3. **IR Builder Bridge（Python <-> C++）**
   - 从 Python 侧构建 Taichi IR 的薄接口。
   - 让后端与 DSL 前端语言解耦。

4. **能力映射（Capability Mapping）**
   - 区分可移植操作与设备专有操作（如 shared memory、warp shuffle）。
   - 设备专有操作降级为 Taichi intrinsic 或 extern。

### 抽象基类（Python 侧）
```python
class DSLToTaichiTranslator:
    def __init__(self, context):
        self.context = context

    def parse(self, dsl_ir):
        """解析/规范化 DSL IR -> 最小 kernel IR。"""
        raise NotImplementedError

    def lower_types(self):
        """DSL 类型 -> Taichi 类型映射。"""
        raise NotImplementedError

    def lower_intrinsics(self):
        """DSL intrinsic -> Taichi 操作或 extern 映射。"""
        raise NotImplementedError

    def build_taichi_ir(self):
        """通过 IR builder 构建 Taichi IR。"""
        raise NotImplementedError

    def finalize(self):
        """验证、诊断与收尾。"""
        raise NotImplementedError
```

### IR Builder Bridge 方案
- **方案 A：Python 侧 builder（迭代快）**
  - 通过 pybind 或 C-API 暴露最小 IR builder。
  - 优点：快、易调试。
  - 缺点：ABI 稳定性与性能约束。

- **方案 B：C++ 侧翻译（稳定）**
  - 降低跨语言调用，将 lowering 放 C++。
  - 优点：ABI 稳定，性能可控。
  - 缺点：开发复杂度更高。

- **方案 C：序列化 IR（JSON/Protobuf）**
  - 前端输出序列化 IR，C++ 重建 Taichi IR。
  - 优点：解耦、语言无关。
  - 缺点：需设计格式与校验。

**建议：** 先用方案 A 快速迭代，IR builder 设计保持小而稳定，后续可无痛迁移到方案 B。

## Numba 前端（阶段 1）

### 选择 Numba 的原因
- 输入主要是 NumPy；Taichi 已有 NumPy IO 支持。
- Numba 有 typed IR，类型信息明确。
- Numba 仅 CPU/CUDA 后端；Taichi 可复用多后端能力。

### 数据来源
- 直接用 Numba typed IR（及类型注解），不走 LLVM/PTX。
- 覆盖 kernel 子集：循环、算术、数组索引、归约、简单分支。

### CPU vs CUDA 解析路径
- **共享部分：** typed IR 提取与规范化。
- **分歧部分：** CUDA intrinsic 与 memory space 的处理。
  - CPU：普通循环、矢量化、归约。
  - CUDA：线程/块索引、shared memory、同步、warp intrinsic。

结论：大部分共用，CUDA 在 intrinsic handler 上分支。

### 初始映射规则（MVP）
- 标量：int/float/bool -> Taichi 标量类型。
- 数组：NumPy 数组 -> Taichi ndarray / external array 参数。
- 循环：`cuda.grid(1)` + `if i < n` -> Taichi `range` for（IR builder 直接构造）。
- 数学操作：基础二元运算（+/-/*//与比较）；不支持时给出诊断。

### 设备专有处理
- shared memory：若 Taichi 有对应语义则直接映射；否则标记为设备专有并要求 CUDA 后端。
- warp ops（如 `shfl`）：映射为 Taichi intrinsic 或 extern。
- 无等价：给出明确错误并 fail fast。

## 扩展计划（Warp / Devito）
- Warp：其 kernel AST 可规范化为相同的 kernel IR。
- Devito：聚焦 stencil 与结构化网格。
- 复用同一基类与能力映射。

## 里程碑
1. **脚手架**
   - 新增 `dsl_to_taichi` 模块：基类 + registry。
   - 提供 Python 侧最小 IR builder 接口。

2. **Numba CPU MVP**
   - typed IR -> 规范化 -> 基本 lowering。
   - 支持 elementwise + 简单循环。

3. **Numba CUDA MVP**
   - CUDA intrinsic 映射 + memory space 处理。
   - 支持基础 kernel 与线程/块索引。

4. **验证与测试**
   - 与 Numba 对齐的输出测试（小型 kernel 集合）。
   - 归约、边界条件测试。

5. **扩展到 Warp / Devito**
   - 复用基类与 IR builder。

## 近期实现计划
- 已完成：建立 `dsl_to_taichi` 脚手架（基类、registry、诊断接口）。
- 已完成：新增 `taichi.lang.numba_typed_frontend` 入口与 typed IR→Taichi 前端 IR 翻译器（不走 AST 重写）。
- 已完成：扩展 typed IR lowering（grid-stride、二维/三维 grid、更多算子、原子操作）。
- 进行中：修复纯 `range` 嵌套（`test_heat_step_cpu`）导致的前端 IR 崩溃问题。
- 接下来：继续补齐 CUDA 特性与诊断测试（shared/shfl/sync 等）。

## 调试输出（IR Dump）
- `TI_NUMBA_TF_FORCE_LOCAL=1`：强制使用仓库内 `./numba` 源码。
- `TI_NUMBA_TF_USE_LOCAL=1`：优先使用仓库内 `./numba`，失败则回退系统 numba；默认仅使用系统 numba。
- `TI_NUMBA_TF_DUMP_TYPED_IR=/path/to/typed_ir.txt`：输出 Numba typed IR（文本）。
- `TI_NUMBA_TF_DUMP_TAICHI_IR=/path/to/taichi_ir.txt`：输出翻译过程日志（stmt_id、block/index、loc、action、stmt），用于对照 typed IR。
- `TI_NUMBA_TF_DUMP_MAP=/path/to/ir_map.txt`：输出语句映射（stmt_id → loc → stmt）。
- `TI_NUMBA_TF_DUMP_TIMING=/path/to/timing.txt`：输出各阶段耗时（typed IR 获取、翻译、dump 等）。
- `TI_NUMBA_TF_DUMP_BUILDER=/path/to/builder.txt`：输出 lowering 过程（AST builder 调用 + ops/subscript/atomic 等细粒度动作）。
- `ti.njit(target="cuda"|"cpu")`：指定 typed IR 解析的后端语义上下文；未指定时自动根据是否使用 `numba.cuda` 进行判断。运行后端仍由 `ti.init(arch=...)` 决定。
- `ti.njit(serialize=True)`：强制 range for 串行执行（等价于 `ti.loop_config(serialize=True)`），用于需要顺序语义的算法（如存在共享写入/数据依赖的内层循环）。

## 未决问题
- IR builder 应该放在哪（现有 C-API 还是新 pybind）？
- 需要暴露哪些最小 Taichi IR 节点？
- 设备专有操作应该如何标记（能力标签 vs 明确 intrinsic）？

## 实现记录
- 2025-09-07：创建中文方案文档（本文件），尚未开始具体实现。
- 2025-09-07：新增 `python/taichi/dsl_to_taichi` 包，包含基类、registry 与 diagnostics 骨架。
- 2025-09-07：新增 `python/taichi/lang/numba_typed_frontend` 占位模块与 `ti.njit` 入口。
- 2025-09-07：保存 Numba typed frontend 的测试用例，拆分为 `tests/python/test_numba_typed_frontend_cuda.py` 与 `tests/python/test_numba_typed_frontend_cpu.py`。
- 2025-09-07：新增 typed IR dump 与语句映射输出（通过 `TI_NUMBA_TF_DUMP_*` 环境变量指定文件路径）。
- 2025-09-07：完成最小 typed IR→Taichi 前端 IR 翻译器（`cuda.grid(1)` + 简单分支 + 数组读写 + 基础二元运算），移除 AST 重写路径。
- 2026-01-17：补齐 typed IR→Taichi 前端 IR 的核心语义覆盖（`range/while/if/phi`、grid‑stride、二维/三维 grid、原子操作、bitwise、math/cast、`ndarray.shape/size`），并增加 numba typing 适配与 numpy ufunc 预处理。
- 2026-01-17：CPU 侧已通过多项子集测试（`vec_add`、grid‑stride 变体、mask/atomic/reduce、math/cast/bitwise、stencil/matmul/laplacian 等）；发现 `test_heat_step_cpu` 会触发前端 IR 崩溃（loop var 误当作可写 lvalue）。
- 2026-01-17：修复 `range` loop var 绑定与 alias 处理，避免对 loop index 生成无效 store；`test_heat_step_cpu` 通过。
- 2026-01-17：新增 untyped IR 预检，遇到 slice 访问直接报错（`test_unsupported_slice_is_diagnosed` 通过）。
- 2026-01-17：`tests/python/test_numba_typed_frontend_cpu.py` 全量通过（75 passed, 1 skipped），跳过项是 Numba CUDA 在本机 PTX 版本不匹配导致的编译失败。
- 2026-01-17：`tests/python/test_numba_typed_frontend_cuda.py` 增加 Numba CUDA 编译探针，PTX/驱动不兼容时自动跳过（避免环境问题阻断主流程）。
- 2026-01-17：补齐 CUDA 设备特性映射（shared array、syncthreads、warp shfl/ballot），支持关键字参数调用与 mask/width 校验；支持 `block_dim` 装饰参数；修正 warp mask 类型为 `u32`。
- 2026-01-17：增加 demo 脚本 `numba_typed_ir_demo.py`（同内容在 `docs/design/numba_typed_ir_demo.py`），用于生成 typed IR 与翻译 trace 文件。
- 2026-01-17：增加耗时统计输出（`TI_NUMBA_TF_DUMP_TIMING`），用于定位 typed IR 获取与翻译的耗时分布。
- 2026-01-17：新增 `numba_typed_ir_profile.py`，输出整体耗时分段（import/ti.init/首次调用等）。
- 2026-01-17：新增 `taichi_kernel_profile.py`，用于对比纯 Taichi kernel 的启动与首调开销。
- 2026-01-17：新增 `scripts/run_numba_typed_frontend_matrix.py`，按可用后端运行完整测试矩阵（CPU/Vulkan/CUDA，CUDA 专属用例仅在可用时运行）。
- 2026-01-17：新增 `numba_typed_ir_pipeline_demo.py`，展示 typed IR→translator→Taichi 前端 IR 的完整链路。
- 2026-01-17：新增 `demo/` 目录（`vec_add`/`grid_stride`/`stencil2d`/`atomic_reduce`），并提供 `demo/run_all.py` 一键生成解析输出与 Taichi IR。
- 2026-01-17：新增 `demo/phi_demo.py`，展示 if/else 引入的 phi（lowering 为 `select`）。
- 2026-01-17：新增 `demo/sieve_demo.py`（案例名 `prime_check`），用素数判定做较复杂控制流与 numba 对比验证，避免数据竞争。
- 2026-01-17：移除 `TI_ENABLE_NUMBA_TF` 作为总开关，默认只要 `numba` 可导入就启用。
- 2026-01-17：增加 typed IR 缓存（按函数+参数类型+target hint），避免重复推导；trace 日志标注 `stmt_id=` 或 `synthetic` 以便对齐。
