# Numba Typed IR -> Taichi Front IR 转换报告

本报告面向导师介绍当前 Numba typed IR 到 Taichi 前端 IR 的转换实现、调试产物与覆盖范围。

## 1. 目标与范围

- 目标：将 Numba typed IR（SSA 形式）翻译为 Taichi 前端 IR（通过 ast_builder API 生成），实现 **IR -> IR** 的统一转换路径。
- 当前覆盖：基础算术/比较、数组读写、for/while、break/continue、条件表达式（phi/select）、原子操作、部分 CUDA intrinsic（grid、gridsize、atomic、shared、shfl、syncthreads、ballot 等）。
- 运行后端：typed IR 推导与最终执行后端解耦，typed IR target 由代码中是否使用 `numba.cuda` intrinsic 自动推断；最终执行后端由 `ti.init(arch=...)` 决定。

## 2. 总体流程（从 typed IR 到 Taichi 前端 IR）

1) **Numba typed IR 推导**
   - 调用 Numba typing pipeline，生成 `func_ir` + `typemap` + `calltypes`。
   - 输出：`TI_NUMBA_TF_DUMP_TYPED_IR`（完整 typed IR 文本）。

2) **Translator 解析 SSA**
   - 入口：`python/taichi/lang/numba_typed_frontend/translator.py`
   - 核心：遍历 `func_ir.blocks`，匹配控制流与语义模式。
   - 使用 `env` 维护 SSA 变量到 `Expr/常量/内置对象` 的映射。

3) **调用 ast_builder 生成 Taichi 前端 IR**
   - 通过 `self.ast_builder` 发出 `begin_frontend_range_for` / `begin_frontend_if` / `expr_assign` / `insert_break_stmt` 等调用。
   - 可选记录：`TI_NUMBA_TF_DUMP_BUILDER`（builder API 调用日志）。

4) **Taichi 编译管线继续 lower 到后端**
   - 若打开 `print_ir`，可直接看到前端 IR 与后续 lowering。

## 3. Translator 如何处理 SSA

### 3.1 env 映射与表达式构造

- `env[name]` 保存 SSA 变量的语义值：
  - 常量（Python 值）
  - `Expr`（Taichi 表达式）
  - 特殊对象（如 `_CudaModule`、`_CudaIntrinsic`）
- `_eval` 将 Numba IR 表达式/变量解释为 `Expr/常量/特殊对象`。
- 对常量赋值：
  - **首次**赋值用 `expr_init`
  - **后续**赋值用 `expr_assign`
  - 避免分支内重新初始化导致的未定义引用。

### 3.2 控制流识别

- **range for**：识别 `range` + `getiter/iternext/pair_first` 结构。
- **grid for**：识别 `cuda.grid` + grid-stride 模式。
- **while**：检测回边 (`backedge`) 与 `branch` 模式。
- **phi if**：识别 `branch` 的双分支 phi 合流，转成 `ops.select(cond, a, b)`。

## 4. 示例 1：原子加法 (atomic_reduce)

代码（`demo/atomic_reduce_demo.py`）：
```python
@ti.njit
def reduce_sum(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.add(out, 0, a[i])
```

Typed IR 片段（`demo/output/atomic_reduce_typed_ir.txt`）：
```
i = cuda.grid(1)
if i < n:
    cuda.atomic.add(out, 0, a[i])
```

Builder 调用日志（`demo/output/atomic_reduce_builder.txt`）：
```
builder=begin_frontend_range_for ...
builder=subscript ... detail=getitem
builder=subscript ... detail=atomic_add
builder=op.atomic_add ...
builder=end_frontend_range_for ...
```

说明：
- `cuda.grid` 被识别为 grid-for 模式，转换成 `range_for`。
- 原子加法翻译成 `op.atomic_add`，与 Taichi 前端 atomic op 对应。

## 5. 示例 2：素数判定 (prime_check)

代码（`demo/sieve_demo.py`，案例名 `prime_check`）：
```python
@ti.njit
def ti_prime_check(out, n):
    for i in range(n):
        if i < 2:
            out[i] = 0
            continue
        is_prime = 1
        limit = int(math.sqrt(i)) + 1
        for p in range(2, limit):
            if i % p == 0:
                is_prime = 0
                break
        out[i] = is_prime
```

关键控制流：
- `continue` / `break` 被识别为结构化跳转，并转换为 `insert_continue_stmt` / `insert_break_stmt` 或结构化处理。
- `is_prime` 为局部变量，保证分支内重复赋值使用 `expr_assign`，避免未初始化引用。

运行后对比 NumPy/Numba：
```
match: True
prime_count: 1229
```

## 6. 调试与可观测性产物

每个 demo 会在 `demo/output/` 生成：
- `*_typed_ir.txt`：Numba typed IR
- `*_ir_map.txt`：IR 语句到源码位置映射
- `*_trace.txt`：translator 动作追踪（env/控制流/赋值等）
- `*_builder.txt`：ast_builder API 调用序列
- `*_timing.txt`：typed IR 推导与翻译耗时

此外，`print_ir=1` 会在 stdout 打印 Taichi 前端 IR 与 lower 结果。

## 7. 当前覆盖与限制

已覆盖：
- 算术/比较、数组读写、条件分支、for/while、break/continue、phi/select
- 原子操作、部分 CUDA intrinsic

暂未覆盖（后续扩展方向）：
- 复杂切片/视图、动态 shape、高阶函数、部分高级 NumPy 语义
- 更复杂的控制流图（深层嵌套 + 多 phi 变量）

## 8. 如何接入更多 DSL

当前推荐做法：为每个 DSL 提供一个 `DSLToTaichiTranslator` 适配器，复用统一的转换阶段接口：
1) `parse(dsl_ir)`：读取 DSL 自身 IR，产出最小可翻译形式（或直接保留原 IR）。
2) `lower_types()`：统一映射到 Taichi dtype。
3) `lower_intrinsics()`：将 DSL 的 intrinsic 映射为 Taichi op/extern。
4) `build_taichi_ir()`：调用 `ast_builder` 输出 Taichi 前端 IR。
5) `finalize()`：校验与诊断输出。

参考实现：
- `python/taichi/lang/numba_typed_frontend/dsl_adapter.py`（`NumbaDSLTranslator`）
- `python/taichi/dsl_to_taichi/registry.py`（注册 & 创建）

新 DSL 的最小落地流程：
- 定义 `YourDSLTranslator(DSLToTaichiTranslator)` 并注册。
- 将 DSL IR 解析为 `parse()` 的输入结构。
- 在 `build_taichi_ir()` 中输出相应 builder 调用序列。

## 8. 复现方式

单个 demo：
```
PYTHONPATH=$PWD/python python demo/atomic_reduce_demo.py
PYTHONPATH=$PWD/python python demo/sieve_demo.py
```

设置后端与规模：
```
TI_DEMO_ARCH=cuda TI_DEMO_N=100000 PYTHONPATH=$PWD/python python demo/sieve_demo.py
```

## 9. 结论

当前实现已完成 Numba typed IR -> Taichi front IR 的核心链路，并通过多组 demo 与测试验证正确性。  
后续工作重点是扩展复杂控制流与更多内存模式，并将更多 DSL 融入统一前端 IR 层。
