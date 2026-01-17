import operator
from types import ModuleType

from taichi._lib import core as _ti_core
from taichi.lang import impl, kernel_arguments, ops
from taichi.lang.any_array import AnyArray
from taichi.lang.expr import Expr
from taichi.types import ndarray_type, primitive_types

from .diagnostics import FrontendError


class _CudaModule:
    pass


class _CudaIntrinsic:
    def __init__(self, name):
        self.name = name


class _CudaDim3:
    def __init__(self, kind):
        self.kind = kind


class _CudaAtomic:
    pass


class _CudaShared:
    pass


class _GlobalFn:
    def __init__(self, fn):
        self.fn = fn


class _GlobalModule:
    def __init__(self, module):
        self.module = module


class _GridIndex:
    def __init__(self, dim, axis=0):
        self.dim = dim
        self.axis = axis


class _GridSize:
    def __init__(self, dim, axis=0):
        self.dim = dim
        self.axis = axis


class _GridExpr:
    def __init__(self, reason):
        self.reason = reason


class _Range:
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step


class _RangeIter:
    def __init__(self, rng):
        self.rng = rng


class _RangeIterNext:
    def __init__(self, rng):
        self.rng = rng


class NumbaIRTranslator:
    def __init__(self, func_ir, typemap, calltypes, kernel, options=None):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.kernel = kernel
        self.options = options or {}
        self.ast_builder = kernel.ast_builder()
        self._serialize = bool(self.options.get("serialize", False))
        self.env = {}
        self._grid_vars = {}
        self._grid_size_vars = set()
        self._loop_stack = []
        self._loop_var_stack = []
        self._force_grid_for = False
        self._grid_axes = []
        self._grid_bounds = []
        self.trace_lines = []
        self.builder_calls = []
        self._stmt_meta = self._build_stmt_meta(func_ir)
        self._visited = set()

    def translate(self):
        with impl.get_runtime().src_info_guard("<numba>"):
            block_dim = self.options.get("block_dim")
            if block_dim is not None:
                self._record_builder_call("block_dim", detail=str(int(block_dim)))
                self.ast_builder.block_dim(int(block_dim))
            self._declare_args()
            entry = 0
            if entry not in self.func_ir.blocks:
                raise FrontendError("unsupported entry block layout")
            if self._prepare_default_grid_for():
                self._emit_default_grid_for(entry)
            else:
                self._translate_block(entry)

    def dump_trace(self, path):
        if not path:
            return
        text = "\n".join(self.trace_lines)
        if not text.endswith("\n"):
            text += "\n"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def dump_builder_calls(self, path):
        if not path:
            return
        text = "\n".join(self.builder_calls)
        if not text.endswith("\n"):
            text += "\n"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _build_stmt_meta(self, func_ir):
        stmt_meta = {}
        stmt_id = 0
        for label, block in sorted(func_ir.blocks.items()):
            for index, stmt in enumerate(block.body):
                stmt_id += 1
                stmt_meta[id(stmt)] = {
                    "id": stmt_id,
                    "block": label,
                    "index": index,
                    "stmt": str(stmt),
                }
        return stmt_meta

    def _declare_args(self):
        impl.get_runtime().compiling_callable.finalize_rets()
        for arg in self.kernel.arguments:
            anno = arg.annotation
            if isinstance(anno, ndarray_type.NdarrayType):
                value = kernel_arguments.decl_ndarray_arg(
                    anno.dtype,
                    anno.ndim,
                    arg.name,
                    bool(anno.needs_grad),
                    anno.boundary,
                )
            else:
                value = kernel_arguments.decl_scalar_arg(anno, arg.name, 0)
            self.env[arg.name] = value
        impl.get_runtime().compiling_callable.finalize_params()

    def _translate_block(self, label):
        if label in self._visited:
            return
        self._visited.add(label)
        block = self.func_ir.blocks[label]
        for stmt in block.body:
            if self._try_translate_ctrl(stmt, label):
                return
            self._translate_stmt(stmt)

    def _try_translate_ctrl(self, stmt, label):
        from numba.core import ir

        if isinstance(stmt, ir.Branch):
            for_match = self._match_for_loop(stmt, label)
            if for_match is not None:
                self._emit_range_for(for_match, stmt.loc)
                self._emit_trace(stmt, "range_for")
                return True
            if not self._force_grid_for:
                grid_match = self._match_grid_branch(stmt)
                if grid_match is not None:
                    self._emit_range_for(grid_match, stmt.loc)
                    self._emit_trace(stmt, "grid_for")
                    return True
            while_match = self._match_while_loop(stmt, label)
            if while_match is not None:
                self._emit_while_loop(while_match, stmt.loc)
                self._emit_trace(stmt, "while")
                return True
            phi_match = self._match_phi_if(stmt)
            if phi_match is not None:
                name, cond, true_val, false_val, join = phi_match
                self.env[name] = ops.select(cond, true_val, false_val)
                self._emit_trace(stmt, "phi_select")
                self._translate_block(join)
                return True
            cond = self._eval(stmt.cond)
            if not isinstance(cond, Expr):
                raise FrontendError("if condition is not an expression")
            dbg_info = self._dbg_info(stmt.loc)
            self._record_builder_call("begin_frontend_if", stmt.loc)
            impl.begin_frontend_if(self.ast_builder, cond.ptr, dbg_info)
            self._record_builder_call("begin_frontend_if_true", stmt.loc)
            self.ast_builder.begin_frontend_if_true()
            self._translate_block(stmt.truebr)
            self._record_builder_call("pop_scope", stmt.loc)
            self.ast_builder.pop_scope()
            self._record_builder_call("begin_frontend_if_false", stmt.loc)
            self.ast_builder.begin_frontend_if_false()
            self._translate_block(stmt.falsebr)
            self._record_builder_call("pop_scope", stmt.loc)
            self.ast_builder.pop_scope()
            self._emit_trace(stmt, "branch")
            return True
        if isinstance(stmt, ir.Jump):
            if self._loop_stack:
                loop = self._loop_stack[-1]
                block = self.func_ir.blocks[label]
                is_last = block.body and block.body[-1] is stmt
                if stmt.target == loop["continue"]:
                    if is_last:
                        self._emit_trace(stmt, "continue_struct")
                        return True
                    self._record_builder_call("insert_continue_stmt", stmt.loc)
                    self.ast_builder.insert_continue_stmt(self._dbg_info(stmt.loc))
                    self._emit_trace(stmt, "continue")
                    return True
                if self._is_trampoline_to_header(stmt.target, loop["continue"]):
                    if is_last:
                        self._emit_trace(stmt, "continue_trampoline_struct")
                        return True
                    self._record_builder_call("insert_continue_stmt", stmt.loc)
                    self.ast_builder.insert_continue_stmt(self._dbg_info(stmt.loc))
                    self._emit_trace(stmt, "continue_trampoline")
                    return True
                if stmt.target == loop["break"]:
                    if loop.get("kind") != "while":
                        self._emit_trace(stmt, "break_struct")
                        return True
                    if loop.get("body") == label and is_last:
                        self._emit_trace(stmt, "break_struct")
                        return True
                    self._record_builder_call("insert_break_stmt", stmt.loc)
                    self.ast_builder.insert_break_stmt(self._dbg_info(stmt.loc))
                    self._emit_trace(stmt, "break")
                    return True
            self._emit_trace(stmt, f"jump {stmt.target}")
            self._translate_block(stmt.target)
            return True
        if isinstance(stmt, ir.Return):
            self._emit_trace(stmt, "return")
            return True
        return False

    def _match_phi_if(self, stmt):
        from numba.core import ir

        if not isinstance(stmt, ir.Branch):
            return None
        true_block = self.func_ir.blocks[stmt.truebr]
        false_block = self.func_ir.blocks[stmt.falsebr]

        def get_phi_assign(block):
            if not block.body or not isinstance(block.body[-1], ir.Jump):
                return None
            jump = block.body[-1]
            assigns = [s for s in block.body[:-1] if isinstance(s, ir.Assign)]
            if not assigns:
                return None
            phi_assign = assigns[-1]
            if not isinstance(phi_assign.target, ir.Var):
                return None
            if not phi_assign.target.name.startswith("$phi"):
                return None
            local_consts = {}
            for s in assigns:
                if isinstance(s.value, ir.Const) and s.target.name.startswith("$"):
                    local_consts[s.target.name] = s.value.value

            value = phi_assign.value
            if isinstance(value, ir.Var) and value.name in local_consts:
                return phi_assign.target.name, local_consts[value.name], jump.target
            if isinstance(value, ir.Const):
                return phi_assign.target.name, value.value, jump.target
            try:
                resolved = self._eval(value)
            except FrontendError:
                return None
            return phi_assign.target.name, resolved, jump.target

        true_info = get_phi_assign(true_block)
        false_info = get_phi_assign(false_block)
        if not true_info or not false_info:
            return None
        if true_info[0] != false_info[0]:
            return None
        if true_info[2] != false_info[2]:
            return None
        cond = self._eval(stmt.cond)
        if not isinstance(cond, Expr):
            return None
        true_val = true_info[1]
        false_val = false_info[1]
        if not isinstance(true_val, Expr):
            true_val = Expr(true_val)
        if not isinstance(false_val, Expr):
            false_val = Expr(false_val)
        self._record_builder_call("op.select", stmt.loc)
        return true_info[0], cond, true_val, false_val, true_info[2]

    def _is_trampoline_to_header(self, target_label, header_label):
        from numba.core import ir

        block = self.func_ir.blocks.get(target_label)
        if block is None or not block.body:
            return False
        if len(block.body) != 1:
            return False
        stmt = block.body[0]
        return isinstance(stmt, ir.Jump) and stmt.target == header_label

    def _translate_stmt(self, stmt):
        from numba.core import ir

        if isinstance(stmt, ir.Assign):
            self._assign(stmt)
            return
        if isinstance(stmt, ir.SetItem):
            self._set_item(stmt)
            return
        raise FrontendError(f"unsupported statement: {stmt}")

    def _assign(self, stmt):
        from numba.core import ir

        target = stmt.target
        value = stmt.value
        if isinstance(value, ir.Const):
            if value.value is None:
                self.env[target.name] = None
                self._emit_trace(stmt, "const_none")
                return
            if target.name.startswith("$") and not target.name.startswith("$phi"):
                self.env[target.name] = value.value
                self._emit_trace(stmt, "const")
                return
            rhs_const = Expr(value.value)
            lhs = self.env.get(target.name)
            if isinstance(lhs, Expr):
                dbg_info = self._dbg_info(stmt.loc)
                self._record_builder_call("expr_assign", stmt.loc)
                self.ast_builder.expr_assign(lhs.ptr, rhs_const.ptr, dbg_info)
                self._emit_trace(stmt, "assign")
            else:
                self._record_builder_call("expr_init", stmt.loc)
                lhs = impl.expr_init(rhs_const)
                self.env[target.name] = lhs
                self._emit_trace(stmt, "const_init")
            return
        if isinstance(value, ir.Arg):
            self.env[target.name] = self.env[value.name]
            self._emit_trace(stmt, f"alias arg {value.name}")
            return
        if self._loop_var_stack and target.name in self._loop_var_stack[-1]:
            rhs = self._eval(value)
            if isinstance(rhs, Expr):
                self.env[target.name] = rhs
            elif isinstance(rhs, ir.Var) and rhs.name in self.env:
                self.env[target.name] = self.env[rhs.name]
            else:
                self.env[target.name] = rhs
            self._emit_trace(stmt, "loop_var_alias")
            return
        if isinstance(value, ir.Var) and value.name in self.env:
            existing = self.env[value.name]
            if isinstance(existing, ir.Expr):
                self.env[target.name] = existing
                self._emit_trace(stmt, "alias")
                return
        rhs = self._eval(value)
        if rhs is None:
            self._emit_trace(stmt, "none")
            return
        if self._force_grid_for and target.name in self._grid_axes:
            if isinstance(rhs, (tuple, _GridIndex, _GridExpr)):
                self._emit_trace(stmt, "grid_skip")
                return
        if isinstance(rhs, tuple):
            if not target.name.startswith("$"):
                raise FrontendError("cuda.grid tuple must be unpacked")
            self.env[target.name] = rhs
            self._emit_trace(stmt, "tuple")
            return
        if isinstance(rhs, _GridIndex):
            self.env[target.name] = rhs
            self._grid_vars[target.name] = rhs
            self._emit_trace(stmt, "grid")
            return
        if isinstance(rhs, _GridExpr):
            self.env[target.name] = rhs
            self._grid_vars[target.name] = rhs
            self._emit_trace(stmt, "grid_expr")
            return
        if isinstance(rhs, _GridSize):
            self.env[target.name] = rhs
            self._grid_size_vars.add(target.name)
            self._emit_trace(stmt, "grid_size")
            return
        if isinstance(rhs, (_Range, _RangeIter, _RangeIterNext)):
            self.env[target.name] = rhs
            self._emit_trace(stmt, "range")
            return
        if isinstance(
            rhs,
            (
                _CudaModule,
                _CudaIntrinsic,
                _CudaDim3,
                _CudaAtomic,
                _CudaShared,
                _GlobalFn,
                _GlobalModule,
                AnyArray,
                impl.SharedArray,
            ),
        ):
            self.env[target.name] = rhs
            self._emit_trace(stmt, "bind global")
            return
        if target.name.startswith("$") and not target.name.startswith("$phi"):
            self.env[target.name] = rhs
            self._emit_trace(stmt, "temp")
            return
        if not isinstance(rhs, Expr):
            rhs = Expr(rhs)
        lhs = self.env.get(target.name)
        if lhs is None or not isinstance(lhs, Expr):
            self._record_builder_call("expr_init", stmt.loc)
            lhs = impl.expr_init(rhs)
            self.env[target.name] = lhs
            self._emit_trace(stmt, "init")
            return
        dbg_info = self._dbg_info(stmt.loc)
        self._record_builder_call("expr_assign", stmt.loc)
        self.ast_builder.expr_assign(lhs.ptr, rhs.ptr, dbg_info)
        self._emit_trace(stmt, "assign")

    def _set_item(self, stmt):
        target = self._eval(stmt.target)
        index = self._eval(stmt.index)
        value = self._eval(stmt.value)
        if not isinstance(value, Expr):
            value = Expr(value)
        self._record_builder_call("subscript", stmt.loc, detail="setitem")
        lvalue = impl.subscript(self.ast_builder, target, index)
        dbg_info = self._dbg_info(stmt.loc)
        self._record_builder_call("expr_assign", stmt.loc)
        self.ast_builder.expr_assign(lvalue.ptr, value.ptr, dbg_info)
        self._emit_trace(stmt, "setitem")

    def _eval(self, value):
        from numba.core import ir

        if isinstance(value, ir.Var):
            if value.name not in self.env:
                raise FrontendError(f"undefined variable: {value.name}")
            return self.env[value.name]
        if isinstance(value, ir.Const):
            if value.value is None:
                return None
            return value.value
        if isinstance(value, ir.FreeVar):
            return self._eval_global(value.value)
        if isinstance(value, ir.Arg):
            if value.name not in self.env:
                raise FrontendError(f"undefined argument: {value.name}")
            return self.env[value.name]
        if isinstance(value, ir.Global):
            return self._eval_global(value.value)
        if isinstance(value, ir.Expr):
            return self._eval_expr(value)
        raise FrontendError(f"unsupported value: {value}")

    def _eval_global(self, value):
        if isinstance(value, ModuleType) and value.__name__.startswith("numba.cuda"):
            return _CudaModule()
        try:
            from taichi.lang.field import Field
        except Exception:  # pragma: no cover - defensive
            Field = ()
        if Field and isinstance(value, Field):
            return value
        if isinstance(value, ModuleType):
            if value.__name__ in ("math", "numpy"):
                return _GlobalModule(value)
        if callable(value):
            return _GlobalFn(value)
        raise FrontendError(f"unsupported global: {value}")

    def _eval_expr(self, expr):
        if expr.op == "getattr":
            base = self._eval(expr.value)
            if isinstance(base, _CudaModule):
                if expr.attr == "grid":
                    return _CudaIntrinsic("cuda.grid")
                if expr.attr == "gridsize":
                    return _CudaIntrinsic("cuda.gridsize")
                if expr.attr in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
                    return _CudaDim3(expr.attr)
                if expr.attr == "atomic":
                    return _CudaAtomic()
                if expr.attr == "shared":
                    return _CudaShared()
                if expr.attr in (
                    "syncthreads",
                    "syncthreads_count",
                    "syncthreads_and",
                    "syncthreads_or",
                    "shfl_sync",
                    "shfl_up_sync",
                    "shfl_down_sync",
                    "shfl_xor_sync",
                    "ballot_sync",
                ):
                    return _CudaIntrinsic(f"cuda.{expr.attr}")
            if isinstance(base, _CudaDim3):
                if expr.attr in ("x", "y", "z"):
                    return _GridExpr(f"{base.kind}.{expr.attr}")
            if isinstance(base, _CudaAtomic):
                return _CudaIntrinsic(f"cuda.atomic.{expr.attr}")
            if isinstance(base, _CudaShared):
                return _CudaIntrinsic(f"cuda.shared.{expr.attr}")
            if isinstance(base, AnyArray):
                if expr.attr == "shape":
                    return base.shape
                if expr.attr == "size":
                    shape = base.shape
                    if not shape:
                        return Expr(1)
                    size = shape[0]
                    for dim in shape[1:]:
                        size = size * dim
                    return size
            if isinstance(base, _GlobalModule):
                try:
                    return _GlobalFn(getattr(base.module, expr.attr))
                except AttributeError:
                    raise FrontendError(f"unsupported getattr: {expr.attr}") from None
            raise FrontendError(f"unsupported getattr: {expr.attr}")
        if expr.op == "call":
            fn = self._eval(expr.func)
            args = [self._eval(arg) for arg in expr.args]
            kwargs = {name: self._eval(var) for name, var in expr.kws}
            return self._call(expr, fn, args, kwargs)
        if expr.op == "binop":
            self._record_builder_call("op.binop", expr.loc, detail=getattr(expr.fn, "__name__", str(expr.fn)))
            return self._binop(expr.fn, self._eval(expr.lhs), self._eval(expr.rhs))
        if expr.op == "unary":
            self._record_builder_call("op.unary", expr.loc, detail=getattr(expr.fn, "__name__", str(expr.fn)))
            return self._unary(expr.fn, self._eval(expr.value))
        if expr.op == "build_slice":
            raise FrontendError("slice is not supported")
        if expr.op == "getitem":
            base = self._eval(expr.value)
            index = self._eval(expr.index)
            self._record_builder_call("subscript", expr.loc, detail="getitem")
            return impl.subscript(self.ast_builder, base, index)
        if expr.op == "static_getitem":
            base = self._eval(expr.value)
            return base[expr.index]
        if expr.op == "exhaust_iter":
            value = self._eval(expr.value)
            if isinstance(value, tuple):
                return value
            return value
        if expr.op == "build_tuple":
            return tuple(self._eval(item) for item in expr.items)
        if expr.op == "getiter":
            value = self._eval(expr.value)
            if isinstance(value, _Range):
                return _RangeIter(value)
            return value
        if expr.op == "iternext":
            value = self._eval(expr.value)
            if isinstance(value, _RangeIter):
                return _RangeIterNext(value.rng)
            return value
        if expr.op == "pair_first":
            value = self._eval(expr.value)
            if isinstance(value, tuple):
                return value[0]
            return value
        if expr.op == "pair_second":
            value = self._eval(expr.value)
            if isinstance(value, tuple):
                return value[1]
            return value
        if expr.op == "inplace_binop":
            lhs = self._eval(expr.lhs)
            rhs = self._eval(expr.rhs)
            return self._binop(expr.immutable_fn, lhs, rhs)
        if expr.op == "cast":
            value = self._eval(expr.value)
            target = expr._kws.get("typ")
            if target is None:
                return value
            self._record_builder_call("op.cast", expr.loc, detail=str(target))
            return ops.cast(value, self._numba_type_to_taichi(target))
        raise FrontendError(f"unsupported expr op: {expr.op}")

    def _call(self, expr, fn, args, kwargs):
        if isinstance(fn, _CudaIntrinsic):
            if fn.name == "cuda.grid":
                if len(args) != 1:
                    raise FrontendError("cuda.grid expects one argument")
                dim = args[0]
                if isinstance(dim, Expr):
                    raise FrontendError("cuda.grid requires constant dimension")
                dim_value = int(dim)
                if dim_value not in (1, 2, 3):
                    raise FrontendError("cuda.grid only supports dim=1/2/3")
                if dim_value == 1:
                    return _GridIndex(dim_value, 0)
                return tuple(_GridIndex(dim_value, axis) for axis in range(dim_value))
            if fn.name == "cuda.gridsize":
                if len(args) != 1:
                    raise FrontendError("cuda.gridsize expects one argument")
                dim = args[0]
                if isinstance(dim, Expr):
                    raise FrontendError("cuda.gridsize requires constant dimension")
                dim_value = int(dim)
                if dim_value not in (1, 2, 3):
                    raise FrontendError("cuda.gridsize only supports dim=1/2/3")
                if dim_value == 1:
                    return _GridSize(dim_value, 0)
                return tuple(_GridSize(dim_value, axis) for axis in range(dim_value))
            if fn.name.startswith("cuda.atomic."):
                op = fn.name.split(".")[-1]
                if len(args) != 3:
                    raise FrontendError("cuda.atomic expects (array, index, value)")
                target = args[0]
                index = args[1]
                value = args[2]
                self._record_builder_call("subscript", expr.loc, detail=f"atomic_{op}")
                lvalue = impl.subscript(self.ast_builder, target, index)
                if op == "add":
                    self._record_builder_call("op.atomic_add", expr.loc)
                    return ops.atomic_add(lvalue, value)
                if op == "min":
                    self._record_builder_call("op.atomic_min", expr.loc)
                    return ops.atomic_min(lvalue, value)
                if op == "max":
                    self._record_builder_call("op.atomic_max", expr.loc)
                    return ops.atomic_max(lvalue, value)
                if op == "sub":
                    self._record_builder_call("op.atomic_sub", expr.loc)
                    return ops.atomic_sub(lvalue, value)
                if op == "and_":
                    self._record_builder_call("op.atomic_and", expr.loc)
                    return ops.atomic_and(lvalue, value)
                if op == "or_":
                    self._record_builder_call("op.atomic_or", expr.loc)
                    return ops.atomic_or(lvalue, value)
                if op == "xor":
                    self._record_builder_call("op.atomic_xor", expr.loc)
                    return ops.atomic_xor(lvalue, value)
                raise FrontendError(f"unsupported cuda.atomic op: {op}")
            if fn.name.startswith("cuda.shared."):
                if fn.name != "cuda.shared.array":
                    raise FrontendError(f"unsupported cuda.shared intrinsic: {fn.name}")
                shape = None
                dtype = None
                if args:
                    shape = args[0]
                if "shape" in kwargs:
                    shape = kwargs["shape"]
                if "dtype" in kwargs:
                    dtype = kwargs["dtype"]
                if shape is None or dtype is None:
                    raise FrontendError("cuda.shared.array requires shape and dtype")
                if isinstance(dtype, _GlobalFn):
                    dtype = dtype.fn
                shape_tuple = self._normalize_shared_shape(shape)
                dtype = self._normalize_dtype(dtype)
                from taichi.lang.simt import block

                return block.SharedArray(shape_tuple, dtype)
            if fn.name.startswith("cuda.syncthreads"):
                from taichi.lang.simt import block

                if fn.name == "cuda.syncthreads":
                    if args or kwargs:
                        raise FrontendError("cuda.syncthreads takes no arguments")
                    return block.sync()
                if not args:
                    raise FrontendError(f"{fn.name} requires a predicate")
                predicate = args[0]
                if fn.name == "cuda.syncthreads_count":
                    if isinstance(predicate, Expr):
                        predicate = ops.cast(predicate, primitive_types.i32)
                    return block.sync_count_nonzero(predicate)
                if fn.name == "cuda.syncthreads_and":
                    if isinstance(predicate, Expr):
                        predicate = ops.cast(predicate, primitive_types.i32)
                    return block.sync_all_nonzero(predicate)
                if fn.name == "cuda.syncthreads_or":
                    if isinstance(predicate, Expr):
                        predicate = ops.cast(predicate, primitive_types.i32)
                    return block.sync_any_nonzero(predicate)
                raise FrontendError(f"unsupported syncthreads intrinsic: {fn.name}")
            if fn.name.startswith("cuda.shfl") or fn.name.startswith("cuda.ballot"):
                return self._lower_warp_intrinsic(expr, fn.name, args, kwargs)
        if isinstance(fn, _GlobalFn):
            if fn.fn is bool:
                return args[0]
            if fn.fn in (int, float):
                ret_ty = self.calltypes.get(expr)
                if ret_ty is not None:
                    return ops.cast(args[0], self._numba_type_to_taichi(ret_ty.return_type))
                return args[0]
            if fn.fn is abs:
                return ops.abs(args[0])
            if fn.fn is range:
                if len(args) == 1:
                    return _Range(0, args[0], 1)
                if len(args) == 2:
                    return _Range(args[0], args[1], 1)
                if len(args) == 3:
                    return _Range(args[0], args[1], args[2])
                raise FrontendError("range expects 1-3 arguments")
            if callable(fn.fn):
                import math
                try:
                    import numpy as np
                except Exception:  # pragma: no cover - numpy not available
                    np = None

                math_map = {
                    math.sin: ops.sin,
                    math.cos: ops.cos,
                    math.sqrt: ops.sqrt,
                    math.exp: ops.exp,
                    math.log: ops.log,
                }
                if fn.fn in math_map:
                    return math_map[fn.fn](args[0])
                if np is not None:
                    np_map = {
                        np.exp: ops.exp,
                        np.log: ops.log,
                        np.sqrt: ops.sqrt,
                    }
                    if fn.fn in np_map:
                        return np_map[fn.fn](args[0])
                if kwargs:
                    return fn.fn(*args, **kwargs)
                return fn.fn(*args)
        if isinstance(fn, _Range):
            raise FrontendError("range call should be lowered by Numba IR")
        raise FrontendError(f"unsupported call: {fn}")

    def _normalize_shared_shape(self, shape):
        if isinstance(shape, Expr):
            raise FrontendError("cuda.shared.array shape must be a constant")
        if isinstance(shape, int):
            return (shape,)
        if isinstance(shape, (tuple, list)) and all(isinstance(s, int) for s in shape):
            return tuple(shape)
        raise FrontendError(f"cuda.shared.array shape must be int or tuple, got {shape!r}")

    def _normalize_dtype(self, dtype):
        import numpy as np
        from taichi._lib import core as _ti_core

        if isinstance(dtype, _ti_core.DataType):
            return dtype
        mapping = {
            np.dtype("float32"): primitive_types.f32,
            np.dtype("float64"): primitive_types.f64,
            np.dtype("int32"): primitive_types.i32,
            np.dtype("int64"): primitive_types.i64,
            np.dtype("uint32"): primitive_types.u32,
            np.dtype("uint64"): primitive_types.u64,
        }
        if isinstance(dtype, np.dtype):
            return mapping.get(dtype)
        if isinstance(dtype, type) and dtype in (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64):
            return mapping.get(np.dtype(dtype))
        raise FrontendError(f"unsupported shared array dtype: {dtype!r}")

    def _lower_warp_intrinsic(self, expr, name, args, kwargs):
        from numba.core import types
        from taichi.lang.simt import warp

        def ensure_full_mask(mask):
            if isinstance(mask, Expr):
                return
            if isinstance(mask, (int,)) and mask in (0xFFFFFFFF, -1):
                return
            raise FrontendError("mask must be 0xFFFFFFFF for warp intrinsics")

        def ensure_width(width):
            if width is None:
                return
            if isinstance(width, int) and width == 32:
                return
            raise FrontendError("width must be 32 for warp intrinsics")

        sig = self.calltypes.get(expr)
        arg_types = sig.args if sig is not None else ()

        if name == "cuda.ballot_sync":
            if len(args) < 2:
                raise FrontendError("cuda.ballot_sync expects (mask, predicate)")
            ensure_full_mask(args[0])
            predicate = args[1]
            if isinstance(predicate, Expr):
                predicate = ops.cast(predicate, primitive_types.i32)
            return warp.ballot(predicate)

        if not args and kwargs:
            ordered = ["mask", "value", "src_lane", "width"]
            args = [kwargs[k] for k in ordered if k in kwargs]
        if len(args) < 3:
            raise FrontendError(f"{name} expects (mask, value, offset)")

        mask = args[0]
        value = args[1]
        offset = args[2]
        width = None
        if len(args) >= 4:
            width = args[3]
        if "width" in kwargs:
            width = kwargs["width"]

        ensure_full_mask(mask)
        if isinstance(mask, int):
            mask = mask & 0xFFFFFFFF
            mask = Expr(mask, dtype=primitive_types.u32)
        ensure_width(width)

        value_type = None
        if arg_types and len(arg_types) >= 2:
            value_type = arg_types[1]
        if value_type is None and isinstance(value, Expr):
            dtype = value.element_type()
            if dtype in (primitive_types.i32, primitive_types.u32):
                value_type = types.int32
            elif dtype == primitive_types.f32:
                value_type = types.float32

        def is_i32(t):
            return t == types.int32

        def is_f32(t):
            return t == types.float32

        if name == "cuda.shfl_sync":
            if value_type is None:
                raise FrontendError("cuda.shfl_sync requires typed value")
            if is_i32(value_type):
                return warp.shfl_sync_i32(mask, value, offset)
            if is_f32(value_type):
                return warp.shfl_sync_f32(mask, value, offset)
            raise FrontendError("cuda.shfl_sync requires i32 or f32")
        if name == "cuda.shfl_up_sync":
            if value_type is None:
                raise FrontendError("cuda.shfl_up_sync requires typed value")
            if is_i32(value_type):
                return warp.shfl_up_i32(mask, value, offset)
            if is_f32(value_type):
                return warp.shfl_up_f32(mask, value, offset)
            raise FrontendError("cuda.shfl_up_sync requires i32 or f32")
        if name == "cuda.shfl_down_sync":
            if value_type is None:
                raise FrontendError("cuda.shfl_down_sync requires typed value")
            if is_i32(value_type):
                return warp.shfl_down_i32(mask, value, offset)
            if is_f32(value_type):
                return warp.shfl_down_f32(mask, value, offset)
            raise FrontendError("cuda.shfl_down_sync requires i32 or f32")
        if name == "cuda.shfl_xor_sync":
            if value_type is None:
                raise FrontendError("cuda.shfl_xor_sync requires typed value")
            if is_i32(value_type):
                return warp.shfl_xor_i32(mask, value, offset)
            raise FrontendError("cuda.shfl_xor_sync requires i32")
        raise FrontendError(f"unsupported warp intrinsic: {name}")

    def _binop(self, fn, lhs, rhs):
        op_name = getattr(fn, "__name__", None)
        op = getattr(operator, op_name, None)
        if op is None:
            raise FrontendError(f"unsupported binary op: {fn}")
        if isinstance(lhs, (_GridIndex, _GridSize, _GridExpr)) or isinstance(rhs, (_GridIndex, _GridSize, _GridExpr)):
            return _GridExpr(op_name or "grid")
        return op(lhs, rhs)

    def _unary(self, fn, value):
        op_name = getattr(fn, "__name__", None)
        if op_name == "neg":
            return -value
        if op_name == "pos":
            return value
        if op_name == "not_":
            return ops.logical_not(value)
        if op_name == "invert":
            return ~value
        raise FrontendError(f"unsupported unary op: {fn}")

    def _numba_type_to_taichi(self, nb_type):
        from numba.core import types

        if nb_type == types.int32:
            return primitive_types.i32
        if nb_type == types.int64:
            return primitive_types.i64
        if nb_type == types.uint32:
            return primitive_types.u32
        if nb_type == types.uint64:
            return primitive_types.u64
        if nb_type == types.float32:
            return primitive_types.f32
        if nb_type == types.float64:
            return primitive_types.f64
        if nb_type == types.boolean:
            return primitive_types.u1
        raise FrontendError(f"unsupported numba type: {nb_type}")

    def _dbg_info(self, loc):
        if loc is None:
            return _ti_core.DebugInfo()
        filename = loc.filename or "<numba>"
        if loc.col is None:
            return _ti_core.DebugInfo(f"{filename}:{loc.line}")
        return _ti_core.DebugInfo(f"{filename}:{loc.line}:{loc.col}")

    def _record_builder_call(self, name, loc=None, detail=None):
        loc_str = "unknown"
        if loc is not None:
            if loc.col is None:
                loc_str = f"{loc.filename}:{loc.line}"
            else:
                loc_str = f"{loc.filename}:{loc.line}:{loc.col}"
        if detail:
            self.builder_calls.append(f"builder={name} loc={loc_str} detail={detail}")
        else:
            self.builder_calls.append(f"builder={name} loc={loc_str}")

    def _emit_trace(self, stmt, action):
        meta = self._stmt_meta.get(id(stmt))
        loc = getattr(stmt, "loc", None)
        loc_str = "unknown"
        if loc is not None:
            if loc.col is None:
                loc_str = f"{loc.filename}:{loc.line}"
            else:
                loc_str = f"{loc.filename}:{loc.line}:{loc.col}"
        if meta is None:
            stmt_tag = "synthetic"
            block = "?"
            index = "?"
            stmt_text = "synthetic"
        else:
            stmt_tag = f"{meta['id']:04d}"
            block = str(meta["block"])
            index = str(meta["index"])
            stmt_text = meta["stmt"].replace("\n", " ")
        self.trace_lines.append(
            f"stmt_id={stmt_tag} block={block} index={index} loc={loc_str} action={action} stmt={stmt_text}"
        )

    def _prepare_default_grid_for(self):
        grid_dim, axes = self._scan_grid_axes()
        if grid_dim is None or not axes:
            return False
        if self._has_entry_grid_branch(axes):
            return False
        scalar_args = [arg.name for arg in self.kernel.arguments if not isinstance(arg.annotation, ndarray_type.NdarrayType)]
        if len(scalar_args) < grid_dim:
            return False
        bounds = scalar_args[-grid_dim:]
        self._grid_axes = axes
        self._grid_bounds = bounds
        self._force_grid_for = True
        return True

    def _has_entry_grid_branch(self, axes):
        from numba.core import ir

        entry = 0
        block = self.func_ir.blocks.get(entry)
        if block is None:
            return False
        for stmt in block.body:
            if isinstance(stmt, ir.Branch):
                cond = stmt.cond
                if not isinstance(cond, ir.Var):
                    continue
                expr = self._resolve_def(cond)
                if not isinstance(expr, ir.Expr) or expr.op != "binop":
                    continue
                if getattr(expr.fn, "__name__", None) not in ("lt", "le"):
                    continue
                if isinstance(expr.lhs, ir.Var) and expr.lhs.name in axes:
                    return True
        return False

    def _emit_default_grid_for(self, entry_label):
        axes = self._grid_axes
        bounds = self._grid_bounds

        def nest(axis):
            loop_var = Expr(self.ast_builder.make_id_expr(""))
            prev = self.env.get(axes[axis])
            self.env[axes[axis]] = loop_var
            begin = Expr(0)
            end = self.env[bounds[axis]]
            dbg_info = self._dbg_info(None)
            self.ast_builder.begin_frontend_range_for(loop_var.ptr, begin.ptr, end.ptr, dbg_info)
            if axis + 1 == len(axes):
                self._translate_block(entry_label)
            else:
                nest(axis + 1)
            self.ast_builder.end_frontend_range_for()
            if prev is None:
                self.env.pop(axes[axis], None)
            else:
                self.env[axes[axis]] = prev

        nest(0)

    def _scan_grid_axes(self):
        from numba.core import ir

        grid_call_vars = {}
        grid_iter_vars = {}
        grid_dim = None
        axes = {}
        axis_tmps = {}

        for block in self.func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                    value = stmt.value
                    if value.op == "call":
                        func = self._resolve_def(value.func)
                        is_cuda_grid = False
                        if isinstance(func, ir.Expr) and func.op == "getattr":
                            if func.attr == "grid":
                                is_cuda_grid = True
                        if is_cuda_grid:
                            arg0 = value.args[0] if value.args else None
                            if isinstance(arg0, ir.Const):
                                grid_dim = int(arg0.value)
                            elif isinstance(arg0, ir.Var):
                                resolved = self._resolve_def(arg0)
                                if isinstance(resolved, ir.Const):
                                    grid_dim = int(resolved.value)
                            if not stmt.target.name.startswith("$") and grid_dim == 1:
                                axes[0] = stmt.target.name
                            grid_call_vars[stmt.target.name] = grid_dim
                    if value.op == "exhaust_iter":
                        if isinstance(value.value, ir.Var) and value.value.name in grid_call_vars:
                            grid_iter_vars[stmt.target.name] = grid_call_vars[value.value.name]
                    if value.op == "static_getitem":
                        if isinstance(value.value, ir.Var) and value.value.name in grid_iter_vars:
                            if stmt.target.name.startswith("$"):
                                axis_tmps[int(value.index)] = stmt.target.name
                            else:
                                axes[int(value.index)] = stmt.target.name
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                    if stmt.value.name in axis_tmps.values() and not stmt.target.name.startswith("$"):
                        for axis, name in axis_tmps.items():
                            if name == stmt.value.name:
                                axes[axis] = stmt.target.name
        if grid_dim is None:
            return None, []
        axes_list = [axes.get(i) for i in range(grid_dim)]
        if any(name is None for name in axes_list):
            return None, []
        return grid_dim, axes_list

    def _is_grid_related(self, value):
        return isinstance(value, (_GridIndex, _GridSize, _GridExpr))

    def _match_for_loop(self, stmt, label):
        from numba.core import ir

        if not isinstance(stmt, ir.Branch):
            return None
        cond_var = stmt.cond
        if not isinstance(cond_var, ir.Var):
            return None
        cond_def = self._resolve_def(cond_var)
        if not isinstance(cond_def, ir.Expr) or cond_def.op != "pair_second":
            return None
        iternext_var = cond_def.value
        iternext_def = self._resolve_def(iternext_var)
        if not isinstance(iternext_def, ir.Expr) or iternext_def.op != "iternext":
            return None
        iter_var = iternext_def.value
        range_expr = self._resolve_range_expr(iter_var)
        if range_expr is None:
            return None
        range_args = [self._eval(arg) for arg in range_expr.args]
        if len(range_args) == 1:
            begin, end, step = 0, range_args[0], 1
        elif len(range_args) == 2:
            begin, end, step = range_args[0], range_args[1], 1
        elif len(range_args) == 3:
            begin, end, step = range_args[0], range_args[1], range_args[2]
        else:
            raise FrontendError("range expects 1-3 arguments")

        grid_stride = self._is_grid_related(begin) or self._is_grid_related(step)
        if grid_stride:
            begin = 0
            step = 1
        if not grid_stride:
            if isinstance(step, Expr):
                pass
            elif step not in (1, None):
                raise FrontendError("range step != 1 is not supported yet")

        header_block = self.func_ir.blocks[label]
        loop_vars = self._collect_loop_vars(header_block, iternext_var, stmt.truebr)
        return {
            "kind": "range",
            "header": label,
            "body": stmt.truebr,
            "exit": stmt.falsebr,
            "begin": begin,
            "end": end,
            "step": step,
            "grid_stride": grid_stride,
            "loop_vars": loop_vars,
        }

    def _resolve_range_expr(self, iter_var):
        from numba.core import ir

        current = self._resolve_def(iter_var)
        if isinstance(current, ir.Expr) and current.op == "getiter":
            current = self._resolve_def(current.value)
        if isinstance(current, ir.Expr) and current.op == "call":
            func = self._resolve_def(current.func)
            if isinstance(func, ir.Global) and func.value is range:
                return current
        return None

    def _collect_loop_vars(self, header_block, iternext_var, body_label=None):
        from numba.core import ir

        loop_vars = set()
        for stmt in header_block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if stmt.value.op == "pair_first" and stmt.value.value == iternext_var:
                    loop_vars.add(stmt.target.name)
        blocks = [header_block]
        if body_label is not None:
            body_block = self.func_ir.blocks.get(body_label)
            if body_block is not None:
                blocks.append(body_block)
        changed = True
        while changed:
            changed = False
            for block in blocks:
                for stmt in block.body:
                    if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                        if stmt.value.name in loop_vars and stmt.target.name not in loop_vars:
                            loop_vars.add(stmt.target.name)
                            changed = True
        return loop_vars

    def _match_grid_branch(self, stmt):
        from numba.core import ir

        cond_var = stmt.cond
        if not isinstance(cond_var, ir.Var):
            return None
        expr = self._resolve_def(cond_var)
        if not isinstance(expr, ir.Expr) or expr.op != "binop":
            return None
        if getattr(expr.fn, "__name__", None) != "lt":
            return None
        if not isinstance(expr.lhs, ir.Var):
            return None
        grid_name = expr.lhs.name
        if grid_name not in self._grid_vars:
            return None
        upper = self._eval(expr.rhs)
        return {
            "kind": "grid",
            "header": None,
            "body": stmt.truebr,
            "exit": stmt.falsebr,
            "begin": 0,
            "end": upper,
            "loop_vars": {grid_name},
        }

    def _match_while_loop(self, stmt, label):
        from numba.core import ir

        if not isinstance(stmt, ir.Branch):
            return None
        if not self._has_backedge(label):
            return None
        body = stmt.truebr
        exit_label = stmt.falsebr
        if not self._has_jump_to(body, label, exit_label):
            return None
        cond_expr = self._eval(stmt.cond)
        if not isinstance(cond_expr, Expr):
            raise FrontendError("while condition is not an expression")
        return {"header": label, "body": body, "exit": exit_label, "cond": cond_expr}

    def _has_jump_to(self, start_label, target_label, stop_label):
        from numba.core import ir

        stack = [start_label]
        visited = set()
        while stack:
            label = stack.pop()
            if label in visited:
                continue
            visited.add(label)
            block = self.func_ir.blocks[label]
            if not block.body:
                continue
            last = block.body[-1]
            if isinstance(last, ir.Jump):
                if last.target == target_label:
                    return True
                if last.target != stop_label:
                    stack.append(last.target)
                continue
            if isinstance(last, ir.Branch):
                for br in (last.truebr, last.falsebr):
                    if br == target_label:
                        return True
                    if br != stop_label:
                        stack.append(br)
        return False

    def _has_backedge(self, header_label):
        from numba.core import ir

        for block in self.func_ir.blocks.values():
            if not block.body:
                continue
            last = block.body[-1]
            if isinstance(last, ir.Jump) and last.target == header_label:
                return True
        return False

    def _resolve_def(self, var):
        from numba.core import ir

        current = var
        seen = set()
        while isinstance(current, ir.Var):
            if current.name in seen:
                return None
            seen.add(current.name)
            defs = self.func_ir._definitions.get(current.name, [])
            if not defs:
                return None
            current = defs[-1]
        return current

    def _emit_range_for(self, info, loc):
        begin = info["begin"]
        end = info["end"]
        step = info.get("step", 1)
        grid_stride = info.get("grid_stride", False)
        begin_expr = begin if isinstance(begin, Expr) else Expr(begin)
        end_expr = end if isinstance(end, Expr) else Expr(end)
        begin_expr = ops.cast(begin_expr, primitive_types.i32)
        end_expr = ops.cast(end_expr, primitive_types.i32)
        if isinstance(step, Expr):
            step_expr = step
        else:
            step_expr = Expr(step)
        step_expr = ops.cast(step_expr, primitive_types.i32)
        dbg_info = self._dbg_info(loc)
        prev = {name: self.env.get(name) for name in info["loop_vars"]}
        if not grid_stride and isinstance(step, Expr):
            loop_var = impl.expr_init(begin_expr)
            for name in info["loop_vars"]:
                self.env[name] = loop_var
            prev_loop_vars = self._loop_var_stack[-1] if self._loop_var_stack else set()
            self._loop_var_stack.append(prev_loop_vars | set(info["loop_vars"]))
            self._loop_stack.append(
                {
                    "kind": "while",
                    "continue": info.get("header") or info["body"],
                    "break": info["exit"],
                    "body": info["body"],
                }
            )
            stmt_dbg = self._dbg_info(loc)
            self._record_builder_call("begin_frontend_while", loc)
            self.ast_builder.begin_frontend_while(Expr(1).ptr, stmt_dbg)
            cond = ops.cmp_lt(loop_var, end_expr)
            self._record_builder_call("begin_frontend_if", loc)
            impl.begin_frontend_if(self.ast_builder, cond.ptr, stmt_dbg)
            self._record_builder_call("begin_frontend_if_true", loc)
            self.ast_builder.begin_frontend_if_true()
            self._record_builder_call("pop_scope", loc)
            self.ast_builder.pop_scope()
            self._record_builder_call("begin_frontend_if_false", loc)
            self.ast_builder.begin_frontend_if_false()
            self._record_builder_call("insert_break_stmt", loc)
            self.ast_builder.insert_break_stmt(stmt_dbg)
            self._record_builder_call("pop_scope", loc)
            self.ast_builder.pop_scope()
            self._translate_block(info["body"])
            next_val = ops.add(loop_var, step_expr)
            self._record_builder_call("expr_assign", loc)
            self.ast_builder.expr_assign(loop_var.ptr, next_val.ptr, stmt_dbg)
            self._record_builder_call("pop_scope", loc)
            self.ast_builder.pop_scope()
            self._loop_stack.pop()
            self._loop_var_stack.pop()
            for name, value in prev.items():
                if value is None:
                    self.env.pop(name, None)
                else:
                    self.env[name] = value
            self._translate_block(info["exit"])
            return

        loop_var = Expr(self.ast_builder.make_id_expr(""))
        if self._serialize:
            self._record_builder_call("parallelize", loc, detail="1")
            self.ast_builder.parallelize(1)
            self._record_builder_call("strictly_serialize", loc)
            self.ast_builder.strictly_serialize()
        for name in info["loop_vars"]:
            self.env[name] = loop_var
        self._loop_stack.append(
            {
                "kind": "range",
                "continue": info.get("header") or info["body"],
                "break": info["exit"],
                "body": info["body"],
            }
        )
        prev_loop_vars = self._loop_var_stack[-1] if self._loop_var_stack else set()
        self._loop_var_stack.append(prev_loop_vars | set(info["loop_vars"]))
        self._record_builder_call("begin_frontend_range_for", loc)
        self.ast_builder.begin_frontend_range_for(loop_var.ptr, begin_expr.ptr, end_expr.ptr, dbg_info)
        self._translate_block(info["body"])
        self._record_builder_call("end_frontend_range_for", loc)
        self.ast_builder.end_frontend_range_for()
        self._loop_stack.pop()
        self._loop_var_stack.pop()
        for name, value in prev.items():
            if value is None:
                self.env.pop(name, None)
            else:
                self.env[name] = value
        self._translate_block(info["exit"])

    def _emit_while_loop(self, info, loc):
        stmt_dbg = self._dbg_info(loc)
        self._loop_stack.append(
            {
                "kind": "while",
                "continue": info["header"],
                "break": info["exit"],
                "body": info["body"],
            }
        )
        self._record_builder_call("begin_frontend_while", loc)
        self.ast_builder.begin_frontend_while(Expr(1).ptr, stmt_dbg)
        self._record_builder_call("begin_frontend_if", loc)
        impl.begin_frontend_if(self.ast_builder, info["cond"].ptr, stmt_dbg)
        self._record_builder_call("begin_frontend_if_true", loc)
        self.ast_builder.begin_frontend_if_true()
        self._record_builder_call("pop_scope", loc)
        self.ast_builder.pop_scope()
        self._record_builder_call("begin_frontend_if_false", loc)
        self.ast_builder.begin_frontend_if_false()
        self._record_builder_call("insert_break_stmt", loc)
        self.ast_builder.insert_break_stmt(stmt_dbg)
        self._record_builder_call("pop_scope", loc)
        self.ast_builder.pop_scope()
        self._translate_block(info["body"])
        self._record_builder_call("pop_scope", loc)
        self.ast_builder.pop_scope()
        self._loop_stack.pop()
        self._translate_block(info["exit"])
