import functools
import inspect
import os
import time
from types import ModuleType

from .config import import_numba, is_available
from .diagnostics import FrontendError
from .translator import NumbaIRTranslator

_ENV_DUMP_TYPED_IR = "TI_NUMBA_TF_DUMP_TYPED_IR"
_ENV_DUMP_TAICHI_IR = "TI_NUMBA_TF_DUMP_TAICHI_IR"
_ENV_DUMP_MAP = "TI_NUMBA_TF_DUMP_MAP"
_ENV_DUMP_TIMING = "TI_NUMBA_TF_DUMP_TIMING"
_ENV_DUMP_BUILDER = "TI_NUMBA_TF_DUMP_BUILDER"


def _load_numba():
    try:
        return import_numba()
    except Exception as exc:
        raise FrontendError(f"numba import failed: {exc}") from exc


def _dump_text(path, text):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _ordered_args(pyfunc, args, kwargs):
    sig = inspect.signature(pyfunc)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    ordered = []
    for name in sig.parameters:
        ordered.append(bound.arguments[name])
    return tuple(ordered)


def _taichi_dtype_to_numpy(dtype):
    import numpy as np

    mapping = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "u32": np.uint32,
        "u64": np.uint64,
        "u1": np.bool_,
    }
    return mapping.get(str(dtype))


def _typeof_arg(arg, numba_mod):
    Purpose = numba_mod.core.typing.typeof.Purpose
    typeof = numba_mod.core.typing.typeof.typeof
    try:
        from taichi.lang._ndarray import ScalarNdarray
    except Exception:  # pragma: no cover - defensive
        ScalarNdarray = ()
    if ScalarNdarray and isinstance(arg, ScalarNdarray):
        np_dtype = _taichi_dtype_to_numpy(arg.dtype)
        if np_dtype is None:
            raise FrontendError(f"unsupported taichi ndarray dtype: {arg.dtype}")
        nb_dtype = numba_mod.np.numpy_support.from_dtype(np_dtype)
        ndim = len(arg.shape) if hasattr(arg, "shape") else 0
        return numba_mod.core.types.Array(nb_dtype, ndim, "C")
    return typeof(arg, Purpose.argument)


_CUDA_ATTRS_INSTALLED = set()


def _ensure_cuda_attr_typing(numba_mod=None, cuda_mod=None):
    global _CUDA_ATTRS_INSTALLED
    if numba_mod is None:
        numba_mod = import_numba()
    if cuda_mod is None:
        import importlib

        cuda_mod = importlib.import_module("numba.cuda")
    intrinsics = cuda_mod.intrinsics
    cuda_target = cuda_mod.descriptor.cuda_target
    types = numba_mod.core.types
    templates = numba_mod.core.typing.templates

    registry = templates.Registry()

    @registry.register_attr
    class CudaModuleExtra(templates.AttributeTemplate):
        key = types.Module(cuda_mod)

        def resolve_grid(self, mod):
            return cuda_target.typing_context.resolve_value_type(intrinsics.grid)

        def resolve_gridsize(self, mod):
            return cuda_target.typing_context.resolve_value_type(intrinsics.gridsize)

    ctx = cuda_target.typing_context
    if id(ctx) in _CUDA_ATTRS_INSTALLED:
        return
    ctx.install_registry(registry)
    _CUDA_ATTRS_INSTALLED.add(id(ctx))


def _get_untyped_ir(pyfunc, numba_mod):
    return numba_mod.core.compiler.run_frontend(pyfunc)


def _strip_bool_globals(func_ir, numba_mod):
    ir_utils = numba_mod.core.ir_utils
    ir = numba_mod.core.ir

    for block in func_ir.blocks.values():
        bool_vars = set()
        new_body = []
        for stmt in block.body:
            if (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Global)
                and stmt.value.value is bool
            ):
                bool_vars.add(stmt.target.name)
                continue
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and stmt.value.op == "call":
                call = stmt.value
                if isinstance(call.func, ir.Var) and call.func.name in bool_vars and len(call.args) == 1:
                    new_body.append(ir.Assign(call.args[0], stmt.target, stmt.loc))
                    continue
            new_body.append(stmt)
        block.body = new_body
    func_ir._definitions = ir_utils.build_definitions(func_ir.blocks)


def _rewrite_numpy_ufuncs(func_ir, numba_mod):
    try:
        import numpy as np
        import math
    except Exception:  # pragma: no cover - numpy/math not available
        return

    ir = numba_mod.core.ir
    ir_utils = numba_mod.core.ir_utils
    mapping = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
    }
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if not isinstance(stmt, ir.Assign):
                continue
            value = stmt.value
            if isinstance(value, ir.Expr) and value.op == "getattr":
                base = value.value
                base_global = None
                if isinstance(base, ir.Global):
                    base_global = base
                elif isinstance(base, ir.Var):
                    defs = func_ir._definitions.get(base.name, [])
                    if defs and isinstance(defs[-1], ir.Global):
                        base_global = defs[-1]
                if base_global is not None and base_global.value is np:
                    repl = mapping.get(value.attr)
                    if repl is not None:
                        stmt.value = ir.Global(repl.__name__, repl, stmt.loc)
    func_ir._definitions = ir_utils.build_definitions(func_ir.blocks)


def _find_cuda_module(func_ir, numba_mod):
    ir = numba_mod.core.ir
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Global):
                value = stmt.value.value
                if getattr(value, "__name__", "") == "numba.cuda":
                    return value
    return None


def _get_typed_ir(pyfunc, args, kwargs, target=None, arg_types=None):
    numba = _load_numba()
    func_ir = _get_untyped_ir(pyfunc, numba)
    if target is None:
        target = _infer_target_from_ir(func_ir)
    cuda_mod_in_ir = _find_cuda_module(func_ir, numba)
    if target == "cuda":
        import importlib

        cuda_mod = cuda_mod_in_ir or importlib.import_module("numba.cuda")
        cuda_target = cuda_mod.descriptor.cuda_target
        typingctx = cuda_target.typing_context
        targetctx = cuda_target.target_context
    else:
        cpu_target = numba.core.registry.cpu_target
        typingctx = cpu_target.typing_context
        targetctx = cpu_target.target_context

    try:
        typingctx.refresh()
        targetctx.refresh()
    except TypeError as exc:
        if "cannot augment" not in str(exc):
            raise
    if target == "cuda":
        _ensure_cuda_attr_typing(numba, cuda_mod_in_ir or cuda_mod)

    typed_passes = numba.core.typed_passes
    type_annotations = numba.core.annotations.type_annotations

    if arg_types is None:
        ordered_args = _ordered_args(pyfunc, args, kwargs)
        arg_types = tuple(_typeof_arg(arg, numba) for arg in ordered_args)

    _strip_bool_globals(func_ir, numba)
    _rewrite_numpy_ufuncs(func_ir, numba)
    try:
        typemap, return_type, calltypes, _ = typed_passes.type_inference_stage(
            typingctx, targetctx, func_ir, arg_types, None
        )
        type_annotation = type_annotations.TypeAnnotation(
            func_ir=func_ir,
            typemap=typemap,
            calltypes=calltypes,
            lifted=(),
            lifted_from=None,
            args=arg_types,
            return_type=return_type,
            html_output=None,
        )
    except numba.core.errors.TypingError as exc:
        msg = str(exc)
        allowed = (
            "Untyped global name" in msg
            or "syncthreads_and" in msg
            or "syncthreads_or" in msg
            or "syncthreads_count" in msg
            or "CallConstraint" in msg
            or "Error at driver init" in msg
        )
        if not allowed:
            raise
        typemap = {}
        calltypes = {}
        type_annotation = None
    return func_ir, typemap, calltypes, type_annotation, target


def _infer_target_from_ir(func_ir):
    from numba.core import ir

    def is_cuda_module(value):
        return isinstance(value, ModuleType) and value.__name__.startswith("numba.cuda")

    for defs in func_ir._definitions.values():
        for definition in defs:
            if isinstance(definition, ir.Global) and is_cuda_module(definition.value):
                return "cuda"
            if isinstance(definition, ir.FreeVar) and is_cuda_module(definition.value):
                return "cuda"
    return "cpu"


def _validate_untyped_ir(pyfunc, options=None):
    numba = _load_numba()
    func_ir = _get_untyped_ir(pyfunc, numba)
    from numba.core import ir

    def resolve_def(var):
        current = var
        seen = set()
        while isinstance(current, ir.Var):
            if current.name in seen:
                return None
            seen.add(current.name)
            defs = func_ir._definitions.get(current.name, [])
            if not defs:
                return None
            current = defs[-1]
        return current

    def is_cuda_attr(call_expr, attr):
        if not isinstance(call_expr, ir.Expr) or call_expr.op != "call":
            return False
        func = resolve_def(call_expr.func)
        if not isinstance(func, ir.Expr) or func.op != "getattr":
            return False
        if func.attr != attr:
            return False
        base = resolve_def(func.value)
        return isinstance(base, ir.Global) and getattr(base.value, "__name__", "") == "numba.cuda"

    for block in func_ir.blocks.values():
        for stmt in block.body:
            value = None
            if isinstance(stmt, ir.Assign):
                value = stmt.value
            elif isinstance(stmt, ir.Expr):
                value = stmt
            if isinstance(value, ir.Expr):
                if value.op == "build_slice":
                    raise FrontendError("slice is not supported")
                if value.op == "call":
                    fn = value.func
                    if isinstance(fn, ir.Global) and fn.value is slice:
                        raise FrontendError("slice is not supported")
                    if is_cuda_attr(value, "shfl_xor_sync"):
                        if not options or "sig" not in options:
                            raise FrontendError("cuda.shfl_xor_sync requires explicit signature")
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Global):
                if stmt.value.value is slice:
                    raise FrontendError("slice is not supported")


def _dump_typed_ir(pyfunc, target, func_ir=None, type_annotation=None, err=None):
    dump_path = os.environ.get(_ENV_DUMP_TYPED_IR, "").strip()
    map_path = os.environ.get(_ENV_DUMP_MAP, "").strip()
    if not dump_path and not map_path:
        return None

    if dump_path:
        header = f"func={pyfunc.__name__} target={target}\n"
        if type_annotation is not None:
            text = header + str(type_annotation)
        elif err is not None:
            text = header + f"typed IR failed: {err}\n"
        else:
            text = header + "typed IR unavailable\n"
        _dump_text(dump_path, text)

    if map_path and func_ir is not None:
        lines = []
        stmt_id = 0
        for label, block in sorted(func_ir.blocks.items()):
            for index, stmt in enumerate(block.body):
                stmt_id += 1
                loc = getattr(stmt, "loc", None)
                if loc is None:
                    loc_str = "unknown"
                elif getattr(loc, "col", None) is not None:
                    loc_str = f"{loc.filename}:{loc.line}:{loc.col}"
                else:
                    loc_str = f"{loc.filename}:{loc.line}"
                lines.append(f"{stmt_id:04d} block={label} index={index} loc={loc_str} stmt={stmt}")
        _dump_text(map_path, "\n".join(lines))


def _dump_timing(path, pyfunc, target, timings, err=None):
    if not path or not timings:
        return
    lines = [f"func={pyfunc.__name__} target={target}"]
    for key, value in timings.items():
        if value is None:
            continue
        lines.append(f"{key}={value:.6f}s")
    if err is not None:
        lines.append(f"error={err}")
    _dump_text(path, "\n".join(lines))

def _numpy_dtype_to_ti(dtype, ti):
    import numpy as np

    mapping = {
        np.dtype("float32"): ti.f32,
        np.dtype("float64"): ti.f64,
        np.dtype("int32"): ti.i32,
        np.dtype("int64"): ti.i64,
        np.dtype("uint32"): ti.u32,
        np.dtype("uint64"): ti.u64,
        np.dtype("bool"): ti.u1,
    }
    return mapping.get(dtype)


def _infer_annotations(args, ti):
    import numpy as np

    annotations = {}
    for index, value in enumerate(args):
        if isinstance(value, np.ndarray):
            dtype = _numpy_dtype_to_ti(value.dtype, ti)
            if dtype is None:
                raise FrontendError(f"unsupported ndarray dtype: {value.dtype}")
            annotations[index] = ti.types.ndarray(dtype=dtype, ndim=value.ndim)
        elif hasattr(value, "dtype") and hasattr(value, "shape"):
            dtype = value.dtype
            if dtype is None:
                raise FrontendError("unsupported ndarray dtype: None")
            ndim = len(value.shape) if hasattr(value, "shape") else 0
            annotations[index] = ti.types.ndarray(dtype=dtype, ndim=ndim)
        elif isinstance(value, (int, np.integer)):
            annotations[index] = ti.i32
        elif isinstance(value, (float, np.floating)):
            annotations[index] = ti.f32
        else:
            raise FrontendError(f"unsupported argument type: {type(value)!r}")
    return annotations


def _build_stub(pyfunc, annotations):
    arg_names = list(inspect.signature(pyfunc).parameters.keys())
    args_src = ", ".join(arg_names)
    src = f"def {pyfunc.__name__}({args_src}):\n    pass\n"
    namespace = {}
    exec(src, namespace)
    stub = namespace[pyfunc.__name__]
    stub.__annotations__ = {name: annotations[i] for i, name in enumerate(arg_names)}
    return stub


class _NumbaKernelPlaceholder:
    def __init__(self, fn, options):
        self._fn = fn
        self._options = dict(options)
        self._kernel = None
        functools.update_wrapper(self, fn)
        if is_available():
            _validate_untyped_ir(self._fn, self._options)

    def __call__(self, *args, **kwargs):
        if not is_available():
            raise FrontendError("numba typed frontend is not enabled")
        import taichi as ti
        if self._kernel is None:
            ordered_args = _ordered_args(self._fn, args, kwargs)
            annotations = _infer_annotations(ordered_args, ti)
            stub = _build_stub(self._fn, annotations)
            self._kernel = _NumbaTranslatedKernel(self._fn, stub, self._options)
        return self._kernel(*args, **kwargs)


class _NumbaTranslatedKernel:
    def __init__(self, pyfunc, stub, options):
        from taichi.lang.kernel_impl import AutodiffMode, Kernel

        self._pyfunc = pyfunc
        self._options = dict(options)
        self._kernel = _NumbaKernel(stub, pyfunc, self._options, AutodiffMode.NONE)

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)


class _NumbaKernel:
    def __init__(self, stub, pyfunc, options, autodiff_mode):
        from taichi.lang.kernel_impl import Kernel

        self._kernel = Kernel(stub, autodiff_mode)
        self._pyfunc = pyfunc
        self._options = dict(options)
        self._kernel.materialize = self._materialize  # type: ignore[method-assign]
        self._typed_ir_cache = {}

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)

    def _materialize(self, key=None, args=None, arg_features=None):
        from taichi.lang import impl

        if key is None:
            key = (self._kernel.func, 0, self._kernel.autodiff_mode)
        self._kernel.runtime.materialize()
        if key in self._kernel.compiled_kernels:
            return

        kernel_name = f"{self._kernel.func.__name__}_c{self._kernel.kernel_counter}_{key[1]}"
        target_hint = self._options.get("target")
        taichi_dump_path = os.environ.get(_ENV_DUMP_TAICHI_IR, "").strip()
        builder_dump_path = os.environ.get(_ENV_DUMP_BUILDER, "").strip()
        timing_path = os.environ.get(_ENV_DUMP_TIMING, "").strip()

        def taichi_ast_generator(kernel_cxx):
            if self._kernel.runtime.inside_kernel:
                raise FrontendError("nested kernel translation is not supported")
            self._kernel.kernel_cpp = kernel_cxx
            self._kernel.runtime.inside_kernel = True
            self._kernel.runtime.current_kernel = self._kernel
            self._kernel.runtime.compiling_callable = kernel_cxx
            func_ir = None
            type_annotation = None
            err = None
            target_name = target_hint or "auto"
            timing = None
            t_total_start = None
            if timing_path:
                timing = {
                    "typed_ir": None,
                    "translate": None,
                    "dump_trace": None,
                    "dump_typed_ir": None,
                    "total": None,
                }
                t_total_start = time.perf_counter()
            try:
                ordered_args = tuple(args) if args is not None else ()
                cache_key = None
                cache_entry = None
                arg_types = None
                try:
                    numba = _load_numba()
                    arg_types = tuple(_typeof_arg(arg, numba) for arg in ordered_args)
                    cache_key = (arg_types, target_hint)
                    cache_entry = self._typed_ir_cache.get(cache_key)
                except Exception:
                    cache_entry = None
                if cache_entry is not None:
                    func_ir, typemap, calltypes, type_annotation, resolved_target = cache_entry
                    target_name = resolved_target or target_name
                    if timing is not None:
                        timing["typed_ir"] = 0.0
                else:
                    t0 = time.perf_counter()
                    func_ir, typemap, calltypes, type_annotation, resolved_target = _get_typed_ir(
                        self._pyfunc, ordered_args, {}, target=target_hint, arg_types=arg_types
                    )
                    target_name = resolved_target or target_name
                    t1 = time.perf_counter()
                    if timing is not None:
                        timing["typed_ir"] = t1 - t0
                    if cache_key is not None:
                        self._typed_ir_cache[cache_key] = (
                            func_ir,
                            typemap,
                            calltypes,
                            type_annotation,
                            resolved_target,
                        )
                translator = NumbaIRTranslator(
                    func_ir=func_ir,
                    typemap=typemap,
                    calltypes=calltypes,
                    kernel=self._kernel,
                    options=self._options,
                )
                t0 = time.perf_counter()
                translator.translate()
                t1 = time.perf_counter()
                if timing is not None:
                    timing["translate"] = t1 - t0
                if taichi_dump_path:
                    t0 = time.perf_counter()
                    translator.dump_trace(taichi_dump_path)
                    t1 = time.perf_counter()
                    if timing is not None:
                        timing["dump_trace"] = t1 - t0
                if builder_dump_path:
                    translator.dump_builder_calls(builder_dump_path)
            except Exception as exc:
                err = exc
                raise FrontendError(str(exc)) from exc
            finally:
                t0 = time.perf_counter()
                _dump_typed_ir(self._pyfunc, target_name, func_ir=func_ir, type_annotation=type_annotation, err=err)
                t1 = time.perf_counter()
                if timing is not None:
                    timing["dump_typed_ir"] = t1 - t0
                    timing["total"] = t1 - (t_total_start or t1)
                    _dump_timing(timing_path, self._pyfunc, target_name, timing, err=err)
                self._kernel.runtime.inside_kernel = False
                self._kernel.runtime.current_kernel = None
                self._kernel.runtime.compiling_callable = None

        taichi_kernel = impl.get_runtime().prog.create_kernel(
            taichi_ast_generator, kernel_name, self._kernel.autodiff_mode
        )
        self._kernel.compiled_kernels[key] = taichi_kernel


def njit(func=None, **options):
    """Placeholder for the Numba typed frontend entrypoint."""

    def decorator(fn):
        return _NumbaKernelPlaceholder(fn, options)

    if func is None:
        return decorator
    return decorator(func)
