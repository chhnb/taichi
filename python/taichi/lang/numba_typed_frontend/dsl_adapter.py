from taichi.dsl_to_taichi import DSLToTaichiTranslator, register_translator

from .api import _get_typed_ir
from .diagnostics import FrontendError
from .translator import NumbaIRTranslator


class NumbaDSLTranslator(DSLToTaichiTranslator):
    """Adapter that exposes the existing NumbaIRTranslator via the DSL API."""

    def __init__(self, context, diagnostics=None):
        super().__init__(context, diagnostics=diagnostics)
        self._dsl_ir = None
        self._pyfunc = None
        self._func_ir = None
        self._typemap = None
        self._calltypes = None
        self._type_annotation = None
        self._target = None
        self._kernel = None
        self._options = {}
        self._translator = None

    def parse(self, dsl_ir):
        if not isinstance(dsl_ir, dict):
            raise FrontendError("dsl_ir must be a dict")
        self._dsl_ir = dsl_ir
        self._pyfunc = dsl_ir.get("pyfunc")
        self._target = dsl_ir.get("target") or getattr(self.context, "target", None)
        self._options = dict(dsl_ir.get("options", {}))
        self._kernel = dsl_ir.get("kernel") or getattr(self.context, "kernel", None)
        if self._kernel is None:
            raise FrontendError("kernel is required for Numba DSL translation")

        if "func_ir" in dsl_ir:
            self._func_ir = dsl_ir["func_ir"]
            self._typemap = dsl_ir.get("typemap", {})
            self._calltypes = dsl_ir.get("calltypes", {})
            self._type_annotation = dsl_ir.get("type_annotation")
            return

        if self._pyfunc is None:
            raise FrontendError("pyfunc is required when func_ir is not provided")
        args = dsl_ir.get("args", ())
        kwargs = dsl_ir.get("kwargs", {})
        arg_types = dsl_ir.get("arg_types")
        (
            self._func_ir,
            self._typemap,
            self._calltypes,
            self._type_annotation,
            self._target,
        ) = _get_typed_ir(self._pyfunc, args, kwargs, target=self._target, arg_types=arg_types)

    def lower_types(self):
        if self.context.target is None and self._target is not None:
            self.context.target = self._target

    def lower_intrinsics(self):
        # Numba-specific intrinsics are handled in NumbaIRTranslator.
        return None

    def build_taichi_ir(self):
        if self._func_ir is None:
            raise FrontendError("typed IR is not available; call parse() first")
        self._translator = NumbaIRTranslator(
            func_ir=self._func_ir,
            typemap=self._typemap,
            calltypes=self._calltypes,
            kernel=self._kernel,
            options=self._options,
        )
        self._translator.translate()
        return self._translator

    def finalize(self):
        return None


try:
    register_translator("numba_typed", NumbaDSLTranslator)
except KeyError:
    pass
