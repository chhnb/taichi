from abc import ABC, abstractmethod

from .diagnostics import DiagnosticSink


class TranslatorContext:
    def __init__(self, target=None, ir_builder=None, capabilities=None, kernel=None):
        self.target = target
        self.ir_builder = ir_builder
        self.capabilities = set(capabilities) if capabilities else set()
        self.kernel = kernel


class DSLToTaichiTranslator(ABC):
    def __init__(self, context, diagnostics=None):
        self.context = context
        self.diagnostics = diagnostics or DiagnosticSink()

    @abstractmethod
    def parse(self, dsl_ir):
        """Parse/normalize DSL IR to a minimal kernel IR."""

    @abstractmethod
    def lower_types(self):
        """Map DSL types to Taichi types."""

    @abstractmethod
    def lower_intrinsics(self):
        """Map DSL intrinsics to Taichi ops or externs."""

    @abstractmethod
    def build_taichi_ir(self):
        """Emit Taichi IR nodes using the IR builder."""

    @abstractmethod
    def finalize(self):
        """Validation and diagnostics."""

    def translate(self, dsl_ir):
        self.parse(dsl_ir)
        self.lower_types()
        self.lower_intrinsics()
        ir = self.build_taichi_ir()
        self.finalize()
        return ir
