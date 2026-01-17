class DiagnosticLevel:
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Diagnostic:
    def __init__(self, level, message, location=None):
        self.level = level
        self.message = message
        self.location = location

    def __repr__(self):
        return f"Diagnostic(level={self.level!r}, message={self.message!r}, location={self.location!r})"


class DiagnosticSink:
    def __init__(self):
        self._items = []

    def add(self, level, message, location=None):
        self._items.append(Diagnostic(level, message, location))

    def error(self, message, location=None):
        self.add(DiagnosticLevel.ERROR, message, location)

    def warning(self, message, location=None):
        self.add(DiagnosticLevel.WARNING, message, location)

    def info(self, message, location=None):
        self.add(DiagnosticLevel.INFO, message, location)

    def has_errors(self):
        return any(item.level == DiagnosticLevel.ERROR for item in self._items)

    def items(self):
        return list(self._items)
