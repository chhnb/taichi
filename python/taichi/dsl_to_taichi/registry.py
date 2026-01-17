from .base import DSLToTaichiTranslator


_TRANSLATOR_REGISTRY = {}


def register_translator(name, translator_cls):
    if not issubclass(translator_cls, DSLToTaichiTranslator):
        raise TypeError("translator_cls must subclass DSLToTaichiTranslator")
    if name in _TRANSLATOR_REGISTRY:
        raise KeyError(f"translator already registered: {name}")
    _TRANSLATOR_REGISTRY[name] = translator_cls


def get_translator(name):
    return _TRANSLATOR_REGISTRY[name]


def list_translators():
    return sorted(_TRANSLATOR_REGISTRY.keys())


def create_translator(name, context, diagnostics=None):
    translator_cls = get_translator(name)
    return translator_cls(context, diagnostics=diagnostics)
