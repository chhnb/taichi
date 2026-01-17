import os
import sys


_ENV_FORCE_LOCAL = "TI_NUMBA_TF_FORCE_LOCAL"
_ENV_USE_LOCAL = "TI_NUMBA_TF_USE_LOCAL"


def _repo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "../../../.."))


def _maybe_add_local_numba():
    repo_root = _repo_root()
    candidate = os.path.join(repo_root, "numba")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
        return True
    return False


def _is_local_numba(mod):
    path = getattr(mod, "__file__", "")
    if not path:
        return False
    repo_root = _repo_root()
    return os.path.abspath(path).startswith(os.path.join(repo_root, "numba"))


def _remove_local_numba_path(drop_repo_root=False):
    repo_root = _repo_root()
    candidate = os.path.join(repo_root, "numba")
    cleaned = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(candidate)]
    if drop_repo_root:
        cleaned = [p for p in cleaned if os.path.abspath(p) != os.path.abspath(repo_root)]
    sys.path[:] = cleaned


def ensure_local_numba():
    if "numba" in sys.modules and not _is_local_numba(sys.modules["numba"]):
        for name in list(sys.modules):
            if name == "numba" or name.startswith("numba."):
                del sys.modules[name]
    _maybe_add_local_numba()


def import_numba():
    force_local = os.environ.get(_ENV_FORCE_LOCAL) == "1"
    use_local = os.environ.get(_ENV_USE_LOCAL) == "1"
    if force_local:
        ensure_local_numba()
        import numba  # noqa: F401
        return numba

    if use_local:
        ensure_local_numba()
        try:
            import numba  # noqa: F401
            return numba
        except Exception as exc_local:
            _remove_local_numba_path()
            for name in list(sys.modules):
                if name == "numba" or name.startswith("numba."):
                    del sys.modules[name]
            try:
                import numba  # noqa: F401
                return numba
            except Exception as exc_system:
                raise ImportError(
                    f"failed to import numba (local={exc_local}; system={exc_system})"
                ) from exc_system

    if "numba" in sys.modules and not _is_local_numba(sys.modules["numba"]):
        return sys.modules["numba"]

    original_path = list(sys.path)
    _remove_local_numba_path(drop_repo_root=True)
    for name in list(sys.modules):
        if name == "numba" or name.startswith("numba."):
            del sys.modules[name]
    try:
        import numba  # noqa: F401
        return numba
    finally:
        sys.path[:] = original_path


def _can_import_numba():
    try:
        import_numba()
        return True
    except Exception:
        return False


def is_available():
    return _can_import_numba()
