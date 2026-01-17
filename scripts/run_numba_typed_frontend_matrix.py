import os
import subprocess
import sys
from pathlib import Path


def _run(cmd, env):
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "python"))

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root / "python"))

    try:
        import taichi as ti
    except Exception as exc:
        print(f"Failed to import taichi: {exc}")
        raise SystemExit(1) from exc

    archs = ["cpu"]
    if hasattr(ti._lib.core, "with_cuda") and ti._lib.core.with_cuda():
        archs.append("cuda")
    if hasattr(ti._lib.core, "with_vulkan") and ti._lib.core.with_vulkan():
        archs.append("vulkan")

    only = env.get("TI_ARCH_MATRIX")
    if only:
        requested = [a.strip() for a in only.split(",") if a.strip()]
        archs = [a for a in archs if a in requested]

    for arch in archs:
        env["TI_TEST_ARCH"] = arch
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(repo_root / "tests/python/test_numba_typed_frontend_cpu.py"),
            ],
            env,
        )

    if "cuda" in archs:
        try:
            from numba import cuda as nb_cuda
        except Exception:
            nb_cuda = None
        if nb_cuda is not None and nb_cuda.is_available():
            _run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(repo_root / "tests/python/test_numba_typed_frontend_cuda.py"),
                ],
                env,
            )
        else:
            print("Numba CUDA not available, skip test_numba_typed_frontend_cuda.py")


if __name__ == "__main__":
    main()
