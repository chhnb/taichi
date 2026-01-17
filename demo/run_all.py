import os
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = repo_root / "demo"
    output_dir = demo_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    demos = [
        "vec_add_demo.py",
        "grid_stride_demo.py",
        "stencil2d_demo.py",
        "atomic_reduce_demo.py",
        "phi_demo.py",
        "sieve_demo.py",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root / "python"))

    for script in demos:
        name = script.replace("_demo.py", "")
        out_path = output_dir / f"{name}_taichi_ir.txt"
        cmd = [sys.executable, str(demo_dir / script)]
        print(" ".join(cmd))
        with open(out_path, "w", encoding="utf-8") as handle:
            subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
