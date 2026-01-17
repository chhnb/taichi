import numpy as np
from numba import cuda
import taichi as ti

from common import init_taichi, setup_demo_env


def main():
    paths = setup_demo_env("phi")
    arch = init_taichi(ti)

    @ti.njit
    def ternary_sign(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = 1.0 if a[i] >= 0 else -1.0

    n = 8
    a = (np.arange(n, dtype=np.float32) - 3) * 0.5
    out = np.zeros_like(a)
    ternary_sign(a, out, n)

    print(out)
    print("arch:", arch)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
