import numpy as np
from numba import cuda
import taichi as ti

from common import init_taichi, setup_demo_env


def main():
    paths = setup_demo_env("vec_add")
    arch = init_taichi(ti)

    @ti.njit
    def vec_add(a, b, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] + b[i]

    n = 8
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 2
    out = np.zeros_like(a)
    vec_add(a, b, out, n)

    print(out)
    print("arch:", arch)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
