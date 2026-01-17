import numpy as np
from numba import cuda
import taichi as ti

from common import init_taichi, setup_demo_env


def main():
    paths = setup_demo_env("atomic_reduce")
    arch = init_taichi(ti)

    @ti.njit
    def reduce_sum(a, out, n):
        i = cuda.grid(1)
        if i < n:
            cuda.atomic.add(out, 0, a[i])

    n = 32
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    reduce_sum(a, out, n)

    print(out)
    print("arch:", arch)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
