import numpy as np
from numba import cuda
import taichi as ti

from common import init_taichi, setup_demo_env


def main():
    paths = setup_demo_env("stencil2d")
    arch = init_taichi(ti)

    @ti.njit
    def stencil(inp, out, n, m):
        i, j = cuda.grid(2)
        if 0 < i < n - 1 and 0 < j < m - 1:
            out[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )

    n, m = 8, 6
    inp = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(inp)
    stencil(inp, out, n, m)

    print(out)
    print("arch:", arch)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
