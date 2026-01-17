import numpy as np
from numba import cuda
import taichi as ti

from common import init_taichi, setup_demo_env


def main():
    paths = setup_demo_env("grid_stride")
    arch = init_taichi(ti)

    @ti.njit
    def grid_stride(a, out, n):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for i in range(i, n, stride):
            out[i] = a[i] * 3

    n = 16
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    grid_stride(a, out, n)

    print(out)
    print("arch:", arch)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
