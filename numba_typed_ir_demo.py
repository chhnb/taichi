import os

import numpy as np
from numba import cuda
import taichi as ti
ti.init(arch=ti.cpu)

def main():
    # os.environ.setdefault("TI_NUMBA_TF_DUMP_TYPED_IR", "/tmp/numba_typed_ir_demo.txt")
    # os.environ.setdefault("TI_NUMBA_TF_DUMP_TAICHI_IR", "/tmp/taichi_ir_demo.txt")

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
    # print("done")
    # print("typed IR:", os.environ["TI_NUMBA_TF_DUMP_TYPED_IR"])
    # print("taichi trace:", os.environ["TI_NUMBA_TF_DUMP_TAICHI_IR"])


if __name__ == "__main__":
    main()
