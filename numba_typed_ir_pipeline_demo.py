import os

import numpy as np
from numba import cuda
import taichi as ti


def main():
    os.environ.setdefault("TI_NUMBA_TF_DUMP_TYPED_IR", "/tmp/numba_typed_ir_pipeline.txt")
    os.environ.setdefault("TI_NUMBA_TF_DUMP_TAICHI_IR", "/tmp/taichi_frontend_trace.txt")
    os.environ.setdefault("TI_NUMBA_TF_DUMP_MAP", "/tmp/numba_typed_ir_map.txt")
    os.environ.setdefault("TI_NUMBA_TF_DUMP_TIMING", "/tmp/numba_typed_ir_timing.txt")

    arch_name = os.environ.get("TI_DEMO_ARCH") or os.environ.get("TI_ARCH") or "cpu"
    arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    arch = arch_map.get(arch_name, ti.cpu)

    ti.init(arch=arch, print_ir=True)

    @ti.njit(target="cuda")
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
    print("arch:", arch_name)
    print("typed IR:", os.environ["TI_NUMBA_TF_DUMP_TYPED_IR"])
    print("typed IR map:", os.environ["TI_NUMBA_TF_DUMP_MAP"])
    print("frontend trace:", os.environ["TI_NUMBA_TF_DUMP_TAICHI_IR"])
    print("timing:", os.environ["TI_NUMBA_TF_DUMP_TIMING"])
    print("taichi front IR: printed to stdout (use shell redirect to capture)")


if __name__ == "__main__":
    main()
