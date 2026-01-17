import math
import os
from pathlib import Path

import numpy as np
import numba
import taichi as ti

from common import init_taichi, setup_demo_env


def main():
    paths = setup_demo_env("prime_check")
    builder_path = Path(paths["typed_ir"]).with_name("prime_check_builder.txt")
    os.environ.setdefault("TI_NUMBA_TF_DUMP_BUILDER", str(builder_path))
    arch = init_taichi(ti)

    @ti.njit
    def ti_prime_check(out, n):
        for i in range(n):
            if i < 2:
                out[i] = 0
                continue
            is_prime = 1
            limit = int(math.sqrt(i)) + 1
            for p in range(2, limit):
                if i % p == 0:
                    is_prime = 0
                    break
            out[i] = is_prime

    @numba.njit
    def nb_prime_check(out, n):
        for i in range(n):
            if i < 2:
                out[i] = 0
                continue
            is_prime = 1
            limit = int(math.sqrt(i)) + 1
            for p in range(2, limit):
                if i % p == 0:
                    is_prime = 0
                    break
            out[i] = is_prime

    n = int(os.environ.get("TI_DEMO_N", "10000"))
    out_nb = np.zeros(n, dtype=np.int32)
    out_ti = np.zeros(n, dtype=np.int32)
    nb_prime_check(out_nb, n)
    ti_prime_check(out_ti, n)

    print("match:", np.array_equal(out_nb, out_ti))
    print("prime_count:", int(out_ti.sum()))
    print("arch:", arch)
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"builder: {builder_path}")


if __name__ == "__main__":
    main()
