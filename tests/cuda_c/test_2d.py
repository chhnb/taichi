import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_fill_2d_field():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    n = 5
    x = ti.field(dtype=ti.i32, shape=(n, n))

    @ti.kernel
    def init():
        for i, j in x:
            x[i, j] = i * n + j

    init()

    expected = np.arange(n * n, dtype=np.int32).reshape(n, n)
    np.testing.assert_array_equal(x.to_numpy(), expected)

    ti.reset()
