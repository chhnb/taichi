import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_transform_func():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    n = 100
    x = ti.field(dtype=ti.f32, shape=n)

    @ti.func
    def transform(val: ti.f32) -> ti.f32:
        return val * 2.0

    @ti.kernel
    def init_x():
        for i in range(n):
            x[i] = transform(i * 1.0 + 5.0)

    init_x()
    expected = (np.arange(n, dtype=np.float32) + 5.0) * 2.0
    np.testing.assert_allclose(x.to_numpy(), expected)

    ti.reset()
