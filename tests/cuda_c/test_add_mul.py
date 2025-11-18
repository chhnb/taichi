import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_vector_add_and_mul():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    n = 1000  # 10000 也行，但 pytest 下可以用稍小的尺寸提速
    a = ti.field(dtype=ti.f32, shape=n)
    b = ti.field(dtype=ti.f32, shape=n)
    c = ti.field(dtype=ti.f32, shape=n)

    @ti.kernel
    def init_fields():
        for i in range(n):
            a[i] = i * 1.0 + 10.0
            b[i] = i * 2.0

    @ti.kernel
    def vector_add():
        for i in range(n):
            c[i] = a[i] + b[i]

    @ti.kernel
    def vector_mul():
        for i in range(n):
            c[i] = a[i] * b[i]

    init_fields()
    vector_add()
    np.testing.assert_allclose(
        c.to_numpy(), np.arange(n, dtype=np.float32) * 3 + 10.0
    )

    vector_mul()
    expected_mul = (np.arange(n, dtype=np.float32) + 10.0) * (
        np.arange(n, dtype=np.float32) * 2.0
    )
    np.testing.assert_allclose(c.to_numpy(), expected_mul)

    ti.reset()
