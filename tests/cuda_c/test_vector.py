import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_fill_vector_field():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    n = 16
    vec = ti.Vector.field(2, dtype=ti.f32, shape=n)

    @ti.kernel
    def fill(offset: ti.f32):
        for i in range(n):
            base = ti.Vector([ti.cast(i, ti.f32), ti.cast(i, ti.f32) * 2.0])
            vec[i] = base + ti.Vector([offset, -offset])

    fill(1.5)

    host = vec.to_numpy()
    expected = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        base = np.array([i, i * 2], dtype=np.float32)
        expected[i] = base + np.array([1.5, -1.5], dtype=np.float32)

    np.testing.assert_allclose(host, expected)

    ti.reset()
