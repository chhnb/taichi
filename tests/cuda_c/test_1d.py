import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_add_length():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    @ti.kernel
    def add_length(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        n = arr.shape[0]
        for i in range(n):
            arr[i] += n

    data = np.arange(16, dtype=np.int32)
    add_length(data)
    np.testing.assert_array_equal(data, np.arange(16, dtype=np.int32) + 16)

    ti.reset()
