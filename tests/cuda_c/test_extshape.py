import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


pytestmark = pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)


def setup_module():
    ti.reset()
    ti.init(arch=ti.cuda_c)


def teardown_module():
    ti.reset()


def test_external_ndarray_shape_1d():
    @ti.kernel
    def fill(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        n = arr.shape[0]
        arr[0] = n
        arr[1] = n * 2.0
        for i in range(2, n):
            arr[i] = n + i

    data = np.zeros(11, dtype=np.float32)
    fill(data)
    expected = np.arange(11, dtype=np.float32) + 11
    expected[0] = 11
    expected[1] = 22
    np.testing.assert_allclose(data, expected)


def test_external_ndarray_shape_2d():
    @ti.kernel
    def fill(arr: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        n0 = arr.shape[0]
        n1 = arr.shape[1]
        for i in range(n0):
            for j in range(n1):
                arr[i, j] = n0 * 10 + n1 + i * 2 + j

    h, w = 3, 5
    data = np.zeros((h, w), dtype=np.float32)
    fill(data)
    expected = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            expected[i, j] = h * 10 + w + i * 2 + j
    np.testing.assert_allclose(data, expected)


def test_external_ndarray_shape_3d():
    @ti.kernel
    def fill(arr: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        n0 = arr.shape[0]
        n1 = arr.shape[1]
        n2 = arr.shape[2]
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    arr[i, j, k] = n0 * 100 + n1 * 10 + n2 + i + j * 2 + k * 3

    dims = (2, 3, 4)
    data = np.zeros(dims, dtype=np.float32)
    fill(data)
    expected = np.zeros(dims, dtype=np.float32)
    n0, n1, n2 = dims
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                expected[i, j, k] = n0 * 100 + n1 * 10 + n2 + i + j * 2 + k * 3
    np.testing.assert_allclose(data, expected)
