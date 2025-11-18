import numpy as np
import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_check_prime_kernel():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    @ti.kernel
    def check_prime(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(arr.shape[0]):
            num = arr[i]
            if num <= 1:
                arr[i] = 0
            elif num == 2:
                arr[i] = 1
            elif num % 2 == 0:
                arr[i] = 0
            else:
                is_prime = 1
                j = 3
                while j * j <= num:
                    if num % j == 0:
                        is_prime = 0
                        break
                    j += 2
                arr[i] = is_prime

    nums = np.arange(1, 100, dtype=np.int32)
    check_prime(nums)
    expected = np.array(
        [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        dtype=np.int32,
    )  # 只需覆盖到你检查的范围
    np.testing.assert_array_equal(nums[: expected.size], expected)

    ti.reset()
