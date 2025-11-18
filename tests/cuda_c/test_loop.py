import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_loop_kernel():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    @ti.kernel
    def loop_test(n: ti.i32) -> ti.i32:
        i = 0
        acc = 0
        while True:
            if i >= n:
                break
            i += 1
            if i % 2 == 0:
                continue
            acc += i
        return acc

    assert loop_test(5) == 9
    assert loop_test(1) == 1
    assert loop_test(0) == 0

    ti.reset()
