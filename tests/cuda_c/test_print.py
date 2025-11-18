import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_kernel_print(capfd):
    ti.reset()
    ti.init(arch=ti.cuda_c)

    @ti.kernel
    def foo() -> ti.i32:
        tmp = 0
        tmp = tmp + 1
        tmp = tmp * 2
        print("tmp =", tmp)
        return tmp

    capfd.readouterr()
    assert foo() == 2
    ti.sync()
    out = capfd.readouterr().out
    assert "tmp = 2" in out

    ti.reset()
