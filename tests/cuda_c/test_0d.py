import pytest
import taichi as ti
from taichi.lang import misc as ti_misc


@pytest.mark.skipif(
    not ti_misc.is_arch_supported(ti.cuda_c),
    reason="CUDA C backend not available on this machine",
)
def test_scalar_field_init():
    ti.reset()
    ti.init(arch=ti.cuda_c)

    x = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def init_x():
        x[None] = 100.0

    init_x()
    assert x[None] == pytest.approx(100.0)
