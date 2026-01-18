import math
import os

import numpy as np
import pytest

import taichi as ti
from taichi.lang.numba_typed_frontend import is_available as _NUMBA_TF_AVAILABLE
from taichi.lang.numba_typed_frontend.diagnostics import FrontendError

try:
    from numba import cuda as _real_cuda

    _CUDA_AVAILABLE = _real_cuda.is_available()
    cuda = _real_cuda
except Exception:  # pragma: no cover - fallback for environments without numba or cuda
    _CUDA_AVAILABLE = False

    class _CudaStub:
        @staticmethod
        def grid(dim):
            # Should never execute; used only for AST parsing in tests.
            raise RuntimeError("cuda grid should be rewritten before execution")

    cuda = _CudaStub()

pytestmark = pytest.mark.skipif(not _NUMBA_TF_AVAILABLE(), reason="numba typed frontend not enabled")

_sparse_field = None
_ad_x = None
_ad_loss = None


@ti.njit
def _math_ops(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = (
            math.sin(a[i])
            + math.cos(a[i])
            + math.sqrt(abs(a[i]))
            + np.exp(a[i])
            - math.log(a[i] + 1.0)
        )


@ti.njit(sig={"out": (ti.i32, 1)})
def _int_cast(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = int(a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.f32, 1)})
def _float_cast(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = float(a[i])


@ti.njit
def _while_sum(a, out, n):
    i = 0
    acc = 0.0
    while i < n:
        acc += a[i]
        i += 1
    out[0] = acc


@ti.njit
def _ternary_sign(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = 1.0 if a[i] >= 0 else -1.0


@ti.njit
def _break_first_positive(a, out, n):
    i = 0
    val = -1.0
    while i < n:
        if a[i] > 0:
            val = a[i]
            break
        i += 1
    out[0] = val


@ti.njit
def _continue_even(a, out, n):
    for i in range(n):
        if i % 2 != 0:
            continue
        out[i] = a[i] * 2


def _square_plus_one(x):
    return x * x + 1.0


def _helper_with_side_effect(x):
    y = x + 1.0
    return y


@ti.njit
def _inline_helper_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = _square_plus_one(a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})
def _atomic_min_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.min(out, 0, a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})
def _atomic_max_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.max(out, 0, a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})
def _atomic_sub_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.sub(out, 0, a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})
def _atomic_and_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.and_(out, 0, a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})
def _atomic_or_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.or_(out, 0, a[i])


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})
def _atomic_xor_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        cuda.atomic.xor(out, 0, a[i])


@ti.njit
def _vec_add(a, b, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = a[i] + b[i]


@ti.njit
def _vec_add_grid2(a, b, out, n, m):
    i, j = cuda.grid(2)
    if i < n and j < m:
        out[i, j] = a[i, j] + b[i, j]


@ti.njit
def _grid_stride_double(a, out, n):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < n:
        out[i] = a[i] * 2


@ti.njit
def _grid_stride_double_y(a, out, n):
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if j < n:
        out[j] = a[j] * 2


@ti.njit
def _grid_stride_double_z(a, out, n):
    k = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    if k < n:
        out[k] = a[k] * 2


@ti.njit
def _grid_stride_direct(a, out, n):
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        out[i] = a[i] * 2


@ti.njit
def _grid_stride_gridsize_var(a, out, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, n, stride):
        out[i] = a[i] + 1


@ti.njit
def _grid_stride_gridsize_prefix(a, out, n):
    stride = cuda.gridsize(1)
    for i in range(cuda.grid(1), n, stride):
        out[i] = a[i] + 2


@ti.njit
def _grid_stride_gridsize_scaled(a, out, n):
    stride = cuda.gridsize(1) * 2
    for i in range(cuda.grid(1), n, stride):
        out[i] = a[i] + 3


@ti.njit
def _grid_stride_block_scaled(a, out, n):
    stride = cuda.blockDim.x * cuda.gridDim.x * 2
    for i in range(cuda.grid(1), n, stride):
        out[i] = a[i] + 4


@ti.njit
def _grid_stride_gridsize_2d(a, out, n, m):
    i, j = cuda.grid(2)
    stride_i, stride_j = cuda.gridsize(2)
    for i in range(i, n, stride_i):
        for j in range(j, m, stride_j):
            out[i, j] = a[i, j] + 5


@ti.njit
def _grid_stride_gridsize_2d_scaled(a, out, n, m):
    i, j = cuda.grid(2)
    stride_i, stride_j = cuda.gridsize(2)
    stride_i2, stride_j2 = stride_i * 2, stride_j * 2
    for i in range(i, n, stride_i2):
        for j in range(j, m, stride_j2):
            out[i, j] = a[i, j] + 6


@ti.njit
def _grid_stride_gridsize_3d(a, out, n0, n1, n2):
    i, j, k = cuda.grid(3)
    stride_i, stride_j, stride_k = cuda.gridsize(3)
    for i in range(i, n0, stride_i):
        for j in range(j, n1, stride_j):
            for k in range(k, n2, stride_k):
                out[i, j, k] = a[i, j, k] + 6


@ti.njit
def _grid_stride_gridsize_3d_scaled(a, out, n0, n1, n2):
    i, j, k = cuda.grid(3)
    stride_i, stride_j, stride_k = cuda.gridsize(3)
    stride_i2, stride_j2, stride_k2 = stride_i * 2, stride_j * 2, stride_k * 2
    for i in range(i, n0, stride_i2):
        for j in range(j, n1, stride_j2):
            for k in range(k, n2, stride_k2):
                out[i, j, k] = a[i, j, k] + 7


@ti.njit
def _grid_stride_gridsize_expr(a, out, n):
    stride = cuda.gridsize(1) * (2 + 2)
    for i in range(cuda.grid(1), n, stride):
        out[i] = a[i] + 7


@ti.njit
def _grid_stride_gridsize_add(a, out, n):
    stride = cuda.gridsize(1)
    stride2 = stride + stride
    for i in range(cuda.grid(1), n, stride2):
        out[i] = a[i] + 8


@ti.njit
def _grid_stride_gridsize_mul_identity(a, out, n):
    stride = cuda.gridsize(1)
    stride2 = stride * 1
    for i in range(cuda.grid(1), n, stride2):
        out[i] = a[i] + 9


@ti.njit(sig={"mask": (ti.i32, 1)})
def _masked_relu(a, mask, out, n):
    i = cuda.grid(1)
    if i < n:
        if not mask[i] or a[i] < 0:
            out[i] = 0.0
        else:
            out[i] = a[i]


@ti.njit
def _grid3_add(a, b, out, n0, n1, n2):
    i, j, k = cuda.grid(3)
    if i < n0 and j < n1 and k < n2:
        out[i, j, k] = a[i, j, k] + b[i, j, k]


@ti.njit
def _laplacian3d(inp, out, n0, n1, n2):
    i, j, k = cuda.grid(3)
    if 0 < i < n0 - 1 and 0 < j < n1 - 1 and 0 < k < n2 - 1:
        out[i, j, k] = (
            -6.0 * inp[i, j, k]
            + inp[i - 1, j, k]
            + inp[i + 1, j, k]
            + inp[i, j - 1, k]
            + inp[i, j + 1, k]
            + inp[i, j, k - 1]
            + inp[i, j, k + 1]
        )


@ti.njit(sig={"mask": (ti.i32, 1)})
def _grid_stride_masked(a, mask, out, n):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(i, n, stride):
        if mask[i]:
            out[i] = a[i] * 3


@ti.njit(sig={"mask": (ti.i32, 1)})
def _grid_stride_continue(a, mask, out, n):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(i, n, stride):
        if mask[i] == 0:
            continue
        out[i] = a[i] * 5


@ti.njit
def _grid_stride_2d(a, out, n, m):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    stride_i = cuda.blockDim.x * cuda.gridDim.x
    stride_j = cuda.blockDim.y * cuda.gridDim.y
    for i in range(i, n, stride_i):
        for j in range(j, m, stride_j):
            out[i, j] = a[i, j] * 2


@ti.njit
def _grid_stride_3d(a, out, n0, n1, n2):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    k = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    stride_i = cuda.blockDim.x * cuda.gridDim.x
    stride_j = cuda.blockDim.y * cuda.gridDim.y
    stride_k = cuda.blockDim.z * cuda.gridDim.z
    for i in range(i, n0, stride_i):
        for j in range(j, n1, stride_j):
            for k in range(k, n2, stride_k):
                out[i, j, k] = a[i, j, k] + 1.0


@ti.njit
def _grid_stride_2d_continue(a, out, n, m):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    stride_i = cuda.blockDim.x * cuda.gridDim.x
    stride_j = cuda.blockDim.y * cuda.gridDim.y
    for i in range(i, n, stride_i):
        for j in range(j, m, stride_j):
            if (i + j) % 2 != 0:
                continue
            out[i, j] = a[i, j] * 2


@ti.njit
def _grid_stride_3d_continue(a, out, n0, n1, n2):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    k = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    stride_i = cuda.blockDim.x * cuda.gridDim.x
    stride_j = cuda.blockDim.y * cuda.gridDim.y
    stride_k = cuda.blockDim.z * cuda.gridDim.z
    for i in range(i, n0, stride_i):
        for j in range(j, n1, stride_j):
            for k in range(k, n2, stride_k):
                if (i + j + k) % 3 != 0:
                    continue
                out[i, j, k] = a[i, j, k] + 1.0


@ti.njit
def _mod_mask(a, out, n):
    i = cuda.grid(1)
    if i < n:
        if i % 2 == 0:
            out[i] = a[i % n]


@ti.njit
def _bitand_mask(a, out, n):
    i = cuda.grid(1)
    if i < n:
        if i & 1 == 0:
            out[i] = a[i]


@ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 2)})
def _bitwise_ops(a, out, n):
    i = cuda.grid(1)
    if i < n:
        x = a[i]
        out[i, 0] = x | 1
        out[i, 1] = x ^ 3
        out[i, 2] = x << 1
        out[i, 3] = x >> 1


@ti.njit(sig={"mask": (ti.i32, 1), "dst_idx": (ti.i32, 1)})
def _masked_scatter(data, dst_idx, mask, out, n):
    i = cuda.grid(1)
    if i < n:
        if mask[i]:
            cuda.atomic.add(out, dst_idx[i], data[i])


@ti.njit(sig={"mask": (ti.i32, 2)})
def _mask2d(inp, mask, out, n0, n1):
    i, j = cuda.grid(2)
    if i < n0 and j < n1:
        if mask[i, j]:
            out[i, j] = inp[i, j]
        else:
            out[i, j] = 0.0


@ti.njit
def _reduce3d(inp, out, n0, n1, n2):
    i, j, k = cuda.grid(3)
    if i < n0 and j < n1 and k < n2:
        cuda.atomic.add(out, 0, inp[i, j, k])


@ti.njit
def _heat_step(u, u_new, n0, n1):
    for i in range(1, n0 - 1):
        for j in range(1, n1 - 1):
            u_new[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
    for i in range(n0):
        for j in range(n1):
            u[i, j] = u_new[i, j]


@ti.njit(sig={"pos": (ti.i32, 2), "weight": (ti.f32, 1), "grid": (ti.f32, 1)})
def _point_scatter(pos, weight, grid, n, grid_res):
    i = cuda.grid(1)
    if i < n:
        gx = pos[i, 0]
        gy = pos[i, 1]
        if 0 <= gx < grid_res and 0 <= gy < grid_res:
            idx = gx * grid_res + gy
            cuda.atomic.add(grid, idx, weight[i])


@ti.njit(sig={"a": (ti.f64, 1), "b": (ti.f64, 1), "out": (ti.f64, 1), "n": ti.i32})
def _sig_vec_add(a, b, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = a[i] + b[i]


@ti.njit
def _range_add(a, b, out):
    for i in range(a.shape[0]):
        out[i] = a[i] + b[i]


@ti.njit
def _prefix_sum(a, out):
    out[0] = a[0]
    for i in range(1, a.shape[0]):
        out[i] = out[i - 1] + a[i]


@ti.njit
def _stencil2d(inp, out):
    for i in range(1, inp.shape[0] - 1):
        for j in range(1, inp.shape[1] - 1):
            out[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )


@ti.njit
def _matmul(a, b, out):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            acc = 0.0
            for k in range(a.shape[1]):
                acc += a[i, k] * b[k, j]
            out[i, j] = acc


def _ti_cpu():
    ti.reset()
    ti.init(arch=ti.cpu)


def test_vec_add_ir_and_correctness():
    _ti_cpu()
    n = 16
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 2
    out = np.zeros(n, dtype=np.float32)
    _vec_add(a, b, out, n)
    assert np.allclose(out, a + b)


def test_multi_backend_consistency():
    if not hasattr(ti._lib.core, "with_cuda") or not hasattr(ti._lib.core, "with_vulkan"):
        pytest.skip("backend availability query not supported")

    def run_on(arch, a_np):
        ti.reset()
        try:
            ti.init(arch=arch)
        except Exception as exc:
            pytest.skip(f"{arch} init failed: {exc}")
        n = a_np.shape[0]
        a = ti.ndarray(dtype=ti.f32, shape=n)
        out = ti.ndarray(dtype=ti.f32, shape=n)
        a.from_numpy(a_np)

        @ti.njit
        def kernel(a, out, n):
            i = cuda.grid(1)
            if i < n:
                out[i] = a[i] * 2.0 + 1.5

        kernel(a, out, n)
        return out.to_numpy()

    a = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    cpu_out = run_on(ti.cpu, a)

    if ti._lib.core.with_cuda() and _CUDA_AVAILABLE:
        cuda_out = run_on(ti.cuda, a)
        assert np.allclose(cuda_out, cpu_out)

    if ti._lib.core.with_vulkan():
        vk_out = run_on(ti.vulkan, a)
        assert np.allclose(vk_out, cpu_out)


def test_unsupported_slice_is_diagnosed():
    def bad(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i:n]

    with pytest.raises(FrontendError):
        ti.njit(bad)


def test_range_negative_step_is_diagnosed():
    _ti_cpu()

    def bad(out):
        for i in range(10, 0, -1):
            out[i] = 1

    with pytest.raises(FrontendError):
        ti.njit(bad)(np.zeros(11, dtype=np.int32))


def _arch_from_env():
    name = (os.environ.get("TI_TEST_ARCH") or "").strip().lower()
    if not name:
        return None
    mapping = {
        "cpu": ti.cpu,
        "x64": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
        "gles": ti.gles,
    }
    return mapping.get(name)


def _available_archs():
    archs = [ti.cpu]
    if ti._lib.core.with_cuda() and _CUDA_AVAILABLE:
        archs.append(ti.cuda)
    if ti._lib.core.with_vulkan():
        archs.append(ti.vulkan)
    return archs


def _run_on_arch(arch, fn, *args):
    ti.reset()
    ti.init(arch=arch, offline_cache=False)
    try:
        return fn(*args)
    finally:
        ti.reset()


def test_multi_backend_elemwise():
    arch_env = _arch_from_env()
    archs = [arch_env] if arch_env is not None else _available_archs()

    a_np = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    @ti.njit
    def kernel(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] * 2.0 + 1.5

    def run(arch):
        n = a_np.shape[0]
        a = ti.ndarray(dtype=ti.f32, shape=n)
        out = ti.ndarray(dtype=ti.f32, shape=n)
        a.from_numpy(a_np)
        kernel(a, out, n)
        return out.to_numpy()

    ref = None
    for arch in archs:
        out = _run_on_arch(arch, run, arch)
        if ref is None:
            ref = out
        else:
            assert np.allclose(out, ref)


def test_multi_backend_atomic_sum():
    arch_env = _arch_from_env()
    archs = [arch_env] if arch_env is not None else _available_archs()

    a_np = np.arange(128, dtype=np.float32)
    expected = np.sum(a_np)

    @ti.njit
    def kernel(a, out, n):
        i = cuda.grid(1)
        if i < n:
            cuda.atomic.add(out, 0, a[i])

    def run(arch):
        n = a_np.shape[0]
        a = ti.ndarray(dtype=ti.f32, shape=n)
        out = ti.ndarray(dtype=ti.f32, shape=1)
        a.from_numpy(a_np)
        out.fill(0.0)
        kernel(a, out, n)
        return out.to_numpy()[0]

    for arch in archs:
        out = _run_on_arch(arch, run, arch)
        assert np.allclose(out, expected)


def test_multi_backend_ndarray_copy():
    arch_env = _arch_from_env()
    archs = [arch_env] if arch_env is not None else _available_archs()

    a_np = (np.arange(32, dtype=np.float32) - 3.0).reshape(8, 4)

    @ti.njit
    def kernel(a, out, n, m):
        i, j = cuda.grid(2)
        if i < n and j < m:
            out[i, j] = a[i, j]

    def run(arch):
        n, m = a_np.shape
        a = ti.ndarray(dtype=ti.f32, shape=(n, m))
        out = ti.ndarray(dtype=ti.f32, shape=(n, m))
        a.from_numpy(a_np)
        kernel(a, out, n, m)
        return out.to_numpy()

    ref = None
    for arch in archs:
        out = _run_on_arch(arch, run, arch)
        if ref is None:
            ref = out
        else:
            assert np.allclose(out, ref)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available for numba comparison")
def test_vec_add_matches_numba_cuda():
    ti.reset()
    ti.init(arch=ti.cuda)
    n = 1024
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 3
    out_ti = np.zeros_like(a)

    _vec_add(a, b, out_ti, n)

    @_real_cuda.jit
    def nb_vec_add(x, y, out):
        i = _real_cuda.grid(1)
        if i < out.size:
            out[i] = x[i] + y[i]

    d_a = _real_cuda.to_device(a)
    d_b = _real_cuda.to_device(b)
    d_out = _real_cuda.device_array_like(a)
    threads = 256
    blocks = (n + threads - 1) // threads
    try:
        nb_vec_add[blocks, threads](d_a, d_b, d_out)
        out_nb = d_out.copy_to_host()
    except Exception as exc:
        pytest.skip(f"numba cuda failed: {exc}")

    assert np.allclose(out_ti, out_nb)


def test_range_add_cpu():
    _ti_cpu()
    n = 8
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 5
    out = np.zeros(n, dtype=np.float32)
    _range_add(a, b, out)
    assert np.allclose(out, a + b)


def test_prefix_sum_cpu():
    _ti_cpu()
    n = 10
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    _prefix_sum(a, out)
    assert np.allclose(out, np.cumsum(a))


def test_axpy_cpu():
    _ti_cpu()
    n = 16
    alpha = 3.0
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32) * 2
    out = np.zeros_like(x)
    _vec_add(alpha * x, y, out, n)
    assert np.allclose(out, alpha * x + y)


def test_grid2_add_cpu():
    _ti_cpu()
    n, m = 4, 5
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    b = np.ones_like(a) * 3
    out = np.zeros_like(a)
    _vec_add_grid2(a, b, out, n, m)
    assert np.allclose(out, a + b)


def test_grid_stride_double_cpu():
    _ti_cpu()
    n = 17
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_double(a, out, n)
    assert np.allclose(out, a * 2)


def test_grid_stride_double_y_cpu():
    _ti_cpu()
    n = 9
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_double_y(a, out, n)
    assert np.allclose(out, a * 2)


def test_grid_stride_double_z_cpu():
    _ti_cpu()
    n = 5
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_double_z(a, out, n)
    assert np.allclose(out, a * 2)


def test_grid_stride_direct_cpu():
    _ti_cpu()
    n = 11
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_direct(a, out, n)
    assert np.allclose(out, a * 2)


def test_grid_stride_gridsize_var_cpu():
    _ti_cpu()
    n = 13
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_gridsize_var(a, out, n)
    assert np.allclose(out, a + 1)


def test_grid_stride_gridsize_prefix_cpu():
    _ti_cpu()
    n = 15
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_gridsize_prefix(a, out, n)
    assert np.allclose(out, a + 2)


def test_grid_stride_gridsize_scaled_cpu():
    _ti_cpu()
    n = 17
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_gridsize_scaled(a, out, n)
    assert np.allclose(out, a + 3)


def test_grid_stride_block_scaled_cpu():
    _ti_cpu()
    n = 19
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_block_scaled(a, out, n)
    assert np.allclose(out, a + 4)


def test_grid_stride_gridsize_2d_cpu():
    _ti_cpu()
    n, m = 6, 5
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(a)
    _grid_stride_gridsize_2d(a, out, n, m)
    assert np.allclose(out, a + 5)


def test_grid_stride_gridsize_2d_scaled_cpu():
    _ti_cpu()
    n, m = 6, 5
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(a)
    _grid_stride_gridsize_2d_scaled(a, out, n, m)
    assert np.allclose(out, a + 6)


def test_grid_stride_gridsize_3d_cpu():
    _ti_cpu()
    n0, n1, n2 = 4, 3, 5
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    out = np.zeros_like(a)
    _grid_stride_gridsize_3d(a, out, n0, n1, n2)
    assert np.allclose(out, a + 6)


def test_grid_stride_gridsize_3d_scaled_cpu():
    _ti_cpu()
    n0, n1, n2 = 4, 3, 5
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    out = np.zeros_like(a)
    _grid_stride_gridsize_3d_scaled(a, out, n0, n1, n2)
    assert np.allclose(out, a + 7)


def test_grid_stride_gridsize_expr_cpu():
    _ti_cpu()
    n = 21
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_gridsize_expr(a, out, n)
    assert np.allclose(out, a + 7)


def test_grid_stride_gridsize_add_cpu():
    _ti_cpu()
    n = 23
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_gridsize_add(a, out, n)
    assert np.allclose(out, a + 8)


def test_grid_stride_gridsize_mul_identity_cpu():
    _ti_cpu()
    n = 25
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _grid_stride_gridsize_mul_identity(a, out, n)
    assert np.allclose(out, a + 9)


def test_grid_stride_2d_cpu():
    _ti_cpu()
    n, m = 7, 5
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(a)
    _grid_stride_2d(a, out, n, m)
    assert np.allclose(out, a * 2)


def test_grid_stride_3d_cpu():
    _ti_cpu()
    n0, n1, n2 = 4, 3, 5
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    out = np.zeros_like(a)
    _grid_stride_3d(a, out, n0, n1, n2)
    assert np.allclose(out, a + 1.0)


def test_grid_stride_2d_continue_cpu():
    _ti_cpu()
    n, m = 8, 6
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(a)
    _grid_stride_2d_continue(a, out, n, m)
    ref = np.zeros_like(a)
    for i in range(n):
        for j in range(m):
            if (i + j) % 2 != 0:
                continue
            ref[i, j] = a[i, j] * 2
    assert np.allclose(out, ref)


def test_grid_stride_3d_continue_cpu():
    _ti_cpu()
    n0, n1, n2 = 5, 4, 3
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    out = np.zeros_like(a)
    _grid_stride_3d_continue(a, out, n0, n1, n2)
    ref = np.zeros_like(a)
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                if (i + j + k) % 3 != 0:
                    continue
                ref[i, j, k] = a[i, j, k] + 1.0
    assert np.allclose(out, ref)


def test_math_funcs_cpu():
    _ti_cpu()
    n = 6
    a = np.linspace(0.1, 0.6, n, dtype=np.float32)
    out = np.zeros_like(a)
    _math_ops(a, out, n)
    expected = np.sin(a) + np.cos(a) + np.sqrt(np.abs(a)) + np.exp(a) - np.log(a + 1.0)
    assert np.allclose(out, expected, atol=1e-6)


def test_casts_cpu():
    _ti_cpu()
    n = 8
    a = np.linspace(-1.5, 3.4, n, dtype=np.float32)
    out_int = np.zeros(n, dtype=np.int32)
    out_float = np.zeros(n, dtype=np.float32)
    _int_cast(a, out_int, n)
    _float_cast(out_int, out_float, n)
    assert np.array_equal(out_int, a.astype(np.int32))
    assert np.allclose(out_float, out_int.astype(np.float32))


def test_while_and_break_continue_cpu():
    _ti_cpu()
    n = 10
    a = np.linspace(-3.0, 6.0, n, dtype=np.float32)
    out_sum = np.zeros(1, dtype=np.float32)
    _while_sum(a, out_sum, n)
    assert np.allclose(out_sum[0], np.sum(a))

    out_break = np.zeros(1, dtype=np.float32)
    _break_first_positive(a, out_break, n)
    expected_first = a[a > 0][0]
    assert np.allclose(out_break[0], expected_first)

    out_continue = np.zeros_like(a)
    _continue_even(a, out_continue, n)
    ref = np.zeros_like(a)
    ref[::2] = a[::2] * 2
    assert np.allclose(out_continue, ref)


def test_ternary_expr_cpu():
    _ti_cpu()
    n = 7
    a = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    out = np.zeros_like(a)
    _ternary_sign(a, out, n)
    expected = np.where(a >= 0, 1.0, -1.0)
    assert np.allclose(out, expected)


def test_inline_helper_cpu():
    _ti_cpu()
    n = 8
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _inline_helper_kernel(a, out, n)
    expected = a * a + 1.0
    assert np.allclose(out, expected)


def test_inline_helper_allows_complex_fn():
    _ti_cpu()

    def bad(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = _helper_with_side_effect(a[i])

    kernel = ti.njit(bad)
    n = 5
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    kernel(a, out, n)
    assert np.allclose(out, a + 1.0)


def test_signature_overrides_dtype_ndim():
    _ti_cpu()
    n = 12
    a = np.arange(n, dtype=np.float64)
    b = np.arange(n, dtype=np.float64) * 2
    out = np.zeros(n, dtype=np.float64)
    _sig_vec_add(a, b, out, n)
    assert np.allclose(out, a + b)


def test_stencil2d_cpu():
    _ti_cpu()
    n, m = 6, 5
    inp = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(inp)
    _stencil2d(inp, out)
    ref = np.zeros_like(inp)
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            ref[i, j] = (
                inp[i, j] * 4 - inp[i - 1, j] - inp[i + 1, j] - inp[i, j - 1] - inp[i, j + 1]
            )
    assert np.allclose(out, ref)


def test_matmul_cpu():
    _ti_cpu()
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    out = np.zeros((2, 2), dtype=np.float32)
    _matmul(a, b, out)
    assert np.allclose(out, a @ b)


def test_masked_relu_cpu():
    _ti_cpu()
    n = 10
    a = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    mask = (np.arange(n) % 2).astype(np.int32)
    out = np.zeros_like(a)
    _masked_relu(a, mask, out, n)
    expected = np.where((mask != 0) & (a >= 0), a, 0.0)
    assert np.allclose(out, expected)


def test_grid3_add_cpu():
    _ti_cpu()
    n0, n1, n2 = 3, 2, 4
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    b = np.ones_like(a) * 2
    out = np.zeros_like(a)
    _grid3_add(a, b, out, n0, n1, n2)
    assert np.allclose(out, a + b)


def test_laplacian3d_cpu():
    _ti_cpu()
    n0, n1, n2 = 5, 4, 6
    inp = np.random.rand(n0, n1, n2).astype(np.float32)
    out = np.zeros_like(inp)
    _laplacian3d(inp, out, n0, n1, n2)
    ref = np.zeros_like(inp)
    for i in range(1, n0 - 1):
        for j in range(1, n1 - 1):
            for k in range(1, n2 - 1):
                ref[i, j, k] = (
                    -6.0 * inp[i, j, k]
                    + inp[i - 1, j, k]
                    + inp[i + 1, j, k]
                    + inp[i, j - 1, k]
                    + inp[i, j + 1, k]
                    + inp[i, j, k - 1]
                    + inp[i, j, k + 1]
                )
    assert np.allclose(out, ref)


def test_grid_stride_masked_cpu():
    _ti_cpu()
    n = 64
    a = np.arange(n, dtype=np.float32)
    mask = (np.arange(n) % 3 == 0).astype(np.int32)
    out = np.zeros_like(a)
    _grid_stride_masked(a, mask, out, n)
    ref = np.zeros_like(a)
    ref[mask.astype(bool)] = a[mask.astype(bool)] * 3
    assert np.allclose(out, ref)


def test_grid_stride_continue_cpu():
    _ti_cpu()
    n = 51
    a = np.arange(n, dtype=np.float32)
    mask = (np.arange(n) % 4 == 0).astype(np.int32)
    out = np.zeros_like(a)
    _grid_stride_continue(a, mask, out, n)
    ref = np.zeros_like(a)
    ref[mask.astype(bool)] = a[mask.astype(bool)] * 5
    assert np.allclose(out, ref)


def test_sparse_write_cpu():
    ti.reset()
    ti.init(arch=ti.cpu)
    global _sparse_field
    _sparse_field = ti.field(dtype=ti.f32)
    pointer = ti.root.pointer(ti.ijk, 2)
    pointer.dense(ti.ijk, 4).place(_sparse_field)

    @ti.njit(sig={"indices": (ti.i32, 2), "values": (ti.f32, 1), "mask": (ti.i32, 1)})
    def write_sparse(indices, values, mask, n):
        i = cuda.grid(1)
        if i < n:
            x = indices[i, 0]
            y = indices[i, 1]
            z = indices[i, 2]
            if mask[i] > 0:
                _sparse_field[x, y, z] += values[i]

    n = 6
    indices = np.array(
        [[0, 0, 0], [1, 1, 1], [1, 0, 2], [0, 1, 3], [1, 1, 2], [0, 0, 1]], dtype=np.int32
    )
    values = np.arange(n, dtype=np.float32)
    mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)
    write_sparse(indices, values, mask, n)
    expected = np.zeros((8, 8, 8), dtype=np.float32)
    for idx, val, m in zip(indices, values, mask):
        if m > 0:
            expected[tuple(idx)] += val
    for idx, val in np.ndenumerate(expected):
        if val != 0:
            assert np.isclose(_sparse_field[idx], val)


def test_cuda_intrinsic_diagnosed():
    _ti_cpu()

    def bad_kernel(a, out):
        i = cuda.grid(1)
        out[i] = cuda.shfl_sync(0xFFFFFFFF, a[i], 0, 16)

    with pytest.raises(FrontendError):
        ti.njit(bad_kernel)(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32))


def test_cuda_grid_dim_mismatch_diagnosed():
    _ti_cpu()

    def bad_kernel():
        i = cuda.grid(2)

    with pytest.raises(FrontendError):
        ti.njit(bad_kernel)()


def test_cuda_grid_dim_non_const_diagnosed():
    _ti_cpu()

    def bad_kernel(dim):
        i = cuda.grid(dim)

    with pytest.raises(FrontendError):
        ti.njit(bad_kernel)(1)


def test_syncthreads_args_diagnosed():
    _ti_cpu()

    def bad_sync():
        cuda.syncthreads(1)

    with pytest.raises(FrontendError):
        ti.njit(bad_sync)()


def test_shfl_missing_offset_diagnosed():
    _ti_cpu()

    def bad_shfl(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_sync(0xFFFFFFFF, a[i])

    with pytest.raises(FrontendError):
        ti.njit(bad_shfl)(
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1,
        )


def test_shared_array_requires_dtype():
    _ti_cpu()

    def bad_shared():
        buf = cuda.shared.array(32)

    with pytest.raises(FrontendError):
        ti.njit(bad_shared)()


def test_for_else_supported():
    _ti_cpu()

    def bad_for(out):
        for i in range(3):
            out[i] = i
        else:
            out[0] = 1

    kernel = ti.njit(bad_for)
    out = np.zeros(3, dtype=np.int32)
    kernel(out)
    assert np.allclose(out, np.array([1, 1, 2], dtype=np.int32))


def test_while_else_supported():
    _ti_cpu()

    def bad_while(out):
        i = 0
        while i < 3:
            i += 1
        else:
            out[0] = i

    kernel = ti.njit(bad_while)
    out = np.zeros(1, dtype=np.int32)
    kernel(out)
    assert out[0] == 3


def test_syncthreads_count_requires_predicate():
    _ti_cpu()

    def bad_count():
        cuda.syncthreads_count()

    with pytest.raises(FrontendError):
        ti.njit(bad_count)()


def test_syncthreads_and_requires_predicate():
    _ti_cpu()

    def bad_and():
        cuda.syncthreads_and()

    with pytest.raises(FrontendError):
        ti.njit(bad_and)()


def test_syncthreads_or_requires_predicate():
    _ti_cpu()

    def bad_or():
        cuda.syncthreads_or()

    with pytest.raises(FrontendError):
        ti.njit(bad_or)()


def test_gridsize_used_outside_step():
    _ti_cpu()

    def bad_gridsize(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.gridsize(1)

    with pytest.raises(FrontendError):
        ti.njit(bad_gridsize)(
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1,
        )


def test_bitshift_requires_int_operand():
    _ti_cpu()

    def bad_shift(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] << 1

    with pytest.raises(FrontendError):
        ti.njit(bad_shift, sig={"a": (ti.f32, 1), "out": (ti.f32, 1)})(
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1,
        )


def test_bitshift_requires_int_offset():
    _ti_cpu()

    def bad_shift(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] << 1.5

    with pytest.raises(FrontendError):
        ti.njit(bad_shift, sig={"a": (ti.i32, 1), "out": (ti.i32, 1)})(
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            1,
        )


def test_bitwise_requires_int_operand():
    _ti_cpu()

    def bad_and(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] & 1

    with pytest.raises(FrontendError):
        ti.njit(bad_and, sig={"a": (ti.f32, 1), "out": (ti.f32, 1)})(
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1,
        )


def test_shfl_requires_i32_or_f32():
    _ti_cpu()

    def bad_shfl(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_sync(0xFFFFFFFF, a[i], 1)

    with pytest.raises(FrontendError):
        ti.njit(bad_shfl, sig={"a": (ti.i64, 1), "out": (ti.i64, 1)})(
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            1,
        )


def test_mod_mask_cpu():
    _ti_cpu()
    n = 17
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _mod_mask(a, out, n)
    ref = np.zeros_like(a)
    ref[::2] = a[::2]
    assert np.allclose(out, ref)


def test_bitand_mask_cpu():
    _ti_cpu()
    n = 18
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    _bitand_mask(a, out, n)
    ref = np.zeros_like(a)
    ref[::2] = a[::2]
    assert np.allclose(out, ref)


def test_bitwise_ops_cpu():
    _ti_cpu()
    n = 8
    a = np.arange(n, dtype=np.int32)
    out = np.zeros((n, 4), dtype=np.int32)
    _bitwise_ops(a, out, n)
    expected = np.zeros_like(out)
    expected[:, 0] = a | 1
    expected[:, 1] = a ^ 3
    expected[:, 2] = a << 1
    expected[:, 3] = a >> 1
    assert np.allclose(out, expected)


def test_atomic_exch_diagnosed():
    _ti_cpu()

    def bad_exch(out):
        cuda.atomic.exch(out, 0, 1)

    with pytest.raises(FrontendError):
        ti.njit(bad_exch)(np.zeros(1, dtype=np.int32))


def test_ballot_mask_diagnosed():
    _ti_cpu()

    def bad_ballot(out):
        i = cuda.grid(1)
        out[i] = cuda.ballot_sync(0x0, i < 1)

    with pytest.raises(FrontendError):
        ti.njit(bad_ballot)(np.zeros(1, dtype=np.int32))


def test_block_local_hint_cpu():
    _ti_cpu()
    n0, n1 = 6, 6

    @ti.njit(block_local=["inp"])
    def stencil_bl(inp, out, n0, n1):
        i, j = cuda.grid(2)
        if 0 < i < n0 - 1 and 0 < j < n1 - 1:
            out[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )

    inp = np.arange(n0 * n1, dtype=np.float32).reshape(n0, n1)
    out = np.zeros_like(inp)
    stencil_bl(inp, out, n0, n1)
    expected = np.zeros_like(inp)
    for i in range(1, n0 - 1):
        for j in range(1, n1 - 1):
            expected[i, j] = (
                inp[i, j] * 4 - inp[i - 1, j] - inp[i + 1, j] - inp[i, j - 1] - inp[i, j + 1]
            )
    assert np.allclose(out, expected)


def test_masked_scatter_cpu():
    _ti_cpu()
    n = 12
    data = np.arange(n, dtype=np.float32)
    dst = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    mask = (np.arange(n) % 2).astype(np.int32)
    out = np.zeros(3, dtype=np.float32)
    _masked_scatter(data, dst, mask, out, n)
    expected = np.zeros_like(out)
    for i in range(n):
        if mask[i]:
            expected[dst[i]] += data[i]
    assert np.allclose(out, expected)


def test_mask2d_cpu():
    _ti_cpu()
    n0, n1 = 4, 5
    inp = np.arange(n0 * n1, dtype=np.float32).reshape(n0, n1)
    mask = ((np.arange(n0)[:, None] + np.arange(n1)) % 2 == 0).astype(np.int32)
    out = np.zeros_like(inp)
    _mask2d(inp, mask, out, n0, n1)
    expected = np.where(mask.astype(bool), inp, 0.0)
    assert np.allclose(out, expected)


def test_reduce3d_cpu():
    _ti_cpu()
    n0, n1, n2 = 3, 4, 5
    inp = np.random.rand(n0, n1, n2).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    _reduce3d(inp, out, n0, n1, n2)
    assert np.allclose(out[0], np.sum(inp), atol=1e-5)


def test_atomic_min_max_cpu():
    _ti_cpu()
    n = 10
    a = np.arange(-3, n - 3, dtype=np.int32)
    out_min = np.array([999], dtype=np.int32)
    out_max = np.array([-999], dtype=np.int32)
    _atomic_min_kernel(a, out_min, n)
    _atomic_max_kernel(a, out_max, n)
    assert out_min[0] == np.min(a)
    assert out_max[0] == np.max(a)


def test_atomic_sub_cpu():
    _ti_cpu()
    a = np.array([7], dtype=np.int32)
    out = np.array([7], dtype=np.int32)
    _atomic_sub_kernel(a, out, 1)
    assert out[0] == 0


def test_atomic_and_or_xor_cpu():
    _ti_cpu()
    a = np.array([0xF0, 0x0F, 0xAA], dtype=np.int32)
    out = np.array([0xFF], dtype=np.int32)
    _atomic_and_kernel(a, out, a.shape[0])
    assert out[0] == (0xF0 & 0x0F & 0xAA)

    out = np.array([0x00], dtype=np.int32)
    _atomic_or_kernel(a, out, a.shape[0])
    assert out[0] == (0xF0 | 0x0F | 0xAA)

    out = np.array([0x00], dtype=np.int32)
    _atomic_xor_kernel(a, out, a.shape[0])
    assert out[0] == (0xF0 ^ 0x0F ^ 0xAA)


def test_heat_step_cpu():
    _ti_cpu()
    n0, n1 = 8, 8
    steps = 2
    u_init = np.random.rand(n0, n1).astype(np.float32)
    u = u_init.copy()
    u_new = np.zeros_like(u_init)
    for _ in range(steps):
        u_new[:, :] = u
        _heat_step(u, u_new, n0, n1)
    ref = u_init.copy()
    buf = ref.copy()
    for _ in range(steps):
        for i in range(1, n0 - 1):
            for j in range(1, n1 - 1):
                buf[i, j] = 0.25 * (ref[i - 1, j] + ref[i + 1, j] + ref[i, j - 1] + ref[i, j + 1])
        ref = buf.copy()
    assert np.allclose(u, ref)


def test_point_scatter_cpu():
    _ti_cpu()
    n = 16
    grid_res = 8
    pos = np.random.randint(0, grid_res, size=(n, 2)).astype(np.int32)
    weight = np.random.rand(n).astype(np.float32)
    grid = np.zeros(grid_res * grid_res, dtype=np.float32)
    _point_scatter(pos, weight, grid, n, grid_res)
    expected = np.zeros_like(grid)
    for i in range(n):
        gx, gy = int(pos[i, 0]), int(pos[i, 1])
        expected[gx * grid_res + gy] += weight[i]
    assert np.allclose(grid, expected)


def test_ad_squared_sum_cpu():
    ti.reset()
    ti.init(arch=ti.cpu)
    global _ad_x, _ad_loss
    n = 8
    _ad_x = ti.field(dtype=ti.f32, needs_grad=True)
    _ad_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    ti.root.dense(ti.i, n).place(_ad_x, _ad_x.grad)

    @ti.njit
    def compute_loss(n):
        i = cuda.grid(1)
        if i < n:
            _ad_loss[None] += _ad_x[i] * _ad_x[i]

    data = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    _ad_x.from_numpy(data)
    _ad_loss[None] = 0.0
    compute_loss(n)
    expected_loss = np.sum(data * data)
    assert np.allclose(_ad_loss[None], expected_loss, atol=1e-6)
