import numpy as np
import pytest

import taichi as ti
from taichi.lang.numba_typed_frontend import is_available as _NUMBA_TF_AVAILABLE
from taichi.lang.numba_typed_frontend.diagnostics import FrontendError

try:
    from numba import cuda

    _CUDA_AVAILABLE = cuda.is_available()
    _CUDA_COMPAT = False
    if _CUDA_AVAILABLE:
        try:
            @cuda.jit
            def _probe_kernel(a):
                i = cuda.grid(1)
                if i < a.size:
                    a[i] += 1

            _probe = np.zeros(1, dtype=np.int32)
            _d_probe = cuda.to_device(_probe)
            _probe_kernel[1, 1](_d_probe)
            _d_probe.copy_to_host()
            _CUDA_COMPAT = True
        except Exception:  # pragma: no cover - env-specific
            _CUDA_COMPAT = False
    _CUDA_AVAILABLE = _CUDA_AVAILABLE and _CUDA_COMPAT
except Exception:  # pragma: no cover - fallback
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _NUMBA_TF_AVAILABLE(), reason="numba typed frontend not enabled")


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_vec_add_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_vec_add(a, b, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] + b[i]

    @cuda.jit
    def nb_vec_add(a, b, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i]

    n = 1 << 16
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 3
    out_ti = np.zeros_like(a)

    ti_vec_add(a, b, out_ti, n)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_vec_add[blocks, threads](d_a, d_b, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_double(a, out, n):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if i < n:
            out[i] = a[i] * 2

    @cuda.jit
    def nb_double(a, out):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if i < out.size:
            out[i] = a[i] * 2

    @ti.njit
    def ti_double_y(a, out, n):
        j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        if j < n:
            out[j] = a[j] * 2

    @cuda.jit
    def nb_double_y(a, out):
        j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        if j < out.size:
            out[j] = a[j] * 2

    n = 1 << 15
    a = np.arange(n, dtype=np.float32)
    out_ti = np.zeros_like(a)
    out_ti_y = np.zeros_like(a)

    ti_double(a, out_ti, n)
    ti_double_y(a, out_ti_y, n)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    d_out_y = cuda.device_array_like(a)
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_double[blocks, threads](d_a, d_out)
    nb_double_y[(1, blocks, 1), (1, threads, 1)](d_a, d_out_y)
    out_nb = d_out.copy_to_host()
    out_nb_y = d_out_y.copy_to_host()

    assert np.allclose(out_ti, out_nb)
    assert np.allclose(out_ti_y, out_nb_y)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_for_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_stride_for(a, out, n):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for i in range(i, n, stride):
            out[i] = a[i] * 3

    @cuda.jit
    def nb_stride_for(a, out):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for idx in range(i, out.size, stride):
            out[idx] = a[idx] * 3

    n = 1 << 15
    a = np.arange(n, dtype=np.float32)
    out_ti = np.zeros_like(a)

    ti_stride_for(a, out_ti, n)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_stride_for[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_continue_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"mask": (ti.i32, 1)})
    def ti_stride_continue(a, mask, out, n):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for i in range(i, n, stride):
            if mask[i] == 0:
                continue
            out[i] = a[i] * 5

    @cuda.jit
    def nb_stride_continue(a, mask, out):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for idx in range(i, out.size, stride):
            if mask[idx] == 0:
                continue
            out[idx] = a[idx] * 5

    n = 1 << 15
    a = np.arange(n, dtype=np.float32)
    mask = (np.arange(n) % 7 == 0).astype(np.int32)
    out_ti = np.zeros_like(a)

    ti_stride_continue(a, mask, out_ti, n)

    d_a = cuda.to_device(a)
    d_mask = cuda.to_device(mask)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_stride_continue[blocks, threads](d_a, d_mask, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_2d_for_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_stride_2d(a, out, n, m):
        i, j = cuda.grid(2)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        for i in range(i, n, stride_i):
            for j in range(j, m, stride_j):
                out[i, j] = a[i, j] * 2

    @cuda.jit
    def nb_stride_2d(a, out):
        i, j = cuda.grid(2)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        for ii in range(i, out.shape[0], stride_i):
            for jj in range(j, out.shape[1], stride_j):
                out[ii, jj] = a[ii, jj] * 2

    n, m = 64, 48
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out_ti = np.zeros_like(a)

    ti_stride_2d(a, out_ti, n, m)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = (16, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    nb_stride_2d[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_2d_continue_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_stride_2d_continue(a, out, n, m):
        i, j = cuda.grid(2)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        for i in range(i, n, stride_i):
            for j in range(j, m, stride_j):
                if (i + j) % 2 != 0:
                    continue
                out[i, j] = a[i, j] * 2

    @cuda.jit
    def nb_stride_2d_continue(a, out):
        i, j = cuda.grid(2)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        for ii in range(i, out.shape[0], stride_i):
            for jj in range(j, out.shape[1], stride_j):
                if (ii + jj) % 2 != 0:
                    continue
                out[ii, jj] = a[ii, jj] * 2

    n, m = 64, 48
    a = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out_ti = np.zeros_like(a)

    ti_stride_2d_continue(a, out_ti, n, m)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = (16, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    nb_stride_2d_continue[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_3d_for_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_stride_3d(a, out, n0, n1, n2):
        i, j, k = cuda.grid(3)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        stride_k = cuda.blockDim.z * cuda.gridDim.z
        for i in range(i, n0, stride_i):
            for j in range(j, n1, stride_j):
                for k in range(k, n2, stride_k):
                    out[i, j, k] = a[i, j, k] + 1.0

    @cuda.jit
    def nb_stride_3d(a, out):
        i, j, k = cuda.grid(3)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        stride_k = cuda.blockDim.z * cuda.gridDim.z
        for ii in range(i, out.shape[0], stride_i):
            for jj in range(j, out.shape[1], stride_j):
                for kk in range(k, out.shape[2], stride_k):
                    out[ii, jj, kk] = a[ii, jj, kk] + 1.0

    n0, n1, n2 = 16, 12, 8
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    out_ti = np.zeros_like(a)

    ti_stride_3d(a, out_ti, n0, n1, n2)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = (8, 4, 2)
    blocks = (
        (n0 + threads[0] - 1) // threads[0],
        (n1 + threads[1] - 1) // threads[1],
        (n2 + threads[2] - 1) // threads[2],
    )
    nb_stride_3d[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_3d_continue_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_stride_3d_continue(a, out, n0, n1, n2):
        i, j, k = cuda.grid(3)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        stride_k = cuda.blockDim.z * cuda.gridDim.z
        for i in range(i, n0, stride_i):
            for j in range(j, n1, stride_j):
                for k in range(k, n2, stride_k):
                    if (i + j + k) % 3 != 0:
                        continue
                    out[i, j, k] = a[i, j, k] + 1.0

    @cuda.jit
    def nb_stride_3d_continue(a, out):
        i, j, k = cuda.grid(3)
        stride_i = cuda.blockDim.x * cuda.gridDim.x
        stride_j = cuda.blockDim.y * cuda.gridDim.y
        stride_k = cuda.blockDim.z * cuda.gridDim.z
        for ii in range(i, out.shape[0], stride_i):
            for jj in range(j, out.shape[1], stride_j):
                for kk in range(k, out.shape[2], stride_k):
                    if (ii + jj + kk) % 3 != 0:
                        continue
                    out[ii, jj, kk] = a[ii, jj, kk] + 1.0

    n0, n1, n2 = 16, 12, 8
    a = np.arange(n0 * n1 * n2, dtype=np.float32).reshape(n0, n1, n2)
    out_ti = np.zeros_like(a)

    ti_stride_3d_continue(a, out_ti, n0, n1, n2)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = (8, 4, 2)
    blocks = (
        (n0 + threads[0] - 1) // threads[0],
        (n1 + threads[1] - 1) // threads[1],
        (n2 + threads[2] - 1) // threads[2],
    )
    nb_stride_3d_continue[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()

    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_stencil2d_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_stencil(inp, out, n, m):
        i, j = cuda.grid(2)
        if 0 < i < n - 1 and 0 < j < m - 1:
            out[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )

    @cuda.jit
    def nb_stencil(inp, out):
        i, j = cuda.grid(2)
        if 0 < i < out.shape[0] - 1 and 0 < j < out.shape[1] - 1:
            out[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )

    n, m = 32, 32
    inp = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out_ti = np.zeros_like(inp)
    ti_stencil(inp, out_ti, n, m)

    d_in = cuda.to_device(inp)
    d_out = cuda.to_device(np.zeros_like(inp))
    threads = (16, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    nb_stencil[blocks, threads](d_in, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_matmul_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_matmul(a, b, out, n, k, m):
        i, j = cuda.grid(2)
        if i < n and j < m:
            acc = 0.0
            for t in range(k):
                acc += a[i, t] * b[t, j]
            out[i, j] = acc

    @cuda.jit
    def nb_matmul(a, b, out):
        i, j = cuda.grid(2)
        if i < out.shape[0] and j < out.shape[1]:
            acc = 0.0
            for t in range(a.shape[1]):
                acc += a[i, t] * b[t, j]
            out[i, j] = acc

    n, k, m = 32, 16, 24
    a = np.arange(n * k, dtype=np.float32).reshape(n, k)
    b = np.arange(k * m, dtype=np.float32).reshape(k, m)
    out_ti = np.zeros((n, m), dtype=np.float32)
    ti_matmul(a, b, out_ti, n, k, m)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(out_ti)
    threads = (16, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    nb_matmul[blocks, threads](d_a, d_b, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-4)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_axpy_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": ti.f32})
    def ti_axpy(a, x, y, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a * x[i] + y[i]

    @cuda.jit
    def nb_axpy(a, x, y, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a * x[i] + y[i]

    n = 1024 * 1024
    alpha = 2.5
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    out_ti = np.zeros_like(x)

    ti_axpy(alpha, x, y, out_ti, n)

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array_like(x)
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_axpy[blocks, threads](alpha, d_x, d_y, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-6)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_conv3x3_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_conv(inp, filt, out, n, m):
        i, j = cuda.grid(2)
        if 1 <= i < n - 1 and 1 <= j < m - 1:
            acc = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    acc += inp[i + di, j + dj] * filt[di + 1, dj + 1]
            out[i, j] = acc

    @cuda.jit
    def nb_conv(inp, filt, out):
        i, j = cuda.grid(2)
        if 1 <= i < out.shape[0] - 1 and 1 <= j < out.shape[1] - 1:
            acc = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    acc += inp[i + di, j + dj] * filt[di + 1, dj + 1]
            out[i, j] = acc

    n, m = 64, 64
    inp = np.random.rand(n, m).astype(np.float32)
    filt = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    out_ti = np.zeros_like(inp)

    ti_conv(inp, filt, out_ti, n, m)

    d_in = cuda.to_device(inp)
    d_f = cuda.to_device(filt)
    d_out = cuda.to_device(np.zeros_like(inp))
    threads = (16, 8)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    nb_conv[blocks, threads](d_in, d_f, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_row_sum_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_row_sum(a, out, n, m):
        i = cuda.grid(1)
        if i < n:
            acc = 0.0
            for j in range(m):
                acc += a[i, j]
            out[i] = acc

    @cuda.jit
    def nb_row_sum(a, out):
        i = cuda.grid(1)
        if i < out.size:
            acc = 0.0
            for j in range(a.shape[1]):
                acc += a[i, j]
            out[i] = acc

    n, m = 256, 128
    a = np.random.rand(n, m).astype(np.float32)
    out_ti = np.zeros(n, dtype=np.float32)

    ti_row_sum(a, out_ti, n, m)

    d_a = cuda.to_device(a)
    d_out = cuda.device_array_like(out_ti)
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_row_sum[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_hist_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"img": (ti.i32, 1), "out": (ti.i32, 1)})
    def ti_hist(img, out, n):
        idx = cuda.grid(1)
        if idx < n:
            v = img[idx]
            cuda.atomic.add(out, v, 1)

    @cuda.jit
    def nb_hist(img, out):
        idx = cuda.grid(1)
        if idx < img.size:
            v = img[idx]
            cuda.atomic.add(out, v, 1)

    n = 1 << 16
    bins = 32
    img = np.random.randint(0, bins, size=n).astype(np.int32)
    out_ti = np.zeros(bins, dtype=np.int32)

    ti_hist(img, out_ti, n)

    d_img = cuda.to_device(img)
    d_out = cuda.to_device(np.zeros_like(out_ti))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_hist[blocks, threads](d_img, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_atomic_min_max_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out_min": (ti.i32, 1), "out_max": (ti.i32, 1)})
    def ti_atomic_min_max(a, out_min, out_max, n):
        i = cuda.grid(1)
        if i < n:
            cuda.atomic.min(out_min, 0, a[i])
            cuda.atomic.max(out_max, 0, a[i])

    @cuda.jit
    def nb_atomic_min_max(a, out_min, out_max):
        i = cuda.grid(1)
        if i < a.size:
            cuda.atomic.min(out_min, 0, a[i])
            cuda.atomic.max(out_max, 0, a[i])

    n = 1 << 12
    a = np.random.randint(-5000, 5000, size=n).astype(np.int32)
    out_min_ti = np.array([2**30], dtype=np.int32)
    out_max_ti = np.array([-(2**30)], dtype=np.int32)

    ti_atomic_min_max(a, out_min_ti, out_max_ti, n)

    d_a = cuda.to_device(a)
    d_min = cuda.to_device(np.array([2**30], dtype=np.int32))
    d_max = cuda.to_device(np.array([-(2**30)], dtype=np.int32))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_atomic_min_max[blocks, threads](d_a, d_min, d_max)
    out_min_nb = d_min.copy_to_host()
    out_max_nb = d_max.copy_to_host()

    assert out_min_ti[0] == out_min_nb[0] == np.min(a)
    assert out_max_ti[0] == out_max_nb[0] == np.max(a)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_scatter_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"dst_idx": (ti.i32, 1), "weight": (ti.f32, 1), "out": (ti.f32, 1)})
    def ti_scatter(dst_idx, weight, out, m):
        e = cuda.grid(1)
        if e < m:
            cuda.atomic.add(out, dst_idx[e], weight[e])

    @cuda.jit
    def nb_scatter(dst_idx, weight, out):
        e = cuda.grid(1)
        if e < dst_idx.size:
            cuda.atomic.add(out, dst_idx[e], weight[e])

    m = 1 << 16
    bins = 128
    dst_idx = np.random.randint(0, bins, size=m).astype(np.int32)
    weight = np.random.rand(m).astype(np.float32)
    out_ti = np.zeros(bins, dtype=np.float32)

    ti_scatter(dst_idx, weight, out_ti, m)

    d_dst = cuda.to_device(dst_idx)
    d_w = cuda.to_device(weight)
    d_out = cuda.to_device(np.zeros_like(out_ti))
    threads = 256
    blocks = (m + threads - 1) // threads
    nb_scatter[blocks, threads](d_dst, d_w, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_vec3_cross_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_cross(a, b, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i, 0] = a[i, 1] * b[i, 2] - a[i, 2] * b[i, 1]
            out[i, 1] = a[i, 2] * b[i, 0] - a[i, 0] * b[i, 2]
            out[i, 2] = a[i, 0] * b[i, 1] - a[i, 1] * b[i, 0]

    @cuda.jit
    def nb_cross(a, b, out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i, 0] = a[i, 1] * b[i, 2] - a[i, 2] * b[i, 1]
            out[i, 1] = a[i, 2] * b[i, 0] - a[i, 0] * b[i, 2]
            out[i, 2] = a[i, 0] * b[i, 1] - a[i, 1] * b[i, 0]

    n = 1 << 12
    a = np.random.rand(n, 3).astype(np.float32)
    b = np.random.rand(n, 3).astype(np.float32)
    out_ti = np.zeros_like(a)

    ti_cross(a, b, out_ti, n)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_cross[blocks, threads](d_a, d_b, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_jacobi_step_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_jacobi(u, unew, nx, ny):
        i, j = cuda.grid(2)
        if 1 <= i < nx - 1 and 1 <= j < ny - 1:
            unew[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

    @cuda.jit
    def nb_jacobi(u, unew):
        i, j = cuda.grid(2)
        if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
            unew[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

    nx, ny = 64, 64
    u = np.random.rand(nx, ny).astype(np.float32)
    unew_ti = np.zeros_like(u)

    ti_jacobi(u, unew_ti, nx, ny)

    d_u = cuda.to_device(u)
    d_unew = cuda.to_device(np.zeros_like(u))
    threads = (16, 8)
    blocks = ((nx + threads[0] - 1) // threads[0], (ny + threads[1] - 1) // threads[1])
    nb_jacobi[blocks, threads](d_u, d_unew)
    unew_nb = d_unew.copy_to_host()
    assert np.allclose(unew_ti, unew_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grayscott_reaction_diffusion_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_grayscott(u, v, u_new, v_new, n, m):
        i, j = cuda.grid(2)
        if 1 <= i < n - 1 and 1 <= j < m - 1:
            u_ij = u[i, j]
            v_ij = v[i, j]
            lap_u = u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4.0 * u_ij
            lap_v = v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4.0 * v_ij
            uvv = u_ij * v_ij * v_ij
            u_new[i, j] = u_ij + (0.16 * lap_u - uvv + 0.035 * (1.0 - u_ij))
            v_new[i, j] = v_ij + (0.08 * lap_v + uvv - (0.035 + 0.065) * v_ij)

    @cuda.jit
    def nb_grayscott(u, v, u_new, v_new, n, m):
        i, j = cuda.grid(2)
        if 1 <= i < n - 1 and 1 <= j < m - 1:
            u_ij = u[i, j]
            v_ij = v[i, j]
            lap_u = u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4.0 * u_ij
            lap_v = v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4.0 * v_ij
            uvv = u_ij * v_ij * v_ij
            u_new[i, j] = u_ij + (0.16 * lap_u - uvv + 0.035 * (1.0 - u_ij))
            v_new[i, j] = v_ij + (0.08 * lap_v + uvv - (0.035 + 0.065) * v_ij)

    n, m = 64, 64
    steps = 5
    u0 = np.ones((n, m), dtype=np.float32)
    v0 = np.zeros_like(u0)
    r = 6
    cx, cy = n // 2, m // 2
    u0[cx - r : cx + r, cy - r : cy + r] = 0.0
    v0[cx - r : cx + r, cy - r : cy + r] = 1.0

    u = u0.copy()
    v = v0.copy()
    u_new = u0.copy()
    v_new = v0.copy()
    for _ in range(steps):
        ti_grayscott(u, v, u_new, v_new, n, m)
        u, u_new = u_new, u
        v, v_new = v_new, v

    d_u = cuda.to_device(u0)
    d_v = cuda.to_device(v0)
    d_u_new = cuda.to_device(u0.copy())
    d_v_new = cuda.to_device(v0.copy())
    threads = (16, 16)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    for _ in range(steps):
        nb_grayscott[blocks, threads](d_u, d_v, d_u_new, d_v_new, n, m)
        d_u, d_u_new = d_u_new, d_u
        d_v, d_v_new = d_v_new, d_v
    u_nb = d_u.copy_to_host()
    v_nb = d_v.copy_to_host()
    assert np.allclose(u, u_nb, atol=1e-4)
    assert np.allclose(v, v_nb, atol=1e-4)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_fdtd2d_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_fdtd_h(ez, hx, hy, n, m):
        i, j = cuda.grid(2)
        if i < n - 1 and j < m - 1:
            hx[i, j] -= 0.5 * (ez[i, j + 1] - ez[i, j])
            hy[i, j] += 0.5 * (ez[i + 1, j] - ez[i, j])

    @ti.njit
    def ti_fdtd_e(ez, hx, hy, n, m):
        i, j = cuda.grid(2)
        if 1 <= i < n - 1 and 1 <= j < m - 1:
            ez[i, j] += 0.5 * ((hy[i, j] - hy[i - 1, j]) - (hx[i, j] - hx[i, j - 1]))

    @cuda.jit
    def nb_fdtd_h(ez, hx, hy, n, m):
        i, j = cuda.grid(2)
        if i < n - 1 and j < m - 1:
            hx[i, j] -= 0.5 * (ez[i, j + 1] - ez[i, j])
            hy[i, j] += 0.5 * (ez[i + 1, j] - ez[i, j])

    @cuda.jit
    def nb_fdtd_e(ez, hx, hy, n, m):
        i, j = cuda.grid(2)
        if 1 <= i < n - 1 and 1 <= j < m - 1:
            ez[i, j] += 0.5 * ((hy[i, j] - hy[i - 1, j]) - (hx[i, j] - hx[i, j - 1]))

    n, m = 48, 48
    steps = 6
    ez0 = np.zeros((n, m), dtype=np.float32)
    ez0[n // 2, m // 2] = 1.0
    hx0 = np.zeros_like(ez0)
    hy0 = np.zeros_like(ez0)

    ez = ez0.copy()
    hx = hx0.copy()
    hy = hy0.copy()
    for _ in range(steps):
        ti_fdtd_h(ez, hx, hy, n, m)
        ti_fdtd_e(ez, hx, hy, n, m)

    d_ez = cuda.to_device(ez0)
    d_hx = cuda.to_device(hx0)
    d_hy = cuda.to_device(hy0)
    threads = (16, 16)
    blocks = ((n + threads[0] - 1) // threads[0], (m + threads[1] - 1) // threads[1])
    for _ in range(steps):
        nb_fdtd_h[blocks, threads](d_ez, d_hx, d_hy, n, m)
        nb_fdtd_e[blocks, threads](d_ez, d_hx, d_hy, n, m)
    ez_nb = d_ez.copy_to_host()
    hx_nb = d_hx.copy_to_host()
    hy_nb = d_hy.copy_to_host()
    assert np.allclose(ez, ez_nb, atol=1e-4)
    assert np.allclose(hx, hx_nb, atol=1e-4)
    assert np.allclose(hy, hy_nb, atol=1e-4)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_laplacian3d_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_laplacian(inp, out, n0, n1, n2):
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

    @cuda.jit
    def nb_laplacian(inp, out):
        i, j, k = cuda.grid(3)
        if 0 < i < out.shape[0] - 1 and 0 < j < out.shape[1] - 1 and 0 < k < out.shape[2] - 1:
            out[i, j, k] = (
                -6.0 * inp[i, j, k]
                + inp[i - 1, j, k]
                + inp[i + 1, j, k]
                + inp[i, j - 1, k]
                + inp[i, j + 1, k]
                + inp[i, j, k - 1]
                + inp[i, j, k + 1]
            )

    n0, n1, n2 = 16, 12, 10
    inp = np.random.rand(n0, n1, n2).astype(np.float32)
    out_ti = np.zeros_like(inp)

    ti_laplacian(inp, out_ti, n0, n1, n2)

    d_inp = cuda.to_device(inp)
    d_out = cuda.to_device(np.zeros_like(inp))
    threads = (4, 4, 2)
    blocks = (
        (n0 + threads[0] - 1) // threads[0],
        (n1 + threads[1] - 1) // threads[1],
        (n2 + threads[2] - 1) // threads[2],
    )
    nb_laplacian[blocks, threads](d_inp, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-4)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid_stride_masked_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"mask": (ti.i32, 1)})
    def ti_stride_masked(a, mask, out, n):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for i in range(i, n, stride):
            if mask[i]:
                out[i] = a[i] * 3

    @cuda.jit
    def nb_stride_masked(a, mask, out):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.blockDim.x * cuda.gridDim.x
        for idx in range(i, out.size, stride):
            if mask[idx]:
                out[idx] = a[idx] * 3

    n = 1 << 15
    a = np.random.rand(n).astype(np.float32)
    mask = (np.random.randint(0, 2, size=n)).astype(np.int32)
    out_ti = np.zeros_like(a)

    ti_stride_masked(a, mask, out_ti, n)

    d_a = cuda.to_device(a)
    d_mask = cuda.to_device(mask)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_stride_masked[blocks, threads](d_a, d_mask, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-6)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_mod_mask_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_mod_mask(a, out, n):
        i = cuda.grid(1)
        if i < n:
            if i % 2 == 0:
                out[i] = a[i % n]

    @cuda.jit
    def nb_mod_mask(a, out):
        i = cuda.grid(1)
        if i < out.size:
            if i % 2 == 0:
                out[i] = a[i % a.size]

    n = 1 << 12
    a = np.random.rand(n).astype(np.float32)
    out_ti = np.zeros_like(a)

    ti_mod_mask(a, out_ti, n)

    d_a = cuda.to_device(a)
    d_out = cuda.to_device(np.zeros_like(a))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_mod_mask[blocks, threads](d_a, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-6)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_masked_scatter_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"mask": (ti.i32, 1), "dst_idx": (ti.i32, 1)})
    def ti_masked_scatter(data, dst_idx, mask, out, n):
        i = cuda.grid(1)
        if i < n:
            if mask[i]:
                cuda.atomic.add(out, dst_idx[i], data[i])

    @cuda.jit
    def nb_masked_scatter(data, dst_idx, mask, out):
        i = cuda.grid(1)
        if i < data.size:
            if mask[i]:
                cuda.atomic.add(out, dst_idx[i], data[i])

    n = 1 << 14
    bins = 16
    data = np.random.rand(n).astype(np.float32)
    dst_idx = np.random.randint(0, bins, size=n).astype(np.int32)
    mask = (np.random.randint(0, 2, size=n)).astype(np.int32)
    out_ti = np.zeros(bins, dtype=np.float32)

    ti_masked_scatter(data, dst_idx, mask, out_ti, n)

    d_data = cuda.to_device(data)
    d_dst = cuda.to_device(dst_idx)
    d_mask = cuda.to_device(mask)
    d_out = cuda.to_device(np.zeros_like(out_ti))
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_masked_scatter[blocks, threads](d_data, d_dst, d_mask, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-4)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_mask2d_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"mask": (ti.i32, 2)})
    def ti_mask2d(inp, mask, out, n0, n1):
        i, j = cuda.grid(2)
        if i < n0 and j < n1:
            if mask[i, j]:
                out[i, j] = inp[i, j]
            else:
                out[i, j] = 0.0

    @cuda.jit
    def nb_mask2d(inp, mask, out):
        i, j = cuda.grid(2)
        if i < out.shape[0] and j < out.shape[1]:
            if mask[i, j]:
                out[i, j] = inp[i, j]
            else:
                out[i, j] = 0.0

    n0, n1 = 64, 48
    inp = np.random.rand(n0, n1).astype(np.float32)
    mask = (np.random.randint(0, 2, size=(n0, n1))).astype(np.int32)
    out_ti = np.zeros_like(inp)

    ti_mask2d(inp, mask, out_ti, n0, n1)

    d_inp = cuda.to_device(inp)
    d_mask = cuda.to_device(mask)
    d_out = cuda.device_array_like(inp)
    threads = (16, 8)
    blocks = ((n0 + threads[0] - 1) // threads[0], (n1 + threads[1] - 1) // threads[1])
    nb_mask2d[blocks, threads](d_inp, d_mask, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_reduce3d_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_reduce3d(inp, out, n0, n1, n2):
        i, j, k = cuda.grid(3)
        if i < n0 and j < n1 and k < n2:
            cuda.atomic.add(out, 0, inp[i, j, k])

    @cuda.jit
    def nb_reduce3d(inp, out):
        i, j, k = cuda.grid(3)
        if i < inp.shape[0] and j < inp.shape[1] and k < inp.shape[2]:
            cuda.atomic.add(out, 0, inp[i, j, k])

    n0, n1, n2 = 20, 12, 10
    inp = np.random.rand(n0, n1, n2).astype(np.float32)
    out_ti = np.zeros(1, dtype=np.float32)

    ti_reduce3d(inp, out_ti, n0, n1, n2)

    d_inp = cuda.to_device(inp)
    d_out = cuda.to_device(np.zeros_like(out_ti))
    threads = (4, 4, 2)
    blocks = (
        (n0 + threads[0] - 1) // threads[0],
        (n1 + threads[1] - 1) // threads[1],
        (n2 + threads[2] - 1) // threads[2],
    )
    nb_reduce3d[blocks, threads](d_inp, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-3)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_point_scatter_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"pos": (ti.i32, 2), "weight": (ti.f32, 1), "grid": (ti.f32, 1)})
    def ti_point_scatter(pos, weight, grid, n, grid_res):
        i = cuda.grid(1)
        if i < n:
            gx = pos[i, 0]
            gy = pos[i, 1]
            if 0 <= gx < grid_res and 0 <= gy < grid_res:
                cuda.atomic.add(grid, gx * grid_res + gy, weight[i])

    @cuda.jit
    def nb_point_scatter(pos, weight, grid, grid_res):
        i = cuda.grid(1)
        if i < pos.shape[0]:
            gx = pos[i, 0]
            gy = pos[i, 1]
            if 0 <= gx < grid_res and 0 <= gy < grid_res:
                cuda.atomic.add(grid, gx * grid_res + gy, weight[i])

    n = 1 << 12
    grid_res = 64
    pos = np.random.randint(0, grid_res, size=(n, 2)).astype(np.int32)
    weight = np.random.rand(n).astype(np.float32)
    grid_ti = np.zeros(grid_res * grid_res, dtype=np.float32)

    ti_point_scatter(pos, weight, grid_ti, n, grid_res)
    ti.sync()

    threads = 256
    blocks = (n + threads - 1) // threads
    d_pos = cuda.to_device(pos)
    d_weight = cuda.to_device(weight)
    d_grid = cuda.to_device(np.zeros_like(grid_ti))
    nb_point_scatter[blocks, threads](d_pos, d_weight, d_grid, grid_res)
    grid_nb = d_grid.copy_to_host()
    assert np.allclose(grid_ti, grid_nb, atol=1e-4)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_masked_relu_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"mask": (ti.i32, 1)})
    def ti_masked_relu(a, mask, out, n):
        i = cuda.grid(1)
        if i < n:
            if not mask[i] or a[i] < 0:
                out[i] = 0.0
            else:
                out[i] = a[i]

    @cuda.jit
    def nb_masked_relu(a, mask, out):
        i = cuda.grid(1)
        if i < out.size:
            if not mask[i] or a[i] < 0:
                out[i] = 0.0
            else:
                out[i] = a[i]

    n = 1 << 12
    a = (np.random.rand(n).astype(np.float32) - 0.5) * 2
    mask = (np.random.randint(0, 2, size=n)).astype(np.int32)
    out_ti = np.zeros_like(a)

    ti_masked_relu(a, mask, out_ti, n)

    d_a = cuda.to_device(a)
    d_mask = cuda.to_device(mask)
    d_out = cuda.device_array_like(a)
    threads = 256
    blocks = (n + threads - 1) // threads
    nb_masked_relu[blocks, threads](d_a, d_mask, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_grid3_add_numba_vs_taichi():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit
    def ti_grid3_add(a, b, out, n0, n1, n2):
        i, j, k = cuda.grid(3)
        if i < n0 and j < n1 and k < n2:
            out[i, j, k] = a[i, j, k] + b[i, j, k]

    @cuda.jit
    def nb_grid3_add(a, b, out):
        i, j, k = cuda.grid(3)
        if i < out.shape[0] and j < out.shape[1] and k < out.shape[2]:
            out[i, j, k] = a[i, j, k] + b[i, j, k]

    n0, n1, n2 = 5, 4, 6
    a = np.random.rand(n0, n1, n2).astype(np.float32)
    b = np.random.rand(n0, n1, n2).astype(np.float32)
    out_ti = np.zeros_like(a)

    ti_grid3_add(a, b, out_ti, n0, n1, n2)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(a)
    threads = (4, 2, 2)
    blocks = (
        (n0 + threads[0] - 1) // threads[0],
        (n1 + threads[1] - 1) // threads[1],
        (n2 + threads[2] - 1) // threads[2],
    )
    nb_grid3_add[blocks, threads](d_a, d_b, d_out)
    out_nb = d_out.copy_to_host()
    assert np.allclose(out_ti, out_nb, atol=1e-5)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shared_array_syncthreads():
    ti.reset()
    ti.init(arch=ti.cuda)
    block_dim = 32

    @ti.njit(block_dim=block_dim, sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shared_shift(a, out, n):
        i = cuda.grid(1)
        if i < n:
            buf = cuda.shared.array(32, dtype=np.int32)
            tid = i % 32
            buf[tid] = a[i]
            cuda.syncthreads()
            out[i] = buf[(tid + 1) % 32]

    n = block_dim * 2
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shared_shift(a, out, n)
    expected = np.roll(a.reshape(-1, block_dim), -1, axis=1).reshape(-1)
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shared_array_2d_keywords():
    ti.reset()
    ti.init(arch=ti.cuda)
    block_dim = 16

    @ti.njit(block_dim=block_dim, sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shared_2d(a, out, n):
        i = cuda.grid(1)
        if i < n:
            buf = cuda.shared.array(shape=(16, 2), dtype=np.int32)
            tid = i % 16
            buf[tid, 0] = a[i]
            buf[tid, 1] = a[i] + 1
            cuda.syncthreads()
            out[i] = buf[(tid + 1) % 16, 0] + buf[tid, 1]

    n = block_dim * 2
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shared_2d(a, out, n)
    expected = np.zeros_like(a)
    for blk in range(n // block_dim):
        base = blk * block_dim
        for tid in range(block_dim):
            expected[base + tid] = a[base + ((tid + 1) % block_dim)] + (a[base + tid] + 1)
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_block_local_stencil_cuda():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(block_local=["inp"])
    def ti_stencil_bl(inp, out, n, m):
        i, j = cuda.grid(2)
        if 0 < i < n - 1 and 0 < j < m - 1:
            out[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )

    n, m = 32, 32
    inp = np.arange(n * m, dtype=np.float32).reshape(n, m)
    out = np.zeros_like(inp)
    ti_stencil_bl(inp, out, n, m)
    expected = np.zeros_like(inp)
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            expected[i, j] = (
                inp[i, j] * 4
                - inp[i - 1, j]
                - inp[i + 1, j]
                - inp[i, j - 1]
                - inp[i, j + 1]
            )
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shfl_sync_i32():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shfl(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_sync(0xFFFFFFFF, a[i], 0)

    n = 64
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shfl(a, out, n)
    expected = np.repeat(a.reshape(-1, 32)[:, :1], 32, axis=1).reshape(-1)
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shfl_sync_keywords_width():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shfl_kw(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_sync(mask=0xFFFFFFFF, value=a[i], src_lane=0, width=32)

    n = 64
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shfl_kw(a, out, n)
    expected = np.repeat(a.reshape(-1, 32)[:, :1], 32, axis=1).reshape(-1)
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shfl_up_i32():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shfl_up(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_up_sync(0xFFFFFFFF, a[i], 1)

    n = 64
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shfl_up(a, out, n)
    expected = a.copy()
    for base in range(0, n, 32):
        expected[base + 1 : base + 32] = a[base : base + 31]
    mask = (np.arange(n) % 32) >= 1
    assert np.allclose(out[mask], expected[mask])


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shfl_down_f32():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.f32, 1), "out": (ti.f32, 1), "n": ti.i32})
    def ti_shfl_down(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_down_sync(0xFFFFFFFF, a[i], 1)

    n = 64
    a = np.arange(n, dtype=np.float32)
    out = np.zeros_like(a)
    ti_shfl_down(a, out, n)
    expected = a.copy()
    for base in range(0, n, 32):
        expected[base : base + 31] = a[base + 1 : base + 32]
    mask = (np.arange(n) % 32) <= 30
    assert np.allclose(out[mask], expected[mask])


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shfl_xor_i32():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shfl_xor(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_xor_sync(0xFFFFFFFF, a[i], 1)

    n = 64
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shfl_xor(a, out, n)
    expected = a.copy()
    lanes = np.arange(32)
    for base in range(0, n, 32):
        expected[base : base + 32] = a[base + (lanes ^ 1)]
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_ballot_sync_full_mask():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"out": (ti.i32, 1), "n": ti.i32})
    def ti_ballot(out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.ballot_sync(0xFFFFFFFF, i % 32 < 16)

    n = 64
    out = np.zeros(n, dtype=np.int32)
    ti_ballot(out, n)
    expected = np.full(n, 0xFFFF, dtype=np.int32)
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shfl_xor_requires_i32():
    ti.reset()
    ti.init(arch=ti.cuda)

    def bad_shfl(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.shfl_xor_sync(0xFFFFFFFF, a[i], 1)

    with pytest.raises(FrontendError):
        ti.njit(bad_shfl)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_bitwise_ops_cuda():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 2), "n": ti.i32})
    def ti_bitwise(a, out, n):
        i = cuda.grid(1)
        if i < n:
            x = a[i]
            out[i, 0] = x | 1
            out[i, 1] = x ^ 3
            out[i, 2] = x << 1
            out[i, 3] = x >> 1

    n = 64
    a = np.arange(n, dtype=np.int32)
    out = np.zeros((n, 4), dtype=np.int32)
    ti_bitwise(a, out, n)
    expected = np.zeros_like(out)
    expected[:, 0] = a | 1
    expected[:, 1] = a ^ 3
    expected[:, 2] = a << 1
    expected[:, 3] = a >> 1
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_shift_negative_value_cuda():
    ti.reset()
    ti.init(arch=ti.cuda)

    @ti.njit(sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_shift(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = a[i] >> 1

    n = 32
    a = np.arange(-n // 2, n // 2, dtype=np.int32)
    out = np.zeros_like(a)
    ti_shift(a, out, n)
    expected = a >> 1
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_syncthreads_count():
    ti.reset()
    ti.init(arch=ti.cuda)
    block_dim = 32

    @ti.njit(block_dim=block_dim, sig={"a": (ti.i32, 1), "out": (ti.i32, 1), "n": ti.i32})
    def ti_sync_count(a, out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.syncthreads_count(a[i] & 1)

    n = 64
    a = np.arange(n, dtype=np.int32)
    out = np.zeros_like(a)
    ti_sync_count(a, out, n)
    expected = np.full(n, 16, dtype=np.int32)
    assert np.allclose(out, expected)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_syncthreads_and_or():
    ti.reset()
    ti.init(arch=ti.cuda)
    block_dim = 32

    @ti.njit(block_dim=block_dim, sig={"out": (ti.i32, 1), "n": ti.i32})
    def ti_sync_all(out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.syncthreads_and(i < n)

    @ti.njit(block_dim=block_dim, sig={"out": (ti.i32, 1), "n": ti.i32})
    def ti_sync_any(out, n):
        i = cuda.grid(1)
        if i < n:
            out[i] = cuda.syncthreads_or(i == 0)

    n = 32
    out_all = np.zeros(n, dtype=np.int32)
    out_any = np.zeros(n, dtype=np.int32)
    ti_sync_all(out_all, n)
    ti_sync_any(out_any, n)
    assert np.all(out_all != 0)
    assert np.all(out_any != 0)
