import importlib
import os
import time


def main():
    marks = []

    def mark(label):
        marks.append((label, time.perf_counter()))

    mark("start")

    np = importlib.import_module("numpy")
    mark("import:numpy")
    ti = importlib.import_module("taichi")
    mark("import:taichi")

    arch_name = os.environ.get("TI_DEMO_ARCH") or os.environ.get("TI_ARCH") or "cuda"
    arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    arch = arch_map.get(arch_name, ti.cuda)

    ti.init(arch=arch)
    mark("ti.init")

    @ti.kernel
    def vec_add(a: ti.types.ndarray(dtype=ti.f32, ndim=1),
                b: ti.types.ndarray(dtype=ti.f32, ndim=1),
                out: ti.types.ndarray(dtype=ti.f32, ndim=1),
                n: ti.i32):
        for i in range(n):
            out[i] = a[i] + b[i]

    mark("kernel_defined")

    n = 8
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 2
    out = np.zeros_like(a)
    mark("arrays_ready")

    vec_add(a, b, out, n)
    mark("first_call")
    vec_add(a, b, out, n)
    mark("second_call")

    print(out)
    print("arch:", arch_name)
    print("timing (seconds):")
    prev = marks[0][1]
    for label, t_cur in marks[1:]:
        print(f"  {label}: {t_cur - prev:.4f}")
        prev = t_cur
    print(f"  total: {prev - marks[0][1]:.4f}")


if __name__ == "__main__":
    main()
