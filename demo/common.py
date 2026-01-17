import os
from pathlib import Path


def setup_demo_env(case_name):
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "typed_ir": output_dir / f"{case_name}_typed_ir.txt",
        "ir_map": output_dir / f"{case_name}_ir_map.txt",
        "trace": output_dir / f"{case_name}_trace.txt",
        "timing": output_dir / f"{case_name}_timing.txt",
    }
    os.environ.setdefault("TI_NUMBA_TF_DUMP_TYPED_IR", str(paths["typed_ir"]))
    os.environ.setdefault("TI_NUMBA_TF_DUMP_TAICHI_IR", str(paths["trace"]))
    os.environ.setdefault("TI_NUMBA_TF_DUMP_MAP", str(paths["ir_map"]))
    os.environ.setdefault("TI_NUMBA_TF_DUMP_TIMING", str(paths["timing"]))
    return paths


def init_taichi(ti):
    arch_name = os.environ.get("TI_DEMO_ARCH") or os.environ.get("TI_ARCH") or "cpu"
    arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    print_ir = os.environ.get("TI_DEMO_PRINT_IR", "1") != "0"
    ti.init(arch=arch_map.get(arch_name, ti.cpu), print_ir=print_ir)
    return arch_name
