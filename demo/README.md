Demo set for the Numba typed IR frontend.

What this shows:
- Typed IR dump (Numba typed IR).
- Statement map (typed IR stmt id -> source).
- Translation trace (builder actions).
- Timing for typed IR + translation.
- Taichi front IR printed to stdout (CHI IR).

Quick run (all demos, capture stdout to files):
```
PYTHONPATH=$PWD/python python demo/run_all.py
```

Run a single demo (stdout prints Taichi front IR):
```
PYTHONPATH=$PWD/python python demo/vec_add_demo.py
```

Phi example (if/else -> phi/select):
```
PYTHONPATH=$PWD/python python demo/phi_demo.py
```

Prime-check example (numba vs ti.njit correctness check):
```
PYTHONPATH=$PWD/python python demo/sieve_demo.py
```

Select backend:
```
TI_DEMO_ARCH=cpu python demo/vec_add_demo.py
TI_DEMO_ARCH=cuda python demo/stencil2d_demo.py
TI_DEMO_ARCH=vulkan python demo/atomic_reduce_demo.py
```

Output files (per demo, in demo/output):
- `<case>_typed_ir.txt`
- `<case>_ir_map.txt`
- `<case>_trace.txt`
- `<case>_timing.txt`
- `<case>_taichi_ir.txt` (captured stdout by demo/run_all.py)

Notes:
- `TI_DEMO_PRINT_IR=0` disables `print_ir` if you only want the trace files.
- `TI_DEMO_N` controls the workload size for `demo/sieve_demo.py` (case name `prime_check`).
