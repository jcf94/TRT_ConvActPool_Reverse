---
mode: agent
description: 'Scaffold a new bench_resnet_stem_vN benchmark attempt: create src/bench_resnet_stem_vN.cu from the shared harness, wire its CMakeLists.txt block, build, and append a README version row.'
---

# Add a new benchmark version

Create the next `bench_resnet_stem_vN` optimization attempt. Optional input: a
short description of the kernel idea to try (`${input:idea}`).

## Steps

1. Determine `N`: the highest existing `src/bench_resnet_stem_v*.cu` + 1.
2. Create `src/bench_resnet_stem_vN.cu`:
   - `#include "resnet_stem_common.cuh"` for shapes, `CUDA_CHECK`, `Args`,
     `parse_args`, `cpu_reference`, `max_abs_err`, `print_result`,
     `time_kernel`, `pack_weights_mma4`, and packed-MMA constants.
   - Keep only this version's unique kernel(s) and `run_*_case` sweep cases.
     Do not copy the harness into the file (follow the v24+ layout).
   - `main` must: parse args, fill inputs with `std::mt19937 rng(1234)`,
     compute `cpu_reference`, run each case, and print via `print_result`.
3. Append a matching block to `CMakeLists.txt`:
   ```cmake
   add_executable(bench_resnet_stem_vN src/bench_resnet_stem_vN.cu)
   target_compile_options(bench_resnet_stem_vN PRIVATE
     $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
     $<$<COMPILE_LANGUAGE:CXX>:-O3>
   )
   ```
   Add `target_link_libraries(bench_resnet_stem_vN PRIVATE cublas)` only if the
   kernel uses cuBLAS.
4. Build with `./scripts/build.sh` and run
   `./build/bench_resnet_stem_vN --iters 1000 --warmup 100`.
5. Confirm `max_abs_err=0` for every correct case, then add a row to the README
   version table and a bullet describing vN, citing the best timing.

## Constraints

- sm_86 only; `nvcc` is in `/usr/local/cuda-11.8`. The target is the fused
  `CaskConvActPool` core (~0.01 ms), not whole-engine time.
- Keep two-space indent, uppercase constants, snake_case helpers, `_kernel`
  suffix on kernels.
