# Repository Guidelines

## Project Structure & Module Organization

This repository is a CUDA benchmark harness for the INT8 ResNet stem operator
(`Conv 7x7 -> ReLU -> MaxPool`). CUDA/C++ benchmark implementations live in
`src/`; each optimization attempt is a separate executable source such as
`src/bench_resnet_stem_v8.cu`. Build configuration is in `CMakeLists.txt`.
Operational scripts live in `scripts/`, with TensorRT/ONNX reproduction helpers
and benchmark runners. `docs/` contains profiling notes. Generated directories
such as `build/`, `results/`, and `models/` are outputs and should not be
treated as source.

Shared shapes, `CUDA_CHECK`, arg parsing, CPU reference, timing, and packed-MMA
weight prep live in [src/resnet_stem_common.cuh](src/resnet_stem_common.cuh).
Read [README.md](README.md) first: it is the authoritative log of every `v*`
attempt, current best timings, and the gap-to-TensorRT plan. Profiling detail is
in [docs/tensorrt_fused_core_profile.md](docs/tensorrt_fused_core_profile.md) and
[docs/trt_kernel_gap_plan.md](docs/trt_kernel_gap_plan.md). The README version
table is more current than the source-file count, so trust it over guessing.

## Build, Test, and Development Commands

- `./scripts/build.sh`: configures a Release CMake build for CUDA architecture
  `86` and builds all benchmark executables.
- `./build/bench_resnet_stem --iters 1000 --warmup 100`: runs the baseline
  benchmark interactively.
- `./scripts/run_resnet_stem.sh`: runs the baseline benchmark and writes
  `results/resnet_stem.csv`.
- `python3 scripts/check_tensorrt.py`: reports Python TensorRT, `trtexec`, and
  `libnvinfer` availability.
- `./scripts/run_tensorrt_stem.sh`: generates ONNX models, runs TensorRT
  profiling, and writes outputs under `results/`.

## Coding Style & Naming Conventions

Use C++17/CUDA17, two-space indentation, and the existing compact kernel style.
Keep constants uppercase (`CONV_OH`, `POOL_OW`), helper functions snake_case
(`parse_args`, `conv_point`), and kernel names descriptive with a `_kernel`
suffix. New benchmark attempts should continue the versioned pattern:
`src/bench_resnet_stem_vN.cu`, reusing the harness in
`resnet_stem_common.cuh` and keeping only that version's unique kernels/cases.

Ablation or isolation experiments on an existing version use the
`src/bench_resnet_stem_vN_ablation.cu` suffix, reuse that version's conv core via
a shared templated `__device__` helper (see `v72_conv<RELU>` in
`bench_resnet_stem_v72_ablation.cu`), and print one timed line per variant. Each
variant must still validate `max_abs_err=0` against a matching CPU reference.
Then append a matching executable block in `CMakeLists.txt`:

```cmake
add_executable(bench_resnet_stem_vN src/bench_resnet_stem_vN.cu)
target_compile_options(bench_resnet_stem_vN PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
(`max_abs_err` must be 0) and timings against the baseline. For TensorRT changes,
run `./scripts/run_tensorrt_stem.sh` and update profiling notes only when results
are reproducible on the target GPU. Record hardware, CUDA, TensorRT version, and
iteration counts when adding performance claims; the comparison target is the
fused `CaskConvActPool` layer (~`0.01 ms`), not whole-engine time. Note env
constraints: GPU is RTX 3080 Ti (`sm_86`), `nvcc` is in `/usr/local/cuda-11.8`,
and `ncu` hardware counters are blocked, so SASS/resource usage is the main
profiling signal
Preserve `CUDA_CHECK`-style error handling and explicit benchmark arguments such
as `--iters`, `--warmup`, and `--csv`. Targets that need cuBLAS also need
`target_link_libraries(... PRIVATE cublas)` (see `bench_resnet_stem_v4`).

## Testing Guidelines

There is no separate unit test framework. Validation is benchmark-driven:
build successfully, run the relevant executable, and compare correctness checks
and timings against the baseline. For TensorRT changes, run
`./scripts/run_tensorrt_stem.sh` and update profiling notes only when results
are reproducible on the target GPU. Record hardware, CUDA, TensorRT version, and
iteration counts when adding performance claims.

## Commit & Pull Request Guidelines

The Git history uses short imperative subjects, for example `Add v3 interior
fast path benchmark` and `Document TensorRT SASS extraction`. Keep commits
focused on one benchmark, script, or documentation change. Pull requests should
state the benchmark goal, list commands run, summarize before/after timings, and
note any environment constraints such as missing `trtexec` or blocked profiler
counters. Include generated CSV/profile snippets only when they support the
change; avoid committing bulky generated artifacts.
