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
`src/bench_resnet_stem_v10.cu`, then add the matching executable in
`CMakeLists.txt`. Preserve `CUDA_CHECK`-style error handling and explicit
benchmark arguments such as `--iters`, `--warmup`, and `--csv`.

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
