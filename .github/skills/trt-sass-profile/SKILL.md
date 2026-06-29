---
name: trt-sass-profile
description: 'Use when extracting and analyzing TensorRT engine SASS for this repo: dump the embedded fatbin from a serialized engine, disassemble with cuobjdump (sm_86), and summarize IMMA/DP4A/LDG instruction mix to compare custom kernels against the CaskConvActPool fused core. Use for "TensorRT SASS", "fatbin", "IMMA count", "instruction mix", or kernel-gap profiling.'
---

# TensorRT SASS Profiling

`ncu` hardware counters are blocked on the target GPU, so SASS/resource usage is
the primary profiling signal. Comparison target is the fused `CaskConvActPool`
layer (~0.01 ms), not whole-engine time.

## Workflow

1. Generate ONNX + TensorRT engine (writes under `results/`):
   ```bash
   python3 scripts/make_resnet_stem_onnx.py
   ./scripts/run_tensorrt_stem.sh
   ```
2. Extract the embedded fatbin from the serialized engine:
   ```bash
   python3 scripts/extract_engine_fatbin.py results/<engine>.plan \
     -o results/engine_elf/engine_fatbin.bin
   ```
   The script cuts from the fatbin magic `0xBA55ED50`; a trailing "Invalid
   fatbin header" from cuobjdump is expected after useful SASS is emitted.
3. Disassemble and summarize:
   ```bash
   cuobjdump -sass -arch sm_86 results/engine_elf/engine_fatbin.bin
   python3 scripts/sass_summary.py --cuobjdump cuobjdump --arch sm_86 \
     results/engine_elf/engine_fatbin.bin
   ```
   `sass_summary.py` counts families: `IMMA, DP4A, HMMA, F2IP, I2FP, FFMA, LDG,
   LDS, STS, STG, BAR`. Pass `--sass` if input is already disassembled text.
4. Compare against custom kernels: dump a bench executable's SASS the same way
   and contrast IMMA density, byte vs wide loads (`LDG.E.128/64` vs `LDG.E.U8`),
   registers, and shared memory. Record findings in the README table and
   `docs/tensorrt_fused_core_profile.md` only when reproducible on the GPU.

## Notes

- TensorRT core kernel: `sm80_trt_conv_act_pool_v3_tile_rows_8_tile_cols_120...`,
  uses `IMMA.16816.S8.S8`, no `DP4A`.
- `nvcc`/`cuobjdump` are under `/usr/local/cuda-11.8/bin` on sm_86.
