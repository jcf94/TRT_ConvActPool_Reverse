# CUDA Operator Benchmark

This repository is a reproducible CUDA benchmark harness for iterating on a
high-performance INT8 ResNet stem operator and comparing it with TensorRT.

Current operator:
`Conv 7x7 stride 2 pad 3 -> ReLU -> MaxPool 3x3 stride 2 pad 1`.

Shapes:

- input: `1x3x224x224`
- conv output: `1x64x112x112`
- maxpool output: `1x64x56x56`

## Remote Build

```bash
cd /root/cuda_op_bench
./scripts/build.sh
./build/bench_resnet_stem --iters 1000 --warmup 100
./scripts/run_resnet_stem.sh
```

`scripts/build.sh` builds the active benchmark targets listed in
`CMakeLists.txt`. Older exploratory sources that are no longer part of the
default build live under `src/legacy/`.

## TensorRT Reproduction

The original observation is that TensorRT INT8 can fuse the first ResNet
`Conv + ReLU + MaxPool` into one kernel. Inserting `QuantizeLinear` and
`DequantizeLinear` between `Conv` and `ReLU` blocks that fusion and splits the
profile into multiple kernels.

Generate and run the TensorRT reproduction:

```bash
python3 scripts/make_resnet_stem_onnx.py
./scripts/run_tensorrt_stem.sh
```

On the current remote machine, pip TensorRT 10.10 provides Python bindings and
runtime libraries but not `trtexec`, so `run_tensorrt_stem.sh` falls back to
`scripts/run_tensorrt_python.py`.

Latest measured TensorRT results on the RTX 3080 Ti:

| model | engine mean | main profile entries |
| --- | ---: | --- |
| plain `Conv->ReLU->MaxPool` | `0.033801 ms` | fused layer `node_of_conv_out + node_of_relu_out + node_of_output`: `0.008938 ms`; input reformat `0.007369 ms`; output reformat `0.007671 ms` |
| Q/DQ-blocked model | `0.064032 ms` | `node_of_conv_out`: `0.029057 ms`; `node_of_conv_q`: `0.005975 ms`; `node_of_conv_dq`: `0.004217 ms`; `PWN(node_of_relu_out)`: `0.004846 ms`; `node_of_output`: `0.004740 ms` |

Conclusion: the plain graph is compiled into one fused TensorRT layer, while the
Q/DQ graph is split and is roughly 2x slower at the engine level.

See [TensorRT Fused Core Profile Notes](docs/tensorrt_fused_core_profile.md) for
the detailed profiling record. `ncu` hardware-counter collection is blocked in
the current container by host driver setting `RmProfilingAdminOnly=1` and
missing `CAP_SYS_ADMIN`, so timing plus SASS/resource inspection are the main
profiling signals.

The fused TensorRT cubin contains
`sm80_trt_conv_act_pool_v3_tile_rows_8_tile_cols_120_execute_kernel_trt`, uses
`IMMA.16816.S8.S8` INT8 tensor-core instructions, and has no `DP4A`
instructions. Matching TensorRT therefore requires an INT8 MMA schedule, not
more scalar-DP4A tuning.

## Current Standing

Best reproducible hand-written CUDA kernel: `bench_resnet_stem_v38`, about
`0.0183 ms`, `max_abs_err=0`. This is about 1.7x slower than TensorRT's fused
core (`~0.0108 ms` in the SASS re-analysis, `0.008938 ms` in the layer profile).

Best generic-library comparison: `bench_resnet_stem_v44` with CUTLASS 4.6 INT8
implicit-GEMM conv is slower here (`0.0685 ms` conv only, `0.0877 ms` conv+pool).
The shape is pathological for stock CUTLASS because RGB input is padded from
3 to 16 channels, the 64x64 problem underfills the generic tile, and pooling is
not fused.

The remaining gap is architectural, not a simple parameter issue:

- TensorRT uses a bespoke 8x120-style INT8 MMA tile with about 240 IMMA
  instructions per CTA, wide loads, and a fused pool/ReLU/quant epilogue.
- The custom packed-MMA family tops out around `0.018 ms`; attempts to remove
  individual bottlenecks either lose occupancy or reduce IMMA density.
- `v39`, `v41`, `v42`, and `v43` are important negative results: eliminating
  shared pool re-reads or moving toward register pooling is not enough unless the
  wide high-IMMA tile is preserved at useful occupancy.

## Active Source Layout

- `src/resnet_stem_common.cuh`: shared shapes, `CUDA_CHECK`, arg parsing, CPU
  reference, timing, result printing, and packed-MMA weight preparation.
- `src/bench_resnet_stem.cu`: baseline direct CUDA implementation.
- `src/bench_resnet_stem_v*.cu`: retained milestone versions that represent a
  distinct implementation strategy, a best result, or a decisive negative
  result.
- `src/legacy/bench_resnet_stem_v*.cu`: archived versions that were mainly
  parameter sweeps, superseded intermediate experiments, or low-signal
  regressions. They are kept for reference but are not built by default.

## Version Summary

| version | status | best relevant result | note |
| --- | --- | ---: | --- |
| baseline | active | `fused_tiled_14x14 = 0.123176 ms` | direct DP4A baseline |
| v2 | active | `fused_tiled_14x14 = 0.081843 ms` | prepacked DP4A weights |
| v3 | legacy | `fused_tiled_56x56 = 0.108958 ms` | interior fast path regressed |
| v4 | active | `cublas_int8_gemm = 0.019873 ms`; `cublas_gemm_relu_pool = 0.039794 ms` | tensor-core upper-bound with prebuilt im2col, not fused |
| v5 | legacy | `fused_tiled_oc4_14x14 = 0.044329 ms` | OC4 direct-DP4A step, superseded |
| v6 | active | `fused_tiled_oc8_8x8 = 0.032353 ms` | first strong OC8 direct-DP4A result |
| v7 | legacy | `fused_tiled_oc16_14x14 = 0.038880 ms` | OC16 pressure regressed |
| v8 | active | `fused_tiled_oc8_8x8_b128 = 0.031077 ms` | best direct-DP4A milestone |
| v9-v11 | legacy | `0.034-0.036 ms` | OC8 tile/block/input-staging sweeps did not beat v8 |
| v12 | active | `ptx_mma_oc16_8x8_w8_b256 = 0.055870 ms` | first correct inline PTX INT8 MMA kernel |
| v13-v14 | legacy | `0.043-0.050 ms` | shared-B/tile-size steps, superseded by v15 |
| v15 | active | `ptx_mma_oc64_smem_b_4x4_w4_b128 = 0.041651 ms` | reuses one B tile across all 64 output channels |
| v16 | legacy | `0.044665 ms` | channel-split pooling epilogue regressed |
| v17 | active | `ptx_mma_oc32_dual_n_4x4_w4_b128 = 0.038609 ms` | dual-N schedule reduces loop/fragment overhead |
| v18-v19 | legacy | `0.032719-0.038691 ms` | dual-N and packed-B steps, superseded by v20 |
| v20 | active | `ptx_mma_oc32_dual_n_packed_ab_4x4 = 0.026198 ms` | packed A/B operands; first custom MMA path faster than DP4A |
| v21-v23 | legacy | `0.028-0.511 ms` | packed epilogue, accumulator atomics, and simple pool-owner designs regressed |
| v24 | active | `ptx_mma_oc32_dual_n_packed_ab_4x4_w8_b256 = 0.023772 ms` | refactored packed A/B MMA baseline using shared harness |
| v25-v30 | legacy | `0.024-0.056 ms` | static-N, parallel/register pool, pool-owner, and raw-acc int32 sweeps |
| v31 | active | `ptx_mma_oc32_raw_acc16_pool_4x4 = ~0.021-0.025 ms` | int16 raw-acc pool probe; useful precursor to OC64 path |
| v32-v34 | legacy | `0.023-0.046 ms` | OC-group epilogue sweeps, no decisive gain |
| v35 | active | `ptx_mma_oc64_raw_acc16_pool_4x4_w4_b256 = ~0.0196 ms` | one CTA owns full 64-OC slab, TRT-style weight reuse |
| v36 | legacy | `~0.029-0.032 ms` | shared weights and transposed accumulator layout regressed |
| v37 | active | `v37_oc64_wide_pool_4x4_b256 ~= 0.019 ms` | dynamic shared memory and wider tile probe |
| v38 | active | `v38_oc64_wide_pool_4x4_b256 ~= 0.0183 ms` | best hand kernel; int8 pool tile confirms plateau |
| v39 | active | `~0.021 ms` | transposed `[N][OC]` cuts pool LDS to TRT-like count but slows down |
| v40 | legacy | `~0.029 ms` | 4-N/warp straight-line attempt regressed |
| v41 | active | `v41_regpool_oc64 = 0.0266 ms` | register-strip pool without conv tile; correct but recomputes too much |
| v42 | active | `v42_implgemm_pc8 = 0.0214 ms` | implicit-GEMM strip cuts pool LDS but loses IMMA density |
| v43 | active | `v43_8x8_reg64pool = 0.0378 ms` | 8-row tile plus large register pool spills/regresses |
| v44 | active | `v44_cutlass_conv = 0.0685 ms`; `v44_cutlass_conv_pool = 0.0877 ms` | stock CUTLASS comparison, not competitive on this shape |
| v45 | active | `v45_trt_replica_8x7 = 0.0414 ms`, err=0 | direct SASS reverse: 8-row tile, IMMA mainloop, packed-byte register pool; correct but smem-bound, ~2.3x v38 (see `docs/trt_sass_reverse_v45.md`) |
| v46 | active | `v46_t8 = 0.0418 ms`, err=0 | transposed [N][OC] pool + vmax4; strided MMA store regresses (negative) |
| v47 | active | `v47_3x3 = 0.0218 ms`, err=0 | occupancy sweep, confirms ~0.0216 floor (negative) |
| v48 | active | `v48_240imma_64x96 = 0.034 ms`, err=0 | first instruction-count match: 240 IMMA via 64OC x 96 conv-pts/CTA, K160; REG255 spills, no epilogue yet |
| v49 | active | `v49_240imma_pool = 0.045 ms`, err=2 | 240 IMMA + I2FP dequant + 3x3 pool epilogue; float scale rounding vs int ref = err2; LDS/STG still high |## Source Retention Policy
| v50 | active | `v50_240imma_pool = 0.0255 ms`, err=2 | 240 IMMA + REG116(no spill) + register pool, LDS696->156; err2 boundary (112%12); STS/STG still high |
| v51 | active | `v51_240imma = 0.044 ms`, err=2 | halo-stepped 240-IMMA tile, REG114; more CTAs regressed vs v50; STS/STG still high |Keep a version in `src/` when it is one of:
| v52 | active | `v52 = 0.080 ms`, err=1 | 2-warp/CTA dup smem 43KB -> 1 CTA/SM, regressed; smem is the occupancy wall (negative) |
- a current best or reproducible comparison target,
- the first implementation of a new strategy,
- a decisive negative result that changes the optimization direction,
- a compact refactored baseline for future work.

Move a version to `src/legacy/` when it is mainly:

- a tile, warp, block-size, or channel-count sweep,
- a local parameter tweak that was superseded by a later milestone,
- a large copied historical file whose result is already captured by a smaller
  refactored version or the README table.

Archived in this cleanup:

```text
v3 v5 v7 v9 v10 v11 v13 v14 v16 v18 v19 v21 v22 v23
v25 v26 v27 v28 v29 v30 v32 v33 v34 v36 v40
```

## Notes

- Target GPU: RTX 3080 Ti (`sm_86`).
- CUDA: `/usr/local/cuda-11.8`.
- TensorRT: `tensorrt-cu11==10.10.0.31` via pip on the remote machine.
- SASS extraction is reproducible with `scripts/extract_engine_fatbin.py`
  followed by `cuobjdump -sass -arch sm_86` on the extracted fatbin.
- The comparison target is TensorRT's fused `CaskConvActPool` layer
  (`~0.01 ms`), not whole-engine time.
