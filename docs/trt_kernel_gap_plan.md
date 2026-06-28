# TensorRT CaskConvActPool Gap Plan

Date: 2026-06-28

Target host:

```text
ssh -p 13879 root@connect.westb.seetacloud.com
remote repo: /root/cuda_op_bench
GPU: NVIDIA GeForce RTX 3080 Ti
driver: 535.104.05
TensorRT: 10.10.0.31
```

## Comparison Target

Use the TensorRT fused layer timing, not whole-engine timing. The current
engine profile contains input/output reformat layers around the core tactic.

Latest useful numbers from the remote machine:

```text
TensorRT CaskConvActPool layer:          0.009854 ms
TensorRT engine_mean_ms:                 0.035513 ms
v24 ptx_mma_oc32_dual_n_packed_ab best:  0.023461 ms
```

The correct target is `CaskConvActPool`, so v24 is about `2.38x` slower than
TensorRT's core, not faster than the whole TensorRT engine.

## Current SASS Gap

Static SASS/resource comparison:

| kernel | registers | shared | IMMA | LDG | LDS | STS | STG | BAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorRT CASK | 128 | 9008 B | 240 | 56 | 50 | 20 | 2 | 3 |
| v24 TILE4 | 48 | 15552 B | 20 | 48 | 308 | 18 | 32 | 2 |

Interpretation:

- TensorRT spends many more registers to keep a much larger straight-line IMMA
  schedule live.
- v24 uses a compact looped MMA body. It has far fewer static IMMA instructions
  and much more shared-memory traffic per unit of useful tensor-core work.
- v24 materializes a `conv_relu_tile` in shared memory, then reloads it for
  pooling. TensorRT's `F2IP.S8.F32.NTZ.RELU` and tiny final `STG` footprint
  suggest a more register-heavy epilogue that writes final pooled output only.

The unique convolution work is `118,013,952 MAC` or `236,027,904 int8 ops` if
counting multiply and add separately. At `0.009854 ms`, TensorRT achieves about
`23.95 TOPS` on this small fused layer. At `0.023461 ms`, v24 achieves about
`10.06 TOPS`. TILE=4 recomputes about `1.266x` conv points because neighboring
pool tiles overlap, so recomputation alone does not explain the gap.

`ncu` can attach on the remote host, but hardware counters are currently blocked
by the driver:

```text
ERR_NVGPUCTRPERM
```

Until host profiling permissions change, use CUDA event timing,
`cuobjdump -res-usage`, `cuobjdump -sass`, and `scripts/sass_summary.py`.

## Iteration Plan

### v25: Static N-Group Schedule

Start from the best v24 shape: `TILE=4`, `WARPS=8`, `block=256`.

Hard-code the N-group work assignment so each active warp enters a compile-time
specialized N-pair path. This should expand the compact looped MMA body into
more straight-line SASS and reduce branch/control overhead.

Acceptance:

```text
./build/bench_resnet_stem_v25 --iters 1000 --warmup 200 --csv
max_abs_err == 0
best timing < v24 0.023461 ms, or SASS shows a useful IMMA/LDS direction
```

Implemented result:

```text
ptx_mma_oc32_static_n_4x4_w8_b256: 0.024841 ms, max_abs_err=0
resource: REG=56, SHARED=15552 B
SASS: IMMA=110, LDG=248, LDS=398, STS=86, STG=32, BAR=2
```

This increased static IMMA as intended, but it also inflated memory
instructions and regressed runtime. Do not continue with this exact static
N-group expansion.

### v26: Wider TensorRT-Like Tile

First test a lower-risk epilogue change before changing the tile shape:
parallelize pooling across `(pool output, output channel)` elements instead of
having one thread compute all 32 channels for a pool output. If this improves
the shared-load bottleneck, keep it as the epilogue baseline for the wider-tile
experiment.

After that, use the TensorRT tactic clue `tile_rows_8_tile_cols_120` to test a
wider N tile. The goal is to reduce TILE=4 overlap and amortize weight/input
setup over more conv points.

Initial result: the parallel epilogue reduces static `LDS` substantially, but
the first TILE=4 measurement regressed. Keep the larger TILE variants in v26 to
separate epilogue effects from recomputation effects.

Implemented result:

```text
ptx_mma_oc32_parallel_pool_4x4_w8_b256: 0.027730 ms, max_abs_err=0
ptx_mma_oc32_parallel_pool_4x4_w8_b224: 0.028336 ms, max_abs_err=0
ptx_mma_oc32_parallel_pool_4x4_w4_b256: 0.028194 ms, max_abs_err=0
ptx_mma_oc32_parallel_pool_5x5_w4_b128: 0.037710 ms, max_abs_err=0
ptx_mma_oc32_parallel_pool_6x6_w4_b128: 0.044803 ms, max_abs_err=0
ptx_mma_oc32_parallel_pool_7x7_w4_b128: 0.055235 ms, max_abs_err=0
resource, TILE4 W8: REG=40, SHARED=15552 B
SASS, TILE4 W8: IMMA=20, LDG=48, LDS=29, STS=18, STG=1, BAR=2
```

The parallel epilogue proves static shared-load count is not the only limiting
factor. It likely hurts store coalescing and warp-level utilization. Keep v26 as
a negative result and return to the v24 epilogue shape for the next attempt.

### v27: Wide Staging Loads

Replace scalar/byte-style fragment construction with wider global/shared loads
where alignment permits. Track the target pattern seen in TensorRT:
`LDG.E.128`, `LDG.E.64`, and low `LDS` counts.

### v28: Register-First Pooling Epilogue

Avoid writing the complete ReLUed conv tile to shared memory and reading it back
for pooling. Keep pool candidates in registers or write packed partials in a
layout that minimizes `LDS.S8`. Do not use `atomicMax`; v22 already showed that
path regresses.

## Reproduction Commands

```bash
./scripts/build.sh
./build/bench_resnet_stem_v24 --iters 1000 --warmup 200 --csv
./build/bench_resnet_stem_v25 --iters 1000 --warmup 200 --csv

/usr/local/cuda-11.8/bin/cuobjdump -res-usage -arch sm_86 \
  build/bench_resnet_stem_v25

python3 scripts/sass_summary.py --cuobjdump /usr/local/cuda-11.8/bin/cuobjdump \
  build/bench_resnet_stem_v25

python3 scripts/sass_summary.py --sass results/trt_conv_act_pool_sm86.sass
```
