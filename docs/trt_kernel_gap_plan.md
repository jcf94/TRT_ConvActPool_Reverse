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

## Algebraic Fusion

The activation and pooling can be merged algebraically:

```text
MaxPool(ReLU(conv)) = ReLU(MaxPool(conv))
```

For this benchmark, the int8 output quantization can also be delayed:

```text
MaxPool(clamp_relu_i8(acc >> shift))
  = clamp_relu_i8(max(acc) >> shift)
```

This is valid because `ReLU`, the common right shift, and upper clamp are
monotonic. Padding remains equivalent because the pooled value is initialized
to zero, matching ReLU's floor. The target epilogue should therefore keep an
`int32 best_acc` per `(output channel, pool output)` and run `clamp_relu_i8`
once at final writeback.

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

### v27: Register-First Pooling Epilogue Prototype

Implemented before v28 as the first register-first attempt. Each warp owns one
N-pair, computes local pool candidates from MMA accumulators in registers,
reduces across each 4-lane row subgroup with shuffle instructions, writes one
partial max per `(N-pair, channel, pool output)`, then performs a final 6-way
shared-memory reduction. This avoids atomics and avoids materializing the full
`conv_relu_tile`.

Implemented result:

```text
ptx_mma_oc32_register_pool_4x4_w8_b256: 0.056778 ms, max_abs_err=0
resource: REG=42, SHARED=16032 B
SASS: IMMA=20, LDG=48, LDS=26, STS=67, STG=1, BAR=2
```

This confirms that simply replacing `conv_relu_tile` with per-N-pair register
pooling is not viable. Although static `LDS` and `STG` drop, the warp-shuffle
and per-output membership logic dominate. A future register-first attempt must
make pool ownership part of the MMA N mapping instead of post-processing every
N-pair against every pool output.

### v28: Delayed ReLU/Quantization Accumulator Pool

Use the algebraic fusion directly: keep shared pool values as raw int32
accumulators, use `atomicMax` on int32 conv accumulators, and apply
`clamp_relu_i8(best_acc, shift)` only once during final output writeback. This
isolates the benefit of delaying ReLU/quantization from the larger pool-owner
mapping problem.

Implemented result:

```text
ptx_mma_oc32_delayed_relu_pool_4x4_w4_b128: 0.038386 ms, max_abs_err=0
ptx_mma_oc32_delayed_relu_pool_4x4_w8_b256: 0.030933 ms, max_abs_err=0
resource, W8: REG=40, SHARED=15008 B
SASS, W8: IMMA=20, LDG=48, LDS=21, STS=3, STG=1, BAR=2
```

Delaying ReLU/quantization is correct and improves the old accumulator-pool
idea, but shared `atomicMax` and conv-to-pool membership control still dominate.

### v29: Pool-Owner N-Lane Utilization

Improve on the older pool-owner idea by using the MMA N dimension for pool
candidates. Each warp owns one pool output and computes the first 8 candidates
in one MMA N tile, then computes candidate 9 in a second MMA tile. This avoids
atomic updates and avoids the v23 issue where each MMA used only one of eight N
lanes.

Implemented result:

```text
ptx_mma_oc32_pool_owner_8n_w4_b128: 0.048035 ms, max_abs_err=0
ptx_mma_oc32_pool_owner_8n_w8_b256: 0.053641 ms, max_abs_err=0
resource, W4/W8: REG=69, SHARED=0 B
SASS: IMMA=20, LDG=118, LDS=0, STS=0, STG=4, BAR=0
```

This is much better than the old v23 pool-owner mapping, but still too slow.
The no-shared design rereads input/weight fragments per pool output and loses
the cross-output reuse that makes v24 faster. A useful next direction is not
pure pool ownership; it must combine delayed ReLU with TensorRT-like wide
input/weight staging across several neighboring pool outputs.

### v30: Raw Accumulator Tile in v24 Shape

Keep the v24 high-reuse MMA/tile structure, but replace the int8
`conv_relu_tile` with an int32 `conv_acc_tile`. Pooling takes the max over raw
accumulators, then applies `clamp_relu_i8` once per final output. This isolates
whether algebraic delayed ReLU helps when the rest of the data-reuse structure
matches the current best custom kernel.

Implemented result:

```text
ptx_mma_oc32_raw_acc_pool_4x4_w8_b256: 0.025424 ms, max_abs_err=0
ptx_mma_oc32_raw_acc_pool_4x4_w4_b256: 0.025570 ms, max_abs_err=0
ptx_mma_oc32_raw_acc_pool_4x4_w8_b192: 0.029558 ms, max_abs_err=0
resource: REG=48, SHARED=23328 B
SASS: IMMA=20, LDG=48, LDS=212, STS=18, STG=32, BAR=2
```

This confirms delayed ReLU is not enough in the v24 tile if raw accumulators are
stored as int32. The increased shared footprint and 32-bit shared traffic
outweigh the saved per-conv-point clamp.

### v31: int16 Raw Accumulator Tile Probe

Use the v30 structure but store raw accumulators as int16 in shared memory. This
is safe for the benchmark's current random input/weight range `[-8, 8]`, where
the maximum absolute accumulator is about 9408. It is not safe for arbitrary
full-range int8 inputs, so treat it as a performance probe for narrower raw
shared storage rather than a final general solution.

Implemented result:

```text
ptx_mma_oc32_raw_acc16_pool_4x4_w8_b256: 0.024975 ms, max_abs_err=0
ptx_mma_oc32_raw_acc16_pool_4x4_w4_b256: 0.024941 ms, max_abs_err=0
ptx_mma_oc32_raw_acc16_pool_4x4_w8_b192: 0.027007 ms, max_abs_err=0
resource: REG=48, SHARED=18144 B
SASS: IMMA=20, LDG=48, LDS=308, STS=18, STG=32, BAR=2
```

This improves over v30 but still loses to v24. The shared-load instruction
count returns to the same level as v24, while shared footprint remains larger.
Delayed ReLU/clamp should only be pursued further if the epilogue avoids the
shared conv tile or packs/reduces raw accumulators in registers.

### v32-v33: Channel-Grouped Pool Epilogue

Return to the v24 int8 `conv_relu_tile`, but vary how many output channels each
pool epilogue task owns. v24 owns all 32 channels per pool output. v32 owns 4
channels per task, reducing per-thread registers and static shared-load count.
v33 owns 8 channels per task, attempting to balance lower task count with lower
register pressure.

Implemented result:

```text
v24 group32 baseline, 4x4_w8_b256: 0.023428-0.023806 ms
v32 group4,  4x4_w8_b256:          0.023522 ms, max_abs_err=0
v33 group8,  4x4_w8_b256:          0.023112-0.023406 ms, max_abs_err=0
v34 group16, 4x4_w8_b256:          0.022947-0.023376 ms, max_abs_err=0
```

SASS/resource for the best `TILE=4, WARPS=8` shape:

| epilogue owner | REG | SHARED | LDS | STG |
| --- | ---: | ---: | ---: | ---: |
| group32/v24 | 48 | 15552 B | 308 | 32 |
| group4/v32 | 40 | 15552 B | 56 | 4 |
| group8/v33 | 40 | 15552 B | 92 | 8 |
| group16/v34 | 40 | 15552 B | 164 | 16 |

The best balance is currently group16: it keeps lower register pressure than
v24 while avoiding the higher task scheduling overhead of group4/group8.

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
