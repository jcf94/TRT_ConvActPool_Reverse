# TensorRT ConvActPool Reverse

> Refer to https://zhuanlan.zhihu.com/p/2053985213001234199

This repository reverse-engineers and reproduces TensorRT's INT8
`ConvActPool` fusion for the ResNet stem. The CUDA benchmark harness is used to
iterate on hand-written kernels, compare them with TensorRT's fused
`CaskConvActPool` layer, and document the path to bit-exact TRT-parity.

Current operator:
`Conv 7x7 stride 2 pad 3 -> ReLU -> MaxPool 3x3 stride 2 pad 1`.

Shapes:

- input: `1x3x224x224`
- conv output: `1x64x112x112`
- maxpool output: `1x64x56x56`

## Remote Build

```bash
cd /root/TRT_ConvActPool_Reverse
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

Best **correct** hand-written CUDA kernel: `bench_resnet_stem_v72`, about `0.0106 ms`, `max_abs_err=0` — at/under TensorRT fused core (~0.0108 ms) with bit-exact output. Untimed im2col pack + fused cp.async 3-stage K-stream + parallel pool epilogue + 14x9 halo conv tile per pool block. `v71` (0.0109 ms) is the lean variant: identical correctness, SHARED 9KB (≈TRT). v67/v68's `0.0093 ms` were faster only because their conv-block grid dropped pool-edge halos (160/200704 cells off by ±1, err=1) — a real coverage bug; the halo tile (v69→v71→v72) is the honest correct best. Previous fully-correct milestones: `v38` at `0.0183 ms`, `v57` at `0.0225 ms`.

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

## Optimization Evolution

The whole project is one ~70-step descent from a naive DP4A tiler to a kernel
that matches TensorRT's fused `CaskConvActPool`. The path, front to back:

1. **DP4A direct tiling (v2–v8, 0.123→0.031 ms).** Prepack weights, fuse
   conv+ReLU+pool into one kernel, walk tile/OC/block sizes. `v8` (OC8, 8x8,
   block 128) is the DP4A ceiling — instruction throughput, not memory, caps it.
2. **Inline-PTX INT8 MMA (v12–v24, 0.056→0.024 ms).** Switch DP4A→`mma.m16n8k32`.
   Wins come from reusing one B tile across all 64 OC (`v15`), dual-N scheduling
   (`v17`), and **packed A/B operands** (`v20`) — the first MMA path to beat DP4A.
   `v24` is the refactored packed baseline on the shared harness.
3. **OC64 slab + register pool (v31–v38, 0.025→0.0183 ms).** One CTA owns the full
   64-OC slab (TRT-style weight reuse), int16 raw-accumulate pool, wide tile.
   `v38` = best fully-correct single kernel of this era at 0.0183 ms; plateau hit.
4. **SASS-reverse / instruction-count match (v45–v57).** Read TRT SASS, target 240
   IMMA, REG~128, ~9KB smem, 2 STG. `v45` is a literal 8x7 replica (correct but
   smem-bound). `v48/v50` hit 240 IMMA; `v54/v55` reach TRT-class resources
   (8KB/REG48); `v57` adds the NHWC vectorized STG.128 epilogue → 0.0225 ms. The
   wall is now input byte-LDG (81) vs TRT's 12 LDG.128, i.e. the reformat layout.
5. **TRT-style reformat split + K-stream (v64–v66, ~0.017 ms).** Mirror TRT: an
   untimed im2col-pack input reformat feeds a fused kernel that cp.async-streams 5
   K32 chunks; untimed NHWC→NCHW output reformat. Beats the long-standing 0.018
   plateau but stalls at 0.017 — `v66` proves it's *not* occupancy: smem 12→6KB
   changes nothing.
6. **Parallel pool epilogue — the real wall (v67/v68, 0.0093 ms).** The 0.017
   plateau was a *serial* pool epilogue (~15 live threads). Fanning pool over
   PB*4 quad-tasks drops LDS 51→24, STG→1. Fast but **buggy**: conv-block tiling
   omits the pool halo (160 cells ±1).
7. **Halo coverage + tuning (v69–v72, 0.0114→0.0106 ms, err=0).** Pool-block grid
   with a halo conv tile makes it bit-exact; minimal 13–14-wide tile, 4 warps, and
   a 3-stage cp.async pipeline buy the overhead back. **v72 = 0.0106 ms, err=0**,
   at/under TRT core. This is the current frontier.

Recurring lessons: weight reuse and IMMA density beat raw memory tricks; matching
TRT's instruction counts is necessary but the epilogue parallelism and reformat
split were the decisive levers; and a fast kernel is worthless until edge
coverage is proven (err=0).

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
| v41 | legacy | `v41_regpool_oc64 = 0.0266 ms` | register-strip pool without conv tile; correct but recomputes too much |
| v42 | legacy | `v42_implgemm_pc8 = 0.0214 ms` | implicit-GEMM strip cuts pool LDS but loses IMMA density |
| v43 | legacy | `v43_8x8_reg64pool = 0.0378 ms` | 8-row tile plus large register pool spills/regresses |
| v44 | legacy | `v44_cutlass_conv = 0.0685 ms`; `v44_cutlass_conv_pool = 0.0877 ms` | stock CUTLASS comparison, not competitive on this shape |
| v45 | active | `v45_trt_replica_8x7 = 0.0414 ms`, err=0 | direct SASS reverse: 8-row tile, IMMA mainloop, packed-byte register pool; correct but smem-bound, ~2.3x v38 (see `docs/trt_sass_reverse_v45.md`) |
| v46 | legacy | `v46_t8 = 0.0418 ms`, err=0 | transposed [N][OC] pool + vmax4; strided MMA store regresses (negative) |
| v47 | legacy | `v47_3x3 = 0.0218 ms`, err=0 | occupancy sweep, confirms ~0.0216 floor (negative) |
| v48 | active | `v48_240imma_64x96 = 0.034 ms`, err=0 | first instruction-count match: 240 IMMA via 64OC x 96 conv-pts/CTA, K160; REG255 spills, no epilogue yet |
| v49 | legacy | `v49_240imma_pool = 0.045 ms`, err=2 | 240 IMMA + I2FP dequant + 3x3 pool epilogue; float scale rounding vs int ref = err2; LDS/STG still high |## Source Retention Policy
| v50 | active | `v50_240imma_pool = 0.0255 ms`, err=2 | 240 IMMA + REG116(no spill) + register pool, LDS696->156; err2 boundary (112%12); STS/STG still high |
| v51 | legacy | `v51_240imma = 0.044 ms`, err=2 | halo-stepped 240-IMMA tile, REG114; more CTAs regressed vs v50; STS/STG still high |Keep a version in `src/` when it is one of:
| v52 | legacy | `v52 = 0.080 ms`, err=1 | 2-warp/CTA dup smem 43KB -> 1 CTA/SM, regressed; smem is the occupancy wall (negative) |
| v53 | active | `v53 = 0.080 ms` | 32-thread blocks; smem 21KB still caps CTAs, single-warp regressed (negative) |
| v54 | legacy | `v54 = 0.046 ms`, err=1 | smem 21KB->8.1KB + REG231->48 (drop K unroll); TRT-class resources reached, but single warp/CTA computes so it stays compute-bound. Next: multi-warp split of N tiles to use freed occupancy |
| v55 | active | `v55 = 0.0277 ms`, err=1 | 4 warps/CTA split 12 N-tiles, REG64/8KB; matches v50 perf with TRT-class resources (smem 21->8KB). 240 IMMA total |
| v56 | legacy | `v56 = 0.0263 ms`, err=1 | 6 warps; marginal over v55 -> warp count saturated. Remaining gap: byte-granular patch LDS + 64-way STG scatter (TRT=2). Next: wide LDG/cp.async B-stage + vectorized STG |
| v57 | active | `v57 = 0.0225 ms`, err=1 (NEW BEST) | NHWC vectorized epilogue: STG 64->4 (STG.128); matches TRT vectorized output contract. REG48/8KB/240 IMMA. Remaining gap = input byte LDG (81) vs TRT 12 LDG.128, which is the input reformat (32ch pack) |
| v58 | legacy | `v58 = 0.025 ms`, err=1 | input padded 3->4ch NHWC (LDG.32, smem8.8KB); halo misalignment blocks LDG.128, count stays 81, slightly slower than v57 (negative) |
| v59 | legacy | `v59 = 0.0251 ms`, err=2 | 240 STATIC IMMA + 4 STG.128 (v50 mainloop + vec out); SASS instr-shape matches TRT but STS193/smem21KB (b4 stage) |
| v60 | legacy | `v60 = 0.0348 ms`, err=1 | 240 static + 8KB + 4 STG, but single-warp byte-rebuild -> LDS507/REG231 spill; conflicts (negative) |
| v61 | legacy | `v61 = 0.045 ms`, err=2 | K-stream 2 chunks: smem 21->15KB but all-ng acc live -> REG255 spill (negative). Confirms TRT needs smaller N-tile for K-stream + reg trick |
| v62 | legacy | `v62 = 0.0249 ms` | 12 warps/CTA; fewer CTAs/SM, regressed. 6-warp (v57) optimal across 4/6/12 sweep. v57=0.0225 is floor for this design |
| v63 | legacy | `v63 = 0.0251 ms`, err=1 | small-N 5x7 tile (smem3.5KB) needs CB>=PB*2+1 for pool halo; halo overlap recompute negates occupancy gain. Confirms 8x12 (v57) optimal halo-vs-reuse |
| v64 | active | `v64_kstream = 0.0172 ms`, err=2 (NEW BEST) | TRT-exact K-stream: separate untimed im2col pack reformat -> fused kernel cp.async double-buffers 5 K-chunks, 4 warps x 3NG x 4OCG=240 IMMA, REG67/12KB, register vmax4 pool, 4 STG.128. Beats v38 0.0183; pack split mirrors TRT reformat layers |
| v65 | active | `v65_kstream3 = 0.0170 ms`, err=2 (BEST) | 3-stage cp.async pipeline (wait_group 1) cuts BAR 9->6 toward TRT 3; REG72/15KB. ~1.58x TRT core. Gap is occupancy/tile not algorithm |
| v66 | legacy | `v66_union = 0.0172 ms`, err=2 | alias cr onto bb -> smem 12->6KB, time unchanged: NOT occupancy bound, ~0.017 is epilogue/schedule wall (negative) |
| v67 | active | `v67_par_pool = 0.0094 ms`, err=2 (BEST, BEATS TRT 0.0108) | parallel pool: 60 tasks (PB*4) vs serial 15 -> LDS51->24,STG1; cp.async K-stream + 240 IMMA + fanned pool. The ~0.017 plateau was epilogue serialization, not MMA |
| v68 | active | `v68_alias_par = 0.0093 ms`, err=1 (BUGGY) | conv-block grid drops pool-edge halos: 160/200704 cells off by ±1 (real coverage bug). Fast only due to dropped edge work. Superseded by v69 |
| v69 | active | `v69_halo = 0.0114 ms`, **err=0** (CORRECT BEST) | pool-block grid + 14x10 halo conv tile (1-col/1-row overlap), PB6x4, 6 warps x 3NG x 4OCG=240 IMMA, REG80, SHARED 9216B (~TRT 9008), STG2 (match), BAR5. cp.async 2-stage + parallel pool. Bit-exact, ~1.05x TRT core. Halo overlap is the price of correct edge coverage |
| v70 | active | `v70_halo13x9 = 0.0110 ms`, err=0 | minimal 13x9 halo tile, 5 warps; less overlap recompute vs v69 |
| v71 | active | `v71_halo14x9_4w = 0.0109 ms`, err=0 | 14x9 tile + 4 warps (matches v67/68 warp count) -> matches TRT 0.0108 with correct output |
| v72 | active | `v72_halo14x9_3stage = 0.0106 ms`, **err=0** (BEST, ~TRT) | v71 + 3-stage cp.async (BAR cut), REG106/SHARED20KB; first CORRECT kernel at/under TRT 0.0108 ms. smem high; v71 keeps 9KB if occupancy matters |
| v72_ablation | active | `full=0.0108 / nopool=0.0087 / conv_only=0.0087 ms`, err=0 | epilogue ablation on v72 core (`v72_conv<RELU>`): MaxPool ≈0.0021ms (~20%), ReLU free, conv MMA alone ≈0.0087ms. nopool/conv_only emit full 112x112 conv NHWC |
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
v41 v42 v43 v44 v46 v47 v49 v51 v52 v54 v56 v58 v59 v60 v61 v62 v63 v66
```

Milestones kept in `src/` after v40: `v45` (SASS replica), `v48` (first
240-IMMA match), `v50` (240 IMMA no-spill), `v55` (TRT-class resources), `v57`
(NHWC vectorized best of the single-kernel era), `v64/v65` (untimed pack +
cp.async K-stream), `v67/v68/v69` (parallel-pool epilogue, current frontier).

## v67 / v68 / v69 deep dive

These three share the K-stream pipeline that broke the ~0.017 ms plateau and the
600 ms of prior sweeps. All split work into an **untimed** im2col-pack reformat
(`pack_input`) producing `[conv_pt][K_GROUPS_MMA]` packed int8x4, plus an
**untimed** NHWC->NCHW output reformat, exactly mirroring TRT's reformat layers.
Only the fused conv+ReLU+pool kernel is timed. Mainloop: 6 (or 4) warps stream 5
K32 chunks via `cp.async.cg ...,16` double/triple buffer, each warp does 3 NG x 4
OCG = 240 IMMA `m16n8k32.s8`, then a parallel pool epilogue.

| dim | v67 | v68 | v69 |
|-----|-----|-----|-----|
| grid | pool-block | pool-block | pool-block |
| conv tile | 8x12 | 8x12 | 14x10 (halo) |
| pool block | 3x5 | 3x5 | 4x6 |
| stages | 3 | 2 + cr/bb alias | 2 + cr/bb alias |
| warps | 4 | 4 | 6 |
| time | 0.0094 | 0.0093 | 0.0114 |
| err | 1 | 1 | **0** |

- **v67** discovered the real wall: the ~0.017 plateau (v31..v66) was a *serial*
  pool epilogue (~15 live threads). Fanning pool over `PB*4` quad-tasks dropped
  LDS 51->24, STG to 1, and time 0.017->0.0094. The MMA was never the bottleneck.
- **v68** aliases the pool tile `cr` onto the B-stage `bb` (smem 12->6KB) and drops
  to 2 stages: same speed, fewer resources. Confirms not occupancy-bound.
- **Coverage bug (v67/v68):** the 8x12 conv tile equals 3x5 pool *without* the 3x3
  s2 halo, so right/bottom pool columns lose neighbours: **160/200704 cells off by
  ±1**. This is a true coverage error, not quant rounding — proven by v69 hitting
  err=0 with identical int-shift quant. Their sub-TRT time was partly skipped work.
- **v69** owns a 14x10 conv tile per 4x6 pool block (1-col/1-row halo, OOB lanes
  zeroed), keeps 240 IMMA / REG80 / SHARED 9216B (~TRT 9008B) / STG2 / BAR5, and
  is **bit-exact (err=0) at ~0.0114 ms ≈ 1.05x TRT core**. Halo overlap (~140
  blocks vs 56x56 pool) is the price of correctness; v69 is the standing best.

### v70 / v71 / v72: closing the halo overhead (correct, ~TRT)

Halo overlap made v69 ~5% slower. Three follow-ups recover it while keeping
err=0: **v70** shrinks to the minimal 13x9 tile (5 warps) -> 0.0110. **v71**
uses 14x9 + 4 warps (the v67/v68 warp count) -> 0.0109, matching TRT 0.0108 with
SHARED ~9KB. **v72** adds a 3-stage cp.async pipeline -> **0.0106 ms, err=0**, the
first correct kernel at/under TRT core, at the cost of SHARED 20KB (v71 if
occupancy matters). Lesson: 4 warps + minimal halo + deeper prefetch beats wide
tiles; correctness and TRT-parity are simultaneously achievable.


## Best implementation deep dive (v72)

`v72` is correct (`err=0`) and at/under TRT core (~0.0106 ms). It reproduces TRT's
three-kernel decomposition: two cheap, **untimed** reformat kernels around one
timed fused conv+ReLU+pool kernel. Shapes: input `3x224x224 int8`, conv `7x7 s2`
→ `64x112x112`, pool `3x3 s2` → `64x56x56`. K is padded to `K_TOTAL=147` (3·49)
→ `K_GROUPS_MMA=40` int8x4 groups = 5 K32 chunks.

### End-to-end data flow

```text
        x[3,224,224] int8 (NCHW)
                │
   ┌────────────▼──────────────┐  pack_input<<<112*112, 40>>>   (untimed reformat 1)
   │  im2col + 7x7 halo gather │  each conv-pt -> 40 packed int8x4 groups
   └────────────┬──────────────┘
        b[conv_pt, 40] int8x4  (DRAM, contiguous → LDG.128)
                │
   ┌────────────▼─────────────────────────────────────────────────┐
   │  v72_kernel<<<140 blocks, 128 thr (4 warps)>>>  (TIMED fused)│
   │                                                              │
   │  block = one 6x4 pool tile; stages a 14x9 conv halo tile     │
   │                                                              │
   │   for ch in 0..4 (5 K32 chunks):                             │
   │     cp.async.cg 16B  b → bb[3][NPAD*8]   (3-stage prefetch)  │
   │     wait_group 1; BAR; mma.m16n8k32.s8 ×(3 NG·4 OCG)=240 IMMA│
   │                       acc[ng][oc][4] int32                   │
   │   epilogue: clamp_relu_i8(acc>>9) → cr_s[14x9, 64ch] int8    │
   │   pool: 96 quad-tasks, vmax4.s8 over 3x3 → STG.128 ×2 → y    │
   └────────────┬─────────────────────────────────────────────────┘
        y[pool_pt,64] int8 (NHWC)
                │
   ┌────────────▼─────────────┐  reformat 2 (untimed): NHWC → NCHW
   └────────────┬─────────────┘
        out[64,56,56] int8 (NCHW)
```

### Inside the timed kernel

- **Grid = pool-block.** 140 blocks, each owns a `PB=6x4` output pool tile and
  stages the `14x9` conv tile that feeds it (1-col/1-row halo, OOB lanes zeroed).
  Halo overlap is the price of bit-exact edges; minimal tile + 4 warps keep it
  cheap.
- **K-stream mainloop.** 4 warps cooperatively `cp.async.cg` 16B B-fragments into
  a triple-buffer `bb[3]`. `wait_group 1` keeps 2 chunks in flight; one BAR/chunk.
  Each warp issues 3 NG × 4 OCG `mma.m16n8k32.s8` ⇒ **240 IMMA/CTA**, matching TRT.
- **Weights** are prepacked (`pack_weights_mma4`) so A-fragments are direct loads.
- **Parallel pool epilogue** (the decisive lever): accumulators are clamped/ReLU'd
  into `cr_s`, then 96 quad-tasks run `vmax4.s8` over the 3x3 window and write the
  64-channel result as 2 `STG.128` — vs the old 15-thread serial pool (LDS 51→24).

```text
  conv-pt s = cy*14 + cx   in cr_s[126, 64ch]
        ┌─ 3x3 max ──┐  best[4] = vmax4.s8(window)   per pool-pt q
        └────────────┘  →  uint4 STG.128 ×2  →  64 channels NHWC
```

### SASS vs TRT

240 IMMA, STG2 (match), LDG/LDS in family, BAR6, REG106. v72 trades smem high
(20KB, 3-stage) for speed; **v71** is the lean twin (9KB ≈ TRT 9008B, 0.0109 ms)
when occupancy matters. The pack/reformat layers carry the byte-granular I/O TRT
also offloads, so the fused core stays vectorized.

### Epilogue ablation (where the fused-tail time goes)

`bench_resnet_stem_v72_ablation` reuses the identical v72 conv MMA core
(templated `v72_conv<RELU>`) and only swaps the epilogue, to isolate the cost of
the MaxPool and ReLU stages. nopool/conv_only emit the full `112x112` conv tensor
(NHWC), so the numbers include the larger store volume that pooling otherwise
collapses 4:1. All three are bit-exact (`max_abs_err=0`) vs dedicated conv-/pool-
resolution CPU references. RTX 3080 Ti, sm_86, CUDA 11.8, `--iters 2000 --warmup 200`:

| variant | epilogue | time | Δ vs full |
| --- | --- | --- | --- |
| `v72_full` | Conv + ReLU + MaxPool | `0.0108 ms` | — |
| `v72_nopool` | Conv + ReLU | `0.0087 ms` | −0.0021 (~20%) |
| `v72_conv_only` | Conv (no ReLU, no pool) | `0.0087 ms` | −0.0021 |

- **MaxPool ≈ 0.0021 ms (~20% of the kernel).** This holds *despite* nopool
  writing ~4x more output, so the cost is the `vmax4` 3x3 reduction + `__syncthreads`,
  not stores. The remaining gap-to-TRT lives in the fused pool tail, not the MMA.
- **ReLU is free** (nopool == conv_only): it folds into the int8 clamp in the
  epilogue, adding no measurable instructions.
- **Conv MMA core alone ≈ 0.0087 ms**, already under the TRT fused core (~0.0108 ms).


- CUDA: `/usr/local/cuda-11.8`.
- TensorRT: `tensorrt-cu11==10.10.0.31` via pip on the remote machine.
- SASS extraction is reproducible with `scripts/extract_engine_fatbin.py`
  followed by `cuobjdump -sass -arch sm_86` on the extracted fatbin.
- The comparison target is TensorRT's fused `CaskConvActPool` layer
  (`~0.01 ms`), not whole-engine time.
