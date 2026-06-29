# v65 vs TRT SASS diff and next steps

Date: 2026-06-29 · RTX 3080 Ti sm_86 · CUDA 11.8 · TRT 10.10
Best hand: v65 0.0170 ms (1.58x TRT core 0.0108). v64 0.0172.

## Instruction mix (fused kernel only; input pack untimed)

| op | TRT | v65 | gap |
|---|---:|---:|---|
| IMMA | 240 | 12 (looped) | cosmetic; unroll for static 240 |
| LDG.128 / LDG.64 | 12 / 26 | 0 / 0 (cp.async) | TRT stages via wide LDG+STS, NOT cp.async |
| STS | 20 | 51 | **v65 stores cr int8 tile then re-reads** |
| LDS | 50 | 51 | similar, but v65 LDS is pool re-read; TRT feeds MMA |
| STG | 2 | 4 | minor |
| BAR | 3 | 6 | 3-stage pipeline barriers |
| F2IP/I2FP/FFMA | 40/80/80 | 0 | int8 vmax4 pool vs TRT float dequant+F2IP |
| REG / smem | 128 / 9008 | 72 / 15360 | TRT no cr tile -> 9KB, higher REG ILP |

## Conclusions
1. TRT does single-stage staging (BAR3) + register pool (F2IP max chain), no cp.async.
   v65 spends smem on a `cr` conv-relu tile (96x64 int8 = 6KB) + pool re-read
   (51 STS / 51 LDS). Removing cr via register pool frees ~6KB -> more CTAs/SM.
2. cp.async helped us (0.0225->0.017) but is not the TRT pattern; TRT just overlaps
   wide LDG with IMMA. Our bottleneck is the smem pool round-trip, not loads.
3. Next: v66 register/warp-shuffle pool (no cr smem), keep cp.async B-stream, 2 STG.

## conv+maxpool with ReLU moved to output reformat
ReLU(MaxPool(conv)) == MaxPool(ReLU(conv)) (both monotonic, pad=0 == ReLU floor),
so ReLU can move to the untimed output reformat: fused does conv->int8 quant(no
relu)->signed maxpool(vmax4.s32)->store; reformat clamps neg->0. Cost analysis:
vmax4 cost identical for signed/unsigned; clamp drops one max-with-0 only. No IMMA
or smem change -> negligible fused gain, and it de-fuses what TRT keeps fused
(F2IP relu+pool+quant in one). Verdict: not beneficial; keep ReLU fused.

## v66: alias cr onto bb (12->6KB) — not occupancy bound
v66 aliases the pool tile onto the B-stage smem: smem 12->6KB, REG67, time 0.0172
= unchanged. So ~0.017 is NOT smem/occupancy bound; the wall is the serial pool
epilogue (only 15 threads, 51 LDS re-read) + per-CTA IMMA schedule, matching the
earlier v38/v39 plateau finding. cp.async removed the gather; next real lever is a
register/warp pool that overlaps with MMA (TRT F2IP), historically regressive.
v65 0.0170 remains best.

## v67: parallel pool epilogue — BEATS TRT (0.0094ms)
The whole ~0.017 plateau (v31..v66) was the SERIAL pool epilogue: 15 active threads.
v67 spreads pool over PB_H*PB_W*4=60 tasks (one quad/thread), best[4] not best[16].
LDS 51->24, STG 1, REG80. Time 0.0170->0.0094 ms < TRT 0.0108. 12+ prior versions
mis-attributed the wall to MMA/tile; it was epilogue parallelism. KEY: keep cp.async
K-stream + 240 IMMA AND fan the pool across all warps. err=2 (int shift vs TRT float).

## v68 correctness audit + v69 halo fix — CORRECT BEST (0.0114ms, err=0)
DEFINITIVE: dumped both outputs vs CPU reference. v67/v68 use a CONV-BLOCK grid
(8x12 conv tile = exactly 6x4 pool, no halo). Pool 3x3 s2 needs 1-col/1-row
overlap, so each block lost its right/bottom pool edge: 160/200704 cells off by
±1 (maxd=1). This is a real coverage bug, NOT integer-shift rounding — proven by
v69 reaching err=0 with identical quant. v68's 0.0093ms was fast only because it
skipped that edge work.
v69: POOL-BLOCK grid; each block owns PB6x4 pool and stages a 14x10 halo conv tile
(bx=gx0*2-1) with OOB lanes ZEROED (not copied). 240 IMMA preserved, REG80,
SHARED 9216B(~TRT 9008), STG2(match), BAR5. err=0, 0.0114ms (~1.05x TRT core).
The price of bit-exact edges is halo overlap (~140 blocks). Correctness > speed:
v69 is the new standing best.
