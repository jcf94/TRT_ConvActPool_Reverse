# TRT `CaskConvActPool` Kernel: Deep SASS Re-analysis & Reproduction Schemes

Date: 2026-06-29 · GPU RTX 3080 Ti (sm_86) · CUDA 11.8 · TensorRT 10.10

Kernel: `sm80_trt_conv_act_pool_v3_tile_rows_8_tile_cols_120_execute_kernel_trt`.
Core layer ≈ `0.0108 ms`. This file augments
[tensorrt_fused_core_profile.md](tensorrt_fused_core_profile.md) and
[trt_kernel_gap_plan.md](trt_kernel_gap_plan.md) with a fresh disassembly of
[../results/trt_conv_act_pool_sm86.sass](../results/trt_conv_act_pool_sm86.sass).

## 1. Verified resource & instruction mix

REG 128 · SHARED 9008 B · LOCAL/STACK 0 · CONST[0] 424 B.

| family | count | note |
|---|---:|---|
| `IMMA.16816.S8.S8` | 240 | Ampere INT8 tensorcore, m16 n8 k16, `ROW`×`COL`, RZ-seeded then chained |
| `DP4A`/`HMMA` | 0 | pure IMMA path |
| `LDG.E.128` | 12 | wide weight/activation loads |
| `LDG.E.64` | 26 | weight bias/scale + activation loads |
| `STS.128`/`STS.64` | 9/11 | stage packed fragments to smem |
| `LDS`/`LDS.64` | 48/2 | feed MMA operands |
| `STG.E.64` | 2 | only final 64×56×56 pooled output, tiny footprint |
| `I2FP.F32.S32` | 80 | dequant: acc int32 → f32 |
| `FFMA.FTZ` | 80 | scale·acc + bias |
| `F2IP.S8.F32.NTZ.RELU` | 40 | **pool-max + ReLU + int8 quant fused in one op** |
| `BAR.SYNC` | 3 | single staged mainloop, two reorg barriers |

## 2. Decoded structure

Prologue (`0x000–0x370`): tid decode, packed weight load via 8×`LDG.E.64`, 12×
`LDG.E.128` activation/weight tiles into registers (R6 = activation base, R8 =
weight base from `c[0x0][0x180]`). CTAID.X/Y/Z select the 8×120 output tile and
OC slab; a `@!P0 BRA` splits a fast interior path from a bordered path.

Mainloop (`~0x2310+`): one long straight-line IMMA region. Operands stay in
registers and are reused (`R88.reuse.ROW`, etc.); accumulators R20/R52/R56/R88…
seed from `RZ` then chain. 240 IMMA = 10 K-groups (147→160 ipw, k16) ×
24 (M,N) frags covering tile_rows_8 OC rows over the 120 conv columns. No
`LDSM`; operands are byte-packed in registers, fed by wide LDG.

Epilogue: per accumulator `I2FP` → `FFMA.FTZ`(scale,bias) → then **pool fold**:
`F2IP.S8.F32.NTZ.RELU Rd, Ra, Rb, Rc` quantizes max(Ra,Rb) with ReLU and chains
the running pool max in `Rc`. 40 F2IP chain the 3×3 conv outputs per pooled pixel
to int8. Only 2 `STG.E.64` — writes pooled output directly, never the 64×112×112.

## 3. Key reproduction principles

1. Fuse `ReLU(MaxPool(conv))`; never materialize conv. Chain pool-max into the
   int8 quant (mimics F2IP) so dequant happens once, pool is on f32 candidates.
2. Wide tile: 8 OC-rows × 120 conv-cols/CTA, sized so each weight fragment
   feeds many conv points → straight-line IMMA, high reuse, ~240 MMA/CTA.
3. K padded to 160, packed int8x4, fed from registers/smem; wide LDG.128/64.
4. Single shared stage + ≤3 barriers; no atomics, ≤2 STG of pooled output.

## 4. Candidate schemes (ranked)

- **S1 (chosen, v35):** OC32 packed-A/B PTX `mma.m16n8k32`, 4×4 pool tile,
  pool-max chained on accumulators before single shift+ReLU+clamp store.
  Extends v24/v34; lowest risk to err=0. Target < 0.020 ms.
- **S2:** widen N to ~120 conv-cols/CTA with cp.async double-buffer; closer to
  TRT but high register/smem risk on sm_86.
- **S3:** mma.sync wide-OC (64) single-pass + register pool; matches F2IP folding
  but pressure heavy. Future after S1 validated.

Validation: `max_abs_err=0` vs baseline; compare to v24 `0.0238` / v34 `0.0229`.

## 5. v35 result

`bench_resnet_stem_v35.cu` implements S1 with one CTA owning the full 64-OC slab
(`OC_GROUPS=4`) for TRT-style weight reuse + int16 raw-acc pool, single store.
Measured RTX 3080 Ti: `oc64_raw_acc16_pool_4x4_w4_b256 ≈ 0.0196 ms`, err=0 —
best custom kernel, ~1.8× the TRT core (`0.0108`). 6x6 tile regresses (smem
pressure). Next: cp.async double-buffer + widen N toward 120 (S2).

## 6. v35 SASS vs TRT — remaining gaps

| metric | TRT core | v35 oc64 4x4 | gap driver |
|---|---:|---:|---|
| REG | 128 | 78 | v35 leaves ILP/work on table |
| IMMA/CTA | 240 | ~40 | TRT 8×120 tile, straight-line; v35 small tile |
| LDG.128/64 | 12/26 | 0/0 | **v35 byte-only loads (88× LDG.E)** |
| LDS/STS | 50/20 | ~251/~34 | **v35 restages conv-acc in smem, pools by re-read** |
| store | 2 STG | direct | ok |

v36 plan: (1) vectorize input/weight loads to `LDG.128/64`; (2) pool in registers
via warp shuffle instead of int16 smem tile — drop one smem buffer + barrier,
mimic F2IP max chain; (3) more work/thread to lift REG→~128 & IMMA/CTA; then
widen N toward 120 with cp.async double-buffer (S2).

## 7. v36 attempts (regressed; v35 stays best)

- Stage 64-OC weights into shared via uint4 → 0.029 ms (weights are L2-cached &
  not the bottleneck; added LDS for A made it worse).
- Transpose acc tile [N][OC] for vectorized pool → 0.032 ms (strided MMA-epilogue
  stores + bank conflicts dominate). Conclusion: bottleneck is not weight loads
  or pool layout; it's tile size / occupancy. Real gain needs S2: 8×120 wide-N
  tile + cp.async double-buffer + straight-line IMMA (matches 240 IMMA/CTA). v35
  `0.0196 ms` remains best (1.8× TRT).

## 8. v37 wide-tile (S2) result

Dynamic shared lets one CTA take a bigger tile. Measured: 4x4 `~0.0184 ms`
(best), 7x7 `~0.027`, 8x8 `~0.033` — **bigger tiles regress**: int16 conv-acc +
b4 staging needs 15–83 KB shared, crushing occupancy. TRT fits in 9 KB by NOT
staging a conv-acc tile; it pools in registers via the F2IP max chain. The final
~1.7× gap is the shared-tile pool. Parity needs register/warp-shuffle pool (no
int16 smem tile), 8-N-batched; v37 4x4 is new best.

## 9. v38 SASS pinpoint — pool re-read is the wall

v38 (int8 tile, dynamic smem) ties v37 at ~0.018 ms; SASS of the 4x4 kernel:

| metric | TRT core | v38 4x4 | gap |
|---|---:|---:|---|
| IMMA | 240 | 40 | small tile |
| LDS | 50 | **596** | each pool pixel re-reads 9×64 from smem |
| STG | 2 | 64 | byte stores, not vectorized |
| LDG.128/64 | 12/26 | 0/0 | byte loads |
| BAR | 3 | 2 | ok |

Reducing shared (int16→int8) did NOT speed up → not shared-size bound; the
plateau (~0.018, all of v31/v35/v37/v38) is the **596 pool LDS**. TRT avoids it
by pooling on registers via the F2IP max chain (50 LDS). Parity requires
eliminating the smem conv tile: lane-remap so a warp's 8 MMA N-points form pool
windows, chain max in registers, vectorize stores → drop LDS 596→~50, STG 64→2.

## 10. v39 — pool LDS is NOT the limiter (key result)

Transposed tile [N][OC] cut pool **LDS 596→56 (matches TRT's 50)** but ran
*slower* (~0.021 vs 0.018): strided int8 MMA-epilogue stores cost more than the
LDS saved. Decisive: ~0.018 ms (all of v31/v35/v37/v38) is a true plateau —
limited by per-CTA MMA schedule (40 IMMA + overhead), not by pool reads. Closing
to TRT 0.0108 needs the full implicit-GEMM rewrite: 8×120 tile, 240 straight-line
IMMA, 128 reg, cp.async, F2IP register pool — a CUTLASS-scale kernel. Plateau
best ~0.018 ms ≈ 1.7× TRT; v36/v39 are instructive regressions.

## 11. v40 — 4-N/warp straight-line (regressed)

Hoisting A reuse to 4 N-pairs/warp pushed acc to 64 regs and idled 5/8 warps at
4x4 → 0.029 ms. Sixth distinct scheme to plateau/regress (v31/35/37/38 ≈0.018;
v36/39/40 worse). Verdict: this packed-MMA+shared family is exhausted at ~0.018;
TRT parity needs a ground-up implicit-GEMM kernel (8×120, 240 straight IMMA, 128
reg, cp.async pipeline, register F2IP pool). Best stable: **v38 ≈ 0.0183 ms**.
