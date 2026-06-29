# TRT `CaskConvActPool` SASS: Full Reverse Walkthrough + Direct Replica (v45)

Date: 2026-06-29 · RTX 3080 Ti (sm_86) · CUDA 11.8 · TensorRT 10.10
Kernel: `sm80_trt_conv_act_pool_v3_tile_rows_8_tile_cols_120_execute_kernel_trt`
SASS: [../results/trt_conv_act_pool_sm86.sass](../results/trt_conv_act_pool_sm86.sass)
(2552 lines, single function). Core layer ≈ `0.0108 ms`.

This supersedes the summary-level notes in
[trt_kernel_implementation_schemes.md](trt_kernel_implementation_schemes.md);
it decodes the kernel address-range by address-range and maps each phase to the
`Conv7x7s2 -> ReLU -> MaxPool3x3s2` math, then specifies v45 as a from-scratch
replica.

---

## 0. Resource & instruction census

`REG 128 · SHARED 9008 B · LOCAL/STACK 0 · CONST[0] 424 B`.

| family | count | role |
|---|---:|---|
| `IMMA.16816.S8.S8` | 240 | mainloop, m16·n8·k16, ROW×COL, RZ-seeded then chained |
| `DP4A`/`HMMA` | 0 | pure tensorcore path |
| `LDG.E.128` | 12 | activation tile rows (one row = 0x200 stride) |
| `LDG.E.64` | 26 | 8 packed-weight loads + bias/scale + tails |
| `STS.128`/`STS.64` | 9/11 | stage packed A/B fragments to smem |
| `LDS`/`LDS.64` | 48/2 | feed IMMA operands |
| `STG.E.64` | 2 | **only** writes 64×56×56 pooled output |
| `I2FP.F32.S32` | 80 | acc(int32)→f32 dequant |
| `FFMA.FTZ` | 80 | `f32 = scale·acc + bias` |
| `F2IP.S8.F32.NTZ.RELU` | 40 | **dequant→ReLU→int8 + running pool-max, fused** |
| `BAR.SYNC` | 3 | one staged mainloop + two epilogue reorg barriers |

Two STG for the whole layer is the headline: the conv result (64×112×112) is
never materialized in DRAM; pooling collapses it before the store.

---

## 1. Constant-bank parameter map (`c[0x0][...]`)

Decoded from how each slot feeds addresses/loops:

| const | meaning |
|---|---|
| `0x178/0x17c` | activation (input) base ptr → `R6` |
| `0x180` | packed-weight base ptr → `R8` (8×`LDG.E.64` = 64 B/CTA prologue) |
| `0x198/0x19c` | output base ptr (final store) |
| `0x164` | conv/pool width param (used `/2` and `*-4` in tile math) |
| `0x168` | conv/pool height param (`/2`, `*-0x3c=-60`) → 60 cols, 8 rows |
| `0x1a0` | mode flag (=1 single-CTA-Z), gates the border vs interior split |
| `0x10`,`0x18` | grid extents for tile-cols/rows bound checks |
| `0x118` | 64-bit env/descriptor base |

`0x3c=60` and `8` confirm the tile name: **8 rows × 60 dual-col = 120 conv
cols**. CONST[0]=424 B holds per-OC bias+scale used by FFMA/I2FP.

---

## 2. Phase map by address range

```
0x000-0x370  Prologue: tid decode, 8x LDG.E.64 weights, 12x LDG.E.128 acts,
             CTAID.{X,Y,Z} -> tile select, @!P0 BRA 0x11a0 (border vs interior)
0x380-0x1190 Staging loop: pack acts -> smem (STS.64/.128), loop BRA 0x400,
             border-path fixups; merges at 0x11a0
0x11a0-0x1ee0 second (border) staging path -> BAR.SYNC 0x1ef0
0x1f00-0x2050 tile/range bound checks, empty-tile @P0 BRA 0x4e20 (EXIT)
0x2270-0x2300 LDS load A/B fragments
0x2310-0x4260 MAINLOOP: 240 IMMA straight-line, interleaved I2FP/FFMA/F2IP
0x2830 first F2IP, ... epilogue interleaved; BAR.SYNC 0x4450/0x4a10
0x45b0-0x4740 epilogue 1: I2FP->FFMA->F2IP pool-max, packed-byte max via 0x80..
0x4750-0x4910 STS staging of pooled bytes + packed max across pool window
0x4920-0x4e10 final 2x STG.E.64 of 64x56x56 int8 output
0x4e20 EXIT
```

Single `BSSY/BSYNC` pair; no `LDSM`, no `cp.async` in the sm_86 cubin (the
mainloop is fully unrolled, fed by wide LDG + smem stage).

---

## 3. Prologue (0x000–0x370): one CTA grabs an 8×120 tile

- `R100=tid`, `R0=tid>>5` (warp), shuffle-broadcast → CTA-uniform; `tid*16`,
  `tid*8` build smem offsets.
- `R8 = c[0x180] + tile*0x20`; eight `LDG.E.64 [R8+0..0x38]` = 64 B = the OC
  slab's packed int8 weights into R110–R118 (reused all mainloop, never reloaded
  → the 12/26 wide-LDG count, vs custom kernels' byte loads).
- `R6 = c[0x178] + tile*0x1800`; twelve `LDG.E.128 [R6+0x000..0x1600]`
  (stride 0x200) = 12 activation rows × 16 B. 8×120 tile reuses these per OC.
- CTAID.X→cols, CTAID.Z→OC slab (`R8*0x164`), CTAID.Y→rows. `R6=R6*3` (×IC).
- `@!P0 BRA 0x11a0`: P0 from `c[0x168]&7` → fast interior vs bordered (pad)
  path, exactly the v38 `interior` predicate.

## 4. Staging (0x400–0x1ee0): pack acts into 9 KB smem, single BAR

Loop body packs RGB activations as `int8x4` (`STS.64`, `STS.128`) into the 9008 B
buffer; weights stay in registers. `BAR.SYNC 0x1ef0` is the only mainloop
barrier. Bound checks at 0x1f00–0x2050; degenerate tiles jump to EXIT.

## 5. Mainloop (0x2310–0x4260): 240 IMMA, register reuse

```
LDS R88..R11        ; B (acts) fragments from smem, 0x330 stride = next OC sub
IMMA R56, R88.reuse.ROW, R28.COL, RZ   ; seed
IMMA R20, R88.reuse.ROW, R32.COL, RZ
IMMA R52, R88.reuse.ROW, R24.COL, RZ
IMMA R88, R88.ROW,      R36.COL, RZ
IMMA R56, R8.reuse.ROW, R29.COL, R56   ; chain k+1 ...
```
240 IMMA = 10 K-steps (147→160, k16) × 24 (M,N) frags for 8 OC-rows × 120 conv
cols. `.reuse` keeps one A op live across 4 N → straight-line, ~128 reg ILP. RZ
on first k, chain on rest. No per-iter pointer math (fully unrolled).

## 6. Epilogue: dequant + ReLU + pool fused (the key)

Per acc: `I2FP R,acc` (int32→f32) → `FFMA R, scale, R, bias`. Then the fold:
```
F2IP.S8.F32.NTZ.RELU R88, R89, R88, RZ   ; int8(relu(max(R89,R88)))
F2IP.S8.F32.NTZ.RELU R88, R93, R92, R88  ; chain prev pool-max in Rc
```
`F2IP(d,a,b,c)` = quantize `max(a,b)` with ReLU to int8 and pool-max into c.
40 F2IP chain the 9 conv values per pooled pixel. Remaining cross-pixel maxes
use a **packed int8x4 max** trick (no float): `x^0x80808080; (x>>7); sub` —
select-max on 4 signed bytes at once (0x4640–0x4910). Final pooled bytes packed
and written with just two `STG.E.64` (0x4db0, 0x4e10).

`MaxPool(ReLU(conv)) = ReLU(MaxPool(conv))` lets the shift/clamp run once; ReLU
folds into F2IP, so pad=0 == ReLU floor. This is the v45 epilogue spec.

---

## 7. Why it wins (vs hand v38 ≈0.0183) and what v45 must copy

| | TRT | v38 | replica goal |
|---|---:|---:|---|
| tile | 8×120 | 8×8 | wide 8-row tile, weights reused all cols |
| IMMA/CTA | 240 | 40 | unroll to ~120 PTX mma (=240 SASS) |
| pool | F2IP reg | 596 LDS | packed-byte reg max, no conv smem |
| STG | 2 | 64 | vectorized STG.64 |

v45 = transcribe this: 8-row tile, single OC-64 slab, IMMA mainloop, then
register packed-byte pool max + vectorized store (no conv_acc smem tile).

## 8. v45 result (measured, RTX 3080 Ti)

`./build/bench_resnet_stem_v45 --iters 1000 --warmup 200`:

```
v45_trt_replica_8x7_b256   0.041355 ms  max_abs_err=0
v45_trt_replica_8x10_b256  0.051955 ms  max_abs_err=0
v45_trt_replica_8x12_b256  0.059179 ms  max_abs_err=0
```

REG 128, IMMA 20, LDS 586, STG 64, BAR 2. Correct but ~2.3× slower than v38
(0.0183). The wider 8-row tile makes the smem stage 57–95 KB, which crushes
occupancy, and the register packed pool still re-reads conv_acc from smem
(586 LDS) so the mainloop stays looped (20 IMMA, not 240 straight-line). This
reproduces the plateau verdict: copying tile shape alone regresses; TRT 0.0108
needs 240 unrolled IMMA + cp.async multistage + true F2IP register pool (no smem
re-read) at once — a CUTLASS-scale rewrite, not a transcription.

