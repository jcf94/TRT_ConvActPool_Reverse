# Blackwell (RTX 5090, sm_120) TensorRT ConvActPool Profile & Reproduction

This note records the Blackwell counterpart of the sm_86 reverse-engineering
work: profiling TensorRT's fused INT8 `ConvActPool` on an RTX 5090 and
reproducing (and beating) it with a hand-written CUDA kernel.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, compute capability **12.0 (sm_120)** |
| Driver | 595.71.05 |
| OS | Ubuntu 22.04.5 LTS |
| CUDA toolkit | 12.8 (`nvcc` V12.8.93), `cuobjdump` for sm_120 SASS |
| TensorRT | **11.1.0.106** (`tensorrt` pip wheel, cu13 libs bundled) in conda env `torch_env` (Python 3.11) |
| Host | `ssh -p 47244 root@connect.westc.seetacloud.com`, workspace `~/autodl-tmp/trt_reverse` |

`ncu` hardware counters are unavailable, so SASS / resource usage and event
timing are the profiling signals — same constraint as the sm_86 box.

## Operator (unchanged)

`Conv 7x7 s2 p3 -> ReLU -> MaxPool 3x3 s2 p1`, INT8.
Input `1x3x224x224` -> conv `1x64x112x112` -> pool `1x64x56x56`.

## TensorRT 11 differences that matter

TensorRT 11 removed the **implicit-quantization** path used on sm_86:
`BuilderFlag.INT8` / `FP16` and the `IInt8EntropyCalibrator2` API are gone.
TensorRT 11 is strongly typed — INT8 must be expressed entirely with ONNX Q/DQ
nodes. Consequences observed:

- The repo's `resnet_stem_block_qdq.onnx` only quantizes the **conv output**, so
  TRT 11 ran the convolution itself in **TF32**
  (`sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f32f32_tf32f32_f32 ...`) plus an
  FP32 `sm50_xmma_pooling_max` — *not* a fused INT8 kernel.
- To obtain a fused INT8 `ConvActPool`, every conv operand must be quantized.
  `src/blackwell/make_stem_int8_qdq_onnx.py` emits a fully-quantized graph
  (Q/DQ on input, per-output-channel DQ on int8 weights, Q/DQ on the ReLU output
  before the pool). This is what triggers the fused cask kernel below.

Runner: `src/blackwell/run_trt_sm120.py` (TRT-11-compatible; no calibrator, no
INT8 flag, optional `--strongly-typed`).

## TensorRT fused-core profile

Fully-quantized engine `results/stem_int8_sm120.plan`, per-layer GPU time via
`IProfiler` (20k iters):

| Layer (myelin) | Role | mean ms |
| --- | --- | --- |
| `__myl_MulMinMaxRounCast` | input quantize/reformat | ~0.0061 |
| `__mye*` (x2) | small reformats | ~0.0014 / ~0.0020 |
| **`__mye326_conv_act_pool`** | **fused Conv+ReLU+MaxPool (INT8)** | **~0.0082–0.0092** |
| `__myl_CastMulMoveResh` | output reformat | ~0.0034 |
| whole engine | — | ~0.046 |

The fused core kernel is, as on sm_86, a cask `conv_act_pool` kernel — TRT has no
native sm_120 variant and reuses the Ampere kernel compiled to sm_120 SASS:

```
sm80_trt_conv_act_pool_v3_tile_rows_4_tile_cols_116_execute_kernel_trt
  relu=true relu_low=0  pool 3x3 pad1  source=cask5
  REG:122  SHARED:8064  CONSTANT:968
  IMMA=144  I2FP=48  F2IP=24  FFMA=48  LDG=26  LDS=37  STS=4  STG=1  BAR=3
  (no DP4A, no HMMA — INT8 IMMA.16816 tensor cores)
```

Note the tile differs from sm_86 (`tile_rows_8_tile_cols_120`): Blackwell selected
`tile_rows_4_tile_cols_116`. SASS saved at
`results/blackwell/trt_conv_act_pool_sm120.sass`.

**Comparison target: the fused core, ~0.0082–0.0092 ms** (stable ~0.0092 ms at
20k iters), not whole-engine time.

## Hand-kernel reproduction

Sources live in `src/blackwell/`. They reuse the existing harness
(`src/resnet_stem_common.cuh`: shapes, `CUDA_CHECK`, CPU reference, event timing,
packed-MMA weights) and the v72 fused design (cp.async K-stream + IMMA core +
fanned `vmax4` pool epilogue, NHWC vectorized stores). Only the timed fused core
is measured; input im2col packing is an untimed reformat, mirroring TRT's
separate quantize/reformat layers.

### Porting fix (v1)

The sm_86 `v72` kernel crashed on Blackwell with an illegal memory access. Cause:
the post-conv shared tile `cr_s` was sized for `N_TILE = 126` cells, but the
epilogue indexes up to `n = 127` (two halo cells that are written but never read
by pooling). Benign on sm_86, faulted on sm_120 (`compute-sanitizer`: "Invalid
__shared__ write"). Fix: size `cr_s` to `CR_CELLS = NG*8 = 128` cells.

### Iteration ladder (RTX 5090, sm_120, 20k iters, `max_abs_err=0`)

| Version | Change | Warps / threads | mean ms | vs TRT core |
| --- | --- | --- | --- | --- |
| TRT core | `sm80_trt_conv_act_pool_v3` 4x116 | — | ~0.0092 | 1.00x |
| `sm120_v1` | v72 port + `cr_s` bounds fix | 4 / 128 | 0.00560 | 1.64x |
| **`sm120_v2`** | **8-warp occupancy** | **8 / 256** | **0.00499** | **1.84x** |
| `sm120_v3` | 16-warp occupancy | 16 / 512 | 0.00589 | 1.56x |

`sm120_v2` is the Blackwell optimum: REG:61, SHARED:21504 B, `max_abs_err=0`,
stable at **0.00499–0.00500 ms** across runs. Blackwell prefers more warps per CTA
than sm_86 (8 vs v72's 4); 16 warps over-subscribes and regresses. The 5090 has
~170 SMs and the grid is 140 CTAs (one wave), so the win is intra-CTA latency
hiding, not more blocks.

### Conclusion

On Blackwell the hand-written fused kernel **already exceeds** TensorRT's fused
`ConvActPool` core by ~1.8x with bit-exact output, consistent with the sm_86
finding that TRT's serial pool epilogue is the bottleneck and a fanned-pool +
cp.async IMMA core beats it. TRT did not specialize the kernel for Blackwell (it
reuses the `sm80_` cask kernel), which widens the gap relative to sm_86.

## Validation: standalone / un-fused INT8 conv

To confirm where the time goes, `src/blackwell/make_stem_int8_unfusable_onnx.py`
denies TRT the Conv->ReLU->MaxPool fusion two ways and we re-profile (20k iters):

| Graph | Conv kernel chosen | conv ms | pool ms | notes |
| --- | --- | --- | --- | --- |
| fused (reference) | `sm80_trt_conv_act_pool_v3_4x116` (cask) | — | — | whole conv+relu+pool = **~0.0092** |
| **conv only** (no pool) | `sm80_xmma_fprop_first_layer_i8i8_i8i32 ... tensor16x8x16_r7s7_u2v2` | **0.00869** | — | genuine INT8 conv (REG:240, SHARED:1024) |
| **no-fuse** (conv + separate pool) | same INT8 `first_layer` conv | 0.00908 | 0.00669 (`sm50_xmma_pooling_tiled_INT8NCxHW4`) | + several Move/Tran/Cast reformat kernels |

Whole-engine means: fused 0.046 ms, conv-only 0.058 ms, no-fuse **0.094 ms**.

Takeaways:

1. **The pool is nearly free inside the fused kernel.** TRT's standalone INT8
   conv (~0.0087 ms) costs essentially the same as the fused Conv+ReLU+MaxPool
   (~0.0092 ms): folding ReLU+MaxPool into the conv epilogue adds only ~0.0005 ms.
2. **Un-fusing is expensive.** Forced apart, TRT must add a separate INT8 pool
   kernel (~0.0067 ms) plus NCHW<->NHWC reformat/transpose kernels, nearly
   doubling end-to-end time (0.094 vs 0.046 ms).
3. **The hand kernel does the whole op faster than TRT does the conv alone.**
   `sm120_v2` (0.00499 ms, conv+relu+pool fused, err=0) beats TRT's *standalone
   INT8 conv* (0.00869 ms) by ~1.74x. The advantage is the tighter fused schedule
   (cp.async IMMA core + fanned `vmax4` pool epilogue), not skipped work — TRT's
   own fused cask core is the right comparison and the hand kernel still leads it
   by ~1.8x.

Reproduce:

```bash
python src/blackwell/make_stem_int8_unfusable_onnx.py
python src/blackwell/run_trt_sm120.py --onnx models/resnet_stem_int8_conv_only.onnx \
  --save-engine results/stem_conv_only_sm120.plan --warmup 200 --iters 20000
python src/blackwell/run_trt_sm120.py --onnx models/resnet_stem_int8_no_fuse.onnx \
  --save-engine results/stem_no_fuse_sm120.plan --warmup 200 --iters 20000
```

## Reproduce

```bash
# On the 5090 host (env torch_env, CUDA 12.8):
export LD_LIBRARY_PATH=$(python -c 'import tensorrt,pathlib;print(pathlib.Path(tensorrt.__file__).resolve().parent.parent/"tensorrt_libs")'):$LD_LIBRARY_PATH

# 1. Fully-quantized INT8 QDQ ONNX -> engine + per-layer profile
python src/blackwell/make_stem_int8_qdq_onnx.py
python src/blackwell/run_trt_sm120.py --onnx models/resnet_stem_int8_qdq.onnx \
  --save-engine results/stem_int8_sm120.plan --warmup 200 --iters 5000

# 2. Extract fused-core SASS
python scripts/extract_engine_fatbin.py results/stem_int8_sm120.plan \
  -o results/engine_elf/int8_fatbin.bin
cuobjdump -sass -arch sm_120 results/engine_elf/int8_fatbin.bin \
  > results/blackwell/trt_conv_act_pool_sm120.sass
python scripts/sass_summary.py --cuobjdump cuobjdump --arch sm_120 \
  results/engine_elf/int8_fatbin.bin

# 3. Build + run the hand kernels
src/blackwell/build_sm120.sh src/blackwell/bench_stem_sm120_v2.cu --warmup 500 --iters 20000
```
