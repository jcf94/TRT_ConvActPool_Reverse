# CUDA Operator Benchmark

This is a reproducible CUDA benchmark harness for iterating on high-performance
operators and comparing them with a framework/TensorRT baseline.

Current operator: ResNet stem in int8:
`Conv 7x7 stride 2 pad 3 -> ReLU -> MaxPool 3x3 stride 2 pad 1`.

The input/output shapes match the first stage of ResNet:

- input: `1x3x224x224`
- conv output: `1x64x112x112`
- maxpool output: `1x64x56x56`

## Remote build

```bash
cd /root/cuda_op_bench
./scripts/build.sh
./build/bench_resnet_stem --iters 1000 --warmup 100
./scripts/run_resnet_stem.sh
```

## TensorRT Reproduction

The original observation is that TensorRT INT8 can fuse the first ResNet
`Conv + ReLU + MaxPool` into one kernel. Inserting `QuantizeLinear` and
`DequantizeLinear` between `Conv` and `ReLU` blocks that fusion and makes the
profile split into multiple kernels.

This repo generates two ONNX models for that test:

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
| plain `Conv->ReLU->MaxPool` | `0.035513 ms` | fused layer `node_of_conv_out + node_of_relu_out + node_of_output`: `0.009854 ms`; input reformat `0.007416 ms`; output reformat `0.007835 ms` |
| Q/DQ-blocked model | `0.073361 ms` | `node_of_conv_out`: `0.031945 ms`; `node_of_conv_q`: `0.006278 ms`; `node_of_conv_dq`: `0.004865 ms`; `PWN(node_of_relu_out)`: `0.005055 ms`; `node_of_output`: `0.005161 ms` |

Conclusion: the reproduction matches the article's behavior. The plain graph is
compiled into one fused TensorRT layer, while the Q/DQ graph is split and is
roughly 2x slower at the engine level.

See [TensorRT Fused Core Profile Notes](docs/tensorrt_fused_core_profile.md) for
the detailed profiling record. ncu was installed and attempted, but hardware
counter collection is blocked in the current container by host driver setting
`RmProfilingAdminOnly=1` and missing `CAP_SYS_ADMIN`.

The TensorRT engine was also inspected at the SASS level by extracting the
embedded fatbin from the serialized engine. The fused core cubin contains
`sm80_trt_conv_act_pool_v3_tile_rows_8_tile_cols_120_execute_kernel_trt`, uses
`IMMA.16816.S8.S8` INT8 tensor-core instructions, and has no `DP4A`
instructions. This confirms that matching TensorRT's core requires an INT8 MMA
schedule, not just further scalar-DP4A tuning.

## CUDA Baseline

The custom CUDA benchmark currently includes:

- `conv_only_i32`: direct INT8 convolution writing `int32` output.
- `conv_relu_pool_separate`: direct convolution + ReLU writes the full
  `64x112x112` intermediate, then a separate maxpool kernel writes `64x56x56`.
- `fused_recompute_v1`: correctness baseline that writes only the final pooled
  output but recomputes overlapping convolution points.
- `fused_tiled_*`: computes a convolution tile into shared memory, applies
  ReLU, then pools from shared memory.

Latest 200-iteration sanity run:

| kernel | mean |
| --- | ---: |
| `conv_only_i32` | `0.118656 ms` |
| `conv_relu_pool_separate` | `0.122035 ms` |
| `fused_recompute_v1` | `0.264417 ms` |
| `fused_tiled_8x8` | `0.152492 ms` |
| `fused_tiled_14x14` | `0.129264 ms` |
| `fused_tiled_28x28` | `0.145203 ms` |
| `fused_tiled_56x56` | `0.172999 ms` |

The best custom fused kernel is currently close to the separate direct-CUDA
baseline, but still far from TensorRT's fused kernel. The gap is expected:
TensorRT is almost certainly using a more specialized INT8 convolution strategy
than this direct per-output-point DP4A implementation.

Latest optimization results:

| version | best relevant result | note |
| --- | ---: | --- |
| baseline | `fused_tiled_14x14 = 0.123176 ms` | direct DP4A, weights packed inside each output-point calculation |
| v2 | `fused_tiled_14x14 = 0.081843 ms` | prepacked DP4A weights; currently best custom fused direct kernel |
| v3 | `fused_tiled_56x56 = 0.108958 ms` | interior fast path was a regression versus v2 |
| v4 | `cublas_int8_gemm = 0.019873 ms`; `cublas_gemm_relu_pool = 0.039794 ms` | tensor-core upper-bound experiment with prebuilt im2col; not a fused operator |
| v5 | `fused_tiled_oc4_14x14 = 0.044329 ms` | computes 4 output channels per thread for each conv point; reuses input packing |
| v6 | `fused_tiled_oc8_8x8 = 0.032353 ms` | computes 8 output channels per thread; best direct-DP4A fused kernel so far |
| v7 | `fused_tiled_oc16_14x14 = 0.038880 ms` | OC16 increases pressure and regresses versus OC8 |
| v8 | `fused_tiled_oc8_8x8_b128 = 0.031077 ms` | OC8 with 128 threads per block; current best |
| v9 | `fused_tiled_oc8_10x10_b128 = 0.035623 ms` | 10x10/12x12 spatial tile sweep did not beat v8 |
| v10 | `oc8_smem_input_8x8_b128 = 0.034136 ms` | staging the input tile in shared memory regressed versus v8 |
| v11 | `fused_tiled_oc8_8x8_b128 = 0.030482 ms` | additional 6x6/7x7/16x16 and block-size sweeps did not beat v8 |
| v12 | `ptx_mma_oc16_8x8_w8_b256 = 0.055870 ms` | first inline PTX `mma.sync.aligned.m16n8k32` fused kernel; correct but fragment construction dominates |
| v13 | `ptx_mma_oc32_smem_b_6x6_w8_b256 = 0.049618 ms` | stages activation/im2col B tile in shared memory and reuses it across two OC16 MMA groups |
| v14 | `ptx_mma_oc32_smem_b_4x4_w4_b128 = 0.043008 ms` | smaller tile improves shared-B staging cost and warp utilization |
| v15 | `ptx_mma_oc64_smem_b_4x4_w4_b128 = 0.041651 ms` | reuses one B tile across all 64 output channels; best PTX MMA so far |
| v16 | `ptx_mma_oc64_smem_b_4x4_w4_b128 = 0.044665 ms` | channel-split pooling epilogue regressed versus v15 |

The v4 result shows that tensor-core INT8 convolution is necessary to approach
TensorRT. It also shows why plain GEMM is not enough: materializing the full
`64x112x112` convolution output and then pooling it costs about another
`0.02 ms`.

The v5-v9 results show the useful limit of the direct-DP4A path on this
operator: reusing input packing across output channels is important, but the
best direct kernel is still about 3x slower than TensorRT's CASK fused core.
Further large gains require an INT8 MMA/tensor-core schedule.

The SASS dump suggests TensorRT is not just using a larger DP4A tile. The fused
CASK kernel has 128 registers/thread, about 9 KB of shared memory, and a long
sequence of `IMMA.16816.S8.S8` instructions fed by wide global loads with
register reuse. The likely implementation is an implicit-GEMM convolution
mainloop over output channels and spatial positions, using INT8 tensor cores and
keeping the ReLU/MaxPool epilogue close enough to the accumulator data that the
full `64x112x112` intermediate is never written. v10 and v11 tested two cheaper
DP4A hypotheses from that analysis: explicit shared-memory input staging and
more launch-geometry sweep. Both regressed, so the remaining gap is primarily
the tensor-core mainloop and hand-scheduled data movement, not a simple cache or
CTA-shape issue.

v12 starts the tensor-core path with inline PTX
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. Each warp computes a
`16 output-channel x 8 conv-position` tile over `K=160` padded elements, writes
the ReLUed conv tile to shared memory, and then performs MaxPool. This validates
the lane/register fragment mapping with `max_abs_err=0`, but it is slower than
the DP4A kernel because every lane rebuilds A/B MMA fragments directly from
global/input indexing for each K step. The next optimization should keep this
PTX MMA instruction path but add staged/reused activation fragments and a less
expensive epilogue.

v13-v15 confirm that activation-fragment reuse matters: staging the B/im2col
tile once per CTA and reusing it across OC groups improves the PTX MMA path from
`0.055870 ms` to `0.041651 ms`. However, it is still slower than the direct DP4A
kernel. The remaining gap is likely in the mainloop details that TensorRT's CASK
kernel hand-schedules: register-level fragment construction, fewer shared-memory
round trips, better instruction interleaving around `IMMA`, and larger spatial
tiles without exploding epilogue cost. v16 tried to split the pooling epilogue by
channel to reduce per-thread registers, but the added shared-memory traffic and
thread scheduling regressed.

## Optimization Plan

The next target is TensorRT's fused layer time, about `0.01 ms` on the current
machine. The planned sequence is:

1. Remove overhead in the direct fused implementation: split interior and border
   paths so the common case avoids padding branches and repeated index checks.
2. Improve data layout: prepack weights and load input vectors in a layout that
   maps cleanly to DP4A.
3. Increase work per CTA: compute multiple output channels and/or neighboring
   spatial points per block to reuse input data and reduce scheduling overhead.
4. Move beyond scalar DP4A if needed: use an implicit-GEMM or tensor-core INT8
   strategy, then fuse ReLU and pooling around that schedule.

Each optimization attempt lives in a separate source file:

- `src/bench_resnet_stem.cu`: baseline implementation.
- `src/bench_resnet_stem_v2.cu`: v2 attempt with prepacked DP4A weights.
- `src/bench_resnet_stem_v3.cu`: v3 attempt with prepacked DP4A weights plus
  an interior fast path that avoids padding checks.
- `src/bench_resnet_stem_v4.cu`: v4 tensor-core upper-bound experiment using
  prebuilt im2col plus cuBLAS INT8 GEMM, followed by a ReLU+MaxPool kernel.
- `src/bench_resnet_stem_v5.cu`: v5 OC4 direct-DP4A fused tile attempt. Each
  thread computes four output channels for the same spatial conv point to reuse
  input packing.
- `src/bench_resnet_stem_v6.cu`: v6 OC8 direct-DP4A fused tile attempt.
- `src/bench_resnet_stem_v7.cu`: v7 OC16 direct-DP4A fused tile attempt.
- `src/bench_resnet_stem_v8.cu`: v8 OC8 block-size sweep for 256/128/64
  threads per block.
- `src/bench_resnet_stem_v9.cu`: v9 OC8 spatial tile-size sweep for 10x10 and
  12x12 tiles.
- `src/bench_resnet_stem_v10.cu`: v10 shared-memory input staging experiment
  for the OC8 DP4A fused tile.
- `src/bench_resnet_stem_v11.cu`: v11 extra OC8 DP4A tile/block sweep.
- `src/bench_resnet_stem_v12.cu`: v12 first inline PTX INT8 MMA fused kernel
  using `m16n8k32`.
- `src/bench_resnet_stem_v13.cu`: v13 OC32 PTX MMA with shared-memory B-tile
  reuse.
- `src/bench_resnet_stem_v14.cu`: v14 PTX MMA tile-size sweep for shared-B
  staging.
- `src/bench_resnet_stem_v15.cu`: v15 OC64 PTX MMA shared-B reuse, current best
  PTX MMA version.
- `src/bench_resnet_stem_v16.cu`: v16 channel-split pooling epilogue experiment.
- Future attempts should use `src/bench_resnet_stem_v17.cu`,
  `src/bench_resnet_stem_v18.cu`, and so on.

## Notes

- The remote machine has an RTX 3080 Ti, so the default CUDA architecture is
  `sm_86`.
- `nvcc` is under `/usr/local/cuda-11.8/bin`.
- TensorRT was not preinstalled at the start of this run. The remote now has
  `tensorrt-cu11==10.10.0.31` installed via pip.
- SASS extraction is reproducible with `scripts/extract_engine_fatbin.py`
  followed by `cuobjdump -sass -arch sm_86` on the extracted fatbin.
- The first fused kernel is a correctness baseline: it avoids writing the
  intermediate `64x112x112` tensor, but recomputes overlapping convolution
  points across adjacent pool windows. Use it as a starting point for tiled
  shared-memory/register reuse optimization.
