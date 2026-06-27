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

The v4 result shows that tensor-core INT8 convolution is necessary to approach
TensorRT. It also shows why plain GEMM is not enough: materializing the full
`64x112x112` convolution output and then pooling it costs about another
`0.02 ms`.

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
- Future attempts should use `src/bench_resnet_stem_v5.cu`,
  `src/bench_resnet_stem_v6.cu`, and so on.

## Notes

- The remote machine has an RTX 3080 Ti, so the default CUDA architecture is
  `sm_86`.
- `nvcc` is under `/usr/local/cuda-11.8/bin`.
- TensorRT was not preinstalled at the start of this run. The remote now has
  `tensorrt-cu11==10.10.0.31` installed via pip.
- The first fused kernel is a correctness baseline: it avoids writing the
  intermediate `64x112x112` tensor, but recomputes overlapping convolution
  points across adjacent pool windows. Use it as a starting point for tiled
  shared-memory/register reuse optimization.
