#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

constexpr int N = 1;
constexpr int IC = 3;
constexpr int IH = 224;
constexpr int IW = 224;
constexpr int OC = 64;
constexpr int KH = 7;
constexpr int KW = 7;
constexpr int CONV_OH = 112;
constexpr int CONV_OW = 112;
constexpr int POOL_OH = 56;
constexpr int POOL_OW = 56;
constexpr int K_TOTAL = IC * KH * KW;
constexpr int K_GROUPS = (K_TOTAL + 3) / 4;

struct Args {
  int warmup = 100;
  int iters = 1000;
  bool csv = false;
};

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto take_int = [&](int& dst) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[i]);
        std::exit(2);
      }
      dst = std::atoi(argv[++i]);
    };
    if (!std::strcmp(argv[i], "--warmup")) {
      take_int(args.warmup);
    } else if (!std::strcmp(argv[i], "--iters")) {
      take_int(args.iters);
    } else if (!std::strcmp(argv[i], "--csv")) {
      args.csv = true;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  return args;
}

__host__ __device__ __forceinline__ int8_t clamp_relu_i8(int acc, int shift) {
  acc = max(acc >> shift, 0);
  acc = min(acc, 127);
  return static_cast<int8_t>(acc);
}

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ int conv_point(const int8_t* __restrict__ x,
                                          const int32_t* __restrict__ w4,
                                          int oc, int oh, int ow) {
  int acc = 0;
#pragma unroll 37
  for (int g = 0; g < K_GROUPS; ++g) {
    int k = g * 4;
    int xp = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int kk = k + j;
      int xv = 0;
      if (kk < K_TOTAL) {
        int kx = kk % KW;
        int ky = (kk / KW) % KH;
        int ic = kk / (KH * KW);
        int iy = oh * 2 + ky - 3;
        int ix = ow * 2 + kx - 3;
        if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
          xv = int(static_cast<unsigned char>(x[(ic * IH + iy) * IW + ix]));
        }
      }
      xp |= xv << (j * 8);
    }
    acc = dp4a_s32(xp, w4[oc * K_GROUPS + g], acc);
  }
  return acc;
}

__global__ void conv_only_kernel(const int8_t* __restrict__ x,
                                 const int32_t* __restrict__ w4,
                                 int32_t* __restrict__ conv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = OC * CONV_OH * CONV_OW;
  if (idx >= total) return;
  int ow = idx % CONV_OW;
  int oh = (idx / CONV_OW) % CONV_OH;
  int oc = idx / (CONV_OH * CONV_OW);
  conv[idx] = conv_point(x, w4, oc, oh, ow);
}

__global__ void conv_relu_kernel(const int8_t* __restrict__ x,
                                 const int32_t* __restrict__ w4,
                                 int8_t* __restrict__ conv_relu, int shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = OC * CONV_OH * CONV_OW;
  if (idx >= total) return;
  int ow = idx % CONV_OW;
  int oh = (idx / CONV_OW) % CONV_OH;
  int oc = idx / (CONV_OH * CONV_OW);
  conv_relu[idx] = clamp_relu_i8(conv_point(x, w4, oc, oh, ow), shift);
}

__global__ void maxpool_kernel(const int8_t* __restrict__ conv_relu,
                               int8_t* __restrict__ y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = OC * POOL_OH * POOL_OW;
  if (idx >= total) return;
  int px = idx % POOL_OW;
  int py = (idx / POOL_OW) % POOL_OH;
  int oc = idx / (POOL_OH * POOL_OW);

  int best = 0;
  for (int ky = 0; ky < 3; ++ky) {
    int cy = py * 2 + ky - 1;
    if (cy < 0 || cy >= CONV_OH) continue;
    for (int kx = 0; kx < 3; ++kx) {
      int cx = px * 2 + kx - 1;
      if (cx < 0 || cx >= CONV_OW) continue;
      int cidx = (oc * CONV_OH + cy) * CONV_OW + cx;
      best = max(best, int(conv_relu[cidx]));
    }
  }
  y[idx] = static_cast<int8_t>(best);
}

__global__ void fused_conv_relu_pool_kernel(const int8_t* __restrict__ x,
                                            const int32_t* __restrict__ w4,
                                            int8_t* __restrict__ y,
                                            int shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = OC * POOL_OH * POOL_OW;
  if (idx >= total) return;
  int px = idx % POOL_OW;
  int py = (idx / POOL_OW) % POOL_OH;
  int oc = idx / (POOL_OH * POOL_OW);

  int best = 0;
  for (int ky = 0; ky < 3; ++ky) {
    int cy = py * 2 + ky - 1;
    if (cy < 0 || cy >= CONV_OH) continue;
    for (int kx = 0; kx < 3; ++kx) {
      int cx = px * 2 + kx - 1;
      if (cx < 0 || cx >= CONV_OW) continue;
      best = max(best, int(clamp_relu_i8(conv_point(x, w4, oc, cy, cx),
                                         shift)));
    }
  }
  y[idx] = static_cast<int8_t>(best);
}

template <int TILE>
__global__ void fused_tiled_conv_relu_pool_kernel(const int8_t* __restrict__ x,
                                                  const int32_t* __restrict__ w4,
                                                  int8_t* __restrict__ y,
                                                  int shift) {
  constexpr int CONV_TILE = TILE * 2 + 1;
  __shared__ int8_t conv_relu_tile[CONV_TILE * CONV_TILE];

  int tile_px = blockIdx.x * TILE;
  int tile_py = blockIdx.y * TILE;
  int oc = blockIdx.z;
  int tid = threadIdx.x;

  for (int i = tid; i < CONV_TILE * CONV_TILE; i += blockDim.x) {
    int tx = i % CONV_TILE;
    int ty = i / CONV_TILE;
    int cx = tile_px * 2 - 1 + tx;
    int cy = tile_py * 2 - 1 + ty;
    int8_t v = 0;
    if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH) {
      v = clamp_relu_i8(conv_point(x, w4, oc, cy, cx), shift);
    }
    conv_relu_tile[i] = v;
  }
  __syncthreads();

  for (int out_i = tid; out_i < TILE * TILE; out_i += blockDim.x) {
    int lx = out_i % TILE;
    int ly = out_i / TILE;
    int px = tile_px + lx;
    int py = tile_py + ly;
    if (px < POOL_OW && py < POOL_OH) {
      int base_x = lx * 2;
      int base_y = ly * 2;
      int best = 0;
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          best = max(best, int(conv_relu_tile[(base_y + ky) * CONV_TILE +
                                              base_x + kx]));
        }
      }
      y[(oc * POOL_OH + py) * POOL_OW + px] = static_cast<int8_t>(best);
    }
  }
}

template <int TILE>
__global__ void fused_tiled_oc4_conv_relu_pool_kernel(
    const int8_t* __restrict__ x, const int32_t* __restrict__ w4,
    int8_t* __restrict__ y, int shift) {
  constexpr int OC_PER_BLOCK = 4;
  constexpr int CONV_TILE = TILE * 2 + 1;
  __shared__ int8_t conv_relu_tile[OC_PER_BLOCK][CONV_TILE * CONV_TILE];

  int tile_px = blockIdx.x * TILE;
  int tile_py = blockIdx.y * TILE;
  int oc_base = blockIdx.z * OC_PER_BLOCK;
  int tid = threadIdx.x;

  for (int i = tid; i < CONV_TILE * CONV_TILE; i += blockDim.x) {
    int tx = i % CONV_TILE;
    int ty = i / CONV_TILE;
    int cx = tile_px * 2 - 1 + tx;
    int cy = tile_py * 2 - 1 + ty;
    int acc0 = 0;
    int acc1 = 0;
    int acc2 = 0;
    int acc3 = 0;

    if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH) {
#pragma unroll 37
      for (int g = 0; g < K_GROUPS; ++g) {
        int k = g * 4;
        int xp = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int kk = k + j;
          int xv = 0;
          if (kk < K_TOTAL) {
            int kx = kk % KW;
            int ky = (kk / KW) % KH;
            int ic = kk / (KH * KW);
            int iy = cy * 2 + ky - 3;
            int ix = cx * 2 + kx - 3;
            if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
              xv = int(static_cast<unsigned char>(x[(ic * IH + iy) * IW + ix]));
            }
          }
          xp |= xv << (j * 8);
        }
        int w_base = oc_base * K_GROUPS + g;
        acc0 = dp4a_s32(xp, w4[w_base], acc0);
        acc1 = dp4a_s32(xp, w4[w_base + K_GROUPS], acc1);
        acc2 = dp4a_s32(xp, w4[w_base + 2 * K_GROUPS], acc2);
        acc3 = dp4a_s32(xp, w4[w_base + 3 * K_GROUPS], acc3);
      }
    }
    conv_relu_tile[0][i] = clamp_relu_i8(acc0, shift);
    conv_relu_tile[1][i] = clamp_relu_i8(acc1, shift);
    conv_relu_tile[2][i] = clamp_relu_i8(acc2, shift);
    conv_relu_tile[3][i] = clamp_relu_i8(acc3, shift);
  }
  __syncthreads();

  for (int out_i = tid; out_i < TILE * TILE; out_i += blockDim.x) {
    int lx = out_i % TILE;
    int ly = out_i / TILE;
    int px = tile_px + lx;
    int py = tile_py + ly;
    if (px < POOL_OW && py < POOL_OH) {
      int base_x = lx * 2;
      int base_y = ly * 2;
      int best0 = 0;
      int best1 = 0;
      int best2 = 0;
      int best3 = 0;
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          int sidx = (base_y + ky) * CONV_TILE + base_x + kx;
          best0 = max(best0, int(conv_relu_tile[0][sidx]));
          best1 = max(best1, int(conv_relu_tile[1][sidx]));
          best2 = max(best2, int(conv_relu_tile[2][sidx]));
          best3 = max(best3, int(conv_relu_tile[3][sidx]));
        }
      }
      int y_base = py * POOL_OW + px;
      y[(oc_base + 0) * POOL_OH * POOL_OW + y_base] = static_cast<int8_t>(best0);
      y[(oc_base + 1) * POOL_OH * POOL_OW + y_base] = static_cast<int8_t>(best1);
      y[(oc_base + 2) * POOL_OH * POOL_OW + y_base] = static_cast<int8_t>(best2);
      y[(oc_base + 3) * POOL_OH * POOL_OW + y_base] = static_cast<int8_t>(best3);
    }
  }
}

template <int TILE>
__global__ void fused_tiled_oc8_conv_relu_pool_kernel(
    const int8_t* __restrict__ x, const int32_t* __restrict__ w4,
    int8_t* __restrict__ y, int shift) {
  constexpr int OC_PER_BLOCK = 8;
  constexpr int CONV_TILE = TILE * 2 + 1;
  __shared__ int8_t conv_relu_tile[OC_PER_BLOCK][CONV_TILE * CONV_TILE];

  int tile_px = blockIdx.x * TILE;
  int tile_py = blockIdx.y * TILE;
  int oc_base = blockIdx.z * OC_PER_BLOCK;
  int tid = threadIdx.x;

  for (int i = tid; i < CONV_TILE * CONV_TILE; i += blockDim.x) {
    int tx = i % CONV_TILE;
    int ty = i / CONV_TILE;
    int cx = tile_px * 2 - 1 + tx;
    int cy = tile_py * 2 - 1 + ty;
    int acc[OC_PER_BLOCK] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH) {
#pragma unroll 37
      for (int g = 0; g < K_GROUPS; ++g) {
        int k = g * 4;
        int xp = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int kk = k + j;
          int xv = 0;
          if (kk < K_TOTAL) {
            int kx = kk % KW;
            int ky = (kk / KW) % KH;
            int ic = kk / (KH * KW);
            int iy = cy * 2 + ky - 3;
            int ix = cx * 2 + kx - 3;
            if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
              xv = int(static_cast<unsigned char>(x[(ic * IH + iy) * IW + ix]));
            }
          }
          xp |= xv << (j * 8);
        }
        int w_base = oc_base * K_GROUPS + g;
#pragma unroll
        for (int c = 0; c < OC_PER_BLOCK; ++c) {
          acc[c] = dp4a_s32(xp, w4[w_base + c * K_GROUPS], acc[c]);
        }
      }
    }
#pragma unroll
    for (int c = 0; c < OC_PER_BLOCK; ++c) {
      conv_relu_tile[c][i] = clamp_relu_i8(acc[c], shift);
    }
  }
  __syncthreads();

  for (int out_i = tid; out_i < TILE * TILE; out_i += blockDim.x) {
    int lx = out_i % TILE;
    int ly = out_i / TILE;
    int px = tile_px + lx;
    int py = tile_py + ly;
    if (px < POOL_OW && py < POOL_OH) {
      int base_x = lx * 2;
      int base_y = ly * 2;
      int best[OC_PER_BLOCK] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          int sidx = (base_y + ky) * CONV_TILE + base_x + kx;
#pragma unroll
          for (int c = 0; c < OC_PER_BLOCK; ++c) {
            best[c] = max(best[c], int(conv_relu_tile[c][sidx]));
          }
        }
      }
      int y_base = py * POOL_OW + px;
#pragma unroll
      for (int c = 0; c < OC_PER_BLOCK; ++c) {
        y[(oc_base + c) * POOL_OH * POOL_OW + y_base] =
            static_cast<int8_t>(best[c]);
      }
    }
  }
}

template <int TILE>
__global__ void fused_tiled_oc8_smem_input_kernel(
    const int8_t* __restrict__ x, const int32_t* __restrict__ w4,
    int8_t* __restrict__ y, int shift) {
  constexpr int OC_PER_BLOCK = 8;
  constexpr int CONV_TILE = TILE * 2 + 1;
  constexpr int INPUT_TILE = CONV_TILE * 2 + 5;
  __shared__ int8_t input_tile[IC][INPUT_TILE * INPUT_TILE];
  __shared__ int8_t conv_relu_tile[OC_PER_BLOCK][CONV_TILE * CONV_TILE];

  int tile_px = blockIdx.x * TILE;
  int tile_py = blockIdx.y * TILE;
  int oc_base = blockIdx.z * OC_PER_BLOCK;
  int tid = threadIdx.x;
  int input_start_x = tile_px * 4 - 5;
  int input_start_y = tile_py * 4 - 5;

  for (int i = tid; i < IC * INPUT_TILE * INPUT_TILE; i += blockDim.x) {
    int lx = i % INPUT_TILE;
    int ly = (i / INPUT_TILE) % INPUT_TILE;
    int ic = i / (INPUT_TILE * INPUT_TILE);
    int gx = input_start_x + lx;
    int gy = input_start_y + ly;
    int8_t v = 0;
    if (gx >= 0 && gx < IW && gy >= 0 && gy < IH) {
      v = x[(ic * IH + gy) * IW + gx];
    }
    input_tile[ic][ly * INPUT_TILE + lx] = v;
  }
  __syncthreads();

  for (int i = tid; i < CONV_TILE * CONV_TILE; i += blockDim.x) {
    int tx = i % CONV_TILE;
    int ty = i / CONV_TILE;
    int cx = tile_px * 2 - 1 + tx;
    int cy = tile_py * 2 - 1 + ty;
    int acc[OC_PER_BLOCK] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH) {
#pragma unroll 37
      for (int g = 0; g < K_GROUPS; ++g) {
        int k = g * 4;
        int xp = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int kk = k + j;
          int xv = 0;
          if (kk < K_TOTAL) {
            int kx = kk % KW;
            int ky = (kk / KW) % KH;
            int ic = kk / (KH * KW);
            int local_y = (cy * 2 + ky - 3) - input_start_y;
            int local_x = (cx * 2 + kx - 3) - input_start_x;
            xv = int(static_cast<unsigned char>(
                input_tile[ic][local_y * INPUT_TILE + local_x]));
          }
          xp |= xv << (j * 8);
        }
        int w_base = oc_base * K_GROUPS + g;
#pragma unroll
        for (int c = 0; c < OC_PER_BLOCK; ++c) {
          acc[c] = dp4a_s32(xp, w4[w_base + c * K_GROUPS], acc[c]);
        }
      }
    }
#pragma unroll
    for (int c = 0; c < OC_PER_BLOCK; ++c) {
      conv_relu_tile[c][i] = clamp_relu_i8(acc[c], shift);
    }
  }
  __syncthreads();

  for (int out_i = tid; out_i < TILE * TILE; out_i += blockDim.x) {
    int lx = out_i % TILE;
    int ly = out_i / TILE;
    int px = tile_px + lx;
    int py = tile_py + ly;
    if (px < POOL_OW && py < POOL_OH) {
      int base_x = lx * 2;
      int base_y = ly * 2;
      int best[OC_PER_BLOCK] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          int sidx = (base_y + ky) * CONV_TILE + base_x + kx;
#pragma unroll
          for (int c = 0; c < OC_PER_BLOCK; ++c) {
            best[c] = max(best[c], int(conv_relu_tile[c][sidx]));
          }
        }
      }
      int y_base = py * POOL_OW + px;
#pragma unroll
      for (int c = 0; c < OC_PER_BLOCK; ++c) {
        y[(oc_base + c) * POOL_OH * POOL_OW + y_base] =
            static_cast<int8_t>(best[c]);
      }
    }
  }
}

static void cpu_reference(const std::vector<int8_t>& x,
                          const std::vector<int8_t>& w,
                          std::vector<int8_t>& y, int shift) {
  std::vector<int8_t> conv_relu(OC * CONV_OH * CONV_OW);
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < CONV_OH; ++oh) {
      for (int ow = 0; ow < CONV_OW; ++ow) {
        int acc = 0;
        for (int ic = 0; ic < IC; ++ic) {
          for (int ky = 0; ky < KH; ++ky) {
            int iy = oh * 2 + ky - 3;
            if (iy < 0 || iy >= IH) continue;
            for (int kx = 0; kx < KW; ++kx) {
              int ix = ow * 2 + kx - 3;
              if (ix < 0 || ix >= IW) continue;
              int x_idx = (ic * IH + iy) * IW + ix;
              int w_idx = ((oc * IC + ic) * KH + ky) * KW + kx;
              acc += int(x[x_idx]) * int(w[w_idx]);
            }
          }
        }
        conv_relu[(oc * CONV_OH + oh) * CONV_OW + ow] =
            clamp_relu_i8(acc, shift);
      }
    }
  }

  for (int oc = 0; oc < OC; ++oc) {
    for (int py = 0; py < POOL_OH; ++py) {
      for (int px = 0; px < POOL_OW; ++px) {
        int best = 0;
        for (int ky = 0; ky < 3; ++ky) {
          int cy = py * 2 + ky - 1;
          if (cy < 0 || cy >= CONV_OH) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int cx = px * 2 + kx - 1;
            if (cx < 0 || cx >= CONV_OW) continue;
            best = std::max(best, int(conv_relu[(oc * CONV_OH + cy) * CONV_OW + cx]));
          }
        }
        y[(oc * POOL_OH + py) * POOL_OW + px] = static_cast<int8_t>(best);
      }
    }
  }
}

template <typename Launch>
static float time_kernel(Launch launch, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

static int max_abs_err(const std::vector<int8_t>& a,
                       const std::vector<int8_t>& b) {
  int err = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    err = std::max(err, std::abs(int(a[i]) - int(b[i])));
  }
  return err;
}

static std::vector<int32_t> pack_weights_dp4a(const std::vector<int8_t>& w) {
  std::vector<int32_t> w4(OC * K_GROUPS);
  for (int oc = 0; oc < OC; ++oc) {
    for (int g = 0; g < K_GROUPS; ++g) {
      int packed = 0;
      for (int j = 0; j < 4; ++j) {
        int k = g * 4 + j;
        int v = 0;
        if (k < K_TOTAL) {
          v = int(static_cast<unsigned char>(w[oc * K_TOTAL + k]));
        }
        packed |= v << (j * 8);
      }
      w4[oc * K_GROUPS + g] = packed;
    }
  }
  return w4;
}

static void print_result(const Args& args, const char* name, float ms, int err) {
  if (args.csv) {
    std::printf("%s,%.6f,%d\n", name, ms, err);
  } else {
    std::printf("%-24s %.6f ms max_abs_err=%d\n", name, ms, err);
  }
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  constexpr int shift = 9;
  constexpr int x_count = N * IC * IH * IW;
  constexpr int w_count = OC * IC * KH * KW;
  constexpr int conv_count = OC * CONV_OH * CONV_OW;
  constexpr int y_count = OC * POOL_OH * POOL_OW;

  std::vector<int8_t> h_x(x_count), h_w(w_count), h_ref(y_count), h_y(y_count);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> dist(-8, 8);
  for (auto& v : h_x) v = static_cast<int8_t>(dist(rng));
  for (auto& v : h_w) v = static_cast<int8_t>(dist(rng));
  std::vector<int32_t> h_w4 = pack_weights_dp4a(h_w);
  cpu_reference(h_x, h_w, h_ref, shift);

  int8_t *d_x = nullptr, *d_conv_relu = nullptr, *d_y = nullptr;
  int32_t* d_w4 = nullptr;
  int32_t* d_conv_i32 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x_count));
  CUDA_CHECK(cudaMalloc(&d_w4, h_w4.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_conv_relu, conv_count));
  CUDA_CHECK(cudaMalloc(&d_conv_i32, conv_count * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_y, y_count));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w4, h_w4.data(), h_w4.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  constexpr int block = 256;
  int conv_grid = (conv_count + block - 1) / block;
  int y_grid = (y_count + block - 1) / block;

  float conv_ms = time_kernel([&] {
    conv_only_kernel<<<conv_grid, block>>>(d_x, d_w4, d_conv_i32);
  }, args.warmup, args.iters);
  print_result(args, "conv_only_i32", conv_ms, 0);

  float separate_ms = time_kernel([&] {
    conv_relu_kernel<<<conv_grid, block>>>(d_x, d_w4, d_conv_relu, shift);
    maxpool_kernel<<<y_grid, block>>>(d_conv_relu, d_y);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "conv_relu_pool_separate", separate_ms,
               max_abs_err(h_ref, h_y));

  float fused_ms = time_kernel([&] {
    fused_conv_relu_pool_kernel<<<y_grid, block>>>(d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_recompute_v1", fused_ms, max_abs_err(h_ref, h_y));

  dim3 tiled8_grid((POOL_OW + 7) / 8, (POOL_OH + 7) / 8, OC);
  float fused_tiled8_ms = time_kernel([&] {
    fused_tiled_conv_relu_pool_kernel<8><<<tiled8_grid, block>>>(d_x, d_w4, d_y,
                                                                shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_8x8", fused_tiled8_ms, max_abs_err(h_ref, h_y));

  dim3 tiled14_grid((POOL_OW + 13) / 14, (POOL_OH + 13) / 14, OC);
  float fused_tiled14_ms = time_kernel([&] {
    fused_tiled_conv_relu_pool_kernel<14><<<tiled14_grid, block>>>(d_x, d_w4, d_y,
                                                                  shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_14x14", fused_tiled14_ms,
               max_abs_err(h_ref, h_y));

  dim3 tiled28_grid((POOL_OW + 27) / 28, (POOL_OH + 27) / 28, OC);
  float fused_tiled28_ms = time_kernel([&] {
    fused_tiled_conv_relu_pool_kernel<28><<<tiled28_grid, block>>>(d_x, d_w4, d_y,
                                                                  shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_28x28", fused_tiled28_ms,
               max_abs_err(h_ref, h_y));

  dim3 tiled56_grid(1, 1, OC);
  float fused_tiled56_ms = time_kernel([&] {
    fused_tiled_conv_relu_pool_kernel<56><<<tiled56_grid, block>>>(d_x, d_w4, d_y,
                                                                  shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_56x56", fused_tiled56_ms,
               max_abs_err(h_ref, h_y));

  dim3 oc4_tiled8_grid((POOL_OW + 7) / 8, (POOL_OH + 7) / 8, OC / 4);
  float oc4_tiled8_ms = time_kernel([&] {
    fused_tiled_oc4_conv_relu_pool_kernel<8><<<oc4_tiled8_grid, block>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc4_8x8", oc4_tiled8_ms,
               max_abs_err(h_ref, h_y));

  dim3 oc4_tiled14_grid((POOL_OW + 13) / 14, (POOL_OH + 13) / 14, OC / 4);
  float oc4_tiled14_ms = time_kernel([&] {
    fused_tiled_oc4_conv_relu_pool_kernel<14><<<oc4_tiled14_grid, block>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc4_14x14", oc4_tiled14_ms,
               max_abs_err(h_ref, h_y));

  dim3 oc4_tiled28_grid((POOL_OW + 27) / 28, (POOL_OH + 27) / 28, OC / 4);
  float oc4_tiled28_ms = time_kernel([&] {
    fused_tiled_oc4_conv_relu_pool_kernel<28><<<oc4_tiled28_grid, block>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc4_28x28", oc4_tiled28_ms,
               max_abs_err(h_ref, h_y));

  dim3 oc8_tiled8_grid((POOL_OW + 7) / 8, (POOL_OH + 7) / 8, OC / 8);
  float oc8_tiled8_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<8><<<oc8_tiled8_grid, block>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_8x8", oc8_tiled8_ms,
               max_abs_err(h_ref, h_y));

  dim3 oc8_tiled14_grid((POOL_OW + 13) / 14, (POOL_OH + 13) / 14, OC / 8);
  float oc8_tiled14_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<14><<<oc8_tiled14_grid, block>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_14x14", oc8_tiled14_ms,
               max_abs_err(h_ref, h_y));

  dim3 oc8_tiled28_grid((POOL_OW + 27) / 28, (POOL_OH + 27) / 28, OC / 8);
  float oc8_tiled28_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<28><<<oc8_tiled28_grid, block>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_28x28", oc8_tiled28_ms,
               max_abs_err(h_ref, h_y));

  constexpr int block128 = 128;
  float oc8_tiled8_b128_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<8><<<oc8_tiled8_grid, block128>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_8x8_b128", oc8_tiled8_b128_ms,
               max_abs_err(h_ref, h_y));

  float oc8_tiled14_b128_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<14><<<oc8_tiled14_grid, block128>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_14x14_b128", oc8_tiled14_b128_ms,
               max_abs_err(h_ref, h_y));

  float oc8_smem_input8_b128_ms = time_kernel([&] {
    fused_tiled_oc8_smem_input_kernel<8><<<oc8_tiled8_grid, block128>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "oc8_smem_input_8x8_b128", oc8_smem_input8_b128_ms,
               max_abs_err(h_ref, h_y));

  float oc8_smem_input14_b128_ms = time_kernel([&] {
    fused_tiled_oc8_smem_input_kernel<14><<<oc8_tiled14_grid, block128>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "oc8_smem_input_14x14_b128", oc8_smem_input14_b128_ms,
               max_abs_err(h_ref, h_y));

  constexpr int block64 = 64;
  float oc8_tiled8_b64_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<8><<<oc8_tiled8_grid, block64>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_8x8_b64", oc8_tiled8_b64_ms,
               max_abs_err(h_ref, h_y));

  float oc8_tiled14_b64_ms = time_kernel([&] {
    fused_tiled_oc8_conv_relu_pool_kernel<14><<<oc8_tiled14_grid, block64>>>(
        d_x, d_w4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "fused_tiled_oc8_14x14_b64", oc8_tiled14_b64_ms,
               max_abs_err(h_ref, h_y));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_w4));
  CUDA_CHECK(cudaFree(d_conv_relu));
  CUDA_CHECK(cudaFree(d_conv_i32));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
