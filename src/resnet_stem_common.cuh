#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr int K_PAD_MMA = ((K_TOTAL + 31) / 32) * 32;
constexpr int K_GROUPS_MMA = K_PAD_MMA / 4;

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

__device__ __forceinline__ uint32_t pack_s8x4(int v0, int v1, int v2, int v3) {
  return (uint32_t(v0) & 0xffu) | ((uint32_t(v1) & 0xffu) << 8) |
         ((uint32_t(v2) & 0xffu) << 16) | ((uint32_t(v3) & 0xffu) << 24);
}

__device__ __forceinline__ void mma_m16n8k32_s8(int32_t d[4],
                                                const uint32_t a[4],
                                                const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
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
            best = std::max(
                best, int(conv_relu[(oc * CONV_OH + cy) * CONV_OW + cx]));
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

static std::vector<uint32_t> pack_weights_mma4(const std::vector<int8_t>& w) {
  std::vector<uint32_t> w_mma4(OC * K_GROUPS_MMA, 0);
  for (int oc = 0; oc < OC; ++oc) {
    for (int kg = 0; kg < K_GROUPS_MMA; ++kg) {
      int vals[4] = {0, 0, 0, 0};
      for (int e = 0; e < 4; ++e) {
        int k = kg * 4 + e;
        if (k < K_TOTAL) {
          vals[e] = int(w[oc * K_TOTAL + k]);
        }
      }
      w_mma4[oc * K_GROUPS_MMA + kg] =
          (uint32_t(vals[0]) & 0xffu) | ((uint32_t(vals[1]) & 0xffu) << 8) |
          ((uint32_t(vals[2]) & 0xffu) << 16) |
          ((uint32_t(vals[3]) & 0xffu) << 24);
    }
  }
  return w_mma4;
}

static void print_result(const Args& args, const char* name, float ms, int err) {
  if (args.csv) {
    std::printf("%s,%.6f,%d\n", name, ms, err);
  } else {
    std::printf("%-44s %.6f ms max_abs_err=%d\n", name, ms, err);
  }
}
