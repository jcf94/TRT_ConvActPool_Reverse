#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
                   int(st__));                                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

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
constexpr int K_PAD = 160;
constexpr int SPATIAL = CONV_OH * CONV_OW;

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

__global__ void relu_pool_from_gemm_kernel(const int32_t* __restrict__ conv,
                                           int8_t* __restrict__ y, int shift) {
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
      int spatial = cy * CONV_OW + cx;
      int v = int(clamp_relu_i8(conv[oc + spatial * OC], shift));
      best = max(best, v);
    }
  }
  y[idx] = static_cast<int8_t>(best);
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

static void build_col_major_weight(const std::vector<int8_t>& w,
                                   std::vector<int8_t>& a) {
  a.assign(OC * K_PAD, 0);
  for (int oc = 0; oc < OC; ++oc) {
    for (int k = 0; k < K_TOTAL; ++k) {
      a[oc + k * OC] = w[oc * K_TOTAL + k];
    }
  }
}

static void build_im2col(const std::vector<int8_t>& x, std::vector<int8_t>& b) {
  b.assign(K_PAD * SPATIAL, 0);
  for (int oh = 0; oh < CONV_OH; ++oh) {
    for (int ow = 0; ow < CONV_OW; ++ow) {
      int spatial = oh * CONV_OW + ow;
      for (int ic = 0; ic < IC; ++ic) {
        for (int ky = 0; ky < KH; ++ky) {
          int iy = oh * 2 + ky - 3;
          if (iy < 0 || iy >= IH) continue;
          for (int kx = 0; kx < KW; ++kx) {
            int ix = ow * 2 + kx - 3;
            if (ix < 0 || ix >= IW) continue;
            int k = (ic * KH + ky) * KW + kx;
            b[k + spatial * K_PAD] = x[(ic * IH + iy) * IW + ix];
          }
        }
      }
    }
  }
}

template <typename Launch>
static float time_op(Launch launch, int warmup, int iters) {
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
  constexpr int x_count = IC * IH * IW;
  constexpr int w_count = OC * IC * KH * KW;
  constexpr int y_count = OC * POOL_OH * POOL_OW;

  std::vector<int8_t> h_x(x_count), h_w(w_count), h_ref(y_count), h_y(y_count);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> dist(-8, 8);
  for (auto& v : h_x) v = static_cast<int8_t>(dist(rng));
  for (auto& v : h_w) v = static_cast<int8_t>(dist(rng));
  cpu_reference(h_x, h_w, h_ref, shift);

  std::vector<int8_t> h_a, h_b;
  build_col_major_weight(h_w, h_a);
  build_im2col(h_x, h_b);

  int8_t *d_a = nullptr, *d_b = nullptr, *d_y = nullptr;
  int32_t* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size()));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size()));
  CUDA_CHECK(cudaMalloc(&d_c, OC * SPATIAL * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_y, y_count));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size(), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  int32_t alpha = 1;
  int32_t beta = 0;

  auto gemm = [&] {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, OC, SPATIAL,
                              K_PAD, &alpha, d_a, CUDA_R_8I, OC, d_b,
                              CUDA_R_8I, K_PAD, &beta, d_c, CUDA_R_32I, OC,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  float gemm_ms = time_op(gemm, args.warmup, args.iters);
  print_result(args, "cublas_int8_gemm", gemm_ms, 0);

  constexpr int block = 256;
  int y_grid = (y_count + block - 1) / block;
  float end_to_end_ms = time_op([&] {
    gemm();
    relu_pool_from_gemm_kernel<<<y_grid, block>>>(d_c, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count, cudaMemcpyDeviceToHost));
  print_result(args, "cublas_gemm_relu_pool", end_to_end_ms,
               max_abs_err(h_ref, h_y));

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
