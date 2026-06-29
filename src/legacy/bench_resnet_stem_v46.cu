#include "resnet_stem_common.cuh"

namespace {

// v46: cut the TRT pool-LDS/STG gap. conv-relu tile is stored transposed
// [N][OC] so each pool window reads 64 contiguous OC as 16x int8x4 and pools
// with vmax4 (mirrors F2IP byte-max), then stores 64 OC as vectorized words.
__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("vmax4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0));
  return r;
}

template <int TILE, int WARPS>
__global__ void v46_kernel(const int8_t* __restrict__ x,
                           const uint32_t* __restrict__ w_mma4,
                           int8_t* __restrict__ y, int shift) {
  constexpr int OCG = 4, OPG = 16, OCB = 64;
  constexpr int CT = TILE * 2 + 1, NT = CT * CT, NG = (NT + 7) / 8;
  extern __shared__ unsigned char sm[];
  uint32_t* b4 = reinterpret_cast<uint32_t*>(sm);
  uint32_t* acc4 = b4 + NT * K_GROUPS_MMA;  // [NT][16]  64 int8 per N
  int tpx = blockIdx.x * TILE, tpy = blockIdx.y * TILE, tid = threadIdx.x;
  int wid = tid >> 5, lid = tid & 31, gid = lid >> 2, lig = lid & 3;
  bool inr = tpx > 0 && tpy > 0 && tpx + TILE < POOL_OW && tpy + TILE < POOL_OH;
  for (int i = tid; i < NT * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA;
    int cx = tpx * 2 - 1 + n % CT, cy = tpy * 2 - 1 + n / CT, v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e; v[e] = 0;
      if (k < K_TOTAL) {
        int ic = k / 49, iy = cy * 2 + (k / 7) % 7 - 3, ix = cx * 2 + k % 7 - 3;
        if (inr || (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH &&
                    iy >= 0 && iy < IH && ix >= 0 && ix < IW))
          v[e] = x[(ic * IH + iy) * IW + ix];
      }
    }
    b4[n * K_GROUPS_MMA + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
  __syncthreads();
  if (wid < WARPS)
    for (int ng = wid; ng < NG; ng += WARPS) {
      int ns = ng * 8; int32_t acc[OCG][4] = {};
#pragma unroll
      for (int kb = 0; kb < K_PAD_MMA; kb += 32) {
        int kgb = kb >> 2; uint32_t b[2];
#pragma unroll
        for (int br = 0; br < 2; ++br) {
          int n0 = ns + gid;
          b[br] = (n0 < NT) ? b4[n0 * K_GROUPS_MMA + kgb + lig + br * 4] : 0;
        }
#pragma unroll
        for (int og = 0; og < OCG; ++og) {
          uint32_t a[4];
#pragma unroll
          for (int ar = 0; ar < 4; ++ar) {
            int row = (ar & 1) ? gid + 8 : gid, kg = kgb + lig + (ar >= 2 ? 4 : 0);
            a[ar] = w_mma4[(og * OPG + row) * K_GROUPS_MMA + kg];
          }
          mma_m16n8k32_s8(acc[og], a, b);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? gid : gid + 8, n = ns + lig * 2 + (i & 1);
        if (n < NT)
#pragma unroll
          for (int og = 0; og < OCG; ++og)
            ((int8_t*)acc4)[n * OCB + og * OPG + row] = clamp_relu_i8(acc[og][i], shift);
      }
    }
  __syncthreads();
  for (int o = tid; o < TILE * TILE; o += blockDim.x) {
    int lx = o % TILE, ly = o / TILE, px = tpx + lx, py = tpy + ly;
    if (px >= POOL_OW || py >= POOL_OH) continue;
    int bx = lx * 2, by = ly * 2; uint32_t best[16] = {};
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        int s = (by + ky) * CT + bx + kx;
#pragma unroll
        for (int c = 0; c < 16; ++c) best[c] = vmax_s8x4(best[c], acc4[s * 16 + c]);
      }
    int yb = py * POOL_OW + px;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
      uint32_t v = best[c];
      y[(c * 4 + 0) * POOL_OH * POOL_OW + yb] = v;
      y[(c * 4 + 1) * POOL_OH * POOL_OW + yb] = v >> 8;
      y[(c * 4 + 2) * POOL_OH * POOL_OW + yb] = v >> 16;
      y[(c * 4 + 3) * POOL_OH * POOL_OW + yb] = v >> 24;
    }
  }
}

template <int TILE, int WARPS, int BLK>
static void run_case(const Args& a, const char* nm, const int8_t* dx,
                     const uint32_t* dw, int8_t* dy, int sh,
                     const std::vector<int8_t>& r, std::vector<int8_t>& hy) {
  constexpr int CT = TILE * 2 + 1, NT = CT * CT;
  size_t sm = NT * K_GROUPS_MMA * 4 + NT * 64;
  auto k = v46_kernel<TILE, WARPS>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  dim3 g((POOL_OW + TILE - 1) / TILE, (POOL_OH + TILE - 1) / TILE, 1);
  float ms = time_kernel([&] { k<<<g, BLK, sm>>>(dx, dw, dy, sh); }, a.warmup, a.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size(), cudaMemcpyDeviceToHost));
  print_result(a, nm, ms, max_abs_err(r, hy));
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); constexpr int sh = 9;
  std::vector<int8_t> hx(IC * IH * IW), hw(OC * IC * KH * KW),
      hr(OC * POOL_OH * POOL_OW), hy(OC * POOL_OH * POOL_OW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8, 8);
  for (auto& v : hx) v = d(rng); for (auto& v : hw) v = d(rng);
  auto hw4 = pack_weights_mma4(hw); cpu_reference(hx, hw, hr, sh);
  int8_t *dx, *dy; uint32_t* dw;
  CUDA_CHECK(cudaMalloc(&dx, hx.size())); CUDA_CHECK(cudaMalloc(&dy, hy.size()));
  CUDA_CHECK(cudaMalloc(&dw, hw4.size() * 4));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw4.data(), hw4.size() * 4, cudaMemcpyHostToDevice));
  run_case<4, 8, 256>(a, "v46_t4_b256", dx, dw, dy, sh, hr, hy);
  run_case<6, 8, 256>(a, "v46_t6_b256", dx, dw, dy, sh, hr, hy);
  run_case<8, 8, 256>(a, "v46_t8_b256", dx, dw, dy, sh, hr, hy);
  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dw));
  return 0;
}
