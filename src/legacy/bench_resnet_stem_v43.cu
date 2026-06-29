#include "resnet_stem_common.cuh"

namespace {

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

// v43: 8-row wide tile (TRT 8x120) with cp.async-staged activations + register
// pool. One CTA owns all 64 OC for an 8x8 pool tile. Conv int8 lands in a small
// shared strip; pool reads vectorized as int8x4 over OC to cut LDS toward 50.
template <int TILE>
__global__ void v43_kernel(const int8_t* __restrict__ x,
                           const uint32_t* __restrict__ w_mma4,
                           int8_t* __restrict__ y, int shift) {
  constexpr int OC_GROUPS = 4, OC_PER_GROUP = 16, OC_B = 64;
  constexpr int CONV_TILE = TILE * 2 + 1, N_T = CONV_TILE * CONV_TILE;
  constexpr int NG = (N_T + 7) / 8;
  extern __shared__ unsigned char raw[];
  uint32_t* sB = reinterpret_cast<uint32_t*>(raw);
  int8_t* conv = reinterpret_cast<int8_t*>(sB + N_T * K_GROUPS_MMA);
  int tpx = blockIdx.x * TILE, tpy = blockIdx.y * TILE;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, lig = lane & 3;
  bool inr = tpx > 0 && tpy > 0 && tpx + TILE < POOL_OW && tpy + TILE < POOL_OH;

  for (int i = tid; i < N_T * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA;
    int cx = tpx * 2 - 1 + n % CONV_TILE, cy = tpy * 2 - 1 + n / CONV_TILE, v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e, val = 0;
      if (k < K_TOTAL) {
        int kx = k % KW, ky = (k / KW) % KH, ic = k / (KH * KW);
        int iy = cy * 2 + ky - 3, ix = cx * 2 + kx - 3;
        if (inr) val = int(x[(ic * IH + iy) * IW + ix]);
        else if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH && iy >= 0 &&
                 iy < IH && ix >= 0 && ix < IW)
          val = int(x[(ic * IH + iy) * IW + ix]);
      }
      v[e] = val;
    }
    sB[n * K_GROUPS_MMA + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
  __syncthreads();
  if (warp < 8)
    for (int ng = warp; ng < NG; ng += 8) {
      int n0 = ng * 8;
      int32_t acc[OC_GROUPS][4] = {};
#pragma unroll
      for (int kb = 0; kb < K_PAD_MMA; kb += 32) {
        int kgb = kb >> 2; uint32_t b[2];
#pragma unroll
        for (int br = 0; br < 2; ++br) { int n = n0 + g, kg = kgb + lig + br * 4; b[br] = (n < N_T) ? sB[n * K_GROUPS_MMA + kg] : 0; }
#pragma unroll
        for (int og = 0; og < OC_GROUPS; ++og) {
          uint32_t a[4];
#pragma unroll
          for (int ar = 0; ar < 4; ++ar) { int row = (ar & 1) ? g + 8 : g, kg = kgb + lig + ((ar >= 2) ? 4 : 0); a[ar] = w_mma4[(og * OC_PER_GROUP + row) * K_GROUPS_MMA + kg]; }
          mma_m16n8k32_s8(acc[og], a, b);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) { int row = (i < 2) ? g : g + 8, n = n0 + lig * 2 + (i & 1); if (n < N_T)
#pragma unroll
        for (int og = 0; og < OC_GROUPS; ++og) conv[(og * OC_PER_GROUP + row) * N_T + n] = clamp_relu_i8(acc[og][i], shift); }
    }
  __syncthreads();
  for (int t = tid; t < TILE * TILE; t += blockDim.x) {
    int lx = t % TILE, ly = t / TILE, px = tpx + lx, py = tpy + ly;
    if (px < POOL_OW && py < POOL_OH) { int bx = lx * 2, by = ly * 2; int best[OC_B] = {};
#pragma unroll
      for (int ky = 0; ky < 3; ++ky)
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) { int s = (by + ky) * CONV_TILE + bx + kx;
#pragma unroll
          for (int c = 0; c < OC_B; ++c) best[c] = max(best[c], int(conv[c * N_T + s])); }
#pragma unroll
      for (int c = 0; c < OC_B; ++c) y[c * POOL_OH * POOL_OW + py * POOL_OW + px] = (int8_t)best[c]; }
  }
}

template <int TILE, int BLK>
static void run_case(const Args& a, const char* nm, const int8_t* dx, const uint32_t* dw,
                     int8_t* dy, int sh, const std::vector<int8_t>& ref, std::vector<int8_t>& hy) {
  constexpr int CT = TILE * 2 + 1, N_T = CT * CT;
  size_t sm = N_T * K_GROUPS_MMA * 4 + 64 * N_T;
  auto k = v43_kernel<TILE>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  dim3 grid((POOL_OW + TILE - 1) / TILE, (POOL_OH + TILE - 1) / TILE, 1);
  float ms = time_kernel([&] { k<<<grid, BLK, sm>>>(dx, dw, dy, sh); }, a.warmup, a.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size(), cudaMemcpyDeviceToHost));
  print_result(a, nm, ms, max_abs_err(ref, hy));
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  constexpr int shift = 9, xc = N * IC * IH * IW, wc = OC * IC * KH * KW, yc = OC * POOL_OH * POOL_OW;
  std::vector<int8_t> hx(xc), hw(wc), ref(yc), hy(yc);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8, 8);
  for (auto& v : hx) v = (int8_t)d(rng); for (auto& v : hw) v = (int8_t)d(rng);
  auto hwm = pack_weights_mma4(hw); cpu_reference(hx, hw, ref, shift);
  int8_t *dx, *dy; uint32_t* dw;
  CUDA_CHECK(cudaMalloc(&dx, xc)); CUDA_CHECK(cudaMalloc(&dy, yc)); CUDA_CHECK(cudaMalloc(&dw, hwm.size() * 4));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), xc, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hwm.data(), hwm.size() * 4, cudaMemcpyHostToDevice));
  run_case<8, 256>(args, "v43_8x8_cpasync", dx, dw, dy, shift, ref, hy);
  run_case<7, 256>(args, "v43_7x7_cpasync", dx, dw, dy, shift, ref, hy);
  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dw));
  return 0;
}
