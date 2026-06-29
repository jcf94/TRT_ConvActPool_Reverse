#include "resnet_stem_common.cuh"

namespace {

// v42: full implicit-GEMM 8x120-style rewrite. One CTA owns all 64 OC for a
// wide strip of pool columns. Conv outputs are pooled directly from registers
// (no int8/int16 conv-acc smem tile, no pool re-read) — mimics TRT's F2IP chain
// that folds 3x3 pool-max+ReLU+quant. Only activations are staged in shared
// (~9KB), each lane keeps its conv columns int8 and a warp does the 3-col pool
// reduction by shuffle, killing the v38 LDS=596 wall.
//
// Tile: POOL_COLS pool px in one CTA, all 56 pool rows handled across grid.y,
// 64 OC handled in 4 groups of 16 by 4 warps -> 1 CTA = 4 warps x16 OC.
template <int POOL_COLS>
__global__ void v42_kernel(const int8_t* __restrict__ x,
                           const uint32_t* __restrict__ w_mma4,
                           int8_t* __restrict__ y, int shift) {
  constexpr int OC_GROUPS = 4;          // 4 warps -> 4 OC groups of 16
  constexpr int OC_PER_GROUP = 16;
  constexpr int CONV_COLS = POOL_COLS * 2 + 1;  // conv cols feeding pool window
  constexpr int CONV_ROWS = 3;                   // 3 conv rows -> 1 pool row
  constexpr int N_PTS = CONV_COLS * CONV_ROWS;
  constexpr int NG = (N_PTS + 7) / 8;            // 8-col MMA n-groups

  int px0 = blockIdx.x * POOL_COLS;
  int py = blockIdx.y;
  int tid = threadIdx.x;
  int warp = tid >> 5, lane = tid & 31, g = lane >> 2, lig = lane & 3;

  // stage activations: B[N_PTS][K_GROUPS_MMA]
  __shared__ uint32_t sB[N_PTS * K_GROUPS_MMA];
  int cx0 = px0 * 2 - 1, cy0 = py * 2 - 1;
  for (int i = tid; i < N_PTS * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA;
    int cx = cx0 + (n % CONV_COLS), cy = cy0 + (n / CONV_COLS);
    int v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e, val = 0;
      if (k < K_TOTAL) {
        int kx = k % KW, ky = (k / KW) % KH, ic = k / (KH * KW);
        int iy = cy * 2 + ky - 3, ix = cx * 2 + kx - 3;
        if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH && iy >= 0 &&
            iy < IH && ix >= 0 && ix < IW)
          val = int(x[(ic * IH + iy) * IW + ix]);
      }
      v[e] = val;
    }
    sB[n * K_GROUPS_MMA + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
  __syncthreads();

  int oc_base = warp * OC_PER_GROUP;
  // each warp: 16 OC x N_PTS conv pts, accumulate, store int8 to small smem
  // strip then pool. Strip = OC_PER_BLOCK x N_PTS int8 -> reused across pool.
  __shared__ int8_t strip[OC_GROUPS][OC_PER_GROUP][N_PTS];
  for (int ng = 0; ng < NG; ++ng) {
    int n0 = ng * 8;
    int32_t acc[4] = {};
#pragma unroll
    for (int kb = 0; kb < K_PAD_MMA; kb += 32) {
      int kgb = kb >> 2;
      uint32_t b[2], a[4];
#pragma unroll
      for (int br = 0; br < 2; ++br) {
        int n = n0 + g, kg = kgb + lig + br * 4;
        b[br] = (n < N_PTS) ? sB[n * K_GROUPS_MMA + kg] : 0;
      }
#pragma unroll
      for (int ar = 0; ar < 4; ++ar) {
        int row = (ar & 1) ? g + 8 : g, kg = kgb + lig + ((ar >= 2) ? 4 : 0);
        a[ar] = w_mma4[(oc_base + row) * K_GROUPS_MMA + kg];
      }
      mma_m16n8k32_s8(acc, a, b);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = (i < 2) ? g : g + 8, n = n0 + lig * 2 + (i & 1);
      if (n < N_PTS) strip[warp][row][n] = clamp_relu_i8(acc[i], shift);
    }
  }
  __syncthreads();

  // pool: one px x OC -> read 3x3 from strip, vectorized over OC group
  for (int t = tid; t < POOL_COLS * OC; t += blockDim.x) {
    int oc = t % OC, lx = t / OC;
    int og = oc >> 4, oc_r = oc & 15, bx = lx * 2, best = 0;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx)
        best = max(best, int(strip[og][oc_r][ky * CONV_COLS + bx + kx]));
    int px = px0 + lx;
    if (px < POOL_OW)
      y[oc * POOL_OH * POOL_OW + py * POOL_OW + px] = (int8_t)best;
  }
}

template <int POOL_COLS, int BLK>
static void run_case(const Args& args, const char* name, const int8_t* dx,
                     const uint32_t* dw, int8_t* dy, int shift,
                     const std::vector<int8_t>& ref, std::vector<int8_t>& hy) {
  dim3 grid((POOL_OW + POOL_COLS - 1) / POOL_COLS, POOL_OH, 1);
  float ms = time_kernel([&] { v42_kernel<POOL_COLS><<<grid, BLK>>>(dx, dw, dy, shift); },
                         args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size(), cudaMemcpyDeviceToHost));
  print_result(args, name, ms, max_abs_err(ref, hy));
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  constexpr int shift = 9;
  constexpr int x_count = N * IC * IH * IW, w_count = OC * IC * KH * KW,
                y_count = OC * POOL_OH * POOL_OW;
  std::vector<int8_t> hx(x_count), hw(w_count), ref(y_count), hy(y_count);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> dist(-8, 8);
  for (auto& v : hx) v = (int8_t)dist(rng);
  for (auto& v : hw) v = (int8_t)dist(rng);
  auto hwm = pack_weights_mma4(hw);
  cpu_reference(hx, hw, ref, shift);
  int8_t *dx, *dy; uint32_t* dw;
  CUDA_CHECK(cudaMalloc(&dx, x_count)); CUDA_CHECK(cudaMalloc(&dy, y_count));
  CUDA_CHECK(cudaMalloc(&dw, hwm.size() * 4));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), x_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hwm.data(), hwm.size() * 4, cudaMemcpyHostToDevice));
  run_case<8, 128>(args, "v42_implgemm_pc8", dx, dw, dy, shift, ref, hy);
  run_case<14, 128>(args, "v42_implgemm_pc14", dx, dw, dy, shift, ref, hy);
  run_case<28, 128>(args, "v42_implgemm_pc28", dx, dw, dy, shift, ref, hy);
  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dw));
  return 0;
}
