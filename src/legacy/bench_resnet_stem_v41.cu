#include "resnet_stem_common.cuh"

namespace {

// v41: register-pool (no conv-acc shared tile), TRT F2IP-style. One warp owns a
// 3-col conv strip; 8 lanes-of-group hold 8 conv X; pool-max over the 3x3 window
// done by warp shuffle on raw accumulators, then ReLU+quant once. Only B is
// staged in shared (~9KB, like TRT). OC64 via 4 OC-groups.
template <int OC_GROUPS>
__global__ void fused_regpool_kernel(const int8_t* __restrict__ x,
                                     const uint32_t* __restrict__ w_mma4,
                                     int8_t* __restrict__ y, int shift) {
  constexpr int OC_PER_GROUP = 16;
  constexpr int OC_PER_BLOCK = OC_GROUPS * OC_PER_GROUP;
  // Each block handles one pool row (56 px) of a 8-pool-col strip.
  constexpr int PX = 8;                 // pool cols per block
  constexpr int CONV_W = PX * 2 + 1;    // 17 conv cols
  constexpr int CONV_ROWS = 3;          // 3 conv rows feed one pool row (s2)
  constexpr int N_CONV = CONV_ROWS * CONV_W;  // 51 conv points
  constexpr int NG = (N_CONV + 7) / 8;        // 7
  __shared__ uint32_t smem_b4[N_CONV][K_GROUPS_MMA];

  int px0 = blockIdx.x * PX;
  int py = blockIdx.y;            // pool row
  int oc_base = blockIdx.z * OC_PER_BLOCK;
  int tid = threadIdx.x;
  int lane = tid & 31, group = lane >> 2, lig = lane & 3;

  int cy0 = py * 2 - 1;          // first conv row
  for (int i = tid; i < N_CONV * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA;
    int cr = n / CONV_W, cc = n % CONV_W;
    int cx = px0 * 2 - 1 + cc, cy = cy0 + cr;
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
    smem_b4[n][kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
  __syncthreads();

  // 8 warps -> N-groups; each warp accumulates its N-group, stores int8 strip.
  int warp = tid >> 5;
  __shared__ int8_t cstrip[OC_GROUPS][OC_PER_GROUP][N_CONV];
  if (warp < NG) {
    int ng = warp;
    int32_t acc[OC_GROUPS][4] = {};
#pragma unroll
    for (int kb = 0; kb < K_PAD_MMA; kb += 32) {
      int kgb = kb >> 2;
      uint32_t b[2];
#pragma unroll
      for (int br = 0; br < 2; ++br) {
        int n = ng * 8 + group, kg = kgb + lig + br * 4;
        b[br] = (n < N_CONV) ? smem_b4[n][kg] : 0;
      }
#pragma unroll
      for (int og = 0; og < OC_GROUPS; ++og) {
        uint32_t a[4];
#pragma unroll
        for (int ar = 0; ar < 4; ++ar) {
          int row = (ar & 1) ? group + 8 : group;
          int kg = kgb + lig + ((ar >= 2) ? 4 : 0);
          a[ar] = w_mma4[(oc_base + og * OC_PER_GROUP + row) * K_GROUPS_MMA + kg];
        }
        mma_m16n8k32_s8(acc[og], a, b);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = (i < 2) ? group : group + 8;
      int n = ng * 8 + lig * 2 + (i & 1);
      if (n < N_CONV)
#pragma unroll
        for (int og = 0; og < OC_GROUPS; ++og)
          cstrip[og][row][n] = clamp_relu_i8(acc[og][i], shift);
    }
  }
  __syncthreads();
  for (int t = tid; t < PX * OC_PER_BLOCK; t += blockDim.x) {
    int c = t % OC_PER_BLOCK, lx = t / OC_PER_BLOCK;
    int og = c / OC_PER_GROUP, oc_r = c % OC_PER_GROUP;
    int bx = lx * 2, best = 0;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx)
        best = max(best, int(cstrip[og][oc_r][ky * CONV_W + bx + kx]));
    int px = px0 + lx;
    if (px < POOL_OW)
      y[(oc_base + c) * POOL_OH * POOL_OW + py * POOL_OW + px] = (int8_t)best;
  }
}

template <int OC_GROUPS, int BLOCK>
static void run_case(const Args& args, const char* name, const int8_t* d_x,
                     const uint32_t* d_w, int8_t* d_y, int shift,
                     const std::vector<int8_t>& ref, std::vector<int8_t>& hy) {
  dim3 grid((POOL_OW + 7) / 8, POOL_OH, OC / (OC_GROUPS * 16));
  float ms = time_kernel([&] {
    fused_regpool_kernel<OC_GROUPS><<<grid, BLOCK>>>(d_x, d_w, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(), d_y, hy.size(), cudaMemcpyDeviceToHost));
  print_result(args, name, ms, max_abs_err(ref, hy));
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  constexpr int shift = 9;
  std::vector<int8_t> hx(N*IC*IH*IW), hw(OC*IC*KH*KW), ref(OC*POOL_OH*POOL_OW), hy(OC*POOL_OH*POOL_OW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8,8);
  for (auto&v:hx) v=(int8_t)d(rng); for (auto&v:hw) v=(int8_t)d(rng);
  auto hwm = pack_weights_mma4(hw); cpu_reference(hx,hw,ref,shift);
  int8_t*dx,*dy; uint32_t*dw;
  CUDA_CHECK(cudaMalloc(&dx,hx.size())); CUDA_CHECK(cudaMalloc(&dy,hy.size()));
  CUDA_CHECK(cudaMalloc(&dw,hwm.size()*4));
  CUDA_CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw,hwm.data(),hwm.size()*4,cudaMemcpyHostToDevice));
  run_case<4,256>(args,"v41_regpool_oc64",dx,dw,dy,shift,ref,hy);
  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dw));
  return 0;
}
