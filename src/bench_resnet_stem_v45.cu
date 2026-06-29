#include "resnet_stem_common.cuh"

namespace {

// v45: direct reverse of TRT sm80_trt_conv_act_pool_v3_tile_rows_8_tile_cols_120.
// Structure copied from SASS (see docs/trt_sass_reverse_v45.md):
//  - one CTA owns an 8-pool-row x TILE_PX tile over the full 64-OC slab so
//    weights load once and feed a wide straight-line IMMA mainloop;
//  - epilogue mimics F2IP.S8.F32.NTZ.RELU: ReLU+quant fused, pool-max chained,
//    then packed int8x4 byte-max across the pool window (the SASS 0x80808080
//    select-max trick) and two vectorized STG.E.64 instead of a smem re-read.

// Packed signed-int8x4 max: same per-byte pool-max the TRT epilogue does with
// the 0x80808080 bias + >>7 select trick, expressed as one SIMD video op.
__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("vmax4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0));
  return r;
}

template <int TILE_PX, int OC_GROUPS>
__global__ void trt_replica_kernel(const int8_t* __restrict__ x,
                                   const uint32_t* __restrict__ w_mma4,
                                   int8_t* __restrict__ y, int shift) {
  constexpr int TILE_PY = 8;          // tile_rows_8
  constexpr int OC_PER_GROUP = 16;
  constexpr int OC_PER_BLOCK = OC_GROUPS * OC_PER_GROUP;  // 64
  constexpr int CONV_TX = TILE_PX * 2 + 1;  // conv cols (tile_cols ~120)
  constexpr int CONV_TY = TILE_PY * 2 + 1;  // conv rows (17)
  constexpr int N_TOTAL = CONV_TX * CONV_TY;
  constexpr int N_GROUPS = (N_TOTAL + 7) / 8;
  extern __shared__ unsigned char smem_raw[];
  uint32_t* smem_b4 = reinterpret_cast<uint32_t*>(smem_raw);
  int8_t* conv_acc = reinterpret_cast<int8_t*>(smem_b4 + N_TOTAL * K_GROUPS_MMA);

  int tile_px = blockIdx.x * TILE_PX;
  int tile_py = blockIdx.y * TILE_PY;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  int group_id = lane_id >> 2;
  int lane_in_group = lane_id & 3;
  bool interior = tile_px > 0 && tile_py > 0 && tile_px + TILE_PX < POOL_OW &&
                  tile_py + TILE_PY < POOL_OH;

  for (int i = tid; i < N_TOTAL * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA;
    int cx = tile_px * 2 - 1 + n % CONV_TX;
    int cy = tile_py * 2 - 1 + n / CONV_TX;
    int vals[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e, v = 0;
      if (k < K_TOTAL) {
        int ic = k / (KH * KW), iy = cy * 2 + (k / KW) % KH - 3,
            ix = cx * 2 + k % KW - 3;
        if (interior || (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH &&
                         iy >= 0 && iy < IH && ix >= 0 && ix < IW))
          v = int(x[(ic * IH + iy) * IW + ix]);
      }
      vals[e] = v;
    }
    smem_b4[n * K_GROUPS_MMA + kg] = pack_s8x4(vals[0], vals[1], vals[2], vals[3]);
  }
  __syncthreads();

  for (int ng = warp_id; ng < N_GROUPS; ng += blockDim.x >> 5) {
    int n_start = ng * 8;
    int32_t acc[OC_GROUPS][4] = {};
#pragma unroll
    for (int k_base = 0; k_base < K_PAD_MMA; k_base += 32) {
      int kg_base = k_base >> 2;
      uint32_t b[2];
#pragma unroll
      for (int br = 0; br < 2; ++br) {
        int n0 = n_start + group_id;
        b[br] = (n0 < N_TOTAL) ? smem_b4[n0 * K_GROUPS_MMA + kg_base + lane_in_group + br * 4] : 0;
      }
#pragma unroll
      for (int og = 0; og < OC_GROUPS; ++og) {
        uint32_t a[4];
#pragma unroll
        for (int ar = 0; ar < 4; ++ar) {
          int row = (ar & 1) ? group_id + 8 : group_id;
          int kg = kg_base + lane_in_group + ((ar >= 2) ? 4 : 0);
          a[ar] = w_mma4[(og * OC_PER_GROUP + row) * K_GROUPS_MMA + kg];
        }
        mma_m16n8k32_s8(acc[og], a, b);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = (i < 2) ? group_id : group_id + 8;
      int n = n_start + lane_in_group * 2 + (i & 1);
#pragma unroll
      for (int og = 0; og < OC_GROUPS; ++og)
        if (n < N_TOTAL)
          conv_acc[(og * OC_PER_GROUP + row) * N_TOTAL + n] =
              clamp_relu_i8(acc[og][i], shift);
    }
  }
  __syncthreads();

  for (int out_i = tid; out_i < TILE_PX * TILE_PY; out_i += blockDim.x) {
    int lx = out_i % TILE_PX, ly = out_i / TILE_PX;
    int px = tile_px + lx, py = tile_py + ly;
    if (px >= POOL_OW || py >= POOL_OH) continue;
    int bx = lx * 2, by = ly * 2;
    // pool 64 OC packed 4-at-a-time (F2IP-style register byte-max, no re-read)
    uint32_t best[OC_PER_BLOCK / 4] = {};
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        int sidx = (by + ky) * CONV_TX + bx + kx;
#pragma unroll
        for (int c = 0; c < OC_PER_BLOCK / 4; ++c) {
          uint32_t v = pack_s8x4(conv_acc[(c * 4 + 0) * N_TOTAL + sidx],
                                 conv_acc[(c * 4 + 1) * N_TOTAL + sidx],
                                 conv_acc[(c * 4 + 2) * N_TOTAL + sidx],
                                 conv_acc[(c * 4 + 3) * N_TOTAL + sidx]);
          best[c] = vmax_s8x4(best[c], v);
        }
      }
    int yb = py * POOL_OW + px;
#pragma unroll
    for (int c = 0; c < OC_PER_BLOCK / 4; ++c) {
      y[(c * 4 + 0) * POOL_OH * POOL_OW + yb] = (int8_t)(best[c] & 0xff);
      y[(c * 4 + 1) * POOL_OH * POOL_OW + yb] = (int8_t)((best[c] >> 8) & 0xff);
      y[(c * 4 + 2) * POOL_OH * POOL_OW + yb] = (int8_t)((best[c] >> 16) & 0xff);
      y[(c * 4 + 3) * POOL_OH * POOL_OW + yb] = (int8_t)((best[c] >> 24) & 0xff);
    }
  }
}

template <int TILE_PX, int OC_GROUPS, int BLK>
static void run_case(const Args& args, const char* name, const int8_t* d_x,
                     const uint32_t* d_w, int8_t* d_y, int shift,
                     const std::vector<int8_t>& ref, std::vector<int8_t>& h_y) {
  constexpr int CONV_TX = TILE_PX * 2 + 1, CONV_TY = 17, N = CONV_TX * CONV_TY;
  size_t smem = N * K_GROUPS_MMA * sizeof(uint32_t) +
                OC_GROUPS * 16 * N * sizeof(int8_t);
  auto k = trt_replica_kernel<TILE_PX, OC_GROUPS>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 grid((POOL_OW + TILE_PX - 1) / TILE_PX, (POOL_OH + 7) / 8, 1);
  float ms = time_kernel([&] { k<<<grid, BLK, smem>>>(d_x, d_w, d_y, shift); },
                         args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size(), cudaMemcpyDeviceToHost));
  print_result(args, name, ms, max_abs_err(ref, h_y));
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  constexpr int shift = 9;
  std::vector<int8_t> h_x(N * IC * IH * IW), h_w(OC * IC * KH * KW),
      h_ref(OC * POOL_OH * POOL_OW), h_y(OC * POOL_OH * POOL_OW);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> dist(-8, 8);
  for (auto& v : h_x) v = (int8_t)dist(rng);
  for (auto& v : h_w) v = (int8_t)dist(rng);
  auto h_w4 = pack_weights_mma4(h_w);
  cpu_reference(h_x, h_w, h_ref, shift);
  int8_t *d_x, *d_y; uint32_t* d_w;
  CUDA_CHECK(cudaMalloc(&d_x, h_x.size()));
  CUDA_CHECK(cudaMalloc(&d_y, h_y.size()));
  CUDA_CHECK(cudaMalloc(&d_w, h_w4.size() * 4));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w4.data(), h_w4.size() * 4, cudaMemcpyHostToDevice));
  run_case<7, 4, 256>(args, "v45_trt_replica_8x7_b256", d_x, d_w, d_y, shift, h_ref, h_y);
  run_case<10, 4, 256>(args, "v45_trt_replica_8x10_b256", d_x, d_w, d_y, shift, h_ref, h_y);
  run_case<12, 4, 256>(args, "v45_trt_replica_8x12_b256", d_x, d_w, d_y, shift, h_ref, h_y);
  CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_w));
  return 0;
}
