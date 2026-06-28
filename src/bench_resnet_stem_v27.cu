#include "resnet_stem_common.cuh"

namespace {

constexpr int TILE = 4;
constexpr int BLOCK_THREADS = 256;
constexpr int OC_GROUPS = 2;
constexpr int OC_PER_GROUP = 16;
constexpr int OC_PER_BLOCK = OC_GROUPS * OC_PER_GROUP;
constexpr int CONV_TILE = TILE * 2 + 1;
constexpr int N_TOTAL = CONV_TILE * CONV_TILE;
constexpr int GROUP_PAIRS = 6;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ bool n_in_pool(int n, int out_i) {
  int conv_x = n % CONV_TILE;
  int conv_y = n / CONV_TILE;
  int lx = out_i % TILE;
  int ly = out_i / TILE;
  int base_x = lx * 2;
  int base_y = ly * 2;
  return conv_x >= base_x && conv_x < base_x + 3 && conv_y >= base_y &&
         conv_y < base_y + 3;
}

__device__ __forceinline__ int subgroup_max4(int v) {
  v = max(v, __shfl_xor_sync(FULL_MASK, v, 1));
  v = max(v, __shfl_xor_sync(FULL_MASK, v, 2));
  return v;
}

__global__ void fused_ptx_mma_oc32_register_pool_kernel(
    const int8_t* __restrict__ x, const uint32_t* __restrict__ w_mma4,
    int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t smem_b4[N_TOTAL][K_GROUPS_MMA];
  __shared__ int8_t partial_pool[GROUP_PAIRS][OC_PER_BLOCK][TILE * TILE];

  int tile_px = blockIdx.x * TILE;
  int tile_py = blockIdx.y * TILE;
  int oc_base = blockIdx.z * OC_PER_BLOCK;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  int group_id = lane_id >> 2;
  int lane_in_group = lane_id & 3;
  bool interior = tile_px > 0 && tile_py > 0 && tile_px + TILE < POOL_OW &&
                  tile_py + TILE < POOL_OH;

  for (int i = tid; i < GROUP_PAIRS * OC_PER_BLOCK * TILE * TILE;
       i += blockDim.x) {
    partial_pool[0][0][i] = 0;
  }

  for (int i = tid; i < N_TOTAL * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA;
    int n = i / K_GROUPS_MMA;
    int conv_x = n % CONV_TILE;
    int conv_y = n / CONV_TILE;
    int cx = tile_px * 2 - 1 + conv_x;
    int cy = tile_py * 2 - 1 + conv_y;
    int vals[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e;
      int v = 0;
      if (k < K_TOTAL) {
        int kx = k % KW;
        int ky = (k / KW) % KH;
        int ic = k / (KH * KW);
        int iy = cy * 2 + ky - 3;
        int ix = cx * 2 + kx - 3;
        if (interior) {
          v = int(x[(ic * IH + iy) * IW + ix]);
        } else if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH) {
          if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
            v = int(x[(ic * IH + iy) * IW + ix]);
          }
        }
      }
      vals[e] = v;
    }
    smem_b4[n][kg] = pack_s8x4(vals[0], vals[1], vals[2], vals[3]);
  }
  __syncthreads();

  if (warp_id < GROUP_PAIRS) {
    int ng0 = warp_id * 2;
    int n_start0 = ng0 * 8;
    int n_start1 = n_start0 + 8;
    bool has_n1 = (ng0 + 1) < ((N_TOTAL + 7) / 8);
    int32_t acc[2][OC_GROUPS][4] = {};

#pragma unroll
    for (int k_base = 0; k_base < K_PAD_MMA; k_base += 32) {
      int kg_base = k_base >> 2;
      uint32_t b0[2];
      uint32_t b1[2];
#pragma unroll
      for (int br = 0; br < 2; ++br) {
        int kg = kg_base + lane_in_group + br * 4;
        int n0 = n_start0 + group_id;
        int n1 = n_start1 + group_id;
        b0[br] = (n0 < N_TOTAL) ? smem_b4[n0][kg] : 0;
        b1[br] = (has_n1 && n1 < N_TOTAL) ? smem_b4[n1][kg] : 0;
      }

#pragma unroll
      for (int og = 0; og < OC_GROUPS; ++og) {
        uint32_t a[4];
#pragma unroll
        for (int ar = 0; ar < 4; ++ar) {
          int row = (ar & 1) ? group_id + 8 : group_id;
          int kg = kg_base + lane_in_group + ((ar >= 2) ? 4 : 0);
          a[ar] = w_mma4[(oc_base + og * OC_PER_GROUP + row) * K_GROUPS_MMA +
                         kg];
        }
        mma_m16n8k32_s8(acc[0][og], a, b0);
        mma_m16n8k32_s8(acc[1][og], a, b1);
      }
    }

#pragma unroll
    for (int out_i = 0; out_i < TILE * TILE; ++out_i) {
#pragma unroll
      for (int og = 0; og < OC_GROUPS; ++og) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          int row = group_id + half * 8;
          int local_best = 0;
#pragma unroll
          for (int bit = 0; bit < 2; ++bit) {
            int i = half * 2 + bit;
            int n0 = n_start0 + lane_in_group * 2 + bit;
            if (n0 < N_TOTAL && n_in_pool(n0, out_i)) {
              local_best = max(local_best, int(clamp_relu_i8(acc[0][og][i],
                                                             shift)));
            }
            int n1 = n_start1 + lane_in_group * 2 + bit;
            if (has_n1 && n1 < N_TOTAL && n_in_pool(n1, out_i)) {
              local_best = max(local_best, int(clamp_relu_i8(acc[1][og][i],
                                                             shift)));
            }
          }
          int best = subgroup_max4(local_best);
          if (lane_in_group == 0) {
            partial_pool[warp_id][og * OC_PER_GROUP + row][out_i] =
                static_cast<int8_t>(best);
          }
        }
      }
    }
  }
  __syncthreads();

  constexpr int OUT_ELEMS = TILE * TILE * OC_PER_BLOCK;
  for (int idx = tid; idx < OUT_ELEMS; idx += blockDim.x) {
    int c = idx & (OC_PER_BLOCK - 1);
    int out_i = idx >> 5;
    int lx = out_i % TILE;
    int ly = out_i / TILE;
    int px = tile_px + lx;
    int py = tile_py + ly;
    if (px < POOL_OW && py < POOL_OH) {
      int best = 0;
#pragma unroll
      for (int gp = 0; gp < GROUP_PAIRS; ++gp) {
        best = max(best, int(partial_pool[gp][c][out_i]));
      }
      y[(oc_base + c) * POOL_OH * POOL_OW + py * POOL_OW + px] =
          static_cast<int8_t>(best);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  constexpr int shift = 9;
  constexpr int x_count = N * IC * IH * IW;
  constexpr int w_count = OC * IC * KH * KW;
  constexpr int y_count = OC * POOL_OH * POOL_OW;

  std::vector<int8_t> h_x(x_count), h_w(w_count), h_ref(y_count), h_y(y_count);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> dist(-8, 8);
  for (auto& v : h_x) v = static_cast<int8_t>(dist(rng));
  for (auto& v : h_w) v = static_cast<int8_t>(dist(rng));
  std::vector<uint32_t> h_w_mma4 = pack_weights_mma4(h_w);
  cpu_reference(h_x, h_w, h_ref, shift);

  int8_t* d_x = nullptr;
  int8_t* d_y = nullptr;
  uint32_t* d_w_mma4 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x_count));
  CUDA_CHECK(cudaMalloc(&d_y, y_count));
  CUDA_CHECK(cudaMalloc(&d_w_mma4, h_w_mma4.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w_mma4, h_w_mma4.data(),
                        h_w_mma4.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  dim3 grid((POOL_OW + TILE - 1) / TILE, (POOL_OH + TILE - 1) / TILE, OC / 32);
  float ms = time_kernel([&] {
    fused_ptx_mma_oc32_register_pool_kernel<<<grid, BLOCK_THREADS>>>(
        d_x, d_w_mma4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size(), cudaMemcpyDeviceToHost));
  print_result(args, "ptx_mma_oc32_register_pool_4x4_w8_b256", ms,
               max_abs_err(h_ref, h_y));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_w_mma4));
  return 0;
}
