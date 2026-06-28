#include "resnet_stem_common.cuh"

namespace {

constexpr int TILE = 4;
constexpr int BLOCK_THREADS = 256;
constexpr int OC_GROUPS = 2;
constexpr int OC_PER_GROUP = 16;
constexpr int OC_PER_BLOCK = OC_GROUPS * OC_PER_GROUP;
constexpr int CONV_TILE = TILE * 2 + 1;
constexpr int N_TOTAL = CONV_TILE * CONV_TILE;

template <int NG0, bool HAS_N1>
__device__ __forceinline__ void mma_n_pair(
    const uint32_t* __restrict__ w_mma4, int oc_base, int group_id,
    int lane_in_group, int shift, const uint32_t smem_b4[N_TOTAL][K_GROUPS_MMA],
    int8_t conv_relu_tile[OC_PER_BLOCK][N_TOTAL]) {
  constexpr int N_START0 = NG0 * 8;
  constexpr int N_START1 = N_START0 + 8;
  int32_t acc0[OC_GROUPS][4] = {};
  int32_t acc1[OC_GROUPS][4] = {};

#pragma unroll
  for (int k_base = 0; k_base < K_PAD_MMA; k_base += 32) {
    int kg_base = k_base >> 2;
    uint32_t b0[2];
    uint32_t b1[2];
#pragma unroll
    for (int br = 0; br < 2; ++br) {
      int kg = kg_base + lane_in_group + br * 4;
      int n0 = N_START0 + group_id;
      int n1 = N_START1 + group_id;
      b0[br] = (n0 < N_TOTAL) ? smem_b4[n0][kg] : 0;
      if constexpr (HAS_N1) {
        b1[br] = (n1 < N_TOTAL) ? smem_b4[n1][kg] : 0;
      } else {
        b1[br] = 0;
      }
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
      mma_m16n8k32_s8(acc0[og], a, b0);
      if constexpr (HAS_N1) {
        mma_m16n8k32_s8(acc1[og], a, b1);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = (i < 2) ? group_id : group_id + 8;
    int n0 = N_START0 + lane_in_group * 2 + (i & 1);
    int n1 = N_START1 + lane_in_group * 2 + (i & 1);
#pragma unroll
    for (int og = 0; og < OC_GROUPS; ++og) {
      int c = og * OC_PER_GROUP + row;
      if (n0 < N_TOTAL) {
        conv_relu_tile[c][n0] = clamp_relu_i8(acc0[og][i], shift);
      }
      if constexpr (HAS_N1) {
        if (n1 < N_TOTAL) {
          conv_relu_tile[c][n1] = clamp_relu_i8(acc1[og][i], shift);
        }
      }
    }
  }
}

__global__ void fused_ptx_mma_oc32_static_n_kernel(
    const int8_t* __restrict__ x, const uint32_t* __restrict__ w_mma4,
    int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t smem_b4[N_TOTAL][K_GROUPS_MMA];
  __shared__ int8_t conv_relu_tile[OC_PER_BLOCK][N_TOTAL];

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

  if (warp_id == 0) {
    mma_n_pair<0, true>(w_mma4, oc_base, group_id, lane_in_group, shift,
                        smem_b4, conv_relu_tile);
  } else if (warp_id == 1) {
    mma_n_pair<2, true>(w_mma4, oc_base, group_id, lane_in_group, shift,
                        smem_b4, conv_relu_tile);
  } else if (warp_id == 2) {
    mma_n_pair<4, true>(w_mma4, oc_base, group_id, lane_in_group, shift,
                        smem_b4, conv_relu_tile);
  } else if (warp_id == 3) {
    mma_n_pair<6, true>(w_mma4, oc_base, group_id, lane_in_group, shift,
                        smem_b4, conv_relu_tile);
  } else if (warp_id == 4) {
    mma_n_pair<8, true>(w_mma4, oc_base, group_id, lane_in_group, shift,
                        smem_b4, conv_relu_tile);
  } else if (warp_id == 5) {
    mma_n_pair<10, false>(w_mma4, oc_base, group_id, lane_in_group, shift,
                          smem_b4, conv_relu_tile);
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
      int best[OC_PER_BLOCK] = {};
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
    fused_ptx_mma_oc32_static_n_kernel<<<grid, BLOCK_THREADS>>>(d_x, d_w_mma4,
                                                                d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size(), cudaMemcpyDeviceToHost));
  print_result(args, "ptx_mma_oc32_static_n_4x4_w8_b256", ms,
               max_abs_err(h_ref, h_y));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_w_mma4));
  return 0;
}
