#include "resnet_stem_common.cuh"

namespace {

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ int subgroup_max4(int v) {
  v = max(v, __shfl_xor_sync(FULL_MASK, v, 1));
  v = max(v, __shfl_xor_sync(FULL_MASK, v, 2));
  return v;
}

template <int WARPS>
__global__ void fused_ptx_mma_oc32_pool_owner_8n_kernel(
    const int8_t* __restrict__ x, const uint32_t* __restrict__ w_mma4,
    int8_t* __restrict__ y, int shift) {
  constexpr int OC_GROUPS = 2;
  constexpr int OC_PER_GROUP = 16;
  constexpr int OC_PER_BLOCK = OC_GROUPS * OC_PER_GROUP;
  int task = blockIdx.x * WARPS + (threadIdx.x >> 5);
  int lane_id = threadIdx.x & 31;
  int group_id = lane_id >> 2;
  int lane_in_group = lane_id & 3;

  int total_tasks = POOL_OH * POOL_OW * (OC / OC_PER_BLOCK);
  if (task >= total_tasks) return;

  int oc_group_block = task % (OC / OC_PER_BLOCK);
  int spatial = task / (OC / OC_PER_BLOCK);
  int px = spatial % POOL_OW;
  int py = spatial / POOL_OW;
  int oc_base = oc_group_block * OC_PER_BLOCK;
  int best[OC_GROUPS][2] = {};

#pragma unroll
  for (int cand_group = 0; cand_group < 2; ++cand_group) {
    int32_t acc[OC_GROUPS][4] = {};

#pragma unroll
    for (int k_base = 0; k_base < K_PAD_MMA; k_base += 32) {
      int kg_base = k_base >> 2;
      uint32_t b[2];
#pragma unroll
      for (int br = 0; br < 2; ++br) {
        int vals[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int k = k_base + lane_in_group * 4 + e + br * 16;
          int cand = (cand_group == 0) ? group_id : 8;
          int v = 0;
          if (k < K_TOTAL) {
            int wx = cand % 3;
            int wy = cand / 3;
            int cx = px * 2 + wx - 1;
            int cy = py * 2 + wy - 1;
            if (cx >= 0 && cx < CONV_OW && cy >= 0 && cy < CONV_OH) {
              int kx = k % KW;
              int ky = (k / KW) % KH;
              int ic = k / (KH * KW);
              int iy = cy * 2 + ky - 3;
              int ix = cx * 2 + kx - 3;
              if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
                v = int(x[(ic * IH + iy) * IW + ix]);
              }
            }
          }
          vals[e] = v;
        }
        b[br] = pack_s8x4(vals[0], vals[1], vals[2], vals[3]);
      }

#pragma unroll
      for (int og = 0; og < OC_GROUPS; ++og) {
        uint32_t a[4];
#pragma unroll
        for (int ar = 0; ar < 4; ++ar) {
          int row = (ar & 1) ? group_id + 8 : group_id;
          int kg = kg_base + lane_in_group + ((ar >= 2) ? 4 : 0);
          a[ar] = w_mma4[(oc_base + og * OC_PER_GROUP + row) *
                             K_GROUPS_MMA +
                         kg];
        }
        mma_m16n8k32_s8(acc[og], a, b);
      }
    }

#pragma unroll
    for (int og = 0; og < OC_GROUPS; ++og) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int local_best = best[og][half];
#pragma unroll
        for (int bit = 0; bit < 2; ++bit) {
          int i = half * 2 + bit;
          local_best = max(local_best, acc[og][i]);
        }
        if (cand_group == 0) {
          best[og][half] = subgroup_max4(local_best);
        } else {
          // Candidate 8 is broadcast into all N columns, so every lane in the
          // 4-lane subgroup observes the same two values.
          best[og][half] = max(best[og][half], local_best);
        }
      }
    }
  }

  if (lane_in_group == 0) {
#pragma unroll
    for (int og = 0; og < OC_GROUPS; ++og) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int row = group_id + half * 8;
        int oc = oc_base + og * OC_PER_GROUP + row;
        y[oc * POOL_OH * POOL_OW + py * POOL_OW + px] =
            clamp_relu_i8(best[og][half], shift);
      }
    }
  }
}

template <int WARPS>
static void run_case(const Args& args, const char* name, const int8_t* d_x,
                     const uint32_t* d_w_mma4, int8_t* d_y, int shift,
                     const std::vector<int8_t>& h_ref,
                     std::vector<int8_t>& h_y) {
  int tasks = POOL_OH * POOL_OW * (OC / 32);
  int block = WARPS * 32;
  int grid = (tasks + WARPS - 1) / WARPS;
  float ms = time_kernel([&] {
    fused_ptx_mma_oc32_pool_owner_8n_kernel<WARPS>
        <<<grid, block>>>(d_x, d_w_mma4, d_y, shift);
  }, args.warmup, args.iters);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size(), cudaMemcpyDeviceToHost));
  print_result(args, name, ms, max_abs_err(h_ref, h_y));
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

  run_case<4>(args, "ptx_mma_oc32_pool_owner_8n_w4_b128", d_x, d_w_mma4, d_y,
              shift, h_ref, h_y);
  run_case<8>(args, "ptx_mma_oc32_pool_owner_8n_w8_b256", d_x, d_w_mma4, d_y,
              shift, h_ref, h_y);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_w_mma4));
  return 0;
}
