#include "resnet_stem_common.cuh"

namespace {

// v48: match TRT IMMA count/behavior. TRT 240 IMMA = (M/16)*(N/8)*(K/16)
// = 4*6*10: one CTA computes 64 OC x 48 conv-points, K 147->160. Fully unrolled
// (no N-group loop) so static IMMA = 4 OC-grp * 6 N * 5(k32) PTX = 240 SASS.
// One warp owns the whole 64x48 slab; goal is instruction-count parity first.
constexpr int N_TILE = 96;           // conv points per CTA (6 x n8)
constexpr int NG6 = N_TILE / 8;      // 6
constexpr int OCG = 4;               // 64 OC / 16

__global__ void v48_kernel(const int8_t* __restrict__ x,
                           const uint32_t* __restrict__ w_mma4,
                           int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t b4[N_TILE * K_GROUPS_MMA];
  int n0 = blockIdx.x * N_TILE;
  int tid = threadIdx.x, lid = tid & 31, gid = lid >> 2, lig = lid & 3;
  for (int i = tid; i < N_TILE * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA, cn = n0 + n;
    int cx = cn % CONV_OW, cy = cn / CONV_OW, v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e; v[e] = 0;
      if (k < K_TOTAL) {
        int ic = k / 49, iy = cy * 2 + (k / 7) % 7 - 3, ix = cx * 2 + k % 7 - 3;
        if (cy < CONV_OH && iy >= 0 && iy < IH && ix >= 0 && ix < IW)
          v[e] = x[(ic * IH + iy) * IW + ix];
      }
    }
    b4[n * K_GROUPS_MMA + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
  __syncthreads();
  if (tid < 32) {
    int32_t acc[OCG][NG6][4] = {};
#pragma unroll
    for (int kb = 0; kb < K_PAD_MMA; kb += 32) {
      int kgb = kb >> 2;
#pragma unroll
      for (int ng = 0; ng < NG6; ng++) {
        uint32_t b[2];
#pragma unroll
        for (int br = 0; br < 2; ++br)
          b[br] = b4[(ng * 8 + gid) * K_GROUPS_MMA + kgb + lig + br * 4];
#pragma unroll
        for (int og = 0; og < OCG; ++og) {
          uint32_t a[4];
#pragma unroll
          for (int ar = 0; ar < 4; ++ar) {
            int row = (ar & 1) ? gid + 8 : gid, kg = kgb + lig + (ar >= 2 ? 4 : 0);
            a[ar] = w_mma4[(og * 16 + row) * K_GROUPS_MMA + kg];
          }
          mma_m16n8k32_s8(acc[og][ng], a, b);
        }
      }
    }
#pragma unroll
    for (int ng = 0; ng < NG6; ng++)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? gid : gid + 8, n = ng * 8 + lig * 2 + (i & 1), cn = n0 + n;
        int cx = cn % CONV_OW, cy = cn / CONV_OW;
        if (cy < CONV_OH)
#pragma unroll
          for (int og = 0; og < OCG; ++og)
            y[(og * 16 + row) * CONV_OH * CONV_OW + cy * CONV_OW + cx] =
                clamp_relu_i8(acc[og][ng][i], shift);
      }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); constexpr int sh = 9;
  std::vector<int8_t> hx(IC * IH * IW), hw(OC * IC * KH * KW),
      hcr(OC * CONV_OH * CONV_OW), hy(OC * CONV_OH * CONV_OW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8, 8);
  for (auto& v : hx) v = d(rng); for (auto& v : hw) v = d(rng);
  auto hw4 = pack_weights_mma4(hw);
  // conv+relu reference only (pool comes in later versions)
  for (int oc = 0; oc < OC; ++oc)
    for (int oh = 0; oh < CONV_OH; ++oh)
      for (int ow = 0; ow < CONV_OW; ++ow) {
        int acc = 0;
        for (int ic = 0; ic < IC; ++ic)
          for (int ky = 0; ky < KH; ++ky) { int iy = oh * 2 + ky - 3; if (iy < 0 || iy >= IH) continue;
            for (int kx = 0; kx < KW; ++kx) { int ix = ow * 2 + kx - 3; if (ix < 0 || ix >= IW) continue;
              acc += int(hx[(ic * IH + iy) * IW + ix]) * int(hw[((oc * IC + ic) * KH + ky) * KW + kx]); } }
        hcr[(oc * CONV_OH + oh) * CONV_OW + ow] = clamp_relu_i8(acc, sh);
      }
  int8_t *dx, *dy; uint32_t* dw;
  CUDA_CHECK(cudaMalloc(&dx, hx.size())); CUDA_CHECK(cudaMalloc(&dy, hy.size()));
  CUDA_CHECK(cudaMalloc(&dw, hw4.size() * 4));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw4.data(), hw4.size() * 4, cudaMemcpyHostToDevice));
  dim3 g((CONV_OH * CONV_OW + N_TILE - 1) / N_TILE, 1, 1);
  float ms = time_kernel([&] { v48_kernel<<<g, 256>>>(dx, dw, dy, sh); }, a.warmup, a.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size(), cudaMemcpyDeviceToHost));
  print_result(a, "v48_240imma_64x96_conv", ms, max_abs_err(hcr, hy));
  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dw));
  return 0;
}
