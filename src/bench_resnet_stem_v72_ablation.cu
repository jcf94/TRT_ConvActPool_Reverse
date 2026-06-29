#include "resnet_stem_common.cuh"

namespace {

// v72 ablation: isolate the cost of the MaxPool and ReLU epilogue stages.
//
// All three kernels share the identical v72 Conv 7x7 MMA core (240 IMMA, 3-stage
// cp.async K-stream on the 14x9 halo tile). They differ only in the epilogue:
//   1. full      : Conv + ReLU + MaxPool, pooled NHWC output (the v72 baseline).
//   2. nopool    : Conv + ReLU, writes the conv-resolution NHWC tile (no pool).
//   3. conv_only : Conv only (no ReLU, no pool), writes conv-resolution NHWC.
//
// nopool/conv_only emit the full 112x112 conv tensor, so the comparison includes
// the larger store volume that pooling would otherwise collapse 4:1. Halo conv
// points are written by multiple CTAs with identical values, so the result stays
// bit-exact against the matching CPU reference.
constexpr int CB_H = 9, CB_W = 14, N_TILE = CB_H * CB_W, NPAD = 128, NG = NPAD / 8, OCG = 4;
constexpr int PB_H = 4, PB_W = 6;
constexpr int KC = 8;                          // kg per k32 chunk
constexpr int NCH = K_GROUPS_MMA / KC;         // 5 chunks
constexpr int WARPS = 4, NG_PW = NG / WARPS;   // 3 ng/warp -> 240 IMMA total
constexpr int NPTS = CONV_OH * CONV_OW;

// Symmetric int8 quantization without the ReLU floor, used by the conv_only case.
__host__ __device__ __forceinline__ int8_t clamp_i8(int acc, int shift) {
  acc = acc >> shift;
  acc = max(acc, -128);
  acc = min(acc, 127);
  return static_cast<int8_t>(acc);
}

__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a, uint32_t b) {
  uint32_t r; asm("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(0)); return r;
}

__device__ __forceinline__ void cpasync16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst), "l"(src));
}

__global__ void pack_input(const int8_t* __restrict__ x, uint32_t* __restrict__ b) {
  int n = blockIdx.x; int cx = n % CONV_OW, cy = n / CONV_OW;
  for (int kg = threadIdx.x; kg < K_GROUPS_MMA; kg += blockDim.x) {
    int v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) { int k = kg * 4 + e; v[e] = 0; if (k < K_TOTAL) {
      int ic = k / 49, iy = cy * 2 + (k / 7) % 7 - 3, ix = cx * 2 + k % 7 - 3;
      if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) v[e] = x[(ic * IH + iy) * IW + ix]; } }
    b[n * K_GROUPS_MMA + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
}

// Shared v72 conv core: runs the cp.async K-stream MMA pipeline and writes the
// quantized 14x9 conv halo tile into cr (NHWC int8). RELU toggles the activation.
template <bool RELU>
__device__ __forceinline__ void v72_conv(const uint32_t* __restrict__ b,
                                         const uint32_t* __restrict__ w,
                                         uint32_t* cr, int shift, int bx, int by) {
  constexpr int ST = 3;
  __shared__ uint32_t bb[3][NPAD * KC];
  int tid = threadIdx.x, lid = tid & 31, gid = lid >> 2, lig = lid & 3, warp = tid >> 5;
  int32_t acc[NG_PW][OCG][4] = {};

  for (int s = 0; s < ST - 1; ++s) {
    for (int i = tid; i < NPAD * 2; i += blockDim.x) { int n = i >> 1, h = i & 1; bool ok_=n<N_TILE&&(by+n/CB_W)>=0&&(by+n/CB_W)<CONV_OH&&(bx+n%CB_W)>=0&&(bx+n%CB_W)<CONV_OW; int gn=ok_?(by+n/CB_W)*CONV_OW+(bx+n%CB_W):0; if(ok_)cpasync16((uint32_t)__cvta_generic_to_shared(&bb[s][n*KC+h*4]), &b[gn * K_GROUPS_MMA + s * KC + h * 4]); else{bb[s][n*KC+h*4]=0;bb[s][n*KC+h*4+1]=0;bb[s][n*KC+h*4+2]=0;bb[s][n*KC+h*4+3]=0;} }
    asm volatile("cp.async.commit_group;\n" ::);
  }

  for (int ch = 0; ch < NCH; ++ch) { int cur = ch % ST, nx = ch + ST - 1;
    if (nx < NCH) { int wb = nx % ST;
      for (int i = tid; i < NPAD * 2; i += blockDim.x) { int n = i >> 1, h = i & 1; bool ok_=n<N_TILE&&(by+n/CB_W)>=0&&(by+n/CB_W)<CONV_OH&&(bx+n%CB_W)>=0&&(bx+n%CB_W)<CONV_OW; int gn=ok_?(by+n/CB_W)*CONV_OW+(bx+n%CB_W):0; if(ok_)cpasync16((uint32_t)__cvta_generic_to_shared(&bb[wb][n*KC+h*4]), &b[gn * K_GROUPS_MMA + nx * KC + h * 4]); else{bb[wb][n*KC+h*4]=0;bb[wb][n*KC+h*4+1]=0;bb[wb][n*KC+h*4+2]=0;bb[wb][n*KC+h*4+3]=0;} }
      asm volatile("cp.async.commit_group;\n" ::); }
    asm volatile("cp.async.wait_group 1;\n" ::); __syncthreads();
#pragma unroll
    for (int j = 0; j < NG_PW; ++j) { int ng = warp + j * WARPS; uint32_t bf[2];
#pragma unroll
      for (int br = 0; br < 2; ++br) bf[br] = bb[cur][(ng * 8 + gid) * KC + lig + br * 4];
#pragma unroll
      for (int og = 0; og < OCG; ++og) { uint32_t a[4];
#pragma unroll
        for (int ar = 0; ar < 4; ++ar) { int row = (ar & 1) ? gid + 8 : gid, kg = ch * KC + lig + (ar >= 2 ? 4 : 0);
          a[ar] = w[(og * 16 + row) * K_GROUPS_MMA + kg]; }
        mma_m16n8k32_s8(acc[j][og], a, bf); } }
  }

#pragma unroll
  for (int j = 0; j < NG_PW; ++j) { int ng = warp + j * WARPS;
#pragma unroll
    for (int i = 0; i < 4; ++i) { int row = (i < 2) ? gid : gid + 8, n = ng * 8 + lig * 2 + (i & 1);
#pragma unroll
      for (int og = 0; og < OCG; ++og)
        ((int8_t*)cr)[n * 64 + og * 16 + row] =
            RELU ? clamp_relu_i8(acc[j][og][i], shift) : clamp_i8(acc[j][og][i], shift); } }
  __syncthreads();
}

// Block geometry helper: one CTA owns one PB_H x PB_W pool tile; bx/by are the
// top-left conv coords including the 3x3/s2 pool halo.
__device__ __forceinline__ void block_origin(int& gx0, int& gy0, int& bx, int& by) {
  constexpr int GX = (POOL_OW + PB_W - 1) / PB_W;
  gx0 = (blockIdx.x % GX) * PB_W; gy0 = blockIdx.x / GX * PB_H; bx = gx0 * 2 - 1; by = gy0 * 2 - 1;
}

// 1. Full v72: Conv + ReLU + MaxPool.
__global__ void full_kernel(const uint32_t* __restrict__ b, const uint32_t* __restrict__ w,
                            int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t cr_s[N_TILE * 16]; uint32_t* cr = cr_s;
  int gx0, gy0, bx, by; block_origin(gx0, gy0, bx, by);
  v72_conv<true>(b, w, cr, shift, bx, by);
  for (int t = threadIdx.x; t < PB_H * PB_W * 4; t += blockDim.x) {
    int o = t >> 2, q = t & 3, px = o % PB_W, py = o / PB_W, gx = gx0 + px, gy = gy0 + py;
    if (gx >= POOL_OW || gy >= POOL_OH) continue; uint32_t best[4] = {};
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) { int cx = px * 2 + kx, cy = py * 2 + ky;
        if (cx < 0 || cx >= CB_W || cy < 0 || cy >= CB_H) continue; int s = cy * CB_W + cx;
#pragma unroll
        for (int c = 0; c < 4; ++c) best[c] = vmax_s8x4(best[c], cr[s * 16 + q * 4 + c]); }
    int yb = (gy * POOL_OW + gx) * OC; uint4* yo = (uint4*)(y + yb);
    yo[q] = make_uint4(best[0], best[1], best[2], best[3]);
  }
}

// 2/3. No-pool ablation: write the conv-resolution NHWC tile, ReLU toggled.
template <bool RELU>
__global__ void nopool_kernel(const uint32_t* __restrict__ b, const uint32_t* __restrict__ w,
                              int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t cr_s[N_TILE * 16]; uint32_t* cr = cr_s;
  int gx0, gy0, bx, by; block_origin(gx0, gy0, bx, by);
  v72_conv<RELU>(b, w, cr, shift, bx, by);
  for (int t = threadIdx.x; t < N_TILE * 4; t += blockDim.x) {
    int o = t >> 2, q = t & 3, cx = bx + o % CB_W, cy = by + o / CB_W;
    if (cx < 0 || cx >= CONV_OW || cy < 0 || cy >= CONV_OH) continue;
    int yb = (cy * CONV_OW + cx) * OC; uint4* yo = (uint4*)(y + yb);
    yo[q] = make_uint4(cr[o * 16 + q * 4 + 0], cr[o * 16 + q * 4 + 1],
                       cr[o * 16 + q * 4 + 2], cr[o * 16 + q * 4 + 3]);
  }
}

// Conv-resolution NHWC reference, ReLU toggled, matching the ablation kernels.
static void conv_reference_nhwc(const std::vector<int8_t>& x, const std::vector<int8_t>& w,
                                std::vector<int8_t>& y_nhwc, int shift, bool relu) {
  for (int oc = 0; oc < OC; ++oc)
    for (int oh = 0; oh < CONV_OH; ++oh)
      for (int ow = 0; ow < CONV_OW; ++ow) {
        int acc = 0;
        for (int ic = 0; ic < IC; ++ic)
          for (int ky = 0; ky < KH; ++ky) { int iy = oh * 2 + ky - 3; if (iy < 0 || iy >= IH) continue;
            for (int kx = 0; kx < KW; ++kx) { int ix = ow * 2 + kx - 3; if (ix < 0 || ix >= IW) continue;
              acc += int(x[(ic * IH + iy) * IW + ix]) * int(w[((oc * IC + ic) * KH + ky) * KW + kx]); } }
        y_nhwc[(oh * CONV_OW + ow) * OC + oc] = relu ? clamp_relu_i8(acc, shift) : clamp_i8(acc, shift);
      }
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); constexpr int sh = 9;

  std::vector<int8_t> hx(IC*IH*IW), hw(OC*IC*KH*KW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8,8);
  for (auto&v:hx) v=d(rng); for (auto&v:hw) v=d(rng);
  auto hw4 = pack_weights_mma4(hw);

  // References: pooled NHWC (full) and conv-resolution NHWC (nopool/conv_only).
  std::vector<int8_t> hr_pool(OC*POOL_OH*POOL_OW); cpu_reference(hx, hw, hr_pool, sh);
  std::vector<int8_t> ref_pool_n(OC*POOL_OH*POOL_OW);
  for (int c=0;c<OC;++c) for (int p=0;p<POOL_OH*POOL_OW;++p) ref_pool_n[p*OC+c]=hr_pool[c*POOL_OH*POOL_OW+p];
  std::vector<int8_t> ref_relu_n(OC*CONV_OH*CONV_OW), ref_conv_n(OC*CONV_OH*CONV_OW);
  conv_reference_nhwc(hx, hw, ref_relu_n, sh, true);
  conv_reference_nhwc(hx, hw, ref_conv_n, sh, false);

  int8_t *dx; uint32_t *dw, *db; int8_t *dy_pool, *dy_conv;
  CUDA_CHECK(cudaMalloc(&dx, hx.size()));
  CUDA_CHECK(cudaMalloc(&dw, hw4.size()*4));
  CUDA_CHECK(cudaMalloc(&db, (size_t)NPTS*K_GROUPS_MMA*4));
  CUDA_CHECK(cudaMalloc(&dy_pool, (size_t)OC*POOL_OH*POOL_OW));
  CUDA_CHECK(cudaMalloc(&dy_conv, (size_t)OC*CONV_OH*CONV_OW));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw4.data(), hw4.size()*4, cudaMemcpyHostToDevice));
  pack_input<<<NPTS,40>>>(dx, db);

  dim3 g(((POOL_OW+PB_W-1)/PB_W)*((POOL_OH+PB_H-1)/PB_H),1,1);
  std::vector<int8_t> hy_pool(OC*POOL_OH*POOL_OW), hy_conv(OC*CONV_OH*CONV_OW);

  float ms_full = time_kernel([&]{ full_kernel<<<g,WARPS*32>>>(db,dw,dy_pool,sh); }, a.warmup, a.iters);
  CUDA_CHECK(cudaMemcpy(hy_pool.data(), dy_pool, hy_pool.size(), cudaMemcpyDeviceToHost));
  print_result(a, "v72_full(conv+relu+pool)", ms_full, max_abs_err(ref_pool_n, hy_pool));

  float ms_nopool = time_kernel([&]{ nopool_kernel<true><<<g,WARPS*32>>>(db,dw,dy_conv,sh); }, a.warmup, a.iters);
  CUDA_CHECK(cudaMemcpy(hy_conv.data(), dy_conv, hy_conv.size(), cudaMemcpyDeviceToHost));
  print_result(a, "v72_nopool(conv+relu)", ms_nopool, max_abs_err(ref_relu_n, hy_conv));

  float ms_conv = time_kernel([&]{ nopool_kernel<false><<<g,WARPS*32>>>(db,dw,dy_conv,sh); }, a.warmup, a.iters);
  CUDA_CHECK(cudaMemcpy(hy_conv.data(), dy_conv, hy_conv.size(), cudaMemcpyDeviceToHost));
  print_result(a, "v72_conv_only(no relu,no pool)", ms_conv, max_abs_err(ref_conv_n, hy_conv));

  CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dy_pool)); CUDA_CHECK(cudaFree(dy_conv));
  return 0;
}
