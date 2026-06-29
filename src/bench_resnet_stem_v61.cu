#include "resnet_stem_common.cuh"

namespace {

// v50: keep 240 IMMA, cut REG/STG. NG-outer loop frees acc[4][4] each column so
// only ~16 live -> low REG (TRT 128). int8 conv-relu staged once, pool reads
// 64 OC contiguous (vmax4) with vectorized stores. integer shift => err=0.
constexpr int CB_H = 8, CB_W = 12, N_TILE = CB_H * CB_W, NG = N_TILE / 8, OCG = 4;
constexpr int PB_H = 3, PB_W = 5;

__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a, uint32_t b) {
  uint32_t r; asm("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(0)); return r;
}

__global__ void v61_kernel(const int8_t* __restrict__ x, const uint32_t* __restrict__ w,
                           int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t b4[N_TILE * 24];
  __shared__ uint32_t cr[N_TILE * 16];  // [n][64oc] packed int8x4
  int bx = (blockIdx.x % (CONV_OW / CB_W)) * CB_W, by = blockIdx.x / (CONV_OW / CB_W) * CB_H;
  int tid = threadIdx.x, lid = tid & 31, gid = lid >> 2, lig = lid & 3;
  int32_t acc[NG][OCG][4] = {};
  for (int kc = 0; kc < 2; ++kc) { int kg0 = kc * 20;
    for (int i = tid; i < N_TILE * 20; i += blockDim.x) {
      int kg = i % 20, n = i / 20, cx = bx + n % CB_W, cy = by + n / CB_W, v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) { int k = (kg0+kg) * 4 + e; v[e] = 0; if (k < K_TOTAL) {
        int ic = k / 49, iy = cy * 2 + (k / 7) % 7 - 3, ix = cx * 2 + k % 7 - 3;
        if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) v[e] = x[(ic * IH + iy) * IW + ix]; } }
      b4[n * 20 + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();
    if (tid < 32)
#pragma unroll
    for (int ng = 0; ng < NG; ng++) {
#pragma unroll
      for (int kb = 0; kb < 80; kb += 32) { int kgb = kb >> 2; uint32_t b[2];
#pragma unroll
        for (int br = 0; br < 2; ++br) b[br] = b4[(ng * 8 + gid) * 20 + kgb + lig + br * 4];
#pragma unroll
        for (int og = 0; og < OCG; ++og) { uint32_t a[4];
#pragma unroll
          for (int ar = 0; ar < 4; ++ar) { int row = (ar & 1) ? gid + 8 : gid, kg = kg0 + kgb + lig + (ar >= 2 ? 4 : 0);
            a[ar] = w[(og * 16 + row) * K_GROUPS_MMA + kg]; }
          mma_m16n8k32_s8(acc[ng][og], a, b); } }
    }
    __syncthreads();
  }
  if (tid < 32)
#pragma unroll
    for (int ng = 0; ng < NG; ng++)
#pragma unroll
      for (int i = 0; i < 4; ++i) { int row = (i < 2) ? gid : gid + 8, n = ng * 8 + lig * 2 + (i & 1);
#pragma unroll
        for (int og = 0; og < OCG; ++og)
          ((int8_t*)cr)[n * 64 + og * 16 + row] = clamp_relu_i8(acc[ng][og][i], shift); }
  __syncthreads();
  for (int o = tid; o < PB_H * PB_W; o += blockDim.x) {
    int px = o % PB_W, py = o / PB_W, gx = bx / 2 + px, gy = by / 2 + py;
    if (gx >= POOL_OW || gy >= POOL_OH) continue; uint32_t best[16] = {};
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) { int cx = px * 2 + kx - 1, cy = py * 2 + ky - 1;
        if (cx < 0 || cx >= CB_W || cy < 0 || cy >= CB_H) continue; int s = cy * CB_W + cx;
#pragma unroll
        for (int c = 0; c < 16; ++c) best[c] = vmax_s8x4(best[c], cr[s * 16 + c]); }
    int yb = (gy * POOL_OW + gx)*OC; uint4* yo=(uint4*)(y+yb);
#pragma unroll
    for (int q=0;q<4;++q) yo[q]=make_uint4(best[q*4],best[q*4+1],best[q*4+2],best[q*4+3]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); constexpr int sh = 9;
  std::vector<int8_t> hx(IC*IH*IW), hw(OC*IC*KH*KW), hr(OC*POOL_OH*POOL_OW), hy(OC*POOL_OH*POOL_OW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8,8);
  for (auto&v:hx) v=d(rng); for (auto&v:hw) v=d(rng);
  auto hw4=pack_weights_mma4(hw); cpu_reference(hx,hw,hr,sh);
  std::vector<int8_t> hr_n(OC*POOL_OH*POOL_OW);for(int c=0;c<OC;++c)for(int p=0;p<POOL_OH*POOL_OW;++p)hr_n[p*OC+c]=hr[c*POOL_OH*POOL_OW+p];
  int8_t *dx,*dy; uint32_t* dw;
  CUDA_CHECK(cudaMalloc(&dx,hx.size())); CUDA_CHECK(cudaMalloc(&dy,hy.size())); CUDA_CHECK(cudaMalloc(&dw,hw4.size()*4));
  CUDA_CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw,hw4.data(),hw4.size()*4,cudaMemcpyHostToDevice));
  dim3 g((CONV_OW/CB_W)*(CONV_OH/CB_H),1,1);
  float ms=time_kernel([&]{v61_kernel<<<g,256>>>(dx,dw,dy,sh);},a.warmup,a.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(),dy,hy.size(),cudaMemcpyDeviceToHost));
  print_result(a,"v61_kstream",ms,max_abs_err(hr_n,hy));
  CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dy));CUDA_CHECK(cudaFree(dw)); return 0;
}
