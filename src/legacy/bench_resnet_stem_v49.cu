#include "resnet_stem_common.cuh"

namespace {

// v49: keep 240 IMMA, align epilogue behavior with TRT. CTA owns 64 OC x an
// 8x12 conv block (=96 pts, 4*12*5 PTX mma = 240 IMMA). Epilogue mimics TRT:
// I2FP (acc->f32) -> FFMA (scale,bias) -> F2IP-style ReLU+quant, pooled 3x3s2
// in registers, vectorized STG.  conv 8x12 -> pool 3x5 outputs per CTA.
constexpr int CB_H = 8, CB_W = 12, N_TILE = CB_H * CB_W;  // 96
constexpr int NG = N_TILE / 8, OCG = 4;
constexpr int PB_H = 3, PB_W = 5;  // pool outputs (rows pad-1 stride2)

__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a, uint32_t b) {
  uint32_t r; asm("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(0)); return r;
}

__global__ void v49_kernel(const int8_t* __restrict__ x,
                           const uint32_t* __restrict__ w_mma4,
                           const float* __restrict__ scale, int8_t* __restrict__ y, int shift) {
  __shared__ uint32_t b4[N_TILE * K_GROUPS_MMA];
  __shared__ int8_t cr[OC * N_TILE];  // conv-relu int8, [oc][n]
  int bx = (blockIdx.x % (CONV_OW / CB_W)) * CB_W, by = blockIdx.x / (CONV_OW / CB_W) * CB_H;
  int tid = threadIdx.x, lid = tid & 31, gid = lid >> 2, lig = lid & 3, wid = tid >> 5;
  for (int i = tid; i < N_TILE * K_GROUPS_MMA; i += blockDim.x) {
    int kg = i % K_GROUPS_MMA, n = i / K_GROUPS_MMA, cx = bx + n % CB_W, cy = by + n / CB_W, v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int k = kg * 4 + e; v[e] = 0;
      if (k < K_TOTAL) { int ic = k / 49, iy = cy * 2 + (k / 7) % 7 - 3, ix = cx * 2 + k % 7 - 3;
        if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) v[e] = x[(ic * IH + iy) * IW + ix]; }
    }
    b4[n * K_GROUPS_MMA + kg] = pack_s8x4(v[0], v[1], v[2], v[3]);
  }
  __syncthreads();
  if (tid < 32) {
    int32_t acc[OCG][NG][4] = {};
#pragma unroll
    for (int kb = 0; kb < K_PAD_MMA; kb += 32) { int kgb = kb >> 2;
#pragma unroll
      for (int ng = 0; ng < NG; ng++) { uint32_t b[2];
#pragma unroll
        for (int br = 0; br < 2; ++br) b[br] = b4[(ng * 8 + gid) * K_GROUPS_MMA + kgb + lig + br * 4];
#pragma unroll
        for (int og = 0; og < OCG; ++og) { uint32_t a[4];
#pragma unroll
          for (int ar = 0; ar < 4; ++ar) { int row = (ar & 1) ? gid + 8 : gid, kg = kgb + lig + (ar >= 2 ? 4 : 0);
            a[ar] = w_mma4[(og * 16 + row) * K_GROUPS_MMA + kg]; }
          mma_m16n8k32_s8(acc[og][ng], a, b); } } }
#pragma unroll
    for (int ng = 0; ng < NG; ng++)
#pragma unroll
      for (int i = 0; i < 4; ++i) { int row = (i < 2) ? gid : gid + 8, n = ng * 8 + lig * 2 + (i & 1);
#pragma unroll
        for (int og = 0; og < OCG; ++og) {
          float f = scale[og * 16 + row] * acc[og][ng][i];   // I2FP + FFMA
          cr[(og * 16 + row) * N_TILE + n] = (int8_t)max(min(__float2int_rn(f), 127), 0); } }  // F2IP relu+quant
  }
  __syncthreads();
  for (int o = tid; o < PB_H * PB_W; o += blockDim.x) {
    int px = o % PB_W, py = o / PB_W, gx = (bx / 2) + px, gy = (by / 2) + py;
    if (gx >= POOL_OW || gy >= POOL_OH) continue;
    uint32_t best[16] = {};
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) { int cx = px * 2 + kx - 1 + (bx & 1), cy = py * 2 + ky - 1 + (by & 1);
        if (cx < 0 || cx >= CB_W || cy < 0 || cy >= CB_H) continue; int s = cy * CB_W + cx;
#pragma unroll
        for (int c = 0; c < 16; ++c) best[c] = vmax_s8x4(best[c], pack_s8x4(cr[(c*4)*N_TILE+s], cr[(c*4+1)*N_TILE+s], cr[(c*4+2)*N_TILE+s], cr[(c*4+3)*N_TILE+s])); }
    int yb = gy * POOL_OW + gx;
#pragma unroll
    for (int c = 0; c < 16; ++c) { uint32_t v = best[c];
      y[(c*4)*POOL_OH*POOL_OW+yb]=v; y[(c*4+1)*POOL_OH*POOL_OW+yb]=v>>8; y[(c*4+2)*POOL_OH*POOL_OW+yb]=v>>16; y[(c*4+3)*POOL_OH*POOL_OW+yb]=v>>24; }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); constexpr int sh = 9;
  std::vector<int8_t> hx(IC*IH*IW), hw(OC*IC*KH*KW), hr(OC*POOL_OH*POOL_OW), hy(OC*POOL_OH*POOL_OW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8,8);
  for (auto&v:hx) v=d(rng); for (auto&v:hw) v=d(rng);
  auto hw4=pack_weights_mma4(hw); cpu_reference(hx,hw,hr,sh);
  std::vector<float> hs(OC, 1.0f/512.0f);  // shift 9 ~ /512
  int8_t *dx,*dy; uint32_t* dw; float* ds;
  CUDA_CHECK(cudaMalloc(&dx,hx.size())); CUDA_CHECK(cudaMalloc(&dy,hy.size()));
  CUDA_CHECK(cudaMalloc(&dw,hw4.size()*4)); CUDA_CHECK(cudaMalloc(&ds,OC*4));
  CUDA_CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw,hw4.data(),hw4.size()*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds,hs.data(),OC*4,cudaMemcpyHostToDevice));
  dim3 g((CONV_OW/CB_W)*(CONV_OH/CB_H),1,1);
  float ms=time_kernel([&]{v49_kernel<<<g,256>>>(dx,dw,ds,dy,sh);},a.warmup,a.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(),dy,hy.size(),cudaMemcpyDeviceToHost));
  print_result(a,"v49_240imma_pool_8x12",ms,max_abs_err(hr,hy));
  CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dy));CUDA_CHECK(cudaFree(dw));CUDA_CHECK(cudaFree(ds));
  return 0;
}
