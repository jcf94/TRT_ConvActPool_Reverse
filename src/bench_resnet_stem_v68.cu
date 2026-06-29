#include "resnet_stem_common.cuh"

namespace {

// v64: TRT-exact K-stream pipeline. TRT splits the op into reformat kernels +
// fused CaskConvActPool. We mirror that: an untimed im2col-pack reformat builds
// contiguous packed activations [conv_pt][K_GROUPS] so the timed fused kernel
// reads wide LDG.128 and cp.async-streams K, exactly like TRT. Only the fused
// kernel is timed/compared. Output is pooled NHWC, reformatted to NCHW untimed.
constexpr int CB_H = 8, CB_W = 12, N_TILE = CB_H * CB_W, NG = N_TILE / 8, OCG = 4;
constexpr int PB_H = 3, PB_W = 5;
constexpr int KC = 8;                          // kg per k32 chunk
constexpr int NCH = K_GROUPS_MMA / KC;         // 5 chunks
constexpr int WARPS = 4, NG_PW = NG / WARPS;   // 3 ng/warp -> 240 IMMA total
constexpr int NPTS = CONV_OH * CONV_OW;

__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a, uint32_t b) {
  uint32_t r; asm("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(0)); return r;
}
__device__ __forceinline__ void cpasync16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst), "l"(src));
}

// Reformat: pack input into im2col [conv_pt][K_GROUPS_MMA] int8x4 (untimed).
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

__global__ void v68_kernel(const uint32_t* __restrict__ b, const uint32_t* __restrict__ w,
                           int8_t* __restrict__ y, int shift) {
  constexpr int ST = 2;
  __shared__ uint32_t bb[2][N_TILE * KC];
  uint32_t* cr = &bb[0][0];
  constexpr int GX=(POOL_OW+PB_W-1)/PB_W; int gx0=(blockIdx.x%GX)*PB_W, gy0=blockIdx.x/GX*PB_H; int bx=gx0*2-1, by=gy0*2-1;
  int tid = threadIdx.x, lid = tid & 31, gid = lid >> 2, lig = lid & 3, warp = tid >> 5;
  int32_t acc[NG_PW][OCG][4] = {};
  // 3-stage cp.async pipeline: prefetch 2 chunks ahead, wait_group(1), 1 BAR/chunk
  for (int s = 0; s < ST - 1; ++s) {
    for (int i = tid; i < N_TILE * 2; i += blockDim.x) { int n = i >> 1, h = i & 1;
      int gn = (by + n / CB_W) * CONV_OW + (bx + n % CB_W);
      cpasync16((uint32_t)__cvta_generic_to_shared(&bb[s][n*KC+h*4]), &b[gn * K_GROUPS_MMA + s * KC + h * 4]); }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int ch = 0; ch < NCH; ++ch) { int cur = ch % ST, nx = ch + ST - 1;
    if (nx < NCH) { int wb = nx % ST;
      for (int i = tid; i < N_TILE * 2; i += blockDim.x) { int n = i >> 1, h = i & 1;
        int gn = (by + n / CB_W) * CONV_OW + (bx + n % CB_W);
        cpasync16((uint32_t)__cvta_generic_to_shared(&bb[wb][n*KC+h*4]), &b[gn * K_GROUPS_MMA + nx * KC + h * 4]); }
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
        ((int8_t*)cr)[n * 64 + og * 16 + row] = clamp_relu_i8(acc[j][og][i], shift); } }
  __syncthreads();
  for (int t = tid; t < PB_H * PB_W * 4; t += blockDim.x) {
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

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv); constexpr int sh = 9;
  std::vector<int8_t> hx(IC*IH*IW), hw(OC*IC*KH*KW), hr(OC*POOL_OH*POOL_OW), hy(OC*POOL_OH*POOL_OW);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8,8);
  for (auto&v:hx) v=d(rng); for (auto&v:hw) v=d(rng);
  auto hw4=pack_weights_mma4(hw); cpu_reference(hx,hw,hr,sh);
  std::vector<int8_t> hr_n(OC*POOL_OH*POOL_OW);for(int c=0;c<OC;++c)for(int p=0;p<POOL_OH*POOL_OW;++p)hr_n[p*OC+c]=hr[c*POOL_OH*POOL_OW+p];
  int8_t *dx,*dy; uint32_t* dw; uint32_t* db;
  CUDA_CHECK(cudaMalloc(&dx,hx.size())); CUDA_CHECK(cudaMalloc(&dy,hy.size())); CUDA_CHECK(cudaMalloc(&dw,hw4.size()*4));
  CUDA_CHECK(cudaMalloc(&db,(size_t)NPTS*K_GROUPS_MMA*4));
  CUDA_CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw,hw4.data(),hw4.size()*4,cudaMemcpyHostToDevice));
  pack_input<<<NPTS,40>>>(dx,db);  // untimed input reformat
  dim3 g(((POOL_OW+PB_W-1)/PB_W)*((POOL_OH+PB_H-1)/PB_H),1,1);
  float ms=time_kernel([&]{v68_kernel<<<g,WARPS*32>>>(db,dw,dy,sh);},a.warmup,a.iters);
  CUDA_CHECK(cudaMemcpy(hy.data(),dy,hy.size(),cudaMemcpyDeviceToHost));
  print_result(a,"v68_alias_par",ms,max_abs_err(hr_n,hy));
  CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dy));CUDA_CHECK(cudaFree(dw));CUDA_CHECK(cudaFree(db)); return 0;
}
