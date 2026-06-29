#include "resnet_stem_common.cuh"
namespace {
// v54: shrink smem to ~3KB. Stage raw input patch (3ch) once; build B fragments
// in registers per ng -> no 15KB packed-B. cr int8 6KB pool. occupancy lever.
constexpr int CB_H=8,CB_W=12,N_TILE=CB_H*CB_W,NG=N_TILE/8,OCG=4,PB_H=3,PB_W=5;
constexpr int PH=CB_H*2+6,PW=CB_W*2+6; // input patch 22x30
__device__ __forceinline__ uint32_t vmax_s8x4(uint32_t a,uint32_t b){uint32_t r;asm("vmax4.s32.s32.s32 %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(0));return r;}
__global__ void v56_kernel(const int8_t* __restrict__ x,const uint32_t* __restrict__ w,int8_t* __restrict__ y,int shift){
 __shared__ int8_t patch[3*PH*PW]; __shared__ uint32_t cr[N_TILE*16];
 int lane=threadIdx.x&31,gid=lane>>2,lig=lane&3;constexpr int GX=(POOL_OW+PB_W-1)/PB_W;
 int gx0=(blockIdx.x%GX)*PB_W,gy0=blockIdx.x/GX*PB_H,bx=gx0*2-1,by=gy0*2-1;
 int ox=bx*2-3,oy=by*2-3; // top-left input
 for(int i=threadIdx.x;i<3*PH*PW;i+=blockDim.x){int p=i%(PH*PW),ic=i/(PH*PW),iy=oy+p/PW,ix=ox+p%PW;patch[i]=(iy>=0&&iy<IH&&ix>=0&&ix<IW)?x[(ic*IH+iy)*IW+ix]:0;}
 __syncthreads();
 int warp=threadIdx.x>>5;
 for(int ng=warp;ng<NG;ng+=6){int32_t acc[OCG][4]={};
  for(int kb=0;kb<K_PAD_MMA;kb+=32){int kgb=kb>>2;uint32_t b[2];
   #pragma unroll
   for(int br=0;br<2;++br){int n=ng*8+gid,cx=n%CB_W,cy=n/CB_W,v[4];
    #pragma unroll
    for(int e=0;e<4;++e){int k=(kgb+lig+br*4)*4+e;v[e]=0;if(k<K_TOTAL){int ic=k/49,py=cy*2+(k/7)%7,px=cx*2+k%7;v[e]=patch[ic*PH*PW+py*PW+px];}}
    b[br]=pack_s8x4(v[0],v[1],v[2],v[3]);}
   #pragma unroll
   for(int og=0;og<OCG;++og){uint32_t a[4];
    #pragma unroll
    for(int ar=0;ar<4;++ar){int row=(ar&1)?gid+8:gid,kg=kgb+lig+(ar>=2?4:0);a[ar]=w[(og*16+row)*K_GROUPS_MMA+kg];}
    mma_m16n8k32_s8(acc[og],a,b);}}
  #pragma unroll
  for(int i=0;i<4;++i){int row=(i<2)?gid:gid+8,n=ng*8+lig*2+(i&1);
   #pragma unroll
   for(int og=0;og<OCG;++og)((int8_t*)cr)[n*64+og*16+row]=clamp_relu_i8(acc[og][i],shift);}}
 __syncthreads();
 for(int o=threadIdx.x;o<PB_H*PB_W;o+=blockDim.x){int px=o%PB_W,py=o/PB_W,gx=gx0+px,gy=gy0+py;if(gx>=POOL_OW||gy>=POOL_OH)continue;uint32_t best[16]={};
  #pragma unroll
  for(int ky=0;ky<3;++ky)
  #pragma unroll
  for(int kx=0;kx<3;++kx){int cx=px*2+kx,cy=py*2+ky,s=cy*CB_W+cx;
   #pragma unroll
   for(int c=0;c<16;++c)best[c]=vmax_s8x4(best[c],cr[s*16+c]);}
  int yb=gy*POOL_OW+gx;
  #pragma unroll
  for(int c=0;c<16;++c){uint32_t v=best[c];y[(c*4)*POOL_OH*POOL_OW+yb]=v;y[(c*4+1)*POOL_OH*POOL_OW+yb]=v>>8;y[(c*4+2)*POOL_OH*POOL_OW+yb]=v>>16;y[(c*4+3)*POOL_OH*POOL_OW+yb]=v>>24;}}
}
}
int main(int argc,char**argv){Args a=parse_args(argc,argv);constexpr int sh=9;
 std::vector<int8_t> hx(IC*IH*IW),hw(OC*IC*KH*KW),hr(OC*POOL_OH*POOL_OW),hy(OC*POOL_OH*POOL_OW);
 std::mt19937 rng(1234);std::uniform_int_distribution<int> d(-8,8);for(auto&v:hx)v=d(rng);for(auto&v:hw)v=d(rng);
 auto hw4=pack_weights_mma4(hw);cpu_reference(hx,hw,hr,sh);
 int8_t *dx,*dy;uint32_t*dw;CUDA_CHECK(cudaMalloc(&dx,hx.size()));CUDA_CHECK(cudaMalloc(&dy,hy.size()));CUDA_CHECK(cudaMalloc(&dw,hw4.size()*4));
 CUDA_CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dw,hw4.data(),hw4.size()*4,cudaMemcpyHostToDevice));
 int GX=(POOL_OW+PB_W-1)/PB_W,GY=(POOL_OH+PB_H-1)/PB_H;dim3 g(GX*GY,1,1);
 float ms=time_kernel([&]{v56_kernel<<<g,192>>>(dx,dw,dy,sh);},a.warmup,a.iters);
 CUDA_CHECK(cudaMemcpy(hy.data(),dy,hy.size(),cudaMemcpyDeviceToHost));print_result(a,"v56_6warp",ms,max_abs_err(hr,hy));
 CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dy));CUDA_CHECK(cudaFree(dw));return 0;}
