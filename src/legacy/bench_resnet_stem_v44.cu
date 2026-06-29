// v44: CUTLASS Ampere int8 implicit-GEMM conv (NHWC) + ReLU/scale epilogue,
// then a fused 3x3 s2 maxpool kernel. This links the real CUTLASS tensorop conv
// (the family TRT generates) to chase the 0.0108 ms core. Channels padded 3->16
// for IMMA alignment. Times conv, pool, and conv+pool against TRT.
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/util/device_memory.h"

#define CK(s) do{ cutlass::Status e=(s); if(e!=cutlass::Status::kSuccess){ \
  printf("cutlass err %d @%d\n",int(e),__LINE__); std::exit(1);} }while(0)
#define CU(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ \
  printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__); std::exit(1);} }while(0)

using ElIn = int8_t; using ElOut = int8_t; using ElAcc = int32_t; using ElCmp = float;
using LIn = cutlass::layout::TensorNHWC; using LOut = cutlass::layout::TensorNHWC;
using Epi = cutlass::epilogue::thread::LinearCombinationRelu<
    ElOut, 16, ElAcc, ElCmp, cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
using Kern = cutlass::conv::kernel::DefaultConv2dFprop<
    ElIn, LIn, ElIn, LIn, ElOut, LOut, ElAcc, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<128,128,64>,
    cutlass::gemm::GemmShape<64,64,64>, cutlass::gemm::GemmShape<16,8,32>, Epi,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kern>;

constexpr int IH=224,IW=224,ICp=16,OC=64,KH=7,KW=7,OH=112,OW=112,PH=56,PW=56;

__global__ void pool_kernel(const int8_t* c,int8_t* y){
  int px=blockIdx.x*blockDim.x+threadIdx.x, py=blockIdx.y, oc=blockIdx.z;
  if(px>=PW) return; int best=0;
  for(int ky=0;ky<3;++ky){int cy=py*2+ky-1; if(cy<0||cy>=OH)continue;
    for(int kx=0;kx<3;++kx){int cx=px*2+kx-1; if(cx<0||cx>=OW)continue;
      best=max(best,int(c[(cy*OW+cx)*OC+oc]));}}
  y[(py*PW+px)*OC+oc]=(int8_t)best;
}

int main(int argc,char**argv){
  int iters=1000,warm=100;
  cutlass::conv::Conv2dProblemSize ps(1,IH,IW,ICp,OC,KH,KW,OH,OW,3,3,2,2,1,1,
      cutlass::conv::Mode::kCrossCorrelation,1);
  std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-8,8);
  std::vector<int8_t> hx(1*IH*IW*ICp,0),hw(OC*KH*KW*ICp,0);
  for(int i=0;i<IH*IW;++i)for(int c=0;c<3;++c)hx[i*ICp+c]=(int8_t)d(rng);
  for(int o=0;o<OC;++o)for(int r=0;r<KH*KW;++r)for(int c=0;c<3;++c)hw[(o*KH*KW+r)*ICp+c]=(int8_t)d(rng);
  int8_t*dx,*dw,*dc,*dy; CU(cudaMalloc(&dx,hx.size())); CU(cudaMalloc(&dw,hw.size()));
  CU(cudaMalloc(&dc,size_t(OH)*OW*OC)); CU(cudaMalloc(&dy,size_t(PH)*PW*OC));
  CU(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dw,hw.data(),hw.size(),cudaMemcpyHostToDevice));
  typename Conv::Arguments args{ps,
    {dx,LIn::packed({1,IH,IW,ICp})},{dw,LIn::packed({OC,KH,KW,ICp})},
    {dc,LOut::packed({1,OH,OW,OC})},{dc,LOut::packed({1,OH,OW,OC})},
    {1.0f/512.0f}};
  Conv op; size_t ws=op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> wsp(ws);
  CK(op.can_implement(args)); CK(op.initialize(args,wsp.get()));
  dim3 pg((PW+31)/32,PH,OC),pb(32);
  auto pool=[&]{pool_kernel<<<pg,pb>>>(dc,dy);};
  for(int i=0;i<warm;++i){CK(op());pool();} CU(cudaDeviceSynchronize());
  cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
  CU(cudaEventRecord(a)); for(int i=0;i<iters;++i)CK(op()); CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
  float mc; CU(cudaEventElapsedTime(&mc,a,b));
  CU(cudaEventRecord(a)); for(int i=0;i<iters;++i){CK(op());pool();} CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
  float mt; CU(cudaEventElapsedTime(&mt,a,b));
  printf("v44_cutlass_conv         %.6f ms\n",mc/iters);
  printf("v44_cutlass_conv_pool    %.6f ms\n",mt/iters);
  return 0;
}
