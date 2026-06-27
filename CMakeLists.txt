cmake_minimum_required(VERSION 3.18)
project(cuda_op_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(bench_resnet_stem src/bench_resnet_stem.cu)
target_compile_options(bench_resnet_stem PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v2 src/bench_resnet_stem_v2.cu)
target_compile_options(bench_resnet_stem_v2 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v3 src/bench_resnet_stem_v3.cu)
target_compile_options(bench_resnet_stem_v3 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v4 src/bench_resnet_stem_v4.cu)
target_link_libraries(bench_resnet_stem_v4 PRIVATE cublas)
target_compile_options(bench_resnet_stem_v4 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
