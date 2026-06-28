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

add_executable(bench_resnet_stem_v5 src/bench_resnet_stem_v5.cu)
target_compile_options(bench_resnet_stem_v5 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v6 src/bench_resnet_stem_v6.cu)
target_compile_options(bench_resnet_stem_v6 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v7 src/bench_resnet_stem_v7.cu)
target_compile_options(bench_resnet_stem_v7 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v8 src/bench_resnet_stem_v8.cu)
target_compile_options(bench_resnet_stem_v8 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v9 src/bench_resnet_stem_v9.cu)
target_compile_options(bench_resnet_stem_v9 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v10 src/bench_resnet_stem_v10.cu)
target_compile_options(bench_resnet_stem_v10 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v11 src/bench_resnet_stem_v11.cu)
target_compile_options(bench_resnet_stem_v11 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v12 src/bench_resnet_stem_v12.cu)
target_compile_options(bench_resnet_stem_v12 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v13 src/bench_resnet_stem_v13.cu)
target_compile_options(bench_resnet_stem_v13 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v14 src/bench_resnet_stem_v14.cu)
target_compile_options(bench_resnet_stem_v14 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v15 src/bench_resnet_stem_v15.cu)
target_compile_options(bench_resnet_stem_v15 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v16 src/bench_resnet_stem_v16.cu)
target_compile_options(bench_resnet_stem_v16 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v17 src/bench_resnet_stem_v17.cu)
target_compile_options(bench_resnet_stem_v17 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v18 src/bench_resnet_stem_v18.cu)
target_compile_options(bench_resnet_stem_v18 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v19 src/bench_resnet_stem_v19.cu)
target_compile_options(bench_resnet_stem_v19 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v20 src/bench_resnet_stem_v20.cu)
target_compile_options(bench_resnet_stem_v20 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v21 src/bench_resnet_stem_v21.cu)
target_compile_options(bench_resnet_stem_v21 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v22 src/bench_resnet_stem_v22.cu)
target_compile_options(bench_resnet_stem_v22 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v23 src/bench_resnet_stem_v23.cu)
target_compile_options(bench_resnet_stem_v23 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v24 src/bench_resnet_stem_v24.cu)
target_compile_options(bench_resnet_stem_v24 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
