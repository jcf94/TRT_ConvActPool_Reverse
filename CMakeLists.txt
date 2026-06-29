cmake_minimum_required(VERSION 3.18)
project(cuda_op_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

function(add_resnet_stem_bench target source)
  add_executable(${target} ${source})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endfunction()

add_resnet_stem_bench(bench_resnet_stem src/bench_resnet_stem.cu)
add_resnet_stem_bench(bench_resnet_stem_v2 src/bench_resnet_stem_v2.cu)
add_resnet_stem_bench(bench_resnet_stem_v4 src/bench_resnet_stem_v4.cu)
target_link_libraries(bench_resnet_stem_v4 PRIVATE cublas)
add_resnet_stem_bench(bench_resnet_stem_v6 src/bench_resnet_stem_v6.cu)
add_resnet_stem_bench(bench_resnet_stem_v8 src/bench_resnet_stem_v8.cu)
add_resnet_stem_bench(bench_resnet_stem_v12 src/bench_resnet_stem_v12.cu)
add_resnet_stem_bench(bench_resnet_stem_v15 src/bench_resnet_stem_v15.cu)
add_resnet_stem_bench(bench_resnet_stem_v17 src/bench_resnet_stem_v17.cu)
add_resnet_stem_bench(bench_resnet_stem_v20 src/bench_resnet_stem_v20.cu)
add_resnet_stem_bench(bench_resnet_stem_v24 src/bench_resnet_stem_v24.cu)
add_resnet_stem_bench(bench_resnet_stem_v31 src/bench_resnet_stem_v31.cu)
add_resnet_stem_bench(bench_resnet_stem_v35 src/bench_resnet_stem_v35.cu)
add_resnet_stem_bench(bench_resnet_stem_v37 src/bench_resnet_stem_v37.cu)
add_resnet_stem_bench(bench_resnet_stem_v38 src/bench_resnet_stem_v38.cu)
add_resnet_stem_bench(bench_resnet_stem_v39 src/bench_resnet_stem_v39.cu)
add_resnet_stem_bench(bench_resnet_stem_v41 src/bench_resnet_stem_v41.cu)
add_resnet_stem_bench(bench_resnet_stem_v42 src/bench_resnet_stem_v42.cu)
add_resnet_stem_bench(bench_resnet_stem_v43 src/bench_resnet_stem_v43.cu)
add_resnet_stem_bench(bench_resnet_stem_v45 src/bench_resnet_stem_v45.cu)

add_executable(bench_resnet_stem_v44 src/bench_resnet_stem_v44.cu)
target_include_directories(bench_resnet_stem_v44 PRIVATE
  ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
  ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
)
target_compile_options(bench_resnet_stem_v44 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
