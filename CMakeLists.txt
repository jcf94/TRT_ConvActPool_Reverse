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

add_executable(bench_resnet_stem_v25 src/bench_resnet_stem_v25.cu)
target_compile_options(bench_resnet_stem_v25 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v26 src/bench_resnet_stem_v26.cu)
target_compile_options(bench_resnet_stem_v26 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v27 src/bench_resnet_stem_v27.cu)
target_compile_options(bench_resnet_stem_v27 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v28 src/bench_resnet_stem_v28.cu)
target_compile_options(bench_resnet_stem_v28 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v29 src/bench_resnet_stem_v29.cu)
target_compile_options(bench_resnet_stem_v29 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v30 src/bench_resnet_stem_v30.cu)
target_compile_options(bench_resnet_stem_v30 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v31 src/bench_resnet_stem_v31.cu)
target_compile_options(bench_resnet_stem_v31 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v32 src/bench_resnet_stem_v32.cu)
target_compile_options(bench_resnet_stem_v32 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v33 src/bench_resnet_stem_v33.cu)
target_compile_options(bench_resnet_stem_v33 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v34 src/bench_resnet_stem_v34.cu)
target_compile_options(bench_resnet_stem_v34 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v35 src/bench_resnet_stem_v35.cu)
target_compile_options(bench_resnet_stem_v35 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v36 src/bench_resnet_stem_v36.cu)
target_compile_options(bench_resnet_stem_v36 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v37 src/bench_resnet_stem_v37.cu)
target_compile_options(bench_resnet_stem_v37 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v38 src/bench_resnet_stem_v38.cu)
target_compile_options(bench_resnet_stem_v38 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v39 src/bench_resnet_stem_v39.cu)
target_compile_options(bench_resnet_stem_v39 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v40 src/bench_resnet_stem_v40.cu)
target_compile_options(bench_resnet_stem_v40 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v41 src/bench_resnet_stem_v41.cu)
target_compile_options(bench_resnet_stem_v41 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(bench_resnet_stem_v42 src/bench_resnet_stem_v42.cu)
target_compile_options(bench_resnet_stem_v42 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
add_executable(bench_resnet_stem_v43 src/bench_resnet_stem_v43.cu)
target_compile_options(bench_resnet_stem_v43 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
