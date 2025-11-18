# ./taichi/runtime/program_impls/cuda_c/CMakeLists.txt

add_library(cuda_c_program_impl)
target_sources(cuda_c_program_impl
  PRIVATE
    cuda_c_program.cpp
  )

target_include_directories(cuda_c_program_impl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  )

