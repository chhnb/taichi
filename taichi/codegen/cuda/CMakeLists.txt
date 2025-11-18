# ./taichi/codegen/cuda/CMakeLists.txt

add_library(cuda_codegen)
target_sources(cuda_codegen
  PRIVATE
    codegen_cuda.cpp
    ../cuda_c/codegen_cuda_c.cpp
    ../cuda_c/cuda_c_kernel_compiler.cpp
  )

target_include_directories(cuda_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(cuda_codegen PRIVATE taichi_util)
target_link_libraries(cuda_codegen PRIVATE cuda_runtime)
