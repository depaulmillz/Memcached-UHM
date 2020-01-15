cmake_minimum_required (VERSION 3.10)

project(UHM CXX CUDA)

set(default_build_type "Debug")

SET(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -G -pg")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

if(NOT DEFINED CUDA_ARCH)
    message("WARNING CUDA_ARCH IS BEING SET TO DEFAULT")
    set(CUDA_ARCH 61)
endif(NOT DEFINED CUDA_ARCH)

string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fopenmp -Xptxas=-v -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -rdc=true -Wreorder")

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)

add_library(StdUHM STATIC simplImpl.cu)
add_library(MemcUHM SHARED memcimpl.cu)
target_link_libraries(MemcUHM -lcudadevrt -lcudart_static)

