cmake_minimum_required (VERSION 3.10)

project(UHMTests CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

string(APPEND CMAKE_CUDA_FLAGS " -G")
message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")

add_executable(AllocatorTest AllocatorTest.cu)
target_link_libraries(AllocatorTest -lgomp)
add_test(AllocatorTest AllocatorTest)

add_executable(HashMapTest HashMapTest.cu)
target_link_libraries(HashMapTest -lgomp)
add_test(HashMapTest HashMapTest)

add_executable(StdImplTest StdImplTest.cc)
target_link_libraries(StdImplTest StdUHM)
add_test(StdImplTest StdImplTest)
