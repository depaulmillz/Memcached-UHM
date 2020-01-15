cmake_minimum_required (VERSION 3.10)

project(MemcUHM CXX CUDA)

add_custom_target(memcached COMMAND make WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/memcached)

ENABLE_TESTING()
ADD_SUBDIRECTORY(uhm)
