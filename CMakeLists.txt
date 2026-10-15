cmake_minimum_required(VERSION 3.22)

# GPU architecture: default Hopper (GH200 = sm_90). Override with
#   cmake -DCOLDCORE_CUDA_ARCHS=80   (A100)  or  "80;90"  etc.
set(COLDCORE_CUDA_ARCHS "90" CACHE STRING "CUDA architectures to compile for")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${COLDCORE_CUDA_ARCHS})
endif()

project(coldcore VERSION 0.3.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(COLDCORE_BUILD_PLUGINS "Build problem plugins" ON)

# ------------------------------------------------------------------ core
# libcoldcore: state-core utilities shared by host and plugins —
# device probing (coherence, HBM size) and mode-tagged allocation
# (HBM / ATS-LPDDR / managed) implementing the automatic placement policy.
add_library(coldcore SHARED src/core/core.cu)
target_include_directories(coldcore PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)
set_target_properties(coldcore PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ------------------------------------------------------------------ plugins
if(COLDCORE_BUILD_PLUGINS)
  add_subdirectory(src/plugins/covering)
  add_subdirectory(src/plugins/sparse)
endif()

message(STATUS "coldcore: CUDA archs = ${CMAKE_CUDA_ARCHITECTURES}, "
               "CUDA compiler = ${CMAKE_CUDA_COMPILER}")
