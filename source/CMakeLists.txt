cmake_minimum_required(VERSION 3.20)

project(BilateralGPU LANGUAGES CXX CUDA)

add_library(bilateralgpu_kernel OBJECT kernel.cu)
set_target_properties(bilateralgpu_kernel PROPERTIES
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

add_library(bilateralgpu_source OBJECT bilateral.cpp)
target_include_directories(bilateralgpu_source PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bilateralgpu_source PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_library(bilateralgpu SHARED)
target_link_libraries(bilateralgpu bilateralgpu_source bilateralgpu_kernel)
