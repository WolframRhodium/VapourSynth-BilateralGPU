cmake_minimum_required(VERSION 3.20)

project(BilateralSYCL LANGUAGES CXX)

find_package(IntelSYCL REQUIRED CONFIG)

add_library(bilateral_sycl SHARED source.cpp)

target_include_directories(bilateral_sycl PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})

set_target_properties(bilateral_sycl PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(
    TARGET bilateral_sycl
    SOURCES kernel.cpp)

install(TARGETS bilateral_sycl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})