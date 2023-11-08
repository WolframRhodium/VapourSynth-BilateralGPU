cmake_minimum_required(VERSION 3.20)

project(VapourSynth-BilateralGPU LANGUAGES CXX)

set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend")
set(ENABLE_CUDA_RTC ON CACHE BOOL "Enable CUDA NVRTC backend")
set(USE_NVRTC_STATIC ON CACHE BOOL "Whether to use NVRTC static library")

set(ENABLE_SYCL OFF CACHE BOOL "Enable SYCL backend")

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

if(NOT VAPOURSYNTH_INCLUDE_DIRECTORY)
    find_package(PkgConfig QUIET MODULE)

    if(PKG_CONFIG_FOUND)
        pkg_search_module(VS vapoursynth)

        if(VS_FOUND)
            message(STATUS "Found VapourSynth r${VS_VERSION}")

            set(VAPOURSYNTH_INCLUDE_DIRECTORY ${VS_INCLUDE_DIRS})

            cmake_path(APPEND VS_INSTALL_DIR ${VS_LIBDIR} vapoursynth)
        endif()
    endif()
endif()


find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    if(VCS_TAG)
        string(STRIP ${VCS_TAG} VCS_TAG)
    endif()
endif()


if(VCS_TAG)
    message(STATUS "VapourSynth-BilateralGPU ${VCS_TAG}")
else()
    message(WARNING "unknown plugin version")
    set(VCS_TAG "unknown")
endif()

configure_file(config.h.in config.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(ENABLE_CUDA)
    add_subdirectory(source)
endif() # ENABLE_CUDA

if(ENABLE_CUDA_RTC)
    add_subdirectory(rtc_source)
endif() # ENABLE_CUDA_RTC

if(ENABLE_SYCL)
    add_subdirectory(sycl_source)
endif() # ENABLE_CUDA_RTC
