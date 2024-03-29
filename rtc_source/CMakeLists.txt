cmake_minimum_required(VERSION 3.26)

project(BilateralGPU_RTC LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_library(bilateralgpu_rtc SHARED bilateral.cpp)

target_include_directories(bilateralgpu_rtc PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(bilateralgpu_rtc PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(bilateralgpu_rtc PRIVATE CUDA::cuda_driver)

if (USE_NVRTC_STATIC AND (CUDAToolkit_VERSION_MAJOR GREATER_EQUAL "12" OR (
        CUDAToolkit_VERSION_MAJOR EQUAL "11" AND
        CUDAToolkit_VERSION_MINOR GREATER_EQUAL "5")))

    target_link_libraries(bilateralgpu_rtc PRIVATE CUDA::nvrtc_static)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_property(TARGET bilateralgpu_rtc PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()
else()
    if (USE_NVRTC_STATIC)
        message(WARNING "NVRTC static library is not used")
    endif()

    target_link_libraries(bilateralgpu_rtc PRIVATE CUDA::nvrtc)
endif()

if(WIN32)
    install(TARGETS bilateralgpu_rtc RUNTIME)
elseif(DEFINED VS_INSTALL_DIR)
    install(TARGETS bilateralgpu_rtc LIBRARY DESTINATION ${VS_INSTALL_DIR})
else()
    install(TARGETS bilateralgpu_rtc
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
