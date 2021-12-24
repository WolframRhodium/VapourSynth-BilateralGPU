cmake_minimum_required(VERSION 3.20)

project(BilateralGPU_RTC LANGUAGES CXX)

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(WARNING "NVRTC is not available for 32-bit system")
else()
    FIND_PACKAGE(CUDAToolkit REQUIRED)

    add_library(bilateralgpu_rtc SHARED bilateral.cpp)
    target_include_directories(bilateralgpu_rtc PRIVATE
        ${VAPOURSYNTH_INCLUDE_DIRECTORY}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(bilateralgpu_rtc PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)
    target_link_libraries(bilateralgpu_rtc PRIVATE CUDA::cuda_driver CUDA::nvrtc)

    install(TARGETS bilateralgpu_rtc
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
