if(WIN32)
    cmake_minimum_required(VERSION 3.23)
else()
    cmake_minimum_required(VERSION 3.20.5)
endif()

project(BilateralSYCL LANGUAGES CXX)

find_package(IntelSYCL REQUIRED CONFIG)

add_library(bilateralsycl SHARED source.cpp kernel.cpp)

target_include_directories(bilateralsycl PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})

set_target_properties(bilateralsycl PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(
    TARGET bilateralsycl
    SOURCES source.cpp kernel.cpp)

if(WIN32)
    install(TARGETS bilateralsycl RUNTIME)
elseif(DEFINED VS_INSTALL_DIR)
    install(TARGETS bilateralsycl LIBRARY DESTINATION ${VS_INSTALL_DIR})
else()
    install(TARGETS bilateralsycl
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
