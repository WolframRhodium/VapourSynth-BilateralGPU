project(BilateralGPU_RTC LANGUAGES CXX)

add_library(bilateralgpu_cl SHARED bilateral.cpp)

target_include_directories(bilateralgpu_cl PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})

find_package(OpenCL MODULE)
if(OpenCL_FOUND)
    target_include_directories(bilateralgpu_cl PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(bilateralgpu_cl PRIVATE ${OpenCL_LIBRARIES})
endif() # OpenCL_FOUND

set_target_properties(bilateralgpu_cl PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

if(WIN32)
    install(TARGETS bilateralgpu_cl RUNTIME)
else()
    install(TARGETS bilateralgpu_cl LIBRARY DESTINATION ${VS_INSTALL_DIR})
endif()
