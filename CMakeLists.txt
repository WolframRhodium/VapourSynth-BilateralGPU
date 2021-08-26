cmake_minimum_required(VERSION 3.20)

project(VapourSynth-BilateralGPU LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

add_subdirectory(source)
add_subdirectory(rtc_source)
