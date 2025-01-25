cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(miniWeather CUDA CXX)

include(cmake/CPM.cmake)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG main
    # The following is required to make the `CCCL::cudax` target available:
    OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# If you're building an executable
add_executable(miniWeather_cudastf miniWeather_cudastf.cu)

target_link_libraries(miniWeather_cudastf PUBLIC cuda)

if (CMAKE_CUDA_COMPILER)
    target_compile_options(miniWeather_cudastf PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_options(miniWeather_cudastf PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

target_link_libraries(miniWeather_cudastf PRIVATE CCCL::CCCL CCCL::cudax)
