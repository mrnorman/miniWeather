cmake_minimum_required(VERSION 3.8)
project(miniWeather)

enable_language(CXX)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

enable_testing()

include_directories(${PNETCDF_PATH}/include)


############################################################
## Set Parameters
############################################################
if ("${NX}" STREQUAL "")
  SET(NX 100)
endif()
if ("${NZ}" STREQUAL "")
  SET(NZ 50)
endif()
if ("${SIM_TIME}" STREQUAL "")
  SET(SIM_TIME 1000)
endif()
if ("${OUT_FREQ}" STREQUAL "")
  SET(OUT_FREQ 10)
endif()
if ("${DATA_SPEC}" STREQUAL "")
  SET(DATA_SPEC DATA_SPEC_THERMAL)
endif()
SET(EXE_DEFS "-D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}")
SET(TEST_DEFS "-D_NX=100 -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=400 -D_DATA_SPEC=DATA_SPEC_THERMAL")


############################################################
## Check for Parallel NetCDF
############################################################
if ("${PNETCDF_PATH}" STREQUAL "")
  MESSAGE(FATAL_ERROR "ERROR: PNETCDF_PATH CMake variable not set.\n"
                      "miniWeather requires a parallel-netcdf installation <https://trac.mcs.anl.gov/projects/parallel-netcdf>.\n"
                      "Specify -DPNETCDF_PATH=/path/to/pnetcdf in your cmake configure line.\n\n")
endif()


############################################################
## Append CXXFLAGS
############################################################
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAGS}")


############################################################
## Add YAKL 
############################################################
add_subdirectory(YAKL)
if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
  set_property(TARGET yakl PROPERTY CUDA_ARCHITECTURES OFF)
  set_property(TARGET yakl PROPERTY CUDA_STANDARD 14)
endif()
include_directories(YAKL)

link_libraries(yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")


############################################################
## Compile the serial version
############################################################
add_executable(serial miniWeather_serial.cpp)
set_target_properties(serial PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(serial_test miniWeather_serial.cpp)
set_target_properties(serial_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

add_test(NAME SERIAL_TEST COMMAND ./check_output.sh ./serial_test 1e-9 4.5e-5 )


############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.cpp)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(mpi_test miniWeather_mpi.cpp)
set_target_properties(mpi_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

add_test(NAME MPI_TEST COMMAND ./check_output.sh ./mpi_test 1e-9 4.5e-5 ) 


############################################################
## YAKL parallel_for Version
############################################################
add_executable(parallelfor miniWeather_mpi_parallelfor.cpp)
set_target_properties(parallelfor PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(parallelfor_test miniWeather_mpi_parallelfor.cpp)
set_target_properties(parallelfor_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

add_test(NAME YAKL_TEST COMMAND ./check_output.sh ./parallelfor_test 1e-9 4.5e-5 )


############################################################
## YAKL parallel_for Version
############################################################
add_executable(parallelfor_simd miniWeather_mpi_parallelfor_simd.cpp)
set_target_properties(parallelfor_simd PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(parallelfor_simd_test miniWeather_mpi_parallelfor_simd.cpp)
set_target_properties(parallelfor_simd_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

add_test(NAME YAKL_SIMD_TEST COMMAND ./check_output.sh ./parallelfor_test 1e-9 4.5e-5 )


include(YAKL/yakl_utils.cmake)
yakl_process_target(serial)
yakl_process_target(serial_test)
yakl_process_target(mpi)
yakl_process_target(mpi_test)
yakl_process_target(parallelfor)
yakl_process_target(parallelfor_test)
yakl_process_target(parallelfor_simd)
yakl_process_target(parallelfor_simd_test)


if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(serial serial_test mpi mpi_test parallelfor parallelfor_test parallelfor_simd parallelfor_simd_test PROPERTIES LINKER_LANGUAGE CXX)
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_target_properties(serial serial_test mpi mpi_test parallelfor parallelfor_test parallelfor_simd parallelfor_simd_test PROPERTIES CUDA_ARCHITECTURES OFF CUDA_STANDARD 14)
  endif()
endif()



