cmake_minimum_required(VERSION 3.8)
project(miniWeather CXX)

if ("${ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

enable_testing()


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
#include_directories(cub)
include_directories(hipCUB)
include_directories(rocPRIM)


############################################################
## Compile the serial version
############################################################
add_executable(serial miniWeather_serial.cpp)
set_source_files_properties(miniWeather_serial.cpp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} -I${PNETCDF_PATH}/include")
target_link_libraries(serial yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")

add_executable(serial_test miniWeather_serial.cpp)
set_source_files_properties(miniWeather_serial.cpp PROPERTIES COMPILE_FLAGS "${TEST_DEFS} -I${PNETCDF_PATH}/include")
target_link_libraries(serial_test yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")

add_test(NAME SERIAL_TEST COMMAND ./check_output.sh ./serial_test 1e-9 4.5e-5 )


############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.cpp)
set_source_files_properties(miniWeather_mpi.cpp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} -I${PNETCDF_PATH}/include")
target_link_libraries(mpi yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")

add_executable(mpi_test miniWeather_mpi.cpp)
set_source_files_properties(miniWeather_mpi.cpp PROPERTIES COMPILE_FLAGS "${TEST_DEFS} -I${PNETCDF_PATH}/include")
target_link_libraries(mpi_test yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")

add_test(NAME MPI_TEST COMMAND ./check_output.sh ./mpi_test 1e-9 4.5e-5 ) 


############################################################
## YAKL Version
############################################################
add_executable(yaklexe miniWeather_mpi_parallelfor.cpp)
set_target_properties(yaklexe PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(yaklexe yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")
if ("${ARCH}" STREQUAL "CUDA")
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_property(TARGET yaklexe PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET yaklexe PROPERTY CUDA_STANDARD 11)
  endif()
  set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES LANGUAGE CUDA)
endif()
set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} -I${PNETCDF_PATH}/include ${YAKL_CXX_FLAGS}")

add_executable(yaklexe_test miniWeather_mpi_parallelfor.cpp)
set_target_properties(yaklexe_test PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(yaklexe_test yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")
if ("${ARCH}" STREQUAL "CUDA")
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_property(TARGET yaklexe_test PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET yaklexe_test PROPERTY CUDA_STANDARD 11)
  endif()
  set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES LANGUAGE CUDA)
endif()
set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES COMPILE_FLAGS "${TEST_DEFS} -I${PNETCDF_PATH}/include ${YAKL_CXX_FLAGS}")

add_test(NAME YAKL_TEST COMMAND ./check_output.sh ./yaklexe_test 1e-9 4.5e-5 )

############################################################
## Compile the C++17 Parallel Agorithms version
############################################################
SET(DO_STDPAR FALSE)
if("${STDPAR_FLAGS}" STREQUAL "")
  MESSAGE(STATUS "No flags provided for C++ standard parallelism, default C++ flags will be used.")
  MESSAGE(STATUS "    Add -DSTDPAR_FLAGS=\"...\" to your cmake configure line to add flags.")
endif()
add_executable(mpistdpar miniWeather_mpi_stdpar.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI" AND "${STDPAR_FLAGS}" STREQUAL "")
  set(STDPAR_FLAGS "-stdpar=gpu")
  MESSAGE(STATUS "    Detected NVIDIA compiler, adding \"-stdpar=gpu\" to CXX flags. Set STDPAR_FLAGS to override.")
endif()
set_source_files_properties(miniWeather_mpi_stdpar.cpp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} -I${PNETCDF_PATH}/include ${STDPAR_FLAGS}")
target_link_libraries(mpistdpar "-L${PNETCDF_PATH}/lib -lpnetcdf ${STDPAR_FLAGS}")

add_executable(mpistdpar_test miniWeather_mpi_stdpar.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI" AND "${STDPAR_FLAGS}" STREQUAL "")
  set(STDPAR_FLAGS "-stdpar=gpu")
endif()
set_source_files_properties(miniWeather_mpi_stdpar.cpp PROPERTIES COMPILE_FLAGS "${TEST_DEFS} -I${PNETCDF_PATH}/include ${STDPAR_FLAGS}")
target_link_libraries(mpistdpar_test yakl "-L${PNETCDF_PATH}/lib -lpnetcdf ${STDPAR_FLAGS}")

add_test(NAME MPISTDPAR_TEST COMMAND ./check_output.sh ./mpistdpar_test 1e-9 4.5e-5 ) 