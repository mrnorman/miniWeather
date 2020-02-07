cmake_minimum_required(VERSION 3.0)
project(miniWeather CXX)

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
include_directories(YAKL)
include_directories(cub)
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
  enable_language(CUDA)
  set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES LANGUAGE CUDA)
endif()
set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} -I${PNETCDF_PATH}/include ${YAKL_CXX_FLAGS}")

add_executable(yaklexe_test miniWeather_mpi_parallelfor.cpp)
set_target_properties(yaklexe_test PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(yaklexe_test yakl "-L${PNETCDF_PATH}/lib -lpnetcdf")
if ("${ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
  set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES LANGUAGE CUDA)
endif()
set_source_files_properties(miniWeather_mpi_parallelfor.cpp PROPERTIES COMPILE_FLAGS "${TEST_DEFS} -I${PNETCDF_PATH}/include ${YAKL_CXX_FLAGS}")

add_test(NAME YAKL_TEST COMMAND ./check_output.sh ./yaklexe_test 1e-9 4.5e-5 )



