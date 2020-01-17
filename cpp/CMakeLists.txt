cmake_minimum_required(VERSION 2.8)
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


############################################################
## Add Kokkos 
############################################################
add_subdirectory(kokkos)
include_directories(kokkos)


############################################################
## Compile the serial version
############################################################
add_executable(serial miniWeather_serial.cpp)
target_include_directories(serial PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(serial PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(serial yakl "${_LIBS}")

add_executable(serial_test miniWeather_serial.cpp)
target_include_directories(serial_test PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(serial_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")
target_link_libraries(serial_test yakl "${_LIBS}")

add_test(NAME SERIAL_TEST COMMAND ./check_output.sh ./serial_test 1e-9 4.5e-5 ) 


############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.cpp)
target_include_directories(mpi PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(mpi yakl "${_LIBS}")

add_executable(mpi_test miniWeather_mpi.cpp)
target_include_directories(mpi_test PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(mpi_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")
target_link_libraries(mpi_test yakl "${_LIBS}")

add_test(NAME MPI_TEST COMMAND ./check_output.sh ./mpi_test 1e-9 4.5e-5 ) 


############################################################
## YAKL Version
############################################################
add_executable(yakl miniWeather_mpi_yakl.cpp)
target_include_directories(yakl PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(yakl PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENMP_FLAGS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS} ${OPENMP_FLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(yakl "${_LIBS}")

add_executable(yakl_test miniWeather_mpi_yakl.cpp)
target_include_directories(yakl_test PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(yakl_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${OPENMP_FLAGS}")
target_link_libraries(yakl_test "${_LIBS}")

add_test(NAME OPENMP_TEST COMMAND ./check_output.sh ./yakl_test 1e-9 4.5e-5 ) 



