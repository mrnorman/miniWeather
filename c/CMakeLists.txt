cmake_minimum_required(VERSION 3.0)
project(miniWeather CXX)

enable_testing()

include(utils.cmake)


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
## Compile the serial version
############################################################
add_executable(serial miniWeather_serial.cpp)
target_include_directories(serial PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(serial PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(serial "${_LIBS}")

add_executable(serial_test miniWeather_serial.cpp)
target_include_directories(serial_test PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(serial_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")
target_link_libraries(serial_test "${_LIBS}")

add_test(NAME SERIAL_TEST COMMAND ./check_output.sh ./serial_test 1e-13 4.5e-5 ) 


############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.cpp)
target_include_directories(mpi PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(mpi "${_LIBS}")

add_executable(mpi_test miniWeather_mpi.cpp)
target_include_directories(mpi_test PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(mpi_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")
target_link_libraries(mpi_test "${_LIBS}")

add_test(NAME MPI_TEST COMMAND ./check_output.sh ./mpi_test 1e-13 4.5e-5 ) 


############################################################
## Compile the MPI + OpenMP version
############################################################
SET(DO_OPENMP TRUE)
if ("${OPENMP_FLAGS}" STREQUAL "")
  MESSAGE(STATUS "WARNING: OPENMP_FLAGS CMake variable not set; therefore, not compiling OpenMP threading code")
  MESSAGE(STATUS "    Add -DOPENMP_FLAGS=\"...\" to enable OpenMP threaded code.")
  set(DO_OPENMP FALSE)
endif()
if (${DO_OPENMP})
  add_executable(openmp miniWeather_mpi_openmp.cpp)
  target_include_directories(openmp PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENMP_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS} ${OPENMP_FLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(openmp "${_LIBS}")

  add_executable(openmp_test miniWeather_mpi_openmp.cpp)
  target_include_directories(openmp_test PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${OPENMP_FLAGS}")
  target_link_libraries(openmp_test "${_LIBS}")

  add_test(NAME OPENMP_TEST COMMAND ./check_output.sh ./openmp_test 1e-13 4.5e-5 ) 
endif()



############################################################
## Compile the MPI + OpenACC version
############################################################
determine_openacc()
if (${DO_OPENACC})
  if ("${OPENACC_LINK_FLAGS}" STREQUAL "")
    SET(OPENACC_LINK_FLAGS ${OPENACC_FLAGS})
  endif()

  add_executable(openacc miniWeather_mpi_openacc.cpp)
  target_include_directories(openacc PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openacc PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENACC_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${OPENACC_LINK_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(openacc "${_LIBS}")

  add_executable(openacc_test miniWeather_mpi_openacc.cpp)
  target_include_directories(openacc_test PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openacc_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${OPENACC_FLAGS}")
  target_link_libraries(openacc_test "${_LIBS}")

  add_test(NAME OPENACC_TEST COMMAND ./check_output.sh ./openacc_test 1e-13 4.5e-5 ) 
endif()



############################################################
## Compile the MPI + OpenMP4.5 version
############################################################
determine_openmp45()
if (${DO_OPENMP45})
  if ("${OPENMP45_LINK_FLAGS}" STREQUAL "")
    SET(OPENMP45_LINK_FLAGS ${OPENMP45_FLAGS})
  endif()

  add_executable(openmp45 miniWeather_mpi_openmp45.cpp)
  target_include_directories(openmp45 PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp45 PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENMP45_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${OPENMP45_LINK_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(openmp45 "${_LIBS}")

  add_executable(openmp45_test miniWeather_mpi_openmp45.cpp)
  target_include_directories(openmp45_test PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp45_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${OPENMP45_FLAGS}")
  target_link_libraries(openmp45_test "${_LIBS}")

  # The XL compiler dumps out non-unique filenames that screw up parallel compilation
  # So it must compile the test at a different time than the original executable
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    add_dependencies(openmp45_test openmp45)
  endif()

  add_test(NAME OPENMP45_TEST COMMAND ./check_output.sh ./openmp45_test 1e-13 4.5e-5 ) 
endif()



