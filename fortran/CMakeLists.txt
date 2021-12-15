cmake_minimum_required(VERSION 3.0)
project(miniWeather Fortran)

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
## Append FFLAGS
############################################################
if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
  SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none")
endif()
if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI" OR "${CMAKE_Fortran_COMPILER_ID}" STREQUAL "NVHPC" )
  SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mextend")
endif()
SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FFLAGS}")


############################################################
## Compile the serial version
############################################################
add_executable(serial miniWeather_serial.F90)
target_include_directories(serial PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(serial PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(serial "${_LIBS}")

add_executable(serial_test miniWeather_serial.F90)
target_include_directories(serial_test PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(serial_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")
target_link_libraries(serial_test "${_LIBS}")

add_test(NAME SERIAL_TEST COMMAND ./check_output.sh ./serial_test 1e-13 4.5e-5 ) 


############################################################
## Compile the MPI version
############################################################
add_executable(mpi miniWeather_mpi.F90)
target_include_directories(mpi PUBLIC ${PNETCDF_PATH}/include)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS}")
string(STRIP "${_LIBS}" _LIBS)
target_link_libraries(mpi "${_LIBS}")

add_executable(mpi_test miniWeather_mpi.F90)
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
  add_executable(openmp miniWeather_mpi_openmp.F90)
  target_include_directories(openmp PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENMP_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${LDFLAGS} ${OPENMP_FLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(openmp "${_LIBS}")

  add_executable(openmp_test miniWeather_mpi_openmp.F90)
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

  add_executable(openacc miniWeather_mpi_openacc.F90)
  target_include_directories(openacc PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openacc PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENACC_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${OPENACC_LINK_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(openacc "${_LIBS}")

  add_executable(openacc_test miniWeather_mpi_openacc.F90)
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

  add_executable(openmp45 miniWeather_mpi_openmp45.F90)
  target_include_directories(openmp45 PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp45 PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${OPENMP45_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${OPENMP45_LINK_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(openmp45 "${_LIBS}")

  add_executable(openmp45_test miniWeather_mpi_openmp45.F90)
  target_include_directories(openmp45_test PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(openmp45_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${OPENMP45_FLAGS}")
  target_link_libraries(openmp45_test "${_LIBS}")

  # The XL compiler dumps out non-unique filenames that screw up parallel compilation
  # So it must compile the test at a different time than the original executable
  if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "XL")
    add_dependencies(openmp45_test openmp45)
  endif()

  add_test(NAME OPENMP45_TEST COMMAND ./check_output.sh ./openmp45_test 1e-13 4.5e-5 ) 
endif()



############################################################
## Compile the MPI + do concurrent version
############################################################
determine_do_concurrent()
if (${DO_DO_CONCURRENT})
  if ("${DO_CONCURRENT_LINK_FLAGS}" STREQUAL "")
    SET(DO_CONCURRENT_LINK_FLAGS ${DO_CONCURRENT_FLAGS})
  endif()

  add_executable(do_concurrent miniWeather_mpi_do_concurrent.F90)
  target_include_directories(do_concurrent PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(do_concurrent PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${DO_CONCURRENT_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${DO_CONCURRENT_LINK_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(do_concurrent "${_LIBS}")

  add_executable(do_concurrent_test miniWeather_mpi_do_concurrent.F90)
  target_include_directories(do_concurrent_test PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(do_concurrent_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${DO_CONCURRENT_FLAGS}")
  target_link_libraries(do_concurrent_test "${_LIBS}")

  add_test(NAME DO_CONCURRENT_TEST COMMAND ./check_output.sh ./do_concurrent_test 1e-13 4.5e-5 ) 
endif()



