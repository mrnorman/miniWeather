cmake_minimum_required(VERSION 2.8)
project(miniWeather Fortran)

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
## Append FFLAGS
############################################################
if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
  SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none")
endif()
if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
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
## Check to see if the compiler is OpenACC-capable
############################################################
# If user didn't specify an OpenACC-capable compiler list, then create one
if ("${OPENACC_LIST}" STREQUAL "")
  SET(OPENACC_LIST GNU PGI)
  MESSAGE(STATUS "OpenACC-capable list is: ${OPENACC_LIST}")
endif()

# Check compiler against openacc-capable list
SET(DO_OPENACC FALSE)
list(FIND OPENACC_LIST ${CMAKE_Fortran_COMPILER_ID} _ind)
if( ${_ind} GREATER -1 )
  SET(DO_OPENACC TRUE)
  # GNU is only openacc-capable with 8.0 onward
  if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_Fortran_COMPILER_VERSION} LESS 8 )
    MESSAGE(STATUS "WARNING: GNU can only compile OpenACC code with versions >= 8.0. Not compiling OpenACC code.")
    SET(DO_OPENACC FALSE)
  endif()
  # If OPENACC_FLAGS isn't set, then do not compile the openacc code
  if ("${OPENACC_FLAGS}" STREQUAL "")
    MESSAGE(STATUS "WARNING: OPENACC_FLAGS CMake variable must be set to compile OpenACC code.")
    MESSAGE(STATUS "    Add -DOPENACC_FLAGS=\"...\" to your cmake configure line to enable OpenACC compilation.")
    SET(DO_OPENACC FALSE)
  endif()
endif()


if (${DO_OPENACC})
  MESSAGE(STATUS "Compiling OpenACC code with ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} using the flags: ${OPENACC_FLAGS}")
  ############################################################
  ## Compile the MPI + OpenACC version
  ############################################################
  if ("${OPENACC_LINK_FLAGS}" STREQUAL "")
    MESSAGE(STATUS "WARNING: OPENACC_LINK_FLAGS CMake variable not specified. ")
    MESSAGE(STATUS "    Using OPENACC_FLAGS in place of OPENACC_LINK_FLAGS in the link phase. ")
    MESSAGE(STATUS "    Add -DOPENACC_LINK_FLAGS to change this behavior. ")
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
else() 
  MESSAGE(STATUS "WARNING: OpenACC code will not be compiled.")
endif()


############################################################
## Check to see if the compiler is OpenMP-offload-capable
############################################################
# If user didn't specify an OpenMP4.5-capable compiler list, then create one
if ("${OPENMP45_LIST}" STREQUAL "")
  SET(OPENMP45_LIST XL GNU)
  MESSAGE(STATUS "OpenMP4.5-capable list is: ${OPENMP45_LIST}")
endif()

# Check compiler against openmp4.5-capable list
SET(DO_OPENMP45 FALSE)
list(FIND OPENMP45_LIST ${CMAKE_Fortran_COMPILER_ID} _ind)
if( ${_ind} GREATER -1 )
  SET(DO_OPENMP45 TRUE)
  # GNU is only openmp4.5-capable with 8.0 onward
  if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_Fortran_COMPILER_VERSION} LESS 8)
    MESSAGE(WARNING "WARNING: GNU can only compile OpenMP4.5 code with versions >= 8.0"
                    "Not compiling OpenMP4.5 code.")
    SET(DO_OPENMP45 FALSE)
  endif()
  # If OPENMP45_FLAGS isn't set, then do not compile the openmp4.5 code
  if ("${OPENMP45_FLAGS}" STREQUAL "")
    MESSAGE(WARNING "WARNING: OPENMP45_FLAGS CMake variable must be set to compile OpenMP4.5 code."
                    "add -DOPENMP45_FLAGS=\"...\" to your cmake configure line to enable OpenMP4.5 compilation.")
    SET(DO_OPENMP45 FALSE)
  endif()
endif()

if (${DO_OPENMP45})
  MESSAGE(STATUS "Compiling OpenMP45 code with ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} using the flags: ${OPENMP45_FLAGS}")
  ############################################################
  ## Compile the MPI + OpenMP4.5 version
  ############################################################
  if ("${OPENMP45_LINK_FLAGS}" STREQUAL "")
    MESSAGE(STATUS "WARNING: OPENMP45_LINK_FLAGS CMake variable not specified. ")
    MESSAGE(STATUS "    Using OPENMP45_FLAGS in place of OPENMP45_LINK_FLAGS in the link phase. ")
    MESSAGE(STATUS "    Add -DOPENMP45_LINK_FLAGS to change this behavior. ")
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
else() 
  MESSAGE(STATUS "WARNING: OpenMP4.5 code will not be compiled.")
endif()


############################################################
## Check to see if the compiler supports DO CONCURRENT
############################################################
# If user didn't specify an OpenACC-capable compiler list, then create one
if ("${CONCURRENT_LIST}" STREQUAL "")
  SET(CONCURRENT_LIST PGI)
  MESSAGE(STATUS "Concurrent-capable list is: ${CONCURRENT_LIST}")
endif()

# Check compiler against openacc-capable list
SET(DO_CONCURRENT FALSE)
list(FIND CONCURRENT_LIST ${CMAKE_Fortran_COMPILER_ID} _ind)
if( ${_ind} GREATER -1 )
  SET(DO_CONCURRENT TRUE)
  #if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI" )
  #  SET(CONCURRENT_FLAGS "-stdpar=gpu -Minfo=accel")
  #else()
  #  SET(DO_CONCURRENT FALSE)
  #endif()
  # If CONCURRENT_FLAGS isn't set, then do not compile the DO CONCURRENT code
  if ("${CONCURRENT_FLAGS}" STREQUAL "")
    MESSAGE(STATUS "WARNING: CONCURRENT_FLAGS CMake variable must be set to compile DO CONCURRENT code.")
    MESSAGE(STATUS "    Add -DCONCURRENT_FLAGS=\"...\" to your cmake configure line to enable DO CONCURRENT compilation.")
    SET(DO_CONCURRENT FALSE)
  endif()

endif()

if (${DO_CONCURRENT})
  add_executable(concurrent miniWeather_mpi_concurrent.F90)
  target_include_directories(concurrent PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(concurrent PROPERTIES COMPILE_FLAGS "${EXE_DEFS} ${CONCURRENT_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${CONCURRENT_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(concurrent "${_LIBS}")

  add_executable(concurrent_test miniWeather_mpi_concurrent.F90)
  target_include_directories(concurrent_test PUBLIC ${PNETCDF_PATH}/include)
  set_target_properties(concurrent_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS} ${CONCURRENT_FLAGS}")
  set(_LIBS "-L${PNETCDF_PATH}/lib -lpnetcdf ${CONCURRENT_FLAGS} ${LDFLAGS}")
  string(STRIP "${_LIBS}" _LIBS)
  target_link_libraries(concurrent_test "${_LIBS}")

  add_test(NAME CONCURRENT_TEST COMMAND ./check_output.sh ./concurrent_test 1e-13 4.5e-5 ) 
endif()

